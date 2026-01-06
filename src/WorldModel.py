from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

import controller
import memory
import vision
from Model import Model
from utils.registry import discover_modules

VISION_REGISTRY: dict = discover_modules(vision)
MEMORY_REGISTRY: dict = discover_modules(memory)
CONTROLLER_REGISTRY: dict = discover_modules(controller)


def describe_action_space(space: Any) -> dict:
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": space.n, "low": 0, "high": space.n - 1}

    elif isinstance(space, gym.spaces.Box):
        return {"type": "Box", "shape": space.shape, "low": space.low, "high": space.high}

    elif isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": space.nvec,
            "low": np.zeros_like(space.nvec),
            "high": space.nvec - 1,
        }

    elif isinstance(space, gym.spaces.MultiBinary):
        return {
            "type": "MultiBinary",
            "n": space.n,
            "low": np.zeros(space.n, dtype=int),
            "high": np.ones(space.n, dtype=int),
        }
    # Below are cases not supported yet.
    elif isinstance(space, gym.spaces.Tuple):
        return {"type": "Tuple", "spaces": [describe_action_space(s) for s in space.spaces]}

    elif isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {k: describe_action_space(v) for k, v in space.spaces.items()},
        }

    else:
        return {"type": "Unknown", "details": str(space)}


def squash_to_action_space(raw_action: torch.Tensor, action_space: Any) -> torch.Tensor:
    """
    Mappe une action non bornée dans les bornes de action_space (gym.spaces.Box).
    """
    decrypted_action_space = describe_action_space(action_space)
    low = torch.as_tensor(
        decrypted_action_space["low"], dtype=torch.float32, device=raw_action.device
    )
    high = torch.as_tensor(
        decrypted_action_space["high"], dtype=torch.float32, device=raw_action.device
    )
    scaled = 0.5 * ((torch.tanh(raw_action) + 1) * (high - low)) + low
    return scaled


class WorldModel(Model):
    def __init__(
        self,
        vision_model: vision.VisionModel,
        memory_model: memory.MemoryModel,
        controller_model: controller.ControllerModel,
        input_shape: tuple[int, ...],
        vision_args: dict,
        memory_args: dict,
        controller_args: dict,
    ) -> None:
        super().__init__()
        self.iter_num = 0  # The number of training iterations tied to this model.
        self.nb_experiments = 0
        self.vision = vision_model(input_shape, **vision_args)

        # The input to the memory model is the output of the vision model
        self.memory = memory_model(**memory_args)

        # The controller takes the output of both the vision and memory models
        # Note: self.memory.transformer.d_model might be a more robust way to get h_dim
        self.memory_d_model = memory_args.get("d_model", 128)  # Default to 128 if not specified
        self.action_dim = controller_args["action_dim"]
        controller_h_dim = self.memory_d_model
        self.controller = controller_model(
            z_dim=self.vision.embed_dim, h_dim=controller_h_dim, **controller_args
        )

        self.a_prev = None

    def forward(
        self,
        input: torch.Tensor,
        action_space: Any,
        is_image_based: bool,
        return_losses: bool = False,
        last_reward: torch.Tensor | None = None,
    ) -> torch.Tensor | dict:
        """
        Args:
            input: observation actuelle
            action_space: espace d'actions
            return_losses: retourner les losses
            last_reward: dernière récompense
            z_next_actual: (B, latent_dim, 1, 1) - vrai prochain z pour training
        """
        # === VISION MODEL ===
        recon, vq_loss = self.vision(input)
        z_e = self.vision.encode(input, is_image_based=is_image_based)  # (B, latent_dim, H, W)

        # Flatten spatial dimensions pour obtenir le vecteur latent
        if is_image_based:
            z_t = z_e.mean(dim=(2, 3))  # (B, latent_dim)
        else:
            z_t = z_e

        # === MEMORY MODEL ===
        # Prédire le prochain état latent z_{t+1} et obtenir l'état caché
        # z_next_pred, h_t = self.memory(z_t)
        # z_next_pred: (B, latent_dim, 1, 1)
        # h_t: (B, d_model=128)
        if self.a_prev is None:
            sample_shape = np.shape(action_space.sample())
            if sample_shape == ():
                sample_shape = (self.action_dim,)
            self.a_prev = torch.zeros(
                (input.shape[0], *sample_shape), device=input.device, dtype=torch.float32
            )

        h_t = self.memory.update_memory(z_t, self.a_prev)

        # === CONTROLLER ===
        action, log_probs, value, _entropy = self.controller(z_t, h_t)
        if isinstance(action_space, gym.spaces.Discrete):
            n: int = int(action_space.n)

            device = input.device
            action = action.to(device)
            log_probs = log_probs.to(device)

            a_prev_onehot = (
                torch.nn.functional.one_hot(action.long(), num_classes=n).float().to(device)
            )
            self.a_prev = a_prev_onehot.detach()

            z_next_pred = self.memory.predict_next(z_t, self.a_prev, h_t)
        else:
            action = squash_to_action_space(action, action_space)
            self.a_prev = action.detach()
            z_next_pred = self.memory.predict_next(z_t, action, h_t)

        if not return_losses:
            return action

        # === LOSSES ===
        # Vision loss depends on model type
        if "no_reconstruction" in self.vision.tags:
            # JEPA: vq_loss already contains the full training loss (VICReg)
            # No reconstruction loss since JEPA predicts in latent space
            recon_loss = torch.zeros(input.shape[0], device=input.device)
            total_loss = vq_loss.mean()
        else:
            # VQ-VAE: compute reconstruction loss between recon and input
            recon_loss = torch.nn.functional.mse_loss(recon, input, reduction="none").mean(
                dim=tuple(range(1, recon.dim()))
            )
            total_loss = recon_loss.mean() + vq_loss.mean()

        outputs = {
            "memory_prediction": z_next_pred,  # (B, latent_dim) - prédiction de z_{t+1}
            "memory_hidden": h_t,  # (B, d_model) - état caché du transformer
            "action": action,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "total_loss": total_loss,
            "log_probs": log_probs,
            "value": value,
        }

        return outputs

    def reset_env_memory(self, env_idx: int | torch.Tensor) -> None:
        self.memory.reset_env_memory(env_idx)
        if self.a_prev is not None:
            self.a_prev[env_idx] = 0

    def export_hyperparams(self) -> dict:
        hyperparams_dict = {
            "vision_model": self.vision.__class__.__name__,
            "vision_args": self.vision.export_hyperparams(),
            "memory_model": self.memory.__class__.__name__,
            "memory_args": self.memory.export_hyperparams(),
            "controller_model": self.controller.__class__.__name__,
            "controller_args": self.controller.export_hyperparams(),
        }

        return hyperparams_dict

    def save(self, path: Path, obs_space: Any, action_space: Any) -> None:
        saving_dict = {
            "iter_num": self.iter_num,
            "nb_experiments": self.nb_experiments,
            "obs_space": obs_space,
            "action_space": action_space,
            "vision_model": self.vision.__class__.__name__,
            "vision_args": self.vision.export_hyperparams(),
            "vision_dict": self.vision.save_state(),
            "memory_model": self.memory.__class__.__name__,
            "memory_args": self.memory.export_hyperparams(),
            "memory_dict": self.memory.save_state(),
            "controller_model": self.controller.__class__.__name__,
            "controller_args": self.controller.export_hyperparams(),
            "controller_dict": self.controller.save_state(),
        }

        torch.save(saving_dict, path)

    def load(self, path: Path, obs_space: Any, action_space: Any, device: torch.device) -> None:
        self.patch_load(path, "vmc", obs_space, action_space, device)

    def patch_load(
        self,
        patch_path: Path,
        patches: str,
        obs_space: Any,
        action_space: Any,
        device: torch.device,
    ) -> None:
        # TODO: Check input/output shape consistencies between components before loading the weights ?
        saved_dict = torch.load(patch_path, weights_only=False, map_location=device)

        self.iter_num = saved_dict["iter_num"]
        self.nb_experiments = saved_dict["nb_experiments"]

        if "v" in patches:
            if saved_dict["obs_space"] != obs_space:
                print(
                    "\nObservation space of the vision to load does not match those of the current environment.\n"
                )
            self.vision = VISION_REGISTRY[saved_dict["vision_model"]](**saved_dict["vision_args"])
            self.vision.load(saved_dict["vision_dict"])

        if "m" in patches:
            self.memory = MEMORY_REGISTRY[saved_dict["memory_model"]](**saved_dict["memory_args"])
            self.memory.load(saved_dict["memory_dict"])

        if "c" in patches:
            if saved_dict["action_space"] != action_space:
                print(
                    "\nAction space of the controller to load does not match those of the current environment.\n"
                )
            self.controller = CONTROLLER_REGISTRY[saved_dict["controller_model"]](
                **saved_dict["controller_args"]
            )
            self.controller.load(saved_dict["controller_dict"])


def render_first_env(envs: Any, title: str = "") -> None:
    import cv2

    frames = envs.render()
    cv2.imshow(title + envs.spec.id, frames[0][..., ::-1])  # RGB -> BGR for OpenCV
    cv2.waitKey(1)
