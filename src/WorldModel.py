import torch
import gymnasium as gym

from Model import Model
from memory.CPC import info_nce_loss
from memory.MemoryModel import flatten_vision_latents

import vision
import memory
import controller
import reward_predictor

def describe_action_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete",
            "n": space.n,
            "low": 0,
            "high": space.n - 1
        }

    elif isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "shape": space.shape,
            "low": space.low,
            "high": space.high
        }

    elif isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": space.nvec,
            "low": np.zeros_like(space.nvec),
            "high": space.nvec - 1
        }

    elif isinstance(space, gym.spaces.MultiBinary):
        return {
            "type": "MultiBinary",
            "n": space.n,
            "low": np.zeros(space.n, dtype=int),
            "high": np.ones(space.n, dtype=int)
        }
    # Below are cases not supported yet.
    elif isinstance(space, gym.spaces.Tuple):
        return {
            "type": "Tuple",
            "spaces": [describe_action_space(s) for s in space.spaces]
        }

    elif isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {k: describe_action_space(v) for k, v in space.spaces.items()}
        }

    else:
        return {"type": "Unknown", "details": str(space)}

def squash_to_action_space(raw_action, action_space):
    """
    Mappe une action non bornée dans les bornes de action_space (gym.spaces.Box)
    """
    decrypted_action_space = describe_action_space(action_space)
    low = torch.as_tensor(decrypted_action_space["low"], dtype=torch.float32, device=raw_action.device)
    high = torch.as_tensor(decrypted_action_space["high"], dtype=torch.float32, device=raw_action.device)
    scaled = torch.sigmoid(raw_action) * (high - low) + low
    return scaled.squeeze()


class WorldModel(Model):

    def __init__(self,
                 vision_model,
                 memory_model,
                 controller_model,
                 input_shape,
                 vision_args,
                 memory_args,
                 controller_args,
                 cpc_model: Model | None = None,
                 cpc_args: dict | None = None) -> None:
        super().__init__()
        self.iter_num = 0  # The number of training iterations tied to this model.
        self.nb_experiments = 0
        self.vision = vision_model(input_shape, **vision_args)

        # The input to the memory model is the output of the vision model
        memory_input_dim = self.vision.embed_dim
        self.memory = memory_model(input_dim=memory_input_dim, **memory_args)

        # The controller takes the output of both the vision and memory models
        # Note: self.memory.transformer.d_model might be a more robust way to get h_dim
        self.memory_d_model = memory_args.get("d_model", 128) # Default to 128 if not specified
        self.action_dim = controller_args["action_dim"]
        controller_h_dim = self.memory_d_model
        self.controller = controller_model(z_dim=self.vision.embed_dim, h_dim=controller_h_dim, **controller_args)

        if cpc_args is None:
            cpc_args = {"context_dim": 128, "prediction_steps": 12}
        if cpc_model is not None:
            self.cpc = cpc_model(self.vision.embed_dim, **cpc_args)
        else:
            self.cpc = None

        self.reward_predictor = None

    def forward(self, input, action_space, use_cpc: bool = False, return_losses: bool = False, last_reward=None):
        recon, vq_loss = self.vision(input)
        z_q = self.vision.encode(input)

        # For image-based models, z_q is (B, D, H, W). We need a vector for the memory model.
        # For vector-based models (Identity), z_q is (B, D, 1, 1).
        z_t = z_q.mean(dim=(2, 3))  # This works for both cases

        h_t = self.memory(z_t.detach().clone())

        action, log_probs = self.controller(z_t.detach().clone(), h_t)

        action = squash_to_action_space(action, action_space)

        if not return_losses:
            return action

        recon_loss = torch.nn.functional.mse_loss(recon, input)
        total_loss = recon_loss + vq_loss
        outputs = {
            "action": action,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "total_loss": total_loss,
            "log_probs": log_probs
        }

        if use_cpc and self.cpc is not None:
            z_seq = flatten_vision_latents(z_q)
            preds = self.cpc(z_seq)
            cpc_loss = info_nce_loss(z_seq, preds)
            total_loss = total_loss + cpc_loss
            outputs["cpc_loss"] = cpc_loss
            outputs["total_loss"] = total_loss

        if self.reward_predictor and last_reward is not None:
            # Need to detach and clone h_t ?
            outputs["predicted_reward"] = self.reward_predictor(z_t.detach().clone(), h_t, log_probs, last_reward)
        return outputs

    def export_hyperparams(self):
        pass

    def save(self, path, obs_space, action_space):
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

        if self.cpc is not None:
            saving_dict["cpc_model"] = self.cpc.__class__.__name__
            saving_dict["cpc_args"] = self.cpc.export_hyperparams()
            saving_dict["cpc_dict"] = self.cpc.save_state()

        if self.reward_predictor is not None:
            saving_dict["reward_predictor_model"] = self.reward_predictor.__class__.__name__
            saving_dict["reward_predictor_args"] = self.reward_predictor.export_hyperparams()
            saving_dict["reward_predictor_dict"] = self.reward_predictor.save_state()
        torch.save(saving_dict, path)

    def load(self, path, obs_space, action_space):
        # TODO: Check input/output shape consistencies between components before loading the weights ?
        saved_dict = torch.load(path, weights_only=False)

        assert obs_space == saved_dict["obs_space"] and action_space == saved_dict["action_space"], "Obeservation space and/or action space of the saved model do not match those of the current environment."
        self.iter_num = saved_dict["iter_num"]
        self.nb_experiments = saved_dict["nb_experiments"]

        self.vision = getattr(vision, saved_dict["vision_model"])(**saved_dict["vision_args"])
        self.vision.load(saved_dict["vision_dict"])

        self.memory = getattr(memory, saved_dict["memory_model"])(**saved_dict["memory_args"])
        self.memory.load(saved_dict["memory_dict"])

        self.controller = getattr(controller, saved_dict["controller_model"])(**saved_dict["controller_args"])
        self.controller.load(saved_dict["controller_dict"])

        if "CPC_dict" in saved_dict:
            self.cpc = getattr(memory, saved_dict["cpc_model"])(**saved_dict["cpc_args"])
            self.cpc.load(saved_dict["cpc_dict"])
        if "reward_predictor_dict" in saved_dict:
            self.reward_predictor = getattr(reward_predictor, saved_dict["reward_predictor_model"])(**saved_dict["reward_predictor_args"])
            self.reward_predictor.load(saved_dict["reward_predictor_dict"])

    def set_reward_predictor(self, reward_predictor_class: Model, **kwargs):
        self.reward_predictor = reward_predictor_class(z_dim=self.vision.embed_dim, h_dim=self.memory_d_model, action_dim=self.action_dim, **kwargs)

