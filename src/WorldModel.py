import torch
import gymnasium as gym

from Model import Model
from memory.CPC import info_nce_loss
from memory.MemoryModel import flatten_vision_latents

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
        self.vision = vision_model(input_shape, **vision_args)

        # The input to the memory model is the output of the vision model
        memory_input_dim = self.vision.embed_dim
        self.memory = memory_model(input_dim=memory_input_dim, **memory_args)

        # The controller takes the output of both the vision and memory models
        # Note: self.memory.transformer.d_model might be a more robust way to get h_dim
        controller_h_dim = memory_args.get("d_model", 128)  # Default to 128 if not specified
        self.controller = controller_model(z_dim=self.vision.embed_dim, h_dim=controller_h_dim, **controller_args)

        self.cpc_model = cpc_model
        if cpc_args is None:
            cpc_args = {"context_dim": 128, "prediction_steps": 12}
        self.cpc_args = cpc_args
        if cpc_model is not None:
            self.cpc = cpc_model(self.vision.embed_dim, **cpc_args)
        else:
            self.cpc = None

    def forward(self, input, action_space, use_cpc: bool = False, return_losses: bool = False):
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

        return outputs

    def export_hyperparam(self):
        pass
