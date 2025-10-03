# src/WorldModel.py

import torch
import gymnasium as gym

from Model import Model
from memory.CPC import info_nce_loss
from memory.MemoryModel import flatten_vision_latents

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
        self.memory = memory_model(input_dim=self.vision.embed_dim, **memory_args)
        controller_h_dim = memory_args.get("d_model", 128)
        self.controller = controller_model(z_dim=self.vision.embed_dim, h_dim=controller_h_dim, **controller_args)

        self.cpc_model = cpc_model
        if cpc_args is None:
            cpc_args = {"context_dim": 128, "prediction_steps": 12}
        self.cpc_args = cpc_args
        if cpc_model is not None:
            self.cpc = cpc_model(self.vision.embed_dim, **cpc_args)
        else:
            self.cpc = None

    def forward(self, input, action_space: gym.spaces.Box, use_cpc: bool = False, return_losses: bool = False):
        vision_out = self.vision(input)
        recon, vq_loss, z_q = vision_out["recon"], vision_out["vq_loss"], vision_out["z_q"]

        z_t = z_q.mean(dim=(2, 3))
        h_t = self.memory(z_t)
        
        # Controller now returns a tanh-squashed action in [-1, 1]
        raw_action, log_probs = self.controller(z_t, h_t)

        # Scale and shift the action to fit the environment's action space
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=raw_action.device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=raw_action.device)
        action_high[2] = 0
        action = action_low + (0.5 * (raw_action + 1.0) * (action_high - action_low))

        if not return_losses:
            return action

        recon_loss = torch.nn.functional.mse_loss(recon, input)
        outputs = {
            "action": action,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "log_probs": log_probs,
        }

        if use_cpc and self.cpc is not None:
            z_seq = flatten_vision_latents(z_q)
            preds = self.cpc(z_seq)
            cpc_loss = info_nce_loss(z_seq, preds)
            total_loss += cpc_loss + recon_loss + vq_loss
            outputs["cpc_loss"] = cpc_loss
            outputs["total_loss"] = total_loss

        return outputs

    def export_hyperparam(self):
        pass
