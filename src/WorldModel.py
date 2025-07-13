import torch

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
        self.memory = memory_model(**memory_args)
        self.controller = controller_model(**controller_args)

        self.cpc_model = cpc_model
        if cpc_args == None:
            cpc_args = {"context_dim": 128, "prediction_steps": 12}
        self.cpc_args = cpc_args
        if cpc_model is not None:
            self.cpc = cpc_model(self.vision.embed_dim, **cpc_args)
        else:
            self.cpc = None

    def forward(self, input, use_cpc: bool = False, return_losses: bool = False):
        recon, vq_loss = self.vision(input)
        z_q = self.vision.encode(input)
        z_t = z_q.mean(dim=(2, 3))

        h_t = self.memory(z_t)

        action = self.controller(z_t, h_t)

        if not return_losses:
            return action

        recon_loss = torch.nn.functional.mse_loss(recon, input)

        total_loss = recon_loss + vq_loss

        outputs = {"action": action, "recon_loss": recon_loss, "vq_loss": vq_loss, "total_loss": total_loss}

        if use_cpc and self.cpc is not None:
            z_seq = flatten_vision_latents(z_q)
            preds = self.cpc(z_seq)
            cpc_loss = info_nce_loss(z_seq, preds)
            total_loss = total_loss + cpc_loss
            outputs["cpc_loss"] = cpc_loss
            outputs["total_loss"] = total_loss

        return outputs
