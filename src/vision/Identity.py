# src/vision/Identity.py

import torch
from Model import Model

class Identity(Model):
    """
    A dummy vision model that acts as an identity function for 1D vector-based environments.
    It mimics the output of a real vision model to be compatible with the WorldModel class.
    """
    def __init__(self, input_shape, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        # This is needed for the WorldModel's CPC logic and controller input
        self.embed_dim = input_shape[0]

    def forward(self, input):
        # Return a dictionary consistent with VQ_VAE
        return {
            "recon": input,
            "vq_loss": torch.tensor(0.0, device=input.device),
            "z_q": input.unsqueeze(-1).unsqueeze(-1)  # Add dummy H, W dims for flatten_vision_latents
        }

    def export_hyperparam(self):
        return {
            "class_name": self.__class__.__name__,
            "input_shape": self.input_shape,
        }