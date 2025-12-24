import torch

from vision import VisionModel

class Identity(VisionModel):
    """
    A dummy vision model that acts as an identity function for 1D vector-based environments.
    It mimics the output of a real vision model to be compatible with the WorldModel class.
    """

    tags = ["vector_based"]

    def __init__(self, input_shape, **_kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = input_shape[0]

    def forward(self, input):
        # Return the input as the "reconstruction" and a zero loss
        return input, torch.tensor(0.0, device=input.device)

    def encode(self, input, is_image_based: bool):
        if is_image_based:
            return input.unsqueeze(-1).unsqueeze(-1) # Add dummy H, W dims for flatten_vision_latents
        return input

    def export_hyperparams(self):
        return {
            "input_shape": self.input_shape,
        }

    def save_state(self):
        return {}

    def load(self, _state_dict):
        pass
