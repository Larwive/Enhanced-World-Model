import torch

from Model import Model
from MemoryModel import flatten_vision_latents


class TemporalTransformer(Model):

    def __init__(self, input_shape, num_layers=4):
        super().__init__()
        input_dim = input_shape[0]
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input):
        input = flatten_vision_latents(input)  # (B, D, H, W) -> (B, T, D)
        # input: (B, T, D) → needs (T, B, D)
        input = input.transpose(0, 1)
        memory = self.transformer(input)  # (T, B, D)
        return memory.transpose(0, 1)
