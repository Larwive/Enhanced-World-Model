import torch

from Model import Model
from .MemoryModel import flatten_vision_latents

class TemporalTransformer(Model):

    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # This layer projects the input from the vision model (or identity)
        # to the dimension required by the transformer.
        self.input_proj = torch.nn.Linear(input_dim, d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input):
        # We don't need to flatten here, as the input from the WorldModel will be (B, D)
        # input shape: (B, D) from vision model
        projected_input = self.input_proj(input) # (B, d_model)
        # print(f"Projected input shape: {projected_input.shape} and value: {projected_input}")
        
        # Transformer expects a sequence. We'll treat the single vector as a sequence of length 1.
        # (B, D) -> (B, 1, D)
        projected_input = projected_input.unsqueeze(1)

        # print(f"Input to transformer shape: {projected_input.shape} and value: {projected_input}")
        
        memory = self.transformer(projected_input)  # (B, 1, d_model)

        # print(f"Output from transformer shape: {memory.shape} and value: {memory}")
        
        # Return the output, removing the sequence dimension
        return memory.squeeze(1) # (B, d_model)

    def export_hyperparam(self):
        # We'll let the WorldModel handle hyperparameter exports for its sub-modules
        pass
