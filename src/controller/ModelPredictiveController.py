import torch

from Model import Model


class ModelPredictiveController(Model):

    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Linear(z_dim + h_dim, action_dim)

    def forward(self, z_t, h_t):
        # z_t: (B, z_dim) and h_t: (B, h_dim)
        x = torch.cat([z_t, h_t], dim=-1)
        return self.fc(x)  # (B, action_dim)
