import torch

from Model import Model


class ModelPredictiveController(Model):

    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        self.fc = torch.nn.Linear(z_dim + h_dim, action_dim)

    def forward(self, z_t, h_t):
        # z_t: (B, z_dim) and h_t: (B, h_dim)
        x = torch.cat([z_t, h_t], dim=-1)
        return self.fc(x), self.fc(x)  # (B, action_dim)

    def export_hyperparams(self):
        return {
            "z_dim": self.z_dim,
            "h_dim": self.h_dim,
            "action_dim": self.action_dim
        }

    def save_state(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)
