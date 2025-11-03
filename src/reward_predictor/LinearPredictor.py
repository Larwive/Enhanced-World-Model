import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from Model import Model



class LinearPredictorModel(Model):

    def __init__(self, z_dim, h_dim, action_dim, **_kwargs):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        self.linear = torch.nn.Linear(z_dim + h_dim + action_dim + 1, 1) # Suspicious dim calculation. Need proper proof.

    def forward(self, z_t, h_t, log_prob, last_reward):
        x = torch.cat([z_t, h_t, log_prob.unsqueeze(-1), torch.tensor([[last_reward]])], dim=-1)  # Potential incorrect cat
        return self.linear(x)

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
