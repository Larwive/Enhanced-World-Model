import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from Model import Model



class LinearPredictorModel(Model):

    def __init__(self, z_dim, h_dim, action_dim, **_kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(z_dim + h_dim + 2, 1) # Suspicious dim calculation. Need proper proof.

    def forward(self, z_t, h_t, log_prob, last_reward):
        x = torch.cat([z_t, h_t, log_prob.unsqueeze(-1), torch.tensor([[last_reward]])], dim=-1)  # Potential incorrect cat
        return self.linear(x)

    def export_hyperparam(self):
        return {
            "class_name": self.__class__.__name__,
        }
