import torch
from torch.distributions import Categorical

from controller import ControllerModel


class DiscreteModelPredictiveController(ControllerModel):
    tags = ["discrete", "stochastic"]

    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        self.policy = torch.nn.Linear(z_dim + h_dim, action_dim)

        self.value = torch.nn.Linear(h_dim, 1)

    def forward(self, z_t, h_t):
        # z_t: (B, z_dim) and h_t: (B, h_dim)
        x = torch.cat([z_t, h_t], dim=-1)
        logits = self.policy(x)               # (B, action_dim)
        dist = Categorical(logits=logits)

        action = dist.sample()                # (B,)
        logp = dist.log_prob(action).unsqueeze(-1)  # (B, 1)
        value = self.value(h_t)               # (B, 1)
        entropy = dist.entropy().unsqueeze(-1)

        return action, logp, value, entropy

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
