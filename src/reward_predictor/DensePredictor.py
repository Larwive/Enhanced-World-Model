import torch
from torch import nn
from Model import Model


class DensePredictorModel(Model):
    def __init__(self, z_dim, h_dim, action_dim=None, **_kwargs):
        super().__init__()
        input_dim = z_dim + h_dim + 2  # Need formal proof.

        self.net = nn.Sequential( # Hard coded for now.
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z_t, h_t, log_prob, last_reward):
        # Ensure log_prob and last_reward are column vectors [batch_size, 1]
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(-1)
        if last_reward.dim() == 1:
            last_reward = last_reward.unsqueeze(-1)

        x = torch.cat([z_t, h_t, log_prob, last_reward], dim=-1)
        return self.net(x)

    def export_hyperparam(self):
        pass
