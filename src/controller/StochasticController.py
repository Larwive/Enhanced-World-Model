import torch

from Model import Model


class StochasticController(Model):

    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()
        self.fc_mean = torch.nn.Linear(z_dim + h_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))  # learnable std

    def forward(self, z_t, h_t):
        x = torch.cat([z_t, h_t], dim=-1)
        mu = self.fc_mean(x)  # Mean of Gaussian
        std = torch.exp(self.log_std)  # Std of Gaussian
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()  # Reparameterized sample
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions
        return action, log_prob
