import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from controller import ControllerModel


class StochasticController(ControllerModel):

    def __init__(self, z_dim, h_dim, action_dim, **_kwargs):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        self.fc_mean = torch.nn.Linear(z_dim + h_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))  # learnable std


    def forward(self, z_t, h_t):
        x = torch.cat([z_t, h_t], dim=-1)

        mu = self.fc_mean(x)
        std = torch.exp(self.log_std)
        # Create a Normal distribution and apply a Tanh transform to squash actions to [-1, 1]
        base_dist = Normal(mu, std)

        transform = TanhTransform(cache_size=1)
        action_dist = TransformedDistribution(base_dist, transform)

        # rsample() gets a sample that gradients can flow through
        action = action_dist.sample()
        # log_prob() correctly computes the log probability of the squashed action
        log_prob = action_dist.log_prob(action).sum(-1)

        return action, log_prob

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
