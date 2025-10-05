import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


from Model import Model



class StochasticController(Model):

    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()
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
        action = action_dist.rsample()

        # log_prob() correctly computes the log probability of the squashed action
        log_prob = action_dist.log_prob(action).sum(-1)

        return action, log_prob

    def export_hyperparam(self):
        return {
            "class_name": self.__class__.__name__,
        }
