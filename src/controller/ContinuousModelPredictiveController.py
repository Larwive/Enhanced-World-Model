import torch
from torch.distributions import Normal

from controller import ControllerModel


class ContinuousModelPredictiveController(ControllerModel):
    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        hidden_dim = z_dim + h_dim

        self.mu_layer = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = torch.nn.Linear(hidden_dim, action_dim)

        self.value = torch.nn.Linear(h_dim, 1)

    def forward(self, z_t, h_t):
        x = torch.cat([z_t, h_t], dim=-1)  # (B, z+h)

        mu = self.mu_layer(x)  # (B, action_dim)
        log_std = self.log_std_layer(x)  # (B, action_dim)
        log_std = torch.clamp(log_std, -10, 2)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        raw_a = dist.rsample()

        action = torch.tanh(raw_a)

        logp = dist.log_prob(raw_a).sum(-1, keepdim=True)
        value = self.value(h_t)  # (B,1)
        entropy = dist.entropy().unsqueeze(-1)

        return action, logp, value, entropy

    def evaluate_actions(self, z_t, h_t, actions):
        """
        Re-evaluate actions for PPO update.
        Args:
            z_t: latent state (B, z_dim)
            h_t: hidden state (B, h_dim)
            actions: actions taken, already in [-1, 1] from tanh (B, action_dim)
        Returns:
            log_probs: log probability of actions (B,)
            values: value estimates (B,)
            entropy: entropy of the distribution (B,)
        """
        x = torch.cat([z_t, h_t], dim=-1)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -10, 2)
        std = torch.exp(log_std)

        dist = Normal(mu, std)

        # Inverse tanh to get raw actions (atanh = 0.5 * log((1+x)/(1-x)))
        # Clamp to avoid numerical issues at boundaries
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        raw_actions = torch.atanh(actions_clamped)

        # Log prob with tanh correction
        log_probs = dist.log_prob(raw_actions).sum(-1)
        # Tanh squashing correction: subtract log(1 - tanh^2(a))
        log_probs = log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)

        values = self.value(h_t).squeeze(-1)
        entropy = dist.entropy().sum(-1)

        return log_probs, values, entropy

    def export_hyperparams(self):
        return {"z_dim": self.z_dim, "h_dim": self.h_dim, "action_dim": self.action_dim}

    def save_state(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)
