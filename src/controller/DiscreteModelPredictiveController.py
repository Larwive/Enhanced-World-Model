import torch
from torch.distributions import Categorical

from controller import ControllerModel


class DiscreteModelPredictiveController(ControllerModel):
    def __init__(self, z_dim, h_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        self.policy = torch.nn.Linear(z_dim + h_dim, action_dim)
        self.value = torch.nn.Linear(h_dim, 1)

        # Temperature for exploration control (lower = more deterministic)
        # Start at 1.0 (standard), can be annealed down during training
        self.temperature = 1.0

    def set_temperature(self, temp: float):
        """Set temperature for action sampling. Lower = more deterministic."""
        self.temperature = max(0.01, temp)  # Prevent division by zero

    def forward(self, z_t, h_t):
        # z_t: (B, z_dim) and h_t: (B, h_dim)
        x = torch.cat([z_t, h_t], dim=-1)
        logits = self.policy(x)  # (B, action_dim)

        # Apply temperature scaling (lower temp = sharper distribution)
        scaled_logits = logits / self.temperature
        dist = Categorical(logits=scaled_logits)

        action = dist.sample()  # (B,)
        # Log prob computed with original logits for correct gradients
        original_dist = Categorical(logits=logits)
        logp = original_dist.log_prob(action).unsqueeze(-1)  # (B, 1)
        value = self.value(h_t)  # (B, 1)
        entropy = original_dist.entropy().unsqueeze(-1)

        return action, logp, value, entropy

    def evaluate_actions(self, z_t, h_t, actions):
        """Evaluate log probability and entropy for given actions."""
        x = torch.cat([z_t, h_t], dim=-1)
        logits = self.policy(x)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions.long())
        value = self.value(h_t).squeeze(-1)
        entropy = dist.entropy()

        return log_prob, value, entropy

    def get_deterministic_action(self, z_t, h_t):
        """Get greedy action (no sampling) for evaluation."""
        x = torch.cat([z_t, h_t], dim=-1)
        logits = self.policy(x)
        return logits.argmax(dim=-1)

    def export_hyperparams(self):
        return {"z_dim": self.z_dim, "h_dim": self.h_dim, "action_dim": self.action_dim}

    def save_state(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)
