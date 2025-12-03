import torch
import torch.nn as nn
from torch.distributions import Normal

from Model import Model


class ImprovedContinuousController(Model):
    """
    Improved continuous controller with:
    1. Multi-layer MLP (better representation)
    2. Model-predictive planning using the world model
    3. Proper value function for A2C
    """

    def __init__(self, z_dim, h_dim, action_dim, hidden_dims=[128, 64],
                 use_planning=True, planning_horizon=5, num_action_samples=8):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.use_planning = use_planning
        self.planning_horizon = planning_horizon
        self.num_action_samples = num_action_samples

        # Build policy network (MLP with layer norm)
        input_dim = z_dim + h_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.shared_policy = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

        # Value network (uses temporal context h_t)
        value_layers = []
        prev_dim = h_dim
        for hidden_dim in hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value = nn.Sequential(*value_layers)

        # Planning value head (estimates value of imagined rollouts)
        if use_planning:
            self.planning_value = nn.Sequential(
                nn.Linear(z_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], 1)
            )

    def forward(self, z_t, h_t, memory_model=None, deterministic=False):
        """
        Args:
            z_t: (B, z_dim) - current latent state
            h_t: (B, h_dim) - memory hidden state
            memory_model: TemporalTransformer instance for planning
            deterministic: if True, return mean action

        Returns:
            action: (B, action_dim) - sampled actions (tanh squashed)
            logp: (B, 1) - log probabilities
            value: (B, 1) - state value estimate
            entropy: (B, 1) - policy entropy
        """
        x = torch.cat([z_t, h_t], dim=-1)
        features = self.shared_policy(x)

        mu = self.mu_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -10, 2)
        std = torch.exp(log_std)

        # Apply planning if enabled and components provided
        if self.use_planning and memory_model is not None and self.training:
            # Do model-predictive planning to refine mean action
            planning_mu = self._plan_actions(z_t, h_t, memory_model, mu)
            # Blend: 70% base policy + 30% planning
            mu = 0.7 * mu + 0.3 * planning_mu

        dist = Normal(mu, std)

        if deterministic:
            raw_action = mu
        else:
            raw_action = dist.rsample()

        action = torch.tanh(raw_action)

        # Correct log prob for tanh squashing
        logp = dist.log_prob(raw_action).sum(-1, keepdim=True)
        # Tanh correction: log_prob(tanh(x)) = log_prob(x) - log(1 - tanh(x)^2)
        logp = logp - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        value = self.value(h_t)
        entropy = dist.entropy().sum(-1, keepdim=True)

        return action, logp, value, entropy

    def _plan_actions(self, z_t, h_t, memory_model, base_mu):
        """
        Perform short-horizon planning using the world model.
        Uses CEM (Cross-Entropy Method) to find good actions.

        Returns:
            mu: (B, action_dim) - refined action mean based on planning
        """
        B = z_t.shape[0]
        device = z_t.device

        # Sample candidate action sequences
        candidate_actions = torch.randn(
            B, self.num_action_samples, self.action_dim, device=device
        ) * 0.5 + base_mu.unsqueeze(1)
        candidate_actions = torch.tanh(candidate_actions)  # Squash to [-1, 1]

        # Evaluate each candidate sequence
        action_values = torch.zeros(B, self.num_action_samples, device=device)

        for i in range(self.num_action_samples):
            action = candidate_actions[:, i]  # (B, action_dim)

            # Simulate forward
            z_curr = z_t
            h_curr = h_t
            total_value = 0.0
            gamma = 0.95

            for step in range(self.planning_horizon):
                # Predict next state
                z_next = memory_model.predict_next(z_curr, action, h_curr)

                # Estimate value of next state using planning value head
                # Note: reward_predictor is designed for (z, h, last_reward), not actions
                step_value = self.planning_value(z_next).squeeze(-1)

                total_value = total_value + (gamma ** step) * step_value

                # Update for next iteration
                z_curr = z_next.detach()
                # For simplicity, keep h_curr constant (or could update via memory)

            action_values[:, i] = total_value

        # Select best actions (top-k=1 for now, but could use weighted average)
        best_idx = torch.argmax(action_values, dim=1)  # (B,)
        best_actions = candidate_actions[torch.arange(B, device=device), best_idx]  # (B, action_dim)

        return best_actions

    def export_hyperparams(self):
        return {
            "z_dim": self.z_dim,
            "h_dim": self.h_dim,
            "action_dim": self.action_dim,
            "use_planning": self.use_planning,
            "planning_horizon": self.planning_horizon,
            "num_action_samples": self.num_action_samples
        }

    def save_state(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)
