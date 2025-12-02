import torch
import torch.nn as nn
from torch.distributions import Categorical

from Model import Model


class ImprovedDiscreteController(Model):
    """
    Improved discrete controller with:
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
        print(f"[DEBUG ImprovedDiscreteController.__init__] z_dim={z_dim}, h_dim={h_dim}, input_dim={input_dim}, action_dim={action_dim}")
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy = nn.Sequential(*layers)

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

    def forward(self, z_t, h_t, memory_model=None, reward_predictor=None, deterministic=False):
        """
        Args:
            z_t: (B, z_dim) - current latent state
            h_t: (B, h_dim) - memory hidden state
            memory_model: TemporalTransformer instance for planning
            reward_predictor: optional reward predictor for planning
            deterministic: if True, take argmax instead of sampling

        Returns:
            action: (B,) - sampled actions
            logp: (B, 1) - log probabilities
            value: (B, 1) - state value estimate
            entropy: (B, 1) - policy entropy
        """
        # Get base policy
        print(f"[DEBUG ImprovedDiscreteController.forward] z_t.shape={z_t.shape}, h_t.shape={h_t.shape}")
        x = torch.cat([z_t, h_t], dim=-1)
        print(f"[DEBUG ImprovedDiscreteController.forward] x.shape after cat={x.shape}, expected={self.z_dim + self.h_dim}")
        logits = self.policy(x)  # (B, action_dim)

        # Apply planning if enabled and components provided
        if self.use_planning and memory_model is not None and self.training:
            # Do model-predictive planning to refine action distribution
            planning_logits = self._plan_actions(z_t, h_t, memory_model, reward_predictor)
            # Blend: 70% base policy + 30% planning
            logits = 0.7 * logits + 0.3 * planning_logits

        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        logp = dist.log_prob(action).unsqueeze(-1)
        value = self.value(h_t)
        entropy = dist.entropy().unsqueeze(-1)

        return action, logp, value, entropy

    def _plan_actions(self, z_t, h_t, memory_model, reward_predictor):
        """
        Perform short-horizon planning using the world model.

        Returns:
            logits: (B, action_dim) - refined action preferences based on planning
        """
        B = z_t.shape[0]
        device = z_t.device

        # For each possible action, do a rollout and estimate return
        action_values = torch.zeros(B, self.action_dim, device=device)

        for a in range(self.action_dim):
            # Create one-hot action
            action_onehot = torch.zeros(B, self.action_dim, device=device)
            action_onehot[:, a] = 1.0

            # Simulate forward
            z_curr = z_t
            h_curr = h_t
            total_value = 0.0
            gamma = 0.95  # discount factor

            for step in range(self.planning_horizon):
                # Predict next state
                z_next = memory_model.predict_next(z_curr, action_onehot, h_curr)

                # Estimate reward (if predictor available, else use value)
                if reward_predictor is not None:
                    # Reward predictor should take (z, h, action) and return reward
                    step_value = reward_predictor(z_curr, h_curr, action_onehot).squeeze(-1)
                else:
                    # Use value function as proxy
                    step_value = self.planning_value(z_next).squeeze(-1)

                total_value = total_value + (gamma ** step) * step_value

                # Update for next iteration (detach to prevent long backprop)
                z_curr = z_next.detach()
                # For planning, we'd need to update h_curr too, but for short horizons
                # we can approximate by keeping it constant

            action_values[:, a] = total_value

        # Convert values to logits (with temperature)
        temperature = 2.0  # Higher = softer planning signal
        planning_logits = action_values / temperature

        return planning_logits

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
