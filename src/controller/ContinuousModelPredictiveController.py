from typing import Any, cast
import torch
from torch.distributions import Normal

from controller import ControllerModel


class ContinuousModelPredictiveController(ControllerModel):
    tags = ["continuous", "stochastic"]

    def __init__(self, z_dim: int, h_dim: int, action_dim: int) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

        hidden_dim = z_dim + h_dim

        self.mu_layer = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = torch.nn.Linear(hidden_dim, action_dim)

        self.value = torch.nn.Linear(h_dim, 1)

    def forward(
        self, z_t: torch.Tensor, h_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def export_hyperparams(self) -> dict[str, int]:
        return {"z_dim": self.z_dim, "h_dim": self.h_dim, "action_dim": self.action_dim}

    def save_state(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
