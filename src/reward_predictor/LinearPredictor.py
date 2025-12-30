from typing import Any, cast

import torch

from reward_predictor import RewardPredictorModel


class LinearPredictor(RewardPredictorModel):
    tags: list[str] = []

    def __init__(self, z_dim: int, h_dim: int, **_kwargs: Any):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim

        # Suspicious dim calculation. Need proper proof.
        self.linear = torch.nn.Linear(z_dim + h_dim + 1, 1)

    def forward(
        self, z_t: torch.Tensor, h_t: torch.Tensor, last_reward: torch.Tensor
    ) -> torch.Tensor:
        if last_reward.dim() == 1:
            last_reward = last_reward.unsqueeze(-1)

        x = torch.cat([z_t, h_t, last_reward], dim=-1)
        return self.linear(x)

    def export_hyperparams(self) -> dict[str, int]:
        return {
            "z_dim": self.z_dim,
            "h_dim": self.h_dim,
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
