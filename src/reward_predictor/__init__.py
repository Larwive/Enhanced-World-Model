from abc import abstractmethod
from typing import Any
import torch

from Model import Model

REGISTRY = {}


class RewardPredictorModel(Model):
    """
    Base class for reward predictor (R) models.

    Responsible for predicting future rewards from latent and hidden states.

    Attributes:
        z_dim: Dimension of latent state (from vision)
        h_dim: Dimension of hidden state (from memory)
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__] = cls

    @abstractmethod
    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        last_reward: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Predict next reward from current state.

        Args:
            z_t: Current latent state, shape (B, z_dim)
            h_t: Current hidden state, shape (B, h_dim)
            last_reward: Previous reward, shape (B,) or (B, 1)

        Returns:
            predicted_reward: Predicted reward for next timestep, shape (B, 1)
        """
        pass

    @abstractmethod
    def save_state(self) -> dict[str, torch.Tensor]:
        """
        Save model state dict to file.

        Args:
            path: Path to save state dict to
        """
        pass
