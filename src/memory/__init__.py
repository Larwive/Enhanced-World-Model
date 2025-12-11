from abc import abstractmethod

import torch

from Model import Model

REGISTRY = {}


class MemoryModel(Model):
    """
    Base class for memory (M) models.

    Responsible for temporal modeling and next-state prediction.

    Attributes:
        input_dim: Dimension of latent input (from vision)
        output_dim: Dimension of hidden state (d_model)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__] = cls

    @abstractmethod
    def forward(
        self, z: torch.Tensor, h_prev: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: update memory and predict next state.

        Args:
            z: Current latent state z_t, shape (B, input_dim)
            h_prev: Previous hidden state h_{t-1}, shape (B, output_dim)
            **kwargs: Must include 'action' (previous/current), may include others

        Returns:
            x_next: Predicted next latent, shape (B, input_dim)
            h: Hidden state, shape (B, output_dim)
        """
        pass

    @abstractmethod
    def update(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Update internal memory with new observation.

        Args:
            x: Current latent state z_t, shape (B, input_dim)
            **kwargs: Must include 'action' (previous action)

        Returns:
            h: Updated hidden state, shape (B, output_dim)
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predict next latent state.

        Args:
            x: Current latent state z_t, shape (B, input_dim)
            h: Current hidden state, shape (B, output_dim)
            **kwargs: Must include 'action' (current action)

        Returns:
            x_next: Predicted next latent, shape (B, input_dim)
        """
        pass

    @abstractmethod
    def reset(self, batch_idx: int | None = None) -> None:
        """Reset memory buffer (for specific env or all)."""
        pass
