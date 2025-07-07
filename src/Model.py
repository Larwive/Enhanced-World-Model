import abc
from typing import Any, Tuple
from torch.nn import Module
import torch


class Model(Module):

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass. Must be implemented in subclasses.

        :param input: Input tensor
        :return: Tuple of tensors (e.g., reconstruction, loss)
        """
        pass

    @abc.abstractmethod
    def export_hyperparam(self) -> dict:
        """
        Export model hyperparameters for saving or logging.
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the model's weights and config.

        :param path: Path to save the model checkpoint (.pt file)
        """
        torch.save({
            "state_dict": self.state_dict(),
            "hyperparam": self.export_hyperparam(),
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "Model":
        """
        Load model from a checkpoint.

        :param path: Path to the checkpoint
        :return: An instance of the model
        """
        checkpoint = torch.load(path, map_location=kwargs.get("map_location", "cpu"))
        model = cls(**checkpoint["hyperparam"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
