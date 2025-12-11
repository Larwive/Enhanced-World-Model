from abc import abstractmethod

import torch

from Model import Model

REGISTRY = {}


class ControllerModel(Model):
    """
    Base class for controller (C) models.

    Responsible for action selection and policy learning.

    Attributes:
        z_dim: Dimension of latent state (from vision)
        h_dim: Dimension of hidden state (from memory)
        action_dim: Dimension of action space
    """

    z_dim: int
    h_dim: int
    action_dim: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__] = cls

    @abstractmethod
    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Compute action and policy information.

        Args:
            z_t: Current latent state, shape (B, z_dim)
            h_t: Current hidden state, shape (B, h_dim)

        Returns:
            For discrete actions: (action, log_prob, value, entropy)
                - action: Selected actions, shape (B,)
                - log_prob: Log probability of actions, shape (B, 1)
                - value: State value estimate, shape (B, 1)
                - entropy: Policy entropy, shape (B, 1)

            For continuous actions: (action, log_prob, value, entropy)
                - action: Selected actions, shape (B, action_dim)
                - log_prob: Log probability of actions, shape (B, 1)
                - value: State value estimate, shape (B, 1)
                - entropy: Policy entropy, shape (B, 1)
        """
        pass
