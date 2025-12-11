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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__] = cls

    @abstractmethod
    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ...]:
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

    @abstractmethod
    def evaluate_actions(
        self, z_t: torch.Tensor, h_t: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions (used in PPO).

        Args:
            z_t: Current latent state, shape (B, z_dim)
            h_t: Current hidden state, shape (B, h_dim)
            actions: Actions to evaluate, shape (B,) for discrete or (B, action_dim) for continuous

        Returns:
            log_prob: Log probability of the actions, shape (B,)
            value: State value estimate, shape (B,)
            entropy: Policy entropy, shape (B,)
        """
        pass
