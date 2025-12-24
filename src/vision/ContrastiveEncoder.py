"""
Contrastive Vision Encoder using SimCLR-style contrastive learning.

A simpler alternative to JEPA that learns representations by:
- Maximizing agreement between augmented views of the same image
- NO reconstruction loss - learns semantic features only

Reference: SimCLR (Chen et al., 2020) - https://arxiv.org/abs/2002.05709
"""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision import VisionModel


class ResidualBlock(nn.Module):
    """Residual block for encoder architecture."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ContrastiveEncoder(VisionModel):
    """
    Contrastive Vision Encoder (SimCLR-style).

    Architecture:
        - Encoder: CNN backbone producing spatial features
        - Projector: MLP head for contrastive learning (only used during training)

    Training:
        - Apply random augmentations to create two views of each image
        - Encode both views with shared encoder
        - Project to contrastive space
        - NT-Xent loss encourages matching views, repels non-matching

    Benefits:
        - Simpler than JEPA (no EMA, no predictor)
        - Learns semantic features invariant to augmentations
        - No reconstruction needed
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        embed_dim: int = 64,
        projector_dim: int = 128,
        kernel_size: int = 4,
        stride: int = 2,
        temperature: float = 0.1,
        augment_strength: float = 0.5,
    ) -> None:
        """
        Initialize Contrastive Encoder.

        Args:
            input_shape: Shape of input images (C, H, W)
            hidden_dim: Hidden dimension in encoder
            embed_dim: Latent embedding dimension (used downstream)
            projector_dim: Dimension of projection head (only for contrastive loss)
            kernel_size: Convolution kernel size
            stride: Convolution stride
            temperature: Temperature for NT-Xent loss (lower = harder negatives)
            augment_strength: Strength of random augmentations (0-1)
        """
        super().__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.projector_dim = projector_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.temperature = temperature
        self.augment_strength = augment_strength

        nb_channels = input_shape[0]
        assert len(input_shape) == 3, "ContrastiveEncoder supports 2D inputs only (images)"

        # Encoder backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(nb_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

        # Projection head (only used during training)
        # Maps from embed_dim to projector_dim for contrastive loss
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projector_dim),
        )

    def _random_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations for contrastive learning.

        Augmentations (applied with probability based on augment_strength):
        - Random horizontal flip
        - Random brightness/contrast adjustment
        - Random Gaussian noise
        """
        batch_size = x.shape[0]
        device = x.device

        # Random horizontal flip
        if torch.rand(1).item() < self.augment_strength:
            flip_mask = torch.rand(batch_size, device=device) < 0.5
            x = torch.where(flip_mask.view(-1, 1, 1, 1), x.flip(-1), x)

        # Random brightness adjustment
        if torch.rand(1).item() < self.augment_strength:
            brightness = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5) * 0.4
            x = x * brightness

        # Random contrast adjustment
        if torch.rand(1).item() < self.augment_strength:
            contrast = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5) * 0.4
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * contrast + mean

        # Random Gaussian noise
        if torch.rand(1).item() < self.augment_strength:
            noise_scale = 0.05 * self.augment_strength
            noise = torch.randn_like(x) * noise_scale
            x = x + noise

        return x.clamp(0, 1)

    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        This is the core contrastive loss from SimCLR.

        Args:
            z1: Projections from view 1 (B, projector_dim)
            z2: Projections from view 2 (B, projector_dim)

        Returns:
            loss: NT-Xent loss (scalar)
        """
        batch_size = z1.shape[0]

        # L2 normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate projections: [z1_0, z1_1, ..., z2_0, z2_1, ...]
        z = torch.cat([z1, z2], dim=0)  # (2B, projector_dim)

        # Compute similarity matrix
        sim = z @ z.T / self.temperature  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i) for each i in [0, B)
        # Labels indicate which column is the positive for each row
        labels = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)],
            dim=0,
        ).to(z.device)

        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)

        return loss

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Creates two augmented views, encodes both, and computes contrastive loss.

        Args:
            input: Input images (B, C, H, W)

        Returns:
            z: Encoded representation (for compatibility, not a reconstruction)
            loss: NT-Xent contrastive loss
        """
        if self.training:
            # Create two augmented views
            view1 = self._random_augment(input)
            view2 = self._random_augment(input)

            # Encode both views
            z1_spatial = self.encoder(view1)  # (B, embed_dim, H', W')
            z2_spatial = self.encoder(view2)

            # Global average pool for projection head
            z1 = z1_spatial.mean(dim=(2, 3))  # (B, embed_dim)
            z2 = z2_spatial.mean(dim=(2, 3))

            # Project to contrastive space
            p1 = self.projector(z1)  # (B, projector_dim)
            p2 = self.projector(z2)

            # Compute contrastive loss
            loss = self._nt_xent_loss(p1, p2)

            # Return encoded view1 as "reconstruction" for interface compatibility
            return z1_spatial, loss
        else:
            # During evaluation, just encode (no augmentation, no loss)
            z = self.encoder(input)
            return z, torch.tensor(0.0, device=input.device)

    def encode(self, input: torch.Tensor, is_image_based: bool) -> torch.Tensor:
        """
        Encode observation to latent representation.

        Args:
            input: Input images (B, C, H, W)
            is_image_based: Whether to preserve spatial dimensions

        Returns:
            latent: Encoded representation
                - If is_image_based=True: (B, embed_dim, H', W')
                - If is_image_based=False: (B, embed_dim)
        """
        z = self.encoder(input)
        if not is_image_based:
            z = z.mean(dim=(2, 3))
        return z

    def export_hyperparams(self) -> dict[str, Any]:
        """Export hyperparameters for checkpoint saving."""
        return {
            "input_shape": self.input_shape,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "projector_dim": self.projector_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "temperature": self.temperature,
            "augment_strength": self.augment_strength,
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        """Save model state for checkpointing."""
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model state from checkpoint."""
        self.load_state_dict(state_dict)
