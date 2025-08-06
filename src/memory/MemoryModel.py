import torch


def flatten_vision_latents(z: torch.Tensor) -> torch.Tensor:
    # z: (B, D, H, W) → (B, T, D)
    B, D, H, W = z.shape
    return z.permute(0, 2, 3, 1).reshape(B, H * W, D)




class MemoryModel:
    """
    The base class for memory (M) models.
    """
    def __init__(self) -> None:
        pass
