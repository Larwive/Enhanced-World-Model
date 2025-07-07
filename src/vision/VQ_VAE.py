import torch
import torch.nn.functional as F

from src.Model import Model


class ResidualBlock(torch.nn.Module):
    """
    An implementation of a residual block.
    """

    def __init__(self, channels):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                         torch.nn.ReLU(), torch.nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(torch.nn.Module):
    """
    Implementation of the vector quantizer.
    """

    def __init__(self, num_embeddings: int, embed_dim: int, beta: float = 0.25):
        """
        Implements vector quantization layer (VQ) as described in the VQ-VAE paper.

        :param num_embeddings: Number of vectors in the codebook (K)
        :param embed_dim: Dimensionality of each embedding vector (D)
        :param beta: Weight for commitment loss (equation (3) in the paper)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.beta = beta

        # Codebook: (K, D)
        self.embedding = torch.nn.Parameter(torch.randn(num_embeddings, embed_dim) * 0.1)

    def forward(self, z_e):
        """
        Forward pass of vector quantizer.

        :param z_e: Encoder output of shape (B, D, ...)
        :return: quantized tensor z_q, commitment loss, and indices
        """
        # Flatten input → (B*..., D)
        z_e_flat = z_e.permute(0, *range(2, z_e.ndim), 1).contiguous()
        flat_shape = z_e_flat.shape
        z_e_flat = z_e_flat.view(-1, self.embed_dim)  # (N, D)

        # Compute distances to embeddings: (N, K)
        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) - 2 * z_e_flat @ self.embedding.t() + torch.sum(
            self.embedding**2, dim=1, keepdim=True).t()
        # Get nearest code indices (N,)
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantized vectors: (N, D)
        z_q_flat = self.embedding[encoding_indices]

        # Reshape back to original input shape
        z_q = z_q_flat.view(*flat_shape)
        z_q = z_q.permute(0, -1, *range(1, z_e.ndim - 1)).contiguous()  # (B, D, ...)

        # Straight-through estimator (copy gradients)
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss
        loss = F.mse_loss(z_e.detach(), z_q) + self.beta * F.mse_loss(z_e, z_q.detach())

        return z_q_st, loss, encoding_indices.view(z_q.shape[0], *z_q.shape[2:])


class VQ_VAE(Model):
    """
    Implementation of the VQ-VAE model (https://arxiv.org/pdf/1711.00937).
    """

    def __init__(self, input_shape, hidden_dim=256, output_dim=3, num_embed=512, embed_dim=64, kernel_size=4, stride=2):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride

        nb_channels = input_shape[0]
        assert len(input_shape) == 3, "This version supports 2D inputs only (e.g. images)"

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(nb_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),  # → 64x64
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),  # → 32x32
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            torch.nn.Conv2d(hidden_dim, embed_dim, kernel_size=1))

        self.vq = VectorQuantizer(num_embeddings=num_embed, embed_dim=embed_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            torch.nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                     padding=1),  # → 64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=kernel_size, stride=stride,
                                     padding=1),  # → 128x128
            #torch.nn.Sigmoid()  # ATTENTION: Temporaire
        )

    def forward(self, input):
        z_e = self.encoder(input)
        z_q, vq_loss, _ = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

    def export_hyperparam(self):
        return {
            "input_shape": self.input_shape,
            "hidden_dim": self.hidden_dim,
            "num_embed": self.num_embed,
            "embed_dim": self.embed_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }
