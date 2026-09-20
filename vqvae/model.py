"""
Complete VQ-VAE (Vector Quantized Variational Autoencoder) Architecture.
Discretizes continuous RGB images into 2D discrete token grids and decodes them back to high-fidelity images.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock, Downsample2d, Upsample2d
from .quantizer import VectorQuantizer


class Encoder(nn.Module):
    """
    Hierarchical Convolutional Encoder.
    Downsamples an image (e.g. 256x256x3) by factor of 2^num_downsamples (e.g. 16x16)
    into a continuous feature map of dimension embedding_dim.
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_res_blocks: int = 2,
        num_downsamples: int = 4,
    ):
        super().__init__()
        
        # Initial input projection
        layers = [
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        ]
        
        # Downsampling hierarchy
        curr_dim = hidden_dim
        for i in range(num_downsamples):
            next_dim = min(curr_dim * 2, hidden_dim * 4) if i < 2 else curr_dim
            layers.append(Downsample2d(curr_dim, next_dim))
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(next_dim, next_dim))
            curr_dim = next_dim
            
        # Final residual processing and projection to embedding dim
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(curr_dim, curr_dim))
            
        layers.append(nn.GroupNorm(min(32, curr_dim), curr_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(curr_dim, embedding_dim, kernel_size=1, stride=1, padding=0))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Hierarchical Convolutional Decoder.
    Upsamples a discrete latent feature map (e.g. 16x16xembedding_dim)
    back to the original image resolution (e.g. 256x256x3).
    """
    def __init__(
        self,
        out_channels: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_res_blocks: int = 2,
        num_upsamples: int = 4,
    ):
        super().__init__()
        
        # Initial projection from codebook embedding dimension
        curr_dim = hidden_dim * 2 if num_upsamples >= 2 else hidden_dim
        layers = [
            nn.Conv2d(embedding_dim, curr_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        ]
        
        # Initial residual processing
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(curr_dim, curr_dim))
            
        # Upsampling hierarchy
        for i in range(num_upsamples):
            next_dim = max(curr_dim // 2, hidden_dim) if i >= num_upsamples - 2 else curr_dim
            layers.append(Upsample2d(curr_dim, next_dim))
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(next_dim, next_dim))
            curr_dim = next_dim
            
        # Output projection to RGB channels [-1, 1]
        layers.append(nn.GroupNorm(min(32, curr_dim), curr_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(curr_dim, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())  # Normalized image output in range [-1, 1]
        
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VQVAE(nn.Module):
    """
    Full VQ-VAE model bridging image space and discrete token space.
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        codebook_size: int = 4096,
        num_res_blocks: int = 2,
        num_downsamples: int = 4,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            num_downsamples=num_downsamples,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            num_upsamples=num_downsamples,
        )
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_downsamples = num_downsamples

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes raw image x into quantized features, vq_loss, and discrete token indices."""
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decodes continuous quantized feature map back into RGB image."""
        return self.decoder(z_q)

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts image tensor (B, 3, H, W) to flat token indices (B, N).
        E.g. 256x256 image -> (B, 256) visual tokens.
        """
        try:
            device = next(self.parameters()).device
            if x.device != device:
                x = x.to(device)
        except StopIteration:
            pass
        _, _, indices = self.encode(x)
        return indices.view(x.size(0), -1)

    def decode_from_indices(self, indices: torch.Tensor, grid_size: Optional[int] = None) -> torch.Tensor:
        """
        Converts flat token indices (B, N) or (B, H, W) into an RGB image (B, 3, H, W).
        """
        try:
            device = next(self.parameters()).device
            if indices.device != device:
                indices = indices.to(device)
        except StopIteration:
            pass
        if indices.dim() == 2:
            B, N = indices.shape
            H = int(N ** 0.5) if grid_size is None else grid_size
            indices = indices.view(B, H, H)
        z_q = self.quantizer.get_codebook_entry(indices)
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass for training.
        Returns:
            recon_x (torch.Tensor): Reconstructed image in [-1, 1].
            total_loss (torch.Tensor): Reconstruction Loss (MSE/L1) + VQ commitment Loss.
            metrics (dict): Breakdown of individual loss components.
        """
        z_q, vq_loss, indices = self.encode(x)
        recon_x = self.decode(z_q)
        
        # 1. Pixel Reconstruction Loss (L1 + MSE)
        l1_loss = F.l1_loss(recon_x, x)
        mse_loss = F.mse_loss(recon_x, x)
        
        # 2. Spatial Edge Gradient Loss (forces sharp borders, contours, and geometric shapes)
        dx_recon = recon_x[:, :, :, 1:] - recon_x[:, :, :, :-1]
        dx_target = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy_recon = recon_x[:, :, 1:, :] - recon_x[:, :, :-1, :]
        dy_target = x[:, :, 1:, :] - x[:, :, :-1, :]
        edge_loss = F.l1_loss(dx_recon, dx_target) + F.l1_loss(dy_recon, dy_target)

        recon_loss = l1_loss * 1.5 + mse_loss * 0.5 + edge_loss * 1.0
        total_loss = recon_loss + vq_loss
        
        metrics = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "edge_loss": edge_loss.detach(),
            "vq_loss": vq_loss.detach(),
        }
        return recon_x, total_loss, metrics
