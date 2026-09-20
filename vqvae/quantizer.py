"""
Vector Quantization (VQ) module with Straight-Through Estimator (STE).
Quantizes continuous latent feature maps into discrete codebook indices.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantization with codebook loss and commitment loss.
    
    Args:
        num_embeddings (int): Size of the codebook dictionary (e.g. 4096).
        embedding_dim (int): Dimensionality of each codebook vector (e.g. 64).
        commitment_cost (float): Beta weight for the commitment loss.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook weights initialized uniformly
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Vector Quantization.
        
        Args:
            inputs (torch.Tensor): Continuous latent features of shape (B, C, H, W) where C == embedding_dim.
            
        Returns:
            quantized (torch.Tensor): Quantized features with straight-through gradient of shape (B, C, H, W).
            loss (torch.Tensor): Scalar VQ loss (codebook loss + commitment loss).
            encoding_indices (torch.Tensor): Discrete codebook token indices of shape (B, H, W) or (B, H*W).
        """
        # Convert (B, C, H, W) -> (B, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self.embedding_dim)  # (B*H*W, D)

        # L2-Normalized Cosine Distance: prevents codebook collapse and variance drift
        flat_input_norm = F.normalize(flat_input, dim=-1)
        embed_norm = F.normalize(self.embedding.weight, dim=-1)
        distances = 2.0 - 2.0 * torch.matmul(flat_input_norm, embed_norm.t())

        # Get nearest codebook index for each latent vector
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        quantized_flat = self.embedding(encoding_indices)  # (B*H*W, D)
        quantized = quantized_flat.view(input_shape)       # (B, H, W, D)

        # Compute Losses:
        # 1. Codebook Loss: moves codebook embeddings towards encoder outputs
        loss_codebook = F.mse_loss(quantized, inputs_permuted.detach())
        # 2. Commitment Loss: prevents encoder outputs from growing uncontrollably
        loss_commitment = F.mse_loss(quantized.detach(), inputs_permuted)
        
        vq_loss = loss_codebook + self.commitment_cost * loss_commitment

        # Straight-Through Estimator: copies gradients from quantized to inputs_permuted
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()

        # Convert back to (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Reshape indices to (B, H, W)
        batch_size, height, width, _ = input_shape
        indices = encoding_indices.view(batch_size, height, width)

        return quantized, vq_loss, indices

    def get_codebook_entry(self, indices: torch.Tensor, shape: Tuple[int, int, int, int] = None) -> torch.Tensor:
        """
        Retrieves quantized feature map directly from discrete indices.
        
        Args:
            indices (torch.Tensor): Token indices of shape (B, H, W) or (B, N).
            shape (Tuple, optional): Target feature shape (B, C, H, W).
            
        Returns:
            quantized (torch.Tensor): Feature tensor of shape (B, C, H, W).
        """
        # (B, H, W, D) or (B, N, D)
        quantized = self.embedding(indices)
        
        if shape is not None:
            B, C, H, W = shape
            quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            if quantized.dim() == 4:
                quantized = quantized.permute(0, 3, 1, 2).contiguous()
            elif quantized.dim() == 3:
                # Assuming square grid
                B, N, D = quantized.shape
                H = int(N ** 0.5)
                quantized = quantized.view(B, H, H, D).permute(0, 3, 1, 2).contiguous()
                
        return quantized
