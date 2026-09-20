"""
Root Mean Square Layer Normalization (RMSNorm) from scratch.
Offers computational efficiency and numerical stability used in modern LLMs (LLaMA, Mistral, Claude).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization layer.
    
    Args:
        dim (int): Model hidden dimension.
        eps (float): Small constant for numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt( mean( x^2 ) + eps )
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
