"""
SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network from scratch.
Offers superior non-linear representational capacity compared to standard GeLU/ReLU MLPs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Block.
    Computes: Output = (w1(x) * SiLU(w3(x))) @ w2
    
    Args:
        dim (int): Input and output hidden dimension.
        hidden_dim (int, optional): Intermediate projection dimension.
        multiple_of (int): Rounds hidden_dim to a multiple of this value for GPU alignment.
        ffn_dim_multiplier (float): Multiplier for hidden dimension scaling (default ~2.67).
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 64,
        ffn_dim_multiplier: float = 2.67,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim / 3) * (ffn_dim_multiplier / 2.67))
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU gating
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
