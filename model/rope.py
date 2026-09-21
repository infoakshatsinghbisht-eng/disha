"""
Rotary Positional Embeddings (RoPE) implementation from scratch.
Applies rotation matrices to query and key vectors for relative positional awareness.
"""

from typing import Tuple
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes complex exponential frequency tensor cis(m * theta_i) for RoPE.
    
    Args:
        dim (int): Head dimension (must be even).
        end (int): Maximum sequence length to precompute.
        theta (float): Base frequency scaling constant.
        
    Returns:
        freqs_cis (torch.Tensor): Tensor of complex frequencies of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64: e^(i * freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    
    Args:
        xq: Query tensor of shape (B, SeqLen, NumHeads, HeadDim)
        xk: Key tensor of shape (B, SeqLen, NumKVHeads, HeadDim)
        freqs_cis: Precomputed frequencies of shape (SeqLen, HeadDim // 2)
        
    Returns:
        xq_out, xk_out with rotary embeddings applied.
    """
    # Reshape to complex representation (B, SeqLen, NumHeads, HeadDim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis: (1, SeqLen, 1, HeadDim // 2)
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))
    
    # Complex multiplication rotates the vectors
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """
    Module wrapping precomputed RoPE frequencies and dynamic caching.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0,
        seq_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = xq.shape[1]
        needed_len = start_pos + seq_len
        if needed_len > self.freqs_cis.shape[0]:
            target_len = max(needed_len + 512, self.freqs_cis.shape[0] * 2)
            self.freqs_cis = precompute_freqs_cis(self.dim, target_len, self.theta).to(xq.device)
        freqs_cis = self.freqs_cis[start_pos : needed_len].to(xq.device)
        return apply_rotary_emb(xq, xk, freqs_cis)
