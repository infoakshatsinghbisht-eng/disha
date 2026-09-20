"""
Multi-Head and Grouped Query Attention (GQA) with KV-Cache from scratch.
Implements causal attention masking and dynamic KV caching for autoregressive generation.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands KV heads to match Query heads when using Grouped Query Attention (GQA).
    (B, SeqLen, NumKVHeads, HeadDim) -> (B, SeqLen, NumQueryHeads, HeadDim)
    """
    if n_rep == 1:
        return x
    B, SeqLen, NumKVHeads, HeadDim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(B, SeqLen, NumKVHeads, n_rep, HeadDim)
        .reshape(B, SeqLen, NumKVHeads * n_rep, HeadDim)
    )


class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) / Multi-Head Attention module.
    
    Args:
        dim (int): Model embedding dimension.
        num_heads (int): Number of Query attention heads.
        num_kv_heads (int, optional): Number of Key/Value heads. If None, equals num_heads.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Projections
        self.wq = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # KV Cache for fast inference
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        """Clears the KV cache."""
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for causal attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, SeqLen, Dim).
            freqs_cis (torch.Tensor, optional): Precomputed RoPE frequencies.
            mask (torch.Tensor, optional): Causal attention mask.
            use_cache (bool): Whether to utilize/update KV cache.
            start_pos (int): Starting position in sequence for caching.
            
        Returns:
            torch.Tensor: Attention output of shape (B, SeqLen, Dim).
        """
        B, SeqLen, _ = x.shape

        # Linear projections
        xq = self.wq(x).view(B, SeqLen, self.num_heads, self.head_dim)
        xk = self.wk(x).view(B, SeqLen, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, SeqLen, self.num_kv_heads, self.head_dim)

        # Apply Rotary Positional Embeddings
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Update and use KV-Cache during generation
        if use_cache:
            if self.cache_k is None or start_pos == 0:
                self.cache_k = xk
                self.cache_v = xv
            else:
                self.cache_k = torch.cat([self.cache_k, xk], dim=1)
                self.cache_v = torch.cat([self.cache_v, xv], dim=1)
            keys = self.cache_k
            values = self.cache_v
        else:
            keys = xk
            values = xv

        # Repeat KV heads for GQA
        keys = repeat_kv(keys, self.n_rep)       # (B, TotalSeqLen, NumHeads, HeadDim)
        values = repeat_kv(values, self.n_rep)   # (B, TotalSeqLen, NumHeads, HeadDim)

        # Transpose for PyTorch attention computation: (B, NumHeads, SeqLen, HeadDim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled Dot-Product Attention: Q * K^T / sqrt(head_dim)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        probs = self.dropout(probs)

        # Weighted sum: Probs * V -> (B, NumHeads, SeqLen, HeadDim)
        output = torch.matmul(probs, values)
        # Reshape back to (B, SeqLen, Dim)
        output = output.transpose(1, 2).contiguous().view(B, SeqLen, -1)

        return self.wo(output)
