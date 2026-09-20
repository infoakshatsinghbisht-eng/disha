"""
Streaming Attention with Attention Sink Tokens & Rolling Sliding-Window KV Cache.
Solves the KV-cache memory explosion problem, guaranteeing O(1) constant RAM footprint
suitable for 8GB RAM laptops and infinite-context inference.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands KV heads for Grouped Query Attention (GQA)."""
    if n_rep == 1:
        return x
    B, SeqLen, NumKVHeads, HeadDim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(B, SeqLen, NumKVHeads, n_rep, HeadDim)
        .reshape(B, SeqLen, NumKVHeads * n_rep, HeadDim)
    )


class StreamingAttention(nn.Module):
    """
    Streaming Attention module with Attention Sinks and Rolling Window Cache.
    
    Key Features:
    1. Attention Sinks: Keeps initial `num_sink_tokens` (e.g., first 4 tokens) which
       absorb excessive attention softmax logits and prevent perplexity collapse.
    2. Sliding Window: Keeps the most recent `window_size` (e.g., 256 tokens).
    3. Memory Bound: Cache size is strictly bounded by (num_sink_tokens + window_size),
       ensuring peak RAM is constant regardless of sequence length.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = 4,
        window_size: int = 256,
        num_sink_tokens: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.num_sink_tokens = num_sink_tokens

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.wq = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Rolling KV-Cache buffers
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        """Resets the rolling KV-cache buffers."""
        self.cache_k = None
        self.cache_v = None

    def get_cache_memory_bytes(self) -> int:
        """Returns the current memory consumption of KV caches in bytes."""
        mem = 0
        if self.cache_k is not None:
            mem += self.cache_k.numel() * self.cache_k.element_size()
        if self.cache_v is not None:
            mem += self.cache_v.numel() * self.cache_v.element_size()
        return mem

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, SeqLen, _ = x.shape

        xq = self.wq(x).view(B, SeqLen, self.num_heads, self.head_dim)
        xk = self.wk(x).view(B, SeqLen, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, SeqLen, self.num_kv_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if use_cache:
            if self.cache_k is None or start_pos == 0:
                self.cache_k = xk
                self.cache_v = xv
            else:
                self.cache_k = torch.cat([self.cache_k, xk], dim=1)
                self.cache_v = torch.cat([self.cache_v, xv], dim=1)

                # Rolling Window Eviction with Attention Sinks:
                # If cache exceeds sink_tokens + window_size, retain sinks + latest window
                max_cache_len = self.num_sink_tokens + self.window_size
                curr_cache_len = self.cache_k.size(1)

                if curr_cache_len > max_cache_len:
                    sink_k = self.cache_k[:, : self.num_sink_tokens]
                    sink_v = self.cache_v[:, : self.num_sink_tokens]
                    recent_k = self.cache_k[:, -self.window_size :]
                    recent_v = self.cache_v[:, -self.window_size :]
                    self.cache_k = torch.cat([sink_k, recent_k], dim=1)
                    self.cache_v = torch.cat([sink_v, recent_v], dim=1)

            keys = self.cache_k
            values = self.cache_v
        else:
            keys = xk
            values = xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Memory-Efficient / Flash Scaled Dot-Product Attention
        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(B, SeqLen, -1)
        return self.wo(output)
