"""
Unified Multimodal Autoregressive Transformer Core.
Natively processes text and discrete visual tokens using unified vocabulary embeddings,
pre-normalized transformer blocks (RMSNorm + RoPE + GQA + SwiGLU), and causal prediction heads.
"""

from typing import Optional, Tuple, Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .attention import Attention
from .streaming_attention import StreamingAttention
from .mlp import SwiGLU


class TransformerBlock(nn.Module):
    """
    Standard Pre-Norm Transformer Decoder Block with optional Constant-Memory Streaming Attention.
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        ffn_dim_multiplier: float = 2.67,
        multiple_of: int = 64,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_streaming: bool = True,
        window_size: int = 256,
        num_sink_tokens: int = 4,
    ):
        super().__init__()
        if use_streaming:
            self.attention = StreamingAttention(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                window_size=window_size,
                num_sink_tokens=num_sink_tokens,
                dropout=dropout,
            )
        else:
            self.attention = Attention(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
            )
        self.feed_forward = SwiGLU(
            dim=dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # Pre-norm Self-Attention
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        # Pre-norm Feed-Forward
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MultimodalTransformer(nn.Module):
    """
    Multimodal Large Language Model Transformer.
    Integrates text tokens, image codebook tokens, and special control tokens into
    a single unified autoregressive language modeling space.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = 4,
        max_seq_len: int = 512,
        ffn_dim_multiplier: float = 2.67,
        multiple_of: int = 64,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len

        # Token Embedding Table (Unified Text + Image Tokens)
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_dim_multiplier=ffn_dim_multiplier,
                multiple_of=multiple_of,
                norm_eps=norm_eps,
                dropout=dropout,
                use_streaming=True,
                window_size=256,
                num_sink_tokens=4,
            )
            for _ in range(num_layers)
        ])

        # Final RMSNorm and LM Head
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying between embedding and output projection
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies
        self.rope_theta = rope_theta
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def reset_caches(self):
        """Resets KV caches across all attention layers."""
        for layer in self.layers:
            layer.attention.reset_cache()

    def get_total_kv_cache_bytes(self) -> int:
        """Returns total memory consumed by KV caches across all layers in bytes."""
        total_bytes = 0
        for layer in self.layers:
            if hasattr(layer.attention, "get_cache_memory_bytes"):
                total_bytes += layer.attention.get_cache_memory_bytes()
        return total_bytes

    def build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Constructs upper triangular causal mask filled with -inf."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or inference.
        
        Args:
            tokens (torch.Tensor): Token IDs of shape (B, SeqLen).
            targets (torch.Tensor, optional): Target token IDs for cross-entropy loss.
            start_pos (int): Starting position for RoPE and KV-cache.
            use_cache (bool): Whether to cache K/V tensors.
            
        Returns:
            logits (torch.Tensor): Output vocabulary logits of shape (B, SeqLen, VocabSize).
            loss (torch.Tensor, optional): Causal cross entropy loss if targets provided.
        """
        B, seq_len = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))

        # Dynamic RoPE frequency expansion if sequence length exceeds buffer
        needed_len = start_pos + seq_len
        if needed_len > self.freqs_cis.shape[0]:
            target_len = max(needed_len + 512, self.freqs_cis.shape[0] * 2)
            theta = getattr(self, "rope_theta", 10000.0)
            self.freqs_cis = precompute_freqs_cis(self.head_dim, target_len, theta).to(tokens.device)

        # Retrieve RoPE frequencies for the sequence slice
        freqs_cis = self.freqs_cis[start_pos : needed_len].to(tokens.device)

        # Causal mask only needed when not step-by-step single token decoding
        mask = None
        if seq_len > 1:
            mask = self.build_causal_mask(seq_len, tokens.device)

        for layer in self.layers:
            if self.training and targets is not None and not use_cache:
                def make_custom_forward(mod):
                    def custom_forward(hidden, freqs, attn_mask):
                        return mod(hidden, freqs_cis=freqs, mask=attn_mask, use_cache=False, start_pos=0)
                    return custom_forward

                h = torch.utils.checkpoint.checkpoint(
                    make_custom_forward(layer),
                    h, freqs_cis, mask,
                    use_reentrant=False,
                )
            else:
                h = layer(
                    h,
                    freqs_cis=freqs_cis,
                    mask=mask,
                    use_cache=use_cache,
                    start_pos=start_pos,
                )

        h = self.norm(h)
        logits = self.output(h)  # (B, SeqLen, VocabSize)

        loss = None
        if targets is not None:
            # Shift tokens for next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.85,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.0,
        consecutive_penalty: float = 0.0,
        eos_token_id: Optional[int] = None,
        image_end_token_id: Optional[int] = None,
        allowed_token_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching, temperature, and top-p (nucleus) filtering.
        
        Args:
            prompt_tokens (torch.Tensor): Prompt token IDs (B, PromptLen).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Softmax sampling temperature.
            top_k (int): Top-K truncation parameter.
            top_p (float): Nucleus Top-P cumulative probability threshold.
            repetition_penalty (float): Repetition penalty factor (>1.0 reduces token repetition).
            consecutive_penalty (float): Penalty subtracted from logits of recently repeated tokens.
            eos_token_id (int, optional): Stop generation on this token.
            image_end_token_id (int, optional): Stop generation when image completes.
            allowed_token_range (tuple, optional): (min_id, max_id) to strictly constrain sampling.
            
        Returns:
            torch.Tensor: Full token sequence (B, PromptLen + GeneratedLen).
        """
        self.reset_caches()
        curr_tokens = prompt_tokens
        B, prompt_len = prompt_tokens.shape

        # Initial forward pass to populate KV cache for prompt
        logits, _ = self.forward(curr_tokens, use_cache=True, start_pos=0)
        next_logits = logits[:, -1, :]  # (B, VocabSize)

        generated = []

        for step in range(max_new_tokens):
            cur_logits = next_logits.clone()
            if allowed_token_range is not None:
                min_id, max_id = allowed_token_range
                cur_logits[:, :min_id] = -float("Inf")
                cur_logits[:, max_id:] = -float("Inf")

            # Apply repetition penalty to prevent token repetition loops
            if repetition_penalty != 1.0 and generated:
                prev_tokens = torch.cat(generated, dim=1)  # (B, step)
                for b in range(B):
                    unique_toks = torch.unique(prev_tokens[b])
                    for tok_id in unique_toks:
                        val = cur_logits[b, tok_id]
                        if val > 0:
                            cur_logits[b, tok_id] = val / repetition_penalty
                        else:
                            cur_logits[b, tok_id] = val * repetition_penalty

            # Suppress identical consecutive token repeats if consecutive_penalty > 0
            if consecutive_penalty > 0.0 and len(generated) >= 2:
                last_tok = generated[-1]
                second_last_tok = generated[-2]
                for b in range(B):
                    if last_tok[b, 0] == second_last_tok[b, 0]:
                        cur_logits[b, last_tok[b, 0]] -= consecutive_penalty

            if temperature > 0:
                scaled_logits = cur_logits / temperature

                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = torch.argmax(cur_logits, dim=-1, keepdim=True)  # (B, 1)

            generated.append(next_token)

            # Check termination
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if image_end_token_id is not None and (next_token == image_end_token_id).all():
                break

            # Forward pass for single step with KV cache
            logits, _ = self.forward(
                next_token,
                use_cache=True,
                start_pos=prompt_len + step,
            )
            next_logits = logits[:, -1, :]

        self.reset_caches()
        if generated:
            gen_tensor = torch.cat(generated, dim=1)
            return torch.cat([prompt_tokens, gen_tensor], dim=1)
        return prompt_tokens
