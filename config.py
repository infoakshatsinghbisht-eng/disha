"""
Global Configuration for 2.0 Billion Parameter Multimodal Foundation LLM from Scratch
Includes configs for:
- Discrete Visual Tokenizer (VQ-VAE with 2048 Codebook and Sobel Gradient Loss)
- Text Tokenizer (Byte BPE 8K)
- 2.0 Billion Parameter Multimodal Transformer Core
- Training & Inference Pipeline
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VQVAEConfig:
    in_channels: int = 3
    hidden_dim: int = 64
    num_res_blocks: int = 2
    num_downsamples: int = 3          # 64x64 -> 8x8 (64 visual tokens)
    codebook_size: int = 2048         # High-capacity discrete codebook
    embedding_dim: int = 64           # Dimension of each codebook vector
    commitment_cost: float = 0.25     # Beta weight for commitment loss
    decay: float = 0.99               # EMA decay factor
    image_size: int = 64              # Target image resolution
    latent_grid_size: int = 8         # 8x8 = 64 visual tokens per image


@dataclass
class TokenizerConfig:
    byte_vocab_size: int = 256
    special_tokens: Tuple[str, ...] = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<image_start>",
        "<image_end>",
        "<text_start>",
        "<text_end>",
        "<unk>",
    )


@dataclass
class LLMConfig:
    # 2.0 Billion Parameter Architecture (dim=2048, layers=36, heads=32, kv_heads=8, SwiGLU multiplier=3.5)
    text_vocab_size: int = 8000       # Expanded text vocabulary
    image_vocab_size: int = 2048      # Matches VQ-VAE codebook size
    max_seq_len: int = 256            # Context window
    
    dim: int = 2048                   # Model Hidden Dimension
    num_layers: int = 36              # Number of Transformer Decoder layers (~1.98B parameters)
    num_heads: int = 32               # Number of Query attention heads
    num_kv_heads: Optional[int] = 8   # Grouped Query Attention (GQA) heads
    ffn_dim_multiplier: float = 3.5   # SwiGLU hidden dim multiplier (intermediate_dim = 7168)
    multiple_of: int = 64             # Dimension multiple
    norm_eps: float = 1e-6            # RMSNorm epsilon
    rope_theta: float = 10000.0       # Base frequency for RoPE
    dropout: float = 0.0              # Dropout rate
    
    # Multimodal Tokens
    image_token_len: int = 64         # 8x8 tokens representing one image


@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    grad_clip: float = 1.0
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class GenerationConfig:
    temperature: float = 0.75
    top_k: int = 40
    top_p: float = 0.90
    cfg_scale: float = 1.5
    repetition_penalty: float = 1.05
    max_new_tokens: int = 64
