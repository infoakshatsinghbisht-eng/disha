# 🚀 Disha Multimodal Foundation Model: Dual-Engine Studio

A state-of-the-art dual-engine multimodal AI platform providing **both** a 100% custom from-scratch PyTorch architecture and an Ultra-HD 1024x1024 LoRA fine-tuning pipeline.

---

## 🏛️ Dual-Engine Architecture

| Feature | **Engine 1: Native Scratch Model** | **Engine 2: Disha Ultra-HD LoRA** |
| :--- | :--- | :--- |
| **Paradigm** | 100% Scratch PyTorch Foundation Model | Fine-Tuned Custom LoRA Adapter |
| **Image Resolution** | $256 \times 256$ Crisp Native | $1024 \times 1024$ Commercial Studio Grade |
| **Visual Encoding** | Hierarchical 4-stage VQ-VAE ($16 \times 16 = 256$ tokens) | Latent Diffusion VAE |
| **Core Architecture** | RoPE + FlashAttention + GQA + SwiGLU Transformer | Pre-trained Diffusion Backbone + Custom LoRA (~80MB) |
| **Inference Time** | Autoregressive Next-Token Sampling | 1-2 Step Turbo Sampling (~1.5s on GPU) |
| **Execution Script** | `python generate.py --prompt "..."` | `python generate_hd.py --prompt "..." --lora ...` |
| **Training Script** | `python train_2b_master.py` | `python train_disha_full.py` |

---

## 🌟 Key Architecture Innovations (Engine 1: Scratch Core)

1. **Discrete Visual Tokenizer (VQ-VAE with Codebook Quantization)**
   - Maps continuous $256 \times 256 \times 3$ images into a discrete grid of $16 \times 16 = 256$ visual tokens.
   - Vector Quantizer with Euclidean distance, Straight-Through Estimator (STE), and Codebook + Commitment Loss ($\beta = 0.25$).
   - Convolutional Hierarchical Decoder reconstructs RGB images from codebook token sequences.

2. **Custom Byte-Level & BPE Tokenizer**
   - Zero-dependency UTF-8 byte tokenizer with subword BPE pair merging algorithm from scratch.
   - Special control tokens: `<bos>`, `<eos>`, `<pad>`, `<image_start>`, `<image_end>`, `<text_start>`, `<text_end>`, `<unk>`.

3. **Unified Multimodal Transformer Core**
   - **Unified Embedding Table**: $V_{total} = V_{text} + V_{image} + V_{special}$.
   - **RMSNorm**: Root Mean Square Layer Normalization for stability.
   - **RoPE (Rotary Position Embeddings)**: Dynamic complex exponential rotary position embeddings ($cis(m \theta_i)$).
   - **Grouped Query Attention (GQA)** with memory-efficient `scaled_dot_product_attention` and Key-Value caching.
   - **SwiGLU Non-Linear MLP**: $(W_1 x \odot \text{silu}(W_3 x)) W_2$.
   - **Activation Checkpointing**: Gradient checkpointing across all decoder layers to minimize VRAM footprint.

---

## 📂 Project Structure

```
bllm/
├── config.py              # Central hyperparameters configuration
├── vqvae/
│   ├── __init__.py
│   ├── layers.py          # Residual blocks, Downsampling, Upsampling
│   ├── quantizer.py       # VectorQuantizer with Straight-Through Estimator
│   └── model.py           # Complete VQ-VAE Encoder-Decoder
├── tokenizer/
│   ├── __init__.py
│   └── text_tokenizer.py  # Byte-level BPE Tokenizer
├── model/
│   ├── __init__.py
│   ├── norm.py            # RMSNorm
│   ├── rope.py            # Rotary Position Embeddings (RoPE)
│   ├── attention.py       # Grouped Query Attention (GQA) & KV-Cache
│   ├── mlp.py             # SwiGLU Feed-Forward Network
│   └── transformer.py     # Multimodal Transformer Decoder
├── agent/                 # 🤖 Agentic Intelligence & Cognitive Layer
│   ├── __init__.py
│   ├── schema.py          # ToolCall, ToolResponse, CriticReport, ThinkingStep
│   ├── critic.py          # Visual Quality Inspector (sharpness, contrast, score)
│   ├── tools.py           # Autonomous tool dispatch registry
│   └── reasoner.py        # CoT (<think>) reasoner & ReAct reflection loop
├── pipeline/
│   ├── __init__.py
│   ├── dataset.py         # Synthetic dataset generator & causal data loader
│   ├── sampler.py         # End-to-end Text-to-Image Generation Pipeline
│   └── agentic_pipeline.py# End-to-end Agentic ReAct Pipeline
├── train_vqvae.py         # Visual tokenizer pre-training script
├── train_llm.py           # Multimodal LLM causal pre-training script
├── generate.py            # CLI text-to-image inference script
├── generate_agentic.py    # 🤖 CLI Agentic ReAct generation script
├── app_production.py      # Production Web Studio with Live CoT Inspector
├── tests/
│   ├── test_all.py        # Core model test suite
│   └── test_agentic.py    # 🤖 Agentic & Visual Critic test suite
└── requirements.txt       # Dependencies
```

---

## ⚡ Quick Start

### 1. Run Unit Tests
```bash
python -m unittest tests/test_all.py
```

### 2. Train the Visual Tokenizer (VQ-VAE)
```bash
python train_vqvae.py --epochs 5 --batch_size 8 --num_samples 200
```

### 3. Train the Multimodal LLM
```bash
python train_llm.py --epochs 5 --batch_size 4 --num_samples 200 --vqvae_ckpt checkpoints/vqvae.pt
```

### 4. Generate Images via CLI
```bash
python generate.py --prompt "a vibrant red glowing circle on dark background" --checkpoint checkpoints/multimodal_llm.pt --output output.png
```

### 5. Launch Interactive Web Studio
```bash
python app_production.py
```
Open your browser at `http://localhost:7860`. Includes real-time **🤖 Agentic Reasoning & Critic Mode** toggle with live `<think>` inspection.

### 6. Run Agentic Image Generation via CLI (Recommendation 3)
```bash
# Run autonomous ReAct loop with Chain-of-Thought (<think>) & Visual Quality Critic
python generate_agentic.py --prompt "a cybernetic tiger in a rainy neon alley" --engine scratch

# Ultra-HD Studio Diffusion with photorealistic style
python generate_agentic.py --prompt "majestic snowy mountain landscape" --engine hd --style "Photorealistic"

# Run Agentic test suite
python -m unittest tests/test_agentic.py
```
