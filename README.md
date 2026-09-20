# 🚀 Multimodal LLM from Scratch (Native Text-to-Image Foundation Model)

A 100% scratch-built Multimodal Large Language Model in **pure PyTorch** without relying on third-party LLM APIs or wrapped model checkpoints.

---

## 🌟 Key Architecture Innovations

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
   - **Grouped Query Attention (GQA)** with full Key-Value caching for fast auto-regressive generation.
   - **SwiGLU Non-Linear MLP**: $(W_1 x \odot \text{silu}(W_3 x)) W_2$.

4. **Generation & Sampling Pipeline**
   - Temperature control, Top-K filtering, Top-P Nucleus sampling, and Classifier-Free Guidance (CFG).

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
├── pipeline/
│   ├── __init__.py
│   ├── dataset.py         # Synthetic dataset generator & causal data loader
│   └── sampler.py         # End-to-end Text-to-Image Generation Pipeline
├── train_vqvae.py         # Visual tokenizer pre-training script
├── train_llm.py           # Multimodal LLM causal pre-training script
├── generate.py            # CLI text-to-image inference script
├── demo_web.py            # Modern interactive web UI
├── tests/
│   └── test_all.py        # Comprehensive test suite
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
python demo_web.py
```
Open your browser at `http://localhost:7860`.
