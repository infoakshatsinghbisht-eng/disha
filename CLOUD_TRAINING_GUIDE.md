# ☁️ Disha Multimodal Foundation Model: Cloud GPU Training Guide

This guide details how to train **Disha (2.0B Multimodal Foundation Model + Agentic Pipeline)** on high-performance Cloud GPUs (Google Colab, RunPod, Kaggle, Lambda Labs, or AWS).

---

## ⚡ 1. Recommended Cloud Hardware

| Platform | Recommended GPU | VRAM | Cost | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Google Colab** | Nvidia T4 / A100 | 16 GB / 40 GB | Free (T4) / $10/mo | Quick experiments & fine-tuning |
| **Kaggle** | 2x Nvidia T4 | 16 GB | 100% Free (30 hrs/wk) | Free full training |
| **RunPod / Vast.ai** | RTX 4090 / A5000 | 24 GB | ~$0.30 - $0.45 / hr | Fast commercial-grade training |
| **Lambda Labs** | Nvidia A10 / A100 | 24 GB / 80 GB | ~$0.60 - $1.29 / hr | Full 2B parameter convergence |

---

## 📦 2. Dataset Preparation & Upload

The local generator created:
- **`dataset_cloud/train/`**: 600 verified 256x256 image-caption pairs across 15+ domains.
- **`dataset_cloud/val/`**: 60 validation pairs.
- **`dataset_cloud/metadata.jsonl`**: Detailed category and prompt annotations.
- **`disha_cloud_dataset.tar.gz`**: Single compressed archive for ultra-fast cloud transfer.

### Option A: Upload `disha_cloud_dataset.tar.gz` to Google Drive / Cloud
In Colab or Cloud Terminal:
```bash
# Extract the dataset in 2 seconds
tar -xzf disha_cloud_dataset.tar.gz
```

### Option B: Generate Directly on Cloud (Zero Upload Needed!)
If you prefer not to upload local files, simply run the builder script directly on your cloud instance (takes ~45 seconds over cloud gigabit connection):
```bash
python prepare_cloud_dataset.py
```

---

## 🚀 3. Step-by-Step Training Execution

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/infoakshatsinghbisht-eng/disha.git
cd disha
git pull origin main
pip install -r requirements.txt
```

### Step 2: Phase 1 — Train the Discrete Visual Tokenizer (VQ-VAE)
Trains the 2048-codebook hierarchical convolutional tokenizer to map $256 \times 256$ RGB images into $16 \times 16 = 256$ discrete visual tokens with Sobel edge gradient loss:
```bash
python train_vqvae.py \
    --data_dir dataset_cloud/train \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.0005 \
    --output_path checkpoints/vqvae.pt
```

### Step 3: Phase 2 — Train the 2.0B Multimodal Transformer Core
Trains the autoregressive next-token decoder (RMSNorm + RoPE + GQA + SwiGLU) to predict visual tokens from text captions:
```bash
python train_llm.py \
    --data_dir dataset_cloud/train \
    --vqvae_ckpt checkpoints/vqvae.pt \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.0003 \
    --checkpoint_dir checkpoints
```

*(Optional) All-in-One Automated 2B Training Script:*
```bash
python train_2b_master.py --train_samples 600 --vq_epochs 10 --llm_epochs 10 --batch_size 8
```

---

## 🤖 4. Testing the Trained Model with Agentic ReAct Engine

Once checkpoints are saved in `checkpoints/multimodal_llm.pt`, run the full Agentic system:

```bash
# Run with Chain-of-Thought (<think>) and Visual Critic
python generate_agentic.py \
    --prompt "a celestial phoenix soaring above frozen crystal glaciers at sunset" \
    --checkpoint checkpoints/multimodal_llm.pt \
    --engine scratch \
    --output cloud_result.png
```

---

## 🏆 5. 3-Month Foundation Model Pre-Training Blueprint

For long-term commercial pre-training (scaling to tens/hundreds of thousands of multimodal pairs across A100/H100 clusters):

### Phase A: Mass Sharded Ingestion (WebDataset Standard)
Streams and packages data into sequential 500-sample `.tar` shards with automatic Agentic CoT `<think>` recaptioning:
```bash
# Ingest and shard 50,000+ multimodal samples directly into GPU-streamable TAR archives
python pipeline/mass_dataset_ingest.py \
    --target_samples 50000 \
    --shard_size 500 \
    --output_dir dataset_sharded
```

### Phase B: 3-Stage Training Curriculum (`train_foundation_3months.py`)

#### Month 1: Visual Tokenizer Pre-Training (VQ-VAE)
Trains the discrete 2048-codebook visual representation on millions of patches:
```bash
python train_foundation_3months.py \
    --stage 1 \
    --steps 100000 \
    --batch_size 16 \
    --lr 5e-4 \
    --shards_dir dataset_sharded \
    --checkpoint_dir checkpoints_cloud
```

#### Month 2: 2.0B Transformer Core Autoregressive Pre-Training
Learns unified text-to-visual causal sequence modeling with GQA, RoPE, and SwiGLU:
```bash
python train_foundation_3months.py \
    --stage 2 \
    --steps 250000 \
    --batch_size 16 \
    --lr 3e-4 \
    --shards_dir dataset_sharded \
    --checkpoint_dir checkpoints_cloud
```

#### Month 3: High-Aesthetic & Agentic ReAct SFT Post-Training
Aligns the model with `<think>` planning traces and fine-tunes on top-aesthetic filtered samples:
```bash
python train_foundation_3months.py \
    --stage 3 \
    --steps 50000 \
    --batch_size 16 \
    --lr 1e-4 \
    --shards_dir dataset_sharded \
    --checkpoint_dir checkpoints_cloud
```

*(Note: `train_foundation_3months.py` features automatic spot-instance recovery; if your cloud instance is preempted or restarted, it will automatically resume from the latest saved checkpoint step without loss of progress!)*

---

## 🛠️ Memory & Speed Optimization Tips for Cloud GPUs
1. **Mixed Precision (FP16 / BF16)**: Already active in `config.py` (`mixed_precision=True`). Reduces VRAM usage by ~50%.
2. **FlashAttention**: Enabled automatically via PyTorch's native `F.scaled_dot_product_attention`.
3. **Batch Size Tuning**:
   - 16GB GPU (T4 / P100): Use `batch_size = 4` to `8`.
   - 24GB GPU (RTX 4090 / A5000): Use `batch_size = 16`.
   - 40GB/80GB GPU (A100): Use `batch_size = 32` to `64`.
