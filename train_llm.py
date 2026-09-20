"""
Training script for the Multimodal Large Language Model (Decoder-only Transformer).
Trains causal next-token prediction across text tokens and discrete visual tokens.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from typing import Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import LLMConfig, VQVAEConfig, TrainingConfig
from vqvae.model import VQVAE
from model.transformer import MultimodalTransformer
from tokenizer.text_tokenizer import ByteTokenizer
from pipeline.dataset import TextImageDataset


def train_multimodal_llm(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    vqvae_checkpoint: str = "checkpoints/vqvae.pt",
    output_path: str = "checkpoints/multimodal_llm.pt",
    num_samples: int = 300,
    data_dir: str = None,
    dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    force_2b: bool = False,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[*] Starting Multimodal LLM Training on device: {device}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    llm_cfg = LLMConfig()

    # Dynamic VRAM scaling: If on consumer GPU (< 20GB VRAM like Colab T4) and not force_2b
    if not force_2b and dim is None:
        if device.startswith("cuda") and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 20.0:
                print(f"[*] Detected {vram_gb:.1f} GB VRAM (Tesla T4 / standard GPU).")
                print("[+] Auto-optimizing Transformer core (dim=512, layers=8, heads=8) to prevent OOM and maximize throughput.")
                llm_cfg.dim = 512
                llm_cfg.num_layers = 8
                llm_cfg.num_heads = 8
                llm_cfg.num_kv_heads = 4
    else:
        if dim is not None:
            llm_cfg.dim = dim
        if num_layers is not None:
            llm_cfg.num_layers = num_layers
        if num_heads is not None:
            llm_cfg.num_heads = num_heads
            llm_cfg.num_kv_heads = max(1, num_heads // 4)

    vq_cfg = VQVAEConfig()
    train_cfg = TrainingConfig()

    # 1. Initialize Tokenizer
    tokenizer = ByteTokenizer()
    total_vocab_size = llm_cfg.text_vocab_size + llm_cfg.image_vocab_size + len(tokenizer.SPECIAL_TOKENS)

    # 2. Initialize or Load VQ-VAE
    vqvae = VQVAE(
        in_channels=vq_cfg.in_channels,
        hidden_dim=vq_cfg.hidden_dim,
        embedding_dim=vq_cfg.embedding_dim,
        codebook_size=vq_cfg.codebook_size,
        num_res_blocks=vq_cfg.num_res_blocks,
        num_downsamples=vq_cfg.num_downsamples,
    ).to(device)

    if os.path.exists(vqvae_checkpoint):
        print(f"[+] Loading VQ-VAE weights from {vqvae_checkpoint}")
        ckpt = torch.load(vqvae_checkpoint, map_location=device, weights_only=False)
        vqvae.load_state_dict(ckpt["vqvae_state_dict"])
    else:
        print("[!] No pre-trained VQ-VAE found. Using randomly initialized visual tokenizer.")
    
    vqvae.eval()

    # 3. Initialize Multimodal Transformer Core
    llm = MultimodalTransformer(
        vocab_size=total_vocab_size,
        dim=llm_cfg.dim,
        num_layers=llm_cfg.num_layers,
        num_heads=llm_cfg.num_heads,
        num_kv_heads=llm_cfg.num_kv_heads,
        max_seq_len=llm_cfg.max_seq_len,
        ffn_dim_multiplier=llm_cfg.ffn_dim_multiplier,
        multiple_of=llm_cfg.multiple_of,
        norm_eps=llm_cfg.norm_eps,
        rope_theta=llm_cfg.rope_theta,
    ).to(device)

    total_params = sum(p.numel() for p in llm.parameters())
    print(f"[*] Multimodal LLM Architecture Initialized: {total_params / 1e6:.2f}M Parameters")

    # 4. Prepare Dataset & DataLoader
    dataset = TextImageDataset(
        tokenizer=tokenizer,
        vqvae=vqvae,
        image_size=vq_cfg.image_size,
        num_samples=num_samples,
        text_vocab_size=llm_cfg.text_vocab_size,
        image_vocab_size=llm_cfg.image_vocab_size,
        max_seq_len=llm_cfg.max_seq_len,
        data_dir=data_dir,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if device == "cpu":
        torch.set_num_threads(min(14, os.cpu_count() or 8))
        print(f"[+] Configured PyTorch CPU threads: {torch.get_num_threads()}")

    optimizer = optim.AdamW(llm.parameters(), lr=lr, weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    use_amp = (device == "cuda" or (isinstance(device, str) and device.startswith("cuda")))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[+] Enabled Mixed Precision (FP16 AMP) for GPU acceleration.")

    # 5. Training Loop with Gradient Accumulation
    grad_accum_steps = 2
    llm.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        optimizer.zero_grad()
        step_count = 0

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, loss = llm(input_ids, targets=target_ids)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(llm.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            step_count += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg": f"{epoch_loss / step_count:.4f}"})

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch} complete | Avg Causal Loss: {avg_epoch_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"[*] LLM Training completed in {elapsed:.2f}s. Saving master model to {output_path}")

    # 6. Save Complete Master Checkpoint
    torch.save({
        "llm_config": llm_cfg,
        "llm_state_dict": llm.state_dict(),
        "vqvae_config": vq_cfg,
        "vqvae_state_dict": vqvae.state_dict(),
        "tokenizer_vocab": {
            "merges": tokenizer.merges,
            "special_tokens": tokenizer.SPECIAL_TOKENS,
        },
    }, output_path)
    print("[+] Master 200M Multimodal LLM model successfully saved! Ready for generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal LLM from Scratch")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--vqvae_ckpt", type=str, default="checkpoints/vqvae.pt", help="VQVAE checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/multimodal_llm.pt", help="Output checkpoint")
    args = parser.parse_args()

    train_multimodal_llm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        vqvae_checkpoint=args.vqvae_ckpt,
        output_path=args.output,
        num_samples=args.num_samples,
    )
