"""
Training script for the Discrete Visual Tokenizer (VQ-VAE).
Trains codebook embeddings and reconstruction capability on image datasets.
"""

import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import VQVAEConfig
from vqvae.model import VQVAE
from tokenizer.text_tokenizer import ByteTokenizer
from pipeline.dataset import TextImageDataset


def train_vqvae(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = "checkpoints/vqvae.pt",
    num_samples: int = 500,
    data_dir: str = None,
):
    print(f"[*] Starting VQ-VAE Training on device: {device}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cfg = VQVAEConfig()
    model = VQVAE(
        in_channels=cfg.in_channels,
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        codebook_size=cfg.codebook_size,
        num_res_blocks=cfg.num_res_blocks,
        num_downsamples=cfg.num_downsamples,
        commitment_cost=cfg.commitment_cost,
    ).to(device)

    if device == "cpu":
        torch.set_num_threads(min(14, os.cpu_count() or 8))
        print(f"[+] Configured PyTorch CPU threads for VQ-VAE: {torch.get_num_threads()}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (num_samples // batch_size + 1))

    tokenizer = ByteTokenizer()
    dataset = TextImageDataset(
        tokenizer=tokenizer,
        image_size=cfg.image_size,
        num_samples=num_samples,
        data_dir=data_dir,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            images = batch["image_tensor"].to(device)
            optimizer.zero_grad()

            recon_x, loss, metrics = model(images)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += metrics["total_loss"].item()
            total_recon += metrics["recon_loss"].item()
            total_vq += metrics["vq_loss"].item()

            pbar.set_postfix({
                "Loss": f"{metrics['total_loss'].item():.4f}",
                "Recon": f"{metrics['recon_loss'].item():.4f}",
                "VQ": f"{metrics['vq_loss'].item():.4f}",
            })

        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.2f}s. Saving checkpoint to {output_path}")

    torch.save({
        "vqvae_config": cfg,
        "vqvae_state_dict": model.state_dict(),
    }, output_path)
    print("[+] VQ-VAE model successfully saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE Visual Tokenizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of synthetic samples")
    parser.add_argument("--output", type=str, default="checkpoints/vqvae.pt", help="Output path")
    args = parser.parse_args()

    train_vqvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        num_samples=args.num_samples,
    )
