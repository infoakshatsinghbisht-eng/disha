"""
All-In-One Automated End-to-End Training and Evaluation Pipeline.
Builds the dataset, trains the VQ-VAE visual tokenizer, trains the Multimodal LLM,
and validates sample outputs on unseen test prompts.
"""

import os
import time
import argparse
import torch
from build_dataset import MultimodalDatasetBuilder
from train_vqvae import train_vqvae
from train_llm import train_multimodal_llm
from pipeline.sampler import MultimodalGeneratorPipeline
from config import GenerationConfig


def run_full_pipeline(
    train_samples: int = 300,
    val_samples: int = 40,
    vqvae_epochs: int = 4,
    llm_epochs: int = 5,
    batch_size: int = 8,
    image_size: int = 64,
):
    start_total = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"[*] Starting Multimodal LLM Training Pipeline on: {device.upper()}")
    print("=" * 65)

    # Step 1: Build Dataset
    print("\n--- [Stage 1/4] Building Multimodal Dataset ---")
    MultimodalDatasetBuilder.build_dataset_to_disk(
        root_dir="dataset",
        train_samples=train_samples,
        val_samples=val_samples,
        image_size=image_size,
    )

    # Step 2: Train Visual Tokenizer (VQ-VAE)
    print("\n--- [Stage 2/4] Training Discrete Visual Tokenizer (VQ-VAE) ---")
    train_vqvae(
        epochs=vqvae_epochs,
        batch_size=batch_size,
        lr=4e-4,
        device=device,
        output_path="checkpoints/vqvae.pt",
        num_samples=train_samples,
    )

    # Step 3: Train Multimodal LLM Transformer
    print("\n--- [Stage 3/4] Training Multimodal Transformer LLM ---")
    train_multimodal_llm(
        epochs=llm_epochs,
        batch_size=batch_size,
        lr=4e-4,
        device=device,
        vqvae_checkpoint="checkpoints/vqvae.pt",
        output_path="checkpoints/multimodal_llm.pt",
        num_samples=train_samples,
    )

    # Step 4: Run Evaluation on Unseen Test Prompts
    print("\n--- [Stage 4/4] Generating Validation Test Samples ---")
    eval_dir = "eval_results"
    os.makedirs(eval_dir, exist_ok=True)

    pipeline = MultimodalGeneratorPipeline.from_pretrained("checkpoints/multimodal_llm.pt", device=device)
    
    test_prompts = [
        "a glowing ruby red celestial sun with stars on sunset background",
        "a centered electric blue circle on dark background",
        "a sharp emerald green diamond emblem on cyberpunk background",
        "a minimalist landscape with a golden pyramid on sunset background",
    ]

    for i, prompt in enumerate(test_prompts):
        out_file = os.path.join(eval_dir, f"eval_sample_{i+1}.png")
        print(f"[*] Generating: \"{prompt}\" -> {out_file}")
        pipeline.generate_image(prompt, gen_config=GenerationConfig(temperature=0.8), save_path=out_file)

    total_time = time.time() - start_total
    print("\n" + "=" * 65)
    print(f"[+] Complete Pipeline Finished in: {total_time:.2f} seconds ({total_time/60:.2f} minutes)!")
    print(f"[+] Model checkpoint saved: checkpoints/multimodal_llm.pt")
    print(f"[+] Evaluation samples saved in: {eval_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full End-to-End Training Pipeline")
    parser.add_argument("--train_samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=30, help="Number of validation samples")
    parser.add_argument("--vq_epochs", type=int, default=3, help="VQ-VAE epochs")
    parser.add_argument("--llm_epochs", type=int, default=4, help="LLM epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    run_full_pipeline(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        vqvae_epochs=args.vq_epochs,
        llm_epochs=args.llm_epochs,
        batch_size=args.batch_size,
    )
