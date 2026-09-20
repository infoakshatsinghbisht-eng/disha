"""
High-Capacity Master Training and Evaluation Pipeline with Real-World Web Dataset.
Trains VQ-VAE and 200M Multimodal LLM to high convergence on the Real Web Dataset.
"""

import os
import time
import argparse
import torch

from fetch_web_multimodal_dataset import fetch_and_build_web_dataset
from train_vqvae import train_vqvae
from train_llm import train_multimodal_llm
from pipeline.sampler import MultimodalGeneratorPipeline
from config import GenerationConfig


def run_master_training(
    num_train: int = 350,
    num_val: int = 40,
    vq_epochs: int = 3,
    llm_epochs: int = 4,
    batch_size: int = 4,
):
    start_total = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"[*] STARTING MASTER MULTIMODAL MODEL TRAINING ON {device.upper()}")
    print("=" * 70)

    # 1. Dataset Check or Fetch
    print("\n--- [1/4] Checking & Preparing Real-World Web Dataset ---")
    train_dir = "data_master/train"
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) < 100:
        fetch_and_build_web_dataset(
            output_dir="data_master",
            num_train=num_train,
            num_val=num_val,
            image_size=64,
        )
    else:
        print(f"[+] Found existing real web dataset in data_master/ ({len(os.listdir(train_dir))//2} pairs).")

    # 2. VQ-VAE Codebook Training
    print("\n--- [2/4] Training Discrete Visual Tokenizer (VQ-VAE) on Real Web Images ---")
    train_vqvae(
        epochs=vq_epochs,
        batch_size=batch_size,
        lr=5e-4,
        device=device,
        output_path="checkpoints/vqvae.pt",
        num_samples=num_train,
        data_dir="data_master/train",
    )

    # 3. Multimodal Transformer LLM Training
    print("\n--- [3/4] Training 200M Multimodal Transformer LLM on Real Web Data ---")
    train_multimodal_llm(
        epochs=llm_epochs,
        batch_size=batch_size,
        lr=4e-4,
        device=device,
        vqvae_checkpoint="checkpoints/vqvae.pt",
        output_path="checkpoints/multimodal_llm.pt",
        num_samples=num_train,
        data_dir="data_master/train",
    )

    # 4. Multi-Category Evaluation Generation
    print("\n--- [4/4] Generating Multi-Category Validation Images ---")
    eval_dir = "master_eval"
    os.makedirs(eval_dir, exist_ok=True)

    pipeline = MultimodalGeneratorPipeline.from_pretrained(
        checkpoint_path="checkpoints/multimodal_llm.pt",
        device=device,
    )

    test_prompts = [
        ("mountain_snow", "a photorealistic majestic mountain peak covered in snow under clear blue sky"),
        ("autumn_forest", "a scenic view of autumn forest trees with golden and red foliage"),
        ("ocean_sunset", "a tranquil coastal ocean wave breaking on sandy tropical beach at sunrise"),
        ("modern_city", "a modern illuminated architectural glass skyscraper reaching into night sky"),
        ("domestic_dog", "a close-up portrait of a domestic dog with loyal expressive eyes"),
        ("sports_car", "a sleek vintage sports car parked on an open asphalt road"),
    ]

    for name, prompt in test_prompts:
        out_path = os.path.join(eval_dir, f"{name}.png")
        print(f"[*] Generating: '{prompt}' -> {out_path}")
        pipeline.generate_image(
            prompt=prompt,
            gen_config=GenerationConfig(temperature=0.75, top_p=0.90),
            save_path=out_path,
        )

    elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"[+] MASTER MODEL TRAINING COMPLETED IN {elapsed:.2f}s ({elapsed/60:.2f} minutes)!")
    print(f"[+] Final Checkpoint: checkpoints/multimodal_llm.pt")
    print(f"[+] Evaluation Images: {eval_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=350)
    parser.add_argument("--val_samples", type=int, default=40)
    parser.add_argument("--vq_epochs", type=int, default=3)
    parser.add_argument("--llm_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_master_training(
        num_train=args.train_samples,
        num_val=args.val_samples,
        vq_epochs=args.vq_epochs,
        llm_epochs=args.llm_epochs,
        batch_size=args.batch_size,
    )
