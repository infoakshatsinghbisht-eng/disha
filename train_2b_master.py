"""
2.0 Billion Parameter Master Multimodal LLM Training Pipeline.
Trains VQ-VAE and 2.0B Transformer on Scraped Internet Web Dataset.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import argparse
from typing import Optional
import torch

from web_scraper_dataset import scrape_internet_dataset
from train_vqvae import train_vqvae
from train_llm import train_multimodal_llm
from pipeline.sampler import MultimodalGeneratorPipeline
from config import GenerationConfig, LLMConfig, VQVAEConfig


def run_2b_training(
    num_train: int = 400,
    num_val: int = 50,
    vq_epochs: int = 4,
    llm_epochs: int = 4,
    batch_size: int = 4,
    dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    force_2b: bool = False,
):
    start_total = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"[*] STARTING 2.0 BILLION PARAMETER MULTIMODAL MODEL TRAINING ON {device.upper()}")
    print("=" * 75)

    data_dir = "data_web_scraped"
    train_dir = os.path.join(data_dir, "train")

    # 1. Verify / Scrape Web Dataset
    print("\n--- [1/4] Verifying Internet Scraped Dataset ---")
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) < 100:
        scrape_internet_dataset(
            output_dir=data_dir,
            num_train=num_train,
            num_val=num_val,
            image_size=64,
        )
    else:
        print(f"[+] Internet dataset ready in {data_dir}/ with {len(os.listdir(train_dir)) // 2} image-text pairs.")

    # 2. Train VQ-VAE on Scraped Web Photos
    print("\n--- [2/4] Training Discrete Visual Tokenizer (VQ-VAE) with Sobel Edge Loss ---")
    train_vqvae(
        epochs=vq_epochs,
        batch_size=batch_size,
        lr=5e-4,
        device=device,
        output_path="checkpoints/vqvae.pt",
        num_samples=num_train,
        data_dir=train_dir,
    )

    # 3. Train 2.0 Billion Parameter Multimodal Transformer
    print("\n--- [3/4] Training 2.0B Parameter Multimodal Transformer LLM ---")
    train_multimodal_llm(
        epochs=llm_epochs,
        batch_size=batch_size,
        lr=3e-4,
        device=device,
        vqvae_checkpoint="checkpoints/vqvae.pt",
        output_path="checkpoints/multimodal_llm.pt",
        num_samples=num_train,
        data_dir=train_dir,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        force_2b=force_2b,
    )

    # 4. Multi-Domain Validation Generation
    print("\n--- [4/4] Generating Multi-Domain Validation Images ---")
    eval_dir = "master_eval"
    os.makedirs(eval_dir, exist_ok=True)

    pipeline = MultimodalGeneratorPipeline.from_pretrained(
        checkpoint_path="checkpoints/multimodal_llm.pt",
        device=device,
    )

    eval_prompts = [
        ("majestic_mountain", "a photorealistic majestic snow covered mountain peak with sharp rock ridges under clear azure sky"),
        ("autumn_forest", "a breathtaking autumn mountain landscape with golden birch forest and crystalline lake reflection"),
        ("ocean_wave", "a tranquil coastal ocean wave crashing softly on sandy tropical beach at sunrise"),
        ("cyberpunk_city", "a cyberpunk neon night city street with glowing holographic signs and reflection on wet asphalt"),
        ("sports_car", "a sleek vintage red sports car parked on an open scenic coastal highway"),
        ("celestial_sun", "a glowing celestial golden sun with radiant solar flares on deep cosmic void background"),
    ]

    for name, prompt in eval_prompts:
        out_path = os.path.join(eval_dir, f"{name}.png")
        print(f"[*] Generating: '{prompt[:45]}...' -> {out_path}")
        pipeline.generate_image(
            prompt=prompt,
            gen_config=GenerationConfig(temperature=0.35, top_p=0.85),
            save_path=out_path,
        )

    elapsed = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"[+] 2.0B MULTIMODAL MODEL TRAINING FINISHED IN {elapsed:.2f}s ({elapsed/60:.2f} minutes)!")
    print(f"[+] Checkpoint: checkpoints/multimodal_llm.pt")
    print(f"[+] Evaluation Outputs: {eval_dir}/")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=400)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--vq_epochs", type=int, default=4)
    parser.add_argument("--llm_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=None, help="Hidden dimension (default: auto-optimized for GPU)")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--force_2b", action="store_true", help="Force full 2B architecture (requires 40GB+ A100 GPU)")
    args = parser.parse_args()

    run_2b_training(
        num_train=args.train_samples,
        num_val=args.val_samples,
        vq_epochs=args.vq_epochs,
        llm_epochs=args.llm_epochs,
        batch_size=args.batch_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        force_2b=args.force_2b,
    )
