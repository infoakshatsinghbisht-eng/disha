"""
1-Click Master Training Pipeline for Disha Foundation Model.
Step 1: Scrapes hundreds of high-res 512x512 / 1024x1024 images in parallel (Himalayas, culture, tech, architecture, nature).
Step 2: Trains custom LoRA adapter weights on top of Pre-Trained Foundation Core.
Step 3: Automatically generates high-definition sample evaluation images.
"""

import os
import time
import argparse
from autonomous_scraper import run_autonomous_scraping
from train_custom_lora import train_disha_lora
from generate_hd import generate_hd_image


def run_full_pipeline(
    num_scrape: int = 500,
    image_size: int = 512,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    engine: str = "sdxl-turbo",
):
    start_time = time.time()
    data_dir = "data_hd_scraped/train"
    lora_dir = "checkpoints/disha_lora"
    eval_dir = "disha_eval_hd"

    # Stage 1: Scrape Fresh HD Dataset
    print("\n" + "=" * 75)
    print("--- [STAGE 1/3] Autonomous High-Resolution Dataset Scraping ---")
    print("=" * 75)
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) < (num_scrape * 2):
        run_autonomous_scraping(
            output_dir=data_dir,
            num_images=num_scrape,
            image_size=image_size,
            max_workers=30,
        )
    else:
        print(f"[+] Found existing HD dataset in {data_dir} ({len(os.listdir(data_dir))//2} pairs).")

    # Stage 2: Train Disha Custom LoRA Adapter
    print("\n" + "=" * 75)
    print(f"--- [STAGE 2/3] Fine-Tuning Disha Model ({engine}) with LoRA on GPU ---")
    print("=" * 75)
    base_model_id = "stabilityai/sdxl-turbo" if "xl" in engine.lower() else "stabilityai/sd-turbo"
    train_disha_lora(
        data_dir=data_dir,
        output_dir=lora_dir,
        base_model_id=base_model_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Stage 3: Generate Validation HD Samples
    print("\n" + "=" * 75)
    print("--- [STAGE 3/3] Generating Crystal-Clear Test Outputs ---")
    print("=" * 75)
    os.makedirs(eval_dir, exist_ok=True)

    test_prompts = [
        ("himalayan_mountain", "a photorealistic majestic snow covered Himalayan mountain peak with traditional Kumaoni village in Uttarakhand, golden hour, 8k"),
        ("developer_setup", "a sleek modern developer workspace setup with glowing laptop displaying python code with Himalayan view window"),
        ("traditional_cottage", "a traditional rustic Himalayan stone cottage nestled in green terraced hills with slate roof and vibrant flowers"),
    ]

    for name, prompt in test_prompts:
        out_file = os.path.join(eval_dir, f"{name}.png")
        print(f"[*] Generating: '{prompt[:45]}...' -> {out_file}")
        try:
            generate_hd_image(
                prompt=prompt,
                output_path=out_file,
                lora_path=lora_dir,
                engine=engine,
                num_inference_steps=2,
            )
        except Exception as e:
            print(f"[!] Warning: Sample generation fallback: {e}")

    total_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"[+] COMPLETE DISHA TRAINING PIPELINE FINISHED IN {total_time/60:.2f} MINUTES!")
    print(f"[+] Scraped Images: {data_dir}/")
    print(f"[+] Trained Model Weights: {lora_dir}/")
    print(f"[+] HD Output Samples: {eval_dir}/")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Click Disha Master Training Pipeline")
    parser.add_argument("--scrape_count", type=int, default=500, help="Number of HD images to scrape")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--engine", type=str, default="sdxl-turbo", choices=["sdxl-turbo", "sd-turbo"], help="Engine architecture")
    args = parser.parse_args()

    run_full_pipeline(
        num_scrape=args.scrape_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        engine=args.engine,
    )
