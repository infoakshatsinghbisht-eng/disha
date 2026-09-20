"""
Disha Custom LoRA Fine-Tuning Pipeline.
Trains lightweight custom adapter weights (~80MB) on top of Pre-Trained Foundation Model
using your custom photos and prompts (e.g. Kumaoni culture, Himalayan landscapes, custom products).
"""

import os
import argparse
import time
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class CustomImagePromptDataset(Dataset):
    """Loads image-caption pairs from directory (e.g., sample_01.png & sample_01.txt)."""
    def __init__(self, data_dir: str, size: int = 512):
        self.samples: List[Tuple[str, str]] = []
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        if os.path.exists(data_dir):
            for fname in sorted(os.listdir(data_dir)):
                base, ext = os.path.splitext(fname)
                if ext.lower() in valid_exts:
                    img_path = os.path.join(data_dir, fname)
                    txt_path = os.path.join(data_dir, base + ".txt")
                    caption = base.replace("_", " ")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                    self.samples.append((img_path, caption))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, caption


def train_disha_lora(
    data_dir: str = "data_web_scraped/train",
    output_dir: str = "checkpoints/disha_lora",
    base_model_id: str = "stabilityai/sdxl-turbo",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    lora_rank: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 75)
    print(f"[*] STARTING DISHA PRE-TRAINED LORA FINE-TUNING ON {device.upper()}")
    print(f"[*] Base Foundation Model: {base_model_id}")
    print(f"[*] Dataset: {data_dir}")
    print(f"[*] Output: {output_dir}")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = CustomImagePromptDataset(data_dir, size=512)
    if len(dataset) == 0:
        print(f"[!] Warning: No images found in {data_dir}. Please place .png + .txt pairs there.")
        return

    print(f"[+] Found {len(dataset)} training image-prompt pairs.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 2. Load Base Diffusion Model & Components
    from diffusers import AutoPipelineForText2Image
    from peft import LoraConfig, get_peft_model

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print("[*] Downloading / Loading Pre-trained Foundation Pipeline...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)

    # Freeze base components
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)

    # Enable gradient checkpointing to reduce VRAM by up to 70%
    if hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    # 3. Inject LoRA Adapters into UNet
    print(f"[*] Injecting LoRA Adapters (Rank={lora_rank}) into Attention Layers...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(pipe.unet, lora_config)
    unet.print_trainable_parameters()

    # 4. Optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # 5. Training Loop
    unet.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"LoRA Epoch {epoch}/{epochs}")

        for step, (images, captions) in enumerate(pbar):
            images = images.to(device, dtype=dtype)
            optimizer.zero_grad()

            with torch.no_grad():
                # Encode images into latent space
                latents = pipe.vae.encode(images).latent_dist.sample() * pipe.vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Encode text prompts
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=list(captions),
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

            # Predict noise with LoRA-adapted UNet
            with torch.amp.autocast("cuda", enabled=(dtype == torch.float16)):
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": torch.zeros((latents.shape[0], 6), device=device, dtype=dtype),
                } if hasattr(pipe.unet, "add_embedding") and pipe.unet.add_embedding is not None else {}

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"==> LoRA Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n[+] LoRA Fine-Tuning Completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")

    # 6. Save LoRA Weights (~80MB)
    out_lora_path = os.path.join(output_dir, "disha_custom_lora.safetensors")
    unet.save_pretrained(output_dir)
    print(f"[+] Disha LoRA Weights Saved to: {output_dir}/")
    print(f"[+] Total size is lightweight (~80MB) — ready for VPS deployment!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disha Custom LoRA Fine-Tuning")
    parser.add_argument("--data_dir", type=str, default="data_web_scraped/train", help="Path to training images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/disha_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (2 recommended for T4 GPU)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--engine", type=str, default="sdxl-turbo", choices=["sdxl-turbo", "sd-turbo"], help="Engine architecture")
    args = parser.parse_args()

    base_model_id = "stabilityai/sdxl-turbo" if "xl" in args.engine.lower() else "stabilityai/sd-turbo"

    train_disha_lora(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model_id=base_model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
