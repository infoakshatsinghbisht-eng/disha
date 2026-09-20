"""
Disha Ultra-HD 1024x1024 Foundation Image Generator.
Uses Pre-Trained State-of-the-Art Diffusion Core (SDXL Turbo) with optional Disha Custom LoRA.
"""

import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
from PIL import Image


def generate_hd_image(
    prompt: str,
    output_path: str = "disha_hd_output.png",
    lora_path: str = None,
    engine: str = "sdxl-turbo",
    num_inference_steps: int = 2,
    guidance_scale: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Image.Image:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from diffusers import AutoPipelineForText2Image

    model_id = "stabilityai/sdxl-turbo" if "xl" in engine.lower() else "stabilityai/sd-turbo"
    print(f"[*] Loading HD Foundation Model ({model_id}) on {device}...")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    if device.startswith("cuda"):
        try:
            pipe.to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"[*] VRAM optimization triggered ({e}). Activating Model CPU Offloading...")
            gc.collect()
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Enable memory optimizations
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    # Load custom trained LoRA weights if provided
    if lora_path and os.path.exists(lora_path):
        print(f"[+] Loading Disha Custom LoRA Adapter from: {lora_path}")
        pipe.load_lora_weights(lora_path)

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    resolution = "1024x1024" if "xl" in engine.lower() else "512x512"
    print(f"[*] Generating {resolution} Ultra-HD Image for: '{prompt}'")
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    print(f"[*] Decoding final {resolution} RGB image...")
    img = result.images[0]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"[+] Success! Crystal-clear HD image saved to: {output_path}")
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disha Ultra-HD Image Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="disha_hd_output.png", help="Output file path")
    parser.add_argument("--lora", type=str, default=None, help="Path to custom trained LoRA weights")
    parser.add_argument("--engine", type=str, default="sdxl-turbo", choices=["sdxl-turbo", "sd-turbo"], help="Engine architecture")
    parser.add_argument("--steps", type=int, default=2, help="Inference steps (1-2 for fast Turbo models)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_hd_image(
        prompt=args.prompt,
        output_path=args.output,
        lora_path=args.lora,
        engine=args.engine,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
