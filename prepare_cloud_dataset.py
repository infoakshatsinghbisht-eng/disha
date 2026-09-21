"""
Cloud Training Dataset Builder & Packager for Disha Multimodal Foundation Model.
Assembles 256x256 multi-domain image-caption pairs (both real-world photographic
and procedural high-fidelity scenes), generates metadata index, and packages everything
into a compressed archive (disha_cloud_dataset.tar.gz) ready for 1-click cloud GPU training.
"""

import os
import sys
import io
import ssl
import json
import time
import tarfile
import random
import urllib.request
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from PIL import Image, ImageDraw, ImageFilter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from build_dataset import MultimodalDatasetBuilder
from pipeline.prompt_enhancer import PromptEnhancer

ssl_context = ssl._create_unverified_context()

# Curated High-Diversity Photographic Categories for Cloud Training
CURATED_WEB_DOMAINS = [
    # 1. Majestic Nature & Landscapes
    {
        "category": "Landscapes",
        "caption": "a photorealistic majestic snow-covered Himalayan mountain peak with crystalline glaciers under azure sky",
        "ids": [10, 28, 29, 30, 42, 54, 60, 65, 76, 88, 93, 102, 114, 126, 138, 149],
    },
    {
        "category": "Landscapes",
        "caption": "a breathtaking autumn forest with vibrant golden and scarlet foliage reflected in a serene mountain lake",
        "ids": [15, 16, 17, 18, 19, 24, 35, 45, 55, 68, 79, 87, 98, 109, 121, 133],
    },
    {
        "category": "Landscapes",
        "caption": "a tranquil tropical coastline with turquoise ocean waves rolling onto white sandy beach during golden sunset",
        "ids": [38, 48, 49, 56, 57, 58, 62, 77, 89, 92, 104, 116, 128, 140, 151, 163],
    },
    {
        "category": "Landscapes",
        "caption": "a lush emerald rainforest with misty cascading waterfall rushing over mossy ancient rocks",
        "ids": [70, 71, 72, 73, 74, 75, 84, 95, 105, 115, 125, 136, 147, 158, 169, 180],
    },
    {
        "category": "Landscapes",
        "caption": "vast golden desert sand dunes under dramatic sunset light with wind ripples and deep shadows",
        "ids": [80, 81, 82, 83, 85, 96, 107, 118, 129, 141, 153, 164, 175, 186, 197, 208],
    },

    # 2. Architecture & Urban Cities
    {
        "category": "Architecture",
        "caption": "a towering modern glass skyscraper with geometric facades reflecting orange evening sunset light",
        "ids": [100, 101, 102, 103, 104, 112, 122, 132, 142, 152, 162, 173, 184, 195, 206, 217],
    },
    {
        "category": "Cyberpunk",
        "caption": "a futuristic cyberpunk street illuminated by neon pink and cyan holographic signs reflected on wet asphalt",
        "ids": [110, 111, 113, 114, 125, 135, 145, 155, 165, 175, 185, 196, 207, 218, 229, 240],
    },
    {
        "category": "Architecture",
        "caption": "a historic classical stone bridge spanning across a calm European river with quaint pastel houses",
        "ids": [120, 121, 122, 123, 124, 134, 144, 154, 166, 177, 188, 199, 211, 222, 233, 244],
    },

    # 3. Wildlife & Animals
    {
        "category": "Wildlife",
        "caption": "a close-up portrait of a loyal golden retriever dog with gentle expressive brown eyes in soft natural sunlight",
        "ids": [237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251, 253, 255],
    },
    {
        "category": "Wildlife",
        "caption": "a fluffy domestic cat curled up peacefully on a cozy wooden windowsill with warm morning sunbeams",
        "ids": [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 212, 214, 216, 219, 223, 227],
    },
    {
        "category": "Wildlife",
        "caption": "a noble wild stallion horse galloping freely through an open lush green alpine meadow at sunrise",
        "ids": [210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 225, 230, 234, 238, 242],
    },
    {
        "category": "Wildlife",
        "caption": "a vivid exotic tropical macaw parrot with radiant red, blue and yellow plumage on a jungle tree branch",
        "ids": [220, 221, 222, 223, 224, 226, 228, 231, 235, 239, 243, 248, 252, 256, 260, 264],
    },

    # 4. Vehicles & Technology
    {
        "category": "Vehicles",
        "caption": "a sleek classic red vintage sports car with polished chrome bumpers parked on scenic coastal highway",
        "ids": [1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084],
    },
    {
        "category": "Vehicles",
        "caption": "a commercial passenger airplane soaring gracefully through warm golden hour clouds above endless horizon",
        "ids": [1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063],
    },

    # 5. Macro & Botanical
    {
        "category": "Botanical",
        "caption": "a vibrant ruby red rose in full bloom with delicate glistening morning dew drops on velvet petals",
        "ids": [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 165, 167, 169, 171, 173],
    },
    {
        "category": "Botanical",
        "caption": "a macro close-up photograph of lush green monstera leaf showing intricate organic vein patterns in morning dew",
        "ids": [170, 171, 172, 173, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196],
    },

    # 6. Celestial & Space
    {
        "category": "Celestial",
        "caption": "a brilliant celestial sun with radiant golden flares emitting warm coronal light in deep cosmic void",
        "ids": [250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 263, 265, 267, 269, 271],
    },
    {
        "category": "Celestial",
        "caption": "an interstellar spiral galaxy with glittering star clusters and vibrant purple nebula gas clouds",
        "ids": [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 321, 323, 325, 327, 329, 331],
    },
]


def fetch_web_photo(image_id: int, size: int = 256, timeout: int = 6) -> Optional[Image.Image]:
    """Downloads a photo from CDN with timeout and Lanczos anti-aliasing."""
    url = f"https://picsum.photos/id/{image_id}/{size * 2}/{size * 2}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"},
    )
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as resp:
            data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return img.resize((size, size), Image.Resampling.LANCZOS)
    except Exception:
        return None


def generate_agentic_caption(base_caption: str, category: str) -> str:
    """
    Enriches base caption with fine-art descriptors, lighting, and camera optics
    so the model learns high-fidelity representations during pre-training.
    """
    optics = [
        "shot on 35mm lens, f/1.8 aperture, natural volumetric lighting",
        "hyperdetailed textures, sharp focus, 8k resolution photograph",
        "cinematic color grading, shallow depth of field, award-winning composition",
        "clean architectural framing, crisp focus, soft natural illumination",
    ]
    return f"{base_caption}, {random.choice(optics)}"


def build_cloud_dataset(
    output_dir: str = "dataset_cloud",
    target_train_count: int = 600,
    target_val_count: int = 60,
    image_size: int = 256,
    max_workers: int = 12,
):
    """
    Builds the full multi-domain 256x256 training & validation dataset.
    Combines web photographic images and procedural super-sampled graphics.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("=" * 75)
    print(f"[*] BUILDING MASTER CLOUD DATASET IN '{output_dir}'")
    print(f"[*] Target: {target_train_count} Train + {target_val_count} Val ({image_size}x{image_size} RGB)")
    print("=" * 75)

    # Collect existing scraped files if available in data_web_scraped
    existing_samples: List[Tuple[Image.Image, str, str]] = []
    source_scraped = "data_web_scraped/train"
    if os.path.exists(source_scraped):
        print(f"[+] Importing existing verified samples from '{source_scraped}'...")
        for fname in os.listdir(source_scraped):
            if fname.endswith(".png"):
                base = os.path.splitext(fname)[0]
                txt_file = os.path.join(source_scraped, base + ".txt")
                img_file = os.path.join(source_scraped, fname)
                if os.path.exists(txt_file):
                    try:
                        with open(txt_file, "r", encoding="utf-8") as f:
                            cap = f.read().strip()
                        img = Image.open(img_file).convert("RGB")
                        if img.size != (image_size, image_size):
                            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        existing_samples.append((img, cap, "WebScraped"))
                    except Exception:
                        pass
        print(f"[+] Loaded {len(existing_samples)} valid pre-cached images.")

    # Fetch new web photos concurrently
    download_tasks = []
    for domain in CURATED_WEB_DOMAINS:
        for img_id in domain["ids"]:
            download_tasks.append((img_id, domain["caption"], domain["category"]))

    random.shuffle(download_tasks)
    print(f"[*] Downloading diverse real-world photographic images (Threads: {max_workers})...")
    fetched_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_web_photo, tid, image_size): (cap, cat)
            for tid, cap, cat in download_tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            cap, cat = future_map[future]
            try:
                img = future.result()
                if img is not None:
                    enriched_cap = generate_agentic_caption(cap, cat)
                    existing_samples.append((img, enriched_cap, cat))
                    fetched_count += 1
            except Exception:
                pass

    print(f"[+] Downloaded {fetched_count} new web images.")

    # Complement with procedural anti-aliased samples to reach target
    needed_train = target_train_count - int(len(existing_samples) * 0.9)
    if needed_train > 0:
        print(f"[*] Synthesizing {needed_train} procedural high-fidelity scenes with Lanczos supersampling...")
        for _ in range(needed_train + target_val_count):
            img, prompt = MultimodalDatasetBuilder.generate_sample(image_size)
            existing_samples.append((img, prompt, "ProceduralFineArt"))

    random.shuffle(existing_samples)

    # Split into Train and Val
    val_count = min(target_val_count, max(20, int(len(existing_samples) * 0.1)))
    val_set = existing_samples[:val_count]
    train_set = existing_samples[val_count: val_count + target_train_count]

    print(f"[*] Writing {len(train_set)} training samples to '{train_dir}'...")
    metadata_records = []

    for i, (img, cap, cat) in enumerate(train_set):
        name = f"train_{i:05d}"
        img_p = os.path.join(train_dir, f"{name}.png")
        txt_p = os.path.join(train_dir, f"{name}.txt")
        img.save(img_p, format="PNG", optimize=True)
        with open(txt_p, "w", encoding="utf-8") as f:
            f.write(cap)
        metadata_records.append({
            "split": "train",
            "id": name,
            "category": cat,
            "caption": cap,
            "width": image_size,
            "height": image_size,
        })

    print(f"[*] Writing {len(val_set)} validation samples to '{val_dir}'...")
    for i, (img, cap, cat) in enumerate(val_set):
        name = f"val_{i:05d}"
        img_p = os.path.join(val_dir, f"{name}.png")
        txt_p = os.path.join(val_dir, f"{name}.txt")
        img.save(img_p, format="PNG", optimize=True)
        with open(txt_p, "w", encoding="utf-8") as f:
            f.write(cap)
        metadata_records.append({
            "split": "val",
            "id": name,
            "category": cat,
            "caption": cap,
            "width": image_size,
            "height": image_size,
        })

    # Save metadata.jsonl
    meta_path = os.path.join(output_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in metadata_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[+] Dataset assembled successfully: {len(train_set)} train + {len(val_set)} val pairs.")
    print(f"[+] Metadata index written to: {meta_path}")

    # Package into tar.gz for ultra-fast cloud transfer
    archive_path = "disha_cloud_dataset.tar.gz"
    print(f"\n[*] Packaging dataset into compressed archive '{archive_path}'...")
    t0 = time.time()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    
    archive_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"[*] Cloud Package Ready: {archive_path} ({archive_mb:.1f} MB) in {time.time()-t0:.1f}s")
    return archive_path, len(train_set), len(val_set)


if __name__ == "__main__":
    build_cloud_dataset()
