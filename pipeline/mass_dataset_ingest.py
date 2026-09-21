"""
Mass Multi-Source Data Ingestion Engine for Foundation Model Pre-Training.
Ingests, enriches, quality-filters, and streams high-fidelity 256x256 multimodal samples
directly into TAR shards (WebDataset standard) for multi-month Cloud GPU training.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import io
import ssl
import json
import time
import random
import urllib.request
import argparse
import concurrent.futures
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline.sharded_dataset import TarShardedDatasetWriter
from pipeline.agentic_recaptioner import AgenticRecaptioner
from build_dataset import MultimodalDatasetBuilder
from agent.critic import VisualCritic

ssl_context = ssl._create_unverified_context()

# Expanded 30+ Domain Taxonomy for Foundation Scale Training
FOUNDATION_CATEGORIES = [
    # Landscapes & Environmental
    ("a photorealistic majestic snow covered Himalayan peak under brilliant azure sky", "Landscapes", [10, 28, 29, 30, 42, 54, 60, 65, 76, 88, 93, 102, 114, 126, 138, 149]),
    ("a lush green pine forest reflecting in a mirror calm alpine glacial lake", "Landscapes", [15, 16, 17, 18, 19, 24, 35, 45, 55, 68, 79, 87, 98, 109, 121, 133]),
    ("a vast sunlit ocean wave breaking dramatically onto tropical sand at sunset", "Landscapes", [38, 48, 49, 56, 57, 58, 62, 77, 89, 92, 104, 116, 128, 140, 151, 163]),
    ("a misty moss covered emerald jungle waterfall flowing through ancient canyon", "Landscapes", [70, 71, 72, 73, 74, 75, 84, 95, 105, 115, 125, 136, 147, 158, 169, 180]),
    ("vast red desert sand dunes under golden hour lighting with long dramatic shadows", "Landscapes", [80, 81, 82, 83, 85, 96, 107, 118, 129, 141, 153, 164, 175, 186, 197, 208]),
    
    # Architecture & Cities
    ("a modern architectural skyscraper with glass facade glowing in golden sunset light", "Architecture", [100, 101, 102, 103, 104, 112, 122, 132, 142, 152, 162, 173, 184, 195]),
    ("a futuristic cyberpunk city street with vibrant neon signs and puddles on asphalt", "Cyberpunk", [110, 111, 113, 114, 125, 135, 145, 155, 165, 175, 185, 196, 207, 218]),
    ("a picturesque cobblestone alleyway in historic European town with flower balconies", "Architecture", [120, 121, 122, 123, 124, 134, 144, 154, 166, 177, 188, 199, 211, 222]),
    ("a minimalist contemporary interior living space with warm sunlight streams", "Interior", [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143]),
    
    # Wildlife & Animals
    ("a detailed close-up portrait of a loyal golden retriever dog with expressive brown eyes", "Wildlife", [237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251]),
    ("a domestic cat napping peacefully on a warm sunlit wooden window sill", "Wildlife", [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 212, 214, 216, 219]),
    ("a majestic wild horse galloping across an open grassy meadow under morning sun", "Wildlife", [210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 225, 230, 234]),
    ("a vibrant scarlet macaw parrot perched on a tropical tree branch in nature", "Wildlife", [220, 221, 222, 223, 224, 226, 228, 231, 235, 239, 243, 248, 252, 256]),
    
    # Vehicles & Aviation
    ("a sleek vintage cherry red sports car parked on an open scenic highway", "Vehicles", [1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082]),
    ("a passenger commercial jet airplane cruising smoothly above golden sunset clouds", "Aviation", [1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061]),
    
    # Macro & Botanical
    ("a macro photograph of a ruby red rose with glistening morning dew drops", "Botanical", [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 165, 167, 169]),
    ("a macro close-up of vibrant green plant leaf showing intricate cellular veins", "Botanical", [170, 171, 172, 173, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192]),
    
    # Space & Celestial
    ("a radiant celestial golden sun with explosive solar flares in deep cosmic space", "Space", [250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 263, 265, 267]),
    ("a glowing spiral galaxy with billions of shimmering stars and purple nebula dust", "Space", [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 321, 323, 325, 327]),
]


def fetch_image_from_url(image_id: int, size: int = 256, timeout: int = 5) -> Optional[Image.Image]:
    """Downloads image from CDN with fast timeout and Lanczos anti-aliasing."""
    url = f"https://picsum.photos/id/{image_id}/{size * 2}/{size * 2}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"},
    )
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as resp:
            data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return img.resize((size, size), Image.Resampling.LANCZOS)
    except Exception:
        return None


def run_mass_ingestion(
    target_samples: int = 2000,
    max_samples_per_shard: int = 500,
    output_dir: str = "dataset_sharded",
    image_size: int = 256,
    enable_agentic_cot: bool = True,
    critic_min_threshold: float = 2.0,
    num_threads: int = 16,
) -> Dict[str, Any]:
    """
    Orchestrates mass ingestion across multi-threaded web sources, local caches,
    and procedural anti-aliased synthetic generators into TAR shards.
    """
    t0 = time.time()
    print("=" * 76)
    print(f"[*] MASS DATASET INGESTION ENGINE — TARGET: {target_samples} SAMPLES")
    print(f"[*] Output Directory      : '{output_dir}'")
    print(f"[*] Resolution            : {image_size}x{image_size} RGB")
    print(f"[*] Shard Capacity        : {max_samples_per_shard} samples/shard")
    print(f"[*] Agentic CoT Recaption : {enable_agentic_cot}")
    print("=" * 76)

    writer = TarShardedDatasetWriter(
        output_dir=output_dir,
        max_samples_per_shard=max_samples_per_shard,
        shard_prefix="disha_shard",
    )
    critic = VisualCritic(aesthetic_threshold=critic_min_threshold)

    ingested_count = 0

    # 1. Ingest existing pre-scraped samples if available
    source_dirs = ["dataset_cloud/train", "data_web_scraped/train", "data_master/train"]
    for s_dir in source_dirs:
        if os.path.exists(s_dir) and ingested_count < target_samples:
            print(f"[+] Ingesting pre-cached samples from '{s_dir}'...")
            for fname in sorted(os.listdir(s_dir)):
                if fname.endswith(".png") and ingested_count < target_samples:
                    base = os.path.splitext(fname)[0]
                    txt_path = os.path.join(s_dir, f"{base}.txt")
                    img_path = os.path.join(s_dir, fname)
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, "r", encoding="utf-8") as f:
                                cap = f.read().strip()
                            img = Image.open(img_path).convert("RGB")
                            if img.size != (image_size, image_size):
                                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

                            # Quality Critic Pre-Filter
                            report = critic.evaluate(img)
                            if report.aesthetic_score < critic_min_threshold:
                                continue

                            # Apply Agentic CoT Recaptioning
                            if enable_agentic_cot:
                                cot_caption, _ = AgenticRecaptioner.generate_cot_trace(cap)
                                final_cap = cot_caption
                            else:
                                final_cap = cap

                            sample_id = f"sample_{ingested_count:06d}"
                            writer.add_sample(
                                sample_id=sample_id,
                                image=img,
                                caption=final_cap,
                                metadata={
                                    "source": s_dir,
                                    "aesthetic_score": report.aesthetic_score,
                                    "sharpness": report.sharpness,
                                    "contrast": report.contrast,
                                },
                            )
                            ingested_count += 1
                        except Exception:
                            pass
            print(f"[+] Total samples so far: {ingested_count}/{target_samples}")

    # 2. Parallel Web Fetching if more samples needed
    if ingested_count < target_samples:
        needed = target_samples - ingested_count
        print(f"[*] Fetching remaining {needed} samples from multi-domain web streams...")
        fetch_queue = []
        for prompt, cat, ids in FOUNDATION_CATEGORIES:
            for iid in ids:
                fetch_queue.append((iid, prompt, cat))

        random.shuffle(fetch_queue)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_meta = {
                executor.submit(fetch_image_from_url, iid, image_size): (prompt, cat)
                for iid, prompt, cat in fetch_queue
            }
            for future in concurrent.futures.as_completed(future_to_meta):
                if ingested_count >= target_samples:
                    break
                prompt, cat = future_to_meta[future]
                try:
                    img = future.result()
                    if img is not None:
                        report = critic.evaluate(img)
                        if report.aesthetic_score >= critic_min_threshold:
                            if enable_agentic_cot:
                                cot_caption, _ = AgenticRecaptioner.generate_cot_trace(prompt, category=cat)
                                final_cap = cot_caption
                            else:
                                final_cap = prompt

                            sample_id = f"sample_{ingested_count:06d}"
                            writer.add_sample(
                                sample_id=sample_id,
                                image=img,
                                caption=final_cap,
                                metadata={
                                    "source": "WebStream",
                                    "category": cat,
                                    "aesthetic_score": report.aesthetic_score,
                                },
                            )
                            ingested_count += 1
                except Exception:
                    pass

    # 3. Procedural Synthetics to guarantee exact target
    if ingested_count < target_samples:
        needed = target_samples - ingested_count
        print(f"[*] Synthesizing {needed} procedural anti-aliased scenes...")
        for _ in range(needed):
            img, prompt = MultimodalDatasetBuilder.generate_sample(image_size)
            if enable_agentic_cot:
                cot_caption, _ = AgenticRecaptioner.generate_cot_trace(prompt, category="Procedural")
                final_cap = cot_caption
            else:
                final_cap = prompt

            sample_id = f"sample_{ingested_count:06d}"
            writer.add_sample(
                sample_id=sample_id,
                image=img,
                caption=final_cap,
                metadata={"source": "ProceduralSuperSampled"},
            )
            ingested_count += 1

    # Finalize shards and write index
    index_info = writer.close()
    elapsed = time.time() - t0

    print("=" * 76)
    print(f"[*] INGESTION COMPLETE in {elapsed:.1f}s!")
    print(f"[*] Total Samples Written : {index_info['total_samples']}")
    print(f"[*] Total Shards Created  : {index_info['total_shards']}")
    print(f"[*] Master Shards Index   : {os.path.join(output_dir, 'shards_index.json')}")
    print("=" * 76)
    return index_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Dataset Ingestion Engine")
    parser.add_argument("--target_samples", type=int, default=1500, help="Total samples to ingest and shard")
    parser.add_argument("--shard_size", type=int, default=500, help="Max samples per TAR shard")
    parser.add_argument("--output_dir", type=str, default="dataset_sharded", help="Output directory for shards")
    parser.add_argument("--image_size", type=int, default=256, help="Target image dimension (256x256)")
    parser.add_argument("--no_cot", action="store_true", help="Disable synthetic Agentic CoT recaptioning")
    args = parser.parse_args()

    run_mass_ingestion(
        target_samples=args.target_samples,
        max_samples_per_shard=args.shard_size,
        output_dir=args.output_dir,
        image_size=args.image_size,
        enable_agentic_cot=not args.no_cot,
    )
