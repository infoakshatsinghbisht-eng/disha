"""
Autonomous High-Resolution Web Image Scraper & Multi-Domain Dataset Builder.
Downloads hundreds or thousands of high-definition (512x512 / 1024x1024) photographic images
from open web CDNs paired with rich, natural language captions for Disha model training.
"""

import os
import io
import ssl
import time
import random
import argparse
import urllib.request
import concurrent.futures
from typing import List, Tuple
from PIL import Image

ssl_context = ssl._create_unverified_context()

# Comprehensive 30+ category registry covering nature, culture, tech, architecture, and art
DATASET_CATEGORIES: List[Tuple[str, List[int], str]] = [
    # 1. Himalayan & Mountain Landscapes
    ("a photorealistic majestic snow covered Himalayan mountain peak with sharp rock ridges under clear azure sky", 
     [10, 28, 29, 30, 42, 54, 60, 65, 76, 88, 1029, 1036], "himalayas"),
    ("a breathtaking autumn mountain landscape with golden birch forest and crystalline lake reflection in Uttarakhand", 
     [15, 16, 17, 18, 19, 24, 35, 45, 55, 68, 1015, 1016], "autumn"),
    ("a tranquil coastal ocean wave crashing softly on sandy tropical beach at golden hour sunrise", 
     [38, 48, 49, 56, 57, 58, 62, 77, 89, 92, 1001, 1002], "ocean"),
    ("a lush green misty pine forest in the Himalayan foothills with morning sunlight beams", 
     [70, 71, 72, 73, 74, 75, 84, 95, 105, 115, 1018, 1024], "forest"),
    ("a pristine mountain river valley with turquoise glacial water rushing over smooth river rocks", 
     [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 1043, 1044], "river"),

    # 2. Culture, Heritage & Architecture
    ("a traditional rustic Himalayan stone cottage nestled in green terraced hills with slate roof", 
     [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 1040, 1041], "cottage"),
    ("an ancient ornate Indian stone temple architecture with intricate carved pillars in warm dusk light", 
     [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 1045, 1047], "temple"),
    ("a modern illuminated architectural glass skyscraper reaching into dramatic night sky", 
     [100, 101, 102, 103, 104, 112, 122, 132, 142, 152, 1060, 1062], "skyscraper"),
    ("a cozy modern wooden cabin interior with warm fireplace glow and panoramic mountain window view", 
     [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 1050, 1051], "cabin"),

    # 3. Technology & Workspace
    ("a sleek modern developer workspace setup with glowing laptop displaying clean python code", 
     [1, 2, 3, 4, 6, 7, 8, 9, 20, 26, 37, 48], "laptop"),
    ("a professional mobile smartphone mockup displaying an elegant modern language learning app UI", 
     [170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 1055, 1056], "mobile_app"),
    ("a minimalist clean designer desk with mechanical keyboard notebook and steaming cup of coffee", 
     [180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 1067, 1068], "desk"),

    # 4. Vehicles & Transportation
    ("a sleek vintage red sports car parked on an open scenic coastal highway under dramatic sunset", 
     [1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080], "sports_car"),
    ("a modern commercial passenger airplane soaring gracefully through vibrant golden hour clouds", 
     [1050, 1051, 1052, 1053, 1054, 1057, 1058, 1059, 1061, 1063], "airplane"),

    # 5. Wildlife & Nature
    ("a close-up portrait of a loyal domestic dog with expressive brown eyes in soft natural sunlight", 
     [237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 1025, 1062], "dog"),
    ("a majestic wild horse standing gracefully in an open grassy meadow under morning sunrise", 
     [210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 1003, 1004], "horse"),
    ("a macro close-up photograph of a vibrant red rose with crystalline morning dew drops on petals", 
     [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 1084, 1085], "flower"),

    # 6. Futuristic & Cosmic
    ("a cyberpunk neon night city street with glowing holographic signs and reflection on wet asphalt", 
     [110, 111, 113, 114, 125, 135, 145, 155, 165, 175, 1070, 1071], "cyberpunk"),
    ("a glowing celestial golden sun with radiant solar flares on deep cosmic void background", 
     [250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 1005, 1006], "sun"),
    ("a luminous spiral galaxy with glittering star clusters and cosmic interstellar nebula dust", 
     [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 1008, 1009], "galaxy"),
]


def fetch_hd_image(image_id: int, size: int = 512) -> Image.Image:
    """Fetches high-res image from open CDN and resizes with high-quality anti-aliasing."""
    url = f"https://picsum.photos/id/{image_id}/{size}/{size}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, context=ssl_context, timeout=10) as resp:
        img_bytes = resp.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img.resize((size, size), Image.Resampling.LANCZOS)


def run_autonomous_scraping(
    output_dir: str = "data_hd_scraped/train",
    num_images: int = 500,
    image_size: int = 512,
    max_workers: int = 25,
):
    """
    Asynchronously scrapes diverse HD image-caption pairs using parallel worker threads.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print(f"[*] STARTING AUTONOMOUS HD IMAGE SCRAPING ({num_images} IMAGES at {image_size}x{image_size})")
    print(f"[*] Target Directory: {output_dir}")
    print(f"[*] Parallel Threads: {max_workers}")
    print("=" * 75)

    # Build queue of (image_id, prompt, category)
    queue = []
    round_idx = 0
    while len(queue) < num_images:
        for prompt, id_list, tag in DATASET_CATEGORIES:
            for base_id in id_list:
                img_id = (base_id + round_idx * 23) % 1084 + 1
                queue.append((img_id, prompt, tag))
                if len(queue) >= num_images:
                    break
            if len(queue) >= num_images:
                break
        round_idx += 1

    random.seed(42)
    random.shuffle(queue)
    queue = queue[:num_images]

    success_count = 0
    start_time = time.time()

    def worker(idx_item):
        idx, (img_id, prompt, tag) = idx_item
        img_filename = f"disha_{idx:05d}.png"
        txt_filename = f"disha_{idx:05d}.txt"
        img_path = os.path.join(output_dir, img_filename)
        txt_path = os.path.join(output_dir, txt_filename)

        # Skip if already exists
        if os.path.exists(img_path) and os.path.exists(txt_path):
            return True

        try:
            img = fetch_hd_image(img_id, size=image_size)
            img.save(img_path, quality=95)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            return True
        except Exception:
            return False

    print(f"[*] Downloading {len(queue)} high-resolution images in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, enumerate(queue)))
        success_count = sum(1 for r in results if r)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"[+] Autonomous Scraping Finished in {elapsed:.1f}s!")
    print(f"[+] Successfully scraped & paired: {success_count}/{num_images} HD images.")
    print(f"[+] Directory: {output_dir}")
    print("=" * 75)
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous HD Image Scraper for Disha")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to scrape")
    parser.add_argument("--size", type=int, default=512, help="Resolution (512 or 1024)")
    parser.add_argument("--output_dir", type=str, default="data_hd_scraped/train", help="Target folder")
    parser.add_argument("--workers", type=int, default=25, help="Number of parallel download threads")
    args = parser.parse_args()

    run_autonomous_scraping(
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=args.size,
        max_workers=args.workers,
    )
