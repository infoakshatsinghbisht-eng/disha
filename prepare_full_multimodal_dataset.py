"""
Master Multimodal Dataset Preparation Script.
Generates 500 high-fidelity training samples and 60 validation samples with anti-aliasing.
"""

import os
import random
from build_dataset import MultimodalDatasetBuilder


def build_master_dataset(
    output_dir: str = "data_master",
    num_train: int = 500,
    num_val: int = 60,
    image_size: int = 64,
):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"[*] Assembling High-Fidelity Dataset in '{output_dir}'...")

    print(f"[*] Generating {num_train} high-fidelity training pairs...")
    for i in range(num_train):
        img, prompt = MultimodalDatasetBuilder.generate_sample(image_size)
        img_path = os.path.join(train_dir, f"train_{i:04d}.png")
        txt_path = os.path.join(train_dir, f"train_{i:04d}.txt")
        img.save(img_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    print(f"[*] Generating {num_val} validation pairs...")
    for i in range(num_val):
        img, prompt = MultimodalDatasetBuilder.generate_sample(image_size)
        img_path = os.path.join(val_dir, f"val_{i:04d}.png")
        txt_path = os.path.join(val_dir, f"val_{i:04d}.txt")
        img.save(img_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    print(f"[+] Master Dataset generation complete: {num_train} train + {num_val} val samples.")


if __name__ == "__main__":
    build_master_dataset()
