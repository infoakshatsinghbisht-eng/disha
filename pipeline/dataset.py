"""
Dataset pipeline and synthetic multimodal data generator for training and testing.
Provides both synthetic procedural datasets and general image-caption loaders.
"""

from typing import List, Tuple, Dict, Optional
import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch.utils.data import Dataset

from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE


class SyntheticMultimodalDataGenerator:
    """
    Generates procedural synthetic image-caption pairs for immediate training and validation.
    Supports diverse shapes, gradients, landscapes, geometric compositions, and color combinations.
    """
    SHAPES = ["circle", "square", "triangle", "star", "diamond", "heart"]
    COLORS = {
        "red": (230, 40, 40),
        "blue": (40, 100, 240),
        "green": (40, 200, 60),
        "yellow": (250, 220, 30),
        "purple": (160, 50, 230),
        "cyan": (30, 220, 230),
        "orange": (250, 130, 20),
        "magenta": (240, 30, 180),
        "white": (245, 245, 245),
        "dark": (20, 20, 25),
    }
    BACKGROUNDS = ["black", "dark blue", "white", "gray", "sunset gradient", "forest gradient", "neon glow"]

    @staticmethod
    def generate_single_sample(image_size: int = 256) -> Tuple[Image.Image, str]:
        """Generates a procedural image and its matching descriptive text prompt."""
        shape = random.choice(SyntheticMultimodalDataGenerator.SHAPES)
        color_name = random.choice(list(SyntheticMultimodalDataGenerator.COLORS.keys()))
        color_rgb = SyntheticMultimodalDataGenerator.COLORS[color_name]
        bg_choice = random.choice(SyntheticMultimodalDataGenerator.BACKGROUNDS)

        # Base canvas
        img = Image.new("RGB", (image_size, image_size), color=(15, 15, 20))
        draw = ImageDraw.Draw(img)

        # Draw Background
        if "sunset" in bg_choice:
            for y in range(image_size):
                r = int(240 * (1 - y / image_size) + 40 * (y / image_size))
                g = int(80 * (1 - y / image_size) + 20 * (y / image_size))
                b = int(120 * (1 - y / image_size) + 60 * (y / image_size))
                draw.line([(0, y), (image_size, y)], fill=(r, g, b))
            bg_desc = "sunset gradient background"
        elif "neon" in bg_choice:
            for y in range(image_size):
                r = int(20 + 30 * math.sin(y / 20))
                g = int(10 + 20 * math.cos(y / 25))
                b = int(40 + 40 * math.sin(y / 30))
                draw.line([(0, y), (image_size, y)], fill=(max(0, r), max(0, g), max(0, b)))
            bg_desc = "glowing neon background"
        elif "white" in bg_choice:
            draw.rectangle([(0, 0), (image_size, image_size)], fill=(240, 240, 245))
            bg_desc = "clean white background"
        elif "dark blue" in bg_choice:
            draw.rectangle([(0, 0), (image_size, image_size)], fill=(10, 20, 45))
            bg_desc = "deep blue background"
        else:
            draw.rectangle([(0, 0), (image_size, image_size)], fill=(18, 18, 22))
            bg_desc = "dark background"

        # Center coordinates & scale
        cx, cy = image_size // 2, image_size // 2
        radius = random.randint(image_size // 6, image_size // 3)

        # Draw shape
        if shape == "circle":
            draw.ellipse(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                fill=color_rgb,
                outline=(255, 255, 255),
                width=3,
            )
        elif shape == "square":
            draw.rectangle(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                fill=color_rgb,
                outline=(255, 255, 255),
                width=3,
            )
        elif shape == "triangle":
            points = [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ]
            draw.polygon(points, fill=color_rgb, outline=(255, 255, 255))
        elif shape == "diamond":
            points = [
                (cx, cy - radius),
                (cx + radius, cy),
                (cx, cy + radius),
                (cx - radius, cy),
            ]
            draw.polygon(points, fill=color_rgb, outline=(255, 255, 255))
        elif shape == "star":
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius // 2
                angle = i * math.pi / 5 - math.pi / 2
                points.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
            draw.polygon(points, fill=color_rgb, outline=(255, 255, 255))
        else:
            # Simple geometric sphere with shadow
            draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=color_rgb)

        # Add subtle soft glow effect
        img = img.filter(ImageFilter.SMOOTH_MORE)

        # Compose natural language prompt variations
        templates = [
            f"a {color_name} {shape} on a {bg_desc}",
            f"an image of a vibrant {color_name} {shape} centered with {bg_desc}",
            f"minimalist artwork featuring a {color_name} {shape} against a {bg_desc}",
            f"a clean render of a {color_name} {shape}, {bg_desc}",
        ]
        prompt = random.choice(templates)
        return img, prompt


class TextImageDataset(Dataset):
    """
    Multimodal Dataset that prepares aligned Text + Visual token sequences for causal training.
    
    Sequence layout:
    [<bos>, text_tokens..., <image_start>, image_codebook_tokens (e.g. 256 tokens)..., <image_end>, <eos>]
    """
    def __init__(
        self,
        tokenizer: ByteTokenizer,
        vqvae: Optional[VQVAE] = None,
        image_size: int = 256,
        num_samples: int = 1000,
        text_vocab_size: int = 5000,
        image_vocab_size: int = 4096,
        max_seq_len: int = 512,
        data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vqvae = vqvae
        self.image_size = image_size
        self.num_samples = num_samples
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir

        # Pre-generate synthetic samples or list directory
        self.samples: List[Tuple[Image.Image, str]] = []
        if data_dir and os.path.exists(data_dir):
            # Load from directory
            valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
            for fname in os.listdir(data_dir):
                base, ext = os.path.splitext(fname)
                if ext.lower() in valid_exts:
                    img_path = os.path.join(data_dir, fname)
                    txt_path = os.path.join(data_dir, base + ".txt")
                    prompt = base.replace("_", " ")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            prompt = f.read().strip()
                    try:
                        img = Image.open(img_path).convert("RGB")
                        self.samples.append((img, prompt))
                    except Exception:
                        pass
        
        # Fallback / augment with synthetic data generator
        if len(self.samples) == 0:
            for _ in range(num_samples):
                img, prompt = SyntheticMultimodalDataGenerator.generate_single_sample(image_size)
                self.samples.append((img, prompt))

    def _process_image(self, img: Image.Image) -> torch.Tensor:
        """Resizes and normalizes image to tensor in range [-1, 1] with shape (3, H, W)."""
        img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        return img_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, prompt = self.samples[idx]
        img_tensor = self._process_image(img)  # (3, H, W) in [-1, 1]

        # Text Tokenization
        text_tokens = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)

        # Visual Tokenization (if VQ-VAE provided, or placeholder discrete grid)
        if self.vqvae is not None:
            with torch.no_grad():
                img_batch = img_tensor.unsqueeze(0)
                img_tokens = self.vqvae.encode_to_indices(img_batch).squeeze(0)  # (256,)
                # Offset image tokens to avoid collision with text vocabulary
                img_tokens = img_tokens + self.text_vocab_size
        else:
            # Pseudo deterministic tokens for initial staging
            img_tokens = torch.randint(
                self.text_vocab_size,
                self.text_vocab_size + self.image_vocab_size,
                (256,),
                dtype=torch.long,
            )

        # Assemble unified sequence:
        # [ <bos>, text_tokens, <image_start>, image_tokens (256), <image_end>, <eos> ]
        full_seq = (
            [self.tokenizer.bos_id]
            + text_tokens
            + [self.tokenizer.image_start_id]
            + img_tokens.tolist()
            + [self.tokenizer.image_end_id, self.tokenizer.eos_id]
        )

        # Truncate or pad to max_seq_len
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[:self.max_seq_len]
        
        seq_len = len(full_seq)
        pad_len = self.max_seq_len - seq_len

        input_ids = torch.tensor(full_seq + [self.tokenizer.pad_id] * pad_len, dtype=torch.long)
        
        # Labels for causal language modeling: targets are input_ids shifted by 1
        # Pad positions set to -100 so loss ignores them
        target_ids = torch.tensor(
            full_seq[1:] + [self.tokenizer.pad_id] * (pad_len + 1),
            dtype=torch.long,
        )
        target_ids[seq_len - 1 :] = -100  # Mask out padding from loss

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "image_tensor": img_tensor,
            "prompt": prompt,
        }
