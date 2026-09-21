"""
Child-to-Genius Nursery Curriculum Training Engine for Disha Multimodal Model.
Progressively trains one geometric primitive at a time like teaching a child:
  Lesson 1: The Dot (Single center point & spatial grounding)
  Lesson 2: The Line (Horizontal, Vertical, Diagonal lines)
  Lesson 3: The Circle (Curvature, radius, radial symmetry)
  Lesson 4: The Square (90-degree orthogonal edges & sharp corners)
  Lesson 5: The Triangle (Diagonal slopes & vertices)

Each lesson trains focused high-contrast samples and immediately runs an Exam
to visually inspect if the model has mastered the shape.
"""

import os
import sys
import math
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import LLMConfig, VQVAEConfig, GenerationConfig
from model.transformer import MultimodalTransformer
from vqvae.model import VQVAE
from tokenizer.text_tokenizer import ByteTokenizer
from pipeline.sampler import MultimodalGeneratorPipeline


# -----------------------------------------------------------------------------
# 1. High-Contrast Procedural Primitive Generator
# -----------------------------------------------------------------------------
COLORS = [
    ("red", (220, 30, 40)),
    ("blue", (30, 110, 230)),
    ("green", (35, 170, 75)),
    ("black", (20, 24, 30)),
    ("purple", (140, 40, 210)),
    ("orange", (240, 120, 25)),
]


def render_primitive(shape: str, color_name: str, rgb: Tuple[int, int, int], size: int = 256) -> Tuple[Image.Image, str]:
    """Generates an ultra-crisp geometric primitive on pure white canvas."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    center = size // 2

    if shape == "dot":
        r = random.randint(16, 28)
        draw.ellipse([center - r, center - r, center + r, center + r], fill=rgb)
        prompt = f"a solid {color_name} dot centered on white canvas"

    elif shape == "line_horizontal":
        w = random.randint(8, 14)
        y = center + random.randint(-10, 10)
        draw.line([(35, y), (size - 35, y)], fill=rgb, width=w)
        prompt = f"a clean {color_name} horizontal line on white background"

    elif shape == "line_vertical":
        w = random.randint(8, 14)
        x = center + random.randint(-10, 10)
        draw.line([(x, 35), (x, size - 35)], fill=rgb, width=w)
        prompt = f"a clean {color_name} vertical line on white background"

    elif shape == "circle":
        r = random.randint(55, 80)
        draw.ellipse([center - r, center - r, center + r, center + r], fill=rgb)
        prompt = f"a solid {color_name} circle on white background"

    elif shape == "square":
        half = random.randint(50, 75)
        draw.rectangle([center - half, center - half, center + half, center + half], fill=rgb)
        prompt = f"a solid {color_name} square on white background"

    elif shape == "triangle":
        h = random.randint(60, 85)
        pts = [(center, center - h), (center - int(h * 1.05), center + int(h * 0.7)), (center + int(h * 1.05), center + int(h * 0.7))]
        draw.polygon(pts, fill=rgb)
        prompt = f"a solid {color_name} triangle on white background"

    else:
        r = 65
        draw.ellipse([center - r, center - r, center + r, center + r], fill=rgb)
        prompt = f"a solid {color_name} circle on white background"

    return img, prompt


# -----------------------------------------------------------------------------
# 2. In-Memory Progressive Shape Dataset
# -----------------------------------------------------------------------------
class ProgressiveShapeDataset(Dataset):
    def __init__(self, allowed_shapes: List[str], samples_per_shape: int = 25, vqvae: Optional[VQVAE] = None, tokenizer: Optional[ByteTokenizer] = None, device: str = "cpu"):
        self.samples = []
        self.tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"[*] Synthesizing clean nursery dataset for shapes: {allowed_shapes}...")
        for shape in allowed_shapes:
            for _ in range(samples_per_shape):
                color_name, rgb = random.choice(COLORS)
                img, prompt = render_primitive(shape, color_name, rgb)

                if vqvae is not None:
                    with torch.no_grad():
                        t_img = transform(img).unsqueeze(0).to(device)
                        indices = vqvae.encode_to_indices(t_img)[0].cpu()
                else:
                    indices = torch.zeros(256, dtype=torch.long)

                text_tokens = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
                prefix = [self.tokenizer.bos_id] + text_tokens + [self.tokenizer.image_start_id]
                full_seq = prefix + (indices + 8000).tolist() + [self.tokenizer.image_end_id, self.tokenizer.eos_id]

                input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)
                target_ids = torch.tensor(full_seq[1:], dtype=torch.long)

                # Focus 100% of the loss gradient on drawing the visual tokens correctly
                text_len = len(prefix)
                target_ids[:text_len - 1] = -100

                self.samples.append({
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "prompt": prompt,
                })

        print(f"[+] Dataset ready with {len(self.samples)} focused high-contrast training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_shape_batch(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_batch = torch.full((len(batch), max_len), 0, dtype=torch.long)
    target_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_batch[i, :seq_len] = item["input_ids"]
        target_batch[i, :seq_len] = item["target_ids"]

    return input_batch, target_batch


# -----------------------------------------------------------------------------
# 3. Progressive Lesson Runner
# -----------------------------------------------------------------------------
LESSONS: Dict[str, List[str]] = {
    "1_dot": ["dot"],
    "2_line": ["line_horizontal", "line_vertical"],
    "3_circle": ["circle"],
    "4_square": ["square"],
    "5_triangle": ["triangle"],
    "all_primitives": ["dot", "line_horizontal", "line_vertical", "circle", "square", "triangle"],
}

EXAM_PROMPTS: Dict[str, str] = {
    "1_dot": "a solid red dot centered on white canvas",
    "2_line": "a clean blue horizontal line on white background",
    "3_circle": "a solid red circle on white background",
    "4_square": "a solid green square on white background",
    "5_triangle": "a solid purple triangle on white background",
    "all_primitives": "a solid red circle on white background",
}


def train_lesson(
    lesson_key: str = "3_circle",
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 3e-4,
    checkpoint_path: str = "checkpoints/multimodal_llm.pt",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 75)
    print(f"👶 DISHA NURSERY CURRICULUM: LESSON '{lesson_key.upper()}'")
    print(f"[*] Target Hardware: {device.upper()} | Epochs: {epochs} | LR: {lr}")
    print("=" * 75)

    shapes = LESSONS.get(lesson_key, ["circle"])

    # Load Pipeline
    print(f"\n[1/4] Loading foundation checkpoint: {checkpoint_path}")
    pipeline = MultimodalGeneratorPipeline.from_pretrained(checkpoint_path, device=device)
    vqvae = pipeline.vqvae.eval()
    llm = pipeline.llm.train()
    tokenizer = pipeline.tokenizer

    # Prepare Dataset
    print(f"\n[2/4] Generating focused samples for lesson '{lesson_key}'...")
    samples_per_shape = 30 if len(shapes) <= 2 else 15
    dataset = ProgressiveShapeDataset(
        allowed_shapes=shapes,
        samples_per_shape=samples_per_shape,
        vqvae=vqvae,
        tokenizer=tokenizer,
        device=device,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_shape_batch)

    # Optimizer
    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training Loop
    print(f"\n[3/4] Teaching the model (Loss should drop as shape features lock in)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits, _ = llm(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"  [Epoch {epoch:2d}/{epochs}] Visual Cross-Entropy Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # Save Updated Checkpoint
    print(f"\n[+] Saving graduated model weights to: {checkpoint_path}")
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt["llm_state_dict"] = llm.state_dict()
    torch.save(ckpt, checkpoint_path)

    # 4. Immediate Visual Exam
    print("\n[4/4] 🎓 TAKING VISUAL EXAMINATION (Aaya ya nahi test karte hain)...")
    exam_prompt = EXAM_PROMPTS.get(lesson_key, f"a solid red {shapes[0]} on white background")
    print(f"[*] Exam Prompt: \"{exam_prompt}\"")

    llm.eval()
    gen_cfg = GenerationConfig(temperature=0.7, top_k=30, top_p=0.88, repetition_penalty=1.12)
    exam_img, tokens = pipeline.generate_image_with_tokens(exam_prompt, gen_config=gen_cfg)

    exam_filename = f"exam_{lesson_key}.png"
    exam_img.save(exam_filename)
    print(f"[+] Exam output image saved to: {os.path.abspath(exam_filename)}")
    print(f"[+] Generated unique visual codebook tokens: {len(set(tokens))} / 256")

    # Auto-display in Colab/Jupyter
    try:
        from IPython.display import display
        display(exam_img)
    except Exception:
        pass

    return exam_img


def main():
    parser = argparse.ArgumentParser(description="Disha Nursery Curriculum Training")
    parser.add_argument(
        "--lesson",
        type=str,
        default="3_circle",
        choices=["1_dot", "2_line", "3_circle", "4_square", "5_triangle", "all_primitives"],
        help="Syllabus lesson to train",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / cpu)")

    args = parser.parse_args()
    train_lesson(
        lesson_key=args.lesson,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
