"""
Child-to-Genius Nursery Curriculum Training Engine for Disha Multimodal Model.
Progressively trains one geometric primitive at a time like teaching a child:
  Lesson 1: The Dot (Single center point & spatial grounding)
  Lesson 2: The Line (Horizontal, Vertical lines)
  Lesson 3: The Circle (Curvature, radius, radial symmetry)
  Lesson 4: The Square (90-degree orthogonal edges & sharp corners)
  Lesson 5: The Triangle (Diagonal slopes & vertices)

Equipped with a Target-Driven Reinforcement Mastery Loop:
Continues training iteratively until the model's visual codebook token predictions
reach the target accuracy threshold (>= 98% / 100%) against ground truth.
Immediately produces a side-by-side Target vs. Learned visual exam card!
"""

import os
import sys
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
import numpy as np
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
# 1. Color Palette & Geometry Helpers
# -----------------------------------------------------------------------------
COLORS = {
    "red": (220, 30, 40),
    "blue": (30, 110, 230),
    "green": (35, 170, 75),
    "purple": (140, 40, 210),
    "orange": (240, 120, 25),
    "black": (20, 24, 30),
}


def render_canonical_primitive(lesson_key: str, color_name: str = "default", size: int = 256) -> Tuple[Image.Image, str]:
    """
    Renders an authoritative, mathematically precise ground-truth primitive.
    """
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    center = size // 2

    if lesson_key == "1_dot":
        col = COLORS.get(color_name, COLORS["red"])
        actual_name = color_name if color_name in COLORS else "red"
        r = 24
        draw.ellipse([center - r, center - r, center + r, center + r], fill=col)
        prompt = f"a solid {actual_name} dot centered on white canvas"

    elif lesson_key == "2_line":
        col = COLORS.get(color_name, COLORS["blue"])
        actual_name = color_name if color_name in COLORS else "blue"
        draw.line([(35, center), (size - 35, center)], fill=col, width=12)
        prompt = f"a clean {actual_name} horizontal line on white background"

    elif lesson_key == "3_circle":
        col = COLORS.get(color_name, COLORS["red"])
        actual_name = color_name if color_name in COLORS else "red"
        r = 65
        draw.ellipse([center - r, center - r, center + r, center + r], fill=col)
        prompt = f"a solid {actual_name} circle on white background"

    elif lesson_key == "4_square":
        col = COLORS.get(color_name, COLORS["green"])
        actual_name = color_name if color_name in COLORS else "green"
        half = 60
        draw.rectangle([center - half, center - half, center + half, center + half], fill=col)
        prompt = f"a solid {actual_name} square on white background"

    elif lesson_key == "5_triangle":
        col = COLORS.get(color_name, COLORS["purple"])
        actual_name = color_name if color_name in COLORS else "purple"
        h = 75
        pts = [
            (center, center - h),
            (center - int(h * 1.05), center + int(h * 0.7)),
            (center + int(h * 1.05), center + int(h * 0.7)),
        ]
        draw.polygon(pts, fill=col)
        prompt = f"a solid {actual_name} triangle on white background"

    else:
        col = COLORS["red"]
        r = 65
        draw.ellipse([center - r, center - r, center + r, center + r], fill=col)
        prompt = "a solid red circle on white background"

    return img, prompt


def decode_tokens_to_image(vqvae: VQVAE, tokens: List[int], device: str) -> Image.Image:
    """Decodes a 256-length list of VQ-VAE codebook indices into a PIL Image."""
    clamped = [max(0, min(vqvae.codebook_size - 1, t)) for t in tokens]
    idx_tensor = torch.tensor([clamped], dtype=torch.long, device=device)
    with torch.no_grad():
        recon = vqvae.decode_from_indices(idx_tensor)
    recon_np = recon[0].detach().cpu().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
    recon_np = ((recon_np + 1.0) * 127.5).astype(np.uint8)
    img = Image.fromarray(recon_np)
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
    return img


def create_exam_card(
    target_img: Image.Image,
    learned_img: Image.Image,
    lesson_key: str,
    prompt: str,
    match_pct: float,
    step: int,
) -> Image.Image:
    """
    Creates a clear, side-by-side inspection card:
    [ TARGET GROUND TRUTH ]  vs  [ DISHA MODEL OUTPUT ]
    """
    w, h = 256, 256
    pad = 16
    header_h = 60
    card_w = w * 2 + pad * 3
    card_h = h + header_h + pad * 2

    card = Image.new("RGB", (card_w, card_h), (242, 245, 248))
    draw = ImageDraw.Draw(card)

    is_mastered = match_pct >= 98.0
    status_text = "MASTERED 🎯" if is_mastered else f"REINFORCING ({match_pct:.1f}%)"
    status_color = (20, 140, 50) if is_mastered else (200, 90, 20)

    draw.text((pad, 10), f"DISHA NURSERY CURRICULUM: LESSON '{lesson_key.upper()}'", fill=(20, 30, 50))
    draw.text((pad, 26), f"Prompt: \"{prompt}\"", fill=(80, 90, 100))
    draw.text((pad, 42), f"Token Match: {match_pct:.1f}% (Step {step}) | Status: {status_text}", fill=status_color)

    # Border & Badges for Target (Left)
    draw.rectangle([pad - 1, header_h + pad - 1, pad + w, header_h + pad + h], outline=(210, 215, 225), width=1)
    card.paste(target_img, (pad, header_h + pad))
    draw.rectangle([pad + 6, header_h + pad + 6, pad + 155, header_h + pad + 24], fill=(245, 247, 250))
    draw.text((pad + 10, header_h + pad + 8), "TARGET (Ground Truth)", fill=(40, 50, 60))

    # Border & Badges for Learned (Right)
    right_x = w + pad * 2
    draw.rectangle([right_x - 1, header_h + pad - 1, right_x + w, header_h + pad + h], outline=(210, 215, 225), width=1)
    card.paste(learned_img, (right_x, header_h + pad))
    draw.rectangle([right_x + 6, header_h + pad + 6, right_x + 185, header_h + pad + 24], fill=(245, 247, 250))
    draw.text((right_x + 10, header_h + pad + 8), "DISHA  OUTPUT  (Learned)", fill=(40, 50, 60))

    return card


# -----------------------------------------------------------------------------
# 2. Reinforcement Mastery Training Loop
# -----------------------------------------------------------------------------
LESSONS = ["1_dot", "2_line", "3_circle", "4_square", "5_triangle"]

ANCHOR_COLORS = {
    "1_dot": ["red", "blue", "green", "purple"],
    "2_line": ["blue", "red", "green", "purple"],
    "3_circle": ["red", "blue", "green", "orange"],
    "4_square": ["green", "red", "blue", "purple"],
    "5_triangle": ["purple", "red", "blue", "green"],
}


def train_reinforcement_mastery(
    lesson_key: str = "1_dot",
    target_acc: float = 98.0,
    max_steps: int = 100,
    lr: float = 2.5e-4,
    checkpoint_path: str = "checkpoints/multimodal_llm.pt",
    device: Optional[str] = None,
) -> Image.Image:
    """
    Target-driven Reinforcement Loop:
    Keeps training with AdamW until the model's visual codebook token predictions
    match the ground truth target tokens with >= target_acc percentage!
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 76)
    print(f"🎯 DISHA REINFORCEMENT MASTERY ENGINE: LESSON '{lesson_key.upper()}'")
    print(f"[*] Target Accuracy Threshold : {target_acc:.1f}%")
    print(f"[*] Maximum Allowed Steps     : {max_steps}")
    print(f"[*] Learning Rate             : {lr}")
    print(f"[*] Compute Device            : {device.upper()}")
    print("=" * 76)

    # 1. Load Foundation Pipeline
    print(f"\n[1/4] Loading foundation checkpoint: {checkpoint_path}")
    pipeline = MultimodalGeneratorPipeline.from_pretrained(checkpoint_path, device=device)
    vqvae = pipeline.vqvae.eval()
    llm = pipeline.llm.train()
    tokenizer = pipeline.tokenizer

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 2. Build Authoritative Target and Contrastive Anchors
    print(f"\n[2/4] Synthesizing authoritative target & contrastive anchors...")
    colors = ANCHOR_COLORS.get(lesson_key, ["red", "blue", "green"])
    target_img, target_prompt = render_canonical_primitive(lesson_key, colors[0])

    batch_samples = []
    for col in colors:
        img, prompt = render_canonical_primitive(lesson_key, col)
        t_img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            indices = vqvae.encode_to_indices(t_img)[0].cpu()

        text_tokens = tokenizer.encode(prompt, add_bos=False, add_eos=False)
        prefix = [tokenizer.bos_id] + text_tokens + [tokenizer.image_start_id]
        full_seq = prefix + (indices + 8000).tolist() + [tokenizer.image_end_id, tokenizer.eos_id]

        inp = torch.tensor(full_seq[:-1], dtype=torch.long)
        tgt = torch.tensor(full_seq[1:], dtype=torch.long)
        img_pos = len(prefix) - 1
        tgt[:img_pos] = -100
        tgt[img_pos + 256:] = -100

        batch_samples.append({
            "input": inp,
            "target": tgt,
            "img_pos": img_pos,
            "gt_tokens": indices,
            "prompt": prompt,
            "is_primary": (col == colors[0]),
        })

    # Prepare primary target reference for verification
    primary = batch_samples[0]
    target_gt_tokens = primary["gt_tokens"].to(device)
    primary_inp = primary["input"].unsqueeze(0).to(device)
    primary_tgt = primary["target"].unsqueeze(0).to(device)
    target_img_pos = primary["img_pos"]

    # Collate batch
    max_len = max(s["input"].size(0) for s in batch_samples)
    B = len(batch_samples)
    batch_inp = torch.full((B, max_len), 0, dtype=torch.long)
    batch_tgt = torch.full((B, max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch_samples):
        slen = s["input"].size(0)
        batch_inp[i, :slen] = s["input"]
        batch_tgt[i, :slen] = s["target"]
    batch_inp = batch_inp.to(device)
    batch_tgt = batch_tgt.to(device)

    print(f"[+] Canonical Target Prompt: \"{target_prompt}\"")
    print(f"[+] Ground Truth Visual Codebook Tokens: 256 tokens ready.")

    # 3. Iterative Reinforcement Loop
    print(f"\n[3/4] Reinforcing until target reproduction is achieved ('Jab tak same na banaye tab tak learn')...")
    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_time = time.time()
    best_match = 0.0
    final_step = 0
    pred_tokens_best = None

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        logits, _ = llm(batch_inp)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()

        final_step = step

        # Evaluate Exact Token Match on Primary Target
        with torch.no_grad():
            pred_visual = logits[0, target_img_pos : target_img_pos + 256].argmax(dim=-1) - 8000
            matched = (pred_visual == target_gt_tokens).sum().item()
            match_pct = (matched / 256.0) * 100.0

        if match_pct > best_match:
            best_match = match_pct
            pred_tokens_best = pred_visual.detach().cpu().tolist()

        status = "🎯 MASTERED!" if match_pct >= target_acc else "REINFORCING..."
        if step % 2 == 0 or step == 1 or match_pct >= target_acc:
            print(f"  [Step {step:3d}/{max_steps}] Loss: {loss.item():.4f} | Token Match: {match_pct:5.1f}% ({matched:3d}/256) | {status}")

        # Stopping Condition: Exit as soon as target reproduction is mastered
        if match_pct >= target_acc and loss.item() < 0.25:
            elapsed = time.time() - start_time
            print(f"\n[+] 🎯 TARGET MASTERED IN {step} STEPS! ({elapsed:.1f}s)")
            print(f"[+] Exact token accuracy: {match_pct:.1f}% ({matched}/256 tokens identical)")
            break

    # 4. Save Graduated Checkpoint
    print(f"\n[+] Saving reinforced model checkpoint to: {checkpoint_path}")
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = {}
    else:
        ckpt = {}

    ckpt["llm_config"] = LLMConfig()
    ckpt["vqvae_config"] = VQVAEConfig()
    ckpt["llm_state_dict"] = llm.state_dict()
    ckpt["vqvae_state_dict"] = vqvae.state_dict()
    ckpt["tokenizer_vocab"] = {"merges": tokenizer.merges}
    torch.save(ckpt, checkpoint_path)

    # 5. Decode & Generate Side-by-Side Exam Card
    print("\n[4/4] 🎓 TAKING VISUAL EXAMINATION (Aaya ya nahi test karte hain)...")
    llm.eval()
    if pred_tokens_best is None:
        pred_tokens_best = pred_visual.detach().cpu().tolist()

    learned_img = decode_tokens_to_image(vqvae, pred_tokens_best, device=device)
    exam_card = create_exam_card(
        target_img=target_img,
        learned_img=learned_img,
        lesson_key=lesson_key,
        prompt=target_prompt,
        match_pct=best_match,
        step=final_step,
    )

    exam_filename = f"exam_{lesson_key}.png"
    exam_card.save(exam_filename)
    learned_img.save(f"output_{lesson_key}.png")

    print(f"[+] Exam Comparison Card Saved : {os.path.abspath(exam_filename)}")
    print(f"[+] Standalone Output Image Saved: {os.path.abspath(f'output_{lesson_key}.png')}")
    print(f"[+] Best Match Accuracy Achieved : {best_match:.1f}%")

    # Inline display for Colab / Jupyter
    try:
        from IPython.display import display
        display(exam_card)
    except Exception:
        pass

    return exam_card


def main():
    parser = argparse.ArgumentParser(description="Disha Nursery Curriculum Training & Reinforcement")
    parser.add_argument(
        "--lesson",
        type=str,
        default="1_dot",
        choices=["1_dot", "2_line", "3_circle", "4_square", "5_triangle"],
        help="Nursery syllabus lesson to train",
    )
    parser.add_argument("--target_acc", type=float, default=98.0, help="Target token match threshold percentage (e.g. 98.0 or 100.0)")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum reinforcement steps")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / cpu)")

    args = parser.parse_args()
    train_reinforcement_mastery(
        lesson_key=args.lesson,
        target_acc=args.target_acc,
        max_steps=args.max_steps,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
