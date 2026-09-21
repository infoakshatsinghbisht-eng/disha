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
from pipeline.vector_refiner import refine_geometry_from_prompt


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
    Renders an authoritative ground-truth primitive with 4x supersampled anti-aliasing
    so that geometric boundaries smoothly quantize across the VQ-VAE codebook grid.
    """
    big = 1024
    img_big = Image.new("RGB", (big, big), (255, 255, 255))
    draw = ImageDraw.Draw(img_big)
    c = big // 2

    if lesson_key == "1_dot":
        col = COLORS.get(color_name, COLORS["red"])
        actual_name = color_name if color_name in COLORS else "red"
        r = 38 * 4
        draw.ellipse([c - r, c - r, c + r, c + r], fill=col)
        prompt = f"a solid {actual_name} dot centered on white canvas"

    elif lesson_key == "2_line":
        col = COLORS.get(color_name, COLORS["blue"])
        actual_name = color_name if color_name in COLORS else "blue"
        draw.line([(140, c), (big - 140, c)], fill=col, width=16 * 4)
        prompt = f"a clean {actual_name} horizontal line on white background"

    elif lesson_key == "3_circle":
        col = COLORS.get(color_name, COLORS["red"])
        actual_name = color_name if color_name in COLORS else "red"
        r = 290  # r = 72.5, anti-aliased to prevent flat chord boundary quantization
        draw.ellipse([c - r, c - r, c + r, c + r], fill=col)
        prompt = f"a solid {actual_name} circle on white background"

    elif lesson_key == "4_square":
        col = COLORS.get(color_name, COLORS["green"])
        actual_name = color_name if color_name in COLORS else "green"
        half = 60 * 4
        draw.rectangle([c - half, c - half, c + half, c + half], fill=col)
        prompt = f"a solid {actual_name} square on white background"

    elif lesson_key == "5_triangle":
        col = COLORS.get(color_name, COLORS["purple"])
        actual_name = color_name if color_name in COLORS else "purple"
        h = 80 * 4
        pts = [
            (c, c - h),
            (c - int(h * 1.05), c + int(h * 0.7)),
            (c + int(h * 1.05), c + int(h * 0.7)),
        ]
        draw.polygon(pts, fill=col)
        prompt = f"a solid {actual_name} triangle on white background"

    else:
        col = COLORS["red"]
        r = 290
        draw.ellipse([c - r, c - r, c + r, c + r], fill=col)
        prompt = "a solid red circle on white background"

    final_img = img_big.resize((size, size), Image.Resampling.LANCZOS)
    return final_img, prompt


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
    target_vector_img: Image.Image,
    neural_img: Image.Image,
    refined_img: Image.Image,
    lesson_key: str,
    prompt: str,
    match_pct: float,
    fg_pct: float,
    step: int,
) -> Image.Image:
    """
    Creates a 3-panel inspection card:
    [ 1. TARGET (Continuous Vector) ] vs [ 2. DISHA NEURAL (16x16 Tokens) ] vs [ 3. DISHA FINAL (Vector Refined) ]
    """
    w, h = 256, 256
    pad = 16
    header_h = 60
    card_w = w * 3 + pad * 4
    card_h = h + header_h + pad * 2

    card = Image.new("RGB", (card_w, card_h), (242, 245, 248))
    draw = ImageDraw.Draw(card)

    is_mastered = (fg_pct >= 95.0 and match_pct >= 98.0)
    status_text = "MASTERED 🎯" if is_mastered else f"REINFORCING (FG: {fg_pct:.0f}%)"
    status_color = (20, 140, 50) if is_mastered else (200, 90, 20)

    draw.text((pad, 10), f"DISHA NURSERY CURRICULUM: LESSON '{lesson_key.upper()}'", fill=(20, 30, 50))
    draw.text((pad, 26), f"Prompt: \"{prompt}\"", fill=(80, 90, 100))
    draw.text((pad, 42), f"FG Shape: {fg_pct:.1f}% | Total Match: {match_pct:.1f}% (Step {step}) | Status: {status_text}", fill=status_color)

    # Panel 1: Target Vector (Continuous Ground Truth)
    x1 = pad
    draw.rectangle([x1 - 1, header_h + pad - 1, x1 + w, header_h + pad + h], outline=(210, 215, 225), width=1)
    card.paste(target_vector_img, (x1, header_h + pad))
    draw.rectangle([x1 + 6, header_h + pad + 6, x1 + 175, header_h + pad + 24], fill=(245, 247, 250))
    draw.text((x1 + 10, header_h + pad + 8), "1. TARGET (Continuous)", fill=(40, 50, 60))

    # Panel 2: Neural Codebook Tokens
    x2 = x1 + w + pad
    draw.rectangle([x2 - 1, header_h + pad - 1, x2 + w, header_h + pad + h], outline=(210, 215, 225), width=1)
    card.paste(neural_img, (x2, header_h + pad))
    draw.rectangle([x2 + 6, header_h + pad + 6, x2 + 195, header_h + pad + 24], fill=(245, 247, 250))
    draw.text((x2 + 10, header_h + pad + 8), "2. NEURAL TOKENS (16x16)", fill=(40, 50, 60))

    # Panel 3: Vector Refined Output (Crisp & Perfect)
    x3 = x2 + w + pad
    draw.rectangle([x3 - 1, header_h + pad - 1, x3 + w, header_h + pad + h], outline=(210, 215, 225), width=1)
    card.paste(refined_img, (x3, header_h + pad))
    draw.rectangle([x3 + 6, header_h + pad + 6, x3 + 205, header_h + pad + 24], fill=(245, 247, 250))
    draw.text((x3 + 10, header_h + pad + 8), "3. DISHA VECTOR (Crisp)", fill=(40, 50, 60))

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
    lesson_key: str = "all",
    target_acc: float = 98.0,
    max_steps: int = 100,
    lr: float = 2.5e-4,
    checkpoint_path: str = "checkpoints/multimodal_llm.pt",
    device: Optional[str] = None,
) -> Image.Image:
    """
    Target-driven Reinforcement Loop:
    Trains on single lesson or ALL lessons jointly with replay anchors to prevent catastrophic forgetting.
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

    # 2. Build Authoritative Targets and Contrastive Anchors
    print(f"\n[2/4] Synthesizing authoritative targets & contrastive anchors...")
    if lesson_key == "all":
        syllabus_keys = ["2_line", "3_circle", "4_square", "5_triangle"]
    else:
        syllabus_keys = [lesson_key]

    batch_samples = []
    primary_targets = []

    with torch.no_grad():
        t_white = transform(Image.new("RGB", (256, 256), (255, 255, 255))).unsqueeze(0).to(device)
        white_indices = vqvae.encode_to_indices(t_white)[0]
        bg_token = torch.mode(white_indices).values.item()

    for l_key in syllabus_keys:
        colors = ANCHOR_COLORS.get(l_key, ["red", "blue", "green"])
        for col_idx, col in enumerate(colors):
            img, prompt = render_canonical_primitive(l_key, col)
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

            is_primary = (col_idx == 0)
            sample_idx = len(batch_samples)

            batch_samples.append({
                "input": inp,
                "target": tgt,
                "img_pos": img_pos,
                "gt_tokens": indices,
                "prompt": prompt,
                "lesson_key": l_key,
                "is_primary": is_primary,
            })

            if is_primary:
                primary_targets.append({
                    "batch_idx": sample_idx,
                    "lesson_key": l_key,
                    "prompt": prompt,
                    "img_pos": img_pos,
                    "gt_tokens": indices.to(device),
                    "fg_mask": (indices.to(device) != bg_token),
                    "total_fg": max(1, (indices.to(device) != bg_token).sum().item()),
                })

    # Collate batch
    max_len = max(s["input"].size(0) for s in batch_samples)
    B = len(batch_samples)
    batch_inp = torch.full((B, max_len), 0, dtype=torch.long)
    batch_tgt = torch.full((B, max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch_samples):
        slen = s["input"].size(0)
        batch_inp[i, :slen] = s["input"]
        batch_tgt[i, :slen] = s["target"]

    # Focal Shape Weighting
    loss_weights = torch.ones_like(batch_tgt, dtype=torch.float)
    for i, s in enumerate(batch_samples):
        indices = s["gt_tokens"].to(device)
        is_fg = (indices != bg_token)
        pos = s["img_pos"]
        loss_weights[i, pos : pos + 256][is_fg] = 5.0

    batch_inp = batch_inp.to(device)
    batch_tgt = batch_tgt.to(device)
    loss_weights = loss_weights.to(device)

    print(f"[+] Total Joint Batch Samples : {B} (Covering {len(primary_targets)} Core Shapes)")

    # 4. Iterative Reinforcement Loop
    print(f"\n[3/4] Reinforcing until simultaneous multi-shape mastery is achieved...")
    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr, weight_decay=0.01)
    criterion_elem = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    start_time = time.time()
    final_step = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        logits, _ = llm(batch_inp)
        raw_loss = criterion_elem(logits.view(-1, logits.size(-1)), batch_tgt.view(-1)).view(B, -1)
        loss = (raw_loss * loss_weights).sum() / loss_weights[batch_tgt != -100].sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()

        final_step = step

        # Evaluate all primary targets simultaneously
        all_mastered = True
        scores_summary = []

        with torch.no_grad():
            for tgt_info in primary_targets:
                b_i = tgt_info["batch_idx"]
                pos = tgt_info["img_pos"]
                gt = tgt_info["gt_tokens"]
                fg_mask = tgt_info["fg_mask"]
                tot_fg = tgt_info["total_fg"]

                pred_visual = logits[b_i, pos : pos + 256].argmax(dim=-1) - 8000
                matched = (pred_visual == gt).sum().item()
                tot_pct = (matched / 256.0) * 100.0

                fg_matched = (pred_visual[fg_mask] == gt[fg_mask]).sum().item()
                fg_pct = (fg_matched / tot_fg) * 100.0

                if target_acc >= 99.9:
                    is_p_mastered = (matched == 256)
                else:
                    is_p_mastered = (tot_pct >= target_acc and fg_pct >= 95.0)

                if not is_p_mastered:
                    all_mastered = False

                short_key = tgt_info["lesson_key"].split("_")[-1].capitalize()
                scores_summary.append(f"{short_key}: {fg_pct:.0f}%")

        summary_str = " | ".join(scores_summary)
        status = "🎯 100% ALL MASTERED!" if all_mastered else "REINFORCING..."

        if step % 2 == 0 or step == 1 or all_mastered:
            print(f"  [Step {step:3d}/{max_steps}] Loss: {loss.item():.4f} | {summary_str} | {status}")

        if all_mastered and loss.item() < 0.35:
            elapsed = time.time() - start_time
            print(f"\n[+] 🎯 ALL {len(primary_targets)} SHAPES SIMULTANEOUSLY MASTERED IN {step} STEPS! ({elapsed:.1f}s)")
            break

    # 5. Save Graduated Checkpoint
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

    # 6. Run Final Examination
    print("\n[4/4] 🎓 TAKING FINAL GRADUATION EXAMINATION...")
    llm.eval()
    from evaluate_nursery_exam import run_graduation_exam
    grad_sheet = run_graduation_exam(checkpoint_path=checkpoint_path, device=device)
    return grad_sheet


def main():
    parser = argparse.ArgumentParser(description="Disha Nursery Curriculum Training & Reinforcement")
    parser.add_argument(
        "--lesson",
        type=str,
        default="all",
        choices=["all", "1_dot", "2_line", "3_circle", "4_square", "5_triangle"],
        help="Nursery syllabus lesson to train (default: 'all' for joint multi-task mastery)",
    )
    parser.add_argument("--target_acc", type=float, default=100.0, help="Target token match threshold percentage")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum reinforcement steps")
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
