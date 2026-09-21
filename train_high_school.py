"""
Disha Level 4: High School Curriculum (Environmental Landscapes & Creative Styling).
Trains Disha to break free from plain white backgrounds and synthesize realistic
two-tone environments, atmospheric lighting, and color-styled objects:
1. Sunny Day Landscape ("landscape_day")
2. Peaceful Night Landscape ("landscape_night")
3. Blue Sports Car on the Road ("car_road")
4. Customized Cottage Colors ("cottage_colors")
"""

import os
import time
import argparse
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torchvision.transforms as T

from config import LLMConfig, VQVAEConfig
from tokenizer.text_tokenizer import ByteTokenizer
from model.transformer import MultimodalTransformer
from vqvae.model import VQVAE
from pipeline.sampler import MultimodalGeneratorPipeline
from pipeline.vector_refiner import refine_geometry_from_prompt
from train_primary_school import (
    render_canonical_scene,
)
from train_kindergarten import (
    render_canonical_composite,
)
from train_nursery_geometry import (
    render_canonical_primitive,
    decode_tokens_to_image,
    create_exam_card,
)


def render_canonical_landscape(scene_name: str, size: int = 256) -> Tuple[Image.Image, str]:
    """
    Renders an authoritative ground-truth atmospheric landscape with 4x supersampling.
    """
    big = 1024
    img_big = Image.new("RGB", (big, big), (255, 255, 255))
    draw = ImageDraw.Draw(img_big)
    cx, cy = big // 2, big // 2

    sn = scene_name.lower()

    if "night" in sn or "moon" in sn:
        # Scene 2: Midnight Navy Sky + Dark Ground + Moon + Glowing Window
        draw.rectangle([0, 0, big, int(big * 0.65)], fill=(15, 23, 42))
        draw.rectangle([0, int(big * 0.65), big, big], fill=(30, 41, 59))
        draw.ellipse([cx + 240, cy - 380, cx + 380, cy - 240], fill=(254, 240, 138))
        # House
        draw.rectangle([cx - 180, cy + 60, cx + 180, cy + 320], fill=(51, 65, 85))
        draw.polygon([(cx, cy - 140), (cx - 210, cy + 70), (cx + 210, cy + 70)], fill=(147, 51, 234))
        draw.rectangle([cx - 50, cy + 140, cx + 50, cy + 240], fill=(250, 204, 21))
        prompt = "a house under a glowing moon in dark night sky"

    elif "road" in sn or ("car" in sn and ("street" in sn or "blue" in sn)):
        # Scene 3: Sky + Grass + Charcoal Grey Road + Blue Sports Car
        draw.rectangle([0, 0, big, int(big * 0.50)], fill=(125, 211, 252))
        draw.rectangle([0, int(big * 0.50), big, int(big * 0.70)], fill=(74, 222, 128))
        draw.rectangle([0, int(big * 0.70), big, big], fill=(71, 85, 105))
        draw.rectangle([0, int(big * 0.84), big, int(big * 0.86)], fill=(250, 204, 21))
        draw.ellipse([cx + 260, cy - 380, cx + 400, cy - 240], fill=(250, 205, 30))
        # Tree on Grass
        draw.rectangle([cx - 300, cy + 40, cx - 260, cy + 190], fill=(139, 69, 19))
        draw.polygon([(cx - 280, cy - 120), (cx - 390, cy + 60), (cx - 170, cy + 60)], fill=(34, 160, 68))
        # Blue Car on Road
        draw.rectangle([cx + 40, cy + 110, cx + 420, cy + 240], fill=(37, 99, 235))
        draw.rectangle([cx + 140, cy + 20, cx + 320, cy + 110], fill=(37, 99, 235))
        draw.ellipse([cx + 120 - 36, cy + 240 - 36, cx + 120 + 36, cy + 240 + 36], fill=(30, 30, 30))
        draw.ellipse([cx + 340 - 36, cy + 240 - 36, cx + 340 + 36, cy + 240 + 36], fill=(30, 30, 30))
        prompt = "a blue sports car on grey road with green trees under sunny sky"

    elif "cottage" in sn or "purple" in sn:
        # Scene 4: Blue Sky + Green Grass + Pastel Yellow Cottage + Purple Roof
        draw.rectangle([0, 0, big, int(big * 0.60)], fill=(125, 211, 252))
        draw.rectangle([0, int(big * 0.60), big, big], fill=(74, 222, 128))
        draw.ellipse([cx + 260, cy - 380, cx + 400, cy - 240], fill=(250, 205, 30))
        # Yellow Cottage with Purple Roof
        draw.rectangle([cx - 400, cy + 50, cx - 120, cy + 320], fill=(254, 240, 138))
        draw.polygon([(cx - 260, cy - 150), (cx - 430, cy + 60), (cx - 90, cy + 60)], fill=(168, 85, 247))
        # Tree
        draw.rectangle([cx + 200, cy + 100, cx + 260, cy + 320], fill=(139, 69, 19))
        draw.polygon([(cx + 230, cy - 180), (cx + 90, cy + 120), (cx + 370, cy + 120)], fill=(34, 160, 68))
        prompt = "a yellow cottage with purple roof and green tree on sunny day"

    else:
        # Scene 1 Default: Blue Sky + Green Grass + House + Tree + Sun
        draw.rectangle([0, 0, big, int(big * 0.60)], fill=(125, 211, 252))
        draw.rectangle([0, int(big * 0.60), big, big], fill=(74, 222, 128))
        draw.ellipse([cx + 260, cy - 380, cx + 400, cy - 240], fill=(250, 205, 30))
        # House
        draw.rectangle([cx - 400, cy + 50, cx - 120, cy + 320], fill=(254, 243, 199))
        draw.polygon([(cx - 260, cy - 150), (cx - 430, cy + 60), (cx - 90, cy + 60)], fill=(225, 29, 72))
        # Tree
        draw.rectangle([cx + 200, cy + 100, cx + 260, cy + 320], fill=(139, 69, 19))
        draw.polygon([(cx + 230, cy - 180), (cx + 90, cy + 120), (cx + 370, cy + 120)], fill=(34, 160, 68))
        prompt = "a house and green tree on green grass under blue sky"

    final_img = img_big.resize((size, size), Image.Resampling.LANCZOS)
    return final_img, prompt


ALL_HIGH_SCHOOL_SCENES = ["landscape_day", "landscape_night", "car_road", "cottage_colors"]


def train_high_school(
    scene_name: str = "all",
    target_acc: float = 96.0,
    max_steps: int = 35,
    lr: float = 2.5e-4,
    checkpoint_path: str = "checkpoints/multimodal_llm.pt",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 76)
    print(f"🌄 DISHA HIGH SCHOOL MASTERY: ATMOSPHERIC SCENE '{scene_name.upper()}'")
    print(f"[*] Target Accuracy Threshold : {target_acc:.1f}%")
    print(f"[*] Maximum Allowed Steps     : {max_steps}")
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

    # 2. Build Replay Buffer + High School Primary Targets
    print(f"\n[2/4] Building Multi-Task Replay Buffer (Nursery + Kindergarten + Primary + High School)...")
    batch_samples = []

    # Primary High School Targets
    if scene_name == "all":
        primary_keys = ALL_HIGH_SCHOOL_SCENES
    else:
        primary_keys = [scene_name]

    for s_key in primary_keys:
        target_img, target_prompt = render_canonical_landscape(s_key)
        t_target = transform(target_img).unsqueeze(0).to(device)
        with torch.no_grad():
            target_gt = vqvae.encode_to_indices(t_target)[0].cpu()

        text_tokens = tokenizer.encode(target_prompt, add_bos=False, add_eos=False)
        prefix = [tokenizer.bos_id] + text_tokens + [tokenizer.image_start_id]
        full_seq = prefix + (target_gt + 8000).tolist() + [tokenizer.image_end_id, tokenizer.eos_id]

        inp = torch.tensor(full_seq[:-1], dtype=torch.long)
        tgt = torch.tensor(full_seq[1:], dtype=torch.long)
        img_pos = len(prefix) - 1
        tgt[:img_pos] = -100
        tgt[img_pos + 256:] = -100

        batch_samples.append({
            "key": s_key,
            "input": inp,
            "target": tgt,
            "img_pos": img_pos,
            "gt_tokens": target_gt,
            "prompt": target_prompt,
            "target_img": target_img,
            "is_primary": True,
        })

    # Comprehensive Replay Buffer (Nursery + Kindergarten + Primary School)
    replay_items = [
        ("primitive", "2_line"),
        ("primitive", "3_circle"),
        ("primitive", "4_square"),
        ("primitive", "5_triangle"),
        ("composite", "tree"),
        ("composite", "house"),
        ("composite", "smiley"),
        ("composite", "car"),
        ("primary_scene", "house_tree"),
        ("primary_scene", "house_sun"),
        ("primary_scene", "car_house"),
        ("primary_scene", "tree_car"),
    ]

    for kind, key in replay_items:
        if kind == "primitive":
            img_p, p_prompt = render_canonical_primitive(key)
        elif kind == "composite":
            img_p, p_prompt = render_canonical_composite(key)
        else:
            img_p, p_prompt = render_canonical_scene(key)

        t_p = transform(img_p).unsqueeze(0).to(device)
        with torch.no_grad():
            indices_p = vqvae.encode_to_indices(t_p)[0].cpu()

        t_toks = tokenizer.encode(p_prompt, add_bos=False, add_eos=False)
        p_prefix = [tokenizer.bos_id] + t_toks + [tokenizer.image_start_id]
        p_full = p_prefix + (indices_p + 8000).tolist() + [tokenizer.image_end_id, tokenizer.eos_id]

        inp_p = torch.tensor(p_full[:-1], dtype=torch.long)
        tgt_p = torch.tensor(p_full[1:], dtype=torch.long)
        p_img_pos = len(p_prefix) - 1
        tgt_p[:p_img_pos] = -100
        tgt_p[p_img_pos + 256:] = -100

        batch_samples.append({
            "key": key,
            "input": inp_p,
            "target": tgt_p,
            "img_pos": p_img_pos,
            "gt_tokens": indices_p,
            "prompt": p_prompt,
            "target_img": img_p,
            "is_primary": False,
        })

    num_samples = len(batch_samples)
    print(f"[+] Total Multi-Task Batch Size: {num_samples} ({len(primary_keys)} High School Scenes + {len(replay_items)} Replay Anchors)")

    # 3. Micro-Batch Reinforcement Loop
    print(f"\n[3/4] Reinforcing atmospheric landscapes with zero-OOM gradient accumulation...")
    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr, weight_decay=0.01)
    criterion_elem = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    best_mean_tot = 0.0
    final_step = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        total_loss = 0.0
        primary_preds = {}

        for s in batch_samples:
            inp_s = s["input"].unsqueeze(0).to(device)
            tgt_s = s["target"].unsqueeze(0).to(device)
            s_pos = s["img_pos"]

            logits_s, _ = llm(inp_s)
            raw_loss_s = criterion_elem(logits_s.view(-1, logits_s.size(-1)), tgt_s.view(-1))

            loss_s = raw_loss_s.sum() / ((tgt_s.view(-1) != -100).sum() * num_samples)
            loss_s.backward()
            total_loss += loss_s.item() * num_samples

            if s["is_primary"]:
                with torch.no_grad():
                    primary_preds[s["key"]] = (logits_s[0, s_pos : s_pos + 256].argmax(dim=-1) - 8000).detach().cpu()

        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()
        final_step = step

        # Evaluate Accuracy
        tot_accs = []
        with torch.no_grad():
            for s in batch_samples:
                if s["is_primary"]:
                    pred_t = primary_preds[s["key"]]
                    gt_t = s["gt_tokens"]
                    tot_m = (pred_t == gt_t).sum().item()
                    tot_accs.append((tot_m / 256.0) * 100.0)

        mean_tot = sum(tot_accs) / len(tot_accs)
        if mean_tot > best_mean_tot:
            best_mean_tot = mean_tot

        is_mastered = (mean_tot >= target_acc)
        status = "🎯 MASTERED!" if is_mastered else f"REINFORCING ({mean_tot:.1f}%)..."

        if step % 2 == 0 or step == 1 or is_mastered:
            print(f"  [Step {step:3d}/{max_steps}] Loss: {total_loss / num_samples:.4f} | Canvas Fidelity: {mean_tot:5.1f}% | {status}")

        if is_mastered and (total_loss / num_samples) < 0.35:
            elapsed = time.time() - start_time
            print(f"\n[+] 🎯 HIGH SCHOOL ATMOSPHERIC SCENES MASTERED IN {step} STEPS! ({elapsed:.1f}s)")
            break

    # 4. Save Checkpoint
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

    # 5. Render Inspection Cards
    print("\n[4/4] 🎓 RENDERING HIGH SCHOOL INSPECTION CARDS...")
    llm.eval()
    for s in batch_samples:
        if s["is_primary"]:
            s_key = s["key"]
            pred_toks = primary_preds[s_key].tolist()
            learned_img = decode_tokens_to_image(vqvae, pred_toks, device=device)
            refined_img = refine_geometry_from_prompt(learned_img, s["prompt"])

            gt_t = s["gt_tokens"]
            pred_t = primary_preds[s_key]
            tot_p = ((pred_t == gt_t).sum().item() / 256.0) * 100.0

            card = create_exam_card(
                target_vector_img=s["target_img"],
                neural_img=learned_img,
                refined_img=refined_img,
                lesson_key=f"high_school_{s_key}",
                prompt=s["prompt"],
                match_pct=tot_p,
                fg_pct=tot_p,
                step=final_step,
            )
            card_name = f"exam_landscape_{s_key}.png"
            card.save(card_name)
            refined_img.save(f"output_landscape_{s_key}.png")
            print(f"[+] Exam Card Saved: {card_name}")

    print("\n[+] High School Joint Atmospheric Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Disha High School Atmospheric Landscape Trainer")
    parser.add_argument(
        "--scene",
        type=str,
        default="all",
        choices=["all", "landscape_day", "landscape_night", "car_road", "cottage_colors"],
        help="High school landscape to train (default: 'all')",
    )
    parser.add_argument("--target_acc", type=float, default=96.0, help="Target accuracy threshold")
    parser.add_argument("--max_steps", type=int, default=35, help="Maximum reinforcement steps")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")

    args = parser.parse_args()
    train_high_school(
        scene_name=args.scene,
        target_acc=args.target_acc,
        max_steps=args.max_steps,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
