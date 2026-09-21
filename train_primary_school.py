"""
Disha Level 3: Primary School Curriculum (Multi-Object Scene Composition).
Trains Disha to compose multiple composite objects and environmental elements together on a single canvas:
1. House next to a Green Tree ("house_tree")
2. House under Bright Yellow Sun ("house_sun")
3. Red Car next to a House ("car_house")
4. Green Tree and Red Car under Sun ("tree_car")
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
from train_kindergarten import (
    render_canonical_composite,
)
from train_nursery_geometry import (
    render_canonical_primitive,
    decode_tokens_to_image,
    create_exam_card,
)


def render_canonical_scene(scene_name: str, size: int = 256) -> Tuple[Image.Image, str]:
    """
    Renders an authoritative ground-truth multi-object scene with 4x supersampling.
    """
    big = 1024
    img_big = Image.new("RGB", (big, big), (255, 255, 255))
    draw = ImageDraw.Draw(img_big)
    cx, cy = big // 2, big // 2

    sn = scene_name.lower()

    if "house" in sn and "tree" in sn:
        # Scene 1: House on Left, Tree on Right
        # 1. House (Left)
        draw.rectangle([cx - 380, cy + 20, cx - 80, cy + 320], fill=(56, 189, 248))
        draw.polygon([(cx - 230, cy - 180), (cx - 410, cy + 30), (cx - 50, cy + 30)], fill=(225, 30, 40))
        # 2. Tree (Right)
        draw.rectangle([cx + 190, cy + 100, cx + 250, cy + 320], fill=(139, 69, 19))
        draw.polygon([(cx + 220, cy - 200), (cx + 80, cy + 120), (cx + 360, cy + 120)], fill=(34, 160, 68))
        prompt = "a house next to a green tree on white background"

    elif "house" in sn and ("sun" in sn or "sky" in sn):
        # Scene 2: House under Bright Yellow Sun
        # 1. Sun (Top-Right)
        draw.ellipse([cx + 220, cy - 380, cx + 380, cy - 220], fill=(250, 205, 30))
        # 2. House (Center-Bottom)
        draw.rectangle([cx - 160, cy, cx + 160, cy + 300], fill=(56, 189, 248))
        draw.polygon([(cx, cy - 220), (cx - 200, cy + 10), (cx + 200, cy + 10)], fill=(225, 30, 40))
        prompt = "a house under a bright yellow sun on white background"

    elif ("car" in sn or "auto" in sn) and "house" in sn:
        # Scene 3: House on Left, Red Car on Right
        # 1. House (Left)
        draw.rectangle([cx - 400, cy - 40, cx - 120, cy + 260], fill=(56, 189, 248))
        draw.polygon([(cx - 260, cy - 240), (cx - 430, cy - 30), (cx - 90, cy - 30)], fill=(225, 30, 40))
        # 2. Car (Right)
        draw.rectangle([cx + 60, cy + 100, cx + 420, cy + 230], fill=(230, 40, 40))
        draw.rectangle([cx + 150, cy + 10, cx + 330, cy + 100], fill=(230, 40, 40))
        draw.ellipse([cx + 120 - 36, cy + 230 - 36, cx + 120 + 36, cy + 230 + 36], fill=(30, 30, 30))
        draw.ellipse([cx + 360 - 36, cy + 230 - 36, cx + 360 + 36, cy + 230 + 36], fill=(30, 30, 30))
        prompt = "a red car next to a house on white background"

    elif ("tree" in sn or "pine" in sn) and "car" in sn:
        # Scene 4: Tree on Left, Red Car on Bottom-Right, Sun Top-Right
        # 1. Sun (Top-Right)
        draw.ellipse([cx + 240, cy - 380, cx + 380, cy - 240], fill=(250, 205, 30))
        # 2. Tree (Left)
        draw.rectangle([cx - 280, cy + 80, cx - 220, cy + 300], fill=(139, 69, 19))
        draw.polygon([(cx - 250, cy - 220), (cx - 390, cy + 100), (cx - 110, cy + 100)], fill=(34, 160, 68))
        # 3. Car (Bottom-Right)
        draw.rectangle([cx + 60, cy + 140, cx + 420, cy + 260], fill=(230, 40, 40))
        draw.rectangle([cx + 150, cy + 50, cx + 330, cy + 140], fill=(230, 40, 40))
        draw.ellipse([cx + 120 - 36, cy + 260 - 36, cx + 120 + 36, cy + 260 + 36], fill=(30, 30, 30))
        draw.ellipse([cx + 360 - 36, cy + 260 - 36, cx + 360 + 36, cy + 260 + 36], fill=(30, 30, 30))
        prompt = "a green tree and red car under yellow sun"

    else:
        # Default: House + Tree
        draw.rectangle([cx - 380, cy + 20, cx - 80, cy + 320], fill=(56, 189, 248))
        draw.polygon([(cx - 230, cy - 180), (cx - 410, cy + 30), (cx - 50, cy + 30)], fill=(225, 30, 40))
        draw.rectangle([cx + 190, cy + 100, cx + 250, cy + 320], fill=(139, 69, 19))
        draw.polygon([(cx + 220, cy - 200), (cx + 80, cy + 120), (cx + 360, cy + 120)], fill=(34, 160, 68))
        prompt = "a house next to a green tree on white background"

    final_img = img_big.resize((size, size), Image.Resampling.LANCZOS)
    return final_img, prompt


ALL_PRIMARY_SCENES = ["house_tree", "house_sun", "car_house", "tree_car"]


def train_primary_school(
    scene_name: str = "all",
    target_acc: float = 98.0,
    max_steps: int = 35,
    lr: float = 2.5e-4,
    checkpoint_path: str = "checkpoints/multimodal_llm.pt",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 76)
    print(f"🏡 DISHA PRIMARY SCHOOL MASTERY: SCENE '{scene_name.upper()}'")
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

    # 2. Build Replay Buffer + Primary Target Scenes
    print(f"\n[2/4] Building Multi-Task Replay Buffer (Nursery + Kindergarten + Primary Scenes)...")
    batch_samples = []

    t_white = transform(Image.new("RGB", (256, 256), (255, 255, 255))).unsqueeze(0).to(device)
    with torch.no_grad():
        bg_token = torch.mode(vqvae.encode_to_indices(t_white)[0]).values.item()

    # Determine primary targets
    if scene_name == "all":
        primary_scene_keys = ALL_PRIMARY_SCENES
    else:
        primary_scene_keys = [scene_name]

    for s_key in primary_scene_keys:
        target_img, target_prompt = render_canonical_scene(s_key)
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

    # Add Replay Anchors (4 Primitives + 4 Kindergarten Objects)
    replay_items = [
        ("primitive", "2_line"),
        ("primitive", "3_circle"),
        ("primitive", "4_square"),
        ("primitive", "5_triangle"),
        ("composite", "tree"),
        ("composite", "house"),
        ("composite", "smiley"),
        ("composite", "car"),
    ]

    for kind, key in replay_items:
        if kind == "primitive":
            img_p, p_prompt = render_canonical_primitive(key)
        else:
            img_p, p_prompt = render_canonical_composite(key)

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
    print(f"[+] Total Multi-Task Batch Size: {num_samples} ({len(primary_scene_keys)} Primary Scenes + {len(replay_items)} Replay Anchors)")

    # 3. Micro-Batch Reinforcement Loop
    print(f"\n[3/4] Reinforcing multi-object spatial compositions with zero-OOM gradient accumulation...")
    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr, weight_decay=0.01)
    criterion_elem = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    best_mean_fg = 0.0
    final_step = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        total_loss = 0.0
        primary_preds = {}

        for s in batch_samples:
            inp_s = s["input"].unsqueeze(0).to(device)
            tgt_s = s["target"].unsqueeze(0).to(device)
            s_pos = s["img_pos"]
            gt_toks = s["gt_tokens"].to(device)

            logits_s, _ = llm(inp_s)
            raw_loss_s = criterion_elem(logits_s.view(-1, logits_s.size(-1)), tgt_s.view(-1))

            is_fg = (gt_toks != bg_token)
            w_s = torch.ones_like(tgt_s.view(-1), dtype=torch.float)
            w_s[s_pos : s_pos + 256][is_fg] = 5.0

            loss_s = (raw_loss_s * w_s).sum() / (w_s[tgt_s.view(-1) != -100].sum() * num_samples)
            loss_s.backward()
            total_loss += loss_s.item() * num_samples

            if s["is_primary"]:
                with torch.no_grad():
                    primary_preds[s["key"]] = (logits_s[0, s_pos : s_pos + 256].argmax(dim=-1) - 8000).detach().cpu()

        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()
        final_step = step

        # Evaluate Primary Scenes Accuracy
        fg_accs = []
        tot_accs = []
        with torch.no_grad():
            for s in batch_samples:
                if s["is_primary"]:
                    pred_t = primary_preds[s["key"]]
                    gt_t = s["gt_tokens"]
                    tot_m = (pred_t == gt_t).sum().item()
                    tot_accs.append((tot_m / 256.0) * 100.0)

                    fg_m = (gt_t != bg_token)
                    tot_fg = max(1, fg_m.sum().item())
                    fg_match_cnt = (pred_t[fg_m] == gt_t[fg_m]).sum().item()
                    fg_accs.append((fg_match_cnt / tot_fg) * 100.0)

        mean_fg = sum(fg_accs) / len(fg_accs)
        mean_tot = sum(tot_accs) / len(tot_accs)

        if mean_fg > best_mean_fg:
            best_mean_fg = mean_fg

        is_mastered = (mean_fg >= 99.0 and mean_tot >= 99.0) if target_acc >= 99.9 else (mean_fg >= 95.0 and mean_tot >= target_acc)
        status = "🎯 100% MASTERED!" if mean_fg >= 99.9 else ("🎯 MASTERED!" if is_mastered else f"REINFORCING (FG: {mean_fg:.0f}%)...")

        if step % 2 == 0 or step == 1 or is_mastered:
            print(f"  [Step {step:3d}/{max_steps}] Loss: {total_loss / num_samples:.4f} | Mean FG: {mean_fg:5.1f}% | Mean Total: {mean_tot:5.1f}% | {status}")

        if is_mastered and (total_loss / num_samples) < 0.35:
            elapsed = time.time() - start_time
            print(f"\n[+] 🎯 PRIMARY SCHOOL SCENES MASTERED IN {step} STEPS! ({elapsed:.1f}s)")
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
    print("\n[4/4] 🎓 RENDERING PRIMARY SCHOOL INSPECTION CARDS...")
    llm.eval()
    for s in batch_samples:
        if s["is_primary"]:
            s_key = s["key"]
            pred_toks = primary_preds[s_key].tolist()
            learned_img = decode_tokens_to_image(vqvae, pred_toks, device=device)
            refined_img = refine_geometry_from_prompt(learned_img, s["prompt"])

            gt_t = s["gt_tokens"]
            pred_t = primary_preds[s_key]
            fg_m = (gt_t != bg_token)
            tot_fg = max(1, fg_m.sum().item())
            fg_cnt = (pred_t[fg_m] == gt_t[fg_m]).sum().item()
            fg_p = (fg_cnt / tot_fg) * 100.0
            tot_p = ((pred_t == gt_t).sum().item() / 256.0) * 100.0

            card = create_exam_card(
                target_vector_img=s["target_img"],
                neural_img=learned_img,
                refined_img=refined_img,
                lesson_key=f"primary_{s_key}",
                prompt=s["prompt"],
                match_pct=tot_p,
                fg_pct=fg_p,
                step=final_step,
            )
            card_name = f"exam_scene_{s_key}.png"
            card.save(card_name)
            refined_img.save(f"output_scene_{s_key}.png")
            print(f"[+] Exam Card Saved: {card_name}")

    print("\n[+] Primary School Joint Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Disha Primary School Multi-Object Scene Trainer")
    parser.add_argument(
        "--scene",
        type=str,
        default="all",
        choices=["all", "house_tree", "house_sun", "car_house", "tree_car"],
        help="Primary school scene to train (default: 'all')",
    )
    parser.add_argument("--target_acc", type=float, default=100.0, help="Target accuracy threshold")
    parser.add_argument("--max_steps", type=int, default=35, help="Maximum reinforcement steps")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")

    args = parser.parse_args()
    train_primary_school(
        scene_name=args.scene,
        target_acc=args.target_acc,
        max_steps=args.max_steps,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
