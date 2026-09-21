"""
Disha Level 2: Kindergarten Curriculum (Composite Multi-Part Objects).
Trains Disha to compose basic geometric primitives into recognizable real-world objects:
1. Pine Tree (Green Triangle Canopy + Brown Rectangle Trunk)
2. Simple House (Square Walls + Triangle Roof)
3. Smiling Emoji (Yellow Circle + 2 Dot Eyes + Curved Mouth)
4. Toy Car (Rectangle Body + 2 Circle Wheels)
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
from train_nursery_geometry import (
    render_canonical_primitive,
    decode_tokens_to_image,
    create_exam_card,
)


def render_canonical_composite(object_name: str, size: int = 256) -> Tuple[Image.Image, str]:
    """
    Renders an authoritative ground-truth composite object with 4x supersampling.
    """
    big = 1024
    img_big = Image.new("RGB", (big, big), (255, 255, 255))
    draw = ImageDraw.Draw(img_big)
    cx, cy = big // 2, big // 2

    obj = object_name.lower()

    if "tree" in obj or "pine" in obj:
        # 1. Trunk (Brown Rectangle)
        trunk_w = 70
        trunk_top = cy + 60
        trunk_bot = cy + 320
        draw.rectangle([cx - trunk_w // 2, trunk_top, cx + trunk_w // 2, trunk_bot], fill=(139, 69, 19))

        # 2. Canopy (Green Triangle)
        canopy_w = 420
        canopy_top = cy - 320
        canopy_bot = trunk_top + 40
        draw.polygon([(cx, canopy_top), (cx - canopy_w // 2, canopy_bot), (cx + canopy_w // 2, canopy_bot)], fill=(34, 160, 68))
        prompt = "a simple green pine tree on white background"

    elif "house" in obj:
        # 1. Walls (Cyan/Yellow Square)
        wall_half = 160
        wall_top = cy - 40
        wall_bot = cy + 280
        draw.rectangle([cx - wall_half, wall_top, cx + wall_half, wall_bot], fill=(56, 189, 248))

        # 2. Roof (Red Triangle)
        roof_top = cy - 320
        roof_half_w = 200
        draw.polygon([(cx, roof_top), (cx - roof_half_w, wall_top + 10), (cx + roof_half_w, wall_top + 10)], fill=(225, 30, 40))
        prompt = "a simple house with red roof on white background"

    elif "smiley" in obj or "face" in obj:
        # 1. Face (Yellow Circle)
        r = 280
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 205, 30))

        # 2. Eyes (Two Black Dots)
        eye_r = 30
        draw.ellipse([cx - 100 - eye_r, cy - 80 - eye_r, cx - 100 + eye_r, cy - 80 + eye_r], fill=(20, 20, 20))
        draw.ellipse([cx + 100 - eye_r, cy - 80 - eye_r, cx + 100 + eye_r, cy - 80 + eye_r], fill=(20, 20, 20))

        # 3. Smile (Curved Arc)
        draw.arc([cx - 120, cy - 40, cx + 120, cy + 140], start=20, end=160, fill=(20, 20, 20), width=24)
        prompt = "a cartoon yellow smiling face on white background"

    elif "car" in obj:
        # 1. Car Body (Red Rectangle + Cabin)
        body_top = cy - 20
        body_bot = cy + 140
        body_half_w = 260
        draw.rectangle([cx - body_half_w, body_top, cx + body_half_w, body_bot], fill=(230, 40, 40))
        # Cabin (Top box)
        cabin_top = cy - 140
        cabin_half_w = 140
        draw.rectangle([cx - cabin_half_w, cabin_top, cx + cabin_half_w, body_top], fill=(230, 40, 40))
        # 2. Wheels (Two Black Circles)
        wheel_r = 50
        wheel_y = body_bot
        draw.ellipse([cx - 150 - wheel_r, wheel_y - wheel_r, cx - 150 + wheel_r, wheel_y + wheel_r], fill=(30, 30, 30))
        draw.ellipse([cx + 150 - wheel_r, wheel_y - wheel_r, cx + 150 + wheel_r, wheel_y + wheel_r], fill=(30, 30, 30))
        prompt = "a simple red toy car on white background"

    else:
        # Default Tree
        trunk_w = 70
        trunk_top = cy + 60
        trunk_bot = cy + 320
        draw.rectangle([cx - trunk_w // 2, trunk_top, cx + trunk_w // 2, trunk_bot], fill=(139, 69, 19))
        canopy_w = 420
        canopy_top = cy - 320
        canopy_bot = trunk_top + 40
        draw.polygon([(cx, canopy_top), (cx - canopy_w // 2, canopy_bot), (cx + canopy_w // 2, canopy_bot)], fill=(34, 160, 68))
        prompt = "a simple green pine tree on white background"

    final_img = img_big.resize((size, size), Image.Resampling.LANCZOS)
    return final_img, prompt


def train_composite_object(
    object_name: str = "tree",
    target_acc: float = 98.0,
    max_steps: int = 35,
    lr: float = 2.5e-4,
    checkpoint_path: str = "checkpoints/multimodal_llm.pt",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 76)
    print(f"🌲 DISHA KINDERGARTEN MASTERY: COMPOSITE OBJECT '{object_name.upper()}'")
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

    # 2. Build Replay Buffer + Primary Target
    print(f"\n[2/4] Building Multi-Task Replay Buffer (Retaining Primitives + Teaching '{object_name}')...")
    batch_samples = []

    # Primary Composite Target
    target_img, target_prompt = render_canonical_composite(object_name)
    t_target = transform(target_img).unsqueeze(0).to(device)
    with torch.no_grad():
        target_gt = vqvae.encode_to_indices(t_target)[0].cpu()
        t_white = transform(Image.new("RGB", (256, 256), (255, 255, 255))).unsqueeze(0).to(device)
        bg_token = torch.mode(vqvae.encode_to_indices(t_white)[0]).values.item()

    text_tokens = tokenizer.encode(target_prompt, add_bos=False, add_eos=False)
    prefix = [tokenizer.bos_id] + text_tokens + [tokenizer.image_start_id]
    full_seq = prefix + (target_gt + 8000).tolist() + [tokenizer.image_end_id, tokenizer.eos_id]

    inp = torch.tensor(full_seq[:-1], dtype=torch.long)
    tgt = torch.tensor(full_seq[1:], dtype=torch.long)
    img_pos = len(prefix) - 1
    tgt[:img_pos] = -100
    tgt[img_pos + 256:] = -100

    batch_samples.append({
        "input": inp,
        "target": tgt,
        "img_pos": img_pos,
        "gt_tokens": target_gt,
        "prompt": target_prompt,
        "is_primary": True,
    })

    # Add Replay Anchors (Primitives + all previously mastered composite objects)
    replay_items = [
        ("primitive", "2_line"),
        ("primitive", "3_circle"),
        ("primitive", "4_square"),
        ("primitive", "5_triangle"),
    ]
    learned_order = ["tree", "house", "smiley", "car"]
    if object_name in learned_order:
        idx = learned_order.index(object_name)
        for prev_obj in learned_order[:idx]:
            replay_items.append(("composite", prev_obj))

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
            "input": inp_p,
            "target": tgt_p,
            "img_pos": p_img_pos,
            "gt_tokens": indices_p,
            "prompt": p_prompt,
            "is_primary": False,
        })

    target_fg_mask = (target_gt.to(device) != bg_token)
    total_fg = max(1, target_fg_mask.sum().item())
    print(f"[+] Replay Buffer Size : {len(batch_samples)} (1 Composite Target + {len(replay_items)} Replay Anchors)")
    print(f"[+] Canonical Prompt   : \"{target_prompt}\"")
    print(f"[+] Foreground Tokens  : {total_fg} shape tokens")

    # 3. Micro-Batch Reinforcement Loop
    print(f"\n[3/4] Reinforcing composite hierarchy with zero-OOM gradient accumulation...")
    optimizer = torch.optim.AdamW(llm.parameters(), lr=lr, weight_decay=0.01)
    criterion_elem = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    num_samples = len(batch_samples)
    best_fg_match = 0.0
    best_total_match = 0.0
    pred_tokens_best = None
    final_step = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        total_loss = 0.0
        primary_pred = None

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
                    primary_pred = logits_s[0, s_pos : s_pos + 256].argmax(dim=-1) - 8000

        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()
        final_step = step

        # Evaluate Composite Target Accuracy
        with torch.no_grad():
            matched = (primary_pred == target_gt.to(device)).sum().item()
            tot_pct = (matched / 256.0) * 100.0

            fg_matched = (primary_pred[target_fg_mask] == target_gt.to(device)[target_fg_mask]).sum().item()
            fg_pct = (fg_matched / total_fg) * 100.0

        if fg_pct > best_fg_match or (fg_pct == best_fg_match and tot_pct > best_total_match):
            best_fg_match = fg_pct
            best_total_match = tot_pct
            pred_tokens_best = primary_pred.detach().cpu().tolist()

        is_mastered = (matched == 256) if target_acc >= 99.9 else (tot_pct >= target_acc and fg_pct >= 95.0)
        status = "🎯 100% MASTERED!" if matched == 256 else ("🎯 MASTERED!" if is_mastered else f"REINFORCING (FG: {fg_pct:.0f}%)...")

        if step % 2 == 0 or step == 1 or is_mastered:
            print(f"  [Step {step:3d}/{max_steps}] Loss: {total_loss / num_samples:.4f} | FG Shape: {fg_pct:5.1f}% ({fg_matched:2d}/{total_fg}) | Total: {tot_pct:5.1f}% ({matched:3d}/256) | {status}")

        if is_mastered and (total_loss / num_samples) < 0.35:
            elapsed = time.time() - start_time
            print(f"\n[+] 🎯 COMPOSITE OBJECT '{object_name.upper()}' MASTERED IN {step} STEPS! ({elapsed:.1f}s)")
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

    # 5. Render 3-Panel Inspection Card
    print("\n[4/4] 🎓 RENDERING 3-PANEL EXAMINATION CARD...")
    llm.eval()
    if pred_tokens_best is None:
        pred_tokens_best = primary_pred.detach().cpu().tolist()

    learned_img = decode_tokens_to_image(vqvae, pred_tokens_best, device=device)
    refined_img = refine_geometry_from_prompt(learned_img, target_prompt)

    exam_card = create_exam_card(
        target_vector_img=target_img,
        neural_img=learned_img,
        refined_img=refined_img,
        lesson_key=f"kindergarten_{object_name}",
        prompt=target_prompt,
        match_pct=best_total_match,
        fg_pct=best_fg_match,
        step=final_step,
    )

    card_name = f"exam_{object_name}.png"
    exam_card.save(card_name)
    refined_img.save(f"output_{object_name}.png")
    learned_img.save(f"output_{object_name}_neural.png")

    print(f"[+] 3-Panel Exam Card Saved     : {os.path.abspath(card_name)}")
    print(f"[+] Crisp Vector Output Saved   : {os.path.abspath(f'output_{object_name}.png')}")
    print(f"\n[💡] Colab Notebook me card dekhne ke liye run karein:")
    print(f"     from IPython.display import Image, display; display(Image('{card_name}'))\n")

    return exam_card


def main():
    parser = argparse.ArgumentParser(description="Disha Kindergarten Composite Object Trainer")
    parser.add_argument(
        "--object",
        type=str,
        default="tree",
        choices=["tree", "house", "smiley", "car"],
        help="Composite object to train (default: 'tree')",
    )
    parser.add_argument("--target_acc", type=float, default=100.0, help="Target accuracy threshold")
    parser.add_argument("--max_steps", type=int, default=35, help="Maximum reinforcement steps")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")

    args = parser.parse_args()
    train_composite_object(
        object_name=args.object,
        target_acc=args.target_acc,
        max_steps=args.max_steps,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
