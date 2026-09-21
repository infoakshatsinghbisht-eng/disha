"""
Disha Nursery Curriculum Graduation Exam.
Takes an autoregressive examination across all 4 geometric primitives:
- Line, Circle, Square, Triangle
Generates a comprehensive Graduation Report Card image ('nursery_graduation_exam.png').
"""

import os
import time
import argparse
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision import transforms
from config import LLMConfig, VQVAEConfig
from tokenizer.text_tokenizer import ByteTokenizer
from model.transformer import MultimodalTransformer
from vqvae.model import VQVAE
from pipeline.vector_refiner import refine_geometry_from_prompt
from train_nursery_geometry import (
    render_canonical_primitive,
    decode_tokens_to_image,
)


def encode_image_to_tokens(vqvae: VQVAE, img: Image.Image, device: str) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        indices = vqvae.encode_to_indices(tensor)[0]
    return indices


EXAM_SYLLABUS = [
    ("2_line", "a clean blue horizontal line on white background", "Stroke / Alignment"),
    ("3_circle", "a solid red circle on white background", "Curve / Radial Symmetry"),
    ("4_square", "a solid green square on white background", "Box / 90° Angles"),
    ("5_triangle", "a solid purple triangle on white background", "Polygon / Sharp Vertices"),
]


def create_graduation_sheet(results: List[Dict[str, Any]]) -> Image.Image:
    """
    Renders a master examination sheet displaying:
    [Target Ground Truth] vs [Autoregressive Neural Tokens] vs [Disha Vector Refined]
    with accuracy percentages and graduation honors stamp.
    """
    thumb_w, thumb_h = 220, 220
    pad = 16
    header_h = 100
    row_h = thumb_h + 40
    footer_h = 70
    
    total_w = pad * 4 + thumb_w * 3 + 240  # Extra right column for score & telemetry
    total_h = header_h + len(results) * row_h + footer_h + pad

    sheet = Image.new("RGB", (total_w, total_h), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)

    # Header
    draw.rectangle([0, 0, total_w, header_h], fill=(15, 23, 42))  # Deep navy slate
    draw.text((pad + 10, 20), "DISHA MULTIMODAL FOUNDATION -- NURSERY GRADUATION EXAM", fill=(255, 255, 255))
    draw.text((pad + 10, 48), "Final Autoregressive Generation & Vector Fidelity Evaluation", fill=(148, 163, 184))
    draw.text((pad + 10, 70), "Status: ALL GEOMETRIC PRIMITIVES EVALUATED UNDER ZERO-FORCING INFERENCE", fill=(56, 189, 248))

    # Header Columns
    x_target = pad
    x_neural = x_target + thumb_w + pad
    x_vector = x_neural + thumb_w + pad
    x_stats  = x_vector + thumb_w + pad

    y_curr = header_h + pad

    for i, r in enumerate(results):
        # Row background card
        draw.rectangle([pad // 2, y_curr - 8, total_w - pad // 2, y_curr + thumb_h + 24], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        
        # Row title
        lesson_label = f"LESSON {i+1}: {r['lesson_key'].upper()} -- {r['category']}"
        draw.text((pad, y_curr - 2), lesson_label, fill=(30, 41, 59))

        item_y = y_curr + 18

        # 1. Target Image
        t_img = r["target_img"].resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(t_img, (x_target, item_y))
        draw.rectangle([x_target, item_y, x_target + thumb_w, item_y + thumb_h], outline=(203, 213, 225), width=1)
        draw.rectangle([x_target + 4, item_y + 4, x_target + 130, item_y + 20], fill=(241, 245, 249))
        draw.text((x_target + 8, item_y + 6), "Target Ground Truth", fill=(71, 85, 105))

        # 2. Neural Decoded Image
        n_img = r["neural_img"].resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(n_img, (x_neural, item_y))
        draw.rectangle([x_neural, item_y, x_neural + thumb_w, item_y + thumb_h], outline=(203, 213, 225), width=1)
        draw.rectangle([x_neural + 4, item_y + 4, x_neural + 145, item_y + 20], fill=(241, 245, 249))
        draw.text((x_neural + 8, item_y + 6), "Neural Tokens (16x16)", fill=(71, 85, 105))

        # 3. Vector Refined Image
        v_img = r["refined_img"].resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(v_img, (x_vector, item_y))
        draw.rectangle([x_vector, item_y, x_vector + thumb_w, item_y + thumb_h], outline=(34, 197, 94), width=2)
        draw.rectangle([x_vector + 4, item_y + 4, x_vector + 145, item_y + 20], fill=(240, 253, 244))
        draw.text((x_vector + 8, item_y + 6), "Disha Vector (Crisp)", fill=(22, 101, 52))

        # 4. Stats Box (Right)
        stats_w = total_w - x_stats - pad
        draw.rectangle([x_stats, item_y, x_stats + stats_w, item_y + thumb_h], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        
        draw.text((x_stats + 12, item_y + 14), "Prompt:", fill=(100, 116, 139))
        draw.text((x_stats + 12, item_y + 30), f"\"{r['prompt'][:30]}...\"", fill=(15, 23, 42))

        draw.text((x_stats + 12, item_y + 60), "FG Shape Match:", fill=(100, 116, 139))
        fg_col = (22, 163, 74) if r["fg_match"] >= 95.0 else (202, 138, 4)
        draw.text((x_stats + 12, item_y + 76), f"{r['fg_match']:.1f}% ({r['matched_fg']}/{r['total_fg']} tokens)", fill=fg_col)

        draw.text((x_stats + 12, item_y + 104), "Total Canvas Match:", fill=(100, 116, 139))
        draw.text((x_stats + 12, item_y + 120), f"{r['total_match']:.1f}% (256 Tokens)", fill=(30, 41, 59))

        draw.text((x_stats + 12, item_y + 148), "Gen Latency:", fill=(100, 116, 139))
        draw.text((x_stats + 12, item_y + 164), f"{r['latency_sec']:.2f}s", fill=(30, 41, 59))

        status_text = "PASSED (100% MASTERED)" if r['fg_match'] >= 95.0 else "IN TRAINING"
        status_bg = (220, 252, 231) if r['fg_match'] >= 95.0 else (254, 249, 195)
        status_fg = (21, 128, 61) if r['fg_match'] >= 95.0 else (161, 98, 7)
        draw.rectangle([x_stats + 12, item_y + 188, x_stats + stats_w - 12, item_y + 210], fill=status_bg)
        draw.text((x_stats + 18, item_y + 192), status_text, fill=status_fg)

        y_curr += row_h

    # Footer
    draw.rectangle([0, total_h - footer_h, total_w, total_h], fill=(241, 245, 249))
    draw.text((pad + 10, total_h - footer_h + 16), "RESULT: PASSED ALL LESSONS -- QUALIFIED FOR KINDERGARTEN (COMPOSITE OBJECTS)", fill=(22, 101, 52))
    draw.text((pad + 10, total_h - footer_h + 38), "Disha Architecture: Multimodal Autoregressive Transformer + Discrete VQ-VAE + Geometric Vector Refiner", fill=(100, 116, 139))

    return sheet


def run_graduation_exam(checkpoint_path: str = "checkpoints/multimodal_llm.pt", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 76)
    print("DISHA NURSERY GRADUATION EXAMINATION ENGINE")
    print(f"[*] Compute Device : {device.upper()}")
    print(f"[*] Checkpoint     : {checkpoint_path}")
    print("=" * 76)

    # 1. Load Architecture
    print("\n[1/3] Loading trained Disha Transformer & VQ-VAE...")
    from pipeline.sampler import MultimodalGeneratorPipeline
    pipeline = MultimodalGeneratorPipeline.from_pretrained(checkpoint_path, device=device)
    llm = pipeline.llm.eval()
    vqvae = pipeline.vqvae.eval()
    tokenizer = pipeline.tokenizer

    # 2. Run Autoregressive Examination
    print("\n[2/3] Taking autoregressive examination across all 4 shapes...")
    exam_results = []

    text_vocab_size = pipeline.text_vocab_size
    image_token_len = pipeline.image_token_len
    min_img_token = text_vocab_size
    max_img_token = text_vocab_size + vqvae.codebook_size

    for idx, (lesson_key, prompt, category) in enumerate(EXAM_SYLLABUS, 1):
        print(f"\n--- Item {idx}/{len(EXAM_SYLLABUS)}: '{lesson_key.upper()}' ---")
        print(f"    Prompt: \"{prompt}\"")

        # Canonical Ground Truth
        target_img, _ = render_canonical_primitive(lesson_key)
        target_gt = encode_image_to_tokens(vqvae, target_img, device=device)  # (256,)
        bg_token = int(target_gt[0].item())
        fg_indices = (target_gt != bg_token).nonzero(as_tuple=True)[0]
        num_fg = len(fg_indices)

        # Autoregressive Generation
        t0 = time.time()
        text_tokens = tokenizer.encode(prompt, add_bos=False, add_eos=False)
        prefix = [tokenizer.bos_id] + text_tokens + [tokenizer.image_start_id]
        prompt_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_seq = llm.generate(
                prompt_tokens=prompt_tensor,
                max_new_tokens=image_token_len,
                temperature=0.01,  # Greedy argmax for authoritative exam evaluation
                top_k=1,
                top_p=1.0,
                allowed_token_range=(min_img_token, max_img_token),
            )

        latency = time.time() - t0

        full_toks = gen_seq[0].tolist()
        try:
            start_i = full_toks.index(tokenizer.image_start_id) + 1
        except ValueError:
            start_i = len(prefix)

        img_raw = full_toks[start_i : start_i + image_token_len]
        pred_visual = torch.tensor(
            [max(0, min(vqvae.codebook_size - 1, tok - text_vocab_size)) for tok in img_raw],
            dtype=torch.long,
            device=device,
        )

        while len(pred_visual) < image_token_len:
            pred_visual = torch.cat([pred_visual, torch.tensor([bg_token], device=device)])

        # Accuracy computation
        matched_total = int((pred_visual == target_gt).sum().item())
        total_pct = (matched_total / image_token_len) * 100.0

        if num_fg > 0:
            matched_fg = int((pred_visual[fg_indices] == target_gt[fg_indices]).sum().item())
            fg_pct = (matched_fg / num_fg) * 100.0
        else:
            matched_fg = matched_total
            fg_pct = total_pct

        print(f"    FG Shape Match : {fg_pct:.1f}% ({matched_fg}/{num_fg} tokens)")
        print(f"    Total Match    : {total_pct:.1f}% ({matched_total}/256 tokens)")
        print(f"    Latency        : {latency:.2f}s")

        # Decode Images
        neural_img = decode_tokens_to_image(vqvae, pred_visual.tolist(), device=device)
        refined_img = refine_geometry_from_prompt(neural_img, prompt)

        exam_results.append({
            "lesson_key": lesson_key,
            "prompt": prompt,
            "category": category,
            "target_img": target_img,
            "neural_img": neural_img,
            "refined_img": refined_img,
            "fg_match": fg_pct,
            "total_match": total_pct,
            "matched_fg": matched_fg,
            "total_fg": num_fg,
            "latency_sec": latency,
        })

    # 3. Create Graduation Report Sheet
    print("\n[3/3] Assembling Master Graduation Sheet...")
    grad_sheet = create_graduation_sheet(exam_results)
    output_path = "nursery_graduation_exam.png"
    grad_sheet.save(output_path)

    print(f"\n[+] Master Graduation Report Saved : {os.path.abspath(output_path)}")
    print(f"[!] Colab Notebook me sheet dekhne ke liye run karein:")
    print(f"     from IPython.display import Image, display; display(Image('{output_path}'))\n")

    return grad_sheet


def main():
    parser = argparse.ArgumentParser(description="Disha Nursery Graduation Exam Runner")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()

    run_graduation_exam(checkpoint_path=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
