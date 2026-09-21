"""
CLI Interface for Disha Agentic Multimodal Image Generation.
Displays live Chain-of-Thought (<think>) reasoning, tool dispatching (<tool_call>),
automated visual critique reports, and saves the final synthesized image.

Usage:
    python generate_agentic.py --prompt "a cybernetic tiger in a rainy neon alley" --engine scratch
    python generate_agentic.py --prompt "majestic snowy mountain landscape" --engine hd --style "Photorealistic"
"""

import os
import sys
import argparse
import time
import torch
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import AgenticConfig
from pipeline.agentic_pipeline import AgenticImagePipeline
from agent.schema import ThinkingStep


def print_banner():
    print("""
+--------------------------------------------------------------------------+
|          DISHA AGENTIC MULTIMODAL IMAGE GENERATION STUDIO                |
|    Cognitive CoT (<think>) * Dynamic Tool Calling * Visual Critic        |
+--------------------------------------------------------------------------+
""")


def on_thinking_step(step: ThinkingStep):
    if step.phase == "ANALYSIS":
        print(f"\n[THINK - STEP {step.step_number}: COGNITIVE INTENT ANALYSIS & PLANNING]")
        print(f"{step.thought}")
    elif step.phase == "GENERATION":
        print(f"\n[TOOL - STEP {step.step_number}: TOOL INVOCATION]")
        print(f"{step.thought}")
        if step.tool_response:
            print(f"[Tool Response]: {step.tool_response.output}")
    elif step.phase == "CRITIQUE":
        print(f"\n[CRITIC - STEP {step.step_number}: VISUAL CRITIC EVALUATION]")
        print(f"{step.thought}")
    elif step.phase == "REFLECTION":
        print(f"\n[REFLECT - STEP {step.step_number}: AUTONOMOUS REFLECTION & SELF-CORRECTION]")
        print(f"{step.thought}")
    elif step.phase == "CONCLUSION":
        print(f"\n[GOAL - STEP {step.step_number}: GOAL ACHIEVED & FINAL APPROVAL]")
        print(f"{step.thought}")


def main():
    parser = argparse.ArgumentParser(description="Disha Agentic Image Generation")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a glowing celestial phoenix rising above frozen crystal mountain peaks",
        help="Input text prompt",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="scratch",
        choices=["scratch", "hd", "sdxl-turbo"],
        help="Generation engine: 'scratch' (Native PyTorch) or 'hd' (Ultra-HD Diffusion)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        choices=[
            "Photorealistic",
            "Cyberpunk / Sci-Fi",
            "Anime / Studio Ghibli",
            "3D Octane Render",
            "Dark Fantasy",
            "Synthwave / Retro",
        ],
        help="Optional artistic style modifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_agentic.png",
        help="Output image file path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="Aesthetic score threshold for critic approval (1.0 - 10.0)",
    )
    parser.add_argument(
        "--max_refinements",
        type=int,
        default=2,
        help="Maximum self-correction refinement loops",
    )
    parser.add_argument(
        "--no_refine",
        action="store_true",
        help="Disable autonomous ReAct refinement loop",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/multimodal_llm.pt",
        help="Path to trained scratch model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda / cpu)",
    )

    args = parser.parse_args()
    print_banner()

    engine_alias = "sdxl-turbo" if args.engine == "hd" else args.engine
    print(f"[*] Input Prompt     : \"{args.prompt}\"")
    print(f"[*] Selected Engine  : {args.engine.upper()} ({engine_alias})")
    print(f"[*] Device           : {args.device}")
    print(f"[*] Critic Threshold : {args.threshold}/10.0")
    print(f"[*] Auto-Refine      : {not args.no_refine} (Max: {args.max_refinements} steps)")
    print("-" * 74)

    # Configure Agent
    cfg = AgenticConfig(
        max_refinement_steps=args.max_refinements,
        aesthetic_threshold=args.threshold,
        default_engine=args.engine,
        auto_refine=not args.no_refine,
    )

    print("[+] Initializing Agentic Multimodal Pipeline...")
    pipeline = AgenticImagePipeline.from_scratch_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config=cfg,
    )

    print("[+] Launching Autonomous ReAct Generation Cycle...")
    trace = pipeline.run(
        prompt=args.prompt,
        engine=args.engine,
        style=args.style,
        auto_refine=not args.no_refine,
        on_step_callback=on_thinking_step,
    )

    # Save output
    if trace.final_image is not None:
        trace.final_image.save(args.output)
        print("\n" + "=" * 74)
        print(f"[*] Generation Complete in {trace.execution_time_sec}s!")
        print(f"[*] Final Image Saved to : {os.path.abspath(args.output)}")
        print(f"[*] Total ReAct Loops    : {trace.total_iterations}")
        if trace.critic_reports:
            final_report = trace.critic_reports[-1]
            print(f"[*] Final Aesthetic Score: {final_report.aesthetic_score}/10.0 ({final_report.status})")
            print(f"[*] Final Sharpness      : {final_report.sharpness:.1f} var")
            print(f"[*] Final Contrast       : {final_report.contrast:.1f} std")
        print("=" * 74)
    else:
        print("[!] No image produced by agentic pipeline.")


if __name__ == "__main__":
    main()
