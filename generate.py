"""
Inference script to generate images from natural language text prompts using the trained Multimodal LLM.
"""

import os
import argparse
import torch

from pipeline.sampler import MultimodalGeneratorPipeline
from config import GenerationConfig, LLMConfig, VQVAEConfig
from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE
from model.transformer import MultimodalTransformer


def main():
    parser = argparse.ArgumentParser(description="Generate image from text prompt using Multimodal LLM")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_llm.pt", help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output image file path")
    parser.add_argument("--temp", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.92, help="Top-p nucleus sampling parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"[*] Initializing Multimodal Generator on {args.device}...")
    
    if os.path.exists(args.checkpoint):
        print(f"[+] Loading checkpoint from {args.checkpoint}")
        pipeline = MultimodalGeneratorPipeline.from_pretrained(args.checkpoint, device=args.device)
    else:
        print("[!] No checkpoint found at specified path. Initializing model for demonstration.")
        llm_cfg = LLMConfig()
        vq_cfg = VQVAEConfig()
        tokenizer = ByteTokenizer()
        total_vocab = llm_cfg.text_vocab_size + llm_cfg.image_vocab_size + len(tokenizer.SPECIAL_TOKENS)
        
        # Adaptive dimensions for CPU demonstration vs GPU
        dim = 256 if args.device == "cpu" else llm_cfg.dim
        num_layers = 4 if args.device == "cpu" else llm_cfg.num_layers
        num_heads = 4 if args.device == "cpu" else llm_cfg.num_heads
        num_kv_heads = 2 if args.device == "cpu" else (llm_cfg.num_kv_heads or 4)

        llm = MultimodalTransformer(
            vocab_size=total_vocab,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=llm_cfg.max_seq_len,
            ffn_dim_multiplier=llm_cfg.ffn_dim_multiplier,
            multiple_of=llm_cfg.multiple_of,
            norm_eps=llm_cfg.norm_eps,
            rope_theta=llm_cfg.rope_theta,
        )
        vqvae = VQVAE(
            in_channels=vq_cfg.in_channels,
            hidden_dim=vq_cfg.hidden_dim,
            embedding_dim=vq_cfg.embedding_dim,
            codebook_size=vq_cfg.codebook_size,
            num_res_blocks=vq_cfg.num_res_blocks,
            num_downsamples=vq_cfg.num_downsamples,
            commitment_cost=vq_cfg.commitment_cost,
        )
        pipeline = MultimodalGeneratorPipeline(
            llm=llm,
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_vocab_size=llm_cfg.text_vocab_size,
            image_token_len=llm_cfg.image_token_len,
            device=args.device,
        )

    gen_cfg = GenerationConfig(
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(f"[*] Generating image for prompt: \"{args.prompt}\"")
    img = pipeline.generate_image(prompt=args.prompt, gen_config=gen_cfg, save_path=args.output)
    print(f"[+] Image successfully generated and saved to: {args.output}")


if __name__ == "__main__":
    main()
