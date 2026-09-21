"""
High-level Inference and Sampling Pipeline for Multimodal Image Generation.
Connects Tokenizer -> Multimodal Transformer -> VQ-VAE Decoder.
"""

from typing import Optional, Union, List
import os
import torch
import numpy as np
from PIL import Image

from tokenizer.text_tokenizer import ByteTokenizer
from model.transformer import MultimodalTransformer
from vqvae.model import VQVAE
from config import LLMConfig, VQVAEConfig, GenerationConfig


class MultimodalGeneratorPipeline:
    """
    End-to-End Pipeline to generate high-fidelity images from natural language text prompts.
    """
    def __init__(
        self,
        llm: MultimodalTransformer,
        vqvae: VQVAE,
        tokenizer: ByteTokenizer,
        text_vocab_size: int = 5000,
        image_token_len: int = 256,
        device: Union[str, torch.device] = "cpu",
    ):
        self.llm = llm.to(device).eval()
        self.vqvae = vqvae.to(device).eval()
        self.tokenizer = tokenizer
        self.text_vocab_size = text_vocab_size
        self.image_token_len = image_token_len
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cpu",
    ) -> "MultimodalGeneratorPipeline":
        # Search candidate paths across nested directories (e.g. Colab /content/disha, /content/disha/disha)
        candidate_paths = [
            checkpoint_path,
            os.path.join(os.getcwd(), checkpoint_path),
            os.path.join("..", checkpoint_path),
            os.path.join("../..", checkpoint_path),
            os.path.join("/content/disha", checkpoint_path),
            os.path.join("/content/disha/disha", checkpoint_path),
            os.path.join("/content", checkpoint_path),
            "/content/disha/checkpoints/multimodal_llm.pt",
            "/content/disha/disha/checkpoints/multimodal_llm.pt",
            "/content/checkpoints/multimodal_llm.pt",
        ]
        resolved_path = None
        for cand in candidate_paths:
            if cand and os.path.exists(cand):
                resolved_path = cand
                break

        if resolved_path is None:
            import glob
            matches = glob.glob("**/multimodal_llm.pt", recursive=True) + glob.glob("/content/**/multimodal_llm.pt", recursive=True)
            if matches:
                resolved_path = matches[0]

        tokenizer = ByteTokenizer()
        llm_cfg = LLMConfig()
        vq_cfg = VQVAEConfig()
        ckpt = {}

        if resolved_path and os.path.exists(resolved_path):
            print(f"[+] Loading foundation checkpoint from: {os.path.abspath(resolved_path)}")
            try:
                ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
                if "tokenizer_vocab" in ckpt:
                    tokenizer.merges = ckpt["tokenizer_vocab"].get("merges", {})
                llm_cfg = ckpt.get("llm_config", LLMConfig())
                vq_cfg = ckpt.get("vqvae_config", VQVAEConfig())
            except Exception as e:
                print(f"[!] Error reading {resolved_path}: {e}")
                ckpt = {}
        else:
            print(f"[!] Checkpoint '{checkpoint_path}' not found. Initializing fresh model for baby-steps training...")

        total_vocab_size = getattr(llm_cfg, "text_vocab_size", 8000) + getattr(llm_cfg, "image_vocab_size", 2048) + len(tokenizer.SPECIAL_TOKENS)

        llm = MultimodalTransformer(
            vocab_size=total_vocab_size,
            dim=getattr(llm_cfg, "dim", 512),
            num_layers=getattr(llm_cfg, "num_layers", 8),
            num_heads=getattr(llm_cfg, "num_heads", 8),
            num_kv_heads=getattr(llm_cfg, "num_kv_heads", 4),
            max_seq_len=getattr(llm_cfg, "max_seq_len", 512),
            ffn_dim_multiplier=getattr(llm_cfg, "ffn_dim_multiplier", 3.5),
            multiple_of=getattr(llm_cfg, "multiple_of", 64),
            norm_eps=getattr(llm_cfg, "norm_eps", 1e-6),
            rope_theta=getattr(llm_cfg, "rope_theta", 10000.0),
        )
        if "llm_state_dict" in ckpt:
            state_dict = dict(ckpt["llm_state_dict"])
            model_sd = llm.state_dict()
            for key in ["tok_embeddings.weight", "output.weight"]:
                if key in state_dict and key in model_sd:
                    ckpt_w = state_dict[key]
                    target_w = model_sd[key]
                    if ckpt_w.shape != target_w.shape and ckpt_w.shape[1] == target_w.shape[1]:
                        min_rows = min(ckpt_w.shape[0], target_w.shape[0])
                        new_w = target_w.clone()
                        new_w[:min_rows] = ckpt_w[:min_rows]
                        state_dict[key] = new_w
            llm.load_state_dict(state_dict, strict=False)

        vqvae = VQVAE(
            in_channels=getattr(vq_cfg, "in_channels", 3),
            hidden_dim=getattr(vq_cfg, "hidden_dim", 128),
            embedding_dim=getattr(vq_cfg, "embedding_dim", 64),
            codebook_size=getattr(vq_cfg, "codebook_size", 2048),
            num_res_blocks=getattr(vq_cfg, "num_res_blocks", 2),
            num_downsamples=getattr(vq_cfg, "num_downsamples", 4),
            commitment_cost=getattr(vq_cfg, "commitment_cost", 0.25),
        )
        if "vqvae_state_dict" in ckpt:
            vqvae.load_state_dict(ckpt["vqvae_state_dict"])
        else:
            # Search candidate paths for standalone vqvae.pt
            for vq_cand in [
                "checkpoints/vqvae.pt",
                "/content/disha/checkpoints/vqvae.pt",
                "/content/disha/disha/checkpoints/vqvae.pt",
                "/content/checkpoints/vqvae.pt",
                "../checkpoints/vqvae.pt",
            ]:
                if os.path.exists(vq_cand):
                    try:
                        vq_ckpt = torch.load(vq_cand, map_location=device, weights_only=False)
                        sd = vq_ckpt["vqvae_state_dict"] if "vqvae_state_dict" in vq_ckpt else vq_ckpt
                        vqvae.load_state_dict(sd)
                        print(f"[+] Loaded pretrained VQ-VAE tokenizer from: {vq_cand}")
                        break
                    except Exception:
                        pass

        return cls(
            llm=llm,
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_vocab_size=getattr(llm_cfg, "text_vocab_size", 8000),
            image_token_len=getattr(llm_cfg, "image_token_len", 256),
            device=device,
        )

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Generates an image from a natural language text prompt.
        
        Args:
            prompt (str): Text prompt (e.g. 'a glowing red circle on dark background').
            gen_config (GenerationConfig, optional): Sampling parameters.
            save_path (str, optional): File path to save output PNG.
            
        Returns:
            PIL.Image.Image: Generated RGB image.
        """
        if gen_config is None:
            gen_config = GenerationConfig()

        # 1. Text Tokenization
        text_tokens = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        
        # Guard against sequence length overflow: preserve most relevant prompt tokens
        max_prompt_len = max(1, self.llm.max_seq_len - self.image_token_len - 4)
        if len(text_tokens) > max_prompt_len:
            text_tokens = text_tokens[-max_prompt_len:]

        # 2. Construct Prefix: [ <bos>, text_tokens, <image_start> ]
        prefix = [self.tokenizer.bos_id] + text_tokens + [self.tokenizer.image_start_id]
        prompt_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)

        # 3. Autoregressive Image Token Generation
        # Constrain visual token sampling strictly to the VQ-VAE codebook space
        min_img_token = self.text_vocab_size
        max_img_token = self.text_vocab_size + self.vqvae.codebook_size

        rep_pen = getattr(gen_config, "repetition_penalty", 1.08)
        generated_seq = self.llm.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=self.image_token_len,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=rep_pen,
            consecutive_penalty=2.5,
            allowed_token_range=(min_img_token, max_img_token),
        )

        # 4. Extract generated image token slice
        full_tokens = generated_seq[0].tolist()
        try:
            start_idx = full_tokens.index(self.tokenizer.image_start_id) + 1
        except ValueError:
            start_idx = len(prefix)

        img_tokens_raw = full_tokens[start_idx : start_idx + self.image_token_len]

        # 5. Map tokens back to VQ-VAE codebook space [0, codebook_size - 1]
        img_tokens = [
            max(0, min(self.vqvae.codebook_size - 1, tok - self.text_vocab_size))
            for tok in img_tokens_raw
        ]
        while len(img_tokens) < self.image_token_len:
            pad_val = img_tokens[-1] if img_tokens else (self.vqvae.codebook_size // 2)
            img_tokens.append(pad_val)

        # 6. Decode visual tokens into continuous RGB image tensor via VQ-VAE
        indices_tensor = torch.tensor([img_tokens], dtype=torch.long, device=self.device)
        recon_tensor = self.vqvae.decode_from_indices(indices_tensor)  # (1, 3, 256, 256) in [-1, 1]

        # 7. Convert tensor to PIL Image: [-1, 1] -> [0, 255]
        recon_np = recon_tensor[0].detach().cpu().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
        recon_np = ((recon_np + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(recon_np)
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.Resampling.LANCZOS)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            img.save(save_path)

        return img

    @torch.no_grad()
    def generate_image_with_tokens(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        save_path: Optional[str] = None,
    ):
        """Generates an image and returns both the PIL.Image and the visual token list."""
        if gen_config is None:
            gen_config = GenerationConfig()

        text_tokens = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        max_prompt_len = max(1, self.llm.max_seq_len - self.image_token_len - 4)
        if len(text_tokens) > max_prompt_len:
            text_tokens = text_tokens[-max_prompt_len:]

        prefix = [self.tokenizer.bos_id] + text_tokens + [self.tokenizer.image_start_id]
        prompt_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)

        min_img_token = self.text_vocab_size
        max_img_token = self.text_vocab_size + self.vqvae.codebook_size

        rep_pen = getattr(gen_config, "repetition_penalty", 1.08)
        generated_seq = self.llm.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=self.image_token_len,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=rep_pen,
            consecutive_penalty=2.5,
            allowed_token_range=(min_img_token, max_img_token),
        )

        full_tokens = generated_seq[0].tolist()
        try:
            start_idx = full_tokens.index(self.tokenizer.image_start_id) + 1
        except ValueError:
            start_idx = len(prefix)

        img_tokens_raw = full_tokens[start_idx : start_idx + self.image_token_len]

        img_tokens = [
            max(0, min(self.vqvae.codebook_size - 1, tok - self.text_vocab_size))
            for tok in img_tokens_raw
        ]
        while len(img_tokens) < self.image_token_len:
            pad_val = img_tokens[-1] if img_tokens else (self.vqvae.codebook_size // 2)
            img_tokens.append(pad_val)

        indices_tensor = torch.tensor([img_tokens], dtype=torch.long, device=self.device)
        recon_tensor = self.vqvae.decode_from_indices(indices_tensor)

        recon_np = recon_tensor[0].detach().cpu().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
        recon_np = ((recon_np + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(recon_np)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            img.save(save_path)

        return img, img_tokens
