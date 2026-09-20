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
        """Loads model weights and tokenizers from a checkpoint file."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        tokenizer = ByteTokenizer()
        if "tokenizer_vocab" in ckpt:
            # Reconstruct merges if saved
            tokenizer.merges = ckpt["tokenizer_vocab"].get("merges", {})

        llm_cfg = ckpt.get("llm_config", LLMConfig())
        vq_cfg = ckpt.get("vqvae_config", VQVAEConfig())

        total_vocab_size = llm_cfg.text_vocab_size + llm_cfg.image_vocab_size + len(tokenizer.SPECIAL_TOKENS)

        llm = MultimodalTransformer(
            vocab_size=total_vocab_size,
            dim=llm_cfg.dim,
            num_layers=llm_cfg.num_layers,
            num_heads=llm_cfg.num_heads,
            num_kv_heads=llm_cfg.num_kv_heads,
            max_seq_len=llm_cfg.max_seq_len,
        )
        if "llm_state_dict" in ckpt:
            llm.load_state_dict(ckpt["llm_state_dict"])

        vqvae = VQVAE(
            in_channels=vq_cfg.in_channels,
            hidden_dim=vq_cfg.hidden_dim,
            embedding_dim=vq_cfg.embedding_dim,
            codebook_size=vq_cfg.codebook_size,
            num_res_blocks=vq_cfg.num_res_blocks,
            num_downsamples=vq_cfg.num_downsamples,
        )
        if "vqvae_state_dict" in ckpt:
            vqvae.load_state_dict(ckpt["vqvae_state_dict"])

        return cls(
            llm=llm,
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_vocab_size=llm_cfg.text_vocab_size,
            image_token_len=llm_cfg.image_token_len,
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
        
        # 2. Construct Prefix: [ <bos>, text_tokens, <image_start> ]
        prefix = [self.tokenizer.bos_id] + text_tokens + [self.tokenizer.image_start_id]
        prompt_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)

        # 3. Autoregressive Image Token Generation
        generated_seq = self.llm.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=self.image_token_len,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            image_end_token_id=self.tokenizer.image_end_id,
        )

        # 4. Extract generated image token slice
        full_tokens = generated_seq[0].tolist()
        # Find index of <image_start>
        try:
            start_idx = full_tokens.index(self.tokenizer.image_start_id) + 1
        except ValueError:
            start_idx = len(prefix)

        img_tokens_raw = full_tokens[start_idx : start_idx + self.image_token_len]
        
        # If fewer tokens generated, pad with random codebook tokens
        while len(img_tokens_raw) < self.image_token_len:
            img_tokens_raw.append(self.text_vocab_size)

        # 5. Map tokens back to VQ-VAE codebook space [0, codebook_size - 1]
        img_tokens = [
            max(0, min(self.vqvae.codebook_size - 1, tok - self.text_vocab_size))
            for tok in img_tokens_raw
        ]

        # 6. Decode visual tokens into continuous RGB image tensor via VQ-VAE
        indices_tensor = torch.tensor([img_tokens], dtype=torch.long, device=self.device)
        recon_tensor = self.vqvae.decode_from_indices(indices_tensor)  # (1, 3, 256, 256) in [-1, 1]

        # 7. Convert tensor to PIL Image: [-1, 1] -> [0, 255]
        recon_np = recon_tensor[0].detach().cpu().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
        recon_np = ((recon_np + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(recon_np)

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
        prefix = [self.tokenizer.bos_id] + text_tokens + [self.tokenizer.image_start_id]
        prompt_tensor = torch.tensor([prefix], dtype=torch.long, device=self.device)

        generated_seq = self.llm.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=self.image_token_len,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            image_end_token_id=self.tokenizer.image_end_id,
        )

        full_tokens = generated_seq[0].tolist()
        try:
            start_idx = full_tokens.index(self.tokenizer.image_start_id) + 1
        except ValueError:
            start_idx = len(prefix)

        img_tokens_raw = full_tokens[start_idx : start_idx + self.image_token_len]
        while len(img_tokens_raw) < self.image_token_len:
            img_tokens_raw.append(self.text_vocab_size)

        img_tokens = [
            max(0, min(self.vqvae.codebook_size - 1, tok - self.text_vocab_size))
            for tok in img_tokens_raw
        ]

        indices_tensor = torch.tensor([img_tokens], dtype=torch.long, device=self.device)
        recon_tensor = self.vqvae.decode_from_indices(indices_tensor)

        recon_np = recon_tensor[0].detach().cpu().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
        recon_np = ((recon_np + 1.0) * 127.5).astype(np.uint8)
        img = Image.fromarray(recon_np)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            img.save(save_path)

        return img, img_tokens
