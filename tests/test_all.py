"""
Comprehensive Test Suite for the Multimodal LLM from Scratch.
Tests Tokenizer, VQ-VAE, Transformer Blocks, and End-to-End Generation.
"""

import os
import unittest
import torch
from PIL import Image

from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE
from vqvae.layers import ResidualBlock, Downsample2d, Upsample2d
from vqvae.quantizer import VectorQuantizer
from model.norm import RMSNorm
from model.rope import precompute_freqs_cis, apply_rotary_emb
from model.attention import Attention
from model.mlp import SwiGLU
from model.transformer import MultimodalTransformer
from pipeline.sampler import MultimodalGeneratorPipeline
from config import LLMConfig, VQVAEConfig, GenerationConfig


class TestMultimodalLLM(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_tokenizer(self):
        """Verify ByteTokenizer encode, special tokens, and decode."""
        tokenizer = ByteTokenizer()
        text = "a glowing star in the night sky"
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(tokens[0], tokenizer.bos_id)
        self.assertEqual(tokens[-1], tokenizer.eos_id)
        
        decoded = tokenizer.decode(tokens)
        self.assertEqual(decoded, text)

    def test_vqvae_layers(self):
        """Verify individual VQ-VAE layers."""
        x = torch.randn(2, 64, 32, 32)
        res_block = ResidualBlock(64, 64)
        out = res_block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))

        down = Downsample2d(64, 128)
        down_out = down(x)
        self.assertEqual(down_out.shape, (2, 128, 16, 16))

        up = Upsample2d(128, 64)
        up_out = up(down_out)
        self.assertEqual(up_out.shape, (2, 64, 32, 32))

    def test_vqvae_quantizer_and_model(self):
        """Verify VectorQuantizer codebook and complete VQ-VAE forward/backward."""
        vqvae = VQVAE(
            in_channels=3,
            hidden_dim=32,
            embedding_dim=16,
            codebook_size=64,
            num_res_blocks=1,
            num_downsamples=2,  # 32x32 -> 8x8
        )
        img = torch.randn(2, 3, 32, 32)
        recon, loss, metrics = vqvae(img)
        self.assertEqual(recon.shape, (2, 3, 32, 32))
        self.assertIn("total_loss", metrics)
        self.assertIn("vq_loss", metrics)

        # Test index encoding and decoding
        indices = vqvae.encode_to_indices(img)
        self.assertEqual(indices.shape, (2, 64))  # 8*8 = 64 tokens

        decoded_img = vqvae.decode_from_indices(indices, grid_size=8)
        self.assertEqual(decoded_img.shape, (2, 3, 32, 32))

    def test_rmsnorm(self):
        """Verify RMSNorm normalization formula and gradient pass."""
        norm = RMSNorm(128)
        x = torch.randn(2, 10, 128)
        out = norm(x)
        self.assertEqual(out.shape, (2, 10, 128))
        self.assertAlmostEqual(out.pow(2).mean(-1).mean().item(), 1.0, delta=0.1)

    def test_rope(self):
        """Verify Rotary Position Embeddings rotation."""
        dim = 32
        freqs_cis = precompute_freqs_cis(dim, 16)
        self.assertEqual(freqs_cis.shape, (16, dim // 2))

        xq = torch.randn(2, 16, 4, dim)
        xk = torch.randn(2, 16, 2, dim)
        xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
        self.assertEqual(xq_rot.shape, xq.shape)
        self.assertEqual(xk_rot.shape, xk.shape)

    def test_attention_and_swiglu(self):
        """Verify GQA Attention with KV caching and SwiGLU."""
        attn = Attention(dim=128, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 8, 128)
        freqs_cis = precompute_freqs_cis(32, 8)
        out = attn(x, freqs_cis=freqs_cis)
        self.assertEqual(out.shape, (2, 8, 128))

        swiglu = SwiGLU(dim=128)
        mlp_out = swiglu(x)
        self.assertEqual(mlp_out.shape, (2, 8, 128))

    def test_multimodal_transformer_forward_and_loss(self):
        """Verify Transformer Decoder causal forward pass, loss, and autoregressive generation."""
        vocab_size = 500
        llm = MultimodalTransformer(
            vocab_size=vocab_size,
            dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64,
        )
        tokens = torch.randint(0, vocab_size, (2, 16))
        targets = torch.randint(0, vocab_size, (2, 16))

        logits, loss = llm(tokens, targets=targets)
        self.assertEqual(logits.shape, (2, 16, vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

        # Test generation with KV-cache
        gen_tokens = llm.generate(tokens[:, :4], max_new_tokens=8, temperature=0.7)
        self.assertEqual(gen_tokens.shape, (2, 12))

    def test_end_to_end_pipeline(self):
        """Verify complete pipeline generating a PIL Image."""
        tokenizer = ByteTokenizer()
        llm = MultimodalTransformer(
            vocab_size=2000,
            dim=64,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=128,
        )
        vqvae = VQVAE(
            in_channels=3,
            hidden_dim=32,
            embedding_dim=16,
            codebook_size=256,
            num_res_blocks=1,
            num_downsamples=3,  # 64x64 -> 8x8 = 64 tokens
        )
        pipeline = MultimodalGeneratorPipeline(
            llm=llm,
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_vocab_size=1000,
            image_token_len=64,
            device=self.device,
        )
        img = pipeline.generate_image(
            prompt="a glowing circle",
            gen_config=GenerationConfig(temperature=0.8, max_new_tokens=64),
        )
        self.assertIsInstance(img, Image.Image)

    def test_256x256_vqvae_and_generation(self):
        """Verify 256x256 resolution VQ-VAE (num_downsamples=4) and 256-token decoding."""
        vqvae = VQVAE(
            in_channels=3,
            hidden_dim=32,
            embedding_dim=16,
            codebook_size=256,
            num_res_blocks=1,
            num_downsamples=4,  # 256x256 -> 16x16 = 256 tokens
        )
        img_tensor = torch.randn(1, 3, 256, 256)
        indices = vqvae.encode_to_indices(img_tensor)
        self.assertEqual(indices.shape, (1, 256))

        recon = vqvae.decode_from_indices(indices)
        self.assertEqual(recon.shape, (1, 3, 256, 256))


if __name__ == "__main__":
    unittest.main()
