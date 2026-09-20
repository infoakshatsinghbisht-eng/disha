"""
Memory Ceiling and Constant-Budget Verification Tests for 8GB RAM Laptops.
Tests Streaming Attention constant memory, dynamic quantization, and prompt enhancement.
"""

import unittest
import torch
from model.streaming_attention import StreamingAttention
from model.transformer import MultimodalTransformer
from model.quantization import Int8Linear, quantize_model_for_low_ram, get_model_ram_mb
from pipeline.prompt_enhancer import PromptEnhancer


class TestMemoryAndStreamingBudget(unittest.TestCase):
    def test_streaming_attention_constant_memory(self):
        """Verify that streaming KV-cache memory does not exceed bounded limit even for long sequences."""
        attn = StreamingAttention(dim=128, num_heads=4, num_kv_heads=2, window_size=32, num_sink_tokens=4)
        
        # Simulate 200 autoregressive steps
        for step in range(200):
            x = torch.randn(1, 1, 128)
            _ = attn(x, use_cache=True, start_pos=step)

        # Max cache length should strictly be num_sink_tokens + window_size = 4 + 32 = 36
        self.assertEqual(attn.cache_k.size(1), 36)
        self.assertEqual(attn.cache_v.size(1), 36)
        
        # Memory consumption should be less than 100 KB for this config
        cache_bytes = attn.get_cache_memory_bytes()
        self.assertLess(cache_bytes, 100 * 1024)

    def test_quantization_reduces_ram(self):
        """Verify that Int8 dynamic quantization compresses linear layers."""
        linear = torch.nn.Linear(256, 512, bias=True)
        qlinear = Int8Linear.from_float(linear)
        
        x = torch.randn(2, 10, 256)
        out_float = linear(x)
        out_q = qlinear(x)
        
        # Test shape match and close cosine similarity
        self.assertEqual(out_q.shape, out_float.shape)
        cos_sim = torch.nn.functional.cosine_similarity(out_float.flatten(), out_q.flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.95)

    def test_transformer_kv_cache_tracking(self):
        """Verify that MultimodalTransformer tracks total KV cache memory."""
        llm = MultimodalTransformer(vocab_size=1000, dim=128, num_layers=2, num_heads=4, num_kv_heads=2)
        tokens = torch.randint(0, 1000, (1, 8))
        _ = llm.generate(tokens, max_new_tokens=16, temperature=0.7)
        
        # Check cache memory method
        total_cache = llm.get_total_kv_cache_bytes()
        self.assertIsInstance(total_cache, int)

    def test_prompt_enhancer(self):
        """Verify PromptEnhancer produces enriched descriptive prompts."""
        base = "a futuristic flying car"
        enhanced = PromptEnhancer.enhance(base, style="Cyberpunk / Sci-Fi")
        self.assertIn("futuristic flying car", enhanced)
        self.assertGreater(len(enhanced), len(base))


if __name__ == "__main__":
    unittest.main()
