"""
Comprehensive Unit Tests for Disha Agentic Multimodal System.
Verifies Tokenizer Agent Tokens, Visual Critic, Tool Registry, Reasoner, and ReAct Pipeline.
"""

import unittest
import numpy as np
import torch
from PIL import Image

from tokenizer.text_tokenizer import ByteTokenizer
from agent.schema import ToolCall, ToolResponse, CriticReport, ThinkingStep
from agent.critic import VisualCritic
from agent.tools import ToolRegistry
from agent.reasoner import AgentReasoner
from pipeline.agentic_pipeline import AgenticImagePipeline
from config import AgenticConfig


class TestAgenticSystem(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_agentic_special_tokens(self):
        """Verify agentic control tokens are present in ByteTokenizer."""
        tokenizer = ByteTokenizer()
        agent_tokens = [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<plan>",
            "</plan>",
        ]
        for tok in agent_tokens:
            self.assertIn(tok, tokenizer.SPECIAL_TOKENS)
            tok_id = tokenizer.special_to_id[tok]
            self.assertIsInstance(tok_id, int)
            self.assertGreaterEqual(tok_id, 0)
            self.assertEqual(tokenizer.id_to_special[tok_id], tok)

        self.assertIsNotNone(tokenizer.think_start_id)
        self.assertIsNotNone(tokenizer.think_end_id)
        self.assertIsNotNone(tokenizer.tool_call_start_id)
        self.assertIsNotNone(tokenizer.tool_call_end_id)

    def test_visual_critic_metrics(self):
        """Verify VisualCritic on flat vs textured images."""
        critic = VisualCritic(aesthetic_threshold=6.5)

        # 1. Flat Uniform Image (Zero detail)
        flat_img = Image.new("RGB", (128, 128), color=(100, 100, 100))
        flat_report = critic.evaluate(flat_img)
        self.assertLess(flat_report.sharpness, 1.0)
        self.assertLess(flat_report.contrast, 1.0)
        self.assertEqual(flat_report.status, "NEEDS_REFINEMENT")
        self.assertGreater(len(flat_report.suggested_refinements), 0)

        # 2. High-Frequency Textured Pattern (Checkerboard + noise)
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[::8, :, :] = 255
        arr[:, ::8, :] = 255
        arr[32:96, 32:96, 0] = 200
        arr[32:96, 32:96, 1] = 50
        textured_img = Image.fromarray(arr)
        tex_report = critic.evaluate(textured_img)
        self.assertGreater(tex_report.sharpness, 50.0)
        self.assertGreater(tex_report.contrast, 20.0)
        self.assertGreater(tex_report.aesthetic_score, flat_report.aesthetic_score)

    def test_tool_registry(self):
        """Verify ToolRegistry tool registration and dispatch."""
        registry = ToolRegistry(critic_threshold=7.0)
        schemas = registry.get_schemas()
        tool_names = [s["name"] for s in schemas]
        self.assertIn("analyze_image_quality", tool_names)
        self.assertIn("refine_prompt", tool_names)

        # Execute prompt refiner
        call = ToolCall(
            name="refine_prompt",
            arguments={
                "current_prompt": "cyberpunk city",
                "refinement_suggestions": ["Inject 'crisp focus'", "Inject 'high contrast'"],
                "style_boost": "volumetric lighting",
            },
        )
        resp = registry.execute(call)
        self.assertTrue(resp.success)
        self.assertIn("cyberpunk city", resp.output)
        self.assertIn("crisp focus", resp.output)
        self.assertIn("volumetric lighting", resp.output)

    def test_agent_reasoner_cycle(self):
        """Verify AgentReasoner ReAct cycle execution."""
        registry = ToolRegistry(critic_threshold=5.0)
        reasoner = AgentReasoner(
            tool_registry=registry,
            aesthetic_threshold=5.0,
            max_refinement_steps=1,
        )

        # Mock generator returning test image
        def mock_gen(prompt, args):
            arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            return Image.fromarray(arr)

        steps_logged = []
        trace = reasoner.run_react_cycle(
            prompt="futuristic neon sports car",
            engine="scratch",
            generate_fn=mock_gen,
            auto_refine=True,
            on_step_callback=lambda s: steps_logged.append(s),
        )

        self.assertGreaterEqual(len(trace.thinking_steps), 3)
        self.assertIsNotNone(trace.final_image)
        self.assertGreaterEqual(trace.total_iterations, 1)

        # Verify phases occurred
        phases = [s.phase for s in trace.thinking_steps]
        self.assertIn("ANALYSIS", phases)
        self.assertIn("GENERATION", phases)
        self.assertIn("CRITIQUE", phases)

    def test_agentic_image_pipeline_baseline(self):
        """Verify AgenticImagePipeline baseline execution."""
        cfg = AgenticConfig(
            max_refinement_steps=1,
            aesthetic_threshold=5.0,
            auto_refine=False,
        )
        pipeline = AgenticImagePipeline.from_scratch_checkpoint(
            checkpoint_path="non_existent_ckpt.pt",
            device=self.device,
            config=cfg,
            dim=64,
            num_layers=2,
        )
        self.assertIsNotNone(pipeline.scratch_pipeline)
        pipeline.scratch_pipeline.image_token_len = 16

        trace = pipeline.run(
            prompt="a glowing crystal in the dark",
            engine="scratch",
            auto_refine=False,
        )
        self.assertIsNotNone(trace.final_image)
        self.assertIsInstance(trace.final_image, Image.Image)
        self.assertEqual(trace.final_image.size, (256, 256))
        self.assertGreater(len(trace.thinking_steps), 0)


if __name__ == "__main__":
    unittest.main()
