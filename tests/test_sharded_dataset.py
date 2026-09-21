"""
Unit Tests for the TAR-Sharded Streaming Dataset Engine and Agentic Recaptioner.
Verifies Shard Creation, In-Memory Unpacking, Streaming PyTorch DataLoader, and CoT traces.
"""

import unittest
import os
import shutil
import tempfile
import torch
from torch.utils.data import DataLoader
from PIL import Image

from pipeline.sharded_dataset import TarShardedDatasetWriter, ShardedMultimodalIterableDataset
from pipeline.agentic_recaptioner import AgenticRecaptioner
from tokenizer.text_tokenizer import ByteTokenizer


class TestShardedDatasetEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_shards_")
        self.tokenizer = ByteTokenizer()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_agentic_recaptioner(self):
        """Verify AgenticRecaptioner produces valid CoT thinking tags and tool calls."""
        raw_prompt = "a majestic golden lion resting on rock"
        cot_trace, expanded = AgenticRecaptioner.generate_cot_trace(raw_prompt)

        self.assertIn("<think>", cot_trace)
        self.assertIn("</think>", cot_trace)
        self.assertIn("<tool_call: generate_image", cot_trace)
        self.assertIn(raw_prompt, expanded)

    def test_tar_sharded_writer_and_reader(self):
        """Verify writing and streaming from TAR shards with zero disk extraction."""
        writer = TarShardedDatasetWriter(
            output_dir=self.test_dir,
            max_samples_per_shard=3,
            shard_prefix="test_shard",
        )

        # Write 5 test samples across 2 shards
        for i in range(5):
            img = Image.new("RGB", (64, 64), color=(i * 40, 100, 150))
            cap = f"synthetic test scene number {i}"
            writer.add_sample(f"sample_{i:04d}", img, cap, {"category": "UnitTest"})

        index_info = writer.close()
        self.assertEqual(index_info["total_samples"], 5)
        self.assertEqual(index_info["total_shards"], 2)

        # Verify shards exist on disk
        shard_files = [os.path.join(self.test_dir, s) for s in index_info["shards"]]
        for sf in shard_files:
            self.assertTrue(os.path.exists(sf))

        # Test ShardedMultimodalIterableDataset (mode='vqvae')
        dataset_vq = ShardedMultimodalIterableDataset(
            shard_paths=shard_files,
            image_size=64,
            mode="vqvae",
            shuffle_shards=False,
        )
        samples_vq = list(dataset_vq)
        self.assertEqual(len(samples_vq), 5)
        self.assertEqual(samples_vq[0]["image"].shape, (3, 64, 64))

        # Test ShardedMultimodalIterableDataset (mode='llm')
        dataset_llm = ShardedMultimodalIterableDataset(
            shard_paths=shard_files,
            tokenizer=self.tokenizer,
            image_size=64,
            max_seq_len=300,
            mode="llm",
            shuffle_shards=False,
        )
        samples_llm = list(dataset_llm)
        self.assertEqual(len(samples_llm), 5)
        self.assertIn("input_ids", samples_llm[0])
        self.assertIn("targets", samples_llm[0])
        self.assertEqual(samples_llm[0]["input_ids"].shape[0], 299)

        # Test PyTorch DataLoader batching
        loader = DataLoader(dataset_llm, batch_size=2)
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["image"].shape, (2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
