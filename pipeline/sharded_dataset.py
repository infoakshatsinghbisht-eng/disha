"""
Industrial-Grade TAR-Sharded Dataset Engine (WebDataset Standard).
Enables streaming millions of multimodal image-caption pairs directly into Cloud GPU VRAM
without disk bottlenecks, inode exhaustion, or filesystem degradation.
"""

from typing import Iterator, Dict, Any, List, Optional, Tuple, Union
import os
import io
import time
import random
import tarfile
import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import numpy as np

from tokenizer.text_tokenizer import ByteTokenizer
from vqvae.model import VQVAE


class TarShardedDatasetWriter:
    """
    Writes multimodal image-caption pairs into sequential .tar archives (WebDataset standard).
    Each shard contains:
        - {sample_id}.png : Compressed 256x256 image bytes
        - {sample_id}.txt : Natural language caption or CoT trace
        - {sample_id}.json: Rich metadata (category, aesthetic score, camera specs)
    """

    def __init__(
        self,
        output_dir: str = "dataset_sharded",
        max_samples_per_shard: int = 500,
        shard_prefix: str = "shard",
    ):
        self.output_dir = output_dir
        self.max_samples_per_shard = max_samples_per_shard
        self.shard_prefix = shard_prefix
        os.makedirs(output_dir, exist_ok=True)

        self.current_shard_idx = 0
        self.current_shard_count = 0
        self.total_samples_written = 0
        self._current_tar: Optional[tarfile.TarFile] = None
        self._open_next_shard()

    def _get_shard_path(self, idx: int) -> str:
        return os.path.join(self.output_dir, f"{self.shard_prefix}_{idx:05d}.tar")

    def _open_next_shard(self):
        if self._current_tar is not None:
            self._current_tar.close()
        
        shard_path = self._get_shard_path(self.current_shard_idx)
        self._current_tar = tarfile.open(shard_path, "w")
        self.current_shard_count = 0

    def add_sample(
        self,
        sample_id: str,
        image: Image.Image,
        caption: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Adds a single sample (image + caption + metadata) to the active shard."""
        if self.current_shard_count >= self.max_samples_per_shard:
            self.current_shard_idx += 1
            self._open_next_shard()

        # 1. Encode PNG to bytes
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format="PNG", optimize=True)
        img_bytes = img_bytes_io.getvalue()

        img_ti = tarfile.TarInfo(name=f"{sample_id}.png")
        img_ti.size = len(img_bytes)
        img_ti.mtime = int(time.time())
        self._current_tar.addfile(img_ti, io.BytesIO(img_bytes))

        # 2. Text Caption
        txt_bytes = caption.encode("utf-8")
        txt_ti = tarfile.TarInfo(name=f"{sample_id}.txt")
        txt_ti.size = len(txt_bytes)
        txt_ti.mtime = int(time.time())
        self._current_tar.addfile(txt_ti, io.BytesIO(txt_bytes))

        # 3. Metadata JSON
        meta_dict = metadata or {}
        meta_dict.setdefault("sample_id", sample_id)
        meta_dict.setdefault("caption", caption)
        meta_dict.setdefault("width", image.width)
        meta_dict.setdefault("height", image.height)
        json_bytes = json.dumps(meta_dict, ensure_ascii=False).encode("utf-8")

        json_ti = tarfile.TarInfo(name=f"{sample_id}.json")
        json_ti.size = len(json_bytes)
        json_ti.mtime = int(time.time())
        self._current_tar.addfile(json_ti, io.BytesIO(json_bytes))

        self.current_shard_count += 1
        self.total_samples_written += 1

    def close(self) -> Dict[str, Any]:
        """Finalizes all active shards and writes the master index."""
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None

        total_shards = self.current_shard_idx + (1 if self.current_shard_count > 0 else 0)
        index_data = {
            "total_samples": self.total_samples_written,
            "total_shards": total_shards,
            "max_samples_per_shard": self.max_samples_per_shard,
            "shards": [f"{self.shard_prefix}_{i:05d}.tar" for i in range(total_shards)],
        }
        index_path = os.path.join(self.output_dir, "shards_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        return index_data


class ShardedMultimodalIterableDataset(IterableDataset):
    """
    High-Performance PyTorch IterableDataset streaming directly from TAR shards.
    Unpacks samples in memory stream with zero temporary disk extraction.
    Supports both VQ-VAE reconstruction training and Multimodal Transformer autoregressive training.
    """

    def __init__(
        self,
        shard_paths: List[str],
        tokenizer: Optional[ByteTokenizer] = None,
        vqvae: Optional[VQVAE] = None,
        image_size: int = 256,
        max_seq_len: int = 512,
        text_vocab_size: int = 8000,
        image_vocab_size: int = 2048,
        mode: str = "llm",  # 'llm' (text+image tokens) or 'vqvae' (raw image tensors)
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.vqvae = vqvae
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.mode = mode
        self.shuffle_shards = shuffle_shards

    def _process_image(self, img: Image.Image) -> torch.Tensor:
        """Resizes and normalizes image to tensor in range [-1, 1] with shape (3, H, W)."""
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

    def _stream_shard(self, shard_path: str) -> Iterator[Tuple[Image.Image, str, Dict[str, Any]]]:
        """Streams samples from a single TAR file in memory."""
        with tarfile.open(shard_path, "r") as tar:
            samples: Dict[str, Dict[str, Any]] = {}
            for member in tar:
                if not member.isfile():
                    continue
                name_parts = os.path.splitext(member.name)
                base = name_parts[0]
                ext = name_parts[1].lower()

                if base not in samples:
                    samples[base] = {}

                f = tar.extractfile(member)
                if f is not None:
                    data = f.read()
                    if ext == ".png":
                        try:
                            samples[base]["image"] = Image.open(io.BytesIO(data)).convert("RGB")
                        except Exception:
                            pass
                    elif ext == ".txt":
                        samples[base]["caption"] = data.decode("utf-8", errors="replace")
                    elif ext == ".json":
                        try:
                            samples[base]["metadata"] = json.loads(data.decode("utf-8"))
                        except Exception:
                            samples[base]["metadata"] = {}

                # Yield complete samples immediately to conserve RAM
                if "image" in samples[base] and "caption" in samples[base]:
                    img = samples[base]["image"]
                    cap = samples[base]["caption"]
                    meta = samples[base].get("metadata", {})
                    del samples[base]
                    yield img, cap, meta

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Multi-worker shard partitioning
        worker_info = torch.utils.data.get_worker_info()
        shards = list(self.shard_paths)

        if self.shuffle_shards:
            import random
            random.shuffle(shards)

        if worker_info is not None:
            # Partition shards among parallel dataloader workers
            shards = [s for i, s in enumerate(shards) if i % worker_info.num_workers == worker_info.id]

        for shard_path in shards:
            if not os.path.exists(shard_path):
                continue
            for img, caption, meta in self._stream_shard(shard_path):
                img_tensor = self._process_image(img)

                if self.mode == "vqvae":
                    yield {
                        "image": img_tensor,
                        "caption": caption,
                    }
                else:
                    # Multimodal LLM Autoregressive Token Sequence
                    text_tokens = self.tokenizer.encode(caption, add_bos=False, add_eos=False)

                    if self.vqvae is not None:
                        with torch.no_grad():
                            vq_dev = next(self.vqvae.parameters()).device
                            batch_img = img_tensor.unsqueeze(0).to(vq_dev)
                            img_tokens = self.vqvae.encode_to_indices(batch_img).squeeze(0).cpu()
                            img_tokens = (img_tokens + self.text_vocab_size).tolist()
                    else:
                        img_tokens = list(range(self.text_vocab_size, self.text_vocab_size + 256))

                    # [ <bos>, text_tokens, <image_start>, visual_tokens (256), <image_end>, <eos> ]
                    seq = (
                        [self.tokenizer.bos_id]
                        + text_tokens
                        + [self.tokenizer.image_start_id]
                        + img_tokens
                        + [self.tokenizer.image_end_id, self.tokenizer.eos_id]
                    )

                    if len(seq) > self.max_seq_len:
                        seq = seq[:self.max_seq_len]

                    seq_len = len(seq)
                    pad_len = self.max_seq_len - seq_len
                    padded_seq = seq + [self.tokenizer.pad_id] * pad_len

                    input_ids = torch.tensor(padded_seq[:-1], dtype=torch.long)
                    target_ids = torch.tensor(padded_seq[1:], dtype=torch.long)
                    target_ids[seq_len - 1:] = -100  # Cross-entropy ignore index

                    yield {
                        "input_ids": input_ids,
                        "targets": target_ids,
                        "image": img_tensor,
                        "caption": caption,
                    }
