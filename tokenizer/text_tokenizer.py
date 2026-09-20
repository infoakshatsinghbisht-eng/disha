"""
Custom Byte-level and Subword Tokenizer from Scratch.
Implements UTF-8 byte mappings, BPE pair merges, and multimodal special tokens.
"""

from typing import List, Dict, Tuple, Optional
import json
import os


class ByteTokenizer:
    """
    Byte-level Tokenizer with Special Multimodal Token Management.
    Maps arbitrary UTF-8 text strings to integer token IDs and back.
    Includes reserved token ranges for multimodal markers (<image_start>, <image_end>, etc.).
    """
    
    SPECIAL_TOKENS = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<image_start>",
        "<image_end>",
        "<text_start>",
        "<text_end>",
        "<unk>",
    ]

    def __init__(self, vocab_file: Optional[str] = None):
        # 1. Special tokens get IDs 0 .. len(SPECIAL_TOKENS) - 1
        self.special_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_special: Dict[int, str] = {i: tok for tok, i in self.special_to_id.items()}
        
        self.pad_id = self.special_to_id["<pad>"]
        self.bos_id = self.special_to_id["<bos>"]
        self.eos_id = self.special_to_id["<eos>"]
        self.image_start_id = self.special_to_id["<image_start>"]
        self.image_end_id = self.special_to_id["<image_end>"]
        self.text_start_id = self.special_to_id["<text_start>"]
        self.text_end_id = self.special_to_id["<text_end>"]
        self.unk_id = self.special_to_id["<unk>"]
        
        # 2. Byte tokens start right after special tokens
        self.byte_offset = len(self.SPECIAL_TOKENS)
        self.byte_to_id: Dict[int, int] = {b: self.byte_offset + b for b in range(256)}
        self.id_to_byte: Dict[int, int] = {self.byte_offset + b: b for b in range(256)}
        
        # 3. Merges for BPE
        self.merges: Dict[Tuple[int, int], int] = {}
        self.id_to_token_bytes: Dict[int, bytes] = {
            self.byte_to_id[b]: bytes([b]) for b in range(256)
        }
        
        self.vocab_size = self.byte_offset + 256
        
        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encodes text to a list of integer token IDs."""
        raw_bytes = text.encode("utf-8")
        tokens = [self.byte_to_id[b] for b in raw_bytes]
        
        # Apply BPE merges if present
        if self.merges:
            tokens = self._apply_merges(tokens)
            
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
            
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a sequence of token IDs back into a UTF-8 string."""
        byte_stream = bytearray()
        
        for tok in tokens:
            if tok in self.id_to_special:
                if not skip_special_tokens:
                    byte_stream.extend(self.id_to_special[tok].encode("utf-8"))
                continue
            elif tok in self.id_to_token_bytes:
                byte_stream.extend(self.id_to_token_bytes[tok])
            elif tok in self.id_to_byte:
                byte_stream.append(self.id_to_byte[tok])
            else:
                # Unknown / Out of range
                continue
                
        return byte_stream.decode("utf-8", errors="replace")

    def _apply_merges(self, tokens: List[int]) -> List[int]:
        """Greedily applies BPE pair merges."""
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            # Find earliest learned merge
            min_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if min_pair not in self.merges:
                break
            
            new_id = self.merges[min_pair]
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == min_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def train_bpe(self, corpus: List[str], target_vocab_size: int = 2000):
        """
        Trains BPE merges from a text corpus from scratch.
        """
        # Convert all corpus sentences to initial byte token sequences
        sequences = [[self.byte_to_id[b] for b in text.encode("utf-8")] for text in corpus]
        
        num_merges = max(0, target_vocab_size - self.vocab_size)
        
        for _ in range(num_merges):
            # Count pair frequencies
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                # No more frequent pairs
                break
                
            new_id = self.vocab_size
            self.merges[best_pair] = new_id
            
            # Form token bytes
            b1 = self.id_to_token_bytes.get(best_pair[0], bytes([self.id_to_byte[best_pair[0]]]))
            b2 = self.id_to_token_bytes.get(best_pair[1], bytes([self.id_to_byte[best_pair[1]]]))
            self.id_to_token_bytes[new_id] = b1 + b2
            
            self.vocab_size += 1
            
            # Apply merge to all sequences
            new_sequences = []
            for seq in sequences:
                i = 0
                new_seq = []
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

    def save(self, filepath: str):
        """Saves tokenizer state and merges to JSON."""
        data = {
            "special_tokens": self.SPECIAL_TOKENS,
            "merges": [list(k) + [v] for k, v in self.merges.items()],
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Loads tokenizer state and merges from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.merges = {}
        for item in data.get("merges", []):
            p1, p2, new_id = item
            pair = (p1, p2)
            self.merges[pair] = new_id
            b1 = self.id_to_token_bytes.get(p1, bytes([self.id_to_byte.get(p1, 0)]))
            b2 = self.id_to_token_bytes.get(p2, bytes([self.id_to_byte.get(p2, 0)]))
            self.id_to_token_bytes[new_id] = b1 + b2
            
        self.vocab_size = data.get("vocab_size", self.byte_offset + 256)
