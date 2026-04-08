"""DataCollator for auto-charter dataset batching.

Handles variable-length sequences:
  - tokens: pad to max_tokens in batch (right-pad with PAD=0)
  - mert_embeddings: pad to max_beats × 768
  - logmel_frames: pad to max_beats × 32 × 128

All padding masks are 1=valid, 0=padded (standard HuggingFace convention).

Usage with HuggingFace Trainer:
    collator = AutoCharterCollator()
    trainer = Trainer(..., data_collator=collator)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from auto_charter.vocab.tokens import Vocab


class AutoCharterCollator:
    """Collate a batch of dataset rows into padded tensors.

    Args:
        pad_token_id: Token ID used for padding (default Vocab.PAD = 0).
        max_tokens: Hard cap on token sequence length (truncates if needed).
        return_tensors: "pt" for PyTorch tensors, "np" for numpy arrays.
    """

    def __init__(
        self,
        pad_token_id: int = Vocab.PAD,
        max_tokens: int = 16384,
        return_tensors: str = "pt",
    ) -> None:
        self.pad_token_id = pad_token_id
        self.max_tokens = max_tokens
        self.return_tensors = return_tensors

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # ── Token sequences ────────────────────────────────────────────────────
        token_seqs = [row["tokens"][:self.max_tokens] for row in batch]
        max_tok = max(len(s) for s in token_seqs)
        token_array = np.full((len(batch), max_tok), self.pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((len(batch), max_tok), dtype=np.int32)
        for i, seq in enumerate(token_seqs):
            token_array[i, :len(seq)] = seq
            attention_mask[i, :len(seq)] = 1

        # ── MERT embeddings ────────────────────────────────────────────────────
        mert_list = [np.array(row["mert_embeddings"], dtype=np.float32)
                     if row["mert_embeddings"] else np.zeros((0, 768), dtype=np.float32)
                     for row in batch]
        max_beats = max(m.shape[0] for m in mert_list) if mert_list else 0
        mert_array = np.zeros((len(batch), max_beats, 768), dtype=np.float32)
        beat_mask = np.zeros((len(batch), max_beats), dtype=np.int32)
        for i, m in enumerate(mert_list):
            if m.shape[0] > 0:
                mert_array[i, :m.shape[0]] = m
                beat_mask[i, :m.shape[0]] = 1

        # ── Log-mel frames ─────────────────────────────────────────────────────
        logmel_list = [np.array(row["logmel_frames"], dtype=np.float32)
                       if row["logmel_frames"] else np.zeros((0, 32, 128), dtype=np.float32)
                       for row in batch]
        logmel_array = np.zeros((len(batch), max_beats, 32, 128), dtype=np.float32)
        for i, lm in enumerate(logmel_list):
            if lm.shape[0] > 0:
                logmel_array[i, :lm.shape[0]] = lm

        # ── Scalar/metadata fields ─────────────────────────────────────────────
        result: dict[str, Any] = {
            "input_ids": token_array,
            "attention_mask": attention_mask,
            "mert_embeddings": mert_array,
            "logmel_frames": logmel_array,
            "beat_attention_mask": beat_mask,
            "labels": token_array.copy(),  # for language-model loss (shift inside model)
        }

        # Pass through scalar metadata
        for key in ("song_id", "instrument", "source_format", "song_name",
                    "artist", "genre", "year", "bpm_mean", "bpm_std",
                    "notes_per_beat_mean", "chord_ratio", "difficulty"):
            if key in batch[0]:
                result[key] = [row.get(key) for row in batch]

        if self.return_tensors == "pt":
            if not _TORCH_AVAILABLE:
                raise ImportError("torch is required for return_tensors='pt'")
            import torch
            for k in ("input_ids", "attention_mask", "mert_embeddings",
                      "logmel_frames", "beat_attention_mask", "labels"):
                if k in result:
                    result[k] = torch.from_numpy(result[k])

        return result
