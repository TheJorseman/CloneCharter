"""AutoCharterTrainCollator — collates dataset rows into model-ready tensors.

Extends the base AutoCharterCollator logic with:
  - beat_ids [B, T]: maps each token position to its beat index
    (scanned from BEAT_BOUNDARY tokens, incremented after each occurrence)
  - instrument_ids [B]: integer instrument class
  - difficulty_ids [B]: integer difficulty level (0–6)
  - beat timing tensors [B, N]: bpm_at_beat, time_sig_num_at_beat,
    time_sig_den_at_beat, beat_duration_s — padded to max_beats in batch
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from auto_charter.vocab.tokens import Vocab

_INSTR_TO_ID = {"guitar": 0, "bass": 1, "drums": 2}


def _to_float32_array(v: Any, fallback_shape: tuple) -> np.ndarray:
    """Convert any audio array to a contiguous float32 ndarray.

    Handles the three formats pandas produces when reading nested parquet columns:
      - Already a float32 ndarray → return as-is (or cast)
      - object-dtype array of float arrays (singly nested, e.g. mert [N, 768])
        → np.stack(v)
      - object-dtype array of object-dtype arrays (doubly nested, e.g. logmel
        [N, 32, 128]) → stack each beat, then stack all beats
      - Python list of lists → np.array
    """
    if isinstance(v, np.ndarray):
        if v.dtype != object:
            return v if v.dtype == np.float32 else v.astype(np.float32)
        if len(v) == 0:
            return np.zeros(fallback_shape, np.float32)
        sample = v.flat[0]
        if isinstance(sample, np.ndarray) and sample.dtype == object:
            # Doubly nested (e.g. logmel: num_beats × (32,) object each holding (128,) arrays)
            return np.array(
                [np.stack(sub).astype(np.float32) for sub in v], dtype=np.float32
            )
        # Singly nested (e.g. mert: num_beats × (768,) float32 arrays)
        return np.stack(v).astype(np.float32)
    if v is not None and len(v) > 0:
        return np.array(v, dtype=np.float32)
    return np.zeros(fallback_shape, dtype=np.float32)


class AutoCharterTrainCollator:
    """Collate a batch of dataset rows into padded tensors for AutoCharterModel.

    Args:
        pad_token_id: Token ID for padding (default Vocab.PAD = 0).
        max_tokens: Hard cap on sequence length (truncates longer sequences).
        max_beats: Hard cap on beat sequence length.
    """

    def __init__(
        self,
        pad_token_id: int = Vocab.PAD,
        max_tokens: int = 16384,
        max_beats: int = 1024,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.max_tokens = max_tokens
        self.max_beats = max_beats

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        B = len(batch)

        # ── Token sequences ────────────────────────────────────────────────────
        token_seqs = [list(row["tokens"])[: self.max_tokens] for row in batch]
        max_tok = max(len(s) for s in token_seqs)

        token_array = np.full((B, max_tok), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((B, max_tok), dtype=np.bool_)
        beat_ids_array = np.zeros((B, max_tok), dtype=np.int64)

        for i, seq in enumerate(token_seqs):
            T = len(seq)
            token_array[i, :T] = seq
            attention_mask[i, :T] = True
            beat_ids = self.compute_beat_ids(seq, T)
            beat_ids_array[i, :T] = beat_ids
            # Pad positions get the last valid beat_id
            if T < max_tok:
                last_bid = int(beat_ids[-1]) if T > 0 else 0
                beat_ids_array[i, T:] = last_bid

        # ── Beat-level audio features ──────────────────────────────────────────
        mert_list = []
        for row in batch:
            m = row["mert_embeddings"]
            arr = _to_float32_array(m, fallback_shape=(0,))
            mert_list.append(arr[: self.max_beats])

        logmel_list = []
        for row in batch:
            lm = row["logmel_frames"]
            arr = _to_float32_array(lm, fallback_shape=(0,))
            logmel_list.append(arr[: self.max_beats])

        max_beats_batch = max(m.shape[0] for m in mert_list) if mert_list else 0
        max_beats_batch = max(max_beats_batch, 1)  # avoid zero-size

        # Use max dim across the batch — handles datasets with mixed MERT model versions
        mert_dim = max((m.shape[1] for m in mert_list if m.ndim >= 2 and m.shape[0] > 0), default=1024)
        logmel_shape = next((lm.shape[1:] for lm in logmel_list if lm.ndim >= 3 and lm.shape[0] > 0), (32, 128))

        mert_array = np.zeros((B, max_beats_batch, mert_dim), dtype=np.float32)
        logmel_array = np.zeros((B, max_beats_batch, *logmel_shape), dtype=np.float32)
        beat_mask = np.zeros((B, max_beats_batch), dtype=np.bool_)

        for i, (m, lm) in enumerate(zip(mert_list, logmel_list)):
            nb = m.shape[0]
            if nb > 0:
                d = m.shape[1]  # actual mert dim for this row (may be < mert_dim)
                mert_array[i, :nb, :d] = m
                lm_shape = lm.shape[1:]
                logmel_array[i, :nb, :lm_shape[0], :lm_shape[1]] = lm
                beat_mask[i, :nb] = True

        # ── Beat timing tensors ────────────────────────────────────────────────
        bpm_array = np.ones((B, max_beats_batch), dtype=np.float32) * 120.0
        ts_num_array = np.full((B, max_beats_batch), 4, dtype=np.int64)
        ts_den_array = np.full((B, max_beats_batch), 4, dtype=np.int64)
        dur_array = np.full((B, max_beats_batch), 0.5, dtype=np.float32)

        for i, row in enumerate(batch):
            nb = min(len(row.get("bpm_at_beat", [])), max_beats_batch)
            if nb > 0:
                bpm_array[i, :nb] = list(row["bpm_at_beat"])[:nb]
                ts_num = list(row.get("time_sig_num_at_beat", [4] * nb))[:nb]
                ts_den = list(row.get("time_sig_den_at_beat", [4] * nb))[:nb]
                dur = list(row.get("beat_durations_s", [0.5] * nb))[:nb]
                ts_num_array[i, :nb] = ts_num
                ts_den_array[i, :nb] = ts_den
                dur_array[i, :nb] = dur

        # ── Conditioning ───────────────────────────────────────────────────────
        instrument_ids = np.array(
            [_INSTR_TO_ID.get(row.get("instrument", "guitar"), 0) for row in batch],
            dtype=np.int64,
        )
        difficulty_ids = np.array(
            [max(0, min(6, int(row.get("difficulty", 0)))) for row in batch],
            dtype=np.int64,
        )

        return {
            "input_ids": torch.from_numpy(token_array),
            "labels": torch.from_numpy(token_array.copy()),
            "attention_mask": torch.from_numpy(attention_mask),
            "beat_ids": torch.from_numpy(beat_ids_array),
            "mert_embeddings": torch.from_numpy(mert_array),
            "logmel_frames": torch.from_numpy(logmel_array),
            "beat_attention_mask": torch.from_numpy(beat_mask),
            "beat_padding_mask": torch.from_numpy(beat_mask),
            "bpm_at_beat": torch.from_numpy(bpm_array),
            "time_sig_num": torch.from_numpy(ts_num_array),
            "time_sig_den": torch.from_numpy(ts_den_array),
            "beat_duration_s": torch.from_numpy(dur_array),
            "instrument_ids": torch.from_numpy(instrument_ids),
            "difficulty_ids": torch.from_numpy(difficulty_ids),
        }

    @staticmethod
    def compute_beat_ids(tokens: list[int], pad_to: int) -> np.ndarray:
        """Assign a beat_id to each token position.

        The beat_id starts at 0 and increments AFTER each BEAT_BOUNDARY token.
        The BEAT_BOUNDARY token itself gets the current beat_id before increment.

        Example:
          tokens:   [BOS, INSTR, BEAT, WAIT, NOTE, SUS, BEAT, NOTE, EOS]
          beat_ids: [  0,    0,    0,    1,    1,   1,    1,    2,   2]
        """
        result = np.zeros(pad_to, dtype=np.int64)
        beat_id = 0
        for i, tok in enumerate(tokens):
            if i >= pad_to:
                break
            result[i] = beat_id
            if tok == Vocab.BEAT_BOUNDARY:
                beat_id += 1
        return result
