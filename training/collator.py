from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

from models.conditioning import INSTRUMENT_MAP, DIFFICULTY_MAP


@dataclass
class ChartCollator:
    """
    Collates a list of dataset items into a batch ready for ChartTransformer.

    Log-mel spectrograms are padded in the time dimension to ``enc_max_frames``
    so that AudioCNNFrontEnd always produces exactly ``enc_max_frames // 16``
    audio tokens (512 by default).

    Token sequences are padded to ``dec_max_len``.  The encoder padding mask
    marks positions added by padding as True (PyTorch convention: True = ignore).
    The 7 prefix tokens are never masked.

    Decoder labels use -100 for PAD positions so CrossEntropyLoss ignores them.
    """

    pad_token_id: int = 3
    enc_max_frames: int = 8192    # log-mel time frames  →  8192 // 16 = 512 enc tokens
    dec_max_len: int = 2048
    instrument_map: Dict[str, int] = field(
        default_factory=lambda: INSTRUMENT_MAP
    )
    difficulty_map: Dict[str, int] = field(
        default_factory=lambda: DIFFICULTY_MAP
    )

    def __call__(self, batch: List[dict]) -> dict:
        B = len(batch)

        # ── Log-mel: pad time axis to enc_max_frames ─────────────────────
        n_mels = batch[0]["log_mel"].shape[0]
        log_mel = torch.zeros(B, n_mels, self.enc_max_frames)
        for i, item in enumerate(batch):
            mel = item["log_mel"]                      # [512, T]
            T = min(mel.shape[1], self.enc_max_frames)
            log_mel[i, :, :T] = mel[:, :T]

        # ── Encoder padding mask [B, enc_max_len] ────────────────────────
        # enc_max_len = 7 prefix tokens + enc_max_frames // 16 audio tokens
        n_audio_tokens = self.enc_max_frames // 16
        enc_max_len = 7 + n_audio_tokens                  # 519 (trimmed to 512 in model)
        enc_max_len = min(enc_max_len, 512)

        enc_padding_mask = torch.zeros(B, enc_max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            T = item["log_mel"].shape[1]
            filled_audio = min(T // 16, n_audio_tokens)
            padded_audio = n_audio_tokens - filled_audio
            # Mask the padded audio positions (prefix 7 tokens are always valid)
            start_mask = 7 + filled_audio
            if start_mask < enc_max_len:
                enc_padding_mask[i, start_mask:] = True

        # ── Decoder token sequences ──────────────────────────────────────
        max_toks = min(
            max(item["token_ids"].shape[0] for item in batch),
            self.dec_max_len + 1,  # +1 because we shift for input/label
        )

        decoder_input_ids = torch.full((B, max_toks - 1), self.pad_token_id, dtype=torch.long)
        decoder_labels = torch.full((B, max_toks - 1), -100, dtype=torch.long)
        decoder_attention_mask = torch.zeros(B, max_toks - 1, dtype=torch.long)

        for i, item in enumerate(batch):
            toks = item["token_ids"]
            L = min(toks.shape[0], max_toks)
            # input  = tokens[0 : L-1]
            # labels = tokens[1 : L]
            inp = toks[:L - 1]
            lbl = toks[1:L]
            decoder_input_ids[i, : len(inp)] = inp
            decoder_labels[i, : len(lbl)] = lbl
            decoder_attention_mask[i, : len(inp)] = 1

        # Mask PAD positions in labels
        pad_positions = decoder_input_ids == self.pad_token_id
        decoder_labels[pad_positions] = -100

        # ── Scalar conditioning ──────────────────────────────────────────
        mert_emb = torch.stack([item["mert_emb"] for item in batch])      # [B, 768]
        bpm = torch.stack([item["bpm"] for item in batch])                # [B, 1]
        ts = torch.stack([item["ts"] for item in batch])                  # [B, 1]
        resolution = torch.stack([item["resolution"] for item in batch])  # [B, 1]
        offset = torch.stack([item["offset"] for item in batch])          # [B, 1]

        # ── Categorical conditioning ─────────────────────────────────────
        instrument_idx = torch.tensor(
            [self.instrument_map.get(item["instrument"], 0) for item in batch],
            dtype=torch.long,
        )
        difficulty_idx = torch.tensor(
            [self.difficulty_map.get(item["difficulty"], 0) for item in batch],
            dtype=torch.long,
        )

        return {
            "log_mel": log_mel,                              # [B, 512, 8192]
            "enc_padding_mask": enc_padding_mask,            # [B, 512]
            "mert_emb": mert_emb,                            # [B, 768]
            "bpm": bpm,                                      # [B, 1]
            "ts": ts,                                        # [B, 1]
            "resolution": resolution,                        # [B, 1]
            "offset": offset,                                # [B, 1]
            "instrument_idx": instrument_idx,                # [B]
            "difficulty_idx": difficulty_idx,                # [B]
            "decoder_input_ids": decoder_input_ids,          # [B, S]
            "decoder_labels": decoder_labels,                # [B, S]  -100 at pads
            "decoder_attention_mask": decoder_attention_mask,# [B, S]
        }
