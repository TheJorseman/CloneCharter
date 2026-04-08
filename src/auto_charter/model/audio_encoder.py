"""AudioEncoder — Transformer encoder over beat-level audio features.

Inputs (per beat):
  mert_embeddings   [B, N, 768]        MERT mean-pooled per beat
  logmel_frames     [B, N, 32, 128]    Log-mel resampled per beat
  bpm_at_beat       [B, N]             BPM value at each beat
  time_sig_num      [B, N] int         Time signature numerator (1–16)
  time_sig_den      [B, N] int         Time signature denominator (1,2,4,8,16→ idx 0–4)
  beat_duration_s   [B, N]             Duration of each beat in seconds
  beat_padding_mask [B, N] bool        True = valid beat, False = padding

Output:
  audio_context     [B, N, d_model]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from auto_charter.model.config import AutoCharterConfig


class AudioEncoder(nn.Module):
    def __init__(self, config: AutoCharterConfig) -> None:
        super().__init__()
        d = config.d_model
        d4 = d // 4

        # MERT projection: [B, N, 768] → [B, N, d]
        self.mert_proj = nn.Linear(config.mert_dim, d)

        # LogMel: apply Linear(128→d) per frame, then mean-pool over 32 frames
        self.logmel_frame_proj = nn.Linear(config.logmel_mels, d)

        # BPM: scalar log(bpm/120) → MLP(1 → d/4 → d)
        self.bpm_mlp = nn.Sequential(
            nn.Linear(1, d4),
            nn.GELU(),
            nn.Linear(d4, d),
        )

        # Time signature: numerator (1–16) + denominator index (0–4)
        self.ts_num_emb = nn.Embedding(17, d4)   # 1..16
        self.ts_den_emb = nn.Embedding(5, d4)    # log2(den): 1→0, 2→1, 4→2, 8→3, 16→4
        self.ts_proj = nn.Linear(d4 * 2, d)

        # Beat duration: log(duration)
        self.dur_mlp = nn.Sequential(
            nn.Linear(1, d4),
            nn.GELU(),
            nn.Linear(d4, d),
        )

        # Learned beat positional encoding
        self.beat_pos_enc = nn.Embedding(config.max_beats, d)

        self.layer_norm_in = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.layers = nn.TransformerEncoder(enc_layer, num_layers=config.n_enc_layers, enable_nested_tensor=False)
        self.layer_norm_out = nn.LayerNorm(d)

    def forward(
        self,
        mert_embeddings: Tensor,      # [B, N, 768]
        logmel_frames: Tensor,        # [B, N, 32, 128]
        bpm_at_beat: Tensor,          # [B, N]
        time_sig_num: Tensor,         # [B, N] int64
        time_sig_den: Tensor,         # [B, N] int64  (raw denominator: 1/2/4/8/16)
        beat_duration_s: Tensor,      # [B, N]
        beat_padding_mask: Tensor,    # [B, N] bool, True=valid
    ) -> Tensor:                      # [B, N, d_model]
        B, N, _ = mert_embeddings.shape
        device = mert_embeddings.device

        # 1. MERT
        mert_feat = self.mert_proj(mert_embeddings)                          # [B, N, d]

        # 2. LogMel: project each of the 32 frames then mean-pool
        # logmel_frames: [B, N, 32, 128]
        lm = self.logmel_frame_proj(logmel_frames)                           # [B, N, 32, d]
        logmel_feat = lm.mean(dim=2)                                         # [B, N, d]

        # 3. BPM
        bpm_input = torch.log(bpm_at_beat.float().clamp(min=1.0) / 120.0).unsqueeze(-1)  # [B, N, 1]
        bpm_feat = self.bpm_mlp(bpm_input)                                   # [B, N, d]

        # 4. Time signature
        # Clamp numerator to valid embedding range [1, 16]
        ts_num_clamped = time_sig_num.long().clamp(1, 16)
        # Convert denominator to index: log2(den).  Clamp to [0,4].
        ts_den_idx = torch.log2(time_sig_den.float().clamp(min=1.0)).long().clamp(0, 4)
        ts_feat = self.ts_proj(
            torch.cat([self.ts_num_emb(ts_num_clamped), self.ts_den_emb(ts_den_idx)], dim=-1)
        )                                                                     # [B, N, d]

        # 5. Beat duration
        dur_input = torch.log(beat_duration_s.float().clamp(min=1e-6)).unsqueeze(-1)  # [B, N, 1]
        dur_feat = self.dur_mlp(dur_input)                                   # [B, N, d]

        # 6. Beat positional encoding
        pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B, N]
        beat_pos = self.beat_pos_enc(pos_ids)                                 # [B, N, d]

        # 7. Combine all features
        x = mert_feat + logmel_feat + bpm_feat + ts_feat + dur_feat + beat_pos  # [B, N, d]
        x = self.layer_norm_in(x)
        x = self.dropout(x)

        # 8. Transformer encoder (src_key_padding_mask: True = IGNORE, so invert)
        padding_mask = ~beat_padding_mask.bool()  # [B, N], True = pad (ignore)
        x = self.layers(x, src_key_padding_mask=padding_mask)

        return self.layer_norm_out(x)
