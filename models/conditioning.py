import torch
import torch.nn as nn
from torch import Tensor


# Instrument and difficulty index mappings (must match tokenizer vocabulary)
INSTRUMENT_MAP = {
    "Single": 0,
    "DoubleRhythm": 0,
    "GuitarCoop": 0,
    "DoubleBass": 1,
    "Drums": 2,
}

DIFFICULTY_MAP = {
    "Expert": 0,
    "Hard": 1,
    "Medium": 2,
    "Easy": 3,
}


class ConditioningEncoder(nn.Module):
    """
    Projects scalar song metadata and categorical labels into d_model-dimensional
    prefix tokens that are prepended to the audio token sequence before the encoder.

    Produces 7 prefix tokens:
        [mert, bpm, ts, resolution, offset, instrument, difficulty]
    """

    def __init__(self, d_model: int = 768, mert_dim: int = 768):
        super().__init__()
        self.d_model = d_model

        self.mert_proj = nn.Sequential(
            nn.Linear(mert_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Scalar projections (input is [B, 1])
        self.bpm_proj = nn.Linear(1, d_model)
        self.ts_proj = nn.Linear(1, d_model)
        self.resolution_proj = nn.Linear(1, d_model)
        self.offset_proj = nn.Linear(1, d_model)

        # Categorical embeddings
        self.instrument_emb = nn.Embedding(3, d_model)  # guitar / bass / drums
        self.difficulty_emb = nn.Embedding(4, d_model)  # expert / hard / medium / easy

    @staticmethod
    def normalize_scalars(
        bpm: Tensor,
        ts: Tensor,
        resolution: Tensor,
        offset: Tensor,
    ):
        """Normalize scalars to roughly zero-mean, unit-variance range."""
        bpm_n = (bpm - 120.0) / 60.0
        ts_n = (ts - 4.0) / 2.0
        res_n = (resolution - 192.0) / 96.0
        off_n = offset / 1.0
        return bpm_n, ts_n, res_n, off_n

    def forward(
        self,
        mert_emb: Tensor,       # [B, mert_dim]
        bpm: Tensor,            # [B, 1]
        ts: Tensor,             # [B, 1]
        resolution: Tensor,     # [B, 1]
        offset: Tensor,         # [B, 1]
        instrument_idx: Tensor, # [B]  long
        difficulty_idx: Tensor, # [B]  long
    ) -> Tensor:
        """
        Returns:
            prefix: [B, 7, d_model]
        """
        bpm_n, ts_n, res_n, off_n = self.normalize_scalars(bpm, ts, resolution, offset)

        mert_tok = self.mert_proj(mert_emb).unsqueeze(1)           # [B, 1, d]
        bpm_tok = self.bpm_proj(bpm_n).unsqueeze(1)                # [B, 1, d]
        ts_tok = self.ts_proj(ts_n).unsqueeze(1)                   # [B, 1, d]
        res_tok = self.resolution_proj(res_n).unsqueeze(1)         # [B, 1, d]
        off_tok = self.offset_proj(off_n).unsqueeze(1)             # [B, 1, d]
        inst_tok = self.instrument_emb(instrument_idx).unsqueeze(1) # [B, 1, d]
        diff_tok = self.difficulty_emb(difficulty_idx).unsqueeze(1) # [B, 1, d]

        return torch.cat(
            [mert_tok, bpm_tok, ts_tok, res_tok, off_tok, inst_tok, diff_tok],
            dim=1,
        )  # [B, 7, d_model]
