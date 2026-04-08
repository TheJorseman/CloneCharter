"""AutoCharterConfig — single source of truth for all model hyperparameters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AutoCharterConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 187
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    beat_boundary_id: int = 4

    # ── Architecture ──────────────────────────────────────────────────────────
    d_model: int = 256
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.2
    max_seq_len: int = 16384
    max_beats: int = 1024

    # ── Audio input dims ──────────────────────────────────────────────────────
    mert_dim: int = 768
    logmel_frames: int = 32   # TARGET_FRAMES in LogMelExtractor
    logmel_mels: int = 128    # N_MELS

    # ── Conditioning ─────────────────────────────────────────────────────────
    n_instruments: int = 3    # guitar=0, bass=1, drums=2
    n_difficulties: int = 7   # 0–6

    # ── Training ─────────────────────────────────────────────────────────────
    label_smoothing: float = 0.1
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    learning_rate: float = 3e-4
    lr_scheduler: str = "cosine"

    # ── Generation defaults ───────────────────────────────────────────────────
    max_new_tokens: int = 8192
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AutoCharterConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "AutoCharterConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
