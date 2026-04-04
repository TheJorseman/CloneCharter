from __future__ import annotations

import json
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset
import datasets as hf_datasets


class CloneHeroDataset(Dataset):
    """
    Wraps the Arrow dataset produced by scripts/tokenize_chart_dataset.py.

    The dataset contains multiple rows per unique song (one per instrument ×
    difficulty combination). To prevent data leakage between train and
    validation splits, we split by unique song identifier (extracted from the
    ``metadata`` JSON field), so every row belonging to a given song goes
    entirely to train OR validation.

    Args:
        dataset_path:  Path to the Arrow dataset directory.
        indices:       Row indices to expose (set by ``build_splits``).
    """

    def __init__(self, dataset_path: str, indices: List[int]):
        self.hf = hf_datasets.load_from_disk(dataset_path)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row = self.hf[self.indices[idx]]

        # log_mel: stored as Sequence[Sequence[Sequence[float64]]]
        # Shape interpretation: [n_mels, n_frames] (the outer Sequence wraps a single matrix)
        logmel_raw = row["logmel_spectrogram"]
        if isinstance(logmel_raw[0][0], list):
            # nested list → take first element
            logmel_raw = logmel_raw[0]
        log_mel = torch.tensor(logmel_raw, dtype=torch.float32)  # [n_mels, n_frames]

        # song_tokens: [n_ticks, 1, variable_len] → flatten to 1-D sequence
        token_ids = self._flatten_tokens(row["song_tokens"])

        return {
            "log_mel": log_mel,                                                # [512, T]
            "token_ids": token_ids,                                            # [seq_len]
            "mert_emb": torch.tensor(row["song_embeddings"], dtype=torch.float32),  # [768]
            "bpm": torch.tensor([row["bpm"]], dtype=torch.float32),            # [1]
            "ts": torch.tensor([row["time_signature"]], dtype=torch.float32),  # [1]
            "resolution": torch.tensor([row["resolution"]], dtype=torch.float32),   # [1]
            "offset": torch.tensor([row["offset"]], dtype=torch.float32),      # [1]
            "instrument": row["instrument"],
            "difficulty": row["difficulty"],
        }

    @staticmethod
    def _flatten_tokens(song_tokens) -> torch.Tensor:
        """
        song_tokens shape in Arrow: [n_ticks, 1, variable_len]
        Flatten all ticks into a single token sequence.
        """
        flat: List[int] = []
        for tick in song_tokens:
            for subtick in tick:
                flat.extend(subtick)
        return torch.tensor(flat, dtype=torch.long)


def build_splits(
    dataset_path: str,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[CloneHeroDataset, CloneHeroDataset]:
    """
    Load the full dataset and split it into train/val by unique song.

    Returns:
        (train_dataset, val_dataset)
    """
    hf = hf_datasets.load_from_disk(dataset_path)
    n = len(hf)

    # Extract a song identifier from the metadata JSON field
    song_ids: List[str] = []
    for i in range(n):
        meta = hf[i]["metadata"]
        try:
            info = json.loads(meta)
            # Use "Name + Artist" as unique key; fall back to raw metadata string
            key = f"{info.get('Name', '')}|{info.get('Artist', '')}"
        except (json.JSONDecodeError, TypeError):
            key = str(meta)
        song_ids.append(key)

    # Unique songs in deterministic order
    unique_songs = list(dict.fromkeys(song_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_songs)

    n_val = max(1, int(len(unique_songs) * val_fraction))
    val_songs = set(unique_songs[-n_val:])

    train_idx, val_idx = [], []
    for i, sid in enumerate(song_ids):
        if sid in val_songs:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return (
        CloneHeroDataset(dataset_path, train_idx),
        CloneHeroDataset(dataset_path, val_idx),
    )
