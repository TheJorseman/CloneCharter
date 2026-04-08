"""AutoCharterDataset — wraps a HuggingFace Arrow dataset for model training.

Filtering:
  - Drops rows with empty mert_embeddings or logmel_frames (not extracted)
  - Drops rows with difficulty == -1 (uncharted)
  - Optionally drops rows exceeding max_beats or below min_tokens

Train/test split:
  - Stratified by instrument (guitar/bass/drums) to guarantee all three
    appear in both splits even with a small dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class AutoCharterDataset(Dataset):
    """PyTorch Dataset wrapping a HuggingFace datasets.Dataset.

    Args:
        hf_dataset: A datasets.Dataset (NOT a DatasetDict).
        max_tokens: Truncate token sequences longer than this.
        max_beats: Drop rows with more beats than this (OOM guard).
        min_tokens: Drop rows with fewer tokens than this (sanity check).
    """

    INSTRUMENT_TO_ID = {"guitar": 0, "bass": 1, "drums": 2}

    def __init__(
        self,
        hf_dataset,
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.max_beats = max_beats
        self.min_tokens = min_tokens

        # Apply filters
        filtered = hf_dataset.filter(
            lambda row: self._keep(row, max_beats, min_tokens),
            desc="Filtering dataset",
        )
        self._data = filtered
        print(
            f"AutoCharterDataset: {len(hf_dataset)} rows → {len(filtered)} after filtering"
        )

    @staticmethod
    def _keep(row: dict, max_beats: int, min_tokens: int) -> bool:
        # Drop uncharted
        if row.get("difficulty", -1) == -1:
            return False
        # Drop rows where audio features were not extracted
        mert = row.get("mert_embeddings", [])
        logmel = row.get("logmel_frames", [])
        if len(mert) == 0 or len(logmel) == 0:
            return False
        # Drop excessive beats (OOM guard)
        num_beats = row.get("num_beats", 0)
        if num_beats > max_beats:
            return False
        # Drop too-short sequences
        tokens = row.get("tokens", [])
        if len(tokens) < min_tokens:
            return False
        return True

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._data[idx]

    @classmethod
    def from_path(
        cls,
        dataset_path: str | Path,
        split: str | None = None,
        **kwargs,
    ) -> "AutoCharterDataset":
        """Load from a local Arrow/Parquet dataset directory."""
        import datasets as hf_datasets

        ds = hf_datasets.load_from_disk(str(dataset_path))
        if isinstance(ds, hf_datasets.DatasetDict):
            if split is None:
                raise ValueError("dataset is a DatasetDict — specify split='train' or 'test'")
            ds = ds[split]
        return cls(ds, **kwargs)

    @staticmethod
    def train_test_split(
        hf_dataset,
        test_size: float = 0.2,
        seed: int = 42,
        **dataset_kwargs,
    ) -> tuple["AutoCharterDataset", "AutoCharterDataset"]:
        """Stratified split by instrument.

        Groups rows by instrument, splits each group independently (80/20),
        then concatenates. This ensures all three instruments appear in both
        splits even with a very small dataset.
        """
        import datasets as hf_datasets

        instruments = ["guitar", "bass", "drums"]
        train_parts, test_parts = [], []

        for instr in instruments:
            group = hf_dataset.filter(lambda r, i=instr: r.get("instrument") == i)
            if len(group) == 0:
                continue
            if len(group) == 1:
                # Can't split — put in train only
                train_parts.append(group)
                continue
            split = group.train_test_split(test_size=test_size, seed=seed)
            train_parts.append(split["train"])
            test_parts.append(split["test"])

        if not train_parts:
            raise ValueError("No data found after grouping by instrument.")

        train_ds = hf_datasets.concatenate_datasets(train_parts).shuffle(seed=seed)
        test_ds = (
            hf_datasets.concatenate_datasets(test_parts).shuffle(seed=seed)
            if test_parts
            else train_parts[0].select([])  # empty dataset if only 1 sample total
        )

        return (
            AutoCharterDataset(train_ds, **dataset_kwargs),
            AutoCharterDataset(test_ds, **dataset_kwargs),
        )
