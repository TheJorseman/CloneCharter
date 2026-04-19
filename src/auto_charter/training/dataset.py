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

import ctypes
import gc
import itertools
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset


def _release_ram() -> None:
    """Force the OS to reclaim freed pages after a shard is done.

    gc.collect() frees Python objects but the allocator keeps the pages
    reserved in the process. This function goes further:
      - PyArrow pool: release_unused() returns Arrow-managed pages
      - Windows: SetProcessWorkingSetSize(-1,-1) trims the working set
      - Linux:   malloc_trim(0) returns glibc heap pages to the kernel
    """
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass


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

    @classmethod
    def from_streaming(
        cls,
        iterable_dataset,
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
    ) -> "StreamingAutoCharterDataset":
        """Wrap a HuggingFace IterableDataset for streaming training."""
        return StreamingAutoCharterDataset(
            iterable_dataset,
            max_tokens=max_tokens,
            max_beats=max_beats,
            min_tokens=min_tokens,
        )

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


class StreamingAutoCharterDataset(IterableDataset):
    """Streaming PyTorch IterableDataset wrapping a HuggingFace IterableDataset.

    Filters are applied lazily — no data is loaded into memory upfront.
    Shuffle via HuggingFace's buffer-based shuffle before wrapping.

    Args:
        iterable_dataset: A datasets.IterableDataset (with streaming=True).
        max_tokens: Truncate token sequences longer than this.
        max_beats: Drop rows with more beats than this.
        min_tokens: Drop rows with fewer tokens than this.
    """

    def __init__(
        self,
        iterable_dataset,
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.max_beats = max_beats
        self.min_tokens = min_tokens
        self._data = iterable_dataset.filter(
            lambda row: AutoCharterDataset._keep(row, max_beats, min_tokens)
        )

    def __iter__(self):
        for row in self._data:
            yield row

    @classmethod
    def materialize_val(
        cls,
        iterable_dataset,
        n_samples: int,
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
    ) -> "AutoCharterDataset":
        """Take `n_samples` from a streaming split and return a regular Dataset.

        Materializing validation avoids re-streaming every eval loop and
        ensures reproducible validation metrics across epochs.
        """
        import datasets as hf_datasets

        filtered = iterable_dataset.filter(
            lambda row: AutoCharterDataset._keep(row, max_beats, min_tokens)
        )
        samples = list(itertools.islice(filtered, n_samples))
        if not samples:
            raise ValueError(
                "No valid samples found in the validation stream after filtering. "
                "Check that mert/logmel embeddings are present."
            )
        hf_ds = hf_datasets.Dataset.from_list(samples)
        print(f"StreamingAutoCharterDataset: materialized {len(hf_ds)} val samples")
        return AutoCharterDataset(hf_ds, max_tokens=max_tokens, max_beats=max_beats, min_tokens=min_tokens)


class ShardedParquetDataset(IterableDataset):
    """Streams local parquet shards one file at a time with explicit memory release.

    Unlike HuggingFace streaming (which caches Arrow buffers internally and leaks
    RAM linearly), this class loads one shard, yields its rows, deletes the table,
    and calls gc.collect() before moving to the next shard. RAM stays flat.

    Shuffle is shard-level (shard order is randomized each epoch) plus optional
    row-level shuffle within each shard — no global shuffle buffer needed.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.max_tokens = max_tokens
        self.max_beats = max_beats
        self.min_tokens = min_tokens
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # pandas is used instead of pyarrow.read_table + as_py() because as_py()
        # on large list columns creates millions of Python float objects (24 bytes
        # each), fragmenting the heap permanently. pandas stores those same arrays
        # as numpy (C-heap, contiguous) which malloc_trim/SetProcessWorkingSetSize
        # can actually release back to the OS.
        import pandas as pd

        shards = list(self.shard_paths)
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(shards)

        for shard_path in shards:
            df = pd.read_parquet(str(shard_path))
            col_names = list(df.columns)
            indices = list(range(len(df)))
            if self.shuffle:
                rng.shuffle(indices)

            for i in indices:
                row = {col: df[col].iat[i] for col in col_names}
                if AutoCharterDataset._keep(row, self.max_beats, self.min_tokens):
                    yield row

            del df, indices
            _release_ram()

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        pattern: str = "*.parquet",
        **kwargs,
    ) -> "ShardedParquetDataset":
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {directory}")
        print(f"ShardedParquetDataset: found {len(paths)} shards in {directory}")
        return cls(paths, **kwargs)

    @classmethod
    def train_val_split(
        cls,
        directory: Path | str,
        val_shards: int = 10,
        pattern: str = "*.parquet",
        seed: int = 42,
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
    ) -> tuple["ShardedParquetDataset", "AutoCharterDataset"]:
        """Split shard list into train/val without loading all data into RAM.

        Val shards are fully materialized (small fixed set). Train shards stream
        one file at a time — RAM stays flat throughout training.
        """
        import pandas as pd
        import datasets as hf_datasets

        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {directory}")

        rng = random.Random(seed)
        shuffled = list(paths)
        rng.shuffle(shuffled)
        val_paths = shuffled[:val_shards]
        train_paths = shuffled[val_shards:]

        print(f"ShardedParquetDataset: {len(train_paths)} train shards, {len(val_paths)} val shards")

        ds_kwargs = dict(max_tokens=max_tokens, max_beats=max_beats, min_tokens=min_tokens)

        # Materialize val rows — pandas avoids Python float object fragmentation
        val_rows = []
        for p in val_paths:
            df = pd.read_parquet(str(p))
            col_names = list(df.columns)
            for i in range(len(df)):
                row = {col: df[col].iat[i] for col in col_names}
                if AutoCharterDataset._keep(row, max_beats, min_tokens):
                    val_rows.append(row)
            del df
            _release_ram()

        if not val_rows:
            raise ValueError("No valid rows found in val shards after filtering.")

        val_hf = hf_datasets.Dataset.from_list(val_rows)
        del val_rows
        _release_ram()

        print(f"  → Val materialized: {len(val_hf)} rows")

        train_ds = cls(train_paths, shuffle=True, seed=seed, **ds_kwargs)
        val_ds = AutoCharterDataset(val_hf, **ds_kwargs)
        return train_ds, val_ds
