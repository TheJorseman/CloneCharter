"""AutoCharterDataset — shard-indexed training datasets with bounded RAM.

ShardIndexedDataset memory architecture:
  Builds a manifest from parquet footer metadata (zero data deserialized).
  Loads shards on demand into an LRU cache bounded by max_shards_in_memory.
  Use with ShardGroupedSampler to keep only 1 shard "hot" at a time.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Sampler


class PreFilteredDataset(Dataset):
    """Lightweight Dataset wrapping a pre-loaded list of row dicts.

    Avoids all HuggingFace Arrow overhead. Use for val data already in RAM.
    """

    def __init__(self, rows: list[dict]) -> None:
        super().__init__()
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


class ShardGroupedSampler(Sampler):
    """Yields indices in shard order so the LRU cache only holds 1 hot shard.

    All indices from shard A are emitted before any from shard B. Within each
    shard the order is shuffled. Shard order is reshuffled each epoch via
    set_epoch(), which the trainer calls automatically.
    """

    def __init__(
        self,
        dataset: "ShardIndexedDataset",
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        shard_to_indices: dict[int, list[int]] = {}
        for global_idx, (shard_idx, _) in enumerate(self.dataset._manifest):
            shard_to_indices.setdefault(shard_idx, []).append(global_idx)

        rng = random.Random(self.seed + self._epoch)
        shard_order = list(shard_to_indices.keys())
        if self.shuffle:
            rng.shuffle(shard_order)

        for shard_idx in shard_order:
            indices = list(shard_to_indices[shard_idx])
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices

        self._epoch += 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


class ShardIndexedDataset(Dataset):
    """Map-style dataset with on-demand shard loading and bounded LRU cache.

    Architecture
    ------------
    1. Manifest (built at startup): reads only parquet footer metadata to get
       row counts. Stores (shard_idx, local_row_idx) for every row — no data
       deserialized.

    2. LRU shard cache: at most `max_shards_in_memory` pandas DataFrames live
       in RAM. On eviction, the shard is explicitly deleted and gc.collect()
       is called.

    3. Use with ShardGroupedSampler so all rows from shard A are accessed before
       shard B — only 1 shard is "hot" at a time.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        max_shards_in_memory: int = 2,
    ) -> None:
        super().__init__()
        self.max_shards_in_memory = max(1, max_shards_in_memory)
        self._shard_paths: list[str] = [str(p) for p in shard_paths]
        self._manifest: list[tuple[int, int]] = []  # (shard_idx, local_row_idx)
        self._cache: dict[int, Any] = {}            # shard_idx → pd.DataFrame
        self._cache_lru: list[int] = []             # front = oldest
        self._build_manifest()

    def _build_manifest(self) -> None:
        import pyarrow.parquet as pq

        n = len(self._shard_paths)
        print(f"Building shard manifest from {n} shards (reading row counts only)...")
        for shard_idx, path in enumerate(self._shard_paths):
            try:
                meta = pq.read_metadata(path)
                for local_idx in range(meta.num_rows):
                    self._manifest.append((shard_idx, local_idx))
            except Exception as e:
                print(f"  skip shard {shard_idx} ({path}): {e}")
            if (shard_idx + 1) % 200 == 0:
                print(f"  scanned {shard_idx + 1}/{n} shards — {len(self._manifest)} rows indexed")
        print(f"Manifest ready: {len(self._manifest)} rows across {n} shards")

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, local_idx = self._manifest[idx]
        df = self._get_shard(shard_idx)
        return {col: df[col].iat[local_idx] for col in df.columns}

    def _get_shard(self, shard_idx: int) -> Any:
        import pandas as pd

        if shard_idx in self._cache:
            self._cache_lru.remove(shard_idx)
            self._cache_lru.append(shard_idx)
            return self._cache[shard_idx]

        while len(self._cache) >= self.max_shards_in_memory:
            evict = self._cache_lru.pop(0)
            del self._cache[evict]
            gc.collect()

        df = pd.read_parquet(self._shard_paths[shard_idx])
        self._cache[shard_idx] = df
        self._cache_lru.append(shard_idx)
        return df

    def evict_all(self) -> None:
        """Release all cached shards. Call between epochs to free RAM."""
        self._cache.clear()
        self._cache_lru.clear()
        gc.collect()

    @classmethod
    def train_val_split(
        cls,
        directory: Path | str,
        val_shards: int = 5,
        pattern: str = "*.parquet",
        seed: int = 42,
        max_shards_in_memory: int = 2,
    ) -> tuple["ShardIndexedDataset", "PreFilteredDataset"]:
        """Reserve val_shards (materialized) and build indexed train dataset."""
        import pandas as pd

        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {directory}")

        rng = random.Random(seed)
        shuffled = list(paths)
        rng.shuffle(shuffled)
        val_paths = shuffled[:val_shards]
        train_paths = shuffled[val_shards:]

        print(f"ShardIndexedDataset: {len(train_paths)} train | {len(val_paths)} val shards")

        val_rows: list[dict] = []
        for p in val_paths:
            df = pd.read_parquet(str(p))
            cols = list(df.columns)
            for i in range(len(df)):
                val_rows.append({col: df[col].iat[i] for col in cols})
            del df
            gc.collect()

        if not val_rows:
            raise ValueError("No rows found in val shards.")

        val_ds = PreFilteredDataset(val_rows)
        print(f"  Val: {len(val_ds)} rows in RAM (no Arrow conversion)")

        train_ds = cls(train_paths, max_shards_in_memory=max_shards_in_memory)
        return train_ds, val_ds
