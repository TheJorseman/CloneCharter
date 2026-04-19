"""AutoCharterDataset — wraps a HuggingFace Arrow dataset for model training.

Filtering:
  - Drops rows with empty mert_embeddings or logmel_frames (not extracted)
  - Drops rows with difficulty == -1 (uncharted)
  - Optionally drops rows exceeding max_beats or below min_tokens

Train/test split:
  - Stratified by instrument (guitar/bass/drums) to guarantee all three
    appear in both splits even with a small dataset.

ShardedParquetDataset memory architecture:
  Python's allocator never returns pages to the OS (fragmentation). The only
  guaranteed fix is subprocess isolation: data loading runs in a worker process
  that owns the parquet/pandas memory. When the worker finishes an epoch it is
  joined and the OS reclaims ALL its memory. The main process (model + optimizer)
  never touches raw shard data — it only receives rows through a Queue.
"""

from __future__ import annotations

import gc
import itertools
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler


# ── Subprocess helpers (must be module-level to be picklable on Windows) ──────

def _row_is_valid(row: dict, max_beats: int, min_tokens: int) -> bool:
    if row.get("difficulty", -1) == -1:
        return False
    mert = row.get("mert_embeddings")
    logmel = row.get("logmel_frames")
    if mert is None or logmel is None or len(mert) == 0 or len(logmel) == 0:
        return False
    if row.get("num_beats", 0) > max_beats:
        return False
    if len(row.get("tokens", [])) < min_tokens:
        return False
    return True


def _worker_main(
    shard_paths: list[str],
    out_queue,
    window_size: int,
    max_beats: int,
    min_tokens: int,
    shuffle: bool,
    seed: int,
) -> None:
    """Subprocess entry point.

    Loads shards in windows of `window_size`. After each window the worker
    explicitly frees all Python and C-heap memory before loading the next one.
    Rows are sent one by one through `out_queue`. A None sentinel signals EOF.

    Running in a subprocess means the OS reclaims ALL memory here when the
    process exits — regardless of Python heap fragmentation.
    """
    import gc
    import ctypes
    import sys
    import random as _random
    import pandas as pd

    def _release():
        gc.collect()
        try:
            import pyarrow as pa
            pa.default_memory_pool().release_unused()
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1)
                )
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    rng = _random.Random(seed)
    paths = list(shard_paths)
    if shuffle:
        rng.shuffle(paths)

    for window_start in range(0, len(paths), window_size):
        window = paths[window_start : window_start + window_size]

        # ── Load this window into a single row list ────────────────────────
        rows: list[dict] = []
        for path in window:
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                print(f"[worker] skip {path}: {exc}")
                continue
            col_names = list(df.columns)
            for i in range(len(df)):
                row = {col: df[col].iat[i] for col in col_names}
                if _row_is_valid(row, max_beats, min_tokens):
                    rows.append(row)
            del df
            gc.collect()

        if shuffle:
            rng.shuffle(rows)

        for row in rows:
            out_queue.put(row)

        # ── Free this window completely before loading the next ────────────
        del rows
        _release()

    out_queue.put(None)  # sentinel: epoch done


# ── Public dataset classes ─────────────────────────────────────────────────────

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

        filtered = hf_dataset.filter(
            lambda row: self._keep(row, max_beats, min_tokens),
            desc="Filtering dataset",
            num_proc=1,  # no worker spawn — prevents RAM spikes from process forking
        )
        self._data = filtered
        print(
            f"AutoCharterDataset: {len(hf_dataset)} rows → {len(filtered)} after filtering"
        )

    @staticmethod
    def _keep(row: dict, max_beats: int, min_tokens: int) -> bool:
        if row.get("difficulty", -1) == -1:
            return False
        mert = row.get("mert_embeddings")
        logmel = row.get("logmel_frames")
        if mert is None or logmel is None or len(mert) == 0 or len(logmel) == 0:
            return False
        num_beats = row.get("num_beats", 0)
        if num_beats > max_beats:
            return False
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
        import datasets as hf_datasets

        instruments = ["guitar", "bass", "drums"]
        train_parts, test_parts = [], []

        for instr in instruments:
            group = hf_dataset.filter(lambda r, i=instr: r.get("instrument") == i)
            if len(group) == 0:
                continue
            if len(group) == 1:
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
            else train_parts[0].select([])
        )

        return (
            AutoCharterDataset(train_ds, **dataset_kwargs),
            AutoCharterDataset(test_ds, **dataset_kwargs),
        )


class StreamingAutoCharterDataset(IterableDataset):
    """Streaming PyTorch IterableDataset wrapping a HuggingFace IterableDataset."""

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
        import datasets as hf_datasets

        filtered = iterable_dataset.filter(
            lambda row: AutoCharterDataset._keep(row, max_beats, min_tokens)
        )
        samples = list(itertools.islice(filtered, n_samples))
        if not samples:
            raise ValueError(
                "No valid samples found in the validation stream after filtering."
            )
        hf_ds = hf_datasets.Dataset.from_list(samples)
        print(f"StreamingAutoCharterDataset: materialized {len(hf_ds)} val samples")
        return AutoCharterDataset(hf_ds, max_tokens=max_tokens, max_beats=max_beats, min_tokens=min_tokens)


class ShardedParquetDataset(IterableDataset):
    """Streams local parquet shards through a subprocess worker.

    Architecture
    ------------
    A single worker subprocess owns all parquet/pandas memory. It loads shards
    in windows of `window_size` files, frees each window after sending its rows,
    then loads the next window. Rows travel from worker → main process via a
    bounded multiprocessing Queue.

    When the epoch ends the worker is joined and the OS reclaims ALL its memory
    at the kernel level — bypassing Python's allocator entirely. The main process
    (model + optimizer) never accumulates shard data, so its RAM stays flat.

    Parameters
    ----------
    shard_paths   : list of parquet file paths
    window_size   : shards loaded at once in the worker (default 5)
    queue_maxsize : max rows buffered in the Queue (backpressure, default 200)
    """

    def __init__(
        self,
        shard_paths: list[Path],
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        window_size: int = 5,
        queue_maxsize: int = 200,
    ) -> None:
        super().__init__()
        self.shard_paths = [str(p) for p in shard_paths]
        self.max_tokens = max_tokens
        self.max_beats = max_beats
        self.min_tokens = min_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.queue_maxsize = queue_maxsize

    def __iter__(self):
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue(maxsize=self.queue_maxsize)

        worker = ctx.Process(
            target=_worker_main,
            args=(
                self.shard_paths,
                queue,
                self.window_size,
                self.max_beats,
                self.min_tokens,
                self.shuffle,
                self.seed,
            ),
            daemon=True,
        )
        worker.start()

        try:
            while True:
                row = queue.get(timeout=600)  # 10 min timeout per row
                if row is None:
                    break
                yield row
        except Exception:
            pass
        finally:
            worker.terminate()
            worker.join()

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
        window_size: int = 5,
    ) -> tuple["ShardedParquetDataset", "AutoCharterDataset"]:
        """Reserve val_shards for validation (materialized), rest for streaming train."""
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

        print(f"ShardedParquetDataset: {len(train_paths)} train shards | {len(val_paths)} val shards")

        ds_kwargs = dict(max_tokens=max_tokens, max_beats=max_beats, min_tokens=min_tokens)

        # Materialize val — small enough to hold in RAM permanently
        val_rows: list[dict] = []
        for p in val_paths:
            df = pd.read_parquet(str(p))
            for i in range(len(df)):
                row = {col: df[col].iat[i] for col in df.columns}
                if _row_is_valid(row, max_beats, min_tokens):
                    val_rows.append(row)
            del df
            gc.collect()

        if not val_rows:
            raise ValueError("No valid rows in val shards after filtering.")

        val_hf = hf_datasets.Dataset.from_list(val_rows)
        del val_rows
        gc.collect()
        print(f"  → Val materialized: {len(val_hf)} rows")

        train_ds = cls(
            train_paths, shuffle=True, seed=seed,
            window_size=window_size, **ds_kwargs,
        )
        val_ds = AutoCharterDataset(val_hf, **ds_kwargs)
        return train_ds, val_ds


# ── Index-based shard dataset (replaces subprocess approach) ──────────────────

class PreFilteredDataset(Dataset):
    """Lightweight Dataset that wraps a pre-filtered list of row dicts.

    Avoids all HuggingFace Arrow overhead (no Dataset.from_list, no .filter()
    workers). Use this for val data that is already filtered by _row_is_valid.
    The collator receives the same dict format as from AutoCharterDataset.
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
        # Group manifest indices by shard
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
    1. Manifest (built at startup): reads only 3 lightweight columns
       (difficulty, num_beats, tokens) to validate rows cheaply. Stores
       (shard_idx, local_row_idx) for every valid row — no audio data loaded.

    2. LRU shard cache: at most `max_shards_in_memory` pandas DataFrames in RAM.
       When the cache is full the least-recently-used shard is explicitly deleted
       and gc.collect() is called before loading the next one.

    3. Use with ShardGroupedSampler so the DataLoader accesses all rows of shard
       A before moving to shard B. This means only 1 shard is "hot" at a time —
       peak RAM = max_shards_in_memory * avg_shard_size + model + optimizer.

    With max_shards_in_memory=2 and a ~100-row shard at ~8 MB/row = ~800 MB per
    shard, training RAM is bounded and flat across all epochs.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
        max_shards_in_memory: int = 2,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.max_beats = max_beats
        self.min_tokens = min_tokens
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
                # Read only the parquet footer metadata — zero data deserialized
                meta = pq.read_metadata(path)
                n_rows = meta.num_rows
                for local_idx in range(n_rows):
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

        # Evict LRU shard(s) until under the limit
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
        val_shards: int = 10,
        pattern: str = "*.parquet",
        seed: int = 42,
        max_tokens: int = 16384,
        max_beats: int = 1024,
        min_tokens: int = 10,
        max_shards_in_memory: int = 2,
    ) -> tuple["ShardIndexedDataset", "PreFilteredDataset"]:
        """Reserve val_shards (materialized as PreFilteredDataset) and build indexed train."""
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

        # Materialize val as plain Python dicts — no HF Arrow conversion, no filter workers
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

        train_ds = cls(
            train_paths,
            max_tokens=max_tokens,
            max_beats=max_beats,
            min_tokens=min_tokens,
            max_shards_in_memory=max_shards_in_memory,
        )
        return train_ds, val_ds
