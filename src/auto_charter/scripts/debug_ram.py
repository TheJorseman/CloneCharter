"""RAM profiler for ShardIndexedDataset training pipeline.

Simulates what the DataLoader + collator do and monitors RSS memory after every
batch. When RAM exceeds the threshold it dumps a tracemalloc snapshot (top
allocators) and stops, so you can see exactly what's growing.

Usage:
    uv run debug-ram \
        --dataset F:/CloneCharter_converted/shards \
        --max-shards-in-memory 2 \
        --batch-size 4 \
        --ram-limit-gb 10 \
        --num-batches 500
"""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from pathlib import Path

import click


@click.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Parquet shard directory")
@click.option("--max-shards-in-memory", default=2, type=int, show_default=True)
@click.option("--batch-size", default=4, type=int, show_default=True)
@click.option("--max-beats", default=1024, type=int, show_default=True)
@click.option("--max-tokens", default=8192, type=int, show_default=True)
@click.option("--ram-limit-gb", default=10.0, type=float, show_default=True, help="Stop and dump diagnostics when RSS exceeds this")
@click.option("--num-batches", default=1000, type=int, show_default=True, help="Stop after this many batches (0 = unlimited)")
@click.option("--log-every", default=20, type=int, show_default=True, help="Print RAM every N batches")
def main(dataset, max_shards_in_memory, batch_size, max_beats, max_tokens, ram_limit_gb, num_batches, log_every):
    """Profile RAM usage while iterating through training batches."""
    try:
        import psutil
    except ImportError:
        click.echo("psutil not installed — run: uv add psutil", err=True)
        sys.exit(1)

    import torch
    from torch.utils.data import DataLoader

    from auto_charter.training.collator import AutoCharterTrainCollator
    from auto_charter.training.dataset import ShardGroupedSampler, ShardIndexedDataset, PreFilteredDataset

    proc = psutil.Process(os.getpid())

    def rss_gb() -> float:
        return proc.memory_info().rss / 1024 ** 3

    def top_tracemalloc(n: int = 20) -> str:
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno")
        lines = [f"  {s}" for s in stats[:n]]
        return "\n".join(lines)

    click.echo(f"Initial RSS: {rss_gb():.2f} GB")
    click.echo(f"RAM limit:   {ram_limit_gb:.1f} GB")
    click.echo(f"Batch size:  {batch_size}")
    click.echo(f"Max shards:  {max_shards_in_memory}")
    click.echo()

    tracemalloc.start()

    dataset_path = Path(dataset)
    click.echo("Building ShardIndexedDataset manifest...")
    train_ds, val_ds = ShardIndexedDataset.train_val_split(
        dataset_path,
        val_shards=5,
        seed=42,
        max_shards_in_memory=max_shards_in_memory,
        max_tokens=max_tokens,
        max_beats=max_beats,
    )

    click.echo(f"Train rows: {len(train_ds)} | Val rows: {len(val_ds)}")
    click.echo(f"RSS after manifest: {rss_gb():.2f} GB")
    click.echo()

    sampler = ShardGroupedSampler(train_ds, shuffle=True, seed=42)
    collator = AutoCharterTrainCollator(max_tokens=max_tokens, max_beats=max_beats)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    peak_gb = rss_gb()
    batch_idx = 0

    for batch in loader:
        batch_idx += 1
        current_gb = rss_gb()
        peak_gb = max(peak_gb, current_gb)

        if batch_idx % log_every == 0:
            # Report current shard cache state
            n_cached = len(train_ds._cache)
            cached_shards = list(train_ds._cache.keys())
            click.echo(
                f"Batch {batch_idx:5d} | RSS {current_gb:.2f} GB (peak {peak_gb:.2f}) | "
                f"shards cached: {n_cached} {cached_shards}"
            )

        # Free the batch tensors explicitly
        del batch
        gc.collect()

        if current_gb > ram_limit_gb:
            click.echo()
            click.echo(f"!!! RAM exceeded {ram_limit_gb} GB at batch {batch_idx} !!!")
            click.echo(f"    RSS = {current_gb:.2f} GB  |  peak = {peak_gb:.2f} GB")
            click.echo()

            click.echo("── Shard cache state ─────────────────────────────────────────")
            for shard_idx, df in train_ds._cache.items():
                path = train_ds._shard_paths[shard_idx]
                mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
                click.echo(f"  shard {shard_idx:4d}: {len(df):5d} rows  {mem_mb:.1f} MB  {path}")

            click.echo()
            click.echo("── Top tracemalloc allocations ───────────────────────────────")
            click.echo(top_tracemalloc(30))

            click.echo()
            click.echo("── Large Python objects (gc tracked) ─────────────────────────")
            try:
                import objgraph
                objgraph.show_most_common_types(limit=15)
            except ImportError:
                click.echo("  (install objgraph for object-type breakdown: uv add objgraph)")

            click.echo()
            click.echo("── torch tensor summary ──────────────────────────────────────")
            total_tensor_mb = 0.0
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor) and obj.is_cuda is False:
                    total_tensor_mb += obj.element_size() * obj.nelement() / 1024 ** 2
            click.echo(f"  CPU tensors total: {total_tensor_mb:.1f} MB")

            sys.exit(1)

        if num_batches > 0 and batch_idx >= num_batches:
            click.echo(f"\nReached {num_batches} batches. Final RSS: {rss_gb():.2f} GB (peak {peak_gb:.2f} GB)")
            break

    tracemalloc.stop()
    click.echo(f"\nDone. Peak RSS: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
