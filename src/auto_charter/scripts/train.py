"""Training script for AutoCharterModel.

Launch with accelerate for multi-GPU support:
  # Single GPU / RTX 5070 Ti
  accelerate launch --mixed_precision fp16 src/auto_charter/scripts/train.py \\
      --dataset ./data/my_dataset --output-dir ./checkpoints/run1

  # Single H100 (bf16)
  accelerate launch --mixed_precision bf16 src/auto_charter/scripts/train.py \\
      --dataset ./data/my_dataset --output-dir ./checkpoints/run1

  # 8x H100
  accelerate launch --num_processes 8 --multi_gpu --mixed_precision bf16 \\
      src/auto_charter/scripts/train.py \\
      --dataset ./data/my_dataset --output-dir ./checkpoints/run1

Or directly (single process, no accelerate launcher):
  uv run train-charter --dataset ./data --output-dir ./runs/run1
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--dataset", "-d", default=None, type=click.Path(), help="Path to local HuggingFace Arrow dataset")
@click.option("--hub-dataset", default=None, help="HuggingFace Hub dataset ID (e.g. user/dataset). Use with --streaming.")
@click.option("--streaming", is_flag=True, help="Stream dataset without downloading (works with --hub-dataset or local parquet shards)")
@click.option("--steps-per-epoch", default=1000, type=int, show_default=True, help="Training steps per epoch for cosine LR schedule (required when streaming)")
@click.option("--val-samples", default=500, type=int, show_default=True, help="Validation samples to materialize from the stream")
@click.option("--shuffle-buffer", default=200, type=int, show_default=True, help="Shuffle buffer size when streaming from Hub")
@click.option("--window-size", default=5, type=int, show_default=True, help="Shards loaded per window in the data worker subprocess (local parquet only)")
@click.option("--output-dir", "-o", required=True, type=click.Path(), help="Directory for checkpoints and logs")
@click.option("--d-model", default=256, type=int, show_default=True, help="Model hidden dim")
@click.option("--n-enc-layers", default=4, type=int, show_default=True, help="Number of encoder layers")
@click.option("--n-dec-layers", default=4, type=int, show_default=True, help="Number of decoder layers")
@click.option("--n-heads", default=8, type=int, show_default=True, help="Number of attention heads")
@click.option("--d-ff", default=512, type=int, show_default=True, help="Feed-forward hidden dim")
@click.option("--dropout", default=0.2, type=float, show_default=True)
@click.option("--batch-size", default=4, type=int, show_default=True)
@click.option("--grad-accum", default=4, type=int, show_default=True, help="Gradient accumulation steps")
@click.option("--num-epochs", default=100, type=int, show_default=True)
@click.option("--lr", default=3e-4, type=float, show_default=True, help="Peak learning rate")
@click.option("--warmup-steps", default=200, type=int, show_default=True)
@click.option("--patience", default=10, type=int, show_default=True, help="Early stopping patience (epochs)")
@click.option("--test-size", default=0.2, type=float, show_default=True, help="Fraction of data for validation")
@click.option("--max-tokens", default=8192, type=int, show_default=True, help="Max token sequence length")
@click.option("--max-beats", default=1024, type=int, show_default=True, help="Max beats per song")
@click.option("--use-wandb", is_flag=True, help="Log to W&B instead of TensorBoard")
@click.option("--mixed-precision", default="bf16", type=click.Choice(["bf16", "fp16", "no"]), show_default=True)
@click.option("--num-workers", default=0, type=int, show_default=True, help="DataLoader workers")
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--log-every", default=10, type=int, show_default=True, help="Log every N gradient steps")
@click.option("--resume-from", default=None, type=click.Path(), help="Resume from checkpoint directory")
@click.option("--max-samples", default=None, type=int, show_default=True, help="Limit dataset to N samples (for quick sanity-check runs)")
def main(
    dataset, hub_dataset, streaming, steps_per_epoch, val_samples, shuffle_buffer, window_size,
    output_dir, d_model, n_enc_layers, n_dec_layers, n_heads, d_ff,
    dropout, batch_size, grad_accum, num_epochs, lr, warmup_steps, patience,
    test_size, max_tokens, max_beats, use_wandb, mixed_precision, num_workers,
    seed, log_every, resume_from, max_samples,
):
    """Train AutoCharterModel on a HuggingFace Arrow dataset."""
    import torch
    import datasets as hf_datasets

    from auto_charter.model.charter_model import AutoCharterModel
    from auto_charter.model.config import AutoCharterConfig
    from auto_charter.training.dataset import AutoCharterDataset, StreamingAutoCharterDataset, ShardedParquetDataset
    from auto_charter.training.trainer import AutoCharterTrainer

    if dataset is None and hub_dataset is None:
        raise click.UsageError("Provide either --dataset (local path) or --hub-dataset (Hub ID).")

    torch.manual_seed(seed)

    ds_kwargs = dict(max_tokens=max_tokens, max_beats=max_beats)

    # ── Streaming mode ────────────────────────────────────────────────────────
    if streaming:
        source = hub_dataset or dataset
        print(f"Loading dataset in streaming mode from: {source}")
        source_path = Path(source) if source else None

        # Local parquet shards → ShardedParquetDataset (subprocess worker, flat RAM)
        if source_path and source_path.exists() and list(source_path.glob("*.parquet")):
            train_ds, val_ds = ShardedParquetDataset.train_val_split(
                source_path,
                val_shards=max(1, val_samples // 20),
                seed=seed,
                window_size=window_size,
                **ds_kwargs,
            )
            print(f"Sharded streaming | Val: {len(val_ds)} rows (RAM: 1 shard at a time)")

        # HuggingFace Hub → HF streaming with shuffle buffer
        else:
            raw_ds = hf_datasets.load_dataset(source, streaming=True)
            if isinstance(raw_ds, hf_datasets.IterableDatasetDict):
                if "train" in raw_ds and ("test" in raw_ds or "validation" in raw_ds):
                    train_split = raw_ds["train"].shuffle(seed=seed, buffer_size=shuffle_buffer)
                    val_split_key = "test" if "test" in raw_ds else "validation"
                    val_split = raw_ds[val_split_key]
                else:
                    only_split = next(iter(raw_ds.values()))
                    train_split = only_split.shuffle(seed=seed, buffer_size=shuffle_buffer)
                    val_split = only_split
            else:
                train_split = raw_ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
                val_split = raw_ds
            print(f"  → Hub shuffle buffer: {shuffle_buffer} rows (~{shuffle_buffer * 8 / 1024:.1f} GB RAM)")
            train_ds = StreamingAutoCharterDataset(train_split, **ds_kwargs)
            val_ds = StreamingAutoCharterDataset.materialize_val(val_split, n_samples=val_samples, **ds_kwargs)
            print(f"Hub streaming | Val: {len(val_ds)} samples materialized")

    # ── Normal (in-memory) mode ───────────────────────────────────────────────
    else:
        if dataset is None:
            raise click.UsageError("--dataset is required when not using --streaming.")
        print(f"Loading dataset from {dataset} ...")
        dataset_path = Path(dataset)

        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if parquet_files:
            print(f"  → Found {len(parquet_files)} parquet shards, loading with load_dataset()")
            raw_ds = hf_datasets.load_dataset(
                "parquet",
                data_files=str(dataset_path / "*.parquet"),
                split="train",
            )
        else:
            try:
                raw_ds = hf_datasets.load_from_disk(str(dataset_path))
            except FileNotFoundError:
                train_subdir = dataset_path / "train"
                if train_subdir.exists():
                    print(f"  → Not a DatasetDict root; loading split from {train_subdir}")
                    raw_ds = hf_datasets.load_from_disk(str(train_subdir))
                else:
                    raise

        if max_samples is not None:
            if isinstance(raw_ds, hf_datasets.DatasetDict):
                raw_ds = hf_datasets.DatasetDict({
                    k: v.select(range(min(max_samples, len(v)))) for k, v in raw_ds.items()
                })
            else:
                raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))
            print(f"  → Sanity-check mode: dataset limited to {max_samples} samples")

        if isinstance(raw_ds, hf_datasets.DatasetDict):
            if "train" in raw_ds and "test" in raw_ds:
                train_ds = AutoCharterDataset(raw_ds["train"], **ds_kwargs)
                val_ds = AutoCharterDataset(raw_ds["test"], **ds_kwargs)
            else:
                raw_ds = next(iter(raw_ds.values()))
                train_ds, val_ds = AutoCharterDataset.train_test_split(
                    raw_ds, test_size=test_size, seed=seed, **ds_kwargs,
                )
        else:
            train_ds, val_ds = AutoCharterDataset.train_test_split(
                raw_ds, test_size=test_size, seed=seed, **ds_kwargs,
            )

        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

        if len(train_ds) == 0:
            raise ValueError(
                "Training dataset is empty after filtering. "
                "Make sure your dataset was built with --extract-mert and --extract-logmel."
            )

    # Build model config
    config = AutoCharterConfig(
        d_model=d_model,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        max_seq_len=max_tokens,
        max_beats=max_beats,
    )

    if resume_from:
        print(f"Loading model from {resume_from} ...")
        model = AutoCharterModel.from_pretrained(Path(resume_from))
    else:
        model = AutoCharterModel(config)

    print(f"Model parameters: {model.num_parameters():,}")

    trainer = AutoCharterTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        num_epochs=num_epochs,
        early_stopping_patience=patience,
        use_wandb=use_wandb,
        log_every_n_steps=log_every,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        resume_from=Path(resume_from) if resume_from else None,
        steps_per_epoch=steps_per_epoch if streaming else 0,
    )
    trainer.train()


if __name__ == "__main__":
    main()
