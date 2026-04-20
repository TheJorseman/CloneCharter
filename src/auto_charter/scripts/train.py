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
@click.option("--dataset", "-d", default=None, type=click.Path(), help="Path to local parquet shards directory")
@click.option("--hub-dataset", default=None, help="HuggingFace Hub dataset ID (e.g. user/dataset)")
@click.option("--val-shards", default=5, type=int, show_default=True, help="Number of shards reserved for validation")
@click.option("--shuffle-buffer", default=200, type=int, show_default=True, help="Shuffle buffer size when streaming from Hub")
@click.option("--max-shards-in-memory", default=2, type=int, show_default=True, help="Max parquet shards in RAM at once")
@click.option("--output-dir", "-o", required=True, type=click.Path(), help="Directory for checkpoints and logs")
@click.option("--d-model", default=256, type=int, show_default=True)
@click.option("--n-enc-layers", default=4, type=int, show_default=True)
@click.option("--n-dec-layers", default=4, type=int, show_default=True)
@click.option("--n-heads", default=8, type=int, show_default=True)
@click.option("--d-ff", default=512, type=int, show_default=True)
@click.option("--dropout", default=0.2, type=float, show_default=True)
@click.option("--batch-size", default=4, type=int, show_default=True)
@click.option("--grad-accum", default=4, type=int, show_default=True)
@click.option("--num-epochs", default=100, type=int, show_default=True)
@click.option("--lr", default=3e-4, type=float, show_default=True)
@click.option("--warmup-steps", default=200, type=int, show_default=True)
@click.option("--patience", default=10, type=int, show_default=True, help="Early stopping patience (epochs)")
@click.option("--max-tokens", default=8192, type=int, show_default=True, help="Max token sequence length")
@click.option("--max-beats", default=1024, type=int, show_default=True, help="Max beats per song")
@click.option("--use-wandb", is_flag=True)
@click.option("--mixed-precision", default="bf16", type=click.Choice(["bf16", "fp16", "no"]), show_default=True)
@click.option("--num-workers", default=0, type=int, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--log-every", default=10, type=int, show_default=True)
@click.option("--resume-from", default=None, type=click.Path())
@click.option("--steps-per-epoch", default=0, type=int, show_default=True, help="Override steps/epoch for LR schedule (0 = auto)")
def main(
    dataset, hub_dataset, val_shards, shuffle_buffer, max_shards_in_memory,
    output_dir, d_model, n_enc_layers, n_dec_layers, n_heads, d_ff,
    dropout, batch_size, grad_accum, num_epochs, lr, warmup_steps, patience,
    max_tokens, max_beats, use_wandb, mixed_precision, num_workers,
    seed, log_every, resume_from, steps_per_epoch,
):
    """Train AutoCharterModel on curated parquet shards."""
    import torch

    from auto_charter.model.charter_model import AutoCharterModel
    from auto_charter.model.config import AutoCharterConfig
    from auto_charter.training.dataset import ShardIndexedDataset, PreFilteredDataset
    from auto_charter.training.trainer import AutoCharterTrainer

    if dataset is None and hub_dataset is None:
        raise click.UsageError("Provide either --dataset (local path) or --hub-dataset (Hub ID).")

    torch.manual_seed(seed)

    source = hub_dataset or dataset
    source_path = Path(source) if source else None

    # ── Local parquet shards ──────────────────────────────────────────────────
    if source_path and source_path.exists() and list(source_path.glob("*.parquet")):
        print(f"Loading dataset from local shards: {source_path}")
        train_ds, val_ds = ShardIndexedDataset.train_val_split(
            source_path,
            val_shards=val_shards,
            seed=seed,
            max_shards_in_memory=max_shards_in_memory,
        )
        print(f"Train rows: {len(train_ds)} | Val rows: {len(val_ds)}")

    # ── HuggingFace Hub streaming ─────────────────────────────────────────────
    else:
        import datasets as hf_datasets

        print(f"Loading dataset from Hub: {source}")
        raw_ds = hf_datasets.load_dataset(source, streaming=True)
        if isinstance(raw_ds, hf_datasets.IterableDatasetDict):
            val_key = "test" if "test" in raw_ds else ("validation" if "validation" in raw_ds else None)
            train_split = raw_ds["train"].shuffle(seed=seed, buffer_size=shuffle_buffer)
            val_split = raw_ds[val_key] if val_key else raw_ds["train"]
        else:
            train_split = raw_ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            val_split = raw_ds

        # Materialize val as plain dicts (no Arrow conversion)
        n_val = val_shards * 20
        val_rows = list(hf_datasets.iterable_dataset.ExamplesIterable(
            iter(val_split.take(n_val))
        )) if hasattr(val_split, "take") else [row for row, _ in zip(val_split, range(n_val))]
        val_ds = PreFilteredDataset(val_rows)

        # Train uses HF IterableDataset directly
        train_ds = train_split
        print(f"Hub streaming | Val: {len(val_ds)} samples materialized")

    # ── Model ─────────────────────────────────────────────────────────────────
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
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()
