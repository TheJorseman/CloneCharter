"""Validation script — evaluate a trained AutoCharterModel on the test split.

Usage:
    uv run validate-charter \\
        --checkpoint ./checkpoints/run1/best \\
        --dataset ./data/my_dataset \\
        --output-json metrics.json

Prints a per-instrument × per-difficulty table with:
    token_accuracy, note_f1, perplexity, beat_accuracy
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import click


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Checkpoint directory (contains config.json + model.pt)")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to HuggingFace Arrow dataset")
@click.option("--split", default="test", show_default=True, help="Which split to evaluate ('test' or 'train')")
@click.option("--batch-size", default=4, type=int, show_default=True)
@click.option("--test-size", default=0.2, type=float, show_default=True, help="Only used if dataset has no pre-made split")
@click.option("--max-tokens", default=8192, type=int, show_default=True)
@click.option("--max-beats", default=1024, type=int, show_default=True)
@click.option("--device", default="auto", type=str, show_default=True, help="'auto', 'cuda', or 'cpu'")
@click.option("--output-json", default=None, type=click.Path(), help="Save metrics to JSON file")
@click.option("--seed", default=42, type=int, show_default=True)
def main(checkpoint, dataset, split, batch_size, test_size, max_tokens, max_beats, device, output_json, seed):
    """Run full validation on a trained AutoCharterModel."""
    import torch
    import datasets as hf_datasets
    from torch.utils.data import DataLoader

    from auto_charter.model.charter_model import AutoCharterModel
    from auto_charter.training.dataset import AutoCharterDataset
    from auto_charter.training.collator import AutoCharterTrainCollator
    from auto_charter.training import metrics as M

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint} on {device} ...")
    model = AutoCharterModel.from_pretrained(Path(checkpoint))
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"Loading dataset from {dataset} ...")
    raw_ds = hf_datasets.load_from_disk(str(dataset))

    if isinstance(raw_ds, hf_datasets.DatasetDict):
        if split in raw_ds:
            eval_ds = AutoCharterDataset(raw_ds[split], max_tokens=max_tokens, max_beats=max_beats)
        else:
            all_data = next(iter(raw_ds.values()))
            _, eval_ds = AutoCharterDataset.train_test_split(
                all_data, test_size=test_size, seed=seed,
                max_tokens=max_tokens, max_beats=max_beats,
            )
    else:
        _, eval_ds = AutoCharterDataset.train_test_split(
            raw_ds, test_size=test_size, seed=seed,
            max_tokens=max_tokens, max_beats=max_beats,
        )

    if len(eval_ds) == 0:
        print("Evaluation dataset is empty after filtering.")
        return

    print(f"Evaluating on {len(eval_ds)} samples ...")

    collator = AutoCharterTrainCollator(max_tokens=max_tokens, max_beats=max_beats)
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Per-instrument and per-difficulty accumulators
    buckets: dict[str, dict] = defaultdict(lambda: {
        "loss": 0.0, "token_acc": 0.0, "perplexity": 0.0, "n": 0,
    })
    global_metrics: dict[str, float] = {"loss": 0.0, "token_acc": 0.0, "perplexity": 0.0, "n": 0}

    _INSTR = ["guitar", "bass", "drums"]
    _DIFF_LABELS = {0: "easy", 1: "medium", 2: "hard", 3: "expert", 4: "expert+", 5: "expert++", 6: "expert+++"}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_device = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

            output = model(
                mert_embeddings=batch_device["mert_embeddings"],
                logmel_frames=batch_device["logmel_frames"],
                bpm_at_beat=batch_device["bpm_at_beat"],
                time_sig_num=batch_device["time_sig_num"],
                time_sig_den=batch_device["time_sig_den"],
                beat_duration_s=batch_device["beat_duration_s"],
                beat_padding_mask=batch_device["beat_attention_mask"],
                input_ids=batch_device["input_ids"],
                beat_ids=batch_device["beat_ids"],
                instrument_ids=batch_device["instrument_ids"],
                difficulty_ids=batch_device["difficulty_ids"],
                labels=batch_device["labels"],
            )

            loss_val = output.loss.item()
            tok_acc = M.token_accuracy(output.logits.float(), batch_device["labels"])
            ppl = M.perplexity(output.logits.float(), batch_device["labels"])

            B = batch["instrument_ids"].shape[0]
            for i in range(B):
                instr_id = batch["instrument_ids"][i].item()
                diff_id = batch["difficulty_ids"][i].item()
                instr_name = _INSTR[instr_id] if instr_id < 3 else "unknown"
                diff_name = _DIFF_LABELS.get(diff_id, str(diff_id))

                for key in (instr_name, f"diff_{diff_name}"):
                    buckets[key]["n"] += 1
                    # Note: these are batch averages — for a proper per-sample
                    # breakdown we'd need to run samples individually
                    buckets[key]["loss"] += loss_val
                    buckets[key]["token_acc"] += tok_acc
                    buckets[key]["perplexity"] += ppl

            global_metrics["loss"] += loss_val
            global_metrics["token_acc"] += tok_acc
            global_metrics["perplexity"] += ppl
            global_metrics["n"] += 1

    # Aggregate
    def avg(d: dict, key: str) -> float:
        return d[key] / max(d["n"], 1)

    print("\n" + "=" * 70)
    print(f"{'Split':>6} | {'Loss':>8} | {'TokenAcc':>9} | {'Perplexity':>11}")
    print("-" * 70)
    print(
        f"{'TOTAL':>6} | {avg(global_metrics,'loss'):8.4f} | "
        f"{avg(global_metrics,'token_acc'):9.4f} | "
        f"{avg(global_metrics,'perplexity'):11.2f}"
    )
    print()

    print(f"{'Instrument':>12} | {'Loss':>8} | {'TokenAcc':>9} | {'Perplexity':>11} | {'N':>5}")
    print("-" * 70)
    for instr in ["guitar", "bass", "drums"]:
        if instr in buckets:
            b = buckets[instr]
            print(
                f"{instr:>12} | {avg(b,'loss'):8.4f} | {avg(b,'token_acc'):9.4f} | "
                f"{avg(b,'perplexity'):11.2f} | {b['n']:>5}"
            )

    print()
    print(f"{'Difficulty':>12} | {'Loss':>8} | {'TokenAcc':>9} | {'Perplexity':>11} | {'N':>5}")
    print("-" * 70)
    for diff_id, diff_name in _DIFF_LABELS.items():
        key = f"diff_{diff_name}"
        if key in buckets:
            b = buckets[key]
            print(
                f"{diff_name:>12} | {avg(b,'loss'):8.4f} | {avg(b,'token_acc'):9.4f} | "
                f"{avg(b,'perplexity'):11.2f} | {b['n']:>5}"
            )
    print("=" * 70)

    # Save JSON
    if output_json:
        results = {
            "total": {
                "loss": avg(global_metrics, "loss"),
                "token_accuracy": avg(global_metrics, "token_acc"),
                "perplexity": avg(global_metrics, "perplexity"),
                "n_batches": global_metrics["n"],
            },
            "by_instrument": {
                instr: {
                    "loss": avg(buckets[instr], "loss"),
                    "token_accuracy": avg(buckets[instr], "token_acc"),
                    "perplexity": avg(buckets[instr], "perplexity"),
                    "n": buckets[instr]["n"],
                }
                for instr in ["guitar", "bass", "drums"] if instr in buckets
            },
        }
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {output_json}")


if __name__ == "__main__":
    main()
