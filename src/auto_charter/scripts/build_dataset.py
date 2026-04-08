"""build_dataset — process a folder of Clone Hero songs and build a HuggingFace dataset.

Usage:
    uv run build-dataset --input-dir PATH/TO/SONGS --output-dir ./out/
    uv run build-dataset --input-dir test_dataset/ --output-dir ./out/ --extract-audio
    uv run build-dataset --input-dir test_dataset/ --push-to-hub username/repo-name

Output: Arrow dataset written to --output-dir, or pushed to the Hub.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--input-dir", "-i", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Root directory containing song folders.")
@click.option("--output-dir", "-o", default=None,
              type=click.Path(file_okay=False),
              help="Where to save the Arrow dataset.")
@click.option("--push-to-hub", default=None,
              help="HuggingFace Hub repo ID to push the dataset (e.g. user/repo).")
@click.option("--extract-audio", is_flag=True, default=False,
              help="Extract MERT and log-mel audio features (slow, requires GPU recommended).")
@click.option("--device", default="cpu",
              help="Torch device for MERT extraction (default: cpu).")
@click.option("--split", default="train",
              help="Dataset split name (default: train).")
@click.option("--instruments", default="guitar,bass,drums",
              help="Comma-separated list of instruments to include.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    input_dir: str,
    output_dir: str | None,
    push_to_hub: str | None,
    extract_audio: bool,
    device: str,
    split: str,
    instruments: str,
    verbose: bool,
) -> None:
    """Build a HuggingFace dataset from Clone Hero song folders."""
    try:
        from datasets import Dataset
    except ImportError:
        click.echo("ERROR: datasets is required: pip install datasets", err=True)
        sys.exit(1)

    from auto_charter.dataset.builder import SongProcessor
    from auto_charter.dataset.schema import get_features

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    allowed_instruments = {i.strip() for i in instruments.split(",")}
    input_path = Path(input_dir)

    song_dirs = sorted(
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not song_dirs:
        click.echo(f"No song directories found in {input_dir}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(song_dirs)} song directories.")

    processor = SongProcessor(
        extract_audio=extract_audio,
        device=device,
    )

    all_rows: list[dict] = []
    for song_dir in song_dirs:
        click.echo(f"Processing: {song_dir.name}")
        try:
            rows = processor.process(song_dir)
        except Exception as e:
            logger.warning("Error processing %s: %s", song_dir.name, e)
            continue

        # Filter to requested instruments
        rows = [r for r in rows if r["instrument"] in allowed_instruments]
        all_rows.extend(rows)
        click.echo(f"  → {len(rows)} rows")

    click.echo(f"\nTotal rows: {len(all_rows)}")

    if not all_rows:
        click.echo("No rows produced. Check that song folders contain notes.chart or notes.mid.",
                   err=True)
        sys.exit(1)

    # Build HuggingFace Dataset
    try:
        features = get_features()
        dataset = Dataset.from_list(all_rows, features=features)
    except Exception as e:
        logger.warning("Could not apply typed schema (%s); building untyped dataset.", e)
        dataset = Dataset.from_list(all_rows)

    click.echo(f"Dataset: {dataset}")

    # Save locally
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(out_path / split))
        click.echo(f"Saved to {out_path / split}")

    # Push to Hub
    if push_to_hub:
        click.echo(f"Pushing to Hub: {push_to_hub} (split={split})")
        dataset.push_to_hub(push_to_hub, split=split)
        click.echo("Done.")

    if not output_dir and not push_to_hub:
        click.echo("\nTip: use --output-dir or --push-to-hub to save the dataset.")


if __name__ == "__main__":
    main()
