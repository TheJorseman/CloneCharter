"""convert-dataset — Convert existing datasets to Parquet format and upload to Hugging Face Hub.

Usage:
    uv run convert-dataset --input ./my_dataset/ --format parquet --push-to-hub thejorseman/CloneHeroCharts
    uv run convert-dataset --input ./dataset.jsonl --format arrow --output ./converted/
    uv run convert-dataset --input ./dataset.parquet --validate-only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

import click
import numpy as np

logger = logging.getLogger(__name__)


def load_dataset_from_path(input_path: Path) -> "Dataset":
    """Load dataset from various formats (Arrow directory, JSONL, Parquet)."""
    try:
        from datasets import Dataset, load_from_disk, DatasetDict
    except ImportError:
        click.echo("ERROR: datasets is required: pip install datasets", err=True)
        sys.exit(1)

    input_path = Path(input_path)

    # Arrow dataset directory
    if input_path.is_dir():
        if (input_path / "dataset_info.json").exists():
            click.echo(f"Loading Arrow dataset from {input_path}")
            return load_from_disk(str(input_path))
        # Check for split subdirectories
        for subdir in input_path.iterdir():
            if subdir.is_dir() and (subdir / "dataset_info.json").exists():
                click.echo(f"Loading Arrow dataset from {subdir}")
                return load_from_disk(str(subdir))

        # Look for JSONL files in directory
        jsonl_files = list(input_path.glob("*.jsonl"))
        if jsonl_files:
            jsonl_file = jsonl_files[0]
            click.echo(f"Loading JSONL dataset from {jsonl_file}")
            return Dataset.from_json(str(jsonl_file))

    # JSONL file (direct path)
    if input_path.suffix == ".jsonl":
        click.echo(f"Loading JSONL dataset from {input_path}")
        return Dataset.from_json(str(input_path))

    # Parquet file
    if input_path.suffix == ".parquet":
        click.echo(f"Loading Parquet dataset from {input_path}")
        import pandas as pd

        df = pd.read_parquet(input_path)
        return Dataset.from_pandas(df)

    # Single JSON file
    if input_path.suffix == ".json":
        click.echo(f"Loading JSON dataset from {input_path}")
        with open(input_path, "r") as f:
            data = json.load(f)
        return Dataset.from_list(data)

    raise ValueError(f"Unsupported input format: {input_path}")


def _align_table_to_schema(table, target_schema):
    """Cast table columns to match target schema, handling null-type mismatches in list columns."""
    import pyarrow as pa

    arrays = []
    for field in target_schema:
        name = field.name
        if name in table.schema.names:
            col = table.column(name)
            if col.type != field.type:
                try:
                    col = col.cast(field.type)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    col = pa.array([None] * len(table), type=field.type)
        else:
            col = pa.array([None] * len(table), type=field.type)
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=target_schema)


def export_to_parquet(
    dataset: "Dataset",
    output_path: Path,
    compression: str = "zstd",
    batch_size: int = 20,
) -> None:
    """Export dataset to Parquet format with compression using batches to reduce memory."""
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        click.echo(f"ERROR: Required libraries not available: {e}", err=True)
        click.echo("Please install: pip install pandas pyarrow", err=True)
        sys.exit(1)

    click.echo(
        f"Exporting dataset to Parquet ({compression} compression, batch_size={batch_size})..."
    )

    # Batch processing with pyarrow
    click.echo("Using batch processing with pyarrow...")

    writer = None
    total_rows = len(dataset)
    schema_path = output_path.parent / f"{output_path.stem}_schema.json"

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataset.select(range(start_idx, end_idx))

        click.echo(
            f"  Processing batch {start_idx // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size} ({start_idx}-{end_idx})"
        )

        # Convert batch to pandas (small batch, manageable memory)
        df_batch = batch.to_pandas()

        # Initialize writer with first batch schema
        if writer is None:
            # Save schema from first batch
            if len(df_batch) > 0:
                schema = {}
                for col in df_batch.columns:
                    dtype = str(df_batch[col].dtype)
                    sample = df_batch[col].iloc[0] if not df_batch[col].empty else None
                    schema[col] = {"dtype": dtype, "sample_type": type(sample).__name__}

                with open(schema_path, "w") as f:
                    json.dump(schema, f, indent=2)

            # Create Parquet writer
            table = pa.Table.from_pandas(df_batch)
            compression_param = None if compression == "none" else compression
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression=compression_param,
                use_dictionary=True,
            )
            writer.write_table(table)
        else:
            # Write subsequent batches
            table = pa.Table.from_pandas(df_batch)
            if table.schema != writer.schema:
                table = _align_table_to_schema(table, writer.schema)
            writer.write_table(table)

        # Force garbage collection
        import gc

        del df_batch, table
        gc.collect()

    if writer:
        writer.close()
        click.echo(f"Saved Parquet file: {output_path}")
        click.echo(f"Dataset schema: {schema_path}")
    else:
        click.echo("ERROR: No data to write to Parquet", err=True)
        sys.exit(1)


def generate_dataset_card(
    dataset: "Dataset",
    output_dir: Path,
    split: str = "train",
    repo_id: str | None = None,
    manifest: dict | None = None,
) -> None:
    """Generate Hugging Face dataset card README.md."""
    try:
        from huggingface_hub import DatasetCard, DatasetCardData
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping README generation")
        return

    # Basic statistics
    num_rows = len(dataset)
    features = dataset.features if hasattr(dataset, "features") else {}

    # Instrument counts if available
    instrument_counts = {}
    if "instrument" in dataset.column_names:
        from collections import Counter

        instrument_counts = dict(Counter(dataset["instrument"]))

    # Create dataset card data
    card_data = DatasetCardData(
        language="en",
        license="mit",
        annotations_creators="expert-generated",
        task_categories=["audio-generation", "token-classification"],
        task_ids=["music-generation", "chart-generation"],
        pretty_name="Clone Hero Charts Dataset",
        tags=["music", "gaming", "clone-hero", "rhythm-game", "audio-generation"],
    )

    # Create card
    repo = repo_id or "clone-hero/charts"
    card = DatasetCard.from_template(card_data, repo_id=repo, pretty=True)

    # Build statistics table
    stats_table = f"- **Total examples**: {num_rows}\n"
    if instrument_counts:
        stats_table += "- **Instrument distribution**:\n"
        for instr, count in instrument_counts.items():
            stats_table += f"  - {instr}: {count} examples\n"

    if manifest:
        stats_table += "- **Processing configuration**:\n"
        for key, value in manifest.items():
            if key not in ["split", "num_rows", "instruments"]:
                stats_table += f"  - {key}: {value}\n"

    # Custom content
    content = f"""# Dataset Card for Clone Hero Charts

## Dataset Description

This dataset contains tokenized Clone Hero charts with audio conditioning features for automatic chart generation.

### Dataset Summary

{stats_table}

### Supported Tasks

- **Chart generation from audio**: Generate Clone Hero charts from audio input
- **Music transcription**: Transcribe audio to guitar/bass/drums notation
- **Beat detection and alignment**: Align audio features with beat timings

### Languages

Metadata is in English. Music is language-agnostic.

## Dataset Structure

### Data Instances

Each row represents one instrument track from one song. A song with guitar, bass, and drums generates three separate rows.

### Data Fields

Key fields include:
- `tokens`: Integer token sequence (vocabulary size: 187)
- `mert_embeddings`: MERT audio embeddings [num_beats, 768] (optional)
- `logmel_frames`: Log-mel spectrogram frames [num_beats, 32, 128] (optional)
- `instrument`: Instrument name ("guitar", "bass", "drums")
- `beat_times_s`: Beat timing in seconds
- `song_name`, `artist`: Song metadata

### Data Splits

- `{split}`: {num_rows} examples

## Dataset Creation

### Source Data

Clone Hero chart files (.chart, .mid) with optional audio stems.

### Personal and Sensitive Information

No personal information included.

## Considerations for Using the Data

### Social Impact of Dataset

Enables research in music generation, automatic charting, and audio-to-symbolic transcription.

### Discussion of Biases

Charts may reflect popularity biases in music selection. Instrument distribution may be skewed toward guitar.

## Additional Information

### Dataset Curators

Auto-generated by auto-charter toolkit.

### Licensing Information

MIT License

### Citation Information

If you use this dataset, please cite the auto-charter project:
```
@software{{auto_charter,
  title = {{Auto-Charter: Clone Hero Dataset Builder}},
  author = {{TheJorseman}},
  year = {{2025}},
  url = {{https://github.com/thejorseman/auto-charter}}
}}
```

"""

    card.content = content
    readme_path = output_dir / "README.md"
    card.save(readme_path)
    click.echo(f"Generated dataset card: {readme_path}")


def validate_dataset(dataset: "Dataset") -> bool:
    """Validate dataset integrity and schema."""
    click.echo("Validating dataset...")

    # Check required columns
    required_columns = {"song_id", "instrument", "tokens", "num_tokens", "num_beats"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        click.echo(f"ERROR: Missing required columns: {missing}", err=True)
        return False

    # Check token sequences are lists of ints
    try:
        sample_tokens = dataset[0]["tokens"]
        if not isinstance(sample_tokens, (list, tuple)):
            click.echo("ERROR: tokens must be a list", err=True)
            return False
        if not all(isinstance(t, int) for t in sample_tokens):
            click.echo("ERROR: tokens must contain integers", err=True)
            return False
    except Exception as e:
        logger.warning("Could not validate tokens: %s", e)

    click.echo("✓ Dataset validation passed")
    return True


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input dataset (Arrow directory, JSONL, Parquet, or JSON)",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(file_okay=False),
    help="Output directory (default: input directory for conversions)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["parquet", "arrow", "hub"]),
    default="parquet",
    help="Output format (default: parquet)",
)
@click.option(
    "--push-to-hub",
    default=None,
    help="HuggingFace Hub repo ID to push to (e.g., thejorseman/CloneHeroCharts)",
)
@click.option(
    "--split",
    default="train",
    help="Dataset split name (default: train)",
)
@click.option(
    "--parquet-compression",
    default="zstd",
    type=click.Choice(["snappy", "gzip", "zstd", "none"]),
    help="Compression algorithm for Parquet export (default: zstd)",
)
@click.option(
    "--generate-readme",
    is_flag=True,
    default=True,
    help="Generate dataset card README.md (default: on)",
)
@click.option(
    "--validate-only",
    is_flag=True,
    default=False,
    help="Only validate dataset without conversion",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output files",
)
@click.option(
    "--batch-size",
    default=20,
    type=int,
    help="Batch size for Parquet export (reduce for memory issues)",
    show_default=True,
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging verbosity",
)
def main(
    input: str,
    output: str | None,
    format: Literal["parquet", "arrow", "hub"],
    push_to_hub: str | None,
    split: str,
    parquet_compression: str,
    generate_readme: bool,
    validate_only: bool,
    overwrite: bool,
    batch_size: int,
    log_level: str,
) -> None:
    """Convert existing datasets to different formats and upload to Hugging Face Hub."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(input)
    output_dir = Path(output) if output else input_path.parent

    # Load dataset
    try:
        dataset = load_dataset_from_path(input_path)
    except Exception as e:
        click.echo(f"ERROR loading dataset: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loaded dataset: {dataset}")
    click.echo(f"Number of rows: {len(dataset)}")

    # Validate
    if not validate_dataset(dataset):
        if validate_only:
            sys.exit(1)
        click.echo("WARNING: Dataset validation failed, continuing anyway...")

    if validate_only:
        click.echo("Validation complete.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest if exists
    manifest = None
    manifest_path = (
        input_path.parent / "manifest.json"
        if input_path.is_file()
        else input_path / "manifest.json"
    )
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            click.echo(f"Loaded manifest from {manifest_path}")
        except Exception as e:
            logger.warning("Could not load manifest: %s", e)

    # Generate README if requested
    if generate_readme:
        generate_dataset_card(dataset, output_dir, split, push_to_hub, manifest)

    # Convert to requested format
    if format == "parquet":
        parquet_path = output_dir / f"{split}.parquet"
        if parquet_path.exists() and not overwrite:
            click.echo(f"ERROR: Parquet file already exists: {parquet_path}", err=True)
            click.echo("Use --overwrite to replace it.", err=True)
            sys.exit(1)

        export_to_parquet(dataset, parquet_path, parquet_compression)

        # Also save as Arrow for compatibility
        arrow_path = output_dir / split
        if not arrow_path.exists() or overwrite:
            dataset.save_to_disk(str(arrow_path))
            click.echo(f"Saved Arrow dataset to {arrow_path}")

    elif format == "arrow":
        arrow_path = output_dir / split
        if arrow_path.exists() and not overwrite:
            click.echo(f"ERROR: Arrow dataset already exists: {arrow_path}", err=True)
            click.echo("Use --overwrite to replace it.", err=True)
            sys.exit(1)

        dataset.save_to_disk(str(arrow_path))
        click.echo(f"Saved Arrow dataset to {arrow_path}")

        # Also export to Parquet for convenience
        parquet_path = output_dir / f"{split}.parquet"
        if not parquet_path.exists() or overwrite:
            export_to_parquet(dataset, parquet_path, parquet_compression, batch_size)

    # Push to Hugging Face Hub
    if push_to_hub:
        click.echo(f"\nPushing to HuggingFace Hub: {push_to_hub} (split={split})...")
        try:
            # Ensure README exists
            readme_path = output_dir / "README.md"
            if not readme_path.exists() and generate_readme:
                generate_dataset_card(dataset, output_dir, split, push_to_hub, manifest)

            # Push dataset
            if format == "hub":
                # Push as new dataset
                dataset.push_to_hub(
                    push_to_hub, split=split, commit_message=f"Add {split} split"
                )
            else:
                # Push existing dataset directory
                from huggingface_hub import HfApi

                api = HfApi()
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=push_to_hub,
                    repo_type="dataset",
                    commit_message=f"Add {split} split",
                )

            click.echo(
                f"✅ Successfully pushed to https://huggingface.co/datasets/{push_to_hub}"
            )

        except Exception as e:
            click.echo(f"ERROR pushing to hub: {e}", err=True)
            sys.exit(1)

    click.echo("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
