"""process-dataset — build a HuggingFace dataset from Clone Hero song folders.
...
(versión modificada con resumen incremental)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import click
import numpy as np

logger = logging.getLogger(__name__)


# ─── Song discovery ───────────────────────────────────────────────────────────


def find_song_dirs(root: Path, recursive: bool = True) -> list[Path]:
    """Find all song directories under root.

    A song directory contains notes.chart or notes.mid (and usually song.ini).
    """
    songs: list[Path] = []
    pattern = "**/" if recursive else ""

    for fname in ("notes.chart", "notes.mid"):
        for chart_file in root.glob(f"{pattern}{fname}"):
            songs.append(chart_file.parent)

    # Deduplicate while preserving discovery order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in songs:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return sorted(unique)


# ─── Checkpoint and incremental saving ────────────────────────────────────────


def load_processed_set(checkpoint_file: Path) -> set[Path]:
    """Load set of already processed song directories."""
    if not checkpoint_file.exists():
        return set()
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return {Path(line.strip()) for line in f if line.strip()}


def save_processed_song(song_dir: Path, checkpoint_file: Path) -> None:
    """Append a song directory to the checkpoint file."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(str(song_dir.absolute()) + "\n")


def save_rows_to_jsonl(rows: list[dict], jsonl_file: Path) -> None:
    """Append rows to a JSON Lines file."""
    with open(jsonl_file, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_all_rows_from_jsonl(jsonl_file: Path) -> list[dict]:
    """Load all rows from a JSON Lines file."""
    if not jsonl_file.exists():
        return []
    rows = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_from_jsonl(jsonl_file: Path, features=None) -> "Dataset":
    """Load dataset directly from JSONL without intermediate list (memory efficient)."""
    try:
        from datasets import Dataset

        dataset = Dataset.from_json(str(jsonl_file), features=features)
        return dataset
    except Exception as e:
        logger.warning(
            "Could not load dataset from JSONL (%s) — falling back to list loading.", e
        )
        # Fallback to original method for compatibility
        rows = load_all_rows_from_jsonl(jsonl_file)
        return (
            Dataset.from_list(rows, features=features)
            if features
            else Dataset.from_list(rows)
        )


# ─── Per-song processing ──────────────────────────────────────────────────────


@dataclass
class ProcessingConfig:
    instruments: list[str]
    separate_stems: bool
    stems_cache: Path | None
    extract_mert: bool
    mert_model: str
    device: str
    logmel_n_mels: int
    logmel_target_frames: int
    include_beat_boundaries: bool
    force_reseparate: bool


@dataclass
class SongResult:
    song_dir: Path
    rows: list[dict] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False
    separation_done: bool = False
    duration_s: float = 0.0


def process_song(
    song_dir: Path,
    cfg: ProcessingConfig,
    separator=None,
    mert_extractor=None,
    logmel_extractor=None,
) -> SongResult:
    """Process a single song directory into dataset rows."""
    t0 = time.perf_counter()
    result = SongResult(song_dir=song_dir)

    from auto_charter.parsers.chart_parser import parse_chart
    from auto_charter.parsers.midi_parser import parse_midi
    from auto_charter.parsers.ini_parser import parse_ini, SongMetadata
    from auto_charter.tokenizer.encoder import encode_track
    from auto_charter.dataset.builder import _compute_stats

    # ── 1. Parse chart ─────────────────────────────────────────────────────────
    chart_path = song_dir / "notes.chart"
    midi_path = song_dir / "notes.mid"

    if chart_path.exists():
        try:
            chart = parse_chart(chart_path)
            source_format = "chart"
        except Exception as e:
            result.error = f"chart parse error: {e}"
            return result
    elif midi_path.exists():
        try:
            chart = parse_midi(midi_path)
            source_format = "midi"
        except Exception as e:
            result.error = f"midi parse error: {e}"
            return result
    else:
        result.skipped = True
        return result

    # ── 2. Parse metadata ──────────────────────────────────────────────────────
    ini_path = song_dir / "song.ini"
    meta: SongMetadata | None = None
    if ini_path.exists():
        try:
            meta = parse_ini(ini_path)
        except Exception:
            pass

    song_name = meta.name if meta else chart.song_meta.get("name", song_dir.name)
    artist = meta.artist if meta else chart.song_meta.get("artist", "")
    song_id = hashlib.md5(f"{artist}|{song_name}".encode()).hexdigest()[:16]

    # ── 3. Beat grid ───────────────────────────────────────────────────────────
    end_tick = chart.end_tick
    if end_tick == 0:
        result.skipped = True
        return result

    beat_ticks, beat_times_s, beat_durations_s, bpm_at_beat, time_sigs = (
        chart.bpm_map.beat_times(end_tick)
    )
    num_beats = len(beat_ticks)

    # ── 4. Stem resolution (Demucs if needed) ──────────────────────────────────
    instruments_in_chart = [
        i for i in cfg.instruments if i in chart.tracks and chart.tracks[i]
    ]
    if not instruments_in_chart:
        result.skipped = True
        return result

    audio_paths: dict[str, Path] = {}
    if cfg.separate_stems and separator is not None:
        from auto_charter.audio.separator import resolve_or_separate

        try:
            audio_paths = resolve_or_separate(
                song_dir=song_dir,
                instruments=instruments_in_chart,
                stems_cache=cfg.stems_cache,
                separator=separator,
            )
            # Check if separation actually ran
            stems_dir = (
                (cfg.stems_cache / song_dir.name)
                if cfg.stems_cache
                else (song_dir / ".stems")
            )
            result.separation_done = any(
                (stems_dir / f).exists() for f in ("other.wav", "bass.wav", "drums.wav")
            )
        except Exception as e:
            logger.warning("[%s] Stem separation error: %s", song_dir.name, e)
            from auto_charter.audio.stem_loader import resolve_stem_path

            audio_paths = {
                i: p
                for i in instruments_in_chart
                if (p := resolve_stem_path(song_dir, i)) is not None
            }
    else:
        from auto_charter.audio.stem_loader import resolve_stem_path, has_dedicated_stem

        audio_paths = {
            i: p
            for i in instruments_in_chart
            if (p := resolve_stem_path(song_dir, i)) is not None
        }

    # ── 5. Per-instrument rows ─────────────────────────────────────────────────
    for instrument in instruments_in_chart:
        notes = chart.tracks[instrument]
        specials = chart.specials.get(instrument, [])

        # Tokens
        try:
            tokens = encode_track(
                chart=chart,
                instrument=instrument,
                include_beat_boundaries=cfg.include_beat_boundaries,
            )
        except Exception as e:
            logger.warning("[%s/%s] Encode error: %s", song_dir.name, instrument, e)
            continue

        # Audio features
        mert_embeddings: list = []
        logmel_frames: list = []
        stem_path = audio_paths.get(instrument)

        if stem_path is not None and num_beats > 0:
            if cfg.extract_mert and mert_extractor is not None:
                try:
                    arr = mert_extractor.extract_per_beat(
                        stem_path, beat_times_s, beat_durations_s
                    )
                    mert_embeddings = arr.tolist()
                except Exception as e:
                    logger.warning(
                        "[%s/%s] MERT error: %s", song_dir.name, instrument, e
                    )

            if logmel_extractor is not None:
                try:
                    arr = logmel_extractor.extract_per_beat(
                        stem_path, beat_times_s, beat_durations_s
                    )
                    logmel_frames = arr.tolist()
                except Exception as e:
                    logger.warning(
                        "[%s/%s] Log-mel error: %s", song_dir.name, instrument, e
                    )

        # Difficulty
        difficulty = -1
        if meta:
            diff_map = {
                "guitar": meta.diff_guitar,
                "bass": meta.diff_bass,
                "drums": meta.diff_drums,
                "rhythm": meta.diff_rhythm,
            }
            difficulty = diff_map.get(instrument, -1)

        from auto_charter.audio.stem_loader import has_dedicated_stem

        stats = _compute_stats(notes, bpm_at_beat)

        row = {
            # Identity
            "song_id": song_id,
            "instrument": instrument,
            "source_format": source_format,
            # Tokens
            "tokens": tokens,
            "num_tokens": len(tokens),
            "num_beats": num_beats,
            # Audio conditioning
            "mert_embeddings": mert_embeddings,
            "logmel_frames": logmel_frames,
            # Beat timing
            "beat_times_s": [float(t) for t in beat_times_s],
            "beat_durations_s": [float(d) for d in beat_durations_s],
            "bpm_at_beat": [float(b) for b in bpm_at_beat],
            "time_sig_num_at_beat": [ts[0] for ts in time_sigs],
            "time_sig_den_at_beat": [ts[1] for ts in time_sigs],
            # Metadata
            "song_name": song_name,
            "artist": artist,
            "genre": meta.genre if meta else chart.song_meta.get("genre", ""),
            "charter": meta.charter if meta else chart.song_meta.get("charter", ""),
            "year": meta.year if meta else 0,
            "song_length_ms": meta.song_length_ms if meta else 0,
            "difficulty": difficulty,
            "resolution": chart.resolution,
            # Flags
            "has_star_power": any(s.kind == "star_power" for s in specials),
            "has_solo": any(s.kind in ("solo", "soloend") for s in specials),
            "has_dedicated_stem": has_dedicated_stem(song_dir, instrument),
            # Stats
            "num_notes": stats["num_notes"],
            "notes_per_beat_mean": stats["notes_per_beat_mean"],
            "chord_ratio": stats["chord_ratio"],
            "sustain_mean_ticks": stats["sustain_mean_ticks"],
            "bpm_mean": stats["bpm_mean"],
            "bpm_std": stats["bpm_std"],
        }
        result.rows.append(row)

    result.duration_s = time.perf_counter() - t0
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--input-dir",
    "-i",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Song root directory (repeatable: -i dir1/ -i dir2/).",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    type=click.Path(file_okay=False),
    help="Where to save the Arrow dataset (optional if --push-to-hub is set).",
)
@click.option(
    "--push-to-hub",
    default=None,
    help="HuggingFace Hub repo ID, e.g. 'username/clone-hero-charts'.",
)
@click.option(
    "--split",
    default="train",
    show_default=True,
    help="Dataset split name.",
)
@click.option(
    "--instruments",
    default="guitar,bass,drums",
    show_default=True,
    help="Comma-separated instrument list to include.",
)
@click.option(
    "--separate-stems",
    is_flag=True,
    default=False,
    help="Run Demucs on songs that lack dedicated stem files.",
)
@click.option(
    "--stems-cache",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to store Demucs-separated stems (default: .stems/ next to song).",
)
@click.option(
    "--force-reseparate",
    is_flag=True,
    default=False,
    help="Re-run Demucs even when stem files already exist.",
)
@click.option(
    "--demucs-model",
    default="htdemucs",
    show_default=True,
    help="Demucs model name.",
)
@click.option(
    "--extract-mert",
    is_flag=True,
    default=False,
    help="Extract MERT embeddings per beat (slow; GPU recommended).",
)
@click.option(
    "--mert-model",
    default="m-a-p/MERT-v1-95M",
    show_default=True,
    help="HuggingFace MERT model ID.",
)
@click.option(
    "--extract-logmel",
    is_flag=True,
    default=True,
    help="Extract log-mel spectrogram per beat (default: on).",
)
@click.option(
    "--no-logmel",
    is_flag=True,
    default=False,
    help="Disable log-mel extraction.",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Torch device for Demucs and MERT ('cpu', 'cuda', 'mps').",
)
@click.option(
    "--no-beat-tokens",
    is_flag=True,
    default=False,
    help="Omit BEAT_BOUNDARY tokens from token sequences.",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="Scan input directories recursively for song folders.",
)
@click.option(
    "--max-songs",
    default=0,
    type=int,
    help="Stop after processing N songs (0 = no limit, useful for testing).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume from last checkpoint (skip already processed songs).",
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Remove existing checkpoint and JSONL before starting (start fresh).",
)
@click.option(
    "--parquet-export",
    is_flag=True,
    default=False,
    help="Export dataset as single Parquet file alongside Arrow format.",
)
@click.option(
    "--parquet-compression",
    default="zstd",
    type=click.Choice(["snappy", "gzip", "zstd", "none"]),
    show_default=True,
    help="Compression algorithm for Parquet export.",
)
@click.option(
    "--parquet-output-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory for Parquet export (default: same as output-dir).",
)
@click.option(
    "--generate-readme",
    is_flag=True,
    default=True,
    help="Generate dataset card README.md (default: on).",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    show_default=True,
)
def main(
    input_dir: tuple[str, ...],
    output_dir: str | None,
    push_to_hub: str | None,
    split: str,
    instruments: str,
    separate_stems: bool,
    stems_cache: str | None,
    force_reseparate: bool,
    demucs_model: str,
    extract_mert: bool,
    mert_model: str,
    extract_logmel: bool,
    no_logmel: bool,
    device: str,
    no_beat_tokens: bool,
    recursive: bool,
    max_songs: int,
    resume: bool,
    clean: bool,
    parquet_export: bool,
    parquet_compression: str,
    parquet_output_dir: str | None,
    generate_readme: bool,
    log_level: str,
) -> None:
    """Build a HuggingFace dataset from one or more Clone Hero song directories."""
    # ── Setup ──────────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from rich.console import Console
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TaskProgressColumn,
            TimeElapsedColumn,
            TextColumn,
            MofNCompleteColumn,
        )
        from rich.table import Table
        from rich import print as rprint

        _RICH = True
    except ImportError:
        _RICH = False

    console = Console() if _RICH else None

    do_logmel = extract_logmel and not no_logmel
    allowed_instruments = [i.strip() for i in instruments.split(",")]
    stems_cache_path = Path(stems_cache) if stems_cache else None

    cfg = ProcessingConfig(
        instruments=allowed_instruments,
        separate_stems=separate_stems,
        stems_cache=stems_cache_path,
        extract_mert=extract_mert,
        mert_model=mert_model,
        device=device,
        logmel_n_mels=128,
        logmel_target_frames=32,
        include_beat_boundaries=not no_beat_tokens,
        force_reseparate=force_reseparate,
    )

    # ── Discover songs ─────────────────────────────────────────────────────────
    all_song_dirs: list[Path] = []
    for d in input_dir:
        found = find_song_dirs(Path(d), recursive=recursive)
        all_song_dirs.extend(found)

    # Deduplicate across input directories
    seen: set[Path] = set()
    song_dirs: list[Path] = []
    for p in all_song_dirs:
        if p not in seen:
            seen.add(p)
            song_dirs.append(p)

    if max_songs > 0:
        song_dirs = song_dirs[:max_songs]

    total = len(song_dirs)
    if total == 0:
        click.echo(
            "ERROR: No song directories found. Check --input-dir paths.", err=True
        )
        sys.exit(1)

    click.echo(f"Found {total} song director{'y' if total == 1 else 'ies'}.")
    click.echo(f"Instruments : {', '.join(allowed_instruments)}")
    click.echo(
        f"Stem sep    : {'yes (' + demucs_model + ')' if separate_stems else 'no'}"
    )
    click.echo(f"Log-mel     : {'yes' if do_logmel else 'no'}")
    click.echo(f"MERT        : {'yes (' + mert_model + ')' if extract_mert else 'no'}")
    click.echo(f"Device      : {device}")
    click.echo()

    # ── Checkpoint setup ───────────────────────────────────────────────────────
    checkpoint_dir = None
    checkpoint_file = None
    rows_jsonl_file = None
    if output_dir:
        checkpoint_dir = Path(output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "processed_songs.txt"
        rows_jsonl_file = checkpoint_dir / "dataset_rows.jsonl"

        if clean:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            if rows_jsonl_file.exists():
                rows_jsonl_file.unlink()
            click.echo("Cleaned existing checkpoint and JSONL.")
    else:
        if resume:
            click.echo("WARNING: --resume requires --output-dir. Disabling resume.")
            resume = False

    processed_set: set[Path] = set()
    if resume and checkpoint_file and checkpoint_file.exists():
        processed_set = load_processed_set(checkpoint_file)
        click.echo(
            f"Resume mode: {len(processed_set)} songs already processed, will skip them."
        )
    else:
        click.echo("Fresh start (no checkpoint or resume disabled).")

    # Filter out already processed songs
    songs_to_process = [p for p in song_dirs if p not in processed_set]
    if resume and len(processed_set) > 0:
        click.echo(
            f"Skipping {len(processed_set)} already processed songs. {len(songs_to_process)} remaining."
        )
    else:
        songs_to_process = song_dirs

    if not songs_to_process:
        click.echo("All songs already processed. Nothing to do.")
        # Still we need to load existing rows and build dataset
        if rows_jsonl_file and rows_jsonl_file.exists():
            # Try efficient loading first
            try:
                dataset = load_dataset_from_jsonl(rows_jsonl_file)
                if len(dataset) == 0:
                    click.echo("No rows found in JSONL file.")
                    return
                click.echo(
                    f"Loaded {len(dataset)} existing rows efficiently from JSONL."
                )
            except Exception as e:
                logger.warning(
                    "Efficient loading failed (%s), falling back to legacy method.", e
                )
                all_rows = load_all_rows_from_jsonl(rows_jsonl_file)
                if not all_rows:
                    click.echo("No rows found in JSONL file.")
                    return
                dataset = all_rows
                click.echo(f"Loaded {len(all_rows)} existing rows using legacy method.")

            _build_and_save_dataset(
                dataset,
                output_dir,
                push_to_hub,
                split,
                allowed_instruments,
                cfg,
                console,
                parquet_export=parquet_export,
                parquet_compression=parquet_compression,
                parquet_output_dir=parquet_output_dir,
                generate_readme=generate_readme,
            )
        return

    # ── Lazy-init heavy models ─────────────────────────────────────────────────
    separator = None
    if separate_stems:
        try:
            from auto_charter.audio.separator import StemSeparator

            separator = StemSeparator(
                model_name=demucs_model,
                device=device,
                segment=None,
                shifts=1,
            )
        except ImportError as e:
            click.echo(
                f"WARNING: Cannot load Demucs ({e}). Stem separation disabled.",
                err=True,
            )
            separator = None

    mert_extractor = None
    if extract_mert:
        try:
            from auto_charter.audio.mert_extractor import MERTExtractor

            mert_extractor = MERTExtractor(model_name=mert_model, device=device)
            click.echo("MERT model loaded once.")
        except ImportError as e:
            click.echo(
                f"WARNING: Cannot load MERT ({e}). MERT extraction disabled.", err=True
            )

    logmel_extractor = None
    if do_logmel:
        try:
            from auto_charter.audio.logmel import LogMelExtractor

            logmel_extractor = LogMelExtractor(n_mels=128, target_frames=32)
        except ImportError as e:
            click.echo(
                f"WARNING: Cannot load librosa ({e}). Log-mel extraction disabled.",
                err=True,
            )

    # ── Process songs ──────────────────────────────────────────────────────────
    n_passed = 0
    n_failed = 0
    n_skipped = 0
    total_notes = 0
    t_pipeline_start = time.perf_counter()

    def _run_songs(song_list: list[Path]):
        nonlocal n_passed, n_failed, n_skipped, total_notes

        if _RICH:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            task = progress.add_task("Processing songs", total=len(song_list))

        with progress if _RICH else _dummy_ctx() as prog:
            for song_dir in song_list:
                desc = song_dir.name[:50]
                if _RICH:
                    progress.update(task, description=f"[cyan]{desc}")

                result = process_song(
                    song_dir=song_dir,
                    cfg=cfg,
                    separator=separator,
                    mert_extractor=mert_extractor,
                    logmel_extractor=logmel_extractor,
                )

                if result.error:
                    n_failed += 1
                    logger.warning("FAIL  %s — %s", song_dir.name, result.error)
                elif result.skipped:
                    n_skipped += 1
                    logger.debug("SKIP  %s", song_dir.name)
                else:
                    n_passed += 1
                    # Save rows incrementally
                    if rows_jsonl_file:
                        save_rows_to_jsonl(result.rows, rows_jsonl_file)
                    if checkpoint_file:
                        save_processed_song(song_dir, checkpoint_file)

                    for row in result.rows:
                        total_notes += row.get("num_notes", 0)

                    sep_tag = " [demucs]" if result.separation_done else ""
                    logger.info(
                        "OK    %-50s  %d rows  %.1fs%s",
                        song_dir.name,
                        len(result.rows),
                        result.duration_s,
                        sep_tag,
                    )

                if _RICH:
                    progress.advance(task)

    _run_songs(songs_to_process)
    pipeline_duration = time.perf_counter() - t_pipeline_start

    # ── Summary ────────────────────────────────────────────────────────────────
    click.echo()
    if _RICH and console:
        table = Table(title="Processing Summary", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Songs processed", str(n_passed))
        table.add_row("Songs failed", str(n_failed))
        table.add_row("Songs skipped", str(n_skipped))
        table.add_row("Total notes", f"{total_notes:,}")
        table.add_row("Pipeline time", f"{pipeline_duration:.1f}s")
        console.print(table)
    else:
        click.echo(f"Processed: {n_passed}  Failed: {n_failed}  Skipped: {n_skipped}")
        click.echo(f"Total notes: {total_notes:,}")
        click.echo(f"Pipeline time: {pipeline_duration:.1f}s")

    # ── Build Dataset from JSONL ───────────────────────────────────────────────
    if output_dir and rows_jsonl_file and rows_jsonl_file.exists():
        # Try efficient loading first, fall back to legacy method if needed
        try:
            dataset = load_dataset_from_jsonl(rows_jsonl_file)
            if len(dataset) == 0:
                click.echo("\nERROR: No rows found in JSONL file.", err=True)
                sys.exit(1)
            click.echo(f"Loaded {len(dataset)} rows efficiently from JSONL.")
        except Exception as e:
            logger.warning(
                "Efficient loading failed (%s), falling back to legacy method.", e
            )
            all_rows = load_all_rows_from_jsonl(rows_jsonl_file)
            if not all_rows:
                click.echo("\nERROR: No rows found in JSONL file.", err=True)
                sys.exit(1)
            dataset = all_rows
            click.echo(f"Loaded {len(all_rows)} rows using legacy method.")

        _build_and_save_dataset(
            dataset,
            output_dir,
            push_to_hub,
            split,
            allowed_instruments,
            cfg,
            console,
            parquet_export=parquet_export,
            parquet_compression=parquet_compression,
            parquet_output_dir=parquet_output_dir,
            generate_readme=generate_readme,
        )
    elif not output_dir and push_to_hub:
        click.echo(
            "ERROR: --push-to-hub requires --output-dir to store intermediate JSONL.",
            err=True,
        )
        sys.exit(1)
    else:
        click.echo("\nNo output directory specified. Dataset not saved.")


def _build_and_save_dataset(
    all_rows,
    output_dir,
    push_to_hub,
    split,
    instruments,
    cfg,
    console,
    parquet_export=False,
    parquet_compression="zstd",
    parquet_output_dir=None,
    generate_readme=True,
):
    """Build HuggingFace Dataset from all_rows and save/push with optional Parquet export and README generation."""
    click.echo("\nBuilding HuggingFace dataset ...")
    try:
        from datasets import Dataset
    except ImportError:
        click.echo("ERROR: datasets is required: pip install datasets", err=True)
        sys.exit(1)

    # Handle input: could be list[dict], Dataset, or Path to JSONL
    if isinstance(all_rows, (str, Path)):
        # Load dataset directly from JSONL without intermediate list
        try:
            from auto_charter.dataset.schema import get_features

            features = get_features()
            dataset = Dataset.from_json(str(all_rows), features=features)
        except Exception as e:
            logger.warning(
                "Could not load dataset from JSONL with schema (%s) — loading without schema.",
                e,
            )
            dataset = Dataset.from_json(str(all_rows))
    elif isinstance(all_rows, Dataset):
        dataset = all_rows
    else:
        # Assume list[dict]
        try:
            from auto_charter.dataset.schema import get_features

            features = get_features()
            dataset = Dataset.from_list(all_rows, features=features)
        except Exception as e:
            logger.warning(
                "Could not apply typed schema (%s) — building untyped dataset.", e
            )
            dataset = Dataset.from_list(all_rows)

    click.echo(f"Dataset: {dataset}")
    _print_dataset_stats(dataset, instruments)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / split
        dataset.save_to_disk(str(save_path))
        click.echo(f"\nSaved Arrow dataset to {save_path}")

        # Export to Parquet if requested
        if parquet_export:
            parquet_dir = Path(parquet_output_dir) if parquet_output_dir else out_path
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / f"{split}.parquet"
            try:
                import pandas as pd

                df = dataset.to_pandas()
                df.to_parquet(
                    parquet_path, compression=parquet_compression, engine="pyarrow"
                )
                click.echo(
                    f"Exported Parquet dataset to {parquet_path} (compression: {parquet_compression})"
                )

                # Also save schema separately for reference
                schema_path = parquet_dir / f"{split}_schema.json"
                import json

                schema = (
                    {str(k): str(type(v).__name__) for k, v in df.iloc[0].items()}
                    if len(df) > 0
                    else {}
                )
                schema_path.write_text(json.dumps(schema, indent=2))
            except Exception as e:
                logger.warning("Failed to export Parquet dataset: %s", e)
                click.echo(f"WARNING: Parquet export failed: {e}")

        # Generate README.md dataset card if requested
        if generate_readme:
            try:
                _generate_dataset_readme(
                    dataset, out_path, split, instruments, cfg, push_to_hub
                )
            except Exception as e:
                logger.warning("Failed to generate README.md: %s", e)
                click.echo(f"WARNING: README generation failed: {e}")

        # Write a manifest JSON next to the dataset for reference
        manifest = {
            "split": split,
            "num_rows": len(dataset),
            "instruments": instruments,
            "extract_mert": cfg.extract_mert,
            "extract_logmel": True,  # we don't have flag here, but can be inferred
            "separate_stems": cfg.separate_stems,
            "demucs_model": cfg.mert_model if cfg.separate_stems else None,
            "parquet_export": parquet_export,
            "parquet_compression": parquet_compression if parquet_export else None,
            "generate_readme": generate_readme,
        }
        manifest_path = out_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        click.echo(f"Manifest: {manifest_path}")

    if push_to_hub:
        click.echo(f"\nPushing to HuggingFace Hub: {push_to_hub} (split={split}) ...")
        try:
            # Generate dataset card before pushing if not already generated
            if generate_readme and output_dir:
                _generate_dataset_readme(
                    dataset, Path(output_dir), split, instruments, cfg, push_to_hub
                )

            # Push with dataset card if README exists
            readme_path = Path(output_dir) / "README.md" if output_dir else None
            if readme_path and readme_path.exists():
                dataset.push_to_hub(
                    push_to_hub, split=split, commit_message=f"Add {split} split"
                )
            else:
                dataset.push_to_hub(push_to_hub, split=split)
            click.echo(f"Pushed successfully.")
        except Exception as e:
            click.echo(f"ERROR pushing to hub: {e}", err=True)
            sys.exit(1)


def _generate_dataset_readme(
    dataset,
    output_dir: Path,
    split: str,
    instruments: list[str],
    cfg,
    repo_id: str | None = None,
) -> None:
    """Generate Hugging Face dataset card README.md."""
    try:
        from huggingface_hub import DatasetCard, DatasetCardData
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping README generation")
        return

    # Basic dataset statistics
    num_rows = len(dataset)
    instrument_counts = {}
    for instr in instruments:
        count = sum(1 for r in dataset if r["instrument"] == instr)
        if count > 0:
            instrument_counts[instr] = count

    # Create dataset card data with YAML front matter
    card_data = DatasetCardData(
        language="en",
        license="mit",
        annotations_creators="expert-generated",
        task_categories=["audio-generation", "token-classification"],
        task_ids=["music-generation", "chart-generation"],
        pretty_name="Clone Hero Charts Dataset",
        tags=["music", "gaming", "clone-hero", "rhythm-game", "audio-generation"],
    )

    # Create card with template
    card = DatasetCard.from_template(
        card_data,
        repo_id=repo_id or "clone-hero/charts",
        pretty=True,
    )

    # Add custom content
    content = f"""# Dataset Card for Clone Hero Charts

## Dataset Description

This dataset contains tokenized Clone Hero charts with optional audio conditioning features (MERT embeddings and log-mel spectrograms).

### Dataset Summary

- **Number of examples**: {num_rows}
- **Instruments**: {", ".join(instruments)}
- **Audio features**: {"MERT embeddings" if cfg.extract_mert else "None"}, {"Log-mel spectrograms" if hasattr(cfg, "extract_logmel") and cfg.extract_logmel else "None"}
- **Stem separation**: {cfg.separate_stems}

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

See `dataset_info.json` for complete schema. Key fields:
- `tokens`: Integer token sequence (vocabulary size: 187)
- `mert_embeddings`: MERT audio embeddings [num_beats, 768]
- `logmel_frames`: Log-mel spectrogram frames [num_beats, 32, 128]
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

If you use this dataset, please cite the auto-charter project.

"""

    # Update card content
    card.content = content

    # Save README.md
    readme_path = output_dir / "README.md"
    card.save(readme_path)
    click.echo(f"Generated dataset card: {readme_path}")


def _print_dataset_stats(dataset, instruments: list[str]) -> None:
    """Print per-instrument row counts and token stats."""
    try:
        from collections import Counter

        instr_counts = Counter(dataset["instrument"])
        click.echo("\nPer-instrument breakdown:")
        for instr in instruments:
            count = instr_counts.get(instr, 0)
            if count == 0:
                continue
            rows = [r for r in dataset if r["instrument"] == instr]
            token_counts = [r["num_tokens"] for r in rows]
            avg_tok = int(np.mean(token_counts)) if token_counts else 0
            click.echo(f"  {instr:<8}  {count:4d} rows  avg {avg_tok:5d} tokens/row")
    except Exception:
        pass  # stats are non-critical


class _dummy_ctx:
    """No-op context manager used when Rich is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


if __name__ == "__main__":
    main()
