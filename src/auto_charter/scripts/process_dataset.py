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
    Uses os.walk for fast directory traversal (much faster than Path.glob on
    large folder trees with 10k+ directories).
    """
    import os

    _CHART_FILES = {"notes.chart", "notes.mid"}
    # Work with plain strings throughout — avoid creating Path objects in the
    # inner loop (33k+ iterations) and defer conversion to the final step.
    seen: set[str] = set()

    for dirpath, _dirs, files in os.walk(str(root)):
        if not recursive:
            _dirs.clear()
        # set.intersection on two small sets is O(1) in practice
        if _CHART_FILES.intersection(files):
            seen.add(dirpath)

    return sorted(Path(p) for p in seen)


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


class ParquetShardWriter:
    """Write dataset rows directly to Parquet shards without an intermediate JSONL file.

    Rows are buffered and flushed when the estimated buffer size exceeds
    `max_shard_mb` MB. This keeps peak RAM usage bounded regardless of how
    many beats each song has (mert+logmel arrays vary greatly in size).
    """

    def __init__(
        self,
        output_dir: Path,
        split: str,
        compression: str = "zstd",
        max_shard_mb: int = 256,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.compression = compression if compression != "none" else None
        self._max_shard_bytes = max_shard_mb * 1024 * 1024
        self._buffer: list[dict] = []
        self._buffer_bytes: int = 0
        self._shard_index = 0
        self._total_rows = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        # Resume: start shard index after any existing shards
        existing = sorted(output_dir.glob(f"{split}-*.parquet"))
        if existing:
            # Extract the highest index from filenames like "train-00003.parquet"
            try:
                last = int(existing[-1].stem.split("-")[-1])
                self._shard_index = last + 1
            except ValueError:
                self._shard_index = len(existing)
            logger.info(
                "ParquetShardWriter: resuming at shard %d (%d existing shards).",
                self._shard_index, len(existing),
            )

    @staticmethod
    def _estimate_row_bytes(row: dict) -> int:
        """Rough byte estimate for a row based on its float arrays."""
        n_beats = row.get("num_beats", 0)
        mert = n_beats * 768 * 4          # float32
        logmel = n_beats * 32 * 128 * 4   # float32
        tokens = row.get("num_tokens", 0) * 4
        return mert + logmel + tokens + 512  # 512 bytes overhead for metadata

    def add(self, rows: list[dict]) -> None:
        """Add rows to the buffer, flushing when buffer exceeds the byte limit."""
        for row in rows:
            self._buffer.append(row)
            self._buffer_bytes += self._estimate_row_bytes(row)
            if self._buffer_bytes >= self._max_shard_bytes:
                self._flush(self._buffer)
                self._buffer = []
                self._buffer_bytes = 0

    @staticmethod
    def _hf_features_to_arrow_schema():
        """Return a PyArrow schema matching the HuggingFace Features definition."""
        try:
            import pyarrow as pa
            from auto_charter.dataset.schema import get_features
            features = get_features()
            return features.arrow_schema
        except Exception:
            return None  # fall back to inference

    def _flush(self, rows: list[dict]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required: pip install pyarrow")

        hf_schema = self._hf_features_to_arrow_schema()
        if hf_schema is not None:
            # Write with HF schema: ensures list fields use 'item' naming (not PyArrow's
            # default 'element'), which is required for Dataset.from_parquet() to load.
            table = pa.Table.from_pylist(rows, schema=hf_schema)
        else:
            # Fallback: infer schema from data, then cast list<element:x> -> list<item:x>
            # so the output is compatible with HuggingFace datasets regardless of PyArrow version.
            logger.warning(
                "HF schema unavailable — writing shard with inferred schema. "
                "Run repair_shards.py on the output if Dataset.from_parquet fails."
            )
            table = pa.Table.from_pylist(rows)
            # Cast inferred list types to use 'item' naming
            item_schema = pa.schema([
                pa.field(f.name, self._list_element_to_item(f.type), nullable=f.nullable)
                for f in table.schema
            ])
            try:
                table = table.cast(item_schema)
            except Exception:
                pass  # leave as-is if cast fails; repair_shards can fix later

        path = self.output_dir / f"{self.split}-{self._shard_index:05d}.parquet"
        pq.write_table(table, str(path), compression=self.compression)
        self._total_rows += len(rows)
        self._shard_index += 1
        logger.debug("Wrote shard %s (%d rows)", path.name, len(rows))

    @staticmethod
    def _list_element_to_item(t) -> "pa.DataType":
        """Recursively replace list<element:x> with list<item:x>."""
        import pyarrow as pa
        if pa.types.is_list(t):
            return pa.list_(ParquetShardWriter._list_element_to_item(t.value_type))
        if pa.types.is_large_list(t):
            return pa.large_list(ParquetShardWriter._list_element_to_item(t.value_type))
        return t

    def close(self) -> int:
        """Flush remaining buffered rows and return total rows written."""
        if self._buffer:
            self._flush(self._buffer)
            self._buffer = []
        return self._total_rows

    @property
    def total_rows(self) -> int:
        return self._total_rows + len(self._buffer)

    def shard_paths(self) -> list[Path]:
        return sorted(self.output_dir.glob(f"{self.split}-*.parquet"))


def _diagnose_parquet_shards(shards: list[Path]) -> list[Path]:
    """Try loading each shard with datasets.Dataset to find bad ones. Returns bad shard paths."""
    from datasets import Dataset

    bad: list[Path] = []
    for shard in shards:
        try:
            Dataset.from_parquet(str(shard))
        except Exception as exc:
            bad.append(shard)
            logger.error("  BAD SHARD %s: %s", shard.name, exc)

    if not bad:
        logger.error(
            "All %d shards load individually — the failure happens only when loading them together. "
            "This usually means one shard has incompatible Arrow list sizes (e.g. logmel shape mismatch). "
            "Try running with --log-level DEBUG and check for MERT/logmel shape warnings.",
            len(shards),
        )
    else:
        logger.error("%d bad shard(s) identified. Delete them and re-run with --resume to reprocess.", len(bad))
        for p in bad:
            logger.error("  delete: %s", p)
    return bad


def load_dataset_from_parquet_shards(parquet_dir: Path, split: str) -> "Dataset":
    """Load a HuggingFace Dataset from all Parquet shards via PyArrow.

    Uses PyArrow directly instead of Dataset.from_parquet() to avoid:
      - HuggingFace cache writes (~/.cache/huggingface/datasets) which can fill disk
      - 'element' vs 'item' list field naming incompatibility between Parquet spec and HF schema

    Falls back to per-shard loading + concat if bulk read fails.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset

    shards = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found in {parquet_dir}")
    paths = [str(p) for p in shards]

    # 1. Bulk PyArrow load -> Dataset (no HF cache, no element/item issues)
    try:
        table = pq.read_table(paths)
        return Dataset(table)
    except Exception as e:
        logger.warning("PyArrow bulk load failed (%s) — falling back to shard-by-shard.", e)

    # 4. Load each shard individually via PyArrow and concatenate
    tables: list[pa.Table] = []
    failed: list[Path] = []
    for shard in shards:
        try:
            tables.append(pq.read_table(str(shard)))
        except Exception as exc:
            failed.append(shard)
            logger.error("  Skipping bad shard %s: %s", shard.name, exc)

    if failed:
        logger.error(
            "%d shard(s) could not be read. Delete them and re-run with --resume to reprocess.",
            len(failed),
        )
        for p in failed:
            logger.error("  delete: %s", p)

    if not tables:
        _diagnose_parquet_shards(shards)
        raise RuntimeError(f"All {len(shards)} Parquet shards failed to load.")

    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return Dataset(combined)


# ── Legacy JSONL helpers (kept for backwards-compatibility with existing outputs) ──

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
        rows = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    import json as _json
                    rows.append(_json.loads(line))
        from datasets import Dataset
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
    delete_stems_after: bool = False
    max_duration_s: float = 0.0  # 0 = no limit; drop songs longer than this


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

    # ── 3b. Duration guard — drop songs that are too long ─────────────────────
    if cfg.max_duration_s > 0 and len(beat_times_s) > 0:
        estimated_duration_s = float(beat_times_s[-1]) + float(beat_durations_s[-1])
        if estimated_duration_s > cfg.max_duration_s:
            result.skipped = True
            logger.debug(
                "[%s] Skipped: duration %.1fs > max %.1fs",
                song_dir.name, estimated_duration_s, cfg.max_duration_s,
            )
            return result

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

        # Skip rows with missing required audio features
        if not logmel_frames:
            logger.warning(
                "[%s/%s] Skipping: logmel_frames is empty (no audio or extraction failed).",
                song_dir.name, instrument,
            )
            continue
        if cfg.extract_mert and not mert_embeddings:
            logger.warning(
                "[%s/%s] Skipping: mert_embeddings is empty (MERT extraction failed).",
                song_dir.name, instrument,
            )
            continue

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

    # ── 6. Borrar stems generados por Demucs (opcional) ───────────────────────
    if cfg.delete_stems_after and cfg.separate_stems and result.separation_done:
        stems_dir = (
            (cfg.stems_cache / song_dir.name)
            if cfg.stems_cache
            else (song_dir / ".stems")
        )
        deleted_mb = 0.0
        for wav in stems_dir.glob("*.wav"):
            try:
                deleted_mb += wav.stat().st_size / (1024 * 1024)
                wav.unlink()
            except Exception as e:
                logger.warning("[%s] No se pudo borrar %s: %s", song_dir.name, wav.name, e)
        # Borrar el directorio si quedó vacío
        try:
            stems_dir.rmdir()
        except OSError:
            pass  # no estaba vacío o no existe
        if deleted_mb > 0:
            logger.debug("[%s] Stems borrados: %.1f MB", song_dir.name, deleted_mb)

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
    "--delete-stems",
    is_flag=True,
    default=False,
    help="Borrar los WAV generados por Demucs tras extraer los features (ahorra espacio).",
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
    default="m-a-p/MERT-v1-330M",
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
    "--max-duration-s",
    default=0.0,
    type=float,
    show_default=True,
    help="Drop songs whose estimated duration exceeds this many seconds (0 = no limit). "
         "Checked from the beat grid before any audio is loaded.",
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
    "--parquet-compression",
    default="zstd",
    type=click.Choice(["snappy", "gzip", "zstd", "none"]),
    show_default=True,
    help="Compression for Parquet shards (written incrementally during processing).",
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
    delete_stems: bool,
    demucs_model: str,
    extract_mert: bool,
    mert_model: str,
    extract_logmel: bool,
    no_logmel: bool,
    device: str,
    no_beat_tokens: bool,
    recursive: bool,
    max_songs: int,
    max_duration_s: float,
    resume: bool,
    clean: bool,
    parquet_compression: str,
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
        delete_stems_after=delete_stems,
        max_duration_s=max_duration_s,
    )

    # ── Discover songs ─────────────────────────────────────────────────────────
    song_list_cache = Path(output_dir) / "song_list.txt" if output_dir else None

    if song_list_cache and song_list_cache.exists() and not clean:
        click.echo(f"Loading song list from cache: {song_list_cache}")
        with open(song_list_cache, "r", encoding="utf-8") as f:
            all_song_dirs = [Path(l.strip()) for l in f if l.strip()]
        click.echo(f"  {len(all_song_dirs)} songs loaded from cache.")
    else:
        all_song_dirs = []
        for d in input_dir:
            click.echo(f"Scanning {d} ... (this may take a while on HDD)")
            found = find_song_dirs(Path(d), recursive=recursive)
            all_song_dirs.extend(found)
            click.echo(f"  {len(found)} songs found.")
        if song_list_cache:
            with open(song_list_cache, "w", encoding="utf-8") as f:
                f.writelines(str(p) + "\n" for p in all_song_dirs)
            click.echo(f"Song list cached to {song_list_cache}")

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
    parquet_shard_dir = None
    shard_writer: ParquetShardWriter | None = None
    if output_dir:
        checkpoint_dir = Path(output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "processed_songs.txt"
        parquet_shard_dir = checkpoint_dir / "shards"

        if clean:
            import shutil
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            if parquet_shard_dir.exists():
                shutil.rmtree(parquet_shard_dir)
            click.echo("Cleaned existing checkpoint and Parquet shards.")
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
        if parquet_shard_dir and parquet_shard_dir.exists():
            try:
                dataset = load_dataset_from_parquet_shards(parquet_shard_dir, split)
                click.echo(f"Loaded {len(dataset)} existing rows from Parquet shards.")
                _build_and_save_dataset(
                    dataset,
                    output_dir,
                    push_to_hub,
                    split,
                    allowed_instruments,
                    cfg,
                    console,
                    parquet_shard_dir=parquet_shard_dir,
                    parquet_compression=parquet_compression,
                    generate_readme=generate_readme,
                )
            except FileNotFoundError:
                click.echo("No Parquet shards found. Nothing to do.")
        return

    # ── Lazy-init heavy models ─────────────────────────────────────────────────
    separator = None
    if separate_stems:
        click.echo("Loading Demucs model ...", nl=False)
        try:
            from auto_charter.audio.separator import StemSeparator

            separator = StemSeparator(
                model_name=demucs_model,
                device=device,
                segment=None,
                shifts=1,
            )
            click.echo(" done.")
        except ImportError as e:
            click.echo(f" FAILED ({e}). Stem separation disabled.", err=True)
            separator = None

    mert_extractor = None
    if extract_mert:
        click.echo(f"Loading MERT model ({mert_model}) ...", nl=False)
        try:
            from auto_charter.audio.mert_extractor import MERTExtractor

            mert_extractor = MERTExtractor(model_name=mert_model, device=device)
            click.echo(" done.")
        except ImportError as e:
            click.echo(f" FAILED ({e}). MERT extraction disabled.", err=True)

    logmel_extractor = None
    if do_logmel:
        click.echo("Initializing log-mel extractor ...", nl=False)
        try:
            from auto_charter.audio.logmel import LogMelExtractor

            logmel_extractor = LogMelExtractor(n_mels=128, target_frames=32)
            click.echo(" done.")
        except ImportError as e:
            click.echo(f" FAILED ({e}). Log-mel extraction disabled.", err=True)

    # ── Process songs ──────────────────────────────────────────────────────────
    # Initialise Parquet shard writer
    if parquet_shard_dir is not None:
        shard_writer = ParquetShardWriter(
            output_dir=parquet_shard_dir,
            split=split,
            compression=parquet_compression,
        )
        click.echo(f"Writing directly to Parquet shards in {parquet_shard_dir}")

    n_passed = 0
    n_failed = 0
    n_skipped = 0
    n_rows_ok = 0
    n_rows_dropped_logmel = 0
    n_rows_dropped_mert = 0
    n_demucs_used = 0
    total_notes = 0
    fail_reasons: dict[str, int] = {}
    t_pipeline_start = time.perf_counter()
    _PROGRESS_INTERVAL = 10  # print a status line every N songs

    def _run_songs(song_list: list[Path]):
        nonlocal n_passed, n_failed, n_skipped, n_rows_ok
        nonlocal n_rows_dropped_logmel, n_rows_dropped_mert, n_demucs_used, total_notes

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

        total_songs = len(song_list)
        with progress if _RICH else _dummy_ctx() as prog:
            for idx, song_dir in enumerate(song_list, 1):
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
                    # Bucket errors by their prefix (e.g. "midi parse error", "chart parse error")
                    bucket = result.error.split(":")[0].strip()
                    fail_reasons[bucket] = fail_reasons.get(bucket, 0) + 1
                    logger.warning("FAIL  %s — %s", song_dir.name, result.error)
                elif result.skipped:
                    n_skipped += 1
                    logger.debug("SKIP  %s", song_dir.name)
                else:
                    n_passed += 1
                    if result.separation_done:
                        n_demucs_used += 1
                    # Count dropped rows (filtered out in process_song for missing audio features)
                    expected_rows = len([
                        i for i in cfg.instruments
                        if i in result.rows[0].get("instrument", "") or True
                    ]) if result.rows else 0
                    n_rows_ok += len(result.rows)

                    if shard_writer is not None:
                        shard_writer.add(result.rows)
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

                # Periodic plain-text status (visible even without Rich)
                if idx % _PROGRESS_INTERVAL == 0 or idx == total_songs:
                    pct = idx / total_songs * 100
                    elapsed = time.perf_counter() - t_pipeline_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (total_songs - idx) / rate if rate > 0 else 0
                    click.echo(
                        f"[{idx:>6}/{total_songs}  {pct:5.1f}%]  "
                        f"ok={n_passed}  fail={n_failed}  skip={n_skipped}  "
                        f"rows={n_rows_ok}  notes={total_notes:,}  "
                        f"eta={eta/60:.1f}min"
                    )

                # Release any cached GPU memory between songs
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

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
        table.add_row("Songs OK", str(n_passed))
        table.add_row("Songs failed", str(n_failed))
        table.add_row("Songs skipped (no chart)", str(n_skipped))
        table.add_row("Rows saved", str(n_rows_ok))
        table.add_row("Demucs separations", str(n_demucs_used))
        table.add_row("Total notes", f"{total_notes:,}")
        table.add_row("Pipeline time", f"{pipeline_duration:.1f}s")
        if fail_reasons:
            table.add_section()
            for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                table.add_row(f"  {reason}", str(count))
        console.print(table)
    else:
        click.echo(f"Songs   ok={n_passed}  fail={n_failed}  skip={n_skipped}")
        click.echo(f"Rows saved : {n_rows_ok}")
        click.echo(f"Demucs used: {n_demucs_used}")
        click.echo(f"Total notes: {total_notes:,}")
        click.echo(f"Pipeline time: {pipeline_duration:.1f}s")
        if fail_reasons:
            click.echo("Failure breakdown:")
            for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                click.echo(f"  {count:>6}x  {reason}")

    # ── Flush remaining shard buffer and build final dataset ───────────────────
    if shard_writer is not None:
        total_written = shard_writer.close()
        shards = shard_writer.shard_paths()
        click.echo(f"\nFlushed {total_written} rows across {len(shards)} Parquet shard(s).")

        if not shards:
            click.echo("ERROR: No rows were written. Dataset not saved.", err=True)
            sys.exit(1)

        click.echo("Loading dataset from Parquet shards ...")
        try:
            dataset = load_dataset_from_parquet_shards(parquet_shard_dir, split)
        except Exception as e:
            click.echo(f"ERROR loading Parquet shards: {e}", err=True)
            sys.exit(1)

        click.echo(f"Loaded {len(dataset)} rows.")
        _build_and_save_dataset(
            dataset,
            output_dir,
            push_to_hub,
            split,
            allowed_instruments,
            cfg,
            console,
            parquet_shard_dir=parquet_shard_dir,
            parquet_compression=parquet_compression,
            generate_readme=generate_readme,
        )
    elif not output_dir and push_to_hub:
        click.echo(
            "ERROR: --push-to-hub requires --output-dir.",
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
    parquet_shard_dir=None,
    parquet_compression="zstd",
    generate_readme=True,
):
    """Save dataset to Arrow format and push to Hub. Parquet shards are already written."""
    from datasets import Dataset

    dataset = all_rows if isinstance(all_rows, Dataset) else Dataset.from_list(all_rows)

    click.echo(f"Dataset: {dataset}")
    _print_dataset_stats(dataset, instruments)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / split
        dataset.save_to_disk(str(save_path))
        click.echo(f"\nSaved Arrow dataset to {save_path}")

        if parquet_shard_dir:
            shards = sorted(Path(parquet_shard_dir).glob(f"{split}-*.parquet"))
            click.echo(f"Parquet shards: {len(shards)} file(s) in {parquet_shard_dir}")

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
            "extract_logmel": True,
            "separate_stems": cfg.separate_stems,
            "parquet_compression": parquet_compression,
            "parquet_shard_dir": str(parquet_shard_dir) if parquet_shard_dir else None,
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
