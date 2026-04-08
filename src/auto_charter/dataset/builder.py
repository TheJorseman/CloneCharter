"""SongProcessor: converts a Clone Hero song folder into dataset rows.

One row is produced per charted instrument per song.

Usage:
    processor = SongProcessor(extract_audio=True)
    rows = processor.process(song_dir)
    # rows is a list of dicts matching the FEATURES schema
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from auto_charter.parsers.chart_parser import parse_chart, ChartData, NoteEvent, SpecialEvent
from auto_charter.parsers.midi_parser import parse_midi
from auto_charter.parsers.ini_parser import parse_ini, SongMetadata
from auto_charter.tokenizer.encoder import encode_track
from auto_charter.audio.stem_loader import resolve_stem_path, has_dedicated_stem

logger = logging.getLogger(__name__)

# Instruments to extract from charts (Expert difficulty only)
_EXPERT_INSTRUMENTS = ["guitar", "bass", "drums", "rhythm"]


@dataclass
class SongProcessor:
    """Process a song folder into dataset rows.

    Args:
        extract_audio: If True, extract MERT and log-mel features.
            Requires torch, transformers, librosa to be installed.
        logmel_n_mels: Number of mel bins (default 128).
        logmel_target_frames: Frames per beat for log-mel (default 32).
        include_beat_boundaries: Emit BEAT_BOUNDARY tokens in sequences.
        mert_model_name: HuggingFace MERT model ID.
        device: Torch device for MERT inference.
    """

    extract_audio: bool = False
    logmel_n_mels: int = 128
    logmel_target_frames: int = 32
    include_beat_boundaries: bool = True
    mert_model_name: str = "m-a-p/MERT-v1-95M"
    device: str = "cpu"

    _mert: Any = field(default=None, init=False, repr=False)
    _logmel: Any = field(default=None, init=False, repr=False)

    def _get_mert(self):
        if self._mert is None:
            from auto_charter.audio.mert_extractor import MERTExtractor
            self._mert = MERTExtractor(model_name=self.mert_model_name, device=self.device)
        return self._mert

    def _get_logmel(self):
        if self._logmel is None:
            from auto_charter.audio.logmel import LogMelExtractor
            self._logmel = LogMelExtractor(
                n_mels=self.logmel_n_mels,
                target_frames=self.logmel_target_frames,
            )
        return self._logmel

    def process(self, song_dir: str | Path) -> list[dict]:
        """Process a single song directory. Returns a list of dataset rows."""
        song_dir = Path(song_dir)
        rows: list[dict] = []

        # ── Parse chart or MIDI ────────────────────────────────────────────────
        chart_path = song_dir / "notes.chart"
        midi_path = song_dir / "notes.mid"

        if chart_path.exists():
            chart = parse_chart(chart_path)
            source_format = "chart"
        elif midi_path.exists():
            try:
                chart = parse_midi(midi_path)
                source_format = "midi"
            except Exception as e:
                logger.warning("Failed to parse MIDI %s: %s", midi_path, e)
                return []
        else:
            logger.warning("No chart or MIDI found in %s", song_dir)
            return []

        # ── Parse metadata ─────────────────────────────────────────────────────
        ini_path = song_dir / "song.ini"
        meta: SongMetadata | None = None
        if ini_path.exists():
            try:
                meta = parse_ini(ini_path)
            except Exception as e:
                logger.warning("Failed to parse INI %s: %s", ini_path, e)

        song_name = meta.name if meta else chart.song_meta.get("name", "")
        artist = meta.artist if meta else chart.song_meta.get("artist", "")
        song_id = hashlib.md5(f"{artist}|{song_name}".encode()).hexdigest()[:16]

        # ── Build beat grid ────────────────────────────────────────────────────
        end_tick = chart.end_tick
        beat_ticks, beat_times_s, beat_durations_s, bpm_at_beat, time_sigs_at_beat = \
            chart.bpm_map.beat_times(end_tick)

        num_beats = len(beat_ticks)
        time_sig_num = [ts[0] for ts in time_sigs_at_beat]
        time_sig_den = [ts[1] for ts in time_sigs_at_beat]

        # ── Per-instrument processing ──────────────────────────────────────────
        for instrument in _EXPERT_INSTRUMENTS:
            if instrument not in chart.tracks or not chart.tracks[instrument]:
                continue

            notes = chart.tracks[instrument]
            specials = chart.specials.get(instrument, [])

            # Encode tokens
            try:
                tokens = encode_track(
                    chart=chart,
                    instrument=instrument,
                    include_beat_boundaries=self.include_beat_boundaries,
                )
            except Exception as e:
                logger.warning("Encoding failed for %s/%s: %s", song_dir.name, instrument, e)
                continue

            # Track statistics
            stats = _compute_stats(notes, bpm_at_beat)

            # Difficulty from metadata
            difficulty = -1
            if meta:
                diff_map = {
                    "guitar": meta.diff_guitar,
                    "bass": meta.diff_bass,
                    "drums": meta.diff_drums,
                    "rhythm": meta.diff_rhythm,
                }
                difficulty = diff_map.get(instrument, -1)

            # Resolve audio stem
            stem_path = resolve_stem_path(song_dir, instrument)
            dedicated_stem = has_dedicated_stem(song_dir, instrument)

            # Audio features (optional)
            mert_embeddings = []
            logmel_frames = []

            if self.extract_audio and stem_path is not None and num_beats > 0:
                try:
                    mert_arr = self._get_mert().extract_per_beat(
                        stem_path, beat_times_s, beat_durations_s
                    )
                    mert_embeddings = mert_arr.tolist()
                except Exception as e:
                    logger.warning("MERT extraction failed for %s: %s", stem_path, e)
                    mert_embeddings = []

                try:
                    logmel_arr = self._get_logmel().extract_per_beat(
                        stem_path, beat_times_s, beat_durations_s
                    )
                    logmel_frames = logmel_arr.tolist()
                except Exception as e:
                    logger.warning("Log-mel extraction failed for %s: %s", stem_path, e)
                    logmel_frames = []

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
                "time_sig_num_at_beat": time_sig_num,
                "time_sig_den_at_beat": time_sig_den,

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
                "has_dedicated_stem": dedicated_stem,

                # Stats
                "num_notes": stats["num_notes"],
                "notes_per_beat_mean": stats["notes_per_beat_mean"],
                "chord_ratio": stats["chord_ratio"],
                "sustain_mean_ticks": stats["sustain_mean_ticks"],
                "bpm_mean": stats["bpm_mean"],
                "bpm_std": stats["bpm_std"],
            }
            rows.append(row)

        return rows


def _compute_stats(notes: list[NoteEvent], bpm_at_beat: list[float]) -> dict:
    """Compute per-track statistics for curriculum learning."""
    if not notes:
        return {
            "num_notes": 0,
            "notes_per_beat_mean": 0.0,
            "chord_ratio": 0.0,
            "sustain_mean_ticks": 0.0,
            "bpm_mean": float(np.mean(bpm_at_beat)) if bpm_at_beat else 0.0,
            "bpm_std": float(np.std(bpm_at_beat)) if bpm_at_beat else 0.0,
        }

    num_notes = len(notes)
    chord_count = sum(1 for n in notes if len(n.pitches) > 1)
    sustain_mean = float(np.mean([n.sustain for n in notes]))
    bpm_arr = np.array(bpm_at_beat, dtype=np.float32)

    return {
        "num_notes": num_notes,
        "notes_per_beat_mean": num_notes / max(1, len(bpm_at_beat)),
        "chord_ratio": chord_count / num_notes,
        "sustain_mean_ticks": sustain_mean,
        "bpm_mean": float(bpm_arr.mean()) if len(bpm_arr) > 0 else 0.0,
        "bpm_std": float(bpm_arr.std()) if len(bpm_arr) > 0 else 0.0,
    }
