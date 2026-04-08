"""HuggingFace datasets.Features schema for the auto-charter dataset.

One row = one instrument track from one song.
A song with guitar + bass generates 2 rows; all three = 3 rows.

Audio conditioning arrays use None for the variable beat dimension:
  mert_embeddings:  [num_beats, 768]    — MERT mean-pooled per beat
  logmel_frames:    [num_beats, 32, 128] — log-mel resampled per beat

These are stored as lists of lists in Arrow; the DataCollator pads them per batch.
"""

from __future__ import annotations

try:
    from datasets import Features, Sequence, Value, Array2D, Array3D
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    Features = Sequence = Value = Array2D = Array3D = None  # type: ignore


def get_features() -> "Features":
    """Return the datasets.Features schema."""
    if not _DATASETS_AVAILABLE:
        raise ImportError("datasets is required: pip install datasets")

    return Features({
        # ── Identity ──────────────────────────────────────────────────────────
        "song_id":              Value("string"),   # hash of artist+title
        "instrument":           Value("string"),   # "guitar" | "bass" | "drums" | ...
        "source_format":        Value("string"),   # "chart" | "midi"

        # ── Token sequence (the model target) ─────────────────────────────────
        "tokens":               Sequence(Value("int32")),  # variable length
        "num_tokens":           Value("int32"),
        "num_beats":            Value("int32"),

        # ── Beat-level audio conditioning ──────────────────────────────────────
        # Stored as lists; shape [num_beats, 768] and [num_beats, 32, 128]
        # Use Sequence(Sequence(...)) since Array2D/3D need fixed first dim
        "mert_embeddings":      Sequence(Sequence(Value("float32"))),   # [B, 768]
        "logmel_frames":        Sequence(Sequence(Sequence(Value("float32")))),  # [B, 32, 128]

        # ── Beat timing (for position encoding and loss weighting) ─────────────
        "beat_times_s":         Sequence(Value("float32")),
        "beat_durations_s":     Sequence(Value("float32")),
        "bpm_at_beat":          Sequence(Value("float32")),
        "time_sig_num_at_beat": Sequence(Value("int32")),
        "time_sig_den_at_beat": Sequence(Value("int32")),

        # ── Song metadata ──────────────────────────────────────────────────────
        "song_name":            Value("string"),
        "artist":               Value("string"),
        "genre":                Value("string"),
        "charter":              Value("string"),
        "year":                 Value("int32"),
        "song_length_ms":       Value("int32"),
        "difficulty":           Value("int32"),    # 0–6; -1 = uncharted
        "resolution":           Value("int32"),    # 192 after normalisation

        # ── Chart feature flags ────────────────────────────────────────────────
        "has_star_power":       Value("bool"),
        "has_solo":             Value("bool"),
        "has_dedicated_stem":   Value("bool"),     # instrument-specific audio stem exists

        # ── Derived statistics (for curriculum learning / filtering) ───────────
        "num_notes":            Value("int32"),
        "notes_per_beat_mean":  Value("float32"),
        "chord_ratio":          Value("float32"),  # fraction of notes that are chords
        "sustain_mean_ticks":   Value("float32"),
        "bpm_mean":             Value("float32"),
        "bpm_std":              Value("float32"),  # 0 = constant BPM song
    })


# Module-level alias for convenience
FEATURES = None  # populated lazily to avoid import errors when datasets is missing


def _get_features_safe():
    """Return Features or None if datasets is not installed."""
    try:
        return get_features()
    except ImportError:
        return None
