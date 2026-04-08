"""Tests for audio feature extraction shapes.

These tests run without MERT (no torch/transformers needed for log-mel tests)
and verify the output shapes and dtypes are correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
EL_RESCATE = REPO_ROOT / "Grupo Marca Registrada - El Rescate"
EL_PRECIO = REPO_ROOT / "El Precio de la Soledad"


def requires_fixture(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"Fixture not found: {path}")


def requires_librosa():
    try:
        import librosa  # noqa: F401
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="librosa not installed")


# ─── Beat aligner unit tests (no audio needed) ────────────────────────────────

def test_slice_beats_shape():
    from auto_charter.audio.beat_aligner import slice_beats

    # Simulate 10 seconds of 100Hz features, 128-dim
    T, F = 1000, 128
    features = np.random.rand(T, F).astype(np.float32)

    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    beat_durations = [0.5] * 5
    result = slice_beats(features, beat_times, beat_durations, feature_rate_hz=100.0,
                         target_frames=32)

    assert result.shape == (5, 32, 128)
    assert result.dtype == np.float32


def test_slice_beats_variable_bpm():
    """Beat durations vary (simulating variable BPM) — output shape must be fixed."""
    from auto_charter.audio.beat_aligner import slice_beats

    T, F = 2000, 64
    features = np.random.rand(T, F).astype(np.float32)

    # Variable beat durations: 0.3s (200 BPM) to 0.75s (80 BPM)
    beat_times = [0.0, 0.3, 0.65, 1.1, 1.5, 2.0, 2.5, 3.25]
    beat_durations = [0.3, 0.35, 0.45, 0.4, 0.5, 0.5, 0.75, 0.5]
    result = slice_beats(features, beat_times, beat_durations, feature_rate_hz=100.0,
                         target_frames=16)

    assert result.shape == (8, 16, 64)


def test_mean_pool_beats_shape():
    from auto_charter.audio.beat_aligner import mean_pool_beats

    T, F = 500, 768
    features = np.random.rand(T, F).astype(np.float32)
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    beat_durations = [0.5] * 5

    result = mean_pool_beats(features, beat_times, beat_durations, feature_rate_hz=100.0)
    assert result.shape == (5, 768)
    assert result.dtype == np.float32


def test_mean_pool_beats_beyond_audio():
    """Beat times extending beyond audio length should not crash."""
    from auto_charter.audio.beat_aligner import mean_pool_beats

    T, F = 100, 768  # only 1 second of audio at 100Hz
    features = np.zeros((T, F), dtype=np.float32)
    features[:50] = 1.0  # first 0.5s has value 1.0

    beat_times = [0.0, 0.5, 1.0, 1.5]  # beats extend beyond audio
    beat_durations = [0.5] * 4

    result = mean_pool_beats(features, beat_times, beat_durations, feature_rate_hz=100.0)
    assert result.shape == (4, 768)
    # First beat: frames 0..50 at 100Hz, frames 0..49 = 1.0, frame 50 = 0.0 → mean ≈ 0.98
    assert result[0].mean() > 0.95
    # Last beat: beyond audio → zeros
    assert result[3].sum() == 0.0


# ─── LogMelExtractor tests (requires librosa + audio files) ───────────────────

@requires_fixture(EL_RESCATE)
@pytest.mark.skipif(
    not (REPO_ROOT / "Grupo Marca Registrada - El Rescate" / "guitar.ogg").exists(),
    reason="guitar.ogg not found"
)
def test_logmel_full_shape():
    try:
        from auto_charter.audio.logmel import LogMelExtractor
    except ImportError:
        pytest.skip("librosa not installed")

    extractor = LogMelExtractor(n_mels=128)
    log_mel = extractor.extract(EL_RESCATE / "guitar.ogg")

    assert log_mel.ndim == 2
    assert log_mel.shape[1] == 128
    assert log_mel.dtype == np.float32
    assert log_mel.shape[0] > 100  # at least 1 second of audio


@requires_fixture(EL_RESCATE)
@pytest.mark.skipif(
    not (REPO_ROOT / "Grupo Marca Registrada - El Rescate" / "guitar.ogg").exists(),
    reason="guitar.ogg not found"
)
def test_logmel_per_beat_shape():
    try:
        from auto_charter.audio.logmel import LogMelExtractor
        from auto_charter.parsers.chart_parser import parse_chart
    except ImportError:
        pytest.skip("librosa not installed")

    chart = parse_chart(EL_RESCATE / "notes.chart")
    beat_ticks, beat_times_s, beat_durations_s, _, _ = chart.bpm_map.beat_times(chart.end_tick)

    extractor = LogMelExtractor(n_mels=128, target_frames=32)
    result = extractor.extract_per_beat(
        EL_RESCATE / "guitar.ogg",
        beat_times_s=beat_times_s,
        beat_durations_s=beat_durations_s,
    )

    num_beats = len(beat_times_s)
    assert result.shape == (num_beats, 32, 128), f"Expected ({num_beats}, 32, 128), got {result.shape}"
    assert result.dtype == np.float32


# ─── Stem loader tests ─────────────────────────────────────────────────────────

@requires_fixture(EL_RESCATE)
def test_stem_loader_dedicated():
    from auto_charter.audio.stem_loader import resolve_stem_path, has_dedicated_stem

    guitar_stem = resolve_stem_path(EL_RESCATE, "guitar")
    assert guitar_stem is not None
    assert guitar_stem.name == "guitar.ogg"
    assert has_dedicated_stem(EL_RESCATE, "guitar") is True

    bass_stem = resolve_stem_path(EL_RESCATE, "bass")
    assert bass_stem is not None
    assert bass_stem.name == "bass.ogg"


@requires_fixture(EL_PRECIO)
def test_stem_loader_fallback():
    from auto_charter.audio.stem_loader import resolve_stem_path, has_dedicated_stem

    # El Precio has no guitar.ogg, only song.ogg
    guitar_stem = resolve_stem_path(EL_PRECIO, "guitar")
    assert guitar_stem is not None
    assert guitar_stem.name == "song.ogg"  # fallback to mix
    assert has_dedicated_stem(EL_PRECIO, "guitar") is False


def test_stem_loader_missing():
    from auto_charter.audio.stem_loader import resolve_stem_path

    result = resolve_stem_path("/nonexistent/path", "guitar")
    assert result is None
