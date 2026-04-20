"""Robust audio loading for librosa.

soundfile (used by librosa by default) does not support .opus or some .mp3
files. This helper tries soundfile first, then falls back to an ffmpeg
subprocess that decodes to a temporary WAV — eliminating the slow audioread
path and its FutureWarning.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, sr: int, mono: bool = True) -> np.ndarray:
    """Load audio as a float32 numpy array at the requested sample rate.

    Tries librosa's default backend (soundfile) first. If that raises an
    exception (common for .opus and some .mp3 files), converts to WAV via
    ffmpeg and retries — avoiding the deprecated audioread fallback.

    Args:
        path: Path to the audio file.
        sr:   Target sample rate in Hz.
        mono: If True, mix down to mono.

    Returns:
        Float32 numpy array of shape [samples] (mono) or [channels, samples].
    """
    import librosa
    import soundfile as sf

    path = Path(path)

    # Fast path: soundfile handles .ogg, .wav, .flac natively
    try:
        sf.info(str(path))  # quick format check — raises on unsupported formats
        y, _ = librosa.load(str(path), sr=sr, mono=mono)
        return y
    except Exception:
        pass

    # Slow path: decode via ffmpeg to a temporary WAV
    logger.debug("soundfile cannot read '%s' — using ffmpeg fallback.", path.name)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        channels = "1" if mono else "2"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(path),
            "-ar", str(sr),
            "-ac", channels,
            "-f", "wav", tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}"
            )
        y, _ = librosa.load(tmp_path, sr=sr, mono=mono)
        return y
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
