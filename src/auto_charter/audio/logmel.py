"""Beat-synchronous log-mel spectrogram extraction.

Extracts a log-mel spectrogram from an audio file and slices it into
per-beat windows resampled to a fixed number of frames.

Output shape: [num_beats, target_frames, n_mels]

Dependencies: librosa, soundfile, numpy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

from .beat_aligner import slice_beats

# Defaults
SR = 22050          # sample rate
N_MELS = 128        # mel filterbank size
N_FFT = 2048        # FFT window
HOP_LENGTH = 220    # ~10ms hop at 22050Hz → ~100 frames/sec
TARGET_FRAMES = 32  # frames per beat in output


class LogMelExtractor:
    """Extract beat-synchronous log-mel spectrograms.

    Args:
        sr: Sample rate for audio loading (default 22050).
        n_mels: Number of mel filterbank channels (default 128).
        n_fft: FFT window size (default 2048).
        hop_length: Hop size in samples (default 220, ~10ms at 22050Hz).
        target_frames: Fixed number of frames per beat in output (default 32).
    """

    def __init__(
        self,
        sr: int = SR,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        target_frames: int = TARGET_FRAMES,
    ) -> None:
        if not _LIBROSA_AVAILABLE:
            raise ImportError("librosa is required: pip install librosa")
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.feature_rate_hz = sr / hop_length  # ~100.2 Hz

    def extract(self, audio_path: str | Path) -> np.ndarray:
        """Load audio and compute full log-mel spectrogram.

        Returns:
            Float32 array [T_frames, n_mels].
        """
        from auto_charter.audio.audio_io import load_audio
        y = load_audio(audio_path, sr=self.sr, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # [n_mels, T]
        return log_mel.T.astype(np.float32)              # [T, n_mels]

    def extract_per_beat(
        self,
        audio_path: str | Path,
        beat_times_s: list[float],
        beat_durations_s: list[float],
    ) -> np.ndarray:
        """Extract log-mel and slice into per-beat windows.

        Returns:
            Float32 array [num_beats, target_frames, n_mels].
        """
        log_mel = self.extract(audio_path)  # [T, n_mels]
        return slice_beats(
            features=log_mel,
            beat_times_s=beat_times_s,
            beat_durations_s=beat_durations_s,
            feature_rate_hz=self.feature_rate_hz,
            target_frames=self.target_frames,
        )
