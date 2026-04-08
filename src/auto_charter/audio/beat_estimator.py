"""BeatEstimator — estimate beat grid from audio for inference-time use.

During dataset building, beat timing comes from the .chart SyncTrack (ground truth).
During inference (Gradio demo), we estimate beats from the audio using librosa.

This bridges the gap so that the same encoder inputs can be produced at inference
time without a .chart file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class BeatEstimator:
    """Estimate beat grid from an audio file using librosa beat tracking.

    Returns a dict with the same field names as the HuggingFace dataset schema,
    ready to be passed to the AudioEncoder.

    Algorithm:
        1. Load audio at 22050 Hz mono
        2. librosa.beat.beat_track(y, sr) → tempo, beat_frames
        3. Convert beat_frames to beat_times_s
        4. Compute beat_durations_s as differences (last duration = 60/tempo)
        5. Constant BPM: bpm_at_beat = [tempo] × N
        6. Assume 4/4 time signature throughout

    Notes:
        - This produces a constant-BPM beat grid. For songs with variable BPM
          this is an approximation, but sufficient for inference.
        - Time signature is always assumed 4/4.
        - The domain mismatch between ground-truth chart BPM and estimated BPM
          is partially mitigated by the soft beat-distance bias in cross-attention.
    """

    @staticmethod
    def estimate(
        audio_path: Path | str,
        sr: int = 22050,
    ) -> dict[str, Any]:
        """Estimate beat grid from audio file.

        Args:
            audio_path: Path to audio file (.ogg, .mp3, .wav).
            sr:         Sample rate for loading (default 22050 Hz).

        Returns:
            dict with keys:
                beat_times_s         list[float]   — beat onset times in seconds
                beat_durations_s     list[float]   — duration of each beat
                bpm_at_beat          list[float]   — constant BPM for each beat
                time_sig_num_at_beat list[int]     — always 4
                time_sig_den_at_beat list[int]     — always 4
                num_beats            int
                bpm_mean             float
        """
        import librosa

        y, actual_sr = librosa.load(str(audio_path), sr=sr, mono=True)

        # Beat tracking
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=actual_sr, trim=True)
        # librosa >= 0.10 may return array; ensure scalar
        tempo = float(np.squeeze(tempo_arr)) if hasattr(tempo_arr, "__len__") else float(tempo_arr)

        if len(beat_frames) == 0:
            # Fallback: generate a regular 4/4 grid at the detected tempo
            duration_s = len(y) / actual_sr
            beat_duration = 60.0 / max(tempo, 1.0)
            beat_times = np.arange(0.0, duration_s, beat_duration)
        else:
            beat_times = librosa.frames_to_time(beat_frames, sr=actual_sr)

        beat_times = beat_times.tolist()
        N = len(beat_times)

        if N == 0:
            # Degenerate: return a single beat
            return {
                "beat_times_s": [0.0],
                "beat_durations_s": [60.0 / max(tempo, 1.0)],
                "bpm_at_beat": [tempo],
                "time_sig_num_at_beat": [4],
                "time_sig_den_at_beat": [4],
                "num_beats": 1,
                "bpm_mean": tempo,
            }

        # Beat durations: difference between consecutive beats
        beat_times_arr = np.array(beat_times)
        last_dur = 60.0 / max(tempo, 1.0)
        diffs = np.diff(beat_times_arr, append=beat_times_arr[-1] + last_dur)
        beat_durations = diffs.tolist()

        return {
            "beat_times_s": beat_times,
            "beat_durations_s": beat_durations,
            "bpm_at_beat": [tempo] * N,
            "time_sig_num_at_beat": [4] * N,
            "time_sig_den_at_beat": [4] * N,
            "num_beats": N,
            "bpm_mean": tempo,
        }
