"""Beat-aligned audio frame extraction utilities.

Provides functions to slice a pre-computed feature array (log-mel or MERT frames)
into per-beat segments and resample each segment to a fixed number of frames.

This handles variable BPM: at 80 BPM a beat is 750ms (~75 MERT frames), at 200 BPM
it's 300ms (~30 frames). Resampling to target_frames=32 normalises everything.
"""

from __future__ import annotations

import numpy as np


def slice_beats(
    features: np.ndarray,
    beat_times_s: list[float],
    beat_durations_s: list[float],
    feature_rate_hz: float,
    target_frames: int = 32,
) -> np.ndarray:
    """Slice a feature array into per-beat windows and resample to fixed length.

    Args:
        features: Float array of shape [T, F] where T=time frames, F=feature dim.
        beat_times_s: Start time (seconds) of each beat.
        beat_durations_s: Duration (seconds) of each beat.
        feature_rate_hz: Feature frame rate in Hz (e.g., 75 for MERT, 100 for log-mel).
        target_frames: Number of frames per beat in the output.

    Returns:
        Float32 array of shape [num_beats, target_frames, F].
    """
    T, F = features.shape
    num_beats = len(beat_times_s)
    out = np.zeros((num_beats, target_frames, F), dtype=np.float32)

    for b in range(num_beats):
        t_start = beat_times_s[b]
        t_end = t_start + beat_durations_s[b]

        f_start = max(0, int(t_start * feature_rate_hz))
        f_end = min(T, int(t_end * feature_rate_hz) + 1)

        if f_start >= T:
            continue  # beyond audio — leave zeros

        beat_frames = features[f_start:f_end]  # [n_f, F]
        n_f = len(beat_frames)

        if n_f == 0:
            continue
        elif n_f == target_frames:
            out[b] = beat_frames
        else:
            # Linear interpolation along time axis for each feature dimension
            t_in = np.linspace(0.0, 1.0, n_f)
            t_out = np.linspace(0.0, 1.0, target_frames)
            for feat_dim in range(F):
                out[b, :, feat_dim] = np.interp(t_out, t_in, beat_frames[:, feat_dim])

    return out


def mean_pool_beats(
    features: np.ndarray,
    beat_times_s: list[float],
    beat_durations_s: list[float],
    feature_rate_hz: float,
) -> np.ndarray:
    """Mean-pool a feature array over each beat window.

    Args:
        features: Float array [T, F].
        beat_times_s: Beat start times in seconds.
        beat_durations_s: Beat durations in seconds.
        feature_rate_hz: Feature frame rate in Hz.

    Returns:
        Float32 array [num_beats, F].
    """
    T, F = features.shape
    num_beats = len(beat_times_s)
    out = np.zeros((num_beats, F), dtype=np.float32)

    for b in range(num_beats):
        t_start = beat_times_s[b]
        t_end = t_start + beat_durations_s[b]
        f_start = max(0, int(t_start * feature_rate_hz))
        f_end = min(T, int(t_end * feature_rate_hz) + 1)
        if f_start < T and f_end > f_start:
            out[b] = features[f_start:f_end].mean(axis=0)

    return out
