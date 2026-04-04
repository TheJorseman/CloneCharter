import numpy as np
import scipy

def interpolate_beat_times(
    beat_times: np.ndarray, steps_per_beat: np.ndarray, n_extend: np.ndarray
):
    """
    This method takes beat_times and then interpolates that using `scipy.interpolate.interp1d` and the output is
    then used to convert raw audio to log-mel-spectrogram.

    Args:
        beat_times (`numpy.ndarray`):
            beat_times is passed into `scipy.interpolate.interp1d` for processing.
        steps_per_beat (`int`):
            used as an parameter to control the interpolation.
        n_extend (`int`):
            used as an parameter to control the interpolation.
    """

    beat_times_function = scipy.interpolate.interp1d(
        np.arange(beat_times.size),
        beat_times,
        bounds_error=False,
        fill_value="extrapolate",
    )

    ext_beats = beat_times_function(
        np.linspace(0, beat_times.size + n_extend - 1, beat_times.size * steps_per_beat + n_extend)
    )

    return ext_beats


beat_times = np.load("sera_porque_te_amo_beat_times.npy")
print(beat_times)
print(beat_times.shape)
beatsteps = interpolate_beat_times(beat_times=beat_times, steps_per_beat=2, n_extend=1)
print(beatsteps)
print(beatsteps.shape)