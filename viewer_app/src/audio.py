from typing import Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt
import sys

try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

from PySide6 import QtCore

def bandpass_filter(x: np.ndarray, fs: float, fc=(50.0, 2000.0), order=2, zero_phase=True, axis=-1) -> np.ndarray:
    """
    Applies a Butterworth band-pass filter to the input signal.

    This function filters the input array `x` using a digital Butterworth band-pass filter with specified cutoff frequencies.
    By default, zero-phase filtering is applied using `sosfiltfilt` to avoid phase distortion. The filter is implemented
    using second-order sections (SOS) for numerical stability.

    Parameters
    ----------
    x : np.ndarray
        Input signal array to be filtered.
    fs : float
        Sampling frequency of the input signal in Hz.
    fc : tuple of float, optional
        Tuple specifying the lower and upper cutoff frequencies (in Hz) for the band-pass filter.
        Default is (50.0, 2000.0).
    order : int, optional
        The order of the Butterworth filter. Default is 2.
    zero_phase : bool, optional
        If True (default), applies zero-phase filtering using `sosfiltfilt`. If False, applies causal filtering using `sosfilt`.
    axis : int, optional
        The axis along which to filter the data. Default is -1 (the last axis).

    Returns
    -------
    np.ndarray
        The filtered signal array, with the same shape as the input.
    Notes
    -----
    - The cutoff frequencies are automatically clamped to ensure they do not exceed the Nyquist frequency.
    - Zero-phase filtering is recommended for most applications to prevent phase distortion.

    Butterworth band-pass filter. Zero-phase (sosfiltfilt) standaard.
    """
    x = np.asarray(x)
    # Get cutoff frequencies from arguments
    low, high = float(fc[0]), float(fc[1])
    nyq = fs / 2.0  # Nyquist frequency (half the sampling rate)

    # Safety clamp: ensure 0 < low < high < Nyquist
    high = min(high, nyq - 1e-6)  # Don't allow high cutoff above Nyquist
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Invalid cutoffs: 0 < {low=} < {high=} < {nyq=}")

    # Normalize cutoff frequencies for filter design (as fraction of Nyquist)
    wn = (low / nyq, high / nyq)

    # Design Butterworth band-pass filter (second-order sections for stability)
    sos = butter(int(order), wn, btype='band', output='sos')

    # Apply filter: zero-phase (forward-backward) or causal (one-way)
    if zero_phase:
        # Zero-phase filtering: avoids phase distortion, good for analysis
        return sosfiltfilt(sos, x, axis=axis)
    else:
        # Causal filtering: for real-time use, introduces phase shift
        return sosfilt(sos, x, axis=axis)  # type: ignore

def compute_stft_db(
    y: np.ndarray,
    sr: int,
    nperseg: int = 1024,
    hop: int = 256,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Short-Time Fourier Transform (STFT) of a signal and return its magnitude in dB.

    Args:
        y (np.ndarray): Input audio signal.
        sr (int): Sampling rate of the audio signal.
        nperseg (int, optional): Length of each segment for STFT. Defaults to 1024.
        hop (int, optional): Hop length (step size) between segments. Defaults to 256.
        window (str, optional): Window function to apply to each segment. Defaults to "hann".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, times, and STFT magnitude in dB.
    """

    # Ensure the signal is a float32 NumPy array
    y = np.asarray(y, dtype=np.float32)
    N = int(len(y))
    _dbg(f"[STFT] N={N} sr={sr} nperseg={nperseg} hop={hop}")

    # If there is no data, return empty arrays
    if N == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros((0, 0), np.float32))

    # Create the window function (Hann or flat window)
    win = np.hanning(nperseg).astype(np.float32) if window == "hann" else np.ones(nperseg, np.float32)

    # Calculate how many frames fit in the signal
    n_frames = 1 + int(np.ceil(max(0, N - nperseg) / float(hop)))

    # Total length after adding padding if needed
    total_len = (n_frames - 1) * hop + nperseg
    if total_len > N:
        # Pad signal with zeros to fit complete frames
        y = np.pad(y, (0, total_len - N), mode="constant")

    # Create overlapping frames using stride tricks (fast, no data copy)
    shape = (n_frames, nperseg)
    strides = (y.strides[0] * hop, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    _arr_stats("[STFT] frames (pre-window)", frames[: min(3, len(frames))])

    # Apply window function to each frame
    frames = frames * win[None, :]

    # Compute FFT of each frame (real-valued FFT)
    spec = np.fft.rfft(frames, axis=1)

    # Compute magnitude of the complex spectrum
    mag = np.abs(spec).astype(np.float32)

    # Normalize the magnitude by its maximum value
    mmax = float(mag.max()) if mag.size else 1.0
    _dbg(f"[STFT] mag.max={mmax:.6e}")
    mag_norm = mag / (mmax + 1e-12)
    mag_norm = np.minimum(mag_norm, 1.0)  # Clamp values so they don’t exceed 0 dB

    # Convert normalized magnitude to decibels
    S_db = (20.0 * np.log10(np.maximum(mag_norm, 1e-12))).astype(np.float32)

    # Compute frequency and time vectors
    f = np.fft.rfftfreq(nperseg, d=1.0 / float(sr)).astype(np.float32)
    t = (np.arange(n_frames, dtype=np.float32) * (hop / float(sr)))

    # Transpose for plotting (frequency × time)
    S_db_FT = S_db.T
    _arr_stats("[STFT] S_db (F,T)", S_db_FT)

    # Return frequency bins, time bins, and the decibel-scaled spectrogram
    return f, t, S_db_FT

class Player(QtCore.QObject):
    """
    Simple helper class for audio playback over a specific time window.

    This class uses the sounddevice library to play a segment of an audio
    signal (between start time `t0` and end time `t1`).

    Attributes:
        started (QtCore.Signal): Signal emitted when playback starts, with (t0, t1) as arguments.
        stopped (QtCore.Signal): Signal emitted when playback stops.
        _stream (sd.OutputStream | None): The active audio stream object, or None if not playing.
        playing (bool): True if audio is currently being played.
    """

    # Define Qt signals
    started = QtCore.Signal(float, float)  # emits (t0, t1)
    stopped = QtCore.Signal()

    def __init__(self):
        """Initialize the player with default state and no active audio stream."""
        super().__init__()
        self._stream = None  # Placeholder for the sounddevice stream
        self.playing = False  # Playback state flag

    def play(self, y: np.ndarray, sr: int, t0: float, t1: float):
        """
        Play a section of the signal y between times t0 and t1.

        Args:
            y (np.ndarray): The input audio signal (1D NumPy array).
            sr (int): The sampling rate of the signal in Hz.
            t0 (float): Start time (in seconds) of the segment to play.
            t1 (float): End time (in seconds) of the segment to play.

        Notes:
            - If sounddevice (sd) is not available, the function does nothing.
            - Playback is done once (no looping).
        """
        if not HAVE_SD:
            return

        # Stop any currently playing audio before starting new playback
        self.stop()

        # Convert time range to sample indices
        start = int(max(0, t0) * sr)
        end = int(min(len(y), t1 * sr))

        # Slice the audio data for the selected time window
        data = y[start:end].astype(np.float32)
        pos = 0  # Keep track of current playback position

        # Define callback function for stream
        def _cb(outdata, frames, timeinfo, status):
            """
            Fill the output buffer with the next chunk of audio data.

            This callback is called repeatedly by the sounddevice stream.
            When all data has been played, it raises sd.CallbackStop to end playback.
            """
            nonlocal pos
            n = min(frames, len(data) - pos)  # Number of frames available to play
            if n > 0:
                outdata[:n, 0] = data[pos:pos + n]  # Copy audio data to output buffer
            if frames > n:
                outdata[n:, 0] = 0  # Fill remainder with silence
            pos += n
            if pos >= len(data):
                raise sd.CallbackStop  # Stop when all data is played

        # Create and start an audio output stream
        self._stream = sd.OutputStream(
            channels=1,
            samplerate=sr,
            dtype="float32",
            callback=_cb
        )
        self._stream.start()
        self.playing = True  # Update state
        self.started.emit(t0, t1)  # Notify others that playback has started

    def play_region(self, y: np.ndarray, sr: int, t_start: float, t_end: float):
        """
        Play a region from the provided audio buffer `y` sampled at `sr` between t_start and t_end.
        This avoids relying on an undefined `_backend` attribute and reuses the existing `play` method.
        """
        t_start = max(0.0, float(t_start))
        t_end = max(t_start, float(t_end))
        # Use the existing play method which slices and plays the provided buffer.
        self.play(y, sr, t_start, t_end)

    def stop(self):
        """
        Stop playback and close the audio stream safely.

        This method stops the current sounddevice stream if it exists,
        closes it, and emits the `stopped` signal when playback ends.
        """
        if not HAVE_SD:
            return

        # Safely stop and close the audio stream
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None  # Remove reference to stream

        # If playback was active, reset state and emit signal
        if self.playing:
            self.playing = False
            self.stopped.emit()

#-------#
# Debug helpers
#-------#
DEBUG_STFT = False                 # True = print debug info during STFT
DYNAMIC_SPECTRO_LEVELS = False    # True = auto levels per spectrogram; False = fixed -100..0 dB
GRAYSCALE_DEBUG = True            # true = spectrogram in grayscale (for debugging color issues) --> Color doesn't work
def _dbg(msg: str):
    """
    Debugging helper function.
    Args:
        msg (str): Message to print if debugging is enabled.
    """
    if DEBUG_STFT:
        print(msg); sys.stdout.flush()

def _arr_stats(name: str, a: np.ndarray):
    """
    Prints statistics about the array.
    Args:
        name (str): Name of the array (for debugging).
        a (np.ndarray): The array to analyze.
    """
    # Only print stats if debugging is enabled
    if not DEBUG_STFT: return
    a = np.asarray(a)
    total = a.size
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    if n_finite == 0:
        # No finite values in the array
        _dbg(f"{name}: shape={a.shape} dtype={a.dtype} NO FINITE VALUES"); return
    avals = a[finite]
    p = lambda q: float(np.percentile(avals, q))
    # Print basic statistics: shape, dtype, number of finite values, min, max, mean, percentiles
    _dbg(f"{name}: shape={a.shape} dtype={a.dtype} "
         f"finite={n_finite}/{total} "
         f"min={avals.min():.3f} max={avals.max():.3f} mean={avals.mean():.3f} "
         f"p1={p(1):.3f} p50={p(50):.3f} p99={p(99):.3f}")
