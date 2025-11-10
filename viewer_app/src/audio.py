from typing import Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt
import sys

# Try to import the sounddevice library for playback
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

from PySide6 import QtCore

def bandpass_filter(x: np.ndarray, fs: float, fc=(50.0, 2000.0), order=2, zero_phase=True, axis=-1) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to an input signal.

    Args:
        x (np.ndarray): Input signal to filter.
        fs (float): Sampling rate in Hz.
        fc (tuple of float): Low and high cutoff frequencies in Hz. Default is (50.0, 2000.0).
        order (int): Filter order. Default is 2.
        zero_phase (bool): If True, use zero-phase filtering (no phase shift). Default is True.
        axis (int): Axis to filter along. Default is -1.

    Returns:
        np.ndarray: The filtered signal, same shape as input.

    Notes:
        - Zero-phase filtering avoids phase distortion but cannot be used in real-time.
        - Frequencies are automatically limited to stay below the Nyquist frequency.
    """
    x = np.asarray(x)

    # Unpack cutoff frequencies
    low, high = float(fc[0]), float(fc[1])
    nyq = fs / 2.0  # Nyquist frequency (half the sample rate)

    # Clamp upper cutoff to stay below Nyquist
    high = min(high, nyq - 1e-6)
    if not (0.0 < low < high < nyq):
        raise ValueError(f"Invalid cutoffs: 0 < {low=} < {high=} < {nyq=}")

    # Normalize to Nyquist
    wn = (low / nyq, high / nyq)

    # Create Butterworth band-pass filter
    sos = butter(int(order), wn, btype='band', output='sos')

    # Apply the filter
    if zero_phase:
        # Use forward-backward filtering (no phase shift)
        return sosfiltfilt(sos, x, axis=axis)
    else:
        # Use one-way causal filtering (adds delay)
        return sosfilt(sos, x, axis=axis) # type: ignore

def compute_stft_db(
    y: np.ndarray,
    sr: int,
    nperseg: int = 1024,
    hop: int = 256,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) and return magnitude in decibels.

    Args:
        y (np.ndarray): Input audio signal.
        sr (int): Sampling rate of the signal in Hz.
        nperseg (int): Number of samples per segment. Default is 1024.
        hop (int): Step size between segments. Default is 256.
        window (str): Window type ("hann" or "flat"). Default is "hann".

    Returns:
        (freqs, times, S_db): Frequency bins, time bins, and magnitude in dB.
    """
    # Convert to float32
    y = np.asarray(y, dtype=np.float32)
    N = int(len(y))
    _dbg(f"[STFT] N={N} sr={sr} nperseg={nperseg} hop={hop}")

    # Handle empty input
    if N == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros((0, 0), np.float32))

    # Create window
    win = np.hanning(nperseg).astype(np.float32) if window == "hann" else np.ones(nperseg, np.float32)

    # Calculate number of frames
    n_frames = 1 + int(np.ceil(max(0, N - nperseg) / float(hop)))

    # Pad signal if needed
    total_len = (n_frames - 1) * hop + nperseg
    if total_len > N:
        y = np.pad(y, (0, total_len - N), mode="constant")

    # Create overlapping frames
    shape = (n_frames, nperseg)
    strides = (y.strides[0] * hop, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    _arr_stats("[STFT] frames (pre-window)", frames[: min(3, len(frames))])

    # Apply window
    frames = frames * win[None, :]

    # FFT per frame
    spec = np.fft.rfft(frames, axis=1)

    # Magnitude
    mag = np.abs(spec).astype(np.float32)

    # Normalize
    mmax = float(mag.max()) if mag.size else 1.0
    _dbg(f"[STFT] mag.max={mmax:.6e}")
    mag_norm = mag / (mmax + 1e-12)
    mag_norm = np.minimum(mag_norm, 1.0)

    # Convert to dB
    S_db = (20.0 * np.log10(np.maximum(mag_norm, 1e-12))).astype(np.float32)

    # Frequency and time axes
    f = np.fft.rfftfreq(nperseg, d=1.0 / float(sr)).astype(np.float32)
    t = (np.arange(n_frames, dtype=np.float32) * (hop / float(sr)))

    # Transpose for (freq × time)
    S_db_FT = S_db.T
    _arr_stats("[STFT] S_db (F,T)", S_db_FT)

    return f, t, S_db_FT

class Player(QtCore.QObject):
    """
    Play a section of audio using sounddevice.

    Emits:
        started(t0, t1): When playback starts.
        stopped(): When playback stops.
    """
    started = QtCore.Signal(float, float)
    stopped = QtCore.Signal()

    def __init__(self):
        """Set up the player with no active stream."""
        super().__init__()
        self._stream = None
        self.playing = False

    def play(self, y: np.ndarray, sr: int, t0: float, t1: float):
        """
        Play the part of `y` between times `t0` and `t1` (in seconds).

        Args:
            y (np.ndarray): 1D audio signal.
            sr (int): Sampling rate in Hz.
            t0 (float): Start time in seconds.
            t1 (float): End time in seconds.

        Notes:
            - Does nothing if sounddevice is not installed.
            - Plays once and stops automatically.
        """
        if not HAVE_SD:
            return

        # Stop any current playback
        self.stop()

        # Convert time to sample indices
        start = int(max(0, t0) * sr)
        end = int(min(len(y), t1 * sr))

        # Slice the audio buffer
        data = y[start:end].astype(np.float32)
        pos = 0

        # Callback for feeding data to the audio stream
        def _cb(outdata, frames, timeinfo, status):
            nonlocal pos
            n = min(frames, len(data) - pos)
            if n > 0:
                outdata[:n, 0] = data[pos:pos + n]
            if frames > n:
                outdata[n:, 0] = 0
            pos += n
            if pos >= len(data):
                raise sd.CallbackStop

        # Start the audio stream
        self._stream = sd.OutputStream(
            channels=1,
            samplerate=sr,
            dtype="float32",
            callback=_cb
        )
        self._stream.start()
        self.playing = True
        self.started.emit(t0, t1)

    def play_region(self, y: np.ndarray, sr: int, t_start: float, t_end: float):
        """
        Wrapper for play(). Ensures times are valid and forwards the call.
        """
        t_start = max(0.0, float(t_start))
        t_end = max(t_start, float(t_end))
        self.play(y, sr, t_start, t_end)

    def stop(self):
        """
        Stop playback if active and close the stream.
        """
        if not HAVE_SD:
            return

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self.playing:
            self.playing = False
            self.stopped.emit()


# Debug helpers
DEBUG_STFT = False
DYNAMIC_SPECTRO_LEVELS = False
GRAYSCALE_DEBUG = False

def _dbg(msg: str):
    """
    Print a debug message if DEBUG_STFT is True.
    """
    if DEBUG_STFT:
        print(msg)
        sys.stdout.flush()


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
