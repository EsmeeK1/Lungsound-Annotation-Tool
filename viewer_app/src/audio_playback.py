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


