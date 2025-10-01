# preprocess.py
import numpy as np
from scipy import signal
import librosa

def _bandpass(y, fs, low, high, order=4, zero_phase=True):
    high = min(high, fs/2 - 1.0)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return (signal.sosfiltfilt(sos, y) if zero_phase else signal.sosfilt(sos, y)).astype(np.float32) # type: ignore

def _stft_db_img(y, sr, n_fft, hop, win, t_target):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=True)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)
    S_db = librosa.util.fix_length(S_db, size=int(t_target), axis=1)
    S_db = np.clip(S_db, -80.0, 0.0)
    S01  = (S_db + 80.0)/80.0
    return S01.astype(np.float32)  # (F,T)

def build_preprocessor(reader_factory, cfg, manifest):
    out_shape = tuple(manifest["input_shape"])      # (F,T,1)
    out_dtype = np.dtype(manifest["input_dtype"])   # float32

    fs      = int(manifest.get("sample_rate", getattr(cfg, "FS", 16000)))
    win_sec = float(manifest.get("raw_window_seconds", 3.0))

    stft_cfg = manifest.get("stft", {})
    n_fft  = int(stft_cfg.get("n_fft", 1024))
    hop    = int(stft_cfg.get("hop", 256))
    win    = stft_cfg.get("window", "hann")

    bp     = manifest.get("bandpass", {})
    low_hz = float(bp.get("low_hz", 30.0))
    high_hz= float(bp.get("high_hz", 3000.0))
    order  = int(bp.get("order", 4))
    zero   = bool(bp.get("zero_phase", True))

    norm   = manifest.get("normalization", None)
    mu     = float(norm.get("mean", 0.0))
    sd     = max(float(norm.get("std", 1.0)), 1e-8)

    F, T, _ = out_shape
    t_target = T
    n_samples = int(fs * win_sec)

    def preproc():
        # 1) lees 3s mono int16
        buf = bytearray(); need = n_samples * 2
        with reader_factory() as reader:
            while len(buf) < need:
                raw = reader.read_bytes()
                if raw: buf.extend(raw)
        x = np.frombuffer(memoryview(buf)[:need], dtype=np.int16).astype(np.float32) / 32768.0

        # 2) bandpass
        x = _bandpass(x, fs, low_hz, high_hz, order=order, zero_phase=zero)

        # 3) STFT→dB→[0,1] + vaste T
        S = _stft_db_img(x, fs, n_fft, hop, win, t_target)  # (F,T)

        # 4) shape safety
        if S.shape[0] != F:
            if S.shape[0] > F: S = S[:F, :]
            else:              S = np.vstack([S, np.zeros((F - S.shape[0], S.shape[1]), dtype=S.dtype)])
        if S.shape[1] != T:
            if S.shape[1] > T: S = S[:, :T]
            else:              S = np.hstack([S, np.zeros((S.shape[0], T - S.shape[1]), dtype=S.dtype)])

        # 5) z-score
        S = ((S - mu) / sd).astype(out_dtype)

        # 6) kanaal
        out = S[..., None]  # (F,T,1)
        if out.shape != out_shape or out.dtype != out_dtype:
            raise ValueError(f"preproc shape/dtype mismatch: have {out.shape}/{out.dtype}, want {out_shape}/{out_dtype}")
        return out

    return preproc
