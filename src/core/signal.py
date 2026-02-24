from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SignalState:
    t: list[float]
    y: list[float]
    y_smooth: float = 0.0
    init: bool = False

def ema_update(prev: float, x: float, alpha: float) -> float:
    if alpha <= 0:
        return x
    return alpha * x + (1.0 - alpha) * prev

def detrend(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x
    t = np.arange(len(x), dtype=np.float32)
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + b)

def window_metrics(t: np.ndarray, x: np.ndarray, freq_min: float, freq_max: float) -> dict:
    if x.size < 8:
        return {"amp_pp": 0.0, "rms": 0.0, "f_dom": 0.0}

    amp_pp = float(np.max(x) - np.min(x))
    rms = float(np.sqrt(np.mean(x**2)))

    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if dt <= 0:
        return {"amp_pp": amp_pp, "rms": rms, "f_dom": 0.0}

    X = np.fft.rfft(x * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), d=dt)
    mag = np.abs(X)

    band = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(band):
        return {"amp_pp": amp_pp, "rms": rms, "f_dom": 0.0}

    idx = int(np.argmax(mag[band]))
    f_dom = float(freqs[band][idx])
    return {"amp_pp": amp_pp, "rms": rms, "f_dom": f_dom}
