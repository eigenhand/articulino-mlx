"""Post-processing of the decoder output: a gentle low-shelf and a peak limit.

The released decoder is balanced on its own; by ear, +2 dB below about 180 Hz sounded best (RBJ cookbook shelf,
slope 0.7). Loud reference voices can drive the decoder to full scale, so a chunk whose peak is above 0.95 is scaled
down to 0.95. Both steps can be switched off.

The IIR filter needs no SciPy: its impulse response is computed once (exact recursion, until it has decayed below
1e-20) and applied by overlap-add FFT convolution. The result matches ``scipy.signal.sosfilt`` with zero initial
state to about 1e-13.
"""
from __future__ import annotations

import math

import numpy as np


def shelf_sos(freq: float, gain_db: float, high: bool = False, sample_rate: int = 24000,
              slope: float = 0.7) -> np.ndarray:
    """RBJ cookbook shelving filter as one second-order section [b0, b1, b2, 1, a1, a2]."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * freq / sample_rate
    c = np.cos(w)
    q = 2 * np.sqrt(a) * np.sin(w) / 2 * np.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    z = 1 if high else -1
    b = [a * ((a + 1) + z * (a - 1) * c + q), -2 * z * a * ((a - 1) + z * (a + 1) * c),
         a * ((a + 1) + z * (a - 1) * c - q)]
    den = [(a + 1) - z * (a - 1) * c + q, 2 * z * ((a - 1) - z * (a + 1) * c), (a + 1) - z * (a - 1) * c - q]
    return np.array([x / den[0] for x in b + den])


def _poles(section: np.ndarray) -> np.ndarray:
    a0, a1, a2 = (float(v) for v in section[3:])
    return np.roots([a0, a1, a2]) if (a1 or a2) else np.zeros(0)


def _recursive(x: np.ndarray, section: np.ndarray) -> np.ndarray:
    """One section by direct recursion (zero initial state), in float64."""
    b0, b1, b2, a0, a1, a2 = (float(v) / float(section[3]) for v in section)
    y = np.zeros(len(x))
    x1 = x2 = y1 = y2 = 0.0
    for n, xn in enumerate(x.tolist()):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1, y2, y1 = x1, xn, y1, yn
    return y


class SosFilter:
    """A cascade of second-order sections (rows [b0, b1, b2, a0, a1, a2]) with zero initial state."""

    def __init__(self, sos: np.ndarray, tail: float = 1e-20):
        self.sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        radius = max([abs(p) for s in self.sos for p in _poles(s)] or [0.0])
        if radius >= 1:
            raise ValueError("unstable filter (pole on or outside the unit circle)")
        length = 3 * len(self.sos) + (math.ceil(math.log(tail) / math.log(radius)) if radius > 0 else 0) + 1
        impulse = np.zeros(length)
        impulse[0] = 1.0
        for section in self.sos:
            impulse = _recursive(impulse, section)
        self.impulse = impulse

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter x (1-D) by overlap-add FFT convolution with the impulse response; returns float64."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        h = self.impulse[: max(n, 1)]
        m = len(h)
        if n == 0:
            return x.copy()
        nfft = 1 << max(10, (2 * m - 1).bit_length())
        step = nfft - m + 1                                   # new samples per block; the tail (m - 1) <= step
        nb = -(-n // step)
        blocks = np.zeros(nb * step)
        blocks[:n] = x
        y = np.fft.irfft(np.fft.rfft(blocks.reshape(nb, step), nfft, axis=1) * np.fft.rfft(h, nfft), nfft, axis=1)
        out = np.zeros((nb + 1) * step)
        out[: nb * step] += y[:, :step].ravel()
        tail = np.zeros((nb, step))
        tail[:, : m - 1] = y[:, step: step + m - 1]
        out[step:] += tail.ravel()
        return out[:n]


def limit_peak(wav: np.ndarray, peak: float | None = 0.95) -> np.ndarray:
    """Scale the whole chunk down if its peak exceeds ``peak`` (no clipping, no compression)."""
    if peak is None or len(wav) == 0:
        return wav
    top = float(np.abs(wav).max())
    return wav * (peak / top) if top > peak else wav
