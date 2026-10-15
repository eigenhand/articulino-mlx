"""Reading reference recordings and resampling them to 24 kHz (NumPy only)."""
from __future__ import annotations

import math
import os

import numpy as np


def load_audio(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read an audio file as mono float32 (channels averaged). Returns (samples, sample_rate)."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def resample(wav: np.ndarray, orig_sr: int, new_sr: int, lowpass_filter_width: int = 6,
             rolloff: float = 0.99) -> np.ndarray:
    """Band-limited (Hann-windowed sinc) resampling of mono audio.

    Same algorithm, defaults and float32 arithmetic as ``torchaudio.functional.resample`` (``sinc_interp_hann``)
    and the PyTorch package, without torch."""
    if orig_sr == new_sr:
        return np.asarray(wav, dtype=np.float32)
    f32 = np.float32
    gcd = math.gcd(int(orig_sr), int(new_sr))
    orig, new = int(orig_sr) // gcd, int(new_sr) // gcd
    x = np.asarray(wav, dtype=f32)
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = np.arange(-width, width + orig, dtype=f32)[None, :] / f32(orig)
    t = np.arange(0, -new, -1, dtype=f32)[:, None] / f32(new) + idx
    t = t * f32(base)
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width).astype(f32)
    window = np.cos(t * f32(math.pi) / f32(lowpass_filter_width) / f32(2)) ** 2
    t = t * f32(math.pi)
    scale = f32(base / orig)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernels = np.where(t == 0, f32(1.0), np.sin(t) / t).astype(f32)
    kernels = kernels * window * scale                                       # (new, K)
    length = x.shape[-1]
    padded = np.pad(x, (width, width + orig))
    frames = np.lib.stride_tricks.sliding_window_view(padded, kernels.shape[1])[::orig]   # (n, K)
    y = (frames @ kernels.T).reshape(-1)                                     # sample n·new + c
    target = math.ceil(new * length / orig)
    return y[:target].astype(np.float32)
