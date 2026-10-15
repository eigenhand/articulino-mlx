"""Measures used by the tests and tools to compare audio of different precisions (NumPy only)."""
from __future__ import annotations

import numpy as np

from .codec import HOP, MEL_N_FFT, MEL_WIN, hann_window, mel_filterbank


def snr(x: np.ndarray, reference: np.ndarray) -> float:
    """Signal-to-noise ratio of ``x`` against ``reference`` in dB (same shapes)."""
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if x.shape != r.shape:
        raise ValueError(f"shapes differ: {x.shape} vs {r.shape}")
    noise = ((x - r) ** 2).sum()
    return float("inf") if noise == 0 else float(10 * np.log10((r ** 2).sum() / noise))


def log_mel(wav: np.ndarray) -> np.ndarray:
    """Magnitude mel spectrogram in dB (frames, 100), the encoder's mel bands (float64, NumPy)."""
    x = np.asarray(wav, dtype=np.float64)
    pad = MEL_N_FFT // 2
    x = np.pad(x, pad, mode="reflect") if len(x) > pad else np.pad(x, pad)
    window = np.zeros(MEL_N_FFT)
    off = (MEL_N_FFT - MEL_WIN) // 2
    window[off: off + MEL_WIN] = hann_window(MEL_WIN)
    n = 1 + (len(x) - MEL_N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, MEL_N_FFT)[::HOP][:n]
    mel = np.abs(np.fft.rfft(frames * window, axis=-1)) @ mel_filterbank().astype(np.float64)
    return 20 * np.log10(np.maximum(mel, 1e-9))


def mel_distance(x: np.ndarray, reference: np.ndarray, floor_db: float = 80.0) -> float:
    """Mean absolute difference of the mel spectrograms in dB, over the time-frequency cells within ``floor_db`` of
    the reference's loudest cell (silence ignored). Unlike the waveform SNR, it does not punish phase differences;
    about 0.1 dB is inaudible, a few dB are a clearly different rendering. Lengths may differ by a few frames (the
    shorter one is used)."""
    a, b = log_mel(x), log_mel(reference)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    floor = b.max() - floor_db
    keep = np.maximum(a, b) > floor
    return float(np.abs(np.maximum(a, floor) - np.maximum(b, floor))[keep].mean())
