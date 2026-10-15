"""Durations and the 4-step sampler of the distilled student (DMD2 style, no classifier-free guidance).

From pure noise, each step predicts the clean latents ``x1 = x + (1 - t_k)·v(x, t_k)``; the next step starts from
those latents re-noised to the next time, ``x = (1 - t_{k+1})·noise_{k+1} + t_{k+1}·x1``. The result is ``x1`` of
the last step. The times are fixed by training: t = 0, 0.25, 0.5, 0.75. The update runs in float32, whatever
dtype the DiT computes in.
"""
from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from .model import DiT, VoiceState
from .noise import draw_noise
from .text import CHARSET

__all__ = ["letter_rate", "draw_noise", "sample"]

_IS_LETTER = np.array([False] + [c.isalnum() for c in CHARSET])


def letter_rate(durations: np.ndarray, ids: np.ndarray, start: int = 0,
                target: float | None = None) -> tuple[list[float], np.ndarray]:
    """Speaking rate in letters per second, counted over letters and digits from position ``start`` (pauses at
    spaces and punctuation do not count). Single over-long letters (silence the aligner gave to one letter) are
    capped at three times the median.

    With ``target``, the letter durations are rescaled so the rate becomes exactly ``target`` (returned as the second
    value; otherwise the durations come back unchanged). durations (B, N) in 10 ms units, ids (B, N)."""
    d_all = np.array(durations, dtype=np.float32)
    ids_np = np.asarray(ids)
    rates = []
    for i in range(ids_np.shape[0]):
        m = _IS_LETTER[ids_np[i]].copy()
        m[:start] = False
        if not m.any():
            rates.append(0.0)
            continue
        d = np.maximum(d_all[i, m], 0.5)
        d = np.minimum(d, 3 * np.median(d))
        rate = float(m.sum() / (d.sum() / 100))
        if target:
            d = d * (rate / target)
            rate = target
        d_all[i, m] = d
        rates.append(rate)
    if target is None:
        return rates, np.asarray(durations, dtype=np.float32)
    return rates, d_all


def sample(dit: DiT, frame_cond: mx.array, voice: VoiceState, timesteps: Sequence[float],
           noise: Sequence[np.ndarray | mx.array]) -> mx.array:
    """Generate the latents of the new frames (lazily; evaluate the result).

    frame_cond (B, L, 512): text features per new frame; noise: one array (B, L, 96) per step (from
    :func:`draw_noise`, or given from outside for tests). Returns the clean normalized latents (B, L, 96), float32."""
    if len(noise) < len(timesteps):
        raise ValueError(f"need {len(timesteps)} noise tensors, got {len(noise)}")
    b = frame_cond.shape[0]
    noise = [n if isinstance(n, mx.array) else mx.array(np.asarray(n, dtype=np.float32)) for n in noise]
    x = noise[0].astype(mx.float32)
    x1 = x
    for k, t in enumerate(timesteps):
        v = dit.velocity(x, mx.full((b,), float(t)), frame_cond, voice).astype(mx.float32)
        x1 = x + (1 - t) * v
        if k + 1 < len(timesteps):
            t_next = timesteps[k + 1]
            x = (1 - t_next) * noise[k + 1].astype(mx.float32) + t_next * x1
    return x1
