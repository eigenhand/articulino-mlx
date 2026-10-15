"""Sampling noise that matches ``torch.randn`` on the CPU, without torch.

The reference package draws the noise of every sampling step with ``torch.Generator().manual_seed(seed)`` and
``torch.randn(1, frames, 96)``. PyTorch's CPU generator is a 32-bit Mersenne Twister (MT19937, standard seeding); a
float32 ``randn`` with at least 16 elements first fills the tensor with uniforms ``(r & 0xFFFFFF) / 2^24`` and then
turns every block of 16 into normals by Box-Muller: for j < 8, ``u1 = 1 - x[j]``, ``u2 = x[j + 8]``,
``x[j] = sqrt(-2 ln u1)·cos(2π u2)``, ``x[j + 8] = sqrt(-2 ln u1)·sin(2π u2)``. NumPy's legacy ``RandomState``
uses the same generator and seeding, so the same seed gives the same noise here, up to float rounding (about 1e-7),
and therefore the same audio as the PyTorch package on every device.
"""
from __future__ import annotations

import numpy as np


class TorchCompatibleGenerator:
    """``torch.Generator().manual_seed(seed)`` for float32 ``randn`` (numel a multiple of 16)."""

    def __init__(self, seed: int):
        seed = int(seed)
        if not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be in [0, 2**32)")
        self._bits = np.random.RandomState(seed)._bit_generator        # MT19937, init_genrand(seed)

    def randn(self, *shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        if n % 16:
            raise ValueError("only sizes that are a multiple of 16 are supported")
        raw = self._bits.random_raw(n).astype(np.uint64)
        u = ((raw & 0xFFFFFF).astype(np.float64) * 2.0 ** -24).astype(np.float32).reshape(-1, 2, 8)
        u1 = (np.float32(1.0) - u[:, 0]).astype(np.float64)
        u2 = u[:, 1].astype(np.float64)
        radius = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        out = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        return out.astype(np.float32).reshape(shape)


def draw_noise(seed: int, total_frames: int, prompt_frames: int, steps: int, latent_dim: int = 96) -> list[np.ndarray]:
    """The noise for every step, (1, total_frames - prompt_frames, latent_dim) float32 each.

    Drawn over all frames (prompt included) and then cut, exactly like the original sampler and the PyTorch package,
    so a seed gives the same noise as there."""
    gen = TorchCompatibleGenerator(seed)
    return [gen.randn(1, total_frames, latent_dim)[:, prompt_frames:] for _ in range(steps)]
