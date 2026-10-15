"""Latent codec in MLX: 24 kHz audio <-> 96-dim latents at 25 Hz (weights in ``codec.safetensors``).

A port of ``articulino.codec`` (PyTorch). MLX convolutions work on (B, T, C), so every tensor here is time-major;
convolution weights are stored as (out, kernel, in) (the converter transposes them, see ``weights.py``).

* :class:`LogMel`: the encoder's input features, identical to torchaudio's ``MelSpectrogram`` (n_fft 1024,
  win 960, hop 240, 100 HTK mel bands without filter normalization, magnitude, reflect padding) followed by
  ``log(clamp(x, 1e-5))``; the STFT uses ``mx.fft.rfft``.
* :class:`LatentEncoder`: log-mel (100 Hz) -> ConvNeXt stages with two stride-2 convolutions -> mean latents (25 Hz).
  Only used once per voice.
* :class:`LatentDecoder`: latents (25 Hz) -> 4x transposed convolution (100 Hz) -> 12 ConvNeXt blocks -> log
  magnitude and phase per STFT bin -> inverse STFT (n_fft 960, hop 240, ``mx.fft.irfft``, overlap-add divided by the
  summed squared window like ``torch.istft``) -> 24 kHz audio.

The log-mel and the inverse STFT always run in float32; the networks run in the dtype of their weights.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import Conv1d, weight_dtype

SAMPLE_RATE = 24000
HOP = 240                 # 100 Hz feature frames
N_FFT = 960               # inverse STFT of the decoder
MEL_N_FFT = 1024
MEL_WIN = 960
N_MELS = 100
LATENT_HOP = 960          # samples per latent frame (25 Hz)


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (like ``torch.hann_window(n)``), float32."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n, dtype=np.float64) / n)).astype(np.float32)


def mel_filterbank(n_freqs: int = MEL_N_FFT // 2 + 1, f_min: float = 0.0, f_max: float = SAMPLE_RATE / 2,
                   n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Triangular HTK mel filters without normalization, (n_freqs, n_mels) float32; same numbers as
    ``torchaudio.functional.melscale_fbanks(..., norm=None, mel_scale="htk")``."""
    f32 = np.float32
    all_freqs = np.linspace(0, sample_rate // 2, n_freqs, dtype=f32)
    m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=f32)
    f_pts = (f32(700.0) * (f32(10.0) ** (m_pts / f32(2595.0)) - f32(1.0))).astype(f32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = (f32(-1.0) * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(f32(0.0), np.minimum(down, up)).astype(f32)


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        pad = (MEL_N_FFT - MEL_WIN) // 2
        window = np.zeros(MEL_N_FFT, dtype=np.float32)
        window[pad: pad + MEL_WIN] = hann_window(MEL_WIN)                 # periodic, centred in 1024
        self._window = mx.array(window)
        self._filters = mx.array(mel_filterbank())
        self.freeze()

    def __call__(self, wav: mx.array) -> mx.array:
        """wav (B, T) float32 -> log-mel (B, T // 240 + 1, 100)."""
        t = wav.shape[-1]
        half = MEL_N_FFT // 2
        if t <= half:
            raise ValueError("audio too short for the log-mel (reflect padding)")
        # centre=True with reflect padding, then frames every 240 samples
        idx = np.concatenate([np.arange(half, 0, -1), np.arange(t), t - 2 - np.arange(half)])
        frames = 1 + t // HOP
        gather = idx[np.arange(frames)[:, None] * HOP + np.arange(MEL_N_FFT)[None, :]]
        x = mx.take(wav.astype(mx.float32), mx.array(gather.astype(np.int32)), axis=-1)       # (B, F, 1024)
        spec = mx.abs(mx.fft.rfft(x * self._window, axis=-1))                                   # (B, F, 513)
        return mx.log(mx.maximum(spec @ self._filters, 1e-5))


class DepthwiseConv1d(nn.Module):
    """Depthwise Conv1d (one filter per channel), weight (C, K, 1) as ``nn.Conv1d(..., groups=C)`` stores it;
    computed as a sum of K shifted, scaled copies (faster than a grouped convolution on the CPU)."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = mx.zeros((channels, kernel_size, 1))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:           # x (B, T, C)
        k = self.weight.shape[1]
        n = x.shape[1] + 2 * self.padding - k + 1
        xp = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        w = self.weight[:, :, 0].T                          # (K, C)
        y = xp[:, :n] * w[0]
        for j in range(1, k):
            y = y + xp[:, j: j + n] * w[j]
        return y + self.bias


class Upsample(nn.Module):
    """ConvTranspose1d(dim, dim, kernel, stride, padding) as one matrix product and an overlap-add.

    weight (kernel·out, in): row j·out + o holds the PyTorch weight [:, o, j] (the converter writes it this way);
    every input frame l contributes y[l, j] to output position l·stride + j - padding. Exact, and much faster than
    MLX's transposed convolution on the CPU."""

    def __init__(self, dim: int, kernel_size: int = 8, stride: int = 4, padding: int = 2):
        super().__init__()
        if kernel_size % stride:
            raise ValueError("kernel_size must be a multiple of stride")
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = mx.zeros((kernel_size * dim, dim))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:           # x (B, L, C_in) -> (B, L·stride, C_out)
        b, n, _ = x.shape
        s, parts = self.stride, self.kernel_size // self.stride
        out_ch = self.weight.shape[0] // self.kernel_size
        y = (x @ self.weight.T).reshape(b, n, parts, s, out_ch)
        acc = mx.zeros((b, n + parts - 1, s, out_ch), dtype=y.dtype)
        for j in range(parts):                              # tap block j lands j frames later
            acc = acc + mx.pad(y[:, :, j], [(0, 0), (j, parts - 1 - j), (0, 0), (0, 0)])
        full = acc.reshape(b, (n + parts - 1) * s, out_ch)
        length = (n - 1) * s + self.kernel_size - 2 * self.padding
        return full[:, self.padding: self.padding + length] + self.bias


class ConvNeXtBlock(nn.Module):
    """Depthwise conv (k 7) -> LayerNorm (eps 1e-6) -> Linear ×3 -> GELU -> Linear -> x + scale·y."""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.depthwise_conv = DepthwiseConv1d(dim, 7)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.up = nn.Linear(dim, inner)
        self.down = nn.Linear(inner, dim)
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:           # x (B, T, C)
        y = self.down(nn.gelu(self.up(self.norm(self.depthwise_conv(x)))))
        return x + self.scale * y


class LatentEncoder(nn.Module):
    def __init__(self, dim: int = 512, latent_dim: int = 96, blocks_per_stage: int = 4):
        super().__init__()
        self.blocks_per_stage = blocks_per_stage
        self.mel = LogMel()
        self.input_conv = Conv1d(N_MELS, dim, 7, padding=3)
        self.blocks = [ConvNeXtBlock(dim, 3 * dim) for _ in range(3 * blocks_per_stage)]
        self.downsample = [Conv1d(dim, dim, 4, stride=2, padding=1) for _ in range(2)]
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_proj = nn.Linear(dim, latent_dim)

    def __call__(self, wav: mx.array) -> mx.array:
        """wav (B, T) with T a multiple of 960 -> mean latents (B, T // 960, 96), float32."""
        mel = self.mel(wav)[:, : wav.shape[-1] // HOP]
        x = self.input_conv(mel.astype(weight_dtype(self.input_conv)))
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % self.blocks_per_stage == self.blocks_per_stage - 1 and i // self.blocks_per_stage < 2:
                x = self.downsample[i // self.blocks_per_stage](x)
        return self.output_proj(self.final_norm(x)).astype(mx.float32)


@lru_cache(maxsize=64)
def _istft_envelope(frames: int) -> mx.array:
    """Overlap-added squared window for ``frames`` frames, cut like ``torch.istft(center=True, length=frames·240)``
    (float32, computed like torch)."""
    w2 = hann_window(N_FFT) ** 2
    out = np.zeros(N_FFT + HOP * (frames - 1), dtype=np.float32)
    for j in range(N_FFT // HOP):
        seg = w2[j * HOP: (j + 1) * HOP]
        out[j * HOP: j * HOP + frames * HOP] += np.tile(seg, frames)
    env = out[N_FFT // 2: N_FFT // 2 + frames * HOP]
    if env.min() < 1e-11:
        raise ValueError("window overlap-add too small (iSTFT not invertible)")
    return mx.array(env)


def istft(spec: mx.array, length: int) -> mx.array:
    """Inverse STFT like ``torch.istft(spec, 960, 240, 960, hann_window(960), center=True, length=length)``.

    spec (B, F, 481) complex64 (time-major) -> (B, length) float32, length <= F·240 + 240 (here F·240)."""
    b, f, _ = spec.shape
    window = mx.array(hann_window(N_FFT))
    frames = mx.fft.irfft(spec, n=N_FFT, axis=-1) * window                       # (B, F, 960)
    parts = N_FFT // HOP                                                         # 4 chunks of 240 per frame
    chunks = frames.reshape(b, f, parts, HOP)
    # overlap-add: output chunk m gets chunk j of frame m - j
    out = mx.zeros((b, f + parts - 1, HOP), dtype=mx.float32)
    for j in range(parts):
        out = out + mx.pad(chunks[:, :, j], [(0, 0), (j, parts - 1 - j), (0, 0)])
    out = out.reshape(b, -1)
    start = N_FFT // 2
    if start + length > out.shape[-1]:
        raise ValueError("istft length longer than the signal")
    env_frames = length // HOP
    y = out[:, start: start + length]
    if length == env_frames * HOP and length <= f * HOP:
        return y / _istft_envelope(env_frames)[None]
    raise ValueError("istft length must be a multiple of 240 and at most frames·240")


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 96, dim: int = 512, blocks: int = 12, max_magnitude: float = 100.0):
        super().__init__()
        self.max_magnitude = max_magnitude
        self.input_conv = Conv1d(latent_dim, dim, 3, padding=1)
        self.upsample = Upsample(dim, 8, stride=4, padding=2)                         # 25 -> 100 Hz
        self.blocks = [ConvNeXtBlock(dim, 3 * dim) for _ in range(blocks)]
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, N_FFT + 2)                                          # 481 log-magnitudes + 481 phases

    def spectrum(self, z: mx.array) -> mx.array:
        """z (B, L, 96) de-normalized latents -> complex spectrum (B, 4L, 481), complex64."""
        h = self.upsample(self.input_conv(z.astype(weight_dtype(self.input_conv))))
        for block in self.blocks:
            h = block(h)
        h = self.head(self.final_norm(h)).astype(mx.float32)
        log_mag, phase = mx.split(h, 2, axis=-1)
        mag = mx.minimum(mx.exp(log_mag), self.max_magnitude)
        real, imag = mag * mx.cos(phase), mag * mx.sin(phase)
        return real.astype(mx.complex64) + mx.array(1j, dtype=mx.complex64) * imag.astype(mx.complex64)

    def __call__(self, z: mx.array) -> mx.array:
        """z (B, L, 96) de-normalized latents -> audio (B, L·960), float32."""
        return istft(self.spectrum(z), z.shape[1] * LATENT_HOP)


class Codec(nn.Module):
    def __init__(self, latent_dim: int = 96, encoder_dim: int = 512, decoder_dim: int = 512, decoder_blocks: int = 12,
                 blocks_per_stage: int = 4, max_magnitude: float = 100.0):
        super().__init__()
        self.encoder = LatentEncoder(encoder_dim, latent_dim, blocks_per_stage)
        self.decoder = LatentDecoder(latent_dim, decoder_dim, decoder_blocks, max_magnitude)

    @classmethod
    def from_config(cls, codec_cfg: dict, latent_dim: int = 96) -> "Codec":
        enc, dec = codec_cfg.get("encoder", {}), codec_cfg.get("decoder", {})
        return cls(latent_dim=latent_dim, encoder_dim=enc.get("dim", 512), decoder_dim=dec.get("dim", 512),
                   decoder_blocks=dec.get("blocks", 12), blocks_per_stage=enc.get("blocks_per_stage", 4),
                   max_magnitude=dec.get("max_magnitude", 100.0))
