"""The acoustic model in MLX: text encoder, voice encoder, duration predictor and the 4-step DiT (student).

A port of ``articulino.model`` (PyTorch). Shapes use B = batch, N = characters (incl. the prompt placeholder at
index 0), R = reference frames, L = latent frames (25 Hz), P = prompt frames, D = model width. Parameter names are
the release names of ``model.safetensors``.

* LayerNorm eps 1e-5, exact (erf) GELU, SiLU in the time and voice-vector MLPs.
* Attention: one ``kv_proj`` matrix (keys first, then values), heads split after the projection, softmax(q·kᵀ/√d)·v
  (``mx.fast.scaled_dot_product_attention``; the softmax runs in float32 for every input type).
* Rotary position embedding on interleaved pairs (``mx.fast.rope`` with ``traditional=True``), self-attention only.
* The DiT continues from a clean voice prompt. The prompt frames attend only to themselves, so their keys and values
  are computed once per voice (:meth:`DiT.prompt_cache`) and every sentence only computes its new frames
  (:meth:`DiT.velocity`).

Every module computes in the dtype of its weights (float32 or float16); inputs are cast on entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .config import ArticulinoConfig, DiTConfig, DurationConfig, TextEncoderConfig, VoiceEncoderConfig

KV = tuple[mx.array, mx.array]
_TIME_BASE = 10000.0


# --------------------------------------------------------------------------------------------- building blocks
def rope(x: mx.array, base: float = 10000.0, offset: int = 0) -> mx.array:
    """Rotate interleaved pairs (x[..., 0::2], x[..., 1::2]) of x (B, H, L, head_dim) by position offset + 0..L-1;
    frequencies 1 / base^(2i/head_dim)."""
    return mx.fast.rope(x, x.shape[-1], traditional=True, base=base, scale=1.0, offset=offset)


def attend(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
    """softmax(q·kᵀ/√d)·v over heads; q (B, H, L, d), k/v (B, H, S, d) -> (B, L, H·d).

    ``mask`` (B, S) bool marks valid keys; invalid keys are left out of the softmax."""
    b, h, l, d = q.shape
    if mask is not None:
        mask = mask[:, None, None, :]
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d), mask=mask)
    return out.transpose(0, 2, 1, 3).reshape(b, l, h * d)


def weight_dtype(module: nn.Module) -> mx.Dtype:
    """The dtype a module computes in: the dtype of its first floating-point parameter (quantized layers: their
    scales)."""
    for _, p in tree_flatten(module.parameters()):
        if mx.issubdtype(p.dtype, mx.floating):
            return p.dtype
    return mx.float32


class Conv1d(nn.Conv1d):
    """``nn.Conv1d`` (weight (out, kernel, in), same parameters) computed as one matrix product over the stacked
    shifted inputs (im2col). Fast and exact on every backend; MLX's own CPU convolution is slow for some channel
    counts (e.g. 96 or 100 inputs)."""

    def __call__(self, x: mx.array) -> mx.array:           # x (B, T, C_in)
        out_ch, k, in_ch = self.weight.shape
        pad, stride = self.padding, self.stride
        if pad:
            x = mx.pad(x, [(0, 0), (pad, pad), (0, 0)])
        n = (x.shape[1] - k) // stride + 1
        cols = mx.concatenate([x[:, j: j + (n - 1) * stride + 1: stride] for j in range(k)], axis=-1)
        y = cols @ self.weight.reshape(out_ch, k * in_ch).T
        return y + self.bias if "bias" in self else y


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, context_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(context_dim or dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def keys_values(self, context: mx.array) -> KV:
        """Keys and values (B, H, S, head_dim) of ``context`` (B, S, context_dim)."""
        b, s, _ = context.shape
        kv = self.kv_proj(context).reshape(b, s, 2, self.heads, -1).transpose(2, 0, 3, 1, 4)
        return kv[0], kv[1]

    def queries(self, x: mx.array) -> mx.array:
        b, l, _ = x.shape
        return self.q_proj(x).reshape(b, l, self.heads, -1).transpose(0, 2, 1, 3)

    def __call__(self, x, context=None, mask=None, use_rope: bool = False, kv: KV | None = None):
        q = self.queries(x)
        k, v = kv if kv is not None else self.keys_values(x if context is None else context)
        if use_rope:
            q, k = rope(q), rope(k)
        return self.out_proj(attend(q, k, v, mask))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(dim, dim * mult)
        self.down = nn.Linear(dim * mult, dim)

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x)))


class MLP(nn.Module):
    """Linear -> SiLU -> Linear."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def __call__(self, x):
        return self.fc2(nn.silu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Pre-norm transformer layer with rotary self-attention (text and voice encoders)."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 4, eps: float = 1e-5):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim, eps=eps)
        self.ff = FeedForward(dim, ff_mult)

    def __call__(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask=mask, use_rope=True)
        return x + self.ff(self.ff_norm(x))


# --------------------------------------------------------------------------------------------- text encoder
class TextEncoder(nn.Module):
    """Character ids + language ids -> features per character (B, N, 512)."""

    def __init__(self, vocab_size: int, cfg: TextEncoderConfig):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, cfg.dim)
        self.language_embedding = nn.Embedding(2, cfg.dim)
        self.layers = [EncoderLayer(cfg.dim, cfg.heads, cfg.ff_mult, cfg.layer_norm_eps) for _ in range(cfg.layers)]
        self.final_norm = nn.LayerNorm(cfg.dim, eps=cfg.layer_norm_eps)

    def __call__(self, ids: mx.array, language: mx.array, mask: mx.array) -> mx.array:
        """ids (B, N) int; language (B,) for the whole text or (B, N) per character; mask (B, N) bool, True = real
        character (the prompt placeholder is False)."""
        lang = self.language_embedding(language)
        x = self.char_embedding(ids) + (lang[:, None] if language.ndim == 1 else lang)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x) * mask[..., None].astype(x.dtype)


# --------------------------------------------------------------------------------------------- voice encoder
class PoolLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_mult)


class VoiceEncoder(nn.Module):
    """Normalized reference latents (B, R, 96) -> 32 voice tokens (B, 32, 768) and a voice vector (B, 768)."""

    def __init__(self, cfg: VoiceEncoderConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.dim)
        self.layers = [EncoderLayer(cfg.dim, cfg.heads, cfg.ff_mult) for _ in range(cfg.layers)]
        self.queries = mx.zeros((cfg.num_tokens, cfg.dim))
        self.pool = [PoolLayer(cfg.dim, cfg.heads, cfg.ff_mult) for _ in range(cfg.pool_layers)]
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.token_proj = nn.Linear(cfg.dim, cfg.out_dim)
        self.vector_mlp = MLP(cfg.dim, cfg.out_dim, cfg.out_dim)

    def __call__(self, latents: mx.array, mask: mx.array | None = None) -> tuple[mx.array, mx.array]:
        x = self.input_proj(latents.astype(weight_dtype(self.input_proj)))
        for layer in self.layers:
            x = layer(x, mask)
        q = mx.broadcast_to(self.queries[None], (x.shape[0], *self.queries.shape))
        for p in self.pool:
            q = q + p.attn(p.attn_norm(q), context=x, mask=mask)
            q = q + p.ff(p.ff_norm(q))
        q = self.final_norm(q)
        return self.token_proj(q), self.vector_mlp(q.mean(1))


# --------------------------------------------------------------------------------------------- durations
class DurationPredictor(nn.Module):
    """Character features + voice vector -> log duration per character, in 10 ms units."""

    def __init__(self, cfg: DurationConfig):
        super().__init__()
        c, k = cfg.channels, cfg.kernel_size
        self.voice_proj = nn.Linear(cfg.voice_dim, cfg.input_dim)
        self.convs = [Conv1d(cfg.input_dim, c, k, padding=k // 2), Conv1d(c, c, k, padding=k // 2),
                      Conv1d(c, c, k, padding=k // 2)]
        self.norms = [nn.LayerNorm(c) for _ in range(3)]
        self.out_proj = nn.Linear(c, 1)

    def __call__(self, h: mx.array, voice_vector: mx.array, mask: mx.array) -> mx.array:
        dt = weight_dtype(self.convs[0])
        m = mask[..., None].astype(dt)
        x = (h.astype(dt) + self.voice_proj(voice_vector.astype(dt))[:, None]) * m
        for conv, norm in zip(self.convs, self.norms):
            x = norm(nn.relu(conv(x))) * m
        return self.out_proj(x).squeeze(-1)


class PreciseDurations:
    """The duration predictor evaluated in float64 on the CPU (MLX has no float64 on the GPU).

    The frame boundaries are cumulative sums of the durations, so float32 rounding in the predictor (a few 1e-6)
    shifts the later boundaries of a sentence and, through the DiT, changes the audio more than any other rounding
    in the pipeline. The predictor is tiny (1.9 M parameters, once per sentence), so it runs in float64 and returns
    float32 log-durations as close to the exact values as the float32 reference itself."""

    def __init__(self, predictor: DurationPredictor, cfg: DurationConfig):
        self.module = DurationPredictor(cfg)
        with mx.stream(mx.cpu):                             # every float64 operation must run on the CPU
            self.module.update(tree_map(lambda a: a.astype(mx.float64), predictor.parameters()))
            mx.eval(self.module.parameters())

    def __call__(self, h: mx.array, voice_vector: mx.array, mask: mx.array) -> mx.array:
        with mx.stream(mx.cpu):
            out = self.module(h.astype(mx.float64), voice_vector.astype(mx.float64), mask)
            return out.astype(mx.float32)


def frame_weights(durations: np.ndarray, frames: int, first: int = 0, units_per_frame: int = 4) -> np.ndarray:
    """Overlap weights (B, frames - first, N) that spread character features over latent frames ``first..frames-1``.

    durations (B, N) in 10 ms units (may be fractional), float32. The weight of character n on frame j is the overlap
    of [start_n, end_n) with [4j, 4j + 4), divided by 4 (computed in float32 like the reference)."""
    d = np.asarray(durations, dtype=np.float32)
    end = np.cumsum(d, axis=1, dtype=np.float32)
    start = end - d
    lo = (np.arange(first, frames, dtype=np.float32) * units_per_frame)[None, :, None]
    w = np.minimum(end[:, None, :], lo + units_per_frame) - np.maximum(start[:, None, :], lo)
    return np.maximum(w, 0).astype(np.float32) / np.float32(units_per_frame)


def expand_to_frames(h: mx.array, durations: np.ndarray, frames: int, first: int = 0,
                     units_per_frame: int = 4) -> mx.array:
    """Spread character features h (B, N, C) over latent frames: (B, frames - first, C)."""
    w = mx.array(frame_weights(durations, frames, first, units_per_frame))
    return mx.matmul(w.astype(h.dtype), h)


# --------------------------------------------------------------------------------------------- DiT
class AdaNorm(nn.Module):
    """LayerNorm without weights (eps 1e-5), then x·(1 + scale) + shift with [scale, shift] = Linear(c)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim)

    def modulation(self, c: mx.array) -> tuple[mx.array, mx.array]:
        """c (B, D) -> 1 + scale, shift (B, 1, D): the same for every frame."""
        scale, shift = mx.split(self.proj(c)[:, None], 2, axis=-1)
        return 1 + scale, shift

    @staticmethod
    def apply(x: mx.array, mod: tuple[mx.array, mx.array]) -> mx.array:
        gain, shift = mod
        return mx.fast.layer_norm(x, None, None, 1e-5) * gain + shift


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn_norm = AdaNorm(dim)
        self.attn = Attention(dim, heads)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, heads, context_dim=dim)
        self.ff_norm = AdaNorm(dim)
        self.ff = FeedForward(dim, ff_mult)

    def __call__(self, x, c, offset: int, cross_kv: KV, prompt_kv: KV | None = None, keep: list | None = None,
                 base: float = 10000.0):
        """x (B, L, D); c (B, D) condition (time + voice); offset: position of the first frame of x (rope);
        cross_kv: keys/values of the voice tokens; prompt_kv: keys/values of the prompt frames in front of x
        (attended without mask); keep: list that receives this block's own (k, v) (to build the prompt cache)."""
        xn = AdaNorm.apply(x, self.attn_norm.modulation(c))
        q = rope(self.attn.queries(xn), base, offset)
        k, v = self.attn.keys_values(xn)
        k = rope(k, base, offset)
        if keep is not None:
            keep.append((k, v))
        if prompt_kv is not None:
            k = mx.concatenate([prompt_kv[0].astype(k.dtype), k], axis=2)
            v = mx.concatenate([prompt_kv[1].astype(v.dtype), v], axis=2)
        x = x + self.attn.out_proj(attend(q, k, v))
        x = x + self.cross_attn(self.cross_norm(x), kv=cross_kv)
        return x + self.ff(AdaNorm.apply(x, self.ff_norm.modulation(c)))


@dataclass
class VoiceState:
    """Everything that depends only on the voice, computed once by :meth:`AcousticModel.prepare_voice`."""

    latents: mx.array               # (1, R, 96) normalized reference latents, float32
    tokens: mx.array                # (1, 32, 768) voice tokens
    vector: mx.array                # (1, 768) voice vector
    prompt: mx.array                # (1, P, 96) the first P reference frames, continued from (float32)
    dit_vector: mx.array            # (1, 384) voice vector in DiT width
    cross_kv: list[KV]              # per DiT block: keys/values of the voice tokens
    prompt_kv: list[KV]             # per DiT block: keys/values of the prompt frames (the voice cache)

    @property
    def prompt_frames(self) -> int:
        return self.prompt.shape[1]

    def arrays(self) -> list[mx.array]:
        return [self.latents, self.tokens, self.vector, self.prompt, self.dit_vector,
                *[a for kv in self.cross_kv for a in kv], *[a for kv in self.prompt_kv for a in kv]]


class DiT(nn.Module):
    """Flow-matching velocity v(x_t, t) on normalized 96-dim latents (student: 8 blocks, D = 384, 6 heads)."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.voice_token_proj = nn.Linear(cfg.voice_dim, d)
        self.voice_vector_proj = nn.Linear(cfg.voice_dim, d)
        self.input_proj = nn.Linear(2 * cfg.latent_dim + 1, d)
        self.time_mlp = MLP(cfg.time_embedding_dim, d, d)
        self.blocks = [DiTBlock(d, cfg.heads, cfg.ff_mult) for _ in range(cfg.layers)]
        self.final_norm = AdaNorm(d)
        self.output_proj = nn.Linear(d, cfg.latent_dim)

    @property
    def dtype(self) -> mx.Dtype:
        return weight_dtype(self.input_proj)

    def time_embedding(self, t: mx.array) -> mx.array:
        """Sinusoidal features of t (B,) -> (B, 256), float32: w = t·1000·exp(-ln(10000)·i/128), [sin w, cos w]."""
        half = self.cfg.time_embedding_dim // 2
        f = mx.exp(-math.log(_TIME_BASE) * mx.arange(half, dtype=mx.float32) / half)
        w = t.astype(mx.float32)[:, None] * self.cfg.time_scale * f[None]
        return mx.concatenate([mx.sin(w), mx.cos(w)], axis=-1)

    def condition(self, t: mx.array, dit_vector: mx.array) -> mx.array:
        """Per-sequence condition c = time_mlp(time_embedding(t)) + projected voice vector, (B, D)."""
        return self.time_mlp(self.time_embedding(t).astype(self.dtype)) + dit_vector.astype(self.dtype)

    def prompt_cache(self, prompt: mx.array, dit_tokens: mx.array, dit_vector: mx.array) -> list[KV]:
        """Run the clean prompt frames (B, P, 96) once at t = 1 and keep every block's keys and values."""
        b, p, _ = prompt.shape
        prompt = prompt.astype(self.dtype)
        one = mx.ones((b, p, 1), dtype=prompt.dtype)
        x = self.input_proj(mx.concatenate([mx.zeros_like(prompt), prompt, one], axis=-1))
        c = self.condition(mx.ones((b,)), dit_vector)
        cache: list[KV] = []
        for blk in self.blocks:
            x = blk(x, c, 0, blk.cross_attn.keys_values(dit_tokens), keep=cache, base=self.cfg.rope_base)
        return cache

    def velocity(self, x_t: mx.array, t: mx.array, frame_cond: mx.array, voice: VoiceState) -> mx.array:
        """Velocity for the new frames only. x_t (B, L, 96); t (B,); frame_cond (B, L, 512) text features of
        these frames; the prompt sits in front of them through ``voice.prompt_kv``. Returns (B, L, 96) in the
        model's dtype."""
        dt = self.dtype
        x_t = x_t.astype(dt)
        zero = mx.zeros_like(x_t)
        x = self.input_proj(mx.concatenate([x_t, zero, zero[..., :1]], axis=-1)) + self.text_proj(frame_cond.astype(dt))
        c = self.condition(t, voice.dit_vector)
        p = voice.prompt_frames
        for blk, cross_kv, prompt_kv in zip(self.blocks, voice.cross_kv, voice.prompt_kv):
            x = blk(x, c, p, cross_kv, prompt_kv, base=self.cfg.rope_base)
        return self.output_proj(AdaNorm.apply(x, self.final_norm.modulation(c)))


# --------------------------------------------------------------------------------------------- the whole model
class AcousticModel(nn.Module):
    """Text encoder, voice encoder, duration predictor and DiT; weights in ``model.safetensors``."""

    def __init__(self, cfg: ArticulinoConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg.text.vocab_size, cfg.text_encoder)
        self.voice_encoder = VoiceEncoder(cfg.voice_encoder)
        self.duration_predictor = DurationPredictor(cfg.duration_predictor)
        self.dit = DiT(cfg.dit)

    def prepare_voice(self, latents: mx.array, prompt_frames: int | None = None) -> VoiceState:
        """latents (1, R, 96): normalized reference latents (float32). Computes the voice tokens/vector and the
        voice cache, and evaluates them."""
        p = self.cfg.sampler.prompt_frames if prompt_frames is None else prompt_frames
        latents = latents.astype(mx.float32)
        tokens, vector = self.voice_encoder(latents)
        dit_tokens = self.dit.voice_token_proj(tokens.astype(self.dit.dtype))
        dit_vector = self.dit.voice_vector_proj(vector.astype(self.dit.dtype))
        prompt = latents[:, :p]
        state = VoiceState(
            latents=latents, tokens=tokens, vector=vector, prompt=prompt, dit_vector=dit_vector,
            cross_kv=[blk.cross_attn.keys_values(dit_tokens) for blk in self.dit.blocks],
            prompt_kv=self.dit.prompt_cache(prompt, dit_tokens, dit_vector),
        )
        mx.eval(state.arrays())
        return state
