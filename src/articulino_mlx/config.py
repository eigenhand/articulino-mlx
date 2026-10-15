"""Model configuration, read from the ``config.json`` that ships next to the weights."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields


def _build(cls, data: dict | None):
    """Create a dataclass from a dict, ignoring unknown keys (newer config files stay readable)."""
    data = data or {}
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(frozen=True)
class TextConfig:
    charset: str = " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789"
    pad_id: int = 0
    vocab_size: int = 53
    languages: dict = field(default_factory=lambda: {"de": 0, "en": 1})


@dataclass(frozen=True)
class TextEncoderConfig:
    dim: int = 512
    layers: int = 6
    heads: int = 8
    ff_mult: int = 4
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class VoiceEncoderConfig:
    input_dim: int = 96
    dim: int = 512
    layers: int = 4
    heads: int = 8
    num_tokens: int = 32
    out_dim: int = 768
    ff_mult: int = 4
    pool_layers: int = 2


@dataclass(frozen=True)
class DurationConfig:
    input_dim: int = 512
    voice_dim: int = 768
    channels: int = 384
    kernel_size: int = 3
    max_units: float = 400.0
    unit_ms: int = 10


@dataclass(frozen=True)
class DiTConfig:
    dim: int = 384
    layers: int = 8
    heads: int = 6
    ff_mult: int = 4
    latent_dim: int = 96
    text_dim: int = 512
    voice_dim: int = 768
    time_embedding_dim: int = 256
    time_scale: float = 1000.0
    rope_base: float = 10000.0


@dataclass(frozen=True)
class LatentConfig:
    dim: int = 96
    frame_rate: int = 25
    hop_length: int = 960
    mean: tuple = ()
    std: tuple = ()


@dataclass(frozen=True)
class SamplerConfig:
    timesteps: tuple = (0.0, 0.25, 0.5, 0.75)
    prompt_frames: int = 125
    decoder_context_frames: int = 12
    duration_units_per_frame: int = 4


@dataclass(frozen=True)
class PostprocessConfig:
    low_shelf_db: float = 2.0
    low_shelf_hz: float = 180.0
    shelf_slope: float = 0.7
    peak_limit: float = 0.95


@dataclass(frozen=True)
class SentenceConfig:
    max_chars: int = 180
    min_letters: int = 20


@dataclass(frozen=True)
class ArticulinoConfig:
    """Everything the package needs to know about a released model, as stored in ``config.json``."""

    sample_rate: int = 24000
    text: TextConfig = field(default_factory=TextConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    voice_encoder: VoiceEncoderConfig = field(default_factory=VoiceEncoderConfig)
    duration_predictor: DurationConfig = field(default_factory=DurationConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    latents: LatentConfig = field(default_factory=LatentConfig)
    codec: dict = field(default_factory=dict)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    voice_max_seconds: float = 8.0
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    sentences: SentenceConfig = field(default_factory=SentenceConfig)
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ArticulinoConfig":
        if d.get("model_type", "articulino") != "articulino":
            raise ValueError(f"not an Articulino config (model_type={d.get('model_type')!r})")
        lat = dict(d.get("latents", {}))
        lat["mean"] = tuple(lat.get("mean", ()))
        lat["std"] = tuple(lat.get("std", ()))
        smp = dict(d.get("sampler", {}))
        if "timesteps" in smp:
            smp["timesteps"] = tuple(smp["timesteps"])
        return cls(
            sample_rate=int(d.get("sample_rate", 24000)),
            text=_build(TextConfig, d.get("text")),
            text_encoder=_build(TextEncoderConfig, d.get("text_encoder")),
            voice_encoder=_build(VoiceEncoderConfig, d.get("voice_encoder")),
            duration_predictor=_build(DurationConfig, d.get("duration_predictor")),
            dit=_build(DiTConfig, d.get("dit")),
            latents=_build(LatentConfig, lat),
            codec=dict(d.get("codec", {})),
            sampler=_build(SamplerConfig, smp),
            voice_max_seconds=float(d.get("voice", {}).get("max_seconds", 8.0)),
            postprocess=_build(PostprocessConfig, d.get("postprocess")),
            sentences=_build(SentenceConfig, d.get("sentences")),
            raw=d,
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "ArticulinoConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
