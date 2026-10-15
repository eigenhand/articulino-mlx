"""The high-level API: :class:`TTS` and :class:`Speech` (same API as the PyTorch package ``articulino``)."""
from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import frontend
from .audio import load_audio, resample
from .codec import Codec
from .config import ArticulinoConfig
from .model import AcousticModel, PreciseDurations, VoiceState, expand_to_frames
from .noise import draw_noise
from .postprocess import SosFilter, limit_peak, shelf_sos
from .sampler import letter_rate, sample
from .sentences import SentenceBuffer, carry_language_tags, split_sentences
from .text import encode, open_language_tag, protect_foreign_spans, restore_foreign_spans
from .weights import DTYPES, PRECISIONS, WEIGHT_FILES, info_dtype, is_quantized, load_modules

DEFAULT_REPO = "eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX"
DEFAULT_PRECISION = "fp16"
SENTENCE_PAUSE = 0.15               # seconds of silence between sentences in say()
MIN_SPEED, MAX_SPEED = 0.7, 1.6
MIN_VOICE_SECONDS = 1.0

# Neutral German sentences used once per voice to measure the voice's natural speaking rate (uniform_rate=True).
RATE_CALIBRATION = (
    "Guten Morgen, die Systeme laufen stabil und es gibt nichts Dringendes.",
    "Ich habe die Unterlagen geprüft und fasse das Wichtigste kurz zusammen.",
    "Der Termin am Donnerstag wurde auf den Nachmittag verschoben.",
    "Soll ich die Nachricht jetzt abschicken oder noch einmal vorlesen?",
    "Das Training läuft seit gestern Abend ohne Unterbrechung weiter.",
    "Draußen sind es zwölf Grad, und am Abend soll es regnen.",
)

Watermark = Callable[[np.ndarray, int], np.ndarray]


def _normalize_german(text: str) -> str:
    """The German frontend, except for spans tagged [en]…[/en]: those keep their spelling (only the model's own
    character normalization applies), so English words are not read with German letter names or numbers."""
    masked, spans = protect_foreign_spans(text, "de")
    if not spans:
        return frontend.normalize(text)
    restored = restore_foreign_spans(frontend.normalize(masked), spans)
    return frontend.normalize(text) if restored is None else restored


class _TaggedStreamNormalizer(frontend.StreamNormalizer):
    """StreamNormalizer with [en]…[/en] protection; a tag left open in one released chunk continues in the next."""

    def __init__(self) -> None:
        super().__init__(self._normalize_chunk)
        self._open: str | None = None

    def _normalize_chunk(self, chunk: str) -> str:
        prefix = self._open or ""
        out = _normalize_german(prefix + chunk)
        self._open = open_language_tag(prefix + chunk)
        return out[len(prefix):] if prefix and out.startswith(prefix) else out


class _FromConfig:
    """Default marker: take the value from config.json."""

    def __repr__(self) -> str:
        return "FROM_CONFIG"


FROM_CONFIG = _FromConfig()


@dataclass(frozen=True)
class Speech:
    """Generated audio: float32 mono samples in [-1, 1], their sample rate, length in seconds and the seed."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    seed: int

    def save(self, path: str | os.PathLike, subtype: str = "PCM_16") -> None:
        """Write the audio to a file (format from the extension, e.g. .wav or .flac; 16-bit PCM by default)."""
        import soundfile as sf

        sf.write(str(path), self.audio, self.sample_rate, subtype=subtype)


@dataclass
class Voice:
    """A prepared reference voice (returned by :meth:`TTS.set_voice`, can be passed back to it later)."""

    state: VoiceState
    rate: float | None          # natural speaking rate in letters per second (None if uniform_rate is off)
    seconds: float              # length of the reference that was used


def _device(device: str | mx.Device | None) -> mx.Device:
    if device is None:
        return mx.default_device()              # the GPU on Apple silicon, the CPU elsewhere
    if isinstance(device, mx.Device):
        return device
    name = str(device).lower()
    if name in ("gpu", "metal", "mps"):
        try:
            available = mx.is_available(mx.gpu)
        except AttributeError:                      # older MLX
            available = mx.metal.is_available()
        if not available:
            raise RuntimeError("MLX has no GPU on this machine; use device='cpu'")
        return mx.Device(mx.gpu)
    if name == "cpu":
        return mx.Device(mx.cpu)
    raise ValueError(f"device must be 'gpu' or 'cpu', not {device!r}")


def _is_cpu(device: mx.Device) -> bool:
    return device.type == mx.cpu


def resolve_folder(repo_or_dir: str | os.PathLike, precision: str | None = None, *, revision: str | None = None,
                   cache_dir: str | os.PathLike | None = None, token: str | bool | None = None) -> Path:
    """The folder holding config.json + weights: a model folder itself, ``<folder>/<precision>``, or the
    ``<precision>`` folder of a Hugging Face repository (downloaded with huggingface_hub)."""
    folder = Path(repo_or_dir).expanduser()
    name = str(repo_or_dir)
    local = isinstance(repo_or_dir, os.PathLike) or name.startswith(("/", ".", "~")) or name.count("/") != 1
    if folder.is_dir():
        wanted = precision or DEFAULT_PRECISION
        if (folder / "config.json").is_file():
            with open(folder / "config.json", encoding="utf-8") as f:
                stored = (json.load(f).get("mlx") or {}).get("precision")
            if precision is None or stored is None or stored == precision:
                return folder
            raise ValueError(f"{folder} holds {stored} weights, not {precision}")
        if (folder / wanted / "config.json").is_file():
            return folder / wanted
        raise FileNotFoundError(f"{folder} has no config.json and no {wanted}/ subfolder")
    if local:
        raise FileNotFoundError(f"model folder {folder} not found")
    precision = precision or DEFAULT_PRECISION
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(f"{repo_or_dir!r} is not a local folder; to download it, install huggingface_hub "
                          "(pip install huggingface_hub)") from None
    root = Path(snapshot_download(repo_id=str(repo_or_dir), revision=revision, cache_dir=cache_dir, token=token,
                                  allow_patterns=[f"{precision}/{f}" for f in WEIGHT_FILES]))
    return root / precision


class TTS:
    """German and English text to speech with zero-shot voice cloning, on MLX.

    Create it with :meth:`from_pretrained`, choose a voice with :meth:`set_voice`, then call :meth:`say` for whole
    texts or :meth:`stream` for text that arrives piece by piece.
    """

    def __init__(self, model: AcousticModel, codec: Codec, config: ArticulinoConfig,
                 device: str | mx.Device | None = None, *, uniform_rate: bool = True,
                 low_shelf_db: float | None = FROM_CONFIG, peak_limit: float | None = FROM_CONFIG,
                 watermark: Watermark | None = None, info: dict | None = None):
        """Prefer :meth:`from_pretrained`. Options:

        uniform_rate: bring every sentence to the voice's natural speaking rate (× ``speed``), measured once per voice.
            Keeps the tempo steady from sentence to sentence. With False, the duration predictor's rate is used as is.
        low_shelf_db: gain of the low-shelf at 180 Hz applied after the decoder; default from the config (+2 dB);
            0 or None switches it off.
        peak_limit: scale a sentence down if its peak is above this; default from the config (0.95); None switches
            it off.
        watermark: reserved hook, ``f(audio, sample_rate) -> audio``, applied to every chunk after post-processing.
            Articulino ships no watermark; see the model card on marking synthetic audio.
        """
        self.config = config
        self.device = _device(device)
        self.model = model
        self.codec = codec
        self.info = dict(info or {})
        self.sample_rate = config.sample_rate
        self.uniform_rate = uniform_rate
        pp = config.postprocess
        self.low_shelf_db = float(pp.low_shelf_db if low_shelf_db is FROM_CONFIG else (low_shelf_db or 0.0))
        self.peak_limit = pp.peak_limit if peak_limit is FROM_CONFIG else peak_limit
        self.watermark = watermark
        self._shelf = (SosFilter(shelf_sos(pp.low_shelf_hz, self.low_shelf_db, False, self.sample_rate,
                                           pp.shelf_slope)) if self.low_shelf_db else None)
        self._mean = mx.array(np.asarray(config.latents.mean, dtype=np.float32))
        self._std = mx.array(np.asarray(config.latents.std, dtype=np.float32))
        self._durations = PreciseDurations(model.duration_predictor, config.duration_predictor)
        self.voice: Voice | None = None

    # ------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, repo_or_dir: str | os.PathLike = DEFAULT_REPO, precision: str | None = None,
                        device: str | mx.Device | None = None, *, dtype: str | None = None,
                        dequantize: bool | None = None, revision: str | None = None,
                        cache_dir: str | os.PathLike | None = None, token: str | bool | None = None,
                        **options) -> "TTS":
        """Load the model from a local folder or a Hugging Face repository.

        precision: ``"fp32"``, ``"fp16"`` or ``"q8"`` – the subfolder of the repository (default fp16).
            A folder that holds config.json itself is loaded as it is (an MLX folder or the PyTorch release).
        device: ``"gpu"`` or ``"cpu"``; None uses MLX's default device (the GPU on Apple silicon).
        dtype: the dtype to compute in, ``"float32"`` or ``"float16"``. Default: the stored dtype on the GPU, float32
            on the CPU (the MLX CPU backend has no fast half-precision kernels).
        dequantize: expand quantized layers to plain layers at load time. Default: on the CPU only (no fast
            quantized kernels there); the download stays small, the quantization error stays the same.
        revision, cache_dir, token: passed to ``huggingface_hub.snapshot_download`` (only for repositories).
        Further keyword options go to :class:`TTS` (``uniform_rate``, ``low_shelf_db``, ``peak_limit``,
        ``watermark``)."""
        if precision is not None and precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, not {precision!r}")
        folder = resolve_folder(repo_or_dir, precision, revision=revision, cache_dir=cache_dir, token=token)
        missing = [f for f in WEIGHT_FILES if not (folder / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} lacks {', '.join(missing)}")
        dev = _device(device)
        if dtype is None:
            target = mx.float32 if _is_cpu(dev) else None
        else:
            if dtype not in DTYPES:
                raise ValueError(f"dtype must be one of {sorted(DTYPES)}, not {dtype!r}")
            target = DTYPES[dtype]
        if dequantize is None:
            dequantize = _is_cpu(dev)
        model, codec, config, info = load_modules(folder, target, dequantize)
        info = dict(info)
        info.update(folder=str(folder), compute_dtype=str(target or info_dtype(info)).replace("mlx.core.", ""),
                    quantized_kernels=is_quantized(model) or is_quantized(codec), device=str(dev))
        return cls(model, codec, config, dev, info=info, **options)

    # ------------------------------------------------------------------------------------------- voice
    def set_voice(self, voice: str | os.PathLike | np.ndarray | Voice, sample_rate: int | None = None) -> Voice:
        """Use this voice from now on.

        voice: path of an audio file, a mono array of samples (``sample_rate``, default 24000), or a :class:`Voice`
        returned earlier. 3 to 8 seconds of clean speech work best; longer recordings are cut to 8 seconds, other
        sample rates are resampled to 24 kHz. Only clone voices of people who agreed to it."""
        if isinstance(voice, Voice):
            self.voice = voice
            return voice
        if isinstance(voice, (str, os.PathLike)):
            wav, sr = load_audio(voice)
        else:
            wav = np.asarray(voice).astype(np.float32)
            if wav.ndim == 2:
                wav = wav.mean(0 if wav.shape[0] < wav.shape[1] else 1)
            sr = self.sample_rate if sample_rate is None else int(sample_rate)
        if wav.ndim != 1:
            raise ValueError("the voice must be mono audio")
        if sr != self.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
        hop = self.config.latents.hop_length
        n = min(len(wav), int(self.config.voice_max_seconds * self.sample_rate)) // hop * hop
        seconds = n / self.sample_rate
        if seconds < MIN_VOICE_SECONDS:
            raise ValueError(f"the voice recording is too short ({len(wav) / self.sample_rate:.2f} s); use 3 to 8 s")
        if seconds < 3.0:
            warnings.warn(f"voice recording of {seconds:.1f} s; 3 to 8 s of clean speech work best", stacklevel=2)
        if not np.isfinite(wav[:n]).all():
            raise ValueError("the voice recording contains NaN or infinite samples")
        with mx.stream(self.device):
            x = mx.array(np.ascontiguousarray(wav[:n]))[None]
            latents = (self.codec.encoder(x) - self._mean) / self._std           # (1, R, 96) float32
            state = self.model.prepare_voice(latents)
            rate = self._natural_rate(state) if self.uniform_rate else None
        self.voice = Voice(state=state, rate=rate, seconds=seconds)
        return self.voice

    def _natural_rate(self, state: VoiceState) -> float:
        """Median letters per second the duration predictor gives this voice on neutral sentences."""
        rates: list[float] = []
        lang = mx.zeros((1,), dtype=mx.int32)
        for sentence in RATE_CALIBRATION:
            ids, _ = encode(sentence)
            t = mx.array([ids], dtype=mx.int32)
            mask = mx.ones(t.shape, dtype=mx.bool_)
            h = self.model.text_encoder(t, lang, mask)
            d = np.exp(np.array(self._durations(h, state.vector, mask)))
            rates += letter_rate(d, np.array([ids]))[0]
        return float(np.median(rates))

    # ------------------------------------------------------------------------------------------- speaking
    def say(self, text: str, path: str | os.PathLike | None = None, speed: float = 1.0, seed: int = 0,
            language: str = "de", normalize: bool = True) -> Speech:
        """Speak a whole text.

        The text goes through the German text frontend (numbers, dates, units, abbreviations, …; only for
        ``language="de"`` and with ``normalize=True``), is split into sentences, and every sentence is generated on
        its own with the same ``seed``; sentences are joined with 0.15 s of silence. ``speed`` (0.7 to 1.6) scales
        the speaking rate. With ``path``, the audio is also written as a WAV file."""
        speed, seed = self._check(speed, seed, language)
        if normalize and language == "de":
            text = _normalize_german(text).strip()
        pieces = carry_language_tags(split_sentences(text, True, self.config.sentences.max_chars,
                                                     self.config.sentences.min_letters)[0])
        gap = np.zeros(int(round(SENTENCE_PAUSE * self.sample_rate)), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for piece in pieces:
            audio = self._speak(piece, speed, seed, language)
            if audio.size:
                if chunks:
                    chunks.append(gap)
                chunks.append(audio)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        speech = Speech(audio=audio, sample_rate=self.sample_rate, duration=len(audio) / self.sample_rate, seed=seed)
        if path is not None:
            speech.save(path)
        return speech

    def stream(self, text: str | Iterable[str], speed: float = 1.0, seed: int = 0, language: str = "de",
               normalize: bool = True) -> Iterator[np.ndarray]:
        """Speak text while it is still arriving; yields one float32 chunk per sentence.

        ``text`` is a string or any iterable of text pieces (e.g. tokens from a language model). Incoming text is
        collected until a sentence is complete, then that sentence is spoken at once. Sentences with fewer than 20
        letters wait for the next one; very long ones are split at commas."""
        speed, seed = self._check(speed, seed, language)          # fail here, not at the first next()
        return self._stream([text] if isinstance(text, str) else text, speed, seed, language, normalize)

    def _stream(self, pieces: Iterable[str], speed: float, seed: int, language: str,
                normalize: bool) -> Iterator[np.ndarray]:
        normalizer = _TaggedStreamNormalizer() if normalize and language == "de" else None
        buffer = SentenceBuffer(self.config.sentences.max_chars, self.config.sentences.min_letters)
        for piece in pieces:
            if normalizer is not None:
                piece = normalizer.push(piece)
            if piece:
                for sentence in buffer.push(piece):
                    audio = self._speak(sentence, speed, seed, language)
                    if audio.size:
                        yield audio
        tail = normalizer.flush() if normalizer is not None else ""
        for sentence in buffer.flush(tail):
            audio = self._speak(sentence, speed, seed, language)
            if audio.size:
                yield audio

    def _check(self, speed: float, seed: int, language: str) -> tuple[float, int]:
        if self.voice is None:
            raise RuntimeError("no voice set: call tts.set_voice('voice.wav') first")
        if language not in self.config.text.languages:
            raise ValueError(f"language must be one of {sorted(self.config.text.languages)}, not {language!r}")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a positive number")
        return min(MAX_SPEED, max(MIN_SPEED, speed)), seed

    def _speak(self, sentence: str, speed: float, seed: int, language: str) -> np.ndarray:
        ids, langs = encode(sentence, language)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        audio = self._postprocess(*self.synthesize(ids, langs, speed, seed))
        if self.watermark is not None:
            audio = np.asarray(self.watermark(audio, self.sample_rate), dtype=np.float32)
        return audio

    def plan(self, ids: list[int], languages: list[int] | int = 0, speed: float = 1.0
             ) -> tuple[mx.array, np.ndarray, int]:
        """Text features and durations of one sentence: (h (1, N, 512) lazily, durations (1, N) float32 in 10 ms
        units with the prompt placeholder first, total latent frames incl. the prompt)."""
        if self.voice is None:
            raise RuntimeError("no voice set: call tts.set_voice('voice.wav') first")
        cfg, voice = self.config, self.voice
        state = voice.state
        units = cfg.sampler.duration_units_per_frame
        ids_np = np.array([[cfg.text.pad_id] + list(ids)], dtype=np.int32)
        mask_np = np.array([[False] + [True] * len(ids)])
        if isinstance(languages, int) or len(set(languages)) == 1:
            lang_np = np.array([languages if isinstance(languages, int) else languages[0]], dtype=np.int32)
        else:
            lang_np = np.array([[languages[0]] + list(languages)], dtype=np.int32)
        with mx.stream(self.device):
            mask = mx.array(mask_np)
            h = self.model.text_encoder(mx.array(ids_np), mx.array(lang_np), mask)
            d = np.exp(np.array(self._durations(h, state.vector, mask))) * mask_np
        d = np.minimum(d, np.float32(cfg.duration_predictor.max_units)).astype(np.float32) / np.float32(speed)
        if self.uniform_rate:
            if voice.rate is None:                  # voice prepared by an instance without uniform_rate
                with mx.stream(self.device):
                    voice.rate = self._natural_rate(state)
            d = letter_rate(d, ids_np, start=1, target=voice.rate * speed)[1]
        d[:, 0] = units * state.prompt_frames
        frames = max(1, int(math.ceil(float(d.sum(1, dtype=np.float32)[0]) / units)))
        return h, d, frames

    def synthesize(self, ids: list[int], languages: list[int] | int = 0, speed: float = 1.0, seed: int = 0,
                   noise: list[np.ndarray] | None = None) -> tuple[np.ndarray, int]:
        """Generate one sentence from character ids (no frontend, no splitting, no post-processing).

        languages: one id for the whole sentence or one per character. noise: one array (1, new frames, 96) per
        sampling step instead of drawing it from ``seed`` (for tests). Returns (raw decoder audio including the
        context frames in front, number of context samples to drop)."""
        h, d, frames = self.plan(ids, languages, speed)
        cfg, state = self.config, self.voice.state
        p = state.prompt_frames
        if frames <= p:
            return np.zeros(0, dtype=np.float32), 0
        with mx.stream(self.device):
            frame_cond = expand_to_frames(h, d, frames, p, cfg.sampler.duration_units_per_frame)
            if noise is None:
                noise = draw_noise(seed, frames, p, len(cfg.sampler.timesteps), cfg.latents.dim)
            x1 = sample(self.model.dit, frame_cond, state, cfg.sampler.timesteps, noise)
            ctx = min(p, cfg.sampler.decoder_context_frames)
            z = mx.concatenate([state.prompt[:, p - ctx:], x1], axis=1)
            wav = np.array(self.codec.decoder(z * self._std + self._mean)[0])
        return wav, ctx * cfg.latents.hop_length

    def _postprocess(self, wav: np.ndarray, skip: int) -> np.ndarray:
        """Low-shelf over the whole decoder output (so the filter has settled), drop the context, limit the peak."""
        if self._shelf is not None and len(wav) > skip:
            wav = self._shelf(wav)
        wav = np.asarray(wav[skip:], dtype=np.float32)
        return limit_peak(wav, self.peak_limit).astype(np.float32, copy=False)
