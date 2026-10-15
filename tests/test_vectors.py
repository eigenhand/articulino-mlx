"""Every building block against the reference vectors computed with the original inference code (PyTorch, CPU, float32).

fp32 (docs/SPEC.md): max|diff| / max|ref| < 1e-3 per block, end-to-end audio SNR > 40 dB.
fp16, q8: the SNR of every block is measured and reported; the asserted minimums only catch breakage.

Each block gets the *reference* inputs, so errors do not carry over between blocks. The noise of the four sampling
steps comes from the vectors. The end-to-end case runs the public pipeline (voice wav -> encoder -> voice cache;
sentence -> durations -> sampler -> decoder) with only the noise taken from the vectors.
"""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from conftest import VECTORS, configs, load_tts, parse_config, record

TOL = 1e-3
MIN_AUDIO_SNR = 40.0
MIN_BLOCK_SNR = {"fp32": 60.0, "fp16": 30.0, "q8": 25.0}     # breakage guards, not quality targets
CASES = ["de_sentence", "en_sentence"]
BLOCKS = ["log-mel", "encoder mu", "normalized latents", "text encoder", "voice tokens", "voice vector",
          "duration (log)", "durations", "frame condition", "voice cache (k, v)", "DiT velocity step 0",
          "DiT velocity step 1", "DiT velocity step 2", "DiT velocity step 3", "4-step latents", "decoder (iSTFT)"]


def rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max() / np.abs(b).max())


def snr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    noise = ((a - b) ** 2).sum()
    return float("inf") if noise == 0 else float(10 * np.log10((b ** 2).sum() / noise))


def f32(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


_VECTORS: dict[str, tuple[dict, dict]] = {}


def vectors(case: str):
    if case not in _VECTORS:
        path = VECTORS / f"{case}.npz"
        if not path.is_file():
            pytest.skip(f"reference vectors {path} missing")
        _VECTORS[case] = (dict(np.load(path)), json.loads((VECTORS / f"{case}.json").read_text(encoding="utf-8")))
    return _VECTORS[case]


def a(v, key) -> mx.array:
    return mx.array(v[key])[None]


_OUTPUTS: dict[tuple[str, str], dict] = {}


def outputs(config: str, case: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All blocks of one stage configuration on one reference sentence: {block: (mine, reference)}."""
    key = (config, case)
    if key in _OUTPUTS:
        return _OUTPUTS[key]
    from articulino_mlx.model import VoiceState, expand_to_frames
    from articulino_mlx.sampler import sample

    tts = load_tts(config)
    v, info = vectors(case)
    m, c = tts.model, tts.codec
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with mx.stream(tts.device):
        out["log-mel"] = (np.array(c.encoder.mel(a(v, "voice_wav"))[0]).T, v["voice_logmel"])
        out["encoder mu"] = (np.array(c.encoder(a(v, "voice_wav"))[0]).T, v["voice_mu"])
        ref = (a(v, "voice_mu").transpose(0, 2, 1) - tts._mean) / tts._std
        out["normalized latents"] = (np.array(ref[0]), v["voice_ref"])
        lang = mx.array([tts.config.text.languages[info["language"]]])
        mask = a(v, "text_mask")
        h = m.text_encoder(a(v, "text_ids").astype(mx.int32), lang, mask)
        out["text encoder"] = (f32(h[0]), v["text_h"])
        tokens, vector = m.voice_encoder(a(v, "voice_ref"))
        out["voice tokens"] = (f32(tokens[0]), v["voice_tokens"])
        out["voice vector"] = (f32(vector[0]), v["voice_vector"])
        log_d = tts._durations(a(v, "text_h"), a(v, "voice_vector"), mask)       # what the pipeline uses
        out["duration (log)"] = (np.array(log_d[0]), v["duration_log"])
        d = np.minimum(np.exp(v["duration_log"]) * v["text_mask"], 400).astype(np.float32)[None]
        d[:, 0] = 4 * info["prompt_frames"]
        out["durations"] = (d[0], v["durations"])
        assert int(np.ceil(d.sum() / 4)) == info["frames_total"]
        fc = expand_to_frames(a(v, "text_h").astype(m.dit.dtype), v["durations"][None], info["frames_total"])
        out["frame condition"] = (f32(fc[0]), v["frame_cond"])
        p = info["prompt_frames"]
        state = m.prepare_voice(a(v, "voice_ref"), p)
        out["voice cache (k, v)"] = (np.stack([f32(x[0]) for kv in state.prompt_kv for x in kv]),
                                     np.stack([v[f"cache_{n}{i}"] for i in range(len(state.prompt_kv)) for n in "kv"]))
        # reference voice state (isolates the DiT): reference tokens, vector and cache
        dit, dt = m.dit, m.dit.dtype
        tok, vec = a(v, "voice_tokens"), a(v, "voice_vector")
        dit_tokens = dit.voice_token_proj(tok.astype(dt))
        ref_state = VoiceState(latents=a(v, "voice_ref"), tokens=tok, vector=vec, prompt=a(v, "voice_ref")[:, :p],
                               dit_vector=dit.voice_vector_proj(vec.astype(dt)),
                               cross_kv=[b.cross_attn.keys_values(dit_tokens) for b in dit.blocks],
                               prompt_kv=[(a(v, f"cache_k{i}").astype(dt), a(v, f"cache_v{i}").astype(dt))
                                          for i in range(len(dit.blocks))])
        cond = a(v, "frame_cond")[:, p:]
        for k, tk in enumerate(info["timesteps"]):
            vel = dit.velocity(a(v, f"step{k}_in"), mx.full((1,), float(tk)), cond, ref_state)
            out[f"DiT velocity step {k}"] = (f32(vel[0]), v[f"step{k}_velocity"])
        x1 = sample(dit, cond, ref_state, info["timesteps"], [v[f"noise_{k}"][None] for k in range(4)])
        out["4-step latents"] = (np.array(x1[0]), v["step3_x1"])
        out["decoder (iSTFT)"] = (np.array(c.decoder(a(v, "decoder_in").transpose(0, 2, 1))[0]), v["decoder_out"])
    _OUTPUTS[key] = out
    return out


PARAMS = [(cfg, case) for cfg in configs() for case in CASES]


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("config,case", PARAMS, ids=[f"{c}-{s}" for c, s in PARAMS])
def test_block(config, case, block):
    mine, ref = outputs(config, case)[block]
    stage = parse_config(config)[0]
    if stage == "fp32":
        value = rel(mine, ref)
        record(config, block, case, value)
        assert value < TOL, f"{block}: {value:.3g} >= {TOL}"
    else:
        value = snr(mine, ref)
        record(config, block, case, value)
        assert value > MIN_BLOCK_SNR[stage], f"{block}: SNR {value:.1f} dB"


def test_text_ids():
    from articulino_mlx.text import encode

    for case in CASES:
        v, info = vectors(case)
        ids, _ = encode(info["sentence"], info["language"])
        assert ids == v["text_ids_raw"].tolist()
        assert v["text_ids"].tolist() == [0] + ids and v["text_mask"].tolist() == [False] + [True] * len(ids)


@pytest.mark.parametrize("config,case", PARAMS, ids=[f"{c}-{s}" for c, s in PARAMS])
def test_end_to_end(config, case):
    """Voice wav + sentence + given noise -> audio, through the public pipeline (no uniform rate, no
    post-processing). fp32 must reach 40 dB SNR. For every stage, the mel distance to the reference audio is
    reported too (it ignores phase); if a stage's durations give a different number of frames, the SNR is not
    defined and the frame difference is reported instead (the noise is then cut or extended)."""
    from articulino_mlx.evaluation import mel_distance

    tts = load_tts(config)
    v, info = vectors(case)
    tts.set_voice(v["voice_wav"])
    lang = tts.config.text.languages[info["language"]]
    frames = tts.plan(v["text_ids_raw"].tolist(), lang)[2]                          # own durations
    new = frames - info["prompt_frames"]
    noise = []
    for k in range(4):
        n = v[f"noise_{k}"]
        if new > len(n):
            n = np.concatenate([n, np.random.default_rng(k).standard_normal((new - len(n), n.shape[1]))])
        noise.append(n[:new].astype(np.float32)[None])
    raw, skip = tts.synthesize(v["text_ids_raw"].tolist(), lang, noise=noise)
    audio = raw[skip:]
    record(config, "audio mel distance (dB)", case, mel_distance(audio, v["audio"]))
    if frames != info["frames_total"]:
        record(config, "audio SNR (dB)", case, f"{frames - info['frames_total']:+d} frames")
        assert parse_config(config)[0] != "fp32", "fp32 must give the reference number of frames"
        return
    value = snr(audio, v["audio"])
    record(config, "audio SNR (dB)", case, value)
    if parse_config(config)[0] == "fp32":
        assert value > MIN_AUDIO_SNR
