"""The MLX port against outputs of the PyTorch package (tools/make_torch_refs.py, run once with torch).

Same noise for the same seed (torch's CPU generator reproduced with NumPy), the same resampling as torchaudio, and
say() with the release defaults (frontend, uniform rate, low-shelf, peak limit) giving the same audio.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import TORCH_REFS, load_tts, record

pytestmark = pytest.mark.skipif(not (TORCH_REFS / "say.npz").is_file(),
                                reason=f"no PyTorch outputs in {TORCH_REFS} (run tools/make_torch_refs.py)")


def refs(name: str) -> dict:
    return dict(np.load(TORCH_REFS / f"{name}.npz"))


def test_noise_matches_torch():
    from articulino_mlx.noise import draw_noise

    worst = 0.0
    for key, ref in refs("noise").items():
        seed, frames, step = (int(x) for x in key.replace("seed", "").replace("frames", "").replace("step", "")
                              .split("_"))
        mine = draw_noise(seed, frames, 125, 4, 96)[step]
        assert mine.shape == ref.shape
        worst = max(worst, float(np.abs(mine - ref).max()))
    record("numpy", "noise vs torch (max abs)", "all", worst)
    assert worst < 1e-5


def test_resample_matches_torchaudio():
    from articulino_mlx.audio import resample

    r = refs("resample")
    for key in [k for k in r if k.startswith("in_")]:
        sr = int(key[3:])
        mine = resample(r[key], sr, 24000)
        assert mine.shape == r[f"out_{sr}"].shape
        assert np.abs(mine - r[f"out_{sr}"]).max() < 1e-5, sr


@pytest.mark.slow
def test_say_matches_pytorch(voice_path):
    """fp32: say() gives the audio of the PyTorch package (same length, high SNR, same natural rate)."""
    import sys

    from articulino_mlx.evaluation import mel_distance, snr

    sys.path.insert(0, str(TORCH_REFS.parent.parent / "tools"))
    from make_torch_refs import SAY_CASES

    tts = load_tts("fp32", uniform_rate=True, low_shelf_db=2.0, peak_limit=0.95)
    voice = tts.set_voice(voice_path)
    r = refs("say")
    assert abs(voice.rate - float(r["natural_rate"])) / float(r["natural_rate"]) < 1e-4
    for i, (text, lang, seed, speed) in enumerate(SAY_CASES):
        mine = tts.say(text, language=lang, seed=seed, speed=speed).audio
        ref = r[f"case{i}"]
        assert mine.shape == ref.shape, (i, mine.shape, ref.shape)
        value = snr(mine, ref)
        record("fp32", "say() vs PyTorch SNR", f"case{i}", value)
        record("fp32", "say() vs PyTorch mel dB", f"case{i}", mel_distance(mine, ref))
        assert value > 30, (i, value)
