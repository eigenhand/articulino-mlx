"""Smoke tests of the public API on one stage (ARTICULINO_MLX_API_STAGE, default fp16): say(), stream(),
set_voice(), from_pretrained(), the CLI."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, WEIGHTS, device_name, stage_folder

pytestmark = pytest.mark.slow
STAGE = os.environ.get("ARTICULINO_MLX_API_STAGE", "fp16")


@pytest.fixture(scope="module")
def folder():
    f = stage_folder(STAGE)
    if f is None:
        pytest.skip(f"no MLX weights for {STAGE} in {WEIGHTS}")
    return f


@pytest.fixture(scope="module")
def speaker(folder, voice_path):
    from articulino_mlx import TTS

    tts = TTS.from_pretrained(folder, device=device_name())
    tts.set_voice(voice_path)
    return tts


def test_say(speaker, tmp_path):
    import soundfile as sf

    from articulino_mlx import Speech

    out = tmp_path / "morgen.wav"
    speech = speaker.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path=out)
    assert isinstance(speech, Speech)
    assert speech.sample_rate == 24000 and speech.seed == 0
    assert speech.audio.dtype == np.float32 and speech.audio.ndim == 1
    assert abs(speech.duration - len(speech.audio) / 24000) < 1e-9
    assert 1.5 < speech.duration < 6.0
    assert np.abs(speech.audio).max() <= 0.95 + 1e-6
    data, sr = sf.read(out, dtype="float32")
    assert sr == 24000 and np.abs(data - speech.audio).max() <= 1 / 32768


def test_seed_is_deterministic(speaker):
    a = speaker.say("Das ist ein Satz zum Vergleichen von zwei Läufen.", seed=3).audio
    b = speaker.say("Das ist ein Satz zum Vergleichen von zwei Läufen.", seed=3).audio
    c = speaker.say("Das ist ein Satz zum Vergleichen von zwei Läufen.", seed=4).audio
    assert np.array_equal(a, b)
    assert len(a) != len(c) or not np.allclose(a, c)


def test_speed(speaker):
    text = "Draußen sind es zwölf Grad, und am Abend soll es regnen."
    slow, fast = speaker.say(text, speed=0.8).duration, speaker.say(text, speed=1.4).duration
    assert slow > fast * 1.4
    with pytest.raises(ValueError):
        speaker.say(text, speed=0)


def test_stream_matches_say(speaker):
    """The card's example: stream() yields one chunk per sentence, the same audio as say() without the pauses."""
    pieces = ["Guten Mor", "gen! Heute ", "regnet es ab 18 Uhr. ", "Nimm einen Schirm mit."]
    chunks = list(speaker.stream(iter(pieces)))
    assert len(chunks) == 2 and all(c.dtype == np.float32 for c in chunks)
    whole = speaker.say("".join(pieces)).audio
    gap = np.zeros(int(0.15 * 24000), dtype=np.float32)
    assert np.array_equal(np.concatenate([chunks[0], gap, chunks[1]]), whole)
    assert len(list(speaker.stream("Ein ganzer Text als Zeichenkette, nicht als Liste."))) == 1


def test_language_and_tags(speaker):
    en = speaker.say("The delivery arrives tomorrow between ten and twelve.", language="en")
    assert en.duration > 1.5
    mixed = speaker.say("Das [en]Meeting[/en] ist auf morgen verschoben.")
    assert mixed.duration > 1.0
    with pytest.raises(ValueError):
        speaker.say("Bonjour", language="fr")


def test_options(folder, voice_path):
    from articulino_mlx import TTS

    plain = TTS.from_pretrained(folder, device=device_name(), uniform_rate=False, low_shelf_db=0, peak_limit=None,
                                watermark=lambda a, sr: a * 0.5)
    plain.set_voice(voice_path)
    speech = plain.say("Ein kurzer Satz ohne Nachbearbeitung, bitte.")
    assert speech.duration > 1.0 and np.abs(speech.audio).max() < 0.75


def test_set_voice_variants(speaker, voice_path):
    import soundfile as sf

    from articulino_mlx import Voice
    from articulino_mlx.audio import resample

    wav, sr = sf.read(voice_path, dtype="float32")
    v1 = speaker.set_voice(wav)
    stereo48 = np.stack([resample(wav, 24000, 48000)] * 2, 1)
    v2 = speaker.set_voice(stereo48, sample_rate=48000)
    assert isinstance(v2, Voice) and abs(v2.rate - v1.rate) / v1.rate < 0.05
    assert speaker.set_voice(v1) is v1
    with pytest.raises(ValueError):
        speaker.set_voice(np.zeros(1000, dtype=np.float32))


def test_precision_folders(tmp_path):
    """from_pretrained finds <folder>/<precision>, and a folder with config.json directly."""
    from articulino_mlx.tts import resolve_folder

    import json

    for stage in ("fp32", "fp16", "q8"):
        (tmp_path / stage).mkdir()
        (tmp_path / stage / "config.json").write_text(json.dumps({"mlx": {"precision": stage}}))
    assert resolve_folder(tmp_path) == tmp_path / "fp16"
    assert resolve_folder(tmp_path, "q8") == tmp_path / "q8"
    assert resolve_folder(tmp_path / "q8") == tmp_path / "q8"
    with pytest.raises(FileNotFoundError):
        resolve_folder(tmp_path / "missing")
    with pytest.raises(ValueError):
        resolve_folder(tmp_path / "fp32", "q8")


def test_cli(folder, voice_path, tmp_path):
    out = tmp_path / "cli.wav"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-m", "articulino_mlx", "say", "Hallo, das ist ein kurzer Test.", "--voice",
                        str(voice_path), "-o", str(out), "--model", str(folder), "--device", device_name()],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr
    assert out.is_file() and "real time" in r.stderr
    r = subprocess.run([sys.executable, "-m", "articulino_mlx", "normalize", "Am 15.11. um 15:30 Uhr."],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.strip() == "Am fünfzehnten November um fünfzehn Uhr dreißig."
