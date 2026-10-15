"""Outputs of the PyTorch package for the parity tests of the MLX port (run with torch, not needed at runtime).

Writes build/torch_refs/{noise,resample,say}.npz:
  noise:    torch.Generator(seed) + torch.randn draws as the PyTorch sampler makes them (several seeds and lengths)
  resample: articulino.audio.resample (= torchaudio) of a test signal for common sample rates
  say:      say() with the release defaults (German frontend, uniform rate, low-shelf, peak limit) for a few texts,
            seeds and speeds, plus the voice's natural rate

  python tools/make_torch_refs.py [--model eigenhand/Articulino-DE-EN-Realtime-Stream-TTS] [--package <articulino/src>]
      [--voice tests/data/radiov.wav] [--out build/torch_refs]

Needs the PyTorch package (installed, or its src folder via --package). The outputs shipped in tests/torch_refs were
made this way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SAY_CASES = [  # (text, language, seed, speed)
    ("Guten Morgen! Dein Termin beim Zahnarzt ist heute um 15:30 Uhr in der Königstraße 12.", "de", 0, 1.0),
    ("Die Stromrechnung über 87,40 € ist am 15.11. fällig; das PDF liegt in deinem Postfach.", "de", 3, 1.2),
    ("The delivery arrives tomorrow between ten and twelve.", "en", 7, 0.9),
    ("Das [en]Meeting[/en] ist auf morgen verschoben.", "de", 42, 1.0),
]
NOISE_CASES = [(0, 140), (1, 200), (42, 333), (12345, 126)]          # (seed, total frames)
RESAMPLE_RATES = [48000, 44100, 22050, 16000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=None, help="src folder of the PyTorch package (default: the installed one)")
    ap.add_argument("--model", default="eigenhand/Articulino-DE-EN-Realtime-Stream-TTS")
    ap.add_argument("--voice", default=str(ROOT / "tests" / "data" / "radiov.wav"))
    ap.add_argument("--out", default=str(ROOT / "build" / "torch_refs"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.package:
        sys.path.insert(0, a.package)
    import torch

    from articulino import TTS
    from articulino.audio import resample
    from articulino.sampler import draw_noise

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    noise = {}
    for seed, frames in NOISE_CASES:
        for k, n in enumerate(draw_noise(seed, frames, 125, 4, 96)):
            noise[f"seed{seed}_frames{frames}_step{k}"] = n.numpy()
    np.savez_compressed(out / "noise.npz", **noise)

    t = np.arange(48000 * 2) / 48000
    rng = np.random.default_rng(0)
    signal = {sr: (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(1.5 * sr)) / sr)
                   + 0.05 * rng.standard_normal(int(1.5 * sr))).astype(np.float32) for sr in RESAMPLE_RATES}
    np.savez_compressed(out / "resample.npz", **{f"in_{sr}": x for sr, x in signal.items()},
                        **{f"out_{sr}": resample(x, sr, 24000) for sr, x in signal.items()})
    del t

    tts = TTS.from_pretrained(a.model, device="cpu")
    voice = tts.set_voice(a.voice)
    say = {"natural_rate": np.array(voice.rate)}
    for i, (text, lang, seed, speed) in enumerate(SAY_CASES):
        say[f"case{i}"] = tts.say(text, language=lang, seed=seed, speed=speed).audio
    np.savez_compressed(out / "say.npz", **say)
    print(f"wrote {out} (torch {torch.__version__})")


if __name__ == "__main__":
    main()
