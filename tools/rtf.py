"""Real-time factor, measured like the PyTorch package's tools/rtf.py.

Ten sentences (tests/data/sentences10.txt), one voice (tests/data/radiov.wav), the voice cache built beforehand, one
warm-up sentence, then every sentence through the whole chain (text -> audio incl. post-processing), seed 42 + i.
Prints audio seconds per compute second and the time until the first sentence is ready; keeps the best of
``--repeat`` runs.

  python tools/rtf.py [--weights build/mlx] [--configs fp32,fp16,q8] [--device gpu|cpu] [--repeat 3]
      [--mode api|raw] [--json out.jsonl]

A config is stage[:dtype[:deq]], e.g. ``fp16:float16`` or ``q8:float32:deq``; without dtype, the defaults of
TTS.from_pretrained for the device apply (GPU: stored dtype, quantized kernels; CPU: float32, expanded layers).
--mode api: tts.say() with the defaults (German frontend, sentence splitting, uniform rate, low-shelf).
--mode raw: the same work as the reference script (no frontend, no uniform rate).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

SENTENCES = ROOT / "tests" / "data" / "sentences10.txt"
VOICE = ROOT / "tests" / "data" / "radiov.wav"


def parse_config(config: str) -> tuple[str, str | None, bool | None]:
    parts = config.split(":")
    return parts[0], (parts[1] or None) if len(parts) > 1 else None, (parts[2] == "deq") if len(parts) > 2 else None


def load(weights: Path, config: str, device: str | None, uniform_rate: bool = True):
    from articulino_mlx import TTS

    stage, dtype, deq = parse_config(config)
    folder = weights / stage if (weights / stage / "config.json").is_file() else weights
    return TTS.from_pretrained(folder, device=device, dtype=dtype, dequantize=deq, uniform_rate=uniform_rate)


def measure(tts, sentences: list[str], repeat: int = 3, mode: str = "api") -> dict:
    """Best of ``repeat`` runs over the sentences; returns a dict with rtf, first-sentence time and more."""
    from articulino_mlx.text import encode

    def speak(text: str, seed: int) -> np.ndarray:
        if mode == "api":
            return tts.say(text, seed=seed).audio
        ids, langs = encode(text)
        return tts._postprocess(*tts.synthesize(ids, langs, 1.0, seed))

    speak("Ein kurzer Satz zum Aufwärmen.", 0)
    best = None
    for _ in range(repeat):
        times, seconds = [], 0.0
        for i, s in enumerate(sentences):
            t0 = time.perf_counter()
            w = speak(s, 42 + i)
            times.append(time.perf_counter() - t0)
            seconds += len(w) / tts.sample_rate
        if best is None or sum(times) < sum(best[0]):
            best = (times, seconds)
    times, seconds = best
    return {"rtf": seconds / sum(times), "audio_seconds": seconds, "compute_seconds": sum(times),
            "first_sentence_seconds": times[0], "median_sentence_seconds": float(np.median(times)),
            "runs": repeat, "mode": mode}


def environment() -> dict:
    import mlx.core as mx

    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "mlx": getattr(mx, "__version__", "?"), "default_device": str(mx.default_device())}
    if sys.platform == "darwin":
        import subprocess

        for key, cmd in (("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                         ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
                         ("macos", ["sw_vers", "-productVersion"])):
            try:
                info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:  # noqa: BLE001 - informative only
                pass
    elif hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        info["cpus"] = f"{cpus[0]}-{cpus[-1]} ({len(cpus)})"
    if os.environ.get("LD_PRELOAD"):
        info["ld_preload"] = os.environ["LD_PRELOAD"]
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "build" / "mlx"))
    ap.add_argument("--configs", default="fp32,fp16,q8")
    ap.add_argument("--device", default=None, choices=[None, "gpu", "cpu"])
    ap.add_argument("--voice", default=str(VOICE))
    ap.add_argument("--sentences", default=str(SENTENCES))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--mode", choices=["api", "raw"], default="api")
    ap.add_argument("--json", default=None, help="append one JSON line per config to this file")
    a = ap.parse_args()
    sentences = [z.strip() for z in open(a.sentences, encoding="utf-8") if z.strip()]
    env = environment()
    print(json.dumps(env), flush=True)
    for config in a.configs.split(","):
        tts = load(Path(a.weights), config, a.device, uniform_rate=(a.mode == "api"))
        tts.set_voice(a.voice)
        r = measure(tts, sentences, a.repeat, a.mode)
        r.update(config=config, device=tts.info.get("device"), compute_dtype=tts.info.get("compute_dtype"),
                 quantized_kernels=tts.info.get("quantized_kernels"))
        print(f"{config:<16} {r['device']:<16} {r['compute_dtype']:<8} kernels={'q' if r['quantized_kernels'] else '-'}"
              f" | {r['audio_seconds']:.1f} s audio in {r['compute_seconds']:.2f} s = {r['rtf']:.1f}x real time"
              f" | first sentence {r['first_sentence_seconds']:.2f} s", flush=True)
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps({**r, "environment": env}) + "\n")


if __name__ == "__main__":
    main()
