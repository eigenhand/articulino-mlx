"""Listening check per precision: speak the four sample sentences of the model card with say() (seed 0) and let a
speech recognizer write down what it hears.

The recognizer is any OpenAI-style /v1/audio/transcriptions endpoint (we used Parakeet-TDT 0.6B v3); 0.4 s of
silence is put in front of every clip, otherwise it tends to swallow the first sound.
Besides the character error rate, the mel distance of every clip to the fp32 clip of the same sentence is reported.

  python tools/listen_check.py [--weights build/mlx] [--configs fp32,fp16:float16,q8:float16:deq]
      --asr http://127.0.0.1:8084 [--voice tests/data/radiov.wav] [--out build/listen] [--device cpu]

Comparison: lower case, letters and digits only, character error rate (the recognizer may write numbers as digits
where the frontend spelled them out; read the report). Writes the WAVs and a report.json to --out.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

SENTENCES = [  # (text, language) - the "Listen" table of the model card
    ("Guten Morgen! Dein Termin beim Zahnarzt ist heute um 15:30 Uhr in der Königstraße 12.", "de"),
    ("Die Stromrechnung über 87,40 € ist am 15.11. fällig; das PDF liegt in deinem Postfach.", "de"),
    ("Es gilt E = mc², und die Lichtgeschwindigkeit beträgt rund 300.000 km/s.", "de"),
    ("The delivery arrives tomorrow between ten and twelve.", "en"),
]
def transcribe(url: str, wav: np.ndarray, language: str, path: Path) -> str:
    import soundfile as sf

    padded = np.concatenate([np.zeros(int(0.4 * 24000), dtype=np.float32), wav])
    sf.write(path, padded, 24000, subtype="PCM_16")
    r = subprocess.run(["curl", "-s", f"{url.rstrip('/')}/v1/audio/transcriptions", "-F", f"file=@{path}",
                        "-F", f"language={language}"], capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout).get("text", "")


def _simple(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.lower())
    return re.sub(r"[^\w]", "", s)


def compare(heard: str, target_reading: str, language: str) -> float:
    """Character error rate between what was heard and what should have been said."""
    h, t = _simple(heard), _simple(target_reading)
    d = list(range(len(h) + 1))
    for i, c in enumerate(t, 1):
        p, d[0] = d[0], i
        for j, x in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (c != x))
    return d[-1] / max(1, len(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True, help="base URL of the speech recognizer")
    ap.add_argument("--weights", default=str(ROOT / "build" / "mlx"))
    ap.add_argument("--configs", default="fp32,fp16:float16,q8:float16:deq",
                    help="stage[:dtype[:deq]] per run; the half-precision variants are what a Mac GPU runs")
    ap.add_argument("--voice", default=str(ROOT / "tests" / "data" / "radiov.wav"))
    ap.add_argument("--out", default=str(ROOT / "build" / "listen"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from articulino_mlx import TTS
    from articulino_mlx.evaluation import mel_distance
    from articulino_mlx.frontend import normalize

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    fp32_audio: dict[int, np.ndarray] = {}
    for config in a.configs.split(","):
        parts = config.split(":")
        stage, dtype = parts[0], (parts[1] if len(parts) > 1 else None)
        deq = (parts[2] == "deq") if len(parts) > 2 else None
        tts = TTS.from_pretrained(Path(a.weights) / stage, device=a.device, dtype=dtype, dequantize=deq)
        tts.set_voice(a.voice)
        for i, (text, lang) in enumerate(SENTENCES):
            t0 = time.perf_counter()
            speech = tts.say(text, language=lang, seed=0)
            took = time.perf_counter() - t0
            if stage == "fp32":
                fp32_audio[i] = speech.audio
            reading = normalize(text) if lang == "de" else text
            name = config.replace(":", "-")
            heard = transcribe(a.asr, speech.audio, lang, out / f"{name}-{i}-{lang}.wav")
            cer = compare(heard, reading, lang)
            dist = mel_distance(speech.audio, fp32_audio[i]) if i in fp32_audio else None
            report.append({"config": config, "sentence": i, "text": text, "language": lang, "reading": reading,
                           "heard": heard, "cer": round(cer, 3), "seconds": round(speech.duration, 2),
                           "mel_distance_to_fp32_db": None if dist is None else round(dist, 3),
                           "compute_seconds": round(took, 2)})
            print(f"[{config} {i}] CER {cer:.3f} | {speech.duration:.2f} s | mel to fp32 "
                  f"{'-' if dist is None else f'{dist:.2f} dB'}\n    said:  {reading}\n    heard: {heard}", flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for config in a.configs.split(","):
        rows = [r for r in report if r["config"] == config]
        dists = [r["mel_distance_to_fp32_db"] for r in rows if r["mel_distance_to_fp32_db"] is not None]
        print(f"{config:<18} mean CER {np.mean([r['cer'] for r in rows]):.3f}"
              + (f" | mean mel distance to fp32 {np.mean(dists):.2f} dB" if dists else ""))
    print(f"WAVs and report in {out}")


if __name__ == "__main__":
    main()
