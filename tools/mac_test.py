#!/usr/bin/env python3
"""One-shot test of articulino-mlx on a Mac: install, check, measure, write WAVs, pack the results.

  python3 tools/mac_test.py                                  # weights from the Hugging Face repo (needs access)
  python3 tools/mac_test.py --weights ~/Downloads/mlx        # or a local folder with fp32/ fp16/ q8/

What it does (each step is logged; a failing step is recorded and the next one still runs):
  1. creates a virtual environment next to this folder (.venv-mac-test) and installs the package with
     mlx, numpy, soundfile, num2words, huggingface_hub and pytest (skip with --no-install);
  2. finds the weights (local folder, or downloads fp32/fp16/q8 from the repository);
  3. runs the test suite on the GPU: every building block against the reference vectors per precision, the
     public API, quantized kernels, the German frontend, and say() against the PyTorch package's output;
  4. measures the real-time factor per precision (10 sentences, voice tests/data/radiov.wav, best of 3) on the
     GPU, and fp32 on the CPU for comparison;
  5. writes WAVs per precision (the four sample sentences of the model card and the ten sentences) and the
     mel distance of every precision to fp32;
  6. packs everything into mac_test_results/<name>.zip – send that file back.

Needs Python 3.10 or newer (python.org, Homebrew or uv) on Apple silicon (M1 or newer).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_REPO = "eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX"
STAGES = ["fp32", "fp16", "q8"]
CARD_SENTENCES = [
    ("Guten Morgen! Dein Termin beim Zahnarzt ist heute um 15:30 Uhr in der Königstraße 12.", "de"),
    ("Die Stromrechnung über 87,40 € ist am 15.11. fällig; das PDF liegt in deinem Postfach.", "de"),
    ("Es gilt E = mc², und die Lichtgeschwindigkeit beträgt rund 300.000 km/s.", "de"),
    ("The delivery arrives tomorrow between ten and twelve.", "en"),
]


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# ----------------------------------------------------------------------------------------------- 1. install
def ensure_venv(args, log: Log) -> None:
    """Re-run this script inside a fresh virtual environment with the package installed."""
    if args.in_venv or args.no_install:
        return
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer is needed (this is {platform.python_version()}); "
                 "e.g. `brew install python@3.12` and run it with python3.12")
    venv = Path(args.venv).expanduser().resolve()
    python = venv / "bin" / "python"
    if not python.exists():
        log(f"creating virtual environment {venv}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    log("installing articulino-mlx with mlx, numpy, soundfile, num2words, huggingface_hub, pytest")
    subprocess.run([str(python), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-q", "-e", f"{ROOT}[hub,test]"], check=True)
    cmd = [str(python), str(Path(__file__).resolve()), "--in-venv"] + [a for a in sys.argv[1:] if a != "--in-venv"]
    log("continuing inside the virtual environment")
    raise SystemExit(subprocess.run(cmd).returncode)


# ----------------------------------------------------------------------------------------------- 2. weights
def find_weights(args, log: Log) -> Path:
    local = Path(args.weights).expanduser()
    if local.is_dir():
        stages = [s for s in STAGES if (local / s / "config.json").is_file()]
        if not stages:
            raise FileNotFoundError(f"{local} has none of the subfolders {STAGES}")
        log(f"weights: local folder {local} ({', '.join(stages)})")
        return local
    from huggingface_hub import snapshot_download

    log(f"downloading {args.weights} (fp32, fp16, q8) – a private repository needs `huggingface-cli login` "
        "or HF_TOKEN")
    path = snapshot_download(repo_id=args.weights, allow_patterns=[f"{s}/*" for s in STAGES],
                             token=os.environ.get("HF_TOKEN") or None)
    return Path(path)


# ----------------------------------------------------------------------------------------------- helpers
def gpu() -> str:
    """'gpu' on Apple silicon; 'cpu' where MLX has no GPU (lets the script run end to end elsewhere, e.g. Linux)."""
    import mlx.core as mx

    try:
        return "gpu" if mx.is_available(mx.gpu) else "cpu"
    except AttributeError:
        return "gpu" if mx.metal.is_available() else "cpu"


def environment() -> dict:
    import mlx.core as mx

    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "mlx": getattr(mx, "__version__", "?"), "default_device": str(mx.default_device())}
    for key, cmd in (("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]), ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
                     ("macos", ["sw_vers", "-productVersion"]), ("cores", ["sysctl", "-n", "hw.perflevel0.physicalcpu"])):
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 - informative only
            pass
    if gpu() == "gpu":
        try:
            device_info = mx.device_info if hasattr(mx, "device_info") else mx.metal.device_info
            info["gpu"] = {k: str(v) for k, v in device_info().items()}
        except Exception:  # noqa: BLE001 - informative only
            pass
    return info


def step(name: str, results: dict, log: Log):
    """Decorator-like helper: run fn, record ok/error and the time it took."""
    def run(fn, *a, **kw):
        log(f"--- {name}")
        t0 = time.perf_counter()
        try:
            value = fn(*a, **kw)
            results["steps"][name] = {"ok": True, "seconds": round(time.perf_counter() - t0, 1)}
            return value
        except BaseException as e:  # noqa: BLE001 - keep going, report everything
            if isinstance(e, KeyboardInterrupt):
                raise
            results["steps"][name] = {"ok": False, "seconds": round(time.perf_counter() - t0, 1),
                                      "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            log(f"FAILED: {type(e).__name__}: {e}")
            return None
    return run


# ----------------------------------------------------------------------------------------------- 3. tests
def run_tests(weights: Path, out: Path, device: str, log: Log, quick: bool) -> dict:
    report = out / f"equality_{device}.json"
    env = dict(os.environ, ARTICULINO_MLX_WEIGHTS=str(weights), ARTICULINO_MLX_REPORT=str(report),
               ARTICULINO_MLX_DEVICE=device, PYTHONPATH=str(ROOT / "src"))
    if device == "cpu":
        env["ARTICULINO_MLX_CONFIGS"] = "fp32"
    tests = ["tests/test_vectors.py"] if (quick or device == "cpu") else ["tests"]
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    log(" ".join(cmd) + f"  (device {device})")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    (out / f"pytest_{device}.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    tail = [line for line in r.stdout.splitlines() if line.strip()][-1:] or ["(no output)"]
    log(f"pytest ({device}): {tail[0]}")
    for line in r.stdout.splitlines():
        if line.startswith(("FAILED", "ERROR")):
            log("  " + line)
    return {"returncode": r.returncode, "summary": tail[0],
            "values": json.loads(report.read_text()) if report.is_file() else None}


# ----------------------------------------------------------------------------------------------- 4./5. speed, WAVs
def measure_and_write(weights: Path, out: Path, log: Log, repeat: int, quick: bool) -> dict:
    import numpy as np

    sys.path.insert(0, str(HERE))
    import rtf

    from articulino_mlx.evaluation import mel_distance

    sentences = [z.strip() for z in open(rtf.SENTENCES, encoding="utf-8") if z.strip()]
    g = gpu()
    runs = [(s, g) for s in STAGES] + [("fp32", "cpu")]
    if quick:
        runs = [("fp32", g), ("fp16", g)] + ([("fp32", "cpu")] if g != "cpu" else [])
    speed, card_fp32, results = {}, {}, {}
    for stage, device in runs:
        if not (weights / stage / "config.json").is_file():
            log(f"{stage}: no weights, skipped")
            continue
        key = f"{stage}@{device}"
        try:
            tts = rtf.load(weights, stage, device)
            tts.set_voice(rtf.VOICE)
            r = rtf.measure(tts, sentences, repeat if device == g else 1)
            r.update(compute_dtype=tts.info.get("compute_dtype"), quantized_kernels=tts.info.get("quantized_kernels"))
            speed[key] = r
            log(f"{key:<10} {r['rtf']:6.1f}x real time | first sentence {r['first_sentence_seconds']:.3f} s | "
                f"{r['compute_dtype']}, quantized kernels {r['quantized_kernels']}")
            if device != g and stage != "fp32":
                continue
            wav_dir = out / "wavs" / key.replace("@", "_")
            wav_dir.mkdir(parents=True, exist_ok=True)
            dist = []
            for i, (text, lang) in enumerate(CARD_SENTENCES, 1):
                speech = tts.say(text, language=lang, seed=0, path=wav_dir / f"card-{i}-{lang}.wav")
                if key == f"fp32@{g}":
                    card_fp32[i] = speech.audio
                elif i in card_fp32:
                    dist.append(mel_distance(speech.audio, card_fp32[i]))
            gap = np.zeros(int(0.15 * 24000), dtype=np.float32)
            parts = []
            for i, s in enumerate(sentences):
                parts += [tts.say(s, seed=42 + i).audio, gap]
            import soundfile as sf

            sf.write(wav_dir / "sentences10.wav", np.concatenate(parts), 24000, subtype="PCM_16")
            results[key] = {"mel_distance_to_fp32_gpu_db": round(float(np.mean(dist)), 3) if dist else None}
        except Exception as e:  # noqa: BLE001
            speed[key] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            log(f"{key}: FAILED {type(e).__name__}: {e}")
    return {"speed": speed, "wavs": results}


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default=None,
                    help="local folder with fp32/ fp16/ q8/, or a repository (default: ./weights if it exists, "
                         f"else {DEFAULT_REPO})")
    ap.add_argument("--venv", default=str(ROOT / ".venv-mac-test"))
    ap.add_argument("--out", default=str(ROOT / "mac_test_results"))
    ap.add_argument("--repeat", type=int, default=3, help="speed runs per precision (best is kept)")
    ap.add_argument("--quick", action="store_true", help="only fp16 on the GPU and fp32 on the CPU, fewer tests")
    ap.add_argument("--no-install", action="store_true", help="use the current Python environment as it is")
    ap.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.weights is None:
        bundled = ROOT / "weights"
        args.weights = str(bundled) if (bundled / "fp16" / "config.json").is_file() else DEFAULT_REPO

    out_root = Path(args.out).expanduser().resolve()
    out = out_root / f"{platform.node().split('.')[0] or 'mac'}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    if args.in_venv and os.environ.get("ARTICULINO_MAC_TEST_OUT"):
        out = Path(os.environ["ARTICULINO_MAC_TEST_OUT"])
    os.environ["ARTICULINO_MAC_TEST_OUT"] = str(out)
    name = out.name
    log = Log(out / "log.txt")
    ensure_venv(args, log)

    results: dict = {"name": name, "started": datetime.now().isoformat(timespec="seconds"), "steps": {},
                     "weights": args.weights}
    results["environment"] = step("environment", results, log)(environment)
    log(json.dumps(results["environment"]))
    weights = step("weights", results, log)(find_weights, args, log)
    if weights is not None:
        g = gpu()
        if g != "gpu":
            log("no MLX GPU here: the GPU runs use the CPU")
        results["tests_gpu"] = step("tests on the GPU", results, log)(run_tests, weights, out, g, log, args.quick)
        if g == "gpu":
            results["tests_cpu"] = step("tests on the CPU (fp32)", results, log)(run_tests, weights, out, "cpu", log,
                                                                                 args.quick)
        results.update(step("speed and WAVs", results, log)(measure_and_write, weights, out, log, args.repeat,
                                                            args.quick) or {})
    results["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "result.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = [f"# articulino-mlx Mac test {name}", "", "| step | ok | seconds |", "|---|---|---|"]
    lines += [f"| {k} | {'yes' if v['ok'] else 'NO: ' + v.get('error', '')} | {v['seconds']} |"
              for k, v in results["steps"].items()]
    if results.get("speed"):
        lines += ["", "| precision@device | x real time | first sentence (s) | dtype | quantized kernels |"
                  " mel distance to fp32 (dB) |", "|---|---|---|---|---|---|"]
        for key, r in results["speed"].items():
            if "rtf" in r:
                md = (results.get("wavs", {}).get(key) or {}).get("mel_distance_to_fp32_gpu_db")
                lines.append(f"| {key} | {r['rtf']:.1f} | {r['first_sentence_seconds']:.3f} | {r['compute_dtype']} | "
                             f"{r['quantized_kernels']} | {'-' if md is None else md} |")
            else:
                lines.append(f"| {key} | failed: {r.get('error')} | | | | |")
    for dev in ("gpu", "cpu"):
        t = results.get(f"tests_{dev}")
        if t:
            lines += ["", f"pytest {dev}: {t['summary']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))

    archive = out_root / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out_root))
    log(f"done. Please send back: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    if shutil.which("open"):
        subprocess.run(["open", "-R", str(archive)], check=False)


if __name__ == "__main__":
    main()
