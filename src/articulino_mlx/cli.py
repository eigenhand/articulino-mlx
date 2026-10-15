"""Command line: ``articulino-mlx say "Text" --voice voice.wav -o out.wav``, ``articulino-mlx normalize "Text"`` and
``articulino-mlx convert <pytorch-release-folder> <out> --precision fp16``."""
from __future__ import annotations

import argparse
import sys
import time


def _text(arg: str) -> str:
    return sys.stdin.read() if arg == "-" else arg


def main(argv: list[str] | None = None) -> int:
    from . import DEFAULT_PRECISION, DEFAULT_REPO, __version__

    ap = argparse.ArgumentParser(prog="articulino-mlx",
                                 description="German and English text to speech with voice cloning, on MLX.")
    ap.add_argument("--version", action="version", version=f"articulino-mlx {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    say = sub.add_parser("say", help="speak a text into a WAV file")
    say.add_argument("text", help='the text, or "-" to read it from standard input')
    say.add_argument("--voice", required=True, help="reference recording of the voice (3 to 8 s of clean speech)")
    say.add_argument("-o", "--output", default="speech.wav", help="WAV file to write (default: speech.wav)")
    say.add_argument("--model", default=DEFAULT_REPO, help="local folder or Hugging Face repository")
    say.add_argument("--precision", choices=["fp32", "fp16", "q8"], default=None,
                     help=f"weights to use (default {DEFAULT_PRECISION}; ignored for a folder with config.json)")
    say.add_argument("--device", choices=["gpu", "cpu"], default=None, help="default: MLX's default device")
    say.add_argument("--dtype", choices=["float32", "float16"], default=None, help="dtype to compute in")
    say.add_argument("--speed", type=float, default=1.0, help="speaking rate, 0.7 to 1.6 (default 1.0)")
    say.add_argument("--seed", type=int, default=0, help="the same seed gives the same audio (default 0)")
    say.add_argument("--language", choices=["de", "en"], default="de")
    say.add_argument("--no-normalize", action="store_true", help="skip the German text frontend")

    norm = sub.add_parser("normalize", help="show what the German text frontend makes of a text")
    norm.add_argument("text", help='the text, or "-" to read it from standard input')

    conv = sub.add_parser("convert", help="convert the PyTorch release (safetensors) into MLX weights")
    conv.add_argument("source", help="folder with config.json, model.safetensors, codec.safetensors (PyTorch release)")
    conv.add_argument("output", help="output folder (one precision) or parent folder (with --all)")
    conv.add_argument("--precision", choices=["fp32", "fp16", "q8"], default="fp16")
    conv.add_argument("--all", action="store_true", help="write fp32/, fp16/ and q8/ below the output folder")
    conv.add_argument("--group-size", type=int, default=64)
    conv.add_argument("--no-codec-quantization", action="store_true", help="keep the codec in float16 for q8")
    conv.add_argument("--skip", nargs="*", default=[], help="module paths that stay unquantized")

    a = ap.parse_args(argv)
    if a.command == "normalize":
        from .frontend import normalize

        print(normalize(_text(a.text)))
        return 0

    if a.command == "convert":
        import json
        from pathlib import Path

        from .weights import convert

        stages = ["fp32", "fp16", "q8"] if a.all else [a.precision]
        for stage in stages:
            out = Path(a.output) / stage if a.all else Path(a.output)
            info = convert(a.source, out, stage, a.group_size, not a.no_codec_quantization, tuple(a.skip))
            size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e6
            print(f"{stage}: {out} ({size:.1f} MB) {json.dumps(info.get('quantization'))}", file=sys.stderr)
        return 0

    from . import TTS

    text = _text(a.text)
    tts = TTS.from_pretrained(a.model, a.precision, a.device, dtype=a.dtype)
    tts.set_voice(a.voice)
    start = time.perf_counter()
    speech = tts.say(text, path=a.output, speed=a.speed, seed=a.seed, language=a.language,
                     normalize=not a.no_normalize)
    took = time.perf_counter() - start
    speedup = speech.duration / max(took, 1e-9)
    print(f"{a.output}: {speech.duration:.2f} s of audio in {took:.2f} s ({speedup:.1f}x real time)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
