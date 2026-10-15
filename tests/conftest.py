"""Shared fixtures.

Environment variables:
  ARTICULINO_MLX_WEIGHTS  folder with fp32/, fp16/, q8/ (MLX weights; default: build/mlx); a folder with
                          config.json itself is used as the only stage
  ARTICULINO_VECTORS      folder with the reference vectors (default: tests/vectors)
  ARTICULINO_MLX_DEVICE   gpu or cpu (default: MLX's default device; the GPU on Apple silicon)
  ARTICULINO_MLX_CONFIGS  comma-separated stage[:dtype[:deq]] to test, e.g. "fp32,fp16:float16,q8:float16:deq"
                          (default: every stage with the defaults of TTS.from_pretrained on that device, and on a
                          CPU additionally the half-precision variants a Mac GPU runs by default)
  ARTICULINO_MLX_REPORT   write the measured values as JSON to this file
  ARTICULINO_TORCH_REFS   outputs of the PyTorch package (tools/make_torch_refs.py; default: build/torch_refs,
                          else tests/torch_refs)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

WEIGHTS = Path(os.environ.get("ARTICULINO_MLX_WEIGHTS", ROOT / "build" / "mlx"))
_VECTOR_DIRS = [HERE / "vectors"]
VECTORS = Path(os.environ.get("ARTICULINO_VECTORS") or next((d for d in _VECTOR_DIRS if d.is_dir()), _VECTOR_DIRS[0]))
_TORCH_REF_DIRS = [ROOT / "build" / "torch_refs", HERE / "torch_refs"]    # freshly made, else the copy in the repository
TORCH_REFS = Path(os.environ.get("ARTICULINO_TORCH_REFS")
                  or next((d for d in _TORCH_REF_DIRS if d.is_dir()), _TORCH_REF_DIRS[0]))
DATA = HERE / "data"
VOICE = DATA / "radiov.wav"
STAGES = ("fp32", "fp16", "q8")

# (config, block) -> {case: value}; printed at the end of the run and written to ARTICULINO_MLX_REPORT
RESULTS: dict[tuple[str, str], dict[str, float | str]] = defaultdict(dict)


def record(config: str, block: str, case: str, value) -> None:
    RESULTS[(config, block)][case] = value


def device_name() -> str:
    import mlx.core as mx

    dev = os.environ.get("ARTICULINO_MLX_DEVICE")
    if dev:
        return dev
    return "cpu" if mx.default_device().type == mx.cpu else "gpu"


def stage_folder(stage: str) -> Path | None:
    if (WEIGHTS / "config.json").is_file():
        return WEIGHTS if stage == STAGES[0] else None
    folder = WEIGHTS / stage
    return folder if (folder / "config.json").is_file() else None


def configs() -> list[str]:
    env = os.environ.get("ARTICULINO_MLX_CONFIGS")
    if env:
        return [c.strip() for c in env.split(",") if c.strip()]
    out = list(STAGES)
    if device_name() == "cpu":
        out += ["fp16:float16", "q8:float16:deq"]
    return out


def parse_config(config: str) -> tuple[str, str | None, bool | None]:
    parts = config.split(":")
    stage = parts[0]
    dtype = parts[1] if len(parts) > 1 and parts[1] else None
    deq = (parts[2] == "deq") if len(parts) > 2 else None
    return stage, dtype, deq


def _fmt(value) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def pytest_terminal_summary(terminalreporter):
    if not RESULTS:
        return
    terminalreporter.write_sep("-", "measured values (fp32 blocks: max|diff|/max|ref|; other stages: SNR in dB)")
    for config in dict.fromkeys(c for c, _ in RESULTS):
        terminalreporter.write_line(f"[{config}]")
        for (cfg, block), vals in RESULTS.items():
            if cfg == config:
                cells = "  ".join(f"{case} {_fmt(v):>9}" for case, v in vals.items())
                terminalreporter.write_line(f"  {block:<26}{cells}")
    path = os.environ.get("ARTICULINO_MLX_REPORT")
    if path:
        data = defaultdict(dict)
        for (cfg, block), vals in RESULTS.items():
            data[cfg][block] = vals
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")


_CACHE: dict[str, object] = {}


def load_tts(config: str, **options):
    """TTS for a stage configuration, as in the reference vectors: no uniform rate, no post-processing."""
    key = config + repr(sorted(options.items()))
    if key not in _CACHE:
        from articulino_mlx import TTS

        stage, dtype, deq = parse_config(config)
        folder = stage_folder(stage)
        if folder is None:
            pytest.skip(f"no MLX weights for {stage} in {WEIGHTS} (run: articulino-mlx convert ... --all)")
        opts = dict(uniform_rate=False, low_shelf_db=0, peak_limit=None)
        opts.update(options)
        _CACHE[key] = TTS.from_pretrained(folder, device=device_name(), dtype=dtype, dequantize=deq, **opts)
    return _CACHE[key]


@pytest.fixture(scope="session")
def voice_path() -> Path:
    if not VOICE.is_file():
        pytest.skip(f"test voice {VOICE} missing")
    return VOICE
