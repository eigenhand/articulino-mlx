"""Text side: the German frontend and the other torch-free modules are unchanged copies of the PyTorch package, and
the frontend gives the original output for the whole pronunciation suite."""
from __future__ import annotations

import random

import numpy as np
import pytest

from articulino_mlx.frontend import StreamNormalizer, normalize
from conftest import DATA, ROOT


def _pytorch_package():
    """The PyTorch package to compare the copies with: ARTICULINO_TORCH_PACKAGE (the folder of `articulino`), else the
    installed `articulino` (pip install "articulino @ git+https://github.com/eigenhand/articulino")."""
    import importlib.util
    import os
    from pathlib import Path

    if os.environ.get("ARTICULINO_TORCH_PACKAGE"):
        return Path(os.environ["ARTICULINO_TORCH_PACKAGE"])
    spec = importlib.util.find_spec("articulino")
    return Path(spec.origin).parent if spec and spec.origin else Path("articulino-not-installed")


PYTORCH_PACKAGE = _pytorch_package()
COPIED = ["frontend/__init__.py", "frontend/german.py", "frontend/data/abbreviations.json", "frontend/data/units.json",
          "text.py", "sentences.py", "postprocess.py", "config.py"]


def _cases():
    rows = []
    for line in (DATA / "pronunciation_cases.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(tuple(line.split("\t")))
    return rows


@pytest.mark.parametrize("name", COPIED)
def test_same_files_as_pytorch_package(name):
    """The frontend, text, sentence, post-processing and config modules are byte-identical copies."""
    theirs = PYTORCH_PACKAGE / name
    if not theirs.is_file():
        pytest.skip(f"PyTorch package not found at {PYTORCH_PACKAGE}")
    ours = ROOT / "src" / "articulino_mlx" / name
    assert ours.read_bytes() == theirs.read_bytes()


def test_pronunciation_suite_matches_live_output():
    """All 129 sentences of the pronunciation suite come out exactly as from the original frontend."""
    rows = _cases()
    assert len(rows) == 129
    wrong = [(inp, ref, normalize(inp)) for _, inp, _, ref in rows if normalize(inp) != ref]
    assert not wrong, wrong[:3]


def test_stream_normalizer_matches_one_shot():
    rng = random.Random(0)
    for _, inp, _, ref in _cases():
        cuts = sorted(rng.sample(range(1, len(inp)), min(4, len(inp) - 1))) if len(inp) > 1 else []
        pieces = [inp[i:j] for i, j in zip([0] + cuts, cuts + [len(inp)])]
        s = StreamNormalizer()
        out = "".join(s.push(p) for p in pieces) + s.flush()
        assert out.strip() == normalize(inp).strip(), (inp, pieces)


def test_shelf_and_peak():
    from articulino_mlx.postprocess import SosFilter, limit_peak, shelf_sos

    sos = shelf_sos(180, 2.0, False, 24000, 0.7)
    f = SosFilter(sos)
    t = np.arange(24000) / 24000
    for hz, gain in ((30, 2.0), (4000, 0.0)):
        x = np.sin(2 * np.pi * hz * t)
        y = f(x)[12000:]
        assert abs(20 * np.log10(np.sqrt((y ** 2).mean() / (x[12000:] ** 2).mean())) - gain) < 0.05
    assert np.abs(limit_peak(np.array([0.0, 2.0, -1.0]))).max() == pytest.approx(0.95)
