"""Quantized stages: the quantized kernels (QuantizedLinear, as on a Mac GPU) compute the same as the layers
expanded to plain weights (the default on a CPU), so the quantization error measured in test_vectors.py is the
same on both paths."""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from conftest import VECTORS, device_name, stage_folder


@pytest.mark.parametrize("stage", ["q8"])
def test_quantized_kernels_match_expanded_layers(stage):
    from articulino_mlx import TTS
    from articulino_mlx.evaluation import snr
    from articulino_mlx.weights import is_quantized

    folder = stage_folder(stage)
    if folder is None or not (VECTORS / "de_sentence.npz").is_file():
        pytest.skip("weights or vectors missing")
    v = dict(np.load(VECTORS / "de_sentence.npz"))
    info = json.loads((VECTORS / "de_sentence.json").read_text(encoding="utf-8"))
    dev = device_name()
    dtype = "float32" if dev == "cpu" else "float16"
    kernels = TTS.from_pretrained(folder, device=dev, dtype=dtype, dequantize=False)
    expanded = TTS.from_pretrained(folder, device=dev, dtype=dtype, dequantize=True)
    assert is_quantized(kernels.model) and not is_quantized(expanded.model)
    outs = []
    for tts in (kernels, expanded):
        with mx.stream(tts.device):
            m = tts.model
            h = m.text_encoder(mx.array(v["text_ids"][None].astype(np.int32)), mx.array([0]),
                               mx.array(v["text_mask"][None]))
            state = m.prepare_voice(mx.array(v["voice_ref"][None]), info["prompt_frames"])
            p = info["prompt_frames"]
            vel = m.dit.velocity(mx.array(v["step1_in"][None]), mx.full((1,), 0.25),
                                 mx.array(v["frame_cond"][None, p:]), state)
            dec = tts.codec.decoder(mx.array(v["decoder_in"].T[None]))
            outs.append([np.array(x.astype(mx.float32)) for x in (h, vel, dec)])
    need = 80.0 if dtype == "float32" else 40.0
    for name, a, b in zip(("text encoder", "DiT velocity", "decoder"), *outs):
        assert snr(a, b) > need, (stage, name, snr(a, b))
