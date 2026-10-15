"""Loading, converting and quantizing the weights.

Two kinds of folders can be loaded:

* an **MLX folder** (written by :func:`convert`): ``config.json`` has an ``"mlx"`` section; convolution weights are
  already in MLX layout (out, kernel, in); the tensors are float32 or float16, and quantized layers carry
  ``weight`` (packed uint32), ``scales`` and ``biases`` as written by ``mlx.nn.quantize``;
* the **PyTorch release folder** (``model.safetensors``, ``codec.safetensors``, ``config.json`` of
  Articulino-DE-EN-Realtime-Stream-TTS): float32 in PyTorch layout, transposed while loading.

Tensor names are the release names in both cases (see ``docs/weights.md``).
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path

from .codec import Codec
from .config import ArticulinoConfig
from .model import AcousticModel

WEIGHT_FILES = ("config.json", "model.safetensors", "codec.safetensors")
FORMAT_VERSION = 1
PRECISIONS = {                      # name -> (storage dtype, quantization bits or None)
    "fp32": (mx.float32, None),
    "fp16": (mx.float16, None),
    "q8": (mx.float16, 8),
}
GROUP_SIZE = 64
MIN_QUANT_SIZE = 1 << 16            # Linear layers with fewer weights stay unquantized (output heads, small MLPs)
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def dtype_name(dtype: mx.Dtype) -> str:
    return {v: k for k, v in DTYPES.items()}[dtype]


def to_mlx_layout(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """PyTorch -> MLX layout. Conv1d (out, in, k) -> (out, k, in) (``nn.Conv1d`` layout). The decoder's
    ConvTranspose1d ``upsample`` (in, out, k) -> (k·out, in): computed as one matrix product and an overlap-add
    (see ``codec.Upsample``). Every 3-D tensor of Articulino is a convolution weight."""
    out = {}
    for name, w in weights.items():
        if w.ndim == 3:
            if name.endswith("upsample.weight"):
                w = w.transpose(2, 1, 0).reshape(w.shape[2] * w.shape[1], w.shape[0])
            else:
                w = w.transpose(0, 2, 1)
        out[name] = w
    return out


def quantizable(group_size: int = GROUP_SIZE, min_size: int = MIN_QUANT_SIZE,
                skip: tuple[str, ...] = ()) -> Callable[[str, nn.Module], bool]:
    """Predicate for ``nn.quantize``: Linear layers with at least ``min_size`` weights whose input width is a multiple
    of ``group_size``, except paths starting with one of ``skip``. Norms, embeddings, convolutions, biases and other
    small tensors stay as they are."""

    def predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
            return False
        out_dim, in_dim = module.weight.shape
        if in_dim % group_size or module.weight.size < min_size:
            return False
        return not any(path == s or path.startswith(s + ".") for s in skip)

    return predicate


def dequantize(module: nn.Module, dtype: mx.Dtype = mx.float32) -> nn.Module:
    """Replace every QuantizedLinear by a plain Linear holding the dequantized weights (same numbers, as ``dtype``).
    The quantization error stays; only the kernels change (the MLX CPU backend has no fast quantized matmul)."""

    def expand(path: str, m: nn.Module) -> nn.Module:
        if not isinstance(m, nn.QuantizedLinear):
            return m
        kwargs = {"group_size": m.group_size, "bits": m.bits}
        if getattr(m, "mode", "affine") != "affine":
            kwargs["mode"] = m.mode
        w = mx.dequantize(m.weight, m.scales, m.get("biases"), **kwargs).astype(dtype)
        has_bias = "bias" in m
        lin = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
        lin.weight = w
        if has_bias:
            lin.bias = m.bias.astype(dtype)
        return lin

    module.update_modules(tree_map_with_path(expand, module.leaf_modules(), is_leaf=nn.Module.is_module))
    return module


KEEP_FLOAT32 = ("duration_predictor",)   # tiny, and its outputs are summed up: stays float32, never quantized


def cast(module: nn.Module, dtype: mx.Dtype) -> None:
    """Cast the floating-point parameters (incl. quantization scales) of ``module`` to ``dtype``, except the
    sub-modules in :data:`KEEP_FLOAT32`."""
    children = module.children()
    keep = [name for name in KEEP_FLOAT32 if name in children]
    if not keep:
        module.set_dtype(dtype)
        return
    for name, child in children.items():
        if name not in keep:
            for c in (child if isinstance(child, list) else [child]):
                c.set_dtype(dtype)
    for name, value in module.parameters().items():
        if name not in children and isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
            setattr(module, name, value.astype(dtype))


def read_config(folder: str | os.PathLike) -> dict:
    with open(Path(folder) / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_modules(folder: str | os.PathLike, dtype: mx.Dtype | None = None, dequantize_layers: bool = False,
                 ) -> tuple[AcousticModel, Codec, ArticulinoConfig, dict]:
    """Build the acoustic model and the codec from a folder (MLX or PyTorch release).

    dtype: cast every floating-point parameter (incl. quantization scales) to this dtype; None keeps the stored one.
    dequantize_layers: expand quantized layers to plain Linear layers (in ``dtype`` or the stored dtype).
    Returns (model, codec, config, mlx section of the config)."""
    folder = Path(folder)
    raw = read_config(folder)
    config = ArticulinoConfig.from_dict(raw)
    info = dict(raw.get("mlx") or {})
    model = AcousticModel(config)
    codec = Codec.from_config(config.codec, config.latents.dim)
    for module, file in ((model, "model.safetensors"), (codec, "codec.safetensors")):
        weights = mx.load(str(folder / file))
        if not info:
            weights = to_mlx_layout(weights)
        quant = info.get("quantization")
        if quant:
            nn.quantize(module, group_size=quant["group_size"], bits=quant["bits"],
                        class_predicate=lambda path, m, w=weights: f"{path}.scales" in w)
        module.load_weights(list(weights.items()), strict=True)
        if dtype is not None:
            cast(module, dtype)
        if dequantize_layers and quant:
            dequantize(module, dtype or info_dtype(info))
        mx.eval(module.parameters())
    return model, codec, config, info


def info_dtype(info: dict) -> mx.Dtype:
    return DTYPES.get(info.get("dtype", "float32"), mx.float32)


def is_quantized(module: nn.Module) -> bool:
    return any(isinstance(m, nn.QuantizedLinear) for _, m in module.named_modules())


def save_folder(model: AcousticModel, codec: Codec, raw_config: dict, out: str | os.PathLike, info: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for module, file in ((model, "model.safetensors"), (codec, "codec.safetensors")):
        tensors = dict(tree_flatten(module.parameters()))
        mx.save_safetensors(str(out / file), tensors, metadata={"format": "mlx"})
    cfg = copy.deepcopy(raw_config)
    cfg["mlx"] = info
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert(src: str | os.PathLike, out: str | os.PathLike, precision: str = "fp16", group_size: int = GROUP_SIZE,
            quantize_codec: bool = True, skip: tuple[str, ...] = ()) -> dict:
    """Convert the PyTorch release folder ``src`` into an MLX folder ``out``.

    precision: ``fp32``, ``fp16`` or ``q8``. The quantized stage quantizes the large Linear layers (see
    :func:`quantizable`) from the float32 weights with ``mlx.nn.quantize`` (affine, ``group_size``), and store
    everything else (and the scales) in float16. quantize_codec: also quantize the codec's Linear layers.
    skip: module paths (prefixes) that stay unquantized. Returns the ``mlx`` section written to config.json."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, not {precision!r}")
    dtype, bits = PRECISIONS[precision]
    raw = read_config(src)
    if raw.get("mlx"):
        raise ValueError(f"{src} is already an MLX folder; convert from the PyTorch release")
    model, codec, _, _ = load_modules(src)                          # float32
    info: dict = {"format_version": FORMAT_VERSION, "precision": precision, "layout": "mlx",
                  "dtype": dtype_name(dtype), "quantization": None, "source": str(Path(src).name)}
    counts = {}
    if bits:
        predicate = quantizable(group_size, skip=tuple(skip) + KEEP_FLOAT32)
        targets = [("model", model)] + ([("codec", codec)] if quantize_codec else [])
        for name, module in targets:
            counts[name] = sum(predicate(p, m) for p, m in module.named_modules())
            nn.quantize(module, group_size=group_size, bits=bits, class_predicate=predicate)
        info["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine",
                                "min_size": MIN_QUANT_SIZE, "quantized_layers": counts,
                                "codec": bool(quantize_codec), "skip": list(skip) + list(KEEP_FLOAT32)}
    for module in (model, codec):
        cast(module, dtype)
    mx.eval(model.parameters(), codec.parameters())
    save_folder(model, codec, raw, out, info)
    return info
