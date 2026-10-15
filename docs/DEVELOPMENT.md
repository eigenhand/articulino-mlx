# Development notes: MLX

State of 15 October 2026. Package in `src/articulino_mlx` (src layout), tests in `tests/`, tools in `tools/`, weights
built into `build/mlx/{fp32,fp16,q8}` (not in git). The computation follows the inference specification in the
PyTorch repository ([docs/SPEC.md](https://github.com/eigenhand/articulino/blob/main/docs/SPEC.md)). The port was
built on a Linux CPU (MLX's CPU backend) and verified on Apple silicon afterwards.

## What the package does

- Package `articulino_mlx` (only mlx, numpy, soundfile, num2words; no PyTorch), with the API of the PyTorch package:
  `TTS.from_pretrained(repo_or_dir, precision="fp16", device=None, dtype=None, dequantize=None)`, `set_voice`, `say`,
  `stream` (per sentence), `Speech`, `Voice`, CLI `articulino-mlx say|normalize|convert`.
- Every building block ported to `mlx.core` / `mlx.nn` (time-major, (B, T, C)): log-mel with `mx.fft.rfft`, latent
  encoder, voice encoder and voice cache, text encoder, duration predictor, frame expansion, DiT with cache
  (`mx.fast.rope` interleaved, `mx.fast.scaled_dot_product_attention`), 4-step sampler (injectable noise), decoder and
  inverse STFT (`mx.fft.irfft`, window normalization as `torch.istft`), post-processing, sentence splitting.
- Unchanged copies of the PyTorch package (a test checks them byte for byte when that package is installed):
  `frontend/` (German text frontend and its JSON tables), `text.py`, `sentences.py`, `postprocess.py`, `config.py`.
  `audio.py` (resampling) is rewritten in NumPy.
- **Noise like torch:** `noise.py` reproduces `torch.Generator(seed)` + `torch.randn` (MT19937, Box–Muller in blocks of
  16) in NumPy, so a seed gives the same audio as the PyTorch package (largest difference 1.4e-6).
- **Durations in float64:** the duration predictor (1.9 M parameters) is stored in float32 in every precision and
  computed in float64 on the CPU stream (Metal has no float64). Frame boundaries are cumulative sums; float32
  rounding in MLX (LayerNorm/conv ~5e-6 instead of ~6e-7 in torch) shifted late boundaries and lowered the end-to-end
  SNR (English sentence 42 → 47 dB).
- **Converter** `articulino-mlx convert <pytorch-release> <out> --all` → `fp32/ fp16/ q8/` with `config.json` (plus an
  `mlx` section: precision, dtype, quantization), `model.safetensors`, `codec.safetensors`. Convolutions in MLX layout
  (out, k, in); `decoder.upsample.weight` as (8·512, 512) (matmul + overlap-add instead of `conv_transpose1d`). q8:
  `mlx.nn.quantize`, affine, group 64, every Linear layer with ≥ 65,536 weights and an input width divisible by 64
  (149 in the model, 49 in the codec), the rest float16; the duration predictor is never quantized. The PyTorch
  release folder also loads directly (transposed at load time).
- Compute type at load time: GPU = the stored type (fp16/q8 compute in float16 with quantized kernels); CPU = float32
  with quantized layers expanded (MLX's CPU backend has no fast float16 or quantized kernels).

## Sizes

| Precision | model | codec | total |
|---|---|---|---|
| fp32 | 265.1 MB | 172.7 MB | 437.9 MB |
| fp16 | 136.3 MB | 86.4 MB | 222.7 MB |
| q8 | 76.3 MB | 50.5 MB | 126.9 MB |

## Equality with the reference vectors (Linux CPU, fp32)

max|Δ| / max|ref| per block (limit 1e-3), every block fed with the reference inputs:

| Block | de_sentence | en_sentence |
|---|---|---|
| log-mel | 2.5e-5 | 2.5e-5 |
| encoder mu | 3.8e-6 | 3.8e-6 |
| text encoder | 5.2e-7 | 1.1e-6 |
| voice tokens / vector | 2.6e-6 / 6.8e-7 | 2.6e-6 / 6.8e-7 |
| log duration (float64 path) | 9.2e-7 | 1.6e-6 |
| frame conditioning | 3.0e-5 | 4.1e-5 |
| voice cache (k, v) | 2.1e-6 | 2.1e-6 |
| DiT velocity, steps 0–3 (max) | 4.8e-6 | 1.1e-6 |
| 4-step latents | 3.8e-6 | 2.3e-6 |
| decoder (inverse STFT) | 3.3e-6 | 5.0e-6 |
| **end-to-end audio (SNR)** | **55.5 dB** | **44.8 dB** |

Log-mel and frame conditioning are as close to the float64 value as the reference itself (both ~1.5e-5): float32
noise, not a porting error. The end-to-end SNR is lower than in the PyTorch package (91 / 75 dB, same kernels as the
reference) because the chain amplifies tiny differences in durations and voice tokens (with the reference
conditioning: latents 117–119 dB).

**`say()` against the PyTorch package** (release defaults: frontend, letter rate, shelf, peak; four cases with seeds
0/3/7/42 and speed 0.9–1.2): same length, SNR 48.0 / 58.5 / 55.3 / 62.9 dB, mel distance 0.0006–0.006 dB, the same
base letter rate (< 1e-4). Resampling like torchaudio (< 1e-5).

**Reduced precision** (SNR per block in dB, computed in float16 as on a Mac GPU; German / English):

| Block | fp16 | q8 |
|---|---|---|
| encoder mu | 61.0 / 61.0 | 59.6 / 59.6 |
| text encoder | 60.5 / 56.5 | 48.1 / 43.2 |
| voice tokens | 51.4 / 51.4 | 34.4 / 34.4 |
| voice cache (k, v) | 61.5 / 61.5 | 45.3 / 45.3 |
| DiT steps 0–3 | 56.7–58.5 | 38.9–43.1 |
| 4-step latents | 55.8 / 58.0 | 36.9 / 32.5 |
| decoder | 54.3 / 47.1 | 40.0 / 38.1 |
| audio: mel distance to the reference (dB) | 0.3 / 1.0 | 1.8 / 2.4 |

Waveform SNR is no useful measure for the lossy precisions (phase; the chain amplifies rounding), hence mel distance
and listening checks.

## Listening check (Parakeet-TDT 0.6B v3, 0.4 s of silence in front)

The four sample sentences of the model card with `say()`, seed 0: CER 0.011 in fp32, fp16 and q8; mel distance to fp32
1.44 dB (fp16) and 2.91 dB (q8), against 5.34 dB for fp32 with another seed. The one error is the same as in the
PyTorch package ("… Königstraße 12 Uhr", seed 0).

## 4-bit (dropped)

A uniform 4-bit version (group-wise affine, group 64) reached only 10 dB SNR on the voice tokens and an 11 dB mel
distance to fp32, twice a seed change. Mixed variants (8 bit for the voice encoder, latent encoder, conditioning and
decoder head, group 32) did not help (9.3–12.0 dB). After listening on a Mac it was dropped. A new attempt would need a
different method (for example AWQ/GPTQ, or the voice encoder and output layers in 8 bit) and a listening test.

## Speed

Apple M3 Pro: see the README (fp16 GPU 152×). On a Linux CPU (for orientation only; 12 efficiency cores of an
i9-14900K in a VM, ten sentences, best of three):

| Configuration | × real time | First sentence |
|---|---|---|
| fp32, Linux wheel as shipped (reference BLAS) | 0.3× | 13.7 s |
| fp32, OpenBLAS via `LD_PRELOAD`, 12 threads | 6.6× | 0.67 s |
| fp16 / q8 (CPU default: float32, expanded), OpenBLAS | 6.6× / 6.6× | 0.64–0.68 s |

The PyTorch package does 31.6× on the same cores; on a CPU, use that one. Bottlenecks of MLX's CPU backend:
element-wise operations (erf/GELU ~25 ns per element), `conv_transpose1d` (389 ms → 20 ms as matmul + overlap-add,
rebuilt), convolutions with 96/100 input channels (im2col matmul, rebuilt).

## First run on Apple silicon (15 October 2026, MacBook Pro M3 Pro)

Fresh clone, weights from the Hugging Face repository (SHA256SUMS verified), `tools/mac_test.py`: 159 tests passed on
the GPU (3 skipped because the PyTorch outputs were not yet shipped; they are now in `tests/torch_refs/`), 35 on the
CPU. End-to-end fp32 on the GPU: 54.8 dB (German), 40.3 dB (English, close to the 40 dB limit, as expected from the
float32 noise above); building blocks ~1e-5 or better.

| Precision | × real time | First sentence | Mel distance to fp32 |
|---|---|---|---|
| fp32 GPU | 136× | 31 ms | – |
| fp16 GPU | 152× | 29 ms | 1.3 dB |
| q8 GPU | 144× | 29 ms | 3.0 dB |
| fp32 CPU | 26× | 171 ms | 0.003 dB |

## Open points

- The repeat-on-broken-output guard of the C++ and ONNX engines (seed + 1) is missing, as in the PyTorch package.
- `tests/data/radiov.wav` is the voice used in all tests; its source and consent have to be settled before the
  repository is public.
