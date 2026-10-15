---
language:
- de
- en
license: apache-2.0
pipeline_tag: text-to-speech
library_name: mlx
base_model: eigenhand/Articulino-DE-EN-Realtime-Stream-TTS
tags:
- text-to-speech
- tts
- german
- english
- voice-cloning
- zero-shot
- flow-matching
- dmd2
- mlx
- apple-silicon
---

# Articulino-DE-EN-Realtime-Stream-TTS-MLX

**[Articulino](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS) for Apple silicon, on
[MLX](https://github.com/ml-explore/mlx).** Small, fast German and English text to speech with zero-shot voice
cloning: a 4-step distilled flow-matching model on 25 Hz audio latents, about 110 M parameters with text, voice and
audio codecs. Same model, same API and, for the same seed, the same audio as the PyTorch release; three precisions
from 438 MB down to 127 MB. Code and weights are Apache 2.0.

On a MacBook Pro with M3 Pro, the whole pipeline runs **152× faster than real time** on the GPU (fp16), and the
first sentence is ready after 29 ms.

| | |
|---|---|
| Languages | German (main), English (limited) |
| Voice | any voice from a 3 to 8 second reference recording (zero-shot cloning) |
| Output | 24 kHz mono |
| Precisions | `fp32` 438 MB · `fp16` 223 MB (default) · `q8` 127 MB |
| Streaming | sentence by sentence while the text is still arriving |
| Runtime | `mlx`, `numpy`, `soundfile`, `num2words`; no PyTorch |
| Speed (M3 Pro, GPU) | 152× real time with `fp16`, first sentence after 29 ms; 26× on the CPU |
| License | Apache 2.0 for code and weights |

Model details, training data, evaluation, limitations and responsible use: see the
[main model card](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS).

## Quickstart

```bash
pip install "articulino-mlx[hub] @ git+https://github.com/eigenhand/articulino-mlx"
```

```python
from articulino_mlx import TTS

tts = TTS.from_pretrained("eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX")   # fp16; or precision="q8", …
tts.set_voice("voice.wav")                          # 3–8 s of clean speech from someone who agreed to this
speech = tts.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path="morgen.wav")
print(speech.duration, speech.sample_rate)          # seconds, 24000
```

`say()` returns a `Speech` with `audio` (float32 NumPy array in [-1, 1]), `sample_rate`, `duration` and `seed`; with
`path`, it also writes a WAV file. Options as in the PyTorch package: `language` (`"de"`, `"en"`), `speed` (0.7 to
1.6), `seed`, `normalize` (German text frontend). Words in the other language can be marked:
`"Das [en]Meeting[/en] ist verschoben."`.

### `stream()`: speak while the text is still coming

```python
import sounddevice as sd                          # pip install sounddevice

def llm_tokens():                                 # any iterator of text pieces, e.g. tokens from a language model
    yield from ["Guten Mor", "gen! Heute ", "regnet es ab 18 Uhr. ", "Nimm einen Schirm mit."]

with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as speaker:
    for chunk in tts.stream(llm_tokens()):
        speaker.write(chunk)                      # one chunk per sentence
```

### Command line

```bash
articulino-mlx say "Guten Morgen! Der Termin ist um 15:30 Uhr." --voice voice.wav -o morgen.wav
articulino-mlx say - --voice voice.wav -o out.wav --precision q8 < text.txt
articulino-mlx normalize "Die Rechnung über 87,40 € ist am 15.11. fällig."
```

### Options of `from_pretrained`

| Option | Meaning | Default |
|---|---|---|
| `precision` | `"fp32"`, `"fp16"`, `"q8"`: which subfolder of the repository is loaded | `"fp16"` |
| `device` | `"gpu"` or `"cpu"` | MLX's default device (the GPU on Apple silicon) |
| `dtype` | `"float32"` or `"float16"`: what the networks compute in | stored dtype on the GPU, float32 on the CPU |
| `dequantize` | expand the `q8` layers to plain weights at load time | on the CPU only (no fast quantized CPU kernels) |
| `uniform_rate`, `low_shelf_db`, `peak_limit`, `watermark` | as in the PyTorch package | as there |

A local folder works as well: either a folder with `fp32/ fp16/ q8/`, one of these subfolders, or the PyTorch
release folder itself (converted while loading).

## Precisions

| Precision | Size | What is stored | Per building block, SNR against the reference ¹ | Mel distance to fp32 ² | Parakeet CER ³ |
|---|---|---|---|---|---|
| `fp32` | 438 MB | float32 | relative error ≤ 4·10⁻⁵ (limit 10⁻³) | 0 | 0.011 |
| `fp16` | 223 MB | float16 | 47–62 dB | 1.4 dB | 0.011 |
| `q8` | 127 MB | 8-bit (affine, group 64) for every Linear layer with ≥ 65,536 weights, rest float16 | 33–60 dB | 2.9 dB | 0.011 |

¹ Each building block (log-mel, latent encoder, text encoder, voice encoder, voice cache, DiT velocity at all four
steps, 4-step latents, decoder) fed with the reference inputs, computed in float16 as on a Mac GPU (Linux CPU,
float16 arithmetic; the quantized layers expanded), against reference values from the original float32 code.
² Mean absolute difference of the mel spectrograms of `say()` (four sentences, seed 0) to fp32 in dB. For scale:
fp32 with another seed differs by 5.3 dB. fp16 and q8 stay much closer to fp32 than a change of seed. On the
GPU of an M3 Pro the distances are the same: 1.3 dB (fp16), 3.0 dB (q8).
³ The four sample sentences of the main card, recognised by Parakeet-TDT 0.6B v3; the one error is the same in all
precisions and in the PyTorch release (an extra "Uhr" after "Königstraße zwölf" at seed 0).

The duration predictor (1.9 M parameters) stays float32 in every precision and runs in float64 on the CPU: the
frame boundaries are cumulative sums of its outputs, so its rounding shifts later frames more than anything else in
the pipeline. The noise, the log-mel, the inverse STFT and the sampler update always run in float32.

Recommendation: `fp16`. It is also the fastest on a Mac (see [Speed](#speed)): the model is small enough that memory
bandwidth does not limit it, so `q8` saves download size, not time. `q8` if size matters.

There is no 4-bit MLX version: 4-bit weights (group-wise affine, as in `mlx.nn.quantize`) changed the voice audibly
(11 dB mel distance to fp32, voice tokens at 10 dB SNR) and failed the listening test.

The large evaluation of the main card (320 German and 107 English utterances of unseen speakers, three seeds,
Whisper large-v3, ECAPA, UTMOS, equivalence tests against the PyTorch reference) is being run for `fp32`, `fp16` and
`q8`; this card will be updated with the results. ONNX fp32/fp16/int8 and PyTorch on a GPU passed it as equivalent.

## How it works

The same pipeline as the PyTorch package, ported to `mlx.core` / `mlx.nn` (time-major tensors):

1. **Voice (once per voice):** log-mel (`mx.fft.rfft`) → ConvNeXt latent encoder → normalized latents → voice
   encoder (32 tokens + vector) → voice cache: keys and values of the first 5 s in every DiT block.
2. **Per sentence:** German text frontend → letters → text encoder → durations (float64) → per-frame text features
   → DiT, 4 steps from noise, continuing from the cached prompt → decoder (ConvNeXt at 100 Hz, magnitude and phase,
   inverse STFT with `mx.fft.irfft` and the window normalisation of `torch.istft`) → 24 kHz → low-shelf +2 dB at
   180 Hz, peak limit 0.95.

The noise is drawn exactly like PyTorch's CPU generator (MT19937 + Box-Muller, reproduced with NumPy), so a seed
gives the same audio as the PyTorch release: on a Linux CPU, `say()` in fp32 matches the PyTorch package with 48 to
63 dB SNR (mel distance below 0.01 dB).

## Files

```
fp32/ fp16/ q8/
    config.json          # the release config plus an "mlx" section (precision, dtype, quantization)
    model.safetensors    # acoustic model: text encoder, voice encoder, duration predictor, DiT
    codec.safetensors    # latent encoder and decoder
```

Tensor names are those of the PyTorch release (`model.safetensors`, `codec.safetensors`; mapping in its
[`docs/weights.md`](https://github.com/eigenhand/articulino/blob/main/docs/weights.md)). Differences: convolution weights are stored as (out, kernel, in) (MLX layout); the decoder's
transposed convolution `decoder.upsample.weight` is stored as (8·512, 512), row `j·512 + o` holding PyTorch's
`[:, o, j]` (it is computed as one matrix product and an overlap-add); quantized layers carry `weight` (packed
uint32), `scales` and `biases` as written by `mlx.nn.quantize`.

`articulino-mlx convert <pytorch-release-folder> <out> --all` writes all four folders from the PyTorch release.

## Tests

```bash
git clone https://github.com/eigenhand/articulino-mlx && cd articulino-mlx
pip install -e ".[hub,test]"
hf download eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX --local-dir build/mlx   # or ARTICULINO_MLX_WEIGHTS=…
python -m pytest
```

On a Mac, `python3 tools/mac_test.py` does all of this in its own environment and also measures the speed
([MAC_TEST.md](MAC_TEST.md)). The comparison with the PyTorch package uses its outputs shipped in `tests/torch_refs/`;
the byte-for-byte check of the shared modules runs when the PyTorch package is installed.

Every building block against the reference vectors of the original inference code (per precision), the end-to-end
audio (fp32: SNR above 40 dB; on an M3 Pro GPU 54.8 dB for the German and 40.3 dB for the English test sentence, so
the English case sits close to the limit), the quantized kernels against expanded weights, the noise and resampling against
PyTorch, `say()` against the PyTorch package, the German frontend (all 129 sentences of the pronunciation suite),
and the public API.

## Speed

MacBook Pro with M3 Pro, macOS, Python 3.13, ten sentences, whole pipeline (`tools/mac_test.py`), best of three runs:

| Precision@device | × real time | First sentence | Mel distance to fp32 |
|---|---|---|---|
| `fp32` GPU | 136× | 31 ms | – |
| `fp16` GPU | **152×** | 29 ms | 1.3 dB |
| `q8` GPU | 144× | 29 ms | 3.0 dB |
| `fp32` CPU | 26× | 171 ms | 0.003 dB |

On the same Mac, every test passed: 159 on the GPU (all precisions) and 35 on the CPU (fp32).

For comparison, on a Linux CPU (12 efficiency cores of an i9-14900K in a
virtual machine, ten sentences, whole pipeline, best of three): 6.6× real time when MLX's CPU backend is given
OpenBLAS, 0.3× with the reference BLAS that the Linux wheel ships. The PyTorch package does 31× on the same cores;
on a CPU, use that one.

## Responsible use

Only clone voices with consent, tell people they hear synthetic speech, and mark generated audio where the law asks
for it (EU AI Act, Art. 50). The weights contain no watermark; `watermark=` is a hook for your own. Details in the
[main model card](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS#responsible-use).

## License

Apache License 2.0 for code and weights. Keep the NOTICE file of the main release (training data attribution) when
you redistribute the model.
