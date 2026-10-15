# Testing articulino-mlx on a Mac

One script installs the package into its own virtual environment, checks it against the reference values, measures
the speed of every precision on the GPU and writes audio files. First run on Apple silicon (15 October 2026, MacBook
Pro with M3 Pro): all tests passed, 152× real time with fp16, the whole run took under a minute.

## What you need

- A Mac with Apple silicon (M1 or newer) and macOS 14 or newer.
- Python 3.10 or newer: `python3 --version`. If it is older: `brew install python@3.12` (then use `python3.12`).
- About 2 GB of free disk space and a few minutes (plus the 0.8 GB download of the weights).
- This repository: `git clone https://github.com/eigenhand/articulino-mlx`.
- The weights: by default the script downloads them from
  [eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-MLX)
  or pass a local folder with the
  subfolders `fp32/ fp16/ q8/`.

## Run it

```bash
cd articulino-mlx
python3 tools/mac_test.py                          # weights from ./weights if present, else from Hugging Face
python3 tools/mac_test.py --weights ~/path/to/mlx  # or from a local folder
python3 tools/mac_test.py --quick                  # faster: fp32 + fp16 only, fewer tests
```

The script

1. creates `.venv-mac-test` in this folder and installs the package with `mlx`, `numpy`, `soundfile`, `num2words`,
   `huggingface_hub` and `pytest` (nothing is installed globally; delete the folder afterwards if you like);
2. runs the tests on the GPU: every building block against the reference vectors in fp32 (limit: relative error
   1e-3, audio SNR 40 dB) and fp16, q8 (SNR reported), the public API (`say`, `stream`, voices, CLI), the
   quantized kernels, the German frontend, and `say()` against the output of the PyTorch package;
   then fp32 on the CPU;
3. measures the real-time factor of fp32, fp16 and q8 on the GPU (ten sentences, best of three runs) and fp32
   on the CPU;
4. writes WAV files per precision to `mac_test_results/<name>/wavs/` (the four sample sentences of the model card
   and the ten test sentences);
5. packs everything into `mac_test_results/<name>.zip` and shows it in the Finder.

**The zip file is the result to share.** If something fails, the script still finishes; the zip then contains the log
(`log.txt`), the test output (`pytest_gpu.txt`) and the error.

Worth a listen: `wavs/fp16_gpu/card-1-de.wav` against `wavs/fp32_gpu/card-1-de.wav`, and the same for `q8` (fp16 and q8
stay closer to fp32 than a change of seed).

## Using it afterwards

```python
from articulino_mlx import TTS

tts = TTS.from_pretrained("weights", precision="fp16")    # or the repository; fp32, fp16, q8
tts.set_voice("voice.wav")
tts.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path="morgen.wav")
```

```bash
.venv-mac-test/bin/articulino-mlx say "Guten Morgen!" --voice voice.wav -o morgen.wav --model weights --precision fp16
```
