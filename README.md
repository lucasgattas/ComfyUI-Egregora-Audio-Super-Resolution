# 🎧 ComfyUI — Egregora Audio Super‑Resolution

Bring music up to studio‑grade sample rates right inside ComfyUI.

This repo ships **three production‑oriented upscaling/enhancement nodes** and bundles a set of **integrated utility toolsets** (enhance, evaluation, null‑testing) so you can denoise → upscale → measure without wiring a huge graph.

---

## ✨ What’s inside

```
custom_nodes/
  ComfyUI-Egregora-Audio-Super-Resolution/
    __init__.py
    egregora_audio_super_resolution.py   # FlashSR node
    egregora_fat_llama_gpu.py            # Fat Llama (CUDA/CuPy)
    egregora_fat_llama_cpu.py            # Fat Llama (CPU/FFTW)
    egregora_audio_enhance_extras.py     # RNNoise / DeepFilterNet / WPE / DAC
    egregora_audio_eval_pack.py          # ABX, Loudness/Match, Metrics, HQ Resample
    egregora_null_test_suite.py          # Align, Gain‑Match, Null, Plots
    flashsr_min.py                       # Light wrapper for FlashSR
    install.py                           # Repo + weights/deps bootstrapper
    requirements.txt
    deps/
      FlashSR_Inference/                 # pulled automatically on install
```

### Core nodes

* **Audio Super Resolution (FlashSR)** — one‑step diffusion upsampler (music‑friendly) ⚡
* **Spectral Enhance (Fat Llama — GPU)** — CUDA/CuPy accelerated iterative spectral enhancer 🐍🧪
* **Spectral Enhance (Fat Llama — CPU/FFTW)** — portable CPU fallback using pyFFTW 🧠

### Integrated utility toolsets (used inside the SR nodes)

* **Enhance — Extras**

  * RNNoise Denoise (48 kHz, adaptive mix, strength, post‑gain)
  * DeepFilterNet 2/3 Denoise (48 kHz native)
  * WPE Dereverb (nara‑wpe)
  * DAC Encode/Decode (Descript Audio Codec)
* **Eval Pack**

  * ABX prepare/judge clips
  * Loudness meter (BS.1770), Gain‑Match (LUFS/RMS)
  * Metrics: SI‑SDR, Log‑Spectral Distance (LSD)
  * High‑quality resampler (SciPy/torch fallbacks)
* **Null Test Suite**

  * Align (XCorr GCC‑PHAT), Gain‑Match, Null, difference plots

> These helpers are wired so you can ABX / null‑test right from the SR node panel.

---

## 🧩 Install (ComfyUI portable or venv)

1. **Copy the folder** to `ComfyUI/custom_nodes/` and restart ComfyUI once.

2. **Install Python deps** using ComfyUI’s Python:

```bash
# From ComfyUI root
python -m pip install -r custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/requirements.txt
python custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/install.py
```

* We **do not** install `torch/torchaudio` here to avoid breaking ComfyUI’s CUDA build.
* First run will:

  * clone `deps/FlashSR_Inference/`
  * check for FlashSR weights
  * warm up DeepFilterNet / DAC / RNNoise caches for smoother first use

3. **FlashSR repo & weights**

* The node pulls the upstream inference code automatically into `deps/FlashSR_Inference/`.
* Place weights in `ComfyUI/models/audio/flashsr/` with **exact** filenames:

  * `student_ldm.pth`, `sr_vocoder.pth`, `vae.pth`
* Or set an env var to auto‑download from your HF repo:

```bash
# point to a HF repo containing those three files
# Windows (cmd)
set EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights
# macOS/Linux
export EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights
```

4. **GPU extras (for the Fat‑Llama GPU node)**

Install a CuPy wheel matching your CUDA (example for CUDA 12):

```bash
python -m pip install "cupy-cuda12x>=13.0"
```

If Windows shows NVRTC / `vector_types.h` errors, install the CUDA runtime DLL wheels:

```bash
python -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12
```

5. **FFmpeg**

Ensure FFmpeg is on your PATH for reading/encoding audio.

---

## 📦 Requirements

`requirements.txt` keeps things lean:

* Core: `soundfile`, `numpy`, `tqdm`, `requests`, `huggingface_hub`
* SR/enhance: `fat-llama`, `fat-llama-fftw`, `pyrnnoise`, `deepfilternet` (import as `df`), `nara-wpe` (import as `nara_wpe`), `descript-audio-codec`
* Optional: `scipy` for HQ resampler/metrics

> Booleans in node UIs use the `BOOLEAN` datatype in `INPUT_TYPES` (proper toggle).

---

## 🛠️ Nodes & key settings

### 1) **Audio Super Resolution (FlashSR)**

* Chunks → overlap‑add → stitches to 48 kHz (or chosen target).
* **Inputs**: `chunk_seconds` (default 5.12), `overlap_seconds` (0.5–0.75 if seams), `device`, `target_sr`, `output_format`, `audio_path` / `audio_url`, `flashsr_lowpass` (gentle LPF).
* **Outputs**: **AUDIO** buffer + saved file.

### 2) **Spectral Enhance (Fat Llama — GPU/CPU)**

* Iterative soft‑thresholding with spectral post.
* **Inputs**: `max_iterations`, `threshold_value`, `target_bitrate_kbps`, `toggle_autoscale`, `target_format`, `audio_path` / `audio_url`.
* **Outputs**: **AUDIO** buffer + saved file.

### Utility toolsets (used inside SR nodes)

* **Denoise/Dereverb**: RNNoise, DeepFilterNet 2/3, WPE
* **Codec**: DAC encode/decode
* **Eval**: ABX clips + judge, BS.1770 loudness, gain‑match, SI‑SDR, LSD
* **Null**: Align → match → null + difference plots

---

## 🎚️ Quality tips (music)

* **FlashSR first, Llama second**: upscale to 48k, then a *light* Llama pass (`iterations≈200`, `threshold≈0.5`) if you want a touch of sparkle.
* **Overlap**: If you hear ticks between chunks, raise `overlap_seconds` a bit.
* **Don’t over‑iterate**: very high iterations/threshold can sound brittle.

---

## 🔍 Licenses (upstream projects)

* **Fat‑Llama / fat‑llama‑fftw**: BSD‑3‑Clause (see PyPI).
* **FlashSR_Inference**: check upstream repo for license status.
* This ComfyUI integration is licensed as per this repository’s LICENSE.

---

## 🧪 Troubleshooting

* **FlashSR import error**: delete `deps/FlashSR_Inference/` and restart to re‑bootstrap.
* **Missing FlashSR weights**: place the 3 files in `models/audio/flashsr/` or set `EGREGORA_FLASHSR_HF_REPO`.
* **CUDA/CuPy NVRTC errors (Windows)**: install the `nvidia-*-cu12` runtime wheels listed above and ensure your CuPy wheel matches CUDA.
* **FFmpeg not found**: install FFmpeg and ensure it’s on PATH.

---

## 🙌 Credits

* FlashSR research & inference code by the original authors.
* Fat Llama packages by RaAd (PyPI maintainer).
* ComfyUI integration & node UX by Egregora.

Happy upsampling! 🎶

---

## 📜 Changelog

* **v0.2.0** — Added Enhance/Eval/Null toolsets; new installer + warmups.
* **v0.1.0** — Initial release: FlashSR SR node, Fat Llama GPU/CPU.
