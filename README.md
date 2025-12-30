# ComfyUI Egregora Audio Super Resolution

A focused audio toolkit for ComfyUI: upscale, enhance, and evaluate audio quality with a clean, practical workflow. This pack is built for real-world use: minimal setup, clear node purposes, and tools to verify results.

---

## Project scope (what this is and is not)

**What it is:**
- A set of audio enhancement nodes (FlashSR + Fat Llama) plus evaluation tools (ABX, loudness, null tests).
- Designed to help you *improve* low-quality audio and *measure* changes reliably.

**What it is not:**
- Not a magical "increase bitrate" tool. It enhances signal content and writes a new file at a chosen format/bitrate.
- Not a replacement for professional mastering. Think of it as an audio cleanup/boost stage.

---

## Nodes overview (what each one does)

### 1) Audio Super Resolution (FlashSR)
**Purpose:** Diffusion-based upsampler aimed at musical content. It resamples internally to 48 kHz and can resample output back to your target rate.

**Best for:**
- Low to mid quality music or wideband content
- Improving detail and clarity in band-limited audio

**Inputs:**
- `audio` (AUDIO)
- `lowpass_input` (BOOL) gentle LPF before inference
- `output_sr` (48000 / 44100 / 96000)

**Outputs:**
- One AUDIO buffer

**Use case:**
- Feed an audio file node -> FlashSR -> Preview Audio

---

### 2) Spectral Enhance (Fat Llama GPU)
**Purpose:** Iterative spectral enhancement using CuPy on GPU.

**Best for:**
- Noisy or compressed audio
- Sharpening "sparkle" and spectral detail

**Inputs:**
- `target_format` (wav / flac)
- `max_iterations` (higher = more aggressive, slower)
- `threshold_value` (controls spectral gating)
- `target_bitrate_kbps` (target write bitrate)
- `toggle_normalize` (on by default)
- `toggle_autoscale` (on by default)

**Outputs:**
- One AUDIO buffer

**Use case:**
- Audio -> Fat Llama GPU -> Preview

---

### 3) Spectral Enhance (Fat Llama CPU/FFTW)
**Purpose:** CPU fallback using FFTW. Same idea as GPU but slower.

**Use case:**
- When you don?t have CUDA/CuPy

---

### 4) Enhance Extras
**Purpose:** Denoise, dereverb, and codec tools you can chain in front of FlashSR or Fat Llama.

Includes:
- RNNoise Denoise
- DeepFilterNet 2/3 Denoise
- WPE Dereverb
- DAC encode/decode

---

### 5) Eval Pack
**Purpose:** Measure loudness, distortion, and quality.

Includes:
- Loudness meter (LUFS approx)
- Gain match (LUFS/RMS)
- ABX preparation/judge
- Spectral metrics (SI-SDR, LSD)
- High quality resampler

---

### 6) Null Test Suite
**Purpose:** See exactly what changed between A and B by aligning and subtracting signals.

Includes:
- Alignment (GCC-PHAT)
- Gain match
- Null output and plots

---

## How to combine nodes (common workflows)

### Clean + enhance (recommended chain)
1) Denoise/Dereverb (Extras)
2) FlashSR (optional)
3) Fat Llama (light pass)
4) Eval Pack or Null Test to verify

### FlashSR only
- Audio -> FlashSR -> Preview

### Fat Llama only
- Audio -> Fat Llama -> Preview

---

## Installation

### 1) Copy node pack
Place this folder into:

```
ComfyUI/custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution
```

Restart ComfyUI once.

---

### 2) Install dependencies (recommended)
Use ComfyUI?s embedded Python:

```powershell
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution
equirements.txt
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\install.py
```

Notes:
- Torch/torchaudio are not installed here to avoid breaking ComfyUI.
- On Windows, `install.py` installs NVIDIA CUDA runtime wheels for CuPy.

---

## FlashSR repo and weights

This project does **not** ship FlashSR code or weights. You must provide the weights:

```
ComfyUI/models/audio/flashsr/
  student_ldm.pth
  sr_vocoder.pth
  vae.pth
```

Optional auto-download from your HF repo:

```powershell
set EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights
```

---

## Troubleshooting (quick fixes)

### FlashSR import issues
- The node auto-downloads `deps/FlashSR_Inference/` on first use.
- If it fails, delete the folder and retry:

```powershell
Remove-Item -Recurse -Force .\ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\deps\FlashSR_Inference
```

### CuPy / CUDA root not detected (Fat Llama GPU)
Run this in ComfyUI root:

```powershell
python_embeded\python.exe -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cupy-cuda12x
```

### Numba needs NumPy 1.26 or less

```powershell
python_embeded\python.exe -m pip install "numpy<=1.26.4"
```

---

## License notes

- FlashSR inference code and weights are from upstream authors; check their repo for license status.
- Fat Llama packages are BSD-3-Clause (see PyPI).
- This integration is MIT (see LICENSE).

---

## Changelog

- **v0.2.1**
  - FlashSR auto-bootstrap and clearer diagnostics.
  - Fat Llama CUDA path detection fixes for portable installs.
  - Fat Llama output scaling aligned with upstream behavior.
  - NumPy pinned to `<=1.26.4` for Numba compatibility.

- **v0.2.0** Added Enhance/Eval/Null toolsets; new installer + warmups.
- **v0.1.0** Initial release: FlashSR SR node, Fat Llama GPU/CPU.
