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

**Platform support:**
- Requires NVIDIA CUDA + CuPy.
- Works on Windows/Linux CUDA systems.
- Does **not** run on macOS (including Apple Silicon/MPS).

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
- When you don't have CUDA/CuPy.
- Recommended path on macOS.

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
Use the same Python environment ComfyUI runs with.

**Windows (portable ComfyUI):**
```powershell
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\requirements.txt
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\install.py
```

**macOS / Linux (venv or system Python used by ComfyUI):**
```bash
python3 -m pip install -r ComfyUI/custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/requirements.txt
python3 ComfyUI/custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/install.py
```

Notes:
- Torch/torchaudio are not installed here to avoid breaking ComfyUI.
- On Windows, `install.py` also installs NVIDIA CUDA runtime wheels for CuPy (used by the GPU Fat Llama node).
- On macOS/Linux, CPU/FFTW nodes work normally; GPU Fat Llama still requires NVIDIA CUDA + CuPy.
- On Apple Silicon, PyTorch-based nodes now use `mps` automatically when CUDA is unavailable.
- `fat-llama` (GPU package) is optional and may fail on non-CUDA systems; base install uses `fat-llama-fftw` for CPU compatibility.
- `deepfilternet` and `descript-audio-codec` are treated as optional in shared environments to avoid transitive dependency downgrades.
- To run DeepFilterNet/DAC without polluting ComfyUI's main environment, use a helper venv and set `EGREGORA_HELPER_PYTHON`.

### Optional helper venv (DeepFilterNet + DAC)
Create an isolated helper environment:

```bash
python3 -m venv .egregora-helper-venv
./.egregora-helper-venv/bin/python -m pip install -U pip
./.egregora-helper-venv/bin/python -m pip install deepfilternet descript-audio-codec
```

Set this variable before launching ComfyUI:

```bash
export EGREGORA_HELPER_PYTHON="$(pwd)/.egregora-helper-venv/bin/python"
```

When set, DeepFilterNet and DAC nodes automatically run in the helper venv through a subprocess bridge.

Or let `install.py` create/update it automatically (recommended):

```bash
EGREGORA_SETUP_HELPER_VENV=1 python3 ComfyUI/custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/install.py
```

Optional helper path override:

```bash
EGREGORA_SETUP_HELPER_VENV=1 EGREGORA_HELPER_VENV="$(pwd)/.eg-helper" python3 ComfyUI/custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/install.py
```

After setup, start ComfyUI from a shell where `EGREGORA_HELPER_PYTHON` points to the helper interpreter.

---

## FlashSR repo and weights

The node **auto-downloads the FlashSR inference repo** on first use into `deps/FlashSR_Inference/`:
https://github.com/jakeoneijk/FlashSR_Inference

However, **FlashSR model weights are not included** in this pack due to licensing/redistribution limits. The weights page does not state a license — download at your own discretion.

You must obtain the weights from the FlashSR authors or their official release and place them here:
https://huggingface.co/datasets/jakeoneijk/FlashSR_weights

```
ComfyUI/models/audio/flashsr/
  student_ldm.pth
  sr_vocoder.pth
  vae.pth
```

Optional auto-download (if *you* host the weights in your own HF repo):

```powershell
set EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights
```

```bash
export EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights
```


---

## Troubleshooting (quick fixes)

### FlashSR import issues
- The node auto-downloads `deps/FlashSR_Inference/` on first use.
- If it fails, delete the folder and retry:

```powershell
Remove-Item -Recurse -Force .\ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\deps\FlashSR_Inference
```

```bash
rm -rf ./ComfyUI/custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/deps/FlashSR_Inference
```

### CuPy / CUDA root not detected (Fat Llama GPU)
Run this in ComfyUI root:

```powershell
python_embeded\python.exe -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cupy-cuda12x
```

```bash
python3 -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cupy-cuda12x
```

Then install the optional GPU package:

```bash
python3 -m pip install -U fat-llama
```

### Apple Silicon (MPS) notes
- FlashSR and PyTorch-based Enhance Extras nodes support device auto-selection `cuda -> mps -> cpu`.
- If an upstream model/checkpoint is loaded as FP8, the nodes now auto-cast to FP16 on MPS for compatibility.
- Fat Llama GPU is CUDA/CuPy-only and does **not** run on MPS; use **Fat Llama CPU/FFTW** on macOS.

### NumPy compatibility note
- This pack no longer hard-pins NumPy to `<=1.26.4`.
- Keep the NumPy version used by your ComfyUI environment to avoid conflicts with other nodes (for example OpenCV/rembg stacks that require NumPy 2.x).
- If you install optional packages manually and see resolver warnings, prefer keeping your existing ComfyUI core stack (`numpy`, `packaging`, `protobuf`) unchanged.

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
  - Previously pinned NumPy for older Numba compatibility (now removed in current branch for better cross-node compatibility).

- **v0.2.0** Added Enhance/Eval/Null toolsets; new installer + warmups.
- **v0.1.0** Initial release: FlashSR SR node, Fat Llama GPU/CPU.

