# ComfyUI Egregora Audio Super Resolution

Audio enhancement and evaluation nodes for ComfyUI. Includes FlashSR super-resolution, Fat Llama spectral enhancement (GPU/CPU), and utility toolsets (enhance, eval, null testing).

---

## What is included

```
custom_nodes/
  ComfyUI-Egregora-Audio-Super-Resolution/
    __init__.py
    egregora_audio_super_resolution.py   # FlashSR node
    egregora_fat_llama_gpu.py            # Fat Llama (CUDA/CuPy)
    egregora_fat_llama_cpu.py            # Fat Llama (CPU/FFTW)
    egregora_audio_enhance_extras.py     # RNNoise / DeepFilterNet / WPE / DAC
    egregora_audio_eval_pack.py          # ABX, loudness, metrics, resample
    egregora_null_test_suite.py          # Align, gain-match, null plots
    flashsr_min.py                       # Light wrapper for FlashSR
    install.py                           # Repo + weights/deps bootstrapper
    requirements.txt
    deps/
      FlashSR_Inference/                 # Pulled automatically on first use
```

---

## Install (ComfyUI portable or venv)

1) Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI once.

2) Install dependencies using ComfyUI's Python:

```powershell
# From ComfyUI root
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolutionequirements.txt
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\install.py
```

Notes:
- This does not install torch/torchaudio to avoid breaking ComfyUI's CUDA build.
- On Windows, `install.py` now installs the NVIDIA CUDA runtime wheels required by CuPy.

---

## FlashSR (Audio Super Resolution)

FlashSR is a diffusion-based upsampler designed around 48 kHz. The node resamples input to 48 kHz internally and can resample output back to your chosen rate.

### FlashSR repo and weights

- The node auto-downloads the FlashSR inference repo on first use into `deps/FlashSR_Inference/`.
- This repo does not ship FlashSR code or weights.
- Place weights here with exact filenames:

```
ComfyUI/models/audio/flashsr/
  student_ldm.pth
  sr_vocoder.pth
  vae.pth
```

Optional: set a Hugging Face repo to auto-download weights:

```powershell
# Windows (cmd)
set EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights
```

---

## Fat Llama (Spectral Enhance)

Fat Llama is a spectral enhancer. It does not increase codec bitrate; it processes the signal and writes a new audio file at the chosen format/bitrate.

### GPU node (CUDA/CuPy)

- Requires NVIDIA GPU + CuPy CUDA 12.
- Uses CuPy + CUDA runtime wheels installed by `install.py` on Windows.

### CPU node (FFTW)

- CPU fallback using `fat-llama-fftw`.
- Slower but does not require CUDA.

---

## Recent fixes and corrections

- FlashSR auto-bootstrap:
  - Missing `deps/FlashSR_Inference` is now downloaded automatically on first use.
  - Clearer errors show resolved repo path and environment variable values.
- Fat Llama GPU CUDA detection:
  - The node now wires CUDA paths for portable installs and refreshes CuPy's cached CUDA path when needed.
  - `install.py` installs the NVIDIA runtime wheels on Windows.
- Fat Llama audio scaling:
  - Internal processing uses upstream scaling; output is scaled safely when writing to avoid clipping.
- NumPy/Numba compatibility:
  - NumPy is pinned to `<=1.26.4` to satisfy Numba (required by FlashSR tooling).

---

## Troubleshooting

### FlashSR import / auto-download

Symptoms:
- "FlashSR module not found"
- Missing `deps/FlashSR_Inference/`

Fixes:
1) Restart ComfyUI after installs or environment changes.
2) Check for logs on first run:
   - `[FlashSR] Repo missing; downloading FlashSR_Inference...`
   - `[FlashSR] Download complete; extracting...`
   - `[FlashSR] Repo OK: ...`
3) If you use a custom repo path, verify it:
   ```powershell
   echo $env:EGREGORA_FLASHSR_REPO
   Test-Path $env:EGREGORA_FLASHSR_REPO
   ```
4) Force a clean re-download:
   ```powershell
   Remove-Item -Recurse -Force .\ComfyUI\custom_nodes\ComfyUI-Egregora-Audio-Super-Resolution\deps\FlashSR_Inference
   ```

### CuPy / CUDA root not detected (Fat Llama GPU)

Run this in the ComfyUI root:

```powershell
python_embeded\python.exe -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cupy-cuda12x
```

Then restart ComfyUI.

### Numba needs NumPy 1.26 or less

Run this in the ComfyUI root:

```powershell
python_embeded\python.exe -m pip install "numpy<=1.26.4"
```

Then restart ComfyUI.

---

## License notes

- FlashSR inference code and weights are from upstream authors; check their repo for license status.
- Fat Llama packages are BSD-3-Clause (see PyPI).
- This integration is MIT (see LICENSE).

---

## Changelog

- Unreleased
  - FlashSR auto-bootstrap and clearer diagnostics.
  - Fat Llama CUDA path detection fixes for portable installs.
  - Fat Llama output scaling aligned with upstream behavior.
  - NumPy pinned to `<=1.26.4` for Numba compatibility.
