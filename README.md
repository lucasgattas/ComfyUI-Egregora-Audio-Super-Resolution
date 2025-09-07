# ComfyUI — Egregora Audio Super‑Resolution

> ✨ High‑quality music audio enhancement for ComfyUI: FlashSR super‑resolution + Fat Llama spectral enhancement (GPU & CPU).

---

## 🧩 What is this?

This repository provides **three production‑ready ComfyUI nodes** focused on **music audio** enhancement:

* **🎧 Audio Super Resolution (FlashSR)** — Fast, one‑step diffusion SR to **48 kHz**, chunked with seamless overlap‑add. Auto‑downloads the three required weights on first run.
* **🎛️ Spectral Enhance (Fat Llama — GPU)** — Iterative Soft Thresholding (IST) with CuPy/CUDA for clarity/restoration, with safe autoscaling.
* **🎛️ Spectral Enhance (Fat Llama — CPU/FFTW)** — CPU fallback using FFTW; slower but portable.

Designed to be simple, robust, and practical for music workflows.

---

## 🚀 Quick Install (TL;DR)

```bash
# 1) Clone into your ComfyUI custom nodes folder
cd <ComfyUI>/custom_nodes
git clone https://github.com/lucasgattas/ComfyUI-Egregora-Audio-Super-Resolution.git
cd ComfyUI-Egregora-Audio-Super-Resolution

# 2) Install Python dependencies into ComfyUI's Python
python -m pip install -r requirements.txt

# 3) (Optional) Run the helper installer (fetch vendor deps if you use it)
python install.py

# 4) Restart ComfyUI
```

> 💡 FlashSR weights are **auto‑downloaded** on first use via `huggingface_hub`. No manual step required unless the machine is offline.

---

## 📦 Requirements

**Common**

* Python 3.10+ (ComfyUI’s embedded Python is fine)
* `soundfile`, `numpy`, `torch`, `huggingface_hub` (pulled by `requirements.txt`)

**GPU Fat Llama**

* NVIDIA GPU (Compute Capability ≥ 7 recommended)
* **CuPy for CUDA 12**: `pip install cupy-cuda12x`
* NVIDIA CUDA runtime + NVRTC DLLs (wheels in `requirements.txt` will install `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12`, etc.)

**CPU/FFTW Fat Llama**

* `fat-llama-fftw` (pulled by `requirements.txt`)

**Optional**

* FFmpeg in PATH for wider audio format support (system install or a ComfyUI FFmpeg node)

---

## 📁 Where things go

* **Outputs (sidecar)** → `ComfyUI/output/audio/` (WAV/FLAC written by nodes)
* **FlashSR weights** → `ComfyUI/models/audio/flashsr/`
* **Vendor (optional)** → `custom_nodes/.../deps/`

You can override the models folder with **`EGREGORA_MODELS_DIR`** env var. If you store the FlashSR repo elsewhere, set **`EGREGORA_FLASHSR_REPO`** to that path.

---

## 🔧 Nodes & Settings

### 🎧 Audio Super Resolution (FlashSR)

**Purpose**: Reconstruct wide‑band detail at **48 kHz** using a distilled one‑step SR model + SR vocoder, processed in chunks and stitched by overlap‑add.

**Inputs**

* **chunk\_seconds** *(float, default 5.12)* — Window length per pass. FlashSR is trained/evaluated with **5.12 s**; keep this unless you have a reason.
* **overlap\_seconds** *(float, default 0.50)* — Cross‑fade between chunks. Increase (e.g., **0.8–1.0 s**) if you hear seam clicks; higher overlap = more compute.
* **device** *(auto/cuda/cpu)* — Select GPU when available for speed; CPU is fine for testing.
* **target\_sr** *(auto/48000/44100/...)* — The model **natively outputs 48 kHz**. If you need 44.1 kHz, best practice is to keep 48 kHz here and downsample after with a high‑quality resampler.
* **output\_format** *(wav/flac)* — The node writes a sidecar file to `output/audio/` with this container (WAV=PCM; FLAC=lossless, smaller).
* **audio\_path** *(string)* — Local file input (leave empty if using the upstream AUDIO pipe).
* **audio\_url** *(string)* — Download and process a remote audio file.
* **flashsr\_lowpass** *(bool)* — Light anti‑aliasing on input; can help with very low‑SR or hissy sources. Disable if results feel too dull.

**Outputs**

* **AUDIO** — `{ "waveform": [1,C,T] tensor, "sample_rate": int }` for chaining in ComfyUI.
* **Sidecar** — WAV/FLAC saved to `output/audio/` using your `output_format`.

**Tips**

* Defaults (5.12 s / 0.50 s) are strong. For difficult transitions use **0.8–1.0 s** overlap.
* Keep the model at **48 kHz** and resample at the very end of the chain for best fidelity.

---

### 🎛️ Spectral Enhance (Fat Llama — GPU)

**Purpose**: Fast **Iterative Soft Thresholding (IST)** in the spectral domain using CuPy/CUDA to recover clarity and tame residual noise after SR.

**Inputs**

* **target\_format** *(wav/flac)* — Output container for the sidecar render.
* **max\_iterations** *(int, default 300)* — More iterations = more detail but longer runtime. Typical: **200–400**; push higher (e.g., 500) cautiously.
* **threshold\_value** *(float, default 0.60)* — Soft‑threshold strength. Lower (0.50–0.55) = brighter/risk hiss; higher (0.65–0.75) = cleaner/risk dullness.
* **target\_bitrate\_kbps** *(int, default 1411)* — Used by Fat Llama to compute upscale factor.
* **toggle\_autoscale** *(bool, default true)* — Keep enabled. Prevents clipping and level jumps.
* **audio\_path / audio\_url** — Optional file/URL input.

**Outputs**

* **AUDIO** and a sidecar WAV/FLAC file.

**Notes**

* Normalization is **always enabled internally** for safety; adaptive filter is disabled (too slow for modest gains).
* Requires CuPy and CUDA runtime/NVRTC DLLs. Check your setup with:

  ```bash
  python -c "import cupy as cp; cp.show_config()"
  ```

---

### 🎛️ Spectral Enhance (Fat Llama — CPU/FFTW)

**Purpose**: Same algorithm as GPU, but portable **CPU/FFTW** implementation.

**Inputs**

* **target\_format, max\_iterations, threshold\_value, target\_bitrate\_kbps, toggle\_autoscale** — Same semantics as GPU node.
* **audio\_path / audio\_url** — Optional file/URL input.

**Outputs**

* **AUDIO** and a sidecar WAV/FLAC file.

**When to use**

* No discrete NVIDIA GPU, or your environment can’t satisfy CuPy/CUDA requirements.

---

## 🎼 Recommended Chains (Music‑centric)

* **Quick & Safe** → *FlashSR* → *Fat Llama (GPU/CPU) mild* → *(optional)* HQ downsample to 44.1 kHz.
* **Problematic transitions** → increase FlashSR **overlap\_seconds** toward **0.8–1.0 s**.
* **Noisy material** → enable **flashsr\_lowpass** and/or raise Fat Llama **threshold\_value** slightly.

---

## 🧰 Troubleshooting

**FlashSR can’t import / no weights**

* First run should auto‑download `student_ldm.pth`, `sr_vocoder.pth`, `vae.pth` into `models/audio/flashsr/`.
* Offline? Place those three files there manually.
* If you cloned the FlashSR repo separately, set `EGREGORA_FLASHSR_REPO` to its path.

**CuPy compilation error (e.g., `vector_types.h` missing)**

* Install matching CUDA runtime + NVRTC wheels:

  ```bash
  python -m pip install "nvidia-cuda-runtime-cu12==12.9.*" "nvidia-cuda-nvrtc-cu12==12.9.*" \
                           nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12
  python -m pip install cupy-cuda12x
  python -c "import cupy as cp; cp.show_config()"
  ```
* On Windows‑portable Python, the node adds DLL search dirs at runtime. Restart ComfyUI after installing wheels.

**Clicks at chunk boundaries**

* Raise **overlap\_seconds** (0.8–1.0 s). Longer overlaps increase compute but hide seams.

**Output too bright/dull**

* Fat Llama: lower **threshold\_value** for more bite; raise it for smoother/cleaner output.

---

## 🔒 Licenses & Attribution

**This repository (node code)**

* © mrgattax/egregoralabs. Licensed under the **MIT License** (see `LICENSE`).

**Weights & Third‑party projects (not included / auto‑downloaded)**

* **FlashSR\_Inference** and **FlashSR weights** belong to their respective authors. Check their repos / model cards for **license & usage terms** before commercial use.
* **fat‑llama** / **fat‑llama‑fftw** are third‑party packages. Refer to their PyPI/GitHub pages for license details.

> ⚠️ You are responsible for ensuring you have the rights to use the models/weights in your context (personal, research, commercial). When in doubt, review upstream licenses and model cards.

**Acknowledgements**

* FlashSR authors & contributors.
* Fat Llama authors & contributors.
* The ComfyUI community.

---

## 🧪 Dev Notes

* Env vars:

  * `EGREGORA_MODELS_DIR` → override models root (default: `ComfyUI/models`).
  * `EGREGORA_FLASHSR_REPO` → path to a local FlashSR repo (if not using the internal `deps/`).
* The FlashSR node writes sidecar WAV/FLAC files to `output/audio/` automatically; you can still place a dedicated Save‑Audio node if that suits your workflow.

---

## 🤝 Contributing

PRs and issues are welcome! Please include:

* OS / Python / ComfyUI version
* Repro steps and a short audio sample (if possible)

---

## 📜 Changelog

* **v0.1.0** — Initial release: FlashSR SR node, Fat Llama GPU/CPU.
