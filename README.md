# 🎧 ComfyUI — Egregora Audio Super‑Resolution

Bring music up to studio‑grade sample‑rates right inside ComfyUI. This repo ships **three minimal, production‑oriented nodes**:

* **Audio Super Resolution (FlashSR)** — one‑step diffusion upsampler (music‑friendly) ⚡
* **Spectral Enhance (Fat Llama — GPU)** — CUDA/CuPy accelerated iterative spectral enhancer 🐍🧪
* **Spectral Enhance (Fat Llama — CPU/FFTW)** — portable CPU fallback using pyFFTW 🧠

Each node is designed to be simple, robust, and to play nicely with ComfyUI’s file I/O.

---

## ✨ What’s inside

```
custom_nodes/
  ComfyUI-Egregora-Audio-Super-Resolution/
    __init__.py
    egregora_audio_super_resolution.py   # FlashSR node
    egregora_fat_llama_gpu.py            # Fat Llama (CUDA)
    egregora_fat_llama_cpu.py            # Fat Llama (FFTW)
    flashsr_min.py                       # Light wrapper for FlashSR
    install.py                           # Repo + weights bootstrapper
    requirements.txt
    deps/
      FlashSR_Inference/                 # pulled automatically on install
```

---

## 🧩 Install (ComfyUI portable or venv)

1. **Drop the folder** into `ComfyUI/custom_nodes/` and restart ComfyUI once.

2. **Install Python deps** (ComfyUI’s Python):

```bash
# From ComfyUI root
python -m pip install -r custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/requirements.txt
```

3. **FlashSR repo & weights**

* The node bootstraps the upstream code automatically:

  * Downloads `FlashSR_Inference` zip to `deps/FlashSR_Inference`.
  * Looks for weights at `models/audio/flashsr/` with exact names:

    * `student_ldm.pth`, `sr_vocoder.pth`, `vae.pth`.
* If you want the node to pull weights from your Hugging Face repo, set:

```bash
# optional: point to your private/public HF repo containing the three files
set EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights   # Windows (cmd)
export EGREGORA_FLASHSR_HF_REPO=yourname/flashsr-weights # macOS/Linux
```

> Tip: you can also place the files manually in `ComfyUI/models/audio/flashsr/`.

4. **GPU extras (only for the CUDA Fat‑Llama node)**

* Install a CuPy wheel that matches your CUDA runtime (e.g., CUDA 12):

```bash
python -m pip install "cupy-cuda12x>=13.0"
```

* On Windows, if you hit `vector_types.h` / NVRTC errors, install NVIDIA runtime DLLs:

```bash
python -m pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12
```

* You also need **FFmpeg** on PATH for reading/encoding audio. The simplest path:

  * Windows: `winget install --id Gyan.FFmpeg.Full` or install via \[ffmpeg.org].
  * macOS: `brew install ffmpeg`
  * Linux: `sudo apt-get install ffmpeg`

---

## 📦 Requirements

`requirements.txt` keeps things lean:

* `soundfile`, `numpy`, `tqdm`, `requests`, `huggingface_hub`
* `fat-llama` (GPU node) and `fat-llama-fftw` (CPU node)

Install via the command in the Install section above.

---

## 🛠️ Nodes & settings

### 1) **Audio Super Resolution (FlashSR)**

Music‑friendly diffusion upsampler. Chunks the waveform, overlaps, and stitches seamlessly.

**Inputs**

* `chunk_seconds` *(float, default 5.12)* — window size per pass. 5.12s matches upstream training length; stick with it for best quality.
* `overlap_seconds` *(float, default 0.50)* — crossfade overlap; increase a bit (0.5–0.75) if you hear seams.
* `device` *(auto|cpu|cuda)* — pick GPU for speed if available.
* `target_sr` *(int|auto)* — 48 000 is common for music. `auto` keeps input rate when appropriate.
* `output_format` *(wav|flac)* — select container/codec for the node’s file output.
* `audio_path` / `audio_url` — optional direct file/URL sources.
* `flashsr_lowpass` *(bool)* — applies a light LPF to better match training distribution; toggle if highs feel brittle.

**Outputs**

* **AUDIO** — 48 kHz (or chosen) waveform buffer + file on disk in your chosen format.

**Notes**

* The node will use `models/audio/flashsr/*.pth` automatically.
* On first run, upstream repo is pulled into `deps/FlashSR_Inference/`.

---

### 2) **Spectral Enhance (Fat Llama — GPU)**

Fast CuPy/CUDA implementation of iterative soft‑thresholding + spectral post.

**Inputs**

* `target_format` *(wav|flac)* — output container/codec.
* `max_iterations` *(int, default 300)* — more = longer + potentially more over‑sharpening. 300–600 is a good range.
* `threshold_value` *(0–1, default 0.60)* — lower = gentler, higher = sparser/more aggressive.
* `target_bitrate_kbps` *(int, default 1411)* — used to choose an upscale factor; set near CD PCM (1411) for full‑band music.
* `toggle_autoscale` *(bool, default true)* — auto headroom gain to avoid clipping.
* `audio_path` / `audio_url` — optional sources.

**Outputs**

* **AUDIO** — processed waveform + rendered file.

**GPU gotchas**

* If CuPy compiles kernels at first run, allow a few seconds.
* If you see NVRTC / missing DLL errors on Windows, follow the CUDA extras in **Install**.

---

### 3) **Spectral Enhance (Fat Llama — CPU/FFTW)**

Drop‑in replacement when you don’t have a compatible GPU.

**Inputs** are the same as the GPU node.

**Performance**: expect \~×3–×10 slower than CUDA depending on CPU and audio length.

---

## 🎚️ Quality tips (music)

* **FlashSR first, Llama second**: For low‑rate sources (e.g., 12–24 kHz or lossy), run FlashSR to 48 kHz, then do a light Fat Llama pass (`max_iterations≈200`, `threshold_value≈0.5`) if you want a touch more sparkle.
* **Crossfade overlap**: if you hear subtle ticks between chunks, nudge `overlap_seconds` up by 0.1–0.2.
* **Don’t over‑iterate**: very high iterations or high threshold can sound brittle/phasey.

---

## 🔍 Licenses (upstream projects)

* **Fat‑Llama (GPU)**: published on PyPI under **BSD‑3‑Clause**. See PyPI page for details.
  Ref: fat‑llama on PyPI.
* **Fat‑Llama‑FFTW (CPU)**: published on PyPI under **BSD‑3‑Clause**. See PyPI page for details.
  Ref: fat‑llama‑fftw on PyPI.
* **FlashSR\_Inference** (upstream code): as of this writing, the repository does **not** include a LICENSE file in the root. Check the upstream README/commits for updates.

> This ComfyUI integration code is distributed under the license declared in this repository.

---

## 🧪 Troubleshooting

* **FlashSR import error**: After installing the node, restart ComfyUI so `install.py` can clone `FlashSR_Inference` and create `deps/FlashSR_Inference/`. If you keep seeing the error, remove `deps/FlashSR_Inference/` and restart.
* **Missing FlashSR weights**: Put `student_ldm.pth`, `sr_vocoder.pth`, `vae.pth` in `models/audio/flashsr/` or set `EGREGORA_FLASHSR_HF_REPO` and restart.
* **Windows CUDA compile / `vector_types.h`**: install the `nvidia-*‑cu12` wheels listed in **Install → GPU extras** and ensure your CuPy wheel matches CUDA.
* **FFmpeg not found**: install FFmpeg and ensure it’s on PATH (see **Install**). Most issues with reading MP3/FLAC are solved by a system FFmpeg.

---

## 🙌 Credits

* FlashSR research & inference code by the original authors. See upstream repo.
* Fat Llama packages by RaAd (PyPI maintainer).
* ComfyUI integration & node UX by Egregora.

Happy upsampling! 🎶

---

## 📜 Changelog

* **v0.1.0** — Initial release: FlashSR SR node, Fat Llama GPU/CPU.
