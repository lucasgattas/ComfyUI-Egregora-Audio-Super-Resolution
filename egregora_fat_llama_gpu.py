import os
import sys
import time
import tempfile
import platform
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import torch

RETURN_TYPES = ("AUDIO",)
FUNCTION = "run"
CATEGORY = "Egregora/Audio"

# ---------------- I/O helpers ----------------

def _to_cs(x: np.ndarray) -> np.ndarray:
    """Return channels-first float32 [C,S]; accepts [S], [S,C], [C,S]."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        h, w = a.shape
        if w <= 8 and h > w:  # soundfile often returns [S,C]
            a = a.T
    else:
        a = a.reshape(-1)[None, :]
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if m > 1.0:  # safety clamp if upstream sent > 1.0
        a = a / (m + 1e-8)
    return a.astype(np.float32)

def _save_temp_wav(cs: np.ndarray, sr: int) -> Path:
    p = Path(tempfile.gettempdir()) / f"eg_in_{int(time.time()*1000)}.wav"
    sf.write(str(p), cs.T, int(sr))
    return p

def _normalize_audio_input(AUDIO=None, audio_path: str = "", audio_url: str = "") -> Tuple[np.ndarray, int, Path]:
    """
    Accept ComfyUI AUDIO dict, or a file path/url; return ([C,S], sr, temp_wav_path).
    """
    # ComfyUI's AUDIO: {"waveform": [B,C,T], "sample_rate": sr}
    if isinstance(AUDIO, dict) and "waveform" in AUDIO and "sample_rate" in AUDIO:
        wf: torch.Tensor = AUDIO["waveform"]
        sr = int(AUDIO["sample_rate"])
        if wf.dim() == 3:
            wf = wf[0]  # [C,T]
        if wf.dim() != 2:
            raise RuntimeError(f"Unexpected AUDIO tensor shape: {tuple(wf.shape)} (want [C,T])")
        cs = wf.detach().cpu().float().numpy()
        return cs, sr, _save_temp_wav(cs, sr)

    # (arr, sr) tuple
    if isinstance(AUDIO, (list, tuple)) and len(AUDIO) == 2:
        arr, sr = AUDIO
        cs = _to_cs(np.asarray(arr))
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    # explicit file path
    if audio_path:
        p = Path(audio_path)
        if not p.exists():
            raise RuntimeError(f"audio_path not found: {audio_path}")
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    # URL fetch
    if audio_url:
        import requests
        r = requests.get(audio_url, timeout=60); r.raise_for_status()
        p = Path(tempfile.gettempdir()) / f"eg_url_{int(time.time()*1000)}.wav"
        p.write_bytes(r.content)
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    raise RuntimeError("No AUDIO provided.")

# ---------------- CUDA/CuPy wiring (Windows) ----------------

def _wire_cuda_for_cupy_windows():
    """
    On Windows portable installs, make NVIDIA pip-wheel DLLs & headers discoverable:
      • Add ...\site-packages\nvidia\<package>\bin to the DLL search path
      • Point CUDA_PATH to ...\site-packages\nvidia\cuda_runtime (has include/)
    Must run BEFORE importing cupy.
    """
    if platform.system() != "Windows":
        return

    sp = Path(sys.executable).parent / "Lib" / "site-packages" / "nvidia"
    rt = sp / "cuda_runtime"     # contains include/ and bin/
    nvrtc = sp / "cuda_nvrtc"    # contains bin/

    # Let CuPy find headers at runtime (NVRTC needs CUDA runtime headers >= CUDA 12.2)
    if rt.exists():
        os.environ.setdefault("CUDA_PATH", str(rt))

    # Make DLLs loadable for this process (Python 3.8+)
    for p in (rt / "bin", nvrtc / "bin"):
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                os.environ["PATH"] = f"{str(p)};{os.environ.get('PATH','')}"

# ---------------- Fat Llama wrapper ----------------

def _ensure_gpu_stack():
    """
    Validate CUDA/CuPy presence early and give a friendly error if not available.
    Also ensure DLL search paths & headers are wired so CuPy can load cudart/nvrtc
    and find CUDA runtime headers like vector_types.h.
    """
    _wire_cuda_for_cupy_windows()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Fat Llama (GPU) requires an NVIDIA GPU. "
            "If you need CPU, use the separate Fat Llama — CPU/FFTW node."
        )

    try:
        import cupy  # noqa: F401  (import after wiring)
    except Exception as e:
        raise RuntimeError(
            "CuPy failed to import. Ensure you've installed a CUDA-12 build "
            "(`pip install cupy-cuda12x`) and matching NVIDIA runtime headers & NVRTC "
            "(`pip install \"nvidia-cuda-runtime-cu12==12.X.*\" \"nvidia-cuda-nvrtc-cu12==12.X.*\"`)."
        ) from e

def _fat_llama_upscale(
    in_wav: Path,
    out_path: Path,
    target_format: str,
    max_iterations: int,
    threshold_value: float,
    target_bitrate_kbps: int,
    toggle_autoscale: bool,
):
    """Call the public API: fat_llama.audio_fattener.feed.upscale(...)"""
    from fat_llama.audio_fattener.feed import upscale  # late import

    # Normalize ALWAYS on; Adaptive filter disabled for perf/stability
    upscale(
        input_file_path=str(in_wav),
        output_file_path=str(out_path),
        source_format="wav",
        target_format=target_format,
        max_iterations=int(max_iterations),
        threshold_value=float(threshold_value),
        target_bitrate_kbps=int(target_bitrate_kbps),
        toggle_normalize=True,
        toggle_autoscale=bool(toggle_autoscale),
        toggle_adaptive_filter=False,
    )

# ---------------- ComfyUI Node ----------------

class EgregoraFatLlamaGPU:
    """
    Spectral Enhance (Fat Llama — GPU only)
    - Normalize is always ON (clamps final amplitude and prevents clipping).
    - Adaptive filter disabled for speed (still available in library if you want a "slow" node).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_format": (["wav", "flac"],),
                "max_iterations": ("INT", {"default": 300, "min": 1, "max": 5000}),
                "threshold_value": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_bitrate_kbps": ("INT", {"default": 1411, "min": 64, "max": 5000}),
                "toggle_autoscale": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "AUDIO": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY
    OUTPUT_NODE = False

    def run(
        self,
        target_format,
        max_iterations,
        threshold_value,
        target_bitrate_kbps,
        toggle_autoscale,
        AUDIO=None,
        audio_path="",
        audio_url="",
    ):
        _ensure_gpu_stack()

        # Normalize inbound audio to a temp WAV we can hand to fat_llama
        cs, in_sr, in_wav = _normalize_audio_input(AUDIO, audio_path, audio_url)

        # Choose an output temp path with chosen container
        suffix = ".wav" if target_format == "wav" else ".flac"
        out_path = Path(tempfile.gettempdir()) / f"eg_fatllama_{int(time.time()*1000)}{suffix}"

        # Run Fat Llama with always-on normalization and no adaptive filter
        _fat_llama_upscale(
            in_wav=in_wav,
            out_path=out_path,
            target_format=target_format,
            max_iterations=max_iterations,
            threshold_value=threshold_value,
            target_bitrate_kbps=target_bitrate_kbps,
            toggle_autoscale=toggle_autoscale,
        )

        # Read result back into Comfy
        y, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
        cs_out = _to_cs(y)
        wf = torch.from_numpy(cs_out).unsqueeze(0).contiguous()  # [1,C,T]
        return ({"waveform": wf, "sample_rate": int(sr)},)

# Register node
NODE_CLASS_MAPPINGS = {
    "EgregoraFatLlamaGPU": EgregoraFatLlamaGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraFatLlamaGPU": "🎛️ Spectral Enhance (Fat Llama — GPU)",
}