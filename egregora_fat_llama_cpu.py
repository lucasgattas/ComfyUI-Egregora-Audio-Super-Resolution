import time, tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import torch

RETURN_TYPES = ("AUDIO",)
FUNCTION = "run"
CATEGORY = "Egregora/Audio"

def _to_cs(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        h, w = a.shape
        if w <= 8 and h > w:  # [S,C] -> [C,S]
            a = a.T
    else:
        a = a.reshape(-1)[None, :]
    m = np.max(np.abs(a)) if a.size else 0.0
    if m > 1.0:
        a = a / (m + 1e-8)
    return a.astype(np.float32)

def _save_temp_wav(cs: np.ndarray, sr: int) -> Path:
    p = Path(tempfile.gettempdir()) / f"eg_in_{int(time.time()*1000)}.wav"
    sf.write(str(p), cs.T, int(sr))
    return p

def _normalize_audio_input(AUDIO=None, audio_path: str="", audio_url: str="") -> Tuple[np.ndarray, int, Path]:
    if isinstance(AUDIO, dict) and "waveform" in AUDIO and "sample_rate" in AUDIO:
        wf: torch.Tensor = AUDIO["waveform"]
        sr = int(AUDIO["sample_rate"])
        if wf.dim() == 3:
            wf = wf[0]
        if wf.dim() != 2:
            raise RuntimeError(f"Unexpected AUDIO tensor shape: {tuple(wf.shape)} (want [C,T])")
        cs = wf.detach().cpu().float().numpy()
        return cs, sr, _save_temp_wav(cs, sr)

    if isinstance(AUDIO, (list, tuple)) and len(AUDIO) == 2:
        arr, sr = AUDIO
        cs = _to_cs(np.asarray(arr))
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    if audio_path:
        p = Path(audio_path)
        if not p.exists():
            raise RuntimeError(f"audio_path not found: {audio_path}")
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    if audio_url:
        import requests
        r = requests.get(audio_url, timeout=60); r.raise_for_status()
        p = Path(tempfile.gettempdir()) / f"eg_url_{int(time.time()*1000)}.wav"
        p.write_bytes(r.content)
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        return cs, int(sr), _save_temp_wav(cs, int(sr))

    raise RuntimeError("No AUDIO provided.")

def _ensure_cpu_pkg():
    try:
        import fat_llama_fftw  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: fat-llama-fftw. "
            "Install into ComfyUI's Python: `python -m pip install fat-llama-fftw`."
        ) from e

def _fat_llama_fftw_upscale(
    in_wav: Path,
    out_path: Path,
    target_format: str,
    max_iterations: int,
    threshold_value: float,
    target_bitrate_kbps: int,
):
    # Public API (CPU): from fat_llama_fftw.audio_fattener.feed import upscale
    # Example call & params documented in README/example.py.  :contentReference[oaicite:1]{index=1}
    from fat_llama_fftw.audio_fattener.feed import upscale  # type: ignore
    upscale(
        input_file_path=str(in_wav),
        output_file_path=str(out_path),
        source_format="wav",
        target_format=target_format,
        max_iterations=int(max_iterations),
        threshold_value=float(threshold_value),
        target_bitrate_kbps=int(target_bitrate_kbps),
    )

class EgregoraFatLlamaCPU:
    """
    Spectral Enhance (Fat Llama — CPU/FFTW)
    — Pure CPU path using pyFFTW backend; no CUDA/CuPy required.
    — If you feed non-WAV inputs via path/URL, ffmpeg on PATH may be required by the package.  :contentReference[oaicite:2]{index=2}
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_format": (["wav", "flac"],),
                "max_iterations": ("INT", {"default": 800, "min": 1, "max": 10000}),
                "threshold_value": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_bitrate_kbps": ("INT", {"default": 1411, "min": 64, "max": 5000}),
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
        AUDIO=None,
        audio_path="",
        audio_url="",
    ):
        _ensure_cpu_pkg()

        cs, in_sr, in_wav = _normalize_audio_input(AUDIO, audio_path, audio_url)
        suffix = ".wav" if target_format == "wav" else ".flac"
        out_path = Path(tempfile.gettempdir()) / f"eg_fatllama_cpu_{int(time.time()*1000)}{suffix}"

        _fat_llama_fftw_upscale(
            in_wav=in_wav,
            out_path=out_path,
            target_format=target_format,
            max_iterations=max_iterations,
            threshold_value=threshold_value,
            target_bitrate_kbps=target_bitrate_kbps,
        )

        y, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
        cs_out = _to_cs(y)
        wf = torch.from_numpy(cs_out).unsqueeze(0).contiguous()  # [1,C,T]
        return ({"waveform": wf, "sample_rate": int(sr)},)

NODE_CLASS_MAPPINGS = {"EgregoraFatLlamaCPU": EgregoraFatLlamaCPU}
NODE_DISPLAY_NAME_MAPPINGS = {"EgregoraFatLlamaCPU": "🎛️ Spectral Enhance (Fat Llama — CPU/FFTW)"}