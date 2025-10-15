# 🎧 ComfyUI — Audio Super Resolution (FlashSR)
# Minimal, single-output node with robust shapes and HQ resampling.
# Inputs:  audio (AUDIO), lowpass_input (BOOL), output_sr (enum)
# Output:  audio (AUDIO)
#
# Internals:
# - Normalize to [C, S] consistently (soundfile returns [S, C] -> transpose)
# - Fixed chunking: 5.12 s, overlap: 0.50 s, Hann WOLA stitching
# - Inference at 48 kHz (FlashSR’s design target), optional post-resample
# - HQ SRC cascade: soxr -> scipy.signal.resample_poly -> torchaudio -> linear
#
# SPDX: MIT

import os, sys, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch

FUNCTION = "run"
CATEGORY = "Egregora/Audio"

# ---------- paths ----------
def _custom_root() -> Path:
    return Path(__file__).resolve().parent

def _models_dir() -> Path:
    # .../ComfyUI/models
    return _custom_root().parents[2] / "models"

def _audio_models_subdir(name: str) -> Path:
    d = _models_dir() / "audio" / name
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------- AUDIO helpers ----------
def _make_audio(sr: int, samples_cs: np.ndarray) -> Dict[str, Any]:
    """Build a ComfyUI AUDIO dict from [C, S] float32."""
    s = np.asarray(samples_cs, dtype=np.float32)
    if s.ndim == 1:
        s = s[None, :]
    C, T = s.shape
    wf = torch.from_numpy(s).unsqueeze(0).contiguous()  # [1, C, T]
    return {"waveform": wf, "sample_rate": int(sr)}

def _from_audio_dict(AUDIO: Any) -> Tuple[np.ndarray, int]:
    """
    Accept Comfy AUDIO dict or (ndarray, sr). Return [C, S] float32 and sr.
    """
    # Comfy AUDIO dict
    if isinstance(AUDIO, dict) and "waveform" in AUDIO and "sample_rate" in AUDIO:
        wf: torch.Tensor = AUDIO["waveform"]
        sr = int(AUDIO["sample_rate"])
        if wf.dim() == 3:
            wf = wf[0]  # [C, T]
        if wf.dim() != 2:
            raise RuntimeError(f"Unexpected AUDIO tensor shape {tuple(wf.shape)}; expected [C, T].")
        cs = wf.detach().cpu().float().numpy()  # [C, T]
        return cs, sr
    # (array, sr)
    if isinstance(AUDIO, (list, tuple)) and len(AUDIO) == 2:
        arr, sr = AUDIO
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            # mono [S] -> [1, S]
            cs = arr[None, :]
        elif arr.ndim == 2:
            # could be [S, C] or [C, S]; treat 1st dim as frames if it's much larger
            if arr.shape[0] >= arr.shape[1] and arr.shape[1] <= 8:
                # soundfile/frames-first -> transpose to [C, S]
                cs = arr.T
            else:
                cs = arr  # already [C, S]
        else:
            cs = arr.reshape(1, -1)
        return cs.astype(np.float32), int(sr)
    raise RuntimeError("No valid AUDIO provided.")

# ---------- HQ resampling ----------
def _resample_hq(x_cs: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Prefer soxr -> scipy.signal.resample_poly -> torchaudio -> linear.
    Operates on [C, S] along the sample axis.
    """
    if src_sr == dst_sr:
        return x_cs.astype(np.float32)

    # soxr
    try:
        import soxr  # type: ignore
        out = [soxr.resample(x_cs[c], src_sr, dst_sr) for c in range(x_cs.shape[0])]
        # equalize length (guard)
        L = min(map(len, out))
        out = np.stack([ch[:L] for ch in out], axis=0)
        return out.astype(np.float32)
    except Exception:
        pass

    # SciPy polyphase
    try:
        from math import gcd
        from scipy.signal import resample_poly  # type: ignore
        g = gcd(src_sr, dst_sr)
        up, down = dst_sr // g, src_sr // g
        out = [resample_poly(x_cs[c], up=up, down=down).astype(np.float32) for c in range(x_cs.shape[0])]
        L = min(map(len, out))
        out = np.stack([ch[:L] for ch in out], axis=0)
        return out
    except Exception:
        pass

    # torchaudio windowed-sinc
    try:
        import torchaudio  # type: ignore
        t = torch.from_numpy(x_cs).float()  # [C, S]
        rs = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
        y = rs(t)  # [C, S']
        return y.numpy().astype(np.float32)
    except Exception:
        pass

    # linear interp fallback (lowest quality)
    ratio = dst_sr / float(src_sr)
    n_out = int(round(x_cs.shape[1] * ratio))
    t_in = np.linspace(0.0, 1.0, x_cs.shape[1], endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float64)
    out = np.stack([np.interp(t_out, t_in, ch) for ch in x_cs], axis=0).astype(np.float32)
    return out

# ---------- chunking & WOLA ----------
def _hann(L: int) -> np.ndarray:
    return np.hanning(L).astype(np.float32)

def _iter_chunks(total_samples: int, win: int, hop: int) -> List[Tuple[int, int]]:
    """
    Yield (start, length) for each chunk to cover [0, total_samples).
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < total_samples:
        L = min(win, total_samples - i)
        spans.append((i, L))
        if i + L >= total_samples:
            break
        i += hop
    return spans

def _wola_stitch(chunks_pred: List[Tuple[np.ndarray, int, int]], total_len: int, win: int) -> np.ndarray:
    """
    Overlap-add predicted chunks with Hann window.
    chunks_pred: list of (pred_cs [C, L_pred], start, L_in)
                 L_in = original (unpadded) input length for that chunk
    Returns [C, total_len].
    """
    if not chunks_pred:
        return np.zeros((1, max(1, total_len)), np.float32)

    C = chunks_pred[0][0].shape[0]
    acc = np.zeros((C, total_len), np.float32)
    wsum = np.zeros(total_len, np.float32)
    w_full = _hann(win)

    for y_cs, start, L_in in chunks_pred:
        L_pred = y_cs.shape[1]
        L = min(L_in, L_pred)  # only weight the valid (unpadded) part
        w = w_full[:L] if L <= win else np.ones(L, np.float32)
        acc[:, start:start+L] += y_cs[:, :L] * w[None, :]
        wsum[start:start+L] += w

    wsum[wsum == 0] = 1.0
    out = acc / wsum[None, :]
    return out.astype(np.float32)

# ---------- FlashSR loader ----------
class _FlashSRRunner:
    REQ_SR = 48000
    CHUNK_S = 5.12
    OVERLAP_S = 0.50
    CHUNK_SAMPLES = int(REQ_SR * CHUNK_S)  # 245760

    HF_DATASET = "jakeoneijk/FlashSR_weights"
    HF_FILES = ("student_ldm.pth", "sr_vocoder.pth", "vae.pth")

    def __init__(self, lowpass: bool = False):
        self.lowpass = bool(lowpass)
        self.ckpt_dir = _audio_models_subdir("flashsr")
        self.repo_path = self._resolve_repo_path()
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._FlashSRClass = None
        self._model = None
        self._ensure_weights()
        self._import()
        self._ensure_model()

    def _resolve_repo_path(self) -> Path:
        env_repo = os.environ.get("EGREGORA_FLASHSR_REPO")
        if env_repo:
            return Path(env_repo)
        # default: custom_nodes/ComfyUI-Egregora-Audio-Super-Resolution/deps/FlashSR_Inference
        return _custom_root().parents[0] / "deps" / "FlashSR_Inference"

    def _ensure_weights(self):
        missing = [f for f in self.HF_FILES if not (self.ckpt_dir / f).exists()]
        if not missing:
            return
        # Try huggingface_hub first
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            for fname in missing:
                hf_hub_download(
                    repo_id=self.HF_DATASET,
                    filename=fname,
                    repo_type="dataset",
                    local_dir=str(self.ckpt_dir),
                )
            print(f"[FlashSR] Downloaded via huggingface_hub: {', '.join(missing)}")
            return
        except Exception as e:
            print(f"[FlashSR] huggingface_hub unavailable or failed ({e}); falling back to direct HTTP…")
        # Fallback: direct HTTP
        try:
            import requests  # type: ignore
            for fname in missing:
                url = f"https://huggingface.co/datasets/{self.HF_DATASET}/resolve/main/{fname}?download=true"
                dst = self.ckpt_dir / fname
                with requests.get(url, stream=True, timeout=1800) as r:
                    r.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                print(f"[FlashSR] Downloaded: {dst}")
        except Exception as ee:
            raise RuntimeError(
                "FlashSR weights missing and auto-download failed. "
                "Place these in models/audio/flashsr: student_ldm.pth, sr_vocoder.pth, vae.pth"
            ) from ee

    def _import(self):
        if self._FlashSRClass is not None:
            return
        try:
            from FlashSR.FlashSR import FlashSR  # type: ignore
            self._FlashSRClass = FlashSR
            return
        except Exception:
            cand = self.repo_path
            if (cand / "FlashSR").exists():
                sys.path.insert(0, str(cand))
                from FlashSR.FlashSR import FlashSR  # type: ignore
                self._FlashSRClass = FlashSR
                return
        raise RuntimeError("FlashSR module not found. Install/clone and set EGREGORA_FLASHSR_REPO if needed.")

    def _ensure_model(self):
        if self._model is not None:
            return
        FlashSR = self._FlashSRClass
        s = str(self.ckpt_dir / "student_ldm.pth")
        v = str(self.ckpt_dir / "sr_vocoder.pth")
        vae = str(self.ckpt_dir / "vae.pth")
        model = FlashSR(s, v, vae)
        model.eval()
        try:
            model.to(self._dev)
        except Exception:
            pass
        self._model = model

    def infer(self, x_cs_48k: np.ndarray) -> np.ndarray:
        """
        x_cs_48k: [C, S] float32 at 48 kHz.
        Returns [C, S] float32 at 48 kHz (same length as input slice passed in).
        """
        x = torch.from_numpy(x_cs_48k).to(self._dev).float()  # [C, S]
        with torch.inference_mode():
            y = self._model(x, lowpass_input=self.lowpass)  # [C, S]
        return y.detach().to("cpu").float().numpy()

# ---------- Node ----------
class EgregoraAudioSuperResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "lowpass_input": ("BOOLEAN", {"default": False}),
                "output_sr": (["48000", "44100", "96000"], {"default": "48000"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY
    OUTPUT_NODE = False

    def run(self, audio=None, lowpass_input=False, output_sr="48000"):
        # 1) Normalize input to [C, S]
        in_cs, in_sr = _from_audio_dict(audio)

        # 2) Resample to model SR if needed
        runner = _FlashSRRunner(lowpass=bool(lowpass_input))
        req_sr = runner.REQ_SR
        if in_sr != req_sr:
            in_cs = _resample_hq(in_cs, in_sr, req_sr)
            in_sr = req_sr

        # 3) Chunking params (internal, non-user)
        win = runner.CHUNK_SAMPLES               # 5.12 s @ 48k
        hop = int((runner.CHUNK_S - runner.OVERLAP_S) * req_sr)
        if hop <= 0 or hop >= win:
            # guard-rail: keep a sane overlap in pathological cases
            hop = win // 2

        total = in_cs.shape[1]
        spans = _iter_chunks(total, win=win, hop=hop)

        # 4) Process chunks in-memory and stitch with Hann WOLA
        preds: List[Tuple[np.ndarray, int, int]] = []
        for start, L in spans:
            # slice and pad up to win
            chunk = in_cs[:, start:start+L]
            if L < win:
                pad = np.zeros((in_cs.shape[0], win - L), np.float32)
                chunk = np.concatenate([chunk, pad], axis=1)
            y_pred = runner.infer(chunk)  # [C, win] @ 48k
            preds.append((y_pred, start, L))  # keep original L for proper weighting

        out_48k = _wola_stitch(preds, total_len=total, win=win)  # [C, total]

        # 5) Optional post-resample for delivery
        tgt_sr = int(output_sr)
        if tgt_sr != in_sr:
            out = _resample_hq(out_48k, in_sr, tgt_sr)
            out_sr = tgt_sr
        else:
            out, out_sr = out_48k, in_sr

        # 6) Return single AUDIO
        return (_make_audio(out_sr, out),)

# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "EgregoraAudioUpscaler": EgregoraAudioSuperResolution,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraAudioUpscaler": "🎧 Audio Super Resolution (FlashSR)",
}
