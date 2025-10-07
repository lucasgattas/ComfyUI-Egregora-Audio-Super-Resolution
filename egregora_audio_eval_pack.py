"""
Egregora · Audio Eval Pack (v1)
===============================

Permissive, model-friendly utilities to complement the Null Test Suite:
- ABX Prepare / ABX Judge (double‑blind listening helper)
- Loudness Meter (BS.1770-style*) + Gain Match (LUFS‑I / RMS)
- Metrics: SI‑SDR and LSD (log‑spectral distance)
- Resample Audio (HQ) with optional SciPy/torchaudio backends

*Note: The 1770 implementation here is a practical approximation for
  integrated loudness, momentary/short‑term, LRA, and true‑peak. For
  certification-grade measurement, validate against a reference meter.

All nodes follow ComfyUI conventions:
- AUDIO is a dict with {"waveform": torch.Tensor[B,C,T], "sample_rate": int}
- IMAGE is torch.Tensor[B,H,W,3] in [0,1]

License: MIT
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Optional deps
try:
    import scipy.signal as sps  # resample_poly, firwin
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import torchaudio
    import torchaudio.functional as AF
    _HAVE_TA = True
except Exception:
    _HAVE_TA = False


# -----------------------------
# Utilities
# -----------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_CN(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        if a.shape[0] > a.shape[1]:
            a = a.T
    else:
        t_axis = int(np.argmax(a.shape))
        a = np.moveaxis(a, t_axis, -1)
        C = int(np.prod(a.shape[:-1]))
        N = a.shape[-1]
        a = a.reshape(C, N)
    return a.astype(np.float32)


def make_audio(sr: int, samples_CN: np.ndarray, meta: Optional[dict] = None) -> Dict[str, Any]:
    s = _normalize_CN(samples_CN)
    wf = torch.from_numpy(s).unsqueeze(0)  # [1,C,N]
    return {
        "sr": int(sr),
        "sample_rate": int(sr),
        "samples": s,
        "waveform": wf,
        "meta": dict(meta or {}),
    }


def to_internal_audio(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict) and "waveform" in x and ("sample_rate" in x or "sr" in x or "rate" in x):
        sr = int(x.get("sample_rate") or x.get("sr") or x.get("rate"))
        wf = _to_numpy(x["waveform"])  # [B,C,T] or [C,T]
        if wf.ndim == 3:
            wf = wf[0]
        s = _normalize_CN(wf)
        return make_audio(sr, s, x.get("meta", {}))
    if isinstance(x, dict) and ("sr" in x or "sample_rate" in x):
        sr = int(x.get("sr") or x.get("sample_rate"))
        buf = x.get("samples") or x.get("audio") or x.get("array")
        if buf is None:
            raise ValueError("Audio dict missing samples/waveform")
        return make_audio(sr, _to_numpy(buf), x.get("meta", {}))
    raise ValueError("Unsupported AUDIO object for this node")


def _image_from_figure(fig) -> torch.Tensor:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    try:
        fig.clf()
    except Exception:
        pass
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    arr = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _rms_db(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return 10.0 * math.log10(float(np.mean(x * x) + 1e-20))


# -----------------------------
# 1770 Loudness helpers (practical approximations)
# -----------------------------

def _k_weight(sr: int, x_CN: np.ndarray) -> np.ndarray:
    """Very small K-weight approx: 1st-order HPF ~60 Hz + slight HF tilt.
    This is sufficient for relative matching; not certification-grade.
    """
    x = x_CN
    fc = 60.0 / (sr * 0.5)
    k = math.exp(-2 * math.pi * fc)
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        xn = x[c].astype(np.float32)
        yc = np.zeros_like(xn)
        z = 0.0
        for n in range(xn.shape[0]):
            z = (1 - k) * xn[n] + k * z
            yc[n] = xn[n] - z
        y[c] = yc
    # tiny HF shelf via first difference
    y[:, 1:] += 0.02 * (y[:, 1:] - y[:, :-1])
    return y


def integrated_lufs(audio: Dict[str, Any]) -> float:
    sr = audio["sample_rate"]
    y = _k_weight(sr, audio["samples"])  # [C,N]
    mono = y.mean(axis=0)
    blk = max(1, int(round(0.400 * sr)))
    hop = max(1, int(round(0.100 * sr)))
    frames = 1 + max(0, (mono.shape[0] - blk) // hop)
    if frames <= 0:
        return _rms_db(mono)
    ms = []
    for i in range(frames):
        s = i * hop
        e = s + blk
        seg = mono[s:e].astype(np.float64)
        ms.append(float(np.mean(seg * seg)))
    ms = np.asarray(ms) + 1e-20
    lufs_ungated = -0.691 + 10.0 * np.log10(np.mean(ms))
    gate = lufs_ungated - 10.0
    mask = (-0.691 + 10.0 * np.log10(ms)) >= gate
    if np.any(mask):
        ms = ms[mask]
    return float(-0.691 + 10.0 * np.log10(np.mean(ms)))


def lufs_series(audio: Dict[str, Any], window_s: float, hop_s: float) -> np.ndarray:
    sr = audio["sample_rate"]
    y = _k_weight(sr, audio["samples"]).mean(axis=0)
    w = max(1, int(round(window_s * sr)))
    h = max(1, int(round(hop_s * sr)))
    frames = 1 + max(0, (y.shape[0] - w) // h)
    out = np.empty((frames,), dtype=np.float32)
    for i in range(frames):
        s = i * h
        seg = y[s : s + w].astype(np.float64)
        out[i] = -0.691 + 10.0 * np.log10(float(np.mean(seg * seg)) + 1e-20)
    return out


def lra_short_term(audio: Dict[str, Any]) -> float:
    st = lufs_series(audio, 3.0, 1.0)  # 3s window, 1s hop (EBU R128)
    if st.size == 0:
        return 0.0
    # Simple gating: remove values near silence
    gate = np.percentile(st, 10.0) - 20.0
    pool = st[st > gate]
    if pool.size == 0:
        pool = st
    return float(np.percentile(pool, 95.0) - np.percentile(pool, 10.0))


def true_peak_dbfs(audio: Dict[str, Any], oversample: int = 4) -> float:
    x = audio["samples"].mean(axis=0)
    sr = audio["sample_rate"]
    if _HAVE_SCIPY:
        y = sps.resample_poly(x, oversample, 1)
    else:
        N = x.shape[0]
        t_old = np.linspace(0.0, 1.0, N, endpoint=False)
        t_new = np.linspace(0.0, 1.0, N * oversample, endpoint=False)
        y = np.interp(t_new, t_old, x).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    return 20.0 * math.log10(peak + 1e-20)


# -----------------------------
# ABX helper
# -----------------------------
@dataclass
class ABXMeta:
    x_is: str  # 'A' or 'B'
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {"x_is": self.x_is, "seed": int(self.seed)}


# -----------------------------
# Node: ABX Prepare
# -----------------------------
class ABX_Prepare:
    CATEGORY = "Egregora/Listening"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "DICT")
    RETURN_NAMES = ("audio_A", "audio_B", "audio_X", "abx_meta")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_A": ("AUDIO", {}),
                "audio_B": ("AUDIO", {}),
            },
            "optional": {
                "clip_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10_000.0, "step": 0.1}),
            },
        }

    def _clip(self, a: Dict[str, Any], start_s: float, dur_s: float) -> Dict[str, Any]:
        sr = a["sample_rate"]
        s = int(round(start_s * sr))
        n = int(round(dur_s * sr))
        x = a["samples"]
        if s + n > x.shape[1]:
            n = max(0, x.shape[1] - s)
        y = x[:, s : s + n]
        return make_audio(sr, y, a.get("meta", {}))

    def execute(self, audio_A, audio_B, clip_seconds=10.0, random_seed=0, start_seconds=0.0):
        A = to_internal_audio(audio_A)
        B = to_internal_audio(audio_B)
        n = min(A["samples"].shape[1], B["samples"].shape[1])
        A["samples"] = A["samples"][:, :n]
        B["samples"] = B["samples"][:, :n]

        A_c = self._clip(A, start_seconds, clip_seconds)
        B_c = self._clip(B, start_seconds, clip_seconds)

        rng = random.Random(int(random_seed))
        x_is = rng.choice(["A", "B"])
        X = A_c if x_is == "A" else B_c
        meta = ABXMeta(x_is=x_is, seed=int(random_seed)).to_dict()
        return A_c, B_c, X, meta


# -----------------------------
# Node: ABX Judge
# -----------------------------
class ABX_Judge:
    CATEGORY = "Egregora/Listening"
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("abx_result",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abx_meta": ("DICT", {}),
                "guess": (["A", "B"], {}),
            },
        }

    def execute(self, abx_meta, guess):
        x_is = str(abx_meta.get("x_is", "?")).upper()
        correct = (guess.upper() == x_is)
        return ({"x_is": x_is, "guess": guess.upper(), "correct": bool(correct)},)


# -----------------------------
# Node: Loudness Meter (1770)
# -----------------------------
class Loudness_Meter_1770:
    CATEGORY = "Egregora/Analysis"
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("metrics",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
            "optional": {
                "compute_true_peak": ("BOOLEAN", {"default": True}),
                "oversample": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            },
        }

    def execute(self, audio, compute_true_peak=True, oversample=4):
        a = to_internal_audio(audio)
        metrics: Dict[str, Any] = {}
        metrics["lufs_integrated"] = float(integrated_lufs(a))
        metrics["lufs_momentary"] = float(lufs_series(a, 0.400, 0.100).mean() if a["samples"].size else 0.0)
        metrics["lufs_short_term"] = float(lufs_series(a, 3.0, 1.0).mean() if a["samples"].size else 0.0)
        metrics["lra"] = float(lra_short_term(a))
        if compute_true_peak:
            metrics["true_peak_dbfs"] = float(true_peak_dbfs(a, oversample=int(oversample)))
        return (metrics,)


# -----------------------------
# Node: Audio Gain Match (1770 / RMS)
# -----------------------------
class Audio_Gain_Match_1770:
    CATEGORY = "Egregora/Analysis"
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio_matched", "gain_db", "ref_level", "in_level")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ref": ("AUDIO", {}),
                "audio_in": ("AUDIO", {}),
            },
            "optional": {
                "mode": (["LUFS-I", "RMS"], {}),
                "max_gain_db": ("FLOAT", {"default": 12.0, "min": -60.0, "max": 60.0, "step": 0.1}),
            },
        }

    def execute(self, audio_ref, audio_in, mode="LUFS-I", max_gain_db=12.0):
        ref = to_internal_audio(audio_ref)
        inn = to_internal_audio(audio_in)
        # resample if SR mismatch
        if inn["sample_rate"] != ref["sample_rate"]:
            sr_old = inn["sample_rate"]
            x = inn["samples"]
            C, N = x.shape
            new_N = int(round(N * ref["sample_rate"] / sr_old))
            t_old = np.linspace(0.0, 1.0, N, endpoint=False)
            t_new = np.linspace(0.0, 1.0, new_N, endpoint=False)
            y = np.stack([np.interp(t_new, t_old, x[c]) for c in range(C)], axis=0).astype(np.float32)
            inn = make_audio(ref["sample_rate"], y, inn.get("meta", {}))

        if str(mode).upper().startswith("LUFS"):
            ref_level = integrated_lufs(ref)
            in_level = integrated_lufs(inn)
        else:
            ref_level = _rms_db(ref["samples"].mean(axis=0))
            in_level = _rms_db(inn["samples"].mean(axis=0))
        gain_db = float(np.clip(ref_level - in_level, -abs(max_gain_db), abs(max_gain_db)))
        gain = 10 ** (gain_db / 20.0)
        y = (inn["samples"] * gain).astype(np.float32)
        out = make_audio(inn["sample_rate"], y, inn.get("meta", {}))
        return (out, float(gain_db), float(ref_level), float(in_level))


# -----------------------------
# Metrics: SI‑SDR & LSD
# -----------------------------

def _stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    mono = x if x.ndim == 1 else x.mean(axis=0)
    N = mono.shape[0]
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + max(0, (N - n_fft) // hop)
    S = np.empty((n_fft // 2 + 1, frames), dtype=np.float32)
    for i in range(frames):
        s = i * hop
        frame = mono[s : s + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]))
        X = np.fft.rfft(frame * win)
        S[:, i] = np.abs(X).astype(np.float32)
    return S


def _lsd(SA: np.ndarray, SB: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    LA = 20 * np.log10(SA + eps)
    LB = 20 * np.log10(SB + eps)
    D = (LA - LB) ** 2
    per = np.sqrt(np.mean(D, axis=0) + 1e-12)
    return float(np.mean(per)), float(np.percentile(per, 95))


def _si_sdr(s: np.ndarray, s_hat: np.ndarray) -> float:
    # operate on mono
    s = s.astype(np.float64)
    s_hat = s_hat.astype(np.float64)
    if s.ndim > 1:
        s = s.mean(axis=0)
    if s_hat.ndim > 1:
        s_hat = s_hat.mean(axis=0)
    # match length
    n = min(s.shape[-1], s_hat.shape[-1])
    s = s[:n]
    s_hat = s_hat[:n]
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + 1e-20)
    s_target = alpha * s
    e_noise = s_hat - s_target
    return 10.0 * np.log10((np.dot(s_target, s_target) + 1e-20) / (np.dot(e_noise, e_noise) + 1e-20))


class Metrics_LSD_SISDR:
    CATEGORY = "Egregora/Analysis"
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("metrics",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ref": ("AUDIO", {}),
                "audio_proc": ("AUDIO", {}),
            },
            "optional": {
                "n_fft": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 128}),
                "hop": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "compute_lsd": ("BOOLEAN", {"default": True}),
                "compute_si_sdr": ("BOOLEAN", {"default": True}),
            },
        }

    def execute(self, audio_ref, audio_proc, n_fft=2048, hop=512, compute_lsd=True, compute_si_sdr=True):
        A = to_internal_audio(audio_ref)
        B = to_internal_audio(audio_proc)
        a = A["samples"].mean(axis=0)
        b = B["samples"].mean(axis=0)
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
        out: Dict[str, Any] = {}
        if compute_lsd:
            SA = _stft_mag(a, n_fft=n_fft, hop=hop)
            SB = _stft_mag(b, n_fft=n_fft, hop=hop)
            lsd_mean, lsd_p95 = _lsd(SA, SB)
            out["lsd_mean_db"] = float(lsd_mean)
            out["lsd_p95_db"] = float(lsd_p95)
        if compute_si_sdr:
            out["si_sdr_db"] = float(_si_sdr(a, b))
        return (out,)


# -----------------------------
# Resample Audio (HQ)
# -----------------------------
class Resample_Audio_HQ:
    CATEGORY = "Egregora/Utils"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_out",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        modes = ["auto", "scipy_polyphase", "torchaudio", "linear"]
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "target_sr": ("INT", {"default": 48000, "min": 4000, "max": 384000, "step": 1}),
            },
            "optional": {
                "mode": (modes, {}),
                "kaiser_beta": ("FLOAT", {"default": 14.769, "min": 5.0, "max": 20.0, "step": 0.1}),
            },
        }

    def execute(self, audio, target_sr=48000, mode="auto", kaiser_beta=14.769):
        a = to_internal_audio(audio)
        src_sr = int(a["sample_rate"])
        if src_sr == int(target_sr):
            return (a,)
        x = a["samples"]  # [C,N]
        C, N = x.shape
        if mode == "auto":
            mode = "scipy_polyphase" if _HAVE_SCIPY else ("torchaudio" if _HAVE_TA else "linear")
        if mode == "scipy_polyphase" and _HAVE_SCIPY:
            # rational ratio
            from math import gcd
            g = gcd(src_sr, int(target_sr))
            up = int(target_sr) // g
            down = src_sr // g
            y = np.stack([sps.resample_poly(x[c], up, down) for c in range(C)], axis=0).astype(np.float32)
        elif mode == "torchaudio" and _HAVE_TA:
            wf = torch.from_numpy(x).unsqueeze(0)  # [1,C,N]
            y = AF.resample(wf, src_sr, int(target_sr), lowpass_filter_width=64, rolloff=0.945, resampling_method="kaiser_window", beta=kaiser_beta)
            y = y.squeeze(0).detach().cpu().numpy().astype(np.float32)
        else:
            # fallback: linear interp
            new_N = int(round(N * (int(target_sr) / src_sr)))
            t_old = np.linspace(0.0, 1.0, N, endpoint=False)
            t_new = np.linspace(0.0, 1.0, new_N, endpoint=False)
            y = np.stack([np.interp(t_new, t_old, x[c]) for c in range(C)], axis=0).astype(np.float32)
        return (make_audio(int(target_sr), y, a.get("meta", {})),)


# -----------------------------
# Registration
# -----------------------------
NODE_CLASS_MAPPINGS = {
    "ABX Prepare": ABX_Prepare,
    "ABX Judge": ABX_Judge,
    "Loudness Meter (BS1770)": Loudness_Meter_1770,
    "Audio Gain Match (1770)": Audio_Gain_Match_1770,
    "Metrics (LSD + SI-SDR)": Metrics_LSD_SISDR,
    "Resample Audio (HQ)": Resample_Audio_HQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ABX Prepare": "Egregora ABX Prepare",
    "ABX Judge": "Egregora ABX Judge",
    "Loudness Meter (BS1770)": "Egregora Loudness Meter (BS1770)",
    "Audio Gain Match (1770)": "Egregora Audio Gain Match (1770)",
    "Metrics (LSD + SI-SDR)": "Egregora Metrics (LSD + SI-SDR)",
    "Resample Audio (HQ)": "Egregora Resample Audio (HQ)",
}
