"""
Egregora · Null Test Suite for ComfyUI (v5)
===========================================

This version fixes the UI toggles by using ComfyUI's BOOLEAN widget type
instead of a non-existent BOOL type, and converts some strings to COMBOs.

Added/changed since v4:
- All on/off controls now use ("BOOLEAN", {"default": ...}) so they render as
  real checkboxes in the UI.
- `align_method` is a COMBO (for now just ["gcc-phat"], extensible later).
- `match_mode` is a COMBO: ["LUFS-I", "RMS"].
- Keeps the v4 compute/plot toggles to save FFT/LUFS work when unneeded.

Contracts (per Comfy docs):
- IMAGE: torch.Tensor [B,H,W,3] in 0..1
- AUDIO: dict with keys {"waveform": torch.Tensor [B,C,T], "sample_rate": int}
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# -----------------------------
# Array/Tensor helpers
# -----------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_CN(arr: np.ndarray) -> np.ndarray:
    """Coerce arbitrary shapes to channels-first [C, N] float32."""
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


def _blank_image(h: int = 8, w: int = 8) -> torch.Tensor:
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


# -----------------------------
# Comfy interop: AUDIO / IMAGE
# -----------------------------

def make_audio(sr: int, samples_CN: np.ndarray, meta: Optional[dict] = None) -> Dict[str, Any]:
    s = _normalize_CN(samples_CN)
    wf = torch.from_numpy(s).unsqueeze(0)  # [1,C,N]
    return {
        "sr": int(sr),
        "sample_rate": int(sr),
        "samples": s,              # convenience
        "waveform": wf,            # Comfy contract
        "meta": dict(meta or {}),
    }


def to_internal_audio(x: Any) -> Dict[str, Any]:
    """Accept a ComfyUI AUDIO or similar → {sr, samples[C,N], waveform[1,C,N]}"""
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


def image_from_figure(fig) -> torch.Tensor:
    """Matplotlib figure → IMAGE torch [1,H,W,3] in 0..1."""
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
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3]


# -----------------------------
# DSP helpers
# -----------------------------

def _rms_db(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    e = float(np.mean(x * x) + 1e-20)
    return 10.0 * math.log10(e)


def _k_weight(sr: int, x_CN: np.ndarray) -> np.ndarray:
    # very small K-weight approx: 1st-order HPF @60 Hz + mild HF tilt
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


def _lsd(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    LA = 20 * np.log10(A + eps)
    LB = 20 * np.log10(B + eps)
    D = (LA - LB) ** 2
    per = np.sqrt(np.mean(D, axis=0) + 1e-12)
    return float(np.mean(per)), float(np.percentile(per, 95))


def _band_energy_hi_db(x_CN: np.ndarray, sr: int, lo_hz: float) -> float:
    mono = x_CN.mean(axis=0)
    X = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sr)
    mask = freqs >= lo_hz
    e_hi = float(np.sum(np.abs(X[mask]) ** 2))
    e_all = float(np.sum(np.abs(X) ** 2) + 1e-20)
    return 10.0 * math.log10(e_hi / e_all + 1e-20)


def _pad_or_crop_CN(x: np.ndarray, N: int) -> np.ndarray:
    C, M = x.shape
    if M == N:
        return x
    if M > N:
        return x[:, :N]
    y = np.zeros((C, N), dtype=x.dtype)
    y[:, :M] = x
    return y


def _xcorr_delay(a: np.ndarray, b: np.ndarray, sr: int, max_shift_smp: int) -> float:
    # GCC-PHAT-ish coarse delay + parabolic refine. Returns samples (b lags a > 0)
    n = 1
    total = a.size + b.size
    while n < total:
        n <<= 1
    A = np.fft.rfft(a, n=n)
    B = np.fft.rfft(b, n=n)
    R = B * np.conj(A)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-(n // 2 - 1) :], cc[: n // 2 + 1]))
    center = len(cc) // 2
    sl = center - max_shift_smp
    sh = center + max_shift_smp + 1
    w = cc[sl:sh]
    k = int(np.argmax(w))
    idx = sl + k
    if 1 <= idx < len(cc) - 1:
        y0, y1, y2 = cc[idx - 1], cc[idx], cc[idx + 1]
        denom = 2 * (y0 - 2 * y1 + y2)
        frac = 0.0 if abs(denom) < 1e-12 else (y0 - y2) / denom
    else:
        frac = 0.0
    return float((idx - center) + frac)


def _apply_frac_delay_CN(x: np.ndarray, delay_samples: float, taps: int = 64) -> np.ndarray:
    if abs(delay_samples) < 1e-6:
        return x.copy()
    C, N = x.shape
    int_d = int(math.floor(abs(delay_samples)))
    frac = abs(delay_samples) - int_d
    sign = 1 if delay_samples >= 0 else -1
    y = np.zeros((C, N), dtype=np.float32)
    if sign > 0:
        if int_d < N:
            y[:, int_d:] = x[:, : N - int_d]
    else:
        if int_d < N:
            y[:, : N - int_d] = x[:, int_d:]
    if frac > 1e-6:
        M = max(16, int(taps))
        n = np.arange(M)
        m = (M - 1) / 2.0
        h = np.sinc(n - m - frac)
        w = np.hanning(M)
        h = (h * w).astype(np.float32)
        h /= np.sum(h)
        for c in range(C):
            yc = np.convolve(y[c], h, mode="same")
            y[c] = yc.astype(np.float32)
    return y


# -----------------------------
# Node 1: Audio Align (XCorr)
# -----------------------------
class Audio_Align_XCorr:
    CATEGORY = "Egregora/Analysis"
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("audio_proc_aligned", "delay_samples", "delay_ms", "peak_corr", "debug_image")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ref": ("AUDIO", {}),
                "audio_proc": ("AUDIO", {}),
            },
            "optional": {
                "max_shift_ms": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 1}),
                # COMBO: list[str] => dropdown
                "align_method": (["gcc-phat"], {}),
                "fractional": ("BOOLEAN", {"default": True}),
                "fir_len": ("INT", {"default": 64, "min": 16, "max": 256, "step": 1}),
            },
        }

    def execute(self, audio_ref, audio_proc, max_shift_ms=200, align_method="gcc-phat", fractional=True, fir_len=64):
        ref = to_internal_audio(audio_ref)
        proc = to_internal_audio(audio_proc)
        # resample proc to ref.sr if needed (linear interp is fine for alignment)
        if proc["sample_rate"] != ref["sample_rate"]:
            sr_old = proc["sample_rate"]
            x = proc["samples"]
            C, N = x.shape
            new_N = int(round(N * ref["sample_rate"] / sr_old))
            t_old = np.linspace(0.0, 1.0, N, endpoint=False)
            t_new = np.linspace(0.0, 1.0, new_N, endpoint=False)
            y = np.stack([np.interp(t_new, t_old, x[c]) for c in range(C)], axis=0).astype(np.float32)
            proc = make_audio(ref["sample_rate"], y, proc.get("meta", {}))

        a = ref["samples"].mean(axis=0)
        b = proc["samples"].mean(axis=0)
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]

        max_shift = int(ref["sample_rate"] * (max_shift_ms / 1000.0))
        lag = _xcorr_delay(a, b, ref["sample_rate"], max_shift)  # +ve => proc lags
        delay_samples = float(lag)
        delay_ms = 1000.0 * delay_samples / ref["sample_rate"]

        aligned = _apply_frac_delay_CN(proc["samples"], -delay_samples if fractional else -round(delay_samples), taps=fir_len)
        aligned = _pad_or_crop_CN(aligned, ref["samples"].shape[1])
        out = make_audio(ref["sample_rate"], aligned, proc.get("meta", {}))

        # minimal debug plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.arange(n)
            fig, ax = plt.subplots(1, 1, figsize=(6, 2.2))
            ax.plot(t, a, linewidth=0.5, label="A")
            ax.plot(t, b, linewidth=0.5, label="B")
            ax.legend(); ax.grid(alpha=.2); ax.set_title("Align preview")
            debug_img = image_from_figure(fig)
        except Exception:
            debug_img = _blank_image()

        return (out, float(delay_samples), float(delay_ms), 0.0, debug_img)


# -----------------------------
# Node 2: Audio Gain Match
# -----------------------------
class Audio_Gain_Match:
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
                # COMBO for mode
                "mode": (["LUFS-I", "RMS"], {}),
                "max_gain_db": ("FLOAT", {"default": 12.0, "min": -48.0, "max": 48.0, "step": 0.1}),
            },
        }

    def execute(self, audio_ref, audio_in, mode="LUFS-I", max_gain_db=12.0):
        ref = to_internal_audio(audio_ref)
        inn = to_internal_audio(audio_in)
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
# Node 3: Audio Null Test (with metric toggles)
# -----------------------------
class Audio_Null_Test:
    CATEGORY = "Egregora/Analysis"
    RETURN_TYPES = ("AUDIO", "DICT")
    RETURN_NAMES = ("audio_null", "metrics")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ref": ("AUDIO", {}),
                "audio_proc_aligned_matched": ("AUDIO", {}),
            },
            "optional": {
                "invert_b": ("BOOLEAN", {"default": True}),
                "least_squares_scale": ("BOOLEAN", {"default": False}),
                # Metric toggles
                "compute_corr": ("BOOLEAN", {"default": True}),
                "compute_null_rms": ("BOOLEAN", {"default": True}),
                "compute_null_lufs": ("BOOLEAN", {"default": True}),
                "compute_lsd": ("BOOLEAN", {"default": True}),
                "compute_hf_residual": ("BOOLEAN", {"default": False}),
                # STFT controls (used only if LSD requested)
                "n_fft": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 128}),
                "hop": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "hf_band_hz": ("INT", {"default": 8000, "min": 1000, "max": 20000, "step": 100}),
            },
        }

    def execute(self, audio_ref, audio_proc_aligned_matched, invert_b=True, least_squares_scale=False,
                compute_corr=True, compute_null_rms=True, compute_null_lufs=True,
                compute_lsd=True, compute_hf_residual=False, n_fft=2048, hop=512, hf_band_hz=8000):
        ref = to_internal_audio(audio_ref)
        pro = to_internal_audio(audio_proc_aligned_matched)
        if pro["sample_rate"] != ref["sample_rate"]:
            raise ValueError("Sample rate mismatch after alignment stage")
        A = ref["samples"]
        B = pro["samples"]
        N = min(A.shape[1], B.shape[1])
        A = A[:, :N]
        B = B[:, :N]
        k = 1.0
        if least_squares_scale:
            a = A.mean(axis=0).astype(np.float64)
            b = B.mean(axis=0).astype(np.float64)
            denom = float(np.dot(b, b) + 1e-20)
            k = float(np.dot(a, b) / denom)
            B = (B * k).astype(np.float32)
        if invert_b:
            B = -B
        null = (A + B).astype(np.float32)

        metrics: Dict[str, Any] = {}
        a_m = A.mean(axis=0)
        b_m = (-B).mean(axis=0)

        if compute_corr:
            am = a_m - np.mean(a_m)
            bm = b_m - np.mean(b_m)
            corr = float(np.dot(am, bm) / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-20))
            metrics["corr_coef"] = corr
        if compute_null_rms:
            metrics["null_rms_dbfs"] = float(_rms_db(null.mean(axis=0)))
        if compute_null_lufs:
            metrics["null_lufs"] = float(integrated_lufs(make_audio(ref["sample_rate"], null)))
        if compute_lsd:
            SA = _stft_mag(a_m, n_fft=n_fft, hop=hop)
            SB = _stft_mag(b_m, n_fft=n_fft, hop=hop)
            lsd_mean, lsd_p95 = _lsd(SA, SB)
            metrics["lsd_mean_db"] = float(lsd_mean)
            metrics["lsd_p95_db"] = float(lsd_p95)
        if compute_hf_residual:
            metrics["hf_residual_db"] = float(_band_energy_hi_db(null, ref["sample_rate"], hf_band_hz))
        # Always include safety stats
        overs = int(np.sum(np.abs(null) > 1.0))
        metrics["overshoot_count"] = int(overs)
        metrics["clipped_pct"] = float(100.0 * overs / null.size)
        metrics["scale_k"] = float(k)

        return make_audio(ref["sample_rate"], null, {}), metrics


# -----------------------------
# Node 4: Audio Plotter (with draw toggles)
# -----------------------------
class Audio_Plotter:
    CATEGORY = "Egregora/Visualization"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_waveforms", "image_spectrograms", "image_diffspec")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ref": ("AUDIO", {}),
                "audio_proc": ("AUDIO", {}),
                "audio_null": ("AUDIO", {}),
            },
            "optional": {
                "draw_waveforms": ("BOOLEAN", {"default": True}),
                "draw_spectrograms": ("BOOLEAN", {"default": True}),
                "draw_diffspec": ("BOOLEAN", {"default": True}),
                "n_fft": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 128}),
                "hop": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            },
        }

    def execute(self, audio_ref, audio_proc, audio_null, draw_waveforms=True, draw_spectrograms=True, draw_diffspec=True, n_fft=2048, hop=512):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ref = to_internal_audio(audio_ref)
        pro = to_internal_audio(audio_proc)
        nul = to_internal_audio(audio_null)

        a = ref["samples"].mean(axis=0)
        b = pro["samples"].mean(axis=0)
        n = min(a.size, b.size, nul["samples"].shape[1])
        a = a[:n]
        b = b[:n]
        null = nul["samples"].mean(axis=0)[:n]

        # Waveforms
        if draw_waveforms:
            t = np.arange(n)
            fig1, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            for ax, y, ttl in zip(axes, [a, b, null], ["A: original", "B: processed", "Null: A−B"]):
                ax.plot(t, y, linewidth=0.7)
                ax.set_ylim(-1.05, 1.05)
                ax.set_title(ttl)
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("samples")
            fig1.tight_layout()
            img_wave = image_from_figure(fig1)
        else:
            img_wave = _blank_image(1, 1)

        # Spectrograms (A, B, Null)
        if draw_spectrograms:
            def _spec(y):
                S = _stft_mag(y, n_fft=n_fft, hop=hop)
                return 20 * np.log10(S + 1e-9)
            SA = _spec(a)
            SB = _spec(b)
            SN = _spec(null)
            fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7))
            for ax, S, ttl in zip(axes2, [SA, SB, SN], ["A: spec", "B: spec", "Null: spec"]):
                ax.imshow(S, origin="lower", aspect="auto")
                ax.set_title(ttl)
            fig2.tight_layout()
            img_spec = image_from_figure(fig2)
        else:
            img_spec = _blank_image(1, 1)

        # Diff-spec |A-B|
        if draw_diffspec:
            def _spec(y):
                S = _stft_mag(y, n_fft=n_fft, hop=hop)
                return 20 * np.log10(S + 1e-9)
            SA = _spec(a)
            SB = _spec(b)
            D = np.abs(10 ** (SA / 20.0) - 10 ** (SB / 20.0))
            fig3 = plt.figure(figsize=(10, 3))
            import matplotlib.pyplot as plt  # noqa
            plt.imshow(20 * np.log10(D + 1e-9), origin="lower", aspect="auto")
            plt.title("|Spec(A) − Spec(B)| (dB)")
            plt.tight_layout()
            img_diff = image_from_figure(fig3)
        else:
            img_diff = _blank_image(1, 1)

        return (img_wave, img_spec, img_diff)


# -----------------------------
# Node 5: Null Test (Full) – with toggles exposed
# -----------------------------
class Null_Test_Full:
    CATEGORY = "Egregora/Analysis"
    RETURN_TYPES = ("AUDIO", "AUDIO", "FLOAT", "FLOAT", "DICT", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "audio_proc_aligned_matched",
        "audio_null",
        "delay_ms",
        "gain_db",
        "metrics",
        "image_waveforms",
        "image_spectrograms",
        "image_diffspec",
    )
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ref": ("AUDIO", {}),
                "audio_proc": ("AUDIO", {}),
            },
            "optional": {
                # Align/Gain params
                "align_max_shift_ms": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 1}),
                "align_method": (["gcc-phat"], {}),
                "fractional": ("BOOLEAN", {"default": True}),
                "fir_len": ("INT", {"default": 64, "min": 16, "max": 256, "step": 1}),
                "match_mode": (["LUFS-I", "RMS"], {}),
                "least_squares_scale": ("BOOLEAN", {"default": False}),
                # Metric toggles
                "compute_corr": ("BOOLEAN", {"default": True}),
                "compute_null_rms": ("BOOLEAN", {"default": True}),
                "compute_null_lufs": ("BOOLEAN", {"default": True}),
                "compute_lsd": ("BOOLEAN", {"default": True}),
                "compute_hf_residual": ("BOOLEAN", {"default": False}),
                # Plot toggles
                "draw_waveforms": ("BOOLEAN", {"default": True}),
                "draw_spectrograms": ("BOOLEAN", {"default": True}),
                "draw_diffspec": ("BOOLEAN", {"default": True}),
                # STFT controls
                "n_fft": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 128}),
                "hop": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            },
        }

    def execute(self, audio_ref, audio_proc, align_max_shift_ms=200, align_method="gcc-phat", fractional=True,
                fir_len=64, match_mode="LUFS-I", least_squares_scale=False,
                compute_corr=True, compute_null_rms=True, compute_null_lufs=True,
                compute_lsd=True, compute_hf_residual=False,
                draw_waveforms=True, draw_spectrograms=True, draw_diffspec=True,
                n_fft=2048, hop=512):
        # 1) Align
        align = Audio_Align_XCorr()
        ap_aligned, delay_samples, delay_ms, _pc, _dbg = align.execute(
            audio_ref, audio_proc,
            max_shift_ms=align_max_shift_ms,
            align_method=align_method,
            fractional=fractional,
            fir_len=fir_len,
        )
        # 2) Gain-match
        gm = Audio_Gain_Match()
        ap_matched, gain_db, _ref_lvl, _in_lvl = gm.execute(audio_ref, ap_aligned, mode=match_mode)
        # 3) Null (+ metrics)
        nt = Audio_Null_Test()
        audio_null, metrics = nt.execute(
            audio_ref, ap_matched,
            invert_b=True,
            least_squares_scale=least_squares_scale,
            compute_corr=compute_corr,
            compute_null_rms=compute_null_rms,
            compute_null_lufs=compute_null_lufs,
            compute_lsd=compute_lsd,
            compute_hf_residual=compute_hf_residual,
            n_fft=n_fft, hop=hop,
        )
        # 4) Plots (respect draw toggles)
        pl = Audio_Plotter()
        img_waves, img_spec, img_diff = pl.execute(
            audio_ref, ap_matched, audio_null,
            draw_waveforms=draw_waveforms,
            draw_spectrograms=draw_spectrograms,
            draw_diffspec=draw_diffspec,
            n_fft=n_fft, hop=hop,
        )

        return ap_matched, audio_null, float(delay_ms), float(gain_db), metrics, img_waves, img_spec, img_diff


# -----------------------------
# Registration (original names maintained for retro-compatibility)
# -----------------------------
NODE_CLASS_MAPPINGS = {
    "Audio Align (XCorr)": Audio_Align_XCorr,
    "Audio Gain Match": Audio_Gain_Match,
    "Audio Null Test": Audio_Null_Test,
    "Audio Plotter": Audio_Plotter,
    "Null Test (Full)": Null_Test_Full,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Align (XCorr)": "Audio Align (XCorr)",
    "Audio Gain Match": "Audio Gain Match",
    "Audio Null Test": "Audio Null Test",
    "Audio Plotter": "Audio Plotter",
    "Null Test (Full)": "Null Test (Full)",
}
