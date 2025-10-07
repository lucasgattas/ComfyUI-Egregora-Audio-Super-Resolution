# Egregora Enhance Extras - Fixed Version
# Adds: RNNoise Denoise, WPE Dereverb, DeepFilterNet Denoise, DAC Encode/Decode, ViSQOL Meter
# Licenses:
# - RNNoise wrappers (pyrnnoise): Apache-2.0
# - NARA-WPE: MIT
# - DeepFilterNet: MIT/Apache-2.0 (dual)
# - DAC: MIT
# - ViSQOL (binary) + Audiocraft wrapper docs: Apache-2.0 (wrapper), ViSQOL itself under Apache-2.0

import os
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torchaudio
import numpy as np

# ----------------------------
# Small audio helpers (Comfy-style)
# ----------------------------

def _is_audio(x):
    return isinstance(x, dict) and "waveform" in x and "sample_rate" in x

def _coerce_audio(x):
    # Returns (wave[B,C,T], sr:int, meta:dict)
    if _is_audio(x):
        wav = x["waveform"]
        sr = int(x["sample_rate"])
        meta = x.get("meta", {})
        if wav.dim() == 2:
            # [C,T] -> [1,C,T]
            wav = wav.unsqueeze(0)
        elif wav.dim() == 1:
            # [T] -> [1,1,T]
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() != 3:
            raise ValueError("Audio waveform must be 1D, 2D or 3D [B,C,T].")
        return wav.float(), sr, meta
    # Torch tensor passthrough (assume [C,T] or [B,C,T] with default sr=48000)
    if isinstance(x, torch.Tensor):
        wav = x
        if wav.dim() == 2:  # [C,T] -> [1,C,T]
            wav = wav.unsqueeze(0)
        elif wav.dim() != 3:
            raise ValueError("Tensor audio must be [C,T] or [B,C,T].")
        return wav.float(), 48000, {}
    raise TypeError("Unsupported audio input type.")

def _make_audio(sr: int, wav: torch.Tensor, meta: Optional[dict] = None):
    # Ensure [B,C,T]
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.dim() != 3:
        raise ValueError("samples must be 1D/2D/3D; got shape %r" % (wav.shape,))
    return {
        "waveform": wav.contiguous(),
        "sample_rate": int(sr),
        "meta": meta or {},
    }

def _resample(wav: torch.Tensor, sr_in: int, sr_out: int):
    if sr_in == sr_out:
        return wav, sr_in
    B, C, T = wav.shape
    res = []
    for b in range(B):
        # torchaudio expects [C,T]
        res.append(torchaudio.functional.resample(wav[b], sr_in, sr_out))
    wav_out = torch.stack(res, dim=0)
    return wav_out, sr_out

def _to_mono(wav: torch.Tensor):
    # [B,C,T] -> [B,1,T]
    if wav.size(1) == 1:
        return wav
    return wav.mean(dim=1, keepdim=True)

def _device_for(wav: torch.Tensor):
    return "cuda" if wav.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# RNNoise (pyrnnoise)
# ----------------------------

class Egregora_RNNoise_Denoise:
    """
    RNNoise denoiser (speech-focused), ComfyUI node.
      • Runs at 48 kHz (10 ms = 480 samples).
      • Mono/stereo: per-channel or downmix to mono.
      • Uses pyrnnoise>=0.3.x 'denoise_chunk' API.
      • Adds static strength + adaptive mix (driven by per-frame VAD) + post-gain with ceiling.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_ms": ("INT", {"default": 20, "min": 5, "max": 60, "step": 5}),
                "stereo_mode": (["per_channel", "downmix_mono"], {"default": "per_channel"}),

                # mix controls
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix_curve": (["equal_power", "linear"], {"default": "equal_power"}),

                # adaptive controls
                "adaptive_mode": (["off", "more_on_noise", "more_on_speech", "gate_on_noise"], {"default": "more_on_noise"}),
                "adaptive_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vad_threshold": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vad_smooth_ms": ("INT", {"default": 50, "min": 0, "max": 500, "step": 5}),

                # post gain
                "post_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "limit_ceiling": ("BOOL", {"default": True}),
                "ceiling": ("FLOAT", {"default": 0.999, "min": 0.1, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Enhance"

    # ---------- helpers ----------
    def _silence_destructor(self, rn):
        try:
            type(rn).__del__ = lambda self: None
        except Exception:
            pass

    def _init_rn(self, channels: int):
        from pyrnnoise import RNNoise
        rn = RNNoise(sample_rate=48000)
        try:
            if getattr(rn, "channels", None) in (None, 0):
                setattr(rn, "channels", channels)
        except Exception:
            pass
        return rn

    def _denoise_chunk_with_probs(self, rn, x_i16):
        """
        Preferred path on pyrnnoise>=0.3.x: returns (wet_i16, vad_probs_per_frame)
        where each frame is 480 samples at 48 kHz.
        """
        import numpy as np
        pad = (-len(x_i16)) % 480
        x_pad = np.pad(x_i16, (0, pad), mode="constant") if pad else x_i16

        out_frames, probs = [], []
        x2 = x_pad[np.newaxis, :]  # (1, N)
        for p, den in rn.denoise_chunk(x2):
            # p may be scalar or array-like; we're per-channel, so take float(p)
            try:
                p_val = float(p[0]) if hasattr(p, "__len__") else float(p)
            except Exception:
                p_val = float(p)
            probs.append(p_val)

            den = np.asarray(den, dtype=np.int16)
            if den.ndim == 2 and den.shape[0] == 1:
                den = den[0]
            out_frames.append(den)

        wet = np.concatenate(out_frames, axis=0)
        return wet[:len(x_i16)], np.asarray(probs, dtype=np.float32)

    def _fallback_frame_loop(self, rn, x_i16, frame_len):
        """
        Very old wheels only: try process_frame/filter; else passthrough.
        (No VAD probs here, so adaptive becomes effectively 'off' on fallback.)
        """
        import numpy as np
        call = None
        if hasattr(rn, "process_frame"):
            call = lambda fr: np.asarray(rn.process_frame(fr), dtype=np.int16)
        elif hasattr(rn, "filter"):
            call = lambda fr: np.asarray(rn.filter(fr), dtype=np.int16)
        if call is None:
            return x_i16, None

        frame_len = max(1, frame_len // 480) * 480
        pad = (-len(x_i16)) % frame_len
        x_work = np.pad(x_i16, (0, pad), mode="constant") if pad else x_i16

        outs = []
        for start in range(0, len(x_work), frame_len):
            chunk = x_work[start:start + frame_len]
            pos, sub = 0, []
            while pos < frame_len:
                fr = chunk[pos:pos + 480]
                if fr.shape[0] < 480:
                    fr = np.pad(fr, (0, 480 - fr.shape[0]), mode="constant")
                try:
                    y = call(fr)
                except Exception:
                    y = fr
                sub.append(y)
                pos += 480
            outs.append(np.concatenate(sub, axis=0))
        out = np.concatenate(outs, axis=0)
        return out[:len(x_i16)], None

    def _smooth_vad_probs(self, probs, smooth_ms: int):
        import numpy as np, math
        if probs is None or probs.size == 0 or smooth_ms <= 0:
            return probs
        hop_ms = 10.0  # RNNoise frame = 10 ms @ 48 kHz
        tau = max(1e-3, float(smooth_ms))
        alpha = math.exp(-hop_ms / tau)
        y = np.empty_like(probs)
        acc = probs[0]
        for i, p in enumerate(probs):
            acc = alpha * acc + (1.0 - alpha) * p
            y[i] = acc
        return y

    def _strength_per_frame(self, base_s, vad_smooth, adaptive_mode, adaptive_amount, vad_threshold):
        import numpy as np
        if vad_smooth is None:
            return np.array([base_s], dtype=np.float32)  # will be broadcast
        s0 = float(base_s)
        a = float(adaptive_amount)
        v = np.clip(vad_smooth, 0.0, 1.0)
        if adaptive_mode == "off":
            s_eff = np.full_like(v, s0, dtype=np.float32)
        elif adaptive_mode == "more_on_noise":
            # more denoise when speech-prob low
            s_eff = s0 + a * (1.0 - v) * (1.0 - s0)
        elif adaptive_mode == "more_on_speech":
            # more denoise when speech-prob high
            s_eff = s0 + a * v * (1.0 - s0)
        elif adaptive_mode == "gate_on_noise":
            # if below threshold => denoise-heavy; else denoise-light
            s_noise = s0 + a * (1.0 - s0)       # push toward 1
            s_speech = s0 * (1.0 - a)           # pull toward 0
            s_eff = np.where(v < vad_threshold, s_noise, s_speech).astype(np.float32)
        else:
            s_eff = np.full_like(v, s0, dtype=np.float32)
        return np.clip(s_eff.astype(np.float32), 0.0, 1.0)

    def _gains_from_strength(self, s_eff, curve):
        import numpy as np, math
        s = np.clip(s_eff, 0.0, 1.0).astype(np.float32)
        if curve == "equal_power":
            # equal-power crossfade: keep power ~constant
            g_wet = np.sin(0.5 * math.pi * s, dtype=np.float32)
            g_dry = np.cos(0.5 * math.pi * s, dtype=np.float32)
        else:
            # linear
            g_wet = s
            g_dry = 1.0 - s
        return g_dry.astype(np.float32), g_wet.astype(np.float32)

    # ---------- main ----------
    def execute(
        self,
        audio,
        frame_ms=20,
        stereo_mode="per_channel",
        strength=1.0,
        mix_curve="equal_power",
        adaptive_mode="more_on_noise",
        adaptive_amount=0.5,
        vad_threshold=0.90,
        vad_smooth_ms=50,
        post_gain_db=0.0,
        limit_ceiling=True,
        ceiling=0.999,
    ):
        import numpy as np
        import torch
        import math

        # Coerce to [B,C,T], resample to 48k (RNNoise domain)
        wav, sr, meta = _coerce_audio(audio)
        wav48, _ = _resample(wav, sr, 48000)

        if stereo_mode == "downmix_mono":
            wav48 = _to_mono(wav48)

        B, C, T = wav48.shape
        frame_len = int(48000 * max(5, min(60, frame_ms)) / 1000)

        out_batches = []
        for b in range(B):
            ch_out = []
            for c in range(C):
                dry = wav48[b, c].detach()  # float32 [-1,1] at 48k
                x = dry.cpu().numpy().astype(np.float32)
                x_i16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

                rn = self._init_rn(channels=1)

                if hasattr(rn, "denoise_chunk"):
                    try:
                        wet_i16, probs = self._denoise_chunk_with_probs(rn, x_i16)
                    except Exception:
                        self._silence_destructor(rn)
                        rn = self._init_rn(channels=1)
                        wet_i16, probs = self._fallback_frame_loop(rn, x_i16, frame_len)
                else:
                    wet_i16, probs = self._fallback_frame_loop(rn, x_i16, frame_len)

                wet = torch.from_numpy(wet_i16.astype(np.float32) / 32768.0).to(dry.device)

                # ----- Adaptive mixing -----
                vad_s = self._smooth_vad_probs(probs, vad_smooth_ms)
                s_eff = self._strength_per_frame(strength, vad_s, adaptive_mode, adaptive_amount, vad_threshold)
                # expand per-frame strengths (10 ms) to per-sample gains
                if s_eff.ndim == 0:
                    s_per_sample = np.full(T, float(s_eff), dtype=np.float32)
                else:
                    s_per_sample = np.repeat(s_eff, 480)[:T].astype(np.float32)

                g_dry_np, g_wet_np = self._gains_from_strength(s_per_sample, mix_curve)
                g_dry = torch.from_numpy(g_dry_np).to(dry.device)
                g_wet = torch.from_numpy(g_wet_np).to(dry.device)

                y = g_dry * dry + g_wet * wet
                y = torch.clamp(y, -1.0, 1.0)

                ch_out.append(y)

            y_st = torch.stack(ch_out, dim=0).unsqueeze(0)  # [1,C,T]
            out_batches.append(y_st)

        y48 = torch.cat(out_batches, dim=0)  # [B,C,T]

        # Back to original sample rate
        y, _ = _resample(y48, 48000, sr)

        # ----- Post-gain + optional ceiling limiter -----
        if post_gain_db != 0.0:
            gain = float(10.0 ** (post_gain_db / 20.0))
            y = y * gain

        if limit_ceiling:
            peak = torch.max(torch.abs(y)).item()
            if peak > ceiling and peak > 0:
                y = y * (ceiling / peak)

        y = torch.clamp(y, -1.0, 1.0)

        meta2 = dict(meta)
        meta2["rnnoise"] = {
            "frame_ms": frame_ms,
            "stereo_mode": stereo_mode,
            "strength": strength,
            "mix_curve": mix_curve,
            "adaptive_mode": adaptive_mode,
            "adaptive_amount": adaptive_amount,
            "vad_threshold": vad_threshold,
            "vad_smooth_ms": vad_smooth_ms,
            "post_gain_db": post_gain_db,
            "limit_ceiling": bool(limit_ceiling),
            "ceiling": ceiling,
        }
        return (_make_audio(sr, y, meta2),)

# ----------------------------
# WPE Dereverb (nara_wpe)
# ----------------------------

class Egregora_WPE_Dereverb:
    """
    Weighted Prediction Error dereverberation.
    Works mono or multi-channel. Uses STFT -> WPE -> iSTFT.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "taps": ("INT", {"default": 10, "min": 3, "max": 32}),
                "delay": ("INT", {"default": 3, "min": 1, "max": 16}),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 10}),
                "n_fft": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
                "hop": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 64}),
                "use_float32": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Enhance"

    def execute(self, audio, taps=10, delay=3, iterations=3, n_fft=1024, hop=256, use_float32=True):
        try:
            import numpy as np
            from nara_wpe import wpe as np_wpe
            from nara_wpe.utils import stft, istft
        except Exception as e:
            raise RuntimeError("nara-wpe not installed. pip install nara-wpe") from e

        wav, sr, meta = _coerce_audio(audio)  # [B,C,T]
        B, C, T = wav.shape

        out_list = []
        for b in range(B):
            # nara_wpe expects numpy with shape (channels, samples)
            y = wav[b].cpu().numpy()  # [C,T]
            
            # FIX: Handle memory issues with large arrays by processing in chunks or using float32
            if use_float32:
                y = y.astype(np.float32)
            
            try:
                # STFT: returns shape (frames, freqs, channels)
                Y = stft(y, size=n_fft, shift=hop)
                
                # FIX: Check memory usage and dtype
                if Y.dtype == np.complex128 and use_float32:
                    Y = Y.astype(np.complex64)
                
                # Transpose to (freqs, channels, frames) as expected by wpe()
                Y = np.transpose(Y, (1, 2, 0))
                
                # Apply WPE with memory-conscious settings
                Z = np_wpe.wpe(Y, taps=taps, delay=delay, iterations=iterations)
                
                # Back to (frames, freqs, channels)
                Z = np.transpose(Z, (2, 0, 1))
                z = istft(Z, size=n_fft, shift=hop)  # (channels, samples)
                
            except MemoryError:
                # Fallback: process with reduced precision or skip WPE
                print(f"Warning: WPE processing failed due to memory constraints for batch {b}")
                z = y  # Pass through original audio
            except Exception as e:
                print(f"Warning: WPE processing failed: {e}")
                z = y  # Pass through original audio
            
            z_t = torch.from_numpy(z).to(wav.device).float()  # [C,T]
            out_list.append(z_t.unsqueeze(0))  # [1,C,T]

        out = torch.cat(out_list, dim=0)
        meta2 = dict(meta)
        meta2["wpe"] = {"taps": taps, "delay": delay, "iterations": iterations, "n_fft": n_fft, "hop": hop}
        return (_make_audio(sr, out, meta2),)


# ----------------------------
# DeepFilterNet (DFN/DFN2/DFN3)
# ----------------------------

class Egregora_DeepFilterNet_Denoise:
    """
    DeepFilterNet denoiser (speech enhancement) for ComfyUI.

    • Runs DeepFilterNet at 48 kHz (its native rate), using tensor I/O.
    • Mono or stereo (per-channel or downmix to mono before DFN).
    • Adds 'strength' wet/dry mix with equal-power or linear curve.
    • Adaptive mix driven by VAD (RNNoise if available, else energy/RMS proxy).
    • Post-gain (dB) and a simple peak ceiling limiter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),

                # DFN options
                "dfn_model": (["DeepFilterNet2", "DeepFilterNet3"], {"default": "DeepFilterNet2"}),
                "device": (["auto", "cuda:0", "cpu"], {"default": "auto"}),

                # proper BOOLEAN toggles (not sockets)
                "use_postfilter": ("BOOLEAN", {"default": False, "label_on": "postfilter on", "label_off": "postfilter off"}),
                "limit_ceiling": ("BOOLEAN", {"default": True, "label_on": "limit on", "label_off": "limit off"}),

                # channel / framing
                "stereo_mode": (["per_channel", "downmix_mono"], {"default": "per_channel"}),
                "frame_ms": ("INT", {"default": 20, "min": 5, "max": 60, "step": 5}),

                # mixing
                "strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix_curve": (["equal_power", "linear"], {"default": "equal_power"}),

                # adaptive controls
                "adaptive_vad_source": (["rms", "rnnoise", "none"], {"default": "rms"}),
                "adaptive_mode": (["off", "more_on_noise", "more_on_speech", "gate_on_noise"], {"default": "more_on_noise"}),
                "adaptive_amount": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vad_threshold": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vad_smooth_ms": ("INT", {"default": 60, "min": 0, "max": 500, "step": 5}),

                # post
                "post_gain_db": ("FLOAT", {"default": 0.5, "min": -24.0, "max": 24.0, "step": 0.1}),
                "ceiling": ("FLOAT", {"default": 0.98, "min": 0.1, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "Egregora/Enhance"

    # ------------------------- DFN backend & cache -------------------------
    _DF_CACHE = {}  # (model_name, device) -> (model, df_state)

    def _pick_device(self, choice: str):
        import torch
        if choice == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return choice

    def _df_get(self, model_name: str, device: str):
        from df.enhance import init_df
        key = (model_name, device)
        if key in self._DF_CACHE:
            return self._DF_CACHE[key]
        model, df_state, _ = init_df(model_name, config_allow_defaults=True)
        model = model.to(device).eval()
        self._DF_CACHE[key] = (model, df_state)
        return model, df_state

    # ----------------------------- VAD helpers -----------------------------
    def _vad_probs_rnnoise_48k(self, x48_np):
        import numpy as np
        try:
            from pyrnnoise import RNNoise
        except Exception:
            return None  # RNNoise not installed

        x_i16 = (np.clip(x48_np, -1.0, 1.0) * 32767.0).astype(np.int16)
        rn = RNNoise(sample_rate=48000)
        try:
            if getattr(rn, "channels", None) in (None, 0):
                setattr(rn, "channels", 1)
        except Exception:
            pass

        probs = []
        if hasattr(rn, "denoise_chunk"):
            pad = (-len(x_i16)) % 480
            x_pad = np.pad(x_i16, (0, pad), mode="constant") if pad else x_i16
            X = x_pad[np.newaxis, :]
            for p, _ in rn.denoise_chunk(X):
                try:
                    probs.append(float(p[0]) if hasattr(p, "__len__") else float(p))
                except Exception:
                    probs.append(float(p))
            return np.asarray(probs, dtype=np.float32)
        return None  # fallback APIs don't expose p

    def _vad_probs_rms_48k(self, x48_np):
        import numpy as np
        hop = 480  # 10 ms at 48 kHz
        n = (len(x48_np) + hop - 1) // hop
        rms = []
        for i in range(n):
            fr = x48_np[i*hop:(i+1)*hop]
            rms.append(float(np.sqrt(np.mean(fr*fr))) if len(fr) else 0.0)
        rms = np.asarray(rms, dtype=np.float32)
        p95 = float(np.percentile(rms, 95)) or 1e-6
        return np.clip(rms / p95, 0.0, 1.0).astype(np.float32)

    def _smooth_probs(self, probs, smooth_ms: int):
        import numpy as np, math
        if probs is None or probs.size == 0 or smooth_ms <= 0:
            return probs
        hop_ms = 10.0
        tau = max(1e-3, float(smooth_ms))
        alpha = math.exp(-hop_ms / tau)
        y = np.empty_like(probs)
        acc = probs[0]
        for i, p in enumerate(probs):
            acc = alpha * acc + (1.0 - alpha) * p
            y[i] = acc
        return y

    def _strength_per_frame(self, base_s, vad_smooth, adaptive_mode, adaptive_amount, vad_threshold):
        import numpy as np
        if vad_smooth is None:
            return np.array([float(base_s)], dtype=np.float32)
        s0 = float(base_s)
        a = float(adaptive_amount)
        v = np.clip(vad_smooth, 0.0, 1.0)
        if adaptive_mode == "off":
            s_eff = np.full_like(v, s0, dtype=np.float32)
        elif adaptive_mode == "more_on_noise":
            s_eff = s0 + a * (1.0 - v) * (1.0 - s0)
        elif adaptive_mode == "more_on_speech":
            s_eff = s0 + a * v * (1.0 - s0)
        elif adaptive_mode == "gate_on_noise":
            s_noise = s0 + a * (1.0 - s0)
            s_speech = s0 * (1.0 - a)
            s_eff = (s_noise * (v < vad_threshold) + s_speech * (v >= vad_threshold)).astype(np.float32)
        else:
            s_eff = np.full_like(v, s0, dtype=np.float32)
        return np.clip(s_eff, 0.0, 1.0).astype(np.float32)

    def _gains_from_strength(self, s_eff, curve):
        import numpy as np, math
        s = np.clip(s_eff, 0.0, 1.0).astype(np.float32)
        if curve == "equal_power":
            g_wet = np.sin(0.5 * math.pi * s, dtype=np.float32)
            g_dry = np.cos(0.5 * math.pi * s, dtype=np.float32)
        else:
            g_wet = s
            g_dry = 1.0 - s
        return g_dry.astype(np.float32), g_wet.astype(np.float32)

    # ------------------------------ main op ------------------------------
    def execute(
        self,
        audio,
        dfn_model="DeepFilterNet2",
        device="auto",
        use_postfilter=False,
        limit_ceiling=True,
        stereo_mode="per_channel",
        frame_ms=20,
        strength=0.65,
        mix_curve="equal_power",
        adaptive_vad_source="rms",
        adaptive_mode="more_on_noise",
        adaptive_amount=0.45,
        vad_threshold=0.90,
        vad_smooth_ms=60,
        post_gain_db=0.5,
        ceiling=0.98,
    ):
        import torch, numpy as np
        from df.enhance import enhance
        from df.io import resample  # DFN tensor resampler (48k native)

        # 1) Coerce to [B,C,T], then tensorize & resample to 48 kHz (DFN native)
        wav, sr, meta = _coerce_audio(audio)
        if stereo_mode == "downmix_mono":
            wav = _to_mono(wav)
        B, C, T = wav.shape

        x_ct = wav.reshape(-1, T).to(torch.float32)               # (C,T)
        x48 = resample(x_ct, sr, 48000) if sr != 48000 else x_ct

        # 2) Load DFN once
        dev = self._pick_device(device)
        model, df_state = self._df_get(dfn_model, dev)

        # 3) Run DFN per channel (tensors-in/out)
        wet_ch = []
        with torch.no_grad():
            for ch in range(x48.shape[0]):
                xin = x48[ch:ch+1]                                # (1,T)
                y = enhance(model, df_state, xin)                 # (1,T)
                # Some DFN builds expose post_filter kwarg; keep flag for future wheels
                # if use_postfilter:
                #     y = enhance(model, df_state, xin, post_filter=True)
                wet_ch.append(y)
        wet48 = torch.cat(wet_ch, dim=0)                          # (C,T)

        # 4) Back to original sample rate (tensors)
        wet = resample(wet48, 48000, sr) if sr != 48000 else wet48
        dry = x_ct if sr == 48000 else resample(x_ct, 48000, sr)

        # 5) Adaptive mix (10 ms frame gains expanded to per-sample)
        hop = int(sr * 0.010)  # 10 ms at current sr for expansion
        out_ch = []
        for ch in range(dry.shape[0]):
            dry_np = dry[ch].detach().cpu().numpy()
            wet_np = wet[ch].detach().cpu().numpy()

            # VAD at 48k domain, then expand
            if adaptive_vad_source == "rnnoise":
                x48_np = (resample(dry[ch:ch+1], sr, 48000)[0].cpu().numpy()
                          if sr != 48000 else dry_np)
                probs = self._vad_probs_rnnoise_48k(x48_np)
            elif adaptive_vad_source == "rms":
                x48_np = (resample(dry[ch:ch+1], sr, 48000)[0].cpu().numpy()
                          if sr != 48000 else dry_np)
                probs = self._vad_probs_rms_48k(x48_np)
            else:
                probs = None

            vad_s = self._smooth_probs(probs, vad_smooth_ms)
            s_eff = self._strength_per_frame(strength, vad_s, adaptive_mode, adaptive_amount, vad_threshold)

            if s_eff.ndim == 0:
                s_per = np.full(dry_np.shape[0], float(s_eff), dtype=np.float32)
            else:
                s_per = np.repeat(s_eff, max(1, hop))[:dry_np.shape[0]].astype(np.float32)

            g_dry_np, g_wet_np = self._gains_from_strength(s_per, mix_curve)
            y_np = g_dry_np * dry_np + g_wet_np * wet_np
            y_np = np.clip(y_np, -1.0, 1.0)
            out_ch.append(torch.from_numpy(y_np))

        y = torch.stack(out_ch, dim=0)                            # (C,T)
        y = y.reshape(B, C, -1)

        # 6) Post-gain + limiter
        if post_gain_db != 0.0:
            gain = float(10.0 ** (post_gain_db / 20.0))
            y = y * gain

        if limit_ceiling:
            peak = torch.max(torch.abs(y)).item()
            if peak > ceiling and peak > 0:
                y = y * (ceiling / peak)

        y = torch.clamp(y, -1.0, 1.0)

        meta2 = dict(meta)
        meta2["deepfilternet"] = {
            "model": dfn_model,
            "device": dev,
            "use_postfilter": bool(use_postfilter),
            "stereo_mode": stereo_mode,
            "frame_ms": frame_ms,
            "strength": strength,
            "mix_curve": mix_curve,
            "adaptive_vad_source": adaptive_vad_source,
            "adaptive_mode": adaptive_mode,
            "adaptive_amount": adaptive_amount,
            "vad_threshold": vad_threshold,
            "vad_smooth_ms": vad_smooth_ms,
            "post_gain_db": post_gain_db,
            "limit_ceiling": bool(limit_ceiling),
            "ceiling": ceiling,
        }
        return (_make_audio(sr, y, meta2),)

# ----------------------------
# Descript Audio Codec (DAC) encode/decode
# ----------------------------

class Egregora_DAC_Encode:
    """
    Encodes audio with DAC and returns latent 'z' & metadata in a DICT.
    Auto-downloads weights for chosen model_type on first use.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_type": (["44khz", "24khz", "16khz"], {"default": "44khz"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("codes", "log")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Codecs"

    def execute(self, audio, model_type="44khz", device="auto"):
        try:
            import dac
        except Exception as e:
            raise RuntimeError("descript-audio-codec not installed. pip install descript-audio-codec") from e

        wav, sr, meta = _coerce_audio(audio)  # [B,C,T] float
        B, C, T = wav.shape

        # Auto-download
        ckpt = dac.utils.download(model_type=model_type)
        model = dac.DAC.load(ckpt)

        dev = _device_for(wav) if device == "auto" else device
        model = model.to(dev)

        # FIX: Get model's expected sample rate
        model_sr = model.sample_rate

        # Compress each batch separately, concat codes
        with torch.no_grad():
            z_all = []
            for b in range(B):
                x = wav[b].to(dev)  # [C,T]
                
                # FIX: Resample to model's expected sample rate before preprocessing
                if sr != model_sr:
                    x_resampled = torchaudio.functional.resample(x, sr, model_sr)
                else:
                    x_resampled = x
                
                # preprocess expects the correct sample rate
                x_prep = model.preprocess(x_resampled, model_sr)
                z, codes, latents, _, _ = model.encode(x_prep)
                
                # Store z (list of tensors) into CPU tensors for DICT
                if isinstance(z, (list, tuple)):
                    z_cpu = [t.detach().cpu() for t in z]
                else:
                    z_cpu = [z.detach().cpu()]
                z_all.append(z_cpu)

        codes_dict = {
            "model_type": model_type,
            "sample_rate": sr,  # Store original sample rate
            "model_sample_rate": model_sr,  # Store model's sample rate
            "latents": z_all,  # list over batch of list[tensor]
        }
        log = f"DAC encode ok: model={model_type}, B={B}, C={C}, sr={sr}->{model_sr}"
        return (codes_dict, log)


class Egregora_DAC_Decode:
    """
    Decodes DICT produced by Egregora_DAC_Encode back to AUDIO.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "codes": ("DICT",),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "log")
    FUNCTION = "execute"
    CATEGORY = "Egregora/Codecs"

    def execute(self, codes, device="auto"):
        try:
            import dac
        except Exception as e:
            raise RuntimeError("descript-audio-codec not installed. pip install descript-audio-codec") from e

        model_type = codes.get("model_type", "44khz")
        sr = int(codes.get("sample_rate", 48000))
        model_sr = int(codes.get("model_sample_rate", sr))
        latents_b = codes.get("latents", [])
        if not latents_b:
            raise ValueError("codes.latents empty")

        ckpt = dac.utils.download(model_type=model_type)
        model = dac.DAC.load(ckpt)

        dev = "cuda" if torch.cuda.is_available() and device in ("auto", "cuda") else "cpu"
        model = model.to(dev)

        outs = []
        with torch.no_grad():
            for z_list in latents_b:
                # z_list: list[tensor] shaped as model expects
                z_dev = [t.to(dev).float() for t in z_list]
                y = model.decode(z_dev)  # [C,T] at model's native sr
                outs.append(y.unsqueeze(0).cpu())

        y_cat = torch.cat(outs, dim=0)  # [B,C,T]
        
        # FIX: Resample back to original sample rate if needed
        if model_sr != sr:
            y_resampled, _ = _resample(y_cat, model_sr, sr)
        else:
            y_resampled = y_cat
            
        audio = _make_audio(sr=sr, wav=y_resampled)
        log = f"DAC decode ok: model={model_type}, B={y_cat.size(0)}, C={y_cat.size(1)}, {model_sr}->{sr}"
        return (audio, log)

# ----------------------------
# Node registration
# ----------------------------

NODE_CLASS_MAPPINGS = {
    "Egregora_RNNoise_Denoise": Egregora_RNNoise_Denoise,
    "Egregora_WPE_Dereverb": Egregora_WPE_Dereverb,
    "Egregora_DeepFilterNet_Denoise": Egregora_DeepFilterNet_Denoise,
    "Egregora_DAC_Encode": Egregora_DAC_Encode,
    "Egregora_DAC_Decode": Egregora_DAC_Decode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Egregora_RNNoise_Denoise": "Egregora RNNoise Denoise",
    "Egregora_WPE_Dereverb": "Egregora WPE Dereverb",
    "Egregora_DeepFilterNet_Denoise": "Egregora DeepFilterNet Denoise",
    "Egregora_DAC_Encode": "Egregora DAC Encode",
    "Egregora_DAC_Decode": "Egregora DAC Decode",

}