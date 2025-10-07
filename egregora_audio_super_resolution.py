import os, sys, time, tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import soundfile as sf
import torch

# Extended outputs: main AUDIO + (ABX A,B,X,meta) + (null audio + metrics)
RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "DICT", "AUDIO", "DICT")
FUNCTION = "run"
CATEGORY = "Egregora/Audio"


# ---------- paths ----------
def _custom_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p.parent / "models").exists() or (p.parent / "output").exists():
            return p.parent
        p = p.parent
    # fallback: 3 dirs up (usual ComfyUI layout)
    return Path(__file__).resolve().parents[3]


def _models_dir() -> Path:
    env = os.environ.get("EGREGORA_MODELS_DIR")
    return Path(env) if env else (_custom_root() / "models")


def _audio_models_subdir(name: str) -> Path:
    d = _models_dir() / "audio" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_dir() -> Path:
    # Put sidecar files under ComfyUI/output/audio
    root = _custom_root()
    out = root / "output" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------- I/O helpers ----------
def _to_cs(x: np.ndarray) -> np.ndarray:
    """Return channels-first float32 [C, S]. Accepts [S], [S, C], [C, S]."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        h, w = a.shape
        if w <= 8 and h > w:  # soundfile often gives [S,C]
            a = a.T
    else:
        a = a.reshape(-1)[None, :]
    m = np.max(np.abs(a)) if a.size else 0.0
    if m > 1.0:
        a = a / (m + 1e-8)
    return a.astype(np.float32)


def _save_temp_wav(cs: np.ndarray, sr: int) -> Path:
    p = Path(tempfile.gettempdir()) / f"eg_in_{int(time.time()*1000)}.wav"
    sf.write(str(p), cs.T, sr)
    return p


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampler. x: [C, S]"""
    if src_sr == dst_sr:
        return x
    ratio = dst_sr / src_sr
    n_out = int(round(x.shape[-1] * ratio))
    t_in = np.linspace(0.0, 1.0, x.shape[-1], endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float64)
    if x.ndim == 1:
        return np.interp(t_out, t_in, x).astype(np.float32)
    out = np.stack([np.interp(t_out, t_in, ch) for ch in x], axis=0).astype(np.float32)
    return out


def _make_audio(sr: int, samples_cn: np.ndarray, meta: Optional[dict] = None) -> Dict[str, Any]:
    """Build a ComfyUI-compatible AUDIO dict that also includes 'samples' for our helpers."""
    s = np.asarray(samples_cn, dtype=np.float32)
    if s.ndim == 1:
        s = s[None, :]
    wf = torch.from_numpy(s).unsqueeze(0).contiguous()  # [1,C,T]
    return {"sample_rate": int(sr), "waveform": wf, "samples": s, "meta": dict(meta or {})}


# ---------- chunking ----------
def _chunker(wav_path: Path, seconds: float, overlap: float) -> List[Tuple[Path, float, int]]:
    """
    Split *without downmixing*; preserve channels.
    Returns: list of (chunk_path, start_seconds, sr).
    """
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if y.ndim == 1:
        y = y[:, None]  # [S] -> [S,1]
    n = y.shape[0]
    win = int(seconds * sr)
    if n <= win:
        tmp = Path(tempfile.gettempdir()) / f"eg_chunk_full_{int(time.time()*1000)}.wav"
        sf.write(str(tmp), y, sr)
        return [(tmp, 0.0, sr)]
    hop = max(1, int((seconds - overlap) * sr))
    out = []
    i = 0
    while i < n:
        j = min(n, i + win)
        piece = y[i:j, :]  # [L, C]
        tmp = Path(tempfile.gettempdir()) / f"eg_chunk_{i}_{int(time.time()*1000)}.wav"
        sf.write(str(tmp), piece, sr)
        out.append((tmp, i / sr, sr))
        if j >= n:
            break
        i += hop
    return out


def _overlap_add_multich(chunks: List[Tuple[np.ndarray, int, float]], seconds: float, overlap: float) -> Tuple[np.ndarray, int]:
    """
    Multi-channel overlap-add with Hann window, preserving channels.
    chunks: list of (y_cs [C,S], sr, start_seconds)
    Returns (out_cs [C,S], sr)
    """
    if not chunks:
        return np.zeros((1, 1), np.float32), 48000
    sr = int(chunks[0][1])
    win = max(1, int(seconds * sr))
    ends, C0 = [], None
    for y_cs, _, start in chunks:
        if y_cs.ndim == 1:
            y_cs = y_cs[None, :]
        if C0 is None:
            C0 = y_cs.shape[0]
        L = y_cs.shape[1]
        ends.append(int(start * sr) + L)
    total = max(ends)
    acc = np.zeros((C0, total), np.float32)
    wsum = np.zeros(total, np.float32)
    w = np.hanning(win).astype(np.float32) if win > 4 else np.ones(win, np.float32)
    for y_cs, _, start_s in chunks:
        y_cs = y_cs if y_cs.ndim == 2 else y_cs[None, :]
        start = int(start_s * sr)
        L = y_cs.shape[1]
        ww = w[:L] if L <= w.shape[0] else np.ones(L, np.float32)
        acc[:, start:start+L] += y_cs[:, :L] * ww[None, :]
        wsum[start:start+L] += ww
    wsum[wsum == 0] = 1.0
    out = acc / wsum[None, :]
    return out.astype(np.float32), sr


# ---------- FlashSR runner ----------
class _FlashSRRunner:
    REQ_SR = 48000
    CHUNK_SAMPLES = 245760  # 5.12 s @ 48k
    HF_DATASET = "jakeoneijk/FlashSR_weights"  # dataset repo id
    HF_FILES = ("student_ldm.pth", "sr_vocoder.pth", "vae.pth")  # required files

    def __init__(self, device: str = "auto", lowpass: bool = False, repo: Optional[Path] = None):
        self.device = device
        self.lowpass = lowpass
        self.ckpt_dir = _audio_models_subdir("flashsr")
        env_repo = os.environ.get("EGREGORA_FLASHSR_REPO")
        self.repo_path = Path(env_repo) if env_repo else (repo if repo else (Path(__file__).parents[1] / "deps" / "FlashSR_Inference"))
        self._FlashSRClass = None
        self._model = None
        self._dev = self._pick_device()

        # Ensure weights exist (download if missing)
        self._ensure_weights()

    def _pick_device(self) -> torch.device:
        if self.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_weights(self):
        missing = [n for n in self.HF_FILES if not (self.ckpt_dir / n).exists()]
        if not missing:
            return
        print(f"[FlashSR] Missing weights in {self.ckpt_dir}: {', '.join(missing)} — downloading from Hugging Face…")
        try:
            from huggingface_hub import hf_hub_download  # robust/resumable
            for fname in missing:
                hf_hub_download(
                    repo_id=self.HF_DATASET,
                    filename=fname,
                    repo_type="dataset",
                    local_dir=str(self.ckpt_dir)
                )
                print(f"[FlashSR] Downloaded via huggingface_hub: {fname}")
            return
        except Exception as e:
            print(f"[FlashSR] huggingface_hub unavailable or failed ({e}); falling back to direct HTTP…")
        try:
            import requests
            for fname in missing:
                url = f"https://huggingface.co/datasets/{self.HF_DATASET}/resolve/main/{fname}?download=true"
                dst = self.ckpt_dir / fname
                with requests.get(url, stream=True, timeout=3600) as r:
                    r.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                print(f"[FlashSR] Downloaded: {dst}")
        except Exception as ee:
            raise RuntimeError(
                "FlashSR weights are missing and automatic download failed. "
                "Place these files into models/audio/flashsr: student_ldm.pth, sr_vocoder.pth, vae.pth"
            ) from ee

    def _import(self):
        if self._FlashSRClass is not None:
            return self._FlashSRClass
        try:
            from FlashSR.FlashSR import FlashSR  # type: ignore
            self._FlashSRClass = FlashSR
            return FlashSR
        except Exception:
            cand = self.repo_path
            if (cand / "FlashSR").exists():
                sys.path.insert(0, str(cand))
                from FlashSR.FlashSR import FlashSR  # type: ignore
                self._FlashSRClass = FlashSR
                return FlashSR
            raise RuntimeError(
                "FlashSR repo not importable. Install jakeoneijk/FlashSR_Inference "
                "or place it under deps/FlashSR_Inference, or set EGREGORA_FLASHSR_REPO."
            )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        FlashSR = self._import()
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
        return model

    def run_chunk(self, wav_path: Path) -> Path:
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        cs = _to_cs(y)
        if sr != self.REQ_SR:
            cs = _resample_linear(cs, sr, self.REQ_SR)
        T = cs.shape[1]
        if T < self.CHUNK_SAMPLES:
            pad = np.zeros((cs.shape[0], self.CHUNK_SAMPLES - T), np.float32)
            cs = np.concatenate([cs, pad], axis=1)
        elif T > self.CHUNK_SAMPLES:
            cs = cs[:, : self.CHUNK_SAMPLES]
        audio_t = torch.from_numpy(cs).to(self._dev).float()  # [C,T]
        model = self._ensure_model()
        with torch.inference_mode():
            pred = model(audio_t, lowpass_input=bool(self.lowpass))  # [C,T]
        out = pred.detach().to("cpu").float().numpy()
        out_wav = Path(tempfile.gettempdir()) / f"eg_flashsr_{int(time.time()*1000)}.wav"
        sf.write(str(out_wav), out.T, self.REQ_SR)
        return out_wav


# ---------- Node ----------
class EgregoraAudioSuperResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_seconds": ("FLOAT", {"default": 5.12, "min": 1.0, "max": 120.0, "step": 0.01}),
                "overlap_seconds": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 5.0, "step": 0.01}),
                "device": (["auto", "cuda", "cpu"],),
                "target_sr": (["auto", "48000", "44100", "32000", "22050", "16000"],),
                "output_format": (["wav", "flac"],),  # container decided by extension
            },
            "optional": {
                "AUDIO": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""}),
                "flashsr_lowpass": ("BOOLEAN", {"default": False}),

                # ABX helpers
                "run_abx": ("BOOLEAN", {"default": False}),
                "clip_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),

                # Null-test helper
                "run_null": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def _normalize_audio(self, AUDIO=None, audio_path="", audio_url=""):
        if isinstance(AUDIO, dict) and "waveform" in AUDIO and "sample_rate" in AUDIO:
            wf: torch.Tensor = AUDIO["waveform"]
            sr = int(AUDIO["sample_rate"])
            if wf.dim() == 3:
                wf = wf[0]  # [C,T]
            if wf.dim() != 2:
                raise RuntimeError(f"Unexpected AUDIO shape: {tuple(wf.shape)} (want [C,T])")
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
            p = Path(tempfile.gettempdir()) / f"eg_url_{int(time.time()*1000)}.wav"; p.write_bytes(r.content)
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
            cs = _to_cs(y)
            return cs, int(sr), _save_temp_wav(cs, int(sr))

        raise RuntimeError("No AUDIO provided.")

    def run(
        self, chunk_seconds, overlap_seconds, device, target_sr, output_format,
        AUDIO=None, audio_path="", audio_url="", flashsr_lowpass=False,
        run_abx=False, clip_seconds=10.0, random_seed=0, start_seconds=0.0,
        run_null=False
    ):
        # ---------- Normalize input ----------
        target_sr_int = 0 if target_sr == "auto" else int(target_sr)
        in_cs, in_sr, in_wav = self._normalize_audio(AUDIO, audio_path, audio_url)

        # ---------- FlashSR over chunks ----------
        chunks = _chunker(in_wav, seconds=chunk_seconds, overlap=overlap_seconds)
        rendered: List[Tuple[np.ndarray, int, float]] = []

        runner = _FlashSRRunner(device=device, lowpass=bool(flashsr_lowpass))
        for p, start, _sr in chunks:
            out = runner.run_chunk(p)  # chunk is rendered at 48k by FlashSR
            y, sr = sf.read(str(out), dtype="float32", always_2d=False)
            cs_y = _to_cs(y)  # [C,S]
            if target_sr_int and target_sr_int != sr:
                cs_y = _resample_linear(cs_y, sr, target_sr_int)
                sr = target_sr_int
            rendered.append((cs_y, sr, start))

        if not rendered:
            raise RuntimeError("No chunks rendered; check input and settings.")

        out_cs, out_sr = _overlap_add_multich(rendered, chunk_seconds, overlap_seconds)

        # ---------- Main AUDIO out ----------
        main_audio = _make_audio(int(out_sr), out_cs)

        # write sidecar file to ComfyUI/output/audio in chosen container
        ext = ".wav" if output_format == "wav" else ".flac"
        out_file = _output_dir() / f"flashsr_{int(time.time()*1000)}{ext}"
        sf.write(str(out_file), out_cs.T, int(out_sr))  # container chosen by extension
        print(f"[FlashSR] Wrote: {out_file}")

        # ---------- Optional ABX prep ----------
        abx_A = _make_audio(int(out_sr), _resample_linear(in_cs, in_sr, int(out_sr)))  # resample original to out_sr
        abx_B = main_audio
        abx_X = _make_audio(int(out_sr), np.zeros((abx_B['samples'].shape[0], 1), np.float32))
        abx_meta: Dict[str, Any] = {}

        if bool(run_abx):
            try:
                from .egregora_audio_eval_pack import ABX_Prepare  # local helper node
            except Exception:
                # fallback to plain prep if the module path differs (e.g., running as a single file)
                from egregora_audio_eval_pack import ABX_Prepare  # type: ignore

            A_c, B_c, X, meta = ABX_Prepare().execute(
                abx_A, abx_B, clip_seconds=float(clip_seconds),
                random_seed=int(random_seed), start_seconds=float(start_seconds)
            )
            abx_A, abx_B, abx_X, abx_meta = A_c, B_c, X, dict(meta)

        # ---------- Optional Null test (no plots, just signal+metrics) ----------
        null_audio = _make_audio(int(out_sr), np.zeros((abx_B['samples'].shape[0], 1), np.float32))
        null_metrics: Dict[str, Any] = {}

        if bool(run_null):
            try:
                from .egregora_null_test_suite import Null_Test_Full
            except Exception:
                from egregora_null_test_suite import Null_Test_Full  # type: ignore

            ap_matched, audio_null, delay_ms, gain_db, metrics, _w, _s, _d = Null_Test_Full().execute(
                audio_ref=abx_A,
                audio_proc=main_audio,
                # keep it lightweight: no figures
                draw_waveforms=False, draw_spectrograms=False, draw_diffspec=False
            )
            # we expose only the null and metrics here
            null_audio = audio_null
            # include alignment/gain in metrics for convenience
            mm = dict(metrics or {})
            mm.update({"delay_ms": float(delay_ms), "gain_db": float(gain_db)})
            null_metrics = mm

        # Return all outputs (unused ones are harmless)
        return (main_audio, abx_A, abx_B, abx_X, abx_meta, null_audio, null_metrics)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "EgregoraAudioUpscaler": EgregoraAudioSuperResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraAudioUpscaler": "🎧 Audio Super Resolution (FlashSR)",
}
