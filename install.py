from __future__ import annotations
import sys, subprocess, importlib, os, hashlib, zipfile
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download

# ---------- Paths (kept exactly as you had) ----------
THIS = Path(__file__).resolve()
PKG = THIS.parent
COMFY_ROOT = (PKG.parent.parent if PKG.parent.name == "custom_nodes" else PKG.parent)
DEPS = PKG / "deps"
FLASH_REPO_DIR = DEPS / "FlashSR_Inference"
WEIGHTS_DIR = COMFY_ROOT / "models" / "audio" / "flashsr"

DEPS.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Small helpers ----------
def _download(url: str, dst: Path, sha256: str | None = None):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    data = r.content
    if sha256 and hashlib.sha256(data).hexdigest().lower() != sha256.lower():
        raise RuntimeError(f"SHA256 mismatch for {url}")
    dst.write_bytes(data)

def _pip_install(args: list[str]):
    print("[Egregora] pip", " ".join(args))
    cmd = [sys.executable, "-m", "pip", "install", *args]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("[Egregora] pip failed:", e)

def _ensure(import_name: str, pip_name: str | None = None, extra_args: list[str] | None = None, try_no_deps: bool = False):
    """
    Import a module, installing it if missing. When try_no_deps is True,
    we first attempt '--no-deps' to avoid pulling CPU torch into ComfyUI.
    """
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        pass

    target = pip_name or import_name
    if try_no_deps:
        _pip_install(["--no-deps", target, *(extra_args or [])])
        try:
            importlib.import_module(import_name)
            return True
        except Exception:
            print(f"[Egregora] '{target}' import still failing; retrying with full deps…")

    _pip_install([target, *(extra_args or [])])
    try:
        importlib.import_module(import_name)
        return True
    except Exception as e:
        print(f"[Egregora] Could not import {import_name}: {e}")
        return False

# ---------- Your existing FlashSR bootstrap ----------
def grab_repo_zip():
    if FLASH_REPO_DIR.exists():
        return
    print("[Egregora] Fetching FlashSR_Inference repository…")
    url = "https://github.com/jakeoneijk/FlashSR_Inference/archive/refs/heads/main.zip"
    zpath = DEPS / "FlashSR_Inference.zip"
    _download(url, zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(DEPS)
    inner = next(p for p in DEPS.glob("FlashSR_Inference-*") if p.is_dir())
    inner.rename(FLASH_REPO_DIR)
    zpath.unlink(missing_ok=True)
    print("[Egregora] FlashSR_Inference ready at:", FLASH_REPO_DIR)

def try_fetch_weights():
    # If you host the three weights on HF, set EGREGORA_FLASHSR_HF_REPO
    # (filenames must be: student_ldm.pth, sr_vocoder.pth, vae.pth)
    hf_repo = os.environ.get("EGREGORA_FLASHSR_HF_REPO", "")
    need = ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"]
    if hf_repo:
        for fname in need:
            dst = WEIGHTS_DIR / fname
            if dst.exists():
                continue
            try:
                print(f"[Egregora] Downloading {fname} from HF repo {hf_repo} …")
                hf_hub_download(repo_id=hf_repo, filename=fname, local_dir=WEIGHTS_DIR)
            except Exception as e:
                print(f"[Egregora] HF download failed for {fname}: {e}")

    missing = [n for n in need if not (WEIGHTS_DIR / n).exists()]
    if missing:
        print("\n[Egregora] FlashSR weights missing:", ", ".join(missing))
        print("Place them here:", WEIGHTS_DIR)
        print("Filenames are exactly: student_ldm.pth, sr_vocoder.pth, vae.pth")
        print("See repo for context: https://github.com/jakeoneijk/FlashSR_Inference")
    else:
        print("[Egregora] FlashSR weights present:", WEIGHTS_DIR)

# ---------- New: model/runtime deps + warmups ----------
def ensure_runtime_deps():
    # keep your requirements light; install optional bits here if missing
    _ensure("numpy")
    _ensure("soundfile")
    _ensure("tqdm")
    _ensure("requests")
    _ensure("huggingface_hub")

    # Models / processors used by your integrated nodes
    _ensure("pyrnnoise")  # RNNoise bindings
    _ensure("nara_wpe", pip_name="nara-wpe")  # dereverb
    _ensure("dac", pip_name="descript-audio-codec")  # Descript Audio Codec

    # DeepFilterNet (df). Try --no-deps first to avoid pulling a CPU torch.
    # ComfyUI already has torch/torchaudio.
    _ensure("df", pip_name="deepfilternet", try_no_deps=True)

    # Fat Llama (already in requirements, but double-check)
    _ensure("fat_llama", pip_name="fat-llama")
    _ensure("fat_llama_fftw", pip_name="fat-llama-fftw")

    # Optional: SciPy for HQ resampler/metrics in the Eval Pack
    _ensure("scipy")

def warmup_deepfilternet():
    try:
        import torch
        from df.enhance import init_df, enhance  # type: ignore
        # This triggers model settings + checkpoint discovery and caches them
        model, df_state, sr, _ = init_df()
        x = torch.zeros(1, int(sr * 0.1))  # 100 ms of silence
        with torch.no_grad():
            _y, _ = enhance(model, df_state, x)
        print("[Egregora] DeepFilterNet warmup OK")
    except Exception as e:
        print("[Egregora] DeepFilterNet warmup skipped:", e)

def warmup_dac():
    try:
        import dac
        # Downloads default weights to local cache (~first use)
        _ = dac.utils.download(model_type="44khz")
        print("[Egregora] DAC warmup OK")
    except Exception as e:
        print("[Egregora] DAC warmup skipped:", e)

def warmup_rnnoise():
    # Nothing to download, but a tiny call verifies the backend
    try:
        import numpy as np
        from pyrnnoise import RNNoise
        rn = RNNoise(sample_rate=48000)
        if getattr(rn, "channels", None) in (None, 0):
            setattr(rn, "channels", 1)
        test = np.zeros((1, 4800), dtype=np.int16)  # 100 ms
        _ = list(rn.denoise_chunk(test))  # iterate a few frames
        print("[Egregora] RNNoise warmup OK")
    except Exception as e:
        print("[Egregora] RNNoise warmup skipped:", e)

# ---------- Entry ----------
if __name__ == "__main__":
    ensure_runtime_deps()
    # Keep your original FlashSR bootstrap
    grab_repo_zip()
    try_fetch_weights()

    # Friendly first-run warmups
    warmup_deepfilternet()
    warmup_dac()
    warmup_rnnoise()

    print("[Egregora] Install complete.")
