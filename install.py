#!/usr/bin/env python3
from pathlib import Path
import zipfile, requests, hashlib, os, sys, subprocess

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # we'll fall back to plain HTTP

THIS = Path(__file__).resolve()
PKG = THIS.parent
COMFY_ROOT = (PKG.parent.parent if PKG.parent.name == "custom_nodes" else PKG.parent)
DEPS = PKG / "deps"
FLASH_REPO_DIR = DEPS / "FlashSR_Inference"
WEIGHTS_DIR = COMFY_ROOT / "models" / "audio" / "flashsr"

DEPS.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def _run(cmd, check=True):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=check)

def _download(url: str, dst: Path, sha256: str | None = None, stream=False):
    with requests.get(url, timeout=180, stream=stream) as r:
        r.raise_for_status()
        if stream:
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        else:
            data = r.content
            if sha256 and hashlib.sha256(data).hexdigest().lower() != sha256.lower():
                raise RuntimeError(f"SHA256 mismatch for {url}")
            dst.write_bytes(data)

def grab_repo_zip():
    if FLASH_REPO_DIR.exists() and (FLASH_REPO_DIR / "FlashSR").exists():
        print("[Egregora] FlashSR_Inference already present:", FLASH_REPO_DIR)
        return
    print("[Egregora] Fetching FlashSR_Inference repository…")
    url = "https://github.com/jakeoneijk/FlashSR_Inference/archive/refs/heads/main.zip"
    zpath = DEPS / "FlashSR_Inference.zip"
    _download(url, zpath, stream=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(DEPS)
    zpath.unlink(missing_ok=True)
    # Find extracted dir (FlashSR_Inference-main or similar)
    inner = next((p for p in DEPS.glob("FlashSR_Inference-*") if p.is_dir()), None)
    if not inner:
        raise RuntimeError("Could not locate extracted FlashSR_Inference-* folder.")
    inner.rename(FLASH_REPO_DIR)
    print("[Egregora] FlashSR_Inference ready at:", FLASH_REPO_DIR)

def pip_editable_install():
    """Install FlashSR_Inference as a package so `import FlashSR` works without sys.path hacks."""
    try:
        _run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], check=True)
        _run([sys.executable, "-m", "pip", "install", "-e", str(FLASH_REPO_DIR)], check=True)
        print("[Egregora] Installed FlashSR_Inference as editable package.")
    except Exception as e:
        print(f"[Egregora] Editable install failed ({e}). Will rely on the node's sys.path fallback.")

def try_fetch_weights():
    # Default to the public dataset unless the user overrides via EGREGORA_FLASHSR_HF_REPO
    hf_repo = os.environ.get("EGREGORA_FLASHSR_HF_REPO", "jakeoneijk/FlashSR_weights")
    need = ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"]

    missing = [n for n in need if not (WEIGHTS_DIR / n).exists()]
    if not missing:
        print("[Egregora] FlashSR weights present:", WEIGHTS_DIR)
        return

    print(f"[Egregora] Missing weights {missing} — downloading from {hf_repo} …")
    if hf_hub_download:
        try:
            for fname in missing:
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=fname,
                    repo_type="dataset",  # weights are in a dataset repo
                    local_dir=str(WEIGHTS_DIR),
                )
                print(f"[Egregora] Downloaded {fname} via huggingface_hub.")
            print("[Egregora] All weights ready:", WEIGHTS_DIR)
            return
        except Exception as e:
            print(f"[Egregora] huggingface_hub failed ({e}); falling back to plain HTTP…")

    # Plain HTTP fallback
    for fname in missing:
        url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{fname}?download=true"
        dst = WEIGHTS_DIR / fname
        _download(url, dst, stream=True)
        print(f"[Egregora] Downloaded {fname} via HTTP.")
    print("[Egregora] All weights ready:", WEIGHTS_DIR)

if __name__ == "__main__":
    grab_repo_zip()
    pip_editable_install()   # safe no-op if it fails; node has a fallback
    try_fetch_weights()
    print("\n[Egregora] Install complete. Restart ComfyUI and run the FlashSR node.\n")
