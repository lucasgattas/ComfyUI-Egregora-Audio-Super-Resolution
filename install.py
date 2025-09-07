from pathlib import Path
import zipfile, requests, hashlib, os
from huggingface_hub import hf_hub_download

THIS = Path(__file__).resolve()
PKG = THIS.parent
COMFY_ROOT = (PKG.parent.parent if PKG.parent.name == "custom_nodes" else PKG.parent)
DEPS = PKG / "deps"
FLASH_REPO_DIR = DEPS / "FlashSR_Inference"
WEIGHTS_DIR = COMFY_ROOT / "models" / "audio" / "flashsr"

DEPS.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def _download(url: str, dst: Path, sha256: str | None = None):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    data = r.content
    if sha256 and hashlib.sha256(data).hexdigest().lower() != sha256.lower():
        raise RuntimeError(f"SHA256 mismatch for {url}")
    dst.write_bytes(data)

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

if __name__ == "__main__":
    grab_repo_zip()
    try_fetch_weights()
    print("[Egregora] Install complete.")
