
import argparse, torch, numpy as np, soundfile as sf
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-sr", type=int, default=48000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = "cuda" if args.device in ("auto","cuda") and torch.cuda.is_available() else "cpu"
    wav, sr = sf.read(args.inp, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        if wav.shape[0] < wav.shape[1]:
            wav = wav.T
        wav = wav.mean(axis=0)
    x = torch.from_numpy(wav).float().to(dev)
    x = torch.nn.functional.pad(x, (0, 64))[: wav.shape[0]]
    out = x.detach().cpu().numpy()
    sf.write(args.out, out, args.target_sr)
    print("OK")
if __name__ == "__main__":
    main()
