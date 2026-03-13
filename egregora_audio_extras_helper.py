import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio


def _load_audio_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    wav = torch.from_numpy(d["waveform"]).float()
    sr = int(d["sample_rate"])
    meta = json.loads(str(d.get("meta_json", "{}")))
    return wav, sr, meta


def _save_audio_npz(path: Path, wav: torch.Tensor, sr: int, meta: dict):
    np.savez_compressed(
        path,
        waveform=wav.detach().cpu().numpy().astype(np.float32),
        sample_rate=np.int64(sr),
        meta_json=json.dumps(meta),
    )


def _resample(wav: torch.Tensor, sr_in: int, sr_out: int):
    if sr_in == sr_out:
        return wav, sr_in
    out = []
    for b in range(wav.shape[0]):
        out.append(torchaudio.functional.resample(wav[b], sr_in, sr_out))
    return torch.stack(out, dim=0), sr_out


def _to_mono(wav: torch.Tensor):
    if wav.size(1) == 1:
        return wav
    return wav.mean(dim=1, keepdim=True)


def _pick_device(choice: str):
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return choice


def _vad_probs_rms_48k(x48_np):
    hop = 480
    n = (len(x48_np) + hop - 1) // hop
    rms = []
    for i in range(n):
        fr = x48_np[i * hop:(i + 1) * hop]
        rms.append(float(np.sqrt(np.mean(fr * fr))) if len(fr) else 0.0)
    rms = np.asarray(rms, dtype=np.float32)
    p95 = float(np.percentile(rms, 95)) or 1e-6
    return np.clip(rms / p95, 0.0, 1.0).astype(np.float32)


def _smooth_probs(probs, smooth_ms: int):
    if probs is None or probs.size == 0 or smooth_ms <= 0:
        return probs
    import math
    hop_ms = 10.0
    tau = max(1e-3, float(smooth_ms))
    alpha = math.exp(-hop_ms / tau)
    y = np.empty_like(probs)
    acc = probs[0]
    for i, p in enumerate(probs):
        acc = alpha * acc + (1.0 - alpha) * p
        y[i] = acc
    return y


def _strength_per_frame(base_s, vad_smooth, adaptive_mode, adaptive_amount, vad_threshold):
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


def _gains_from_strength(s_eff, curve):
    import math
    s = np.clip(s_eff, 0.0, 1.0).astype(np.float32)
    if curve == "equal_power":
        g_wet = np.sin(0.5 * math.pi * s, dtype=np.float32)
        g_dry = np.cos(0.5 * math.pi * s, dtype=np.float32)
    else:
        g_wet = s
        g_dry = 1.0 - s
    return g_dry.astype(np.float32), g_wet.astype(np.float32)


def op_deepfilternet(inp: Path, out: Path, params: dict):
    from df.enhance import enhance, init_df
    from df.io import resample

    wav, sr, meta = _load_audio_npz(inp)
    if params.get("stereo_mode", "per_channel") == "downmix_mono":
        wav = _to_mono(wav)
    B, C, T = wav.shape

    x_ct = wav.reshape(-1, T).to(torch.float32)
    x48 = resample(x_ct, sr, 48000) if sr != 48000 else x_ct

    dev = _pick_device(params.get("device", "auto"))
    model, df_state, _ = init_df(params.get("dfn_model", "DeepFilterNet2"), config_allow_defaults=True)
    model = model.to(dev).eval()

    wet_ch = []
    with torch.no_grad():
        for ch in range(x48.shape[0]):
            xin = x48[ch:ch + 1]
            y = enhance(model, df_state, xin)
            wet_ch.append(y)
    wet48 = torch.cat(wet_ch, dim=0)

    wet = resample(wet48, 48000, sr) if sr != 48000 else wet48
    dry = x_ct if sr == 48000 else resample(x_ct, 48000, sr)

    hop = int(sr * 0.010)
    out_ch = []
    for ch in range(dry.shape[0]):
        dry_np = dry[ch].detach().cpu().numpy()
        wet_np = wet[ch].detach().cpu().numpy()
        probs = _vad_probs_rms_48k(dry_np if sr == 48000 else resample(dry[ch:ch + 1], sr, 48000)[0].cpu().numpy())
        vad_s = _smooth_probs(probs, int(params.get("vad_smooth_ms", 60)))
        s_eff = _strength_per_frame(
            params.get("strength", 0.65),
            vad_s,
            params.get("adaptive_mode", "more_on_noise"),
            params.get("adaptive_amount", 0.45),
            params.get("vad_threshold", 0.90),
        )
        if s_eff.ndim == 0:
            s_per = np.full(dry_np.shape[0], float(s_eff), dtype=np.float32)
        else:
            s_per = np.repeat(s_eff, max(1, hop))[:dry_np.shape[0]].astype(np.float32)
        g_dry_np, g_wet_np = _gains_from_strength(s_per, params.get("mix_curve", "equal_power"))
        y_np = np.clip(g_dry_np * dry_np + g_wet_np * wet_np, -1.0, 1.0)
        out_ch.append(torch.from_numpy(y_np))

    y = torch.stack(out_ch, dim=0).reshape(B, C, -1)
    if float(params.get("post_gain_db", 0.0)) != 0.0:
        y = y * float(10.0 ** (float(params["post_gain_db"]) / 20.0))
    if bool(params.get("limit_ceiling", True)):
        ceiling = float(params.get("ceiling", 0.98))
        peak = torch.max(torch.abs(y)).item()
        if peak > ceiling and peak > 0:
            y = y * (ceiling / peak)
    y = torch.clamp(y, -1.0, 1.0)

    meta2 = dict(meta)
    meta2["deepfilternet"] = {"device": dev, "mode": "helper_venv"}
    _save_audio_npz(out, y, sr, meta2)


def op_dac_encode(inp: Path, out: Path, params: dict):
    import dac

    wav, sr, _ = _load_audio_npz(inp)
    model_type = params.get("model_type", "44khz")
    dev = _pick_device(params.get("device", "auto"))
    ckpt = dac.utils.download(model_type=model_type)
    model = dac.DAC.load(ckpt).to(dev)
    model_sr = model.sample_rate

    z_all = []
    with torch.no_grad():
        for b in range(wav.shape[0]):
            x = wav[b].to(dev)
            x_resampled = torchaudio.functional.resample(x, sr, model_sr) if sr != model_sr else x
            x_prep = model.preprocess(x_resampled, model_sr)
            z, _, _, _, _ = model.encode(x_prep)
            z_all.append([t.detach().cpu() for t in (z if isinstance(z, (list, tuple)) else [z])])

    payload = {
        "model_type": model_type,
        "sample_rate": int(sr),
        "model_sample_rate": int(model_sr),
        "latents": z_all,
        "log": f"DAC encode ok (helper): model={model_type}",
    }
    torch.save(payload, out)


def op_dac_decode(inp: Path, out: Path, params: dict):
    import dac

    codes = torch.load(inp, map_location="cpu")
    model_type = codes.get("model_type", "44khz")
    sr = int(codes.get("sample_rate", 48000))
    model_sr = int(codes.get("model_sample_rate", sr))
    latents_b = codes.get("latents", [])
    dev = _pick_device(params.get("device", "auto"))

    ckpt = dac.utils.download(model_type=model_type)
    model = dac.DAC.load(ckpt).to(dev)

    outs = []
    with torch.no_grad():
        for z_list in latents_b:
            z_dev = [t.to(dev).float() for t in z_list]
            y = model.decode(z_dev)
            outs.append(y.unsqueeze(0).cpu())
    y_cat = torch.cat(outs, dim=0)
    if model_sr != sr:
        y_cat, _ = _resample(y_cat, model_sr, sr)
    meta = {"dac": {"mode": "helper_venv", "device": dev}}
    _save_audio_npz(out, y_cat, sr, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=["deepfilternet", "dac_encode", "dac_decode"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    inp = Path(args.inp)
    out = Path(args.out)
    params = json.loads(args.params)

    if args.op == "deepfilternet":
        op_deepfilternet(inp, out, params)
    elif args.op == "dac_encode":
        op_dac_encode(inp, out, params)
    else:
        op_dac_decode(inp, out, params)


if __name__ == "__main__":
    main()
