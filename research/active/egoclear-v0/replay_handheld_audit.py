"""Offline Handheld Replay Regression Evaluator for EgoClear (v6.6 / v10.5 / v10.6 / v11).

Reads recorded `.egoclear` replay clips captured directly from `:egoclear-demo-app` (via the
in-app "录回放" button, pulled from `/sdcard/Android/data/com.linnan.blindassist.egoclear.demo/files/replays/`)
OR single diagnostic frame triplets (`diag_rgb.raw`, `diag_K.raw`, `diag_R.raw`), and replays ANY list of
exported ONNX models side-by-side in seconds with ZERO phone rebuilds needed!

Binary `.egoclear` v1 record format (little-endian):
  Header: magic b"EGC1" (4B) + n_frames (uint32) + rgb_h (uint32) + rgb_w (uint32)
  Per frame:
    - K_scaled:  4 x float32
    - R_c2w:     9 x float32
    - t_world:   3 x float32
    - fwd_world: 3 x float32
    - ds_meters: 1 x float32
    - pitch_deg: 1 x float32
    - roll_deg:  1 x float32
    - rgb_u8:    3 * rgb_h * rgb_w uint8 (NCHW [0..255], reconstructed to ImageNet-normalized float32)
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
Z_EDGES = np.array([0.30 + 0.20 * (i + 1) for i in range(24)], dtype=np.float32)


def load_egoclear_replay(path: Path) -> list[dict]:
    data = path.read_bytes()
    magic, n_frames, rgb_h, rgb_w = struct.unpack_from("<4sIII", data, 0)
    if magic != b"EGC1":
        raise ValueError(f"Unsupported replay magic {magic!r} in {path}")
    off = 16
    rgb_bytes = 3 * rgb_h * rgb_w
    frames = []
    for _ in range(n_frames):
        meta = struct.unpack_from("<22f", data, off)
        off += 22 * 4
        rgb_u8 = np.frombuffer(data, dtype=np.uint8, count=rgb_bytes, offset=off).reshape(3, rgb_h, rgb_w)
        off += rgb_bytes
        rgb_norm = (rgb_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        frames.append(
            {
                "K": np.array(meta[0:4], dtype=np.float32).reshape(1, 4),
                "R": np.array(meta[4:13], dtype=np.float32).reshape(1, 3, 3),
                "t": np.array(meta[13:16], dtype=np.float32).reshape(1, 3),
                "fwd": np.array(meta[16:19], dtype=np.float32).reshape(1, 3),
                "ds": np.array([meta[19]], dtype=np.float32),
                "pitch": float(meta[20]),
                "roll": float(meta[21]),
                "rgb": rgb_norm[None],
            }
        )
    return frames


def replay_model(name: str, enc_onnx: Path, step_onnx: Path, frames: list[dict], thr: float = 0.50) -> dict:
    se = ort.InferenceSession(str(enc_onnx), providers=["CPUExecutionProvider"])
    ss = ort.InferenceSession(str(step_onnx), providers=["CPUExecutionProvider"])
    aux_in_name = next((i.name for i in ss.get_inputs() if i.name.endswith("_t") and i.name != "f2d_t"), "geom_t")
    aux_pfx = aux_in_name[:-2]
    has_omni = any(o.name == "omni_probs" for o in ss.get_outputs())

    mem = np.zeros((1, 128, 24, 15), dtype=np.float32)
    age = np.full((1, 1, 24, 15), 10.0, dtype=np.float32)
    hist: list[tuple] = []

    p05, p13, p19, p41 = [], [], [], []
    drop19, head19, dcl27, unc41 = [], [], [], []
    hit_m_list = []

    for fr in frames:
        f2d, aux = se.run(None, {"rgb": fr["rgb"], "K_scaled": fr["K"], "R_c2w": fr["R"]})
        cur = (f2d, aux, fr["K"], fr["R"], fr["t"], fr["fwd"])
        hist.append(cur)
        if len(hist) > 3:
            hist.pop(0)
        p1 = hist[-2] if len(hist) >= 2 else cur
        p2 = hist[-3] if len(hist) >= 3 else p1

        feed = {
            "f2d_t": cur[0], f"{aux_pfx}_t": cur[1],
            "f2d_1": p1[0], f"{aux_pfx}_1": p1[1],
            "f2d_2": p2[0], f"{aux_pfx}_2": p2[1],
            "K_t": cur[2], "R_t": cur[3], "t_t": cur[4], "fwd_t": cur[5],
            "K_1": p1[2], "R_1": p1[3], "t_1": p1[4], "fwd_1": p1[5],
            "K_2": p2[2], "R_2": p2[3], "t_2": p2[4], "ds_t": fr["ds"],
            "mem_prev": mem, "age_prev": age,
        }
        out = ss.run(None, feed)
        cp = out[2][0]
        unc = 1.0 / (1.0 + np.exp(-float(out[3][0, 18])))
        if has_omni:
            omni = out[4][0]  # (6, 24)
            mem, age = out[5], out[6]
            head19.append(float(omni[3, 7]))
            drop19.append(float(omni[4, 7]))
            dcl27.append(float(omni[5, 11]))
        else:
            mem, age = out[4], out[5]

        p05.append(float(cp[0]))
        p13.append(float(cp[4]))
        p19.append(float(cp[7]))
        p41.append(float(cp[18]))
        unc41.append(unc)
        hit = next((float(Z_EDGES[i]) for i in range(19) if cp[i] >= thr), None)
        hit_m_list.append(hit)

    return {
        "model": name,
        "frames": len(frames),
        "has_omni": has_omni,
        "mean_p0.5m": round(float(np.mean(p05)), 4),
        "mean_p1.3m": round(float(np.mean(p13)), 4),
        "mean_p1.9m": round(float(np.mean(p19)), 4),
        "mean_p4.1m": round(float(np.mean(p41)), 4),
        "alert_rate_1.3m@thr": round(float(np.mean(np.array(p13) >= thr)), 4),
        "alert_rate_1.9m@thr": round(float(np.mean(np.array(p19) >= thr)), 4),
        "alert_rate_4.1m@thr": round(float(np.mean(np.array(p41) >= thr)), 4),
        "mean_unc_4.1m": round(float(np.mean(unc41)), 4),
        "max_drop_1.9m": round(float(np.max(drop19)), 4) if drop19 else None,
        "max_head_1.9m": round(float(np.max(head19)), 4) if head19 else None,
        "max_dyn_closing_2.7m": round(float(np.max(dcl27)), 4) if dcl27 else None,
        "last_hit_m": hit_m_list[-1] if hit_m_list else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline handheld replay audit for EgoClear ONNX models.")
    ap.add_argument("--replay", type=Path, required=True, help="Path to .egoclear replay clip.")
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec formatted as NAME=ENC_ONNX:STEP_ONNX (repeatable).",
    )
    args = ap.parse_args()
    frames = load_egoclear_replay(args.replay)
    rows = []
    for spec in args.model:
        name, paths = spec.split("=", 1)
        enc_s, stp_s = paths.rsplit(":", 1)
        res = replay_model(name, Path(enc_s), Path(stp_s), frames, thr=args.thr)
        rows.append(res)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
