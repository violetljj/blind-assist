"""Replay all 1,209 real ST VL53L8CH H3 hardware CNH frames through v4.1 and v4.2 Unified Omni."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROUTE = Path(__file__).resolve().parent
REPO = ROUTE.parents[2]
ASSETS = REPO / "artifacts.local/work/egogeom-omni-v0"
BRINGUP = REPO / "artifacts.local/hardware-bringup"
CAPTURE = BRINGUP / "captures/simple-20260927T164234Z-cc4d0e"
PILOT = REPO / "artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot"

sys.path[:0] = [
    str(ROUTE),
    str(ROUTE.parent / "dtr-r0/nearfield"),
]

from egogeom_dataset import IMAGENET_MEAN, IMAGENET_STD
from egogeom_losses_and_physics import compute_gravity_aligned_ray_map
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config, batch_host_focal_from_K
from cnh_h3_recording_diagnostics import load_recording


def pack_real_h3_to_20ch(H_arr: np.ndarray, D_mm: np.ndarray, S_arr: np.ndarray, N_arr: np.ndarray) -> np.ndarray:
    """Pack real ST VL53L8CH H3 frames (N,64,16) + distance (N,64) into v4.1/v4.2 20-ch [N,20,8,8]."""
    T = H_arr.shape[0]
    out = np.zeros((T, 20, 8, 8), dtype=np.float32)
    # Valid firmware distance mask: status in {5, 9} and nb_targets > 0 and 50 <= d_mm <= 4800
    valid_d = ((S_arr == 5) | (S_arr == 9)) & (N_arr > 0) & (D_mm >= 50) & (D_mm <= 4800)
    d_m = np.where(valid_d, D_mm.astype(np.float32) / 1000.0, 0.0)
    out[:, 0] = np.log1p(d_m).reshape(T, 8, 8)
    out[:, 1] = valid_d.astype(np.float32).reshape(T, 8, 8)

    # Signed log1p(10 * |h| / scale) matching v4.1/v4.2 contract: sign(h) * log1p(10 * |h_norm|)
    h = H_arr.astype(np.float32)  # [T, 64, 16]
    # Normalize per-frame by RMS magnitude so signed ratios match training contract
    rms = np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + 1e-6)
    h_unit = h / rms
    h_enc = np.sign(h_unit) * np.log1p(10.0 * np.abs(h_unit))
    out[:, 2:18] = h_enc.transpose(0, 2, 1).reshape(T, 16, 8, 8)
    out[:, 18] = 1.0  # cnh_valid
    out[:, 19] = 1.0  # has_tof_ch
    return out


def load_anchor_rgb(device: torch.device):
    """Load deterministic indoor ARKit anchor frame (288x384) with intrinsics and gravity."""
    from eval_egogeom_v3_zeroshot import decode_isotropic_frame

    p = sorted((ASSETS / "zeroshot_eval/arkit_val30_shard00").glob("*.npz"))[0]
    with np.load(p, allow_pickle=False) as d:
        offs, blob = d["rgb_offsets"], d["rgb_blob"]
        orig_hw, K_orig = d["orig_hw"], d["K_orig"]
        poses_R = d["poses_R"]
    rgb_f, K = decode_isotropic_frame(blob[offs[0] : offs[1]].tobytes(), orig_hw, K_orig, dark_scale=1.0)
    g = (poses_R[0].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
    g /= np.linalg.norm(g) + 1e-6
    x = ((rgb_f - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
    rgb_t = torch.from_numpy(x).to(device)
    K_t = torch.from_numpy(K.astype(np.float32)[None]).to(device)
    g_t = torch.from_numpy(g[None]).to(device)
    rays_t = compute_gravity_aligned_ray_map(K_t, g_t, 288, 384)
    hf_t = batch_host_focal_from_K(K_t, 384, 288, 350.0)
    return rgb_t, rays_t, hf_t, p.stem


def load_model(ckpt_path: Path, device: torch.device) -> EgoGeomOmniV41Net:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = EgoGeomOmniV41Net(EgoGeomV41Config(**ck["cfg"]), pretrained_backbone=False).to(device).eval()
    m.load_state_dict(ck["ema"], strict=True)
    return m


def load_all_7_recordings() -> dict[str, dict]:
    recs = {}
    for label, folder in [
        ("cabinet_init", "01-background-attempt1"),
        ("cabinet_obj", "02-object-attempt2"),
        ("cabinet_move", "03-movement-attempt3"),
        ("cabinet_restored", "04-restored-attempt4"),
    ]:
        recs[label] = load_recording(CAPTURE / folder / "tof/raw.bin")
    inv = {r["id"]: r for r in json.loads((PILOT / "inventory.json").read_text(encoding="utf-8"))}
    for label, idx in [("corridor_bright", "4"), ("corridor_dark", "4b"), ("corridor_restart", "9")]:
        recs[label] = load_recording(REPO / inv[idx]["raw_path"])
    return recs


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v41", default=str(ASSETS / "egogeom_omni_v41_unified_best.pt"))
    ap.add_argument("--v42", default=str(ASSETS / "egogeom_omni_v42_unified_best.pt"))
    ap.add_argument("--out", default=str(ASSETS / "real_h3_signed_replay_v42_vs_v41.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_t, rays_t, hf_t, anchor_name = load_anchor_rgb(device)
    models = {
        "v41": load_model(Path(args.v41), device),
        "v42": load_model(Path(args.v42), device),
    }

    # Exact zero-interference verification (has_tof=0 vs None)
    zero_tof = torch.zeros(1, 20, 8, 8, device=device)
    zero_ht = torch.zeros(1, 1, 1, 1, device=device)
    anchor_pure = {}
    zero_int = {}
    for tag, m in models.items():
        d_pure = m(rgb_t, rays_t, None, None, host_focal=hf_t)["depth_m"]
        d_z = m(rgb_t, rays_t, zero_tof, zero_ht, host_focal=hf_t)["depth_m"]
        zero_int[tag] = float((d_pure - d_z).abs().max().item())
        # Center 50% ROI mean (matching ToF 45 deg FOV)
        anchor_pure[tag] = round(float(d_pure[:, :, 72:216, 96:288].mean().item()), 4)

    recs = load_all_7_recordings()
    total_frames = sum(len(d["H"]) for d in recs.values())
    report = {
        "anchor_clip": anchor_name,
        "total_real_h3_frames": total_frames,
        "anchor_pure_rgb_roi_mean_m": anchor_pure,
        "zero_interference_max_abs_m": zero_int,
        "recordings": {},
    }

    for name, d in recs.items():
        T = len(d["H"])
        tof_np = pack_real_h3_to_20ch(d["H"], d["D"], d["S"], d["N"])
        neg_frac = float((d["H"][:, :, 1:] < 0).mean())
        valid_frac = float((((d["S"] == 5) | (d["S"] == 9)) & (d["N"] > 0) & (d["D"] >= 50) & (d["D"] <= 4800)).mean())
        entry = {
            "frames": T,
            "raw_neg_tail_frac": round(neg_frac, 4),
            "raw_dist_valid_frac": round(valid_frac, 4),
        }
        for tag, m in models.items():
            means = []
            nans = 0
            bs = 32
            for i in range(0, T, bs):
                chunk = torch.from_numpy(tof_np[i : i + bs]).to(device)
                B = chunk.shape[0]
                ht = torch.ones(B, 1, 1, 1, device=device)
                out = m(rgb_t.expand(B, -1, -1, -1), rays_t.expand(B, -1, -1, -1), chunk, ht, host_focal=hf_t.expand(B, -1, -1, -1))["depth_m"]
                nans += int((~torch.isfinite(out)).sum().item())
                roi = out[:, :, 72:216, 96:288].mean(dim=(1, 2, 3)).cpu().numpy()
                means.extend(roi.tolist())
            arr = np.asarray(means, dtype=np.float64)
            entry[f"{tag}_roi_mean_m"] = round(float(arr.mean()), 4)
            entry[f"{tag}_roi_std_m"] = round(float(arr.std()), 4)
            entry[f"{tag}_nan_count"] = nans
        report["recordings"][name] = entry
        print(f"  {name:18s} ({T:3d}f): v41={entry['v41_roi_mean_m']:.4f}±{entry['v41_roi_std_m']:.4f}m | v42={entry['v42_roi_mean_m']:.4f}±{entry['v42_roi_std_m']:.4f}m", flush=True)

    report["cabinet_object_insertion_check"] = {
        f"{tag}_init_to_obj_delta_m": round(report["recordings"]["cabinet_obj"][f"{tag}_roi_mean_m"] - report["recordings"]["cabinet_init"][f"{tag}_roi_mean_m"], 4)
        for tag in ("v41", "v42")
    }
    for tag in ("v41", "v42"):
        report["cabinet_object_insertion_check"][f"{tag}_restored_vs_init_err_m"] = round(
            abs(report["recordings"]["cabinet_restored"][f"{tag}_roi_mean_m"] - report["recordings"]["cabinet_init"][f"{tag}_roi_mean_m"]), 4
        )

    out_p = Path(args.out)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
