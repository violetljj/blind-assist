"""Unified v4.2 vs v4.1 Forensic Benchmark Suite (4-Domain Zero-Shot + M6 BEV v3 + Aria Clear-Road Hallucination + Oxford Border).

Runs locally on RTX 5060 (0 CU!) right after v4.2 checkpoints finish downloading from Drive:
  1. Materializes label-free causal `v42_stream/` across all 103 clips / 12,872 frames (4 domains).
  2. Runs standard 4-domain cross-glasses evaluator (`eval_egogeom_v3_zeroshot.py --arch v41`) + BEV v3 M6 38-source cross-fit.
  3. Runs exact Aria local-ground 5-frame 8-voxel readout comparing clear-road close-hallucination frames
     (`v4.1` baseline: 519/954 = 54.4% clear-road frames falsely flagged!) vs `v4.2`!
  4. Runs Oxford 22-frame black-border vs nearest-valid-color depth MAE comparison (`v4.1` baseline: 1.911 -> 1.664 m) vs `v4.2`!
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROUTE = Path(__file__).resolve().parent
REPO = ROUTE.parents[2]
ASSETS = REPO / "artifacts.local/work/egogeom-omni-v0"
sys.path[:0] = [
    str(ROUTE),
    str(ROUTE.parent / "egoclear-v0"),
    str(ROUTE / "math_theory/m6_evt_eprocess_risk"),
    str(ROUTE / "math_theory/m9_pgl4_frobenius_integrability/aria_local_ground_readout"),
    str(ROUTE / "math_theory/m14_conformal_s2_atlas_equivariance"),
]

from egogeom_losses_and_physics import compute_gravity_aligned_ray_map
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config, batch_host_focal_from_K
from egogeom_temporal_bev import TemporalBEVConfig, run_bev_on_clip
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3
from eval_cross_glasses_zeroshot import (
    H_IN,
    IM_MEAN,
    IM_STD,
    W_IN,
    corridor_dist_from_depth,
    decode_isotropic_frame,
    temporal_bev_corridor_distances,
)
from eval_egogeom_v3_zeroshot import score_dataset

SUITES = {
    "arkit_val30_day": ("arkit_val30_shard00", 1.0, 2.0),
    "coda_campus": ("coda_zeroshot_shard000", 1.0, 4.0),
    "aria_gen2": ("aria_g2_20clips", 1.0, 4.0),
    "arkit_val30_dark018": ("arkit_val30_shard00", 0.18, 2.0),
}


def strict_headings(suite: str, packed: np.ndarray, poses_r: np.ndarray, poses_t: np.ndarray) -> np.ndarray:
    fwd = packed.copy()
    if suite == "aria_gen2":
        for i in range(min(2, len(fwd))):
            xy = poses_t[i, :2] - poses_t[0, :2]
            if np.linalg.norm(xy) < 0.15:
                xy = poses_r[i, :2, 2]
            norm = float(np.linalg.norm(xy))
            fwd[i, :2] = xy / norm if norm > 1e-6 else [1.0, 0.0]
            fwd[i, 2] = 0.0
    return fwd


@torch.inference_mode()
def materialize_and_score_4domain(model: EgoGeomOmniV41Net, device: torch.device, stream_root: Path) -> dict:
    bev_cfg = TemporalBEVConfig()
    summary = {}
    for sname, (folder, dscale, h_thr) in SUITES.items():
        npzs = sorted((ASSETS / "zeroshot_eval" / folder).rglob("*.npz"))
        out_dir = stream_root / sname
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = []
        for p in npzs:
            with np.load(p, allow_pickle=False) as d:
                offs, blob = d["rgb_offsets"], d["rgb_blob"]
                orig_hw, K_orig = d["orig_hw"], d["K_orig"]
                poses_R, poses_t, causal_fwd, eval_rows = d["poses_R"], d["poses_t"], d["causal_fwd"], d["eval_rows"]
            T = len(eval_rows)
            true_fwd = strict_headings(sname, causal_fwd, poses_R, poses_t)
            rgbs, Ks, gs = [], [], []
            for i in range(T):
                rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=dscale)
                rgbs.append(((rgb_f - IM_MEAN) / IM_STD).transpose(2, 0, 1))
                Ks.append(K)
                g_cam = (poses_R[i].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
                g_cam /= np.linalg.norm(g_cam) + 1e-6
                gs.append(g_cam)

            d_1f_wz = np.full(T, np.inf, dtype=np.float32)
            d_1f_strict = np.full(T, np.inf, dtype=np.float32)
            obs_wz_list = []
            dep_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
            dh_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
            nrm_s4 = np.empty((T, 3, H_IN // 4, W_IN // 4), dtype=np.float32)
            ls_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)

            for i0 in range(0, T, 16):
                i1 = min(T, i0 + 16)
                x = torch.from_numpy(np.stack(rgbs[i0:i1])).to(device)
                Kt = torch.from_numpy(np.stack(Ks[i0:i1])).to(device)
                gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device)
                rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
                hf = batch_host_focal_from_K(Kt, W_IN, H_IN, model.cfg.canonical_focal_px)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, rays, None, None, host_focal=hf, return_features=False)
                dep = out["depth_m"][:, 0].float().cpu().numpy()
                dh = out["delta_h_m"][:, 0].float().cpu().numpy()
                nrm = out["normal_grav"].float().cpu().numpy()
                ls = out["log_sigma"][:, 0].float().cpu().numpy()
                dep_s4[i0:i1] = dep[:, ::4, ::4]
                dh_s4[i0:i1] = dh[:, ::4, ::4]
                nrm_s4[i0:i1] = nrm[:, :, ::4, ::4]
                ls_s4[i0:i1] = ls[:, ::4, ::4]
                for j in range(i1 - i0):
                    idx = i0 + j
                    dw, ow = corridor_dist_from_depth(dep[j], None, ls[j], Ks[idx], poses_R[idx], causal_fwd[idx], use_pred_dh=False)
                    ds, _ = corridor_dist_from_depth(dep[j], None, ls[j], Ks[idx], poses_R[idx], true_fwd[idx], use_pred_dh=False)
                    d_1f_wz[idx] = dw
                    d_1f_strict[idx] = ds
                    obs_wz_list.append(ow)

            Ks_arr = np.stack(Ks).astype(np.float32)
            d_bev_v1 = temporal_bev_corridor_distances(poses_t, causal_fwd, poses_R, obs_wz_list)
            d_bev_v23, r_bev_v23, r_hyb_v23 = run_bev_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, causal_fwd, bev_cfg)
            d_bev_strict, r_bev_strict, r_hyb_strict = run_bev_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, true_fwd, bev_cfg)

            v3 = EgoGeomTemporalBEVV3()
            r_geom = np.array([v3.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks_arr[i], poses_R[i], poses_t[i], true_fwd[i])[3] for i in range(T)], dtype=np.float32)

            np.savez(
                out_dir / f"{p.stem}.npz",
                depth_s4=dep_s4, dh_s4=dh_s4, nrm_s4=nrm_s4, ls_s4=ls_s4,
                K=Ks_arr, poses_R=poses_R, poses_t=poses_t, causal_fwd=causal_fwd, strict_fwd=true_fwd,
                d_1f_wz=d_1f_wz, d_1f_strict=d_1f_strict,
                d_bev_v23=d_bev_v23, r_bev_v23=r_bev_v23, r_hyb_v23=r_hyb_v23,
                d_bev_strict=d_bev_strict, r_bev_strict=r_bev_strict, r_hyb_strict=r_hyb_strict,
                r_geom_m6=r_geom,
            )
            clips.append({
                "name": p.stem,
                "eval_rows": eval_rows,
                "d_1f_wz": d_1f_wz,
                "d_bev_v1": d_bev_v1,
                "d_bev_v23": d_bev_v23,
                "r_hyb_v23": r_hyb_v23,
                "r_geom_m6": r_geom,
            })

        summary[sname] = {
            "1f_world_z": score_dataset(clips, "d_1f_wz", None, h_thr),
            "stage2_bev_v1_proto": score_dataset(clips, "d_bev_v1", None, h_thr),
            "stage2_bev_v23_hyb": score_dataset(clips, "d_bev_v23", "r_hyb_v23", h_thr),
            "stage2_bev_v3_geom_m6": score_dataset(clips, "d_bev_v23", "r_geom_m6", h_thr),
        }
        print(f"=== {sname} ===", json.dumps(summary[sname], indent=2), flush=True)
    return summary


def score_existing_stream(stream_root: Path) -> dict:
    """Re-score already materialized 4-domain stream without re-running GPU inference."""
    summary = {}
    for sname, (folder, _dark_gain, h_thr) in SUITES.items():
        clips = []
        for p in sorted((stream_root / sname).glob("*.npz")):
            src = ASSETS / "zeroshot_eval" / folder / p.name
            with np.load(src, allow_pickle=False) as d:
                eval_rows = d["eval_rows"]
            with np.load(p, allow_pickle=False) as o:
                clips.append({
                    "name": p.stem,
                    "eval_rows": eval_rows,
                    "d_1f_wz": o["d_1f_wz"],
                    "d_bev_v23": o["d_bev_v23"],
                    "r_hyb_v23": o["r_hyb_v23"],
                    "r_geom_m6": o["r_geom_m6"],
                })
        summary[sname] = {
            "1f_world_z": score_dataset(clips, "d_1f_wz", None, h_thr),
            "stage2_bev_v23_hyb": score_dataset(clips, "d_bev_v23", "r_hyb_v23", h_thr),
            "stage2_bev_v3_geom_m6": score_dataset(clips, "d_bev_v23", "r_geom_m6", h_thr),
        }
    return summary


def run_m6_crossfit_on_stream(tag: str) -> dict:
    """Exact 38-source held-out cross-fit & single-threshold ceiling matching M6/M3 protocol."""
    import replay

    clips, stream = replay.materialize(tag)
    for c in clips:
        with np.load(stream / c.obs["domain"] / (c.obs["name"] + ".npz")) as f:
            c.geometry = f["r_geom_m6"].astype(float)

    cf_geom = replay.crossfit(clips, "geometry")
    cf_conf = replay.crossfit(clips, "confirmed")
    return {
        "m6_geom_crossfit_pooled": cf_geom["pooled"],
        "m6_geom_crossfit_domains": cf_geom["domains"],
        "m6_geom_ceiling_all": replay.ceiling(clips, "geometry"),
        "m6_geom_ceiling_aria": replay.ceiling([c for c in clips if c.obs["domain"] == "aria_gen2"], "geometry"),
        "legacy_confirmed_crossfit_pooled": cf_conf["pooled"],
        "legacy_confirmed_ceiling_all": replay.ceiling(clips, "confirmed"),
    }


def run_aria_clear_road_audit(stream_root: Path) -> dict:
    """Exact Round-3 Aria 5-frame local-ground 8-voxel clear-road hallucination audit (v4.1 baseline: 519/954!)."""
    from collections import deque
    import readout as aria_ro

    aria_dir = stream_root / "aria_gen2"
    in_dir = ASSETS / "math_theory/m9_pgl4_frobenius_integrability/aria_local_ground_readout/input"
    haz_det = haz_clr = haz_unk = 0
    clr_det = clr_clr = clr_unk = 0
    fixed_errs = []
    finite_errs = []
    for p in sorted(aria_dir.glob("*.npz")):
        src_pack = ASSETS / "zeroshot_eval/aria_g2_20clips" / p.name
        with np.load(src_pack, allow_pickle=False) as d:
            er = d["eval_rows"]
        with np.load(p, allow_pickle=False) as o:
            dep_flat = o["depth_s4"].reshape(len(er), -1)
        with np.load(in_dir / p.name, allow_pickle=False) as a:
            r, t, fw, K_nat = a["rot"], a["pos"], a["fw"], a["K"]
        history = deque(maxlen=5)
        pred_d = np.full(len(er), np.nan, dtype=np.float64)
        cov = np.full(len(er), np.nan, dtype=np.float64)
        for i, z0 in enumerate(dep_flat):
            z = z0.astype(np.float64)
            ok = np.isfinite(z) & (z > 0.30) & (z < 12.0)
            fx, fy, cx, cy = K_nat[i]
            pts = np.stack(((aria_ro.NATIVE_U[ok] - cx) * z[ok] / fx, (aria_ro.NATIVE_V[ok] - cy) * z[ok] / fy, z[ok]), axis=1)
            history.append((pts @ r[i].T + t[i]).astype(np.float32))
            d_i, c_i, _ = aria_ro.corridor(np.concatenate(history), t[i].astype(float), fw[i])
            pred_d[i], cov[i] = d_i, c_i

        D_gt = er[:, 0]
        gt_val = (np.isfinite(D_gt) | np.isinf(D_gt)) & ~np.isnan(er[:, 1]) & (er[:, 1] >= 0.6)
        gt_haz = gt_val & np.isfinite(D_gt) & (D_gt <= 4.0)
        gt_clr = gt_val & (~gt_haz)
        p_unk = (~np.isfinite(cov)) | (cov < 0.6)
        p_haz = (~p_unk) & np.isfinite(pred_d) & (pred_d <= 4.0)
        p_clr = (~p_unk) & (~p_haz)

        haz_det += int((gt_haz & p_haz).sum())
        haz_clr += int((gt_haz & p_clr).sum())
        haz_unk += int((gt_haz & p_unk).sum())
        clr_det += int((gt_clr & p_haz).sum())
        clr_clr += int((gt_clr & p_clr).sum())
        clr_unk += int((gt_clr & p_unk).sum())

        if gt_haz.any():
            d_fill = np.where(p_haz[gt_haz], pred_d[gt_haz], 6.0)
            fixed_errs.append(np.abs(d_fill - D_gt[gt_haz]))
            mh = gt_haz & p_haz
            if mh.any():
                finite_errs.append(np.abs(pred_d[mh] - D_gt[mh]))

    return {
        "haz_840_det_clr_unk": [haz_det, haz_clr, haz_unk],
        "clr_954_halluc_clr_unk": [clr_det, clr_clr, clr_unk],
        "clear_road_hallucination_pct": round(100.0 * clr_det / max(1, clr_det + clr_clr + clr_unk), 2),
        "fixed_mae_m": round(float(np.concatenate(fixed_errs).mean()), 3),
        "finite_mae_m": round(float(np.concatenate(finite_errs).mean()), 3) if finite_errs else -1.0,
        "v41_reference_clear_halluc": "519/954 (54.40%), haz=604/840, fixed_mae=1.809m",
    }


@torch.inference_mode()
def run_oxford_border_audit(model: EgoGeomOmniV41Net, device: torch.device) -> dict:
    """Exact M14 Oxford 22-frame rectified-fisheye black-border vs nearest-valid-color MAE audit (v4.1: 1.911 -> 1.664 m)."""
    import importlib.util
    from scipy.ndimage import distance_transform_edt
    from atlas import level_basis, pin_rays, pull

    m14_path = ROUTE / "math_theory/m14_conformal_s2_atlas_equivariance/eval_empirical.py"
    spec = importlib.util.spec_from_file_location("m14_eval_empirical", m14_path)
    m14_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m14_mod)
    oxford_inputs = m14_mod.oxford_inputs

    k = (224.0, 126.0, 192.0, 144.0)
    cam = pin_rays(288, 384, k)
    err_sum = {"black": 0.0, "nearest_valid": 0.0}
    n_pix = 0
    for row in oxford_inputs():
        basis = level_basis(row["down"])
        p, ok = row["project"](cam)
        image, valid = pull(row["image"], p, ok)
        rays = (cam @ basis.T).astype(np.float32)
        gtxy, good = row["refproject"](cam)
        gt, gm = pull(
            cv2.imread(str(row["ref_path"]), cv2.IMREAD_UNCHANGED).astype(np.float32) * row["ref_scale"],
            gtxy, good, nearest=True,
        )
        gm &= valid & (gt >= 0.3) & (gt <= 12.0)
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        variants = {"black": image, "nearest_valid": image[tuple(nearest)]}
        rays_t = torch.from_numpy(rays.transpose(2, 0, 1)[None]).to(device)
        du = rays_t[:, :, :, 1:] - rays_t[:, :, :, :-1]
        inv_fx = torch.sqrt((du * du).sum(dim=1, keepdim=True) + 1e-12).amax(dim=(2, 3), keepdim=True)
        hf = (inv_fx * model.cfg.canonical_focal_px).clamp(0.20, 4.0)
        for name, bgr in variants.items():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(((rgb - IM_MEAN) / IM_STD).transpose(2, 0, 1)[None]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = model(x, rays_t, None, None, host_focal=hf)["depth_m"][0, 0].float().cpu().numpy()
            err_sum[name] += float(np.abs(z[gm] - gt[gm]).sum(dtype=float))
        n_pix += int(gm.sum())
    return {
        "n_pixels": n_pix,
        "black_mae_m": round(err_sum["black"] / max(1, n_pix), 4),
        "nearest_valid_mae_m": round(err_sum["nearest_valid"] / max(1, n_pix), 4),
        "border_gap_m": round((err_sum["black"] - err_sum["nearest_valid"]) / max(1, n_pix), 4),
        "v41_reference": "black=1.911 m, nearest_valid=1.664 m, gap=0.247 m",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="v42")
    ap.add_argument("--reuse-stream", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = EgoGeomOmniV41Net(EgoGeomV41Config(**ck["cfg"]), pretrained_backbone=False).to(device).eval()
    model.load_v4_pure_rgb_trunk(ck["ema"])
    print(f"Loaded {args.tag} checkpoint {Path(args.ckpt).name} on {device}", flush=True)

    stream_root = ASSETS / f"math_theory/{args.tag}_stream"
    if args.reuse_stream and (stream_root / "aria_gen2").exists():
        four_dom = score_existing_stream(stream_root)
    else:
        four_dom = materialize_and_score_4domain(model, device, stream_root)
    res = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "four_domain": four_dom,
        "m6_38source_crossfit": run_m6_crossfit_on_stream(args.tag),
        "aria_clear_road_audit": run_aria_clear_road_audit(stream_root),
        "oxford_border_audit": run_oxford_border_audit(model, device),
    }
    out_path = ASSETS / f"{args.tag}_full_forensic_audit.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print("M6 38-source crossfit:", json.dumps(res["m6_38source_crossfit"], indent=2), flush=True)
    print("Aria clear-road audit:", json.dumps(res["aria_clear_road_audit"], indent=2), flush=True)
    print("Oxford border audit:", json.dumps(res["oxford_border_audit"], indent=2), flush=True)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
