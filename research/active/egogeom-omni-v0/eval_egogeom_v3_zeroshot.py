"""Unified Strict Zero-Shot Cross-Glasses Evaluator for EgoGeom-Omni v2.1 / v3 / v3.x + Stage-2 BEV v2.3.

Evaluates any checkpoint locally on RTX 5060 (0 CU!) across:
  1. Meta Aria Gen 2 (19 real grayscale wide-FOV glasses clips, 1,794 valid / 840 haz<=4m)
  2. UT Austin CODa (24 real campus rover clips, 2,819 valid / 34 haz<=4m)
  3. ARKitScenes val30 Day (30 real indoor clips, 467 valid / 388 haz<=2m)
  4. ARKitScenes val30 Dark 0.18x (30 real indoor dark clips, 467 valid / 388 haz<=2m)

Reports BOTH:
  - Single-frame 3D projection (`1f_world_z`)
  - Stage-2 v1 proto BEV (`bev_v1_proto`, exact apples-to-apples vs v2.1 report!)
  - Stage-2 v2.3 Zone-Split GroundScaleLock BEV (`bev_v23_hyb`, <0.9 ms/frame!)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(r"e:\linnan\linnan")
sys.path.insert(0, str(REPO / "research/active/egogeom-omni-v0"))
sys.path.insert(0, str(REPO / "research/active/egoclear-v0"))

from egogeom_losses_and_physics import compute_gravity_aligned_ray_map
from egogeom_omni_model import EgoGeomConfig, EgoGeomOmniNet
from egogeom_omni_v3_model import EgoGeomOmniV3Net, EgoGeomV3Config
from egogeom_temporal_bev import TemporalBEVConfig, run_bev_on_clip
from eval_cross_glasses_zeroshot import (
    BW_SPEEDS_PROXY,
    H_IN,
    IM_MEAN,
    IM_STD,
    W_IN,
    corridor_dist_from_depth,
    decode_isotropic_frame,
    temporal_bev_corridor_distances,
)
from eval_closed_loop import make_episodes, score_episode


def score_dataset(clips: list[dict], dist_key: str, score_key: str | None, hazard_thr_m: float) -> dict:
    yt_all, ys_all, d_errs = [], [], []
    all_eps = []
    for c in clips:
        er = c["eval_rows"]
        D_gt = er[:, 0]
        val = (np.isfinite(D_gt) | np.isinf(D_gt)) & ~np.isnan(er[:, 1]) & (er[:, 1] >= 0.6)
        D_pr = c[dist_key]
        sc = c[score_key] if score_key is not None else (1.0 / np.clip(np.where(np.isfinite(D_pr), D_pr, 12.0), 0.20, 12.0))
        if val.any():
            yt_all.append((D_gt[val] <= hazard_thr_m).astype(np.int32))
            ys_all.append(sc[val])
            mh = val & np.isfinite(D_gt) & (D_gt >= 0.30) & (D_gt <= 4.0) & np.isfinite(D_pr) & (D_pr <= 6.0)
            if mh.any():
                d_errs.append(np.abs(D_pr[mh] - D_gt[mh]))
        eps = make_episodes(c["name"], er, BW_SPEEDS_PROXY, kind="blind")
        for ep in eps:
            ep["_D_pr"] = D_pr
            all_eps.append(ep)

    yt = np.concatenate(yt_all)
    ys = np.concatenate(ys_all)
    out = {
        "n_valid": int(len(yt)),
        "n_pos": int(yt.sum()),
        "dist_mae_m": round(float(np.concatenate(d_errs).mean()), 3) if d_errs else -1.0,
    }
    if 0 < yt.sum() < len(yt):
        out["auroc"] = round(float(roc_auc_score(yt, ys)), 4)
        out["ap"] = round(float(average_precision_score(yt, ys)), 4)
        neg, pos = ys[yt == 0], ys[yt == 1]
        for fpr_t in (0.05, 0.10):
            thr = float(np.quantile(neg, 1.0 - fpr_t))
            out[f"rec_fpr{int(fpr_t*100)}"] = round(float((pos >= thr).mean()), 4)
    best_fa5, best_fa10 = 100.0, 100.0
    for d_thr in np.linspace(1.4, 4.4, 31):
        late = avoid = fa_ep = 0
        km = 0.0
        for ep in all_eps:
            l, a, _, fe, _, k = score_episode(ep, ep["_D_pr"] <= d_thr)
            late += l; avoid += a; fa_ep += fe; km += k
        if avoid > 0 and km > 1e-4:
            lr = 100.0 * late / avoid
            fa_km = fa_ep / km
            if fa_km <= 5.0 and lr < best_fa5:
                best_fa5 = lr
            if fa_km <= 10.0 and lr < best_fa10:
                best_fa10 = lr
    out["cl_late_fa5_pct"] = round(best_fa5, 1) if best_fa5 < 100.0 else -1.0
    out["cl_late_fa10_pct"] = round(best_fa10, 1) if best_fa10 < 100.0 else -1.0
    return out


@torch.no_grad()
def eval_model_on_suite(model, npz_paths: list[Path], device: torch.device, dark_scale: float = 1.0) -> list[dict]:
    bev_cfg = TemporalBEVConfig()
    clips = []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as d:
            offs, blob = d["rgb_offsets"], d["rgb_blob"]
            orig_hw, K_orig = d["orig_hw"], d["K_orig"]
            poses_R, poses_t, causal_fwd, eval_rows = d["poses_R"], d["poses_t"], d["causal_fwd"], d["eval_rows"]
            T = len(eval_rows)
            rgbs, Ks, gs = [], [], []
            for i in range(T):
                rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=dark_scale)
                rgbs.append(((rgb_f - IM_MEAN) / IM_STD).transpose(2, 0, 1))
                Ks.append(K)
                g_cam = (poses_R[i].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
                g_cam /= np.linalg.norm(g_cam) + 1e-6
                gs.append(g_cam)

            d_1f_wz = np.full(T, np.inf, dtype=np.float32)
            obs_wz_list = []
            dep_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
            dh_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)
            nrm_s4 = np.empty((T, 3, H_IN // 4, W_IN // 4), dtype=np.float32)
            ls_s4 = np.empty((T, H_IN // 4, W_IN // 4), dtype=np.float32)

            for i0 in range(0, T, 32):
                i1 = min(T, i0 + 32)
                x = torch.from_numpy(np.stack(rgbs[i0:i1])).to(device)
                Kt = torch.from_numpy(np.stack(Ks[i0:i1])).to(device)
                gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device)
                rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, rays, None, None, return_features=False)
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
                    d_1f_wz[idx] = dw
                    obs_wz_list.append(ow)

            d_bev_v1 = temporal_bev_corridor_distances(poses_t, causal_fwd, poses_R, obs_wz_list)
            d_bev_v23, r_bev_v23, r_hyb_v23 = run_bev_on_clip(
                dep_s4, dh_s4, nrm_s4, ls_s4, np.stack(Ks), poses_R, poses_t, causal_fwd, bev_cfg
            )
            clips.append({
                "name": p.stem,
                "eval_rows": eval_rows,
                "d_1f_wz": d_1f_wz,
                "d_bev_v1": d_bev_v1,
                "d_bev_v23": d_bev_v23,
                "r_bev_v23": r_bev_v23,
                "r_hyb_v23": r_hyb_v23,
            })
    return clips


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", choices=["v2", "v3"], default="v3")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    if args.arch == "v3":
        model = EgoGeomOmniV3Net(EgoGeomV3Config(**ck["cfg"]), include_teacher_heads=False).to(device).eval()
    else:
        model = EgoGeomOmniNet(EgoGeomConfig(**ck["cfg"]), include_teacher_heads=False).to(device).eval()
    model.load_state_dict(ck["ema"], strict=False)
    print(f"Loaded {args.arch} ckpt {Path(args.ckpt).name} on {torch.cuda.get_device_name(0)}", flush=True)

    zs = REPO / "artifacts.local/work/egogeom-omni-v0/zeroshot_eval"
    suites = [
        ("aria_gen2", sorted((zs / "aria_g2_20clips").rglob("*.npz")), 1.0, 4.0),
        ("coda_campus", sorted((zs / "coda_zeroshot_shard000").rglob("*.npz")), 1.0, 4.0),
        ("arkit_val30_day", sorted((zs / "arkit_val30_shard00").rglob("*.npz")), 1.0, 2.0),
        ("arkit_val30_dark018", sorted((zs / "arkit_val30_shard00").rglob("*.npz")), 0.18, 2.0),
    ]
    summary = {"ckpt": str(args.ckpt), "arch": args.arch, "suites": {}}
    for sname, npzs, dscale, h_thr in suites:
        ts = time.time()
        clips = eval_model_on_suite(model, npzs, device, dark_scale=dscale)
        res = {
            "1f_world_z": score_dataset(clips, "d_1f_wz", None, h_thr),
            "stage2_bev_v1_proto": score_dataset(clips, "d_bev_v1", None, h_thr),
            "stage2_bev_v23_hard": score_dataset(clips, "d_bev_v23", None, h_thr),
            "stage2_bev_v23_hyb": score_dataset(clips, "d_bev_v23", "r_hyb_v23", h_thr),
        }
        summary["suites"][sname] = res
        print(f"\n=== {sname} ({time.time()-ts:.1f}s) ===", flush=True)
        for k, v in res.items():
            print(f"  {k:22s}: AUROC={v.get('auroc',-1):.4f} AP={v.get('ap',-1):.4f} R@5%={v.get('rec_fpr5',-1)*100:5.1f}% R@10%={v.get('rec_fpr10',-1)*100:5.1f}% MAE={v['dist_mae_m']:.3f}m Late@5={v['cl_late_fa5_pct']}%", flush=True)

    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out_json} (total {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
