"""Phase 3 G4 (96 GB Blackwell) Breakthrough Experiment:
EgoClear-v3 (Frozen 62M-Real-Image DAV2-S + Disparity-Expansion TTC + Dynamic-Suppressed Latent Memory)
vs SOTA Pure-Vision Metric Depth Baselines (UniDepth-V2-Small, UniDepth-V2-Large, UniDepth-V2-Large + Explicit 6m Mem, DAV2-S Ground)

Evaluated across 5 strict regimes (ALL EgoClear-v3 models trained ONLY on 627 synthetic sessions -> ZERO-SHOT on Real!):
  1. SYNTH_110_CLEAN        (110 frozen synthetic eval, clean poses)
  2. SYNTH_110_VIO_1X       (realistic ARCore/VIO drift: 2% trans, 0.8 deg/m yaw, 1.5 deg tilt)
  3. SYNTH_110_VIO_2X       (severe VIO drift stress test: 4% trans, 1.6 deg/m yaw, 3.0 deg tilt)
  4. REAL_40_LOCAL_ZEROSHOT (40 real ZED outdoor walking sessions, drift-free +/-0.8s local-window CREStereo GT)
  5. REAL_40_GLOBAL_ZEROSHOT (40 real ZED outdoor walking sessions, original global-fused GT)
Plus calibrated UNKNOWN selective prediction audit (unc_logits Risk-Coverage curve).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, "/content")
ud_root = Path("/content/unidepth_src")
if ud_root.exists():
    sub = next(ud_root.glob("UniDepth-*"), None)
    if sub is not None:
        sys.path.insert(0, str(sub))

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v3_model import EgoClearV3Net
from eval_closed_loop import H, eval_score_curve, make_episodes, score_episode, summarize_at_fa_caps
from pack_sanpo_colab import CELL, DYN, FWD0, HGT, MIN_PTS, MIN_VOX_V2, SKY, VOX, WALK, X_HALF, corridor_d
from train_and_eval_g4 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_PTS_S8,
    decode_rgb_batch,
    depth_map_to_corridor_d,
    ground_anchor_relative_depth,
    load_scene,
)
from train_and_eval_phase2_g4 import (
    SanpoPhase2ClipDataset,
    compute_loss_v2,
    perturb_vio_trajectory,
)

DRIVE_RUNS = Path("/content/drive/MyDrive/BlindAssist/runs/egoclear_v0_phase3")
LOCAL_RUNS = Path("/content/egoclear_runs_p3")


def build_local_window_real_eval_rows(scene_d: dict, half_win: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build drift-free +/-0.8s local-window static CREStereo GT for SANPO-Real (eliminates 25s vertical ZED drift)."""
    n = len(scene_d["eval_rows"])
    K_o = scene_d["K_orig"].astype(np.float64)
    Hh, Ww = float(scene_d["orig_hw"][0]), float(scene_d["orig_hw"][1])
    s8_h, s8_w = scene_d["depth_s8"].shape[1:]
    fx = K_o[0] * s8_w / Ww
    fy = K_o[1] * s8_h / Hh
    cx = K_o[2] * s8_w / Ww
    cy = K_o[3] * s8_h / Hh
    vv, uu = np.mgrid[0:s8_h, 0:s8_w]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)

    Rs = scene_d["poses_R"].astype(np.float64)
    ps = scene_d["poses_t"].astype(np.float64)
    fwds = scene_d["causal_fwd"].astype(np.float64)
    vh = scene_d["valid_heading"] > 0
    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]

    P_all, L_all = [], []
    for i in range(n):
        df = scene_d["depth_s8"][i].astype(np.float32).reshape(-1)
        sf = scene_d["seg_s8"][i].reshape(-1)
        ok = np.isfinite(df) & (df > 0.2) & (df < 25.0) & (sf != SKY)
        P_all.append(((Rs[i] @ (dirs[ok] * df[ok, None]).T).T + ps[i]).astype(np.float32))
        L_all.append(sf[ok])

    local_rows = np.full((n, 4), np.nan, dtype=np.float64)
    local_single_d = np.full(n, np.inf, dtype=np.float32)

    for i in range(n):
        if not vh[i]:
            continue
        vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2]
        if np.linalg.norm(vt) < 0.2:
            continue
        tf = np.array([vt[0], vt[1], 0.0]) / np.linalg.norm(vt)
        tr_ = np.array([tf[1], -tf[0], 0.0])

        lo, hi = max(0, i - half_win), min(n, i + half_win + 1)
        Pw_win = np.concatenate(P_all[lo:hi])
        Lw_win = np.concatenate(L_all[lo:hi])
        Gw = Pw_win[np.isin(Lw_win, WALK)]
        if len(Gw) < 100:
            continue
        # Local ground plane height z_g(x, y) via 0.25m cells within +/-0.8s window (drift < 2cm!)
        keys, inv = np.unique(np.floor(Gw[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv, kind="mergesort")
        z_s = Gw[order, 2]
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        gz = np.array([np.median(g) for g in np.split(z_s, cuts)], dtype=np.float64)
        gtree = cKDTree((keys + 0.5) * CELL)
        cellset = {tuple(k) for k in keys}

        def height_loc(Q: np.ndarray) -> np.ndarray:
            dist, idx_g = gtree.query(Q[:, :2], distance_upper_bound=1.0)
            h = np.full(len(Q), np.nan)
            okg = np.isfinite(dist)
            h[okg] = Q[okg, 2] - gz[idx_g[okg]]
            return h

        stat = Pw_win[~np.isin(Lw_win, WALK + [SKY] + DYN)]
        hs = height_loc(stat)
        stat = stat[(hs >= HGT[0]) & (hs <= HGT[1])]
        if len(stat) > 0:
            vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0)
            vc = (vox + 0.5) * VOX
            for kc in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
                cellset.add(tuple(kc))
            rel_v = vc - ps[i]
            x_t, z_t = rel_v @ tr_, rel_v @ tf
            inc = (np.abs(x_t) <= X_HALF) & (z_t >= FWD0) & (z_t <= H)
            D = corridor_d(z_t[inc], MIN_VOX_V2)
        else:
            D = float(np.inf)

        reach = min(D, H)
        samp = ps[i][:2] + np.outer(np.arange(FWD0, reach + 1e-6, CELL), tf[:2])
        cover = float(np.mean([tuple(np.floor(s_ / CELL).astype(np.int64)) in cellset for s_ in samp])) if len(samp) else 1.0

        P0, L0 = P_all[i], L_all[i]
        h0 = height_loc(P0)
        rel0 = P0 - ps[i]
        x0, z0 = rel0 @ tr_, rel0 @ tf
        dyn_flag = float(
            ((np.abs(x0) <= X_HALF) & (z0 >= FWD0) & (z0 <= H) & np.isin(L0, DYN) & (h0 >= HGT[0]) & (h0 <= HGT[1])).sum()
            >= MIN_PTS
        )
        local_rows[i] = [D, cover, dyn_flag, seg_len[i]]

        # Local single-frame CREStereo reference distance
        fwd = fwds[i]
        right = np.array([fwd[1], -fwd[0], 0.0])
        xq, zq = rel0 @ right, rel0 @ fwd
        strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
        g_est = np.percentile(P0[strip, 2], 5) if strip.sum() >= 50 else ps[i][2] - 1.5
        hq = P0[:, 2] - g_est
        cm = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT[0]) & (hq <= HGT[1])
        local_single_d[i] = corridor_d(zq[cm], MIN_PTS)

    return local_rows, local_single_d


@torch.inference_mode()
def run_unidepth_baselines(
    datasets_map: dict[str, tuple[dict[str, dict], dict[str, np.ndarray]]],
    bw_speeds: dict,
    device: torch.device,
) -> dict:
    """Run UniDepth-V2-Small & UniDepth-V2-Large (+ 6m explicit memory) on Synth-110 and Real-40."""
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    b1_thrs = np.round(np.arange(0.5, 5.01, 0.25), 2)
    eval_dh, eval_dw = 156, 276
    out = {}

    for u_name, hf_id in (
        ("unidepth_v2_small", "lpiccinelli/unidepth-v2-vits14"),
        ("unidepth_v2_large", "lpiccinelli/unidepth-v2-vitl14"),
    ):
        t0 = time.time()
        print(f"[unidepth] loading {u_name} ({hf_id})...", flush=True)
        u_mod = UniDepthV2.from_pretrained(hf_id).to(device).eval()

        # Cache per-scene predictions once so Real-Local and Real-Global reuse identical depth maps!
        cached_single = {}
        cached_mem = {}
        for dkey, (scene_dicts, eval_rows_map) in datasets_map.items():
            pred_d = {}
            pred_mem = {} if u_name == "unidepth_v2_large" else None
            tot_ms, tot_fr = 0.0, 0
            for sc, d in scene_dicts.items():
                n = len(d["eval_rows"])
                if n < 5:
                    continue
                if sc in cached_single:
                    pred_d[sc] = cached_single[sc]
                    if pred_mem is not None:
                        pred_mem[sc] = cached_mem[sc]
                    continue

                rgbs = decode_rgb_batch(d, np.arange(n))  # (n, 3, 294, 518) in [0,1]
                K_o = d["K_orig"].astype(np.float64)
                Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
                fx_r = K_o[0] * RGB_W / Ww
                fy_r = K_o[1] * RGB_H / Hh
                cx_r = K_o[2] * RGB_W / Ww
                cy_r = K_o[3] * RGB_H / Hh
                K_mat = torch.tensor(
                    [[fx_r, 0.0, cx_r], [0.0, fy_r, cy_r], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
                )

                depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
                for st in range(0, n, 32):
                    en = min(n, st + 32)
                    imgs_u8 = torch.from_numpy((rgbs[st:en] * 255.0).clip(0, 255).astype(np.uint8)).to(device)
                    cam_b = Pinhole(K=K_mat.unsqueeze(0).clone())
                    torch.cuda.synchronize()
                    ts = time.perf_counter()
                    pred = u_mod.infer(imgs_u8, cam_b)["depth"]  # (b, 1, 294, 518)
                    torch.cuda.synchronize()
                    tot_ms += (time.perf_counter() - ts) * 1000.0
                    tot_fr += (en - st)
                    pred_rs = F.interpolate(pred.float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1)
                    depths[st:en] = pred_rs.cpu().numpy()

                K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
                Rs = d["poses_R"].astype(np.float64)
                ps = d["poses_t"].astype(np.float64)
                fwds = d["causal_fwd"].astype(np.float64)
                d_arr = np.full(n, np.inf, dtype=np.float32)
                pts_seq = [] if pred_mem is not None else None
                for i in range(n):
                    dh, pw = depth_map_to_corridor_d(depths[i], K_d, Rs[i], ps[i], fwds[i], MIN_PTS_S8)
                    d_arr[i] = dh
                    if pts_seq is not None:
                        pts_seq.append(pw[::4])
                cached_single[sc] = d_arr
                pred_d[sc] = d_arr

                if pred_mem is not None:
                    seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]
                    cum = np.cumsum(seg_len)
                    d_mem = np.full(n, np.inf, dtype=np.float32)
                    for i in range(n):
                        if not d["valid_heading"][i]:
                            continue
                        win = [j for j in range(i + 1) if cum[i] - cum[j] <= 6.0]
                        Q = np.concatenate([pts_seq[j] for j in win]) if win else np.zeros((0, 3), dtype=np.float32)
                        if len(Q) == 0:
                            continue
                        fwd = fwds[i]
                        right = np.array([fwd[1], -fwd[0], 0.0])
                        rq = Q - ps[i]
                        xq, zq = rq @ right, rq @ fwd
                        strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
                        g = np.percentile(Q[strip, 2], 5) if strip.sum() >= 30 else ps[i][2] - 1.5
                        hq = Q[:, 2] - g
                        cm = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT[0]) & (hq <= HGT[1])
                        d_mem[i] = corridor_d(zq[cm], max(5, MIN_PTS_S8 // 4))
                    cached_mem[sc] = d_mem
                    pred_mem[sc] = d_mem

            curve = eval_score_curve(eval_rows_map, pred_d, bw_speeds, b1_thrs, higher_is_alert=False)
            out.setdefault(dkey, {})[u_name] = {
                "curve": curve,
                "summary": summarize_at_fa_caps(curve),
                "ms_per_frame": round(tot_ms / max(tot_fr, 1), 2) if tot_fr else None,
            }
            bf = out[dkey][u_name]["summary"]["best_floor"]
            m80 = out[dkey][u_name]["summary"].get("late_at_fa_fr<=80.0")
            m120 = out[dkey][u_name]["summary"].get("late_at_fa_fr<=120.3")
            m80_s = f"{m80['late_rate']*100:.1f}%" if m80 else "N/A"
            m120_s = f"{m120['late_rate']*100:.1f}%" if m120 else "N/A"
            print(
                f"[{dkey}] {u_name}: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f}) | "
                f"@fr<=80: {m80_s} | @fr<=120: {m120_s}",
                flush=True,
            )
            if pred_mem is not None:
                cm = eval_score_curve(eval_rows_map, pred_mem, bw_speeds, b1_thrs, higher_is_alert=False)
                out[dkey][u_name + "_explicit_mem6m"] = {"curve": cm, "summary": summarize_at_fa_caps(cm)}
                bfm = out[dkey][u_name + "_explicit_mem6m"]["summary"]["best_floor"]
                print(
                    f"[{dkey}] {u_name}_explicit_mem6m: floor={bfm['late']}/{bfm['avoidable']}({bfm['late_rate']*100:.1f}%, "
                    f"ep={bfm['fa_ep_per_km']:.1f}, fr={bfm['fa_fr_per_km']:.1f})",
                    flush=True,
                )

        del u_mod
        torch.cuda.empty_cache()
        print(f"[unidepth] finished {u_name} in {time.time()-t0:.1f}s", flush=True)
    return out


@torch.inference_mode()
def eval_v3_on_regime(
    model: EgoClearV3Net,
    scene_dicts: dict[str, dict],
    eval_rows_override: dict[str, np.ndarray] | None,
    bw_speeds: dict,
    device: torch.device,
    vio_scale: float = 0.0,  # 0.0 = clean, 1.0 = realistic ARCore, 2.0 = severe drift
    compute_uncertainty_audit: bool = False,
) -> dict:
    model.eval()
    eval_rows_map = {}
    corr_probs_map = {}
    unc_probs_map = {}
    stat_ap_num = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
    dyn_fp_frames, dyn_tot_frames = 0, 0
    tot_ms, tot_fr = 0.0, 0

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))

    for sc, d in scene_dicts.items():
        er = eval_rows_override[sc] if eval_rows_override is not None else d["eval_rows"]
        n = len(er)
        if n < 5:
            continue
        rgb = decode_rgb_batch(d, np.arange(n))
        rgb = (rgb - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)

        R_seq = d["poses_R"].astype(np.float32)
        t_seq = d["poses_t"].astype(np.float32)
        fwd_seq = d["causal_fwd"].astype(np.float32)
        seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

        if vio_scale > 0.0:
            seed = int(hashlib.sha256((f"vio{vio_scale}:" + sc).encode()).hexdigest()[:8], 16)
            R_seq, t_seq, fwd_seq, seg_seq = perturb_vio_trajectory(
                R_seq, t_seq, fwd_seq, np.random.default_rng(seed),
                trans_rel_std=0.02 * vio_scale,
                yaw_deg_per_m=0.8 * vio_scale,
                grav_tilt_deg=1.5 * vio_scale,
            )

        batch = {
            "rgb": torch.from_numpy(rgb)[None].to(device),
            "K_scaled": torch.from_numpy(np.repeat(K_s[None], n, axis=0))[None].to(device),
            "R_c2w": torch.from_numpy(R_seq)[None].to(device),
            "t_w": torch.from_numpy(t_seq)[None].to(device),
            "fwd_w": torch.from_numpy(fwd_seq)[None].to(device),
            "seg_len": torch.from_numpy(seg_seq)[None].to(device),
        }

        torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_sequence(**batch)
        torch.cuda.synchronize()
        tot_ms += (time.perf_counter() - ts) * 1000.0
        tot_fr += n

        p_corr = torch.sigmoid(out["corr_logits"][0].float()).cpu().numpy()
        p_unc = torch.sigmoid(out["unc_logits"][0].float()).cpu().numpy()
        p_stat = torch.sigmoid(out["static_logits"][0].float()).cpu().numpy()
        eval_rows_map[sc] = er
        corr_probs_map[sc] = p_corr
        unc_probs_map[sc] = p_unc

        gt_st = d["gt_bev_static"][:, :, :, 6:9] > 0
        pr_st = p_stat[:, :, :, 6:9] >= 0.5
        for b in range(3):
            stat_ap_num[b][0] += int((pr_st[:, b] & gt_st[:, b]).sum())
            stat_ap_num[b][1] += int((pr_st[:, b] & ~gt_st[:, b]).sum())
            stat_ap_num[b][2] += int((~pr_st[:, b] & gt_st[:, b]).sum())

        D_gt = er[:, 0]
        dyn_only = (d["valid_heading"] > 0) & np.isinf(D_gt) & (er[:, 2] > 0.5)
        if np.any(dyn_only):
            dyn_tot_frames += int(dyn_only.sum())
            dyn_fp_frames += int((p_corr[dyn_only, : idx_4m + 1].max(axis=1) >= 0.5).sum())

    fixed_scores = {sc: probs[:, : idx_4m + 1].max(axis=1) for sc, probs in corr_probs_map.items()}
    thrs = np.round(np.concatenate([np.arange(0.02, 0.20, 0.02), np.arange(0.20, 0.96, 0.04)]), 3)
    curve_fixed = eval_score_curve(eval_rows_map, fixed_scores, bw_speeds, thrs, higher_is_alert=True)

    def walker_fn(ep: dict, thr: float) -> np.ndarray:
        probs = corr_probs_map[ep["scene"]]
        k_bin = int(np.clip(np.searchsorted(z_edges, min(ep["dreq"] + 1.0, 4.9)), 0, BEV_NZ - 1))
        return probs[:, : k_bin + 1].max(axis=1) >= thr

    curve_walker = eval_score_curve(eval_rows_map, {}, bw_speeds, thrs, walker_aware_fn=walker_fn)

    hband_f1 = {}
    for b, name in ((0, "LOW"), (1, "BODY"), (2, "HEAD")):
        tp, fp, fn = stat_ap_num[b]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        hband_f1[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "pos_cells": tp + fn}

    unc_audit = None
    if compute_uncertainty_audit:
        # Selective prediction / UNKNOWN risk-coverage curve on frame-level hazard classification (D <= 4.0m)
        all_p, all_y, all_u = [], [], []
        for sc, er in eval_rows_map.items():
            valid = (np.isfinite(er[:, 0]) | np.isinf(er[:, 0])) & np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)
            if not np.any(valid):
                continue
            p_f = corr_probs_map[sc][valid, : idx_4m + 1].max(axis=1)
            u_f = unc_probs_map[sc][valid, : idx_4m + 1].mean(axis=1)
            y_f = (er[valid, 0] <= H).astype(np.int32)
            all_p.append(p_f)
            all_u.append(u_f)
            all_y.append(y_f)
        ap = np.concatenate(all_p)
        au = np.concatenate(all_u)
        ay = np.concatenate(all_y)
        order = np.argsort(au)  # lowest uncertainty first
        unc_audit = {}
        for cov in (1.0, 0.95, 0.90, 0.80, 0.70):
            k = max(1, int(len(order) * cov))
            sel = order[:k]
            pred_b = (ap[sel] >= 0.32).astype(np.int32)
            err = float(np.mean(pred_b != ay[sel]))
            fp_r = float(np.mean(pred_b[ay[sel] == 0] == 1)) if np.any(ay[sel] == 0) else 0.0
            fn_r = float(np.mean(pred_b[ay[sel] == 1] == 0)) if np.any(ay[sel] == 1) else 0.0
            unc_audit[f"cov_{int(cov*100)}"] = {"n_frames": k, "err": round(err, 4), "fpr": round(fp_r, 4), "fnr": round(fn_r, 4)}

    return {
        "ms_per_frame_stream": round(tot_ms / max(tot_fr, 1), 2),
        "hband_corridor_f1_at_0.5": hband_f1,
        "dyn_only_alert_rate_at_0.5": {
            "alerts": dyn_fp_frames,
            "frames": dyn_tot_frames,
            "rate": round(dyn_fp_frames / max(dyn_tot_frames, 1), 4),
        },
        "fixed_4m": {"curve": curve_fixed, "summary": summarize_at_fa_caps(curve_fixed)},
        "walker_aware": {"curve": curve_walker, "summary": summarize_at_fa_caps(curve_walker)},
        "uncertainty_audit": unc_audit,
    }


def train_and_eval_v3(
    mode: str,
    train_dicts: list[dict],
    eval_synth_dicts: dict[str, dict],
    eval_real_dicts: dict[str, dict],
    real_local_rows: dict[str, np.ndarray],
    bw_speeds: dict,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 48,
    clip_len: int = 12,
    clips_per_epoch: int = 1920,
) -> dict:
    t0 = time.time()
    torch.manual_seed(20261120)
    np.random.seed(20261120)

    model = EgoClearV3Net(mode=mode, c_bev=128, freeze_dav2=True).to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n=== Training EgoClear-v3-{mode} (FROZEN DAV2-S, params={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M, bs={batch_size}x{clip_len}={batch_size*clip_len} fr/step) ===",
        flush=True,
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-4, weight_decay=1e-4)
    ds = SanpoPhase2ClipDataset(train_dicts, clip_len=clip_len, clips_per_epoch=clips_per_epoch)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    total_steps = epochs * len(loader)
    warmup = max(20, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    torch.cuda.reset_peak_memory_stats()

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0}
        t_ep = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_sequence(
                    batch["rgb"], batch["K_scaled"], batch["R_c2w"], batch["t_w"], batch["fwd_w"], batch["seg_len"]
                )
                loss, parts = compute_loss_v2(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_loss:
                ep_loss[k] += parts[k]
        for k in ep_loss:
            ep_loss[k] /= len(loader)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_loss.items()}, "sec": round(time.time() - t_ep, 1)})
        if ep in (1, 5, 10, 15, epochs):
            print(
                f"  [EgoClear-v3-{mode} ep {ep:02d}/{epochs}] tot={ep_loss['tot']:.4f} "
                f"(st={ep_loss['st']:.4f} dy={ep_loss['dy']:.4f} co={ep_loss['co']:.4f}) "
                f"{time.time()-t_ep:.1f}s | peak_vram={peak_gb:.1f}GB",
                flush=True,
            )

    ckpt_local = LOCAL_RUNS / f"egoclear_v3_{mode}.pt"
    # Save only trainable weights (~16 MB!) + metadata for ultra-fast saving/deployment
    tr_state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("dav2.")}
    torch.save({"mode": mode, "c_bev": 128, "head_state_dict": tr_state, "history": history}, ckpt_local)
    shutil.copy2(ckpt_local, DRIVE_RUNS / ckpt_local.name)

    regimes = {
        "synth_clean": eval_v3_on_regime(model, eval_synth_dicts, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=(mode == "full")),
        "synth_vio_1x": eval_v3_on_regime(model, eval_synth_dicts, None, bw_speeds, device, vio_scale=1.0),
        "synth_vio_2x": eval_v3_on_regime(model, eval_synth_dicts, None, bw_speeds, device, vio_scale=2.0),
        "real_local_zeroshot": eval_v3_on_regime(model, eval_real_dicts, real_local_rows, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=(mode == "full")),
        "real_global_zeroshot": eval_v3_on_regime(model, eval_real_dicts, None, bw_speeds, device, vio_scale=0.0),
    }

    for rname, ev in regimes.items():
        s = ev["fixed_4m"]["summary"]
        m80 = s.get("late_at_fa_fr<=80.0")
        m120 = s.get("late_at_fa_fr<=120.3")
        m200 = s.get("late_at_fa_fr<=200.0")
        s80 = f"{m80['late']}/{m80['avoidable']}({m80['late_rate']*100:.1f}%,ep={m80['fa_ep_per_km']:.1f},fr={m80['fa_fr_per_km']:.1f})" if m80 else "N/A"
        s120 = f"{m120['late']}/{m120['avoidable']}({m120['late_rate']*100:.1f}%,ep={m120['fa_ep_per_km']:.1f},fr={m120['fa_fr_per_km']:.1f})" if m120 else "N/A"
        s200 = f"{m200['late']}/{m200['avoidable']}({m200['late_rate']*100:.1f}%,ep={m200['fa_ep_per_km']:.1f},fr={m200['fa_fr_per_km']:.1f})" if m200 else "N/A"
        print(
            f"[P3 EgoClear-v3-{mode} | {rname}] @fr<=80: {s80} | @fr<=120: {s120} | @fr<=200: {s200} | "
            f"dyn@0.5={ev['dyn_only_alert_rate_at_0.5']['rate']*100:.1f}% | HEAD_F1={ev['hband_corridor_f1_at_0.5']['HEAD']['f1']:.3f}",
            flush=True,
        )

    del model, opt
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "params_M": round(n_tot / 1e6, 2),
        "trainable_M": round(n_tr / 1e6, 2),
        "train_sec": round(time.time() - t0, 1),
        "history": history,
        "evals": regimes,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[p3] GPU={torch.cuda.get_device_name(0)}", flush=True)
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    train_dicts = []
    eval_synth_dicts = {}
    for sc, sp in m287["splits"].items():
        p = Path("/content/sanpo_287/scenes") / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            if sp == "train":
                train_dicts.append(d)
            else:
                eval_synth_dicts[sc] = d
    for p in sorted(Path("/content/sanpo_p2/extra_scenes").glob("*.npz")):
        train_dicts.append(load_scene(p))

    eval_real_dicts = {}
    for p in sorted(Path("/content/sanpo_p2/real_scenes").glob("*.npz")):
        eval_real_dicts[p.stem] = load_scene(p)

    # Compute (or load cached) drift-free local-window CREStereo GT for SANPO-Real 40 in parallel!
    cache_rloc = Path("/content/sanpo_p2/real_local_gt_cache.npz")
    if cache_rloc.exists():
        z = np.load(cache_rloc, allow_pickle=False)
        real_local_rows = {sc: z[f"r_{sc}"] for sc in eval_real_dicts}
        real_local_single_d = {sc: z[f"d_{sc}"] for sc in eval_real_dicts}
        print(f"[p3] loaded cached REAL_40_LOCAL ({len(real_local_rows)} scenes)", flush=True)
    else:
        print(f"[p3] building drift-free +/-0.8s local-window GT for {len(eval_real_dicts)} SANPO-Real sessions (parallel)...", flush=True)
        t_loc = time.time()
        from concurrent.futures import ProcessPoolExecutor
        items_r = list(eval_real_dicts.items())
        with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 8)) as ex:
            res_r = list(ex.map(build_local_window_real_eval_rows, [v for _, v in items_r]))
        real_local_rows = {sc: lr for (sc, _), (lr, _) in zip(items_r, res_r)}
        real_local_single_d = {sc: lsd for (sc, _), (_, lsd) in zip(items_r, res_r)}
        save_d = {}
        for sc in real_local_rows:
            save_d[f"r_{sc}"] = real_local_rows[sc]
            save_d[f"d_{sc}"] = real_local_single_d[sc]
        np.savez(cache_rloc, **save_d)
        print(f"[p3] REAL_40_LOCAL built & cached in {time.time()-t_loc:.1f}s", flush=True)
    b1_thrs = np.round(np.arange(0.5, 5.01, 0.25), 2)
    c_rloc = eval_score_curve(real_local_rows, real_local_single_d, bw_speeds, b1_thrs, higher_is_alert=False)
    s_rloc = summarize_at_fa_caps(c_rloc)["best_floor"]
    print(
        f"[p3] REAL_40_LOCAL CREStereo ref_single: "
        f"floor={s_rloc['late']}/{s_rloc['avoidable']}({s_rloc['late_rate']*100:.1f}%, ep={s_rloc['fa_ep_per_km']:.1f}, fr={s_rloc['fa_fr_per_km']:.1f})",
        flush=True,
    )

    # 1. Run UniDepth-V2-Small & UniDepth-V2-Large (+ explicit 6m memory) baselines on Synth-110 & Real-40
    datasets_map = {
        "synth_110": (eval_synth_dicts, {sc: d["eval_rows"] for sc, d in eval_synth_dicts.items() if len(d["eval_rows"]) >= 5}),
        "real_40_local": (eval_real_dicts, real_local_rows),
        "real_40_global": (eval_real_dicts, {sc: d["eval_rows"] for sc, d in eval_real_dicts.items() if len(d["eval_rows"]) >= 5}),
    }
    unidepth_res = run_unidepth_baselines(datasets_map, bw_speeds, device)

    # Also evaluate DAV2-S Ground Anchor on real_40_local for complete comparison
    proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    dav2_mod = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
    eval_dh, eval_dw = 156, 276
    dav2_real_d = {}
    for sc, d in eval_real_dicts.items():
        n = len(d["eval_rows"])
        rgbs = decode_rgb_batch(d, np.arange(n))
        depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for st in range(0, n, 32):
                en = min(n, st + 32)
                imgs = [(rgbs[k].transpose(1, 2, 0) * 255.0).astype(np.uint8) for k in range(st, en)]
                inp = proc(images=imgs, return_tensors="pt").to(device)
                pred = dav2_mod(**inp).predicted_depth
                depths[st:en] = F.interpolate(pred.unsqueeze(1).float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        K_o = d["K_orig"].astype(np.float64)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
        Rs, ps, fwds = d["poses_R"].astype(np.float64), d["poses_t"].astype(np.float64), d["causal_fwd"].astype(np.float64)
        nom_h = 1.55 if str(d["cam"]) == "camera_head" else 1.20
        d_arr = np.full(n, np.inf, dtype=np.float32)
        for i in range(n):
            z_map = ground_anchor_relative_depth(depths[i], K_d, Rs[i], nom_h)
            d_arr[i], _ = depth_map_to_corridor_d(z_map, K_d, Rs[i], ps[i], fwds[i], MIN_PTS_S8)
        dav2_real_d[sc] = d_arr
    c_dav_rloc = eval_score_curve(real_local_rows, dav2_real_d, bw_speeds, b1_thrs, higher_is_alert=False)
    unidepth_res["real_40_local"]["dav2_small_ground"] = {"curve": c_dav_rloc, "summary": summarize_at_fa_caps(c_dav_rloc)}
    unidepth_res["real_40_local"]["ref_crestereo_single"] = {"curve": c_rloc, "summary": summarize_at_fa_caps(c_rloc)}
    bf_dav = unidepth_res["real_40_local"]["dav2_small_ground"]["summary"]["best_floor"]
    print(f"[real_40_local] dav2_small_ground: floor={bf_dav['late']}/{bf_dav['avoidable']}({bf_dav['late_rate']*100:.1f}%, ep={bf_dav['fa_ep_per_km']:.1f}, fr={bf_dav['fa_fr_per_km']:.1f})", flush=True)
    del dav2_mod, proc
    torch.cuda.empty_cache()

    # 2. Train & evaluate EgoClear-v3 (Single -> MVS -> Full) strictly on 627 Synthetic -> ZERO-SHOT on Real!
    v3_res = {}
    for mode in ("single", "mvs", "full"):
        v3_res[mode] = train_and_eval_v3(
            mode, train_dicts, eval_synth_dicts, eval_real_dicts, real_local_rows, bw_speeds, device,
            epochs=20, batch_size=48, clip_len=12, clips_per_epoch=1920,
        )

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_sec": round(time.time() - t_all, 1),
        "counts": {"train_synth": len(train_dicts), "eval_synth": len(eval_synth_dicts), "eval_real": len(eval_real_dicts)},
        "baselines": unidepth_res,
        "egoclear_v3": v3_res,
    }
    res_local = LOCAL_RUNS / "phase3_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase3_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[p3] DONE saved phase3_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
