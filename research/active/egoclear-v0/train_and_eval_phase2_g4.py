"""Phase 2 G4 (96 GB Blackwell) full-scale experiment:
1. Saturate RTX PRO 6000 96 GB:
   - 627 SANPO-Synthetic train sessions (~36,000 frames) vs 110 strictly frozen synthetic eval sessions
   - batch_size=32 x clip_len=12 (384 frames/step!), bf16 + TF32 + 12 DataLoader workers
   - EgoClear-v2 fixes:
       (a) Explicit negative supervision (y_corr=0, 2.5x weight) on dynamic-only clear frames
       (b) 2.5x HEAD height-band weight in static occupancy loss
       (c) Strong Sim-to-Real RGB + random focal-length/zoom crop augmentation (0.72x..1.15x, exact K update!)
       (d) Training-time realistic VIO drift injection (50% clips: 2.5% trans, 0.6 deg/step yaw, 2.0 deg tilt)
2. Four Rigorous Evaluations:
   - Test 1: SANPO-Synthetic 110 frozen eval (Clean poses) vs GT-Single, DAV2-S Ground, Focal-Calibrated DAV2-Metric-Large, UniDepth-V2-S/L (if loadable)
   - Test 2: SANPO-Synthetic 110 frozen eval under REALISTIC NOISY VIO DRIFT (2% trans + 0.8 deg/m yaw + 1.5 deg gravity tilt)
   - Test 3: ZERO-SHOT REAL-WORLD OUTDOOR TRANSFER on SANPO-Real (40 real ZED outdoor walking sessions, ZERO real fine-tuning!)
   - Test 4: Dynamic-pedestrian rejection & HEAD/BODY/LOW corridor F1 audit
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, "/content")
from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W, EgoClearNet
from eval_closed_loop import H, eval_score_curve, summarize_at_fa_caps
from train_and_eval_g4 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_PTS_S8,
    decode_rgb_batch,
    depth_map_to_corridor_d,
    ground_anchor_relative_depth,
    load_scene,
)

DRIVE_RUNS = Path("/content/drive/MyDrive/BlindAssist/runs/egoclear_v0_phase2")
LOCAL_RUNS = Path("/content/egoclear_runs_p2")


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rot_xy(rx: float, ry: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    return Ry @ Rx


def perturb_vio_trajectory(
    R_seq: np.ndarray,   # (T, 3, 3)
    t_seq: np.ndarray,   # (T, 3)
    fwd_seq: np.ndarray, # (T, 3)
    rng: np.random.Generator,
    trans_rel_std: float = 0.02,
    yaw_deg_per_m: float = 0.8,
    grav_tilt_deg: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply causal realistic ARCore/VIO drift + gravity tilt noise to a trajectory."""
    T = len(t_seq)
    ds = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)
    # Random walk translation drift proportional to traveled distance + small per-frame jitter
    dt_noise = rng.normal(0.0, 1.0, (T, 2)).astype(np.float32) * (trans_rel_std * np.maximum(ds, 0.05))[:, None]
    dxy_cum = np.cumsum(dt_noise, axis=0)
    t_noisy = t_seq.copy()
    t_noisy[:, :2] += dxy_cum

    # Random walk yaw drift + constant small gravity tilt bias + per-frame tilt jitter
    dyaw = np.cumsum(rng.normal(0.0, 1.0, T).astype(np.float32) * np.deg2rad(yaw_deg_per_m) * np.maximum(ds, 0.05))
    bias_rx = float(np.deg2rad(rng.normal(0.0, grav_tilt_deg)))
    bias_ry = float(np.deg2rad(rng.normal(0.0, grav_tilt_deg)))

    R_noisy = np.empty_like(R_seq)
    fwd_noisy = np.empty_like(fwd_seq)
    for i in range(T):
        Rz = rot_z(float(dyaw[i]))
        Rtilt = rot_xy(
            bias_rx + float(np.deg2rad(rng.normal(0.0, 0.4))),
            bias_ry + float(np.deg2rad(rng.normal(0.0, 0.4))),
        )
        # Apply yaw drift in world + slight gravity tilt perturbation
        R_noisy[i] = Rtilt @ Rz @ R_seq[i]
        # Recompute causal heading from noisy trajectory
        a = max(0, i - 2)
        b = i if i >= 2 else min(T - 1, 2)
        v = (t_noisy[b] - t_noisy[a])[:2]
        if np.linalg.norm(v) >= 0.2:
            f = np.array([v[0], v[1], 0.0], dtype=np.float32) / float(np.linalg.norm(v))
        else:
            cz = R_noisy[i, :, 2].copy()
            cz[2] = 0.0
            n_cz = float(np.linalg.norm(cz))
            f = cz / n_cz if n_cz > 1e-6 else fwd_seq[i]
        fwd_noisy[i] = f
    seg_noisy = np.r_[0.0, np.linalg.norm(np.diff(t_noisy[:, :2], axis=0), axis=1)].astype(np.float32)
    return R_noisy, t_noisy, fwd_noisy, seg_noisy


def apply_focal_zoom_crop(
    rgb: np.ndarray,   # (T, 3, H, W) float32 in [0, 1]
    K_s: np.ndarray,   # (4,) [fx, fy, cx, cy] at (W, H)
    scale: float,      # <1.0 => zoom out (pad/wider effective FOV), >1.0 => zoom in (crop)
) -> tuple[np.ndarray, np.ndarray]:
    """Consistent geometric resize/crop across clip with exact pinhole intrinsics update."""
    T, _, H0, W0 = rgb.shape
    fx, fy, cx, cy = [float(x) for x in K_s]
    if abs(scale - 1.0) < 0.02:
        return rgb, K_s.copy()
    nw = max(32, int(round(W0 * scale)))
    nh = max(32, int(round(H0 * scale)))
    out = np.zeros_like(rgb)
    # Center crop or center pad after resizing
    for t in range(T):
        hwc = rgb[t].transpose(1, 2, 0)
        rs = cv2.resize(hwc, (nw, nh), interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA)
        if scale > 1.0:
            x0 = (nw - W0) // 2
            y0 = (nh - H0) // 2
            canvas = rs[y0 : y0 + H0, x0 : x0 + W0]
        else:
            canvas = np.zeros((H0, W0, 3), dtype=np.float32)
            x0 = (W0 - nw) // 2
            y0 = (H0 - nh) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = rs
        out[t] = canvas.transpose(2, 0, 1)
    # Exact updated intrinsics!
    sx = nw / float(W0)
    sy = nh / float(H0)
    if scale > 1.0:
        dx = -((nw - W0) // 2)
        dy = -((nh - H0) // 2)
    else:
        dx = (W0 - nw) // 2
        dy = (H0 - nh) // 2
    K_new = np.array([fx * sx, fy * sy, cx * sx + dx, cy * sy + dy], dtype=np.float32)
    return out, K_new


class SanpoPhase2ClipDataset(torch.utils.data.Dataset):
    """627-session SANPO-Synthetic train dataset with Sim-to-Real RGB, focal-zoom, and VIO drift augmentation."""

    def __init__(self, scene_dicts: list[dict], clip_len: int = 12, clips_per_epoch: int = 1600):
        self.scenes = [d for d in scene_dicts if len(d["eval_rows"]) >= clip_len + 1]
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        d = self.scenes[rng.integers(len(self.scenes))]
        n = len(d["eval_rows"])
        D_all = d["eval_rows"][:, 0]
        dyn_all = d["eval_rows"][:, 2]
        haz_idx = np.nonzero(np.isfinite(D_all) & (D_all <= H))[0]
        dyn_only_idx = np.nonzero(np.isinf(D_all) & (dyn_all > 0.5))[0]

        u_pick = rng.random()
        if len(haz_idx) > 0 and u_pick < 0.55:
            anchor = int(haz_idx[rng.integers(len(haz_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(dyn_only_idx) > 0 and u_pick < 0.75:
            # Explicitly oversample dynamic-pedestrian clear clips so the network masters dynamic rejection!
            anchor = int(dyn_only_idx[rng.integers(len(dyn_only_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        else:
            st = int(rng.integers(0, n - self.clip_len + 1))
        sl = np.arange(st, st + self.clip_len)

        rgb = decode_rgb_batch(d, sl)  # (T, 3, 294, 518) in [0, 1]
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)

        # 1. Focal-length / FOV augmentation (simulates ZED Mini 76.9 deg, phones, etc.)
        if rng.random() < 0.65:
            zoom = float(rng.uniform(0.72, 1.15))
            rgb, K_s = apply_focal_zoom_crop(rgb, K_s, zoom)

        # 2. Strong Sim-to-Real photometric + sensor noise augmentation
        gamma = float(rng.uniform(0.75, 1.35))
        rgb = np.power(np.clip(rgb, 1e-4, 1.0), gamma)
        # Per-channel white balance + brightness/contrast
        ch_gain = rng.uniform(0.85, 1.15, size=(1, 3, 1, 1)).astype(np.float32)
        c_fac = float(rng.uniform(0.80, 1.25))
        mean = rgb.mean(axis=(2, 3), keepdims=True)
        rgb = np.clip(((rgb - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
        if rng.random() < 0.35:
            # Slight Gaussian sensor noise
            rgb = np.clip(rgb + rng.normal(0.0, float(rng.uniform(0.005, 0.025)), rgb.shape).astype(np.float32), 0.0, 1.0)
        if rng.random() < 0.25:
            # Slight motion/defocus blur
            for t_i in range(self.clip_len):
                hwc = cv2.GaussianBlur(rgb[t_i].transpose(1, 2, 0), (3, 3), 0.8)
                rgb[t_i] = hwc.transpose(2, 0, 1)

        rgb = (rgb - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_seq = np.repeat(K_s[None], self.clip_len, axis=0)

        R_seq = d["poses_R"][sl].astype(np.float32)
        t_seq = d["poses_t"][sl].astype(np.float32)
        fwd_seq = d["causal_fwd"][sl].astype(np.float32)
        seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

        # 3. Training-time realistic VIO drift + gravity tilt injection (50% of clips!)
        if rng.random() < 0.50:
            R_seq, t_seq, fwd_seq, seg_seq = perturb_vio_trajectory(
                R_seq, t_seq, fwd_seq, rng, trans_rel_std=0.025, yaw_deg_per_m=0.8, grav_tilt_deg=2.0
            )

        gt_stat = d["gt_bev_static"][sl].astype(np.float32)
        gt_dyn = d["gt_bev_dynamic"][sl].astype(np.float32)
        vh = d["valid_heading"][sl].astype(np.float32)
        ev_r = d["eval_rows"][sl]
        D_gt = ev_r[:, 0]
        cover = ev_r[:, 1]
        dyn_fl = ev_r[:, 2]

        # FIX 1: DO NOT mask out dyn_only clear frames! Supervise them explicitly as y_corr = 0 with 2.5x weight!
        y_corr = (D_gt[:, None] <= self.z_edges[None, :]).astype(np.float32)
        valid_corr = ((vh > 0.5) & np.isfinite(cover) & (cover >= 0.6)).astype(np.float32)
        dyn_neg_w = np.where((np.isinf(D_gt)) & (dyn_fl > 0.5), 2.5, 1.0).astype(np.float32)

        return {
            "rgb": torch.from_numpy(rgb),
            "K_scaled": torch.from_numpy(K_seq),
            "R_c2w": torch.from_numpy(R_seq),
            "t_w": torch.from_numpy(t_seq),
            "fwd_w": torch.from_numpy(fwd_seq),
            "seg_len": torch.from_numpy(seg_seq),
            "gt_static": torch.from_numpy(gt_stat),
            "gt_dynamic": torch.from_numpy(gt_dyn),
            "y_corr": torch.from_numpy(y_corr),
            "valid_corr": torch.from_numpy(valid_corr),
            "dyn_neg_w": torch.from_numpy(dyn_neg_w),
        }


def compute_loss_v2(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    st_log = out["static_logits"]
    dy_log = out["dynamic_logits"]
    co_log = out["corr_logits"]
    un_log = out["unc_logits"]

    gt_st = batch["gt_static"]
    gt_dy = batch["gt_dynamic"]
    y_co = batch["y_corr"]
    v_co = batch["valid_corr"]
    dn_w = batch["dyn_neg_w"]

    col_w = torch.ones(BEV_NX, device=st_log.device, dtype=st_log.dtype)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    # FIX 2: Boost HEAD height band (index 2) 2.5x for overhanging cane-complementary obstacles
    hb_w = torch.tensor([1.0, 1.2, 2.5], device=st_log.device, dtype=st_log.dtype)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.0, device=st_log.device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()

    bce_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=dy_log.device), reduction="none")
    loss_dy = bce_dy.mean()

    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=co_log.device)
    z_w = torch.where(z_edges <= H + 0.2, 2.0, 0.5)[None, None, :]
    bce_co = F.binary_cross_entropy_with_logits(co_log, y_co, pos_weight=torch.tensor(3.0, device=co_log.device), reduction="none")
    pt_co = torch.exp(-F.binary_cross_entropy_with_logits(co_log, y_co, reduction="none"))
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_co = ((((1.0 - pt_co) ** 1.5) * bce_co) * m_co).sum() / m_co.sum().clamp(min=1.0)

    with torch.no_grad():
        err_tgt = ((torch.sigmoid(co_log) - y_co).abs() > 0.35).float()
    bce_un = F.binary_cross_entropy_with_logits(un_log, err_tgt, reduction="none")
    loss_un = (bce_un * m_co).sum() / m_co.sum().clamp(min=1.0)

    total = loss_st + 0.5 * loss_dy + 1.8 * loss_co + 0.2 * loss_un
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "dy": float(loss_dy.item()),
        "co": float(loss_co.item()),
    }


@torch.inference_mode()
def eval_egoclear(
    model: EgoClearNet,
    eval_scene_dicts: dict[str, dict],
    bw_speeds: dict,
    device: torch.device,
    noisy_vio: bool = False,
) -> dict:
    model.eval()
    eval_rows_map = {}
    corr_probs_map = {}
    stat_ap_num = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
    dyn_fp_frames, dyn_tot_frames = 0, 0
    tot_ms, tot_fr = 0.0, 0

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))

    for sc, d in eval_scene_dicts.items():
        n = len(d["eval_rows"])
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

        if noisy_vio:
            seed = int(hashlib.sha256(("vio_eval:" + sc).encode()).hexdigest()[:8], 16)
            R_seq, t_seq, fwd_seq, seg_seq = perturb_vio_trajectory(
                R_seq, t_seq, fwd_seq, np.random.default_rng(seed),
                trans_rel_std=0.02, yaw_deg_per_m=0.8, grav_tilt_deg=1.5
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
        p_stat = torch.sigmoid(out["static_logits"][0].float()).cpu().numpy()
        eval_rows_map[sc] = d["eval_rows"]
        corr_probs_map[sc] = p_corr

        gt_st = d["gt_bev_static"][:, :, :, 6:9] > 0
        pr_st = p_stat[:, :, :, 6:9] >= 0.5
        for b in range(3):
            stat_ap_num[b][0] += int((pr_st[:, b] & gt_st[:, b]).sum())
            stat_ap_num[b][1] += int((pr_st[:, b] & ~gt_st[:, b]).sum())
            stat_ap_num[b][2] += int((~pr_st[:, b] & gt_st[:, b]).sum())

        D_gt = d["eval_rows"][:, 0]
        dyn_only = (d["valid_heading"] > 0) & np.isinf(D_gt) & (d["eval_rows"][:, 2] > 0.5)
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
    }


def eval_pretrained_baselines_on_set(
    set_name: str,
    scene_dicts: dict[str, dict],
    bw_speeds: dict,
    device: torch.device,
) -> dict:
    """Run Reference-GT-Single + DAV2-S Ground + Focal-Calibrated DAV2-Metric-Outdoor-Large on any eval set."""
    eval_rows_map = {sc: d["eval_rows"] for sc, d in scene_dicts.items() if len(d["eval_rows"]) >= 5}
    gt_single_map = {sc: d["gt_single_d"] for sc, d in scene_dicts.items() if len(d["eval_rows"]) >= 5}
    b1_thrs = np.round(np.arange(0.5, 5.01, 0.25), 2)

    out = {}
    c_gt = eval_score_curve(eval_rows_map, gt_single_map, bw_speeds, b1_thrs, higher_is_alert=False)
    out["ref_depth_single"] = {"curve": c_gt, "summary": summarize_at_fa_caps(c_gt)}
    sg = out["ref_depth_single"]["summary"]["best_floor"]
    print(f"[{set_name}] ref_depth_single: floor={sg['late']}/{sg['avoidable']} ({sg['late_rate']*100:.1f}%), FA_ep={sg['fa_ep_per_km']:.1f}, FA_fr={sg['fa_fr_per_km']:.1f}", flush=True)

    eval_dh, eval_dw = 156, 276
    specs = [
        ("dav2_small_ground", "depth-anything/Depth-Anything-V2-Small-hf", "relative"),
        ("dav2_metric_outdoor_large_focalcal", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "metric_vkitti"),
    ]
    for bname, hf_id, kind in specs:
        proc = AutoImageProcessor.from_pretrained(hf_id)
        mod = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device).eval()
        pred_d = {}
        raw_medians = []
        for sc, d in scene_dicts.items():
            n = len(d["eval_rows"])
            if n < 5:
                continue
            rgbs = decode_rgb_batch(d, np.arange(n))
            depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for st in range(0, n, 32):
                    en = min(n, st + 32)
                    imgs = [(rgbs[k].transpose(1, 2, 0) * 255.0).astype(np.uint8) for k in range(st, en)]
                    inp = proc(images=imgs, return_tensors="pt").to(device)
                    pred = mod(**inp).predicted_depth
                    out_rs = F.interpolate(pred.unsqueeze(1).float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1)
                    depths[st:en] = out_rs.cpu().numpy()
            K_o = d["K_orig"].astype(np.float64)
            Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
            K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
            Rs = d["poses_R"].astype(np.float64)
            ps = d["poses_t"].astype(np.float64)
            fwds = d["causal_fwd"].astype(np.float64)
            nom_h = 1.55 if str(d["cam"]) == "camera_head" else 1.20
            # VKITTI training focal ratio: fx_vkitti=725 @ width=1242 -> fx_norm = 0.5837; cam fx_norm = K_o[0]/Ww
            fx_scale = (K_o[0] / Ww) / (725.0 / 1242.0)
            d_arr = np.full(n, np.inf, dtype=np.float32)
            for i in range(n):
                if kind == "relative":
                    z_map = ground_anchor_relative_depth(depths[i], K_d, Rs[i], nom_h)
                else:
                    # DAV2-Metric-Outdoor-Large predicts depth in meters scaled for VKITTI intrinsics
                    z_map = depths[i] * fx_scale
                    if i == 0:
                        raw_medians.append(float(np.median(depths[i])))
                dh, _ = depth_map_to_corridor_d(z_map, K_d, Rs[i], ps[i], fwds[i], MIN_PTS_S8)
                d_arr[i] = dh
            pred_d[sc] = d_arr
        curve = eval_score_curve(eval_rows_map, pred_d, bw_speeds, b1_thrs, higher_is_alert=False)
        out[bname] = {"curve": curve, "summary": summarize_at_fa_caps(curve), "raw_median_sample": round(float(np.median(raw_medians)), 3) if raw_medians else None}
        bf = out[bname]["summary"]["best_floor"]
        print(f"[{set_name}] {bname}: floor={bf['late']}/{bf['avoidable']} ({bf['late_rate']*100:.1f}%), FA_ep={bf['fa_ep_per_km']:.1f}, FA_fr={bf['fa_fr_per_km']:.1f} (raw_med={out[bname]['raw_median_sample']})", flush=True)
        del mod, proc
        torch.cuda.empty_cache()

    return out


def train_variant_v2(
    mode: str,
    train_dicts: list[dict],
    eval_synth_dicts: dict[str, dict],
    eval_real_dicts: dict[str, dict],
    bw_speeds: dict,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    clip_len: int = 12,
    clips_per_epoch: int = 1600,
) -> dict:
    t0 = time.time()
    torch.manual_seed(20261120)
    np.random.seed(20261120)

    model = EgoClearNet(mode=mode, backbone_name="facebook/dinov2-small", c_bev=128, unfreeze_last_k=6).to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Training EgoClear-v2-{mode} on {len(train_dicts)} sessions (params={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M, bs={batch_size}x{clip_len}={batch_size*clip_len} fr/step) ===", flush=True)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 4e-5, "weight_decay": 1e-4},
            {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4},
        ]
    )
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
                f"  [EgoClear-v2-{mode} ep {ep:02d}/{epochs}] tot={ep_loss['tot']:.4f} "
                f"(st={ep_loss['st']:.4f} dy={ep_loss['dy']:.4f} co={ep_loss['co']:.4f}) "
                f"{time.time()-t_ep:.1f}s | peak_vram={peak_gb:.1f}GB",
                flush=True,
            )

    ckpt_local = LOCAL_RUNS / f"egoclear_v2_{mode}.pt"
    torch.save({"mode": mode, "c_bev": 128, "state_dict": model.state_dict(), "history": history}, ckpt_local)
    shutil.copy2(ckpt_local, DRIVE_RUNS / ckpt_local.name)

    # Evaluate on ALL 3 test regimes!
    ev_clean = eval_egoclear(model, eval_synth_dicts, bw_speeds, device, noisy_vio=False)
    ev_noisy = eval_egoclear(model, eval_synth_dicts, bw_speeds, device, noisy_vio=True)
    ev_real = eval_egoclear(model, eval_real_dicts, bw_speeds, device, noisy_vio=False)

    for tag, ev in (("SYNTH_CLEAN", ev_clean), ("SYNTH_NOISY_VIO", ev_noisy), ("REAL_ZEROSHOT", ev_real)):
        for rk in ("fixed_4m", "walker_aware"):
            s = ev[rk]["summary"]
            m85 = s.get("late_at_fa_fr<=80.0")
            m120 = s.get("late_at_fa_fr<=120.3")
            m85_s = f"{m85['late']}/{m85['avoidable']}({m85['late_rate']*100:.1f}%,ep={m85['fa_ep_per_km']:.1f},fr={m85['fa_fr_per_km']:.1f})" if m85 else "N/A"
            m120_s = f"{m120['late']}/{m120['avoidable']}({m120['late_rate']*100:.1f}%,ep={m120['fa_ep_per_km']:.1f},fr={m120['fa_fr_per_km']:.1f})" if m120 else "N/A"
            print(
                f"[P2 EgoClear-v2-{mode} | {tag} | {rk}] @fr<=80: {m85_s} | @fr<=120: {m120_s} | "
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
        "eval_synth_clean": ev_clean,
        "eval_synth_noisy_vio": ev_noisy,
        "eval_real_zeroshot": ev_real,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[p2] GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    # Load 177 existing train + 110 frozen eval
    train_dicts = []
    eval_synth_dicts = {}
    s287_dir = Path("/content/sanpo_287/scenes")
    for sc, sp in m287["splits"].items():
        p = s287_dir / f"{sc}.npz"
        if not p.exists():
            continue
        d = load_scene(p)
        if sp == "train":
            train_dicts.append(d)
        else:
            eval_synth_dicts[sc] = d

    # Load +450 extra disjoint train scenes
    extra_dir = Path("/content/sanpo_p2/extra_scenes")
    n_extra = 0
    for p in sorted(extra_dir.glob("*.npz")):
        train_dicts.append(load_scene(p))
        n_extra += 1

    # Load 40 SANPO-Real zero-shot sessions
    real_dir = Path("/content/sanpo_p2/real_scenes")
    eval_real_dicts = {}
    for p in sorted(real_dir.glob("*.npz")):
        eval_real_dicts[p.stem] = load_scene(p)

    print(
        f"[p2] Loaded train={len(train_dicts)} (177+{n_extra}), "
        f"synth_eval={len(eval_synth_dicts)}, real_zeroshot={len(eval_real_dicts)}",
        flush=True,
    )

    # Pretrained pure-vision baselines on BOTH synthetic eval & real zero-shot!
    base_synth = eval_pretrained_baselines_on_set("SYNTH_110", eval_synth_dicts, bw_speeds, device)
    base_real = eval_pretrained_baselines_on_set("REAL_ZEROSHOT", eval_real_dicts, bw_speeds, device)

    # Train & evaluate EgoClear-v2-Single vs EgoClear-v2-Full (saturated G4 96 GB!)
    egoclear_v2 = {}
    for mode in ("single", "full"):
        egoclear_v2[mode] = train_variant_v2(
            mode, train_dicts, eval_synth_dicts, eval_real_dicts, bw_speeds, device,
            epochs=20, batch_size=32, clip_len=12, clips_per_epoch=1600,
        )

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "total_wall_sec": round(time.time() - t_all, 1),
        "counts": {
            "train_synth": len(train_dicts),
            "eval_synth": len(eval_synth_dicts),
            "eval_real_zeroshot": len(eval_real_dicts),
        },
        "baselines_synth": base_synth,
        "baselines_real_zeroshot": base_real,
        "egoclear_v2": egoclear_v2,
    }
    res_local = LOCAL_RUNS / "phase2_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase2_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[p2] DONE saved phase2_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
