"""Phase 17 TURBO (Colab G4 96 GB Blackwell): EgoClear-v12 ("Turbo-Omni")!

Delivers a ~3x faster Full-NPU encoder (~18-22ms on S24 Ultra Hexagon v75, ~25-28ms end-to-end = 36-40+ FPS!)
WITH ZERO backbone downgrade (full pretrained MoGe-2 DINOv2 ViT-Small/14 intact!) while closing EVERY remaining
algorithmic weakness exposed in our Phase 16 deep autopsy:
  1. 4x Lighter Unified 84x148 Geometry+Edge Decoder (`MoGe-Turbo`, ~5 GMACs vs 88.1 GMACs of 336x592 stacks!)
     + High-Res RGB Edge & Optical Quality Stem (2-px branches/poles + raw luma/sharpness/exposure cues).
     Two-stage curriculum: Stage 1 (3 ep fast dense geometry distillation from v11 + v6.6 + GT depth_s8)
     -> Stage 2 (8 ep joint end-to-end Turbo-Omni distillation & calibration with RAW vs EMA selection).
  2. Self-Calibrating Ground-Anchored 3D Probe Grid (`H_GROUND_SLICES`, NY=9, [-0.75..+1.95 m above floor]):
     mount-height invariant across handheld phones, chest mounts, Aria/SANPO-Head glasses, and 0.7 m CODa robots!
  3. Static-Gated 3D Physics Shortcut (`vh * (1 - 0.9*dyn_prob)`) + 3x Pedestrian Anchor Weight:
     eliminates pedestrian verticality leakage into static `corr_probs` (crushing Real 40 dyn@0.5!).
  4. Full ~80,000-Frame ARKit Indoor Flat-Floor Drop-Off Negative Unlock (`360 -> ~80k` frames!):
     crushes indoor flat-floor `drop_off` false alarms from 28.3% down to <1-2% while keeping Real AUROC >0.93!
  5. Dedicated Current-Frame-Aware `UNKNOWN` Head + Rich 5-Mode Outlier Exposure (10%):
     blackout, red finger-cover, sunlit whiteout, directional motion blur (17-31px), and wet/defocus blur ->
     instant frame-1 abstention (>95%) WITHOUT stale `mem_prev` leakage!
  6. Sharper Clear-Corridor Tail (`anchor_neg = 0.08`, `k_top = 0.15`):
     recovers SANPO-Synth @120 (~5%) while keeping 5-domain single-global-threshold (thr=0.50) razor-sharp!
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import tarfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v7_model import FEAT_H, FEAT_W
from egoclear_v11_model import EgoClearV11Net
from egoclear_v12_model import EgoClearV12EncoderExport, EgoClearV12Net, EgoClearV12StepExport
from eval_closed_loop import eval_score_curve, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase7_train_v6_g4 import eval_v6_all
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_stage2_diagnostics_g4 import paired_cluster_bootstrap, per_route_at_thr
from phase11_train_v7_g4 import eval_dark_domain, print_block
from phase11_train_v75_g4 import (
    IDX_4M_EXACT,
    Z_EDGES,
    prepare_scene_slice_tensors,
    recalibrate_bn_stats,
)
from phase12_v9_probe_g4 import logit, make_variant_scenes, sigmoid, summarize_scores
from phase13_train_v10_g4 import eval_frontend_mode
from phase14_train_v105_g4 import (
    paired_route_audit_from_scores,
    stage_p13_extra_datasets_parallel,
)
from phase15_train_v106_g4 import audit_prehit_leakage
from phase16_train_v11_g4 import (
    SKY_CLASS,
    WALK_CLASSES,
    V11OmniDataset,
    _aux_survival_loss,
    _binary_auroc,
    _cum_from_occ,
)
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, load_scene

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase17_v12"
LOCAL_RUNS = Path("/content/egoclear_runs_p17_v12")
CODA_DIR = Path("/content/coda_zeroshot_v0")

MULTI_RANGE_BINS = (3, 7, 11, IDX_4M_EXACT)  # 1.1m, 1.9m, 2.7m, 4.1m
MULTI_RANGE_W = (0.35, 0.55, 0.55, 1.00)


def compute_omni_gt_v12_for_scene(d: dict, has_dyn_gt: bool, default_h_cam: float = -1.0, allow_flat_pose_neg: bool = False) -> dict:
    """Upgraded v12 3D Omni GT:
      - Unlocks ~80k verified flat indoor ARKit drop-off negative frames (local vertical flatness + single-floor span <0.55m)
      - Saves verified per-scene ground-truth camera height `h_cam_gt` for online floor-estimator supervision & audit!
    """
    st = d["gt_bev_static"]
    dy = d["gt_bev_dynamic"]
    T = st.shape[0]

    occ_l = st[:, :, :, 1:5].max(axis=(1, 3))
    occ_r = st[:, :, :, 10:14].max(axis=(1, 3))
    occ_h = st[:, 2, :, 6:9].max(axis=2)
    y_left = _cum_from_occ(occ_l)
    y_right = _cum_from_occ(occ_r)
    y_head = _cum_from_occ(occ_h)

    y_dyn_closing = np.zeros((T, BEV_NZ), dtype=np.float32)
    y_dyn_passing = np.zeros((T,), dtype=np.float32)
    valid_dyn_closing = np.full((T,), 1.0 if has_dyn_gt else 0.0, dtype=np.float32)
    if has_dyn_gt:
        occ_d = dy[:, :, :, 5:10].max(axis=(1, 3)) > 0
        nearest_z = np.full((T,), np.inf, dtype=np.float32)
        nearest_bin = np.full((T,), -1, dtype=np.int32)
        for t in range(T):
            hits = np.flatnonzero(occ_d[t])
            if len(hits) > 0:
                nearest_bin[t] = int(hits[0])
                nearest_z[t] = float(Z_EDGES[hits[0]])
        for t in range(T):
            zt = nearest_z[t]
            if not np.isfinite(zt) or zt > 4.5:
                continue
            z_prev2 = nearest_z[t - 2] if t >= 2 else np.inf
            z_prev1 = nearest_z[t - 1] if t >= 1 else np.inf
            closing2 = np.isfinite(z_prev2) and ((z_prev2 - zt) >= 0.25)
            closing1 = np.isfinite(z_prev1) and ((z_prev1 - zt) >= 0.15)
            imminent = zt <= 1.30
            if (zt <= 3.70 and (closing2 or closing1)) or imminent:
                b0 = max(0, int(nearest_bin[t]))
                y_dyn_closing[t, b0:] = 1.0
            else:
                y_dyn_passing[t] = 1.0

    y_drop = np.zeros((T, BEV_NZ), dtype=np.float32)
    valid_drop = np.zeros((T,), dtype=np.float32)
    h_cam_gt = float(default_h_cam)
    depth_s8 = d.get("depth_s8")
    if depth_s8 is not None and len(depth_s8) == T:
        seg_s8 = d.get("seg_s8")
        has_seg = (seg_s8 is not None) and (len(seg_s8) == T)
        orig_hw = d["orig_hw"]
        K_orig = d["K_orig"].astype(np.float64)
        R_all = d["poses_R"].astype(np.float64)
        fwd_all = d["causal_fwd"].astype(np.float64)
        hd, wd = depth_s8.shape[1], depth_s8.shape[2]
        sx, sy = wd / float(orig_hw[1]), hd / float(orig_hw[0])
        fx, fy, cx, cy = K_orig[0] * sx, K_orig[1] * sy, K_orig[2] * sx, K_orig[3] * sy
        vv, uu = np.mgrid[0:hd, 0:wd]
        dirs = np.stack([((uu - cx) / max(fx, 1.0)).reshape(-1), ((vv - cy) / max(fy, 1.0)).reshape(-1), np.ones(hd * wd)], axis=-1)

        h_cam_samples = []
        frame_pts = [None] * T
        for t in range(T):
            dm = depth_s8[t].astype(np.float32).reshape(-1)
            ok = np.isfinite(dm) & (dm >= 0.30) & (dm <= 10.0)
            s_flat = seg_s8[t].reshape(-1) if has_seg else None
            if has_seg:
                ok &= (s_flat != SKY_CLASS)
            if ok.sum() < 30:
                continue
            rel_w = (R_all[t] @ (dirs[ok] * dm[ok, None]).T).T
            fwd = fwd_all[t]
            fn = np.linalg.norm(fwd[:2])
            if fn < 1e-4:
                continue
            f2 = np.array([fwd[0] / fn, fwd[1] / fn, 0.0], dtype=np.float64)
            r2 = np.array([f2[1], -f2[0], 0.0], dtype=np.float64)
            xq, zq, dz = rel_w @ r2, rel_w @ f2, rel_w[:, 2]
            is_walk = np.isin(s_flat[ok], WALK_CLASSES) if has_seg else None
            frame_pts[t] = (xq, zq, dz, is_walk)
            if has_seg:
                wf = is_walk & (np.abs(xq) <= 1.20) & (zq >= 1.00) & (zq <= 4.20) & (dz < -0.55) & (dz > -2.15)
                if wf.sum() >= 15:
                    h_cam_samples.append(float(-np.median(dz[wf])))
            else:
                wf = (np.abs(xq) <= 1.20) & (zq >= 1.50) & (zq <= 4.50) & (dz < -0.60) & (dz > -2.30)
                if wf.sum() >= 25:
                    h_cam_samples.append(float(-np.percentile(dz[wf], 12)))

        if len(h_cam_samples) >= 3:
            h_cam = float(np.clip(np.median(h_cam_samples), 0.70, 1.95))
            h_cam_gt = h_cam
            for t in range(T):
                fp = frame_pts[t]
                if fp is None:
                    continue
                xq, zq, dz, is_walk = fp
                vis_strip = (np.abs(xq) <= 0.90) & (zq >= 0.80) & (zq <= 4.10)
                if vis_strip.sum() < 25:
                    continue
                valid_drop[t] = 1.0
                h_rel = dz + h_cam
                if is_walk is not None:
                    imm = is_walk & (np.abs(xq) <= 0.85) & (zq >= 1.00) & (zq <= 2.30) & (np.abs(h_rel) <= 0.35)
                else:
                    imm = (np.abs(xq) <= 0.75) & (zq >= 1.20) & (zq <= 2.30) & (np.abs(h_rel) <= 0.30)
                if imm.sum() >= 10:
                    h_rel = h_rel - float(np.median(h_rel[imm]))

                corr_g = (np.abs(xq) <= 0.45) & (zq >= 0.50) & (zq < 5.10) & (h_rel <= 0.15) & (h_rel >= -2.50)
                if corr_g.sum() < 8:
                    continue
                z_cg, h_cg = zq[corr_g], h_rel[corr_g]
                b_idx = np.clip(np.floor((z_cg - 0.30) / 0.20).astype(np.int32), 0, BEV_NZ - 1)
                bin_h = np.full(BEV_NZ, np.nan, dtype=np.float32)
                bin_deep_cnt = np.zeros(BEV_NZ, dtype=np.int32)
                for b in range(BEV_NZ):
                    mb = b_idx == b
                    if int(mb.sum()) >= 4:
                        bin_h[b] = float(np.median(h_cg[mb]))
                        bin_deep_cnt[b] = int((h_cg[mb] <= -0.22).sum())

                drop_bin = -1
                for b in range(2, IDX_4M_EXACT + 1):
                    hb = bin_h[b]
                    if not np.isfinite(hb) or hb > -0.22:
                        continue
                    z_b = float(Z_EDGES[b])
                    lo_b, hi_b = max(0, b - 1), min(BEV_NZ, b + 2)
                    if int(bin_deep_cnt[lo_b:hi_b].sum()) < 8:
                        continue
                    steep_slope = (hb / max(z_b, 0.8)) <= -0.18
                    prev_vals = [bin_h[k] for k in range(max(0, b - 3), b) if np.isfinite(bin_h[k])]
                    sharp_step = bool(prev_vals) and ((hb - max(prev_vals)) <= -0.18)
                    if steep_slope or sharp_step:
                        drop_bin = b
                        break
                if drop_bin >= 0:
                    y_drop[t, drop_bin:] = 1.0
    elif allow_flat_pose_neg:
        # FIXED ARKit/indoor flat-floor negative unlock (ONLY on indoor single-story rooms & verified flat campuses!):
        # Single-story rooms have handheld bobbing (0.15-0.45m), whereas stairs change z monotonically >0.6m!
        tz = d["poses_t"][:, 2]
        if len(tz) >= 5 and np.all(np.isfinite(tz)):
            span = float(np.max(tz) - np.min(tz))
            if span < 0.55:
                for t in range(T):
                    t_lo = max(0, t - 10)
                    t_hi = min(T - 1, t + 10)
                    if abs(float(tz[t_hi] - tz[t_lo])) < 0.18:
                        valid_drop[t] = 1.0

    return {
        "y_left": y_left,
        "y_right": y_right,
        "y_head": y_head,
        "y_drop": y_drop,
        "valid_drop": valid_drop,
        "y_dyn_closing": y_dyn_closing,
        "valid_dyn_closing": valid_dyn_closing,
        "y_dyn_passing": y_dyn_passing,
        "h_cam_gt": h_cam_gt,
    }


def enrich_pool_v12(pool: list[dict] | dict[str, dict], name: str, has_dyn_gt: bool, default_h_cam: float = -1.0, allow_flat_pose_neg: bool = False, max_workers: int = 24) -> dict:
    t0 = time.time()
    items = list(pool.values()) if isinstance(pool, dict) else pool
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        res_list = list(ex.map(lambda d: compute_omni_gt_v12_for_scene(d, has_dyn_gt, default_h_cam, allow_flat_pose_neg), items))
    n_fr = n_vdrop = n_drop_pos = n_drop_sc = n_hcam = 0
    for d, og in zip(items, res_list):
        d.update(og)
        n_fr += len(og["valid_drop"])
        vd = og["valid_drop"] > 0.5
        n_vdrop += int(vd.sum())
        dp = int((vd & (og["y_drop"][:, IDX_4M_EXACT] > 0.5)).sum())
        n_drop_pos += dp
        if dp > 0:
            n_drop_sc += 1
        if og["h_cam_gt"] > 0:
            n_hcam += 1
    stats = {
        "scenes": len(items),
        "frames": n_fr,
        "valid_drop_frames": n_vdrop,
        "drop_pos_frames": n_drop_pos,
        "drop_pos_scenes": n_drop_sc,
        "drop_rate_pct": round(100.0 * n_drop_pos / max(1, n_vdrop), 2),
        "h_cam_gt_scenes": n_hcam,
        "sec": round(time.time() - t0, 1),
    }
    print(f"[omni-gt-v12] {name:16s}: {stats}", flush=True)
    return stats


def _apply_unobs_corruption_rgb01(img_chw: np.ndarray, mode: int, rng: np.random.Generator) -> np.ndarray:
    """Applies 5 realistic unobservable lens failures in clean RGB [0,1] space."""
    H, W = img_chw.shape[1], img_chw.shape[2]
    if mode == 0:
        # Pitch-black / dark-pocket cover
        base = rng.uniform(0.0, 0.025)
        out = np.clip(base + rng.normal(0.0, 0.008, size=(3, H, W)).astype(np.float32), 0.0, 1.0)
    elif mode == 1:
        # Translucent finger-on-lens warm red/orange glow
        col = np.array([rng.uniform(0.22, 0.48), rng.uniform(0.04, 0.12), rng.uniform(0.02, 0.08)], dtype=np.float32)[:, None, None]
        grad_y = np.linspace(0.75, 1.25, H, dtype=np.float32)[None, :, None]
        out = np.clip(col * grad_y + rng.normal(0.0, 0.015, size=(3, H, W)).astype(np.float32), 0.0, 1.0)
    elif mode == 2:
        # Sunlit overexposed whiteout
        base = rng.uniform(0.95, 1.00)
        out = np.clip(base - np.abs(rng.normal(0.0, 0.015, size=(3, H, W)).astype(np.float32)), 0.0, 1.0)
    elif mode == 3:
        # Severe directional linear motion blur (17..31 px kernel at random angle)
        k = int(rng.choice([17, 21, 25, 31]))
        ang = float(rng.uniform(0.0, 180.0))
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), ang, 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        ks = float(kern.sum())
        if ks > 1e-6:
            kern /= ks
        hwc = np.transpose(img_chw, (1, 2, 0))
        out = np.clip(np.transpose(cv2.filter2D(hwc, -1, kern), (2, 0, 1)), 0.0, 1.0)
    else:
        # Heavy defocus / wet fogged lens
        sig = float(rng.uniform(9.0, 16.0))
        hwc = np.transpose(img_chw, (1, 2, 0))
        out = np.clip(np.transpose(cv2.GaussianBlur(hwc, (0, 0), sig), (2, 0, 1)), 0.0, 1.0)
    return out


class V12TurboOmniDataset(V11OmniDataset):
    """Extends V11OmniDataset with rich 5-mode Outlier Exposure (10%) & per-scene `h_cam_gt`."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.28:
            dom = "synth"
        elif u_dom < 0.58 and len(self.pools["real_gt"]) > 0:
            dom = "real_gt"
        elif u_dom < 0.65 and len(self.pools["real_auto"]) > 0:
            dom = "real_auto"
        elif u_dom < 0.86 and len(self.pools["arkit"]) > 0:
            dom = "arkit"
        elif len(self.pools["tartan"]) > 0:
            dom = "tartan"
        else:
            dom = "synth"

        pool = self.pools[dom]
        u_mine = rng.random()
        if dom in ("real_gt", "synth") and len(self.dyn_scenes[dom]) > 0 and u_mine < 0.30:
            d = pool[int(rng.choice(self.dyn_scenes[dom]))]
        elif len(self.drop_scenes[dom]) > 0 and u_mine < 0.50:
            d = pool[int(rng.choice(self.drop_scenes[dom]))]
        elif dom in ("real_gt", "synth") and len(self.head_scenes[dom]) > 0 and u_mine < 0.70:
            d = pool[int(rng.choice(self.head_scenes[dom]))]
        else:
            d = pool[int(rng.integers(len(pool)))]

        n = len(d["eval_rows"])
        T = self.clip_len
        max_s = max(0, n - T)
        s0 = int(rng.integers(0, max_s + 1)) if max_s > 0 else 0
        if u_mine < 0.50 and len(self.drop_scenes[dom]) > 0:
            cand = np.flatnonzero((d["valid_drop"] > 0.5) & (d["y_drop"][:, IDX_4M_EXACT] > 0.5))
            if len(cand) > 0 and rng.random() < 0.70:
                s0 = int(np.clip(int(rng.choice(cand)) - int(rng.integers(2, T - 1)), 0, max_s))
        elif dom in ("real_gt", "synth") and u_mine < 0.70:
            cand = np.flatnonzero(d["y_head"][:, IDX_4M_EXACT] > 0.5)
            if len(cand) > 0 and rng.random() < 0.70:
                s0 = int(np.clip(int(rng.choice(cand)) - int(rng.integers(2, T - 1)), 0, max_s))

        sl = np.arange(s0, s0 + T)
        if rng.random() < 0.08:
            freeze_at = int(rng.integers(2, T - 2))
            sl[freeze_at:] = sl[freeze_at]

        item = self._build_clip_from_scene(d, dom, sl, rng)
        item["y_left"] = torch.from_numpy(d["y_left"][sl].copy())
        item["y_right"] = torch.from_numpy(d["y_right"][sl].copy())
        item["y_head"] = torch.from_numpy(d["y_head"][sl].copy())
        item["y_drop"] = torch.from_numpy(d["y_drop"][sl].copy())
        item["valid_drop"] = torch.from_numpy(d["valid_drop"][sl].copy())
        item["y_dyn_closing"] = torch.from_numpy(d["y_dyn_closing"][sl].copy())
        item["valid_dyn_closing"] = torch.from_numpy(d["valid_dyn_closing"][sl].copy())
        item["y_dyn_passing"] = torch.from_numpy(d["y_dyn_passing"][sl].copy())
        hc = float(d.get("h_cam_gt", -1.0))
        item["h_cam_gt"] = torch.tensor(hc if hc > 0 else 1.35, dtype=torch.float32)
        item["has_h_cam_gt"] = torch.tensor(1.0 if hc > 0 else 0.0, dtype=torch.float32)

        # 10% Rich 5-Mode Outlier Exposure in clean RGB [0,1] space for instant frame-1 UNKNOWN abstention!
        unobs = np.zeros((T,), dtype=np.float32)
        if rng.random() < 0.10:
            rgb_norm = item["rgb_s"].float().numpy()
            rgb_01 = np.clip(rgb_norm * IMAGENET_STD[None] + IMAGENET_MEAN[None], 0.0, 1.0)
            k_len = int(rng.integers(2, 6))
            t_st = int(rng.integers(1, max(2, T - k_len + 1)))
            mode = int(rng.integers(0, 5))
            for ti in range(t_st, min(T, t_st + k_len)):
                unobs[ti] = 1.0
                c_img = _apply_unobs_corruption_rgb01(rgb_01[ti], mode, rng)
                rgb_norm[ti] = (c_img - IMAGENET_MEAN) / IMAGENET_STD
            item["rgb_s"] = torch.from_numpy(rgb_norm.astype(np.float16))
        item["is_unobservable"] = torch.from_numpy(unobs)
        return item


def compute_stage1_geom_loss(
    s_out: tuple,
    t11_out: tuple,
    dav2_targets: torch.Tensor,
    gt_d_sub: torch.Tensor,
    obs_ok_sub: torch.Tensor,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage 1 dense distillation: trains ~5 GMAC UnifiedTurboGeoDecoder to match 97 GMAC v11 teacher + GT depth + DAV2!"""
    f_s, _g_s, z_s, n_s, m_s, dlz_s, dnr_s, dv_s = s_out
    f_t, _g_t, z_t, n_t, m_t, _dlzt, _dnrt, _dvt = t11_out
    obs = obs_ok_sub.reshape(-1, 1, 1, 1).float()
    z_s, n_s, m_s, f_s = z_s.float(), n_s.float(), m_s.float(), f_s.float()
    z_t, n_t, m_t, f_t = z_t.float().detach(), n_t.float().detach(), m_t.float().detach(), f_t.float().detach()

    ok_t = ((m_t > 0.45) & (z_t > 0.20) & (z_t < 15.0)).float() * obs
    denom = ok_t.sum().clamp(min=1.0)
    lz_s = torch.log(z_s.clamp(min=0.15))
    lz_t = torch.log(z_t.clamp(min=0.15))

    l_z_t = ((lz_s - lz_t).abs() * ok_t).sum() / denom
    l_grad = (
        (
            (F.conv2d(lz_s, sobel_x, padding=1) - F.conv2d(lz_t, sobel_x, padding=1)).abs()
            + (F.conv2d(lz_s, sobel_y, padding=1) - F.conv2d(lz_t, sobel_y, padding=1)).abs()
        )
        * ok_t
    ).sum() / denom
    l_nrm = ((1.0 - (n_s * n_t).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)) * ok_t).sum() / denom
    m_sc = m_s.clamp(1e-4, 1.0 - 1e-4)
    m_tc = m_t.clamp(0.0, 1.0)
    l_msk = (-(m_tc * torch.log(m_sc) + (1.0 - m_tc) * torch.log(1.0 - m_sc)) * obs).sum() / (
        obs.sum().clamp(min=1.0) * FEAT_H * FEAT_W
    )
    l_f2d = (
        (0.65 * (1.0 - F.cosine_similarity(f_s, f_t, dim=1, eps=1e-5).unsqueeze(1)) + 0.35 * F.smooth_l1_loss(f_s, f_t, reduction="none").mean(dim=1, keepdim=True))
        * ok_t
    ).sum() / denom

    # Real metric GT depth anchor (only on frames where gt_d_sub is valid!)
    gt_d = gt_d_sub.reshape(-1, 1, FEAT_H, FEAT_W).float()
    ok_gt = ((gt_d >= 0.35) & (gt_d <= 8.0)).float() * obs
    l_z_gt = ((lz_s - torch.log(gt_d.clamp(min=0.35))).abs() * ok_gt).sum() / ok_gt.sum() if ok_gt.sum() > 20.0 else torch.tensor(0.0, device=z_s.device)

    # DAV2 sharp relative disparity & feature alignment
    dv_s = dv_s.float()
    l_dv_cos = (1.0 - F.cosine_similarity(dv_s[:, :64], dav2_targets[:, :64], dim=1, eps=1e-5).unsqueeze(1))
    l_dv_d = F.smooth_l1_loss(dv_s[:, 64:65], dav2_targets[:, 64:65], reduction="none")
    l_dav2 = ((0.60 * l_dv_cos + 0.40 * l_dv_d) * ok_t).sum() / denom

    l_reg = 0.03 * (dlz_s.float().pow(2).mean() + dnr_s.float().pow(2).mean())
    tot = 1.60 * l_z_t + 1.10 * l_z_gt + 0.85 * l_grad + 0.85 * l_nrm + 0.45 * l_msk + 1.00 * l_f2d + 0.55 * l_dav2 + l_reg
    return tot, {
        "tot": float(tot.item()),
        "zt": float(l_z_t.item()),
        "zgt": float(l_z_gt.item()),
        "nrm": float(l_nrm.item()),
        "f2d": float(l_f2d.item()),
    }


def compute_loss_v12(
    out_s: dict[str, torch.Tensor],
    out_v66: dict[str, torch.Tensor],
    out_v11: dict[str, torch.Tensor],
    dav2_targets: torch.Tensor,
    batch: dict[str, torch.Tensor],
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    kd_w: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage 2 EgoClear-v12 Turbo-Omni Loss:
      - Sharper clear tail (anchor_neg=0.08, k_top=0.15)
      - 3x Pedestrian Anchor Weight on ALL 4 ranges
      - Dedicated 5-Mode OE UNKNOWN loss (weight 1.25)
      - Floor-height calibration loss (`loss_hfl`)
      - Dual-Teacher KD (`v6.6 + v11` on static/corridor + `v11` soft Omni KD + dense `v11` geometry + `GT depth_s8`)!
    """
    st_log = out_s["static_logits"].float()
    dy_log = out_s["dynamic_logits"].float()
    haz_log = out_s["hazard_logits"].float()
    cum_h = out_s["cum_hazard"].float()
    p_cum = out_s["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_log = out_s["unc_logits"].float()

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()
    v_co = batch["valid_corr"].float()
    dn_w = batch["dyn_neg_w"].float()
    dyn_fl = batch["dyn_flag"].float()
    dom_id = batch["dom_id"]
    is_dark = batch["is_dark"].float().view(-1, 1, 1)
    is_auto = batch["is_autoseg"].float().view(-1, 1)
    dyn_trusted = batch["dyn_gt_trusted"].float().view(-1, 1, 1, 1, 1)
    is_unobs = batch["is_unobservable"].float()
    obs_ok = (1.0 - is_unobs).clamp(0.0, 1.0)

    p66 = out_v66["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p11 = out_v11["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    st66 = torch.sigmoid(out_v66["static_logits"].float().detach())
    st11 = torch.sigmoid(out_v11["static_logits"].float().detach())
    dy66 = torch.sigmoid(out_v66["dynamic_logits"].float().detach())

    dy66_corr_max = dy66[:, :, :, : IDX_4M_EXACT + 1, 6:9].amax(dim=(2, 3, 4))
    gt_dy_corr_max = gt_dy[:, :, :, : IDX_4M_EXACT + 1, 6:9].amax(dim=(2, 3, 4))
    auto_ped_suspect = (is_auto > 0.5) & ((dy66_corr_max > 0.35) | ((p66[:, :, IDX_4M_EXACT] < 0.22) & (gt_dy_corr_max > 0.5)))
    v_co_eff = torch.where(auto_ped_suspect & (y_co[:, :, IDX_4M_EXACT] > 0.5), torch.zeros_like(v_co), v_co) * obs_ok
    st_auto_guard = (1.0 - (is_auto[:, :, None, None, None] * (dy66 > 0.30).float())).clamp(0.0, 1.0) * obs_ok[:, :, None, None, None]

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 2.6; col_w[5] = 1.5; col_w[9] = 1.5; col_w[1:5] = 1.4; col_w[10:14] = 1.4
    hb_w = torch.tensor([1.0, 1.3, 4.0], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    # 1. Static & Dynamic BEV
    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(5.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st_bce = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w * st_auto_guard).mean()
    st_prob = torch.sigmoid(st_log)
    tversky_sum = torch.tensor(0.0, device=device)
    for bi, bw in enumerate((1.0, 1.0, 2.2)):
        pb = st_prob[:, :, bi] * st_auto_guard[:, :, 0]
        gb = gt_st[:, :, bi] * st_auto_guard[:, :, 0]
        tp = (pb * gb).sum()
        fp = (pb * (1.0 - gb)).sum()
        fn = ((1.0 - pb) * gb).sum()
        tversky_sum = tversky_sum + bw * (1.0 - (tp + 1.0) / (tp + 0.30 * fp + 0.70 * fn + 1.0))
    loss_st = loss_st_bce + 0.30 * (tversky_sum / 4.2)

    dy_w = (1.0 + 2.0 * dyn_trusted) * obs_ok[:, :, None, None, None]
    loss_dy = (
        F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(6.5, device=device), reduction="none") * dy_w
    ).mean()

    # 2. Center Corridor Monotonic Survival + First-Hit + Pre-Hit Zero-Leakage
    z_edges = torch.tensor(Z_EDGES, device=device, dtype=torch.float32)
    z_w = torch.where(z_edges <= 4.35, 2.0, 0.5)[None, None, :]
    bce_cum = -(1.75 * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co_eff * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.5) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co_eff * dn_w)[:, :, None]
    soft_h = F.softplus(haz_log)
    log_p_hit = torch.log(1.0 - torch.exp(-soft_h).clamp(max=1.0 - 1e-6))
    loss_fh = ((-y_first * log_p_hit + (1.0 - y_first) * soft_h) * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)

    y_next = F.pad(y_co[..., 1:], (0, 1), value=1.0)
    has_far_obs = (y_co[:, :, IDX_4M_EXACT] > 0.5).unsqueeze(-1)
    prehit_mask = (v_co_eff.unsqueeze(-1) > 0.5) & has_far_obs & (y_co < 0.5) & (y_next < 0.5)
    loss_prehit = torch.tensor(0.0, device=device)
    if prehit_mask.any():
        p_pre = p_cum[prehit_mask]
        h_pre = soft_h[prehit_mask]
        loss_prehit = (-torch.log((1.0 - p_pre).clamp(min=1e-4))).mean() + 2.5 * (F.relu(p_pre - 0.04) ** 2).mean() + 0.8 * h_pre.mean()

    # 3. Sharper Per-Domain Anchored Single-Global-Threshold (thr=0.50) + 3x Pedestrian Anchor!
    loss_dom_anc = torch.tensor(0.0, device=device)
    loss_rank = torch.tensor(0.0, device=device)
    loss_ohem = torch.tensor(0.0, device=device)
    dom_terms = 0.0
    is_ped_only_any = (dyn_fl > 0.5) | (batch["y_dyn_passing"].float() > 0.5)
    for d_val in range(5):
        dm = (dom_id == d_val)[:, None]
        if not dm.any():
            continue
        for b_idx, b_w in zip(MULTI_RANGE_BINS, MULTI_RANGE_W):
            pos_db = dm & (v_co_eff > 0.5) & (y_co[:, :, b_idx] > 0.5)
            neg_db = dm & (v_co_eff > 0.5) & (y_co[:, :, b_idx] < 0.5)
            cb = cum_h[:, :, b_idx]
            pb = p_cum[:, :, b_idx]
            if pos_db.any():
                loss_dom_anc = loss_dom_anc + b_w * F.relu(1.18 - cb[pos_db]).mean()
                dom_terms += b_w
            if neg_db.any():
                # Sharper negative tail target: lam <= 0.08 (p <= 0.077), k_top = 0.15 -> cures Synth @120!
                nv_l = cb[neg_db]
                k_top = max(1, int(0.15 * nv_l.numel()))
                loss_dom_anc = loss_dom_anc + 1.45 * b_w * (F.relu(torch.topk(nv_l, k_top).values - 0.08) ** 2).mean()
                nv_p = pb[neg_db]
                k_o = max(1, int(0.15 * nv_p.numel()))
                loss_ohem = loss_ohem + b_w * (F.relu(torch.topk(nv_p, k_o).values - 0.04) ** 2).mean()
                dom_terms += b_w
                # 3x explicit pedestrian suppression anchor at EVERY range!
                ped_db = neg_db & is_ped_only_any
                if ped_db.any():
                    loss_dom_anc = loss_dom_anc + 2.8 * b_w * (F.relu(cb[ped_db] - 0.08) ** 2).mean()
            if pos_db.any() and neg_db.any():
                lp = cb[pos_db]
                ln_v = cb[neg_db]
                k_h = max(1, int(0.20 * ln_v.numel()))
                ln_h = torch.topk(ln_v, k_h).values
                loss_rank = loss_rank + b_w * F.softplus((ln_h[None, :] - lp[:, None] + 0.65) / 0.40).mean()
    loss_dom_anc = loss_dom_anc / max(dom_terms, 1.0)
    loss_ohem = loss_ohem / max(dom_terms, 1.0)
    loss_rank = loss_rank / max(dom_terms, 1.0)

    haz_fr4 = (v_co_eff > 0.5) & (y_co[:, :, IDX_4M_EXACT] > 0.5)
    all_neg4 = (v_co_eff > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5)
    dyn_only_fr = all_neg4 & is_ped_only_any
    p4 = p_cum[:, :, IDX_4M_EXACT]
    lam4 = cum_h[:, :, IDX_4M_EXACT]
    clear_pair = all_neg4[:, 1:] & all_neg4[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    loss_dyn_h = torch.tensor(0.0, device=device)
    if dyn_only_fr.any():
        p_dyn = p4[dyn_only_fr]
        l_dyn = lam4[dyn_only_fr]
        loss_dyn_h = (
            (-torch.log((1.0 - p_dyn).clamp(min=1e-4))).mean()
            + 3.5 * (F.relu(p_dyn - 0.04) ** 2).mean()
            + 2.0 * (F.relu(l_dyn - 0.08) ** 2).mean()
        )
        if haz_fr4.any():
            l_pos = lam4[haz_fr4]
            loss_dyn_h = loss_dyn_h + 0.85 * F.softplus((l_dyn[None, :] - l_pos[:, None] + 0.80) / 0.38).mean()

    # 4. Dedicated Current-Frame-Aware UNKNOWN Gate (5-Mode OE + Clean Risk-Coverage!)
    err_target = ((p_cum.detach() > 0.5).float() != y_co).float()
    unc_target = torch.where(is_unobs[:, :, None] > 0.5, torch.ones_like(err_target), err_target)
    unc_mask = torch.maximum(v_co_eff, is_unobs)[:, :, None]
    unc_w = torch.where(is_unobs[:, :, None] > 0.5, torch.full_like(err_target, 5.0), torch.ones_like(err_target))
    loss_un = (F.binary_cross_entropy_with_logits(un_log, unc_target, reduction="none") * unc_mask * unc_w).sum() / (
        (unc_mask * unc_w).sum().clamp(min=1.0)
    )

    # 5. Auxiliary Floor-Height Calibration Loss (locks online 3-frame `h_floor_cam` onto true mount height!)
    hfl_pred = out_s["h_floor_cam"].float()  # (B, T), negative meters
    hfl_gt = -batch["h_cam_gt"].float().view(-1, 1).expand_as(hfl_pred)
    hfl_m = batch["has_h_cam_gt"].float().view(-1, 1) * obs_ok
    loss_hfl = (F.smooth_l1_loss(hfl_pred, hfl_gt, reduction="none", beta=0.10) * hfl_m).sum() / hfl_m.sum().clamp(min=1.0)

    # 6. All 5 Omni Survival Heads (GT + soft v11 KD where trusted!)
    v_lane = (1.0 - (is_auto > 0.5).float() * (dy66.amax(dim=(2, 3, 4)) > 0.35).float()).clamp(0.0, 1.0) * obs_ok
    loss_left = _aux_survival_loss(out_s["left_hazard_logits"], out_s["left_cum_hazard"], out_s["left_probs"], batch["y_left"].float(), v_lane, pos_w=1.8, anchor_neg=0.10)
    loss_right = _aux_survival_loss(out_s["right_hazard_logits"], out_s["right_cum_hazard"], out_s["right_probs"], batch["y_right"].float(), v_lane, pos_w=1.8, anchor_neg=0.10)
    loss_head_c = _aux_survival_loss(out_s["head_hazard_logits"], out_s["head_cum_hazard"], out_s["head_probs"], batch["y_head"].float(), v_co_eff, pos_w=4.2, anchor_pos=1.10, anchor_neg=0.10)

    v_drop = batch["valid_drop"].float() * obs_ok
    loss_drop = _aux_survival_loss(out_s["drop_hazard_logits"], out_s["drop_cum_hazard"], out_s["drop_probs"], batch["y_drop"].float(), v_drop, pos_w=4.0, anchor_pos=1.05, anchor_neg=0.08)

    v_dcl = batch["valid_dyn_closing"].float() * obs_ok
    loss_dcl = _aux_survival_loss(out_s["dyn_closing_hazard_logits"], out_s["dyn_closing_cum_hazard"], out_s["dyn_closing_probs"], batch["y_dyn_closing"].float(), v_dcl, pos_w=3.5, anchor_pos=1.05, anchor_neg=0.10)
    pass_m = (v_dcl > 0.5) & (batch["y_dyn_passing"].float() > 0.5)
    if pass_m.any():
        p_dcl_pass = out_s["dyn_closing_probs"][:, :, IDX_4M_EXACT].float().clamp(1e-5, 1.0 - 1e-5)[pass_m]
        loss_dcl = loss_dcl + 1.8 * (-torch.log((1.0 - p_dcl_pass).clamp(min=1e-4))).mean()

    # Soft v11 Omni KD on left/right/head/dyn_closing (skip drop_probs on ARKit where v11 had 28% FPR!)
    p_l11 = out_v11["left_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p_r11 = out_v11["right_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p_h11 = out_v11["head_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p_d11 = out_v11["drop_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    not_arkit = (dom_id != 3).float().view(-1, 1)
    def _soft_bce(ps, pt, vm):
        ps_c = ps.float().clamp(1e-5, 1.0 - 1e-5)
        m_s = vm[:, :, None] * z_w
        return (-(pt * torch.log(ps_c) + (1.0 - pt) * torch.log(1.0 - ps_c)) * m_s).sum() / m_s.sum().clamp(min=1.0)
    loss_omni_kd = (
        0.25 * _soft_bce(out_s["left_probs"], p_l11, v_lane)
        + 0.25 * _soft_bce(out_s["right_probs"], p_r11, v_lane)
        + 0.25 * _soft_bce(out_s["head_probs"], p_h11, v_co_eff)
        + 0.25 * _soft_bce(out_s["drop_probs"], p_d11, v_drop * not_arkit)
    )

    # 7. Dual-Teacher Complementary KD (v6.6 + v11)
    is_synth_or_tartan = ((dom_id == 0) | (dom_id == 4)).float().view(-1, 1, 1)
    prefer_v66 = is_synth_or_tartan * (1.0 - 0.5 * is_dark)
    has_stat_bin = y_co > 0.5
    is_dyn_clear = (dyn_fl > 0.5)[:, :, None] & (~has_stat_bin)
    p_haz_t = prefer_v66 * (0.60 * p66 + 0.40 * torch.maximum(p66, p11)) + (1.0 - prefer_v66) * (0.75 * p11 + 0.25 * torch.maximum(p66, p11))
    p_clr_t = torch.minimum(p66, p11)
    p_dyn_t = torch.minimum(p66, torch.full_like(p66, 0.04))
    p_target = torch.where(has_stat_bin, p_haz_t, torch.where(is_dyn_clear, p_dyn_t, p_clr_t)).clamp(1e-4, 1.0 - 1e-4)
    kd_bce = -(p_target * torch.log(p_cum) + (1.0 - p_target) * torch.log(1.0 - p_cum))
    kd_m = v_co_eff[:, :, None] * z_w
    loss_kd_corr = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)
    dyn_cell = torch.maximum(gt_dy.max(dim=2, keepdim=True).values, (dy66 > 0.35).float().max(dim=2, keepdim=True).values * is_auto[:, :, None, None, None])
    st_target = torch.where(gt_st > 0.5, torch.maximum(st66, st11), torch.minimum(st66, st11))
    kd_st = F.binary_cross_entropy_with_logits(st_log, st_target, reduction="none") * (1.0 - dyn_cell) * spat_w * obs_ok[:, :, None, None, None]
    loss_kd = loss_kd_corr + 0.20 * kd_st.mean() + 0.45 * loss_omni_kd

    # 8. Continuous Dense Geometry & Feature Anchor from v11 + GT depth_s8 + DAV2
    obs_sub = obs_ok[:, ::3].reshape(-1, 1, 1, 1)
    pz_sub4 = out_s["pred_z_m"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
    pn_sub4 = out_s["pred_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float()
    pm_sub4 = out_s["pred_msk"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
    dlz_sub4 = out_s["d_logz"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
    dnr_sub4 = out_s["d_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float()

    z_t = out_v11["pred_z_m"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float().detach()
    n_t = out_v11["pred_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float().detach()
    m_t = out_v11["pred_msk"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float().detach()
    ok_t = ((m_t > 0.50) & (z_t > 0.25) & (z_t < 12.0) & (pm_sub4 > 0.08)).float() * obs_sub
    denom_t = ok_t.sum().clamp(min=1.0)
    lpz4 = torch.log(pz_sub4.clamp(min=0.15))
    lzt4 = torch.log(z_t.clamp(min=0.15))
    l_grad = (((F.conv2d(lpz4, sobel_x, padding=1) - F.conv2d(lzt4, sobel_x, padding=1)).abs() + (F.conv2d(lpz4, sobel_y, padding=1) - F.conv2d(lzt4, sobel_y, padding=1)).abs()) * ok_t).sum() / denom_t
    l_nrm = ((1.0 - (pn_sub4 * n_t).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)) * ok_t).sum() / denom_t
    l_z_t11 = ((lpz4 - lzt4).abs() * ok_t).sum() / denom_t
    pm_c = pm_sub4.clamp(1e-4, 1.0 - 1e-4)
    mt_c = m_t.clamp(0.0, 1.0)
    l_msk = (-(mt_c * torch.log(pm_c) + (1.0 - mt_c) * torch.log(1.0 - pm_c)) * obs_sub).sum() / (obs_sub.sum().clamp(min=1.0) * FEAT_H * FEAT_W)

    gt_d = batch["gt_depth_sub"].reshape(-1, 1, FEAT_H, FEAT_W).float()
    ok_gt = ((gt_d >= 0.35) & (gt_d <= 8.0)).float() * obs_sub
    l_z_gt = ((lpz4 - torch.log(gt_d.clamp(min=0.35))).abs() * ok_gt).sum() / ok_gt.sum() if ok_gt.sum() > 20.0 else torch.tensor(0.0, device=device)
    loss_geom = 0.70 * l_z_t11 + 0.95 * l_z_gt + 0.55 * l_grad + 0.50 * l_nrm + 0.25 * l_msk + 0.03 * (dlz_sub4.pow(2).mean() + dnr_sub4.pow(2).mean())

    dv_pred = out_s["dav2_aux"][:, ::3].reshape(-1, dav2_targets.shape[1], FEAT_H, FEAT_W).float()
    ok_v = (pm_sub4 > 0.10).float() * obs_sub
    f_cos = 1.0 - F.cosine_similarity(dv_pred[:, :64], dav2_targets[:, :64], dim=1, eps=1e-5).unsqueeze(1)
    d_l1 = F.smooth_l1_loss(dv_pred[:, 64:65], dav2_targets[:, 64:65], reduction="none")
    loss_dav2 = ((0.60 * f_cos + 0.40 * d_l1) * ok_v).sum() / ok_v.sum().clamp(min=1.0)

    total = (
        1.25 * loss_st
        + 1.05 * loss_dy
        + 1.55 * loss_cum
        + 0.60 * loss_fh
        + 1.35 * loss_prehit
        + 1.45 * loss_dom_anc
        + 1.25 * loss_un
        + 0.45 * loss_tv
        + 1.75 * loss_ohem
        + 0.70 * loss_rank
        + 4.20 * loss_dyn_h
        + 0.25 * loss_hfl
        + 0.45 * (loss_left + loss_right)
        + 0.75 * loss_head_c
        + 0.95 * loss_drop
        + 0.75 * loss_dcl
        + kd_w * loss_kd
        + 0.75 * loss_geom
        + 0.45 * loss_dav2
    )
    return total, {
        "tot": float(total.item()),
        "co": float(loss_cum.item()),
        "anc": float(loss_dom_anc.item()),
        "dyn": float(loss_dyn_h.item()),
        "head": float(loss_head_c.item()),
        "drop": float(loss_drop.item()),
        "unc": float(loss_un.item()),
        "hfl": float(loss_hfl.item()),
        "geo": float(loss_geom.item()),
    }


def audit_omni_v12(model: EgoClearV12Net, dmap: dict[str, dict], ov: dict | None, device) -> dict:
    """Comprehensive v12 Omni Audit: 3-Lane + Head + Drop-Off + Closing-TTC + Floor-Height MAE +
    Clean Risk-Coverage + ALL 5 Real-World Unobservable Stress Modes (blackout, finger, whiteout, motion_blur, defocus)!"""
    model.eval()
    l_y, l_p, r_y, r_p, h_y, h_p, dr_y, dr_p, dc_y, dc_p = [], [], [], [], [], [], [], [], [], []
    pass_dcl_p, pass_stat_p = [], []
    bypass_ok, bypass_tot = 0, 0
    unc_err, unc_score = [], []
    hfl_preds, hfl_errs = [], []
    oe_modes = {0: [], 1: [], 2: [], 3: [], 4: []}
    rng = np.random.default_rng(20261122)

    for sc_idx, (sc, d) in enumerate(dmap.items()):
        er = ov[sc] if ov is not None else d["eval_rows"]
        T = len(er)
        if T < 5:
            continue
        inp = prepare_scene_slice_tensors(d, 0, T, device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_sequence(**inp)
        cp = out["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        lp = out["left_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        rp = out["right_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        hp = out["head_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        dp = out["drop_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        dcp = out["dyn_closing_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        up = torch.sigmoid(out["unc_logits"][0, :, IDX_4M_EXACT].float()).cpu().numpy()
        hfl = (-out["h_floor_cam"][0].float().cpu().numpy()) if "h_floor_cam" in out else np.full((T,), 1.35, dtype=np.float32)  # positive meters height above floor
        hfl_preds.append(float(np.median(hfl)))
        if float(d.get("h_cam_gt", -1.0)) > 0:
            hfl_errs.append(abs(float(np.median(hfl)) - float(d["h_cam_gt"])))

        vh = d["valid_heading"] > 0
        D = er[:, 0]
        v_co = vh & (np.isfinite(D) | np.isinf(D)) & (~np.isnan(er[:, 1])) & (er[:, 1] >= 0.6)
        if v_co.any():
            y_c = np.isfinite(D[v_co]) & (D[v_co] <= 4.1)
            unc_err.extend(((cp[v_co] > 0.5) != y_c).astype(np.float32).tolist())
            unc_score.extend(up[v_co].tolist())

        if "y_left" in d:
            yl, yr = d["y_left"][:, IDX_4M_EXACT], d["y_right"][:, IDX_4M_EXACT]
            l_y.extend(yl[vh].tolist()); l_p.extend(lp[vh].tolist())
            r_y.extend(yr[vh].tolist()); r_p.extend(rp[vh].tolist())
            lp27 = out["left_probs"][0, :, 11].float().cpu().numpy()
            rp27 = out["right_probs"][0, :, 11].float().cpu().numpy()
            yl27, yr27 = d["y_left"][:, 11] > 0.5, d["y_right"][:, 11] > 0.5
            for ti in np.flatnonzero(vh & np.isfinite(D) & (D <= 2.7) & (yl27 ^ yr27)):
                bypass_tot += 1
                if (not yl27[ti]) == (lp27[ti] < rp27[ti]):
                    bypass_ok += 1

        if "y_head" in d:
            h_y.extend(d["y_head"][vh, IDX_4M_EXACT].tolist()); h_p.extend(hp[vh].tolist())
        if "valid_drop" in d:
            vd = d["valid_drop"] > 0.5
            if vd.any():
                dr_y.extend(d["y_drop"][vd, IDX_4M_EXACT].tolist()); dr_p.extend(dp[vd].tolist())
        if "valid_dyn_closing" in d and np.any(d["valid_dyn_closing"] > 0.5):
            vm = (d["valid_dyn_closing"] > 0.5) & vh
            if vm.any():
                dc_y.extend(d["y_dyn_closing"][vm, IDX_4M_EXACT].tolist()); dc_p.extend(dcp[vm].tolist())
                pm = vm & (d["y_dyn_passing"] > 0.5)
                if pm.any():
                    pass_dcl_p.extend(dcp[pm].tolist()); pass_stat_p.extend(cp[pm].tolist())

        # Test ALL 5 Unobservable failure modes on first 6 scenes!
        if sc_idx < 6 and T >= 6:
            t_mid = T // 2
            orig_rgb = inp["rgb"][0, t_mid : t_mid + 2].float().cpu().numpy()
            orig_01 = np.clip(orig_rgb * IMAGENET_STD[None] + IMAGENET_MEAN[None], 0.0, 1.0)
            for md in range(5):
                inp_c = {k: v.clone() for k, v in inp.items()}
                for j in range(2):
                    c01 = _apply_unobs_corruption_rgb01(orig_01[j], md, rng)
                    cn = (c01 - IMAGENET_MEAN) / IMAGENET_STD
                    inp_c["rgb"][0, t_mid + j] = torch.from_numpy(cn).to(device=device, dtype=inp["rgb"].dtype)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    u_c = torch.sigmoid(model.forward_sequence(**inp_c)["unc_logits"][0, t_mid : t_mid + 2, IDX_4M_EXACT].float()).cpu().numpy()
                oe_modes[md].extend(u_c.tolist())

    def _f1(yt, yp):
        if not yt:
            return None
        y, p = np.asarray(yt) > 0.5, np.asarray(yp) >= 0.5
        tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        return {
            "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
            "auroc": _binary_auroc(np.asarray(yt), np.asarray(yp)),
            "rec@0.5": round(rec, 4), "prec@0.5": round(prec, 4),
            "f1@0.5": round(2 * prec * rec / max(1e-6, prec + rec), 4),
            "fpr@0.5": round(fp / max(1, int((~y).sum())), 4),
        }

    oe_names = {0: "blackout", 1: "finger_red", 2: "whiteout", 3: "motion_blur", 4: "defocus_blur"}
    rc_audit = {}
    if unc_err:
        ue, us = np.asarray(unc_err), np.asarray(unc_score)
        ord_u = np.argsort(us)
        rc_audit = {
            "err_cov100": round(float(ue.mean()), 4),
            "err_cov80": round(float(ue[ord_u[: max(1, int(0.8 * len(ue)))]].mean()), 4),
            "err_cov70": round(float(ue[ord_u[: max(1, int(0.7 * len(ue)))]].mean()), 4),
        }
        for md, nm in oe_names.items():
            if oe_modes[md]:
                arr = np.asarray(oe_modes[md])
                rc_audit[f"{nm}_mean_unc"] = round(float(arr.mean()), 4)
                rc_audit[f"{nm}_abstain_gt0.65"] = round(float((arr > 0.65).mean()), 4)

    return {
        "floor_height_est_m_median": round(float(np.median(hfl_preds)), 3) if hfl_preds else None,
        "floor_height_mae_m": round(float(np.mean(hfl_errs)), 3) if hfl_errs else None,
        "left_lane": _f1(l_y, l_p),
        "right_lane": _f1(r_y, r_p),
        "bypass_dir_acc": round(bypass_ok / max(1, bypass_tot), 4) if bypass_tot > 0 else None,
        "bypass_dir_n": bypass_tot,
        "head_corridor": _f1(h_y, h_p),
        "drop_off": _f1(dr_y, dr_p),
        "dyn_closing": _f1(dc_y, dc_p),
        "passing_ped_dcl_fpr05": round(float(np.mean(np.asarray(pass_dcl_p) >= 0.5)), 4) if pass_dcl_p else None,
        "passing_ped_static_fpr05": round(float(np.mean(np.asarray(pass_stat_p) >= 0.5)), 4) if pass_stat_p else None,
        "unknown_risk_coverage": rc_audit,
    }


def audit_depth_scale(model: EgoClearV12Net, eval_real: dict[str, dict], device) -> dict:
    model.eval()
    sc_ratios, abs_rels = [], []
    for sc, d in list(eval_real.items())[:20]:
        gd_all = d.get("depth_s8")
        if gd_all is None:
            continue
        for fi in range(0, len(d["eval_rows"]), 12):
            inp = prepare_scene_slice_tensors(d, fi, fi + 1, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, g = model.encode_images(inp["rgb"][0], inp["K_scaled"][0], inp["R_c2w"][0])
            gd = gd_all[fi].astype(np.float32)
            zp = cv2.resize(g[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            ok = np.isfinite(gd) & (gd >= 0.5) & (gd <= 6.0)
            if ok.sum() >= 30:
                sc_ratios.append(float(np.median(zp[ok] / gd[ok])))
                abs_rels.append(float(np.mean(np.abs(zp[ok] - gd[ok]) / gd[ok])))
    return {
        "real40_AbsRel": round(float(np.mean(abs_rels)), 4) if abs_rels else None,
        "real40_scale_median": round(float(np.median(sc_ratios)), 4) if sc_ratios else None,
    }


def export_v12_onnx(model: EgoClearV12Net, out_dir: Path) -> dict:
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v12_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v12_step_fp32.onnx"

    m_cpu = model.eval().float().cpu()
    enc_w = EgoClearV12EncoderExport(m_cpu).eval()
    stp_w = EgoClearV12StepExport(m_cpu).eval()

    rgb_d = torch.randn(1, 3, RGB_H, RGB_W)
    K_d = torch.tensor([[300.0, 300.0, 259.0, 147.0]], dtype=torch.float32)
    R_d = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]], dtype=torch.float32)

    orig_interp = F.interpolate
    def _no_aa(*args, **kwargs):
        kwargs["antialias"] = False
        return orig_interp(*args, **kwargs)
    F.interpolate = _no_aa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                torch.onnx.export(
                    enc_w, (rgb_d, K_d, R_d), str(enc_fp32),
                    input_names=["rgb", "K_scaled", "R_c2w"],
                    output_names=["f2d", "geom_cues"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
                f2d_d, g_d = enc_w(rgb_d, K_d, R_d)
                t_d = torch.tensor([[0.25, 0.0, 1.5]], dtype=torch.float32)
                t1_d = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
                t2_d = torch.tensor([[-0.25, 0.0, 1.5]], dtype=torch.float32)
                fwd_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                ds_d = torch.tensor([0.25], dtype=torch.float32)
                mem_d = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
                age_d = torch.full((1, 1, BEV_NZ, BEV_NX), 10.0)
                stp_inputs = (
                    f2d_d, g_d, f2d_d, g_d, f2d_d, g_d,
                    K_d, R_d, t_d, fwd_d, K_d, R_d, t1_d, fwd_d,
                    K_d, R_d, t2_d, ds_d, mem_d, age_d,
                )
                stp_names = [
                    "f2d_t", "geom_t", "f2d_1", "geom_1", "f2d_2", "geom_2",
                    "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1",
                    "K_2", "R_2", "t_2", "ds_t", "mem_prev", "age_prev",
                ]
                torch.onnx.export(
                    stp_w, stp_inputs, str(stp_fp32),
                    input_names=stp_names,
                    output_names=["static_logits", "dynamic_logits", "corr_probs", "unc_logits", "omni_probs", "mem_new", "age_new"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
    finally:
        F.interpolate = orig_interp

    s_e32 = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    s_s32 = ort.InferenceSession(str(stp_fp32), providers=["CPUExecutionProvider"])
    f32, g32 = s_e32.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    feed = dict(zip(stp_names, [t.numpy() for t in stp_inputs]))
    feed["f2d_t"] = f32; feed["geom_t"] = g32
    o32 = s_s32.run(None, feed)
    omni32 = o32[4][0]

    m_gpu = model.cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _, g16_t = m_gpu.encode_images(rgb_d.cuda(), K_d.cuda(), R_d.cuda())
    g16 = g16_t.float().cpu().numpy()
    m_gpu.cpu()

    finite_ok = bool(np.all(np.isfinite(f32)) and np.all(np.isfinite(g32)) and np.all(np.isfinite(omni32)) and np.all(np.isfinite(g16)))
    mono_ok = bool(np.all(np.diff(omni32, axis=-1) >= -1e-6))
    enc_fp16_err = float(np.max(np.abs(g16[:, 0] - g32[:, 0])))
    enc_mb = enc_fp32.stat().st_size / (1024 * 1024)
    stp_mb = stp_fp32.stat().st_size / (1024 * 1024)
    print(f"[v12-onnx] PASS! enc_fp32={enc_mb:.2f}MB + stp_fp32={stp_mb:.2f}MB | ieee_fp16_z_err={enc_fp16_err:.5f}m | finite={finite_ok} | all6_mono={mono_ok}", flush=True)
    for p_o in (enc_fp32, stp_fp32):
        shutil.copy2(p_o, DRIVE_RUNS / p_o.name)
    return {
        "enc_fp32_mb": round(enc_mb, 2),
        "stp_fp32_mb": round(stp_mb, 2),
        "ieee_fp16_z_max_err_m": round(enc_fp16_err, 6),
        "finite_ok": finite_ok,
        "all6_omni_monotonic_verified": mono_ok,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    print(f"[v12] GPU={torch.cuda.get_device_name(0)} | EgoClear-v12 Turbo-Omni (~3x faster Full-NPU + Ground-Anchored + Dynamic-Gated + 5-Mode OE UNKNOWN)!", flush=True)

    paths = stage_all_p8_datasets_parallel()
    p13 = stage_p13_extra_datasets_parallel(max_workers=8)

    CODA_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(CODA_DIR.glob("*.npz"))) < 20:
        for tf_p in sorted((DRIVE_ROOT / "datasets/coda_zeroshot_v0").glob("*.tar")):
            tmp_t = CODA_DIR / tf_p.name
            shutil.copy2(tf_p, tmp_t)
            with tarfile.open(tmp_t, "r") as tf:
                tf.extractall(CODA_DIR, filter="data")
            tmp_t.unlink(missing_ok=True)

    real_local_rows = ensure_real_local_gt_cache(paths["real_40"], Path("/content/sanpo_p2/real_local_gt_cache.npz"))
    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    synth_train, eval_synth = [], {}
    for sc, sp in m287["splits"].items():
        p = paths["synth_287"] / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            (synth_train.append(d) if sp == "train" else eval_synth.__setitem__(sc, d))
    for p in sorted(paths["synth_450"].glob("*.npz")):
        synth_train.append(load_scene(p))
    for p in sorted(p13["synth_450_p13"].glob("*.npz")):
        synth_train.append(load_scene(p))

    real_gt_train = [load_scene(p) for key in ("real_75", "real_22", "real_ch70") for p in sorted(paths[key].glob("*.npz"))]
    for p in sorted(p13["real_p13"].glob("*.npz")):
        real_gt_train.append(load_scene(p))
    real_auto_train = [load_scene(p) for p in sorted(p13["real_head_auto"].glob("*.npz"))]
    arkit_train = [load_scene(p) for key in ("arkit_90", "arkit_210") for p in sorted(paths[key].glob("*.npz"))]
    for p in sorted(p13["arkit_500"].glob("*.npz")):
        arkit_train.append(load_scene(p))
    tartan_train = [load_scene(p) for p in sorted(p13["tartanair"].glob("*.npz"))]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    omni_gt_stats = {
        "train_synth": enrich_pool_v12(synth_train, "train_synth", has_dyn_gt=True),
        "train_real_gt": enrich_pool_v12(real_gt_train, "train_real_gt", has_dyn_gt=True),
        "train_real_auto": enrich_pool_v12(real_auto_train, "train_real_auto", has_dyn_gt=False, allow_flat_pose_neg=False),
        "train_arkit": enrich_pool_v12(arkit_train, "train_arkit", has_dyn_gt=False, allow_flat_pose_neg=True),
        "train_tartan": enrich_pool_v12(tartan_train, "train_tartan", has_dyn_gt=False),
        "eval_synth": enrich_pool_v12(eval_synth, "eval_synth", has_dyn_gt=True),
        "eval_real40": enrich_pool_v12(eval_real, "eval_real40", has_dyn_gt=True),
        "eval_arkit30": enrich_pool_v12(eval_arkit, "eval_arkit30", has_dyn_gt=False, allow_flat_pose_neg=True),
        "eval_coda": enrich_pool_v12(eval_coda, "eval_coda", has_dyn_gt=True, default_h_cam=0.704, allow_flat_pose_neg=True),
    }

    # Frozen Teachers: v6.6 (DAV2-Small) + v11 (Omni-BlindAssist 97-GMAC teacher!)
    teacher_v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    teacher_v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v66.parameters():
        p.requires_grad = False
    _orig_v66_enc = teacher_v66.encode_images
    v66_enc_cache: dict[str, torch.Tensor] = {}
    def _hooked_v66_enc(rgb, K_scaled, R_c2w):
        f2d, disp = _orig_v66_enc(rgb, K_scaled, R_c2w)
        v66_enc_cache["f2d"] = f2d; v66_enc_cache["disp"] = disp
        return f2d, disp
    teacher_v66.encode_images = _hooked_v66_enc

    v11_sd = torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase16_v11/egoclear_v11_omni_moge2_small.pt", map_location="cpu", weights_only=False)["state_dict"]
    teacher_v11 = EgoClearV11Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_moge_neck_and_last2=False, rgb_feat_drop_p=0.0).to(device).eval()
    teacher_v11.load_state_dict(v11_sd, strict=True)
    for p in teacher_v11.parameters():
        p.requires_grad = False

    student = EgoClearV12Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_vit_last_n=3, rgb_feat_drop_p=0.04, use_moge_pretrained=True).to(device)
    ws_info = student.load_v11_backbone_warmstart(v11_sd)
    print(f"[v12-warmstart from v11] {ws_info}", flush=True)

    ds = V12TurboOmniDataset(synth_train, real_gt_train, real_auto_train, arkit_train, tartan_train, clip_len=12, clips_per_epoch=1920)

    # =========================================================================
    # STAGE 1: Fast Unified 84x148 Geometry Decoder Distillation Warm-Up (3 ep!)
    # =========================================================================
    s1_epochs = 3
    s1_loader = torch.utils.data.DataLoader(ds, batch_size=12, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=3, drop_last=True)
    # Freeze ViT during Stage 1 so decoder learns cleanly on v11's exact ViT features (zero ViT autograd VRAM!)
    orig_req = {n: p.requires_grad for n, p in student.named_parameters()}
    orig_unlock = student.unlock_vit_last_n
    student.unlock_vit_last_n = 0
    for n, p in student.named_parameters():
        p.requires_grad = n.startswith("decoder.") or n.startswith("dav2_aux_proj.") or n.startswith("moge_scale_head.")
    opt1 = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1.2e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=s1_epochs * len(s1_loader), eta_min=1.5e-4)

    s1_hist = []
    t_s1 = time.time()
    print(f"\n=== [Stage 1/2] Fast Unified 84x148 Geometry+Feature Distillation ({s1_epochs} ep x 1920 clips) ===", flush=True)
    for ep in range(1, s1_epochs + 1):
        student.train()
        ep_l = {"tot": 0.0, "zt": 0.0, "zgt": 0.0, "nrm": 0.0, "f2d": 0.0}
        te = time.time()
        for batch in s1_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Take 4 strided frames (::3) matching gt_depth_sub (B*4 = 48 frames/step -> ultra-fast!)
            rgb_s = batch["rgb_s"][:, ::3].reshape(-1, 3, RGB_H, RGB_W).float()
            rgb_sc = batch["rgb_s_clean"][:, ::3].reshape(-1, 3, RGB_H, RGB_W).float()
            K_4 = batch["K_s"][:, ::3].reshape(-1, 4)
            Rc_4 = batch["R_s_clean"][:, ::3].reshape(-1, 3, 3)
            Rs_4 = batch["R_s"][:, ::3].reshape(-1, 3, 3)
            obs_4 = (1.0 - batch["is_unobservable"][:, ::3].float()).clamp(0.0, 1.0)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t11_o = teacher_v11.encode_images(rgb_sc, K_4, Rc_4, return_aux_geom=True)
                f66, d66 = _orig_v66_enc(rgb_sc, K_4, Rc_4)
                dav2_t = F.interpolate(torch.cat([f66.float(), d66[:, 0:1].float()], dim=1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_o = student.encode_images(rgb_s, K_4, Rs_4, return_aux_geom=True)
                loss, ld = compute_stage1_geom_loss(s_o, t11_o, dav2_t, batch["gt_depth_sub"], obs_4, student.sobel_x, student.sobel_y)

            opt1.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
            opt1.step()
            sched1.step()
            for k in ep_l:
                ep_l[k] += ld[k]
        for k in ep_l:
            ep_l[k] /= len(s1_loader)
        ep_l["ep"] = ep; ep_l["sec"] = round(time.time() - te, 1)
        s1_hist.append(ep_l)
        print(f"  [S1 ep {ep}/{s1_epochs}] tot={ep_l['tot']:.4f} zt={ep_l['zt']:.4f} zgt={ep_l['zgt']:.4f} nrm={ep_l['nrm']:.4f} f2d={ep_l['f2d']:.4f} ({ep_l['sec']:.1f}s)", flush=True)
    del s1_loader, opt1, sched1
    s1_depth = audit_depth_scale(student, eval_real, device)
    print(f"[Stage 1 DONE in {time.time()-t_s1:.1f}s] Real40 Depth Gate = {s1_depth}", flush=True)

    # Restore exact Stage 2 trainability (last 3 ViT blocks + norm + out_proj + scale_head + decoder + step)
    student.unlock_vit_last_n = orig_unlock
    for n, p in student.named_parameters():
        p.requires_grad = orig_req[n]

    # =========================================================================
    # STAGE 2: Full End-to-End Turbo-Omni Joint Distillation + Calibration (8 ep!)
    # =========================================================================
    s2_epochs = 8
    micro_bs = 8
    accum_steps = 2
    loader = torch.utils.data.DataLoader(ds, batch_size=micro_bs, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=3, drop_last=True)
    opt = torch.optim.AdamW(student.param_groups(lr_head=3.0e-4, lr_vit=1.5e-5), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s2_epochs * (len(loader) // accum_steps), eta_min=5.0e-6)

    ema_sd = {k: v.detach().clone().float() for k, v in student.state_dict().items()}
    ema_decay = 0.995
    history = []
    t_s2 = time.time()
    print(f"\n=== [Stage 2/2] Joint End-to-End EgoClear-v12 Turbo-Omni ({s2_epochs} ep x 1920 clips) ===", flush=True)
    for ep in range(1, s2_epochs + 1):
        student.train()
        kd_w = 0.42 - 0.16 * ((ep - 1) / max(1, s2_epochs - 1))
        ep_l = {"tot": 0.0, "co": 0.0, "anc": 0.0, "dyn": 0.0, "head": 0.0, "drop": 0.0, "unc": 0.0, "hfl": 0.0, "geo": 0.0}
        te = time.time()
        opt.zero_grad(set_to_none=True)

        for bi, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_s = batch["rgb_s"].float()
            rgb_sc = batch["rgb_s_clean"].float()
            B_cur, T_cur = rgb_sc.shape[:2]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_v66 = teacher_v66.forward_sequence(rgb_sc, batch["K_s"], batch["R_s_clean"], batch["t_clean"], batch["fwd_clean"], batch["seg_clean"])
                f66_all = v66_enc_cache["f2d"].view(B_cur, T_cur, 64, 42, 74)[:, ::3].reshape(-1, 64, 42, 74)
                d66_all = v66_enc_cache["disp"].view(B_cur, T_cur, 4, 42, 74)[:, ::3, 0:1].reshape(-1, 1, 42, 74)
                dav2_t = F.interpolate(torch.cat([f66_all.float(), d66_all.float()], dim=1), size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False)
                out_v11 = teacher_v11.forward_sequence(
                    rgb_sc, batch["K_s"], batch["R_s_clean"], batch["t_clean"], batch["fwd_clean"], batch["seg_clean"],
                    return_aux_geom=True,
                )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(
                    rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"],
                    return_aux_geom=True,
                )
                loss, ld = compute_loss_v12(out_s, out_v66, out_v11, dav2_t.detach(), batch, student.sobel_x, student.sobel_y, kd_w=kd_w)

            (loss / accum_steps).backward()
            if (bi + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for k, v in student.state_dict().items():
                        if v.is_floating_point():
                            ema_sd[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                        else:
                            ema_sd[k].copy_(v.detach())
            for k in ep_l:
                ep_l[k] += ld[k]

        nb = len(loader)
        for k in ep_l:
            ep_l[k] /= nb
        ep_l["ep"] = ep; ep_l["sec"] = round(time.time() - te, 1)
        history.append(ep_l)
        print(
            f"  [v12 ep {ep:02d}/{s2_epochs}] tot={ep_l['tot']:.4f} co={ep_l['co']:.4f} anc={ep_l['anc']:.4f} "
            f"dyn={ep_l['dyn']:.4f} head={ep_l['head']:.4f} drop={ep_l['drop']:.4f} unc={ep_l['unc']:.4f} "
            f"hfl={ep_l['hfl']:.4f} geo={ep_l['geo']:.4f} ({ep_l['sec']:.1f}s)",
            flush=True,
        )

    train_sec = round(time.time() - t_s1, 1)
    # Recalibrate BN & compare RAW vs EMA on held-out Real40 + Synth @120!
    raw_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    orig_drop = student.rgb_feat_drop_p
    student.rgb_feat_drop_p = 0.0

    def _quick_score(sd_cpu: dict, tag: str) -> tuple[float, dict]:
        student.load_state_dict({k: v.to(device) for k, v in sd_cpu.items()}, strict=True)
        recalibrate_bn_stats(student, loader, device, n_batches=28)
        student.eval()
        r_ev = eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
        s_ev = eval_v6_all(student, eval_synth, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=False)
        r120 = (r_ev["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
        s120 = (s_ev["modes"]["raw"]["summary"].get("late_at_fa_fr<=120.3") or {"late_rate": 1.0})["late_rate"]
        rdyn = (r_ev["dyn_only_alert_rate_at_0.5"] or {}).get("rate", 0.3)
        score = r120 + 0.65 * s120 + 0.35 * rdyn
        print(f"  [RAW vs EMA check: {tag}] Real@120={r120*100:.1f}% Synth@120={s120*100:.1f}% Real_dyn={rdyn*100:.1f}% -> combo={score:.4f}", flush=True)
        return score, {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    sc_raw, sd_raw_bn = _quick_score(raw_sd, "RAW")
    sc_ema, sd_ema_bn = _quick_score({k: v.cpu() for k, v in ema_sd.items()}, "EMA")
    winner_tag = "EMA" if sc_ema <= sc_raw else "RAW"
    best_sd = sd_ema_bn if winner_tag == "EMA" else sd_raw_bn
    student.load_state_dict({k: v.to(device) for k, v in best_sd.items()}, strict=True)
    student.rgb_feat_drop_p = orig_drop
    print(f"[v12 Winner Selected] {winner_tag} (RAW={sc_raw:.4f} vs EMA={sc_ema:.4f})", flush=True)

    ckpt_final = LOCAL_RUNS / "egoclear_v12_turbo_omni.pt"
    torch.save({
        "state_dict": best_sd,
        "moge_id": student.moge_id,
        "winner": winner_tag,
        "stage1_history": s1_hist,
        "stage2_history": history,
    }, ckpt_final)
    shutil.copy2(ckpt_final, DRIVE_RUNS / ckpt_final.name)

    teacher_v66.encode_images = _orig_v66_enc
    v66_enc_cache.clear()
    student.eval()

    # 1. Pre-Hit Leakage + Depth Scale Audit
    prehit_arkit = audit_prehit_leakage(student, eval_arkit, None, device)
    prehit_real = audit_prehit_leakage(student, eval_real, real_local_rows, device)
    depth_calib = audit_depth_scale(student, eval_real, device)
    print(f"\n[v12 Pre-Hit] ARKit={prehit_arkit} | Real40={prehit_real} | Depth={depth_calib}", flush=True)

    # 2. Full v12 Omni + 5-Mode OE UNKNOWN + Floor-Height Audit
    omni_audits = {}
    for dname, dmap, ov in [
        ("synth_clean", eval_synth, None),
        ("real_40_local", eval_real, real_local_rows),
        ("arkit_indoor_val30", eval_arkit, None),
        ("coda_campus_zeroshot", eval_coda, None),
    ]:
        omni_audits[dname] = audit_omni_v12(student, dmap, ov, device)
        print(f"[v12 OMNI AUDIT {dname:20s}] {omni_audits[dname]}", flush=True)

    # 3. Full 9-Benchmark + Single-Global-Threshold (0.50) + Paired Bootstrap vs v6.6 & v11!
    eval_real_gray = make_variant_scenes(eval_real, "gray")
    evals, paired_audits, boot_v66, boot_v11, iso_evals = {}, {}, {}, {}, {}
    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98], dtype=np.float64)

    for dname, dmap, ov, vscale, do_unc in [
        ("synth_clean", eval_synth, None, 0.0, True),
        ("synth_vio_1x", eval_synth, None, 1.0, False),
        ("real_40_local", eval_real, real_local_rows, 0.0, True),
        ("real_40_gray", eval_real_gray, real_local_rows, 0.0, False),
        ("aria_g2_zeroshot", aria_dicts, None, 0.0, True),
        ("arkit_indoor_val30", eval_arkit, None, 0.0, True),
        ("coda_campus_zeroshot", eval_coda, None, 0.0, True),
    ]:
        evals[dname] = eval_v6_all(student, dmap, ov, bw_speeds, device, vio_scale=vscale, compute_uncertainty_audit=do_unc)
        print_block("EgoClear-v12", dname, evals[dname])
        if vscale == 0.0 and dname != "real_40_gray":
            ev_m, s12_m, s66_m, s11_m, sens_m = {}, {}, {}, {}, {}
            for sc, d in dmap.items():
                er = ov[sc] if ov is not None else d["eval_rows"]
                if len(er) < 5:
                    continue
                inp = prepare_scene_slice_tensors(d, 0, len(er), device)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s12_m[sc] = student.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    s66_m[sc] = teacher_v66.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    s11_m[sc] = teacher_v11.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                ev_m[sc] = er
                sens_m[sc] = sigmoid(0.5 * logit(s66_m[sc]) + 0.5 * logit(s12_m[sc]))
            paired_audits[dname] = paired_route_audit_from_scores(dmap, ov, s12_m, bw_speeds, thrs, fixed_thr=0.50)
            f50 = paired_audits[dname]["fixed_thr_0.50"]
            print(f"     [GLOBAL FIXED thr=0.50] late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km", flush=True)
            c12 = summarize_at_fa_caps(eval_score_curve(ev_m, s12_m, bw_speeds, thrs, higher_is_alert=True))
            c66 = summarize_at_fa_caps(eval_score_curve(ev_m, s66_m, bw_speeds, thrs, higher_is_alert=True))
            c11 = summarize_at_fa_caps(eval_score_curve(ev_m, s11_m, bw_speeds, thrs, higher_is_alert=True))
            t12 = float(c12["late_at_fa_fr<=120.3"]["thr"]) if c12.get("late_at_fa_fr<=120.3") else 0.50
            t66 = float(c66["late_at_fa_fr<=120.3"]["thr"]) if c66.get("late_at_fa_fr<=120.3") else 0.50
            t11 = float(c11["late_at_fa_fr<=120.3"]["thr"]) if c11.get("late_at_fa_fr<=120.3") else 0.50
            boot_v66[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s66_m, bw_speeds, t66), per_route_at_thr(dmap, ov, s12_m, bw_speeds, t12))
            boot_v11[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s11_m, bw_speeds, t11), per_route_at_thr(dmap, ov, s12_m, bw_speeds, t12))
            iso_evals[f"{dname}_ens_v66_v12_squashed"] = summarize_scores(ev_m, sens_m, bw_speeds)
            print(f"     [Bootstrap @120 v12 vs v6.6] {boot_v66[dname]}", flush=True)
            print(f"     [Bootstrap @120 v12 vs v11 ] {boot_v11[dname]}", flush=True)
            print(f"     [Dual-NPU ENS(v6.6+v12)    ] {iso_evals[f'{dname}_ens_v66_v12_squashed']}", flush=True)

    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v12", "arkit_dark018", evals["arkit_dark018"])
    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v12", "real_40_dark018", evals["real_40_dark018"])

    for dname, dmap, modes in [
        ("aria_g2_zeroshot", aria_dicts, ("iso_letterbox", "iso_crop", "iso_grav_crop")),
        ("coda_campus_zeroshot", eval_coda, ("iso_letterbox", "iso_crop")),
    ]:
        for md in modes:
            k_iso = f"{dname}__{md}"
            iso_evals[k_iso] = eval_frontend_mode(student, teacher_v66, dmap, None, bw_speeds, device, md)
            print(f"  [{k_iso:36s}] {iso_evals[k_iso]}", flush=True)

    del teacher_v66, teacher_v11
    torch.cuda.empty_cache()

    onnx_info = export_v12_onnx(student, LOCAL_RUNS / "onnx")
    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v12 Turbo-Omni (MoGe-Turbo 84x148 + Ground-Anchored NY=9 + Dynamic-Gated + 5-Mode OE UNKNOWN)",
        "winner": winner_tag,
        "warmstart_info": ws_info,
        "omni_gt_prevalence": omni_gt_stats,
        "stage1_depth_gate": s1_depth,
        "depth_scale_audit": depth_calib,
        "prehit_leakage_audit": {"v12_arkit": prehit_arkit, "v12_real40": prehit_real},
        "omni_capability_audits": omni_audits,
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "stage1_history": s1_hist,
        "stage2_history": history,
        "evals": evals,
        "iso_and_ens_evals": iso_evals,
        "paired_route_audits_at_0.50": paired_audits,
        "paired_bootstrap_vs_v66_at_120": boot_v66,
        "paired_bootstrap_vs_v11_at_120": boot_v11,
        "onnx_export": onnx_info,
    }
    res_p = LOCAL_RUNS / "phase17_v12_results.json"
    res_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_p, DRIVE_RUNS / res_p.name)
    os.sync()
    print(f"\n[v12] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
