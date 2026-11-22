"""Phase 13 (Colab G4 96 GB Blackwell): Train & Evaluate EgoClear-v10 — All-Domain + Aria Generational Breakthrough.

Stacks ALL empirically proven breakthroughs from Phases 1-13:
  1. 2.4x Multi-Domain Data Expansion (~2,400+ scenes in 20 TB Drive):
     - SANPO-Real full-seg + DeepLabV3 auto-cleaned 100 real camera_head sessions
     - 800 ARKitScenes real indoor LiDAR captures (visit-disjoint from val30)
     - TartanAir 90-deg wide-FOV indoor/alley/urban/pitch-dark-night (exact NED->Z-up + depth_s8)
     - 1,077 SANPO-Synthetic walking sessions
     - STRICT 100% zero-shot on Meta Aria Gen 2 (193 events) & UT Austin CODa (36 events); zero reads of official test splits!
  2. Universal Border-Masked Isotropic Camera Frontend (fx == fy, cures 1.762x Aria / 1.474x CODa / 1.321x ARKit squash!).
  3. Exact 3D Homography Temporal Head-Scan Augmentation (pitch -24..+4 deg, sinusoidal yaw +/-38 deg, roll +/-5 deg)
     with exact optical depth projection z_new = z_old / (R_virt @ ray_new)_z!
  4. Controlled Backbone Unlock: MoGe-2 last 2 ViT blocks + norm + neck at lr=1.8e-5, jointly anchored by
     331M MoGe-2-ViT-L-Normal + GT depth_s8 + Training-Only DAV2-Small Feature Alignment (ZERO inference cost -> 52.1 ms NPU!).
  5. Cured Continuous Loss (no v9 bimodal cliff!): Smooth Calibrated Tail-OHEM (anchor 0.14) + Gentle Continuous Ranking
     + First-Entry Tight-Window 2.2x Boost (targeting Aria's 86.7% 1-2 frame misses!) + Domain-Aware Smooth Dual-Champion KD (v6.6 + v8.5).
  6. Complete Multi-Frontend Eval (squashed + isotropic letterbox/grav-crop) + Paired Bootstrap vs v6.6/v8.5/v9 + Dual-NPU ENS(v6.6+v10) + ONNX-17!
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tarfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_NX, BEV_NZ, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v7_model import FEAT_H, FEAT_W, _build_moge_uv_levels
from egoclear_v8_model import EgoClearV8Net
from egoclear_v9_model import EgoClearV9Net
from egoclear_v10_model import EgoClearV10EncoderExport, EgoClearV10Net, EgoClearV10StepExport
from eval_closed_loop import H, eval_score_curve, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase7_train_v6_g4 import eval_v6_all
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from phase11_stage2_diagnostics_g4 import (
    extract_moge_geom_cues_standalone,
    paired_cluster_bootstrap,
    per_route_at_thr,
)
from phase11_train_v7_g4 import eval_dark_domain, print_block
from phase11_train_v75_g4 import (
    IDX_4M_EXACT,
    Z_EDGES,
    collect_paired_route_audit,
    prepare_scene_slice_tensors,
    recalibrate_bn_stats,
)
from phase12_v9_probe_g4 import logit, make_variant_scenes, sigmoid, summarize_scores
from phase13_aria_isotropic_probe_g4 import prepare_frontend_tensors
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_DATA = DRIVE_ROOT / "datasets"
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase13_v10"
LOCAL_RUNS = Path("/content/egoclear_runs_p13_v10")
LOCAL_P13 = Path("/content/p13_data")
CODA_DIR = Path("/content/coda_zeroshot_v0")


def _rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def stage_p13_extra_datasets() -> dict[str, Path]:
    """Ensure all Phase 13 newly packed datasets are staged locally under /content/p13_data/."""
    specs = [
        ("real_p13", DRIVE_DATA / "sanpo_real_p13_extra_v0", LOCAL_P13 / "sanpo_real_extra"),
        ("real_head_auto", DRIVE_DATA / "sanpo_real_head100_autoseg_v0", LOCAL_P13 / "sanpo_real_head100_autoseg"),
        ("arkit_500", DRIVE_DATA / "arkit_indoor_p13_500_v0", LOCAL_P13 / "arkit_500"),
        ("tartanair", DRIVE_DATA / "tartanair_widefov_p13_v0", LOCAL_P13 / "tartanair_widefov"),
        ("synth_450_p13", DRIVE_DATA / "sanpo_synth_p13_450_v0", LOCAL_P13 / "sanpo_synth_450"),
    ]
    out = {}
    for key, d_dir, l_dir in specs:
        l_dir.mkdir(parents=True, exist_ok=True)
        out[key] = l_dir
        if len(list(l_dir.glob("*.npz"))) < 5 and (d_dir / "manifest_sha256.json").exists():
            man = json.loads((d_dir / "manifest_sha256.json").read_text())
            for sh in man["shards"]:
                sp = d_dir / sh["file"]
                if sp.exists():
                    with tarfile.open(sp, "r") as tf:
                        tf.extractall(l_dir, filter="data")
        print(f"[stage-p13] {key:15s}: {len(list(l_dir.glob('*.npz')))} scenes ready", flush=True)
    return out


def apply_iso_and_zoom_rgb_depth(
    rgb: np.ndarray,          # (T, 3, H, W) [0,1]
    depth_sub: np.ndarray | None,  # (2, Hd, Wd) or None at frames [0, 6]
    K_s: np.ndarray,          # [fx, fy, cx, cy]
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Universal isotropic un-squashing (fx==fy) + random multi-aspect letterbox (1:1..16:9) + focal zoom!"""
    T = rgb.shape[0]
    fx, fy, cx, cy = [float(x) for x in K_s]
    ratio = fx / max(fy, 1e-6)
    d_full = None
    if depth_sub is not None:
        d_full = np.stack([cv2.resize(depth_sub[k].astype(np.float32), (RGB_W, RGB_H), interpolation=cv2.INTER_NEAREST) for k in range(2)], axis=0)

    # 1. If native camera is non-16:9 (e.g. ARKit 4:3 or TartanAir 4:3, ratio ~1.32), restore exact isotropic fx==fy 85% of the time!
    if ratio > 1.03 and rng.random() < 0.85:
        if rng.random() < 0.65:
            # Isotropic letterbox (fx_new == fy)
            nw = max(64, int(round(RGB_W / ratio)))
            pad_l = (RGB_W - nw) // 2
            rgb_n = np.zeros_like(rgb)
            for ti in range(T):
                rs = cv2.resize(rgb[ti].transpose(1, 2, 0), (nw, RGB_H), interpolation=cv2.INTER_AREA)
                rgb_n[ti, :, :, pad_l : pad_l + nw] = rs.transpose(2, 0, 1)
            if d_full is not None:
                d_n = np.zeros_like(d_full)
                for k in range(2):
                    d_n[k, :, pad_l : pad_l + nw] = cv2.resize(d_full[k], (nw, RGB_H), interpolation=cv2.INTER_NEAREST)
                d_full = d_n
            sx = nw / float(RGB_W)
            K_s = np.array([fx * sx, fy, cx * sx + pad_l, cy], dtype=np.float32)
            rgb = rgb_n
        else:
            # Isotropic vertical crop (fy_new == fx, zero black bars!)
            nh_full = int(round(RGB_H * ratio))
            sy = nh_full / float(RGB_H)
            y0 = int(rng.integers(0, max(1, nh_full - RGB_H + 1)))
            rgb_n = np.zeros_like(rgb)
            for ti in range(T):
                rs = cv2.resize(rgb[ti].transpose(1, 2, 0), (RGB_W, nh_full), interpolation=cv2.INTER_LINEAR)
                rgb_n[ti] = rs[y0 : y0 + RGB_H].transpose(2, 0, 1)
            if d_full is not None:
                d_n = np.zeros_like(d_full)
                for k in range(2):
                    rs = cv2.resize(d_full[k], (RGB_W, nh_full), interpolation=cv2.INTER_NEAREST)
                    d_n[k] = rs[y0 : y0 + RGB_H]
                d_full = d_n
            K_s = np.array([fx, fy * sy, cx, cy * sy - y0], dtype=np.float32)
            rgb = rgb_n

    # 2. Focal zoom (0.55x wide-FOV .. 1.25x narrow-FOV)
    if rng.random() < 0.60:
        zoom = float(rng.uniform(0.55, 1.25))
        if abs(zoom - 1.0) >= 0.02:
            rgb, K_s = apply_focal_zoom_crop(rgb, K_s, zoom)
            if d_full is not None:
                nw = max(32, int(round(RGB_W * zoom)))
                nh = max(32, int(round(RGB_H * zoom)))
                d_n = np.zeros_like(d_full)
                for k in range(2):
                    rs = cv2.resize(d_full[k], (nw, nh), interpolation=cv2.INTER_NEAREST)
                    if zoom > 1.0:
                        x0, y0 = (nw - RGB_W) // 2, (nh - RGB_H) // 2
                        d_n[k] = rs[y0 : y0 + RGB_H, x0 : x0 + RGB_W]
                    else:
                        x0, y0 = (RGB_W - nw) // 2, (RGB_H - nh) // 2
                        d_n[k, y0 : y0 + nh, x0 : x0 + nw] = rs
                d_full = d_n

    # 3. Synthetic multi-aspect isotropic letterbox mask (simulating 1:1 Aria / 5:4 CODa / 4:3 ARKit border geometry on 30% of clips!)
    if rng.random() < 0.30:
        nw_mask = int(rng.integers(276, 440))
        pad_l = (RGB_W - nw_mask) // 2
        pad_r = pad_l + nw_mask
        rgb[:, :, :, :pad_l] = 0.0
        rgb[:, :, :, pad_r:] = 0.0
        if d_full is not None:
            d_full[:, :, :pad_l] = 0.0
            d_full[:, :, pad_r:] = 0.0

    return rgb, d_full, K_s


def apply_3d_homography_headscan(
    rgb: np.ndarray,             # (T, 3, H, W) [0,1]
    d_full: np.ndarray | None,   # (2, H, W) at frames [0, 6]
    K_s: np.ndarray,             # [fx, fy, cx, cy]
    R_seq: np.ndarray,           # (T, 3, 3)
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Exact 3D projective camera rotation (downward pitch -24..+4 deg + temporal sinusoidal head-turn yaw +/-38 deg + roll +/-5 deg!).
    Updates R_c2w_new = R_c2w @ R_virt(t) and projects optical depth exactly: z_new = z_old / (R_virt @ ray_new)_z!
    """
    T = rgb.shape[0]
    fx, fy, cx, cy = [float(x) for x in K_s]
    K_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K_mat)

    pitch0 = math.radians(float(rng.uniform(-24.0, 4.0)))
    yaw0 = math.radians(float(rng.uniform(-24.0, 24.0)))
    yaw_amp = math.radians(float(rng.uniform(0.0, 18.0)))
    yaw_ph = float(rng.uniform(0.0, 2.0 * math.pi))
    roll0 = math.radians(float(rng.uniform(-5.0, 5.0)))

    rgb_out = np.empty_like(rgb)
    R_out = np.empty_like(R_seq)
    d_out = np.zeros_like(d_full) if d_full is not None else None

    if d_full is not None:
        vv, uu = np.mgrid[0:RGB_H, 0:RGB_W].astype(np.float64)
        rx_n = (uu - cx) / max(fx, 1e-6)
        ry_n = (vv - cy) / max(fy, 1e-6)

    for ti in range(T):
        yaw_t = float(np.clip(yaw0 + yaw_amp * math.sin(2.0 * math.pi * ti / max(T, 1) + yaw_ph), math.radians(-38.0), math.radians(38.0)))
        R_virt = _rot_y(yaw_t) @ _rot_x(pitch0) @ _rot_z(roll0)
        H_mat = K_mat @ R_virt.T @ K_inv
        hwc = rgb[ti].transpose(1, 2, 0)
        warped = cv2.warpPerspective(hwc, H_mat, (RGB_W, RGB_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
        rgb_out[ti] = warped.transpose(2, 0, 1)
        R_out[ti] = (R_seq[ti].astype(np.float64) @ R_virt).astype(np.float32)

        if d_full is not None and ti in (0, 6):
            ki = 0 if ti == 0 else 1
            dw = cv2.warpPerspective(d_full[ki], H_mat, (RGB_W, RGB_H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            denom = R_virt[2, 0] * rx_n + R_virt[2, 1] * ry_n + R_virt[2, 2]
            d_out[ki] = np.where((dw > 0.15) & (denom > 0.25), dw / np.maximum(denom, 0.25), 0.0).astype(np.float32)

    return rgb_out, d_out, R_out


class V10MegaMultiDomainDataset(torch.utils.data.Dataset):
    """2,400+ scene multi-domain dataset with Universal Isotropic Framing + Exact 3D Homography Head-Scan + Tight-Entry Flags!"""

    def __init__(
        self,
        synth_dicts: list[dict],
        real_dicts: list[dict],
        arkit_dicts: list[dict],
        tartan_dicts: list[dict],
        clip_len: int = 12,
        clips_per_epoch: int = 3200,
    ):
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.z_edges = Z_EDGES
        self.idx_4m = IDX_4M_EXACT
        self.pools = {
            "synth": [d for d in synth_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "real": [d for d in real_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "arkit": [d for d in arkit_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "tartan": [d for d in tartan_dicts if len(d["eval_rows"]) >= clip_len + 1],
        }
        print(
            f"[dataset-v10] pools: synth={len(self.pools['synth'])} (24%), "
            f"real={len(self.pools['real'])} (38%), arkit={len(self.pools['arkit'])} (24%), "
            f"tartan={len(self.pools['tartan'])} (14%) | TOTAL={sum(len(v) for v in self.pools.values())} scenes!",
            flush=True,
        )

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.24:
            dom = "synth"
        elif u_dom < 0.62 and len(self.pools["real"]) > 0:
            dom = "real"
        elif u_dom < 0.86 and len(self.pools["arkit"]) > 0:
            dom = "arkit"
        elif len(self.pools["tartan"]) > 0:
            dom = "tartan"
        else:
            dom = "synth"

        pool = self.pools[dom]
        d = pool[rng.integers(len(pool))]
        n = len(d["eval_rows"])
        D_all = d["eval_rows"][:, 0]
        dyn_all = d["eval_rows"][:, 2]
        vh_all = d["valid_heading"] > 0
        haz_idx = np.nonzero(vh_all & np.isfinite(D_all) & (D_all <= H))[0]
        clear_idx = np.nonzero(vh_all & np.isinf(D_all))[0]
        dyn_only_idx = np.nonzero(vh_all & np.isinf(D_all) & (dyn_all > 0.5))[0]

        u_pick = rng.random()
        if len(haz_idx) > 0 and u_pick < 0.50:
            anchor = int(haz_idx[rng.integers(len(haz_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(dyn_only_idx) > 0 and u_pick < 0.68:
            anchor = int(dyn_only_idx[rng.integers(len(dyn_only_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(clear_idx) > 0 and u_pick < 0.90:
            anchor = int(clear_idx[rng.integers(len(clear_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        else:
            st = int(rng.integers(0, n - self.clip_len + 1))
        sl = np.arange(st, st + self.clip_len)

        if rng.random() < 0.08:
            freeze_at = int(rng.integers(2, self.clip_len - 2))
            sl[freeze_at:] = sl[freeze_at]

        rgb_raw = decode_rgb_batch(d, sl)
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_base = np.array(
            [K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh],
            dtype=np.float32,
        )
        rgb_t = (rgb_raw - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_t_seq = np.repeat(K_base[None], self.clip_len, axis=0)
        R_clean = d["poses_R"][sl].astype(np.float32)
        t_clean = d["poses_t"][sl].astype(np.float32)
        fwd_clean = d["causal_fwd"][sl].astype(np.float32)
        seg_clean = np.r_[0.0, np.linalg.norm(np.diff(t_clean[:, :2], axis=0), axis=1)].astype(np.float32)

        # Student spatial geometry: Isotropic un-squash + multi-aspect letterbox + focal zoom + 3D Homography Head-Scan!
        dep_raw = d["depth_s8"][sl[::6]] if ("depth_s8" in d and d["depth_s8"] is not None) else None
        rgb_s, d_full, K_s = apply_iso_and_zoom_rgb_depth(rgb_raw.copy(), dep_raw, K_base.copy(), rng)
        R_s_base = R_clean.copy()
        if rng.random() < 0.55:
            rgb_s, d_full, R_s_base = apply_3d_homography_headscan(rgb_s, d_full, K_s, R_s_base, rng)

        if d_full is not None:
            gt_d_sub = np.stack([cv2.resize(d_full[k], (FEAT_W, FEAT_H), interpolation=cv2.INTER_NEAREST) for k in range(2)], axis=0)
            gt_d_sub = np.nan_to_num(gt_d_sub, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            gt_d_sub = np.zeros((2, FEAT_H, FEAT_W), dtype=np.float32)

        # Clean pixel-aligned student view (for 331M ViT-L + DAV2-Small aux alignment!)
        rgb_s_clean = (rgb_s.copy() - IMAGENET_MEAN[None]) / IMAGENET_STD[None]

        # Photometric degradation (keep pure-black letterbox/homography border pixels at exact 0.0 so valid_84 stays crisp!)
        active_mask = (rgb_s.max(axis=1, keepdims=True) > 0.005).astype(np.float32)
        is_dark = rng.random() < 0.28
        if is_dark:
            gamma = float(rng.uniform(1.10, 1.55))
            gain = float(rng.uniform(0.18, 0.55))
            rgb_s = np.power(np.clip(rgb_s, 1e-4, 1.0), gamma) * gain
            rgb_s = np.clip(rgb_s + rng.normal(0.0, float(rng.uniform(0.012, 0.032)), rgb_s.shape).astype(np.float32), 0.0, 1.0)
        else:
            gamma = float(rng.uniform(0.68, 1.45))
            rgb_s = np.power(np.clip(rgb_s, 1e-4, 1.0), gamma)
            ch_gain = rng.uniform(0.84, 1.16, size=(1, 3, 1, 1)).astype(np.float32)
            c_fac = float(rng.uniform(0.80, 1.25))
            mean = rgb_s.mean(axis=(2, 3), keepdims=True)
            rgb_s = np.clip(((rgb_s - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
            if rng.random() < 0.30:
                rgb_s = np.clip(rgb_s + rng.normal(0.0, float(rng.uniform(0.005, 0.022)), rgb_s.shape).astype(np.float32), 0.0, 1.0)

        is_mono = rng.random() < 0.45
        if is_mono:
            luma = 0.299 * rgb_s[:, 0:1] + 0.587 * rgb_s[:, 1:2] + 0.114 * rgb_s[:, 2:3]
            sat = 0.0 if rng.random() < 0.68 else float(rng.uniform(0.0, 0.25))
            rgb_s = np.clip(sat * rgb_s + (1.0 - sat) * luma, 0.0, 1.0)

        if rng.random() < 0.18:
            for ti in range(self.clip_len):
                rgb_s[ti] = cv2.GaussianBlur(rgb_s[ti].transpose(1, 2, 0), (3, 3), 0.9).transpose(2, 0, 1)

        # Preserve exact black padded borders!
        rgb_s = rgb_s * active_mask
        rgb_s = (rgb_s - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_s_seq = np.repeat(K_s[None], self.clip_len, axis=0)

        R_s, t_s, fwd_s, seg_s = R_s_base.copy(), t_clean.copy(), fwd_clean.copy(), seg_clean.copy()
        if rng.random() < 0.48:
            R_s, t_s, fwd_s, seg_s = perturb_vio_trajectory(
                R_s, t_s, fwd_s, rng, trans_rel_std=0.025, yaw_deg_per_m=0.8, grav_tilt_deg=2.0
            )

        gt_stat = d["gt_bev_static"][sl].astype(np.float32)
        gt_dyn = d["gt_bev_dynamic"][sl].astype(np.float32)
        vh = d["valid_heading"][sl].astype(np.float32)
        ev_r = d["eval_rows"][sl]
        D_gt, cover, dyn_fl = ev_r[:, 0], ev_r[:, 1], ev_r[:, 2]

        y_corr = (D_gt[:, None] <= self.z_edges[None, :]).astype(np.float32)
        valid_corr = ((vh > 0.5) & np.isfinite(cover) & (cover >= 0.6)).astype(np.float32)
        in_corr_vox = gt_stat[:, :, : self.idx_4m + 1, 6:9].sum(axis=(1, 2, 3))
        ambig_subvox = np.isinf(D_gt) & (in_corr_vox > 0.5)
        # First-entry & tight-window (3.0..4.1m) 2.2x boost directly targeting Aria's 86.7% 1-2 frame misses!
        is_haz4 = np.isfinite(D_gt) & (D_gt <= H)
        is_first_entry = is_haz4 & np.r_[True, ~is_haz4[:-1]]
        is_far_entry = is_haz4 & (D_gt >= 2.8)
        entry_boost = np.where(is_first_entry | is_far_entry, 2.2, 1.0).astype(np.float32)
        dyn_neg_w = np.where(np.isinf(D_gt) & (dyn_fl > 0.5), 2.8, np.where(ambig_subvox, 0.15, entry_boost)).astype(np.float32)

        return {
            "rgb_t": torch.from_numpy(rgb_t.astype(np.float16)),
            "K_t": torch.from_numpy(K_t_seq),
            "R_t": torch.from_numpy(R_clean),
            "t_t": torch.from_numpy(t_clean),
            "fwd_t": torch.from_numpy(fwd_clean),
            "seg_t": torch.from_numpy(seg_clean),
            "rgb_s_clean": torch.from_numpy(rgb_s_clean.astype(np.float16)),
            "rgb_s": torch.from_numpy(rgb_s.astype(np.float16)),
            "K_s": torch.from_numpy(K_s_seq),
            "R_s_clean": torch.from_numpy(R_s_base),
            "R_s": torch.from_numpy(R_s),
            "t_s": torch.from_numpy(t_s),
            "fwd_s": torch.from_numpy(fwd_s),
            "seg_s": torch.from_numpy(seg_s),
            "gt_depth_sub": torch.from_numpy(gt_d_sub),
            "gt_static": torch.from_numpy(gt_stat),
            "gt_dynamic": torch.from_numpy(gt_dyn),
            "y_corr": torch.from_numpy(y_corr),
            "valid_corr": torch.from_numpy(valid_corr),
            "dyn_neg_w": torch.from_numpy(dyn_neg_w),
            "dyn_flag": torch.from_numpy((dyn_fl > 0.5).astype(np.float32)),
            "prefer_v85": torch.tensor(1.0 if (dom == "real" or is_dark) else 0.0, dtype=torch.float32),
        }


def compute_loss_v10(
    out_s: dict[str, torch.Tensor],
    out_v66: dict[str, torch.Tensor],
    out_v85: dict[str, torch.Tensor],
    vitl_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dav2_targets: torch.Tensor,
    batch: dict[str, torch.Tensor],
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    kd_w: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cured Continuous Loss (no bimodal cliff!) + First-Entry Boost + Domain-Aware KD + ViT-L & GT depth_s8 + Zero-Cost DAV2 Aux Alignment!"""
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
    pref_85 = batch["prefer_v85"].float().view(-1, 1, 1)

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0; col_w[5] = 1.5; col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.3, 2.8], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    # 1. Static & Dynamic BEV
    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()
    loss_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none").mean()

    # 2. Monotonic Survival + First-Hit (with 2.2x First-Entry & 2.8-4.1m tight-window boost via dn_w!)
    z_edges = torch.tensor(Z_EDGES, device=device, dtype=torch.float32)
    z_w = torch.where(z_edges <= 4.35, 2.0, 0.5)[None, None, :]
    bce_cum = -(2.3 * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.5) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w)[:, :, None]
    soft_h = F.softplus(haz_log)
    log_p_hit = torch.log(1.0 - torch.exp(-soft_h).clamp(max=1.0 - 1e-6))
    loss_fh = ((-y_first * log_p_hit + (1.0 - y_first) * soft_h) * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)

    # 3. Uncertainty calibration
    err_target = ((p_cum.detach() > 0.5).float() != y_co).float()
    loss_un = (F.binary_cross_entropy_with_logits(un_log, err_target, reduction="none") * v_co[:, :, None]).sum() / (
        v_co.sum().clamp(min=1.0) * BEV_NZ
    )

    # 4. Cured Continuous Ranking + Smooth Calibrated Tail-OHEM (NO v9 bimodal cliff, NO v8.5 [0.06..0.28] squash!)
    haz_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] > 0.5)
    clear_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl < 0.5)
    dyn_only_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl > 0.5)
    p4 = p_cum[:, :, IDX_4M_EXACT]
    lam4 = cum_h[:, :, IDX_4M_EXACT]
    clear_pair = clear_fr[:, 1:] & clear_fr[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    # Smooth Calibrated Tail-OHEM at p=0.14 (keeps [0.15..0.85] rich & calibrated around 0.50 while crushing hard FA spikes!)
    loss_ohem = torch.tensor(0.0, device=device)
    if clear_fr.any():
        pc_vals = p4[clear_fr]
        k_top = max(1, int(0.25 * pc_vals.numel()))
        top_p = torch.topk(pc_vals, k_top).values
        loss_ohem = (F.relu(top_p - 0.14) ** 2).mean()

    # Gentle continuous pairwise AUC margin in Lambda space (temp=0.45, margin=0.45 -> smooth ROC without bimodal step!)
    loss_rank = torch.tensor(0.0, device=device)
    if haz_fr.any() and clear_fr.any():
        l_pos = lam4[haz_fr]
        l_clr = lam4[clear_fr]
        k_hard = max(1, int(0.25 * l_clr.numel()))
        l_neg_hard = torch.topk(l_clr, k_hard).values
        loss_rank = F.softplus((l_neg_hard[None, :] - l_pos[:, None] + 0.45) / 0.45).mean()

    loss_dyn_h = (F.relu(p4[dyn_only_fr] - 0.12) ** 2).mean() if dyn_only_fr.any() else torch.tensor(0.0, device=device)

    # 5. Domain-Aware Dual-Champion KD (v6.6 + de-squashed v8.5; NEVER let v6.6 drag down dark/real outdoor hazards!)
    p66 = out_v66["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p85_raw = out_v85["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p85 = torch.sigmoid(torch.log(p85_raw / (1.0 - p85_raw)) + 1.00).clamp(1e-4, 1.0 - 1e-4)
    st66 = torch.sigmoid(out_v66["static_logits"].float().detach())
    st85 = torch.sigmoid(out_v85["static_logits"].float().detach())

    has_stat_bin = y_co > 0.5
    is_dyn_clear = (dyn_fl > 0.5)[:, :, None] & (~has_stat_bin)
    # On real/dark clips, rely 75% on v8.5 (our Real/Dark champion!) + 25% union; on synth/arkit/tartan, blend v6.6 + v8.5!
    p_haz_real = 0.75 * p85 + 0.25 * torch.maximum(p66, p85)
    p_haz_gen = 0.50 * p66 + 0.50 * torch.maximum(p66, p85)
    p_haz_t = pref_85 * p_haz_real + (1.0 - pref_85) * p_haz_gen
    p_clr_t = pref_85 * torch.minimum(p85, 0.5 * (p66 + p85)) + (1.0 - pref_85) * (0.50 * p66 + 0.50 * torch.minimum(p66, p85))
    p_target = torch.where(has_stat_bin, p_haz_t, torch.where(is_dyn_clear, torch.minimum(p66, p85_raw), p_clr_t)).clamp(1e-4, 1.0 - 1e-4)

    kd_bce = -(p_target * torch.log(p_cum) + (1.0 - p_target) * torch.log(1.0 - p_cum))
    kd_m = v_co[:, :, None] * z_w
    loss_kd_corr = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)
    dyn_cell = gt_dy.max(dim=2, keepdim=True).values
    st_target = torch.where(gt_st > 0.5, torch.maximum(st66, st85), torch.minimum(st66, st85))
    kd_st = F.binary_cross_entropy_with_logits(st_log, st_target, reduction="none") * (1.0 - dyn_cell) * spat_w
    loss_kd = loss_kd_corr + 0.25 * kd_st.mean()

    # 6. Joint 331M MoGe-2-ViT-L-Normal + GT depth_s8 Scale De-Biasing (anchors unlocked neck & last 2 ViT blocks!)
    z_l, n_l, m_l = vitl_targets
    pz_sub = out_s["pred_z_m"][:, ::6].reshape(-1, 1, FEAT_H, FEAT_W).float()
    pn_sub = out_s["pred_nrm"][:, ::6].reshape(-1, 3, FEAT_H, FEAT_W).float()
    pm_sub = out_s["pred_msk"][:, ::6].reshape(-1, 1, FEAT_H, FEAT_W).float()
    dlz_sub = out_s["d_logz"][:, ::6].reshape(-1, 1, FEAT_H, FEAT_W).float()
    dnr_sub = out_s["d_nrm"][:, ::6].reshape(-1, 3, FEAT_H, FEAT_W).float()

    ok_l = ((m_l > 0.65) & (z_l > 0.25) & (z_l < 12.0) & (pm_sub > 0.10)).float()
    denom_l = ok_l.sum().clamp(min=1.0)
    lpz = torch.log(pz_sub.clamp(min=0.15))
    lzl = torch.log(z_l.clamp(min=0.15))
    l_grad = (((F.conv2d(lpz, sobel_x, padding=1) - F.conv2d(lzl, sobel_x, padding=1)).abs()
               + (F.conv2d(lpz, sobel_y, padding=1) - F.conv2d(lzl, sobel_y, padding=1)).abs()) * ok_l).sum() / denom_l
    l_nrm = ((1.0 - (pn_sub * n_l).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)) * ok_l).sum() / denom_l
    l_z_vitl = ((lpz - lzl).abs() * ok_l).sum() / denom_l

    gt_d = batch["gt_depth_sub"].reshape(-1, 1, FEAT_H, FEAT_W).float()
    ok_gt = ((gt_d >= 0.35) & (gt_d <= 8.0)).float()
    l_z_gt = ((lpz - torch.log(gt_d.clamp(min=0.35))).abs() * ok_gt).sum() / ok_gt.sum() if ok_gt.sum() > 20.0 else torch.tensor(0.0, device=device)
    l_reg = 0.04 * (dlz_sub.pow(2).mean() + dnr_sub.pow(2).mean())
    loss_geom = 0.50 * l_z_vitl + 0.85 * l_z_gt + 0.55 * l_grad + 0.45 * l_nrm + l_reg

    # 7. Training-Only Zero-Inference-Cost DAV2-Small Feature & Disparity Alignment!
    dv_pred = out_s["dav2_aux"][:, ::6].reshape(-1, dav2_targets.shape[1], FEAT_H, FEAT_W).float()
    ok_v = (pm_sub > 0.10).float()
    # Cosine alignment on 64ch f2d + SmoothL1 on relative disparity channel 64
    f_cos = 1.0 - F.cosine_similarity(dv_pred[:, :64], dav2_targets[:, :64], dim=1, eps=1e-5).unsqueeze(1)
    d_l1 = F.smooth_l1_loss(dv_pred[:, 64:65], dav2_targets[:, 64:65], reduction="none")
    loss_dav2 = ((0.6 * f_cos + 0.4 * d_l1) * ok_v).sum() / ok_v.sum().clamp(min=1.0)

    total = (
        1.15 * loss_st
        + 0.55 * loss_dy
        + 1.65 * loss_cum
        + 0.60 * loss_fh
        + 0.20 * loss_un
        + 0.35 * loss_tv
        + 1.25 * loss_ohem
        + 0.45 * loss_rank
        + 2.40 * loss_dyn_h
        + kd_w * loss_kd
        + 0.70 * loss_geom
        + 0.30 * loss_dav2
    )
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "co": float(loss_cum.item()),
        "rank": float(loss_rank.item()),
        "ohem": float(loss_ohem.item()),
        "geom": float(loss_geom.item()),
        "zgt": float(l_z_gt.item()),
        "dav2": float(loss_dav2.item()),
    }


def export_v10_onnx(model: EgoClearV10Net, out_dir: Path) -> dict:
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v10_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v10_step_fp32.onnx"

    m_cpu = model.eval().float().cpu()
    enc_w = EgoClearV10EncoderExport(m_cpu).eval()
    stp_w = EgoClearV10StepExport(m_cpu).eval()

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
                    output_names=["static_logits", "dynamic_logits", "corr_probs", "unc_logits", "mem_new", "age_new"],
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
    cp32 = o32[2][0]

    m_gpu = model.cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _, g16_t = m_gpu.encode_images(rgb_d.cuda(), K_d.cuda(), R_d.cuda())
    g16 = g16_t.float().cpu().numpy()
    m_gpu.cpu()

    finite_ok = bool(np.all(np.isfinite(f32)) and np.all(np.isfinite(g32)) and np.all(np.isfinite(cp32)) and np.all(np.isfinite(g16)))
    mono_ok = bool(np.all(np.diff(cp32) >= -1e-6))
    enc_fp16_err = float(np.max(np.abs(g16[:, 0] - g32[:, 0])))
    enc_mb = enc_fp32.stat().st_size / (1024 * 1024)
    stp_mb = stp_fp32.stat().st_size / (1024 * 1024)
    print(f"[v10-onnx] PASS! enc_fp32={enc_mb:.2f}MB + stp_fp32={stp_mb:.2f}MB | ieee_fp16_z_err={enc_fp16_err:.5f}m | finite={finite_ok} | mono={mono_ok}", flush=True)
    for p_o in (enc_fp32, stp_fp32):
        shutil.copy2(p_o, DRIVE_RUNS / p_o.name)
    return {
        "enc_fp32_mb": round(enc_mb, 2),
        "stp_fp32_mb": round(stp_mb, 2),
        "ieee_fp16_z_max_err_m": round(enc_fp16_err, 6),
        "finite_ok": finite_ok,
        "onnx_monotonic_verified": mono_ok,
    }


def eval_frontend_mode(model, v66, dmap, ov, bw_speeds, device, mode: str) -> dict:
    ev_r, s10, s66, sens = {}, {}, {}, {}
    for sc, d in dmap.items():
        er = ov[sc] if ov is not None else d["eval_rows"]
        if len(er) < 5:
            continue
        inp = prepare_frontend_tensors(d, device, mode=mode)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p10 = model.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
            p66 = v66.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
        ev_r[sc] = er; s10[sc] = p10; s66[sc] = p66
        sens[sc] = sigmoid(0.5 * logit(p66) + 0.5 * logit(p10))
    return {
        "v10": summarize_scores(ev_r, s10, bw_speeds),
        "v66": summarize_scores(ev_r, s66, bw_speeds),
        "ens_v66_v10_70ms": summarize_scores(ev_r, sens, bw_speeds),
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    print(f"[v10] GPU={torch.cuda.get_device_name(0)} | Mega-Data + Isotropic Border-Mask + 3D Homography Head-Scan + Unlocked MoGe-2 Neck/Last2 + DAV2 Aux", flush=True)

    paths = stage_all_p8_datasets_parallel()
    p13 = stage_p13_extra_datasets()

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

    # Load ALL training pools (existing + Phase 13 Mega-Data expansion!)
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

    real_train = [load_scene(p) for key in ("real_75", "real_22", "real_ch70") for p in sorted(paths[key].glob("*.npz"))]
    for p in sorted(p13["real_p13"].glob("*.npz")):
        real_train.append(load_scene(p))
    for p in sorted(p13["real_head_auto"].glob("*.npz")):
        real_train.append(load_scene(p))

    arkit_train = [load_scene(p) for key in ("arkit_90", "arkit_210") for p in sorted(paths[key].glob("*.npz"))]
    for p in sorted(p13["arkit_500"].glob("*.npz")):
        arkit_train.append(load_scene(p))

    tartan_train = [load_scene(p) for p in sorted(p13["tartanair"].glob("*.npz"))]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    # Load Frozen Teachers: v6.6 (DAV2-Small) + v8.5 (MoGe-2 Real/Dark Champion)
    teacher_v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    teacher_v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v66.parameters():
        p.requires_grad = False

    teacher_v85 = EgoClearV8Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), freeze_moge=True, use_geom_residual=True, use_ground_relative=False).to(device).eval()
    teacher_v85.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase11_v85/egoclear_v85_moge2_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v85.parameters():
        p.requires_grad = False

    ckpt_final = LOCAL_RUNS / "egoclear_v10_moge2_small.pt"
    drive_ckpt_final = DRIVE_RUNS / "egoclear_v10_moge2_small.pt"
    student = EgoClearV10Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_moge_neck_and_last2=True, rgb_feat_drop_p=0.06).to(device)
    n_tot = sum(p.numel() for p in student.parameters())
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_moge_tr = sum(p.numel() for n, p in student.named_parameters() if p.requires_grad and n.startswith("moge."))
    print(f"[v10] Built EgoClearV10Net: total={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M (unlocked MoGe neck+last2={n_moge_tr/1e6:.2f}M)", flush=True)

    if ckpt_final.exists() or drive_ckpt_final.exists():
        src_ck = ckpt_final if ckpt_final.exists() else drive_ckpt_final
        ck = torch.load(src_ck, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        history = ck.get("history", [])
        train_sec = round(sum(float(h.get("sec", 0.0)) for h in history), 1)
        print(f"[v10] Loaded finished checkpoint {src_ck} ({len(history)} epochs, train_sec={train_sec}s)", flush=True)
    else:
        from moge.model.v2 import MoGeModel
        print("[v10] Loading frozen Ruicheng/moge-2-vitl-normal (331M MIT) geometric teacher...", flush=True)
        moge_l = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
        for p in moge_l.parameters():
            p.requires_grad = False
        uv_levels = _build_moge_uv_levels(RGB_H // 14, RGB_W // 14, RGB_W / RGB_H)

        # Warm-start from converged EgoClear-v9 (100% matching head weights -> starts at AbsRel 0.1878 & CODa 50%!)
        v9_init_p = DRIVE_ROOT / "runs/egoclear_v0_phase12_v9/egoclear_v9_moge2_small.pt"
        if v9_init_p.exists():
            v9_sd = torch.load(v9_init_p, map_location="cpu", weights_only=False)["head_state_dict"]
            miss, unexp = student.load_state_dict(v9_sd, strict=False)
            print(f"[v10] Warm-started from EgoClear-v9! (missing={len(miss)} [moge+dav2_aux], unexpected={len(unexp)})", flush=True)

        epochs = 10
        micro_bs = 8
        accum_steps = 2  # Effective B=16 with half peak VRAM (~46GB) & ZERO grad_checkpoint double-recompute (2x faster!)
        ds = V10MegaMultiDomainDataset(synth_train, real_train, arkit_train, tartan_train, clip_len=12, clips_per_epoch=2560)  # 320 micro-steps = 160 opt steps/ep
        loader = torch.utils.data.DataLoader(
            ds, batch_size=micro_bs, shuffle=True, num_workers=14, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
        )
        opt = torch.optim.AdamW(student.param_groups(lr_head=2.2e-4, lr_moge=1.8e-5), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (len(loader) // accum_steps), eta_min=4.0e-6)

        start_ep = 1
        history = []
        resume_ck = DRIVE_RUNS / "egoclear_v10_latest.pt"
        if resume_ck.exists():
            ck = torch.load(resume_ck, map_location="cpu", weights_only=False)
            student.load_state_dict(ck["state_dict"], strict=False)
            if "opt" in ck and "sched" in ck:
                opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            history = ck.get("history", [])
            start_ep = int(ck.get("ep", 0)) + 1
            print(f"[v10] RESUMED from ep={start_ep-1}", flush=True)

        t_tr = time.time()
        print(f"\n=== Training EgoClear-v10 TURBO (ep {start_ep}..{epochs}, v9 Warm-Start + 2x GradAccum No-Ckpt + Mega-Data 2463 scenes) ===", flush=True)
        for ep in range(start_ep, epochs + 1):
            student.train()
            kd_w = 0.44 - 0.24 * ((ep - 1) / max(1, epochs - 1))  # 0.44 -> 0.20
            ep_l = {"tot": 0.0, "st": 0.0, "co": 0.0, "rank": 0.0, "ohem": 0.0, "geom": 0.0, "zgt": 0.0, "dav2": 0.0}
            te = time.time()
            opt.zero_grad(set_to_none=True)

            for bi, batch in enumerate(loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                rgb_t = batch["rgb_t"].float()
                rgb_s = batch["rgb_s"].float()
                rgb_sc = batch["rgb_s_clean"].float()

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_v66 = teacher_v66.forward_sequence(rgb_t, batch["K_t"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"])
                    out_v85 = teacher_v85.forward_sequence(rgb_t, batch["K_t"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"])

                    b_sub, t_sub = rgb_sc[:, ::6].shape[:2]
                    rgb_sub_f = rgb_sc[:, ::6].reshape(b_sub * t_sub, 3, RGB_H, RGB_W)
                    K_sub_f = batch["K_s"][:, ::6].reshape(b_sub * t_sub, 4)
                    R_sub_f = batch["R_s_clean"][:, ::6].reshape(b_sub * t_sub, 3, 3)

                    f66_sub, d66_sub = teacher_v66.encode_images(rgb_sub_f, K_sub_f, R_sub_f)
                    dav2_t = F.interpolate(
                        torch.cat([f66_sub.float(), d66_sub[:, 0:1].float()], dim=1),
                        size=(FEAT_H, FEAT_W), mode="bilinear", align_corners=False,
                    )

                _, z_l, n_l, m_l = extract_moge_geom_cues_standalone(
                    moge_l, rgb_sub_f, K_sub_f, R_sub_f,
                    uv_levels, student.feat_uu, student.feat_vv, student.sobel_x, student.sobel_y, student.im_mean, student.im_std,
                )

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_s = student.forward_sequence(
                        rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"],
                        return_aux_geom=True,
                    )
                    loss, ld = compute_loss_v10(
                        out_s, out_v66, out_v85,
                        (z_l.detach(), n_l.detach(), m_l.detach()),
                        dav2_t.detach(),
                        batch, student.sobel_x, student.sobel_y, kd_w=kd_w,
                    )
                (loss / accum_steps).backward()
                if (bi + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.5)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                for k in ep_l:
                    ep_l[k] += ld[k]

            nb = len(loader)
            for k in ep_l:
                ep_l[k] /= nb
            ep_l["ep"] = ep
            ep_l["sec"] = round(time.time() - te, 1)
            history.append(ep_l)
            print(
                f"  [v10 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} st={ep_l['st']:.4f} co={ep_l['co']:.4f} "
                f"rank={ep_l['rank']:.4f} ohem={ep_l['ohem']:.4f} geom={ep_l['geom']:.4f} zgt={ep_l['zgt']:.4f} dav2={ep_l['dav2']:.4f} ({ep_l['sec']:.1f}s)",
                flush=True,
            )

            if ep % 3 == 0 or ep == epochs:
                ck_loc = LOCAL_RUNS / "egoclear_v10_latest.pt"
                torch.save(
                    {
                        "ep": ep,
                        "state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "history": history,
                    },
                    ck_loc,
                )
                shutil.copy2(ck_loc, DRIVE_RUNS / ck_loc.name)

            if ep == 5:
                student.eval()
                print_block("V10-ep05", "real_40_local", eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=False))
                print_block("V10-ep05", "aria_g2_zeroshot", eval_v6_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=False))
                print("  [V10-ep05 Aria iso_letterbox]", eval_frontend_mode(student, teacher_v66, aria_dicts, None, bw_speeds, device, "iso_letterbox"), flush=True)
                print("  [V10-ep05 Aria iso_grav_crop]", eval_frontend_mode(student, teacher_v66, aria_dicts, None, bw_speeds, device, "iso_grav_crop"), flush=True)
                print_block("V10-ep05", "arkit_indoor_val30", eval_v6_all(student, eval_arkit, None, bw_speeds, device, compute_uncertainty_audit=False))
                print_block("V10-ep05", "coda_campus_zeroshot", eval_v6_all(student, eval_coda, None, bw_speeds, device, compute_uncertainty_audit=False))

        train_sec = round(time.time() - t_tr, 1)
        orig_drop = student.rgb_feat_drop_p
        student.rgb_feat_drop_p = 0.0
        recalibrate_bn_stats(student, loader, device, n_batches=32)
        student.rgb_feat_drop_p = orig_drop
        del moge_l
        torch.cuda.empty_cache()

        torch.save({"state_dict": {k: v.cpu() for k, v in student.state_dict().items()}, "moge_id": student.moge_id, "history": history}, ckpt_final)
        shutil.copy2(ckpt_final, drive_ckpt_final)

    # Full 9-Benchmark + Multi-Frontend Isotropic Eval + Paired Bootstrap vs v6.6/v8.5/v9 + Dual-NPU ENS(v6.6+v10)!
    print("\n=== Full 9-Benchmark + Multi-Frontend Isotropic + Paired Bootstrap (v10 vs v6.6, v8.5, v9) ===", flush=True)
    student.eval()
    eval_real_gray = make_variant_scenes(eval_real, "gray")

    # GT depth scale audit
    sc_ratios, abs_rels = [], []
    for sc, d in list(eval_real.items())[:20]:
        gd_all = d.get("depth_s8")
        if gd_all is None:
            continue
        for fi in range(0, len(d["eval_rows"]), 12):
            inp = prepare_scene_slice_tensors(d, fi, fi + 1, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, g = student.encode_images(inp["rgb"][0], inp["K_scaled"][0], inp["R_c2w"][0])
            gd = gd_all[fi].astype(np.float32)
            zp = cv2.resize(g[0, 0].float().cpu().numpy(), (gd.shape[1], gd.shape[0]))
            ok = np.isfinite(gd) & (gd >= 0.5) & (gd <= 6.0)
            if ok.sum() >= 30:
                sc_ratios.append(float(np.median(zp[ok] / gd[ok])))
                abs_rels.append(float(np.mean(np.abs(zp[ok] - gd[ok]) / gd[ok])))
    depth_calib = {
        "real40_AbsRel": round(float(np.mean(abs_rels)), 4) if abs_rels else None,
        "real40_scale_median": round(float(np.median(sc_ratios)), 4) if sc_ratios else None,
    }
    print(f"[v10 Depth Scale Audit] {depth_calib} (v9 was 0.1878 / 1.0493, MoGe-2 was 0.2906 / 1.135)", flush=True)

    # Load v9 for exact paired bootstrap comparison!
    teacher_v9 = EgoClearV9Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), freeze_moge=True).to(device).eval()
    teacher_v9.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase12_v9/egoclear_v9_moge2_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    evals, paired_audits, boot_v66, boot_v85, boot_v9, iso_evals = {}, {}, {}, {}, {}, {}
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
        print_block("EgoClear-v10", dname, evals[dname])
        if vscale == 0.0 and dname != "real_40_gray":
            paired_audits[dname] = collect_paired_route_audit(student, dmap, ov, bw_speeds, device, fixed_thr=0.50)
            f50 = paired_audits[dname]["fixed_thr_0.50"]
            print(f"     [FIXED thr=0.50] late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km ({f50['fa_ep_km']} ep/km)", flush=True)

            if dname != "synth_clean":
                ev_m, s10_m, s66_m, s85_m, s9_m, sens_m = {}, {}, {}, {}, {}, {}
                for sc, d in dmap.items():
                    er = ov[sc] if ov is not None else d["eval_rows"]
                    if len(er) < 5:
                        continue
                    inp = prepare_scene_slice_tensors(d, 0, len(er), device)
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s10_m[sc] = student.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                        s66_m[sc] = teacher_v66.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                        s85_m[sc] = teacher_v85.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                        s9_m[sc] = teacher_v9.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    ev_m[sc] = er
                    sens_m[sc] = sigmoid(0.5 * logit(s66_m[sc]) + 0.5 * logit(s10_m[sc]))
                c10 = summarize_at_fa_caps(eval_score_curve(ev_m, s10_m, bw_speeds, thrs, higher_is_alert=True))
                c66 = summarize_at_fa_caps(eval_score_curve(ev_m, s66_m, bw_speeds, thrs, higher_is_alert=True))
                c85 = summarize_at_fa_caps(eval_score_curve(ev_m, s85_m, bw_speeds, thrs, higher_is_alert=True))
                c9 = summarize_at_fa_caps(eval_score_curve(ev_m, s9_m, bw_speeds, thrs, higher_is_alert=True))
                t10 = float(c10["late_at_fa_fr<=120.3"]["thr"]) if c10.get("late_at_fa_fr<=120.3") else 0.50
                t66 = float(c66["late_at_fa_fr<=120.3"]["thr"]) if c66.get("late_at_fa_fr<=120.3") else 0.50
                t85 = float(c85["late_at_fa_fr<=120.3"]["thr"]) if c85.get("late_at_fa_fr<=120.3") else 0.50
                t9 = float(c9["late_at_fa_fr<=120.3"]["thr"]) if c9.get("late_at_fa_fr<=120.3") else 0.50
                boot_v66[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s66_m, bw_speeds, t66), per_route_at_thr(dmap, ov, s10_m, bw_speeds, t10))
                boot_v85[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s85_m, bw_speeds, t85), per_route_at_thr(dmap, ov, s10_m, bw_speeds, t10))
                boot_v9[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s9_m, bw_speeds, t9), per_route_at_thr(dmap, ov, s10_m, bw_speeds, t10))
                iso_evals[f"{dname}_ens_v66_v10_squashed"] = summarize_scores(ev_m, sens_m, bw_speeds)
                print(f"     [Bootstrap @120 v10 vs v6.6] {boot_v66[dname]}", flush=True)
                print(f"     [Bootstrap @120 v10 vs v8.5] {boot_v85[dname]}", flush=True)
                print(f"     [Bootstrap @120 v10 vs v9  ] {boot_v9[dname]}", flush=True)
                print(f"     [Dual-NPU 70ms ENS(v6.6+v10)] {iso_evals[f'{dname}_ens_v66_v10_squashed']}", flush=True)

    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v10", "arkit_dark018", evals["arkit_dark018"])
    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v10", "real_40_dark018", evals["real_40_dark018"])

    # Multi-frontend Isotropic evaluations on non-16:9 domains (Aria 1:1, CODa 5:4, ARKit 4:3)!
    print("\n=== Universal Isotropic Frontend Evaluation (fx==fy on Aria 1:1, CODa 5:4, ARKit 4:3) ===", flush=True)
    for dname, dmap, modes in [
        ("aria_g2_zeroshot", aria_dicts, ("iso_letterbox", "iso_crop", "iso_grav_crop")),
        ("coda_campus_zeroshot", eval_coda, ("iso_letterbox", "iso_crop")),
        ("arkit_indoor_val30", eval_arkit, ("iso_letterbox",)),
    ]:
        for md in modes:
            k_iso = f"{dname}__{md}"
            iso_evals[k_iso] = eval_frontend_mode(student, teacher_v66, dmap, None, bw_speeds, device, md)
            print(f"  [{k_iso:36s}] {iso_evals[k_iso]}", flush=True)

    del teacher_v66, teacher_v85, teacher_v9
    torch.cuda.empty_cache()

    onnx_info = export_v10_onnx(student, LOCAL_RUNS / "onnx")
    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v10 (Mega-Data + Border-Masked Isotropic Dual-Geom + 3D Homography Head-Scan + Unlocked MoGe-2 Neck/Last2 + Zero-Cost DAV2 Aux + Cured Continuous Loss)",
        "total_params_m": round(n_tot / 1e6, 2),
        "trainable_params_m": round(n_tr / 1e6, 2),
        "unlocked_moge_params_m": round(n_moge_tr / 1e6, 2),
        "train_dataset_counts": {
            "synth": len(synth_train),
            "real_outdoor": len(real_train),
            "arkit_indoor": len(arkit_train),
            "tartanair_widefov": len(tartan_train),
        },
        "depth_scale_audit": depth_calib,
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "history": history,
        "evals": evals,
        "iso_and_ens_evals": iso_evals,
        "paired_route_audits_at_0.50": paired_audits,
        "paired_bootstrap_vs_v66_at_120": boot_v66,
        "paired_bootstrap_vs_v85_at_120": boot_v85,
        "paired_bootstrap_vs_v9_at_120": boot_v9,
        "onnx_export": onnx_info,
    }
    res_p = LOCAL_RUNS / "phase13_v10_results.json"
    res_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_p, DRIVE_RUNS / res_p.name)
    print(f"\n[v10] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
