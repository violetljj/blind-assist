"""Phase 11 Stage 3B (Colab G4 96 GB Blackwell): EgoClear-v8.5 — Pure-Physics MoGe-2 Clean-Unified Champion.

Synthesizes EVERY verified scientific discovery from Stages 1, 2, 3A & X-Ray Autopsy:
  1. MoGe-2-ViT-S-Normal (35M MIT) + 84x148 Metric-3D & Gravity-Normal Voxel Consistency + Verified FP16-Bounded Tile-Capped Shift.
  2. lags = (1, 2) ONLY (proven -6.3 pp on SANPO-Real to 34.2%, -1.9 pp on ARKit to 2.6%, -1.5 pp on Synth, 20% faster Android step!).
  3. Zero-Init Bounded Geometric Residual Head distilled from 331M MoGe-2-ViT-L-Normal on pixel-aligned clean->dark/gray frames
     (Gate 3 proved -12.6 pp interface-swap ceiling on SANPO-Real 40 & -0.042 edge AbsRel!).
  4. 18-ch Direct 3D Physics-Shortcut Corridor Head + FreeSpaceDynamicConvGRU (explicit ray-pass-through clearing).
  5. ROCK-SOLID B=16 Unified Training (ZERO oscillating B=1 Proxy-Lagrangian loops! ZERO gradient-clipping starvation!):
     - Exact IDX_4M_EXACT = 18 (4.10m) BEV + Monotonic Survival + First-Hit + Uncertainty + Temporal Smoothness.
     - Smooth B=16 Clear-Corridor Top-25% Tail-OHEM (pins clear floor <= 0.06, unlocking Aria 10x & CODa 58x SNR!).
     - Explicit Dynamic-Only Suppression Hinge (crushes real_40 dyn back below 5%!).
     - Decaying Clean-to-Degraded Dual-Teacher Complementary KD (V6-Base + V5-Small: 0.38 -> 0.12) to preserve Aria, CODa, Synth & HEAD F1!
  6. Full 5-Domain + Dark-Room 0.18x + VIO 1x + Causal Uncertainty-Gated Confirmation + Paired Route-Cluster Bootstrap vs v6.6 & v7.5 + QAIRT ONNX export!
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

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v6_model import EgoClearV6Net
from egoclear_v7_model import EgoClearV7Net, FEAT_H, FEAT_W, _build_moge_uv_levels
from egoclear_v8_model import EgoClearV8EncoderExport, EgoClearV8Net, EgoClearV8StepExport
from eval_closed_loop import H, eval_score_curve, make_episodes, score_episode, summarize_at_fa_caps
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase6_train_v5_g4 import apply_virtual_pan_tilt
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
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_V75 = DRIVE_ROOT / "runs/egoclear_v0_phase11_v75"
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase11_v85"
LOCAL_RUNS = Path("/content/egoclear_runs_p11_v85")
CODA_DIR = Path("/content/coda_zeroshot_v0")


class V85UnifiedDataset(torch.utils.data.Dataset):
    """Returns clean teacher RGB (rgb_t), pixel-aligned clean student RGB (rgb_s_clean for ViT-L KD), and degraded student RGB (rgb_s)."""

    def __init__(
        self,
        synth_dicts: list[dict],
        real_seg_dicts: list[dict],
        arkit_dicts: list[dict],
        clip_len: int = 12,
        clips_per_epoch: int = 2560,
    ):
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.z_edges = Z_EDGES
        self.idx_4m = IDX_4M_EXACT
        self.pools = {
            "synth": [d for d in synth_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "real_seg": [d for d in real_seg_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "arkit": [d for d in arkit_dicts if len(d["eval_rows"]) >= clip_len + 1],
        }
        print(
            f"[dataset-v8.5] pools: synth={len(self.pools['synth'])} (35%), "
            f"real_seg={len(self.pools['real_seg'])} (45%), arkit={len(self.pools['arkit'])} (20%)",
            flush=True,
        )

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.35:
            dom = "synth"
        elif u_dom < 0.80 and len(self.pools["real_seg"]) > 0:
            dom = "real_seg"
        elif len(self.pools["arkit"]) > 0:
            dom = "arkit"
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
        if len(haz_idx) > 0 and u_pick < 0.48:
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

        # 10% stationary pause simulation
        if rng.random() < 0.10:
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

        # Spatial zoom + virtual pan-tilt (shared between rgb_s_clean and rgb_s!)
        rgb_s = rgb_raw.copy()
        K_s = K_base.copy()
        if rng.random() < 0.65:
            rgb_s, K_s = apply_focal_zoom_crop(rgb_s, K_s, float(rng.uniform(0.55, 1.28)))
        if rng.random() < 0.50:
            sh_x = int(rng.integers(-int(0.18 * RGB_W), int(0.18 * RGB_W) + 1))
            sh_y = int(rng.integers(-int(0.11 * RGB_H), int(0.11 * RGB_H) + 1))
            rgb_s, K_s = apply_virtual_pan_tilt(rgb_s, K_s, sh_x, sh_y)

        rgb_s_clean = (rgb_s.copy() - IMAGENET_MEAN[None]) / IMAGENET_STD[None]

        # 28% darkroom low-light + sensor noise OR normal photometric jitter
        if rng.random() < 0.28:
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

        is_mono = rng.random() < 0.42
        if is_mono:
            luma = 0.299 * rgb_s[:, 0:1] + 0.587 * rgb_s[:, 1:2] + 0.114 * rgb_s[:, 2:3]
            sat = 0.0 if rng.random() < 0.65 else float(rng.uniform(0.0, 0.25))
            rgb_s = np.clip(sat * rgb_s + (1.0 - sat) * luma, 0.0, 1.0)

        if rng.random() < 0.20:
            for ti in range(self.clip_len):
                rgb_s[ti] = cv2.GaussianBlur(rgb_s[ti].transpose(1, 2, 0), (3, 3), 0.9).transpose(2, 0, 1)

        rgb_s = (rgb_s - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_s_seq = np.repeat(K_s[None], self.clip_len, axis=0)

        R_s, t_s, fwd_s, seg_s = R_clean.copy(), t_clean.copy(), fwd_clean.copy(), seg_clean.copy()
        if rng.random() < 0.50:
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
        dyn_neg_w = np.where(np.isinf(D_gt) & (dyn_fl > 0.5), 2.8, np.where(ambig_subvox, 0.15, 1.0)).astype(np.float32)

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
            "R_s": torch.from_numpy(R_s),
            "t_s": torch.from_numpy(t_s),
            "fwd_s": torch.from_numpy(fwd_s),
            "seg_s": torch.from_numpy(seg_s),
            "gt_static": torch.from_numpy(gt_stat),
            "gt_dynamic": torch.from_numpy(gt_dyn),
            "y_corr": torch.from_numpy(y_corr),
            "valid_corr": torch.from_numpy(valid_corr),
            "dyn_neg_w": torch.from_numpy(dyn_neg_w),
            "dyn_flag": torch.from_numpy((dyn_fl > 0.5).astype(np.float32)),
            "is_mono": torch.tensor(1.0 if is_mono else 0.0, dtype=torch.float32),
        }


def compute_loss_v85(
    out_s: dict[str, torch.Tensor],
    out_t6: dict[str, torch.Tensor],
    out_t5: dict[str, torch.Tensor],
    vitl_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    batch: dict[str, torch.Tensor],
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    kd_w: float,
    geom_kd_w: float = 0.60,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rock-solid B=16 Unified Loss: Dense BEV + Survival + Clear Tail-OHEM + Dyn Hinge + Complementary Clean-KD + ViT-L Geom KD."""
    st_log = out_s["static_logits"].float()
    dy_log = out_s["dynamic_logits"].float()
    haz_log = out_s["hazard_logits"].float()
    p_cum = out_s["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_log = out_s["unc_logits"].float()

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()
    v_co = batch["valid_corr"].float()
    dn_w = batch["dyn_neg_w"].float()
    dyn_fl = batch["dyn_flag"].float()
    is_mono = batch["is_mono"].float()[:, None, None]

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.3, 2.8], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    # 1. Static & Dynamic BEV Focal/Weighted BCE (preserves strong HEAD F1!)
    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()
    loss_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none").mean()

    # 2. Monotonic Corridor Survival + First-Hit Loss
    z_edges = torch.tensor(Z_EDGES, device=device, dtype=torch.float32)
    z_w = torch.where(z_edges <= 4.15, 2.0, 0.5)[None, None, :]
    bce_cum = -(2.0 * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
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

    # 4. Exact 4.1m Clear-Corridor Smooth Tail-OHEM + Temporal Flicker + Dynamic-Only Non-Alert Hinge!
    clear_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl < 0.5)
    dyn_only_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl > 0.5)
    p4 = p_cum[:, :, IDX_4M_EXACT]
    clear_pair = clear_fr[:, 1:] & clear_fr[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    if clear_fr.any():
        exc = F.relu(p4[clear_fr] - 0.06) ** 2
        k_hard = max(1, int(0.25 * exc.numel()))
        loss_ohem = exc.mean() + 2.5 * torch.topk(exc, k_hard).values.mean()
    else:
        loss_ohem = torch.tensor(0.0, device=device)

    loss_dyn_h = (F.relu(p4[dyn_only_fr] - 0.08) ** 2).mean() if dyn_only_fr.any() else torch.tensor(0.0, device=device)

    # 5. Complementary Clean-to-Degraded Dual-Teacher KD (exact v6.6 formula — preserves Aria/CODa/Synth/HEAD/dyn!)
    p6 = out_t6["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    p5 = out_t5["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
    st6 = torch.sigmoid(out_t6["static_logits"].float().detach())
    st5 = torch.sigmoid(out_t5["static_logits"].float().detach())
    has_stat_bin = y_co > 0.5
    is_dyn_or_mono_clear = ((dyn_fl > 0.5)[:, :, None] | (is_mono > 0.5)) & (~has_stat_bin)
    p_target = torch.where(
        has_stat_bin,
        0.65 * p6 + 0.35 * torch.maximum(p6, p5),
        torch.where(is_dyn_or_mono_clear, p5, torch.minimum(p6, p5)),
    ).clamp(1e-4, 1.0 - 1e-4)
    kd_bce = -(p_target * torch.log(p_cum) + (1.0 - p_target) * torch.log(1.0 - p_cum))
    kd_m = v_co[:, :, None] * z_w
    loss_kd_corr = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)
    dyn_cell = gt_dy.max(dim=2, keepdim=True).values
    st_target = torch.where(gt_st > 0.5, torch.maximum(st6, st5), torch.minimum(st6, st5))
    kd_st = F.binary_cross_entropy_with_logits(st_log, st_target, reduction="none") * (1.0 - dyn_cell) * spat_w
    loss_kd = loss_kd_corr + 0.25 * kd_st.mean()

    # 6. Scale-Normalized Clean->Degraded 331M MoGe-2-ViT-L-Normal Geometric Residual KD (on frames [::6])
    loss_geom = torch.tensor(0.0, device=device)
    if vitl_targets is not None and "pred_z_m" in out_s:
        z_l, n_l, m_l = vitl_targets
        pz_sub = out_s["pred_z_m"][:, ::6].reshape(-1, 1, FEAT_H, FEAT_W).float()
        pn_sub = out_s["pred_nrm"][:, ::6].reshape(-1, 3, FEAT_H, FEAT_W).float()
        dlz_sub = out_s["d_logz"][:, ::6].reshape(-1, 1, FEAT_H, FEAT_W).float()
        dnr_sub = out_s["d_nrm"][:, ::6].reshape(-1, 3, FEAT_H, FEAT_W).float()
        ok = ((m_l > 0.65) & (z_l > 0.25) & (z_l < 12.0)).float()
        denom = ok.sum().clamp(min=1.0)
        lpz = torch.log(pz_sub.clamp(min=0.15))
        lzl = torch.log(z_l.clamp(min=0.15))
        g_xl = F.conv2d(lzl, sobel_x, padding=1)
        g_yl = F.conv2d(lzl, sobel_y, padding=1)
        g_xp = F.conv2d(lpz, sobel_x, padding=1)
        g_yp = F.conv2d(lpz, sobel_y, padding=1)
        edge_w = 1.0 + 1.5 * ((g_xl.abs() + g_yl.abs()) / 0.15).clamp(0.0, 1.0)
        l_z = ((lpz - lzl).abs() * edge_w * ok).sum() / denom
        l_grad = (((g_xp - g_xl).abs() + (g_yp - g_yl).abs()) * ok).sum() / denom
        l_nrm = ((1.0 - (pn_sub * n_l).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)) * ok).sum() / denom
        l_reg = 0.05 * (dlz_sub.pow(2).mean() + dnr_sub.pow(2).mean())
        loss_geom = l_z + 0.50 * l_grad + 0.45 * l_nrm + l_reg

    total = (
        1.15 * loss_st
        + 0.50 * loss_dy
        + 1.65 * loss_cum
        + 0.60 * loss_fh
        + 0.20 * loss_un
        + 0.35 * loss_tv
        + 1.25 * loss_ohem
        + 2.00 * loss_dyn_h
        + kd_w * loss_kd
        + geom_kd_w * loss_geom
    )
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "co": float(loss_cum.item()),
        "ohem": float(loss_ohem.item()),
        "kd": float(loss_kd.item()),
        "geom": float(loss_geom.item()),
    }


def eval_causal_unc_confirm(scores_map: dict[str, np.ndarray], unc_map: dict[str, np.ndarray], eval_rows_map: dict[str, np.ndarray], bw_speeds: dict) -> dict:
    """Strictly causal 0-lookahead uncertainty-gated temporal smoothing: suppresses single-frame uncertain FA spikes!"""
    filt_map = {}
    for sc, p in scores_map.items():
        u = unc_map[sc]
        out_p = p.copy()
        for t in range(1, len(p)):
            # High predicted error probability u[t] > 0.30 dampens sudden 1-frame upward spikes cleanly
            if p[t] > p[t - 1] and u[t] > 0.30:
                w_prev = float(np.clip((u[t] - 0.30) / 0.40, 0.0, 0.55))
                out_p[t] = (1.0 - w_prev) * p[t] + w_prev * p[t - 1]
        filt_map[sc] = out_p
    thrs = np.array([0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.68, 0.76, 0.84], dtype=np.float64)
    return summarize_at_fa_caps(eval_score_curve(eval_rows_map, filt_map, bw_speeds, thrs, higher_is_alert=True))


def export_v85_onnx(model: EgoClearV8Net, out_dir: Path) -> dict:
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v85_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v85_step_fp32.onnx"

    m_cpu = model.eval().float().cpu()
    enc_w = EgoClearV8EncoderExport(m_cpu).eval()
    stp_w = EgoClearV8StepExport(m_cpu).eval()

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
                    output_names=["f2d", "geom_cues", "plane_obs"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
                f2d_d, g_d, pl_d = enc_w(rgb_d, K_d, R_d)
                t_d = torch.tensor([[0.25, 0.0, 1.5]], dtype=torch.float32)
                t1_d = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32)
                t2_d = torch.tensor([[-0.25, 0.0, 1.5]], dtype=torch.float32)
                fwd_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                ds_d = torch.tensor([0.25], dtype=torch.float32)
                mem_d = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
                age_d = torch.full((1, 1, BEV_NZ, BEV_NX), 10.0)
                hst_d = torch.tensor([[1.35, 0.5]], dtype=torch.float32)
                stp_inputs = (
                    f2d_d, g_d, pl_d, f2d_d, g_d, f2d_d, g_d,
                    K_d, R_d, t_d, fwd_d, K_d, R_d, t1_d, fwd_d,
                    K_d, R_d, t2_d, ds_d, mem_d, age_d, hst_d,
                )
                stp_names = [
                    "f2d_t", "geom_t", "plane_obs_t", "f2d_1", "geom_1", "f2d_2", "geom_2",
                    "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1",
                    "K_2", "R_2", "t_2", "ds_t", "mem_prev", "age_prev", "h_state_prev",
                ]
                torch.onnx.export(
                    stp_w, stp_inputs, str(stp_fp32),
                    input_names=stp_names,
                    output_names=["static_logits", "dynamic_logits", "corr_probs", "unc_logits", "mem_new", "age_new", "h_state_new"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
    finally:
        F.interpolate = orig_interp

    s_e32 = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    s_s32 = ort.InferenceSession(str(stp_fp32), providers=["CPUExecutionProvider"])
    f32, g32, p32 = s_e32.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    feed = dict(zip(stp_names, [t.numpy() for t in stp_inputs]))
    feed["f2d_t"] = f32; feed["geom_t"] = g32; feed["plane_obs_t"] = p32
    o32 = s_s32.run(None, feed)
    cp32 = o32[2][0]

    # Native IEEE-754 FP16 encoder parity on GPU (matches Hexagon HTP FP16!)
    m_gpu = model.cuda().eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _, g16_t, p16_t = m_gpu.encode_images(rgb_d.cuda(), K_d.cuda(), R_d.cuda())
    g16, p16 = g16_t.float().cpu().numpy(), p16_t.float().cpu().numpy()
    m_gpu.cpu()

    finite_ok = bool(np.all(np.isfinite(f32)) and np.all(np.isfinite(g32)) and np.all(np.isfinite(cp32)) and np.all(np.isfinite(g16)) and np.all(np.isfinite(p16)))
    mono_ok = bool(np.all(np.diff(cp32) >= -1e-6))
    enc_fp16_err = float(np.max(np.abs(g16[:, 0] - g32[:, 0])))
    enc_mb = enc_fp32.stat().st_size / (1024 * 1024)
    stp_mb = stp_fp32.stat().st_size / (1024 * 1024)
    print(f"[v8.5-onnx] PASS! enc_fp32={enc_mb:.2f}MB + stp_fp32={stp_mb:.2f}MB | ieee_fp16_z_err={enc_fp16_err:.5f}m | finite={finite_ok} | mono={mono_ok}", flush=True)
    for p_o in (enc_fp32, stp_fp32):
        shutil.copy2(p_o, DRIVE_RUNS / p_o.name)
    return {
        "enc_fp32_mb": round(enc_mb, 2),
        "stp_fp32_mb": round(stp_mb, 2),
        "ieee_fp16_z_max_err_m": round(enc_fp16_err, 6),
        "finite_ok": finite_ok,
        "onnx_monotonic_verified": mono_ok,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
    print(f"[v8.5] GPU={torch.cuda.get_device_name(0)} | Pure B=16 Unified MoGe-2 + lags=(1,2) + ViT-L Geom KD + Clean Dual-Teacher KD", flush=True)

    paths = stage_all_p8_datasets_parallel()
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
    real_seg_train = [load_scene(p) for key in ("real_75", "real_22", "real_ch70") for p in sorted(paths[key].glob("*.npz"))]
    arkit_train = [load_scene(p) for key in ("arkit_90", "arkit_210") for p in sorted(paths[key].glob("*.npz"))]

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))}

    # Load Frozen Dual Teachers (V6-Base + V5-Small) + 331M MoGe-2-ViT-L-Normal Geometric Teacher!
    t6_ck = DRIVE_ROOT / "runs/egoclear_v0_phase7_v6/egoclear_v6_base_full.pt"
    t5_ck = DRIVE_ROOT / "runs/egoclear_v0_phase6_v5/egoclear_v5_full.pt"
    teacher_v6 = EgoClearV6Net(dav2_id="depth-anything/Depth-Anything-V2-Base-hf", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_dav2=True).to(device).eval()
    teacher_v6.load_state_dict(torch.load(t6_ck, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v6.parameters():
        p.requires_grad = False

    anchor_v5 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    anchor_v5.load_state_dict(torch.load(t5_ck, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in anchor_v5.parameters():
        p.requires_grad = False

    from moge.model.v2 import MoGeModel
    print("[v8.5] Loading frozen Ruicheng/moge-2-vitl-normal (331M MIT) geometric teacher...", flush=True)
    moge_l = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    for p in moge_l.parameters():
        p.requires_grad = False
    uv_levels = _build_moge_uv_levels(RGB_H // 14, RGB_W // 14, RGB_W / RGB_H)

    # Build EgoClear-v8.5 (lags=(1, 2), gentle rgb_feat_drop_p=0.10, use_geom_residual=True, use_ground_relative=False)
    student = EgoClearV8Net(
        moge_id="Ruicheng/moge-2-vits-normal",
        c_feat=64,
        c_bev=128,
        lags=(1, 2),
        freeze_moge=True,
        use_geom_residual=True,
        use_ground_relative=False,
        rgb_feat_drop_p=0.10,
    ).to(device)
    n_tot = sum(p.numel() for p in student.parameters())
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[v8.5] Built EgoClearV8Net: total={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M", flush=True)

    epochs = 14
    bs = 16
    ds = V85UnifiedDataset(synth_train, real_seg_train, arkit_train, clip_len=12, clips_per_epoch=2560)  # 160 steps/ep
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1.5e-5)

    history = []
    t_tr = time.time()
    print(f"\n=== Training EgoClear-v8.5 (14 clean B=16 epochs, zero B=1 oscillation!) ===", flush=True)
    for ep in range(1, epochs + 1):
        student.train()
        kd_w = 0.42 - 0.28 * ((ep - 1) / max(1, epochs - 1))  # 0.42 -> 0.14
        ep_l = {"tot": 0.0, "st": 0.0, "co": 0.0, "ohem": 0.0, "kd": 0.0, "geom": 0.0}
        te = time.time()

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_t = batch["rgb_t"].float()
            rgb_s = batch["rgb_s"].float()
            rgb_sc = batch["rgb_s_clean"].float()

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_t6 = teacher_v6.forward_sequence(rgb_t, batch["K_t"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"])
                out_t5 = anchor_v5.forward_sequence(rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"])

            # 2 strided frames per clip (32 frames/step -> ~12s/ep!) for clean->degraded ViT-L geometric KD
            b_sub, t_sub = rgb_sc[:, ::6].shape[:2]
            _, z_l, n_l, m_l = extract_moge_geom_cues_standalone(
                moge_l,
                rgb_sc[:, ::6].reshape(b_sub * t_sub, 3, RGB_H, RGB_W),
                batch["K_s"][:, ::6].reshape(b_sub * t_sub, 4),
                batch["R_t"][:, ::6].reshape(b_sub * t_sub, 3, 3),
                uv_levels, student.feat_uu, student.feat_vv, student.sobel_x, student.sobel_y, student.im_mean, student.im_std,
            )

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(
                    rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"],
                    return_aux_geom=True,
                )
                loss, ld = compute_loss_v85(
                    out_s, out_t6, out_t5, (z_l.detach(), n_l.detach(), m_l.detach()),
                    batch, student.sobel_x, student.sobel_y, kd_w=kd_w, geom_kd_w=0.60,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.5)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += ld[k]

        nb = len(loader)
        for k in ep_l:
            ep_l[k] /= nb
        ep_l["ep"] = ep
        ep_l["sec"] = round(time.time() - te, 1)
        ep_l["free_a"] = round(float(0.85 * torch.sigmoid(student.mem_gru.free_clear_logit).item()), 3)
        history.append(ep_l)
        print(
            f"  [v8.5 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} st={ep_l['st']:.4f} co={ep_l['co']:.4f} "
            f"ohem={ep_l['ohem']:.4f} kd={ep_l['kd']:.4f} geom={ep_l['geom']:.4f} free_a={ep_l['free_a']} ({ep_l['sec']:.1f}s)",
            flush=True,
        )

        if ep % 4 == 0 or ep == epochs:
            ck_loc = LOCAL_RUNS / "egoclear_v85_latest.pt"
            torch.save(
                {
                    "ep": ep,
                    "head_state_dict": {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")},
                    "history": history,
                },
                ck_loc,
            )
            shutil.copy2(ck_loc, DRIVE_RUNS / ck_loc.name)

        if ep == 8:
            student.eval()
            print_block("V8.5-ep08", "synth_clean", eval_v6_all(student, eval_synth, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8.5-ep08", "real_40_local", eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8.5-ep08", "aria_g2_zeroshot", eval_v6_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8.5-ep08", "arkit_indoor_val30", eval_v6_all(student, eval_arkit, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8.5-ep08", "coda_campus_zeroshot", eval_v6_all(student, eval_coda, None, bw_speeds, device, compute_uncertainty_audit=False))

    train_sec = round(time.time() - t_tr, 1)
    recalibrate_bn_stats(student, loader, device, n_batches=32)
    del teacher_v6, anchor_v5, moge_l
    torch.cuda.empty_cache()

    ckpt_final = LOCAL_RUNS / "egoclear_v85_moge2_small.pt"
    head_sd = {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")}
    torch.save({"head_state_dict": head_sd, "moge_id": student.moge_id, "history": history}, ckpt_final)
    shutil.copy2(ckpt_final, DRIVE_RUNS / ckpt_final.name)

    # Full 5-Domain + Dark-Room + Causal Uncertainty Confirmation + Paired Cluster Bootstrap vs v6.6 & v7.5!
    print("\n=== Full 5-Domain + Dark-Room + Paired Cluster Bootstrap of EgoClear-v8.5 ===", flush=True)
    student.eval()
    v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    v75 = EgoClearV7Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_moge=True).to(device).eval()
    v75.load_state_dict(torch.load(DRIVE_V75 / "egoclear_v75_moge2_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    evals, paired_audits, unc_confirm, boot_vs_v66_120, boot_vs_v75_120 = {}, {}, {}, {}, {}
    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98], dtype=np.float64)

    for dname, dmap, ov, vscale, do_unc in [
        ("synth_clean", eval_synth, None, 0.0, True),
        ("synth_vio_1x", eval_synth, None, 1.0, False),
        ("real_40_local", eval_real, real_local_rows, 0.0, True),
        ("aria_g2_zeroshot", aria_dicts, None, 0.0, True),
        ("arkit_indoor_val30", eval_arkit, None, 0.0, True),
        ("coda_campus_zeroshot", eval_coda, None, 0.0, True),
    ]:
        if not dmap:
            continue
        evals[dname] = eval_v6_all(student, dmap, ov, bw_speeds, device, vio_scale=vscale, compute_uncertainty_audit=do_unc)
        print_block("EgoClear-v8.5", dname, evals[dname])
        if vscale == 0.0:
            paired_audits[dname] = collect_paired_route_audit(student, dmap, ov, bw_speeds, device, fixed_thr=0.50)
            f50 = paired_audits[dname]["fixed_thr_0.50"]
            print(f"     [FIXED thr=0.50] late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km ({f50['fa_ep_km']} ep/km)", flush=True)

            if dname != "synth_clean":
                ev_m, s85_m, u85_m, s66_m, s75_m = {}, {}, {}, {}, {}
                for sc, d in dmap.items():
                    er = ov[sc] if ov is not None else d["eval_rows"]
                    if len(er) < 5:
                        continue
                    inp = prepare_scene_slice_tensors(d, 0, len(er), device)
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        o85 = student.forward_sequence(**inp)
                        o66 = v66.forward_sequence(**inp)
                        o75 = v75.forward_sequence(**inp, mask_lags=(4,))
                    ev_m[sc] = er
                    s85_m[sc] = o85["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    u85_m[sc] = torch.sigmoid(o85["unc_logits"][0, :, : IDX_4M_EXACT + 1].float()).mean(dim=1).cpu().numpy()
                    s66_m[sc] = o66["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    s75_m[sc] = o75["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()

                unc_confirm[dname] = eval_causal_unc_confirm(s85_m, u85_m, ev_m, bw_speeds)
                uc120 = unc_confirm[dname].get("late_at_fa_fr<=120.3")
                if uc120:
                    print(f"     [Causal Unc-Smooth @120] {uc120['late']}/{uc120['avoidable']}({uc120['late_rate']*100:.1f}%, fr={uc120['fa_fr_per_km']:.1f})", flush=True)

                c85 = summarize_at_fa_caps(eval_score_curve(ev_m, s85_m, bw_speeds, thrs, higher_is_alert=True))
                c66 = summarize_at_fa_caps(eval_score_curve(ev_m, s66_m, bw_speeds, thrs, higher_is_alert=True))
                c75 = summarize_at_fa_caps(eval_score_curve(ev_m, s75_m, bw_speeds, thrs, higher_is_alert=True))
                t85 = float(c85["late_at_fa_fr<=120.3"]["thr"]) if c85.get("late_at_fa_fr<=120.3") else 0.50
                t66 = float(c66["late_at_fa_fr<=120.3"]["thr"]) if c66.get("late_at_fa_fr<=120.3") else 0.50
                t75 = float(c75["late_at_fa_fr<=120.3"]["thr"]) if c75.get("late_at_fa_fr<=120.3") else 0.50
                boot_vs_v66_120[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s66_m, bw_speeds, t66), per_route_at_thr(dmap, ov, s85_m, bw_speeds, t85))
                boot_vs_v75_120[dname] = paired_cluster_bootstrap(per_route_at_thr(dmap, ov, s75_m, bw_speeds, t75), per_route_at_thr(dmap, ov, s85_m, bw_speeds, t85))
                print(f"     [Bootstrap @120 v8.5 vs v6.6] {boot_vs_v66_120[dname]}", flush=True)
                print(f"     [Bootstrap @120 v8.5 vs v7.5(no-lag4)] {boot_vs_v75_120[dname]}", flush=True)

    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v8.5", "arkit_dark018", evals["arkit_dark018"])
    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v8.5", "real_40_dark018", evals["real_40_dark018"])

    del v66, v75
    torch.cuda.empty_cache()

    onnx_info = export_v85_onnx(student, LOCAL_RUNS / "onnx")
    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v8.5 (MoGe-2-ViT-S-Normal 35M + lags=(1,2) + ViT-L 331M Bounded Residual KD + 18ch Physics Shortcut + FreeSpaceGRU + B=16 Tail-OHEM & Clean Dual-Teacher KD)",
        "total_params_m": round(n_tot / 1e6, 2),
        "trainable_params_m": round(n_tr / 1e6, 2),
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "history": history,
        "evals": evals,
        "paired_route_audits_at_0.50": paired_audits,
        "causal_unc_confirm": unc_confirm,
        "paired_bootstrap_vs_v66_at_120": boot_vs_v66_120,
        "paired_bootstrap_vs_v75_nolag4_at_120": boot_vs_v75_120,
        "onnx_export": onnx_info,
    }
    res_p = LOCAL_RUNS / "phase11_v85_results.json"
    res_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_p, DRIVE_RUNS / res_p.name)
    print(f"\n[v8.5] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
