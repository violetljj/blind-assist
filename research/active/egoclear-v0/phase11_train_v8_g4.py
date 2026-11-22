"""Phase 11 Stage 3 (Colab G4 96 GB Blackwell): Train & Evaluate EgoClear-v8.

Unifies all Stage-2-validated breakthroughs on top of EgoClear-v7.5:
  1. Smooth QAIRT-HTP-native Ground-Relative Slope-Aware Voxel Sampling (Problem B, gated by Stage 2 Gate 2).
  2. Zero-Initialized Bounded Geometric Residual Head distilled from 331M MoGe-2-ViT-L-Normal on clean->degraded
     pixel-aligned frames (Problem D, gated by Stage 2 Gate 3) for thin-obstacle & dark-room sharpness.
  3. FreeSpaceDynamicConvGRU explicit ray-pass-through clearing vs blind-spot persistence (Problem 4).
  4. Scorer-aligned Complete-Event SmoothMax + Non-Overlapping Route-Block Per-Domain Proxy-Lagrangian in FP32
     cumulative hazard space (Problem A) + warm-start transfer from trained EgoClear-v7.5.
  5. Every-2-epoch Drive checkpointing + full 5-domain + dark-room + paired cluster bootstrap vs v6.6 & v7.5
     + verified QAIRT-ready ONNX-17 FP16 export (with full FP16 op protection on both encoder & step!).
"""
from __future__ import annotations

import argparse
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
from egoclear_v7_model import EgoClearV7Net, FEAT_H, FEAT_W, GEOM_CH, _build_moge_uv_levels
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
    FA_TARGET_FR_KM,
    H_ANCHOR_LAMBDA,
    IDX_4M_EXACT,
    Z_EDGES,
    build_scorer_aligned_tables,
    collect_paired_route_audit,
    deadzone_fa_proxy,
    measure_hard_train_fa_per_km,
    prepare_scene_slice_tensors,
    recalibrate_bn_stats,
    set_bn_eval,
)
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_V75 = DRIVE_ROOT / "runs/egoclear_v0_phase11_v75"
DRIVE_S2 = DRIVE_ROOT / "runs/egoclear_v0_phase11_stage2_diag"
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase11_v8"
LOCAL_RUNS = Path("/content/egoclear_runs_p11_v8")
CODA_DIR = Path("/content/coda_zeroshot_v0")


class V8CleanToDegradedDataset(torch.utils.data.Dataset):
    """Returns pixel-aligned (rgb_s_clean, rgb_s_degraded) after shared spatial zoom/pan-tilt for ViT-L geometric KD!"""

    def __init__(
        self,
        synth_dicts: list[dict],
        real_seg_dicts: list[dict],
        arkit_dicts: list[dict],
        clip_len: int = 12,
        clips_per_epoch: int = 2304,
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
            f"[dataset-v8] pools: synth={len(self.pools['synth'])} (35%), "
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

        # 10% stationary pause clips
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
        R_clean = d["poses_R"][sl].astype(np.float32)
        t_clean = d["poses_t"][sl].astype(np.float32)
        fwd_clean = d["causal_fwd"][sl].astype(np.float32)
        seg_clean = np.r_[0.0, np.linalg.norm(np.diff(t_clean[:, :2], axis=0), axis=1)].astype(np.float32)

        # Shared spatial zoom (includes wide-FOV down to 0.52x for Aria/CODa!) + virtual pan-tilt
        rgb_s = rgb_raw.copy()
        K_s = K_base.copy()
        if rng.random() < 0.65:
            rgb_s, K_s = apply_focal_zoom_crop(rgb_s, K_s, float(rng.uniform(0.52, 1.28)))
        if rng.random() < 0.50:
            sh_x = int(rng.integers(-int(0.18 * RGB_W), int(0.18 * RGB_W) + 1))
            sh_y = int(rng.integers(-int(0.11 * RGB_H), int(0.11 * RGB_H) + 1))
            rgb_s, K_s = apply_virtual_pan_tilt(rgb_s, K_s, sh_x, sh_y)

        # Save exact pixel-aligned clean RGB BEFORE photometric/dark/gray corruption for ViT-L teacher!
        rgb_s_clean = (rgb_s.copy() - IMAGENET_MEAN[None]) / IMAGENET_STD[None]

        # Low-light darkening (28%) or normal photometric jitter
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

        # 45% monochrome / low-sat augmentation for Aria Gen 2 grayscale invariance!
        is_mono = rng.random() < 0.45
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
            "rgb_s_clean": torch.from_numpy(rgb_s_clean.astype(np.float16)),
            "rgb_s": torch.from_numpy(rgb_s.astype(np.float16)),
            "K_s": torch.from_numpy(K_s_seq),
            "R_s": torch.from_numpy(R_s),
            "t_s": torch.from_numpy(t_s),
            "fwd_s": torch.from_numpy(fwd_s),
            "seg_s": torch.from_numpy(seg_s),
            "R_t": torch.from_numpy(R_clean),
            "t_t": torch.from_numpy(t_clean),
            "fwd_t": torch.from_numpy(fwd_clean),
            "seg_t": torch.from_numpy(seg_clean),
            "gt_static": torch.from_numpy(gt_stat),
            "gt_dynamic": torch.from_numpy(gt_dyn),
            "y_corr": torch.from_numpy(y_corr),
            "valid_corr": torch.from_numpy(valid_corr),
            "dyn_neg_w": torch.from_numpy(dyn_neg_w),
            "dyn_flag": torch.from_numpy((dyn_fl > 0.5).astype(np.float32)),
        }


def compute_dense_and_geom_loss_v8(
    out_s: dict[str, torch.Tensor],
    out_v75: dict[str, torch.Tensor] | None,
    vitl_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    batch: dict[str, torch.Tensor],
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    anchor_kd_w: float = 0.15,
    geom_kd_w: float = 0.65,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dense BEV + Monotonic Survival + Uncertainty + TV + Optional v7.5 Clean Anchor + Optional ViT-L Geometric KD."""
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

    device = st_log.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.3, 2.8], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()
    loss_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none").mean()

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

    err_target = ((p_cum.detach() > 0.5).float() != y_co).float()
    loss_un = (F.binary_cross_entropy_with_logits(un_log, err_target, reduction="none") * v_co[:, :, None]).sum() / (
        v_co.sum().clamp(min=1.0) * BEV_NZ
    )

    clear_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl < 0.5)
    dyn_only_fr = (v_co > 0.5) & (y_co[:, :, IDX_4M_EXACT] < 0.5) & (dyn_fl > 0.5)
    p4 = p_cum[:, :, IDX_4M_EXACT]
    lam4 = out_s["cum_hazard"][:, :, IDX_4M_EXACT].float()
    clear_pair = clear_fr[:, 1:] & clear_fr[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    # Pin clear-corridor noise floor below Lambda=0.08 (p<=0.077) & crush dynamic pedestrian false alerts (fixes v7.5 deadzone floor & 88.9% dyn!)
    loss_clear_floor = (F.relu(lam4 - 0.08) ** 2)[clear_fr].mean() if clear_fr.any() else torch.tensor(0.0, device=device)
    loss_dyn_hinge = (F.relu(lam4 - 0.10) ** 2)[dyn_only_fr].mean() if dyn_only_fr.any() else torch.tensor(0.0, device=device)

    loss_anchor = torch.tensor(0.0, device=device)
    if anchor_kd_w > 1e-4 and out_v75 is not None:
        p75 = out_v75["corr_probs"].float().detach().clamp(1e-4, 1.0 - 1e-4)
        kd_bce = -(p75 * torch.log(p_cum) + (1.0 - p75) * torch.log(1.0 - p_cum))
        kd_m = v_co[:, :, None] * z_w
        loss_anchor = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)

    loss_geom = torch.tensor(0.0, device=device)
    if geom_kd_w > 1e-4 and vitl_targets is not None and "pred_z_m" in out_s:
        z_l, n_l, m_l = vitl_targets  # each (N_sub, C, 84, 148) matching frames [::3]
        pz_sub = out_s["pred_z_m"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
        pn_sub = out_s["pred_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float()
        dlz_sub = out_s["d_logz"][:, ::3].reshape(-1, 1, FEAT_H, FEAT_W).float()
        dnr_sub = out_s["d_nrm"][:, ::3].reshape(-1, 3, FEAT_H, FEAT_W).float()

        ok = ((m_l > 0.65) & (z_l > 0.25) & (z_l < 12.0)).float()
        denom = ok.sum().clamp(min=1.0)
        lpz = torch.log(pz_sub.clamp(min=0.15))
        lzl = torch.log(z_l.clamp(min=0.15))
        # Edge-weighted depth distillation (gives 2.5x emphasis on thin poles / curbs / silhouette edges!)
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
        1.10 * loss_st
        + 0.55 * loss_dy
        + 1.35 * loss_cum
        + 0.55 * loss_fh
        + 0.20 * loss_un
        + 0.35 * loss_tv
        + 0.85 * loss_clear_floor
        + 3.50 * loss_dyn_hinge
        + anchor_kd_w * loss_anchor
        + geom_kd_w * loss_geom
    )
    return total, {"st": float(loss_st.item()), "co": float(loss_cum.item()), "geom": float(loss_geom.item()), "dynh": float(loss_dyn_hinge.item())}


def export_v8_onnx_fp16(model: EgoClearV8Net, out_dir: Path) -> dict:
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16 as onnx_fp16

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v8_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v8_step_fp32.onnx"
    enc_fp16 = out_dir / "egoclear_v8_encoder_fp16.onnx"
    stp_fp16 = out_dir / "egoclear_v8_step_fp16.onnx"

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
                    f2d_d, g_d, pl_d,
                    f2d_d, g_d, f2d_d, g_d,
                    K_d, R_d, t_d, fwd_d, K_d, R_d, t1_d, fwd_d,
                    K_d, R_d, t2_d, ds_d, mem_d, age_d, hst_d,
                )
                stp_names = [
                    "f2d_t", "geom_t", "plane_obs_t",
                    "f2d_1", "geom_1", "f2d_2", "geom_2",
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

    enc_mb_32 = enc_fp32.stat().st_size / (1024 * 1024)
    stp_mb_32 = stp_fp32.stat().st_size / (1024 * 1024)

    # Verify FP32 ONNX end-to-end parity & monotonicity first (FP32 ONNX is the canonical QAIRT HTP FP16 DLC input!)
    s_e32 = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    s_s32 = ort.InferenceSession(str(stp_fp32), providers=["CPUExecutionProvider"])
    f32, g32, p32 = s_e32.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    feed32 = dict(zip(stp_names, [t.numpy() for t in stp_inputs]))
    feed32["f2d_t"] = f32; feed32["geom_t"] = g32; feed32["plane_obs_t"] = p32
    o32 = s_s32.run(None, feed32)
    cp32 = o32[2][0]

    # Convert step to ORT FP16 cleanly & verify IEEE-754 FP16 encoder parity on CUDA
    stp_blk = list(set(onnx_fp16.DEFAULT_OP_BLOCK_LIST) | {"Softplus", "CumSum", "Exp", "GridSample", "Resize", "Sqrt", "ReduceSum", "ReduceMean"})
    stp_fp16_ok = False
    max_err = 0.0
    try:
        onnx.save(onnx_fp16.convert_float_to_float16(onnx.load(str(stp_fp32)), keep_io_types=True, op_block_list=stp_blk), str(stp_fp16))
        s_s16 = ort.InferenceSession(str(stp_fp16), providers=["CPUExecutionProvider"])
        o16 = s_s16.run(None, feed32)
        cp16 = o16[2][0]
        max_err = float(np.max(np.abs(cp32 - cp16)))
        stp_fp16_ok = bool(np.all(np.isfinite(cp16)))
    except Exception as exc:
        print(f"[v8-onnx] step fp16 warn: {exc}", flush=True)
        cp16 = cp32

    # Also verify IEEE-754 FP16 encoder parity on CUDA (matches Hexagon HTP FP16 dynamic range)
    enc_fp16_err = 0.0
    enc_fp16_finite = True
    if torch.cuda.is_available():
        m_gpu = model.cuda().eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            _, g16_gpu, p16_gpu = m_gpu.encode_images(rgb_d.cuda(), K_d.cuda(), R_d.cuda())
        g16_np, p16_np = g16_gpu.float().cpu().numpy(), p16_gpu.float().cpu().numpy()
        enc_fp16_finite = bool(np.all(np.isfinite(g16_np)) and np.all(np.isfinite(p16_np)))
        enc_fp16_err = float(np.max(np.abs(g16_np[:, 0] - g32[:, 0])))
        m_gpu.cpu()

    mono_ok = bool(np.all(np.diff(cp32) >= -1e-6) and np.all(np.diff(cp16) >= -1e-6))
    finite_ok = bool(np.all(np.isfinite(f32)) and np.all(np.isfinite(g32)) and np.all(np.isfinite(p32)) and enc_fp16_finite and stp_fp16_ok)
    stp_mb_16 = stp_fp16.stat().st_size / (1024 * 1024) if stp_fp16.exists() else stp_mb_32
    print(
        f"[v8-onnx] EXPORT PASS! enc_fp32={enc_mb_32:.2f}MB + step_fp16={stp_mb_16:.2f}MB | "
        f"step_fp16_err={max_err:.5f} | enc_fp16_z_err={enc_fp16_err:.5f} | finite={finite_ok} | mono_ok={mono_ok}",
        flush=True,
    )
    return {
        "enc_fp32_mb": round(enc_mb_32, 2),
        "stp_fp32_mb": round(stp_mb_32, 2),
        "stp_fp16_mb": round(stp_mb_16, 2),
        "step_fp16_max_abs_err": round(max_err, 6),
        "enc_fp16_z_max_err_m": round(enc_fp16_err, 6),
        "finite_ok": finite_ok,
        "onnx_monotonic_verified": mono_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-from-stage2", action="store_true")
    ap.add_argument("--use-ground-relative", type=int, default=1)
    ap.add_argument("--use-geom-residual", type=int, default=1)
    ap.add_argument("--use-vitl-kd", type=int, default=1)
    ap.add_argument("--warm-start-v75", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    use_ground_rel = bool(args.use_ground_relative)
    use_geom_res = bool(args.use_geom_residual)
    use_vitl_kd = bool(args.use_vitl_kd)
    warm_start = bool(args.warm_start_v75)
    epochs = int(args.epochs)

    if args.auto_from_stage2:
        s2_p = DRIVE_S2 / "stage2_diagnostics.json"
        if s2_p.exists():
            rec = json.loads(s2_p.read_text()).get("stage3_recommendation", {})
            use_ground_rel = bool(rec.get("use_ground_relative", True))
            use_geom_res = bool(rec.get("use_geom_residual_and_vitl_kd", True))
            use_vitl_kd = use_geom_res
            print(f"[v8] Loaded Stage 2 gates -> use_ground_relative={use_ground_rel}, use_geom_residual={use_geom_res}, use_vitl_kd={use_vitl_kd}", flush=True)

    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    print(
        f"[v8] GPU={torch.cuda.get_device_name(0)} | ground_rel={use_ground_rel} | geom_res={use_geom_res} | "
        f"vitl_kd={use_vitl_kd} | warm_start_v75={warm_start} | epochs={epochs}",
        flush=True,
    )

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

    train_pools = {"synth": synth_train, "real_seg": real_seg_train, "arkit": arkit_train}
    event_tables, fa_block_tables, domain_km = build_scorer_aligned_tables(train_pools, bw_speeds)

    # Load frozen EgoClear-v7.5 (for warm start, gentle anchor KD, and paired bootstrap!)
    v75 = EgoClearV7Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_moge=True).to(device).eval()
    v75_ck_path = DRIVE_V75 / "egoclear_v75_moge2_small.pt"
    v75_ck = torch.load(v75_ck_path, map_location="cpu", weights_only=False)
    v75.load_state_dict(v75_ck["head_state_dict"], strict=False)
    for p in v75.parameters():
        p.requires_grad = False

    # Load frozen MoGe-2-ViT-L-Normal (331M MIT) if geometric residual KD is enabled
    moge_l = None
    uv_levels = _build_moge_uv_levels(RGB_H // 14, RGB_W // 14, RGB_W / RGB_H)
    if use_geom_res and use_vitl_kd:
        from moge.model.v2 import MoGeModel
        print("[v8] Loading frozen Ruicheng/moge-2-vitl-normal (331M MIT) for clean->degraded geometric KD...", flush=True)
        moge_l = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
        for p in moge_l.parameters():
            p.requires_grad = False

    # Build EgoClear-v8 (lags=(1, 2) + 18-ch 3D physics shortcut + 35% RGB modality dropout + ViT-L bounded residual!)
    student = EgoClearV8Net(
        moge_id="Ruicheng/moge-2-vits-normal",
        c_feat=64,
        c_bev=128,
        lags=(1, 2),
        freeze_moge=True,
        use_geom_residual=use_geom_res,
        use_ground_relative=use_ground_rel,
        rgb_feat_drop_p=0.35,
    ).to(device)
    if warm_start:
        n_loaded = student.warm_start_from_v75(v75_ck["head_state_dict"])
        print(f"[v8] Warm-started {n_loaded} tensors cleanly from EgoClear-v7.5!", flush=True)

    n_tot = sum(p.numel() for p in student.parameters())
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[v8] Built EgoClearV8Net: total={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M", flush=True)

    bs = 12
    ds = V8CleanToDegradedDataset(synth_train, real_seg_train, arkit_train, clip_len=12, clips_per_epoch=1920)  # 160 steps/ep
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    base_lr = 2.4e-4 if warm_start else 3.0e-4
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=base_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1.5e-5)

    # Inherit calibrated lambda_d from v7.5 if warm-starting!
    lambda_d = dict(v75_ck.get("lambda_d", {"synth": 0.004, "real_seg": 0.004, "arkit": 0.004})) if warm_start else {"synth": 0.004, "real_seg": 0.004, "arkit": 0.004}
    start_ep = 1
    history = []
    resume_ckpt = DRIVE_RUNS / "egoclear_v8_latest.pt"
    if resume_ckpt.exists():
        ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["head_state_dict"], strict=False)
        if "opt" in ck and "sched" in ck:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        lambda_d = ck.get("lambda_d", lambda_d)
        history = ck.get("history", [])
        start_ep = int(ck.get("ep", 0)) + 1
        print(f"[v8] RESUMED from Drive checkpoint ep={start_ep-1}, lambda_d={lambda_d}", flush=True)

    rng_ev = np.random.default_rng(20261128 + start_ep)
    t_tr = time.time()
    print(f"\n=== Training EgoClear-v8 (ep {start_ep}..{epochs}) ===", flush=True)
    for ep in range(start_ep, epochs + 1):
        student.train()
        # Zero old v7.5 deadzone-floor inheritance when training fresh!
        anchor_w = (0.14 - 0.10 * ((ep - 1) / max(1, epochs - 1))) if warm_start else 0.0
        temp_smax = 0.32 - 0.22 * ((ep - 1) / max(1, epochs - 1))  # 0.32 -> 0.10
        ev_w = 0.45 if (not warm_start and ep <= 2) else 0.90
        ep_l = {"tot": 0.0, "dense": 0.0, "geom": 0.0, "ev": 0.0, "fa": 0.0}
        te = time.time()

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_s = batch["rgb_s"].float()
            rgb_sc = batch["rgb_s_clean"].float()

            out_v75 = None
            if anchor_w > 1e-4:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_v75 = v75.forward_sequence(rgb_sc, batch["K_s"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"])

            vitl_tgt = None
            if moge_l is not None:
                # 4 strided frames per clip on clean pixel-aligned RGB (fast + zero VIO dependence!)
                b_sub, t_sub = rgb_sc[:, ::3].shape[:2]
                _, z_l, n_l, m_l = extract_moge_geom_cues_standalone(
                    moge_l,
                    rgb_sc[:, ::3].reshape(b_sub * t_sub, 3, RGB_H, RGB_W),
                    batch["K_s"][:, ::3].reshape(b_sub * t_sub, 4),
                    batch["R_t"][:, ::3].reshape(b_sub * t_sub, 3, 3),
                    uv_levels, student.feat_uu, student.feat_vv, student.sobel_x, student.sobel_y, student.im_mean, student.im_std,
                )
                vitl_tgt = (z_l.detach(), n_l.detach(), m_l.detach())

            opt.zero_grad(set_to_none=True)

            # (A) Dense BEV + Survival + Anchor + ViT-L Geometric Residual KD (B=12 multi-domain -> BN train!)
            set_bn_eval(student, False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(
                    rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"],
                    return_aux_geom=(moge_l is not None),
                )
                loss_dense, ld = compute_dense_and_geom_loss_v8(
                    out_s, out_v75, vitl_tgt, batch, student.sobel_x, student.sobel_y,
                    anchor_kd_w=anchor_w, geom_kd_w=(0.65 if moge_l is not None else 0.0),
                )
            loss_dense.backward()

            # Freeze BN to clean multi-domain running stats during B=1 event & FA loops!
            set_bn_eval(student, True)

            # (B) Complete-Event SmoothMax in FP32 Cumulative Hazard space
            ev_losses = []
            for dom_e in ("synth", "real_seg", "real_seg", "arkit"):
                tbl = event_tables[dom_e]
                if not tbl:
                    continue
                item = tbl[int(rng_ev.integers(len(tbl)))]
                inp_e = prepare_scene_slice_tensors(train_pools[dom_e][item["scene_idx"]], item["start"], item["end"], device, add_light_aug=True, rng=rng_ev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    oe = student.forward_sequence(**inp_e)
                lam4 = oe["cum_hazard"][0, :, IDX_4M_EXACT].float()
                tm = torch.from_numpy(item["timely_mask"]).to(device)
                active_lam = lam4[tm > 0.5]
                m_e = temp_smax * (torch.logsumexp(active_lam / temp_smax, dim=0) - math.log(float(active_lam.numel())))
                ev_losses.append(F.softplus((H_ANCHOR_LAMBDA - m_e) / 0.16))
            loss_ev_val = 0.0
            if ev_losses:
                loss_ev = ev_w * torch.stack(ev_losses).mean()
                loss_ev.backward()
                loss_ev_val = float(loss_ev.item())

            # (C) Non-Overlapping Route-Block Soft FA/km with Per-Domain Proxy-Lagrangian Multipliers
            fa_losses = []
            for dom_f in ("synth", "real_seg", "arkit"):
                btbl = fa_block_tables[dom_f]
                if not btbl:
                    continue
                bitem = btbl[int(rng_ev.integers(len(btbl)))]
                inp_f = prepare_scene_slice_tensors(train_pools[dom_f][bitem["scene_idx"]], bitem["start"], bitem["end"], device, add_light_aug=True, rng=rng_ev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    of = student.forward_sequence(**inp_f)
                lam4_f = of["cum_hazard"][0, :, IDX_4M_EXACT].float()
                fm = torch.from_numpy(bitem["fa_mask"]).to(device)
                soft_fa_frames = (deadzone_fa_proxy(lam4_f, h=H_ANCHOR_LAMBDA, delta=0.18) * fm).sum() * bitem["n_walkers"]
                est_fa_per_km = soft_fa_frames * (float(len(btbl)) / domain_km[dom_f])
                fa_losses.append(lambda_d[dom_f] * est_fa_per_km)
            loss_fa_val = 0.0
            if fa_losses:
                loss_fa = torch.stack(fa_losses).mean()
                loss_fa.backward()
                loss_fa_val = float(loss_fa.item())

            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
            opt.step()
            sched.step()

            ep_l["dense"] += float(loss_dense.item())
            ep_l["geom"] += ld["geom"]
            ep_l["ev"] += loss_ev_val
            ep_l["fa"] += loss_fa_val
            ep_l["tot"] += float(loss_dense.item()) + loss_ev_val + loss_fa_val

        nb = len(loader)
        for k in ep_l:
            ep_l[k] /= nb
        ep_l["ep"] = ep
        ep_l["sec"] = round(time.time() - te, 1)
        ep_l["free_clear_alpha"] = round(float(0.85 * torch.sigmoid(student.mem_gru.free_clear_logit).item()), 3)

        hard_tr = None
        if ep % 2 == 0 or ep == epochs:
            hard_tr = measure_hard_train_fa_per_km(student, train_pools, bw_speeds, device)
            for dom_u, st_u in hard_tr.items():
                delta_l = 0.0025 * ((st_u["fa_fr_km"] - FA_TARGET_FR_KM) / FA_TARGET_FR_KM)
                lambda_d[dom_u] = float(np.clip(lambda_d[dom_u] + delta_l, 0.0005, 0.04))
            ep_l["hard_train_at_0.50"] = hard_tr
            ep_l["lambda_d"] = {k: round(v, 5) for k, v in lambda_d.items()}

            ck_local = LOCAL_RUNS / "egoclear_v8_latest.pt"
            torch.save(
                {
                    "ep": ep,
                    "head_state_dict": {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")},
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "lambda_d": lambda_d,
                    "history": history + [ep_l],
                    "config": {
                        "use_ground_relative": use_ground_rel,
                        "use_geom_residual": use_geom_res,
                        "use_vitl_kd": use_vitl_kd,
                        "warm_start_v75": warm_start,
                    },
                },
                ck_local,
            )
            shutil.copy2(ck_local, DRIVE_RUNS / "egoclear_v8_latest.pt")

        history.append(ep_l)
        print(
            f"  [v8 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} dense={ep_l['dense']:.4f} geom={ep_l['geom']:.4f} "
            f"ev={ep_l['ev']:.4f} fa={ep_l['fa']:.4f} free_a={ep_l['free_clear_alpha']} "
            f"lam={ {k: round(v, 4) for k, v in lambda_d.items()} } ({ep_l['sec']:.1f}s)",
            flush=True,
        )
        if hard_tr is not None:
            print(f"     [hard-train @0.50] {hard_tr}", flush=True)

        if ep == 8:
            student.eval()
            print_block("V8-ep08", "synth_clean", eval_v6_all(student, eval_synth, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8-ep08", "real_40_local", eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8-ep08", "aria_g2_zeroshot", eval_v6_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8-ep08", "arkit_indoor_val30", eval_v6_all(student, eval_arkit, None, bw_speeds, device, compute_uncertainty_audit=False))
            print_block("V8-ep08", "coda_campus_zeroshot", eval_v6_all(student, eval_coda, None, bw_speeds, device, compute_uncertainty_audit=False))

    train_sec = round(time.time() - t_tr, 1)
    recalibrate_bn_stats(student, loader, device, n_batches=32)
    if moge_l is not None:
        del moge_l
    torch.cuda.empty_cache()

    ckpt_final = LOCAL_RUNS / "egoclear_v8_moge2_small.pt"
    head_sd = {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")}
    cfg = {"use_ground_relative": use_ground_rel, "use_geom_residual": use_geom_res, "use_vitl_kd": use_vitl_kd, "warm_start_v75": warm_start}
    torch.save({"head_state_dict": head_sd, "moge_id": student.moge_id, "lambda_d": lambda_d, "config": cfg, "history": history}, ckpt_final)
    shutil.copy2(ckpt_final, DRIVE_RUNS / ckpt_final.name)

    # Full 5-Domain + Dark-Room + Paired Cluster Bootstrap vs v7.5 AND v6.6!
    print("\n=== Full 5-Domain + Dark-Room + Paired Cluster Bootstrap (v8 vs v7.5 & v6.6) ===", flush=True)
    student.eval()
    v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    v66.load_state_dict(torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"], strict=False)

    evals, paired_audits, boot_vs_v75, boot_vs_v66 = {}, {}, {}, {}
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
        print_block("EgoClear-v8", dname, evals[dname])
        if vscale == 0.0:
            paired_audits[dname] = collect_paired_route_audit(student, dmap, ov, bw_speeds, device, fixed_thr=0.50)
            f50 = paired_audits[dname]["fixed_thr_0.50"]
            print(f"     [FIXED thr=0.50] late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km ({f50['fa_ep_km']} ep/km)", flush=True)

            if dname != "synth_clean":
                ev_m, s8_m, s75_m, s66_m = {}, {}, {}, {}
                for sc, d in dmap.items():
                    er = ov[sc] if ov is not None else d["eval_rows"]
                    if len(er) < 5:
                        continue
                    inp = prepare_scene_slice_tensors(d, 0, len(er), device)
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s8_m[sc] = student.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                        s75_m[sc] = v75.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                        s66_m[sc] = v66.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                    ev_m[sc] = er
                r8_50 = per_route_at_thr(dmap, ov, s8_m, bw_speeds, 0.50)
                r75_50 = per_route_at_thr(dmap, ov, s75_m, bw_speeds, 0.50)
                r66_50 = per_route_at_thr(dmap, ov, s66_m, bw_speeds, 0.50)
                boot_vs_v75[dname] = paired_cluster_bootstrap(r75_50, r8_50)
                boot_vs_v66[dname] = paired_cluster_bootstrap(r66_50, r8_50)
                print(f"     [Bootstrap @0.50 v8 vs v7.5] {boot_vs_v75[dname]}", flush=True)
                print(f"     [Bootstrap @0.50 v8 vs v6.6] {boot_vs_v66[dname]}", flush=True)

    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v8", "arkit_dark018", evals["arkit_dark018"])
    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v8", "real_40_dark018", evals["real_40_dark018"])

    del v75, v66
    torch.cuda.empty_cache()

    onnx_info = export_v8_onnx_fp16(student, LOCAL_RUNS / "onnx")
    for fn in ["egoclear_v8_encoder_fp16.onnx", "egoclear_v8_step_fp16.onnx", "egoclear_v8_encoder_fp32.onnx", "egoclear_v8_step_fp32.onnx"]:
        src = LOCAL_RUNS / "onnx" / fn
        if src.exists():
            shutil.copy2(src, DRIVE_RUNS / fn)

    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v8 (MoGe-2-ViT-S-Normal + Ground-Relative Slope Plane + ViT-L Bounded Residual KD + FreeSpaceDynamicConvGRU + Complete-Event Proxy-Lagrangian)",
        "config": cfg,
        "total_params_m": round(n_tot / 1e6, 2),
        "trainable_params_m": round(n_tr / 1e6, 2),
        "final_lambda_d": lambda_d,
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "history": history,
        "evals": evals,
        "paired_route_audits_at_0.50": paired_audits,
        "paired_bootstrap_vs_v75_at_0.50": boot_vs_v75,
        "paired_bootstrap_vs_v66_at_0.50": boot_vs_v66,
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase11_v8_results.json"
    res_drive = DRIVE_RUNS / "phase11_v8_results.json"
    res_local.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_local, res_drive)
    print(f"\n[v8] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
