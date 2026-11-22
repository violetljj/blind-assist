"""Phase 11 Step 2 (Colab G4, 96 GB Blackwell):
Train & Evaluate EgoClear-v7 (Frozen MoGe-2-ViT-S-Normal 35M MIT + 84x148 Metric-3D & Gravity-Normal Voxel Consistency
+ 3-Lag 1,2,4 Parallax Stereo + Clear-Corridor Tail-OHEM + Clean-to-Degraded Low-Light/Stationary/Monochrome KD).

Directly targets the exact quantitative bottlenecks uncovered in Phase 9/10 autopsy & live S24 Ultra testing:
  1. Eliminates affine-invariant DAV2 scale schism (10x thr shift: 0.08 CODa vs 0.88 ARKit) via MoGe-2 true metric depth + closed-form focal shift.
  2. Injects gravity-aligned surface normal verticality (1 - |n_up|) + walkable floor hit directly into all 2,520 3D corridor voxels.
  3. Upgrades 2D projection from 42x74 to high-res 84x148 (4x pixels) for thin poles, curbs, and HEAD recall.
  4. Adds Clear-Corridor Top-25% Tail-OHEM loss to crush the 35.1 pp false-alarm tax (@500=13.5% vs @120=48.6% on SANPO-Real).
  5. Adds 28% low-light darkening + sensor noise & 10% stationary pause clips for real-world night/handheld robustness.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net, H_CORR
from egoclear_v6_model import EgoClearV6Net
from egoclear_v7_model import EgoClearV7EncoderExport, EgoClearV7Net, EgoClearV7StepExport, FEAT_H, FEAT_W, GEOM_CH
from eval_closed_loop import H
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase6_train_v5_g4 import apply_virtual_pan_tilt, fmt_m
from phase7_train_v6_g4 import eval_v6_all
from phase8_distill_v65_g4 import stage_all_p8_datasets_parallel
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase11_v7"
LOCAL_RUNS = Path("/content/egoclear_runs_p11_v7")
CODA_DIR = Path("/content/coda_zeroshot_v0")


class V7MetricNormalDataset(torch.utils.data.Dataset):
    """1,094 clean full-seg scenes with Clean-to-Degraded low-light, monochrome, zoom, pan-tilt, VIO & stationary clips."""

    def __init__(
        self,
        synth_dicts: list[dict],
        real_seg_dicts: list[dict],
        arkit_dicts: list[dict],
        clip_len: int = 12,
        clips_per_epoch: int = 2880,
    ):
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
        self.idx_4m = int(np.searchsorted(self.z_edges, 4.0))
        self.pools = {
            "synth": [d for d in synth_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "real_seg": [d for d in real_seg_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "arkit": [d for d in arkit_dicts if len(d["eval_rows"]) >= clip_len + 1],
        }
        print(
            f"[dataset-v7] CLEAN full-seg pools: synth={len(self.pools['synth'])} (35%), "
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

        # 10% Stationary / Pause simulation (teaches zero-baseline standing-still stability without ghost lock-in!)
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

        # Clean inputs for privileged Geometry Teacher (V6-Base)
        rgb_t = (rgb_raw - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_t_seq = np.repeat(K_base[None], self.clip_len, axis=0)
        R_clean = d["poses_R"][sl].astype(np.float32)
        t_clean = d["poses_t"][sl].astype(np.float32)
        fwd_clean = d["causal_fwd"][sl].astype(np.float32)
        seg_clean = np.r_[0.0, np.linalg.norm(np.diff(t_clean[:, :2], axis=0), axis=1)].astype(np.float32)

        # Hardened Student inputs: focal zoom, pan-tilt, low-light darkroom, monochrome, blur, VIO drift
        rgb_s = rgb_raw.copy()
        K_s = K_base.copy()
        if rng.random() < 0.60:
            zoom = float(rng.uniform(0.60, 1.28))
            rgb_s, K_s = apply_focal_zoom_crop(rgb_s, K_s, zoom)
        if rng.random() < 0.50:
            sh_x = int(rng.integers(-int(0.18 * RGB_W), int(0.18 * RGB_W) + 1))
            sh_y = int(rng.integers(-int(0.11 * RGB_H), int(0.11 * RGB_H) + 1))
            rgb_s, K_s = apply_virtual_pan_tilt(rgb_s, K_s, sh_x, sh_y)

        # Low-light darkening + sensor noise (28%) OR normal photometric jitter (72%)
        is_dark = rng.random() < 0.28
        if is_dark:
            gamma = float(rng.uniform(1.10, 1.55))
            gain = float(rng.uniform(0.18, 0.55))
            rgb_s = np.power(np.clip(rgb_s, 1e-4, 1.0), gamma) * gain
            noise_std = float(rng.uniform(0.012, 0.032))
            rgb_s = np.clip(rgb_s + rng.normal(0.0, noise_std, rgb_s.shape).astype(np.float32), 0.0, 1.0)
        else:
            gamma = float(rng.uniform(0.68, 1.45))
            rgb_s = np.power(np.clip(rgb_s, 1e-4, 1.0), gamma)
            ch_gain = rng.uniform(0.84, 1.16, size=(1, 3, 1, 1)).astype(np.float32)
            c_fac = float(rng.uniform(0.80, 1.25))
            mean = rgb_s.mean(axis=(2, 3), keepdims=True)
            rgb_s = np.clip(((rgb_s - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
            if rng.random() < 0.30:
                rgb_s = np.clip(rgb_s + rng.normal(0.0, float(rng.uniform(0.005, 0.022)), rgb_s.shape).astype(np.float32), 0.0, 1.0)

        is_mono = rng.random() < 0.35
        if is_mono:
            luma = 0.299 * rgb_s[:, 0:1] + 0.587 * rgb_s[:, 1:2] + 0.114 * rgb_s[:, 2:3]
            sat = 0.0 if rng.random() < 0.65 else float(rng.uniform(0.0, 0.25))
            rgb_s = np.clip(sat * rgb_s + (1.0 - sat) * luma, 0.0, 1.0)

        if rng.random() < 0.20:
            for ti in range(self.clip_len):
                hwc = cv2.GaussianBlur(rgb_s[ti].transpose(1, 2, 0), (3, 3), 0.9)
                rgb_s[ti] = hwc.transpose(2, 0, 1)

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
        is_arkit = 1.0 if dom == "arkit" else 0.0

        return {
            "rgb_t": torch.from_numpy(rgb_t.astype(np.float16)),
            "K_t": torch.from_numpy(K_t_seq),
            "R_t": torch.from_numpy(R_clean),
            "t_t": torch.from_numpy(t_clean),
            "fwd_t": torch.from_numpy(fwd_clean),
            "seg_t": torch.from_numpy(seg_clean),
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
            "is_arkit": torch.tensor(is_arkit, dtype=torch.float32),
        }


def compute_loss_v7(
    out_s: dict[str, torch.Tensor],
    out_t6: dict[str, torch.Tensor] | None,
    out_t5: dict[str, torch.Tensor] | None,
    batch: dict[str, torch.Tensor],
    kd_weight: float = 0.30,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unified Metric-Normal BEV + Monotonic Survival + Top-25% Clear-Corridor Tail-OHEM + Decaying Dual-Teacher KD."""
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

    # 1. Static BEV Focal BCE (boosted BODY/HEAD weight to lift SANPO-Real recall!)
    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.5, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w).mean()

    # 2. Dynamic BEV BCE (100% clean full-seg scenes -> strong pedestrian suppression!)
    loss_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none").mean()

    # 3. Monotonic Cumulative Corridor Survival Focal BCE (balanced pos_w=2.0 to avoid indoor over-firing!)
    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=device, dtype=torch.float32)
    idx_4m = int((z_edges <= H_CORR + 1e-4).sum().item()) - 1
    z_w = torch.where(z_edges <= H_CORR + 0.2, 2.0, 0.5)[None, None, :]

    pos_w = 2.0
    bce_cum = -(pos_w * y_co * torch.log(p_cum) + (1.0 - y_co) * torch.log(1.0 - p_cum))
    pt_cum = y_co * p_cum + (1.0 - y_co) * (1.0 - p_cum)
    m_co = (v_co * dn_w)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.5) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    # 4. Exact Discrete First-Hit Survival NLL
    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w)[:, :, None]
    soft_h = F.softplus(haz_log)
    log_p_hit = torch.log(1.0 - torch.exp(-soft_h).clamp(max=1.0 - 1e-6))
    nll_fh = -y_first * log_p_hit + (1.0 - y_first) * soft_h
    loss_fh = (nll_fh * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)

    # 5. Uncertainty calibration BCE
    err_target = ((p_cum.detach() > 0.5).float() != y_co).float()
    loss_un = (F.binary_cross_entropy_with_logits(un_log, err_target, reduction="none") * v_co[:, :, None]).sum() / (
        v_co.sum().clamp(min=1.0) * BEV_NZ
    )

    # 6. Clear-Corridor Temporal Flicker + Top-25% Hard-Negative Tail-OHEM (directly crushes @120 FA tax!)
    clear_fr = (v_co > 0.5) & (y_co[:, :, idx_4m] < 0.5)
    p4 = p_cum[:, :, idx_4m]
    clear_pair = clear_fr[:, 1:] & clear_fr[:, :-1]
    loss_tv = (p4[:, 1:] - p4[:, :-1]).abs()[clear_pair].mean() if clear_pair.any() else torch.tensor(0.0, device=device)

    if clear_fr.any():
        clear_p4 = p4[clear_fr]
        # Base margin above 0.08 + Top-25% hardest clear-corridor spikes penalized strongly!
        exc = F.relu(clear_p4 - 0.08) ** 2
        k_hard = max(1, int(0.25 * exc.numel()))
        top_exc, _ = torch.topk(exc, k_hard)
        loss_ohem = exc.mean() + 3.0 * top_exc.mean()
    else:
        loss_ohem = torch.tensor(0.0, device=device)

    # 7. Complementary Dual-Teacher Clean-KD (decays over training so MoGe-2 metric/normal geometry can surpass V6-Base!)
    loss_kd = torch.tensor(0.0, device=device)
    if kd_weight > 1e-4 and out_t6 is not None and out_t5 is not None:
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

    total = (
        1.20 * loss_st
        + 0.45 * loss_dy
        + 1.80 * loss_cum
        + 0.65 * loss_fh
        + 0.20 * loss_un
        + 0.40 * loss_tv
        + 1.60 * loss_ohem
        + kd_weight * loss_kd
    )
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "dy": float(loss_dy.item()),
        "co": float(loss_cum.item()),
        "ohem": float(loss_ohem.item()),
        "kd": float(loss_kd.item()),
    }


def eval_dark_domain(model: EgoClearV7Net, scenes: dict[str, dict], eval_rows_ov: dict | None, bw_speeds: dict, device: torch.device, gain: float = 0.18) -> dict:
    """Evaluates low-light (0.18x luma + sensor noise) stability on real scenes without modifying original dicts."""
    import copy

    dark_scenes = {}
    for sc, d in scenes.items():
        dd = dict(d)
        n = len(d["eval_rows"])
        rgb = decode_rgb_batch(d, np.arange(n))
        rng = np.random.default_rng(1234)
        noise = rng.normal(0.0, 0.025, rgb.shape).astype(np.float32)
        rgb_d = np.clip((rgb ** 1.35) * gain + noise, 0.0, 1.0)
        # Re-encode as JPEG blob matching decode_rgb_batch format
        bufs, offs = [], [0]
        for i in range(n):
            bgr = cv2.cvtColor((rgb_d[i].transpose(1, 2, 0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            assert ok
            b = enc.tobytes()
            bufs.append(b)
            offs.append(offs[-1] + len(b))
        dd["rgb_blob"] = np.frombuffer(b"".join(bufs), dtype=np.uint8)
        dd["rgb_offsets"] = np.array(offs, dtype=np.int64)
        dark_scenes[sc] = dd
    return eval_v6_all(model, dark_scenes, eval_rows_ov, bw_speeds, device, compute_uncertainty_audit=True)


def print_block(tag: str, dom: str, ev: dict):
    raw_mode = ev["modes"]["raw"]
    rm = raw_mode["summary"]
    p80 = fmt_m(raw_mode, "late_at_fa_fr<=80.0")
    p120 = fmt_m(raw_mode, "late_at_fa_fr<=120.3")
    p200 = fmt_m(raw_mode, "late_at_fa_fr<=200.0")
    p350 = fmt_m(raw_mode, "late_at_fa_fr<=350.0")
    p500 = fmt_m(raw_mode, "late_at_fa_fr<=500.0")
    hb = ev["hband_corridor_f1_at_0.5"]
    dyn = ev["dyn_only_alert_rate_at_0.5"]
    ds = f"{dyn['rate']*100:.2f}%({dyn['fp']}/{dyn['total']})" if dyn["total"] > 0 else "N/A"
    thr120 = rm["late_at_fa_fr<=120.3"]["thr"] if rm.get("late_at_fa_fr<=120.3") else -1.0
    print(
        f"  [{tag:10s}] {dom:18s} | @80:{p80} | @120:{p120}(thr={thr120:.2f}) | @200:{p200} | @350:{p350} | @500:{p500} | "
        f"HEAD F1={hb['HEAD']['f1']:.3f}(R={hb['HEAD']['recall']:.2f}) | dyn={ds} | mono={ev['monotonicity_violation_rate']*100:.3f}%",
        flush=True,
    )


def export_v7_onnx_fp16(model: EgoClearV7Net, out_dir: Path) -> dict:
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16 as onnx_fp16

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32 = out_dir / "egoclear_v7_encoder_fp32.onnx"
    stp_fp32 = out_dir / "egoclear_v7_step_fp32.onnx"
    enc_fp16 = out_dir / "egoclear_v7_encoder_fp16.onnx"
    stp_fp16 = out_dir / "egoclear_v7_step_fp16.onnx"

    m_cpu = model.eval().float().cpu()
    enc_w = EgoClearV7EncoderExport(m_cpu).eval()
    stp_w = EgoClearV7StepExport(m_cpu).eval()

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
                t4_d = torch.tensor([[-0.75, 0.0, 1.5]], dtype=torch.float32)
                fwd_d = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                ds_d = torch.tensor([0.25], dtype=torch.float32)
                mem_d = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
                age_d = torch.full((1, 1, BEV_NZ, BEV_NX), 10.0)
                stp_inputs = (
                    f2d_d, g_d, f2d_d, g_d, f2d_d, g_d, f2d_d, g_d,
                    K_d, R_d, t_d, fwd_d, K_d, R_d, t1_d, fwd_d,
                    K_d, R_d, t2_d, K_d, R_d, t4_d, ds_d, mem_d, age_d,
                )
                stp_names = [
                    "f2d_t", "geom_t", "f2d_1", "geom_1", "f2d_2", "geom_2", "f2d_4", "geom_4",
                    "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1",
                    "K_2", "R_2", "t_2", "K_4", "R_4", "t_4", "ds_t", "mem_prev", "age_prev",
                ]
                torch.onnx.export(
                    stp_w, stp_inputs, str(stp_fp32),
                    input_names=stp_names,
                    output_names=["static_logits", "dynamic_logits", "corr_probs", "unc_logits", "mem_new", "age_new"],
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
    finally:
        F.interpolate = orig_interp

    # Convert both to FP16 keeping IO fp32 + survival ops safe
    m_enc = onnx_fp16.convert_float_to_float16(onnx.load(str(enc_fp32)), keep_io_types=True)
    onnx.save(m_enc, str(enc_fp16))
    m_stp = onnx_fp16.convert_float_to_float16(
        onnx.load(str(stp_fp32)),
        keep_io_types=True,
        op_block_list=["Softplus", "CumSum", "Exp", "GridSample", "Resize", "Sqrt", "ReduceSum"],
    )
    onnx.save(m_stp, str(stp_fp16))

    # Parity & monotonicity check in ONNX Runtime CPU
    s_e32 = ort.InferenceSession(str(enc_fp32), providers=["CPUExecutionProvider"])
    s_e16 = ort.InferenceSession(str(enc_fp16), providers=["CPUExecutionProvider"])
    s_s32 = ort.InferenceSession(str(stp_fp32), providers=["CPUExecutionProvider"])
    s_s16 = ort.InferenceSession(str(stp_fp16), providers=["CPUExecutionProvider"])

    f32, g32 = s_e32.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    f16, g16 = s_e16.run(None, {"rgb": rgb_d.numpy(), "K_scaled": K_d.numpy(), "R_c2w": R_d.numpy()})
    feed32 = dict(zip(stp_names, [t.numpy() for t in stp_inputs]))
    feed32["f2d_t"] = f32; feed32["geom_t"] = g32
    feed16 = dict(feed32)
    feed16["f2d_t"] = f16; feed16["geom_t"] = g16
    o32 = s_s32.run(None, feed32)
    o16 = s_s16.run(None, feed16)
    cp32, cp16 = o32[2][0], o16[2][0]
    mono_ok = bool(np.all(np.diff(cp16) >= -1e-6))
    max_err = float(np.max(np.abs(cp32 - cp16)))
    enc_mb = enc_fp16.stat().st_size / (1024 * 1024)
    stp_mb = stp_fp16.stat().st_size / (1024 * 1024)
    print(
        f"[v7-onnx] EXPORT PASS! enc_fp16={enc_mb:.2f}MB + step_fp16={stp_mb:.2f}MB = {enc_mb+stp_mb:.2f}MB | "
        f"fp16_max_abs_err={max_err:.5f} | mono_ok={mono_ok}",
        flush=True,
    )
    return {
        "enc_fp32_mb": round(enc_fp32.stat().st_size / (1024 * 1024), 2),
        "stp_fp32_mb": round(stp_fp32.stat().st_size / (1024 * 1024), 2),
        "enc_fp16_mb": round(enc_mb, 2),
        "stp_fp16_mb": round(stp_mb, 2),
        "total_fp16_mb": round(enc_mb + stp_mb, 2),
        "fp16_max_abs_err": round(max_err, 6),
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
    print(f"[v7] GPU={torch.cuda.get_device_name(0)}", flush=True)

    # 1. Stage all cached datasets (1,094 train + 5 eval domains)
    paths = stage_all_p8_datasets_parallel()
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

    real_seg_train = []
    for key in ("real_75", "real_22", "real_ch70"):
        real_seg_train.extend(load_scene(p) for p in sorted(paths[key].glob("*.npz")))
    arkit_train = []
    for key in ("arkit_90", "arkit_210"):
        arkit_train.extend(load_scene(p) for p in sorted(paths[key].glob("*.npz")))

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}
    eval_coda = {p.stem: load_scene(p) for p in sorted(CODA_DIR.glob("*.npz"))} if CODA_DIR.exists() else {}

    print(
        f"[v7] Loaded {len(synth_train)+len(real_seg_train)+len(arkit_train)} clean train scenes "
        f"(synth={len(synth_train)}, real_seg={len(real_seg_train)}, arkit={len(arkit_train)}) + "
        f"5 eval sets (synth={len(eval_synth)}, real={len(eval_real)}, aria={len(aria_dicts)}, arkit={len(eval_arkit)}, coda={len(eval_coda)})",
        flush=True,
    )

    # 2. Load frozen Dual Teachers (V6-Base + V5-Small Anchor) for decaying Clean-KD
    t6_ckpt = LOCAL_RUNS / "egoclear_v6_base_full.pt"
    t5_ckpt = LOCAL_RUNS / "egoclear_v5_full.pt"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase7_v6/egoclear_v6_base_full.pt", t6_ckpt)
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase6_v5/egoclear_v5_full.pt", t5_ckpt)

    teacher_v6 = EgoClearV6Net(dav2_id="depth-anything/Depth-Anything-V2-Base-hf", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_dav2=True).to(device).eval()
    teacher_v6.load_state_dict(torch.load(t6_ckpt, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher_v6.parameters():
        p.requires_grad = False

    anchor_v5 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device).eval()
    anchor_v5.load_state_dict(torch.load(t5_ckpt, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in anchor_v5.parameters():
        p.requires_grad = False

    # 3. Build EgoClear-v7 (MoGe-2-ViT-S-Normal 35M MIT + 84x148 Metric-Normal Voxel Consistency + 3-Lag 1,2,4)
    student = EgoClearV7Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_moge=True).to(device)
    n_tot = sum(p.numel() for p in student.parameters())
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[v7] EgoClearV7Net built: total={n_tot/1e6:.2f}M, trainable_head={n_tr/1e6:.2f}M (MoGe-2 35.1M frozen)", flush=True)

    # 4. Train (16 epochs x 2,880 clips, bs=20 x 12 = 240 fr/step on 96 GB Blackwell G4)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bs = 20 if vram_gb >= 65.0 else 10
    epochs = 16
    ds = V7MetricNormalDataset(synth_train, real_seg_train, arkit_train, clip_len=12, clips_per_epoch=2880)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3.0e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1.5e-5)

    history = []
    t_tr = time.time()
    print(f"\n=== Training EgoClear-v7 ({epochs} ep, bs={bs}x12={bs*12} fr/step, lr=3e-4, decaying KD 0.35->0.10 + Tail-OHEM) ===", flush=True)
    for ep in range(1, epochs + 1):
        student.train()
        kd_w = 0.35 - 0.25 * ((ep - 1) / max(1, epochs - 1))  # 0.35 -> 0.10
        ep_l = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0, "ohem": 0.0, "kd": 0.0}
        te = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            rgb_t = batch["rgb_t"].float()
            rgb_s = batch["rgb_s"].float()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_t6 = teacher_v6.forward_sequence(rgb_t, batch["K_t"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"])
                out_t5 = anchor_v5.forward_sequence(rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"])

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(rgb_s, batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"])
                loss, ld = compute_loss_v7(out_s, out_t6, out_t5, batch, kd_weight=kd_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 2.0)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += ld[k]
        nb = len(loader)
        for k in ep_l:
            ep_l[k] /= nb
        ep_l["ep"] = ep
        ep_l["kd_w"] = round(kd_w, 3)
        ep_l["sec"] = round(time.time() - te, 1)
        history.append(ep_l)
        print(
            f"  [v7 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} st={ep_l['st']:.4f} dy={ep_l['dy']:.4f} "
            f"co={ep_l['co']:.4f} ohem={ep_l['ohem']:.4f} kd={ep_l['kd']:.4f}(w={kd_w:.2f}) ({ep_l['sec']:.1f}s)",
            flush=True,
        )
        # Mid-flight checkpoint & real_40 + aria quick sanity at ep 8
        if ep == 8:
            student.eval()
            ev_r8 = eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=False)
            ev_a8 = eval_v6_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=False)
            print_block("V7-ep08", "real_40_local", ev_r8)
            print_block("V7-ep08", "aria_g2_zeroshot", ev_a8)
            torch.save(
                {"head_state_dict": {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")}, "ep": ep},
                LOCAL_RUNS / "egoclear_v7_ep08.pt",
            )

    train_sec = round(time.time() - t_tr, 1)
    # Free teachers before full eval & ONNX export
    del teacher_v6, anchor_v5
    torch.cuda.empty_cache()

    ckpt_path = LOCAL_RUNS / "egoclear_v7_moge2_small.pt"
    head_sd = {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("moge.")}
    torch.save({"head_state_dict": head_sd, "moge_id": student.moge_id, "history": history}, ckpt_path)
    shutil.copy2(ckpt_path, DRIVE_RUNS / ckpt_path.name)

    # 5. Full 5-Domain + Dark-Room Evaluation (exact apples-to-apples with EgoClear-v6.6!)
    print("\n=== Full 5-Domain + Dark-Room Evaluation of EgoClear-v7 ===", flush=True)
    student.eval()
    evals = {}
    evals["synth_clean"] = eval_v6_all(student, eval_synth, None, bw_speeds, device, compute_uncertainty_audit=True)
    print_block("EgoClear-v7", "synth_clean", evals["synth_clean"])

    evals["synth_vio_1x"] = eval_v6_all(student, eval_synth, None, bw_speeds, device, vio_scale=1.0, compute_uncertainty_audit=False)
    print_block("EgoClear-v7", "synth_vio_1x", evals["synth_vio_1x"])

    evals["real_40_local"] = eval_v6_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=True)
    print_block("EgoClear-v7", "real_40_local", evals["real_40_local"])

    evals["aria_g2_zeroshot"] = eval_v6_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=True)
    print_block("EgoClear-v7", "aria_g2_zeroshot", evals["aria_g2_zeroshot"])

    evals["arkit_indoor_val30"] = eval_v6_all(student, eval_arkit, None, bw_speeds, device, compute_uncertainty_audit=True)
    print_block("EgoClear-v7", "arkit_indoor_val30", evals["arkit_indoor_val30"])

    if eval_coda:
        evals["coda_campus_zeroshot"] = eval_v6_all(student, eval_coda, None, bw_speeds, device, compute_uncertainty_audit=True)
        print_block("EgoClear-v7", "coda_campus_zeroshot", evals["coda_campus_zeroshot"])

    evals["arkit_dark018"] = eval_dark_domain(student, eval_arkit, None, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v7", "arkit_dark018", evals["arkit_dark018"])

    evals["real_40_dark018"] = eval_dark_domain(student, eval_real, real_local_rows, bw_speeds, device, gain=0.18)
    print_block("EgoClear-v7", "real_40_dark018", evals["real_40_dark018"])

    # 6. Export verified ONNX-17 FP32 + FP16 and sync everything to Drive
    onnx_info = export_v7_onnx_fp16(student, LOCAL_RUNS / "onnx")
    for fn in ["egoclear_v7_encoder_fp16.onnx", "egoclear_v7_step_fp16.onnx", "egoclear_v7_encoder_fp32.onnx", "egoclear_v7_step_fp32.onnx"]:
        src = LOCAL_RUNS / "onnx" / fn
        if src.exists():
            shutil.copy2(src, DRIVE_RUNS / fn)

    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "model": "EgoClear-v7 (MoGe-2-ViT-S-Normal 35M + 84x148 Metric-Normal Voxel Consistency + 3-Lag 1,2,4 + Tail-OHEM)",
        "total_params_m": round(n_tot / 1e6, 2),
        "trainable_params_m": round(n_tr / 1e6, 2),
        "train_sec": train_sec,
        "total_wall_sec": round(time.time() - t_all, 1),
        "history": history,
        "evals": evals,
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase11_v7_results.json"
    res_drive = DRIVE_RUNS / "phase11_v7_results.json"
    res_local.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_local, res_drive)
    print(f"\n[v7] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
