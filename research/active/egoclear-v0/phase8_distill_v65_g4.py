"""Phase 8 G4 (96 GB Blackwell): EgoClear-v6.5 Privileged Cross-Architecture Knowledge Distillation.

Teacher (frozen, eval, clean-pose, 202.8 MB FP16):
  - EgoClear-v6-Base (frozen DAV2-Base 97.5M + 3-lag 1,2,4 multi-scale stereo)
  - Real outdoor SOTA: 29.7% @fr<=200, 18.9% @fr<=500; Synth 0.7% @120; Indoor 0.0% @200!
Student (warm-started from EgoClear-v5, 57.22 MB FP16, 3.9 ms/fr):
  - EgoClear-v4/v5 (frozen DAV2-Small 25M + 2-lag 1,2 + Monotonic First-Hit Survival)
  - Already strong on Aria Gen 2 zero-shot (53.9% @120), VIO 1x (5.1%), Real dyn@0.5 (11.1%), Indoor HEAD F1 (0.532).

Privileged Distillation Mechanism (Best of BOTH Worlds!):
  1. Teacher sees CLEAN (unperturbed) trajectories & RGB; Student sees AUGMENTED RGB (30% monochrome/low-sat +
     45% exact Virtual Pan-Tilt head-turn) & 50% VIO-drifted trajectories!
  2. Dynamic-Safe Distillation Gate:
     - Soft corridor survival KL/BCE distillation + BEV static spatial distillation are STRICTLY masked out
       wherever ground-truth dynamic pedestrians exist (or where student dynamic detector fires > 0.20 on
       seg-free head cams), preventing teacher's dynamic-pedestrian leakage from polluting the student!
  3. 1,194-scene unified pool (36% Synth + 36% Real-Seg [167 full-seg!] + 10% Real-Head-Noseg + 18% Indoor [300]).
  4. Exports final 57.22 MB egoclear_v65_distilled_{encoder,step}_fp16.onnx with verified 0.0 monotonic parity!
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net, EgoClearV4StepExport, H_CORR
from egoclear_v6_model import EgoClearV6Net
from eval_closed_loop import H
from phase4_simreal_aria_onnx_g4 import EgoClearV3EncoderExport
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase5b_calibrated_v41_g4 import _copy_and_extract_one
from phase6_train_v5_g4 import apply_virtual_pan_tilt, eval_v5_all, fmt_m
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase8_v65"
LOCAL_RUNS = Path("/content/egoclear_runs_p8_v65")


def stage_all_p8_datasets_parallel() -> dict[str, Path]:
    """8-thread parallel Drive-to-/content staging for ALL 10 cached datasets (1,194 train + 4 eval domains!)."""
    t0 = time.time()
    specs = [
        ("synth_287", DRIVE_ROOT / "datasets/sanpo_synth_287_v0", Path("/content/sanpo_287/scenes"), 280),
        ("synth_450", DRIVE_ROOT / "datasets/sanpo_synth_extra450_v0", Path("/content/sanpo_p2/extra_scenes"), 440),
        ("real_40", DRIVE_ROOT / "datasets/sanpo_real_zeroshot_40_v0", Path("/content/sanpo_p2/real_scenes"), 38),
        ("real_75", DRIVE_ROOT / "datasets/sanpo_real_train76_v0", Path("/content/sanpo_p4/real_train_scenes"), 70),
        ("real_22", DRIVE_ROOT / "datasets/sanpo_real_extra444_v0", Path("/content/p6_data/real_extra444"), 20),
        ("real_ch70", DRIVE_ROOT / "datasets/sanpo_real_chest240_v0", Path("/content/p7_data/real_chest240"), 65),
        ("real_hd100", DRIVE_ROOT / "datasets/sanpo_real_head160_noseg_v0", Path("/content/p7_data/real_head160"), 95),
        ("arkit_90", DRIVE_ROOT / "datasets/arkit_indoor_train90_v0", Path("/content/arkit_p5/train_scenes"), 85),
        ("arkit_210", DRIVE_ROOT / "datasets/arkit_indoor_extra210_v0", Path("/content/p6_data/arkit_extra210"), 200),
        ("arkit_va30", DRIVE_ROOT / "datasets/arkit_indoor_val30_v0", Path("/content/arkit_p5/val_scenes"), 28),
    ]
    jobs, paths = [], {}
    for key, d_dir, l_dir, min_n in specs:
        l_dir.mkdir(parents=True, exist_ok=True)
        paths[key] = l_dir
        if len(list(l_dir.glob("*.npz"))) >= min_n:
            continue
        if not d_dir.exists():
            continue
        man_p = d_dir / "manifest_sha256.json"
        if man_p.exists():
            man = json.loads(man_p.read_text())
            sh_files = [d_dir / sh["file"] for sh in man["shards"]]
        else:
            sh_files = sorted(d_dir.glob("*.tar"))
        for sf in sh_files:
            if sf.exists():
                jobs.append((sf, l_dir))

    aria_dir = Path("/content/aria_g2_scenes")
    aria_dir.mkdir(parents=True, exist_ok=True)
    paths["aria"] = aria_dir
    if len(list(aria_dir.glob("*.npz"))) < 18:
        jobs.append((DRIVE_ROOT / "caches/aria_g2_20clips.tar", aria_dir))

    if jobs:
        print(f"[stage-p8] copying & extracting {len(jobs)} tar shards across 8 threads...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_copy_and_extract_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                name, sec = fut.result()
                done += 1
                if done in (1, len(jobs) // 2, len(jobs)):
                    print(f"  [stage-p8] {done}/{len(jobs)} done ({name} {sec:.1f}s, elapsed {time.time()-t0:.1f}s)", flush=True)
    print(f"[stage-p8] ALL 11 datasets ready locally in {time.time()-t0:.1f}s!", flush=True)
    return paths


class PrivilegedDistillDataset(torch.utils.data.Dataset):
    """Returns BOTH clean-pose/RGB inputs for privileged Teacher (V6-Base) AND augmented/VIO-drifted inputs for Student (V5-Small)."""

    def __init__(
        self,
        synth_dicts: list[dict],
        real_seg_dicts: list[dict],
        real_head_noseg_dicts: list[dict],
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
            "real_head": [d for d in real_head_noseg_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "arkit": [d for d in arkit_dicts if len(d["eval_rows"]) >= clip_len + 1],
        }
        print(
            f"[dataset-v6.5] pools: synth={len(self.pools['synth'])} (36%), "
            f"real_seg={len(self.pools['real_seg'])} (36%), "
            f"real_head={len(self.pools['real_head'])} (10%), "
            f"arkit={len(self.pools['arkit'])} (18%)",
            flush=True,
        )

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.36:
            dom = "synth"
        elif u_dom < 0.72 and len(self.pools["real_seg"]) > 0:
            dom = "real_seg"
        elif u_dom < 0.82 and len(self.pools["real_head"]) > 0:
            dom = "real_head"
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
        if len(haz_idx) > 0 and u_pick < 0.52:
            anchor = int(haz_idx[rng.integers(len(haz_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(dyn_only_idx) > 0 and u_pick < 0.70:
            anchor = int(dyn_only_idx[rng.integers(len(dyn_only_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(clear_idx) > 0 and u_pick < 0.88:
            anchor = int(clear_idx[rng.integers(len(clear_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        else:
            st = int(rng.integers(0, n - self.clip_len + 1))
        sl = np.arange(st, st + self.clip_len)

        rgb_raw = decode_rgb_batch(d, sl)
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_base = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)

        # Clean inputs for privileged Teacher (V6-Base)
        rgb_t = (rgb_raw - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_t_seq = np.repeat(K_base[None], self.clip_len, axis=0)
        R_clean = d["poses_R"][sl].astype(np.float32)
        t_clean = d["poses_t"][sl].astype(np.float32)
        fwd_clean = d["causal_fwd"][sl].astype(np.float32)
        seg_clean = np.r_[0.0, np.linalg.norm(np.diff(t_clean[:, :2], axis=0), axis=1)].astype(np.float32)

        # Strong augmented inputs for Student (V5-Small): focal zoom + virtual pan-tilt + 30% monochrome + 50% VIO drift!
        rgb_s = rgb_raw.copy()
        K_s = K_base.copy()
        if rng.random() < 0.65:
            rgb_s, K_s = apply_focal_zoom_crop(rgb_s, K_s, float(rng.uniform(0.68, 1.22)))
        if rng.random() < 0.45:
            sh_x = int(rng.integers(-int(0.16 * RGB_W), int(0.16 * RGB_W) + 1))
            sh_y = int(rng.integers(-int(0.10 * RGB_H), int(0.10 * RGB_H) + 1))
            rgb_s, K_s = apply_virtual_pan_tilt(rgb_s, K_s, sh_x, sh_y)

        gamma = float(rng.uniform(0.70, 1.42))
        rgb_s = np.power(np.clip(rgb_s, 1e-4, 1.0), gamma)
        ch_gain = rng.uniform(0.84, 1.16, size=(1, 3, 1, 1)).astype(np.float32)
        c_fac = float(rng.uniform(0.80, 1.25))
        mean = rgb_s.mean(axis=(2, 3), keepdims=True)
        rgb_s = np.clip(((rgb_s - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
        if rng.random() < 0.30:
            # 30% monochrome/low-saturation augmentation -> strengthens Aria 79-deg grayscale glasses zero-shot!
            luma = (0.299 * rgb_s[:, 0:1] + 0.587 * rgb_s[:, 1:2] + 0.114 * rgb_s[:, 2:3])
            sat = float(rng.uniform(0.0, 0.35))
            rgb_s = np.clip(sat * rgb_s + (1.0 - sat) * luma, 0.0, 1.0)
        if rng.random() < 0.30:
            rgb_s = np.clip(rgb_s + rng.normal(0.0, float(rng.uniform(0.005, 0.022)), rgb_s.shape).astype(np.float32), 0.0, 1.0)
        if rng.random() < 0.20:
            for t_i in range(self.clip_len):
                rgb_s[t_i] = cv2.GaussianBlur(rgb_s[t_i].transpose(1, 2, 0), (3, 3), 0.8).transpose(2, 0, 1)

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
        ambig_subvox = (np.isinf(D_gt)) & (in_corr_vox > 0.5)
        dyn_neg_w = np.where((np.isinf(D_gt)) & (dyn_fl > 0.5), 2.5, np.where(ambig_subvox, 0.15, 1.0)).astype(np.float32)
        has_dyn = 0.0 if (dom == "real_head" or ("has_dyn_gt" in d and int(d["has_dyn_gt"]) == 0)) else 1.0
        is_dyn_frame = (dyn_fl > 0.5).astype(np.float32)

        return {
            "rgb_t": torch.from_numpy(rgb_t), "K_t": torch.from_numpy(K_t_seq),
            "R_t": torch.from_numpy(R_clean), "t_t": torch.from_numpy(t_clean),
            "fwd_t": torch.from_numpy(fwd_clean), "seg_t": torch.from_numpy(seg_clean),
            "rgb_s": torch.from_numpy(rgb_s), "K_s": torch.from_numpy(K_s_seq),
            "R_s": torch.from_numpy(R_s), "t_s": torch.from_numpy(t_s),
            "fwd_s": torch.from_numpy(fwd_s), "seg_s": torch.from_numpy(seg_s),
            "gt_static": torch.from_numpy(gt_stat), "gt_dynamic": torch.from_numpy(gt_dyn),
            "y_corr": torch.from_numpy(y_corr), "valid_corr": torch.from_numpy(valid_corr),
            "dyn_neg_w": torch.from_numpy(dyn_neg_w), "has_dyn_gt": torch.tensor(has_dyn, dtype=torch.float32),
            "is_dyn_frame": torch.from_numpy(is_dyn_frame),
        }


def compute_loss_v65_distill(
    out_s: dict[str, torch.Tensor],
    out_t: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """Dynamic-Safe Privileged Distillation + Hard V5 Monotonic Survival & Flicker Supervision."""
    st_s = out_s["static_logits"].float()
    dy_s = out_s["dynamic_logits"].float()
    haz_s = out_s["hazard_logits"].float()
    p_s = out_s["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
    un_s = out_s["unc_logits"].float()

    with torch.no_grad():
        p_t = out_t["corr_probs"].float().clamp(1e-5, 1.0 - 1e-5)
        st_t_prob = torch.sigmoid(out_t["static_logits"].float())
        # Student's own dynamic confidence to protect seg-free head clips from dynamic pedestrian distillation!
        s_dyn_max = torch.sigmoid(dy_s).amax(dim=(2, 3, 4))  # (B, T)

    gt_st = batch["gt_static"].float()
    gt_dy = batch["gt_dynamic"].float()
    y_co = batch["y_corr"].float()
    v_co = batch["valid_corr"].float()
    dn_w = batch["dyn_neg_w"].float()
    has_dyn = batch["has_dyn_gt"].float()      # (B,)
    is_dyn = batch["is_dyn_frame"].float()     # (B, T)

    device = st_s.device
    col_w = torch.ones(BEV_NX, device=device, dtype=torch.float32)
    col_w[6:9] = 3.0; col_w[5] = 1.5; col_w[9] = 1.5
    hb_w = torch.tensor([1.0, 1.2, 2.5], device=device, dtype=torch.float32)[None, None, :, None, None]
    spat_w = hb_w * col_w[None, None, None, None, :]

    # 1. Hard static BEV focal BCE (masked by has_dyn on seg-free head only where student doesn't detect moving pedestrian)
    safe_st_frame = torch.where(has_dyn[:, None] > 0.5, torch.ones_like(is_dyn), (s_dyn_max < 0.25).float())
    bce_st = F.binary_cross_entropy_with_logits(st_s, gt_st, pos_weight=torch.tensor(4.0, device=device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_s, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * spat_w * safe_st_frame[:, :, None, None, None]).mean()

    # 2. Hard dynamic BEV BCE (strictly only on full-seg clips with has_dyn == 1.0 -> preserves 11% real dyn accuracy!)
    bce_dy = F.binary_cross_entropy_with_logits(dy_s, gt_dy, pos_weight=torch.tensor(4.0, device=device), reduction="none")
    dy_m = has_dyn[:, None, None, None, None].expand_as(bce_dy)
    loss_dy = (bce_dy * dy_m).sum() / dy_m.sum().clamp(min=1.0)

    # 3. Hard Monotonic Survival Corridor BCE + First-Hit NLL
    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=device, dtype=torch.float32)
    idx_4m = int((z_edges <= H_CORR + 1e-4).sum().item()) - 1
    z_w = torch.where(z_edges <= H_CORR + 0.2, 2.0, 1.2)[None, None, :]

    y_sm = y_co * 0.97 + 0.015
    bce_cum = -(2.2 * y_sm * torch.log(p_s) + (1.0 - y_sm) * torch.log(1.0 - p_s))
    pt_cum = y_co * p_s + (1.0 - y_co) * (1.0 - p_s)
    m_co = (v_co * dn_w * safe_st_frame)[:, :, None] * z_w
    loss_cum = (((1.0 - pt_cum) ** 1.2) * bce_cum * m_co).sum() / m_co.sum().clamp(min=1.0)

    y_prev = F.pad(y_co[..., :-1], (1, 0), value=0.0)
    y_first = (y_co > 0.5).float() * (y_prev < 0.5).float()
    mask_fh = (1.0 - y_prev) * (v_co * dn_w * safe_st_frame)[:, :, None]
    bce_fh = F.binary_cross_entropy_with_logits(haz_s, y_first, pos_weight=torch.tensor(2.6, device=device), reduction="none")
    loss_fh = (bce_fh * mask_fh).sum() / mask_fh.sum().clamp(min=1.0)
    loss_co = loss_cum + 0.30 * loss_fh

    # 4. Dynamic-Safe Privileged Teacher Distillation (Corridor Survival BCE + BEV Static Spatial MSE!)
    # Strictly gated: ONLY distills where NO dynamic pedestrian is present (1 - is_dyn) AND safe_st_frame == 1!
    kd_gate = v_co * (1.0 - is_dyn) * safe_st_frame  # (B, T)
    kd_bce = -(p_t * torch.log(p_s) + (1.0 - p_t) * torch.log(1.0 - p_s))
    kd_m = kd_gate[:, :, None] * z_w
    loss_kd_corr = (kd_bce * kd_m).sum() / kd_m.sum().clamp(min=1.0)

    st_s_prob = torch.sigmoid(st_s)
    bev_kd_mask = (1.0 - gt_dy) * kd_gate[:, :, None, None, None] * spat_w
    loss_kd_bev = (((st_s_prob - st_t_prob).pow(2)) * bev_kd_mask).sum() / bev_kd_mask.sum().clamp(min=1.0)
    loss_kd = 0.70 * loss_kd_corr + 0.30 * loss_kd_bev

    # 5. Clear-frame flicker suppression
    p4 = p_s[:, :, idx_4m]
    y4 = y_co[:, :, idx_4m]
    clear = v_co * (1.0 - y4) * (dn_w >= 0.5).float() * safe_st_frame
    clear_pair = clear[:, 1:] * clear[:, :-1]
    loss_clear_tv = ((p4[:, 1:] - p4[:, :-1]).abs() * clear_pair).sum() / clear_pair.sum().clamp(min=1.0)
    loss_clear_spike = (F.relu(p4 - 0.18).pow(2) * clear * dn_w).sum() / (clear * dn_w).sum().clamp(min=1.0)
    v3 = v_co[:, 2:] * v_co[:, 1:-1] * v_co[:, :-2]
    jerk = ((p4[:, 2:] - p4[:, 1:-1]) - (p4[:, 1:-1] - p4[:, :-2])).abs()
    loss_jerk = (jerk * v3).sum() / v3.sum().clamp(min=1.0)
    loss_flk = 0.50 * loss_clear_tv + 0.40 * loss_clear_spike + 0.15 * loss_jerk

    with torch.no_grad():
        err_tgt = ((p_s - y_co).abs() > 0.35).float()
    bce_un = F.binary_cross_entropy_with_logits(un_s, err_tgt, reduction="none")
    loss_un = (bce_un * m_co).sum() / m_co.sum().clamp(min=1.0)

    total = loss_st + 0.50 * loss_dy + 1.5 * loss_co + 0.85 * loss_kd + 0.35 * loss_flk + 0.2 * loss_un
    return total, {
        "tot": float(total.item()), "st": float(loss_st.item()),
        "dy": float(loss_dy.item()), "co": float(loss_co.item()),
        "kd": float(loss_kd.item()), "flk": float(loss_flk.item()),
    }


def export_v65_onnx_fp16(model: EgoClearV4Net, out_dir: Path) -> dict:
    import onnxruntime as ort
    from edge_quant_and_replay import convert_onnx_weights_to_fp16_safe

    m_cpu = model.cpu().eval()
    enc = EgoClearV3EncoderExport(m_cpu).eval()
    stp = EgoClearV4StepExport(m_cpu).eval()

    rgb = torch.randn(1, 3, RGB_H, RGB_W)
    K = torch.tensor([[449.0, 453.0, 252.0, 155.0]])
    R = torch.eye(3).unsqueeze(0)
    enc_p = out_dir / "egoclear_v65_distilled_encoder.onnx"
    stp_p = out_dir / "egoclear_v65_distilled_step.onnx"

    with torch.no_grad():
        pt_f, pt_d = enc(rgb, K, R)
        torch.onnx.export(enc, (rgb, K, R), str(enc_p), opset_version=17, input_names=["rgb", "K_scaled", "R_c2w"], output_names=["f2d", "disp_cues"], dynamo=False)
        t_t, t_1, t_2 = torch.tensor([[0.5, 0.0, 1.5]]), torch.tensor([[0.25, 0.0, 1.5]]), torch.tensor([[0.0, 0.0, 1.5]])
        fwd, ds = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0.25])
        mem0, age0 = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX), torch.full((1, 1, BEV_NZ, BEV_NX), 1.0)
        stp_in = (pt_f, pt_d, pt_f, pt_d, pt_f, pt_d, K, R, t_t, fwd, K, R, t_1, fwd, K, R, t_2, ds, mem0, age0)
        pt_stp = stp(*stp_in)
        torch.onnx.export(
            stp, stp_in, str(stp_p), opset_version=17,
            input_names=["f2d_t", "dc_t", "f2d_1", "dc_1", "f2d_2", "dc_2", "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1", "K_2", "R_2", "t_2", "ds_t", "mem_prev", "age_prev"],
            output_names=["static_logits", "dynamic_logits", "corr_probs_monotonic", "unc_logits", "mem_new", "age_new"],
            dynamo=False,
        )

    enc_fp16 = out_dir / "egoclear_v65_distilled_encoder_fp16.onnx"
    stp_fp16 = out_dir / "egoclear_v65_distilled_step_fp16.onnx"
    convert_onnx_weights_to_fp16_safe(enc_p, enc_fp16)
    convert_onnx_weights_to_fp16_safe(stp_p, stp_fp16)

    sess_s = ort.InferenceSession(str(stp_fp16), providers=["CPUExecutionProvider"])
    feed_s = {inp.name: val.numpy() for inp, val in zip(sess_s.get_inputs(), stp_in)}
    ort_s = sess_s.run(None, feed_s)
    stp_err = float(np.max(np.abs(ort_s[2] - pt_stp[2].numpy())))
    mono_ok = bool(np.all(np.diff(ort_s[2], axis=-1) >= -1e-6))

    for p in (enc_fp16, stp_fp16):
        dst = DRIVE_RUNS / p.name
        if dst.exists():
            dst.unlink()
        shutil.copy2(p, dst)

    info = {
        "fp32_total_mb": round((enc_p.stat().st_size + stp_p.stat().st_size) / 1e6, 2),
        "fp16_total_mb": round((enc_fp16.stat().st_size + stp_fp16.stat().st_size) / 1e6, 2),
        "fp16_step_corr_max_abs_err": round(stp_err, 6),
        "onnx_monotonic_verified": mono_ok,
    }
    print(f"[onnx-v6.5] EXPORT & FP16 PARITY PASS: {info}", flush=True)
    return info


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    # 1. Stage all 11 Drive datasets in parallel (~100s)
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
    real_head_train = [load_scene(p) for p in sorted(paths["real_hd100"].glob("*.npz"))]
    arkit_train = []
    for key in ("arkit_90", "arkit_210"):
        arkit_train.extend(load_scene(p) for p in sorted(paths[key].glob("*.npz")))

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}

    tot_tr = len(synth_train) + len(real_seg_train) + len(real_head_train) + len(arkit_train)
    print(
        f"[v6.5] Loaded {tot_tr} train scenes: synth={len(synth_train)}, real_seg={len(real_seg_train)}, "
        f"real_head={len(real_head_train)}, arkit={len(arkit_train)}",
        flush=True,
    )

    # 2. Load frozen Teacher (V6-Base 97.5M + 3-lag) & warm-start Student (V5-Small 25M + 2-lag)
    t_ckpt_p = LOCAL_RUNS / "egoclear_v6_base_full.pt"
    s_ckpt_p = LOCAL_RUNS / "egoclear_v5_full.pt"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase7_v6/egoclear_v6_base_full.pt", t_ckpt_p)
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase6_v5/egoclear_v5_full.pt", s_ckpt_p)

    teacher = EgoClearV6Net(dav2_id="depth-anything/Depth-Anything-V2-Base-hf", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_dav2=True).to(device).eval()
    teacher.load_state_dict(torch.load(t_ckpt_p, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    for p in teacher.parameters():
        p.requires_grad = False

    student = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    student.load_state_dict(torch.load(s_ckpt_p, map_location="cpu", weights_only=False)["head_state_dict"], strict=False)
    print("[v6.5] Teacher (V6-Base 101.5M frozen) + Student (V5-Small 28.5M warm-start) READY!", flush=True)

    # 3. Privileged Cross-Architecture Distillation (18 ep x 2,880 clips, bs=32 x 12 = 384 fr/step on 96 GB G4!)
    t_tr = time.time()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2.0e-4, weight_decay=1e-4)
    ds = PrivilegedDistillDataset(synth_train, real_seg_train, real_head_train, arkit_train, clip_len=12, clips_per_epoch=2880)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    epochs = 18
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1e-5)
    history = []
    print(f"\n=== Privileged Distilling EgoClear-v6.5 (V6-Base Teacher -> 57MB V5-Small Student, 18 ep, 384 fr/step) ===", flush=True)
    for ep in range(1, epochs + 1):
        student.train()
        ep_l = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0, "kd": 0.0, "flk": 0.0}
        te = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_t = teacher.forward_sequence(
                    batch["rgb_t"], batch["K_t"], batch["R_t"], batch["t_t"], batch["fwd_t"], batch["seg_t"]
                )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_s = student.forward_sequence(
                    batch["rgb_s"], batch["K_s"], batch["R_s"], batch["t_s"], batch["fwd_s"], batch["seg_s"]
                )
                loss, parts = compute_loss_v65_distill(out_s, out_t, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += parts[k]
        for k in ep_l:
            ep_l[k] /= len(loader)
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_l.items()}, "sec": round(time.time() - te, 1)})
        if ep in (1, 5, 10, 14, epochs):
            print(
                f"  [V6.5 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} "
                f"(st={ep_l['st']:.4f} dy={ep_l['dy']:.4f} co={ep_l['co']:.4f} kd={ep_l['kd']:.4f} flk={ep_l['flk']:.4f}) {time.time()-te:.1f}s",
                flush=True,
            )

    del teacher
    torch.cuda.empty_cache()

    ckpt_p = LOCAL_RUNS / "egoclear_v65_distilled_small.pt"
    tr_state = {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith("dav2.")}
    torch.save({"arch": "EgoClearV4Net-v6.5-Distilled", "mode": "full", "c_bev": 128, "head_state_dict": tr_state, "history": history}, ckpt_p)
    dst_ckpt = DRIVE_RUNS / ckpt_p.name
    if dst_ckpt.exists():
        dst_ckpt.unlink()
    shutil.copy2(ckpt_p, dst_ckpt)

    # 4. Full 4-Domain Evaluation of EgoClear-v6.5 (57 MB Mobile Student!)
    print("\n=== Evaluating EgoClear-v6.5 (57.22 MB Distilled Student) Across ALL 4 Domains ===", flush=True)
    ev_v65 = {
        "synth_clean": eval_v5_all(student, eval_synth, None, bw_speeds, device, compute_uncertainty_audit=True),
        "synth_vio_1x": eval_v5_all(student, eval_synth, None, bw_speeds, device, vio_scale=1.0),
        "real_40_local": eval_v5_all(student, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=True, cam_split=True),
        "aria_g2_zeroshot": eval_v5_all(student, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=True),
        "arkit_indoor_val30": eval_v5_all(student, eval_arkit, None, bw_speeds, device, compute_uncertainty_audit=True),
    }
    for rname, ev in ev_v65.items():
        print(
            f"[V6.5-57MB | {rname}] tv={ev['clear_frame_mean_tv']} mono={ev['monotonicity_violation_rate']} "
            f"dyn={ev['dyn_only_alert_rate_at_0.5']['rate']} headF1={ev['hband_corridor_f1_at_0.5']['HEAD']['f1']} ms={ev['ms_per_frame_stream']}\n"
            f"   RAW 4.0m:   @80={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=80.0')} | @120={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=120.3')} | @200={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=200.0')} | @350={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=350.0')} | @500={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=500.0')}\n"
            f"   SLOPE 4.0m: @80={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=80.0')} | @120={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=120.3')} | @200={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=200.0')} | @350={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=350.0')}\n"
            f"   RAW 4.4m:   @80={fmt_m(ev['modes']['h44m_raw'],'late_at_fa_fr<=80.0')} | @120={fmt_m(ev['modes']['h44m_raw'],'late_at_fa_fr<=120.3')} | @200={fmt_m(ev['modes']['h44m_raw'],'late_at_fa_fr<=200.0')} | @500={fmt_m(ev['modes']['h44m_raw'],'late_at_fa_fr<=500.0')}",
            flush=True,
        )
        if ev.get("by_camera_mount"):
            for ck, cv in ev["by_camera_mount"].items():
                m120 = cv.get("late_at_fa_fr<=120.3")
                m200 = cv.get("late_at_fa_fr<=200.0")
                s120 = f"{m120['late']}/{m120['avoidable']}({m120['late_rate']*100:.1f}%)" if m120 else "N/A"
                s200 = f"{m200['late']}/{m200['avoidable']}({m200['late_rate']*100:.1f}%)" if m200 else "N/A"
                print(f"     [cam={ck}] @fr<=120: {s120} | @fr<=200: {s200}", flush=True)

    onnx_info = export_v65_onnx_fp16(student, LOCAL_RUNS)

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_sec": round(time.time() - t_all, 1),
        "train_sec": round(time.time() - t_tr, 1),
        "counts": {
            "synth_train": len(synth_train), "real_seg_train": len(real_seg_train),
            "real_head_train": len(real_head_train), "arkit_train": len(arkit_train),
            "total_train": tot_tr,
        },
        "egoclear_v65_distilled": {"history": history, "evals": ev_v65},
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase8_v65_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase8_v65_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[v6.5] DONE saved phase8_v65_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
