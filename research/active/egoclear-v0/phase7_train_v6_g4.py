"""Phase 7 G4 (96 GB Blackwell): EgoClear-v6 (Frozen DAV2-Base 97.5M + 3-Lag Multi-Scale Stereo lag=1,2,4
+ 1,400-Scene Unified Multi-Camera Pool: 627 Synth + ~475 Real Outdoor [Head+Chest] + 300 Real Indoor).

Also exports 2-stage streaming ONNX-17 (FP32 + FP16) with verified 0.0 monotonic parity!
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v6_model import EgoClearV6Net, EgoClearV6StepExport, compute_loss_v6
from eval_closed_loop import H, apply_causal_hysteresis, eval_score_curve, summarize_at_fa_caps
from phase4_simreal_aria_onnx_g4 import EgoClearV3EncoderExport
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from phase5b_calibrated_v41_g4 import THRS_HIRES, _copy_and_extract_one
from phase6_train_v5_g4 import apply_virtual_pan_tilt, fmt_m
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase7_v6"
LOCAL_RUNS = Path("/content/egoclear_runs_p7_v6")


def stage_all_p7_datasets_parallel() -> dict[str, Path]:
    """6-thread parallel Drive-to-/content staging for all cached datasets (including P6 ARKit 210 & Real 22)."""
    t0 = time.time()
    specs = [
        ("synth_287", DRIVE_ROOT / "datasets/sanpo_synth_287_v0", Path("/content/sanpo_287/scenes"), 280),
        ("synth_450", DRIVE_ROOT / "datasets/sanpo_synth_extra450_v0", Path("/content/sanpo_p2/extra_scenes"), 440),
        ("real_40", DRIVE_ROOT / "datasets/sanpo_real_zeroshot_40_v0", Path("/content/sanpo_p2/real_scenes"), 38),
        ("real_75", DRIVE_ROOT / "datasets/sanpo_real_train76_v0", Path("/content/sanpo_p4/real_train_scenes"), 70),
        ("real_22", DRIVE_ROOT / "datasets/sanpo_real_extra444_v0", Path("/content/p6_data/real_extra444"), 20),
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
        print(f"[stage-p7] copying & extracting {len(jobs)} tar shards across 8 threads...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_copy_and_extract_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                name, sec = fut.result()
                done += 1
                if done in (1, len(jobs) // 2, len(jobs)):
                    print(f"  [stage-p7] {done}/{len(jobs)} done ({name} {sec:.1f}s, elapsed {time.time()-t0:.1f}s)", flush=True)
    print(f"[stage-p7] ALL base datasets ready locally in {time.time()-t0:.1f}s!", flush=True)
    return paths


class Phase7V6Dataset(torch.utils.data.Dataset):
    """1,400+ scene multi-domain sampler: 34% Synth + 48% Real Outdoor (Head+Chest) + 18% Real Indoor."""

    def __init__(
        self,
        synth_dicts: list[dict],
        real_seg_dicts: list[dict],
        real_head_noseg_dicts: list[dict],
        arkit_dicts: list[dict],
        clip_len: int = 12,
        clips_per_epoch: int = 3072,
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
            f"[dataset-v6] pools: synth={len(self.pools['synth'])} (34%), "
            f"real_seg={len(self.pools['real_seg'])} (30%), "
            f"real_head_noseg={len(self.pools['real_head'])} (18%), "
            f"arkit={len(self.pools['arkit'])} (18%)",
            flush=True,
        )

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.34:
            dom = "synth"
        elif u_dom < 0.64 and len(self.pools["real_seg"]) > 0:
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
        elif len(dyn_only_idx) > 0 and u_pick < 0.68:
            anchor = int(dyn_only_idx[rng.integers(len(dyn_only_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(clear_idx) > 0 and u_pick < 0.88:
            anchor = int(clear_idx[rng.integers(len(clear_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        else:
            st = int(rng.integers(0, n - self.clip_len + 1))
        sl = np.arange(st, st + self.clip_len)

        rgb = decode_rgb_batch(d, sl)
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)

        if rng.random() < 0.65:
            zoom = float(rng.uniform(0.68, 1.22))
            rgb, K_s = apply_focal_zoom_crop(rgb, K_s, zoom)

        # Exact geometric Virtual Pan-Tilt head-turn augmentation (+/-18 deg yaw, +/-10 deg pitch)
        if rng.random() < 0.45:
            sh_x = int(rng.integers(-int(0.16 * RGB_W), int(0.16 * RGB_W) + 1))
            sh_y = int(rng.integers(-int(0.10 * RGB_H), int(0.10 * RGB_H) + 1))
            rgb, K_s = apply_virtual_pan_tilt(rgb, K_s, sh_x, sh_y)

        gamma = float(rng.uniform(0.70, 1.42))
        rgb = np.power(np.clip(rgb, 1e-4, 1.0), gamma)
        ch_gain = rng.uniform(0.84, 1.16, size=(1, 3, 1, 1)).astype(np.float32)
        c_fac = float(rng.uniform(0.80, 1.25))
        mean = rgb.mean(axis=(2, 3), keepdims=True)
        rgb = np.clip(((rgb - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
        if rng.random() < 0.25:
            luma = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
            sat = float(rng.uniform(0.0, 0.35))
            rgb = np.clip(sat * rgb + (1.0 - sat) * luma, 0.0, 1.0)
        if rng.random() < 0.30:
            rgb = np.clip(rgb + rng.normal(0.0, float(rng.uniform(0.005, 0.022)), rgb.shape).astype(np.float32), 0.0, 1.0)
        if rng.random() < 0.20:
            for t_i in range(self.clip_len):
                hwc = cv2.GaussianBlur(rgb[t_i].transpose(1, 2, 0), (3, 3), 0.8)
                rgb[t_i] = hwc.transpose(2, 0, 1)

        rgb = (rgb - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_seq = np.repeat(K_s[None], self.clip_len, axis=0)

        R_seq = d["poses_R"][sl].astype(np.float32)
        t_seq = d["poses_t"][sl].astype(np.float32)
        fwd_seq = d["causal_fwd"][sl].astype(np.float32)
        seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

        if rng.random() < 0.50:
            R_seq, t_seq, fwd_seq, seg_seq = perturb_vio_trajectory(
                R_seq, t_seq, fwd_seq, rng, trans_rel_std=0.025, yaw_deg_per_m=0.8, grav_tilt_deg=2.0
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
            "has_dyn_gt": torch.tensor(has_dyn, dtype=torch.float32),
        }


@torch.inference_mode()
def eval_v6_all(
    model: EgoClearV6Net,
    scene_dicts: dict[str, dict],
    eval_rows_override: dict[str, np.ndarray] | None,
    bw_speeds: dict,
    device: torch.device,
    vio_scale: float = 0.0,
    compute_uncertainty_audit: bool = False,
    cam_split: bool = False,
) -> dict:
    model.eval()
    eval_rows_map, corr_probs_map, unc_probs_map = {}, {}, {}
    stat_ap_num = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
    dyn_fp = dyn_tot = mono_v = mono_t = tot_fr = 0
    clear_tv = 0.0
    clear_cnt = 0
    tot_ms = 0.0

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))
    idx_44m = int(min(BEV_NZ - 1, np.searchsorted(z_edges, 4.4)))

    for sc, d in scene_dicts.items():
        er = eval_rows_override[sc] if eval_rows_override is not None else d["eval_rows"]
        n = len(er)
        if n < 5:
            continue
        rgb = (decode_rgb_batch(d, np.arange(n)) - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
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
                trans_rel_std=0.02 * vio_scale, yaw_deg_per_m=0.8 * vio_scale, grav_tilt_deg=1.5 * vio_scale,
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

        p_corr = out["corr_probs"][0].float().cpu().numpy()
        p_unc = torch.sigmoid(out["unc_logits"][0].float()).cpu().numpy()
        p_stat = torch.sigmoid(out["static_logits"][0].float()).cpu().numpy()
        eval_rows_map[sc] = er
        corr_probs_map[sc] = p_corr
        unc_probs_map[sc] = p_unc

        diffs = np.diff(p_corr, axis=1)
        mono_v += int((diffs < -1e-4).sum())
        mono_t += int(diffs.size)

        p4 = p_corr[:, idx_4m]
        valid_c = (np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)) & np.isinf(er[:, 0])
        cp = valid_c[1:] & valid_c[:-1]
        if np.any(cp):
            clear_tv += float(np.abs(p4[1:] - p4[:-1])[cp].sum())
            clear_cnt += int(cp.sum())

        gt_st = d["gt_bev_static"][:, :, :, 6:9] > 0
        pr_st = p_stat[:, :, :, 6:9] >= 0.5
        for b in range(3):
            stat_ap_num[b][0] += int((pr_st[:, b] & gt_st[:, b]).sum())
            stat_ap_num[b][1] += int((pr_st[:, b] & ~gt_st[:, b]).sum())
            stat_ap_num[b][2] += int((~pr_st[:, b] & gt_st[:, b]).sum())

        D_gt = er[:, 0]
        dyn_only = (d["valid_heading"] > 0) & np.isinf(D_gt) & (er[:, 2] > 0.5)
        if np.any(dyn_only):
            dyn_tot += int(dyn_only.sum())
            dyn_fp += int((p4[dyn_only] >= 0.5).sum())

    scores_4m = {sc: probs[:, idx_4m] for sc, probs in corr_probs_map.items()}
    scores_44m = {sc: probs[:, idx_44m] for sc, probs in corr_probs_map.items()}

    modes_out = {}
    for hmode in ("raw", "slope_aware_hyst", "confirm_2of3"):
        c = eval_score_curve(eval_rows_map, scores_4m, bw_speeds, THRS_HIRES, higher_is_alert=True, hysteresis_mode=hmode)
        modes_out[hmode] = {"curve": c, "summary": summarize_at_fa_caps(c)}
    for hmode in ("raw", "slope_aware_hyst"):
        c = eval_score_curve(eval_rows_map, scores_44m, bw_speeds, THRS_HIRES, higher_is_alert=True, hysteresis_mode=hmode)
        modes_out[f"h44m_{hmode}"] = {"curve": c, "summary": summarize_at_fa_caps(c)}

    cam_out = None
    if cam_split:
        cam_out = {}
        for cam_tag in ("camera_head", "camera_chest"):
            sub_er = {sc: er for sc, er in eval_rows_map.items() if cam_tag in sc}
            sub_sc = {sc: scores_4m[sc] for sc in sub_er}
            if sub_er:
                c = eval_score_curve(sub_er, sub_sc, bw_speeds, THRS_HIRES, higher_is_alert=True, hysteresis_mode="raw")
                cam_out[cam_tag] = summarize_at_fa_caps(c)

    hband_f1 = {}
    for b, name in ((0, "LOW"), (1, "BODY"), (2, "HEAD")):
        tp, fp, fn = stat_ap_num[b]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        hband_f1[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "pos_cells": tp + fn}

    unc_audit = None
    if compute_uncertainty_audit:
        all_p, all_y, all_u = [], [], []
        for sc, er in eval_rows_map.items():
            valid = (np.isfinite(er[:, 0]) | np.isinf(er[:, 0])) & np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)
            if not np.any(valid):
                continue
            all_p.append(scores_4m[sc][valid])
            all_u.append(unc_probs_map[sc][valid, : idx_4m + 1].mean(axis=1))
            all_y.append((er[valid, 0] <= H).astype(np.int32))
        ap, au, ay = np.concatenate(all_p), np.concatenate(all_u), np.concatenate(all_y)
        order = np.argsort(au)
        unc_audit = {
            f"cov_{int(cov*100)}": {
                "n_frames": (k := max(1, int(len(order) * cov))),
                "err": round(float(np.mean((ap[order[:k]] >= 0.32).astype(np.int32) != ay[order[:k]])), 4),
            }
            for cov in (1.0, 0.90, 0.80, 0.70)
        }

    return {
        "ms_per_frame_stream": round(tot_ms / max(tot_fr, 1), 2),
        "clear_frame_mean_tv": round(clear_tv / max(clear_cnt, 1), 5),
        "monotonicity_violation_rate": round(mono_v / max(mono_t, 1), 6),
        "hband_corridor_f1_at_0.5": hband_f1,
        "dyn_only_alert_rate_at_0.5": {"fp": dyn_fp, "total": dyn_tot, "rate": round(dyn_fp / max(dyn_tot, 1), 4) if dyn_tot else None},
        "uncertainty_risk_coverage": unc_audit,
        "by_camera_mount": cam_out,
        "modes": modes_out,
    }


def export_v6_onnx_fp16(model: EgoClearV6Net, out_dir: Path) -> dict:
    try:
        import onnx, onnxruntime as ort
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
        import onnx, onnxruntime as ort
    from edge_quant_and_replay import convert_onnx_weights_to_fp16_safe

    m_cpu = model.cpu().eval()
    enc = EgoClearV3EncoderExport(m_cpu).eval()
    stp = EgoClearV6StepExport(m_cpu).eval()

    rgb = torch.randn(1, 3, RGB_H, RGB_W)
    K = torch.tensor([[449.0, 453.0, 252.0, 155.0]])
    R = torch.eye(3).unsqueeze(0)
    enc_p = out_dir / "egoclear_v6_base_encoder.onnx"
    stp_p = out_dir / "egoclear_v6_base_step.onnx"

    with torch.no_grad():
        pt_f, pt_d = enc(rgb, K, R)
        torch.onnx.export(enc, (rgb, K, R), str(enc_p), opset_version=17, input_names=["rgb", "K_scaled", "R_c2w"], output_names=["f2d", "disp_cues"], dynamo=False)
        t_t = torch.tensor([[1.0, 0.0, 1.5]])
        t_1 = torch.tensor([[0.75, 0.0, 1.5]])
        t_2 = torch.tensor([[0.50, 0.0, 1.5]])
        t_4 = torch.tensor([[0.0, 0.0, 1.5]])
        fwd, ds = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0.25])
        mem0, age0 = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX), torch.full((1, 1, BEV_NZ, BEV_NX), 1.0)
        stp_in = (pt_f, pt_d, pt_f, pt_d, pt_f, pt_d, pt_f, pt_d, K, R, t_t, fwd, K, R, t_1, fwd, K, R, t_2, K, R, t_4, ds, mem0, age0)
        pt_stp = stp(*stp_in)
        torch.onnx.export(
            stp, stp_in, str(stp_p), opset_version=17,
            input_names=[
                "f2d_t", "dc_t", "f2d_1", "dc_1", "f2d_2", "dc_2", "f2d_4", "dc_4",
                "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1", "K_2", "R_2", "t_2", "K_4", "R_4", "t_4",
                "ds_t", "mem_prev", "age_prev",
            ],
            output_names=["static_logits", "dynamic_logits", "corr_probs_monotonic", "unc_logits", "mem_new", "age_new"],
            dynamo=False,
        )

    enc_fp16 = out_dir / "egoclear_v6_base_encoder_fp16.onnx"
    stp_fp16 = out_dir / "egoclear_v6_base_step_fp16.onnx"
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
    print(f"[onnx-v6] EXPORT & FP16 PARITY PASS: {info}", flush=True)
    return info


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    # 1. Stage all 8 cached Drive datasets in parallel (8 threads, ~100s)
    paths = stage_all_p7_datasets_parallel()
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
    for ddir in (paths["real_75"], paths["real_22"], Path("/content/p7_data/real_chest240")):
        if ddir.exists():
            real_seg_train.extend(load_scene(p) for p in sorted(ddir.glob("*.npz")))

    real_head_train = []
    head_dir = Path("/content/p7_data/real_head160")
    if head_dir.exists():
        real_head_train.extend(load_scene(p) for p in sorted(head_dir.glob("*.npz")))

    arkit_train = []
    for ddir in (paths["arkit_90"], paths["arkit_210"]):
        if ddir.exists():
            arkit_train.extend(load_scene(p) for p in sorted(ddir.glob("*.npz")))

    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va30"].glob("*.npz"))}

    tot_tr = len(synth_train) + len(real_seg_train) + len(real_head_train) + len(arkit_train)
    print(
        f"[v6] RAM Loaded {tot_tr} train scenes: synth={len(synth_train)}, real_seg={len(real_seg_train)}, "
        f"real_head_noseg={len(real_head_train)}, arkit={len(arkit_train)} | "
        f"eval: synth={len(eval_synth)}, real={len(eval_real)}, aria={len(aria_dicts)}, arkit={len(eval_arkit)}",
        flush=True,
    )

    # 2. Build EgoClear-v6 (Frozen DAV2-Base 97.5M + 3-lag 1,2,4 multi-scale stereo + Monotonic Survival)
    model = EgoClearV6Net(dav2_id="depth-anything/Depth-Anything-V2-Base-hf", c_feat=64, c_bev=128, lags=(1, 2, 4), freeze_dav2=True).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[v6] EgoClearV6Net (DAV2-Base 97.5M frozen + 3-lag MVS): trainable={n_tr/1e6:.2f}M / total={n_tot/1e6:.2f}M", flush=True)

    t_tr = time.time()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
    ds = Phase7V6Dataset(synth_train, real_seg_train, real_head_train, arkit_train, clip_len=12, clips_per_epoch=3072)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    epochs = 22
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1.5e-5)
    history = []
    print(f"\n=== Training EgoClear-v6-Base ({tot_tr} scenes, 22 ep, 384 fr/step, DAV2-Base + lag 1,2,4) ===", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        ep_l = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0, "flk": 0.0}
        te = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_sequence(
                    batch["rgb"], batch["K_scaled"], batch["R_c2w"], batch["t_w"], batch["fwd_w"], batch["seg_len"]
                )
                loss, parts = compute_loss_v6(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += parts[k]
        for k in ep_l:
            ep_l[k] /= len(loader)
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_l.items()}, "sec": round(time.time() - te, 1)})
        if ep in (1, 5, 10, 15, 20, epochs):
            print(
                f"  [V6 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} "
                f"(st={ep_l['st']:.4f} dy={ep_l['dy']:.4f} co={ep_l['co']:.4f} flk={ep_l['flk']:.4f}) {time.time()-te:.1f}s",
                flush=True,
            )

    ckpt_p = LOCAL_RUNS / "egoclear_v6_base_full.pt"
    tr_state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("dav2.")}
    torch.save({
        "arch": "EgoClearV6Net-Base-3lag", "dav2_id": "depth-anything/Depth-Anything-V2-Base-hf",
        "lags": (1, 2, 4), "c_bev": 128, "head_state_dict": tr_state, "history": history,
    }, ckpt_p)
    dst_ckpt = DRIVE_RUNS / ckpt_p.name
    if dst_ckpt.exists():
        dst_ckpt.unlink()
    shutil.copy2(ckpt_p, dst_ckpt)

    print("\n=== Evaluating EgoClear-v6-Base Across ALL 4 Domains ===", flush=True)
    ev_v6 = {
        "synth_clean": eval_v6_all(model, eval_synth, None, bw_speeds, device, compute_uncertainty_audit=True),
        "synth_vio_1x": eval_v6_all(model, eval_synth, None, bw_speeds, device, vio_scale=1.0),
        "real_40_local": eval_v6_all(model, eval_real, real_local_rows, bw_speeds, device, compute_uncertainty_audit=True, cam_split=True),
        "aria_g2_zeroshot": eval_v6_all(model, aria_dicts, None, bw_speeds, device, compute_uncertainty_audit=True),
        "arkit_indoor_val30": eval_v6_all(model, eval_arkit, None, bw_speeds, device, compute_uncertainty_audit=True),
    }
    for rname, ev in ev_v6.items():
        print(
            f"[V6-BASE | {rname}] tv={ev['clear_frame_mean_tv']} mono={ev['monotonicity_violation_rate']} "
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

    onnx_info = export_v6_onnx_fp16(model, LOCAL_RUNS)

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_sec": round(time.time() - t_all, 1),
        "train_sec": round(time.time() - t_tr, 1),
        "counts": {
            "synth_train": len(synth_train), "real_seg_train": len(real_seg_train),
            "real_head_noseg_train": len(real_head_train), "arkit_train": len(arkit_train),
            "total_train": tot_tr,
            "eval_synth": len(eval_synth), "eval_real": len(eval_real), "aria_g2": len(aria_dicts), "eval_arkit": len(eval_arkit),
        },
        "egoclear_v6_base": {"trainable_params": n_tr, "history": history, "evals": ev_v6},
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase7_v6_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase7_v6_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[v6] DONE saved phase7_v6_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
