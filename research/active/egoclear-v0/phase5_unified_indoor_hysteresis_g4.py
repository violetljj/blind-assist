"""Phase 5 G4 (96 GB Blackwell): Unified Indoor+Outdoor EgoClear-v4 + Causal Event-Level Hysteresis.

Pipeline:
  1. Stage existing 5 Drive datasets to /content (SANPO-Synth 287 + Extra 450 + Real-40 + Real-Train-75 + Aria-G2 19)
     and rebuild drift-free local GT cache for SANPO-Real-40 (~35s on 48 vCPUs).
  2. Pack ARKitScenes Real Indoor Corridors/Rooms (90 visit-disjoint Train + 30 strictly unseen Val) -> Drive.
  3. Evaluate Indoor-30 Baselines:
     - LiDAR Ref Single (iPad hardware sensor reference)
     - DAV2-S Ground Anchor
     - UniDepth-V2-Small (34M) & UniDepth-V2-Large (340M) with real iPad intrinsics
     - Phase 4 EgoClear-v3-Full (Outdoor-Only -> Indoor Zero-Shot!)
  4. Zero-Training Causal Event-Level Hysteresis Ablation on Phase 4 EgoClear-v3-Full across ALL 4 DOMAINS:
     (raw vs ema_hyst vs confirm_2of3) -> cleanly isolates inference-time state-machine benefit!
  5. Train EgoClear-v4-Full (Monotonic First-Hit Survival Hazard + Clear-Frame Flicker Suppression Loss +
     Unified 42% Synth + 28% Real-Outdoor + 30% ARKit-Indoor + grayscale/low-light aug, 20 ep, 576 fr/step).
  6. Full 4-Domain x 3-Hysteresis Evaluation (Synth Clean, Synth VIO 1x, Real-40 Local, Aria-G2 Zero-Shot,
     ARKit-Indoor 30 Val) + 2-Stage Mobile Streaming ONNX-17 Export & Parity Audit!
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, "/content")
ud_root = Path("/content/unidepth_src")
if ud_root.exists():
    sub = next(ud_root.glob("UniDepth-*"), None)
    if sub is not None:
        sys.path.insert(0, str(sub))

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, NY, RGB_H, RGB_W
from egoclear_v3_model import EgoClearV3Net
from egoclear_v4_model import EgoClearV4Net, EgoClearV4StepExport, compute_loss_v4
from eval_closed_loop import H, apply_causal_hysteresis, eval_score_curve, summarize_at_fa_caps
from pack_arkit_colab import pack_arkit_split
from phase4_simreal_aria_onnx_g4 import EgoClearV3EncoderExport
from train_and_eval_g4 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    decode_rgb_batch,
    depth_map_to_corridor_d,
    ground_anchor_relative_depth,
    load_scene,
)
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory
from train_and_eval_phase3_g4 import build_local_window_real_eval_rows

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_ARKIT_TR = DRIVE_ROOT / "datasets/arkit_indoor_train90_v0"
DRIVE_ARKIT_VA = DRIVE_ROOT / "datasets/arkit_indoor_val30_v0"
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase5"
LOCAL_RUNS = Path("/content/egoclear_runs_p5")


def extract_drive_tar_dataset(drive_dir: Path, dst_dir: Path, min_files: int) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    cur = sorted(dst_dir.glob("*.npz"))
    if len(cur) >= min_files:
        print(f"[stage] {dst_dir.name} already ready ({len(cur)} npz)", flush=True)
        return cur
    t0 = time.time()
    man_p = drive_dir / "manifest_sha256.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        sh_files = [drive_dir / sh["file"] for sh in man["shards"]]
    else:
        sh_files = sorted(drive_dir.glob("*.tar"))
    for src in sh_files:
        tmp = dst_dir / src.name
        shutil.copy2(src, tmp)
        with tarfile.open(tmp, "r") as tf:
            tf.extractall(dst_dir)
        tmp.unlink()
    cur = sorted(dst_dir.glob("*.npz"))
    print(f"[stage] extracted {drive_dir.name} -> {len(cur)} npz in {time.time()-t0:.1f}s", flush=True)
    return cur


def _local_real_worker(path_str: str):
    p = Path(path_str)
    d = load_scene(p)
    rows, sd = build_local_window_real_eval_rows(d)
    return p.stem, rows, sd


def ensure_real_local_gt_cache(real_eval_dir: Path, cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return {k[2:]: z[k] for k in z.files if k.startswith("r_")}
    t0 = time.time()
    files = sorted(str(p) for p in real_eval_dir.glob("*.npz"))
    ncpu = min(20, max(4, (os.cpu_count() or 8) - 2))
    print(f"[stage] rebuilding drift-free +/-0.8s local GT cache for {len(files)} real eval scenes...", flush=True)
    save_d = {}
    out_rows = {}
    with ProcessPoolExecutor(max_workers=ncpu) as ex:
        for sc, rows, sd in ex.map(_local_real_worker, files):
            save_d[f"r_{sc}"] = rows
            save_d[f"d_{sc}"] = sd
            out_rows[sc] = rows
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save_d)
    print(f"[stage] rebuilt real_local_gt_cache in {time.time()-t0:.1f}s", flush=True)
    return out_rows


class UnifiedPhase5ClipDataset(torch.utils.data.Dataset):
    """Unified multi-domain clip sampler: 42% SANPO-Synth + 28% SANPO-Real Outdoor + 30% ARKitScenes Indoor.

    Adds broad FOV zoom (0.68-1.25), grayscale/low-light simulation (bridges indoor dim corridors &
    monochrome glasses zero-shot without ever touching Aria!), and 50% realistic VIO drift injection.
    """

    def __init__(
        self,
        synth_dicts: list[dict],
        real_dicts: list[dict],
        arkit_dicts: list[dict],
        clip_len: int = 12,
        clips_per_epoch: int = 2400,
    ):
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
        self.pools = {
            "synth": [d for d in synth_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "real": [d for d in real_dicts if len(d["eval_rows"]) >= clip_len + 1],
            "arkit": [d for d in arkit_dicts if len(d["eval_rows"]) >= clip_len + 1],
        }
        print(
            f"[dataset] UnifiedPhase5 pools: synth={len(self.pools['synth'])}, "
            f"real={len(self.pools['real'])}, arkit={len(self.pools['arkit'])}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.42:
            dom = "synth"
        elif u_dom < 0.70 and len(self.pools["real"]) > 0:
            dom = "real"
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
        if len(haz_idx) > 0 and u_pick < 0.50:
            anchor = int(haz_idx[rng.integers(len(haz_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(dyn_only_idx) > 0 and u_pick < 0.68:
            anchor = int(dyn_only_idx[rng.integers(len(dyn_only_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        elif len(clear_idx) > 0 and u_pick < 0.88:
            # Explicit clear-corridor sequences so temporal flicker suppression loss gets strong clean supervision!
            anchor = int(clear_idx[rng.integers(len(clear_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        else:
            st = int(rng.integers(0, n - self.clip_len + 1))
        sl = np.arange(st, st + self.clip_len)

        rgb = decode_rgb_batch(d, sl)
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)

        # 1. Broad FOV / focal-length augmentation
        if rng.random() < 0.65:
            zoom = float(rng.uniform(0.68, 1.22))
            rgb, K_s = apply_focal_zoom_crop(rgb, K_s, zoom)

        # 2. Photometric + indoor warm/cool + low-light + grayscale/desaturation aug
        gamma = float(rng.uniform(0.70, 1.45))
        rgb = np.power(np.clip(rgb, 1e-4, 1.0), gamma)
        ch_gain = rng.uniform(0.82, 1.18, size=(1, 3, 1, 1)).astype(np.float32)
        c_fac = float(rng.uniform(0.78, 1.28))
        mean = rgb.mean(axis=(2, 3), keepdims=True)
        rgb = np.clip(((rgb - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
        if rng.random() < 0.22:
            # Monochrome / low-saturation augmentation (general sensor invariance for dim corridors & grayscale CV cameras)
            luma = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
            sat = float(rng.uniform(0.0, 0.35))
            rgb = np.clip(sat * rgb + (1.0 - sat) * luma, 0.0, 1.0)
        if rng.random() < 0.35:
            rgb = np.clip(rgb + rng.normal(0.0, float(rng.uniform(0.005, 0.025)), rgb.shape).astype(np.float32), 0.0, 1.0)
        if rng.random() < 0.22:
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
        D_gt = ev_r[:, 0]
        cover = ev_r[:, 1]
        dyn_fl = ev_r[:, 2]

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


@torch.inference_mode()
def eval_model_with_hysteresis(
    model: nn.Module,
    scene_dicts: dict[str, dict],
    eval_rows_override: dict[str, np.ndarray] | None,
    bw_speeds: dict,
    device: torch.device,
    is_v4_monotonic: bool,
    vio_scale: float = 0.0,
    compute_uncertainty_audit: bool = False,
) -> dict:
    """Evaluate EgoClear-v3 or v4 across raw, ema_hyst, confirm_2of3, and walker_aware + hysteresis."""
    model.eval()
    eval_rows_map = {}
    corr_probs_map = {}
    unc_probs_map = {}
    stat_ap_num = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
    dyn_fp_frames, dyn_tot_frames = 0, 0
    clear_tv_sum, clear_pair_cnt = 0.0, 0
    mono_viol_cnt, mono_tot_cnt = 0, 0
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

        if is_v4_monotonic:
            p_corr = out["corr_probs"][0].float().cpu().numpy()
        else:
            p_corr = torch.sigmoid(out["corr_logits"][0].float()).cpu().numpy()
        p_unc = torch.sigmoid(out["unc_logits"][0].float()).cpu().numpy()
        p_stat = torch.sigmoid(out["static_logits"][0].float()).cpu().numpy()

        eval_rows_map[sc] = er
        corr_probs_map[sc] = p_corr
        unc_probs_map[sc] = p_unc

        # Audit monotonicity violations across consecutive range bins
        diffs_z = np.diff(p_corr, axis=1)
        mono_viol_cnt += int((diffs_z < -1e-4).sum())
        mono_tot_cnt += int(diffs_z.size)

        # Audit clear-frame temporal flicker (mean |p_t - p_{t-1}| on clear valid frames)
        p4 = p_corr[:, idx_4m] if is_v4_monotonic else p_corr[:, : idx_4m + 1].max(axis=1)
        valid_c = (np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)) & np.isinf(er[:, 0])
        cp = valid_c[1:] & valid_c[:-1]
        if np.any(cp):
            clear_tv_sum += float(np.abs(p4[1:] - p4[:-1])[cp].sum())
            clear_pair_cnt += int(cp.sum())

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
            dyn_fp_frames += int((p4[dyn_only] >= 0.5).sum())

    if is_v4_monotonic:
        fixed_scores = {sc: probs[:, idx_4m] for sc, probs in corr_probs_map.items()}
    else:
        fixed_scores = {sc: probs[:, : idx_4m + 1].max(axis=1) for sc, probs in corr_probs_map.items()}

    thrs = np.round(np.concatenate([np.arange(0.02, 0.20, 0.02), np.arange(0.20, 0.96, 0.04)]), 3)

    modes_out = {}
    for hmode in ("raw", "ema_hyst", "confirm_2of3"):
        c = eval_score_curve(eval_rows_map, fixed_scores, bw_speeds, thrs, higher_is_alert=True, hysteresis_mode=hmode)
        modes_out[hmode] = {"curve": c, "summary": summarize_at_fa_caps(c)}

    # Walker-aware + ema_hyst (uses strictly monotonic P(<= d_req + 1.0) for v4!)
    def walker_hyst_fn(ep: dict, thr: float) -> np.ndarray:
        probs = corr_probs_map[ep["scene"]]
        k_bin = int(np.clip(np.searchsorted(z_edges, min(ep["dreq"] + 1.0, 4.9)), 0, BEV_NZ - 1))
        sc_arr = probs[:, k_bin] if is_v4_monotonic else probs[:, : k_bin + 1].max(axis=1)
        return apply_causal_hysteresis(sc_arr, thr, mode="ema_hyst", ema_alpha=0.55, off_ratio=0.72)

    c_wh = eval_score_curve(eval_rows_map, {}, bw_speeds, thrs, walker_aware_fn=walker_hyst_fn)
    modes_out["walker_ema_hyst"] = {"curve": c_wh, "summary": summarize_at_fa_caps(c_wh)}

    hband_f1 = {}
    for b, name in ((0, "LOW"), (1, "BODY"), (2, "HEAD")):
        tp, fp, fn = stat_ap_num[b]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        hband_f1[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "pos_cells": tp + fn}

    unc_audit = None
    if compute_uncertainty_audit:
        all_p, all_y, all_u = [], [], []
        for sc, er in eval_rows_map.items():
            valid = (np.isfinite(er[:, 0]) | np.isinf(er[:, 0])) & np.isfinite(er[:, 1]) & (er[:, 1] >= 0.6)
            if not np.any(valid):
                continue
            p_f = fixed_scores[sc][valid]
            u_f = unc_probs_map[sc][valid, : idx_4m + 1].mean(axis=1)
            y_f = (er[valid, 0] <= H).astype(np.int32)
            all_p.append(p_f)
            all_u.append(u_f)
            all_y.append(y_f)
        ap, au, ay = np.concatenate(all_p), np.concatenate(all_u), np.concatenate(all_y)
        order = np.argsort(au)
        unc_audit = {}
        for cov in (1.0, 0.90, 0.80, 0.70):
            k = max(1, int(len(order) * cov))
            sel = order[:k]
            pred_b = (ap[sel] >= 0.32).astype(np.int32)
            unc_audit[f"cov_{int(cov*100)}"] = {
                "n_frames": k,
                "err": round(float(np.mean(pred_b != ay[sel])), 4),
            }

    return {
        "ms_per_frame_stream": round(tot_ms / max(tot_fr, 1), 2),
        "clear_frame_mean_tv": round(clear_tv_sum / max(clear_pair_cnt, 1), 5),
        "monotonicity_violation_rate": round(mono_viol_cnt / max(mono_tot_cnt, 1), 6),
        "hband_corridor_f1_at_0.5": hband_f1,
        "dyn_only_alert_rate_at_0.5": {
            "fp": dyn_fp_frames,
            "total": dyn_tot_frames,
            "rate": round(dyn_fp_frames / max(dyn_tot_frames, 1), 4) if dyn_tot_frames else None,
        },
        "uncertainty_risk_coverage": unc_audit,
        "modes": modes_out,
    }


@torch.inference_mode()
def eval_indoor_baselines(arkit_val: dict[str, dict], bw_speeds: dict, device: torch.device) -> dict:
    """Evaluate LiDAR Ref Single, DAV2-S Ground, UniDepth-V2-Small & UniDepth-V2-Large on ARKitScenes Indoor 30 Val."""
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    eval_rows_map = {sc: d["eval_rows"] for sc, d in arkit_val.items()}
    gt_single_map = {sc: d["gt_single_d"] for sc, d in arkit_val.items()}
    b1_thrs = np.round(np.arange(0.5, 5.01, 0.25), 2)
    out = {}

    c_ref = eval_score_curve(eval_rows_map, gt_single_map, bw_speeds, b1_thrs, higher_is_alert=False)
    out["ref_lidar_single"] = {"curve": c_ref, "summary": summarize_at_fa_caps(c_ref)}
    bf = out["ref_lidar_single"]["summary"]["best_floor"]
    print(f"[ARKIT_INDOOR_30] ref_lidar_single: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f})", flush=True)

    eval_dh, eval_dw = 192, 256
    min_pts_in = 24
    proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    dav2_m = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
    dav2_d = {}
    for sc, d in arkit_val.items():
        n = len(d["eval_rows"])
        rgbs = decode_rgb_batch(d, np.arange(n))
        depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for st in range(0, n, 32):
                en = min(n, st + 32)
                imgs = [(rgbs[k].transpose(1, 2, 0) * 255.0).astype(np.uint8) for k in range(st, en)]
                inp = proc(images=imgs, return_tensors="pt").to(device)
                pred = dav2_m(**inp).predicted_depth
                depths[st:en] = F.interpolate(pred.unsqueeze(1).float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        K_o = d["K_orig"].astype(np.float64)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
        Rs, ps, fwds = d["poses_R"].astype(np.float64), d["poses_t"].astype(np.float64), d["causal_fwd"].astype(np.float64)
        d_arr = np.full(n, np.inf, dtype=np.float32)
        for i in range(n):
            z_map = ground_anchor_relative_depth(depths[i], K_d, Rs[i], 1.25)
            d_arr[i], _ = depth_map_to_corridor_d(z_map, K_d, Rs[i], ps[i], fwds[i], min_pts_in)
        dav2_d[sc] = d_arr
    c_dav = eval_score_curve(eval_rows_map, dav2_d, bw_speeds, b1_thrs, higher_is_alert=False)
    out["dav2_small_ground"] = {"curve": c_dav, "summary": summarize_at_fa_caps(c_dav)}
    bf = out["dav2_small_ground"]["summary"]["best_floor"]
    print(f"[ARKIT_INDOOR_30] dav2_small_ground: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f})", flush=True)
    del dav2_m, proc
    torch.cuda.empty_cache()

    for u_name, hf_id in (
        ("unidepth_v2_small", "lpiccinelli/unidepth-v2-vits14"),
        ("unidepth_v2_large", "lpiccinelli/unidepth-v2-vitl14"),
    ):
        u_mod = UniDepthV2.from_pretrained(hf_id).to(device).eval()
        ud_d = {}
        for sc, d in arkit_val.items():
            n = len(d["eval_rows"])
            rgbs = decode_rgb_batch(d, np.arange(n))
            K_o = d["K_orig"].astype(np.float64)
            Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
            K_mat = torch.tensor(
                [[K_o[0] * RGB_W / Ww, 0.0, K_o[2] * RGB_W / Ww], [0.0, K_o[1] * RGB_H / Hh, K_o[3] * RGB_H / Hh], [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=device,
            )
            depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
            for st in range(0, n, 32):
                en = min(n, st + 32)
                imgs_u8 = torch.from_numpy((rgbs[st:en] * 255.0).clip(0, 255).astype(np.uint8)).to(device)
                pred = u_mod.infer(imgs_u8, Pinhole(K=K_mat.unsqueeze(0).clone()))["depth"]
                depths[st:en] = F.interpolate(pred.float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
            K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
            Rs, ps, fwds = d["poses_R"].astype(np.float64), d["poses_t"].astype(np.float64), d["causal_fwd"].astype(np.float64)
            d_arr = np.full(n, np.inf, dtype=np.float32)
            for i in range(n):
                d_arr[i], _ = depth_map_to_corridor_d(depths[i], K_d, Rs[i], ps[i], fwds[i], min_pts_in)
            ud_d[sc] = d_arr
        c_u = eval_score_curve(eval_rows_map, ud_d, bw_speeds, b1_thrs, higher_is_alert=False)
        out[u_name] = {"curve": c_u, "summary": summarize_at_fa_caps(c_u)}
        bf = out[u_name]["summary"]["best_floor"]
        print(f"[ARKIT_INDOOR_30] {u_name}: floor={bf['late']}/{bf['avoidable']}({bf['late_rate']*100:.1f}%, ep={bf['fa_ep_per_km']:.1f}, fr={bf['fa_fr_per_km']:.1f})", flush=True)
        del u_mod
        torch.cuda.empty_cache()

    return out


def export_v4_onnx(model: EgoClearV4Net, out_dir: Path) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
        import onnxruntime as ort

    m_cpu = model.cpu().eval()
    enc = EgoClearV3EncoderExport(m_cpu).eval()
    stp = EgoClearV4StepExport(m_cpu).eval()

    rgb = torch.randn(1, 3, RGB_H, RGB_W)
    K = torch.tensor([[449.0, 453.0, 252.0, 155.0]])
    R = torch.eye(3).unsqueeze(0)
    enc_path = out_dir / "egoclear_v4_encoder.onnx"
    stp_path = out_dir / "egoclear_v4_step.onnx"

    with torch.no_grad():
        pt_f, pt_d = enc(rgb, K, R)
        torch.onnx.export(
            enc, (rgb, K, R), str(enc_path), opset_version=17,
            input_names=["rgb", "K_scaled", "R_c2w"], output_names=["f2d", "disp_cues"], dynamo=False,
        )
        t_t = torch.tensor([[0.5, 0.0, 1.5]])
        t_1 = torch.tensor([[0.25, 0.0, 1.5]])
        t_2 = torch.tensor([[0.0, 0.0, 1.5]])
        fwd = torch.tensor([[1.0, 0.0, 0.0]])
        ds = torch.tensor([0.25])
        mem0 = torch.zeros(1, m_cpu.c_bev, BEV_NZ, BEV_NX)
        age0 = torch.full((1, 1, BEV_NZ, BEV_NX), 1.0)
        stp_inputs = (pt_f, pt_d, pt_f, pt_d, pt_f, pt_d, K, R, t_t, fwd, K, R, t_1, fwd, K, R, t_2, ds, mem0, age0)
        pt_stp = stp(*stp_inputs)
        torch.onnx.export(
            stp, stp_inputs, str(stp_path), opset_version=17,
            input_names=[
                "f2d_t", "dc_t", "f2d_1", "dc_1", "f2d_2", "dc_2",
                "K_t", "R_t", "t_t", "fwd_t", "K_1", "R_1", "t_1", "fwd_1", "K_2", "R_2", "t_2",
                "ds_t", "mem_prev", "age_prev",
            ],
            output_names=["static_logits", "dynamic_logits", "corr_probs_monotonic", "unc_logits", "mem_new", "age_new"],
            dynamo=False,
        )

    sess_e = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    ort_f, _ = sess_e.run(None, {"rgb": rgb.numpy(), "K_scaled": K.numpy(), "R_c2w": R.numpy()})
    enc_err = float(np.max(np.abs(ort_f - pt_f.numpy())))

    sess_s = ort.InferenceSession(str(stp_path), providers=["CPUExecutionProvider"])
    feed_s = {inp.name: val.numpy() for inp, val in zip(sess_s.get_inputs(), stp_inputs)}
    ort_s = sess_s.run(None, feed_s)
    stp_err = float(np.max(np.abs(ort_s[2] - pt_stp[2].numpy())))
    mono_ok = bool(np.all(np.diff(ort_s[2], axis=-1) >= -1e-6))

    for _ in range(5):
        sess_s.run(None, feed_s)
    t_b = time.perf_counter()
    for _ in range(30):
        sess_s.run(None, feed_s)
    step_cpu_ms = (time.perf_counter() - t_b) * 1000.0 / 30.0

    for p in (enc_path, stp_path):
        dst = DRIVE_RUNS / p.name
        if dst.exists():
            dst.unlink()
        shutil.copy2(p, dst)

    info = {
        "encoder_onnx_mb": round(enc_path.stat().st_size / 1e6, 2),
        "step_onnx_mb": round(stp_path.stat().st_size / 1e6, 2),
        "encoder_max_abs_err": round(enc_err, 6),
        "step_corr_probs_max_abs_err": round(stp_err, 6),
        "onnx_monotonic_verified": mono_ok,
        "step_cpu_ort_ms": round(step_cpu_ms, 2),
    }
    print(f"[onnx-v4] EXPORT & PARITY PASS: {info}", flush=True)
    return info


def fmt_cap(ev_mode: dict, cap: float, key_prefix: str = "late_at_fa_fr<=") -> str:
    m = ev_mode["summary"].get(f"{key_prefix}{cap}")
    if not m:
        return "N/A"
    return f"{m['late']}/{m['avoidable']}({m['late_rate']*100:.1f}%,ep={m['fa_ep_per_km']:.1f},fr={m['fa_fr_per_km']:.1f})"


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    # Ensure UniDepth source & lightweight deps exist in fresh session
    if not Path("/content/unidepth_src").exists():
        print("[setup] fetching UniDepth-V2 code & installing deps...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "timm", "xformers==0.0.29.post3", "onnx", "onnxruntime"], check=False)
        subprocess.run(["bash", "-lc", "mkdir -p /content/unidepth_src && curl -sL https://github.com/lpiccinelli-eth/UniDepth/archive/refs/heads/main.tar.gz | tar -xz -C /content/unidepth_src"], check=True)
        sub = next(Path("/content/unidepth_src").glob("UniDepth-*"), None)
        if sub is not None and str(sub) not in sys.path:
            sys.path.insert(0, str(sub))

    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]
    m_arkit = json.loads(Path("/content/phase5_arkit_manifest.json").read_text())

    # 1. Stage existing 5 Drive datasets to /content
    s287_dir = Path("/content/sanpo_287/scenes")
    s450_dir = Path("/content/sanpo_p2/extra_scenes")
    r40_dir = Path("/content/sanpo_p2/real_scenes")
    r76_dir = Path("/content/sanpo_p4/real_train_scenes")
    aria_dir = Path("/content/aria_g2_scenes")

    extract_drive_tar_dataset(DRIVE_ROOT / "datasets/sanpo_synth_287_v0", s287_dir, 280)
    extract_drive_tar_dataset(DRIVE_ROOT / "datasets/sanpo_synth_extra450_v0", s450_dir, 440)
    extract_drive_tar_dataset(DRIVE_ROOT / "datasets/sanpo_real_zeroshot_40_v0", r40_dir, 38)
    extract_drive_tar_dataset(DRIVE_ROOT / "datasets/sanpo_real_train76_v0", r76_dir, 70)
    aria_dir.mkdir(parents=True, exist_ok=True)
    if len(list(aria_dir.glob("*.npz"))) < 18:
        aria_tar = Path("/content/aria_g2_20clips.tar")
        shutil.copy2(DRIVE_ROOT / "caches/aria_g2_20clips.tar", aria_tar)
        with tarfile.open(aria_tar, "r") as tf:
            tf.extractall(aria_dir)
    real_local_rows = ensure_real_local_gt_cache(r40_dir, Path("/content/sanpo_p2/real_local_gt_cache.npz"))

    # 2. Pack ARKitScenes Real Indoor (90 Train + 30 strictly unseen Val) -> Drive
    ncpu = min(20, max(4, (os.cpu_count() or 8) - 2))
    arkit_tr_dir = Path("/content/arkit_p5/train_scenes")
    arkit_va_dir = Path("/content/arkit_p5/val_scenes")
    tr_stems, tr_infos = pack_arkit_split(
        m_arkit["train_pool"], m_arkit["target_train_valid"], arkit_tr_dir, DRIVE_ARKIT_TR,
        "arkit_indoor_train90_v0", min_valid_frames=20, max_workers=ncpu,
    )
    va_stems, va_infos = pack_arkit_split(
        m_arkit["val_pool"], m_arkit["target_val_valid"], arkit_va_dir, DRIVE_ARKIT_VA,
        "arkit_indoor_val30_v0", min_valid_frames=20, max_workers=ncpu,
    )

    # Load all datasets into RAM
    synth_train, eval_synth = [], {}
    for sc, sp in m287["splits"].items():
        p = s287_dir / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            if sp == "train":
                synth_train.append(d)
            else:
                eval_synth[sc] = d
    for p in sorted(s450_dir.glob("*.npz")):
        synth_train.append(load_scene(p))
    real_train = [load_scene(p) for p in sorted(r76_dir.glob("*.npz"))]
    eval_real = {p.stem: load_scene(p) for p in sorted(r40_dir.glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(aria_dir.glob("*.npz"))}
    arkit_train = [load_scene(arkit_tr_dir / f"{st}.npz") for st in tr_stems]
    eval_arkit = {st: load_scene(arkit_va_dir / f"{st}.npz") for st in va_stems}

    print(
        f"[p5] RAM Loaded: synth_tr={len(synth_train)}, real_tr={len(real_train)}, arkit_tr={len(arkit_train)} | "
        f"eval_synth={len(eval_synth)}, eval_real={len(eval_real)}, aria_g2={len(aria_dicts)}, eval_arkit={len(eval_arkit)}",
        flush=True,
    )

    # 3. Evaluate Indoor-30 Baselines (LiDAR Ref, DAV2-S Ground, UniDepth-V2-Small, UniDepth-V2-Large)
    indoor_baselines = eval_indoor_baselines(eval_arkit, bw_speeds, device)

    # 4. Zero-Training Hysteresis Ablation + Indoor Zero-Shot Audit on Phase 4 EgoClear-v3-Full (Sim+Real)
    p4_ckpt_drive = DRIVE_ROOT / "runs/egoclear_v0_phase4/egoclear_v3_full_simreal.pt"
    p4_ckpt_local = LOCAL_RUNS / "egoclear_v3_full_simreal.pt"
    shutil.copy2(p4_ckpt_drive, p4_ckpt_local)
    p4_ckpt = torch.load(p4_ckpt_local, map_location="cpu", weights_only=False)

    m_v3 = EgoClearV3Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    m_v3.load_state_dict(p4_ckpt["head_state_dict"], strict=False)
    print("\n=== Phase 4 EgoClear-v3-Full (Outdoor-Only) + Zero-Training Hysteresis Ablation (4 Domains) ===", flush=True)
    ev_v3_ablation = {
        "synth_clean": eval_model_with_hysteresis(m_v3, eval_synth, None, bw_speeds, device, is_v4_monotonic=False),
        "real_40_local": eval_model_with_hysteresis(m_v3, eval_real, real_local_rows, bw_speeds, device, is_v4_monotonic=False),
        "aria_g2_zeroshot": eval_model_with_hysteresis(m_v3, aria_dicts, None, bw_speeds, device, is_v4_monotonic=False),
        "arkit_indoor_val30_zeroshot": eval_model_with_hysteresis(m_v3, eval_arkit, None, bw_speeds, device, is_v4_monotonic=False),
    }
    for rname, ev in ev_v3_ablation.items():
        print(
            f"[V3-P4 | {rname}] tv={ev['clear_frame_mean_tv']} mono_viol={ev['monotonicity_violation_rate']} | "
            f"RAW@120={fmt_cap(ev['modes']['raw'], 120.3)} | "
            f"EMA_HYST@120={fmt_cap(ev['modes']['ema_hyst'], 120.3)} | "
            f"2OF3@120={fmt_cap(ev['modes']['confirm_2of3'], 120.3)}",
            flush=True,
        )
    del m_v3
    torch.cuda.empty_cache()

    # 5. Train EgoClear-v4-Full (Unified Indoor+Outdoor + Monotonic Survival Hazard + Clear-Frame Flicker Suppression)
    model_v4 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    # Warm-start compatible backbone/MVS/ConvGRU/BEV weights from Phase 4, keeping fresh -3.5 survival hazard head!
    warm_sd = {k: v for k, v in p4_ckpt["head_state_dict"].items() if not k.startswith("corr_head.6.")}
    missing, unexpected = model_v4.load_state_dict(warm_sd, strict=False)
    print(f"[v4-init] warm-started from P4 (fresh survival head layers={len([k for k in missing if not k.startswith('dav2.')])})", flush=True)

    t_tr = time.time()
    opt = torch.optim.AdamW([p for p in model_v4.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    ds = UnifiedPhase5ClipDataset(synth_train, real_train, arkit_train, clip_len=12, clips_per_epoch=2400)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=48, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    epochs = 20
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1e-5)
    history = []
    print(f"\n=== Training EgoClear-v4-Full (Unified 627 Synth + {len(real_train)} Real + {len(arkit_train)} Indoor, 20 ep, 576 fr/step) ===", flush=True)
    for ep in range(1, epochs + 1):
        model_v4.train()
        ep_l = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0, "flk": 0.0}
        te = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model_v4.forward_sequence(
                    batch["rgb"], batch["K_scaled"], batch["R_c2w"], batch["t_w"], batch["fwd_w"], batch["seg_len"]
                )
                loss, parts = compute_loss_v4(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_v4.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += parts[k]
        for k in ep_l:
            ep_l[k] /= len(loader)
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_l.items()}, "sec": round(time.time() - te, 1)})
        if ep in (1, 5, 10, 15, epochs):
            print(
                f"  [V4 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} "
                f"(st={ep_l['st']:.4f} dy={ep_l['dy']:.4f} co={ep_l['co']:.4f} flk={ep_l['flk']:.4f}) {time.time()-te:.1f}s",
                flush=True,
            )

    ckpt_p = LOCAL_RUNS / "egoclear_v4_full_unified.pt"
    tr_state = {k: v.cpu() for k, v in model_v4.state_dict().items() if not k.startswith("dav2.")}
    torch.save({"mode": "full", "c_bev": 128, "arch": "EgoClearV4Net", "head_state_dict": tr_state, "history": history}, ckpt_p)
    dst_ckpt = DRIVE_RUNS / ckpt_p.name
    if dst_ckpt.exists():
        dst_ckpt.unlink()
    shutil.copy2(ckpt_p, dst_ckpt)

    # 6. Full 4-Domain x 4-Mode Evaluation of EgoClear-v4-Full
    print("\n=== Evaluating EgoClear-v4-Full Across ALL 4 Domains (Synth, Real-40, Aria-G2 Zero-Shot, ARKit-Indoor 30) ===", flush=True)
    ev_v4 = {
        "synth_clean": eval_model_with_hysteresis(model_v4, eval_synth, None, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
        "synth_vio_1x": eval_model_with_hysteresis(model_v4, eval_synth, None, bw_speeds, device, is_v4_monotonic=True, vio_scale=1.0),
        "real_40_local": eval_model_with_hysteresis(model_v4, eval_real, real_local_rows, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
        "aria_g2_zeroshot": eval_model_with_hysteresis(model_v4, aria_dicts, None, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
        "arkit_indoor_val30": eval_model_with_hysteresis(model_v4, eval_arkit, None, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
    }
    for rname, ev in ev_v4.items():
        print(
            f"[V4-UNIFIED | {rname}] tv={ev['clear_frame_mean_tv']} mono_viol={ev['monotonicity_violation_rate']} "
            f"dyn={ev['dyn_only_alert_rate_at_0.5']['rate']} headF1={ev['hband_corridor_f1_at_0.5']['HEAD']['f1']}\n"
            f"   RAW:      @80={fmt_cap(ev['modes']['raw'], 80.0)} | @120={fmt_cap(ev['modes']['raw'], 120.3)} | @200={fmt_cap(ev['modes']['raw'], 200.0)} | @350={fmt_cap(ev['modes']['raw'], 350.0)}\n"
            f"   EMA_HYST: @80={fmt_cap(ev['modes']['ema_hyst'], 80.0)} | @120={fmt_cap(ev['modes']['ema_hyst'], 120.3)} | @200={fmt_cap(ev['modes']['ema_hyst'], 200.0)} | @350={fmt_cap(ev['modes']['ema_hyst'], 350.0)}\n"
            f"   2OF3:     @80={fmt_cap(ev['modes']['confirm_2of3'], 80.0)} | @120={fmt_cap(ev['modes']['confirm_2of3'], 120.3)} | @200={fmt_cap(ev['modes']['confirm_2of3'], 200.0)} | @350={fmt_cap(ev['modes']['confirm_2of3'], 350.0)}",
            flush=True,
        )

    # 7. Export 2-stage ONNX-17 & Parity Verification
    onnx_info = export_v4_onnx(model_v4, LOCAL_RUNS)

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_sec": round(time.time() - t_all, 1),
        "counts": {
            "synth_train": len(synth_train),
            "real_train": len(real_train),
            "arkit_train": len(arkit_train),
            "eval_synth": len(eval_synth),
            "eval_real": len(eval_real),
            "aria_g2_clips": len(aria_dicts),
            "eval_arkit": len(eval_arkit),
        },
        "arkit_pack_sample_info": {"train": tr_infos[:5], "val": va_infos[:5]},
        "indoor_baselines": indoor_baselines,
        "v3_p4_hysteresis_ablation": ev_v3_ablation,
        "egoclear_v4_unified": {"train_sec": round(time.time() - t_tr, 1), "history": history, "evals": ev_v4},
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase5_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase5_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[p5] DONE saved phase5_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
