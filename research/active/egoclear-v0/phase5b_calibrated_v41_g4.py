"""Phase 5b G4 (96 GB Blackwell): EgoClear-v4.1 Calibrated Unified Model + Multi-Threaded Drive Staging + Slope-Aware Hysteresis.

Targets the exact two trade-offs diagnosed in Phase 5:
  1. Multi-domain balance: 45% SANPO-Synth (627) + 40% SANPO-Real Outdoor (75) + 15% ARKit-Indoor (90)
     (restores full 40% outdoor-real weight while retaining indoor HEAD/furniture generalization).
  2. Calibrated survival + flicker loss (compute_loss_v4_calibrated): gentle label smoothing (0.015),
     balanced pos_w=2.1, flicker weight 0.35 -> preserves strict 0.0% monotonicity & low jitter without
     crushing high-recall / cross-hardware probability ranking.
  3. High-resolution threshold sweep up to 0.9985 + new strictly causal `slope_aware_hyst` (0-frame delay
     on rising obstacle ramps, filters isolated 1-frame baseline glitches).
  4. 6-thread parallel Drive-to-/content staging for all 7 cached tar datasets (~3x faster startup).
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/content")

from egoclear_model import BEV_DZ, BEV_NX, BEV_NZ, BEV_Z_MIN, RGB_H, RGB_W
from egoclear_v3_model import EgoClearV3Net
from egoclear_v4_model import EgoClearV4Net, EgoClearV4StepExport, compute_loss_v4_calibrated
from eval_closed_loop import H, apply_causal_hysteresis, eval_score_curve, summarize_at_fa_caps
from phase4_simreal_aria_onnx_g4 import EgoClearV3EncoderExport
from phase5_unified_indoor_hysteresis_g4 import ensure_real_local_gt_cache
from train_and_eval_g4 import IMAGENET_MEAN, IMAGENET_STD, decode_rgb_batch, load_scene
from train_and_eval_phase2_g4 import apply_focal_zoom_crop, perturb_vio_trajectory

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist")
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase5b"
LOCAL_RUNS = Path("/content/egoclear_runs_p5b")

THRS_HIRES = np.array(
    [
        0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
        0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52,
        0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88,
        0.92, 0.94, 0.96, 0.975, 0.985, 0.992, 0.996, 0.9985,
    ],
    dtype=np.float64,
)


def _copy_and_extract_one(args: tuple[Path, Path]) -> tuple[str, float]:
    src, dst_dir = args
    t0 = time.time()
    tmp = dst_dir / f".tmp_{src.name}"
    shutil.copy2(src, tmp)
    with tarfile.open(tmp, "r") as tf:
        tf.extractall(dst_dir)
    tmp.unlink()
    return src.name, time.time() - t0


def stage_all_datasets_parallel() -> dict[str, Path]:
    """Copy & extract all 7 cached tar datasets from Google Drive using 6 concurrent threads."""
    t0 = time.time()
    specs = [
        ("synth_287", DRIVE_ROOT / "datasets/sanpo_synth_287_v0", Path("/content/sanpo_287/scenes"), 280),
        ("synth_450", DRIVE_ROOT / "datasets/sanpo_synth_extra450_v0", Path("/content/sanpo_p2/extra_scenes"), 440),
        ("real_40", DRIVE_ROOT / "datasets/sanpo_real_zeroshot_40_v0", Path("/content/sanpo_p2/real_scenes"), 38),
        ("real_75", DRIVE_ROOT / "datasets/sanpo_real_train76_v0", Path("/content/sanpo_p4/real_train_scenes"), 70),
        ("arkit_tr", DRIVE_ROOT / "datasets/arkit_indoor_train90_v0", Path("/content/arkit_p5/train_scenes"), 85),
        ("arkit_va", DRIVE_ROOT / "datasets/arkit_indoor_val30_v0", Path("/content/arkit_p5/val_scenes"), 28),
    ]
    jobs = []
    paths = {}
    for key, d_dir, l_dir, min_n in specs:
        l_dir.mkdir(parents=True, exist_ok=True)
        paths[key] = l_dir
        if len(list(l_dir.glob("*.npz"))) >= min_n:
            continue
        man_p = d_dir / "manifest_sha256.json"
        if man_p.exists():
            man = json.loads(man_p.read_text())
            sh_files = [d_dir / sh["file"] for sh in man["shards"]]
        else:
            sh_files = sorted(d_dir.glob("*.tar"))
        for sf in sh_files:
            jobs.append((sf, l_dir))

    aria_dir = Path("/content/aria_g2_scenes")
    aria_dir.mkdir(parents=True, exist_ok=True)
    paths["aria"] = aria_dir
    if len(list(aria_dir.glob("*.npz"))) < 18:
        jobs.append((DRIVE_ROOT / "caches/aria_g2_20clips.tar", aria_dir))

    if jobs:
        print(f"[stage-mt] copying & extracting {len(jobs)} tar shards across 6 threads...", flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(_copy_and_extract_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                name, sec = fut.result()
                done += 1
                if done in (1, len(jobs) // 2, len(jobs)):
                    print(f"  [stage-mt] {done}/{len(jobs)} done (latest {name} in {sec:.1f}s, elapsed {time.time()-t0:.1f}s)", flush=True)
    print(f"[stage-mt] ALL 7 datasets ready locally in {time.time()-t0:.1f}s!", flush=True)
    return paths


class CalibratedPhase5bDataset(torch.utils.data.Dataset):
    """Balanced multi-domain sampler: 45% SANPO-Synth + 40% SANPO-Real Outdoor + 15% ARKit-Indoor."""

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
            f"[dataset-v4.1] pools: synth={len(self.pools['synth'])} (45%), "
            f"real={len(self.pools['real'])} (40%), arkit={len(self.pools['arkit'])} (15%)",
            flush=True,
        )

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        u_dom = rng.random()
        if u_dom < 0.45:
            dom = "synth"
        elif u_dom < 0.85 and len(self.pools["real"]) > 0:
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

        rgb = decode_rgb_batch(d, sl)
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)

        if rng.random() < 0.65:
            zoom = float(rng.uniform(0.70, 1.20))
            rgb, K_s = apply_focal_zoom_crop(rgb, K_s, zoom)

        gamma = float(rng.uniform(0.72, 1.40))
        rgb = np.power(np.clip(rgb, 1e-4, 1.0), gamma)
        ch_gain = rng.uniform(0.84, 1.16, size=(1, 3, 1, 1)).astype(np.float32)
        c_fac = float(rng.uniform(0.80, 1.25))
        mean = rgb.mean(axis=(2, 3), keepdims=True)
        rgb = np.clip(((rgb - mean) * c_fac + mean) * ch_gain, 0.0, 1.0)
        if rng.random() < 0.20:
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
def eval_hires(
    model: nn.Module,
    scene_dicts: dict[str, dict],
    eval_rows_override: dict[str, np.ndarray] | None,
    bw_speeds: dict,
    device: torch.device,
    is_v4_monotonic: bool,
    vio_scale: float = 0.0,
    compute_uncertainty_audit: bool = False,
) -> dict:
    model.eval()
    eval_rows_map, corr_probs_map, unc_probs_map = {}, {}, {}
    stat_ap_num = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
    dyn_fp_frames = dyn_tot_frames = 0
    clear_tv_sum = 0.0
    clear_pair_cnt = 0
    mono_viol_cnt = mono_tot_cnt = 0
    tot_ms = 0.0
    tot_fr = 0

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))

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

        p_corr = out["corr_probs"][0].float().cpu().numpy() if is_v4_monotonic else torch.sigmoid(out["corr_logits"][0].float()).cpu().numpy()
        p_unc = torch.sigmoid(out["unc_logits"][0].float()).cpu().numpy()
        p_stat = torch.sigmoid(out["static_logits"][0].float()).cpu().numpy()

        eval_rows_map[sc] = er
        corr_probs_map[sc] = p_corr
        unc_probs_map[sc] = p_unc

        diffs_z = np.diff(p_corr, axis=1)
        mono_viol_cnt += int((diffs_z < -1e-4).sum())
        mono_tot_cnt += int(diffs_z.size)

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

    fixed_scores = {
        sc: (probs[:, idx_4m] if is_v4_monotonic else probs[:, : idx_4m + 1].max(axis=1))
        for sc, probs in corr_probs_map.items()
    }

    modes_out = {}
    for hmode in ("raw", "slope_aware_hyst", "ema_hyst", "confirm_2of3"):
        c = eval_score_curve(eval_rows_map, fixed_scores, bw_speeds, THRS_HIRES, higher_is_alert=True, hysteresis_mode=hmode)
        modes_out[hmode] = {"curve": c, "summary": summarize_at_fa_caps(c)}

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
            all_p.append(fixed_scores[sc][valid])
            all_u.append(unc_probs_map[sc][valid, : idx_4m + 1].mean(axis=1))
            all_y.append((er[valid, 0] <= H).astype(np.int32))
        ap, au, ay = np.concatenate(all_p), np.concatenate(all_u), np.concatenate(all_y)
        order = np.argsort(au)
        unc_audit = {}
        for cov in (1.0, 0.90, 0.80, 0.70):
            k = max(1, int(len(order) * cov))
            sel = order[:k]
            unc_audit[f"cov_{int(cov*100)}"] = {"n_frames": k, "err": round(float(np.mean((ap[sel] >= 0.32).astype(np.int32) != ay[sel])), 4)}

    return {
        "ms_per_frame_stream": round(tot_ms / max(tot_fr, 1), 2),
        "clear_frame_mean_tv": round(clear_tv_sum / max(clear_pair_cnt, 1), 5),
        "monotonicity_violation_rate": round(mono_viol_cnt / max(mono_tot_cnt, 1), 6),
        "hband_corridor_f1_at_0.5": hband_f1,
        "dyn_only_alert_rate_at_0.5": {
            "fp": dyn_fp_frames, "total": dyn_tot_frames,
            "rate": round(dyn_fp_frames / max(dyn_tot_frames, 1), 4) if dyn_tot_frames else None,
        },
        "uncertainty_risk_coverage": unc_audit,
        "modes": modes_out,
    }


def export_v41_onnx(model: EgoClearV4Net, out_dir: Path) -> dict:
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
    enc_path = out_dir / "egoclear_v41_encoder.onnx"
    stp_path = out_dir / "egoclear_v41_step.onnx"

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
    print(f"[onnx-v4.1] EXPORT & PARITY PASS: {info}", flush=True)
    return info


def fmt_m(ev_mode: dict, key: str) -> str:
    m = ev_mode["summary"].get(key)
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

    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    # 1. Parallel 6-thread staging of ALL 7 Drive datasets (no Apple CDN re-download!)
    paths = stage_all_datasets_parallel()
    real_local_rows = ensure_real_local_gt_cache(paths["real_40"], Path("/content/sanpo_p2/real_local_gt_cache.npz"))

    synth_train, eval_synth = [], {}
    for sc, sp in m287["splits"].items():
        p = paths["synth_287"] / f"{sc}.npz"
        if p.exists():
            d = load_scene(p)
            if sp == "train":
                synth_train.append(d)
            else:
                eval_synth[sc] = d
    for p in sorted(paths["synth_450"].glob("*.npz")):
        synth_train.append(load_scene(p))
    real_train = [load_scene(p) for p in sorted(paths["real_75"].glob("*.npz"))]
    eval_real = {p.stem: load_scene(p) for p in sorted(paths["real_40"].glob("*.npz"))}
    aria_dicts = {p.stem: load_scene(p) for p in sorted(paths["aria"].glob("*.npz"))}
    arkit_train = [load_scene(p) for p in sorted(paths["arkit_tr"].glob("*.npz"))]
    eval_arkit = {p.stem: load_scene(p) for p in sorted(paths["arkit_va"].glob("*.npz"))}

    print(
        f"[p5b] Loaded: synth_tr={len(synth_train)}, real_tr={len(real_train)}, arkit_tr={len(arkit_train)} | "
        f"eval_synth={len(eval_synth)}, eval_real={len(eval_real)}, aria_g2={len(aria_dicts)}, eval_arkit={len(eval_arkit)}",
        flush=True,
    )

    # 2. Re-evaluate V4.0 with high-res threshold sweep (up to 0.9985) + slope_aware_hyst for fair comparison
    p5_ckpt_local = LOCAL_RUNS / "egoclear_v4_full_unified.pt"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase5/egoclear_v4_full_unified.pt", p5_ckpt_local)
    p5_ckpt = torch.load(p5_ckpt_local, map_location="cpu", weights_only=False)

    m_v40 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    m_v40.load_state_dict(p5_ckpt["head_state_dict"], strict=False)
    print("\n=== Re-evaluating V4.0 (30% indoor) with Hi-Res Sweep (up to 0.9985) + slope_aware_hyst ===", flush=True)
    ev_v40_hires = {
        "real_40_local": eval_hires(m_v40, eval_real, real_local_rows, bw_speeds, device, is_v4_monotonic=True),
        "aria_g2_zeroshot": eval_hires(m_v40, aria_dicts, None, bw_speeds, device, is_v4_monotonic=True),
        "arkit_indoor_val30": eval_hires(m_v40, eval_arkit, None, bw_speeds, device, is_v4_monotonic=True),
    }
    for rname, ev in ev_v40_hires.items():
        print(
            f"[V4.0-HIRES | {rname}] RAW: @80={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=80.0')} | "
            f"@120={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=120.3')} | @500={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=500.0')}\n"
            f"   SLOPE_HYST: @120={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=120.3')} | @500={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=500.0')}",
            flush=True,
        )
    del m_v40
    torch.cuda.empty_cache()

    # 3. Train EgoClear-v4.1-Calibrated (warm-start from P4 outdoor backbone + fresh calibrated survival head, 16 ep)
    p4_ckpt_local = LOCAL_RUNS / "egoclear_v3_full_simreal.pt"
    shutil.copy2(DRIVE_ROOT / "runs/egoclear_v0_phase4/egoclear_v3_full_simreal.pt", p4_ckpt_local)
    p4_ckpt = torch.load(p4_ckpt_local, map_location="cpu", weights_only=False)

    model_v41 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    warm_sd = {k: v for k, v in p4_ckpt["head_state_dict"].items() if not k.startswith("corr_head.6.")}
    model_v41.load_state_dict(warm_sd, strict=False)

    t_tr = time.time()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bs = 48 if vram_gb >= 48.0 else (32 if vram_gb >= 32.0 else 16)
    lr = 2.5e-4 * (bs / 48.0) ** 0.5
    n_workers = min(12, max(4, (os.cpu_count() or 8) - 2))
    print(f"[hw] GPU={torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM), auto bs={bs}, lr={lr:.2e}, workers={n_workers}", flush=True)
    opt = torch.optim.AdamW([p for p in model_v41.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    ds = CalibratedPhase5bDataset(synth_train, real_train, arkit_train, clip_len=12, clips_per_epoch=2400)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True
    )
    epochs = 16
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=1e-5)
    history = []
    print(f"\n=== Training EgoClear-v4.1-Calibrated (45% Synth + 40% Real + 15% Indoor, 16 ep, 576 fr/step) ===", flush=True)
    for ep in range(1, epochs + 1):
        model_v41.train()
        ep_l = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0, "flk": 0.0}
        te = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model_v41.forward_sequence(
                    batch["rgb"], batch["K_scaled"], batch["R_c2w"], batch["t_w"], batch["fwd_w"], batch["seg_len"]
                )
                loss, parts = compute_loss_v4_calibrated(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_v41.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_l:
                ep_l[k] += parts[k]
        for k in ep_l:
            ep_l[k] /= len(loader)
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_l.items()}, "sec": round(time.time() - te, 1)})
        if ep in (1, 4, 8, 12, epochs):
            print(
                f"  [V4.1 ep {ep:02d}/{epochs}] tot={ep_l['tot']:.4f} "
                f"(st={ep_l['st']:.4f} dy={ep_l['dy']:.4f} co={ep_l['co']:.4f} flk={ep_l['flk']:.4f}) {time.time()-te:.1f}s",
                flush=True,
            )

    ckpt_p = LOCAL_RUNS / "egoclear_v41_full_calibrated.pt"
    tr_state = {k: v.cpu() for k, v in model_v41.state_dict().items() if not k.startswith("dav2.")}
    torch.save({"mode": "full", "c_bev": 128, "arch": "EgoClearV4Net-v4.1", "head_state_dict": tr_state, "history": history}, ckpt_p)
    dst_ckpt = DRIVE_RUNS / ckpt_p.name
    if dst_ckpt.exists():
        dst_ckpt.unlink()
    shutil.copy2(ckpt_p, dst_ckpt)

    # 4. Full 4-Domain x 4-Mode Hi-Res Evaluation of EgoClear-v4.1-Calibrated
    print("\n=== Evaluating EgoClear-v4.1-Calibrated Across ALL 4 Domains ===", flush=True)
    ev_v41 = {
        "synth_clean": eval_hires(model_v41, eval_synth, None, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
        "synth_vio_1x": eval_hires(model_v41, eval_synth, None, bw_speeds, device, is_v4_monotonic=True, vio_scale=1.0),
        "real_40_local": eval_hires(model_v41, eval_real, real_local_rows, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
        "aria_g2_zeroshot": eval_hires(model_v41, aria_dicts, None, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
        "arkit_indoor_val30": eval_hires(model_v41, eval_arkit, None, bw_speeds, device, is_v4_monotonic=True, compute_uncertainty_audit=True),
    }
    for rname, ev in ev_v41.items():
        print(
            f"[V4.1-CAL | {rname}] tv={ev['clear_frame_mean_tv']} mono={ev['monotonicity_violation_rate']} "
            f"dyn={ev['dyn_only_alert_rate_at_0.5']['rate']} headF1={ev['hband_corridor_f1_at_0.5']['HEAD']['f1']}\n"
            f"   RAW:        @80={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=80.0')} | @120={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=120.3')} | @200={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=200.0')} | @500={fmt_m(ev['modes']['raw'],'late_at_fa_fr<=500.0')}\n"
            f"   SLOPE_HYST: @80={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=80.0')} | @120={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=120.3')} | @200={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=200.0')} | @500={fmt_m(ev['modes']['slope_aware_hyst'],'late_at_fa_fr<=500.0')}\n"
            f"   EMA_HYST:   @80={fmt_m(ev['modes']['ema_hyst'],'late_at_fa_fr<=80.0')} | @120={fmt_m(ev['modes']['ema_hyst'],'late_at_fa_fr<=120.3')} | @200={fmt_m(ev['modes']['ema_hyst'],'late_at_fa_fr<=200.0')} | @500={fmt_m(ev['modes']['ema_hyst'],'late_at_fa_fr<=500.0')}",
            flush=True,
        )

    # 5. Export ONNX-17 for EgoClear-v4.1
    onnx_info = export_v41_onnx(model_v41, LOCAL_RUNS)

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_sec": round(time.time() - t_all, 1),
        "train_sec": round(time.time() - t_tr, 1),
        "mix": {"synth": 0.45, "real_outdoor": 0.40, "arkit_indoor": 0.15},
        "v40_hires_recheck": ev_v40_hires,
        "egoclear_v41_calibrated": {"history": history, "evals": ev_v41},
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase5b_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase5b_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[p5b] DONE saved phase5b_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
