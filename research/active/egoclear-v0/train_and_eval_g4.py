"""Phase 1 G4 (96 GB) experiment: Pure-Vision Baselines vs EgoClear-Net (Single / MVS / Full).

Runs on the 287 SANPO-Synthetic dev sessions packed in Drive:
  /content/drive/MyDrive/BlindAssist/datasets/sanpo_synth_287_v0/
Strictly isolates 177 train sessions from 110 held-out eval sessions (official SANPO test split is never read).

Stage 1: Verify SHA-256 and extract tar shards to local /content/sanpo_287/
Stage 2: Exact walk-alert-r0 v2 parity check on GT-Single (all 287 & 110 eval) + GT-Memory references
Stage 3: Pretrained Pure-Vision Baselines on 110 eval sessions:
         - DAV2-Small (ground-anchored)
         - DAV2-Metric-Outdoor-Small
         - DAV2-Metric-Outdoor-Large (335M)
         - DAV2-Metric-Outdoor-Large + Explicit 6m 3D Point Memory
Stage 4: Train & evaluate EgoClear-Net ablations (strictly on 177 train, evaluated on 110 eval):
         - EgoClear-Single
         - EgoClear-MVS (+ 2/3-frame static motion-parallax cost volume)
         - EgoClear-Full (+ metric-warped visibility/contradiction-gated latent BEV memory)
Stage 5: Save checkpoints + full Pareto JSON to Drive & flush_and_unmount().
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
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
from eval_closed_loop import (
    H,
    eval_score_curve,
    make_episodes,
    score_episode,
    summarize_at_fa_caps,
)

DRIVE_DATA = Path("/content/drive/MyDrive/BlindAssist/datasets/sanpo_synth_287_v0")
DRIVE_RUNS = Path("/content/drive/MyDrive/BlindAssist/runs/egoclear_v0_phase1")
LOCAL_DATA = Path("/content/sanpo_287")
LOCAL_RUNS = Path("/content/egoclear_runs")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

# Corridor box constants matching walk-alert-r0
X_HALF, FWD0, HGT_LO, HGT_HI = 0.30, 0.30, 0.15, 1.90
MIN_PTS_S8 = 21
S = 8
DYN = [12, 13, 14, 21]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_extract_dataset() -> dict:
    t0 = time.time()
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    scenes_dir = LOCAL_DATA / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DRIVE_DATA / "manifest_sha256.json").read_text())
    marker = LOCAL_DATA / ".extracted_ok"
    if marker.exists() and len(list(scenes_dir.glob("*.npz"))) >= manifest["n_sessions"]:
        print(f"[data] already extracted locally ({manifest['n_sessions']} scenes)", flush=True)
        return manifest

    for sh in manifest["shards"]:
        src = DRIVE_DATA / sh["file"]
        dst = LOCAL_DATA / sh["file"]
        shutil.copy2(src, dst)
        got = sha256_file(dst)
        if got != sh["sha256"]:
            raise RuntimeError(f"SHA256 mismatch on {sh['file']}: got {got}, want {sh['sha256']}")
        with tarfile.open(dst, "r") as tf:
            tf.extractall(scenes_dir)
        dst.unlink()
    marker.write_text("ok")
    print(
        f"[data] copied, verified, and extracted {manifest['n_sessions']} scenes "
        f"({manifest['n_frames']} frames) in {time.time()-t0:.1f}s",
        flush=True,
    )
    return manifest


def load_scene(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def decode_rgb_batch(scene_d: dict, indices: list[int] | np.ndarray) -> np.ndarray:
    """Return (len(indices), 3, 294, 518) float32 RGB in [0, 1]."""
    blob = scene_d["rgb_blob"]
    offs = scene_d["rgb_offsets"]
    out = np.empty((len(indices), 3, RGB_H, RGB_W), dtype=np.float32)
    for k, idx in enumerate(indices):
        raw = blob[offs[idx] : offs[idx + 1]]
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out[k] = rgb.transpose(2, 0, 1)
    return out


def corridor_d(z: np.ndarray, n_min: int) -> float:
    if len(z) < n_min:
        return float(np.inf)
    z = np.sort(z)
    cnt = np.searchsorted(z, z + 0.30, side="right") - np.arange(len(z))
    ok = np.nonzero(cnt >= n_min)[0]
    return float(z[ok[0]]) if len(ok) else float(np.inf)


def depth_map_to_corridor_d(
    depth_hw: np.ndarray,  # (H_d, W_d) optical z in meters
    K_d: np.ndarray,       # [fx, fy, cx, cy] at (H_d, W_d)
    R_c2w: np.ndarray,     # (3, 3)
    t_w: np.ndarray,       # (3,)
    fwd: np.ndarray,       # (3,)
    min_pts: int,
) -> tuple[float, np.ndarray]:
    """Backproject optical depth map into world points & return (corridor_d, world_pts)."""
    hd, wd = depth_hw.shape
    vv, uu = np.mgrid[0:hd, 0:wd]
    d = depth_hw.reshape(-1)
    ok = np.isfinite(d) & (d > 0.15) & (d < 25.0)
    if not np.any(ok):
        return float(np.inf), np.zeros((0, 3), dtype=np.float32)
    u = uu.reshape(-1)[ok]
    v = vv.reshape(-1)[ok]
    dk = d[ok]
    pts_c = np.stack([(u - K_d[2]) / K_d[0] * dk, (v - K_d[3]) / K_d[1] * dk, dk], axis=-1)
    pts_w = (R_c2w @ pts_c.T).T + t_w
    right = np.array([fwd[1], -fwd[0], 0.0], dtype=np.float64)
    rel = pts_w - t_w
    xq = rel @ right
    zq = rel @ fwd
    strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
    g_est = np.percentile(pts_w[strip, 2], 5) if strip.sum() >= 30 else t_w[2] - 1.5
    hq = pts_w[:, 2] - g_est
    c_m = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT_LO) & (hq <= HGT_HI)
    return corridor_d(zq[c_m], min_pts), pts_w.astype(np.float32)


def ground_anchor_relative_depth(
    rel_disp: np.ndarray,  # (H_d, W_d) higher = closer (DAV2 relative disparity)
    K_d: np.ndarray,
    R_c2w: np.ndarray,
    nom_cam_h: float,
) -> np.ndarray:
    """Causal ground-plane affine anchoring: disp = a * (1 / z_floor) + b using rays pointing down."""
    hd, wd = rel_disp.shape
    vv, uu = np.mgrid[0:hd, 0:wd]
    rays_c = np.stack([(uu - K_d[2]) / K_d[0], (vv - K_d[3]) / K_d[1], np.ones_like(uu)], axis=-1)
    rays_w = rays_c @ R_c2w.T  # (hd, wd, 3)
    # Ray intersects flat ground z_w = -nom_cam_h at scale t = -nom_cam_h / rays_w[..., 2]
    rz = rays_w[..., 2]
    m_down = rz < -0.10
    z_floor = np.where(m_down, -nom_cam_h / np.minimum(rz, -1e-4), np.nan)
    # Candidate ground region: lower half of image, expected floor distance 0.8..6.0 m
    cand = m_down & (vv >= hd * 0.45) & (z_floor >= 0.8) & (z_floor <= 6.0) & np.isfinite(rel_disp)
    inv_z = 1.0 / z_floor[cand]
    disp = rel_disp[cand].astype(np.float64)
    if len(disp) < 80:
        return np.full_like(rel_disp, np.inf, dtype=np.float32)
    # Robust Huber/IRLS or median-split RANSAC for disp = a * inv_z + b
    rng = np.random.default_rng(0)
    best_in, best_ab = -1, None
    idx_all = np.arange(len(disp))
    for _ in range(40):
        i1, i2 = rng.choice(idx_all, 2, replace=False)
        dx = inv_z[i2] - inv_z[i1]
        if abs(dx) < 0.08:
            continue
        a = (disp[i2] - disp[i1]) / dx
        if a <= 1e-3:
            continue
        b = disp[i1] - a * inv_z[i1]
        pred = a * inv_z + b
        resid = np.abs(disp - pred)
        thr = max(0.08 * np.median(np.abs(disp)), 1e-3)
        nin = int((resid <= thr).sum())
        if nin > best_in:
            best_in, best_ab = nin, (a, b)
    if best_ab is None or best_in < 40:
        return np.full_like(rel_disp, np.inf, dtype=np.float32)
    a, b = best_ab
    # Refit on inliers
    resid = np.abs(disp - (a * inv_z + b))
    thr = max(0.08 * np.median(np.abs(disp)), 1e-3)
    inl = resid <= thr
    A_mat = np.column_stack([inv_z[inl], np.ones(inl.sum())])
    ab, *_ = np.linalg.lstsq(A_mat, disp[inl], rcond=None)
    if ab[0] <= 1e-3:
        ab = best_ab
    inv_depth = (rel_disp.astype(np.float64) - ab[1]) / ab[0]
    depth = np.where(inv_depth > 1.0 / 30.0, 1.0 / np.maximum(inv_depth, 1e-4), np.inf)
    return depth.astype(np.float32)


def run_gt_and_pretrained_baselines(scenes_dir: Path, manifest: dict, device: torch.device) -> dict:
    bw = manifest["bw_speeds"]
    splits = manifest["splits"]
    all_scenes = manifest["sessions"]
    eval_scenes = [s["scene"] for s in all_scenes if s["split"] == "eval"]

    # 1. Exact walk-alert-r0 v2 parity check on GT-Single (all 287 & 110 eval) + GT-Memory on 110 eval
    eval_rows_287, gt_single_287 = {}, {}
    eval_rows_110, gt_single_110 = {}, {}
    gt_mem_raw_110, gt_mem_nodyn_110 = {}, {}

    print(f"[baselines] computing GT parity + GT memory references on {len(all_scenes)} scenes...", flush=True)
    for rec in all_scenes:
        sc = rec["scene"]
        d = load_scene(scenes_dir / f"{sc}.npz")
        n = len(d["eval_rows"])
        if n < 5:
            continue
        eval_rows_287[sc] = d["eval_rows"]
        gt_single_287[sc] = d["gt_single_d"]
        if rec["split"] == "eval":
            eval_rows_110[sc] = d["eval_rows"]
            gt_single_110[sc] = d["gt_single_d"]

            # Compute 6m explicit GT memory (raw & oracle-no-dynamic) on 110 eval
            K_o = d["K_orig"].astype(np.float64)
            Hh, Ww = int(d["orig_hw"][0]), int(d["orig_hw"][1])
            vv, uu = np.mgrid[0:Hh:S, 0:Ww:S]
            dirs = np.stack([(uu - K_o[2]) / K_o[0], (vv - K_o[3]) / K_o[1], np.ones(uu.shape)], -1).reshape(-1, 3)
            P_all, L_all = [], []
            ps = d["poses_t"].astype(np.float64)
            Rs = d["poses_R"].astype(np.float64)
            fwds = d["causal_fwd"].astype(np.float64)
            seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]
            cum = np.cumsum(seg_len)
            for i in range(n):
                df = d["depth_s8"][i].astype(np.float32).reshape(-1)
                sf = d["seg_s8"][i].reshape(-1)
                ok = np.isfinite(df) & (df > 0.1) & (df < 30.0)
                P_all.append(((Rs[i] @ (dirs[ok] * df[ok, None]).T).T + ps[i]).astype(np.float32))
                L_all.append(sf[ok])
            d_raw = np.full(n, np.inf, dtype=np.float32)
            d_nod = np.full(n, np.inf, dtype=np.float32)
            for i in range(n):
                if not d["valid_heading"][i]:
                    continue
                win = [j for j in range(i + 1) if cum[i] - cum[j] <= 6.0]
                fwd = fwds[i]
                right = np.array([fwd[1], -fwd[0], 0.0])
                for arr, no_dyn in ((d_raw, False), (d_nod, True)):
                    Q = np.concatenate(
                        [P_all[j] if (not no_dyn or j == i) else P_all[j][~np.isin(L_all[j], DYN)] for j in win]
                    )
                    rq = Q - ps[i]
                    xq, zq = rq @ right, rq @ fwd
                    strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
                    g = np.percentile(Q[strip, 2], 5) if strip.sum() >= 50 else ps[i][2] - 1.5
                    hq = Q[:, 2] - g
                    cm = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT_LO) & (hq <= HGT_HI)
                    arr[i] = corridor_d(zq[cm], MIN_PTS_S8)
            gt_mem_raw_110[sc] = d_raw
            gt_mem_nodyn_110[sc] = d_nod

    b1_thrs = np.round(np.arange(0.5, 5.01, 0.25), 2)
    results = {}
    c_287 = eval_score_curve(eval_rows_287, gt_single_287, bw, np.array([5.0]), higher_is_alert=False)
    print(f"[parity] GT-Single all-287 @ tau=5.0: {c_287[0]}", flush=True)
    results["parity_gt_single_287_tau5"] = c_287[0]

    for name, sc_dict in (
        ("gt_single_110", gt_single_110),
        ("gt_mem_raw_110", gt_mem_raw_110),
        ("gt_mem_nodyn_110", gt_mem_nodyn_110),
    ):
        curve = eval_score_curve(eval_rows_110, sc_dict, bw, b1_thrs, higher_is_alert=False)
        results[name] = {"curve": curve, "summary": summarize_at_fa_caps(curve)}
        s = results[name]["summary"]["best_floor"]
        print(
            f"[baseline] {name}: floor late={s['late']}/{s['avoidable']} ({s['late_rate']*100:.1f}%), "
            f"FA_ep={s['fa_ep_per_km']:.1f}/km, FA_fr={s['fa_fr_per_km']:.1f}/km",
            flush=True,
        )

    # 2. Pretrained Pure-Vision Monocular Depth Baselines on 110 eval sessions (on G4 GPU!)
    models_spec = [
        ("dav2_small_ground", "depth-anything/Depth-Anything-V2-Small-hf", "relative"),
        ("dav2_metric_outdoor_small", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "metric"),
        ("dav2_metric_outdoor_large", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "metric"),
    ]
    # Stride-8 equivalent spatial resolution on 294x518 is ~147x259 -> use 147x259 (~16x fewer pts than full 2208x1242, ~same density as S=8!)
    # Wait: S=8 on 1242x2208 is 156x276 -> exact same point density -> use min_pts = 21!
    eval_dh, eval_dw = 156, 276

    for bname, hf_id, kind in models_spec:
        t0 = time.time()
        print(f"[baseline] loading {bname} ({hf_id})...", flush=True)
        proc = AutoImageProcessor.from_pretrained(hf_id)
        mod = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device).eval()

        pred_d_110 = {}
        pred_mem_110 = {} if bname == "dav2_metric_outdoor_large" else None
        tot_ms, tot_fr = 0.0, 0

        for idx_s, sc in enumerate(eval_scenes, 1):
            d = load_scene(scenes_dir / f"{sc}.npz")
            n = len(d["eval_rows"])
            if n < 5:
                continue
            rgbs = decode_rgb_batch(d, np.arange(n))  # (n, 3, 294, 518) in [0,1]
            # Convert to list of uint8 HWC for AutoImageProcessor in chunks of 32
            depths = np.empty((n, eval_dh, eval_dw), dtype=np.float32)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for st in range(0, n, 32):
                    en = min(n, st + 32)
                    imgs = [(rgbs[k].transpose(1, 2, 0) * 255.0).astype(np.uint8) for k in range(st, en)]
                    inp = proc(images=imgs, return_tensors="pt").to(device)
                    torch.cuda.synchronize()
                    ts = time.perf_counter()
                    out = mod(**inp).predicted_depth  # (b, h, w)
                    torch.cuda.synchronize()
                    tot_ms += (time.perf_counter() - ts) * 1000.0
                    tot_fr += (en - st)
                    out_rs = F.interpolate(out.unsqueeze(1).float(), size=(eval_dh, eval_dw), mode="bilinear", align_corners=False).squeeze(1)
                    depths[st:en] = out_rs.cpu().numpy()

            K_o = d["K_orig"].astype(np.float64)
            Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
            K_d = np.array([K_o[0] * eval_dw / Ww, K_o[1] * eval_dh / Hh, K_o[2] * eval_dw / Ww, K_o[3] * eval_dh / Hh])
            Rs = d["poses_R"].astype(np.float64)
            ps = d["poses_t"].astype(np.float64)
            fwds = d["causal_fwd"].astype(np.float64)
            nom_h = 1.55 if str(d["cam"]) == "camera_head" else 1.20

            d_single = np.full(n, np.inf, dtype=np.float32)
            pts_seq = [] if pred_mem_110 is not None else None
            for i in range(n):
                if kind == "relative":
                    z_map = ground_anchor_relative_depth(depths[i], K_d, Rs[i], nom_h)
                else:
                    z_map = depths[i]
                dh, pw = depth_map_to_corridor_d(z_map, K_d, Rs[i], ps[i], fwds[i], MIN_PTS_S8)
                d_single[i] = dh
                if pts_seq is not None:
                    # Subsample 1/4 points for fast 6m explicit memory
                    pts_seq.append(pw[::4])
            pred_d_110[sc] = d_single

            if pred_mem_110 is not None:
                seg_len = np.r_[0.0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]
                cum = np.cumsum(seg_len)
                d_mem = np.full(n, np.inf, dtype=np.float32)
                for i in range(n):
                    if not d["valid_heading"][i]:
                        continue
                    win = [j for j in range(i + 1) if cum[i] - cum[j] <= 6.0]
                    Q = np.concatenate([pts_seq[j] for j in win]) if win else np.zeros((0, 3), dtype=np.float32)
                    if len(Q) == 0:
                        continue
                    fwd = fwds[i]
                    right = np.array([fwd[1], -fwd[0], 0.0])
                    rq = Q - ps[i]
                    xq, zq = rq @ right, rq @ fwd
                    strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
                    g = np.percentile(Q[strip, 2], 5) if strip.sum() >= 30 else ps[i][2] - 1.5
                    hq = Q[:, 2] - g
                    cm = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT_LO) & (hq <= HGT_HI)
                    d_mem[i] = corridor_d(zq[cm], max(5, MIN_PTS_S8 // 4))
                pred_mem_110[sc] = d_mem

            if idx_s % 30 == 0 or idx_s == len(eval_scenes):
                print(f"  [{bname}] {idx_s}/{len(eval_scenes)} scenes ({time.time()-t0:.0f}s)", flush=True)

        curve = eval_score_curve(eval_rows_110, pred_d_110, bw, b1_thrs, higher_is_alert=False)
        results[bname] = {
            "curve": curve,
            "summary": summarize_at_fa_caps(curve),
            "ms_per_frame_batch32": round(tot_ms / max(tot_fr, 1), 2),
        }
        s = results[bname]["summary"]
        bf = s["best_floor"]
        m22 = s["late_at_fa_ep<=22.6"]
        print(
            f"[baseline] {bname}: floor={bf['late_rate']*100:.1f}% (FA_ep={bf['fa_ep_per_km']:.1f}, FA_fr={bf['fa_fr_per_km']:.1f}), "
            f"@FA<=22.6: {m22['late_rate']*100:.1f}% (FA_ep={m22['fa_ep_per_km']:.1f})" if m22 else f"[baseline] {bname}: floor={bf['late_rate']*100:.1f}%",
            flush=True,
        )
        if pred_mem_110 is not None:
            cmem = eval_score_curve(eval_rows_110, pred_mem_110, bw, b1_thrs, higher_is_alert=False)
            results[bname + "_explicit_mem6m"] = {"curve": cmem, "summary": summarize_at_fa_caps(cmem)}
            sm = results[bname + "_explicit_mem6m"]["summary"]["best_floor"]
            print(
                f"[baseline] {bname}_explicit_mem6m: floor={sm['late_rate']*100:.1f}%, "
                f"FA_ep={sm['fa_ep_per_km']:.1f}/km, FA_fr={sm['fa_fr_per_km']:.1f}/km",
                flush=True,
            )

        del mod, proc
        torch.cuda.empty_cache()

    return results


class SanpoClipDataset(torch.utils.data.Dataset):
    """Preloaded in-memory SANPO-Synthetic train sessions -> random causal clips of length T."""

    def __init__(self, scene_dicts: list[dict], clip_len: int = 10, clips_per_epoch: int = 800, augment: bool = True):
        self.scenes = [d for d in scene_dicts if len(d["eval_rows"]) >= clip_len + 1]
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.augment = augment
        # Precompute z bin upper edges for cumulative corridor hazard supervision
        self.z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)

    def __len__(self) -> int:
        return self.clips_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        d = self.scenes[rng.integers(len(self.scenes))]
        n = len(d["eval_rows"])
        # Bias 60% of clips to include at least one hazard frame (D <= 4.0 m) so positives are rich!
        D_all = d["eval_rows"][:, 0]
        haz_idx = np.nonzero(np.isfinite(D_all) & (D_all <= H))[0]
        if len(haz_idx) > 0 and rng.random() < 0.60:
            anchor = int(haz_idx[rng.integers(len(haz_idx))])
            st = int(np.clip(anchor - rng.integers(0, self.clip_len), 0, n - self.clip_len))
        else:
            st = int(rng.integers(0, n - self.clip_len + 1))
        sl = np.arange(st, st + self.clip_len)

        rgb = decode_rgb_batch(d, sl)  # (T, 3, 294, 518) in [0, 1]
        if self.augment:
            # Photometric jitter (brightness/contrast) preserving exact camera geometry
            b_fac = float(rng.uniform(0.85, 1.15))
            c_fac = float(rng.uniform(0.85, 1.15))
            mean = rgb.mean(axis=(2, 3), keepdims=True)
            rgb = np.clip((rgb - mean) * c_fac + mean * b_fac, 0.0, 1.0)
        rgb = (rgb - IMAGENET_MEAN[None]) / IMAGENET_STD[None]

        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
        K_seq = np.repeat(K_s[None], self.clip_len, axis=0)

        R_seq = d["poses_R"][sl].astype(np.float32)
        t_seq = d["poses_t"][sl].astype(np.float32)
        fwd_seq = d["causal_fwd"][sl].astype(np.float32)
        seg_seq = np.nan_to_num(d["eval_rows"][sl, 3], nan=0.0).astype(np.float32)
        # Recompute seg_len directly from poses_t so standing/invalid frames still have exact step
        seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

        gt_stat = d["gt_bev_static"][sl].astype(np.float32)   # (T, 3, 24, 15)
        gt_dyn = d["gt_bev_dynamic"][sl].astype(np.float32)   # (T, 3, 24, 15)
        vh = d["valid_heading"][sl].astype(np.float32)
        ev_r = d["eval_rows"][sl]
        D_gt = ev_r[:, 0]
        cover = ev_r[:, 1]
        dyn_fl = ev_r[:, 2]

        # Cumulative corridor hazard label y_corr (T, 24): 1 if static D_gt <= z_edge[k]
        y_corr = (D_gt[:, None] <= self.z_edges[None, :]).astype(np.float32)
        # Mask out invalid/low-coverage frames and dynamic-only clear frames from corridor loss
        valid_corr = (
            (vh > 0.5)
            & np.isfinite(cover)
            & (cover >= 0.6)
            & ~((np.isinf(D_gt)) & (dyn_fl > 0.5))
        ).astype(np.float32)

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
            "valid_h": torch.from_numpy(vh),
        }


def compute_egoclear_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    st_log = out["static_logits"]    # (B, T, 3, 24, 15)
    dy_log = out["dynamic_logits"]   # (B, T, 3, 24, 15)
    co_log = out["corr_logits"]      # (B, T, 24)
    un_log = out["unc_logits"]       # (B, T, 24)

    gt_st = batch["gt_static"]
    gt_dy = batch["gt_dynamic"]
    y_co = batch["y_corr"]
    v_co = batch["valid_corr"]       # (B, T)

    # Spatial decision-aligned column weights: central corridor x in [6,7,8] weighted 3.0x, side buffer [5,9] 1.5x
    col_w = torch.ones(BEV_NX, device=st_log.device, dtype=st_log.dtype)
    col_w[6:9] = 3.0
    col_w[5] = 1.5
    col_w[9] = 1.5
    col_w = col_w[None, None, None, None, :]

    # Focal-weighted BCE for static occupancy (pedestrians are 0 here!)
    bce_st = F.binary_cross_entropy_with_logits(st_log, gt_st, pos_weight=torch.tensor(4.0, device=st_log.device), reduction="none")
    pt_st = torch.exp(-F.binary_cross_entropy_with_logits(st_log, gt_st, reduction="none"))
    loss_st = (((1.0 - pt_st) ** 1.5) * bce_st * col_w).mean()

    # Dynamic current-frame occupancy auxiliary loss
    bce_dy = F.binary_cross_entropy_with_logits(dy_log, gt_dy, pos_weight=torch.tensor(4.0, device=dy_log.device), reduction="none")
    loss_dy = bce_dy.mean()

    # Direct decision-aligned corridor hazard loss (asymmetric pos_weight=3.0 inside 4.0m)
    z_edges = torch.tensor([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], device=co_log.device)
    z_w = torch.where(z_edges <= H + 0.2, 2.0, 0.5)[None, None, :]
    bce_co = F.binary_cross_entropy_with_logits(co_log, y_co, pos_weight=torch.tensor(3.0, device=co_log.device), reduction="none")
    pt_co = torch.exp(-F.binary_cross_entropy_with_logits(co_log, y_co, reduction="none"))
    m_co = v_co[:, :, None] * z_w
    loss_co = ((((1.0 - pt_co) ** 1.5) * bce_co) * m_co).sum() / m_co.sum().clamp(min=1.0)

    # Calibrated uncertainty auxiliary head: predicts whether |p_corr - y_corr| > 0.35
    with torch.no_grad():
        err_tgt = ((torch.sigmoid(co_log) - y_co).abs() > 0.35).float()
    bce_un = F.binary_cross_entropy_with_logits(un_log, err_tgt, reduction="none")
    loss_un = (bce_un * m_co).sum() / m_co.sum().clamp(min=1.0)

    total = loss_st + 0.5 * loss_dy + 1.5 * loss_co + 0.2 * loss_un
    return total, {
        "tot": float(total.item()),
        "st": float(loss_st.item()),
        "dy": float(loss_dy.item()),
        "co": float(loss_co.item()),
        "un": float(loss_un.item()),
    }


@torch.inference_mode()
def eval_egoclear_on_sessions(
    model: EgoClearNet,
    eval_scene_dicts: dict[str, dict],
    bw_speeds: dict,
    device: torch.device,
) -> dict:
    model.eval()
    eval_rows_map = {}
    corr_probs_map = {}   # scene -> (T, 24) float32
    stat_ap_num = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}  # tp, fp, fn @ 0.5
    dyn_fp_frames, dyn_tot_frames = 0, 0
    tot_ms, tot_fr = 0.0, 0

    z_edges = np.array([BEV_Z_MIN + (k + 1) * BEV_DZ for k in range(BEV_NZ)], dtype=np.float32)
    idx_4m = int(np.searchsorted(z_edges, 4.0))  # bin covering up to ~4.1 m

    for sc, d in eval_scene_dicts.items():
        n = len(d["eval_rows"])
        if n < 5:
            continue
        rgb = decode_rgb_batch(d, np.arange(n))
        rgb = (rgb - IMAGENET_MEAN[None]) / IMAGENET_STD[None]
        K_o = d["K_orig"].astype(np.float32)
        Hh, Ww = float(d["orig_hw"][0]), float(d["orig_hw"][1])
        K_s = np.array([K_o[0] * RGB_W / Ww, K_o[1] * RGB_H / Hh, K_o[2] * RGB_W / Ww, K_o[3] * RGB_H / Hh], dtype=np.float32)
        t_seq = d["poses_t"].astype(np.float32)
        seg_seq = np.r_[0.0, np.linalg.norm(np.diff(t_seq[:, :2], axis=0), axis=1)].astype(np.float32)

        batch = {
            "rgb": torch.from_numpy(rgb)[None].to(device),
            "K_scaled": torch.from_numpy(np.repeat(K_s[None], n, axis=0))[None].to(device),
            "R_c2w": torch.from_numpy(d["poses_R"].astype(np.float32))[None].to(device),
            "t_w": torch.from_numpy(t_seq)[None].to(device),
            "fwd_w": torch.from_numpy(d["causal_fwd"].astype(np.float32))[None].to(device),
            "seg_len": torch.from_numpy(seg_seq)[None].to(device),
        }

        torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_sequence(**batch)
        torch.cuda.synchronize()
        tot_ms += (time.perf_counter() - ts) * 1000.0
        tot_fr += n

        p_corr = torch.sigmoid(out["corr_logits"][0].float()).cpu().numpy()   # (n, 24)
        p_stat = torch.sigmoid(out["static_logits"][0].float()).cpu().numpy() # (n, 3, 24, 15)
        eval_rows_map[sc] = d["eval_rows"]
        corr_probs_map[sc] = p_corr

        # Per-height-band F1 on central corridor (cols 6..8)
        gt_st = d["gt_bev_static"][:, :, :, 6:9] > 0
        pr_st = p_stat[:, :, :, 6:9] >= 0.5
        for b in range(3):
            stat_ap_num[b][0] += int((pr_st[:, b] & gt_st[:, b]).sum())
            stat_ap_num[b][1] += int((pr_st[:, b] & ~gt_st[:, b]).sum())
            stat_ap_num[b][2] += int((~pr_st[:, b] & gt_st[:, b]).sum())

        # Dynamic-only clear frames: does model falsely alert on passing pedestrians?
        D_gt = d["eval_rows"][:, 0]
        dyn_only = (d["valid_heading"] > 0) & np.isinf(D_gt) & (d["eval_rows"][:, 2] > 0.5)
        if np.any(dyn_only):
            dyn_tot_frames += int(dyn_only.sum())
            dyn_fp_frames += int((p_corr[dyn_only, idx_4m] >= 0.5).sum())

    # 1) Fixed-horizon readout (max prob inside <= 4.1 m, apples-to-apples with B1!)
    fixed_scores = {sc: probs[:, : idx_4m + 1].max(axis=1) for sc, probs in corr_probs_map.items()}
    thrs = np.round(np.concatenate([np.arange(0.02, 0.20, 0.02), np.arange(0.20, 0.96, 0.04)]), 3)
    curve_fixed = eval_score_curve(eval_rows_map, fixed_scores, bw_speeds, thrs, higher_is_alert=True)

    # 2) Walker-speed-aware readout: alert when P(obstacle <= dreq + 1.0 m) >= thr
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


def train_and_eval_variant(
    mode: str,
    train_dicts: list[dict],
    eval_dicts: dict[str, dict],
    bw_speeds: dict,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 8,
    clip_len: int = 10,
) -> dict:
    t0 = time.time()
    torch.manual_seed(20261120)
    np.random.seed(20261120)

    model = EgoClearNet(mode=mode, backbone_name="facebook/dinov2-small", unfreeze_last_k=4).to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Training EgoClear-{mode} (params={n_tot/1e6:.2f}M, trainable={n_tr/1e6:.2f}M) ===", flush=True)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 3e-5, "weight_decay": 1e-4},
            {"params": head_params, "lr": 4e-4, "weight_decay": 1e-4},
        ]
    )
    ds = SanpoClipDataset(train_dicts, clip_len=clip_len, clips_per_epoch=720, augment=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = {"tot": 0.0, "st": 0.0, "dy": 0.0, "co": 0.0, "un": 0.0}
        t_ep = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_sequence(
                    batch["rgb"], batch["K_scaled"], batch["R_c2w"], batch["t_w"], batch["fwd_w"], batch["seg_len"]
                )
                loss, parts = compute_egoclear_loss(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            for k in ep_loss:
                ep_loss[k] += parts[k]
        for k in ep_loss:
            ep_loss[k] /= len(loader)
        history.append({"epoch": ep, **{k: round(v, 4) for k, v in ep_loss.items()}, "sec": round(time.time() - t_ep, 1)})
        if ep in (1, 4, 8, epochs):
            print(
                f"  [EgoClear-{mode} ep {ep:02d}/{epochs}] tot={ep_loss['tot']:.4f} "
                f"(st={ep_loss['st']:.4f} dy={ep_loss['dy']:.4f} co={ep_loss['co']:.4f}) "
                f"{time.time()-t_ep:.1f}s",
                flush=True,
            )

    eval_res = eval_egoclear_on_sessions(model, eval_dicts, bw_speeds, device)
    ckpt_local = LOCAL_RUNS / f"egoclear_{mode}.pt"
    torch.save({"mode": mode, "state_dict": model.state_dict(), "history": history}, ckpt_local)
    shutil.copy2(ckpt_local, DRIVE_RUNS / ckpt_local.name)

    for key in ("fixed_4m", "walker_aware"):
        s = eval_res[key]["summary"]
        bf = s["best_floor"]
        m22 = s["late_at_fa_ep<=22.6"]
        m120 = s["late_at_fa_fr<=120.3"]
        m22_s = f"{m22['late_rate']*100:.1f}%" if m22 else "N/A"
        m120_s = f"{m120['late_rate']*100:.1f}%" if m120 else "N/A"
        print(
            f"[RESULT EgoClear-{mode} | {key}] floor={bf['late_rate']*100:.1f}% (FA_ep={bf['fa_ep_per_km']:.1f}, FA_fr={bf['fa_fr_per_km']:.1f}) | "
            f"@FA_ep<=22.6: {m22_s} | "
            f"@FA_fr<=120.3: {m120_s} | "
            f"dyn_alert@0.5={eval_res['dyn_only_alert_rate_at_0.5']['rate']*100:.1f}% | "
            f"{eval_res['ms_per_frame_stream']} ms/fr",
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
        "eval": eval_res,
    }


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    manifest = stage_extract_dataset()
    scenes_dir = LOCAL_DATA / "scenes"

    # Stage 2 & 3: GT parity + Pretrained Pure-Vision Baselines
    baselines = run_gt_and_pretrained_baselines(scenes_dir, manifest, device)

    # Preload 177 train & 110 eval scenes into RAM for fast G4 training
    train_dicts = []
    eval_dicts = {}
    for rec in manifest["sessions"]:
        d = load_scene(scenes_dir / f"{rec['scene']}.npz")
        if rec["split"] == "train":
            train_dicts.append(d)
        else:
            eval_dicts[rec["scene"]] = d
    print(f"[data] preloaded {len(train_dicts)} train & {len(eval_dicts)} eval scenes into RAM", flush=True)

    # Stage 4: Train & evaluate 3 progressive EgoClear-Net variants
    egoclear_results = {}
    for mode in ("single", "mvs", "full"):
        egoclear_results[mode] = train_and_eval_variant(
            mode, train_dicts, eval_dicts, manifest["bw_speeds"], device, epochs=12, batch_size=12, clip_len=10
        )

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "total_wall_sec": round(time.time() - t_all, 1),
        "train_sessions": len(train_dicts),
        "eval_sessions": len(eval_dicts),
        "baselines": baselines,
        "egoclear": egoclear_results,
    }
    res_local = LOCAL_RUNS / "phase1_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase1_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[done] saved phase1_results.json ({res_local.stat().st_size} B) in {time.time()-t_all:.1f}s", flush=True)

    try:
        from google.colab import drive
        drive.flush_and_unmount()
        print("[done] drive.flush_and_unmount() complete", flush=True)
    except Exception as e:
        print(f"[done] flush note: {e}", flush=True)


if __name__ == "__main__":
    main()
