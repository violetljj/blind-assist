"""Strict Zero-Shot Cross-Glasses & Low-Obstacle Benchmark (Local RTX 5060, 0 CU cost).

Evaluates EgoGeom-Omni v2.1 (12.9M From-Scratch, 12.02 ms S24U Hexagon v75 NPU) against
pretrained Depth-Anything-V2-Small (ViT-S, 24.8M) on THREE strictly unseen public datasets:
  1. Meta Aria Gen 2 (19 real grayscale wide-FOV glasses clips, 1,794 valid / 840 haz<=4m)
  2. UT Austin CODa (24 real campus clips, 2,819 valid / 34 haz<=4m — outdoor low-FA stress test!)
  3. ARKitScenes val30 Day & Dark 0.18x (30 real indoor clips, 467 valid — indoor + dark robustness!)

Also benchmarks a zero-training Stage-2 Prototype: 1-ms Causal Pose-Warped 2.5D BEV Memory
on top of EgoGeom-Omni v2.1's (depth_m, delta_h_m, log_sigma) outputs!
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# Add repo paths cleanly
REPO = Path(r"e:\linnan\linnan")
sys.path.insert(0, str(REPO / "research/active/egogeom-omni-v0"))
sys.path.insert(0, str(REPO / "research/active/egoclear-v0"))

from egogeom_losses_and_physics import compute_gravity_aligned_ray_map
from egogeom_omni_model import EgoGeomConfig, EgoGeomOmniNet
from eval_closed_loop import make_episodes, score_episode

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
H_IN, W_IN = 288, 384
X_HALF, FWD0, H_MAX = 0.30, 0.30, 4.0
HGT_LO, HGT_HI = 0.15, 1.90
MIN_PTS = 18

# Synthetic BlindWays speed distribution proxy (mean ~1.05 m/s, range 0.75-1.35 m/s) for closed-loop parity
BW_SPEEDS_PROXY = {f"p{i}": np.linspace(0.78 + 0.04 * i, 1.15 + 0.04 * i, 10).tolist() for i in range(6)}


def decode_isotropic_frame(jpg_bytes: bytes, orig_hw: np.ndarray, K_orig: np.ndarray, dark_scale: float = 1.0):
    """Undo pack stretching back to true isotropic orig_hw aspect, then aspect-safe crop/resize to 288x384 (fx == fy!)."""
    bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    target_ar = W_IN / float(H_IN)
    orig_ar = ow / float(oh)

    # First map to an isotropic intermediate matching orig_ar
    if orig_ar >= target_ar:
        ih = H_IN
        iw = max(W_IN, int(round(H_IN * orig_ar)))
    else:
        iw = W_IN
        ih = max(H_IN, int(round(W_IN / orig_ar)))
    rgb_iso = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA if iw < rgb.shape[1] else cv2.INTER_LINEAR)
    sx, sy = iw / float(ow), ih / float(oh)
    fx, fy, cx, cy = float(K_orig[0]) * sx, float(K_orig[1]) * sy, float(K_orig[2]) * sx, float(K_orig[3]) * sy

    # Center crop to (H_IN, W_IN) preserving fx == fy!
    y0 = max(0, (ih - H_IN) // 2)
    x0 = max(0, (iw - W_IN) // 2)
    rgb_crop = rgb_iso[y0 : y0 + H_IN, x0 : x0 + W_IN]
    if rgb_crop.shape[:2] != (H_IN, W_IN):
        rgb_crop = cv2.resize(rgb_crop, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
    K_out = np.array([[fx, 0.0, cx - x0], [0.0, fy, cy - y0], [0.0, 0.0, 1.0]], dtype=np.float32)

    rgb_f = rgb_crop.astype(np.float32) / 255.0
    if dark_scale < 0.99:
        rgb_f = np.clip(rgb_f * dark_scale, 0.0, 1.0)
    return rgb_f, K_out


def corridor_dist_from_depth(
    depth: np.ndarray,
    delta_h: np.ndarray | None,
    log_sigma: np.ndarray | None,
    K: np.ndarray,
    R_wc: np.ndarray,
    fwd_w: np.ndarray,
    use_pred_dh: bool = True,
) -> tuple[float, np.ndarray]:
    """Project predicted depth (+ optional delta_h) into world-aligned walking corridor -> (D_pred, obs_pts_rel_w)."""
    # Subsample 4x -> 72x96 (6,912 pts, super fast!)
    d = depth[::4, ::4]
    H4, W4 = d.shape
    v, u = np.mgrid[0:H4, 0:W4].astype(np.float32)
    u = u * 4.0 + 2.0
    v = v * 4.0 + 2.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = d.reshape(-1)
    ok = np.isfinite(z) & (z > 0.25) & (z < 10.0)
    if ok.sum() < 50:
        return float(np.inf), np.zeros((0, 4), dtype=np.float32)

    x_c = (u.reshape(-1)[ok] - cx) * z[ok] / max(fx, 1.0)
    y_c = (v.reshape(-1)[ok] - cy) * z[ok] / max(fy, 1.0)
    P_cam = np.stack([x_c, y_c, z[ok]], axis=-1)
    P_rel = (R_wc @ P_cam.T).T  # relative to camera in world (+Z up)

    fw = np.array([fwd_w[0], fwd_w[1], 0.0], dtype=np.float32)
    nf = float(np.linalg.norm(fw))
    if nf < 1e-4:
        cz = R_wc[:, 2].copy()
        cz[2] = 0.0
        fw = cz / max(float(np.linalg.norm(cz)), 1e-6)
    else:
        fw /= nf
    rt = np.array([fw[1], -fw[0], 0.0], dtype=np.float32)

    x_lat = P_rel @ rt
    z_fwd = P_rel @ fw

    if use_pred_dh and delta_h is not None:
        h_val = delta_h[::4, ::4].reshape(-1)[ok]
    else:
        # Robust local ground Z from lower 10th percentile in front strip
        strip = (np.abs(x_lat) <= 1.2) & (z_fwd >= 0.8) & (z_fwd <= 4.5)
        g0 = float(np.percentile(P_rel[strip, 2], 8)) if strip.sum() >= 40 else -1.50
        h_val = P_rel[:, 2] - g0

    conf = np.exp(-np.clip(log_sigma[::4, ::4].reshape(-1)[ok], -3.0, 2.0)) if log_sigma is not None else np.ones_like(h_val)
    obs_m = (h_val >= HGT_LO) & (h_val <= HGT_HI) & (z_fwd >= FWD0) & (z_fwd <= 5.0)
    obs_pts = np.concatenate([P_rel[obs_m], conf[obs_m, None]], axis=-1) if obs_m.any() else np.zeros((0, 4), dtype=np.float32)

    cm = obs_m & (np.abs(x_lat) <= X_HALF) & (z_fwd <= H_MAX)
    if cm.sum() < MIN_PTS:
        return float(np.inf), obs_pts
    zs = np.sort(z_fwd[cm])
    cnt = np.searchsorted(zs, zs + 0.30, side="right") - np.arange(len(zs))
    hit = np.flatnonzero(cnt >= MIN_PTS)
    return (float(zs[hit[0]]) if len(hit) else float(np.inf)), obs_pts


def temporal_bev_corridor_distances(
    poses_t: np.ndarray,
    causal_fwd: np.ndarray,
    poses_R: np.ndarray,
    obs_pts_list: list[np.ndarray],
    window: int = 6,
    decay: float = 0.72,
    min_score: float = 14.0,
) -> np.ndarray:
    """Stage-2 Prototype (<1 ms/frame): Causal Pose-Warped World 2.5D BEV Memory!

    Accumulates world-anchored obstacle evidence across the last `window` frames (~1.2 s @ 5 Hz)
    with exponential temporal decay + uncertainty weighting, retaining close shallow obstacles
    even after they dip below the camera bottom FOV while filtering single-frame jitter!
    """
    T = len(obs_pts_list)
    out_d = np.full(T, np.inf, dtype=np.float32)
    world_buf: list[tuple[np.ndarray, np.ndarray]] = []  # (pts_world [N,3], weights [N])

    for i in range(T):
        pts = obs_pts_list[i]
        if len(pts) > 0:
            pw = pts[:, :3] + poses_t[i][None, :]
            wt = np.clip(pts[:, 3], 0.25, 2.5)
            world_buf.append((pw, wt))
        else:
            world_buf.append((np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)))
        if len(world_buf) > window:
            world_buf.pop(0)

        fw = np.array([causal_fwd[i, 0], causal_fwd[i, 1], 0.0], dtype=np.float32)
        nf = float(np.linalg.norm(fw))
        if nf < 1e-4:
            cz = poses_R[i, :, 2].copy()
            cz[2] = 0.0
            fw = cz / max(float(np.linalg.norm(cz)), 1e-6)
        else:
            fw /= nf
        rt = np.array([fw[1], -fw[0], 0.0], dtype=np.float32)

        # Accumulate corridor z-histogram (0.20 m bins from 0.30..4.10 m)
        bins = np.zeros(19, dtype=np.float32)
        k_len = len(world_buf)
        for age_idx, (pw, wt) in enumerate(world_buf):
            if len(pw) == 0:
                continue
            age = (k_len - 1) - age_idx
            w_t = decay ** age
            rel = pw - poses_t[i][None, :]
            xl = rel @ rt
            zf = rel @ fw
            m = (np.abs(xl) <= X_HALF) & (zf >= FWD0) & (zf <= H_MAX)
            if not m.any():
                continue
            b_idx = np.clip(((zf[m] - FWD0) / 0.20).astype(np.int32), 0, 18)
            np.add.at(bins, b_idx, wt[m] * w_t)

        # 2-bin sliding sum (0.40 m corridor slice)
        s2 = bins[:-1] + bins[1:]
        hits = np.flatnonzero(s2 >= min_score)
        if len(hits):
            out_d[i] = FWD0 + hits[0] * 0.20
    return out_d


def compute_metrics(clips_data: list[dict], pred_key: str, hazard_thr_m: float = 4.0) -> dict:
    """Compute per-frame AUROC, Recall@5%FPR, Dist MAE, and Closed-Loop Blind-Walker Late@FA<=5/km & 10/km."""
    y_true, y_score, d_errs = [], [], []
    all_eps = []
    for c in clips_data:
        er = c["eval_rows"]
        D_gt = er[:, 0]
        val = (np.isfinite(D_gt) | np.isinf(D_gt)) & ~np.isnan(er[:, 1]) & (er[:, 1] >= 0.6)
        D_pr = c[pred_key]
        # Higher score = closer predicted obstacle (1 / clamp(D_pr, 0.2, 10.0))
        score = 1.0 / np.clip(np.where(np.isfinite(D_pr), D_pr, 12.0), 0.20, 12.0)

        m = val
        if m.any():
            y_true.append((D_gt[m] <= hazard_thr_m).astype(np.int32))
            y_score.append(score[m])
            mh = m & np.isfinite(D_gt) & (D_gt >= 0.30) & (D_gt <= 4.0) & np.isfinite(D_pr) & (D_pr <= 6.0)
            if mh.any():
                d_errs.append(np.abs(D_pr[mh] - D_gt[mh]))

        eps = make_episodes(c["name"], er, BW_SPEEDS_PROXY, kind="blind")
        for ep in eps:
            ep["_score"] = score
            ep["_D_pr"] = D_pr
            all_eps.append(ep)

    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    out = {
        "n_valid": int(len(yt)),
        "n_pos": int(yt.sum()),
        "dist_mae_m": round(float(np.concatenate(d_errs).mean()), 3) if d_errs else -1.0,
    }
    if 0 < yt.sum() < len(yt):
        out["auroc"] = round(float(roc_auc_score(yt, ys)), 4)
        out["ap"] = round(float(average_precision_score(yt, ys)), 4)
        # Recall at 5% and 10% FPR
        neg_scores = ys[yt == 0]
        pos_scores = ys[yt == 1]
        for fpr_t in (0.05, 0.10):
            thr = float(np.quantile(neg_scores, 1.0 - fpr_t))
            out[f"rec_fpr{int(fpr_t*100)}"] = round(float((pos_scores >= thr).mean()), 4)
    else:
        out["auroc"] = -1.0

    # Closed-loop Blind-Walker sweep over alert distance threshold (1.4..4.5 m)
    best_fa5, best_fa10 = 100.0, 100.0
    for d_thr in np.linspace(1.4, 4.4, 31):
        late = avoid = fa_ep = 0
        km = 0.0
        for ep in all_eps:
            al = ep["_D_pr"] <= d_thr
            l, a, _, fe, _, k = score_episode(ep, al)
            late += l
            avoid += a
            fa_ep += fe
            km += k
        if avoid > 0 and km > 1e-4:
            lr = 100.0 * late / avoid
            fa_km = fa_ep / km
            if fa_km <= 5.0 and lr < best_fa5:
                best_fa5 = lr
            if fa_km <= 10.0 and lr < best_fa10:
                best_fa10 = lr
    out["cl_late_fa5_pct"] = round(best_fa5, 1) if best_fa5 < 100.0 else -1.0
    out["cl_late_fa10_pct"] = round(best_fa10, 1) if best_fa10 < 100.0 else -1.0
    return out


@torch.no_grad()
def run_egogeom_on_dataset(model: EgoGeomOmniNet, npz_paths: list[Path], device: torch.device, dark_scale: float = 1.0):
    clips = []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as d:
            offs = d["rgb_offsets"]
            blob = d["rgb_blob"]
            orig_hw = d["orig_hw"]
            K_orig = d["K_orig"]
            poses_R = d["poses_R"]
            poses_t = d["poses_t"]
            causal_fwd = d["causal_fwd"]
            eval_rows = d["eval_rows"]
            T = len(eval_rows)

            rgbs, Ks, gs = [], [], []
            for i in range(T):
                jpg = blob[offs[i] : offs[i + 1]].tobytes()
                rgb_f, K = decode_isotropic_frame(jpg, orig_hw, K_orig, dark_scale=dark_scale)
                rgb_n = (rgb_f - IM_MEAN) / IM_STD
                rgbs.append(rgb_n.transpose(2, 0, 1))
                Ks.append(K)
                g_cam = (poses_R[i].T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
                g_cam /= np.linalg.norm(g_cam) + 1e-6
                gs.append(g_cam)

            # Batched GPU inference (bs=32 -> blazing fast on RTX 5060!)
            d_single_dh = np.full(T, np.inf, dtype=np.float32)
            d_single_wz = np.full(T, np.inf, dtype=np.float32)
            obs_dh_list, obs_wz_list = [], []
            for i0 in range(0, T, 32):
                i1 = min(T, i0 + 32)
                x = torch.from_numpy(np.stack(rgbs[i0:i1])).to(device)
                Kt = torch.from_numpy(np.stack(Ks[i0:i1])).to(device)
                gt = torch.from_numpy(np.stack(gs[i0:i1])).to(device)
                rays = compute_gravity_aligned_ray_map(Kt, gt, H_IN, W_IN)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, rays, None, None, return_features=False)
                dep = out["depth_m"][:, 0].float().cpu().numpy()
                dh = out["delta_h_m"][:, 0].float().cpu().numpy()
                ls = out["log_sigma"][:, 0].float().cpu().numpy()
                for j in range(i1 - i0):
                    idx = i0 + j
                    dd, op = corridor_dist_from_depth(dep[j], dh[j], ls[j], Ks[idx], poses_R[idx], causal_fwd[idx], use_pred_dh=True)
                    d_single_dh[idx] = dd
                    obs_dh_list.append(op)
                    dw, ow = corridor_dist_from_depth(dep[j], None, ls[j], Ks[idx], poses_R[idx], causal_fwd[idx], use_pred_dh=False)
                    d_single_wz[idx] = dw
                    obs_wz_list.append(ow)

            d_bev_dh = temporal_bev_corridor_distances(poses_t, causal_fwd, poses_R, obs_dh_list)
            d_bev_wz = temporal_bev_corridor_distances(poses_t, causal_fwd, poses_R, obs_wz_list)
            clips.append({
                "name": p.stem,
                "eval_rows": eval_rows,
                "ego_1f_dh": d_single_dh,
                "ego_1f_wz": d_single_wz,
                "ego_bev_dh": d_bev_dh,
                "ego_bev_wz": d_bev_wz,
            })
    return clips


@torch.no_grad()
def run_dav2_metric_on_dataset(model_id: str, npz_paths: list[Path], device: torch.device, dark_scale: float = 1.0):
    from transformers import AutoModelForDepthEstimation
    m = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(device)
    clips = []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as d:
            offs, blob = d["rgb_offsets"], d["rgb_blob"]
            orig_hw, K_orig = d["orig_hw"], d["K_orig"]
            poses_R, poses_t, causal_fwd, eval_rows = d["poses_R"], d["poses_t"], d["causal_fwd"], d["eval_rows"]
            T = len(eval_rows)
            rgbs, Ks = [], []
            for i in range(T):
                rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=dark_scale)
                # Resize to 294x392 (divisible by 14 for ViT-S)
                rgb_v = cv2.resize(rgb_f, (392, 294), interpolation=cv2.INTER_LINEAR)
                rgbs.append(((rgb_v - IM_MEAN) / IM_STD).transpose(2, 0, 1))
                Ks.append(K)

            d_dav2 = np.full(T, np.inf, dtype=np.float32)
            for i0 in range(0, T, 24):
                i1 = min(T, i0 + 24)
                x = torch.from_numpy(np.stack(rgbs[i0:i1])).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = m(pixel_values=x).predicted_depth
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(H_IN, W_IN), mode="bilinear", align_corners=False)[:, 0].float().cpu().numpy()
                for j in range(i1 - i0):
                    idx = i0 + j
                    dw, _ = corridor_dist_from_depth(pred[j], None, None, Ks[idx], poses_R[idx], causal_fwd[idx], use_pred_dh=False)
                    d_dav2[idx] = dw
            clips.append({"name": p.stem, "eval_rows": eval_rows, "dav2_1f_wz": d_dav2})
    del m
    torch.cuda.empty_cache()
    return clips


def inspect_real_h3_cnh_reference() -> dict:
    """Non-rigorous diagnostic reference on self-recorded Atom+H3 captures (characterizes real CNH waveform domain gap)."""
    cap_dir = REPO / "artifacts.local/hardware-bringup/captures/live-small-1790532816778"
    if not cap_dir.exists():
        return {"status": "not_found"}
    bins = sorted(cap_dir.rglob("*.bin"))
    jsonls = sorted(cap_dir.rglob("*.jsonl"))
    # Parse CNH stats from jsonl if present
    bin0_ratios, peak_bins, n_frames = [], [], 0
    for jf in jsonls:
        if "cnh" not in jf.name.lower() and "tof" not in jf.name.lower():
            continue
        for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines()[:500]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            hist = obj.get("cnh") or obj.get("histograms") or obj.get("bins")
            if isinstance(hist, list) and len(hist) >= 16:
                arr = np.asarray(hist, dtype=np.float32).reshape(-1, 16)
                tot = np.maximum(arr.sum(axis=-1), 1.0)
                bin0_ratios.append(float(np.median((arr[:, 0] + arr[:, 1]) / tot)))
                peak_bins.append(int(np.argmax(arr.mean(axis=0))))
                n_frames += 1
    return {
        "capture_dir": cap_dir.name,
        "n_bin_files": len(bins),
        "n_jsonl_files": len(jsonls),
        "n_parsed_cnh_frames": n_frames,
        "median_bin01_crosstalk_frac": round(float(np.median(bin0_ratios)), 4) if bin0_ratios else -1.0,
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = REPO / "artifacts.local/work/egogeom-omni-v0"
    zs = work / "zeroshot_eval"

    cfg = EgoGeomConfig.base()
    model = EgoGeomOmniNet(cfg, include_teacher_heads=False).to(device).eval()
    ck = torch.load(work / "egogeom_omni_v21_unified_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["ema"], strict=False)
    print(f"Loaded EgoGeom-Omni v2.1 Unified Best on {torch.cuda.get_device_name(0)}", flush=True)

    suites = [
        ("aria_gen2", sorted((zs / "aria_g2_20clips").rglob("*.npz")), 1.0, "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", 4.0),
        ("coda_campus", sorted((zs / "coda_zeroshot_shard000").rglob("*.npz")), 1.0, "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", 4.0),
        ("arkit_val30_day", sorted((zs / "arkit_val30_shard00").rglob("*.npz")), 1.0, "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", 2.0),
        ("arkit_val30_dark018", sorted((zs / "arkit_val30_shard00").rglob("*.npz")), 0.18, "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", 2.0),
    ]

    summary = {}
    for sname, npzs, dscale, dav2_id, h_thr in suites:
        ts = time.time()
        print(f"\n=== Zero-Shot Suite: {sname} ({len(npzs)} clips, dark={dscale}, pos_thr={h_thr}m) ===", flush=True)
        ego_clips = run_egogeom_on_dataset(model, npzs, device, dark_scale=dscale)
        dav2_clips = run_dav2_metric_on_dataset(dav2_id, npzs, device, dark_scale=dscale)
        res = {
            "egogeom_v21_1f_pred_dh": compute_metrics(ego_clips, "ego_1f_dh", hazard_thr_m=h_thr),
            "egogeom_v21_1f_world_z": compute_metrics(ego_clips, "ego_1f_wz", hazard_thr_m=h_thr),
            "egogeom_v21_stage2_bev_dh": compute_metrics(ego_clips, "ego_bev_dh", hazard_thr_m=h_thr),
            "egogeom_v21_stage2_bev_wz": compute_metrics(ego_clips, "ego_bev_wz", hazard_thr_m=h_thr),
            "dav2_metric_small_1f": compute_metrics(dav2_clips, "dav2_1f_wz", hazard_thr_m=h_thr),
        }
        summary[sname] = res
        for k, v in res.items():
            print(f"  {k:28s}: {v}", flush=True)
        print(f"  (elapsed {time.time()-ts:.1f}s)", flush=True)

    summary["non_rigorous_h3_reference"] = inspect_real_h3_cnh_reference()
    print("\n=== Non-Rigorous H3 Capture Reference ===", flush=True)
    print(f"  {summary['non_rigorous_h3_reference']}", flush=True)

    out_json = work / "cross_glasses_zeroshot_v21.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark -> {out_json} (total {(time.time()-t0):.1f}s)", flush=True)


if __name__ == "__main__":
    main()
