"""Causal Two-Frame Epipolar / Photometric Parallax Verification Gate (Zero-Retraining, Zero-Leakage).

Directly tests whether causal two-frame motion parallax from past frames (I_{t-k}, I_t) + causal ego-pose
(R_{t-k->t}, t_{t-k->t}) breaks the monocular clear-road hallucination floor on Meta Aria Gen 2
(519/954 = 54.40% on v4.1 single-frame monocular vs 39/954 stereo) AND improves M6 38-source cross-fit!

Physics principle (100% causal, zero future/GT leakage):
  Given a candidate close obstacle pixel (u, v) in frame t with predicted monocular depth z_pred < 3.0 m:
    - Pick the most recent strictly-past frame t_prev < t (within 1..5 frames) with translation baseline
      ||t_{t->t_prev}|| >= 0.08 m.
    - Project 3D point X_t(z_pred) into frame t_prev -> p_obs.
    - Project null-hypothesis far/flat-ground surface X_t(z_null) (z_null = max(2.2 * z_pred, 6.0 m) or flat ground)
      into frame t_prev -> p_null.
    - When epipolar parallax separation ||p_obs - p_null||_2 >= tau_px AND local patch has texture gradient:
        Compare zero-mean normalized patch photometric error (ZNCC / census-L1) at p_obs vs p_null (and multi-depth
        epipolar line sweep z in {z_pred, 1.5*z_pred, 2.2*z_pred, 4.0*z_pred, 10.0m}).
        If a strictly farther depth hypothesis matches I_{t_prev} significantly better than z_pred,
        the monocular near-obstacle pixel is epipolarly REFUTED as a clear-road hallucination!
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

ROUTE = Path(__file__).resolve().parent
REPO = ROUTE.parents[2]
ASSETS = REPO / "artifacts.local/work/egogeom-omni-v0"
M9_RO = ROUTE / "math_theory/m9_pgl4_frobenius_integrability/aria_local_ground_readout"
M6 = ROUTE / "math_theory/m6_evt_eprocess_risk"

sys.path[:0] = [str(ROUTE), str(M9_RO), str(M6)]
import readout as aria_ro
import replay
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3

H_S4, W_S4 = 72, 96
H_IN, W_IN = 288, 384


def decode_gray_s4_clip(pack_path: Path, dark_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Decode clip RGBs to grayscale [T, 72, 96] float32 + gradient magnitude [T, 72, 96] + s4 K [T, 3, 3]."""
    from eval_egogeom_v3_zeroshot import decode_isotropic_frame

    with np.load(pack_path, allow_pickle=False) as d:
        offs, blob = d["rgb_offsets"], d["rgb_blob"]
        orig_hw, K_orig = d["orig_hw"], d["K_orig"]
    T = len(offs) - 1
    grays = np.empty((T, H_S4, W_S4), dtype=np.float32)
    grads = np.empty((T, H_S4, W_S4), dtype=np.float32)
    K_s4 = np.empty((T, 3, 3), dtype=np.float32)
    for i in range(T):
        rgb_f, K = decode_isotropic_frame(blob[offs[i] : offs[i + 1]].tobytes(), orig_hw, K_orig, dark_scale=dark_scale)
        g = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]).astype(np.float32)
        g_s4 = cv2.resize(g, (W_S4, H_S4), interpolation=cv2.INTER_AREA)
        # Local zero-mean normalization (3x3 box) makes photometric matching illumination-invariant
        gx = cv2.Sobel(g_s4, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g_s4, cv2.CV_32F, 0, 1, ksize=3)
        grays[i] = g_s4
        grads[i] = np.hypot(gx, gy)
        K_i = K.astype(np.float32).copy()
        K_i[0, :] *= 0.25
        K_i[1, :] *= 0.25
        K_s4[i] = K_i
    return grays, grads, K_s4


def _sample_patch_zncc_cost(
    img_cur: np.ndarray,
    img_prev: np.ndarray,
    u_cur: np.ndarray,
    v_cur: np.ndarray,
    u_prev: np.ndarray,
    v_prev: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Compute vectorized (2r+1)x(2r+1) zero-mean normalized SAD (0=perfect match, 2=opposite) for N pixels."""
    H, W = img_cur.shape
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    du, dv = np.meshgrid(offs, offs)
    du = du.reshape(1, -1)  # [1, P]
    dv = dv.reshape(1, -1)

    uc = u_cur[:, None] + du
    vc = v_cur[:, None] + dv
    up = u_prev[:, None] + du
    vp = v_prev[:, None] + dv

    # Bilinear interpolation helper
    def interp(im, xs, ys):
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        fx = xs - x0
        fy = ys - y0
        x0c = np.clip(x0, 0, W - 2)
        y0c = np.clip(y0, 0, H - 2)
        return (
            (1 - fx) * (1 - fy) * im[y0c, x0c]
            + fx * (1 - fy) * im[y0c, x0c + 1]
            + (1 - fx) * fy * im[y0c + 1, x0c]
            + fx * fy * im[y0c + 1, x0c + 1]
        )

    pc = interp(img_cur, uc, vc)
    pp = interp(img_prev, up, vp)
    pc0 = pc - pc.mean(axis=1, keepdims=True)
    pp0 = pp - pp.mean(axis=1, keepdims=True)
    nc = np.sqrt((pc0 * pc0).sum(axis=1) + 1e-6)
    np_ = np.sqrt((pp0 * pp0).sum(axis=1) + 1e-6)
    zncc = (pc0 * pp0).sum(axis=1) / (nc * np_)
    return 1.0 - zncc  # [N], 0 = identical texture, 1 = uncorrelated, 2 = inverted


def causal_parallax_filter_frame(
    z_s4: np.ndarray,
    grays: np.ndarray,
    grads: np.ndarray,
    K_s4: np.ndarray,
    poses_R: np.ndarray,
    poses_t: np.ndarray,
    t_idx: int,
    max_z_check: float = 4.2,
    min_baseline_m: float = 0.08,
    min_parallax_px: float = 1.0,
    min_grad: float = 0.020,
    cost_margin: float = 0.08,
    max_far_cost: float = 0.35,
    mult_hypotheses: tuple[float, ...] = (1.6, 2.5, 4.5),
    patch_radius: int = 2,
) -> tuple[np.ndarray, dict]:
    """Strictly causal two-frame epipolar check on near-depth pixels (z_s4 < max_z_check).

    Returns (z_filtered [72,96], diag).
    """
    z_out = z_s4.copy()
    diag = {"checked": 0, "parallax_ok": 0, "refuted": 0, "baseline_m": 0.0, "lag": 0}
    if t_idx == 0:
        return z_out, diag

    # Select most recent strictly-past frame (lag 1..6) with sufficient translation baseline
    R_cur = poses_R[t_idx]
    t_cur = poses_t[t_idx]
    best_prev = -1
    best_base = 0.0
    for lag in range(1, min(7, t_idx + 1)):
        p_idx = t_idx - lag
        base = float(np.linalg.norm(t_cur - poses_t[p_idx]))
        if base >= min_baseline_m:
            best_prev = p_idx
            best_base = base
            break
    if best_prev < 0:
        return z_out, diag
    diag["baseline_m"] = best_base
    diag["lag"] = t_idx - best_prev

    # Relative transform from cur camera frame -> prev camera frame:
    # X_world = R_cur X_cur + t_cur  =>  X_prev = R_prev^T (R_cur X_cur + t_cur - t_prev)
    R_prev = poses_R[best_prev]
    t_prev = poses_t[best_prev]
    R_rel = (R_prev.T @ R_cur).astype(np.float32)
    t_rel = (R_prev.T @ (t_cur - t_prev)).astype(np.float32)

    Kc = K_s4[t_idx]
    Kp = K_s4[best_prev]
    fx_c, fy_c, cx_c, cy_c = float(Kc[0, 0]), float(Kc[1, 1]), float(Kc[0, 2]), float(Kc[1, 2])
    fx_p, fy_p, cx_p, cy_p = float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2])

    # Candidate near-depth textured pixels inside safe border
    margin = patch_radius + 1
    cand = np.isfinite(z_s4) & (z_s4 > 0.25) & (z_s4 < max_z_check) & (grads[t_idx] >= min_grad)
    cand[:margin, :] = False
    cand[-margin:, :] = False
    cand[:, :margin] = False
    cand[:, -margin:] = False
    vs, us = np.nonzero(cand)
    if len(us) == 0:
        return z_out, diag
    diag["checked"] = int(len(us))

    z_pred = z_s4[vs, us].astype(np.float32)
    u_f = us.astype(np.float32)
    v_f = vs.astype(np.float32)
    # Unit-z rays in cur camera frame
    rx = (u_f - cx_c) / fx_c
    ry = (v_f - cy_c) / fy_c
    rz = np.ones_like(rx)
    # Rotated rays R_rel @ [rx, ry, 1]^T
    drx = R_rel[0, 0] * rx + R_rel[0, 1] * ry + R_rel[0, 2] * rz
    dry = R_rel[1, 0] * rx + R_rel[1, 1] * ry + R_rel[1, 2] * rz
    drz = R_rel[2, 0] * rx + R_rel[2, 1] * ry + R_rel[2, 2] * rz

    def project_depth(z_vec: np.ndarray):
        px = z_vec * drx + t_rel[0]
        py = z_vec * dry + t_rel[1]
        pz = z_vec * drz + t_rel[2]
        in_front = pz > 0.15
        pz_safe = np.where(in_front, pz, 1.0)
        up = fx_p * (px / pz_safe) + cx_p
        vp = fy_p * (py / pz_safe) + cy_p
        in_img = in_front & (up >= margin) & (up < W_S4 - margin) & (vp >= margin) & (vp < H_S4 - margin)
        return up, vp, in_img

    up_obs, vp_obs, ok_obs = project_depth(z_pred)
    # Reference far hypothesis (e.g. 4.5x z_pred) to check epipolar observability
    up_far_ref, vp_far_ref, ok_far_ref = project_depth(np.minimum(z_pred * mult_hypotheses[-1], 20.0))
    parallax_px = np.hypot(up_obs - up_far_ref, vp_obs - vp_far_ref)
    eval_mask = ok_obs & ok_far_ref & (parallax_px >= min_parallax_px)
    if not np.any(eval_mask):
        return z_out, diag
    diag["parallax_ok"] = int(eval_mask.sum())

    idx_e = np.nonzero(eval_mask)[0]
    u_e, v_e = u_f[idx_e], v_f[idx_e]
    z_e = z_pred[idx_e]
    c_obs = _sample_patch_zncc_cost(
        grays[t_idx], grays[best_prev], u_e, v_e, up_obs[idx_e], vp_obs[idx_e], radius=patch_radius
    )

    best_far_cost = np.full_like(c_obs, 9.0)
    best_far_z = z_e.copy()
    for mult in mult_hypotheses:
        z_h = np.minimum(z_e * mult, 20.0)
        up_h, vp_h, ok_h = project_depth(np.minimum(z_pred * mult, 20.0))
        up_he, vp_he, ok_he = up_h[idx_e], vp_h[idx_e], ok_h[idx_e]
        c_h = np.where(
            ok_he,
            _sample_patch_zncc_cost(grays[t_idx], grays[best_prev], u_e, v_e, up_he, vp_he, radius=patch_radius),
            9.0,
        )
        better = c_h < best_far_cost
        best_far_cost = np.where(better, c_h, best_far_cost)
        best_far_z = np.where(better, z_h, best_far_z)

    # Refute pixel if a strictly farther depth along the causal epipolar line fits significantly better
    # AND actually achieves a genuine match (best_far_cost < max_far_cost)
    refuted = (best_far_cost < max_far_cost) & ((c_obs - best_far_cost) >= cost_margin)
    if np.any(refuted):
        r_idx = idx_e[refuted]
        z_out[vs[r_idx], us[r_idx]] = best_far_z[refuted]
        diag["refuted"] = int(refuted.sum())
    return z_out, diag


def eval_aria_audit(tag: str, margin_grid: list[float], min_parallax_px: float = 1.0) -> dict:
    """Evaluate causal parallax gate directly on Codex Round-3 Aria 8-voxel clear-road audit (1,794 frames)."""
    stream_dir = ASSETS / f"math_theory/{tag}_stream/aria_gen2"
    in_dir = ASSETS / "math_theory/m9_pgl4_frobenius_integrability/aria_local_ground_readout/input"
    pack_dir = ASSETS / "zeroshot_eval/aria_g2_20clips"

    # Preload all 19 clips once
    clips_data = []
    for p in sorted(stream_dir.glob("*.npz")):
        with np.load(pack_dir / p.name, allow_pickle=False) as d:
            er = d["eval_rows"]
        with np.load(p, allow_pickle=False) as o:
            dep_s4 = o["depth_s4"].astype(np.float32)  # [150,72,96]
            dh_s4 = o["dh_s4"].astype(np.float32)
            nrm_s4 = o["nrm_s4"].astype(np.float32)
            ls_s4 = o["ls_s4"].astype(np.float32)
            K_orig_s4 = o["K"].astype(np.float32)
            poses_R = o["poses_R"].astype(np.float32)
            poses_t = o["poses_t"].astype(np.float32)
            strict_fwd = o["strict_fwd"].astype(np.float32)
        with np.load(in_dir / p.name, allow_pickle=False) as a:
            r_ro, t_ro, fw_ro, K_nat = a["rot"], a["pos"], a["fw"], a["K"]
        grays, grads, K_s4 = decode_gray_s4_clip(pack_dir / p.name, dark_scale=1.0)
        clips_data.append({
            "name": p.stem,
            "er": er,
            "dep_s4": dep_s4,
            "dh_s4": dh_s4,
            "nrm_s4": nrm_s4,
            "ls_s4": ls_s4,
            "K_s4": K_s4,
            "K_full": K_orig_s4,
            "poses_R": poses_R,
            "poses_t": poses_t,
            "strict_fwd": strict_fwd,
            "r_ro": r_ro,
            "t_ro": t_ro,
            "fw_ro": fw_ro,
            "K_nat": K_nat,
            "grays": grays,
            "grads": grads,
        })

    results = {}
    for cfg_name, cost_margin, min_par, max_far_cost in [
        ("baseline_no_gate", None, None, None),
        ("parallax_strict_m012", 0.12, 1.0, 0.30),
        ("parallax_balanced_m008", 0.08, 1.0, 0.35),
        ("parallax_agg_m004", 0.04, 0.75, 0.45),
    ]:
        haz_det = haz_clr = haz_unk = 0
        clr_det = clr_clr = clr_unk = 0
        fixed_errs, finite_errs = [], []
        gate_times_ms = []
        tot_checked = tot_par = tot_ref = 0

        for cd in clips_data:
            er = cd["er"]
            history = deque(maxlen=5)
            pred_d = np.full(len(er), np.nan, dtype=np.float64)
            cov = np.full(len(er), np.nan, dtype=np.float64)
            for i in range(len(er)):
                z_map = cd["dep_s4"][i]
                if cost_margin is not None:
                    t0 = time.perf_counter_ns()
                    z_map, diag = causal_parallax_filter_frame(
                        z_map, cd["grays"], cd["grads"], cd["K_s4"],
                        cd["poses_R"], cd["poses_t"], i,
                        min_parallax_px=min_par, cost_margin=cost_margin, max_far_cost=max_far_cost,
                    )
                    gate_times_ms.append((time.perf_counter_ns() - t0) * 1e-6)
                    tot_checked += diag["checked"]
                    tot_par += diag["parallax_ok"]
                    tot_ref += diag["refuted"]

                z = z_map.reshape(-1).astype(np.float64)
                ok = np.isfinite(z) & (z > 0.30) & (z < 12.0)
                fx, fy, cx, cy = cd["K_nat"][i]
                pts = np.stack(((aria_ro.NATIVE_U[ok] - cx) * z[ok] / fx, (aria_ro.NATIVE_V[ok] - cy) * z[ok] / fy, z[ok]), axis=1)
                history.append((pts @ cd["r_ro"][i].T + cd["t_ro"][i]).astype(np.float32))
                d_i, c_i, _ = aria_ro.corridor(np.concatenate(history), cd["t_ro"][i].astype(float), cd["fw_ro"][i])
                pred_d[i], cov[i] = d_i, c_i

            D_gt = er[:, 0]
            gt_val = (np.isfinite(D_gt) | np.isinf(D_gt)) & ~np.isnan(er[:, 1]) & (er[:, 1] >= 0.6)
            gt_haz = gt_val & np.isfinite(D_gt) & (D_gt <= 4.0)
            gt_clr = gt_val & (~gt_haz)
            p_unk = (~np.isfinite(cov)) | (cov < 0.6)
            p_haz = (~p_unk) & np.isfinite(pred_d) & (pred_d <= 4.0)
            p_clr = (~p_unk) & (~p_haz)

            haz_det += int((gt_haz & p_haz).sum())
            haz_clr += int((gt_haz & p_clr).sum())
            haz_unk += int((gt_haz & p_unk).sum())
            clr_det += int((gt_clr & p_haz).sum())
            clr_clr += int((gt_clr & p_clr).sum())
            clr_unk += int((gt_clr & p_unk).sum())
            if gt_haz.any():
                d_fill = np.where(p_haz[gt_haz], pred_d[gt_haz], 6.0)
                fixed_errs.extend(np.abs(d_fill - D_gt[gt_haz]).tolist())
                fin = gt_haz & p_haz
                if fin.any():
                    finite_errs.extend(np.abs(pred_d[fin] - D_gt[fin]).tolist())

        n_clr = clr_det + clr_clr + clr_unk
        n_haz = haz_det + haz_clr + haz_unk
        results[cfg_name] = {
            "haz_840_det_clr_unk": [haz_det, haz_clr, haz_unk],
            "haz_recall_pct": round(100.0 * haz_det / max(1, n_haz), 2),
            "clr_954_halluc_clr_unk": [clr_det, clr_clr, clr_unk],
            "clear_halluc_pct": round(100.0 * clr_det / max(1, n_clr), 2),
            "fixed_mae_m": round(float(np.mean(fixed_errs)), 4),
            "finite_mae_m": round(float(np.mean(finite_errs)), 4),
            "gate_cpu_ms_p50_p95": [round(float(x), 3) for x in np.percentile(gate_times_ms, [50, 95])] if gate_times_ms else [0.0, 0.0],
            "pixels_checked_par_refuted": [tot_checked, tot_par, tot_ref],
        }
        print(f"  [{cfg_name:24s}] haz={haz_det}/{n_haz} ({results[cfg_name]['haz_recall_pct']}%) | "
              f"clear_halluc={clr_det}/{n_clr} ({results[cfg_name]['clear_halluc_pct']}%) | "
              f"fixed_mae={results[cfg_name]['fixed_mae_m']}m | cpu_p50={results[cfg_name]['gate_cpu_ms_p50_p95'][0]}ms", flush=True)
    return results


def eval_m6_crossfit_all_domains(tag: str = "v41", cost_margin: float = 0.08, min_par: float = 1.0) -> dict:
    """Test causal parallax gate + EgoGeomTemporalBEVV3 across ALL 103 clips (38-source held-out M6 cross-fit)."""
    clips, stream = replay.materialize(tag)
    dom_folders = {
        "aria_gen2": ("aria_g2_20clips", 1.0),
        "coda_campus": ("coda_zeroshot_shard000", 1.0),
        "arkit_val30_day": ("arkit_val30_shard00", 1.0),
        "arkit_val30_dark018": ("arkit_val30_shard00", 0.18),
    }
    cpu_gate_ms = []
    for c in clips:
        dom = c.obs["domain"]
        name = c.obs["name"]
        folder, dscale = dom_folders[dom]
        pack_p = ASSETS / "zeroshot_eval" / folder / f"{name}.npz"
        grays, grads, K_s4 = decode_gray_s4_clip(pack_p, dark_scale=dscale)
        with np.load(stream / dom / f"{name}.npz", allow_pickle=False) as o:
            dep_s4 = o["depth_s4"]
            dh_s4 = o["dh_s4"]
            nrm_s4 = o["nrm_s4"]
            ls_s4 = o["ls_s4"]
            K_full = o["K"]
            poses_R = o["poses_R"]
            poses_t = o["poses_t"]
            strict_fwd = o["strict_fwd"]

        bev_base = EgoGeomTemporalBEVV3()
        bev_par = EgoGeomTemporalBEVV3()
        g_base, g_par = [], []
        for i in range(len(dep_s4)):
            _, _, _, r0 = bev_base.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], K_full[i], poses_R[i], poses_t[i], strict_fwd[i])
            t0 = time.perf_counter_ns()
            z_f, _ = causal_parallax_filter_frame(
                dep_s4[i], grays, grads, K_s4, poses_R, poses_t, i,
                min_parallax_px=min_par, cost_margin=cost_margin,
            )
            cpu_gate_ms.append((time.perf_counter_ns() - t0) * 1e-6)
            # If parallax refuted a near pixel (z_f > dep_s4[i] + 0.1), also zero its positive height excess dh_s4
            refuted_px = z_f > (dep_s4[i] + 0.10)
            dh_f = np.where(refuted_px, np.minimum(dh_s4[i], 0.0), dh_s4[i])
            _, _, _, r1 = bev_par.step(z_f, dh_f, nrm_s4[i], ls_s4[i], K_full[i], poses_R[i], poses_t[i], strict_fwd[i])
            g_base.append(r0)
            g_par.append(r1)
        c.geom_base = np.asarray(g_base, dtype=np.float64)
        c.geom_parallax = np.asarray(g_par, dtype=np.float64)

    return {
        "gate_cpu_ms_p50_p95": [round(float(x), 3) for x in np.percentile(cpu_gate_ms, [50, 95])],
        "base_crossfit_pooled": replay.crossfit(clips, "geom_base")["pooled"],
        "parallax_crossfit_pooled": replay.crossfit(clips, "geom_parallax")["pooled"],
        "base_ceiling_all": replay.ceiling(clips, "geom_base"),
        "parallax_ceiling_all": replay.ceiling(clips, "geom_parallax"),
        "base_ceiling_aria": replay.ceiling([c for c in clips if c.obs["domain"] == "aria_gen2"], "geom_base"),
        "parallax_ceiling_aria": replay.ceiling([c for c in clips if c.obs["domain"] == "aria_gen2"], "geom_parallax"),
    }


def main() -> None:
    print("=== [1/2] Causal Two-Frame Epipolar Parallax Gate on Aria Clear-Road Audit (v4.1) ===", flush=True)
    aria_res = eval_aria_audit("v41", [0.12, 0.08, 0.04])
    print("\n=== [2/2] Full 12,872-Frame 38-Source Held-Out M6 Cross-Fit (v4.1 + Parallax Gate) ===", flush=True)
    m6_res = eval_m6_crossfit_all_domains("v41", cost_margin=0.08, min_par=1.0)
    print(json.dumps(m6_res, indent=2), flush=True)

    out = {"aria_clear_road_audit": aria_res, "m6_38source_crossfit": m6_res}
    dst = ASSETS / "causal_parallax_gate_v41_report.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {dst}", flush=True)


if __name__ == "__main__":
    main()
