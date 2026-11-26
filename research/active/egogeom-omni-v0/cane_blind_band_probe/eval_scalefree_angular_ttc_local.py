"""Hypothesis 1 (Local, 0.00 CCU, Zero-Training):
  Part 1A: 5-Stage Aria Overhang Miss Attribution Waterfall (Backbone vs Geom Filter vs 3D Box vs BEV Occ).
  Part 1B: 8-Arm Scale-Free Angular Cone, Log-TTC, Zero-VIO Gyro Epipolar TTC, & Bidirectional Parallax
           Benchmarked on 38-Source 3-Zone Cane-Neutral Crossfit + Aria Pure Overhangs (ovh_midup=386, ovh_head=64).

Zero edits to frozen `egogeom_omni_v41_model.py` or `:app`. Uses cached `v41_stream` (103 clips, 12,872 frames).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
PROBE_DIR = ART / "cane_blind_band_probe"
STREAM_DIR = ART / "math_theory/v41_stream"
ZS_DIR = ART / "zeroshot_eval"

sys.path[:0] = [str(HERE), str(ROUTE), str(ROUTE.parent / "egoclear-v0")]

from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3, TemporalBEVV3Config, _zncc_3x3_batch  # noqa: E402
from eval_cane_neutral_and_ablation import (  # noqa: E402
    FastClipEval3Zone,
    HeadAnchoredGeomBEV,
    exact_frontier_3z,
    frame_metrics_3zone,
    run_2fold_crossfit_3z,
)
from run_cane_blind_colab import (  # noqa: E402
    M6HeadAnchoredRisk,
    assign_source_folds,
    build_bev_band_eval_rows,
    confirm_score,
)

H1_ARMS = (
    "m6_ha70_nopg",
    "m6_ha70_par_refute",
    "angular_cone_rel",
    "angular_cone_par_refute",
    "angular_cone_log_ttc",
    "gyro_ttc_only",
    "angular_cone_plus_gyro_ttc",
    "m6_ha70_bidir_par",
)


# ---------------------------------------------------------------------------
# Helpers: Gravity-Aligned Angular Rays (ZERO depth dependence!)
# ---------------------------------------------------------------------------

def _compute_angular_rays_s4(
    H: int,
    W: int,
    K: np.ndarray,
    R_wc: np.ndarray,
    strict_fwd: np.ndarray,
    stride: int = 4,
    step: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute (az_rad, el_rad, fwd_dot) on subsampled [H//step, W//step] grid using ONLY K + IMU R_wc + fwd."""
    fw = np.asarray(strict_fwd[:2], dtype=np.float32)
    nf = float(np.linalg.norm(fw))
    if nf < 1e-6:
        return None
    fw3 = np.array([fw[0] / nf, fw[1] / nf, 0.0], dtype=np.float32)
    rt3 = np.array([fw3[1], -fw3[0], 0.0], dtype=np.float32)

    hs, ws = (H + step - 1) // step, (W + step - 1) // step
    v_i, u_i = np.indices((hs, ws), dtype=np.float32)
    s_pix = float(stride * step)
    u = s_pix * u_i + 0.5
    v = s_pix * v_i + 0.5

    fx, fy = max(float(K[0, 0]), 1.0), max(float(K[1, 1]), 1.0)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    rx = (u - cx) / fx
    ry = (v - cy) / fy
    rz = np.ones_like(rx)
    inv_n = 1.0 / np.sqrt(rx * rx + ry * ry + 1.0)
    rays_cam = np.stack([rx * inv_n, ry * inv_n, rz * inv_n], axis=-1)  # [hs, ws, 3]
    rays_w = rays_cam @ R_wc.astype(np.float32).T

    fwd_dot = rays_w @ fw3
    lat_dot = rays_w @ rt3
    up_dot = rays_w[..., 2]
    az = np.arctan2(lat_dot, np.maximum(fwd_dot, 1e-5))
    el = np.arcsin(np.clip(up_dot, -1.0, 1.0))
    return az, el, fwd_dot


# ---------------------------------------------------------------------------
# New Scale-Free Readout Engines
# ---------------------------------------------------------------------------

class AngularConeRelRisk:
    """Scale-free gravity-horizon angular cone relative foreground prominence + optional log-TTC boost.

    ZERO metric 3D-box clipping, ZERO `poses_t` required; portable across head/neck mounts via `el_lo_deg, el_hi_deg`.
    """

    def __init__(
        self,
        az_half_deg: float = 16.0,
        el_lo_deg: float = -18.0,
        el_hi_deg: float = 10.0,
        use_log_ttc: bool = False,
        n_az: int = 5,
        n_el: int = 4,
    ):
        self.az_half = float(np.radians(az_half_deg))
        self.el_lo = float(np.radians(el_lo_deg))
        self.el_hi = float(np.radians(el_hi_deg))
        self.use_log_ttc = bool(use_log_ttc)
        self.n_az = int(n_az)
        self.n_el = int(n_el)
        self.reset()

    def reset(self) -> None:
        self._prev_cells: np.ndarray | None = None
        self._prev_log_z: np.ndarray | None = None
        self._floor_dh_ema: float = 0.0

    def step(
        self,
        depth_s4: np.ndarray,
        dh_s4: np.ndarray,
        nrm_s4: np.ndarray,
        ls_s4: np.ndarray,
        K: np.ndarray,
        R_wc: np.ndarray,
        strict_fwd: np.ndarray,
    ) -> float:
        H, W = depth_s4.shape
        ang = _compute_angular_rays_s4(H, W, K, R_wc, strict_fwd, stride=4, step=2)
        if ang is None:
            self._prev_cells = None
            self._prev_log_z = None
            return 0.0
        az, el, fwd_dot = ang

        sub = np.s_[::2, ::2]
        z = depth_s4[sub].astype(np.float32, copy=False)
        dh = dh_s4[sub].astype(np.float32, copy=False)
        nd = nrm_s4[1][sub].astype(np.float32, copy=False)
        ls = ls_s4[sub].astype(np.float32, copy=False)

        valid = np.isfinite(z) & (z > 0.20) & (z < 25.0) & (fwd_dot > 0.25)
        if int(valid.sum()) < 20:
            self._prev_cells = None
            self._prev_log_z = None
            return 0.0

        # Online floor dh bias from clearly downward-looking horizontal surfaces
        fl = valid & (el < -0.25) & (np.abs(nd) >= 0.72)
        if int(fl.sum()) >= 12:
            obs_dh = float(np.clip(np.median(dh[fl]), -0.15, 0.20))
            self._floor_dh_ema = 0.75 * self._floor_dh_ema + 0.25 * obs_dh
        dh_eff = dh - self._floor_dh_ema

        disp = 1.0 / np.clip(z, 0.25, 25.0)
        # Scale-free background disparity reference in the horizon/upper half
        bg_m = valid & (el >= -0.22)
        disp_bg = float(np.percentile(disp[bg_m], 30)) if int(bg_m.sum()) >= 15 else float(np.percentile(disp[valid], 30))
        disp_bg = max(disp_bg, 0.04)

        # Relative foreground jump (scale-invariant!) + mild absolute near-field anchor
        rel_prom = np.clip((disp / disp_bg) - 1.0, 0.0, 6.0)
        near_sig = np.clip(disp - 0.22, 0.0, 3.0)  # gentle preference for <4.5m without hard 3D cutoff
        ls_med = float(np.median(ls[valid]))
        conf = np.exp(-np.clip(ls - ls_med, -2.0, 2.0))
        vert = np.clip(1.0 - nd * nd, 0.0, 1.0)
        dh_w = np.clip(dh_eff / 0.22, 0.0, 1.0)
        # Elevation soft ramp near lower edge of cane-blind cone
        el_ramp = np.clip((el - self.el_lo) / 0.14, 0.0, 1.0)

        ev = (0.55 * rel_prom + 0.45 * near_sig * 2.5) * vert * dh_w * conf * el_ramp

        cone = valid & (np.abs(az) <= self.az_half) & (el >= self.el_lo) & (el <= self.el_hi)
        n_cells = self.n_az * self.n_el
        now_cells = np.zeros(n_cells, dtype=np.float64)
        cur_log_z = np.full(n_cells, np.nan, dtype=np.float32)

        if cone.any():
            ia = np.clip(((az[cone] + self.az_half) / (2.0 * self.az_half) * self.n_az).astype(np.int32), 0, self.n_az - 1)
            ie = np.clip(((el[cone] - self.el_lo) / (self.el_hi - self.el_lo) * self.n_el).astype(np.int32), 0, self.n_el - 1)
            cid = ie * self.n_az + ia
            now_cells = np.bincount(cid, weights=ev[cone], minlength=n_cells).astype(np.float64)
            if self.use_log_ttc:
                lz_vals = np.log(np.clip(z[cone], 0.25, 25.0))
                for c_idx in np.unique(cid):
                    cur_log_z[c_idx] = float(np.percentile(lz_vals[cid == c_idx], 20))

        prev = np.zeros_like(now_cells) if self._prev_cells is None else self._prev_cells
        sup = 0.35 * now_cells + 0.65 * np.minimum(now_cells, prev)

        if self.use_log_ttc and self._prev_log_z is not None:
            ok_lz = np.isfinite(cur_log_z) & np.isfinite(self._prev_log_z)
            # Approach rate -d(log Z)/step > 0 means approaching (scale factor cancels!)
            approach = np.where(ok_lz, np.clip(self._prev_log_z - cur_log_z, 0.0, 0.5), 0.0)
            # Boost cells showing coherent scale-free approach
            sup = sup * (1.0 + 3.0 * approach)

        self._prev_cells = now_cells
        self._prev_log_z = cur_log_z
        return float(np.max(-np.expm1(-sup / 8.0)))


class GyroEpipolarTTCRisk:
    """Zero-VIO, Zero-Depth Pure Camera + IMU Gyroscope Scale-Free TTC Looming Detector.

    Uses ONLY relative rotation `R_rel = R_prev.T @ R_cur` and forward unit heading `fw3` (NO `poses_t`, NO `depth_s4`!).
    Tests scale-free TTC hypotheses (in frame-lag multiples) via 3x3 ZNCC with strict unimodal + 2-frame persistence locks.
    """

    def __init__(
        self,
        lag: int = 2,
        az_half_deg: float = 18.0,
        el_lo_deg: float = -16.0,
        el_hi_deg: float = 12.0,
        ttc_steps: tuple[float, ...] = (32.0, 20.0, 13.0, 8.0),
        min_grad: float = 0.012,
        min_par_px: float = 1.2,
        max_best_cost: float = 0.22,
        min_inf_margin: float = 0.14,
        min_unimodal_margin: float = 0.04,
    ):
        self.lag = int(lag)
        self.az_half = float(np.radians(az_half_deg))
        self.el_lo = float(np.radians(el_lo_deg))
        self.el_hi = float(np.radians(el_hi_deg))
        self.ttc_steps = np.array(ttc_steps, dtype=np.float32)
        self.min_grad = float(min_grad)
        self.min_par_px = float(min_par_px)
        self.max_best_cost = float(max_best_cost)
        self.min_inf_margin = float(min_inf_margin)
        self.min_unimodal_margin = float(min_unimodal_margin)
        self.reset()

    def reset(self) -> None:
        self._hist: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=self.lag + 1)
        self._prev_grid: np.ndarray | None = None

    def step(
        self,
        gray_s4: np.ndarray,
        K_full: np.ndarray,
        R_wc: np.ndarray,
        strict_fwd: np.ndarray,
    ) -> float:
        gray = gray_s4.astype(np.float32) * (1.0 / 255.0) if gray_s4.dtype == np.uint8 else gray_s4.astype(np.float32, copy=False)
        K_s4 = K_full.astype(np.float32).copy()
        K_s4[0, :] *= 0.25
        K_s4[1, :] *= 0.25
        R_f = R_wc.astype(np.float32, copy=False)

        self._hist.append((gray, K_s4, R_f))
        if len(self._hist) <= self.lag:
            self._prev_grid = None
            return 0.0

        prev_gray, Kp, R_prev = self._hist[0]
        H, W = gray.shape
        ang = _compute_angular_rays_s4(H, W, K_full, R_f, strict_fwd, stride=4, step=2)
        if ang is None:
            self._prev_grid = None
            return 0.0
        az, el, fwd_dot = ang

        grad = np.zeros_like(gray)
        grad[1:-1, 1:-1] = np.hypot(gray[1:-1, 2:] - gray[1:-1, :-2], gray[2:, 1:-1] - gray[:-2, 1:-1])

        cone_sub = (np.abs(az) <= self.az_half) & (el >= self.el_lo) & (el <= self.el_hi) & (fwd_dot > 0.3) & (grad[::2, ::2] >= self.min_grad)
        cand = np.zeros((H, W), dtype=bool)
        cand[::2, ::2] = cone_sub
        cand[:2, :] = cand[-2:, :] = cand[:, :2] = cand[:, -2:] = False
        vs, us = np.nonzero(cand)
        n_cells = 15  # 5 az x 3 el
        now_grid = np.zeros(n_cells, dtype=np.float64)

        if len(us) > 0:
            fw = np.asarray(strict_fwd[:2], dtype=np.float32)
            fw /= float(np.linalg.norm(fw)) + 1e-6
            fw3 = np.array([fw[0], fw[1], 0.0], dtype=np.float32)
            # Forward heading in previous camera frame (unit translation direction, ZERO metric baseline!)
            u_fwd_prev = R_prev.T @ fw3
            R_rel = R_prev.T @ R_f

            fx_c, fy_c, cx_c, cy_c = float(K_s4[0, 0]), float(K_s4[1, 1]), float(K_s4[0, 2]), float(K_s4[1, 2])
            fx_p, fy_p, cx_p, cy_p = float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2])
            u_f = us.astype(np.float32)
            v_f = vs.astype(np.float32)
            rx = (u_f - cx_c) / fx_c
            ry = (v_f - cy_c) / fy_c
            drx = R_rel[0, 0] * rx + R_rel[0, 1] * ry + R_rel[0, 2]
            dry = R_rel[1, 0] * rx + R_rel[1, 1] * ry + R_rel[1, 2]
            drz = R_rel[2, 0] * rx + R_rel[2, 1] * ry + R_rel[2, 2]

            # Hypothesis 0: infinite distance (pure gyro rotation, inv_lam = 0)
            # Hypotheses 1..M: inv_lam = lag / ttc_steps
            inv_lams = np.concatenate([np.zeros(1, dtype=np.float32), float(self.lag) / self.ttc_steps])
            px = drx[:, None] + inv_lams[None, :] * u_fwd_prev[0]
            py = dry[:, None] + inv_lams[None, :] * u_fwd_prev[1]
            pz = drz[:, None] + inv_lams[None, :] * u_fwd_prev[2]
            in_f = pz > 0.15
            pz_s = np.where(in_f, pz, 1.0)
            up = fx_p * (px / pz_s) + cx_p
            vp = fy_p * (py / pz_s) + cy_p
            in_img = in_f & (up >= 2.0) & (up < W - 2.0) & (vp >= 2.0) & (vp < H - 2.0)

            # Parallax between pure rotation (col 0) and nearest TTC hypothesis (col -1)
            par_px = np.hypot(up[:, 0] - up[:, -1], vp[:, 0] - vp[:, -1])
            ok = in_img[:, 0] & in_img[:, -1] & (par_px >= self.min_par_px)
            if ok.any():
                idx = np.nonzero(ok)[0]
                costs = _zncc_3x3_batch(gray, prev_gray, u_f[idx], v_f[idx], up[idx], vp[idx])
                costs = np.where(in_img[idx], costs, 9.0)
                c_inf = costs[:, 0]
                c_near = costs[:, 1:]
                best_j = np.argmin(c_near, axis=1)
                c_best = c_near[np.arange(len(idx)), best_j]
                # Unimodality: mask best_j and adjacent hypotheses to check for repetitive-texture ambiguity
                non_adj = np.abs(np.arange(c_near.shape[1])[None, :] - best_j[:, None]) > 1
                c_other = np.where(non_adj, c_near, 9.0).min(axis=1)
                passed = (
                    (c_best < self.max_best_cost)
                    & ((c_inf - c_best) >= self.min_inf_margin)
                    & ((c_other - c_best) >= self.min_unimodal_margin)
                )
                if passed.any():
                    p_idx = idx[passed]
                    w_ttc = 10.0 / self.ttc_steps[best_j[passed]]  # higher risk for shorter TTC
                    v_sub = vs[p_idx] // 2
                    u_sub = us[p_idx] // 2
                    ia = np.clip(((az[v_sub, u_sub] + self.az_half) / (2.0 * self.az_half) * 5).astype(np.int32), 0, 4)
                    ie = np.clip(((el[v_sub, u_sub] - self.el_lo) / (self.el_hi - self.el_lo) * 3).astype(np.int32), 0, 2)
                    cid = ie * 5 + ia
                    cnts = np.bincount(cid, minlength=n_cells)
                    sums = np.bincount(cid, weights=w_ttc, minlength=n_cells)
                    # Spatial cluster guard: require >= 3 agreeing pixels in the angular cell
                    now_grid = np.where(cnts >= 3, sums, 0.0)

        prev_g = np.zeros_like(now_grid) if self._prev_grid is None else self._prev_grid
        sup = 0.35 * now_grid + 0.65 * np.minimum(now_grid, prev_g)
        self._prev_grid = now_grid
        return float(np.max(-np.expm1(-sup / 5.0)))


def _bidir_parallax_rescue_near(
    bev_v3: EgoGeomTemporalBEVV3,
    depth_s4: np.ndarray,
    dh_s4: np.ndarray,
    nrm_s4: np.ndarray,
    gray_s4: np.ndarray,
    K_full: np.ndarray,
    R_wc: np.ndarray,
    t_w: np.ndarray,
    strict_fwd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric Near-Rescue epipolar pass on pixels where v4.1 predicted far (>3.5m) inside upper corridor."""
    # Step 1: Existing Far-Refutation (also updates bev_v3._par_hist)
    z_out, dh_out = bev_v3._causal_parallax_refute(depth_s4, dh_s4, gray_s4, K_full, R_wc, t_w)
    if dh_out is None:
        dh_out = dh_s4
    nrm_out = nrm_s4

    # Find best_prev from bev_v3._par_hist (excluding the frame just appended at [-1])
    if len(bev_v3._par_hist) < 2:
        return z_out, dh_out, nrm_out
    t_f = t_w.astype(np.float32, copy=False)
    best_prev = None
    hist_list = list(bev_v3._par_hist)[:-1]
    for item in reversed(hist_list):
        if float(np.linalg.norm(t_f - item[3])) >= bev_v3.cfg.parallax_min_baseline_m:
            best_prev = item
            break
    if best_prev is None:
        return z_out, dh_out, nrm_out

    prev_gray, Kp, R_prev, t_prev = best_prev
    gray = gray_s4.astype(np.float32) * (1.0 / 255.0) if gray_s4.dtype == np.uint8 else gray_s4.astype(np.float32, copy=False)
    H, W = gray.shape
    ang = _compute_angular_rays_s4(H, W, K_full, R_wc, strict_fwd, stride=4, step=2)
    if ang is None:
        return z_out, dh_out, nrm_out
    az, el, fwd_dot = ang

    grad = np.zeros_like(gray)
    grad[1:-1, 1:-1] = np.hypot(gray[1:-1, 2:] - gray[1:-1, :-2], gray[2:, 1:-1] - gray[:-2, 1:-1])

    # Only test pixels in upper-mid forward corridor where v4.1 predicted FAR (>3.5m)
    cand_sub = (
        (z_out[::2, ::2] > 3.5)
        & (grad[::2, ::2] >= 0.012)
        & (np.abs(az) <= np.radians(18.0))
        & (el >= np.radians(-16.0))
        & (el <= np.radians(12.0))
        & (fwd_dot > 0.3)
    )
    cand = np.zeros((H, W), dtype=bool)
    cand[::2, ::2] = cand_sub
    cand[:2, :] = cand[-2:, :] = cand[:, :2] = cand[:, -2:] = False
    vs, us = np.nonzero(cand)
    if len(us) == 0:
        return z_out, dh_out, nrm_out

    K_s4 = K_full.astype(np.float32).copy()
    K_s4[0, :] *= 0.25
    K_s4[1, :] *= 0.25
    R_f = R_wc.astype(np.float32, copy=False)
    R_rel = R_prev.T @ R_f
    t_rel = R_prev.T @ (t_f - t_prev)
    fx_c, fy_c, cx_c, cy_c = float(K_s4[0, 0]), float(K_s4[1, 1]), float(K_s4[0, 2]), float(K_s4[1, 2])
    fx_p, fy_p, cx_p, cy_p = float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2])

    u_f, v_f = us.astype(np.float32), vs.astype(np.float32)
    rx, ry = (u_f - cx_c) / fx_c, (v_f - cy_c) / fy_c
    drx = R_rel[0, 0] * rx + R_rel[0, 1] * ry + R_rel[0, 2]
    dry = R_rel[1, 0] * rx + R_rel[1, 1] * ry + R_rel[1, 2]
    drz = R_rel[2, 0] * rx + R_rel[2, 1] * ry + R_rel[2, 2]

    z_far = np.minimum(z_out[vs, us].astype(np.float32), 20.0)
    z_nears = np.array((0.9, 1.4, 2.1, 3.0), dtype=np.float32)
    z_grid = np.concatenate([z_far[:, None], np.broadcast_to(z_nears[None, :], (len(us), len(z_nears)))], axis=1)
    px = z_grid * drx[:, None] + t_rel[0]
    py = z_grid * dry[:, None] + t_rel[1]
    pz = z_grid * drz[:, None] + t_rel[2]
    in_f = pz > 0.15
    pz_s = np.where(in_f, pz, 1.0)
    up = fx_p * (px / pz_s) + cx_p
    vp = fy_p * (py / pz_s) + cy_p
    in_img = in_f & (up >= 2.0) & (up < W - 2.0) & (vp >= 2.0) & (vp < H - 2.0)

    par_px = np.hypot(up[:, 0] - up[:, 1], vp[:, 0] - vp[:, 1])
    ok = in_img[:, 0] & in_img[:, 1] & (par_px >= 1.2)
    if not ok.any():
        return z_out, dh_out, nrm_out

    idx = np.nonzero(ok)[0]
    costs = _zncc_3x3_batch(gray, prev_gray, u_f[idx], v_f[idx], up[idx], vp[idx])
    costs = np.where(in_img[idx], costs, 9.0)
    c_far = costs[:, 0]
    c_near = costs[:, 1:]
    best_j = np.argmin(c_near, axis=1)
    c_best = c_near[np.arange(len(idx)), best_j]
    non_adj = np.abs(np.arange(c_near.shape[1])[None, :] - best_j[:, None]) > 1
    c_other = np.where(non_adj, c_near, 9.0).min(axis=1)
    rescued = (c_best < 0.18) & ((c_far - c_best) >= 0.18) & ((c_other - c_best) >= 0.05)
    # Cluster guard: require >= 3 rescued pixels in frame corridor
    if int(rescued.sum()) < 3:
        return z_out, dh_out, nrm_out

    z_out = z_out.copy()
    dh_out = dh_out.copy()
    nrm_out = nrm_out.copy()
    r_idx = idx[rescued]
    z_out[vs[r_idx], us[r_idx]] = z_nears[best_j[rescued]]
    dh_out[vs[r_idx], us[r_idx]] = np.maximum(dh_out[vs[r_idx], us[r_idx]], 0.80)
    nrm_out[1, vs[r_idx], us[r_idx]] = 0.0  # vertical obstacle prior for rescued overhang
    return z_out, dh_out, nrm_out


# ---------------------------------------------------------------------------
# Part 1A: 5-Stage Aria Overhang Attribution Waterfall
# ---------------------------------------------------------------------------

def run_aria_overhang_waterfall(aria_clips_raw: list[dict]) -> dict:
    """Decompose Aria overhang misses into Raw Cone -> Relative Jump -> Geom Filter -> 3D Box -> BEV Occ."""
    subsets = ("grounded_midup", "ovh_midup", "ovh_head", "strict_clear")
    acc = {
        s: {
            "n_frames": 0,
            "s1_cone_raw_le4m": 0,
            "s2_cone_rel_jump_1p5x": 0,
            "s3_cone_geom_le4m": 0,
            "s4_metric_3d_box_le4m": 0,
            "s5_bev_ha85_geom_le4m": 0,
            "cone_d_p10_vals": [],
            "cone_rel_prom_vals": [],
        }
        for s in subsets
    }

    for c in aria_clips_raw:
        dep_s4, dh_s4, nrm_s4 = c["dep_s4"], c["dh_s4"], c["nrm_s4"]
        Ks, poses_R, strict_fwd = c["Ks"], c["poses_R"], c["strict_fwd"]
        d_ha85g = c["d_ha85g"]
        T = len(dep_s4)
        H, W = dep_s4.shape[1], dep_s4.shape[2]
        v_g, u_g = np.indices((H, W), dtype=np.float32)
        u_f = 4.0 * u_g + 0.5
        v_f = 4.0 * v_g + 0.5

        for i in range(T):
            ang = _compute_angular_rays_s4(H, W, Ks[i], poses_R[i], strict_fwd[i], stride=4, step=1)
            if ang is None:
                continue
            az, el, fwd_dot = ang
            z = dep_s4[i]
            dh = dh_s4[i]
            nd = np.abs(nrm_s4[i, 1])
            ok = np.isfinite(z) & (z > 0.25) & (z < 20.0) & (fwd_dot > 0.25)
            disp = np.where(ok, 1.0 / np.clip(z, 0.25, 20.0), 0.0)
            bg_m = ok & (el >= -0.22)
            disp_bg = float(np.percentile(disp[bg_m], 30)) if int(bg_m.sum()) >= 20 else 0.08
            disp_bg = max(disp_bg, 0.04)

            # Metric 3D projection for Stage 4
            fx, fy, cx, cy = float(Ks[i, 0, 0]), float(Ks[i, 1, 1]), float(Ks[i, 0, 2]), float(Ks[i, 1, 2])
            P_cam = np.stack([(u_f - cx) * z / fx, (v_f - cy) * z / fy, z], axis=-1)
            P_rel = P_cam @ poses_R[i].T
            fw = strict_fwd[i, :2]
            fw3 = np.array([fw[0], fw[1], 0.0], dtype=np.float32)
            fw3 /= float(np.linalg.norm(fw3)) + 1e-6
            rt3 = np.array([fw3[1], -fw3[0], 0.0], dtype=np.float32)
            z_fwd = P_rel @ fw3
            x_lat = P_rel @ rt3
            z_rel = P_rel[..., 2]

            for sname in subsets:
                if not c["masks"][sname][i]:
                    continue
                st = acc[sname]
                st["n_frames"] += 1
                el_lo = np.radians(-8.0) if sname == "ovh_head" else np.radians(-22.0)
                el_hi = np.radians(15.0) if sname == "ovh_head" else np.radians(12.0)
                cone = ok & (np.abs(az) <= np.radians(18.0)) & (el >= el_lo) & (el <= el_hi)
                d_p10 = float(np.percentile(z[cone], 10)) if int(cone.sum()) >= 4 else 25.0
                rel_p = float(np.percentile(disp[cone], 90) / disp_bg) if int(cone.sum()) >= 4 else 1.0
                st["cone_d_p10_vals"].append(d_p10)
                st["cone_rel_prom_vals"].append(rel_p)

                if d_p10 <= 4.0:
                    st["s1_cone_raw_le4m"] += 1
                if rel_p >= 1.5:
                    st["s2_cone_rel_jump_1p5x"] += 1

                cone_g = cone & (nd < 0.75) & (dh >= 0.10)
                if int(cone_g.sum()) >= 4 and float(np.percentile(z[cone_g], 10)) <= 4.0:
                    st["s3_cone_geom_le4m"] += 1

                box_m = (
                    ok & (nd < 0.75) & (dh >= 0.10)
                    & (np.abs(x_lat) <= 0.30) & (z_rel >= -0.85) & (z_rel <= 0.25)
                    & (z_fwd >= 0.25) & (z_fwd <= 4.0)
                )
                if int(box_m.sum()) >= 4:
                    st["s4_metric_3d_box_le4m"] += 1

                if np.isfinite(d_ha85g[i]) and d_ha85g[i] <= 4.0:
                    st["s5_bev_ha85_geom_le4m"] += 1

    out = {}
    clr_prom = np.array(acc["strict_clear"]["cone_rel_prom_vals"])
    clr_d = -np.array(acc["strict_clear"]["cone_d_p10_vals"])
    for sname in subsets:
        st = acc[sname]
        n = max(1, st["n_frames"])
        rec = {
            "n_frames": int(st["n_frames"]),
            "s1_cone_raw_le4m_rate": round(st["s1_cone_raw_le4m"] / n, 4),
            "s2_cone_rel_jump_1p5x_rate": round(st["s2_cone_rel_jump_1p5x"] / n, 4),
            "s3_cone_geom_le4m_rate": round(st["s3_cone_geom_le4m"] / n, 4),
            "s4_metric_3d_box_le4m_rate": round(st["s4_metric_3d_box_le4m"] / n, 4),
            "s5_bev_ha85_geom_le4m_rate": round(st["s5_bev_ha85_geom_le4m"] / n, 4),
            "median_cone_d_p10_m": round(float(np.median(st["cone_d_p10_vals"])), 3) if st["n_frames"] else None,
            "median_cone_rel_prom": round(float(np.median(st["cone_rel_prom_vals"])), 3) if st["n_frames"] else None,
        }
        if sname != "strict_clear" and st["n_frames"] > 0 and len(clr_prom) > 0:
            y = np.concatenate([np.ones(st["n_frames"], dtype=np.int32), np.zeros(len(clr_prom), dtype=np.int32)])
            rec["auroc_raw_cone_depth_vs_clear"] = round(float(roc_auc_score(y, np.concatenate([-np.array(st["cone_d_p10_vals"]), clr_d]))), 4)
            rec["auroc_cone_rel_prom_vs_clear"] = round(float(roc_auc_score(y, np.concatenate([np.array(st["cone_rel_prom_vals"]), clr_prom]))), 4)
        out[sname] = rec
    return out


# ---------------------------------------------------------------------------
# Clip Runner for All 8 H1 Arms
# ---------------------------------------------------------------------------

def run_h1_arms_on_clip(
    dep_s4: np.ndarray,
    dh_s4: np.ndarray,
    nrm_s4: np.ndarray,
    ls_s4: np.ndarray,
    gray_s4: np.ndarray,
    Ks: np.ndarray,
    poses_R: np.ndarray,
    poses_t: np.ndarray,
    strict_fwd: np.ndarray,
    timing_acc: dict[str, list[float]] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    T = len(dep_s4)
    bev_ha85_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    m6_base = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    m6_par = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    m6_bidir = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    par_ref = EgoGeomTemporalBEVV3(TemporalBEVV3Config())
    par_bidir = EgoGeomTemporalBEVV3(TemporalBEVV3Config())

    ang_rel = AngularConeRelRisk(use_log_ttc=False)
    ang_par = AngularConeRelRisk(use_log_ttc=False)
    ang_ttc = AngularConeRelRisk(use_log_ttc=True)
    gyro_ttc = GyroEpipolarTTCRisk(lag=2)

    out = {
        arm: {"dist": np.full(T, np.inf, dtype=np.float32), "raw": np.zeros(T, dtype=np.float64)}
        for arm in H1_ARMS
    }
    d_ha85g = np.full(T, np.inf, dtype=np.float32)
    do_t = timing_acc is not None

    for i in range(T):
        args = (dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        d_85, _, _ = bev_ha85_g.step(*args)
        d_ha85g[i] = d_85

        # 1. m6_ha70_nopg (Step 1 baseline)
        t0 = time.perf_counter() if do_t else 0.0
        r_b = m6_base.step(*args)
        if do_t:
            timing_acc["m6_ha70_nopg"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha70_nopg"]["dist"][i] = d_85
        out["m6_ha70_nopg"]["raw"][i] = r_b

        # 2. m6_ha70_par_refute (Existing Far-Refute parallax + m6_ha70)
        t0 = time.perf_counter() if do_t else 0.0
        gray_f = gray_s4[i].astype(np.float32) * (1.0 / 255.0)
        z_ref, dh_ref = par_ref._causal_parallax_refute(dep_s4[i], dh_s4[i], gray_f, Ks[i], poses_R[i], poses_t[i])
        r_pr = m6_par.step(z_ref, dh_ref, nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        if do_t:
            timing_acc["m6_ha70_par_refute"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha70_par_refute"]["dist"][i] = d_85
        out["m6_ha70_par_refute"]["raw"][i] = r_pr

        # 3. angular_cone_rel
        t0 = time.perf_counter() if do_t else 0.0
        r_ar = ang_rel.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        if do_t:
            timing_acc["angular_cone_rel"].append((time.perf_counter() - t0) * 1e3)
        out["angular_cone_rel"]["dist"][i] = d_85
        out["angular_cone_rel"]["raw"][i] = r_ar

        # 4. angular_cone_par_refute
        t0 = time.perf_counter() if do_t else 0.0
        r_ap = ang_par.step(z_ref, dh_ref, nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        if do_t:
            timing_acc["angular_cone_par_refute"].append((time.perf_counter() - t0) * 1e3)
        out["angular_cone_par_refute"]["dist"][i] = d_85
        out["angular_cone_par_refute"]["raw"][i] = r_ap

        # 5. angular_cone_log_ttc
        t0 = time.perf_counter() if do_t else 0.0
        r_at = ang_ttc.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        if do_t:
            timing_acc["angular_cone_log_ttc"].append((time.perf_counter() - t0) * 1e3)
        out["angular_cone_log_ttc"]["dist"][i] = d_85
        out["angular_cone_log_ttc"]["raw"][i] = r_at

        # 6. gyro_ttc_only (Zero-VIO, Zero-Depth!)
        t0 = time.perf_counter() if do_t else 0.0
        r_gt = gyro_ttc.step(gray_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        if do_t:
            timing_acc["gyro_ttc_only"].append((time.perf_counter() - t0) * 1e3)
        out["gyro_ttc_only"]["dist"][i] = d_85
        out["gyro_ttc_only"]["raw"][i] = r_gt

        # 7. angular_cone_plus_gyro_ttc
        out["angular_cone_plus_gyro_ttc"]["dist"][i] = d_85
        out["angular_cone_plus_gyro_ttc"]["raw"][i] = float(min(1.0, r_ap + 0.45 * r_gt))

        # 8. m6_ha70_bidir_par (Far-Refute + Near-Rescue using poses_t)
        t0 = time.perf_counter() if do_t else 0.0
        z_bi, dh_bi, nrm_bi = _bidir_parallax_rescue_near(
            par_bidir, dep_s4[i], dh_s4[i], nrm_s4[i], gray_f, Ks[i], poses_R[i], poses_t[i], strict_fwd[i]
        )
        r_bi = m6_bidir.step(z_bi, dh_bi, nrm_bi, ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        if do_t:
            timing_acc["m6_ha70_bidir_par"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha70_bidir_par"]["dist"][i] = d_85
        out["m6_ha70_bidir_par"]["raw"][i] = r_bi

    for arm in H1_ARMS:
        out[arm]["confirmed"] = confirm_score(out[arm]["raw"])
    return out, d_ha85g


def _episode_hits(mask_pos: np.ndarray, alerts: np.ndarray) -> tuple[int, int]:
    n = len(mask_pos)
    ep = hit = i = 0
    while i < n:
        if mask_pos[i]:
            j = i
            while j + 1 < n and mask_pos[j + 1]:
                j += 1
            if (j - i + 1) >= 2:
                ep += 1
                if alerts[i : j + 1].any():
                    hit += 1
            i = j + 1
        else:
            i += 1
    return ep, hit


def eval_aria_overhangs(aria_clips_raw: list[dict]) -> dict:
    sub_names = ("grounded_midup", "ovh_midup", "ovh_head")
    out = {}
    clr_cat = {
        arm: np.concatenate([c["arms"][arm]["confirmed"][c["masks"]["strict_clear"]] for c in aria_clips_raw])
        for arm in H1_ARMS
    }
    n_clr = sum(int(c["masks"]["strict_clear"].sum()) for c in aria_clips_raw)
    for sn in sub_names:
        n_pos = sum(int(c["masks"][sn].sum()) for c in aria_clips_raw)
        entry = {"n_frames": n_pos, "n_strict_clear": n_clr, "arms": {}}
        y_bin = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_clr, dtype=np.int32)])
        for arm in H1_ARMS:
            sc = np.concatenate([c["arms"][arm]["confirmed"][c["masks"][sn]] for c in aria_clips_raw])
            t5 = float(np.quantile(clr_cat[arm], 0.95))
            au = float(roc_auc_score(y_bin, np.concatenate([sc, clr_cat[arm]])))
            rec = {
                "auroc_confirmed": round(au, 4),
                "recall_at_5pct_clr": round(float((sc >= t5).mean()), 4),
                "t5_clr": round(t5, 5),
                "zero_tie": bool(t5 <= 1e-12),
            }
            if sn == "ovh_midup":
                ep_tot = hit_t5 = 0
                for c in aria_clips_raw:
                    e1, h1 = _episode_hits(c["masks"][sn], c["arms"][arm]["confirmed"] >= t5)
                    ep_tot += e1
                    hit_t5 += h1
                rec["episodes"] = ep_tot
                rec["ep_hit_rate_t5"] = round(hit_t5 / ep_tot, 4) if ep_tot else None
            entry["arms"][arm] = rec
        out[sn] = entry
    return out


def main() -> None:
    t0 = time.time()
    aria_gt_bundle = np.load(PROBE_DIR / "aria_height_band_gt.npz", allow_pickle=False)

    local_suites = [
        ("aria_gen2", "aria_g2_20clips"),
        ("arkit_val30_day", "arkit_val30_shard00"),
        ("arkit_val30_dark018", "arkit_val30_shard00"),
        ("coda_campus", "coda_zeroshot_shard000"),
    ]

    suite_clips: dict[str, list[dict]] = {}
    aria_clips_raw: list[dict] = []
    timing_acc: dict[str, list[float]] = {k: [] for k in H1_ARMS}

    for sname, folder in local_suites:
        ts = time.time()
        src_paths = sorted((ZS_DIR / folder).rglob("*.npz"))
        clips_list = []
        for ci, p in enumerate(src_paths):
            with np.load(p, allow_pickle=False) as d_src:
                eval_rows = d_src["eval_rows"].astype(np.float64)
                gt_bev_static = d_src["gt_bev_static"] if "gt_bev_static" in d_src else None
                cam_name = str(d_src["cam"]) if "cam" in d_src else ("rover_flir" if sname == "coda_campus" else "unknown")

            with np.load(STREAM_DIR / sname / p.name, allow_pickle=False) as d_obs:
                dep_s4 = d_obs["depth_s4"]
                dh_s4 = d_obs["dh_s4"]
                nrm_s4 = d_obs["nrm_s4"]
                ls_s4 = d_obs["ls_s4"]
                gray_s4 = d_obs["gray_s4"]
                Ks_arr = d_obs["K"]
                poses_R = d_obs["poses_R"]
                poses_t = d_obs["poses_t"]
                strict_fwd = d_obs["strict_fwd"]

            use_t = timing_acc if (sname == "aria_gen2" and ci < 6) else None
            arms, d_ha85g = run_h1_arms_on_clip(
                dep_s4, dh_s4, nrm_s4, ls_s4, gray_s4, Ks_arr, poses_R, poses_t, strict_fwd, use_t
            )

            gt_bands: dict[str, np.ndarray] = {}
            if sname == "aria_gen2":
                for st in ("centered_full", "strict_past"):
                    for b in ("full_ground", "low_only_ground", "midup_ground", "head_ground"):
                        key = f"{p.stem}__{st}__{b}"
                        if key in aria_gt_bundle:
                            gt_bands[f"{st}__{b}"] = aria_gt_bundle[key].astype(np.float64)

                rf = gt_bands["centered_full__full_ground"]
                rl = gt_bands["centered_full__low_only_ground"]
                rm = gt_bands["centered_full__midup_ground"]
                rh = gt_bands["centered_full__head_ground"]
                val = (np.isfinite(rf[:, 0]) | np.isinf(rf[:, 0])) & (~np.isnan(rf[:, 1])) & (rf[:, 1] >= 0.6)
                dl, dm, dh_gt, df = rl[:, 0], rm[:, 0], rh[:, 0], rf[:, 0]
                dyn = rf[:, 2] > 0.5
                fin_ld = np.isfinite(dl) & np.isfinite(dm)
                diff_ld = np.where(fin_ld, np.abs(dl - dm), np.inf)
                aria_clips_raw.append({
                    "stem": p.stem,
                    "dep_s4": dep_s4, "dh_s4": dh_s4, "nrm_s4": nrm_s4,
                    "Ks": Ks_arr, "poses_R": poses_R, "strict_fwd": strict_fwd,
                    "d_ha85g": d_ha85g, "arms": arms,
                    "masks": {
                        "grounded_midup": val & (dm <= 4.0) & (dl <= 4.0) & (diff_ld < 0.8),
                        "ovh_midup": val & (dm <= 4.0) & (dl > 4.0),
                        "ovh_head": val & (dh_gt <= 4.0) & (dl > 4.0),
                        "strict_clear": val & (df > 4.0) & (~dyn),
                    },
                })
            else:
                gt_bands["orig_full"] = eval_rows
                gt_bands["bev_full"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0, 1, 2))
                gt_bands["bev_midup"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (1, 2))

            clips_list.append({
                "domain": sname, "name": p.stem, "cam": cam_name,
                "arms": arms, "gt_bands": gt_bands,
            })

        assign_source_folds(clips_list)
        suite_clips[sname] = clips_list
        print(f"[H1] {sname}: {len(clips_list)} clips in {time.time() - ts:.2f}s", flush=True)

    # Part 1A: Waterfall Attribution
    waterfall = run_aria_overhang_waterfall(aria_clips_raw)
    print("\n=== Part 1A: Aria Overhang Attribution Waterfall ===", flush=True)
    for k, v in waterfall.items():
        print(f"  {k:15s} n={v['n_frames']:3d} | S1_cone<=4m={v['s1_cone_raw_le4m_rate']:.3f} | S2_rel>=1.5x={v['s2_cone_rel_jump_1p5x_rate']:.3f} | S3_geom={v['s3_cone_geom_le4m_rate']:.3f} | S4_3dbox={v['s4_metric_3d_box_le4m_rate']:.3f} | S5_bev={v['s5_bev_ha85_geom_le4m_rate']:.3f} | med_d={v['median_cone_d_p10_m']} | auc_d={v.get('auroc_raw_cone_depth_vs_clear')} | auc_rel={v.get('auroc_cone_rel_prom_vs_clear')}", flush=True)

    # Part 1B: Aria pure overhangs across all 8 arms
    aria_ovh_eval = eval_aria_overhangs(aria_clips_raw)

    # Part 1B: 38-source 3-Zone Cane-Neutral Crossfit + Full Ground Parity
    def _make_objs(target_map: dict[str, str], full_map: dict[str, str] | None = None) -> list[FastClipEval3Zone]:
        objs = []
        for s in ("aria_gen2", "arkit_val30_day", "arkit_val30_dark018", "coda_campus"):
            tk = target_map[s]
            fk = tk if full_map is None else full_map[s]
            for c in suite_clips[s]:
                rf = None if fk == tk else c["gt_bands"][fk]
                objs.append(FastClipEval3Zone(c["domain"], c["name"], c["device"], c["source"], c["fold"], c["gt_bands"][tk], rf, c["arms"]))
        return objs

    suites_eval = {
        "full_ground_centered": _make_objs({
            "aria_gen2": "centered_full__full_ground",
            "arkit_val30_day": "orig_full",
            "arkit_val30_dark018": "orig_full",
            "coda_campus": "orig_full",
        }),
        "midup_3zone_centered": _make_objs(
            {
                "aria_gen2": "centered_full__midup_ground",
                "arkit_val30_day": "bev_midup",
                "arkit_val30_dark018": "bev_midup",
                "coda_campus": "bev_midup",
            },
            {
                "aria_gen2": "centered_full__full_ground",
                "arkit_val30_day": "bev_full",
                "arkit_val30_dark018": "bev_full",
                "coda_campus": "bev_full",
            },
        ),
    }

    bench_res = {}
    for bname, objs in suites_eval.items():
        print(f"\n=== Part 1B: 38-Source {bname} ===", flush=True)
        b_entry = {}
        for arm in H1_ARMS:
            fr = exact_frontier_3z(objs, arm)
            cf = run_2fold_crossfit_3z(objs, arm)
            dom_f = {}
            for d in ("aria_gen2", "arkit_val30_day", "arkit_val30_dark018", "coda_campus"):
                d_objs = [o for o in objs if o.domain == d]
                dom_f[d] = frame_metrics_3zone(d_objs, arm, 4.0)
            b_entry[arm] = {"frontier": fr, "crossfit": cf, "domain_frame": dom_f}
            p5 = fr["best_at_5_clear_km"]
            scf = cf["shared_crossfit"]["pooled"]
            dcf = cf["device_crossfit"]["pooled"]
            print(
                f"  {arm:27s} | P5={p5['timely']:3d}/{p5['avoidable']} ({p5['clear_warnings_per_km']:.2f}cw) "
                f"| CF={scf['timely']:3d}/{scf['avoidable']} ({scf['clear_warnings_per_km']:.2f}cw, neu={(scf.get('cane_neutral_alert_rate') or 0.0)*100:.2f}%) "
                f"| DCF={dcf['timely']:3d}/{dcf['avoidable']} ({dcf['clear_warnings_per_km']:.2f}cw) "
                f"| Aria_AUROC3z={dom_f['aria_gen2']['auroc_confirmed_3z']} CODa={dom_f['coda_campus']['auroc_confirmed_3z']}",
                flush=True,
            )
        bench_res[bname] = b_entry

    # Parity check against Step 1 baseline
    b_full_cf = bench_res["full_ground_centered"]["m6_ha70_nopg"]["crossfit"]["shared_crossfit"]["pooled"]["timely"]
    b_full_p5 = bench_res["full_ground_centered"]["m6_ha70_nopg"]["frontier"]["best_at_5_clear_km"]["timely"]
    print(f"\n[parity] m6_ha70_nopg full_ground_centered: CF={b_full_cf}/365 (Step 1=45), P5={b_full_p5}/365 (Step 1=86)", flush=True)

    latency = {
        arm: {
            "mean_ms": round(float(np.mean(v)), 4),
            "p95_ms": round(float(np.percentile(v, 95)), 4),
        }
        for arm, v in timing_acc.items() if v
    }

    report = {
        "meta": {
            "checkpoint": "egogeom_omni_v41_pure_rgb_best.pt",
            "mode": "local_0ccu_cached_streams",
            "total_elapsed_s": round(time.time() - t0, 2),
        },
        "part1a_aria_overhang_waterfall": waterfall,
        "part1b_aria_overhang_arms": aria_ovh_eval,
        "part1b_38source_benchmarks": bench_res,
        "cpu_latency_ms": latency,
    }

    def _scan(obj, p="root"):
        if isinstance(obj, float) and not np.isfinite(obj):
            raise RuntimeError(f"Non-finite at {p}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{p}.{k}")
        elif isinstance(obj, list):
            for idx_i, v in enumerate(obj):
                _scan(v, f"{p}[{idx_i}]")

    _scan(report)
    out_json = PROBE_DIR / "scalefree_angular_ttc_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Saved {out_json} ({out_json.stat().st_size / 1e3:.1f} KB) in {report['meta']['total_elapsed_s']}s; 0 NaN/Inf!", flush=True)


if __name__ == "__main__":
    main()
