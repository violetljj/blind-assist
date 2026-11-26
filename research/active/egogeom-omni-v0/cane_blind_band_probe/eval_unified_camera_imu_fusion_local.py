"""Option A (Local, 0.00 CCU, Zero-Training):
Narrow-Cone Frontal-Normal Angular TTC + Motion-Gated Zero-VIO Gyro TTC + Unified Camera+IMU Fusion.

Controlled ablation on the exact 38-source 3-Zone Cane-Neutral crossfit (103 clips, 12,872 frames)
+ Aria pure overhangs (`ovh_midup=386`, `ovh_head=64`, `grounded_midup=283`, `strict_clear=954`):
  1. m6_ha70_nopg             : Step 1 parity baseline (full CF=45/365, P5=86/365)
  2. m6_ha70_par_refute       : Existing 1.07 ms far-refutation parallax + m6_ha70 (full CF=57/365)
  3. ang16_log_ttc            : H1 wide ±16° cone + log-TTC baseline
  4. ang10_log_ttc            : Narrowed ±10° cone + log-TTC
  5. ang08_log_ttc            : Shoulder-width ±8° cone + log-TTC
  6. ang08_front_log_ttc      : ±8° + frontal-normal weighting (suppresses parallel indoor side walls) + az cosine
  7. gyro_ttc_gated           : Zero-VIO, Zero-Depth ±12° Gyro TTC + IMU rotation-rate & walking coherence gate
  8. cam_imu_novio_fuse       : 100% Zero-VIO, Zero-ToF (ang08_front_log_ttc + gyro_ttc_gated)
  9. fuse_m6par_ang08         : m6_ha70_par_refute + calibrated ang08_front_log_ttc
 10. fuse_m6par_ang08_gyro    : Unified Camera+IMU (m6_ha70_par_refute + ang08_front_log_ttc + gyro_ttc_gated)

Zero edits to frozen `egogeom_omni_v41_model.py` or `:app`.
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
from eval_scalefree_angular_ttc_local import _compute_angular_rays_s4, _episode_hits  # noqa: E402
from run_cane_blind_colab import (  # noqa: E402
    M6HeadAnchoredRisk,
    assign_source_folds,
    build_bev_band_eval_rows,
    confirm_score,
)

OPT_A_ARMS = (
    "m6_ha70_nopg",
    "m6_ha70_par_refute",
    "ang16_log_ttc",
    "ang10_log_ttc",
    "ang08_log_ttc",
    "ang08_front_log_ttc",
    "gyro_ttc_gated",
    "cam_imu_novio_fuse",
    "fuse_m6par_ang08",
    "fuse_m6par_ang08_gyro",
)


class NarrowAngularConeTTC:
    """Scale-free gravity-horizon angular cone with optional frontal-normal wall suppression and log-TTC.

    Uses `normal_grav` (`nrm_s4`: [n_right, n_down, n_fwd] in gravity-aligned camera frame) to suppress
    parallel side walls (`|n_fwd| << |n_right|`) in narrow indoor corridors without metric 3D-box clipping.
    """

    def __init__(
        self,
        az_half_deg: float = 8.0,
        el_lo_deg: float = -18.0,
        el_hi_deg: float = 12.0,
        use_log_ttc: bool = True,
        use_frontality: bool = False,
        use_az_cos: bool = False,
        n_az: int = 3,
        n_el: int = 4,
        support_scale: float = 8.0,
    ):
        self.az_half = float(np.radians(az_half_deg))
        self.el_lo = float(np.radians(el_lo_deg))
        self.el_hi = float(np.radians(el_hi_deg))
        self.use_log_ttc = bool(use_log_ttc)
        self.use_frontality = bool(use_frontality)
        self.use_az_cos = bool(use_az_cos)
        self.n_az = int(n_az)
        self.n_el = int(n_el)
        self.support_scale = float(support_scale)
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
        nx = nrm_s4[0][sub].astype(np.float32, copy=False)
        nd = nrm_s4[1][sub].astype(np.float32, copy=False)
        nz = nrm_s4[2][sub].astype(np.float32, copy=False)
        ls = ls_s4[sub].astype(np.float32, copy=False)

        valid = np.isfinite(z) & (z > 0.20) & (z < 25.0) & (fwd_dot > 0.25)
        if int(valid.sum()) < 20:
            self._prev_cells = None
            self._prev_log_z = None
            return 0.0

        fl = valid & (el < -0.25) & (np.abs(nd) >= 0.72)
        if int(fl.sum()) >= 12:
            obs_dh = float(np.clip(np.median(dh[fl]), -0.15, 0.20))
            self._floor_dh_ema = 0.75 * self._floor_dh_ema + 0.25 * obs_dh
        dh_eff = dh - self._floor_dh_ema

        disp = 1.0 / np.clip(z, 0.25, 25.0)
        bg_m = valid & (el >= -0.22)
        disp_bg = float(np.percentile(disp[bg_m], 30)) if int(bg_m.sum()) >= 15 else float(np.percentile(disp[valid], 30))
        disp_bg = max(disp_bg, 0.04)

        rel_prom = np.clip((disp / disp_bg) - 1.0, 0.0, 6.0)
        near_sig = np.clip(disp - 0.22, 0.0, 3.0)
        ls_med = float(np.median(ls[valid]))
        conf = np.exp(-np.clip(ls - ls_med, -2.0, 2.0))
        vert = np.clip(1.0 - nd * nd, 0.0, 1.0)
        dh_w = np.clip(dh_eff / 0.22, 0.0, 1.0)
        el_ramp = np.clip((el - self.el_lo) / 0.14, 0.0, 1.0)

        ev = (0.55 * rel_prom + 0.45 * near_sig * 2.5) * vert * dh_w * conf * el_ramp

        if self.use_frontality:
            # Parallel corridor walls have |nx| >> |nz|; frontal obstacles/overhangs face the camera (|nz| high).
            horiz_n = np.hypot(nx, nz) + 1e-4
            front_cos = np.abs(nz) / horiz_n
            front_w = 0.15 + 0.85 * (front_cos * front_cos)
            ev = ev * front_w

        if self.use_az_cos:
            az_w = np.cos(np.clip(np.abs(az) / self.az_half, 0.0, 1.0) * (0.5 * np.pi))
            ev = ev * az_w

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
            approach = np.where(ok_lz, np.clip(self._prev_log_z - cur_log_z, 0.0, 0.5), 0.0)
            sup = sup * (1.0 + 3.0 * approach)

        self._prev_cells = now_cells
        self._prev_log_z = cur_log_z
        return float(np.max(-np.expm1(-sup / self.support_scale)))


class GatedGyroEpipolarTTCRisk:
    """Zero-VIO, Zero-Depth Camera+IMU Gyro TTC with IMU rotation-rate gate and forward-walking coherence gate.

    Strict Zero-VIO: uses ONLY `gray_s4`, `K_full`, IMU rotation `R_wc`, and unit forward heading `strict_fwd`.
    Suppresses false positives during rapid head turns (`rot_deg > 5 deg`) and stationary frames.
    """

    def __init__(
        self,
        lag: int = 2,
        az_half_deg: float = 12.0,
        el_lo_deg: float = -16.0,
        el_hi_deg: float = 12.0,
        ttc_steps: tuple[float, ...] = (32.0, 20.0, 13.0, 8.0),
        min_grad: float = 0.012,
        min_par_px: float = 1.2,
        max_best_cost: float = 0.22,
        min_inf_margin: float = 0.14,
        min_unimodal_margin: float = 0.04,
        rot_gate_lo_deg: float = 4.0,
        rot_gate_hi_deg: float = 9.0,
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
        self.rot_gate_lo = float(rot_gate_lo_deg)
        self.rot_gate_hi = float(rot_gate_hi_deg)
        self.reset()

    def reset(self) -> None:
        self._hist: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=self.lag + 1)
        self._prev_grid: np.ndarray | None = None
        self._walk_ema: float = 0.0

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
        R_rel = R_prev.T @ R_f
        # Pure IMU gyroscope rotation magnitude across `lag` frames
        cos_th = float(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
        rot_deg = float(np.degrees(np.arccos(cos_th)))
        rot_gate = float(np.clip((self.rot_gate_hi - rot_deg) / max(self.rot_gate_hi - self.rot_gate_lo, 1e-3), 0.0, 1.0))
        if rot_gate <= 1e-3:
            self._prev_grid = np.zeros(9, dtype=np.float64)
            return 0.0

        H, W = gray.shape
        ang = _compute_angular_rays_s4(H, W, K_full, R_f, strict_fwd, stride=4, step=2)
        if ang is None:
            self._prev_grid = None
            return 0.0
        az, el, fwd_dot = ang

        grad = np.zeros_like(gray)
        grad[1:-1, 1:-1] = np.hypot(gray[1:-1, 2:] - gray[1:-1, :-2], gray[2:, 1:-1] - gray[:-2, 1:-1])

        # Evaluate both center narrow cone AND wider peripheral belt (for zero-VIO walking coherence check)
        wide_sub = (np.abs(az) <= np.radians(28.0)) & (el >= np.radians(-26.0)) & (el <= self.el_hi) & (fwd_dot > 0.3) & (grad[::2, ::2] >= self.min_grad)
        cand = np.zeros((H, W), dtype=bool)
        cand[::2, ::2] = wide_sub
        cand[:2, :] = cand[-2:, :] = cand[:, :2] = cand[:, -2:] = False
        vs, us = np.nonzero(cand)
        n_cells = 9  # 3 az x 3 el (narrow ±12° cone)
        now_grid = np.zeros(n_cells, dtype=np.float64)

        if len(us) > 0:
            fw = np.asarray(strict_fwd[:2], dtype=np.float32)
            fw /= float(np.linalg.norm(fw)) + 1e-6
            fw3 = np.array([fw[0], fw[1], 0.0], dtype=np.float32)
            u_fwd_prev = R_prev.T @ fw3

            fx_c, fy_c, cx_c, cy_c = float(K_s4[0, 0]), float(K_s4[1, 1]), float(K_s4[0, 2]), float(K_s4[1, 2])
            fx_p, fy_p, cx_p, cy_p = float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2])
            u_f = us.astype(np.float32)
            v_f = vs.astype(np.float32)
            rx = (u_f - cx_c) / fx_c
            ry = (v_f - cy_c) / fy_c
            drx = R_rel[0, 0] * rx + R_rel[0, 1] * ry + R_rel[0, 2]
            dry = R_rel[1, 0] * rx + R_rel[1, 1] * ry + R_rel[1, 2]
            drz = R_rel[2, 0] * rx + R_rel[2, 1] * ry + R_rel[2, 2]

            inv_lams = np.concatenate([np.zeros(1, dtype=np.float32), float(self.lag) / self.ttc_steps])
            px = drx[:, None] + inv_lams[None, :] * u_fwd_prev[0]
            py = dry[:, None] + inv_lams[None, :] * u_fwd_prev[1]
            pz = drz[:, None] + inv_lams[None, :] * u_fwd_prev[2]
            in_f = pz > 0.15
            pz_s = np.where(in_f, pz, 1.0)
            up = fx_p * (px / pz_s) + cx_p
            vp = fy_p * (py / pz_s) + cy_p
            in_img = in_f & (up >= 2.0) & (up < W - 2.0) & (vp >= 2.0) & (vp < H - 2.0)

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

                # Zero-VIO walking coherence: fraction of all wide-field pixels whose epipolar match
                # prefers forward translation over pure rotation (distinguishes walking vs standing still!)
                fwd_agree = float(((c_best < 0.25) & ((c_inf - c_best) >= 0.08)).mean())
                self._walk_ema = 0.6 * self._walk_ema + 0.4 * fwd_agree
                walk_gate = float(np.clip((self._walk_ema - 0.025) / 0.075, 0.0, 1.0))

                non_adj = np.abs(np.arange(c_near.shape[1])[None, :] - best_j[:, None]) > 1
                c_other = np.where(non_adj, c_near, 9.0).min(axis=1)
                v_sub_all = vs[idx] // 2
                u_sub_all = us[idx] // 2
                in_center = (
                    (np.abs(az[v_sub_all, u_sub_all]) <= self.az_half)
                    & (el[v_sub_all, u_sub_all] >= self.el_lo)
                    & (el[v_sub_all, u_sub_all] <= self.el_hi)
                )
                passed = (
                    in_center
                    & (c_best < self.max_best_cost)
                    & ((c_inf - c_best) >= self.min_inf_margin)
                    & ((c_other - c_best) >= self.min_unimodal_margin)
                )
                if passed.any() and walk_gate > 1e-3:
                    p_idx = idx[passed]
                    w_ttc = (10.0 / self.ttc_steps[best_j[passed]]) * rot_gate * walk_gate
                    v_sub = vs[p_idx] // 2
                    u_sub = us[p_idx] // 2
                    ia = np.clip(((az[v_sub, u_sub] + self.az_half) / (2.0 * self.az_half) * 3).astype(np.int32), 0, 2)
                    ie = np.clip(((el[v_sub, u_sub] - self.el_lo) / (self.el_hi - self.el_lo) * 3).astype(np.int32), 0, 2)
                    cid = ie * 3 + ia
                    cnts = np.bincount(cid, minlength=n_cells)
                    sums = np.bincount(cid, weights=w_ttc, minlength=n_cells)
                    now_grid = np.where(cnts >= 3, sums, 0.0)

        prev_g = np.zeros_like(now_grid) if self._prev_grid is None else self._prev_grid
        sup = 0.35 * now_grid + 0.65 * np.minimum(now_grid, prev_g)
        self._prev_grid = now_grid
        return float(np.max(-np.expm1(-sup / 5.0)))


def run_opt_a_arms_on_clip(
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
) -> dict[str, dict[str, np.ndarray]]:
    T = len(dep_s4)
    bev_ha85_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    m6_base = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    m6_par = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    par_ref = EgoGeomTemporalBEVV3(TemporalBEVV3Config())

    ang16 = NarrowAngularConeTTC(az_half_deg=16.0, el_lo_deg=-18.0, el_hi_deg=10.0, use_log_ttc=True, use_frontality=False, use_az_cos=False, n_az=5, n_el=4, support_scale=8.0)
    ang10 = NarrowAngularConeTTC(az_half_deg=10.0, el_lo_deg=-18.0, el_hi_deg=12.0, use_log_ttc=True, use_frontality=False, use_az_cos=False, n_az=3, n_el=4, support_scale=8.0)
    ang08 = NarrowAngularConeTTC(az_half_deg=8.0, el_lo_deg=-18.0, el_hi_deg=12.0, use_log_ttc=True, use_frontality=False, use_az_cos=False, n_az=3, n_el=4, support_scale=8.0)
    ang08_front = NarrowAngularConeTTC(az_half_deg=8.0, el_lo_deg=-18.0, el_hi_deg=12.0, use_log_ttc=True, use_frontality=True, use_az_cos=True, n_az=3, n_el=4, support_scale=12.0)
    ang08_front_novio = NarrowAngularConeTTC(az_half_deg=8.0, el_lo_deg=-18.0, el_hi_deg=12.0, use_log_ttc=True, use_frontality=True, use_az_cos=True, n_az=3, n_el=4, support_scale=12.0)
    gyro_g = GatedGyroEpipolarTTCRisk(lag=2, az_half_deg=12.0, el_lo_deg=-16.0, el_hi_deg=12.0)

    out = {
        arm: {"dist": np.full(T, np.inf, dtype=np.float32), "raw": np.zeros(T, dtype=np.float64)}
        for arm in OPT_A_ARMS
    }
    do_t = timing_acc is not None

    for i in range(T):
        args = (dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        d_85, _, _ = bev_ha85_g.step(*args)

        t0 = time.perf_counter() if do_t else 0.0
        r_b = m6_base.step(*args)
        if do_t:
            timing_acc["m6_ha70_nopg"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha70_nopg"]["dist"][i] = d_85
        out["m6_ha70_nopg"]["raw"][i] = r_b

        t0 = time.perf_counter() if do_t else 0.0
        gray_f = gray_s4[i].astype(np.float32) * (1.0 / 255.0)
        z_ref, dh_ref = par_ref._causal_parallax_refute(dep_s4[i], dh_s4[i], gray_f, Ks[i], poses_R[i], poses_t[i])
        r_pr = m6_par.step(z_ref, dh_ref, nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        if do_t:
            timing_acc["m6_ha70_par_refute"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha70_par_refute"]["dist"][i] = d_85
        out["m6_ha70_par_refute"]["raw"][i] = r_pr

        r_a16 = ang16.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        out["ang16_log_ttc"]["dist"][i] = d_85
        out["ang16_log_ttc"]["raw"][i] = r_a16

        r_a10 = ang10.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        out["ang10_log_ttc"]["dist"][i] = d_85
        out["ang10_log_ttc"]["raw"][i] = r_a10

        r_a08 = ang08.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        out["ang08_log_ttc"]["dist"][i] = d_85
        out["ang08_log_ttc"]["raw"][i] = r_a08

        t0 = time.perf_counter() if do_t else 0.0
        r_af = ang08_front.step(z_ref, dh_ref, nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        if do_t:
            timing_acc["ang08_front_log_ttc"].append((time.perf_counter() - t0) * 1e3)
        out["ang08_front_log_ttc"]["dist"][i] = d_85
        out["ang08_front_log_ttc"]["raw"][i] = r_af

        t0 = time.perf_counter() if do_t else 0.0
        r_gg = gyro_g.step(gray_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        if do_t:
            timing_acc["gyro_ttc_gated"].append((time.perf_counter() - t0) * 1e3)
        out["gyro_ttc_gated"]["dist"][i] = d_85
        out["gyro_ttc_gated"]["raw"][i] = r_gg

        # 100% Zero-VIO, Zero-ToF Camera+IMU fusion (raw dep_s4/dh_s4 + gated gyro, NEVER touches poses_t!)
        r_af_nv = ang08_front_novio.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], strict_fwd[i])
        out["cam_imu_novio_fuse"]["dist"][i] = d_85
        out["cam_imu_novio_fuse"]["raw"][i] = float(1.0 - (1.0 - 0.65 * r_af_nv) * (1.0 - 0.45 * r_gg))

        # Unified Camera+IMU Fusion 1: m6_ha70_par_refute (ground/narrow indoor) + calibrated narrow frontal overhang cone
        # Scale ang08_front gently (0.18) so it preserves m6_ha70's clean corridor calibration while rescuing overhangs
        out["fuse_m6par_ang08"]["dist"][i] = d_85
        out["fuse_m6par_ang08"]["raw"][i] = float(1.0 - (1.0 - r_pr) * (1.0 - 0.18 * r_af))

        # Unified Camera+IMU Fusion 2: full 3-channel (m6_ha70_par_refute + narrow frontal cone + motion-gated gyro TTC)
        out["fuse_m6par_ang08_gyro"]["dist"][i] = d_85
        out["fuse_m6par_ang08_gyro"]["raw"][i] = float(1.0 - (1.0 - r_pr) * (1.0 - 0.15 * r_af) * (1.0 - 0.18 * r_gg))

    for arm in OPT_A_ARMS:
        out[arm]["confirmed"] = confirm_score(out[arm]["raw"])
    return out


def eval_aria_overhangs(aria_clips_raw: list[dict]) -> dict:
    sub_names = ("grounded_midup", "ovh_midup", "ovh_head")
    out = {}
    clr_cat = {
        arm: np.concatenate([c["arms"][arm]["confirmed"][c["masks"]["strict_clear"]] for c in aria_clips_raw])
        for arm in OPT_A_ARMS
    }
    n_clr = sum(int(c["masks"]["strict_clear"].sum()) for c in aria_clips_raw)
    for sn in sub_names:
        n_pos = sum(int(c["masks"][sn].sum()) for c in aria_clips_raw)
        entry = {"n_frames": n_pos, "n_strict_clear": n_clr, "arms": {}}
        y_bin = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_clr, dtype=np.int32)])
        for arm in OPT_A_ARMS:
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
    timing_acc: dict[str, list[float]] = {k: [] for k in ("m6_ha70_nopg", "m6_ha70_par_refute", "ang08_front_log_ttc", "gyro_ttc_gated")}

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
            arms = run_opt_a_arms_on_clip(
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
                    "arms": arms,
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
        print(f"[OptA] {sname}: {len(clips_list)} clips in {time.time() - ts:.2f}s", flush=True)

    aria_ovh = eval_aria_overhangs(aria_clips_raw)
    print("\n=== Aria Pure Overhangs (grounded=283, ovh_midup=386, ovh_head=64 vs clear=954) ===", flush=True)
    for arm in OPT_A_ARMS:
        g = aria_ovh["grounded_midup"]["arms"][arm]
        om = aria_ovh["ovh_midup"]["arms"][arm]
        oh = aria_ovh["ovh_head"]["arms"][arm]
        print(
            f"  {arm:24s} | grounded={g['auroc_confirmed']:.4f} "
            f"| ovh_midup={om['auroc_confirmed']:.4f} (ep={om['ep_hit_rate_t5']*100:4.1f}%) "
            f"| ovh_head={oh['auroc_confirmed']:.4f} (R@5={oh['recall_at_5pct_clr']*100:4.1f}%)",
            flush=True,
        )

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
        print(f"\n=== 38-Source {bname} ===", flush=True)
        b_entry = {}
        for arm in OPT_A_ARMS:
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
                f"  {arm:24s} | P5={p5['timely']:3d}/{p5['avoidable']} ({p5['clear_warnings_per_km']:.2f}cw) "
                f"| CF={scf['timely']:3d}/{scf['avoidable']} ({scf['clear_warnings_per_km']:.2f}cw) "
                f"| DCF={dcf['timely']:3d}/{dcf['avoidable']} ({dcf['clear_warnings_per_km']:.2f}cw) "
                f"| Aria3z={dom_f['aria_gen2']['auroc_confirmed_3z']:.4f} ARKitD={dom_f['arkit_val30_day']['auroc_confirmed_3z']:.4f} CODa={dom_f['coda_campus']['auroc_confirmed_3z']:.4f}",
                flush=True,
            )
        bench_res[bname] = b_entry

    latency = {
        arm: {"mean_ms": round(float(np.mean(v)), 4), "p95_ms": round(float(np.percentile(v, 95)), 4)}
        for arm, v in timing_acc.items() if v
    }
    report = {
        "meta": {
            "checkpoint": "egogeom_omni_v41_pure_rgb_best.pt",
            "mode": "local_0ccu_opt_a_narrow_cone_fusion",
            "total_elapsed_s": round(time.time() - t0, 2),
        },
        "aria_overhang_arms": aria_ovh,
        "benchmarks_38source": bench_res,
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
    out_json = PROBE_DIR / "opt_a_narrow_fusion_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] Saved {out_json} ({out_json.stat().st_size / 1e3:.1f} KB) in {report['meta']['total_elapsed_s']}s; 0 NaN/Inf!", flush=True)
    print("[latency]", latency, flush=True)


if __name__ == "__main__":
    main()
