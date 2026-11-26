"""Fast (<1.5 ms CPU) Causal Two-Frame Epipolar Scale Calibrator + Obstacle-Only Parallax Gate.

Strictly causal, zero-retraining, zero-GT/future leakage:
  Stage A (Frame-Level Causal Epipolar Scale s_t):
    On up to 256 top-gradient pixels with observable baseline ||t|| >= 0.08 m, evaluate a 9-point
    global scale grid s in {0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.05, 2.40} by minimizing
    trimmed-mean 3x3 ZNCC photometric residual between I_t and strictly-past I_{t-k}.
    Smooth log(s_t) causally via EMA. Preserves flat ground planes identically (pure similarity scaling!).

  Stage B (Obstacle-Only Epipolar Refutation):
    ONLY on pixels strictly above ground (dh_s4 > 0.15 m and z_cal <= 4.0 m), if a far-plane
    hypothesis (z_far = max(2.5*z_cal, 8.0 m)) matches I_{t-k} significantly better than z_cal,
    suppress that pixel (set NaN) so clear-road hallucinations do NOT corrupt corridor occupancy!
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
from causal_parallax_gate import decode_gray_s4_clip, H_S4, W_S4
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3

SCALE_GRID = np.array([0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.05, 2.40], dtype=np.float32)
OFFS_3X3 = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], dtype=np.float32)


def _zncc_3x3(img_cur: np.ndarray, img_prev: np.ndarray, uc: np.ndarray, vc: np.ndarray, up: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Fast vectorized 3x3 ZNCC cost (1 - zncc) for shapes uc/vc: [N], up/vp: [N, S]."""
    H, W = img_cur.shape
    N, S = up.shape
    du = OFFS_3X3[:, 0]  # [9]
    dv = OFFS_3X3[:, 1]

    # Current patches: integer indexing [N, 9]
    ux = np.clip(np.rint(uc).astype(np.int32)[:, None] + du.astype(np.int32)[None, :], 0, W - 1)
    vy = np.clip(np.rint(vc).astype(np.int32)[:, None] + dv.astype(np.int32)[None, :], 0, H - 1)
    pc = img_cur[vy, ux]  # [N, 9]
    pc0 = pc - pc.mean(axis=1, keepdims=True)
    nc = np.sqrt((pc0 * pc0).sum(axis=1, keepdims=True) + 1e-6)  # [N, 1]

    # Previous patches: bilinear [N, S, 9]
    xs = up[:, :, None] + du[None, None, :]
    ys = vp[:, :, None] + dv[None, None, :]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    x0c = np.clip(x0, 0, W - 2)
    y0c = np.clip(y0, 0, H - 2)
    pp = (
        (1.0 - fx) * (1.0 - fy) * img_prev[y0c, x0c]
        + fx * (1.0 - fy) * img_prev[y0c, x0c + 1]
        + (1.0 - fx) * fy * img_prev[y0c + 1, x0c]
        + fx * fy * img_prev[y0c + 1, x0c + 1]
    )
    pp0 = pp - pp.mean(axis=2, keepdims=True)
    np_ = np.sqrt((pp0 * pp0).sum(axis=2) + 1e-6)  # [N, S]
    dot = (pc0[:, None, :] * pp0).sum(axis=2)  # [N, S]
    zncc = dot / (nc * np_)
    return 1.0 - zncc  # [N, S]


class CausalEpipolarScaleTracker:
    """Online strictly-causal frame scale calibrator + optional obstacle-only parallax suppressor."""

    def __init__(self, ema_alpha: float = 0.25, enable_scale: bool = True, enable_obs_gate: bool = False, obs_margin: float = 0.12):
        self.ema_alpha = ema_alpha
        self.enable_scale = enable_scale
        self.enable_obs_gate = enable_obs_gate
        self.obs_margin = obs_margin
        self.log_s = 0.0
        self.n_updates = 0

    @property
    def scale(self) -> float:
        return float(np.exp(self.log_s))

    def step(
        self,
        z_s4: np.ndarray,
        dh_s4: np.ndarray,
        grays: np.ndarray,
        grads: np.ndarray,
        K_s4: np.ndarray,
        poses_R: np.ndarray,
        poses_t: np.ndarray,
        t_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        diag = {"scale": self.scale, "obs_refuted": 0, "updated": False}
        if t_idx == 0:
            return z_s4 * self.scale, dh_s4 * self.scale, diag

        # Pick most recent strictly-past frame (lag 1..6) with baseline >= 0.10 m
        R_cur, t_cur = poses_R[t_idx], poses_t[t_idx]
        best_prev = -1
        for lag in range(1, min(7, t_idx + 1)):
            p = t_idx - lag
            if float(np.linalg.norm(t_cur - poses_t[p])) >= 0.10:
                best_prev = p
                break
        if best_prev < 0:
            return z_s4 * self.scale, dh_s4 * self.scale, diag

        R_prev, t_prev = poses_R[best_prev], poses_t[best_prev]
        R_rel = (R_prev.T @ R_cur).astype(np.float32)
        t_rel = (R_prev.T @ (t_cur - t_prev)).astype(np.float32)
        Kc, Kp = K_s4[t_idx], K_s4[best_prev]
        fx_c, fy_c, cx_c, cy_c = float(Kc[0, 0]), float(Kc[1, 1]), float(Kc[0, 2]), float(Kc[1, 2])
        fx_p, fy_p, cx_p, cy_p = float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2])

        # Stage A: Global causal epipolar scale estimation over top-256 textured pixels
        if self.enable_scale:
            g_map = grads[t_idx].copy()
            valid = np.isfinite(z_s4) & (z_s4 > 0.40) & (z_s4 < 6.0) & (g_map >= 0.03)
            valid[:2, :] = valid[-2:, :] = valid[:, :2] = valid[:, -2:] = False
            vs, us = np.nonzero(valid)
            if len(us) >= 32:
                if len(us) > 256:
                    top = np.argpartition(g_map[vs, us], -256)[-256:]
                    vs, us = vs[top], us[top]
                u_f, v_f = us.astype(np.float32), vs.astype(np.float32)
                z_v = z_s4[vs, us].astype(np.float32)
                rx = (u_f - cx_c) / fx_c
                ry = (v_f - cy_c) / fy_c
                drx = R_rel[0, 0] * rx + R_rel[0, 1] * ry + R_rel[0, 2]
                dry = R_rel[1, 0] * rx + R_rel[1, 1] * ry + R_rel[1, 2]
                drz = R_rel[2, 0] * rx + R_rel[2, 1] * ry + R_rel[2, 2]

                # Project all 9 scale hypotheses [N, 9]
                z_grid = z_v[:, None] * SCALE_GRID[None, :]  # [N, 9]
                px = z_grid * drx[:, None] + t_rel[0]
                py = z_grid * dry[:, None] + t_rel[1]
                pz = z_grid * drz[:, None] + t_rel[2]
                in_f = pz > 0.15
                pz_s = np.where(in_f, pz, 1.0)
                up = fx_p * (px / pz_s) + cx_p
                vp = fy_p * (py / pz_s) + cy_p
                in_img = in_f & (up >= 2) & (up < W_S4 - 2) & (vp >= 2) & (vp < H_S4 - 2)
                # Require observable parallax across scale sweep (>= 1.5 px between s=0.70 and s=2.40)
                par_span = np.hypot(up[:, 0] - up[:, -1], vp[:, 0] - vp[:, -1])
                ok_pix = in_img.all(axis=1) & (par_span >= 1.5)
                if ok_pix.sum() >= 24:
                    costs = _zncc_3x3(grays[t_idx], grays[best_prev], u_f[ok_pix], v_f[ok_pix], up[ok_pix], vp[ok_pix])
                    # Robust trimmed mean (bottom 60% of pixels per scale) rejects moving objects & occlusions!
                    k_keep = max(12, int(0.60 * costs.shape[0]))
                    part = np.partition(costs, k_keep - 1, axis=0)[:k_keep, :]
                    med_cost = part.mean(axis=0)  # [9]
                    best_i = int(np.argmin(med_cost))
                    # Require meaningful curve sharpness (at least 0.015 better than worst scale) and good match
                    if med_cost[best_i] < 0.35 and (med_cost.max() - med_cost[best_i]) >= 0.015:
                        s_hat = float(SCALE_GRID[best_i])
                        alpha = 0.50 if self.n_updates == 0 else self.ema_alpha
                        self.log_s = (1.0 - alpha) * self.log_s + alpha * float(np.log(s_hat))
                        self.log_s = float(np.clip(self.log_s, np.log(0.65), np.log(2.50)))
                        self.n_updates += 1
                        diag["updated"] = True
                        diag["scale"] = self.scale

        s = self.scale
        z_cal = (z_s4 * s).astype(np.float32)
        dh_cal = (dh_s4 * s).astype(np.float32)

        # Stage B: Optional obstacle-only epipolar verification (NEVER touches ground pixels!)
        if self.enable_obs_gate:
            cand = np.isfinite(z_cal) & (z_cal > 0.30) & (z_cal <= 4.1) & (dh_cal > 0.15) & (grads[t_idx] >= 0.025)
            cand[:2, :] = cand[-2:, :] = cand[:, :2] = cand[:, -2:] = False
            vs, us = np.nonzero(cand)
            if len(us) > 0:
                u_f, v_f = us.astype(np.float32), vs.astype(np.float32)
                z_v = z_cal[vs, us]
                rx = (u_f - cx_c) / fx_c
                ry = (v_f - cy_c) / fy_c
                drx = R_rel[0, 0] * rx + R_rel[0, 1] * ry + R_rel[0, 2]
                dry = R_rel[1, 0] * rx + R_rel[1, 1] * ry + R_rel[1, 2]
                drz = R_rel[2, 0] * rx + R_rel[2, 1] * ry + R_rel[2, 2]
                # Compare z_cal vs z_far = max(2.5 * z_cal, 8.0 m)
                z_pair = np.stack([z_v, np.maximum(z_v * 2.5, 8.0)], axis=1)
                px = z_pair * drx[:, None] + t_rel[0]
                py = z_pair * dry[:, None] + t_rel[1]
                pz = z_pair * drz[:, None] + t_rel[2]
                in_f = pz > 0.15
                pz_s = np.where(in_f, pz, 1.0)
                up = fx_p * (px / pz_s) + cx_p
                vp = fy_p * (py / pz_s) + cy_p
                ok = in_f.all(axis=1) & (up >= 2).all(axis=1) & (up < W_S4 - 2).all(axis=1) & (vp >= 2).all(axis=1) & (vp < H_S4 - 2).all(axis=1)
                par = np.hypot(up[:, 0] - up[:, 1], vp[:, 0] - vp[:, 1])
                ok &= par >= 1.2
                if ok.any():
                    idx = np.nonzero(ok)[0]
                    c2 = _zncc_3x3(grays[t_idx], grays[best_prev], u_f[idx], v_f[idx], up[idx], vp[idx])
                    refuted = (c2[:, 1] < 0.30) & ((c2[:, 0] - c2[:, 1]) >= self.obs_margin)
                    if refuted.any():
                        ri = idx[refuted]
                        z_cal[vs[ri], us[ri]] = np.nan
                        dh_cal[vs[ri], us[ri]] = 0.0
                        diag["obs_refuted"] = int(refuted.sum())
        return z_cal, dh_cal, diag


def main() -> None:
    stream_dir = ASSETS / "math_theory/v41_stream/aria_gen2"
    in_dir = ASSETS / "math_theory/m9_pgl4_frobenius_integrability/aria_local_ground_readout/input"
    pack_dir = ASSETS / "zeroshot_eval/aria_g2_20clips"

    clips_data = []
    for p in sorted(stream_dir.glob("*.npz")):
        with np.load(pack_dir / p.name, allow_pickle=False) as d:
            er = d["eval_rows"]
        with np.load(p, allow_pickle=False) as o:
            dep_s4 = o["depth_s4"].astype(np.float32)
            dh_s4 = o["dh_s4"].astype(np.float32)
            nrm_s4 = o["nrm_s4"].astype(np.float32)
            ls_s4 = o["ls_s4"].astype(np.float32)
            K_full = o["K"].astype(np.float32)
            poses_R = o["poses_R"].astype(np.float32)
            poses_t = o["poses_t"].astype(np.float32)
            strict_fwd = o["strict_fwd"].astype(np.float32)
        with np.load(in_dir / p.name, allow_pickle=False) as a:
            r_ro, t_ro, fw_ro, K_nat = a["rot"], a["pos"], a["fw"], a["K"]
        grays, grads, K_s4 = decode_gray_s4_clip(pack_dir / p.name, dark_scale=1.0)
        clips_data.append(dict(
            name=p.stem, er=er, dep_s4=dep_s4, dh_s4=dh_s4, nrm_s4=nrm_s4, ls_s4=ls_s4,
            K_s4=K_s4, K_full=K_full, poses_R=poses_R, poses_t=poses_t, strict_fwd=strict_fwd,
            r_ro=r_ro, t_ro=t_ro, fw_ro=fw_ro, K_nat=K_nat, grays=grays, grads=grads,
        ))

    print("=== Fast Causal Two-Frame Epipolar Scale + Obstacle-Only Gate on Aria (v4.1) ===", flush=True)
    aria_out = {}
    for name, en_s, en_g in [
        ("baseline_v41", False, False),
        ("epipolar_scale_only", True, False),
        ("obs_gate_only", False, True),
        ("epipolar_scale_plus_obs_gate", True, True),
    ]:
        haz_det = haz_clr = haz_unk = 0
        clr_det = clr_clr = clr_unk = 0
        fixed_errs, finite_errs, cpu_ms, scales = [], [], [], []
        for cd in clips_data:
            er = cd["er"]
            trk = CausalEpipolarScaleTracker(ema_alpha=0.25, enable_scale=en_s, enable_obs_gate=en_g)
            history = deque(maxlen=5)
            pred_d = np.full(len(er), np.nan, dtype=np.float64)
            cov = np.full(len(er), np.nan, dtype=np.float64)
            for i in range(len(er)):
                if en_s or en_g:
                    t0 = time.perf_counter_ns()
                    z_map, _, diag = trk.step(
                        cd["dep_s4"][i], cd["dh_s4"][i], cd["grays"], cd["grads"],
                        cd["K_s4"], cd["poses_R"], cd["poses_t"], i,
                    )
                    cpu_ms.append((time.perf_counter_ns() - t0) * 1e-6)
                    scales.append(diag["scale"])
                else:
                    z_map = cd["dep_s4"][i]
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

        aria_out[name] = {
            "haz_840_det_clr_unk": [haz_det, haz_clr, haz_unk],
            "haz_recall_pct": round(100.0 * haz_det / 840, 2),
            "clr_954_halluc_clr_unk": [clr_det, clr_clr, clr_unk],
            "clear_halluc_pct": round(100.0 * clr_det / 954, 2),
            "fixed_mae_m": round(float(np.mean(fixed_errs)), 4),
            "finite_mae_m": round(float(np.mean(finite_errs)), 4),
            "median_causal_scale": round(float(np.median(scales)), 3) if scales else 1.0,
            "cpu_ms_p50_p95": [round(float(x), 3) for x in np.percentile(cpu_ms, [50, 95])] if cpu_ms else [0.0, 0.0],
        }
        print(
            f"  [{name:28s}] haz={haz_det}/840 ({aria_out[name]['haz_recall_pct']}%) | "
            f"clr_halluc={clr_det}/954 ({aria_out[name]['clear_halluc_pct']}%) | "
            f"fixed_mae={aria_out[name]['fixed_mae_m']}m | s_med={aria_out[name]['median_causal_scale']} | "
            f"cpu_p50={aria_out[name]['cpu_ms_p50_p95'][0]}ms",
            flush=True,
        )

    # Full 12,872-frame 38-source held-out M6 cross-fit on epipolar_scale_only & epipolar_scale_plus_obs_gate
    print("\n=== Full 12,872-Frame 38-Source Held-Out M6 Cross-Fit (Fast Causal Epipolar Scale) ===", flush=True)
    clips, stream = replay.materialize("v41")
    dom_folders = {
        "aria_gen2": ("aria_g2_20clips", 1.0),
        "coda_campus": ("coda_zeroshot_shard000", 1.0),
        "arkit_val30_day": ("arkit_val30_shard00", 1.0),
        "arkit_val30_dark018": ("arkit_val30_shard00", 0.18),
    }
    cpu_all = []
    for c in clips:
        dom, cname = c.obs["domain"], c.obs["name"]
        folder, dscale = dom_folders[dom]
        grays, grads, K_s4 = decode_gray_s4_clip(ASSETS / "zeroshot_eval" / folder / f"{cname}.npz", dark_scale=dscale)
        with np.load(stream / dom / f"{cname}.npz", allow_pickle=False) as o:
            dep_s4, dh_s4, nrm_s4, ls_s4 = o["depth_s4"], o["dh_s4"], o["nrm_s4"], o["ls_s4"]
            K_full, poses_R, poses_t, strict_fwd = o["K"], o["poses_R"], o["poses_t"], o["strict_fwd"]

        bev0 = EgoGeomTemporalBEVV3()
        bev_s = EgoGeomTemporalBEVV3()
        bev_sg = EgoGeomTemporalBEVV3()
        trk_s = CausalEpipolarScaleTracker(ema_alpha=0.25, enable_scale=True, enable_obs_gate=False)
        trk_sg = CausalEpipolarScaleTracker(ema_alpha=0.25, enable_scale=True, enable_obs_gate=True)
        g0, gs, gsg = [], [], []
        for i in range(len(dep_s4)):
            _, _, _, r0 = bev0.step(dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], K_full[i], poses_R[i], poses_t[i], strict_fwd[i])
            t0 = time.perf_counter_ns()
            zs, dhs, _ = trk_s.step(dep_s4[i], dh_s4[i], grays, grads, K_s4, poses_R, poses_t, i)
            cpu_all.append((time.perf_counter_ns() - t0) * 1e-6)
            _, _, _, rs = bev_s.step(zs, dhs, nrm_s4[i], ls_s4[i], K_full[i], poses_R[i], poses_t[i], strict_fwd[i])
            zsg, dhsg, _ = trk_sg.step(dep_s4[i], dh_s4[i], grays, grads, K_s4, poses_R, poses_t, i)
            _, _, _, rsg = bev_sg.step(zsg, dhsg, nrm_s4[i], ls_s4[i], K_full[i], poses_R[i], poses_t[i], strict_fwd[i])
            g0.append(r0)
            gs.append(rs)
            gsg.append(rsg)
        c.geom_base = np.asarray(g0, dtype=np.float64)
        c.geom_scale = np.asarray(gs, dtype=np.float64)
        c.geom_scale_gate = np.asarray(gsg, dtype=np.float64)

    m6_out = {
        "scale_tracker_cpu_ms_p50_p95": [round(float(x), 3) for x in np.percentile(cpu_all, [50, 95])],
        "base_crossfit_pooled": replay.crossfit(clips, "geom_base")["pooled"],
        "scale_crossfit_pooled": replay.crossfit(clips, "geom_scale")["pooled"],
        "scale_gate_crossfit_pooled": replay.crossfit(clips, "geom_scale_gate")["pooled"],
        "base_ceiling_all": replay.ceiling(clips, "geom_base"),
        "scale_ceiling_all": replay.ceiling(clips, "geom_scale"),
        "scale_gate_ceiling_all": replay.ceiling(clips, "geom_scale_gate"),
        "base_ceiling_aria": replay.ceiling([c for c in clips if c.obs["domain"] == "aria_gen2"], "geom_base"),
        "scale_ceiling_aria": replay.ceiling([c for c in clips if c.obs["domain"] == "aria_gen2"], "geom_scale"),
        "scale_gate_ceiling_aria": replay.ceiling([c for c in clips if c.obs["domain"] == "aria_gen2"], "geom_scale_gate"),
    }
    print(json.dumps(m6_out, indent=2), flush=True)
    dst = ASSETS / "causal_epipolar_scale_v41_report.json"
    dst.write_text(json.dumps({"aria_clear_road_audit": aria_out, "m6_38source_crossfit": m6_out}, indent=2), encoding="utf-8")
    print(f"Saved -> {dst}", flush=True)


if __name__ == "__main__":
    main()
