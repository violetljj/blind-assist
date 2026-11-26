"""Step 1 (0 CCU, Zero-Training): Cane-Neutral 3-Zone Re-Eval & Readout Ablation Ladder.

Part 1 (5 domains: Aria, ARKit Day, ARKit Dark, CODa, SANPO-Real ZS40 + 4-domain 38-source):
  Re-evaluates all 8 Round-1 readout arms under the Cane-Neutral 3-Zone protocol:
    - haz:          valid & (D_target <= 4.0)           -> must alert (misses penalized)
    - cane_neutral: valid & (D_target > 4.0) & (D_full <= 4.0) -> cane-covered low obstacle
                    (neither missed-hazard nor false-alarm/clear-warning penalty)
    - strict_clear: valid & (D_full > 4.0) & (~dyn) & (~haz)   -> strict empty road (FA penalized)
  Also logs transparent `cane_neutral_alert_rate = neutral_alert_frames / neutral_frames`.

Part 2 (4 domains with cached stride-4 tensors: 103 clips, 38 sources):
  Tests 7 strictly causal, <1 ms/frame pure-NumPy Stage-2 readout upgrades against
  5 Round-1 reference arms on `full_ground_centered` (2-zone), `midup_3zone_centered`,
  `midup_3zone_strict_past_aria`, and `ha85_3zone_centered`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np
from numba import njit
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
PROBE_DIR = ART / "cane_blind_band_probe"
STREAM_DIR = ART / "math_theory/v41_stream"
ZS_DIR = ART / "zeroshot_eval"

sys.path[:0] = [str(HERE), str(ROUTE), str(ROUTE.parent / "egoclear-v0")]

from egogeom_temporal_bev import EgoGeomTemporalBEV, TemporalBEVConfig, _horizontal_fwd_right  # noqa: E402
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3, TemporalBEVV3Config  # noqa: E402
from eval_closed_loop import H as EVAL_H, MIN_EVENT_FRAMES_V2, REFRACTORY_S, make_episodes, score_episode, weak_mask  # noqa: E402
from run_cane_blind_colab import (  # noqa: E402
    ALL_ARMS as R1_ALL_ARMS,
    BW_SPEEDS_PROXY,
    FA_BUDGET,
    FINE_Q_GRID,
    M6HeadAnchoredRisk,
    Q_GRID,
    assign_source_folds,
    build_bev_band_eval_rows,
    confirm_score,
)

LATE, AVOID, UNAVOID, FALSE, FFRAME, KM, CLEAR_WARN, ALERT, VALID, ON_CLEAR, CLEAR, ON_NEUTRAL, NEUTRAL = range(13)

NEW_ARMS = (
    "bev_midup_cal_thr",
    "bev_ha85_geom",
    "m6_ha85_nopg",
    "m6_ha70_nopg",
    "m6_ha85_dhcal",
    "m6_ha_adapt",
    "m6_ha_adapt_v2",
)
REF_ARMS_PART2 = (
    "bev_full_ground",
    "bev_midup_ground",
    "m6_full_ground",
    "m6_midup_ground",
    "m6_head_anchored_85_pg",
)
PART2_ARMS = REF_ARMS_PART2 + NEW_ARMS


# ---------------------------------------------------------------------------
# Bit-Exact Numba 3-Zone Scorer
# ---------------------------------------------------------------------------

@njit
def _score_ep_3zone_numba(
    alerts_raw: np.ndarray,
    valid: np.ndarray,
    haz: np.ndarray,
    dyn: np.ndarray,
    weak: np.ndarray,
    neutral: np.ndarray,
    ok_full: np.ndarray,
    ds: np.ndarray,
    refr: float,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    seg_has_ok: np.ndarray,
    strict_clear: np.ndarray,
    cum_m: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    n = len(alerts_raw)
    alerts = np.empty(n, dtype=np.bool_)
    for i in range(n):
        alerts[i] = bool(alerts_raw[i]) and valid[i]

    late = 0
    avoid = 0
    unavoid = 0
    for s_idx in range(len(seg_starts)):
        if not seg_has_ok[s_idx]:
            unavoid += 1
            continue
        avoid += 1
        s0 = seg_starts[s_idx]
        s1 = seg_ends[s_idx]
        hit = False
        for k in range(s0, s1):
            if alerts[k] and ok_full[k]:
                hit = True
                break
        if not hit:
            late += 1

    # FA episodes: alert cluster triggers FA iff NO frame in cluster touches haz | dyn | weak | neutral
    fa_ep = 0
    fa_fr = 0
    n_idx = 0
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        if alerts[i]:
            idx[n_idx] = i
            n_idx += 1
            if (not haz[i]) and (not dyn[i]) and (not weak[i]) and (not neutral[i]):
                fa_fr += 1

    start = 0
    while start < n_idx:
        end = start
        while end + 1 < n_idx:
            gap = 0.0
            for t in range(idx[end] + 1, idx[end + 1] + 1):
                gap += ds[t]
            if gap <= refr:
                end += 1
            else:
                break
        any_protected = False
        for u in range(start, end + 1):
            fr = idx[u]
            if haz[fr] or dyn[fr] or weak[fr] or neutral[fr]:
                any_protected = True
                break
        if not any_protected:
            fa_ep += 1
        start = end + 1

    # Repeated clear warnings strictly on strict_clear
    inside = False
    charged = False
    last_m = 0.0
    clear_warn = 0
    for i in range(n):
        if not strict_clear[i]:
            inside = False
            continue
        if not inside:
            inside = True
            charged = False
        if alerts[i] and ((not charged) or (cum_m[i] - last_m > refr)):
            clear_warn += 1
            last_m = cum_m[i]
            charged = True

    return late, avoid, unavoid, fa_ep, fa_fr, clear_warn


def summarize_counts_3zone(x: np.ndarray) -> dict:
    avoid = int(x[AVOID])
    late = int(x[LATE])
    km = float(x[KM])
    clear_n = int(x[CLEAR])
    valid_n = int(x[VALID])
    neutral_n = int(x[NEUTRAL])
    return {
        "timely": avoid - late,
        "late": late,
        "avoidable": avoid,
        "unavoidable": int(x[UNAVOID]),
        "late_pct": round(float(100.0 * late / avoid), 2) if avoid else None,
        "fa_episodes": int(x[FALSE]),
        "walker_km": round(km, 4),
        "fa_per_km": round(float(x[FALSE] / km), 3) if km > 1e-6 else None,
        "repeated_clear_warnings": int(x[CLEAR_WARN]),
        "clear_warnings_per_km": round(float(x[CLEAR_WARN] / km), 3) if km > 1e-6 else None,
        "clear_alert_frames": int(x[ON_CLEAR]),
        "known_clear_frames": clear_n,
        "clear_frame_duty": round(float(x[ON_CLEAR] / clear_n), 4) if clear_n else None,
        "neutral_alert_frames": int(x[ON_NEUTRAL]),
        "cane_neutral_frames": neutral_n,
        "cane_neutral_alert_rate": round(float(x[ON_NEUTRAL] / neutral_n), 4) if neutral_n else None,
        "alert_frames": int(x[ALERT]),
        "valid_frames": valid_n,
        "frame_duty": round(float(x[ALERT] / valid_n), 4) if valid_n else None,
    }


class FastClipEval3Zone:
    def __init__(
        self,
        domain: str,
        name: str,
        device: str,
        source: str,
        fold: int,
        rows_target: np.ndarray,
        rows_full: np.ndarray | None,
        arms: dict,
    ):
        self.domain = domain
        self.name = name
        self.device = device
        self.source = source
        self.fold = fold
        self.rows = rows_target
        self.rows_full = rows_target if rows_full is None else rows_full
        self.arms = arms
        self.episodes = make_episodes(name, rows_target, BW_SPEEDS_PROXY, "blind")
        valid = np.asarray(self.episodes[0]["valid"], dtype=np.bool_)
        self.valid = valid
        D_t = self.rows[:, 0]
        D_f = self.rows_full[:, 0]
        self.haz = valid & (D_t <= EVAL_H)
        self.dyn = self.rows[:, 2] > 0.5
        self.ds = self.rows[:, 3].astype(np.float64)
        self.weak = np.asarray(weak_mask(self.episodes[0]), dtype=np.bool_)
        self.cane_neutral = valid & (D_t > EVAL_H) & (D_f <= EVAL_H)
        self.strict_clear = valid & (D_f > EVAL_H) & (~self.dyn) & (~self.haz)
        self.clear = self.strict_clear  # alias

        # Mandatory pairwise disjointness assertion
        assert not (self.haz & self.cane_neutral).any(), f"Overlap haz & neutral in {domain}/{name}"
        assert not (self.haz & self.strict_clear).any(), f"Overlap haz & clear in {domain}/{name}"
        assert not (self.cane_neutral & self.strict_clear).any(), f"Overlap neutral & clear in {domain}/{name}"

        self.cum_m = np.cumsum(np.where(valid, self.ds, 0.0), dtype=np.float64)
        self.km = float(self.ds[valid].sum() / 1000.0)
        self.n_valid_int = int(valid.sum())
        self.n_clear_int = int(self.strict_clear.sum())
        self.n_neutral_int = int(self.cane_neutral.sum())

        starts, ends = [], []
        i, n = 0, len(D_t)
        while i < n:
            if self.haz[i]:
                j = i
                while j + 1 < n and (self.haz[j + 1] or (j + 2 < n and self.haz[j + 2] and valid[j + 1])):
                    j += 1
                if int(self.haz[i : j + 1].sum()) >= MIN_EVENT_FRAMES_V2:
                    starts.append(i)
                    ends.append(j + 1)
                i = j + 1
            else:
                i += 1
        self.seg_starts = np.asarray(starts, dtype=np.int64)
        self.seg_ends = np.asarray(ends, dtype=np.int64)
        self.ep_fast = []
        for ep in self.episodes:
            ok_full = D_t >= ep["dreq"]
            seg_has_ok = np.array([bool(ok_full[s:e].any()) for s, e in zip(starts, ends)], dtype=np.bool_)
            self.ep_fast.append((ok_full, float(ep["v"] * REFRACTORY_S), seg_has_ok))
        self.cache: dict[bytes, np.ndarray] = {}

    def count(self, alerts: np.ndarray) -> np.ndarray:
        a = np.asarray(alerts, dtype=np.bool_)
        key = np.packbits(a).tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        tot_late = tot_avoid = tot_unavoid = tot_fa_ep = tot_fa_fr = tot_cw = 0
        for ok_full, refr, seg_has_ok in self.ep_fast:
            l, av, un, fe, ff, cw = _score_ep_3zone_numba(
                a, self.valid, self.haz, self.dyn, self.weak, self.cane_neutral, ok_full,
                self.ds, refr, self.seg_starts, self.seg_ends, seg_has_ok,
                self.strict_clear, self.cum_m,
            )
            tot_late += l
            tot_avoid += av
            tot_unavoid += un
            tot_fa_ep += fe
            tot_fa_fr += ff
            tot_cw += cw

        a_val = a & self.valid
        res = np.array([
            float(tot_late),
            float(tot_avoid),
            float(tot_unavoid),
            float(tot_fa_ep),
            float(tot_fa_fr),
            6.0 * self.km,
            float(tot_cw),
            float(a_val.sum()),
            float(self.n_valid_int),
            float((a_val & self.strict_clear).sum()),
            float(self.n_clear_int),
            float((a_val & self.cane_neutral).sum()),
            float(self.n_neutral_int),
        ], dtype=np.float64)
        self.cache[key] = res
        return res


def verify_3zone_2zone_parity(clips: list[FastClipEval3Zone]) -> None:
    rng = np.random.default_rng(20261125)
    checked = 0
    for c in clips[: min(10, len(clips))]:
        assert c.n_neutral_int == 0, "Parity helper expects rows_full == rows_target (neutral=0)"
        n = len(c.rows)
        for p in np.linspace(0.02, 0.80, 20):
            mask = rng.random(n) < p
            fast = c.count(mask)
            ref_frozen = np.sum([score_episode(ep, mask) for ep in c.episodes], axis=0)
            np.testing.assert_allclose(fast[:6], ref_frozen, rtol=0, atol=1e-12)
            checked += 1
    print(f"[parity] FastClipEval3Zone (neutral=0) bit-exact against score_episode across {checked} tests!", flush=True)


# ---------------------------------------------------------------------------
# Causal Readout Engines (Part 2 Ablation Ladder)
# ---------------------------------------------------------------------------

class HeadAnchoredGeomBEV:
    """Zero-ground-dependency Head-Anchored 2.5D BEV with dh_s4 & vertical-normal filtering.

    Supports both fixed z_rel bounds and per-frame adaptive `(z_rel_lo, floor_dh_bias)`.
    """

    def __init__(
        self,
        z_rel_lo: float = -0.85,
        z_rel_hi: float = 0.25,
        occ_thr_pix_eq: float = 0.0308,
        vertical_power: float = 1.0,
        dh_scale: float = 0.25,
    ):
        self.z_rel_lo = float(z_rel_lo)
        self.z_rel_hi = float(z_rel_hi)
        self.vertical_power = float(vertical_power)
        self.dh_scale = float(dh_scale)
        self.cfg = TemporalBEVConfig(use_scale_lock=False, occ_thr_pix_eq=float(occ_thr_pix_eq))
        self.reset()

    def reset(self) -> None:
        self.buf: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._uv_hw: tuple[int, int] | None = None
        self._u_flat: np.ndarray | None = None
        self._v_flat: np.ndarray | None = None

    def _get_uv(self, H4: int, W4: int) -> tuple[np.ndarray, np.ndarray]:
        if self._uv_hw != (H4, W4):
            s = float(self.cfg.stride)
            v, u = np.mgrid[0:H4, 0:W4].astype(np.float32)
            self._u_flat = (u * s + 0.5 * s).reshape(-1)
            self._v_flat = (v * s + 0.5 * s).reshape(-1)
            self._uv_hw = (H4, W4)
        return self._u_flat, self._v_flat

    def step(
        self,
        depth_s4: np.ndarray,
        dh_s4: np.ndarray,
        nrm_s4: np.ndarray,
        log_sigma_s4: np.ndarray,
        K: np.ndarray,
        R_wc: np.ndarray,
        t_w: np.ndarray,
        causal_fwd: np.ndarray,
        z_rel_lo_override: float | None = None,
        floor_dh_bias: float = 0.0,
    ) -> tuple[float, float, float]:
        c = self.cfg
        H4, W4 = depth_s4.shape
        u_f, v_f = self._get_uv(H4, W4)
        z_lo = self.z_rel_lo if z_rel_lo_override is None else float(z_rel_lo_override)

        z = depth_s4.astype(np.float32, copy=False).reshape(-1)
        ok = np.isfinite(z) & (z > 0.25) & (z < 9.5)
        fw, rt = _horizontal_fwd_right(R_wc, causal_fwd)

        occ_thr = c.occ_thr_area_m2 if c.use_area_norm else c.occ_thr_pix_eq
        n_bins = int(round((c.fwd_max_m - c.fwd_min_m) / c.z_bin_m)) + 1
        bin_cz = c.fwd_min_m + (np.arange(n_bins, dtype=np.float32) + 0.5) * c.z_bin_m
        z_mid = c.fwd_min_m + (np.arange(n_bins - 1, dtype=np.float32) + 1.0) * c.z_bin_m
        is_blind_bin = bin_cz < c.blind_zone_m

        risk_1f = 0.0
        if int(ok.sum()) >= 50:
            fx = max(float(K[0, 0]), 1.0)
            fy = max(float(K[1, 1]), 1.0)
            cx, cy = float(K[0, 2]), float(K[1, 2])
            z_ok = z[ok]
            P_cam = np.stack([(u_f[ok] - cx) * z_ok / fx, (v_f[ok] - cy) * z_ok / fy, z_ok], axis=-1)
            P_rel = (R_wc @ P_cam.T).T
            x_lat = P_rel @ rt
            z_fwd = P_rel @ fw
            z_rel = P_rel[:, 2]

            obs_m = (z_rel >= z_lo) & (z_rel <= self.z_rel_hi) & (z_fwd >= c.fwd_min_m) & (z_fwd <= 5.0)
            if obs_m.any():
                dh_ok = dh_s4.astype(np.float32, copy=False).reshape(-1)[ok][obs_m] - float(floor_dh_bias)
                nd_ok = nrm_s4[1].astype(np.float32, copy=False).reshape(-1)[ok][obs_m]
                dh_w = np.clip(dh_ok / self.dh_scale, 0.0, 1.0)
                vert_w = np.clip(1.0 - nd_ok * nd_ok, 0.0, 1.0)
                if self.vertical_power != 1.0:
                    vert_w = np.power(vert_w, self.vertical_power)
                geom_w = dh_w * vert_w

                keep_g = geom_w > 0.01
                if keep_g.any():
                    dA = np.full(int(keep_g.sum()), 0.004, dtype=np.float32)
                    ls_ok = log_sigma_s4.astype(np.float32, copy=False).reshape(-1)[ok][obs_m][keep_g]
                    conf = np.clip(np.exp(-np.clip(ls_ok, -3.0, 2.0)), 0.25, 2.5)
                    wt = (dA * conf * geom_w[keep_g]).astype(np.float32)
                    P_keep = P_rel[obs_m][keep_g]
                    pw = (P_keep + t_w[None, :]).astype(np.float32)
                    self.buf.append((pw, wt, t_w.astype(np.float32), fw.astype(np.float32)))

                    cm = (np.abs(x_lat[obs_m][keep_g]) <= c.x_half_m) & (z_fwd[obs_m][keep_g] <= c.fwd_max_m)
                    if cm.any():
                        b1 = np.zeros(n_bins, dtype=np.float32)
                        bi1 = np.clip(((z_fwd[obs_m][keep_g][cm] - c.fwd_min_m) / c.z_bin_m).astype(np.int32), 0, n_bins - 1)
                        np.add.at(b1, bi1, wt[cm])
                        s1 = b1[:-1] + b1[1:]
                        p1 = 1.0 - np.exp(-s1 / occ_thr)
                        risk_1f = float(np.max(p1 / (z_mid + 0.15)) + 0.06 * np.sum(np.log1p(s1 / occ_thr) / ((z_mid + 0.4) ** 2)))
                else:
                    self.buf.append((np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), t_w.astype(np.float32), fw.astype(np.float32)))
            else:
                self.buf.append((np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), t_w.astype(np.float32), fw.astype(np.float32)))
        else:
            self.buf.append((np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), t_w.astype(np.float32), fw.astype(np.float32)))

        if len(self.buf) > c.window_frames:
            self.buf.pop(0)

        bin_pts_w = t_w[None, :] + bin_cz[:, None] * fw[None, :]
        occ_num = np.zeros(n_bins, dtype=np.float32)
        vis_den = np.zeros(n_bins, dtype=np.float32)
        frm_cnt = np.zeros(n_bins, dtype=np.int32)
        k_len = len(self.buf)

        for age_idx, (pw, wt, t_old, fw_old) in enumerate(self.buf):
            age = (k_len - 1) - age_idx
            w_vis = float(c.decay_vis ** age)
            w_bld = float(c.decay_blind ** age)
            w_bin = np.where(is_blind_bin, w_bld, w_vis).astype(np.float32)

            z_at_old = (bin_pts_w - t_old[None, :]) @ fw_old
            in_vis = (z_at_old >= c.blind_zone_m) & (z_at_old <= 4.8)
            vis_den += w_bin * in_vis.astype(np.float32)

            if len(pw) == 0:
                continue
            rel = pw - t_w[None, :]
            xl = rel @ rt
            zf = rel @ fw
            m = (np.abs(xl) <= c.x_half_m) & (zf >= c.fwd_min_m) & (zf <= c.fwd_max_m)
            if not m.any():
                continue
            b_idx = np.clip(((zf[m] - c.fwd_min_m) / c.z_bin_m).astype(np.int32), 0, n_bins - 1)
            tmp = np.zeros(n_bins, dtype=np.float32)
            np.add.at(tmp, b_idx, wt[m])
            occ_num += tmp * w_bin
            frm_cnt += (tmp >= 0.25 * occ_thr).astype(np.int32)

        occ = occ_num / np.maximum(vis_den, 0.75)
        s2 = occ[:-1] + occ[1:]
        fc2 = np.maximum(frm_cnt[:-1], frm_cnt[1:])
        confirmed = (s2 >= occ_thr) & ((fc2 >= c.min_confirm_frames) | (s2 >= 2.6 * occ_thr))
        hits = np.flatnonzero(confirmed)
        pred_d = float(c.fwd_min_m + hits[0] * c.z_bin_m) if len(hits) else float(np.inf)

        conf_mult = np.where(fc2 >= c.min_confirm_frames, 1.0, 0.55)
        s2_eff = s2 * conf_mult
        p_occ = 1.0 - np.exp(-s2_eff / occ_thr)
        risk_bev = float(np.max(p_occ / (z_mid + 0.15)) + 0.06 * np.sum(np.log1p(s2_eff / occ_thr) / ((z_mid + 0.4) ** 2)))
        risk_hyb = c.hybrid_bev_weight * risk_bev + (1.0 - c.hybrid_bev_weight) * risk_1f
        return pred_d, risk_bev, float(risk_hyb)


class M6HeadAnchoredUpgraded:
    """Unified strictly causal M6 Head-Anchored engine covering dhcal, adapt, and adapt_v2."""

    def __init__(
        self,
        dh_cal: bool = False,
        adaptive_band: bool = False,
        vertical_power: float = 1.0,
        dh_scale: float = 0.25,
        band_ramp_m: float = 0.30,
        z_rel_lo_fixed: float = -0.85,
        z_rel_hi: float = 0.25,
    ):
        self.dh_cal = bool(dh_cal)
        self.adaptive_band = bool(adaptive_band)
        self.vertical_power = float(vertical_power)
        self.dh_scale = float(dh_scale)
        self.band_ramp_m = float(band_ramp_m)
        self.z_rel_lo_fixed = float(z_rel_lo_fixed)
        self.z_rel_hi = float(z_rel_hi)
        self.reset()

    def reset(self) -> None:
        self._grid: tuple[np.ndarray, np.ndarray] | None = None
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self.h_ema: float = 1.60
        self.floor_dh_ema: float = 0.0

    def step(
        self,
        depth_s4: np.ndarray,
        dh_s4: np.ndarray,
        nrm_s4: np.ndarray,
        ls_s4: np.ndarray,
        K: np.ndarray,
        R_wc: np.ndarray,
        t_w: np.ndarray,
        causal_fwd: np.ndarray,
    ) -> tuple[float, float, float]:
        """Returns (risk, z_rel_lo_used, floor_dh_ema_used)."""
        sub = np.s_[::2, ::2]
        z = depth_s4[sub].astype(np.float32, copy=False).ravel()
        if self._grid is None or self._grid[0].shape != z.shape:
            v, u = np.indices(depth_s4[sub].shape, dtype=np.float32)
            self._grid = ((8.0 * u + 0.5).ravel(), (8.0 * v + 0.5).ravel())
        u, v = self._grid

        fx = max(float(K[0, 0]), 1.0)
        fy = max(float(K[1, 1]), 1.0)
        cx, cy = float(K[0, 2]), float(K[1, 2])
        xyz = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1) @ R_wc.T
        valid = np.isfinite(xyz).all(axis=1) & (z > 0.0)
        if not valid.any():
            self._prev = None
            return 0.0, self.z_rel_lo_fixed, self.floor_dh_ema

        z_rel = xyz[:, 2]
        dh = dh_s4[sub].astype(np.float32, copy=False).ravel()
        n_down = nrm_s4[1][sub].astype(np.float32, copy=False).ravel()
        ls = ls_s4[sub].astype(np.float32, copy=False).ravel()

        floor = valid & (z_rel < -0.45) & (np.abs(n_down) >= 0.70)
        n_floor = int(floor.sum())
        if self.adaptive_band and n_floor >= 20:
            h_obs = float(np.clip(-np.percentile(z_rel[floor], 15), 1.10, 1.90))
            self.h_ema = 0.85 * self.h_ema + 0.15 * h_obs
        if (self.dh_cal or self.adaptive_band) and n_floor >= 15:
            obs_dh = float(np.clip(np.median(dh[floor]), -0.15, 0.20))
            self.floor_dh_ema = float(np.clip(0.75 * self.floor_dh_ema + 0.25 * obs_dh, -0.15, 0.20))

        if self.adaptive_band:
            pitch_deg = float(np.degrees(np.arcsin(np.clip(R_wc[2, 2], -1.0, 1.0))))
            z_lo_base = max(-0.85, -(self.h_ema - 0.65))
            pitch_guard = min(0.20, max(0.0, (-15.0 - pitch_deg) * 0.015))
            z_rel_lo = min(-0.35, z_lo_base + pitch_guard)
            scale_ref = self.h_ema
        else:
            z_rel_lo = self.z_rel_lo_fixed
            scale_ref = 1.50

        dh_eff = dh - self.floor_dh_ema if (self.dh_cal or self.adaptive_band) else dh
        ls_med = float(np.median(ls[valid]))
        conf = np.exp(-np.clip(ls - ls_med, -2.0, 2.0))
        band_ramp = np.clip((z_rel - z_rel_lo) / self.band_ramp_m, 0.0, 1.0)
        dh_ramp = np.clip(dh_eff / self.dh_scale, 0.0, 1.0)
        vertical = np.clip(1.0 - n_down * n_down, 0.0, 1.0)
        if self.vertical_power != 1.0:
            vertical = np.power(vertical, self.vertical_power)
        weight = band_ramp * dh_ramp * vertical * conf

        keep = valid & (z_rel >= z_rel_lo) & (z_rel <= self.z_rel_hi) & (weight > 0.0)
        world = xyz[keep, :2] + t_w[:2]
        w_keep = weight[keep]

        fw = np.asarray(causal_fwd[:2], dtype=np.float64)
        nf = float(np.linalg.norm(fw))
        if nf < 1e-7:
            self._prev = None
            return 0.0, z_rel_lo, self.floor_dh_ema
        fw = fw / nf
        rt = np.array([fw[1], -fw[0]], dtype=np.float64)

        n_b = 14
        fwd_lo_h, fwd_hi_h, lat_half_h = 0.20, 3.00, 0.22
        bin_w = (fwd_hi_h - fwd_lo_h) / float(n_b)

        def _bins(pts: np.ndarray, ev: np.ndarray) -> np.ndarray:
            if len(pts) == 0:
                return np.zeros(n_b - 1, dtype=np.float64)
            offset = (pts - t_w[:2]) / scale_ref
            forward = offset @ fw
            lateral = offset @ rt
            mask = (np.abs(lateral) <= lat_half_h) & (forward >= fwd_lo_h) & (forward < fwd_hi_h)
            if not mask.any():
                return np.zeros(n_b - 1, dtype=np.float64)
            idx = ((forward[mask] - fwd_lo_h) / bin_w).astype(np.int32).clip(0, n_b - 1)
            cnt = np.bincount(idx, weights=ev[mask], minlength=n_b)
            return cnt[:-1] + cnt[1:]

        now = _bins(world, w_keep)
        past = np.zeros_like(now) if self._prev is None else _bins(*self._prev)
        self._prev = (world, w_keep)
        support = 0.35 * now + 0.65 * np.minimum(now, past)
        proximity = 1.0 / (1.0 + (fwd_lo_h + bin_w + bin_w * np.arange(len(support), dtype=np.float64)) ** 2)
        risk = float(np.max(-np.expm1(-support / 10.0) * proximity))
        return risk, z_rel_lo, self.floor_dh_ema


def run_ablation_arms_on_clip(
    dep_s4: np.ndarray,
    dh_s4: np.ndarray,
    nrm_s4: np.ndarray,
    ls_s4: np.ndarray,
    Ks: np.ndarray,
    poses_R: np.ndarray,
    poses_t: np.ndarray,
    strict_fwd: np.ndarray,
    timing_acc: dict[str, list[float]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    T = len(dep_s4)

    # Round-1 reference arms (5)
    bev_full = EgoGeomTemporalBEV(TemporalBEVConfig(hgt_lo_m=0.15, hgt_hi_m=1.90, use_scale_lock=True))
    bev_midup = EgoGeomTemporalBEV(TemporalBEVConfig(hgt_lo_m=0.65, hgt_hi_m=1.90, use_scale_lock=True))
    m6_full = EgoGeomTemporalBEVV3(TemporalBEVV3Config(geom_rel_h_lo=0.06, geom_rel_h_hi=1.30))
    m6_midup = EgoGeomTemporalBEVV3(TemporalBEVV3Config(geom_rel_h_lo=0.40, geom_rel_h_hi=1.30))
    m6_ha85_pg = M6HeadAnchoredRisk(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=10.0)

    # 7 New Ablation Arms
    bev_midup_cal = EgoGeomTemporalBEV(TemporalBEVConfig(
        hgt_lo_m=0.65, hgt_hi_m=1.90, use_scale_lock=True,
        occ_thr_pix_eq=0.0308, occ_thr_area_m2=0.0198,
    ))
    bev_ha85_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    bev_ha70_g = HeadAnchoredGeomBEV(z_rel_lo=-0.70, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    bev_ha_adapt_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=1.0, dh_scale=0.25)
    bev_ha_adapt2_g = HeadAnchoredGeomBEV(z_rel_lo=-0.85, z_rel_hi=0.25, occ_thr_pix_eq=0.0308, vertical_power=2.0, dh_scale=0.35)

    m6_ha85_nopg = M6HeadAnchoredRisk(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=180.0)
    m6_ha70_nopg = M6HeadAnchoredRisk(z_rel_lo=-0.70, z_rel_hi=0.25, pitch_gate_deg=180.0)
    m6_ha85_dhcal = M6HeadAnchoredUpgraded(dh_cal=True, adaptive_band=False, vertical_power=1.0, dh_scale=0.25, band_ramp_m=0.30)
    m6_ha_adapt = M6HeadAnchoredUpgraded(dh_cal=True, adaptive_band=True, vertical_power=1.0, dh_scale=0.25, band_ramp_m=0.30)
    m6_ha_adapt_v2 = M6HeadAnchoredUpgraded(dh_cal=True, adaptive_band=True, vertical_power=2.0, dh_scale=0.35, band_ramp_m=0.25)

    # Note: for legacy dist of m6_head_anchored_85_pg on full_ground_centered parity, import HeadAnchoredBEV
    from run_cane_blind_colab import HeadAnchoredBEV
    ha85_pg_legacy = HeadAnchoredBEV(z_rel_lo=-0.85, z_rel_hi=0.25, pitch_gate_deg=10.0)

    out = {
        arm: {"dist": np.full(T, np.inf, dtype=np.float32), "raw": np.zeros(T, dtype=np.float64)}
        for arm in PART2_ARMS
    }

    do_time = timing_acc is not None
    for i in range(T):
        args_bev = (dep_s4[i], dh_s4[i], nrm_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])

        # Reference arms
        d, _, rh = bev_full.step(*args_bev)
        out["bev_full_ground"]["dist"][i], out["bev_full_ground"]["raw"][i] = d, rh

        d, _, rh = bev_midup.step(*args_bev)
        out["bev_midup_ground"]["dist"][i], out["bev_midup_ground"]["raw"][i] = d, rh

        d_f, _, _, rg_f = m6_full.step(*args_bev)
        out["m6_full_ground"]["dist"][i], out["m6_full_ground"]["raw"][i] = d_f, rg_f

        d_m, _, _, rg_m = m6_midup.step(*args_bev)
        out["m6_midup_ground"]["dist"][i], out["m6_midup_ground"]["raw"][i] = d_m, rg_m

        d_hapg, _, _ = ha85_pg_legacy.step(dep_s4[i], ls_s4[i], Ks[i], poses_R[i], poses_t[i], strict_fwd[i])
        out["m6_head_anchored_85_pg"]["dist"][i] = d_hapg
        out["m6_head_anchored_85_pg"]["raw"][i] = m6_ha85_pg.step(*args_bev)

        # 1. bev_midup_cal_thr
        t0 = time.perf_counter() if do_time else 0.0
        d, _, rh = bev_midup_cal.step(*args_bev)
        if do_time:
            timing_acc["bev_midup_cal_thr"].append((time.perf_counter() - t0) * 1e3)
        out["bev_midup_cal_thr"]["dist"][i], out["bev_midup_cal_thr"]["raw"][i] = d, rh

        # 2. bev_ha85_geom
        t0 = time.perf_counter() if do_time else 0.0
        d_85g, _, rh_85g = bev_ha85_g.step(*args_bev)
        if do_time:
            timing_acc["bev_ha85_geom"].append((time.perf_counter() - t0) * 1e3)
        out["bev_ha85_geom"]["dist"][i], out["bev_ha85_geom"]["raw"][i] = d_85g, rh_85g

        # 3. m6_ha85_nopg
        t0 = time.perf_counter() if do_time else 0.0
        rg_85n = m6_ha85_nopg.step(*args_bev)
        if do_time:
            timing_acc["m6_ha85_nopg"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha85_nopg"]["dist"][i] = d_85g
        out["m6_ha85_nopg"]["raw"][i] = rg_85n

        # 4. m6_ha70_nopg
        d_70g, _, _ = bev_ha70_g.step(*args_bev)
        t0 = time.perf_counter() if do_time else 0.0
        rg_70n = m6_ha70_nopg.step(*args_bev)
        if do_time:
            timing_acc["m6_ha70_nopg"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha70_nopg"]["dist"][i] = d_70g
        out["m6_ha70_nopg"]["raw"][i] = rg_70n

        # 5. m6_ha85_dhcal
        t0 = time.perf_counter() if do_time else 0.0
        rg_dhcal, _, _ = m6_ha85_dhcal.step(*args_bev)
        if do_time:
            timing_acc["m6_ha85_dhcal"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha85_dhcal"]["dist"][i] = d_85g
        out["m6_ha85_dhcal"]["raw"][i] = rg_dhcal

        # 6. m6_ha_adapt
        t0 = time.perf_counter() if do_time else 0.0
        rg_ad, zlo_ad, fdh_ad = m6_ha_adapt.step(*args_bev)
        d_ad, _, _ = bev_ha_adapt_g.step(*args_bev, z_rel_lo_override=zlo_ad, floor_dh_bias=fdh_ad)
        if do_time:
            timing_acc["m6_ha_adapt"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha_adapt"]["dist"][i] = d_ad
        out["m6_ha_adapt"]["raw"][i] = rg_ad

        # 7. m6_ha_adapt_v2
        t0 = time.perf_counter() if do_time else 0.0
        rg_ad2, zlo_ad2, fdh_ad2 = m6_ha_adapt_v2.step(*args_bev)
        d_ad2, _, _ = bev_ha_adapt2_g.step(*args_bev, z_rel_lo_override=zlo_ad2, floor_dh_bias=fdh_ad2)
        if do_time:
            timing_acc["m6_ha_adapt_v2"].append((time.perf_counter() - t0) * 1e3)
        out["m6_ha_adapt_v2"]["dist"][i] = d_ad2
        out["m6_ha_adapt_v2"]["raw"][i] = rg_ad2

    for arm in PART2_ARMS:
        out[arm]["confirmed"] = confirm_score(out[arm]["raw"])
    return out


# ---------------------------------------------------------------------------
# Metrics, Frontier & Cross-Fit Helpers
# ---------------------------------------------------------------------------

def frame_metrics_3zone(clip_objs: list[FastClipEval3Zone], arm: str, hazard_thr_m: float) -> dict:
    yt_h, ys_r_all, ys_c_all = [], [], []
    yt_3z, ys_r_3z, ys_c_3z = [], [], []
    err_fin, err_fix6 = [], []
    pos4_pred_fin = pos4_tot = 0

    for c in clip_objs:
        D_t = c.rows[:, 0]
        val = c.valid
        if not val.any():
            continue
        D_pr = c.arms[arm]["dist"]
        sc_r = c.arms[arm]["raw"]
        sc_c = c.arms[arm]["confirmed"]

        # Standard 2-zone frame arrays
        yt_h.append((D_t[val] <= hazard_thr_m).astype(np.int32))
        ys_r_all.append(sc_r[val])
        ys_c_all.append(sc_c[val])

        # 3-zone frame arrays: pos = (val & D_t <= hazard_thr_m), neg = strict_clear
        pos_m = val & (D_t <= hazard_thr_m)
        neg_m = c.strict_clear
        keep3 = pos_m | neg_m
        if keep3.any():
            yt_3z.append(pos_m[keep3].astype(np.int32))
            ys_r_3z.append(sc_r[keep3])
            ys_c_3z.append(sc_c[keep3])

        pos4 = val & np.isfinite(D_t) & (D_t >= 0.30) & (D_t <= 4.0)
        if pos4.any():
            pos4_tot += int(pos4.sum())
            fin = pos4 & np.isfinite(D_pr) & (D_pr <= 6.0)
            pos4_pred_fin += int(fin.sum())
            if fin.any():
                err_fin.append(np.abs(D_pr[fin] - D_t[fin]))
            d_fill = np.where(fin, D_pr, 6.0)
            err_fix6.append(np.abs(d_fill[pos4] - D_t[pos4]))

    y2 = np.concatenate(yt_h)
    r2 = np.concatenate(ys_r_all)
    c2 = np.concatenate(ys_c_all)
    y3 = np.concatenate(yt_3z) if yt_3z else np.zeros(0, dtype=np.int32)
    r3 = np.concatenate(ys_r_3z) if ys_r_3z else np.zeros(0, dtype=np.float64)
    c3 = np.concatenate(ys_c_3z) if ys_c_3z else np.zeros(0, dtype=np.float64)

    out = {
        "n_valid": int(len(y2)),
        "n_pos_eval_thr": int(y2.sum()),
        "pred_finite_rate_on_pos": round(float(pos4_pred_fin / pos4_tot), 4) if pos4_tot else None,
        "mae_finite_m": round(float(np.concatenate(err_fin).mean()), 3) if err_fin else None,
        "mae_fixed6m_m": round(float(np.concatenate(err_fix6).mean()), 3) if err_fix6 else None,
    }
    for tag, sc in (("raw", r2), ("confirmed", c2)):
        if 0 < y2.sum() < len(y2):
            out[f"auroc_{tag}_2z"] = round(float(roc_auc_score(y2, sc)), 4)
            out[f"ap_{tag}_2z"] = round(float(average_precision_score(y2, sc)), 4)
        else:
            out[f"auroc_{tag}_2z"] = None
            out[f"ap_{tag}_2z"] = None
    for tag, sc in (("raw", r3), ("confirmed", c3)):
        if len(y3) > 0 and 0 < y3.sum() < len(y3):
            out[f"auroc_{tag}_3z"] = round(float(roc_auc_score(y3, sc)), 4)
            out[f"ap_{tag}_3z"] = round(float(average_precision_score(y3, sc)), 4)
        else:
            out[f"auroc_{tag}_3z"] = None
            out[f"ap_{tag}_3z"] = None
    return out


def exact_frontier_3z(clip_objs: list[FastClipEval3Zone], arm: str, field: str = "confirmed") -> dict:
    events: dict[float, np.ndarray] = {}
    base = sum((c.count(np.zeros(len(c.rows), dtype=bool)) for c in clip_objs), np.zeros(13, dtype=np.float64))
    for c in clip_objs:
        vals = c.arms[arm][field]
        prev = c.count(np.zeros(len(vals), dtype=bool))
        for cut in np.unique(vals)[::-1]:
            nxt = c.count(vals >= cut)
            diff = nxt - prev
            if np.any(diff != 0):
                events[float(cut)] = events.get(float(cut), np.zeros(13, dtype=np.float64)) + diff
            prev = nxt

    rows = []
    cur = base.copy()

    def _pt(thr: float | None, arr: np.ndarray) -> dict:
        s = summarize_counts_3zone(arr)
        s["threshold"] = thr
        return s

    rows.append(_pt(None, cur))
    for cut in sorted(events, reverse=True):
        cur = cur + events[cut]
        rows.append(_pt(cut, cur))

    km = float(base[KM])
    avoid = int(base[AVOID])
    budget = FA_BUDGET * km
    ok_clear = [r for r in rows if r["repeated_clear_warnings"] <= budget + 1e-9]
    ok_fa = [r for r in rows if r["fa_episodes"] <= budget + 1e-9]
    best_clear = max(ok_clear, key=lambda r: (r["timely"], -r["repeated_clear_warnings"]))
    best_fa = max(ok_fa, key=lambda r: (r["timely"], -r["fa_episodes"]))
    return {
        "avoidable": avoid,
        "walker_km": round(km, 4),
        "known_clear_frames": int(base[CLEAR]),
        "cane_neutral_frames": int(base[NEUTRAL]),
        "best_at_5_clear_km": best_clear,
        "best_at_5_orig_fa_km": best_fa,
    }


def fit_threshold_3z(
    clip_objs: list[FastClipEval3Zone],
    arm: str,
    field: str = "confirmed",
    robust: bool = True,
    q_grid: tuple[float, ...] | np.ndarray = Q_GRID,
):
    thresholds = [-np.inf, np.inf]
    if clip_objs:
        train_scores = np.concatenate([c.arms[arm][field] for c in clip_objs])
        thresholds += list(np.unique(np.quantile(train_scores, q_grid, method="higher")))
    candidates = []
    for thr in thresholds:
        counts = sum((c.count(c.arms[arm][field] >= thr) for c in clip_objs), np.zeros(13, dtype=np.float64))
        if counts[FALSE] > FA_BUDGET * counts[KM]:
            continue
        if robust and counts[CLEAR_WARN] > FA_BUDGET * counts[KM]:
            continue
        candidates.append(((counts[LATE], counts[CLEAR_WARN], counts[ON_CLEAR], counts[ALERT], -thr), thr, counts))
    _, threshold, best = min(candidates, key=lambda row: row[0])
    return threshold, summarize_counts_3zone(best)


def run_2fold_crossfit_3z(clip_objs: list[FastClipEval3Zone], arm: str, field: str = "confirmed") -> dict:
    shared_rows, dev_rows, shared_fine_rows, dev_fine_rows = [], [], [], []
    for f in (0, 1):
        train = [c for c in clip_objs if c.fold != f]
        test = [c for c in clip_objs if c.fold == f]
        if not train or not test:
            continue
        assert {c.source for c in train}.isdisjoint({c.source for c in test})
        t_shared, _ = fit_threshold_3z(train, arm, field=field, robust=True, q_grid=Q_GRID)
        t_shared_f, _ = fit_threshold_3z(train, arm, field=field, robust=True, q_grid=FINE_Q_GRID)
        devs = sorted({c.device for c in clip_objs})
        t_dev, t_dev_f = {}, {}
        for d in devs:
            tr_d = [c for c in train if c.device == d]
            t_dev[d] = fit_threshold_3z(tr_d, arm, field=field, robust=True, q_grid=Q_GRID)[0] if tr_d else np.inf
            t_dev_f[d] = fit_threshold_3z(tr_d, arm, field=field, robust=True, q_grid=FINE_Q_GRID)[0] if tr_d else np.inf
        for c in test:
            vals = c.arms[arm][field]
            shared_rows.append((c.domain, c.count(vals >= t_shared)))
            dev_rows.append((c.domain, c.count(vals >= t_dev[c.device])))
            shared_fine_rows.append((c.domain, c.count(vals >= t_shared_f)))
            dev_fine_rows.append((c.domain, c.count(vals >= t_dev_f[c.device])))

    def _agg(items):
        pooled = sum((x for _, x in items), np.zeros(13, dtype=np.float64))
        doms = sorted({d for d, _ in items})
        return {
            "pooled": summarize_counts_3zone(pooled),
            "domains": {d: summarize_counts_3zone(sum((x for dd, x in items if dd == d), np.zeros(13, dtype=np.float64))) for d in doms},
        }

    return {
        "shared_crossfit": _agg(shared_rows),
        "device_crossfit": _agg(dev_rows),
        "shared_crossfit_fine": _agg(shared_fine_rows),
        "device_crossfit_fine": _agg(dev_fine_rows),
    }


def fixed_dist_diag_3z(clip_objs: list[FastClipEval3Zone], arm: str) -> dict:
    c2 = sum((c.count(c.arms[arm]["dist"] <= 2.0) for c in clip_objs), np.zeros(13, dtype=np.float64))
    c4 = sum((c.count(c.arms[arm]["dist"] <= 4.0) for c in clip_objs), np.zeros(13, dtype=np.float64))
    return {"dist_le_2m": summarize_counts_3zone(c2), "dist_le_4m": summarize_counts_3zone(c4)}


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    aria_gt_bundle = np.load(PROBE_DIR / "aria_height_band_gt.npz", allow_pickle=False)
    sanpo_bundle = np.load(PROBE_DIR / "sanpo_real_zs40_1d_bundle.npz", allow_pickle=False)
    sanpo_meta = json.loads((PROBE_DIR / "sanpo_real_zs40_meta.json").read_text(encoding="utf-8"))
    r1_report = json.loads((PROBE_DIR / "cane_blind_report.json").read_text(encoding="utf-8"))

    # Also run Round-1 8 arms via run_all_readouts_on_clip? Better: Part 2 computes all 12 PART2_ARMS,
    # and Part 1 needs the 3 remaining BEV head-anchored arms on the 4 local domains too.
    # Running run_all_readouts_on_clip + run_ablation_arms_on_clip takes ~25s total; we can merge them cleanly.
    from run_cane_blind_colab import run_all_readouts_on_clip

    suite_clips: dict[str, list[dict]] = {}
    timing_acc: dict[str, list[float]] = {k: [] for k in NEW_ARMS}

    local_suites = [
        ("aria_gen2", "aria_g2_20clips"),
        ("arkit_val30_day", "arkit_val30_shard00"),
        ("arkit_val30_dark018", "arkit_val30_shard00"),
        ("coda_campus", "coda_zeroshot_shard000"),
    ]

    for sname, folder in local_suites:
        ts = time.time()
        src_paths = sorted((ZS_DIR / folder).rglob("*.npz"))
        clips_list = []
        for idx_c, p in enumerate(src_paths):
            with np.load(p, allow_pickle=False) as d_src:
                eval_rows = d_src["eval_rows"].astype(np.float64)
                gt_bev_static = d_src["gt_bev_static"] if "gt_bev_static" in d_src else None
                cam_name = str(d_src["cam"]) if "cam" in d_src else ("rover_flir" if sname == "coda_campus" else "unknown")

            with np.load(STREAM_DIR / sname / p.name, allow_pickle=False) as d_obs:
                dep_s4 = d_obs["depth_s4"]
                dh_s4 = d_obs["dh_s4"]
                nrm_s4 = d_obs["nrm_s4"]
                ls_s4 = d_obs["ls_s4"]
                Ks_arr = d_obs["K"]
                poses_R = d_obs["poses_R"]
                poses_t = d_obs["poses_t"]
                strict_fwd = d_obs["strict_fwd"]

            arms_r1 = run_all_readouts_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, strict_fwd)
            use_timer = timing_acc if (sname == "aria_gen2" and idx_c < 5) else None
            arms_p2 = run_ablation_arms_on_clip(dep_s4, dh_s4, nrm_s4, ls_s4, Ks_arr, poses_R, poses_t, strict_fwd, use_timer)

            # Merge: keep all R1_ALL_ARMS + NEW_ARMS
            arms_merged = dict(arms_r1)
            for k in NEW_ARMS:
                arms_merged[k] = arms_p2[k]
            # Verify Part 2 ref arms bit-exact vs R1
            for rk in REF_ARMS_PART2:
                np.testing.assert_allclose(arms_p2[rk]["raw"], arms_r1[rk]["raw"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(arms_p2[rk]["dist"], arms_r1[rk]["dist"], rtol=0, atol=1e-6)

            gt_bands: dict[str, np.ndarray] = {}
            if sname == "aria_gen2":
                for st in ("centered_full", "strict_past"):
                    for b in ("full_ground", "midup_ground", "head_anchored_85"):
                        gt_bands[f"{st}__{b}"] = aria_gt_bundle[f"{p.stem}__{st}__{b}"].astype(np.float64)
            else:
                gt_bands["orig_full"] = eval_rows
                gt_bands["bev_full"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (0, 1, 2))
                gt_bands["bev_midup"] = build_bev_band_eval_rows(eval_rows, gt_bev_static, (1, 2))

            clips_list.append({
                "domain": sname,
                "name": p.stem,
                "cam": cam_name,
                "arms": arms_merged,
                "gt_bands": gt_bands,
            })

        assign_source_folds(clips_list)
        suite_clips[sname] = clips_list
        print(f"[load] {sname}: {len(clips_list)} clips ready in {time.time() - ts:.2f}s", flush=True)

    # Load SANPO-Real ZS40 1D bundle (8 Round-1 arms)
    sanpo_clips = []
    for cmeta in sanpo_meta["clips"]:
        stem = cmeta["name"]
        arms = {
            arm: {field: sanpo_bundle[f"{stem}__arm__{arm}__{field}"] for field in ("dist", "raw", "confirmed")}
            for arm in R1_ALL_ARMS
        }
        gt_bands = {}
        for gk in ("orig_full", "bev_full", "bev_midup"):
            key = f"{stem}__gt__{gk}"
            if key in sanpo_bundle:
                gt_bands[gk] = sanpo_bundle[key].astype(np.float64)
        sanpo_clips.append({
            "domain": "sanpo_real_zs40",
            "name": stem,
            "cam": cmeta["cam"],
            "arms": arms,
            "gt_bands": gt_bands,
        })
    assign_source_folds(sanpo_clips)
    suite_clips["sanpo_real_zs40"] = sanpo_clips
    print(f"[load] sanpo_real_zs40: {len(sanpo_clips)} clips from Colab 1D bundle", flush=True)

    def make_objs_3z(
        suite_list: list[str],
        target_map: dict[str, str],
        full_map: dict[str, str] | None = None,
        cam_filter: str | None = None,
    ) -> list[FastClipEval3Zone]:
        objs = []
        for s in suite_list:
            tk = target_map[s]
            fk = tk if full_map is None else full_map[s]
            for c in suite_clips[s]:
                if cam_filter is not None and c["cam"] != cam_filter:
                    continue
                if tk not in c["gt_bands"] or fk not in c["gt_bands"]:
                    continue
                rf = None if fk == tk else c["gt_bands"][fk]
                objs.append(FastClipEval3Zone(c["domain"], c["name"], c["device"], c["source"], c["fold"], c["gt_bands"][tk], rf, c["arms"]))
        return objs

    # Bit-exact parity check when rows_full == rows_target (neutral=0)
    verify_3zone_2zone_parity(
        make_objs_3z(["aria_gen2"], {"aria_gen2": "centered_full__full_ground"})[:3]
        + make_objs_3z(["arkit_val30_day"], {"arkit_val30_day": "orig_full"})[:3]
        + make_objs_3z(["sanpo_real_zs40"], {"sanpo_real_zs40": "orig_full"})[:4]
    )

    def eval_cohort_3z(objs: list[FastClipEval3Zone], h_thr: float, arms_list: tuple[str, ...], with_crossfit: bool = False) -> dict:
        res = {}
        for arm in arms_list:
            entry = {
                "frame": frame_metrics_3zone(objs, arm, h_thr),
                "frontier_confirmed": exact_frontier_3z(objs, arm, "confirmed"),
                "fixed_dist": fixed_dist_diag_3z(objs, arm),
            }
            if with_crossfit:
                entry["crossfit_2fold"] = run_2fold_crossfit_3z(objs, arm, "confirmed")
            res[arm] = entry
        return res

    # -----------------------------------------------------------------------
    # Part 1: 5-Domain 2-Zone vs 3-Zone Cane-Neutral Comparison (8 R1 Arms)
    # -----------------------------------------------------------------------
    print("\n[Part 1] Running 5-domain 2-zone vs 3-zone Cane-Neutral evaluation...", flush=True)
    part1_per_domain = {}

    p1_specs = [
        ("aria_gen2", "centered_full__midup_ground", "centered_full__full_ground", 4.0, None),
        ("aria_gen2", "centered_full__head_anchored_85", "centered_full__full_ground", 4.0, None),
        ("aria_gen2", "strict_past__midup_ground", "strict_past__full_ground", 4.0, None),
        ("arkit_val30_day", "bev_midup", "bev_full", 2.0, None),
        ("arkit_val30_dark018", "bev_midup", "bev_full", 2.0, None),
        ("coda_campus", "bev_midup", "bev_full", 4.0, None),
        ("sanpo_real_zs40", "bev_midup", "bev_full", 4.0, None),
        ("sanpo_real_zs40", "bev_midup", "bev_full", 4.0, "camera_head"),
        ("sanpo_real_zs40", "bev_midup", "bev_full", 4.0, "camera_chest"),
    ]
    for sname, tk, fk, h_thr, cam_f in p1_specs:
        key = f"{sname}::{tk}" if cam_f is None else f"{sname}::{cam_f}__{tk}"
        do_cf = (sname == "sanpo_real_zs40" and cam_f is None)
        objs_2z = make_objs_3z([sname], {sname: tk}, None, cam_filter=cam_f)
        objs_3z = make_objs_3z([sname], {sname: tk}, {sname: fk}, cam_filter=cam_f)
        part1_per_domain[key] = {
            "2zone": eval_cohort_3z(objs_2z, h_thr, R1_ALL_ARMS, with_crossfit=do_cf),
            "3zone": eval_cohort_3z(objs_3z, h_thr, R1_ALL_ARMS, with_crossfit=do_cf),
        }

    # 4-domain 38-source pooled (2-zone baseline check + 3-zone configs)
    doms4 = ["aria_gen2", "arkit_val30_day", "arkit_val30_dark018", "coda_campus"]
    p1_4dom_specs = {
        "midup_2zone_matched": (
            {"aria_gen2": "centered_full__midup_ground", "arkit_val30_day": "bev_midup", "arkit_val30_dark018": "bev_midup", "coda_campus": "bev_midup"},
            None,
        ),
        "midup_3zone_centered": (
            {"aria_gen2": "centered_full__midup_ground", "arkit_val30_day": "bev_midup", "arkit_val30_dark018": "bev_midup", "coda_campus": "bev_midup"},
            {"aria_gen2": "centered_full__full_ground", "arkit_val30_day": "bev_full", "arkit_val30_dark018": "bev_full", "coda_campus": "bev_full"},
        ),
        "midup_3zone_strict_past_aria": (
            {"aria_gen2": "strict_past__midup_ground", "arkit_val30_day": "bev_midup", "arkit_val30_dark018": "bev_midup", "coda_campus": "bev_midup"},
            {"aria_gen2": "strict_past__full_ground", "arkit_val30_day": "bev_full", "arkit_val30_dark018": "bev_full", "coda_campus": "bev_full"},
        ),
        "ha85_3zone_centered": (
            {"aria_gen2": "centered_full__head_anchored_85", "arkit_val30_day": "bev_midup", "arkit_val30_dark018": "bev_midup", "coda_campus": "bev_midup"},
            {"aria_gen2": "centered_full__full_ground", "arkit_val30_day": "bev_full", "arkit_val30_dark018": "bev_full", "coda_campus": "bev_full"},
        ),
        "midup_3zone_origfull_clear": (
            {"aria_gen2": "centered_full__midup_ground", "arkit_val30_day": "bev_midup", "arkit_val30_dark018": "bev_midup", "coda_campus": "bev_midup"},
            {"aria_gen2": "centered_full__full_ground", "arkit_val30_day": "orig_full", "arkit_val30_dark018": "orig_full", "coda_campus": "orig_full"},
        ),
    }
    part1_4dom = {}
    for cfg_name, (tmap, fmap) in p1_4dom_specs.items():
        objs4 = make_objs_3z(doms4, tmap, fmap)
        part1_4dom[cfg_name] = {
            arm: {
                "frontier_confirmed": exact_frontier_3z(objs4, arm, "confirmed"),
                "crossfit_2fold": run_2fold_crossfit_3z(objs4, arm, "confirmed"),
                "fixed_dist": fixed_dist_diag_3z(objs4, arm),
            }
            for arm in R1_ALL_ARMS
        }

    # Verify exact Round-1 2-zone parity on midup_ground_matched
    for arm in R1_ALL_ARMS:
        r1_p5 = r1_report["four_domain_38source"]["midup_ground_matched"][arm]["frontier_confirmed"]["best_at_5_clear_km"]["timely"]
        now_p5 = part1_4dom["midup_2zone_matched"][arm]["frontier_confirmed"]["best_at_5_clear_km"]["timely"]
        assert r1_p5 == now_p5, f"Round-1 midup_ground_matched parity mismatch for {arm}: {r1_p5} vs {now_p5}"
    print("[parity] Part 1 midup_2zone_matched matches Round-1 cane_blind_report.json across all 8 arms!", flush=True)

    # -----------------------------------------------------------------------
    # Part 2: 7 New Ablation Arms + 5 Reference Arms on 4 Domains (103 Clips)
    # -----------------------------------------------------------------------
    print("\n[Part 2] Running 7-arm readout ablation ladder on 4 domains (38 sources)...", flush=True)
    part2_4dom_specs = {
        "full_ground_centered": (
            {"aria_gen2": "centered_full__full_ground", "arkit_val30_day": "orig_full", "arkit_val30_dark018": "orig_full", "coda_campus": "orig_full"},
            None,
        ),
        "midup_3zone_centered": p1_4dom_specs["midup_3zone_centered"],
        "midup_3zone_strict_past_aria": p1_4dom_specs["midup_3zone_strict_past_aria"],
        "ha85_3zone_centered": p1_4dom_specs["ha85_3zone_centered"],
    }
    part2_4dom = {}
    for cfg_name, (tmap, fmap) in part2_4dom_specs.items():
        objs4 = make_objs_3z(doms4, tmap, fmap)
        part2_4dom[cfg_name] = {
            arm: {
                "frontier_confirmed": exact_frontier_3z(objs4, arm, "confirmed"),
                "crossfit_2fold": run_2fold_crossfit_3z(objs4, arm, "confirmed"),
                "fixed_dist": fixed_dist_diag_3z(objs4, arm),
            }
            for arm in PART2_ARMS
        }

    # Verify Round-1 exact parity on full_ground_centered / m6_head_anchored_85_pg
    r1_ha = r1_report["four_domain_38source"]["full_ground_centered"]["m6_head_anchored_85_pg"]
    p2_ha = part2_4dom["full_ground_centered"]["m6_head_anchored_85_pg"]
    assert p2_ha["frontier_confirmed"]["best_at_5_clear_km"]["timely"] == r1_ha["frontier_confirmed"]["best_at_5_clear_km"]["timely"] == 51
    assert p2_ha["crossfit_2fold"]["shared_crossfit"]["pooled"]["timely"] == r1_ha["crossfit_2fold"]["shared_crossfit"]["pooled"]["timely"] == 45
    assert p2_ha["crossfit_2fold"]["device_crossfit"]["pooled"]["timely"] == r1_ha["crossfit_2fold"]["device_crossfit"]["pooled"]["timely"] == 117
    print("[parity] Part 2 full_ground_centered/m6_head_anchored_85_pg exact match (51/365, CF 45/365, DCF 117/365)!", flush=True)

    # Part 2 per-domain metrics (midup 3-zone per domain + full_ground on Aria centered)
    part2_per_domain = {}
    p2_dom_specs = [
        ("aria_gen2", "centered_full__midup_ground", "centered_full__full_ground", 4.0),
        ("aria_gen2", "strict_past__midup_ground", "strict_past__full_ground", 4.0),
        ("arkit_val30_day", "bev_midup", "bev_full", 2.0),
        ("arkit_val30_dark018", "bev_midup", "bev_full", 2.0),
        ("coda_campus", "bev_midup", "bev_full", 4.0),
        ("aria_gen2", "centered_full__full_ground", "centered_full__full_ground", 4.0),
    ]
    for sname, tk, fk, h_thr in p2_dom_specs:
        objs_d = make_objs_3z([sname], {sname: tk}, None if fk == tk else {sname: fk})
        part2_per_domain[f"{sname}::{tk}"] = eval_cohort_3z(objs_d, h_thr, PART2_ARMS, with_crossfit=False)

    latency_ms = {
        arm: {
            "n_frames": len(vals),
            "mean_ms": round(float(np.mean(vals)), 3),
            "p95_ms": round(float(np.percentile(vals, 95)), 3),
        }
        for arm, vals in timing_acc.items()
    }

    report = {
        "meta": {
            "total_elapsed_s": round(time.time() - t0, 2),
            "r1_arms": list(R1_ALL_ARMS),
            "part2_arms": list(PART2_ARMS),
            "latency_aria_5clips_ms": latency_ms,
        },
        "part1_3zone_per_domain": part1_per_domain,
        "part1_3zone_4dom_38source": part1_4dom,
        "part2_ablation_4dom_38source": part2_4dom,
        "part2_ablation_per_domain": part2_per_domain,
    }

    # Check zero NaN/Inf
    def _scan(obj, p="root"):
        if isinstance(obj, float):
            if not np.isfinite(obj):
                raise RuntimeError(f"Non-finite value at {p}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{p}.{k}")
        elif isinstance(obj, list):
            for idx_i, v in enumerate(obj):
                _scan(v, f"{p}[{idx_i}]")

    _scan(report)
    out_path = PROBE_DIR / "cane_neutral_ablation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Saved {out_path} ({out_path.stat().st_size / 1e3:.1f} KB) in {report['meta']['total_elapsed_s']}s; 0 NaN/Inf!", flush=True)


if __name__ == "__main__":
    main()
