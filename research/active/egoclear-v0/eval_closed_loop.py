"""Closed-loop blind-walker evaluator with exact walk-alert-r0 v2 parity.

Evaluates any per-frame alert boolean stream (or continuous hazard score / predicted
corridor distance) against the exact SANPO-Synthetic v2 fused static GT truth:
- BlindWays walker speeds (6 walkers per session, RT ~ U(1.16, 2.0) s, a = 2.0 m/s^2)
- Avoidable events (>= 2 hazard frames, >= 8 GT voxels, centered truth heading)
- Metrics: Late/Miss Rate (%), FA episodes / km (2 s refractory), FA frames / km,
  plus matched-FA Late Rate and full operating Pareto curve.
"""
from __future__ import annotations

import hashlib
import numpy as np

H = 4.0
DECEL = 2.0
BODY_MARGIN = 0.30
REFRACTORY_S = 2.0
RT_RANGE = (1.16, 2.0)
K_WALKERS = 6
MIN_EVENT_FRAMES_V2 = 2
SIGHTED = (1.35, 0.12, 1.0, 1.7)


def walkers(kind: str, seed: int, k: int, bw_speeds: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "sighted":
        m, s, lo, hi = SIGHTED
        v = np.clip(rng.normal(m, s, k), lo, hi)
    else:
        keys = sorted(bw_speeds)
        v = np.array([bw_speeds[p][rng.integers(len(bw_speeds[p]))] for p in rng.choice(keys, k)])
    rt = rng.uniform(*RT_RANGE, k)
    return v, rt


def make_episodes(scene: str, eval_rows: np.ndarray, bw_speeds: dict, kind: str = "blind") -> list[dict]:
    """eval_rows: (T, 4) float64 [D_gt, cover, dyn_flag, seg_len]."""
    seed = int(hashlib.sha256(f"{kind}:{scene}".encode()).hexdigest()[:8], 16)
    v, rt = walkers(kind, seed, K_WALKERS, bw_speeds)
    D = eval_rows[:, 0]
    valid = (np.isfinite(D) | np.isinf(D)) & ~np.isnan(eval_rows[:, 1]) & (eval_rows[:, 1] >= 0.6)
    out = []
    for j in range(K_WALKERS):
        dreq = float(v[j] * rt[j] + v[j] ** 2 / (2.0 * DECEL) + BODY_MARGIN)
        out.append({"scene": scene, "v": float(v[j]), "rt": float(rt[j]), "dreq": dreq, "rows": eval_rows, "valid": valid})
    return out


def weak_mask(ep: dict) -> np.ndarray:
    r = ep["rows"]
    valid = ep["valid"]
    haz = valid & (r[:, 0] <= H)
    w = np.zeros(len(haz), dtype=bool)
    i, n = 0, len(haz)
    while i < n:
        if haz[i]:
            j = i
            while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                j += 1
            if haz[i : j + 1].sum() < MIN_EVENT_FRAMES_V2:
                w[i : j + 1] = haz[i : j + 1]
            i = j + 1
        else:
            i += 1
    return w


def score_episode(ep: dict, alerts: np.ndarray) -> tuple[int, int, int, int, int, float]:
    """Return (late, avoidable, unavoidable, fa_episodes, fa_frames, km)."""
    r = ep["rows"]
    D, dyn, ds = r[:, 0], r[:, 2] > 0.5, r[:, 3]
    valid = ep["valid"]
    haz = valid & (D <= H)
    alerts = np.asarray(alerts, dtype=bool) & valid
    late = avoid = unavoid = 0
    i, n = 0, len(D)
    while i < n:
        if haz[i]:
            j = i
            while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                j += 1
            seg = slice(i, j + 1)
            ok = D[seg] >= ep["dreq"]
            if haz[seg].sum() < MIN_EVENT_FRAMES_V2:
                i = j + 1
                continue
            if ok.any():
                avoid += 1
                late += int(not (alerts[seg] & ok).any())
            else:
                unavoid += 1
            i = j + 1
        else:
            i += 1

    weak = weak_mask(ep)
    # FA episodes (2 s refractory)
    fa_ep = 0
    idx = np.nonzero(alerts)[0]
    refr = ep["v"] * REFRACTORY_S
    start = 0
    while start < len(idx):
        end = start
        while end + 1 < len(idx) and ds[idx[end] + 1 : idx[end + 1] + 1].sum() <= refr:
            end += 1
        fr = idx[start : end + 1]
        if not haz[fr].any() and not dyn[fr].any() and not weak[fr].any():
            fa_ep += 1
        start = end + 1

    # FA frames (per-frame non-hazard, non-dynamic, non-weak alerts)
    fa_fr = int((alerts & ~haz & ~dyn & ~weak).sum())
    km = float(ds[valid].sum() / 1000.0)
    return late, avoid, unavoid, fa_ep, fa_fr, km


def apply_causal_hysteresis(
    scores: np.ndarray,
    thr_on: float,
    mode: str = "ema_hyst",
    ema_alpha: float = 0.55,
    off_ratio: float = 0.72,
    fast_bypass_delta: float = 0.25,
) -> np.ndarray:
    """Strictly causal (zero-lookahead) event-level temporal hysteresis & confirmation state machine.

    Modes:
      - 'raw': simple threshold scores >= thr_on
      - 'ema_hyst': causal EMA smoothing + Schmitt trigger dual-threshold (thr_on, off_ratio * thr_on)
      - 'confirm_2of3': requires >=2 of last 3 frames >= thr_on to enter alert (with high-confidence
        immediate bypass >= min(0.98, thr_on + fast_bypass_delta) for sudden close hazards), then
        holds alert until EMA falls below off_ratio * thr_on.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if mode == "raw" or n == 0:
        return s >= thr_on

    thr_off = thr_on * off_ratio
    bypass_thr = min(0.98, thr_on + fast_bypass_delta)
    alerts = np.zeros(n, dtype=bool)
    ema = float(s[0]) if np.isfinite(s[0]) else 0.0
    active = False

    if mode == "ema_hyst":
        for i in range(n):
            val = float(s[i]) if np.isfinite(s[i]) else 0.0
            ema = val if i == 0 else (ema_alpha * val + (1.0 - ema_alpha) * ema)
            if not active:
                if ema >= thr_on or val >= bypass_thr:
                    active = True
            else:
                if ema < thr_off:
                    active = False
            alerts[i] = active
        return alerts

    if mode == "confirm_2of3":
        hits = np.zeros(n, dtype=np.int32)
        for i in range(n):
            val = float(s[i]) if np.isfinite(s[i]) else 0.0
            ema = val if i == 0 else (ema_alpha * val + (1.0 - ema_alpha) * ema)
            hits[i] = 1 if val >= thr_on else 0
            recent_hits = int(hits[max(0, i - 2) : i + 1].sum())
            if not active:
                if recent_hits >= 2 or val >= bypass_thr:
                    active = True
            else:
                if ema < thr_off and val < thr_on:
                    active = False
            alerts[i] = active
        return alerts

    if mode == "slope_aware_hyst":
        # Zero-delay trigger on rising obstacle ramps (prev >= 0.55*thr_on), while filtering isolated 1-frame glitches
        # from clear baseline (prev < 0.55*thr_on and val < bypass_thr); Schmitt hysteresis on release.
        prev_val = 0.0
        for i in range(n):
            val = float(s[i]) if np.isfinite(s[i]) else 0.0
            ema = val if i == 0 else (ema_alpha * val + (1.0 - ema_alpha) * ema)
            if not active:
                if val >= thr_on and (prev_val >= 0.55 * thr_on or val >= bypass_thr):
                    active = True
            else:
                if val < thr_off and ema < thr_off:
                    active = False
            alerts[i] = active
            prev_val = val
        return alerts

    raise ValueError(f"Unknown hysteresis mode: {mode}")


def eval_score_curve(
    sessions_eval_rows: dict[str, np.ndarray],
    per_scene_scores: dict[str, np.ndarray],
    bw_speeds: dict,
    thresholds: np.ndarray,
    higher_is_alert: bool = True,
    kind: str = "blind",
    walker_aware_fn=None,
    hysteresis_mode: str = "raw",
    ema_alpha: float = 0.55,
    off_ratio: float = 0.72,
) -> list[dict]:
    """Sweep thresholds and return closed-loop Pareto curve.

    If walker_aware_fn is provided: walker_aware_fn(ep, thr) -> bool array (T,).
    Otherwise compares per_scene_scores[scene] >= thr (or <= thr if higher_is_alert=False),
    optionally wrapped in a strictly causal event-level hysteresis state machine.
    """
    eps_all = []
    for sc, rows in sessions_eval_rows.items():
        if len(rows) >= 5:
            eps_all.extend(make_episodes(sc, rows, bw_speeds, kind=kind))

    curve = []
    for thr in thresholds:
        tot_late = tot_avoid = tot_unavoid = tot_fa_ep = tot_fa_fr = 0
        tot_km = 0.0
        for ep in eps_all:
            if walker_aware_fn is not None:
                al = walker_aware_fn(ep, float(thr))
            else:
                sc_arr = per_scene_scores[ep["scene"]]
                if higher_is_alert and hysteresis_mode != "raw":
                    al = apply_causal_hysteresis(
                        sc_arr, float(thr), mode=hysteresis_mode, ema_alpha=ema_alpha, off_ratio=off_ratio
                    )
                else:
                    al = (sc_arr >= thr) if higher_is_alert else (sc_arr <= thr)
            l, a, u, fe, ff, km = score_episode(ep, al)
            tot_late += l
            tot_avoid += a
            tot_unavoid += u
            tot_fa_ep += fe
            tot_fa_fr += ff
            tot_km += km
        curve.append(
            {
                "thr": float(thr),
                "late": int(tot_late),
                "avoidable": int(tot_avoid),
                "unavoidable": int(tot_unavoid),
                "late_rate": (tot_late / tot_avoid) if tot_avoid else None,
                "fa_episodes": int(tot_fa_ep),
                "fa_ep_per_km": (tot_fa_ep / tot_km) if tot_km > 0 else None,
                "fa_frames": int(tot_fa_fr),
                "fa_fr_per_km": (tot_fa_fr / tot_km) if tot_km > 0 else None,
                "km": round(tot_km, 4),
            }
        )
    return curve


def summarize_at_fa_caps(
    curve: list[dict],
    fa_ep_caps=(10.0, 15.0, 22.6, 30.0, 40.0, 60.0),
    fa_fr_caps=(40.0, 80.0, 120.3, 200.0, 350.0, 500.0),
) -> dict:
    """Best achievable late_rate under matched FA episode/km and FA frame/km caps."""
    out = {"best_floor": min(curve, key=lambda x: (x["late_rate"] if x["late_rate"] is not None else 1.0, x["fa_ep_per_km"] or 1e9))}
    for cap in fa_ep_caps:
        ok = [p for p in curve if p["fa_ep_per_km"] is not None and p["fa_ep_per_km"] <= cap + 1e-9]
        out[f"late_at_fa_ep<={cap}"] = min(ok, key=lambda x: (x["late_rate"], x["fa_ep_per_km"])) if ok else None
    for cap in fa_fr_caps:
        ok = [p for p in curve if p["fa_fr_per_km"] is not None and p["fa_fr_per_km"] <= cap + 1e-9]
        out[f"late_at_fa_fr<={cap}"] = min(ok, key=lambda x: (x["late_rate"], x["fa_fr_per_km"])) if ok else None
    return out
