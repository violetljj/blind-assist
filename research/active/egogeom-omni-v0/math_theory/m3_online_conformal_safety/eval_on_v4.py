"""Four-domain delayed-audit replay on frozen v4 per-clip predictions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from statistics import NormalDist

import numpy as np
from controller import (
    HORIZON_M,
    OUT,
    RiskConformalController,
    lower_distance,
    standardized,
)

ROUTE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROUTE))
sys.path.insert(0, str(ROUTE.parent / "egoclear-v0"))
from eval_closed_loop import make_episodes, score_episode
from eval_cross_glasses_zeroshot import BW_SPEEDS_PROXY

SUITES = {
    "arkit_val30_day": ("arkit_val30_shard00", 30),
    "arkit_val30_dark018": ("arkit_val30_shard00", 30),
    "aria_gen2": ("aria_g2_20clips", 19),
    "coda_campus": ("coda_zeroshot_shard000", 24),
}
ALPHA = .05
AUDIT = .7
DELAY = 5
ARMS = ("raw", "gaussian", "fixed_aci", "sa_dr")
GRID = np.round(np.linspace(1.4, 4.4, 31), 1)


def load_stream(cache, originals):
    clips = []
    for domain, (folder, count) in SUITES.items():
        paths = sorted((originals / folder).rglob("*.npz"))
        if len(paths) != count:
            raise RuntimeError(f"{domain}: {len(paths)} of {count} source clips")
        for p in paths:
            with np.load(cache / domain / p.name, allow_pickle=False) as stream:
                raw = np.asarray(stream["d_1f_strict"], dtype=float)
                ls = np.median(stream["ls_s4"], axis=(1, 2))
            if np.any(np.isnan(raw)) or np.any(raw < 0) or not np.all(np.isfinite(ls)):
                raise ValueError(f"nonphysical prediction or missing uncertainty in {p}")
            pred = np.minimum(raw, HORIZON_M)
            # v4 trains log_sigma on absolute log-depth residuals. First-order
            # metre conversion is a declared frame-level proxy for the corridor.
            sigma = np.maximum(.05, pred * np.exp(np.clip(ls, -3.0, 2.0)))
            with np.load(p, allow_pickle=False) as src:
                rows = np.array(src["eval_rows"], copy=True)
            if len(rows) != len(raw):
                raise ValueError(f"length mismatch: {p}")
            clips.append({"domain": domain, "name": p.stem, "rows": rows, "raw": raw, "pred": pred, "sigma": sigma})
    return clips


def replay(clips):
    rng = np.random.default_rng(133714)
    control = RiskConformalController(sum(len(c["raw"]) for c in clips), ALPHA, DELAY, AUDIT)
    pending = deque()
    issued = observed = 0
    fixed_q = 0.0

    def disclose(item):
        nonlocal fixed_q, observed
        old, i, token, old_fixed, audit = item
        row = old["rows"][i]
        valid = bool(row[0] >= 0 and row[1] >= .6)
        truth = row[0] if valid and audit else None
        control.reveal(token, valid, truth)
        if truth is not None:
            observed += 1
            error = standardized(token.distance_m, token.sigma_m, truth) > old_fixed
            fixed_q += .05 * (int(error) - ALPHA)

    for c in clips:
        n = len(c["raw"])
        for key in ("fixed_aci", "sa_dr"):
            c[key] = np.empty(n)
            c[key + "_empty"] = np.zeros(n, dtype=bool)
        # Linearized Gaussian baseline with this same scale. The Gaussian law
        # for the corridor residual is a comparator assumption.
        c["gaussian"] = np.maximum(0., c["pred"] - NormalDist().inv_cdf(1 - ALPHA) * c["sigma"])
        for i in range(n):
            token = control.predict(c["pred"][i], c["sigma"][i], propensity=AUDIT)
            c["sa_dr_empty"][i] = token.lower_m > HORIZON_M
            c["sa_dr"][i] = token.action_lower_m
            fixed_lower = lower_distance(token.distance_m, token.sigma_m, fixed_q)
            c["fixed_aci_empty"][i] = fixed_lower > HORIZON_M
            c["fixed_aci"][i] = 0.0 if fixed_lower > HORIZON_M else fixed_lower
            pending.append((c, i, token, fixed_q, rng.random() < AUDIT))
            if len(pending) > DELAY:
                disclose(pending.popleft())
            issued += 1
    while pending:
        disclose(pending.popleft())
    return {"issued": issued, "valid": control.resolved, "random_audits": observed,
            "anytime_95pct_realized_error_upper": control.audit_upper(.05),
            "dr_error_estimate": control.proxy_sum / control.resolved}


def summarize_counts(counts, thresholds):
    out = []
    for thr, (late, avoid, _, fa, _, km) in zip(thresholds, counts):
        out.append({"threshold_m": float(thr), "late": int(late), "avoidable": int(avoid),
                        "false_alarm_episodes": int(fa), "walker_km": float(km),
                        "late_pct": float(100 * late / avoid) if avoid else None,
                        "fa_per_km": float(fa / km) if km else None})
    return out


def metrics(clips):
    curves = {s: np.zeros((len(ARMS), len(GRID), 6)) for s in SUITES}
    ranges = {}
    for domain in SUITES:
        domain_clips = [c for c in clips if c["domain"] == domain]
        rows = np.concatenate([c["rows"] for c in domain_clips])
        truth = np.minimum(rows[:, 0], HORIZON_M)
        valid = (truth >= 0) & (rows[:, 1] >= .6)
        detail = {"clips": len(domain_clips), "frames": len(truth), "valid": int(valid.sum()),
                  "positive_inf_truth_on_valid": int(np.isposinf(rows[valid, 0]).sum()),
                  "no_corridor_detection_on_valid": int(np.isinf(np.concatenate([c["raw"] for c in domain_clips])[valid]).sum())}
        valid_first20 = valid[:20]
        for arm in ARMS:
            # Raw infinity is a no-detection symbol; consistently cap the
            # *range diagnostic* comparator at 12m, with raw alerts unchanged.
            pred = np.concatenate([c["pred" if arm == "raw" else arm] for c in domain_clips])
            e = truth < pred
            detail[arm] = {"violations": int(e[valid].sum()), "violation_rate": float(e[valid].mean()),
                           "first_20_frames_valid": int(valid_first20.sum()),
                           "first_20_frames_violations": int(e[:20][valid_first20].sum()),
                           "alert_rate_at_2m": float((pred[valid] <= 2.).mean())}
            if arm in ("sa_dr", "fixed_aci"):
                empty = np.concatenate([c[arm + "_empty"] for c in domain_clips])
                detail[arm]["empty_nominal_sets_on_valid"] = int(empty[valid].sum())
                detail[arm]["nominal_violations"] = int((e | empty)[valid].sum())
        ranges[domain] = detail
        for c in domain_clips:
            episodes = make_episodes(c["name"], c["rows"], BW_SPEEDS_PROXY, "blind")
            result_cache = {}
            for a, arm in enumerate(ARMS):
                for j, threshold in enumerate(GRID):
                    alerts = c[arm] <= threshold
                    key = np.packbits(alerts).tobytes()
                    if key not in result_cache:
                        result_cache[key] = np.sum([score_episode(ep, alerts) for ep in episodes], axis=0)
                    curves[domain][a, j] += result_cache[key]
    curves["pooled"] = sum(curves.values())
    report = {}
    for s, values in curves.items():
        report[s] = {}
        for a, arm in enumerate(ARMS):
            points = summarize_counts(values[a], GRID)
            feasible = [p for p in points if p["late_pct"] is not None and p["fa_per_km"] <= 5.]
            best = min(feasible, key=lambda p: (p["late_pct"], p["fa_per_km"])) if feasible else None
            report[s][arm] = {"fixed_2m": next(p for p in points if p["threshold_m"] == 2.),
                              "best_at_fa5_grid_oracle": best, "curve": points}
    return {"range_by_domain": ranges, "episode_metrics": report}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=OUT.parent / "v4_stream")
    p.add_argument("--originals", type=Path, default=OUT.parent.parent / "zeroshot_eval")
    args = p.parse_args()
    if not (args.cache / "manifest.json").is_file():
        raise RuntimeError(f"Frozen v4 stream incomplete: {args.cache}")
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    clips = load_stream(args.cache, args.originals)
    result = {"design": {"horizon_m": HORIZON_M, "alpha": ALPHA, "known_random_audit_probability": AUDIT,
                         "feedback_delay_frames": DELAY, "order": list(SUITES), "range": "capped at 12m; infinity means beyond horizon",
                         "uncertainty": "spatial median of frozen log_sigma times capped depth after exponentiation; metre proxy",
                         "operating_point": "retrospective grid for comparison plus fixed 2m"},
              "stream_identity": {k: manifest[k] for k in ("checkpoint_sha256", "protocol_sha256", "batch")},
              "feedback": replay(clips), **metrics(clips)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v4_empirical_report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["feedback"]))
    for s, arms in result["episode_metrics"].items():
        print(s, {a: None if m["best_at_fa5_grid_oracle"] is None else m["best_at_fa5_grid_oracle"]["late_pct"] for a, m in arms.items()})


if __name__ == "__main__":
    main()
