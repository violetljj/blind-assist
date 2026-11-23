"""Group-held-out operating points with the frozen walker scorer and a clear-road cost."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from controller import HORIZON_M, OUT, standardized
from eval_on_v4 import (
    BW_SPEEDS_PROXY,
    SUITES,
    load_stream,
    make_episodes,
    score_episode,
)
from numba import njit

Q_GRID = (.5, .65, .8, .9, .95, .975, .99, .995, .999)
FA_BUDGET = 5.0
ARMS = (
    "always", "never", "bev_distance_2m", "fixed_range_95_2m",
    "raw_hybrid", "confirmed_hybrid", "confirmed_cost", "device_cost", "device_only_cost",
)
# Six-walker episode counts plus frame counts without walker replication.
LATE, AVOID, UNAVOID, FALSE, FFRAME, KM, CLEAR_WARN, ALERT, VALID, ON_CLEAR, CLEAR = range(11)


def source_id(domain, name):
    if domain.startswith("arkit"):
        return "arkit", name
    device = "aria" if domain.startswith("aria") else "coda"
    group = re.fullmatch(r"(.+)_c\d+", name)
    if group is None:
        raise ValueError(f"Unknown source grouping for {name}")
    return device, group.group(1)


def source_folds(clips):
    devices = defaultdict(set)
    for clip in clips:
        dev, src = source_id(clip["domain"], clip["name"])
        devices[dev].add(src)
    folds = {}
    for dev, names in devices.items():
        ordered = sorted(names, key=lambda s: hashlib.sha256(("M3-alert-cost-v2/" + s).encode()).digest())
        for rank, name in enumerate(ordered):
            folds[(dev, name)] = rank % 2
    if {d: len(n) for d, n in devices.items()} != {"arkit": 30, "aria": 2, "coda": 6}:
        raise AssertionError("Unexpected original source count")
    return folds


def confirm_score(score):
    # Two threshold crossings among the latest three; clip boundaries are hard resets.
    padded = np.pad(score, (2, 0), constant_values=0)
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)


@njit
def repeated_clear_warnings(alerts, clear, cumulative_m, refractory_m):
    inside = False
    charged = False
    last = 0.0
    warnings = 0
    for i in range(len(alerts)):
        if not clear[i]:
            inside = False
            continue
        if not inside:
            inside = True
            charged = False
        if alerts[i] and (not charged or cumulative_m[i] - last > refractory_m):
            warnings += 1
            last = cumulative_m[i]
            charged = True
    return warnings


@dataclass
class Clip:
    obs: dict
    device: str
    source: str
    fold: int
    raw: np.ndarray
    confirmed: np.ndarray
    bev_d: np.ndarray
    episodes: list
    clear: np.ndarray
    position: np.ndarray
    counts_cache: dict = field(default_factory=dict)

    def count(self, alerts):
        key = np.packbits(alerts).tobytes()
        if key not in self.counts_cache:
            rows = [score_episode(ep, alerts) for ep in self.episodes]
            frozen = np.sum(rows, axis=0)
            repeated = sum(repeated_clear_warnings(alerts, self.clear, self.position, 2.0 * ep["v"])
                           for ep in self.episodes)
            valid = self.episodes[0]["valid"]
            counts = np.r_[frozen, repeated, (alerts & valid).sum(), valid.sum(),
                           (alerts & self.clear).sum(), self.clear.sum()]
            if counts[CLEAR_WARN] < counts[FALSE]:
                raise AssertionError("Augmented count below original on a pure clear cluster")
            self.counts_cache[key] = counts
        return self.counts_cache[key]


def read_clips():
    cache = OUT.parent / "v4_stream"
    obs = load_stream(cache, OUT.parent.parent / "zeroshot_eval")
    folds = source_folds(obs)
    clips = []
    for c in obs:
        with np.load(cache / c["domain"] / (c["name"] + ".npz"), allow_pickle=False) as pack:
            raw = pack["r_hyb_strict"]
            distance = pack["d_bev_strict"]
        if np.any(~np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("Nonfinite or negative BEV risk")
        dev, src = source_id(c["domain"], c["name"])
        rows = c["rows"]
        episodes = make_episodes(c["name"], rows, BW_SPEEDS_PROXY, "blind")
        clear = episodes[0]["valid"] & (rows[:, 0] > 4.0) & (rows[:, 2] <= .5)
        valid = episodes[0]["valid"]
        if not np.all(np.isfinite(rows[valid, 3])) or np.any(rows[valid, 3] < 0):
            raise ValueError("Invalid length on a scored frame")
        # Native missing rows may have NaN path increments. Invalid rows cut
        # the diagnostic clear segment; only within-segment differences count.
        ds = np.where(valid, rows[:, 3], 0.0)
        clips.append(Clip(c, dev, src, folds[(dev, src)], raw, confirm_score(raw), distance,
                          episodes, clear, np.cumsum(ds, dtype=np.float64)))
    return clips


def summarize(x):
    return {
        "late": int(x[LATE]), "avoidable": int(x[AVOID]), "unavoidable": int(x[UNAVOID]),
        "late_pct": float(100 * x[LATE] / x[AVOID]) if x[AVOID] else None,
        "fa_episodes": int(x[FALSE]), "walker_km": float(x[KM]), "fa_per_km": float(x[FALSE] / x[KM]),
        "repeated_clear_warnings": int(x[CLEAR_WARN]), "clear_warnings_per_km": float(x[CLEAR_WARN] / x[KM]),
        "clear_alert_frames": int(x[ON_CLEAR]), "known_clear_frames": int(x[CLEAR]),
        "clear_frame_duty": float(x[ON_CLEAR] / x[CLEAR]) if x[CLEAR] else None,
        "alert_frames": int(x[ALERT]), "valid_frames": int(x[VALID]), "frame_duty": float(x[ALERT] / x[VALID]),
    }


def threshold_record(t):
    if np.isposinf(t):
        return "never"
    if np.isneginf(t):
        return "always"
    return float(t)


def fit_threshold(clips, score_key, robust, device_only=False):
    thresholds = [-np.inf, np.inf]
    if not device_only:
        train_scores = np.concatenate([getattr(c, score_key) for c in clips])
        thresholds += list(np.unique(np.quantile(train_scores, Q_GRID, method="higher")))
    candidates = []
    for thr in thresholds:
        counts = sum(c.count(getattr(c, score_key) >= thr) for c in clips)
        if counts[FALSE] > FA_BUDGET * counts[KM]:
            continue
        if robust and counts[CLEAR_WARN] > FA_BUDGET * counts[KM]:
            continue
        candidates.append(((counts[LATE], counts[CLEAR_WARN], counts[ON_CLEAR], counts[ALERT], -thr), thr, counts))
    _, threshold, best = min(candidates, key=lambda row: row[0])
    return threshold, {"threshold": threshold_record(threshold), "training": summarize(best)}


def fit_range(clips):
    samples = []
    for c in clips:
        valid = c.episodes[0]["valid"]
        samples.append(standardized(c.obs["pred"][valid], c.obs["sigma"][valid], c.obs["rows"][valid, 0]))
    scores = np.concatenate(samples)
    k = math.ceil(.95 * (len(scores) + 1))
    return np.sort(scores)[k - 1] if k <= len(scores) else 1.0


def ranges(c, q):
    pred, sigma = c.obs["pred"], c.obs["sigma"]
    nominal = np.maximum(0.0, pred - sigma * q / (1.0 - abs(q))) if abs(q) < 1.0 else (
        np.zeros_like(pred) if q >= 1.0 else np.full_like(pred, np.inf))
    action = np.where(nominal <= HORIZON_M, nominal, 0.0)
    valid = c.episodes[0]["valid"]
    truth = np.minimum(c.obs["rows"][:, 0], HORIZON_M)
    counts = np.array([(truth[valid] < nominal[valid]).sum(), (truth[valid] < action[valid]).sum(),
                       (nominal[valid] > HORIZON_M).sum(), valid.sum()], dtype=int)
    return action, counts


def diagnostic():
    n = 19
    rows = np.column_stack([np.r_[np.full(8, 10.), np.full(3, 3.5), np.full(8, 10.)],
                            np.ones(n), np.zeros(n), np.full(n, .5)])
    ep = {"v": 1.0, "dreq": 2.0, "rows": rows, "valid": np.ones(n, dtype=bool)}
    clear = rows[:, 0] > 4
    out = {}
    for name, alerts in {"always": np.ones(n, dtype=bool), "never": np.zeros(n, dtype=bool),
                         "only_two_clear_endpoints": np.isin(np.arange(n), [0, n-1])}.items():
        late, avoid, _, fa, ff, _ = score_episode(ep, alerts)
        out[name] = {"late": late, "avoidable": avoid, "fa": fa, "false_frames": ff,
                     "clear_warnings": repeated_clear_warnings(alerts, clear, np.cumsum(rows[:, 3]), 2.0)}
    assert (out["always"]["fa"], out["always"]["false_frames"], out["always"]["clear_warnings"]) == (0, 16, 4)
    assert out["only_two_clear_endpoints"]["fa"] == 2
    return out


def aggregate_groups(source_rows):
    domains = list(SUITES)
    metrics = {}
    for dom in [*domains, "pooled"]:
        rows = [r for r in source_rows if r["domain"] == dom or dom == "pooled"]
        metrics[dom] = {arm: summarize(sum(r["counts"][arm] for r in rows)) for arm in ARMS}
        cover = np.sum([r["range"] for r in rows], axis=0)
        metrics[dom]["range"] = {
            "nominal_violations": int(cover[0]), "action_violations": int(cover[1]),
            "empty_sets": int(cover[2]), "valid": int(cover[3]),
            "nominal_coverage": float(1-cover[0]/cover[3]), "action_coverage": float(1-cover[1]/cover[3])}
    return metrics


def main():
    clips = read_clips()
    fitted = []
    holdouts = []
    for f in (0, 1):
        train = [c for c in clips if c.fold != f]
        test = [c for c in clips if c.fold == f]
        assert {c.source for c in train}.isdisjoint(c.source for c in test)
        fits = {}
        thresholds = {}
        for arm, attr, robust in [("raw_hybrid", "raw", False), ("confirmed_hybrid", "confirmed", False),
                                  ("confirmed_cost", "confirmed", True)]:
            thresholds[arm], fits[arm] = fit_threshold(train, attr, robust)
        for arm in ("device_cost", "device_only_cost"):
            thresholds[arm], fits[arm] = {}, {}
            for dev in ("arkit", "aria", "coda"):
                t, rec = fit_threshold([c for c in train if c.device == dev], "confirmed", True,
                                       device_only=(arm == "device_only_cost"))
                thresholds[arm][dev], fits[arm][dev] = t, rec
        q = fit_range(train)
        fitted.append({"test_fold": f, "train_sources": sorted({c.source for c in train}),
                       "test_sources": sorted({c.source for c in test}), "threshold_fits": fits,
                       "training_only_range_q": float(q)})
        for c in test:
            lower, coverage_counts = ranges(c, q)
            alarms = {"always": np.ones(len(c.raw), dtype=bool), "never": np.zeros(len(c.raw), dtype=bool),
                      "bev_distance_2m": c.bev_d <= 2., "fixed_range_95_2m": lower <= 2.,
                      "raw_hybrid": c.raw >= thresholds["raw_hybrid"]}
            for arm in ("confirmed_hybrid", "confirmed_cost"):
                alarms[arm] = c.confirmed >= thresholds[arm]
            for arm in ("device_cost", "device_only_cost"):
                alarms[arm] = c.confirmed >= thresholds[arm][c.device]
            holdouts.append({"fold": f, "domain": c.obs["domain"], "source": c.source, "name": c.obs["name"],
                             "counts": {a: c.count(mask) for a, mask in alarms.items()}, "range": coverage_counts})
    summary = aggregate_groups(holdouts)
    report = {"scorer_counterexample": diagnostic(), "folds": fitted, "out_of_fold": summary,
              "fold_aggregates": [aggregate_groups([r for r in holdouts if r["fold"] == f]) for f in (0, 1)],
              "count_vector": ["late", "avoidable", "unavoidable", "fa_episodes", "six_walker_false_frames", "walker_km",
                               "repeated_clear_warnings", "unique_alert_frames", "unique_valid", "unique_alert_on_clear", "unique_clear"],
              "range_vector": ["nominal_violations", "action_violations", "empty_sets", "valid"],
              "by_clip": [{**r, "range": r["range"].tolist(), "counts": {a: x.tolist() for a, x in r["counts"].items()}} for r in holdouts]}
    cache_manifest = json.loads((OUT.parent / "v4_stream" / "manifest.json").read_text(encoding="utf8"))
    report["v4_stream"] = {k: cache_manifest[k] for k in ("protocol_sha256", "checkpoint_sha256")}
    dest = OUT / "operating_point_v2"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "crossfit_alert_report.json").write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
    for arm in ARMS:
        x = summary["pooled"][arm]
        print(arm, f"late {x['late']}/{x['avoidable']} FA/km {x['fa_per_km']:.2f}",
              f"repeated clear/km {x['clear_warnings_per_km']:.2f}, clear duty {100*x['clear_frame_duty']:.1f}%")
    print("range", summary["pooled"]["range"])


if __name__ == "__main__":
    main()
