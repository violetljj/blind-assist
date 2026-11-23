"""Replay causal motion against the frozen walker proxy. No label is read by predict()."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
from barrier import EscapeMargin, View, contact_time, sparse_motion

ROUTE = Path(__file__).resolve().parents[2]
REPO = ROUTE.parents[2]
WORK = REPO / "artifacts.local/work/egogeom-omni-v0"
EVIDENCE = WORK / "math_theory"
sys.path.insert(0, str(ROUTE.parent/"egoclear-v0"))
sys.path.insert(0, str(ROUTE))
from eval_closed_loop import eval_score_curve
from eval_cross_glasses_zeroshot import BW_SPEEDS_PROXY

SUITES = {"coda_campus": "coda_zeroshot_shard000",
          "arkit_val30_day":"arkit_val30_shard00", "arkit_val30_dark018":"arkit_val30_shard00"}


def predict(cache):
    gray, depth, intr, rot, pos, fwd, distance = (
        cache[key] for key in ("gray_s4","depth_s4","K","poses_R","poses_t","causal_fwd","d_1f_wz"))
    engine = EscapeMargin()
    count = len(distance)
    ttc = np.full(count, np.inf)
    margin = np.full(count, np.inf)
    modes = Counter()
    elapsed, eligible_times = [], []
    before = None
    for i in range(count):
        start = perf_counter_ns()
        now = View(gray[i],depth[i],intr[i],rot[i],pos[i])
        state = sparse_motion(before, now, fwd[i], distance[i], .20) if before is not None else None
        if state is None and np.isfinite(distance[i]):
            state = np.array([distance[i],0.]), np.zeros(2), np.eye(2)*.15**2, np.eye(2)*.4**2, "untracked"
        if state is not None:
            p,v,cov_p,cov_v,mode = state
            ttc[i] = contact_time(p,v)
            # Uncalibrated one-standard-deviation sensitivity allowance for 2s replay.
            allowance = np.sqrt(np.trace(cov_p)) + engine.horizon*np.sqrt(np.trace(cov_v))
            margin[i] = engine.clearance(p,v,allowance=allowance)
        else:
            mode = "no_corridor_support"
        before = now
        us = (perf_counter_ns()-start)/1000
        elapsed.append(us)
        if state is not None:
            eligible_times.append(us)
        modes[mode] += 1
    return distance,ttc,margin,modes,elapsed,eligible_times


def warning_margin(scores, threshold, ttc_by_name):
    values = []
    total = 0
    for name, score in scores.items():
        alert = score <= threshold
        onset = alert & ~np.r_[False, alert[:-1]]
        finite = onset & np.isfinite(ttc_by_name[name])
        total += int(onset.sum())
        values.extend((ttc_by_name[name][finite]-.65).tolist())
    return {"alert_onsets": total, "finite_estimated_contact": len(values),
                "predicted_ttc_minus_reaction_s_q10_med_q90": np.quantile(values,[.1,.5,.9]).tolist() if values else None}


def run_suite(suite, cache_root):
    source = sorted((WORK/"zeroshot_eval"/SUITES[suite]).rglob("*.npz"))
    if not source:
        raise FileNotFoundError(SUITES[suite])
    rows, distances, ttcs, margins = {},{},{},{}
    times, eligible_times, modes = [],[],Counter()
    for p in source:
        with np.load(cache_root/suite/p.name,allow_pickle=False) as cache:
            d,t,h,counts,ts,tes = predict(cache)
        distances[p.stem], ttcs[p.stem], margins[p.stem] = d,t,h
        # Predictions and state for this clip are complete before metric-only truth access.
        with np.load(p,allow_pickle=False) as labels:
            rows[p.stem] = labels["eval_rows"]
        if len(d) != len(rows[p.stem]):
            raise ValueError(f"Frame count mismatch: {p.name}")
        times.extend(ts)
        eligible_times.extend(tes)
        modes.update(counts)
    arms = {}
    for label, scores, grid, operating in [
        ("distance_m",distances,np.linspace(1.4,4.4,31),2.),
        ("ttc_s",ttcs,np.linspace(.2,3.2,31),1.5),
        ("escape_clearance_m",margins,np.linspace(-1.5,1.5,31),0.),
    ]:
        grid = np.unique(np.round(np.r_[grid,operating],6))
        curve = eval_score_curve(rows,scores,BW_SPEEDS_PROXY,grid,higher_is_alert=False)
        eligible = [q for q in curve if q["late_rate"] is not None and q["fa_ep_per_km"] is not None and q["fa_ep_per_km"] <= 5.]
        best = min(eligible,key=lambda q:(q["late_rate"],q["fa_ep_per_km"],q["thr"])) if eligible else None
        operating_result = next(q for q in curve if q["thr"] == operating)
        arms[label] = {"fixed_operating": operating_result, "descriptive_best_at_fa5": best,
                           "operating_lead": warning_margin(scores,operating,ttcs),
                           "best_lead": warning_margin(scores,best["thr"],ttcs) if best else None,
                           "curve": curve}
    valid = sum(int(((np.isfinite(v[:,0])|np.isinf(v[:,0])) & (v[:,1]>=.6)).sum()) for v in rows.values())
    return {"clips": len(source),"frames": len(times),"valid": valid, "mode_frame_counts": dict(modes),
                "dt_s": .20, "time_basis": "source packer nominal 5 Hz; ARKit uses minimum time gaps and has no stored timestamps",
                "predict_all_mean_ms": float(np.mean(times)/1000), "predict_eligible_mean_ms": float(np.mean(eligible_times)/1000),
                "predict_eligible_p95_ms": float(np.quantile(eligible_times,.95)/1000), "arms": arms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,default=EVIDENCE/"v4_stream")
    args = p.parse_args()
    cv2.setNumThreads(1)
    suites = {}
    frozen = json.loads((WORK/"cross_glasses_zeroshot_v4_final.json").read_text(encoding="utf-8"))
    for suite in SUITES:
        suites[suite] = run_suite(suite,args.cache)
        suites[suite]["archived_1f_world_z"] = frozen["suites"][suite]["1f_world_z"]
        print(suite, json.dumps({k:v["descriptive_best_at_fa5"] for k,v in suites[suite]["arms"].items()}),flush=True)
    out = EVIDENCE/"m4_stochastic_cbf_ics"/"v4_empirical_report.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    result = {"evidence":"Offline replay; hypothetical walker reaction/stopping model. TTC estimated from predictions.",
              "cache":args.cache.name,"suites":suites}
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")


if __name__ == "__main__":
    main()
