"""Evaluate causal scale corrections on the frozen v4 stream using the owning scorer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep

import numpy as np
from lie_scale import LieScaleIEKF

ROUTE = Path(__file__).resolve().parents[2]
ART = ROUTE.parents[2] / "artifacts.local/work/egogeom-omni-v0"
STREAM = ART / "math_theory/v4_stream"
DEST = ART / "math_theory/m2_lie_group_scale_iekf"
SUITES = {"aria_gen2": ("aria_g2_20clips",4), "coda_campus": ("coda_zeroshot_shard000",4),
          "arkit_val30_day": ("arkit_val30_shard00",2), "arkit_val30_dark018": ("arkit_val30_shard00",2)}
sys.path[:0] = [str(ROUTE),str(ROUTE.parent/"egoclear-v0")]
from egogeom_temporal_bev import EgoGeomTemporalBEV
from eval_closed_loop import make_episodes, score_episode
from eval_cross_glasses_zeroshot import (
    BW_SPEEDS_PROXY,
    corridor_dist_from_depth,
    temporal_bev_corridor_distances,
)
from eval_egogeom_v3_zeroshot import score_dataset


def predict(cache):
    dep, dh, nrm, ls, K, R, t, fwd = (cache[k] for k in
        ("depth_s4","dh_s4","nrm_s4","ls_s4","K","poses_R","poses_t","strict_fwd"))
    flt, bev = LieScaleIEKF(), EgoGeomTemporalBEV()
    distances, bev_dist, risk, scales, times = [], [], [], [], []
    baseline_clouds, canonical_clouds, candidate_clouds = [], [], []
    archived_fwd = cache["causal_fwd"]
    # The reference corridor function only accesses x[::4, ::4].
    H,W = dep.shape[1:]
    unscaled = np.empty((4*H,4*W),np.float32)
    full_ls = np.empty_like(unscaled)
    for i in range(len(dep)):
        t0 = perf_counter_ns()
        s = flt.step(dep[i],dh[i],nrm[i],ls[i],K[i],R[i],t[i])
        times.append((perf_counter_ns()-t0)/1e6)
        scales.append(s)
        scaled = dep[i]*s
        unscaled[::4,::4] = scaled
        full_ls[::4,::4] = ls[i]
        dw, points = corridor_dist_from_depth(unscaled,None,full_ls,K[i],R[i],fwd[i],use_pred_dh=False)
        candidate_clouds.append(points)
        if s != 1:
            unscaled[::4,::4] = dep[i]
            _, points = corridor_dist_from_depth(unscaled,None,full_ls,K[i],R[i],fwd[i],use_pred_dh=False)
        baseline_clouds.append(points)
        if not np.array_equal(fwd[i], archived_fwd[i]):
            _, points = corridor_dist_from_depth(unscaled,None,full_ls,K[i],R[i],archived_fwd[i],use_pred_dh=False)
        canonical_clouds.append(points)
        db, _, rh = bev.step(scaled,dh[i]*s,nrm[i],ls[i],K[i],R[i],t[i],fwd[i])
        distances.append(dw)
        bev_dist.append(db)
        risk.append(rh)
    v1 = temporal_bev_corridor_distances(t, fwd, R, baseline_clouds)
    old_v1 = v1 if np.array_equal(fwd, archived_fwd) else temporal_bev_corridor_distances(t, archived_fwd, R, canonical_clouds)
    return ({"strict_v1":v1,"canonical_v1":old_v1,
             "candidate_v1":temporal_bev_corridor_distances(t,fwd,R,candidate_clouds),
             "candidate_1f": np.array(distances,np.float32),"candidate_bev":np.array(bev_dist,np.float32),
             "candidate_risk":np.array(risk,np.float32)},scales,times,flt.accepted,flt.attempts)


def common_mae(clips, left, right):
    errors, totals, gt_total = [], np.zeros(2,int), 0
    for clip in clips:
        er = clip["eval_rows"]
        gt = er[:,0]
        eligible = np.isfinite(gt) & (gt>=.3) & (gt<=4) & (er[:,1]>=.6)
        a,b = clip[left],clip[right]
        valid_a, valid_b = eligible & np.isfinite(a) & (a<=6), eligible & np.isfinite(b) & (b<=6)
        totals += [sum(valid_a),sum(valid_b)]
        gt_total += int(sum(eligible))
        both = valid_a & valid_b
        errors.append(np.column_stack((np.abs(a[both]-gt[both]),np.abs(b[both]-gt[both]))))
    paired = np.concatenate(errors)
    return {"eligible_labels":gt_total,"baseline_n":int(totals[0]),"candidate_n":int(totals[1]),
            "common_n":len(paired),"common_mae_baseline_candidate_m":paired.mean(0).tolist() if len(paired) else []}


def operating_point(clips, key):
    episodes = [(ep, c[key]) for c in clips for ep in
                make_episodes(c["name"], c["eval_rows"], BW_SPEEDS_PROXY, kind="blind")]
    best = None
    for threshold in np.linspace(1.4,4.4,31):
        counts = np.sum([score_episode(ep, dist<=threshold) for ep,dist in episodes],axis=0)
        late, avoid, _, alarms, _, km = counts
        if avoid <= 0 or km <= 1.e-4 or alarms/km > 5:
            continue
        if best is None or late < best["late"]:
            best = {"threshold_m":round(float(threshold),2),"late":int(late),"avoidable":int(avoid),
                    "false_alarm_episodes":int(alarms),"walker_km":float(km),"fa_per_km":float(alarms/km)}
    return best


def evaluate(name, frozen, wait_seconds):
    directory, hazard = SUITES[name]
    files = sorted((ART/"zeroshot_eval"/directory).rglob("*.npz"))
    expected = {"aria_gen2":19,"coda_campus":24,"arkit_val30_day":30,"arkit_val30_dark018":30}[name]
    if len(files) != expected:
        raise RuntimeError(f"{name}: expected {expected} source clips, got {len(files)}")
    clips, scales, times, per_clip = [], [], [], []
    for source in files:
        target = STREAM/name/source.name
        deadline = monotonic() + wait_seconds
        while not target.exists() and monotonic() < deadline:
            sleep(2)
        with np.load(target, allow_pickle=False) as cache:
            predictions, clip_scales, lat, accepted, attempts = predict(cache)
            identity = str(cache["identity"])
            predictions.update({k:cache[k] for k in ("d_1f_wz","d_bev_v23","r_hyb_v23",
                                                     "d_1f_strict","d_bev_strict","r_hyb_strict")})
        # Truth is opened only after this entire clip's candidate & control outputs exist.
        with np.load(source,allow_pickle=False) as labels:
            predictions.update(name=source.stem,eval_rows=labels["eval_rows"])
        clips.append(predictions)
        scales.extend(clip_scales)
        times.extend(lat)
        per_clip.append({"name":source.stem,"identity":identity,"frames":len(lat),"attempted":attempts,"accepted":accepted,
                         "scale_min":min(clip_scales),"scale_median":float(np.median(clip_scales)),
                         "scale_max":max(clip_scales)})
    def scores(one, v1, bev, risk):
        return {"1f_world_z":score_dataset(clips,one,None,hazard),
                "stage2_bev_v1_proto":score_dataset(clips,v1,None,hazard),
                "stage2_bev_v23_hard":score_dataset(clips,bev,None,hazard),
                "stage2_bev_v23_hyb":score_dataset(clips,bev,risk,hazard)}
    published = frozen[name]
    canonical = scores("d_1f_wz","canonical_v1","d_bev_v23","r_hyb_v23")
    strict = scores("d_1f_strict","strict_v1","d_bev_strict","r_hyb_strict") if name=="aria_gen2" else canonical
    candidate = scores("candidate_1f","candidate_v1","candidate_bev","candidate_risk")
    counts = {}
    if name.startswith("arkit"):
        for key,metrics in [("d_bev_strict",strict),("candidate_bev",candidate)]:
            point = operating_point(clips,key)
            reported = metrics["stage2_bev_v23_hyb"]["cl_late_fa5_pct"]
            if point is not None:
                assert round(100*point["late"]/point["avoidable"],1)==reported
            counts[key] = point
    return {"published":published,"unmodified_protocol_replay":canonical,"strict_unscaled":strict,
            "strict_scaled":candidate,"canonical_exact_rounded_match":published==canonical,
            "arkit_bev_late5_operating_point":counts,
            "scale_p05_median_p95":np.percentile(scales,[5,50,95]).tolist(),
            "cpu_step_ms":{"median":float(np.median(times)),"p95":float(np.percentile(times,95)),
                           "mean":float(np.mean(times)),"n":len(times)},
            "single_frame_mae_support":common_mae(clips,"d_1f_strict","candidate_1f"),
            "bev_mae_support":common_mae(clips,"d_bev_strict","candidate_bev"),"clips":per_clip}


def gates(suites):
    out = {}
    if "aria_gen2" in suites:
        c = suites["aria_gen2"]["strict_scaled"]
        out["aria_1f_mae_le_0.55"] = c["1f_world_z"]["dist_mae_m"]<=.55
        out["aria_bev_mae_le_0.42"] = c["stage2_bev_v23_hyb"]["dist_mae_m"]<=.42
        for arm in ("1f_world_z","stage2_bev_v23_hyb"):
            out[f"aria_{arm}_auroc_ge_0.72"] = c[arm]["auroc"]>=.72
    for name,limit in [("arkit_val30_day",4.5),("arkit_val30_dark018",5.5)]:
        if name in suites:
            pct = suites[name]["strict_scaled"]["stage2_bev_v23_hyb"]["cl_late_fa5_pct"]
            out[f"{name}_bev_late5"] = 0<=pct<=limit
    if "coda_campus" in suites:
        q = suites["coda_campus"]
        for arm in ("1f_world_z","stage2_bev_v23_hyb"):
            b,c = q["strict_unscaled"][arm],q["strict_scaled"][arm]
            out[f"coda_{arm}_auroc_mae_nonworse"] = c["auroc"]>=b["auroc"] and c["dist_mae_m"]<=b["dist_mae_m"]
    return out


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",choices=list(SUITES),action="append",help="Default: all four suites")
    parser.add_argument("--wait-for-cache-seconds",type=int,default=0,help="Optional wait per atomic clip from an active stream build")
    args = parser.parse_args()
    suites = {}
    frozen = json.loads((ART/"cross_glasses_zeroshot_v4_final.json").read_text(encoding="utf-8"))["suites"]
    DEST.mkdir(exist_ok=True,parents=True)
    for name in args.suite or SUITES:
        suites[name] = evaluate(name,frozen,args.wait_for_cache_seconds)
        q = suites[name]
        print(name, q["strict_scaled"], q["cpu_step_ms"],flush=True)
    filename = "v4_empirical_report.json" if set(suites)==set(SUITES) else "+".join(suites)+".json"
    result = {"suites":suites,"criteria":gates(suites),"scope":"reused frozen RGB v4 offline replay on supplied metric camera poses"}
    (DEST/filename).write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
