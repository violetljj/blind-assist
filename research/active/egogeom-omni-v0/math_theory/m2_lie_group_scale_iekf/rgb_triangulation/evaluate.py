"""Frozen v4 RGB/pose replay. Save clip predictions first, then score on the original rows."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
from metric_scale import MetricScale

source_dir = Path(__file__).resolve()
sys.path[:0] = [str(source_dir.parents[1]), str(source_dir.parents[3])]
from eval_cross_glasses_zeroshot import decode_isotropic_frame
from eval_on_v4 import (
    ART,
    BW_SPEEDS_PROXY,
    DEST,
    STREAM,
    SUITES,
    EgoGeomTemporalBEV,
    LieScaleIEKF,
    corridor_dist_from_depth,
    make_episodes,
    operating_point,
    score_dataset,
    score_episode,
)

RUN = DEST/"rgb_triangulation"
ARMS = ("original","prior_m2","rgb_global","rgb_local")


def predict(src,suite,dest):
    with np.load(STREAM/suite/src.name) as obs:
        dep,dh,nrm,ls,K,R,t,fw = (obs[k] for k in
                                  ("depth_s4","dh_s4","nrm_s4","ls_s4","K","poses_R","poses_t","strict_fwd"))
        out = {f"original_{name}":obs[key] for name,key in
               [("one","d_1f_strict"),("bev","d_bev_strict"),("risk","r_hyb_strict")]}
        identity = str(obs["identity"])
    with np.load(src) as pack:
        blob,offs,hw,Korig = (pack[k] for k in ("rgb_blob","rgb_offsets","orig_hw","K_orig"))
    n,h,w = dep.shape
    old,tri = LieScaleIEKF(),MetricScale()
    bevs = {arm:EgoGeomTemporalBEV() for arm in ARMS[1:]}
    for arm in ARMS[1:]:
        for kind in ("one","bev","risk"):
            out[f"{arm}_{kind}"] = np.empty(n,np.float32)
    scales,anchors,times = [],[],[]
    full,full_ls = np.empty((4*h,4*w),np.float32),np.empty((4*h,4*w),np.float32)
    shade = .18 if suite=="arkit_val30_dark018" else 1.
    for i in range(n):
        old_s = old.step(dep[i],dh[i],nrm[i],ls[i],K[i],R[i],t[i])
        rgb,k_observed = decode_isotropic_frame(blob[offs[i]:offs[i+1]].tobytes(),hw,Korig,dark_scale=shade)
        if i==0 and not np.allclose(k_observed,K[i]):
            raise ValueError("RGB intrinsics differ from the cached v4 stream")
        started = perf_counter_ns()
        gray = cv2.cvtColor(np.rint(255*rgb).astype(np.uint8),cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray,(2*w,2*h),interpolation=cv2.INTER_AREA)
        s,local,k = tri.step(gray,dep[i],ls[i],K[i],R[i],t[i])
        times.append((perf_counter_ns()-started)/1.e6)
        scales.append(s)
        anchors.append(k)
        full_ls[::4,::4] = ls[i]
        for arm,correction in zip(ARMS[1:],(old_s,s,local)):
            d = dep[i]*correction
            full[::4,::4] = d
            distance,_ = corridor_dist_from_depth(full,None,full_ls,K[i],R[i],fw[i],False)
            db,_,risk = bevs[arm].step(d,dh[i]*correction,nrm[i],ls[i],K[i],R[i],t[i],fw[i])
            out[f"{arm}_one"][i] = distance
            out[f"{arm}_bev"][i] = db
            out[f"{arm}_risk"][i] = risk
    np.savez_compressed(dest,identity=identity,scale=scales,anchors=anchors,step_ms=times,**out)
    return {"name":src.stem,"identity":identity,"frames":n,"attempts":tri.attempts,"updates":tri.updates,
            "accepted_tracks":tri.n_tracks}


def support_metrics(clips,key):
    base = "original_"+key.split("_")[-1]
    gt = np.concatenate([c["eval_rows"][:,0] for c in clips])
    cov = np.concatenate([c["eval_rows"][:,1] for c in clips])
    b,d = (np.concatenate([c[k] for c in clips]) for k in (base,key))
    eligible = (gt>=.3)&(gt<=4)&(cov>=.6)
    fixed = eligible & np.isfinite(b) & (b<=6)
    valid = eligible & np.isfinite(d) & (d<=6)
    paired = valid & fixed
    error = np.abs(np.minimum(np.where(np.isfinite(d),d,6.),6.)-gt)
    return {"eligible":int(sum(eligible)), "candidate_finite":int(sum(valid)),"baseline_finite":int(sum(fixed)),
            "common_finite":int(sum(paired)),
            "mae_common_candidate_m":float(np.abs(d[paired]-gt[paired]).mean()) if paired.any() else None,
            "mae_common_baseline_m":float(np.abs(b[paired]-gt[paired]).mean()) if paired.any() else None,
            "mae_filled6_eligible_m":float(error[eligible].mean()),"mae_filled6_baseline_finite_m":float(error[fixed].mean())}


def fixed_threshold(clips,key):
    episodes = [(ep,c[key]) for c in clips for ep in make_episodes(c["name"],c["eval_rows"],BW_SPEEDS_PROXY,kind="blind")]
    late,avoidable,_,fa,_,km = np.sum([score_episode(ep,d<=2.0) for ep,d in episodes],axis=0)
    return {"threshold_m":2.,"late":int(late),"avoidable":int(avoidable),"fa_episodes":int(fa),"walker_km":float(km),
            "late_pct":float(100*late/avoidable) if avoidable else None,"fa_per_km":float(fa/km) if km>0 else None}


def run(suite, output):
    folder,hazard = SUITES[suite]
    clips,stats,prices,scales = [],[],[],[]
    dest = output/"predictions"/suite
    dest.mkdir(parents=True,exist_ok=True)
    for i,src in enumerate(sorted((ART/"zeroshot_eval"/folder).rglob("*.npz")),1):
        stats.append(predict(src,suite,dest/src.name))
        with np.load(dest/src.name) as pred:
            c = {f"{a}_{k}":pred[f"{a}_{k}"] for a in ARMS for k in ("one","bev","risk")}
            prices.extend(pred["step_ms"])
            scales.extend(pred["scale"])
        with np.load(src) as truth:
            c.update(name=src.stem,eval_rows=truth["eval_rows"])
        clips.append(c)
        if i%10==0:
            print(suite,i,flush=True)
    metrics = {}
    for arm in ARMS:
        one,bev,risk = [f"{arm}_{s}" for s in ("one","bev","risk")]
        metrics[arm] = {"one":score_dataset(clips,one,None,hazard),"bev":score_dataset(clips,bev,risk,hazard),
                        "one_support":support_metrics(clips,one),"bev_support":support_metrics(clips,bev),
                        "fixed_2m":fixed_threshold(clips,bev),"best_bev_fa5":operating_point(clips,bev)}
    previous = json.loads((DEST/"v4_empirical_report.json").read_text())["suites"][suite]
    for arm,baseline in [("original","strict_unscaled"),("prior_m2","strict_scaled")]:
        for code,full in [("one","1f_world_z"),("bev","stage2_bev_v23_hyb")]:
            assert metrics[arm][code]==previous[baseline][full],(suite,arm,code)
    result = {"metrics":metrics,"clips":stats,"cpu_step_ms_p50_p95_mean":[float(np.median(prices)),float(np.percentile(prices,95)),float(np.mean(prices))],
              "scale_quantiles":np.quantile(scales,[.05,.5,.95]).tolist(),"previous_arms_matched":True}
    (output/(suite+".json")).write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(suite, {arm: [q["one"]["dist_mae_m"],q["one"]["auroc"],q["bev"]["dist_mae_m"],q["bev"]["auroc"]]
                  for arm,q in metrics.items()},flush=True)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RUN, help="Destination for this run's predictions and reports")
    output = parser.parse_args().out_dir
    cv2.setNumThreads(1)
    output.mkdir(parents=True,exist_ok=True)
    files = [Path(__file__),Path(__file__).parent/"metric_scale.py"]
    result = {"identity":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "v4_manifest":hashlib.sha256((STREAM/"manifest.json").read_bytes()).hexdigest(),
              "suites":{suite:run(suite,output) for suite in SUITES}}
    (output/"v4_empirical_report.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
