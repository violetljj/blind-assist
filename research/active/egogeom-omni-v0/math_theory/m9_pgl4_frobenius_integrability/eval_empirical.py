"""Freeze each clip prediction before scoring. Run v4, or pass --version v41."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns

for setting in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[setting] = "1"
import cv2
import numpy as np

from frobenius import FrobeniusFilter

HOME = Path(__file__).resolve().parent
MATH = HOME.parent
ROUTE = MATH.parent
REPO = ROUTE.parents[2]
ART = REPO/"artifacts.local/work/egogeom-omni-v0"
OUTPUT = ART/"math_theory"/HOME.name
M2 = MATH/"m2_lie_group_scale_iekf"/"rgb_triangulation"
OLD = ART/"math_theory"/M2.parent.name/M2.name
sys.path[:0] = [str(M2), str(M2.parent), str(ROUTE), str(ROUTE.parent/"egoclear-v0")]
from evaluate import fixed_threshold, support_metrics
from eval_on_v4 import SUITES, BW_SPEEDS_PROXY, EgoGeomTemporalBEV, corridor_dist_from_depth, make_episodes, score_dataset
from eval_cross_glasses_zeroshot import decode_isotropic_frame
from metric_scale import MetricScale

ARMS = ("original", "shape", "joint")


class GlobalAnchor(MetricScale):
    def _local_scale(self, *args):
        # The RGB fit uses the shape-corrected depths of the accepted matches.
        # The additional non-projective local M2 output is unused here.
        return float(np.exp(self.log_scale))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def store(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def predict(src, cache, suite, dest):
    with np.load(cache) as obs:
        dep, ls, nrm, K, R, t, fw = (obs[key] for key in
            ("depth_s4", "ls_s4", "nrm_s4", "K", "poses_R", "poses_t", "strict_fwd"))
        pred = {f"original_{kind}": obs[key] for kind, key in
            zip(("one","bev","risk"), ("d_1f_strict","d_bev_strict","r_hyb_strict"))}
    with np.load(src) as packed:
        blob, offsets, hw, intr = (packed[key] for key in ("rgb_blob", "rgb_offsets", "orig_hw", "K_orig"))
    n, h, w = dep.shape
    shape, anchor = FrobeniusFilter(), GlobalAnchor()
    bevs = [EgoGeomTemporalBEV(), EgoGeomTemporalBEV()]
    for arm in ARMS[1:]:
        for kind in ("one","bev","risk"):
            pred[arm+"_"+kind] = np.empty(n,np.float32)
    full, unc = [np.empty((4*h,4*w),np.float32) for _ in range(2)]
    timing = np.zeros((n,2))
    params = np.zeros((n,5))
    for i in range(n):
        start = perf_counter_ns()
        corrected, updated, beta = shape.step(dep[i], nrm[i], K[i], R[i])
        timing[i,0] = (perf_counter_ns()-start)/1e6
        shade = .18 if suite=="arkit_val30_dark018" else 1.
        rgb, _ = decode_isotropic_frame(blob[offsets[i]:offsets[i+1]].tobytes(), hw, intr, shade)
        start = perf_counter_ns()
        gray = cv2.cvtColor(np.rint(255*rgb).astype(np.uint8),cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (2*w,2*h), interpolation=cv2.INTER_AREA)
        scale, _, _ = anchor.step(gray, corrected, ls[i], K[i], R[i], t[i])
        timing[i,1] = (perf_counter_ns()-start)/1e6
        params[i] = [*beta,scale,updated]
        unc[::4,::4] = ls[i]
        for arm, bev, d in zip(ARMS[1:], bevs, (corrected, corrected*scale)):
            full[::4,::4] = d
            pred[arm+"_one"][i] = corridor_dist_from_depth(full,None,unc,K[i],R[i],fw[i],False)[0]
            dist, _, risk = bev.step(d,None,None,ls[i],K[i],R[i],t[i],fw[i])
            pred[arm+"_bev"][i], pred[arm+"_risk"][i] = dist, risk
    np.savez_compressed(dest, **pred, timing=timing, params=params)


def warnings(clips, key):
    total = 0
    for c in clips:
        er = c["eval_rows"]
        alert = c[key]<=2.
        for ep in make_episodes(c["name"], er, BW_SPEEDS_PROXY, kind="blind"):
            valid = ep["valid"]
            clear = valid & (er[:,0]>4) & (er[:,2]<=.5)
            pos = np.cumsum(np.where(valid,er[:,3],0.), dtype=np.float64)
            last = -np.inf
            for i in range(len(clear)):
                if not clear[i]:
                    last = -np.inf
                elif alert[i] and pos[i]-last > 2.*ep["v"]:
                    total += 1
                    last = pos[i]
    return total


def evaluate(version, suites):
    cv2.setNumThreads(1)
    out = OUTPUT/version
    out.mkdir(parents=True, exist_ok=True)
    stream = ART/"math_theory"/(version+"_stream")
    manifest = json.loads((stream/"manifest.json").read_text())
    files = [Path(__file__), HOME/"frobenius.py", M2/"evaluate.py", M2/"metric_scale.py",
             ROUTE/"egogeom_temporal_bev.py", ROUTE/"eval_cross_glasses_zeroshot.py",
             ROUTE/"eval_egogeom_v3_zeroshot.py", ROUTE.parent/"egoclear-v0"/"eval_closed_loop.py"]
    full_report = {"version":version, "stream_manifest_sha256":sha(stream/"manifest.json"),
                   "code_sha256":{p.name:sha(p) for p in files}, "suites":{}}
    for suite in suites:
        folder, hazard = SUITES[suite]
        preds = out/suite
        preds.mkdir(exist_ok=True)
        sources = sorted((ART/"zeroshot_eval"/folder).rglob("*.npz"))
        entries = {row["name"]:row for row in manifest["suites"][suite]["clips"]}
        if {p.stem for p in sources} != set(entries):
            raise ValueError("Incomplete stream")
        clips, prices, updated, identities = [], [], [], []
        for src in sources:
            cache = stream/suite/src.name
            src_hash = sha(src)
            if src_hash != entries[src.stem]["input_sha256"]:
                raise ValueError("Source identity mismatch")
            pred = preds/src.name
            predict(src,cache,suite,pred)
            identities.append({"clip":src.stem,"pack_sha256":src_hash,"stream_sha256":sha(cache),
                               "prediction_sha256":sha(pred)})
            with np.load(pred) as p:
                c = {k:p[k] for k in p.files if k not in ("timing","params")}
                prices.extend(p["timing"])
                updated.extend(p["params"][:,-1])
            if version == "v4":
                archive = OLD/"predictions"/suite/src.name
                with np.load(archive) as p:
                    for kind in ("one","bev","risk"):
                        np.testing.assert_array_equal(c["original_"+kind],p["original_"+kind])
                    c.update({k:p[k] for k in p.files if k.startswith(("prior_m2_","rgb_global_","rgb_local_"))})
                identities[-1]["m2_prediction_sha256"] = sha(archive)
            with np.load(src) as truth:
                c.update(name=src.stem,eval_rows=truth["eval_rows"])
            clips.append(c)
        arms = ARMS + (("prior_m2","rgb_global","rgb_local") if version=="v4" else ())
        metrics = {}
        for arm in arms:
            one, bev, risk = [arm+"_"+k for k in ("one","bev","risk")]
            fixed = fixed_threshold(clips,bev)
            fixed["repeated_clear"] = warnings(clips,bev)
            fixed["repeated_clear_per_km"] = fixed["repeated_clear"]/fixed["walker_km"]
            metrics[arm] = {"one":score_dataset(clips,one,None,hazard),"bev":score_dataset(clips,bev,risk,hazard),
                            "one_support":support_metrics(clips,one),"bev_support":support_metrics(clips,bev),
                            "fixed_2m":fixed}
        times = np.array(prices)
        result = {"metrics":metrics,"frames":len(times),"updated_frames":int(sum(updated)),
                  "ms_p50_p95":{"core":np.percentile(times[:,0],[50,95]).tolist(),
                                "rgb_anchor":np.percentile(times[:,1],[50,95]).tolist(),
                                "core_plus_rgb":np.percentile(times.sum(1),[50,95]).tolist()},
                  "inputs":identities}
        store(out/(suite+".json"),result)
        print(version,suite,metrics["joint"]["bev_support"],fixed,flush=True)
        full_report["suites"][suite] = result
    store(out/"v5_empirical_report.json",full_report)
    if version == "v4":
        store(OUTPUT/"v5_empirical_report.json",full_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version",choices=("v4","v41"),default="v4")
    parser.add_argument("--suites",nargs="+",choices=SUITES,default=list(SUITES))
    args = parser.parse_args()
    evaluate(args.version,args.suites)
