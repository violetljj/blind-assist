"""CPU-only, cross-traversal Development screen; no training or protected data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_number(name):
    return int(re.search(r"(\d+)\.png$", name).group(1))


def features(z, names, role):
    values, manifest = [], []
    for i, name in enumerate(names):
        raw = z.read(name)
        im = Image.open(io.BytesIO(raw)).convert("L").resize((64, 32), Image.Resampling.LANCZOS)
        a = np.asarray(im, dtype=np.float32)
        for y in range(0, 32, 8):
            for x in range(0, 64, 8):
                patch = a[y:y+8, x:x+8]
                patch -= patch.mean()
                patch /= max(float(patch.std()), 1.0)
        values.append(a.ravel())
        manifest.append({"role": role, "ordinal_zero_based": i, "archive_member": name,
                         "upstream_frame_number": image_number(name), "sha256": sha(raw)})
    return np.stack(values), manifest


def sequence_cost(d, length, velocities=(0.8, 1.0, 1.2)):
    """Current + past queries only. No GT or future query image is an argument."""
    nq, nr = d.shape
    out = np.full_like(d, np.inf)
    for v in velocities:
        total = np.zeros_like(d)
        valid = np.ones_like(d, dtype=bool)
        for k in range(length):
            off = int(np.rint(k*v))
            shifted = np.full_like(d, np.inf)
            if k < nq and off < nr:
                shifted[k:, off:] = d[:nq-k, :nr-off]
            valid &= np.isfinite(shifted)
            total += shifted
        out = np.minimum(out, np.where(valid, total/length, np.inf))
    return out


def predictions(cost, exclusion_radius=5):
    n, m = cost.shape
    pred = np.argmin(cost, axis=1)
    margin = np.zeros(n, dtype=float)
    for q in range(n):
        row = cost[q].copy()
        best = row[pred[q]]
        row[max(0,pred[q]-exclusion_radius):min(m,pred[q]+exclusion_radius+1)] = np.inf
        second = np.min(row)
        if np.isfinite(best) and np.isfinite(second):
            margin[q] = float(second-best)
    return pred, margin


def report_arm(name, cost, truth_centers, tolerance, cal, known, absent, manifest):
    pred, margin = predictions(cost)
    correct = abs(pred-truth_centers) <= tolerance
    eligible = cal[correct[cal] & np.isfinite(cost[cal].min(axis=1))]
    threshold = float(np.quantile(margin[eligible], .05)) if len(eligible) else None
    accepted = margin >= threshold if threshold is not None else np.zeros(len(pred), dtype=bool)
    accepted &= np.isfinite(cost.min(axis=1))
    rows = []
    for label, ids in [("calibration_known", cal), ("evaluation_known", known), ("evaluation_map_absent", absent)]:
        for q in ids:
            rows.append({"arm": name, "split": label, "query_ordinal": int(q),
                         "query_frame": manifest[q]["upstream_frame_number"],
                         "prediction_ordinal": int(pred[q]), "truth_center_ordinal": float(truth_centers[q]),
                         "absolute_index_error": float(abs(pred[q]-truth_centers[q])),
                         "known_match_correct": bool(correct[q]) if label != "evaluation_map_absent" else None,
                         "confidence_margin": float(margin[q]), "accepted": bool(accepted[q])})
    def stats(ids):
        good = correct[ids] & accepted[ids]
        bad = ~correct[ids] & accepted[ids]
        return {"n":len(ids), "raw_top1_correct":int(correct[ids].sum()),
                "accepted_correct":int(good.sum()), "accepted_wrong":int(bad.sum()),
                "abstained":int((~accepted[ids]).sum()), "coverage":float(accepted[ids].mean()),
                "precision_when_accepted":float(good.sum()/accepted[ids].sum()) if accepted[ids].sum() else None,
                "absolute_index_error_median":float(np.median(abs(pred[ids]-truth_centers[ids]))),
                "absolute_index_error_p95":float(np.quantile(abs(pred[ids]-truth_centers[ids]),.95))}
    result = {"arm":name, "confidence_threshold":threshold,
              "calibration_correct_count":len(eligible), "calibration":stats(cal), "known":stats(known),
              "map_absent":{"n":len(absent), "false_accepts":int(accepted[absent].sum()),
                            "rejected":int((~accepted[absent]).sum())}}
    return result, rows


def main():
    t0=time.perf_counter()
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=600)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(args.archive)
    names=z.namelist()
    groups={s:sorted([n for n in names if f"/{s}/" in n and n.endswith(".png")], key=image_number)
            for s in ["summer","winter"]}
    mats=[n for n in names if n.endswith(".mat")]
    audit={"archive_bytes":args.archive.stat().st_size,"archive_sha256":sha(args.archive.read_bytes()),
           "entry_count":len(names), "seasons":{k:len(v) for k,v in groups.items()}, "mat_files":mats,
           "source_revision":"02c6ed1e623b94c450cdce86271c9cfe5e00f2a5",
           "source_url":"https://raw.githubusercontent.com/qcr/openseqslam2/02c6ed1e623b94c450cdce86271c9cfe5e00f2a5/datasets/samples.zip"}
    gtdata=loadmat(io.BytesIO(z.read(mats[0])))
    gt=np.asarray(gtdata["x"])
    audit["ground_truth"]={"shape":list(gt.shape),"values":np.unique(gt).tolist(),
                            "nonzero_count":int(np.count_nonzero(gt))}
    write_json(args.out/"data_audit.json",audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    if min(len(v) for v in groups.values()) < args.limit:
        raise ValueError("Fewer than requested frames: inspect audit before changing sample")
    # Upstream config samples every tenth file; the MAT dimensions must bind it.
    ref_all, query_all=groups["summer"], groups["winter"]
    if gt.shape==(len(ref_all),len(query_all)):
        gt_layout="all_images"
    elif gt.shape==(len(ref_all[::10]),len(query_all[::10])):
        ref_all,query_all=ref_all[::10],query_all[::10]
        gt_layout="every_tenth_sorted_image"
    else:
        raise ValueError(f"GT shape does not bind archive images: {gt.shape}")
    n=min(args.limit,len(ref_all),len(query_all))
    if n < 358:
        raise ValueError("Fewer than 358 GT-bound image pairs; inspect data contract")
    ref_names,query_names=ref_all[:n],query_all[:n]
    # Both axes are inspected: default Nordland truth is diagonal/symmetric.
    # Require symmetry before adopting the reference-row/query-column layout.
    if gt.shape[0]!=gt.shape[1] or not np.array_equal(gt,gt.T):
        raise ValueError("Ground truth axis convention needs explicit source inspection")
    centers=np.array([np.mean(np.flatnonzero(gt[:,q] == 1)) for q in range(n)])
    if not np.all(np.isfinite(centers)):
        raise ValueError("Missing exact-match ground truth")
    ref,ref_manifest=features(z,ref_names,"summer_reference")
    query,query_manifest=features(z,query_names,"winter_query")
    if set(x["sha256"] for x in ref_manifest)&set(x["sha256"] for x in query_manifest):
        raise ValueError("Exact duplicate image across traversals")
    with (args.out/"sample_manifest.jsonl").open("w",encoding="utf-8") as f:
        for row in ref_manifest+query_manifest:f.write(json.dumps(row)+"\n")
    distance=cdist(query,ref,metric="cityblock")/ref.shape[1]
    np.savez_compressed(args.out/"descriptors_and_distance.npz",reference=ref,query=query,distance=distance)
    # Map supports only first 240 places; later real winter frames are absent.
    map_n=240
    d=distance[:,:map_n]
    mu=uniform_filter1d(d,size=41,axis=1,mode="nearest")
    var=uniform_filter1d(d*d,size=41,axis=1,mode="nearest")-mu*mu
    enhanced=(d-mu)/np.sqrt(np.maximum(var,1e-6))
    cal=np.arange(10,70)
    known=np.arange(90,230)
    absent=np.arange(280,n)
    tolerance=5
    assert np.all(centers[known] < map_n-tolerance)
    assert np.all(centers[absent] > map_n+tolerance)
    arms={"single_frame_raw":d,"single_frame_local_contrast":enhanced,
          "causal_sequence_5":sequence_cost(enhanced,5),"causal_sequence_10":sequence_cost(enhanced,10)}
    reports,rows=[],[]
    for name,cost in arms.items():
        report,records=report_arm(name,cost,centers,tolerance,cal,known,absent,query_manifest)
        reports.append(report);rows+=records
    # Full-map retrieval is descriptive only, using the same held query range.
    full_pred=np.argmin(distance,axis=1)
    full_correct=abs(full_pred[known]-centers[known])<=tolerance
    result={"status":"COMPLETED_DEVELOPMENT_SCREEN","evidence_role":"public_real_cross_season_train_route_development",
            "not_evidence_for":["head_worn_walking","route_deviation","metric_localization","hazard_forecast","user_safety","state_of_the_art_superiority"],
            "data_audit":audit,"gt_binding":gt_layout,"gt_symmetric":True,
            "reference_count":n,"query_count":n,"map_reference_count":map_n,
            "split":{"calibration_known":[10,69],"evaluation_known":[90,229],"evaluation_map_absent":[280,n-1]},
            "tolerance_reference_frames":tolerance,"patch_pixels":[8,8],"feature_pixels":[64,32],
            "local_contrast_window":41,"sequence_velocities":[.8,1.,1.2],
            "threshold_rule":"5th percentile of confidence margins of correct calibration matches; no absent-query calibration",
            "full_map_single_frame":{"n":len(known),"correct":int(full_correct.sum())},
            "arms":reports,"cpu_seconds_script_wall":time.perf_counter()-t0,"gpu_seconds":0,
            "backend":{"placement":"TASK_NOT_GPU_SUITABLE","actual":"numpy/scipy CPU","numpy":np.__version__,"python":sys.version,"host":platform.node()}}
    write_json(args.out/"result.json",result)
    with (args.out/"predictions.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({"wall_seconds":result["cpu_seconds_script_wall"],"arms":reports},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
