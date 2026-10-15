"""Post-hoc Development ablation, preserving baseline run02 and its thresholds."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def main():
    start=time.perf_counter()
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=np.load(a.baseline/"descriptors_and_distance.npz")
    distance=data["distance"]
    rows=[r for r in csv.DictReader((a.baseline/"predictions.csv").open(encoding="utf-8"))
          if r["arm"]=="causal_sequence_10"]
    for row in rows:
        row["residual"]=float(distance[int(row["query_ordinal"]),int(row["prediction_ordinal"])])
    cal=[r for r in rows if r["split"]=="calibration_known" and r["known_match_correct"]=="True"]
    threshold=float(np.quantile([r["residual"] for r in cal],.95))
    output=[]
    metrics={}
    for arm in ["margin_only_baseline","absolute_only","margin_and_absolute"]:
        metrics[arm]={}
        for split in ["calibration_known","evaluation_known","evaluation_map_absent"]:
            subset=[r for r in rows if r["split"]==split]
            accepted=[]
            for r in subset:
                margin=r["accepted"]=="True"
                absolute=r["residual"]<=threshold
                yes=margin if arm=="margin_only_baseline" else absolute if arm=="absolute_only" else margin and absolute
                accepted.append(yes)
                output.append({"arm":arm,"split":split,"query_ordinal":int(r["query_ordinal"]),
                               "absolute_residual":r["residual"],"accepted":yes,
                               "known_match_correct":r["known_match_correct"]})
            accepted=np.array(accepted,dtype=bool)
            if split=="evaluation_map_absent":
                metrics[arm][split]={"n":len(subset),"false_accepts":int(accepted.sum()),"rejected":int((~accepted).sum())}
            else:
                correct=np.array([r["known_match_correct"]=="True" for r in subset])
                metrics[arm][split]={"n":len(subset),"accepted_correct":int((accepted&correct).sum()),
                                    "accepted_wrong":int((accepted&~correct).sum()),"abstained":int((~accepted).sum())}
    result={"status":"COMPLETED_POSTHOC_DEVELOPMENT_ABLATION","hypothesis":"Absolute current-frame residual adds evidence of map membership beyond relative uniqueness",
            "selection_disclosure":"Hypothesis proposed after observing run02 false acceptance; threshold fixed at calibration-correct residual p95 before run03 scores",
            "baseline_result_sha256":hashlib.sha256((a.baseline/"result.json").read_bytes()).hexdigest(),
            "calibration_correct_count":len(cal),"maximum_absolute_residual":threshold,
            "threshold_rule":"p95 raw current-frame residual among correct calibration matches; no absent queries used",
            "metrics":metrics,"cpu_seconds_script_wall":time.perf_counter()-start,"gpu_seconds":0}
    (a.out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (a.out/"predictions.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
