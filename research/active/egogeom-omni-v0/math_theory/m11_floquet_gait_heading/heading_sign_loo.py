"""Post-result sign diagnostic. Train on two complete walks and test on the third."""
import json
import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from eval_empirical import (DATA, HERE, OUT, WORK, infer_session, load, score_visit,
                            sha, summary)


def run():
    fixed_file=OUT/"v5_empirical_report.json"
    fixed=load(fixed_file)
    geom_file=DATA/"screen-hololens/summary.json"
    old_file=DATA/"screen-hololens/causal_comparison.json"
    reasons_file=WORK/"lamar-miss-breakdown-dev-20261117/breakdown.json"
    geometry={v["visit"]:v for v in load(geom_file)["visits"]}
    old={(r["session"],int(r["timestamp_us"])):r for r in load(old_file)["predictions"]}
    reasons={(p["session"],p["episode"]):p for p in load(reasons_file)["episodes"]}
    all_rows=[]; all_episodes=[]; folds={}
    cv2.setNumThreads(1)
    for session in sorted(fixed["cases"]):
        training=[r for name,c in fixed["cases"].items() if name!=session for r in c["per_time"] if r["imu"]]
        # Sum of signed directional agreement. Ties retain the initially assumed sign.
        agreement=float(np.sum(np.cos(np.deg2rad([r["errors"]["IMU"] for r in training]))))
        reverse=agreement<0
        native=DATA/"hololens/CAB/native_depth"/(session+".npz")
        predictions,_,_=infer_session(native,reverse)
        rows,eps=score_visit(geometry[session],predictions,old,reasons)
        folds[session]={"training_available":len(training),"train_cosine_sum":agreement,
                        "reverse":reverse,"test":summary(rows,eps)}
        all_rows.extend(rows);all_episodes.extend(eps)
    return {"design":"Exploratory same-site leave-one-recording-out choice of IMU sign only",
            "folds":folds,"nominal":summary(all_rows,all_episodes),"inputs":{
            p.name:sha(p) for p in (fixed_file,HERE/"heading.py",HERE/"eval_empirical.py",
                                     HERE/"heading_sign_loo.py")}}

if __name__=="__main__":
    with threadpool_limits(limits=1):
        result=run()
    (OUT/"sign_loo_diagnostic.json").write_text(json.dumps(result,indent=1,allow_nan=False)+"\n")
    print(result["nominal"]["fusion"])
