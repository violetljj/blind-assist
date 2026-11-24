"""Compare stopping policies using the original features and M3 source folds."""
import os
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from policy import solve, QVIAlertPolicy

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"m6_evt_eprocess_risk"))
import replay


def evaluate(version):
    clips,stream=replay.materialize(version)
    _,adv,intervals=solve()
    times=[]
    unadjusted=[]
    for c in clips:
        policy=QVIAlertPolicy(adv)
        scores=[]
        observable=(c.raw,c.bev_d,c.obs["pred"],c.obs["sigma"])
        for point in zip(*observable,strict=True):
            begin=time.perf_counter_ns()
            scores.append(policy.step(*point))
            times.append((time.perf_counter_ns()-begin)*1e-6)
        c.stop=np.asarray(scores)
        unadjusted.append(c.count(c.stop>=0))
    prior=replay.OUT.parent/"m6_evt_eprocess_risk"/("v5_empirical_report.json" if version=="v4" else "v5_empirical_report_v41.json")
    baseline=json.loads(prior.read_text())["baseline"] if prior.exists() else replay.baselines(clips)
    return dict(version=version,code_sha256={p.name:replay.digest(p) for p in (
                    Path(__file__),HERE/"policy.py",Path(replay.__file__),replay.M3/"eval_alert_operating_point.py")},
                manifest_sha256=replay.digest(stream/"manifest.json"),
                stream_input_identity="per-source raw hashes and model hash in identified stream manifest",
                baseline=baseline,table_intervals=intervals,unadjusted=replay.score(sum(unadjusted)),
                crossfit=replay.crossfit(clips,"stop"),curve=replay.frontier(clips,"stop"),
                cpu_ms_p50_p95=np.percentile(times,[50,95]).tolist(),frames=len(times),
                support="all eligible frames; policy emits scores and makes no new distance estimate")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--version",choices=("v4","v41"),default="v4")
    version=ap.parse_args().version
    out=evaluate(version)
    path=replay.OUT.parent/HERE.name/("v5_empirical_report.json" if version=="v4" else "v5_empirical_report_v41.json")
    replay.save(path,out)
    print(version,out["unadjusted"],out["crossfit"]["pooled"],out["curve"]["exact_best_at_5_clear_per_km"])
