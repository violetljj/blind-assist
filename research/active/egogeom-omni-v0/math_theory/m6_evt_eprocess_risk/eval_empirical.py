"""Four-domain held-out evaluation of bounded geometry and distance-block bets."""
import argparse
import os
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import time
from pathlib import Path

import numpy as np
import replay
from risk import DistanceBlockBet, InvariantEProcessBEV

HERE = Path(__file__).resolve().parent
FIELDS = ("depth_s4","dh_s4","nrm_s4","ls_s4","K","poses_R","poses_t","strict_fwd")


def evaluate(version):
    clips, stream = replay.materialize(version)
    files = {}
    times, bet_time = [], []
    for c in clips:
        path = stream / c.obs["domain"] / (c.obs["name"]+".npz")
        files[str(path.relative_to(stream))] = replay.digest(path)
        with np.load(path) as pack:
            arrays = [pack[k] for k in FIELDS]
        model = InvariantEProcessBEV()
        scores = []
        for args in zip(*arrays, strict=True):
            begin = time.perf_counter_ns()
            scores.append(model.step(*args))
            times.append((time.perf_counter_ns()-begin)*1e-6)
        c.geometry = np.asarray(scores)
        # Observable odometry, never the truth coverage or walker path increments.
        xy = arrays[6][:,:2].astype(float)
        c.odometry = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    base = replay.baselines(clips)
    if version == "v4":
        assert (base["confirmed_peak"]["detected_at_most"],base["confirmed_shared"]["pooled"]["timely"],
                base["device_cost"]["timely"]) == (13,6,70)
    pulses, calibration = [], []
    for f in (0,1):
        train = [c for c in clips if c.fold != f]
        test = [c for c in clips if c.fold == f]
        clear = np.concatenate([c.geometry[c.clear] for c in train])
        cut = float(np.quantile(clear,.90,method="higher"))
        calibration.append(dict(test_fold=f, clear_cut_strictly_greater=cut, train_clear=len(clear),
                                train_marginal_hit_rate=float((clear>cut).mean())))
        for c in test:
            bet = DistanceBlockBet()
            alerts = []
            for s,m in zip(c.geometry,c.odometry):
                start = time.perf_counter_ns()
                _,alarm = bet.step(s>cut,m)
                bet_time.append((time.perf_counter_ns()-start)*1e-6)
                alerts.append(alarm)
            pulses.append((c, c.count(np.asarray(alerts))))
    return dict(version=version, streams=len(clips), frames=len(times), sources=len({c.source for c in clips}),
                code_sha256={p.name:replay.digest(p) for p in (Path(__file__),HERE/"risk.py",HERE/"replay.py",
                    replay.M3/"eval_alert_operating_point.py",replay.M3/"diagnose_clear_peak_ceiling.py",
                    replay.M3/"eval_on_v4.py",HERE.parents[2]/"egoclear-v0"/"eval_closed_loop.py")},
                stream_sha256=files, manifest_sha256=replay.digest(stream/"manifest.json"),
                baseline=base, geometry=replay.crossfit(clips,"geometry"),
                pooled_peak=replay.ceiling(clips,"geometry"),
                aria_peak=replay.ceiling([c for c in clips if c.device=="aria"],"geometry"),
                exact_geometry_frontier=replay.frontier(clips,"geometry"),
                aria_frontier=replay.frontier([c for c in clips if c.device=="aria"],"geometry"),
                conditional_bet=dict(pooled=replay.score(sum(x for _,x in pulses)),calibration=calibration,
                    domains={dom:replay.score(sum(x for c,x in pulses if c.obs["domain"]==dom))
                             for dom in sorted({c.obs["domain"] for c,_ in pulses})}),
                geometry_cpu_ms_p50_p95=np.percentile(times,[50,95]).tolist(),
                bet_cpu_ms_p50_p95=np.percentile(bet_time,[50,95]).tolist(),
                support="all 5,547 eligible frames scored; no new metric distance predicted")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--version",choices=("v4","v41"),default="v4")
    version=parser.parse_args().version
    report=evaluate(version)
    out=replay.OUT.parent/HERE.name
    name="v5_empirical_report.json" if version=="v4" else "v5_empirical_report_v41.json"
    replay.save(out/name,report)
    print(version,report["exact_geometry_frontier"]["exact_best_at_5_clear_per_km"],
          report["geometry"]["pooled"],report["conditional_bet"]["pooled"],flush=True)
