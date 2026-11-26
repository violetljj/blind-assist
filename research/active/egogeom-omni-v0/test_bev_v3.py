"""Verify EgoGeomTemporalBEVV3 parity with v2.3 + M6 and benchmark CPU latency & 38-source cross-fit."""
from __future__ import annotations

import json
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import sys
import time
from pathlib import Path

import numpy as np

ROUTE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROUTE), str(ROUTE / "math_theory/m6_evt_eprocess_risk")]

from egogeom_temporal_bev import EgoGeomTemporalBEV
from egogeom_temporal_bev_v3 import EgoGeomTemporalBEVV3
import replay
from risk import InvariantEProcessBEV

FIELDS = ("depth_s4", "dh_s4", "nrm_s4", "ls_s4", "K", "poses_R", "poses_t", "strict_fwd")


def verify_version(version: str) -> dict:
    clips, stream = replay.materialize(version)
    max_d_diff, max_hyb_diff, max_geom_diff = 0.0, 0.0, 0.0
    t_v3, t_m6_only = [], []

    for c in clips:
        path = stream / c.obs["domain"] / (c.obs["name"] + ".npz")
        with np.load(path) as pack:
            arrays = [pack[k] for k in FIELDS]
        v3 = EgoGeomTemporalBEVV3()
        v23 = EgoGeomTemporalBEV()
        m6 = InvariantEProcessBEV()

        g_scores = []
        for args in zip(*arrays, strict=True):
            t0 = time.perf_counter_ns()
            d3, rb3, rh3, rg3 = v3.step(*args)
            t_v3.append((time.perf_counter_ns() - t0) * 1e-6)

            d2, rb2, rh2 = v23.step(*args)
            t1 = time.perf_counter_ns()
            rg_ref = m6.step(*args)
            t_m6_only.append((time.perf_counter_ns() - t1) * 1e-6)

            if np.isfinite(d3) and np.isfinite(d2):
                max_d_diff = max(max_d_diff, abs(d3 - d2))
            else:
                assert np.isinf(d3) == np.isinf(d2)
            max_hyb_diff = max(max_hyb_diff, abs(rh3 - rh2))
            max_geom_diff = max(max_geom_diff, abs(rg3 - rg_ref))
            g_scores.append(rg3)
        c.v3_geom = np.asarray(g_scores)

    cf = replay.crossfit(clips, "v3_geom")
    ceil_all = replay.ceiling(clips, "v3_geom")
    res = {
        "version": version,
        "frames": len(t_v3),
        "max_diff_dist_m": max_d_diff,
        "max_diff_hyb_v23": max_hyb_diff,
        "max_diff_geom_m6": max_geom_diff,
        "v3_total_cpu_ms_p50_p95": [round(float(x), 4) for x in np.percentile(t_v3, [50, 95])],
        "m6_only_cpu_ms_p50_p95": [round(float(x), 4) for x in np.percentile(t_m6_only, [50, 95])],
        "crossfit_pooled": cf["pooled"],
        "ceiling_all": ceil_all,
    }
    assert max_d_diff < 1e-6 and max_hyb_diff < 1e-6 and max_geom_diff < 1e-6, res
    return res


if __name__ == "__main__":
    out = {}
    for ver in ("v4", "v41"):
        out[ver] = verify_version(ver)
        print(f"=== BEV v3 [{ver}] ===", json.dumps(out[ver], indent=2), flush=True)
    dst = ROUTE.parents[2] / "artifacts.local/work/egogeom-omni-v0/bev_v3_verification.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {dst}", flush=True)
