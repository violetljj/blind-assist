"""Pre-specified CPU/GPU readout concordance for a scale cohort (v3 addendum rule).

Rescores the given audit units with the frozen CPU scorer and compares every saved
score array with the GPU output: relative error = max |gpu - cpu| / max |cpu| per key.
Fails if any scan-type key exceeds 1e-5 or B0/B1-R exceeds 1e-10.
"""
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import cnh_track_a_scale_fast as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--condition', default='primary-mount-10-snr6')
    p.add_argument('--units', type=int, nargs='+', required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    gpu = a.evidence/'readouts-gpu'/a.condition
    out = a.evidence/'readouts-cpu-check'/a.condition
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(gpu/'bias.npy', out/'bias.npy')
    units = [u for u in a.units if (a.evidence/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json').exists()]
    jobs = [(str(a.evidence/'geometry'), str(a.evidence/'sensor'), str(out), u, -10, 1, a.family, True) for u in units]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(sf.score_unit_fast, jobs))
    worst = {}
    for u in units:
        with np.load(gpu/f'unit{u:02d}.npz') as g, np.load(out/f'unit{u:02d}.npz') as c:
            for k in c.files:
                if c[k].dtype.kind != 'f' or k == 'witness' or k == 'visibility':
                    continue
                rel = float(np.max(np.abs(g[k]-c[k]))/max(np.max(np.abs(c[k])), 1e-12))
                worst[k] = max(worst.get(k, 0.), rel)
    exact = {k: v for k, v in worst.items() if k.startswith('B0') or k.startswith('B1-R')}
    scan = {k: v for k, v in worst.items() if k not in exact}
    ok = all(v <= 1e-10 for v in exact.values()) and all(v <= 1e-5 for v in scan.values())
    report = dict(units=units, pass_gate=ok, max_rel_B0_B1R=max(exact.values()), max_rel_scan=max(scan.values()), per_key=worst)
    (out/'concordance.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('units', 'pass_gate', 'max_rel_B0_B1R', 'max_rel_scan')}))


if __name__ == '__main__':
    main()
