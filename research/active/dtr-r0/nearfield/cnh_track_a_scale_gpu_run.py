"""Scale readouts on GPU (same outputs/keys as cnh_track_a_scale_fast; analysis unchanged).

Each worker process scores whole units with cnh_track_a_gpu_readout; statistics use the
frozen cnh_track_a_scale_evaluate.analyze. Tolerance vs the CPU scorer (pilot unit 9):
B0/B1-R relative error <= 1e-13, scan arms (S1, S1cell, S2, S2r4, memory) <= 5e-7.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np


def score_unit_gpu(job):
    geometry, sensor, output, unit, mount, snr_index, family = job
    import cnh_track_a_scale_evaluate as se
    import cnh_track_a_gpu_readout2 as g
    from cnh_track_a_readout import noisy_poses
    from cnh_track_a_v13_evaluate import observability
    se.sensor_module.FAMILY = family
    target = Path(output)/f'unit{unit:02d}.npz'
    if target.exists():
        return unit
    bias = np.load(Path(output)/'bias.npy')
    split, records, step = se.unit_records(Path(geometry), Path(sensor), unit, mount, snr_index)
    scores = {}
    for rec in records:
        out = g.sequence_readouts(rec['hist'], rec['ambient'], bias, rec['poses'], rec['tq'],
                                  noisy_poses(rec['poses'], rec['ego_seed'], dt=.2), with_r4=True)
        for key, value in out.items():
            scores.setdefault(key, []).append(value)
    arrays = {k.replace('/', '__').replace('|', '@'): np.concatenate(v) for k, v in scores.items()}
    extra = dict(labels=np.concatenate([r['labels'] for r in records]), main=np.concatenate([r['main'] for r in records]),
                 boundary=np.concatenate([r['boundary'] for r in records]),
                 witness=np.concatenate([r['witness'] for r in records]), strata=se.strata_of(records),
                 config=np.repeat([r['config'] for r in records], 12), frame=np.tile(np.arange(12), len(records)),
                 split=np.array(split))
    if split == 'audit':
        extra['visibility'] = observability(Path(sensor), records, mount, 5, step=step)
    tmp = target.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays, **extra)
    tmp.replace(target)
    return unit


def main():
    import cnh_track_a_scale_evaluate as se
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--split-counts', type=int, nargs=3, required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    t0 = time.monotonic()
    ntr, nca, nau = a.split_counts
    n = ntr+nca+nau
    se.sensor_module.FAMILY = a.family
    conds = []
    for spec in a.conditions:
        name, mount, snr = spec.split(':')
        out = a.root/name
        out.mkdir(parents=True, exist_ok=True)
        si = (3, 6, 12).index(int(snr))
        if not (out/'bias.npy').exists():
            calib = []
            for u in range(ntr, ntr+nca):
                if (a.geometry/f'unit{u:02d}'/f'unit{u:02d}.json').exists():
                    calib.extend(r['hist'] for r in se.unit_records(a.geometry, a.sensor, u, int(mount), si)[1])
            np.save(out/'bias.npy', np.median(np.concatenate(calib), axis=0))
        conds.append((name, int(mount), si, out))
    jobs = [(str(a.geometry), str(a.sensor), str(out), u, mount, si, a.family)
            for _, mount, si, out in conds for u in range(ntr, n)
            if (a.geometry/f'unit{u:02d}'/f'unit{u:02d}.json').exists()]
    jobs.sort(key=lambda j: (j[3], j[2]))
    with ProcessPoolExecutor(a.workers) as pool:
        for _ in pool.map(score_unit_gpu, jobs):
            pass
    for name, mount, si, out in conds:
        report = se.analyze(out, n, dict(train=ntr, calib=nca, audit=nau), True)
        report['wall_s_pipeline'] = time.monotonic()-t0
        report['implementation'] = 'cnh_track_a_scale_gpu_run (GPU scorer within declared tolerance)'
        (out/'result.json').write_text(json.dumps(report, indent=1, allow_nan=False)+'\n', encoding='utf-8')
        print(name, json.dumps({arm: {g: round(v[g]['macro_AP'], 4) for g in ('HEAD', 'BODY')} for arm, v in report['arms'].items()}))


if __name__ == '__main__':
    main()
