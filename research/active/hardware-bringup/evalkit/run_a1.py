"""Direction A experiment 1 (DIRECTION_A1_TOF_SCALED_METRIC_DEV_20261117.md) on evaluation-kit v1.
Usage: run_a1.py <out dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from methods import A1_METHODS, Methods

REGIONS = ['near', 'near_nonground'] + [f'corridor_{b}' for b in core.BANDS]
FOV_HALF = np.deg2rad(22.)


def block_median(rows, method, region):
    per = []
    for b in sorted({r['block'] for r in rows}):
        v = [r['depth'][method][region]['absrel'] for r in rows if r['block'] == b and r['depth'][method].get(region)
             and r['depth'][method][region]['absrel'] is not None]
        if v:
            per.append(float(np.median(v)))
    return (float(np.median(per)) if per else None), len(per)


def main(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117); M = Methods(); report = dict(protocol=core.PROTOCOL_VERSION, methods=A1_METHODS)
    for name in ('aria_gen2_walks', 'sanpo_real_local'):
        gen = adapters.ADAPTERS[name]
        n_all = sum(1 for _ in gen()); k = max(1, n_all // 20)
        sc = core.self_check([f for i, f in enumerate(gen()) if i % k == 0][:20], rng)
        report[name] = dict(self_check=sc)
        if not sc['pass']:
            report[name]['status'] = 'STOPPED_AT_SELF_CHECK'; continue
        rows, t0 = [], time.time()
        for f in gen():
            reg = core.regions(f, rng)
            d = core.rays(f.K, *f.ref_depth.shape)
            fov = (np.abs(np.arctan(d[:, 0])) <= FOV_HALF) & (np.abs(np.arctan(d[:, 1])) <= FOV_HALF)
            if 'near_nonground' in reg:
                reg['nng_in_fov'] = reg['near_nonground'] & fov; reg['nng_out_fov'] = reg['near_nonground'] & ~fov
            row = dict(seq=f.seq, t=f.t, block=f.block, truth={b: reg.get(f'truth_{b}') for b in core.BANDS},
                       truth_opened={b: reg.get(f'truth_opened_{b}') for b in core.BANDS},
                       truth_dist={b: reg.get(f'truth_dist_{b}') for b in core.BANDS}, depth={}, alert={})
            for m in A1_METHODS:
                pred = getattr(M, m)(f)
                row['depth'][m] = {r: core.depth_metrics(pred, f.ref_depth, reg[r]) for r in REGIONS + ['nng_in_fov', 'nng_out_fov'] if r in reg}
                row['alert'][m] = core.band_alert(pred, f, reg)
            rows.append(row)
        summ = {m: core.summarize(rows, m) for m in A1_METHODS}
        for m in A1_METHODS:
            for r in ('nng_in_fov', 'nng_out_fov'):
                summ[m][r] = dict(zip(('absrel', 'blocks'), block_median(rows, m, r)))
        report[name].update(frames=len(rows), seconds=round(time.time() - t0), summary=summ)
        (out / f'rows_{name}.json').write_text(json.dumps(rows, default=float), encoding='utf-8')
        print(name, 'done', len(rows), flush=True)
    s = report['aria_gen2_walks']['summary']
    def prim(m):
        return (s[m]['corridor_LOW']['absrel'] + s[m]['corridor_BODY']['absrel']) / 2
    base, new = prim('tof_dav2s'), prim('tof_unidepth')
    dfa = {b: s['tof_unidepth'][f'detect_{b}']['fa_frame_rate'] - s['tof_dav2s'][f'detect_{b}']['fa_frame_rate'] for b in ('LOW', 'BODY')}
    drec = {b: s['tof_unidepth'][f'detect_{b}']['recall'] - s['tof_dav2s'][f'detect_{b}']['recall'] for b in ('LOW', 'BODY')}
    within = all(v <= .03 for v in dfa.values()) and all(v >= -.02 for v in drec.values())
    rel = new / base - 1
    verdict = ('ADOPT_TOF_UNIDEPTH' if rel <= -.15 and within else 'KEEP_TOF_DAV2S' if (rel >= .15 or not within) else 'UNCERTAIN')
    report['decision'] = dict(primary_tof_dav2s=base, primary_tof_unidepth=new, relative_change=rel, fa_delta=dfa, recall_delta=drec,
                              within_tolerance=within, verdict=verdict)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report['decision'], indent=1))
    for name in ('aria_gen2_walks', 'sanpo_real_local'):
        if 'summary' not in report[name]:
            print(name, report[name].get('status')); continue
        print('##', name)
        for m, v in report[name]['summary'].items():
            print(f"{m:18s} AbsRel nng {v['near_nonground']['absrel'] or 0:.3f} in/out FoV {v['nng_in_fov']['absrel'] or 0:.3f}/{v['nng_out_fov']['absrel'] or 0:.3f}"
                  f" | corr L/B/H " + '/'.join(f"{v[f'corridor_{b}']['absrel'] or 0:.3f}" for b in core.BANDS)
                  + ' | recall ' + '/'.join(f"{v[f'detect_{b}']['recall']:.2f}" for b in core.BANDS)
                  + ' | FA strict ' + '/'.join(f"{v[f'detect_{b}']['fa_frame_rate']:.2f}" for b in core.BANDS))


if __name__ == '__main__':
    main(sys.argv[1])
