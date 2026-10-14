"""Task-aligned metric and speed-aware warning distance on the ADT tilt-sweep outputs (plan: PLAN_TASK_METRIC.md).

Truth per query-frame from GT speed v: threat = >= 150 in-box GT px with z50 <= d_warn(v) = 0.5 + 2.0 v (0.7 m when
v < 0.3 m/s); empty = 0 px; occupied-non-threat = other >= 150 px frames. Rules: v6 (blob z50 <= 1.8 m), v6s
(z50 <= clip(0.8 + 2.0 v, 0.8, 1.8) with GT speed), v6w (same with v = 1.0 if GT speed >= 0.3 else 0). Arms: rgb and
fused (ToF alarm of the sweep unchanged); 2-of-3 confirmation.
Usage: adt_task_metric_dev_20261014.py <sweep dir> <prep dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2

KEYS = ('i0', 'd0', 'i30', 'd30')


def d_warn(v):
    return np.where(v < .3, .7, .5 + 2.0 * v)


def zcut(v):
    return np.clip(.8 + 2.0 * v, .8, 1.8)


def main(sweep, prep, out):
    sw = Path(sweep); res = {}
    seqs = sorted(p.name[:-len('.truth.json')] for p in sw.glob('*.truth.json'))
    data = []
    for s in seqs:
        truth = json.loads((sw / f'{s}.truth.json').read_text(encoding='utf8'))
        v = np.nan_to_num(np.load(Path(prep) / s / 'poses.npz')['speed'], nan=0.)
        data.append((s, truth, v, {k: json.loads((sw / f'{s}.{k}.json').read_text(encoding='utf8')) for k in KEYS}))
    for key in KEYS:
        for rule in ('v6', 'v6s', 'v6w'):
            c = {f'{a}_{m}': [0, 0] for a in ('rgb', 'fused') for m in ('threat', 'empty', 'nuisance', 'threat_walk', 'empty_still', 'empty_walk')}
            for s, truth, v, R in data:
                vr = {'v6': None, 'v6s': v, 'v6w': np.where(v >= .3, 1., 0.)}[rule]
                zc = np.full(len(v), 1.8) if vr is None else zcut(vr)
                raw = {a: np.zeros((len(v), 2), bool) for a in ('rgb', 'fused')}
                for i, f in enumerate(R[key]):
                    for q in (0, 1):
                        rgb = sum(b[0] for b in f['blobs'][q] if b[1] <= zc[i]) >= 150
                        raw['rgb'][i, q] = rgb; raw['fused'][i, q] = rgb or f['tof'][q]
                for a in raw:
                    A = np.stack([V2.confirm(raw[a][:, 0]), V2.confirm(raw[a][:, 1])], 1)
                    for i, t in enumerate(truth):
                        dw = d_warn(v[i]); walk = v[i] >= .3
                        for q in (0, 1):
                            tq = t['truth'][q]
                            if tq['n'] == 0:
                                ks = ['empty', 'empty_walk' if walk else 'empty_still']
                            elif tq['n'] >= 150:
                                ks = (['threat'] + (['threat_walk'] if walk else [])) if tq['z50'] <= dw else ['nuisance']
                            else:
                                continue
                            for k in ks:
                                c[f'{a}_{k}'][0] += int(A[i, q]); c[f'{a}_{k}'][1] += 1
            res[f'{key}/{rule}'] = {k: dict(n=x, rate=round(x[0] / max(x[1], 1), 4)) for k, x in c.items()}
            r = res[f'{key}/{rule}']; p = lambda k: r[k]['rate'] * 100
            print(f"{key:4s} {rule:4s} | RGB threat rec {p('rgb_threat'):5.1f} (walk {p('rgb_threat_walk'):5.1f}) empty FA {p('rgb_empty'):4.1f} "
                  f"(still {p('rgb_empty_still'):4.1f} walk {p('rgb_empty_walk'):4.1f}) nuisance {p('rgb_nuisance'):5.1f} | fused threat {p('fused_threat'):5.1f} "
                  f"empty FA {p('fused_empty'):4.1f} nuisance {p('fused_nuisance'):5.1f}", flush=True)
    r = res['i0/v6']
    print('denominators (i0): threat', r['rgb_threat']['n'][1], 'threat_walk', r['rgb_threat_walk']['n'][1], 'empty', r['rgb_empty']['n'][1],
          'empty_still', r['rgb_empty_still']['n'][1], 'nuisance', r['rgb_nuisance']['n'][1])
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
