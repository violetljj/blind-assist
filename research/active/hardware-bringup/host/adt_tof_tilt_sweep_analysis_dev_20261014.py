"""Metrics for adt_tof_tilt_sweep_dev_20261014 output (plan: adt-eval-20261013/PLAN_TILT_SWEEP.md).
Usage: adt_tof_tilt_sweep_analysis_dev_20261014.py <sweep dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2

ARMS = ('rgb', 'tof', 'fused', 'fused_veto')
STRATA = (('p_level', -90, 5), ('p_mid', 5, 20), ('p_down', 20, 90))   # 'p_' keeps them apart from the distance keys
PX_GRID, Z_GRID = (50, 75, 100, 150, 200, 300, 450, 600, 900), (1.2, 1.4, 1.6, 1.8, 2.0)


def raw_alerts(frames, key, px=150, zc=1.8):
    A = {a: np.zeros((len(frames), 2), bool) for a in ARMS}
    for i, f in enumerate(frames):
        r = f['r'][key]
        for q in (0, 1):
            bl = [b for b in r['blobs'][q] if b[1] <= zc]
            rgb = sum(b[0] for b in bl) >= px; rgbv = sum(b[0] for b in bl if not b[2]) >= px
            A['rgb'][i, q] = rgb; A['tof'][i, q] = r['tof'][q]; A['fused'][i, q] = rgb or r['tof'][q]; A['fused_veto'][i, q] = rgbv or r['tof'][q]
    return {a: np.stack([V2.confirm(m[:, 0]), V2.confirm(m[:, 1])], 1) for a, m in A.items()}


def count(seq_frames, key, arms=ARMS):
    c = {}
    for frames in seq_frames:
        A = raw_alerts(frames, key)
        for i, f in enumerate(frames):
            st = next(n for n, lo, hi in STRATA if lo <= f['pitch'] < hi)
            for q in (0, 1):
                t = f['truth'][q]
                if t['n'] == 0:
                    ks = ['fa', f'fa_{st}']
                elif t['n'] >= 150 and t['z50'] <= 1.5:
                    ks = ['rec', f'rec_{st}', 'rec_head' if q == 0 else 'rec_body', 'rec_near' if t['z50'] <= .7 else 'rec_mid']
                else:
                    continue
                for a in arms:
                    for k in ks:
                        e = c.setdefault(a, {}).setdefault(k, [0, 0]); e[0] += int(A[a][i, q]); e[1] += 1
    return {a: {k: dict(n=v, rate=round(v[0] / v[1], 4)) for k, v in d.items()} for a, d in c.items()}


def rgb_sweep(seq_frames, key):
    out = []
    for px in PX_GRID:
        for zc in Z_GRID:
            fa = [0, 0]; rc = [0, 0]
            for frames in seq_frames:
                a = raw_alerts(frames, key, px, zc)['rgb']
                for i, f in enumerate(frames):
                    for q in (0, 1):
                        t = f['truth'][q]
                        if t['n'] == 0:
                            fa[0] += int(a[i, q]); fa[1] += 1
                        elif t['n'] >= 150 and t['z50'] <= 1.5:
                            rc[0] += int(a[i, q]); rc[1] += 1
            out.append((px, zc, fa[0] / fa[1], rc[0] / rc[1]))
    best = {}
    for r in (.80, .85, .90):
        ok = [c for c in out if c[3] >= r]; best[str(r)] = min(ok, key=lambda c: c[2]) if ok else None
    return best


def main(sweep, out):
    seq_frames = []; sw = Path(sweep)
    tilts = sorted({int(p.name.split('.')[-2][1:]) for p in sw.glob('*.i*.json')})
    for p in sorted(sw.glob('*.truth.json')):
        frames = json.loads(p.read_text(encoding='utf8'))
        for f in frames:
            f['r'] = {}
        for v in ('i', 'd'):
            for t in tilts:
                for f, r in zip(frames, json.loads((sw / p.name.replace('.truth.', f'.{v}{t}.')).read_text(encoding='utf8'))):
                    f['r'][f'{v}{t}'] = r
        seq_frames.append(frames)
    if len(seq_frames) < 13:
        print('incomplete', len(seq_frames)); return
    res = {}
    for v in ('i', 'd'):
        for t in tilts:
            key = f'{v}{t}'; res[key] = count(seq_frames, key); res[key]['rgb_sweep'] = rgb_sweep(seq_frames, key)
    for v, name in (('i', 'ideal ToF'), ('d', 'degraded ToF')):
        print(f'\n== {name}: FA / recall (2-of-3 query-frames); fused_veto by head pitch stratum (current-mount ToF axis below horizontal)')
        print('tilt | RGB FA/rec | RGB FA@rec>=.80/.85 | ToF FA/rec | fused FA/rec | fused_veto FA/rec (HEAD/BODY, near/mid) | level<5: FA/rec | 5-20 | >20')
        for t in tilts:
            r = res[f'{v}{t}']; sw = r['rgb_sweep']
            f = lambda a, k: r[a][k]['rate'] * 100
            fs = lambda c: '  -  ' if c is None else f'{c[2]*100:4.1f}'
            print(f"{t:3d} | {f('rgb','fa'):4.1f}/{f('rgb','rec'):4.1f} | {fs(sw['0.8'])}/{fs(sw['0.85'])} | {f('tof','fa'):4.1f}/{f('tof','rec'):4.1f} | "
                  f"{f('fused','fa'):4.1f}/{f('fused','rec'):4.1f} | {f('fused_veto','fa'):4.1f}/{f('fused_veto','rec'):4.1f} "
                  f"({f('fused_veto','rec_head'):4.1f}/{f('fused_veto','rec_body'):4.1f}, {f('fused_veto','rec_near'):4.1f}/{f('fused_veto','rec_mid'):4.1f}) | "
                  + ' | '.join(f"{f('fused_veto','fa_'+s):4.1f}/{f('fused_veto','rec_'+s):4.1f}" for s, _, _ in STRATA))
        base_fa = res[f'{v}0']['fused']['fa']['rate']
        ok = [t for t in tilts if res[f'{v}{t}']['fused_veto']['fa']['rate'] <= base_fa]
        best = max(res[f'{v}{t}']['fused_veto']['rec']['rate'] for t in ok) if ok else None
        rec = [t for t in ok if best is not None and res[f'{v}{t}']['fused_veto']['rec']['rate'] >= best - .01]
        res[f'choice_{v}'] = dict(tilt0_fused_fa=base_fa, eligible=ok, best_recall=best, recommended=rec)
        print(f'{name}: tilt-0 fused FA {base_fa*100:.2f}%; eligible tilts {ok}; recommended (within 1 pt of best {best*100:.1f}%) {rec}')
    n = {s: sum(1 for fr in seq_frames for f in fr if lo <= f['pitch'] < hi) for s, lo, hi in STRATA}
    print('frames per stratum', n)
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
