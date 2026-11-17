"""Diagnosis of the indoor LOW-band loss (DIRECTION_A_INDOOR_ADT_DEV_20261118.md, follow-up): on LOW-positive frames,
the one ToF scale versus the scale each region would need (median ref / raw UniDepth over the region's truth pixels),
and the depth dependence of ref / raw over all valid pixels. Evaluator-side diagnosis, development data.
Usage: diag_adt_low_scale.py <rows json> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from methods import Methods
from run_confirm import PIXELS

A, B = 'tof_dav2s', 'hybrid_in'


def group(r):
    a, b = r['alert'][A]['LOW'], r['alert'][B]['LOW']
    return 'dav2s_only' if a and not b else 'hybrid_only' if b and not a else 'both_hit' if a else 'both_miss'


def main(rows_path, out_path):
    rows = json.loads(Path(rows_path).read_text(encoding='utf-8'))
    want = {(r['seq'], round(r['t'], 3)): group(r) for r in rows if r['truth']['LOW'] == 1}
    rng = np.random.default_rng(0)
    hit = [k for k, g in want.items() if g == 'both_hit']; keep_hit = set(map(tuple, rng.permutation(np.array(hit, dtype=object))[:200]))
    M = Methods(); M.set_unidepth_pixels(PIXELS); res = []
    bins = np.array([.5, 1., 1.5, 2., 2.5, 3., 4., 6.])
    for f in adapters.adt_apartment():
        k = (f.seq, round(f.t, 3)); g = want.get(k)
        if g is None or (g == 'both_hit' and k not in keep_hit):
            continue
        reg = core.regions(f, rng); raw = M._unidepth_cached(f).reshape(-1); ref = f.ref_depth.reshape(-1)
        ok = np.isfinite(raw) & (raw > .05) & np.isfinite(ref)
        rat = lambda m: float(np.median(ref[m & ok] / raw[m & ok])) if (m & ok).sum() >= 50 else None
        d = core.rays(f.K, *f.ref_depth.shape); pts = d * np.where(np.isfinite(ref), ref, 0)[:, None]
        above, _, _ = core._corridor(pts, np.isfinite(ref), reg['axes'], reg['ground'][1])
        ground = np.isfinite(ref) & (np.abs(above) < .03) & (ref < 3)
        db = np.digitize(ref, bins)
        res.append(dict(group=g, seq=f.seq, t=f.t, s_tof=M._tof_scale(f, False), s_low=rat(reg['corridor_LOW']),
                        s_body=rat(reg['corridor_BODY']), s_ground=rat(ground),
                        by_depth={f'{bins[i - 1]}-{bins[i]}': rat(db == i) for i in range(1, len(bins))}))
    Path(out_path).write_text(json.dumps(res, indent=0), encoding='utf-8')
    for g in ('dav2s_only', 'hybrid_only', 'both_hit', 'both_miss'):
        rs = [r for r in res if r['group'] == g]
        med = lambda key: float(np.median([r[key] for r in rs if r[key] is not None])) if rs else None
        rel = [r['s_tof'] / r['s_low'] for r in rs if r['s_tof'] and r['s_low']]
        print(g, len(rs), 's_tof %.3f s_low %.3f s_body %.3f s_ground %.3f' % (med('s_tof'), med('s_low'), med('s_body'), med('s_ground')),
              's_tof/s_low q10/50/90', np.round(np.percentile(rel, [10, 50, 90]), 3).tolist() if rel else None)
        print('   ref/raw by depth', {b: round(float(np.median([r['by_depth'][b] for r in rs if r['by_depth'][b]])), 3)
                                       for b in rs[0]['by_depth'] if any(r['by_depth'][b] for r in rs)})


if __name__ == '__main__':
    main(*sys.argv[1:3])
