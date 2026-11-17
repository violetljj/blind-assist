"""Where do truth LOW-corridor pixels land under a method's prediction (DIRECTION_A_INDOOR_ADT_DEV_20261118 follow-up)?
For LOW-positive frames by discordance group: share of truth LOW pixels predicted inside the LOW band and corridor,
above / below the band, beyond 2.5 m or outside laterally; predicted vs truth LOW area (alert needs >= MIN_AREA after
a 3x3 opening). Evaluator-side diagnosis on development data.
Usage: diag_adt_low_where.py <rows json> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from diag_adt_low_scale import group
from methods import Methods
from run_confirm import PIXELS

ARMS = ('tof_dav2s', 'tof_unidepth', 'unidepth_s')


def main(rows_path, out_path):
    rows = json.loads(Path(rows_path).read_text(encoding='utf-8'))
    want = {(r['seq'], round(r['t'], 3)): group(r) for r in rows if r['truth']['LOW'] == 1}
    rng = np.random.default_rng(0); hit = [k for k, g in want.items() if g == 'both_hit']
    keep_hit = {tuple(x) for x in np.array(hit, dtype=object)[rng.permutation(len(hit))[:150]]}
    M = Methods(); M.set_unidepth_pixels(PIXELS); res = []
    for f in adapters.adt_apartment():
        k = (f.seq, round(f.t, 3)); g = want.get(k)
        if g is None or (g == 'both_hit' and k not in keep_hit):
            continue
        reg = core.regions(f, rng); h, w = f.ref_depth.shape; d = core.rays(f.K, h, w)
        tl = reg['corridor_LOW']; ref = f.ref_depth.reshape(-1)
        t_area = float(core.pixel_area(f.K, h, w, np.where(np.isfinite(ref), ref, 0))[tl].sum())
        row = dict(group=g, truth_area=t_area, truth_px=int(tl.sum()))
        for m in ARMS:
            p = getattr(M, m)(f).reshape(-1); ok = np.isfinite(p) & (p > .05)
            pts = d * np.where(ok, p, 0)[:, None]
            above, along, planar = core._corridor(pts, ok, reg['axes'], reg['ground'][1])
            lo, hi = core.BANDS['LOW']; sel = tl & ok
            inband = planar & (above >= lo) & (above < hi)
            row[m] = dict(in_low=float((sel & inband).sum() / max(1, sel.sum())),
                          above=float((sel & (above >= hi)).sum() / max(1, sel.sum())),
                          below=float((sel & (above < lo)).sum() / max(1, sel.sum())),
                          outside=float((sel & ~planar).sum() / max(1, sel.sum())),
                          pred_area=float(core.pixel_area(f.K, h, w, np.where(ok, p, 0))[core._open(inband, h, w)].sum()),
                          height_err_med=float(np.median(above[sel] - above_truth[sel])) if (above_truth := core._corridor(
                              d * np.where(np.isfinite(ref), ref, 0)[:, None], np.isfinite(ref), reg['axes'], reg['ground'][1])[0]) is not None and sel.any() else None)
        res.append(row)
    Path(out_path).write_text(json.dumps(res, indent=0), encoding='utf-8')
    for g in ('dav2s_only', 'hybrid_only', 'both_hit', 'both_miss'):
        rs = [r for r in res if r['group'] == g]
        print(g, len(rs), 'truth area med %.4f' % np.median([r['truth_area'] for r in rs]))
        for m in ARMS:
            med = lambda key: round(float(np.median([r[m][key] for r in rs if r[m][key] is not None])), 3)
            print('   %-13s in_low %.3f above %.3f below %.3f outside %.3f pred_area %.4f height_err %+.3f' % (
                m, med('in_low'), med('above'), med('below'), med('outside'), med('pred_area'), med('height_err_med')))


if __name__ == '__main__':
    main(*sys.argv[1:3])
