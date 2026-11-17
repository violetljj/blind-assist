"""Post-hoc (after the A1 recheck run): frames without output per method, and the A1 detection comparison re-counted
(a) on frames where both arms have output (paired) and (b) treating "no output" as "no alert".
The frozen rules counted each arm on its own output frames; this script only describes. Usage: a1_paired_posthoc.py <rows json>...
"""
import json
import sys

PAIRS = (('tof_dav2s', 'tof_unidepth'), ('tof_dav2s_out', 'tof_unidepth_out'))


def count(rows, m, band, keep, none_as_false=False):
    tp = fn = fp = tn = 0
    for r in rows:
        if not keep(r):
            continue
        t, a = r['truth'].get(band), r['alert'][m].get(band)
        if t is None or t < 0:
            continue
        if a is None:
            if not none_as_false:
                continue
            a = False
        if t == 1:
            tp += a; fn += (not a)
        else:
            fp += a; tn += (not a)
    return dict(tp=tp, pos=tp + fn, fp=fp, neg=fp + tn)


for path in sys.argv[1:]:
    rows = json.load(open(path, encoding='utf-8'))
    print('##', path, len(rows), 'frames')
    for m in rows[0]['alert']:
        print(f'  {m:18s} no output: {sum(1 for r in rows if r["alert"][m]["LOW"] is None)}')
    for a, b in PAIRS:
        for band in ('LOW', 'BODY'):
            both = lambda r: r['alert'][a][band] is not None and r['alert'][b][band] is not None
            out = []
            for label, keep, naf in (('paired', both, False), ('none=no alert', lambda r: True, True)):
                ca, cb = count(rows, a, band, keep, naf), count(rows, b, band, keep, naf)
                out.append(f"{label}: {a} rec {ca['tp']}/{ca['pos']} FA {ca['fp']}/{ca['neg']} | {b} rec {cb['tp']}/{cb['pos']} FA {cb['fp']}/{cb['neg']}")
            print(f'  {band}:', ' || '.join(out))


def paired_absrel(rows, a, b, region):
    """Block-median AbsRel of both arms on frames where both have a value for the region."""
    import numpy as np
    per = {a: [], b: []}
    for blk in sorted({r['block'] for r in rows}):
        va, vb = [], []
        for r in rows:
            if r['block'] != blk:
                continue
            ma, mb = r['depth'][a].get(region), r['depth'][b].get(region)
            if ma and mb and ma['absrel'] is not None and mb['absrel'] is not None:
                va.append(ma['absrel']); vb.append(mb['absrel'])
        if va:
            per[a].append(np.median(va)); per[b].append(np.median(vb))
    return len(per[a]), (float(np.median(per[a])) if per[a] else None), (float(np.median(per[b])) if per[b] else None)


for path in sys.argv[1:]:
    rows = json.load(open(path, encoding='utf-8'))
    print('## paired AbsRel (blocks, base, new)', path)
    for a, b in PAIRS:
        for region in ('corridor_LOW', 'corridor_BODY', 'near_nonground', 'nng_in_fov', 'nng_out_fov'):
            n, x, y = paired_absrel(rows, a, b, region)
            print(f'  {a:14s} vs {b:16s} {region:15s} blocks {n:3d}  {x if x is None else round(x, 3)}  {y if y is None else round(y, 3)}')
