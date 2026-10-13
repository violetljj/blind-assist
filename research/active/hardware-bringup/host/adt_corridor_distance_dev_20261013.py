"""GT corridor distance per prepared ADT frame (CPU, parallel over sequences; plan PLAN_LEADTIME.md).

d = 5th percentile of z of GT points with |x| <= 0.30, y in [-0.20, 0.90], 0.3 <= z <= 6 m (>= 150 px), else inf.
Usage: adt_corridor_distance_dev_20261013.py <prep dir> <out json>
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def one(seq):
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    with np.load(seq / 'frames.npz') as Z:
        dep, R, t = Z['depth'], Z['R_cc'], Z['t_ns']
    vv, uu = np.mgrid[0:h:2, 0:w:2]                                   # every 2nd pixel; 150 px threshold -> 38
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
    out = []
    for i in range(len(t)):
        z = dep[i, ::2, ::2].reshape(-1) / 1000.; ok = z > .1
        p = (dirs[ok] * z[ok, None]) @ R[i].T
        m = (np.abs(p[:, 0]) <= .3) & (p[:, 1] >= -.2) & (p[:, 1] <= .9) & (p[:, 2] >= .3) & (p[:, 2] <= 6)
        out.append(float(np.percentile(p[m, 2], 5)) if m.sum() >= 38 else None)
    return seq.name, dict(t_ns=t.tolist(), d=out)


if __name__ == '__main__':
    seqs = sorted(p for p in Path(sys.argv[1]).iterdir() if (p / 'frames.npz').exists())
    with ProcessPoolExecutor(7) as ex:
        res = dict(ex.map(one, seqs))
    Path(sys.argv[2]).write_text(json.dumps(res)); print('done', len(res))
