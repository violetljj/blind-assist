"""Re-emulate the tilt-sweep ToF alarms with their depth, so the walk/still rule can also gate the ToF arm
(plan: adt-eval-20261013/PLAN_TASK_METRIC_TOF.md).

Same zone sampling, degraded seeds and alarm rule as adt_tof_tilt_sweep_dev_20261014 (a usable zone with >= 13
GT-valid samples of which >= 30 % return inside the box). Added per box: the alarm depth = the smallest, over alarming
zones, of the median corridor depth z of that zone's in-box returning samples. No DAV2 / blob readout is recomputed.
Output: <out>/<seq>.<key>.tofz.json = [[ [alarm, z], [alarm, z] ] per frame]. Usage: <prep dir> <out dir> <n procs>
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
from adt_tof_tilt_sweep_dev_20261014 import rx, BOXES

KEYS = ('i0', 'd0', 'i30', 'd30')


def one(args):
    prep, out, si, seq = args
    if all((Path(out) / f'{seq}.{k}.tofz.json').exists() for k in KEYS):
        return seq
    meta = json.loads((Path(prep) / seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    with np.load(Path(prep) / seq / 'frames.npz') as Z:
        depth, R = Z['depth'], Z['R_cc']
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays0 = V1.X.zone_rays()
    for key in KEYS:
        v, t = key[0], int(key[1:])
        rc = (rays0 @ rx(-t).T) @ Rct.T; un = rc / np.linalg.norm(rc, axis=-1, keepdims=True)
        zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
        zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h); zu, zv = np.clip(zu, 0, w - 1), np.clip(zv, 0, h - 1)
        rows = []
        for i in range(len(depth)):
            zz = (depth[i] / 1000.)[zv, zu]; ok0 = zin & (zz > .1)
            rr = np.where(ok0, zz / np.maximum(un[..., 2], 1e-6), np.nan)
            if v == 'i':
                use = np.ones(64, bool); ret = ok0
            else:
                g = np.random.default_rng([20261014, si, i, t])
                use = g.random(64) >= .25; rr = rr * (1 + .04 * g.standard_normal(64))[:, None]
                ret = ok0 & use[:, None] & (np.nan_to_num(rr, nan=99.) < 3.)
            nv = ok0.sum(1); sp = (un * np.nan_to_num(rr)[..., None]) @ R[i].T; row = []
            for lo, hi in BOXES:
                inb = ret & (np.abs(sp[..., 0]) <= .3) & (sp[..., 1] >= lo) & (sp[..., 1] <= hi) & (sp[..., 2] >= .3) & (sp[..., 2] <= 2.)
                al = use & (nv >= 13) & (inb.sum(1) >= .3 * np.maximum(nv, 1))
                z = min((float(np.median(sp[k][inb[k], 2])) for k in np.flatnonzero(al)), default=None)
                row.append([bool(al.any()), None if z is None else round(z, 3)])
            rows.append(row)
        (Path(out) / f'{seq}.{key}.tofz.json').write_text(json.dumps(rows), encoding='utf8')
    return seq


def main(prep, out, n):
    Path(out).mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in Path(prep).iterdir() if (p / 'frames.npz').exists())
    with Pool(int(n)) as pool:
        for s in pool.imap_unordered(one, [(prep, out, si, s) for si, s in enumerate(seqs)]):
            print('done', s, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
