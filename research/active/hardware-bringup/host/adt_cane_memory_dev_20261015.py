"""Causal short-term ground memory: can the device tell that the cane will find an obstacle first? (EXPLORE, 2026-10-15)

A head-level camera cannot see the floor just ahead of the feet, but while walking it saw that floor 1-3 s earlier.
Estimator (fixed before results): fuse ground-band points (0.05-0.30 m above floor) from the PAST 3 s only (current frame
included) with the frame poses, 3 cm voxels, corridor |x|<=0.30 m, forward 0..d_body+0.15 m, >=3 voxels => "cane finds it
first" => suppress the v6 alert of that frame. Same band/margin/voxel rule as the truth labels (adt_cane_reach_dev).
Arms:  gt     : GT depth for both memory and d_body (visibility bound of causal memory; GT poses)
       chain  : the device's own anchored DAV2 depth (adt_chain_depth_dump) for memory and d_body; GT poses
Truth: cane_reach labels (+-10 s GT), strata walking / walking without manipulation as the summary script.
Usage: adt_cane_memory_dev_20261015.py <prep> <adt_eval.json> <cane_reach.json> <adt data> <out.json> [arms...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adt_cane_reach_dev_20261015 as M
import adt_cane_reach_summary_dev_20261015 as SUM
import frozen_fusion_eval_v2 as V2

PAST_S = 3.0


def predict(args):
    seq, floor, arm = args
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    S = M.STEP
    vv, uu = np.mgrid[0:h:S, 0:w:S]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        t_ns = Z['t_ns']
        z_all = Z['depth'][:, ::S, ::S].astype(np.float32) / 1000. if arm == 'gt' else None
    if arm == 'chain':
        with np.load(seq / 'chain_depth_s2.npz') as Z:
            z_all = Z['z'].astype(np.float32)
    T = np.load(seq / 'poses.npz')['T_sc']; n = len(t_ns); t = (t_ns - t_ns[0]) / 1e9
    scene, ground = [], []
    for j in range(n):
        z = z_all[j].reshape(-1); ok = np.isfinite(z) & (z > .1) & (z < 10)
        P = (T[j, :3, :3] @ (dirs[ok] * z[ok, None]).T).T + T[j, :3, 3]
        scene.append(P.astype(np.float32))
        hgt = P[:, 1] - floor
        ground.append(np.unique(np.floor(P[(hgt >= M.BAND[0]) & (hgt <= M.BAND[1])] / M.VOX).astype(np.int64), axis=0))
    out = []
    for i in range(n):
        Rc = M.corridor(T[i, :3, :3]); o = T[i, :3, 3]
        c = (Rc @ (scene[i] - o).T).T
        body = (np.abs(c[:, 0]) <= M.X_HALF) & (c[:, 1] >= M.BOX_Y[0]) & (c[:, 1] <= M.BOX_Y[1]) & (c[:, 2] >= M.Z_RANGE[0]) & (c[:, 2] <= M.Z_RANGE[1])
        if body.sum() < 5:
            out.append(None); continue
        d_body = float(np.percentile(c[body, 2], 5))
        win = np.flatnonzero((t <= t[i]) & (t >= t[i] - PAST_S))
        keys = np.concatenate([ground[j] for j in win])
        gc = (Rc @ ((keys + .5) * M.VOX - o).T).T
        sup = (np.abs(gc[:, 0]) <= M.X_HALF) & (gc[:, 2] >= 0) & (gc[:, 2] <= d_body + M.MARGIN)
        k = keys[sup] + (1 << 20); nvox = len(np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2])) if sup.any() else 0
        out.append(bool(nvox >= M.MIN_VOX))
    return seq.name, out


def main(prep, evaluation, cane, adt, out, *arms):
    arms = arms or ('gt', 'chain')
    R = json.loads(Path(evaluation).read_text(encoding='utf8')); L = json.loads(Path(cane).read_text(encoding='utf8'))['labels']
    nd = {s: SUM.nearest_dynamic(Path(prep) / s, Path(adt) / s) for s in L}
    pred = {}
    with ProcessPoolExecutor(7) as ex:
        for arm in arms:
            pred[arm] = dict(ex.map(predict, [(Path(prep) / s, R[s]['floor'], arm) for s in R]))
    res = {}
    for arm in arms:
        for sub in ('walk', 'walk_noManip'):
            c = {}
            for s, v in R.items():
                fr, lab, spd, pr = v['frames'], L[s]['rows'], L[s]['speed'], pred[arm][s]
                al = [np.asarray(V2.confirm([f['arms']['chain'][q][1] for f in fr]), bool) for q in (0, 1)]
                for k, f in enumerate(fr):
                    if spd[k] < M.WALK or (sub == 'walk_noManip' and nd[s][k] < SUM.MANIP_M):
                        continue
                    n = [f['truth'][q]['n'] for q in (0, 1)]
                    if max(n) >= 150 and lab[k] and not lab[k].get('held_only'):
                        cat = 'CANE' if lab[k]['support_vox'] >= M.MIN_VOX else 'MISS'; near = lab[k]['d_body'] <= 1.5
                    elif max(n) == 0:
                        cat, near = 'NEG', False
                    else:
                        continue
                    a = bool(al[0][k] or al[1][k]); kept = a and not bool(pr[k])
                    for key, val in ((f'n|{cat}', 1), (f'v6|{cat}', a), (f'kept|{cat}', kept)):
                        c[key] = c.get(key, 0) + val
                    if near:
                        for key, val in ((f'n_near|{cat}', 1), (f'v6_near|{cat}', a), (f'kept_near|{cat}', kept)):
                            c[key] = c.get(key, 0) + val
            res[f'{arm}|{sub}'] = c
            tot = lambda p: sum(c.get(f'{p}|{x}', 0) for x in ('CANE', 'MISS', 'NEG'))
            print(f"{arm:5s} {sub:12s} alerts v6 {tot('v6')} -> kept {tot('kept')} | CANE {c.get('v6|CANE',0)}->{c.get('kept|CANE',0)} "
                  f"MISS {c.get('v6|MISS',0)}->{c.get('kept|MISS',0)} NEG {c.get('v6|NEG',0)}->{c.get('kept|NEG',0)} | "
                  f"MISS recall<=1.5 {c.get('v6_near|MISS',0)}->{c.get('kept_near|MISS',0)}/{c.get('n_near|MISS',0)}", flush=True)
    Path(out).write_text(json.dumps(dict(past_s=PAST_S, results=res), indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:])
