"""Ground memory under device-like pose errors on ADT (EXPLORE, 2026-10-15).

Plan frozen in adt-eval-20261013/PLAN_POSE_ERR.md before running. Same estimator, labels and strata as
adt_cane_memory_dev_20261015 (chain depth only); only the poses used to fuse past ground points are perturbed:
  A0  GT poses                         S+/S-  horizontal step length x1.2 / x0.8, GT direction
  H   step length GT, direction = current head heading (PDR cannot see the walking direction)
  HS+/HS-  H with length x1.2 / x0.8   Z  rotation only, no translation (lower bound)
  P+/P-  GT translation, constant +-3 deg camera pitch bias (gravity / IMU zero error)
Vertical displacement is kept from GT in every arm (ADT floors are flat).
Usage: adt_cane_memory_pose_dev_20261015.py <prep> <adt_eval.json> <cane_reach.json> <adt data> <out.json>
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
ARMS = ('A0', 'S+', 'S-', 'H', 'HS+', 'HS-', 'Z', 'P+', 'P-')


def perturb(T, arm):
    T = T.copy(); p = T[:, :3, 3].copy(); d = np.diff(p, axis=0); h = d[:, [0, 2]]
    if arm in ('S+', 'S-'):
        h = h * (1.2 if arm == 'S+' else .8)
    elif arm.startswith('H'):
        f = T[1:, [0, 2], 2]; f /= np.linalg.norm(f, axis=1, keepdims=True)    # head heading (horizontal), scene Y-up
        h = np.linalg.norm(h, axis=1, keepdims=True) * f * {'H': 1., 'HS+': 1.2, 'HS-': .8}[arm]
    elif arm == 'Z':
        h = h * 0
    d[:, [0, 2]] = h
    T[:, :3, 3] = np.concatenate([p[:1], p[:1] + np.cumsum(d, 0)])
    if arm in ('P+', 'P-'):
        a = np.radians(3 if arm == 'P+' else -3)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        T[:, :3, :3] = T[:, :3, :3] @ Rx
    return T


def predict(args):
    seq, floor, arm = args
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    S = M.STEP
    vv, uu = np.mgrid[0:h:S, 0:w:S]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        t_ns = Z['t_ns']
    with np.load(seq / 'chain_depth_s2.npz') as Z:
        z_all = Z['z'].astype(np.float32)
    T = perturb(np.load(seq / 'poses.npz')['T_sc'], arm); n = len(t_ns); t = (t_ns - t_ns[0]) / 1e9
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
        out.append(dict(sup=bool(nvox >= M.MIN_VOX), nvox=int(nvox), d_body=round(d_body, 3),
                        d_sup=round(float(np.sort(gc[sup, 2])[min(2, sup.sum() - 1)]), 3) if sup.any() else None))
    return (arm, seq.name), out


def main(prep, evaluation, cane, adt, out):
    R = json.loads(Path(evaluation).read_text(encoding='utf8')); L = json.loads(Path(cane).read_text(encoding='utf8'))['labels']
    nd = {s: SUM.nearest_dynamic(Path(prep) / s, Path(adt) / s) for s in L}
    with ProcessPoolExecutor(7) as ex:
        pred = dict(ex.map(predict, [(Path(prep) / s, R[s]['floor'], a) for a in ARMS for s in R]))
    res, lost = {}, []
    for arm in ARMS:
        for sub in ('walk', 'walk_noManip'):
            c = {}
            for s, v in R.items():
                fr, lab, spd, pr = v['frames'], L[s]['rows'], L[s]['speed'], pred[(arm, s)]
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
                    a = bool(al[0][k] or al[1][k]); kept = a and not bool(pr[k] and pr[k]['sup'])
                    for key, val in ((f'n|{cat}', 1), (f'v6|{cat}', a), (f'kept|{cat}', kept)):
                        c[key] = c.get(key, 0) + val
                    if near:
                        for key, val in ((f'n_near|{cat}', 1), (f'v6_near|{cat}', a), (f'kept_near|{cat}', kept)):
                            c[key] = c.get(key, 0) + val
                        if arm == 'A0' and sub == 'walk_noManip' and cat == 'MISS' and a and not kept:
                            lost.append(dict(seq=s, frame=k, truth=lab[k], pred=pr[k]))
            res[f'{arm}|{sub}'] = c
            tot = lambda p: sum(c.get(f'{p}|{x}', 0) for x in ('CANE', 'MISS', 'NEG'))
            print(f"{arm:4s} {sub:12s} alerts {tot('v6')} -> {tot('kept')} | CANE {c.get('v6|CANE',0)}->{c.get('kept|CANE',0)} "
                  f"MISS {c.get('v6|MISS',0)}->{c.get('kept|MISS',0)} NEG {c.get('v6|NEG',0)}->{c.get('kept|NEG',0)} | "
                  f"MISS near {c.get('v6_near|MISS',0)}->{c.get('kept_near|MISS',0)}/{c.get('n_near|MISS',0)}", flush=True)
    Path(out).write_text(json.dumps(dict(past_s=PAST_S, results=res, a0_lost_miss_near=lost), indent=1), encoding='utf8')
    for x in lost:
        print('lost', x['seq'], x['frame'], x['truth'], x['pred'])


if __name__ == '__main__':
    main(*sys.argv[1:])
