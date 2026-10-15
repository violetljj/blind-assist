"""Ground-memory support rules against filled-in overhangs (EXPLORE, 2026-10-15, consumed ADT development data).

Plan frozen in adt-eval-20261013/PLAN_FALSE_SUPPORT.md before running. Chain depth; estimator as adt_cane_memory_pose:
  A0  >=3 support voxels (3 cm), any observation
  C1  a 3 cm support voxel counts only if its 6 cm cell was observed in >=3 distinct frames of the 3 s window
  C2  only ground-band points observed at >=1.5 m horizontal distance from the camera are fused
Each rule runs with GT poses and with the worst PDR arm (HS+).
Usage: adt_cane_false_support_dev_20261015.py <prep> <adt_eval.json> <cane_reach_v3.json> <adt data> <out.json>
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
import adt_cane_memory_pose_dev_20261015 as PO
import frozen_fusion_eval_v2 as V2

PAST_S, MIN_FRAMES, FAR_M = 3.0, 3, 1.5
RULES, POSES = ('A0', 'C1', 'C2'), ('A0', 'HS+')


def pack(k):
    k = k + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def predict(args):
    seq, floor, rule, pose = args
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    S = M.STEP
    vv, uu = np.mgrid[0:h:S, 0:w:S]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        t_ns = Z['t_ns']
    with np.load(seq / 'chain_depth_s2.npz') as Z:
        z_all = Z['z'].astype(np.float32)
    T = PO.perturb(np.load(seq / 'poses.npz')['T_sc'], pose); n = len(t_ns); t = (t_ns - t_ns[0]) / 1e9
    scene, ground = [], []
    for j in range(n):
        z = z_all[j].reshape(-1); ok = np.isfinite(z) & (z > .1) & (z < 10)
        P = (T[j, :3, :3] @ (dirs[ok] * z[ok, None]).T).T + T[j, :3, 3]
        scene.append(P.astype(np.float32))
        hgt = P[:, 1] - floor; g = P[(hgt >= M.BAND[0]) & (hgt <= M.BAND[1])]
        if rule == 'C2':
            g = g[np.hypot(g[:, 0] - T[j, 0, 3], g[:, 2] - T[j, 2, 3]) >= FAR_M]
        ground.append(np.unique(np.floor(g / M.VOX).astype(np.int64), axis=0))
    out = []
    for i in range(n):
        Rc = M.corridor(T[i, :3, :3]); o = T[i, :3, 3]
        c = (Rc @ (scene[i] - o).T).T
        body = (np.abs(c[:, 0]) <= M.X_HALF) & (c[:, 1] >= M.BOX_Y[0]) & (c[:, 1] <= M.BOX_Y[1]) & (c[:, 2] >= M.Z_RANGE[0]) & (c[:, 2] <= M.Z_RANGE[1])
        if body.sum() < 5:
            out.append(None); continue
        d_body = float(np.percentile(c[body, 2], 5))
        win = np.flatnonzero((t <= t[i]) & (t >= t[i] - PAST_S))
        keys = np.concatenate([ground[j] for j in win]); fid = np.concatenate([np.full(len(ground[j]), j) for j in win])
        gc = (Rc @ ((keys + .5) * M.VOX - o).T).T
        sup = (np.abs(gc[:, 0]) <= M.X_HALF) & (gc[:, 2] >= 0) & (gc[:, 2] <= d_body + M.MARGIN)
        if rule == 'C1' and sup.any():
            coarse = pack(np.floor_divide(keys[sup], 2))
            pairs = np.unique(np.stack([coarse, fid[sup]], 1), axis=0)
            cells, cnt = np.unique(pairs[:, 0], return_counts=True)
            good = np.isin(coarse, cells[cnt >= MIN_FRAMES])
            nvox = len(np.unique(pack(keys[sup][good])))
        else:
            nvox = len(np.unique(pack(keys[sup]))) if sup.any() else 0
        out.append(bool(nvox >= M.MIN_VOX))
    return (rule, pose, seq.name), out


def main(prep, evaluation, cane, adt, out):
    R = json.loads(Path(evaluation).read_text(encoding='utf8')); L = json.loads(Path(cane).read_text(encoding='utf8'))['labels']
    nd = {s: SUM.nearest_dynamic(Path(prep) / s, Path(adt) / s) for s in L}
    with ProcessPoolExecutor(7) as ex:
        pred = dict(ex.map(predict, [(Path(prep) / s, R[s]['floor'], r, p) for r in RULES for p in POSES for s in R]))
    res = {}
    for r in RULES:
        for p in POSES:
            for sub in ('walk', 'walk_noManip'):
                c = {}
                for s, v in R.items():
                    fr, lab, spd, pr = v['frames'], L[s]['rows'], L[s]['speed'], pred[(r, p, s)]
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
                        for key, val in ((f'v6|{cat}', a), (f'kept|{cat}', kept)):
                            c[key] = c.get(key, 0) + val
                        if near:
                            for key, val in ((f'n_near|{cat}', 1), (f'kept_near|{cat}', kept)):
                                c[key] = c.get(key, 0) + val
                res[f'{r}|{p}|{sub}'] = c
                supp = 1 - c.get('kept|CANE', 0) / max(c.get('v6|CANE', 1), 1)
                print(f"{r} {p:3s} {sub:12s} kept {sum(c.get(f'kept|{x}',0) for x in ('CANE','MISS','NEG'))} | "
                      f"CANE suppressed {supp:.1%} | MISS near {c.get('kept_near|MISS',0)}/{c.get('n_near|MISS',0)} | "
                      f"NEG {c.get('v6|NEG',0)}->{c.get('kept|NEG',0)}", flush=True)
    Path(out).write_text(json.dumps(dict(rules=dict(past_s=PAST_S, min_frames=MIN_FRAMES, far_m=FAR_M), results=res), indent=1),
                         encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:])
