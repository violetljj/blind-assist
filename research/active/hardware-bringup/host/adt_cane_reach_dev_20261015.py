"""Cane-reach relabelling of the ADT evaluation: would the white cane touch it before the body does? (EXPLORE, 2026-10-15)

Task re-definition under test: a body-level obstacle needs a device alert only when the cane will not find it first.
Fixed rule (written before reading any result; no tuning):
  body part    : current-frame GT points in the frozen HEAD or BODY box (|x|<=0.30 m, 0.30-2.0 m ahead,
                 0.20 m above to 0.90 m below the camera); d_body = 5th percentile of their forward distance.
  cane support : GT points 0.05-0.30 m above the floor, |x|<=0.30 m, forward 0 .. d_body+0.15 m, fused from all frames
                 within +-10 s (3 cm voxels, the floor itself excluded); >=3 distinct voxels => the cane finds it first.
  categories   : frame positive if either box has >=150 GT pixels (as the 2026-10-13 truth), negative if both have 0;
                 positive -> CANE (support found) or MISS (no support: overhang / protrusion / hanging).
Strata: all frames, and walking frames (horizontal speed >= 0.30 m/s over +-2 frames).
Alerts are the stored v6 chain readouts in adt_eval.json with the same 2/3 confirmation; nothing is re-run on GPU.
Hand-held / moved objects: body-box points inside any ADT dynamic object's box (+3 cm, nearest pose) are removed;
frames left with < 150 full-res px of body points are excluded as held-object-only (added after a visual check
found carried picture frames/baskets labelled MISS).
Usage: adt_cane_reach_dev_20261015.py <prep dir> <adt_eval.json> <out.json> <adt data dir>
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STEP, VOX, WIN_S = 2, .03, 10.
BAND = (.05, .30); MARGIN = .15; MIN_VOX = 3; WALK = .30; BOX_Y = (-.20, .90); X_HALF = .30; Z_RANGE = (.30, 2.0)
MIN_BODY_PTS = 38   # 150 full-res px at STEP 2


def corridor(R_sc):
    up = np.array([0., -1., 0.])                       # same as adt_prep: rows x, down, horizontal forward (scene Y-up)
    f = R_sc[:, 2] - up * (R_sc[:, 2] @ up); f /= np.linalg.norm(f)
    return np.stack([np.cross(up, f), up, f])


def dynamic_boxes(adt_seq):
    """Dynamic ADT objects: uid -> (timestamps, R_wo [k,3,3], t_wo [k,3], local AABB min/max)."""
    import csv
    inst = json.loads((adt_seq / 'instances.json').read_text(encoding='utf8'))
    dyn = {int(k) for k, v in inst.items() if v.get('motion_type') == 'dynamic'}
    box = {}
    with open(adt_seq / '3d_bounding_box.csv', newline='') as f:
        for r in csv.reader(f):
            if r[0].isdigit() and int(r[0]) in dyn and int(r[0]) not in box:
                v = list(map(float, r[2:8])); box[int(r[0])] = (np.array(v[0::2]), np.array(v[1::2]))
    poses = {}
    with open(adt_seq / 'scene_objects.csv', newline='') as f:
        for r in csv.reader(f):
            if r[0].isdigit() and int(r[0]) in box:
                poses.setdefault(int(r[0]), []).append(list(map(float, r[1:9])))
    out = {}
    for uid, rows in poses.items():
        a = np.array(sorted(rows))
        if (a[:, 0] >= 0).any():
            a = a[a[:, 0] >= 0]                     # drop the timestamp -1 default pose when a trajectory exists
        w, x, y, z = a[:, 4], a[:, 5], a[:, 6], a[:, 7]
        R = np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                      np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                      np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], 1)
        out[uid] = (a[:, 0], R, a[:, 1:4], box[uid][0] - .03, box[uid][1] + .03)
    return out


def inside_dynamic(P, t, boxes):
    """Mask of scene points inside any dynamic object's (3 cm enlarged) box at device time t (nearest pose)."""
    m = np.zeros(len(P), bool)
    for ts, R, tr, lo, hi in boxes.values():
        k = 0 if ts[0] < 0 else int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
        if ts[0] >= 0 and abs(ts[k - 1] - t) < abs(ts[k] - t):
            k -= 1                                   # nearest pose in time
        loc = (P - tr[k]) @ R[k]
        m |= np.all((loc >= lo) & (loc <= hi), axis=1)
    return m


def label_sequence(args):
    seq, floor, adt_seq = args
    boxes = dynamic_boxes(adt_seq)
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    vv, uu = np.mgrid[0:h:STEP, 0:w:STEP]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        depth, t_ns = Z['depth'][:, ::STEP, ::STEP], Z['t_ns']
    T = np.load(seq / 'poses.npz')['T_sc']
    n = len(t_ns); scene, ground = [], []
    for j in range(n):
        z = depth[j].reshape(-1) / 1000.; ok = z > .1
        P = (T[j, :3, :3] @ (dirs[ok] * z[ok, None]).T).T + T[j, :3, 3]
        scene.append(P.astype(np.float32))
        hgt = P[:, 1] - floor; g = P[(hgt >= BAND[0]) & (hgt <= BAND[1])]
        ground.append(np.unique(np.floor(g / VOX).astype(np.int64), axis=0))      # voxel keys per frame
    t = (t_ns - t_ns[0]) / 1e9; rows = []
    pos = T[:, [0, 2], 3]; lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    speed = np.linalg.norm(pos[hi] - pos[lo], axis=1) / np.maximum(t[hi] - t[lo], 1e-3)   # horizontal m/s, +-2 frames
    for i in range(n):
        Rc = corridor(T[i, :3, :3]); o = T[i, :3, 3]
        c = (Rc @ (scene[i] - o).T).T                                              # x right, y down from camera, z forward
        body = (np.abs(c[:, 0]) <= X_HALF) & (c[:, 1] >= BOX_Y[0]) & (c[:, 1] <= BOX_Y[1]) & (c[:, 2] >= Z_RANGE[0]) & (c[:, 2] <= Z_RANGE[1])
        if not body.any():
            rows.append(None); continue
        held = inside_dynamic(scene[i][body], int(t_ns[i]), boxes)               # carried / moved objects removed
        if (~held).sum() < MIN_BODY_PTS:
            rows.append(dict(held_only=True, speed=round(float(speed[i]), 3))); continue
        d_body = float(np.percentile(c[body][~held, 2], 5))
        win = np.flatnonzero(np.abs(t - t[i]) <= WIN_S)
        keys = np.concatenate([ground[j] for j in win])                            # duplicates kept; dedup after the cheap filter
        gc = (Rc @ ((keys + .5) * VOX - o).T).T
        sup = (np.abs(gc[:, 0]) <= X_HALF) & (gc[:, 2] >= 0) & (gc[:, 2] <= d_body + MARGIN)
        k = keys[sup] + (1 << 20); nvox = len(np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2])) if sup.any() else 0
        rows.append(dict(d_body=round(d_body, 3), support_vox=int(nvox), speed=round(float(speed[i]), 3),
                         d_support=round(float(gc[sup, 2].min()), 3) if sup.any() else None))
    return seq.name, dict(rows=rows, speed=np.round(speed, 3).tolist())


def confirm(a):
    sys.path.insert(0, str(Path(__file__).parent))
    import frozen_fusion_eval_v2 as V2
    return np.asarray(V2.confirm(np.asarray(a, bool)), bool)


def main(prep, evaluation, out, adt_data):
    R = json.loads(Path(evaluation).read_text(encoding='utf8'))
    jobs = [(Path(prep) / s, R[s]['floor'], Path(adt_data) / s) for s in R]
    with ProcessPoolExecutor(7) as ex:
        labels = dict(ex.map(label_sequence, jobs))
    stats = {}
    def add(k, v):
        d = stats.setdefault(k, [0, 0]); d[0] += int(v); d[1] += 1
    for s, v in R.items():
        fr, lab, spd = v['frames'], labels[s]['rows'], labels[s]['speed']
        assert len(fr) == len(lab), s
        alert = [confirm([f['arms']['chain'][q][1] for f in fr]) for q in (0, 1)]       # v6, 2/3 confirmed
        for k, f in enumerate(fr):
            n = [f['truth'][q]['n'] for q in (0, 1)]
            if max(n) >= 150 and lab[k] is not None and lab[k].get('held_only'):
                add(('all', 'excluded_held_only'), 1); continue
            if max(n) >= 150 and lab[k] is not None:
                cat = 'CANE' if lab[k]['support_vox'] >= MIN_VOX else 'MISS'
                near = 'near' if lab[k]['d_body'] <= 1.5 else 'far'
            elif max(n) == 0:
                cat, near = 'NEG', None
            else:
                continue
            any_alert, head_alert = bool(alert[0][k] or alert[1][k]), bool(alert[0][k])
            for sub in (('all', 'walk') if spd[k] >= WALK else ('all',)):
                add((sub, 'frames', cat), 1)
                if near:
                    add((sub, 'frames', cat, near), 1)
                    add((sub, 'recall_any', cat, near), any_alert); add((sub, 'recall_head_only', cat, near), head_alert)
                add((sub, 'alerts_any', cat), any_alert); add((sub, 'alerts_head_only', cat), head_alert)
    rep = dict(rule=dict(step=STEP, voxel_m=VOX, window_s=WIN_S, band_m=BAND, margin_m=MARGIN, min_vox=MIN_VOX),
               stats={'|'.join(map(str, k)): v for k, v in stats.items()},
               labels={s: l for s, l in labels.items()})
    Path(out).write_text(json.dumps(rep), encoding='utf8')
    for k in sorted(stats, key=str):
        a, b = stats[k]; print('|'.join(map(str, k)), f'{a}/{b}' if k[1] != 'frames' else a)


if __name__ == '__main__':
    main(*sys.argv[1:5])
