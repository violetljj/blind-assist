"""Cane-reach labels on SANPO-Synthetic outdoor sessions (EXPLORE, 2026-10-15): how common are hazards the cane misses?

Local SANPO-Synthetic subset (artifacts.local/datasets/sanpo-synthetic-ba-nfo, <=50 frames/session); only the 40 train +
10 validation sessions of the 2026-09-19 selection are read, the 10 official-test sessions are skipped.
Conventions checked on walkable pixels: depth = optical z in metres (2 header values), pose quaternion = camera->world,
world Z up. Rule fixed before results (same spirit as the ADT relabel, heights made absolute because SANPO has chest and
head cameras):
  local ground : walkable classes (road, curb, sidewalk, crosswalk, paved trail, other walkable, terrain), 0.25 m cells,
                 median Z; a point's height = Z - ground of the nearest cell within 1 m.
  corridor     : horizontal camera heading, |x|<=0.30 m, forward 0.30-2.0 m from the camera.
  body part    : points 0.60-1.90 m above ground; positive if >= 21 points at stride 8 (150 px of a 640x480 image),
                 negative if none; d_body = 5th percentile forward distance.
  cane support : points 0.05-0.30 m above ground, forward 0 .. d_body+0.15 m, fused over all session frames (<=10 s),
                 3 cm voxels, >= 3 voxels => CANE, else MISS. Body classes of MISS frames are tallied.
Usage: sanpo_cane_reach_dev_20261015.py <out.json> [sheet.jpg]
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'artifacts.local/datasets/sanpo-synthetic-ba-nfo'
SELECTION = ROOT / 'artifacts.local/work/ba-nfo-20260919/sanpo-selection.json'
LABELS = json.loads((DATA / 'official-taxonomy/labelmap.json').read_text())
NAMES = {v: k for k, v in LABELS.items()}
WALK = [1, 2, 3, 5, 6, 17, 30]
S, VOX, CELL = 8, .03, .25
BODY = (.60, 1.90); BAND = (.05, .30); MARGIN = .15; MIN_VOX = 3; MIN_BODY = 21; X_HALF = .30; Z_RANGE = (.30, 2.0)


def quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def session(scene):
    cam = next((DATA / scene).glob('camera_*'))
    meta = json.loads((DATA / scene / 'description.json').read_text())['session_camera_details'][0]
    p = meta['left_camera_params']; H, W = p['image_height'], p['image_width']
    vv, uu = np.mgrid[0:H:S, 0:W:S]
    dirs = np.stack([(uu - p['cx']) / p['fx'], (vv - p['cy']) / p['fy'], np.ones(uu.shape)], -1).reshape(-1, 3)
    poses = list(csv.DictReader(open(cam / 'camera_poses.csv')))
    frames = sorted((cam / 'left/depth_maps').glob('*.float16.gz'))
    pts, labs, Rs, ps = [], [], [], []
    for f in frames:
        k = int(f.name[:6]); r = poses[k]
        d = np.frombuffer(gzip.decompress(f.read_bytes()), '<f2')[2:].astype(np.float32).reshape(H, W)[::S, ::S].reshape(-1)
        seg = cv2.imread(str(cam / 'left/segmentation_masks' / f'{k:06d}.png'), cv2.IMREAD_UNCHANGED)[::S, ::S, 2].reshape(-1)
        R = quat(*[float(r[c]) for c in ('q_x', 'q_y', 'q_z', 'q_w')]); t = np.array([float(r[c]) for c in ('pos_x', 'pos_y', 'pos_z')])
        ok = np.isfinite(d) & (d > .1) & (d < 30)
        pts.append((R @ (dirs[ok] * d[ok, None]).T).T + t); labs.append(seg[ok]); Rs.append(R); ps.append(t)
    # local ground grid from walkable points
    G = np.concatenate([P[np.isin(L, WALK)] for P, L in zip(pts, labs)])
    if len(G) < 100:
        return scene, dict(camera=cam.name, frames=[], note='no walkable ground')
    keys, inv = np.unique(np.floor(G[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
    gz = np.array([np.median(G[inv.ravel() == i, 2]) for i in range(len(keys))]); centres = (keys + .5) * CELL
    from scipy.spatial import cKDTree
    tree = cKDTree(centres)

    def height(P):
        dist, idx = tree.query(P[:, :2], distance_upper_bound=1.0)
        hgt = np.full(len(P), np.nan); ok = np.isfinite(dist); hgt[ok] = P[ok, 2] - gz[idx[ok]]
        return hgt
    hts = [height(P) for P in pts]
    ground = np.concatenate([P[(h >= BAND[0]) & (h <= BAND[1])] for P, h in zip(pts, hts)])
    gkeys = np.unique(np.floor(ground / VOX).astype(np.int64), axis=0) if len(ground) else np.zeros((0, 3), np.int64)
    gcent = (gkeys + .5) * VOX
    fps = meta.get('fps', 5.0); out = []
    for i, (P, L, h, R, t) in enumerate(zip(pts, labs, hts, Rs, ps)):
        fwd = R[:, 2].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd); right = np.array([fwd[1], -fwd[0], 0.])
        rel = P - t; x, z = rel @ right, rel @ fwd
        inc = (np.abs(x) <= X_HALF) & (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
        body = inc & (h >= BODY[0]) & (h <= BODY[1])
        cam_h = float(height(t[None])[0])
        row = dict(frame=i, cam_h=round(cam_h, 2) if np.isfinite(cam_h) else None, n_body=int(body.sum()))
        if body.sum() >= MIN_BODY:
            d_body = float(np.percentile(z[body], 5)); row['d_body'] = round(d_body, 3)
            gr = gcent - t; gx, gz_ = gr @ right, gr @ fwd
            sup = (np.abs(gx) <= X_HALF) & (gz_ >= 0) & (gz_ <= d_body + MARGIN)
            row['support_vox'] = int(sup.sum()); row['cat'] = 'CANE' if sup.sum() >= MIN_VOX else 'MISS'
            near = body & (z <= d_body + .3)
            row['classes'] = {NAMES.get(int(c), str(c)): int(n) for c, n in Counter(L[near].tolist()).most_common(3)}
        elif body.sum() == 0:
            row['cat'] = 'NEG'
        out.append(row)
    speed = np.linalg.norm(np.diff(np.array(ps)[:, :2], axis=0), axis=1) * fps if len(ps) > 1 else np.zeros(0)
    return scene, dict(camera=cam.name, fps=fps, median_speed=float(np.median(speed)) if len(speed) else None, frames=out)


def main(out, sheet=None):
    sel = json.loads(SELECTION.read_text())['sessions']
    scenes = [x['scene'] for x in sel if x['split'] != 'test' and (DATA / x['scene'] / 'description.json').exists()]
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(session, scenes))
    c = Counter(); cls = Counter(); cam = Counter()
    for s, v in res.items():
        for r in v['frames']:
            if 'cat' in r:
                c[(v['camera'], r['cat'])] += 1
                if r['cat'] != 'NEG' and r.get('d_body', 9) <= 1.5:
                    c[(v['camera'], r['cat'], 'near')] += 1
                if r['cat'] == 'MISS':
                    cls[max(r['classes'], key=r['classes'].get) if r['classes'] else 'none'] += 1
    for k in sorted(c, key=str):
        print('|'.join(k), c[k])
    print('MISS dominant classes:', cls.most_common())
    Path(out).write_text(json.dumps(dict(sessions=res, counts={'|'.join(k): v for k, v in c.items()}, miss_classes=dict(cls)), indent=1))
    if sheet:
        miss = [(s, r) for s, v in res.items() for r in v['frames'] if r.get('cat') == 'MISS']
        rng = np.random.default_rng(0); tiles = []
        for s, r in [miss[i] for i in rng.choice(len(miss), min(8, len(miss)), replace=False)]:
            cam_dir = next((DATA / s).glob('camera_*'))
            im = cv2.resize(cv2.imread(str(sorted((cam_dir / 'left/video_frames').glob('*.png'))[r['frame']])), (368, 207))
            cv2.putText(im, f"{r['d_body']:.2f} {list(r['classes'])[:1]}", (5, 20), 0, .5, (0, 0, 255), 2); tiles.append(im)
        while len(tiles) < 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(sheet, np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, 8, 2)], 0))


if __name__ == '__main__':
    main(*sys.argv[1:3])
