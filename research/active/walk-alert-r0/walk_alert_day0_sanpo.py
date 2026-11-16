"""walk-alert-r0 step 0 (EXPLORE, 2026-11-16): how many straight-ahead corridor hazard events does local SANPO-Synthetic hold?

Reads only the 40 train + 10 val sessions of artifacts.local/work/ba-nfo-20260919/sanpo-selection.json (10 official-test
sessions skipped). Conventions reused from hardware-bringup/host/sanpo_cane_reach_dev_20261015.py (checked there):
depth = optical z metres after 2 header values, pose quaternion camera->world, world Z up, class id in PNG R channel.
Rule fixed before counting:
  heading   : horizontal walking direction from poses p[t+2]-p[t-2] (clipped at ends); speed < 0.3 m/s => STANDING.
  ground    : walkable-class points, 0.25 m cells, median Z; height = Z - nearest cell ground within 1 m.
  corridor  : |lateral| <= 0.30 m, forward 0.30-4.00 m along heading, height 0.15-1.90 m, non-walkable non-sky classes.
  positive  : >= 21 corridor points at stride 8; d_near = 5th percentile forward distance.
  event     : run of positive frames (gaps of 1 frame bridged); DYNAMIC if dominant class pedestrian/rider/animal/vehicle.
This is evaluator-side labelling (uses future poses); it is not a detector. Usage: walk_alert_day0_sanpo.py <out_dir> [local|new]
(new = sessions of artifacts.local/downloads/sanpo-synthetic-walk-alert-20261116/selection.json)
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'artifacts.local/datasets/sanpo-synthetic-ba-nfo'
SELECTION = ROOT / 'artifacts.local/work/ba-nfo-20260919/sanpo-selection.json'
NEW = ROOT / 'artifacts.local/downloads/sanpo-synthetic-walk-alert-20261116'
NAMES = {v: k for k, v in json.loads((DATA / 'official-taxonomy/labelmap.json').read_text()).items()}
WALK = [1, 2, 3, 5, 6, 17, 30]
SKY = 27
DYNAMIC = {12, 13, 14, 21}
S, CELL = 8, .25
X_HALF, FWD, HGT, MIN_PTS, STAND = .30, (.30, 4.0), (.15, 1.90), 21, .30
BINS = [(.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def session(arg):
    root, scene = Path(arg[0]), arg[1]
    cam = next((root / scene).glob('camera_*'))
    meta = json.loads((root / scene / 'description.json').read_text())['session_camera_details'][0]
    p = meta['left_camera_params']; H, W = p['image_height'], p['image_width']; fps = float(meta.get('fps', 5.0))
    vv, uu = np.mgrid[0:H:S, 0:W:S]
    dirs = np.stack([(uu - p['cx']) / p['fx'], (vv - p['cy']) / p['fy'], np.ones(uu.shape)], -1).reshape(-1, 3)
    poses = list(csv.DictReader(open(cam / 'camera_poses.csv')))
    frames = sorted((cam / 'left/depth_maps').glob('*.float16.gz'))
    pts, labs, ps = [], [], []
    for f in frames:
        k = int(f.name[:6]); r = poses[k]
        d = np.frombuffer(gzip.decompress(f.read_bytes()), '<f2')[2:].astype(np.float32).reshape(H, W)[::S, ::S].reshape(-1)
        seg = cv2.imread(str(cam / 'left/segmentation_masks' / f'{k:06d}.png'), cv2.IMREAD_UNCHANGED)[::S, ::S, 2].reshape(-1)
        R = quat(*[float(r[c]) for c in ('q_x', 'q_y', 'q_z', 'q_w')])
        t = np.array([float(r[c]) for c in ('pos_x', 'pos_y', 'pos_z')])
        ok = np.isfinite(d) & (d > .1) & (d < 30)
        pts.append((R @ (dirs[ok] * d[ok, None]).T).T + t); labs.append(seg[ok]); ps.append(t)
    n = len(ps); ps = np.array(ps)
    base = dict(scene=scene, camera=cam.name, fps=fps, frames=n, seconds=n / fps,
                path_m=float(np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1).sum()) if n > 1 else 0.)
    G = np.concatenate([P[np.isin(L, WALK)] for P, L in zip(pts, labs)]) if n else np.zeros((0, 3))
    if len(G) < 100 or n < 5:
        return dict(base, note='NOT_EVALUABLE: no walkable ground or too few frames', rows=[], events=[])
    keys, inv = np.unique(np.floor(G[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    gz = np.array([np.median(G[inv == i, 2]) for i in range(len(keys))]); tree = cKDTree((keys + .5) * CELL)
    rows = []
    for i in range(n):
        a, b = max(0, i - 2), min(n - 1, i + 2)
        v = (ps[b] - ps[a])[:2]; speed = float(np.linalg.norm(v) / max(b - a, 1) * fps)
        row = dict(frame=i, speed=round(speed, 3))
        if speed < STAND:
            row['cat'] = 'STANDING'; rows.append(row); continue
        fwd = np.array([v[0], v[1], 0.]) / np.linalg.norm(v); right = np.array([fwd[1], -fwd[0], 0.])
        P, L = pts[i], labs[i]
        dist, idx = tree.query(P[:, :2], distance_upper_bound=1.0)
        h = np.full(len(P), np.nan); okg = np.isfinite(dist); h[okg] = P[okg, 2] - gz[idx[okg]]
        rel = P - ps[i]; x, z = rel @ right, rel @ fwd
        inc = ((np.abs(x) <= X_HALF) & (z >= FWD[0]) & (z <= FWD[1]) & (h >= HGT[0]) & (h <= HGT[1])
               & ~np.isin(L, WALK + [SKY]))
        if inc.sum() >= MIN_PTS:
            d_near = float(np.percentile(z[inc], 5)); near = inc & (z <= d_near + .3)
            cls = Counter(L[near].tolist()).most_common(1)[0][0]
            row.update(cat='POS', d_near=round(d_near, 3), cls=NAMES.get(int(cls), str(cls)), dynamic=int(cls) in DYNAMIC)
        else:
            row['cat'] = 'NEG' if inc.sum() == 0 else 'WEAK'
        rows.append(row)
    events, cur = [], None
    for r in rows:
        if r['cat'] == 'POS':
            if cur and r['frame'] - cur['last'] <= 2:
                cur['last'] = r['frame']; cur['d'].append(r['d_near']); cur['cls'].append(r['cls']); cur['dyn'].append(r['dynamic'])
            else:
                cur = dict(first=r['frame'], last=r['frame'], d=[r['d_near']], cls=[r['cls']], dyn=[r['dynamic']]); events.append(cur)
    ev = [dict(first=e['first'], last=e['last'], seconds=(e['last'] - e['first'] + 1) / fps, d_first=e['d'][0], d_min=min(e['d']),
               cls=Counter(e['cls']).most_common(1)[0][0], dynamic=sum(e['dyn']) * 2 > len(e['dyn']),
               censored_start=e['first'] == 0) for e in events]
    moving = [r['speed'] for r in rows if r['cat'] != 'STANDING']
    return dict(base, median_speed=float(np.median(moving)) if moving else None, rows=rows, events=ev)


def main(out, source='local'):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(SELECTION.read_text())['sessions']
    if source == 'new':
        root = NEW; scenes = json.loads((NEW / 'selection.json').read_text())['sessions']
    else:
        root = DATA; scenes = [s['scene'] for s in sel if s['split'] in ('train', 'val')]
    scenes = [(str(root), s) for s in scenes if (root / s).is_dir()]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(session, scenes))
    (out / 'sessions.json').write_text(json.dumps(res, indent=1))
    cats = Counter(r['cat'] for s in res for r in s['rows'])
    ev = [e for s in res for e in s['events']]
    moving_s = sum(sum(r['cat'] != 'STANDING' for r in s['rows']) / s['fps'] for s in res)
    summ = dict(sessions=len(res), cameras=Counter(s['camera'] for s in res), not_evaluable=sum('note' in s for s in res),
                seconds=round(sum(s['seconds'] for s in res), 1), moving_seconds=round(moving_s, 1), frame_cats=cats,
                events=len(ev), static_events=sum(not e['dynamic'] for e in ev), dynamic_events=sum(e['dynamic'] for e in ev),
                censored_start_events=sum(e['censored_start'] for e in ev),
                events_first_seen_by_bin={f'{a}-{b}': sum(a <= e['d_first'] < b for e in ev) for a, b in BINS},
                events_min_dist_by_bin={f'{a}-{b}': sum(a <= e['d_min'] < b for e in ev) for a, b in BINS},
                static_classes=Counter(e['cls'] for e in ev if not e['dynamic']), dynamic_classes=Counter(e['cls'] for e in ev if e['dynamic']),
                path_km=round(sum(s['path_m'] for s in res) / 1000, 3),
                events_per_km=round(len(ev) / max(sum(s['path_m'] for s in res) / 1000, 1e-9), 1),
                static_per_km=round(sum(not e['dynamic'] for e in ev) / max(sum(s['path_m'] for s in res) / 1000, 1e-9), 1),
                sessions_with_static=sum(any(not e['dynamic'] for e in s['events']) for s in res),
                static_min_dist_by_bin={f'{a}-{b}': sum(a <= e['d_min'] < b for e in ev if not e['dynamic']) for a, b in BINS},
                median_speed_from_metadata_fps_UNRELIABLE=float(np.median([s['median_speed'] for s in res if s.get('median_speed')])))
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=dict))
    print(json.dumps(summ, indent=1, default=dict))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'local')
