"""Cane-reach labels on a MuSoHu bag (helmet ZED2 depth + odometry; CC0) (EXPLORE, 2026-10-15).

Same rule as sanpo_cane_reach_dev_20261015 (body 0.60-1.90 m above ground, cane band 0.05-0.30 m, corridor +-0.30 m x
0.30-2.0 m, +0.15 m margin, 3 cm voxels, >=3 voxels => CANE), with MuSoHu specifics fixed before results:
  frames      : ZED2 depth_registered (compressedDepth) at <= 5 Hz, stride 4; pose = nearest ZED odom (REP-103, Z up),
                optical->base rotation fixed (x_b = z_o, y_b = -x_o, z_b = -y_o), lever arm ignored.
  people      : YOLOv8n person boxes (conf >= 0.3) on the RGB frame; depth pixels inside them are dropped (social
                navigation data is crowded, and moving people break the static-support rule as on SANPO).
  ground      : per 0.25 m cell lower envelope (5th percentile Z) of all kept points; camera height = median of
                camera Z minus the cell ground under the walker; a cell estimate deviating > 0.30 m from the walker's
                ground at the nearest trajectory point is replaced by that trajectory ground.
  support     : ground-band voxels from frames within +-10 s.
Usage: musohu_cane_reach_dev_20261015.py <bag> <out.json> [sheet.jpg]
"""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

S, VOX, CELL, HZ, WIN_S = 4, .03, .25, 5.0, 10.
BODY = (.60, 1.90); BAND = (.05, .30); MARGIN = .15; MIN_VOX = 3; X_HALF = .30; Z_RANGE = (.30, 2.0)
R_BO = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
YOLO = Path('E:/codex-tools/models/ultralytics/yolov8n.pt')


def quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def decode_depth(msg):
    raw = bytes(msg.data)
    if 'compressedDepth' in msg.format:
        _, qa, qb = struct.unpack('<iff', raw[:12])
        img = cv2.imdecode(np.frombuffer(raw[12:], np.uint8), cv2.IMREAD_UNCHANGED)
        if msg.format.startswith('32FC1'):
            v = img.astype(np.float32); z = np.where(v > 0, qa / np.maximum(v - qb, 1e-6), np.nan)
            return z.astype(np.float32)
        return img.astype(np.float32) / 1000.
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32)


def read(bag):
    from rosbags.highlevel import AnyReader
    odom, depth, rgb, K = [], [], [], None
    last = -1e18
    with AnyReader([Path(bag)]) as r:
        conns = [c for c in r.connections if c.topic in ('/zed2/zed_node/odom', '/zed2/zed_node/depth/depth_registered/compressed',
                                                        '/zed2/zed_node/rgb/image_rect_color/compressed', '/zed2/zed_node/depth/camera_info')]
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic.endswith('/odom'):
                p, q = m.pose.pose.position, m.pose.pose.orientation
                odom.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            elif c.topic.endswith('camera_info') and K is None:
                K = (np.array(getattr(m, 'K', None) if hasattr(m, 'K') else m.k).reshape(3, 3), m.width, m.height)
            elif c.topic.endswith('depth_registered/compressed'):
                if t - last >= 1e9 / HZ:
                    depth.append((t, decode_depth(m))); last = t
            elif c.topic.endswith('image_rect_color/compressed'):
                rgb.append((t, bytes(m.data)))
    return np.array(odom), depth, rgb, K


def main(bag, out, sheet=None):
    odom, depth, rgb, (K, W, H) = read(bag)
    from ultralytics import YOLO as Y
    det = Y(str(YOLO))
    rgb_t = np.array([t for t, _ in rgb])
    vv, uu = np.mgrid[0:H:S, 0:W:S]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(uu.shape)], -1).reshape(-1, 3)
    pts, cams, Rs, ts, imgs = [], [], [], [], []
    for t, z in depth:
        k = int(np.argmin(np.abs(odom[:, 0] - t))); o = odom[k]
        R = quat(*o[4:8]) @ R_BO; p = o[1:4]
        j = int(np.argmin(np.abs(rgb_t - t))); im = cv2.imdecode(np.frombuffer(rgb[j][1], np.uint8), cv2.IMREAD_COLOR)
        keep = np.ones((H, W), bool)
        if im is not None:
            sy, sx = H / im.shape[0], W / im.shape[1]
            for b in det.predict(im, classes=[0], conf=.3, verbose=False)[0].boxes.xyxy.cpu().numpy():
                keep[int(b[1] * sy):int(b[3] * sy) + 1, int(b[0] * sx):int(b[2] * sx) + 1] = False
        zz = z[::S, ::S].reshape(-1); ok = np.isfinite(zz) & (zz > .3) & (zz < 15) & keep[::S, ::S].reshape(-1)
        pts.append((R @ (dirs[ok] * zz[ok, None]).T).T + p); cams.append(p); Rs.append(R); ts.append(t / 1e9); imgs.append(im)
    cams, ts = np.array(cams), np.array(ts)
    allp = np.concatenate(pts)
    keys, inv = np.unique(np.floor(allp[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel(); order = np.argsort(inv); bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
    zsorted = allp[order, 2]
    cell_ground = np.array([np.percentile(zsorted[bounds[i]:bounds[i + 1]], 5) for i in range(len(keys))])
    lut = {tuple(k): g for k, g in zip(keys.tolist(), cell_ground)}
    under = [lut.get(tuple(np.floor(c[:2] / CELL).astype(np.int64).tolist())) for c in cams]
    hs = [c[2] - g for c, g in zip(cams, under) if g is not None]
    cam_h = float(np.median(hs)) if hs else 1.75
    traj_ground = cams[:, 2] - cam_h
    from scipy.spatial import cKDTree
    traj = cKDTree(cams[:, :2]); cells = cKDTree((keys + .5) * CELL)
    _, near_traj = traj.query((keys + .5) * CELL)
    ground_fixed = np.where(np.abs(cell_ground - traj_ground[near_traj]) <= .30, cell_ground, traj_ground[near_traj])

    def height(P):
        d, idx = cells.query(P[:, :2], distance_upper_bound=1.0)
        h = np.full(len(P), np.nan); ok = np.isfinite(d); h[ok] = P[ok, 2] - ground_fixed[idx[ok]]
        return h
    hts = [height(P) for P in pts]
    gvox = [np.unique(np.floor(P[(h >= BAND[0]) & (h <= BAND[1])] / VOX).astype(np.int64), axis=0) for P, h in zip(pts, hts)]
    min_body = max(5, int(round(4.88e-4 * len(dirs))))
    rows = []
    for i, (P, h, R, p, t) in enumerate(zip(pts, hts, Rs, cams, ts)):
        fwd = R[:, 2].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd); right = np.array([fwd[1], -fwd[0], 0.])
        rel = P - p; x, z = rel @ right, rel @ fwd
        body = (np.abs(x) <= X_HALF) & (z >= Z_RANGE[0]) & (z <= Z_RANGE[1]) & (h >= BODY[0]) & (h <= BODY[1])
        win = np.flatnonzero(np.abs(ts - t) <= WIN_S)
        spd = float(np.linalg.norm(cams[min(i + 2, len(cams) - 1), :2] - cams[max(i - 2, 0), :2]) /
                    max(ts[min(i + 2, len(ts) - 1)] - ts[max(i - 2, 0)], 1e-3))
        row = dict(i=i, t=round(t - ts[0], 2), speed=round(spd, 2), n_body=int(body.sum()))
        if body.sum() >= min_body:
            d_body = float(np.percentile(z[body], 5)); row['d_body'] = round(d_body, 3)
            g = (np.concatenate([gvox[j] for j in win]) + .5) * VOX - p
            sup = (np.abs(g @ right) <= X_HALF) & (g @ fwd >= 0) & (g @ fwd <= d_body + MARGIN)
            nv = len(np.unique(np.floor(g[sup] / VOX).astype(np.int64), axis=0)) if sup.any() else 0
            row['support_vox'] = nv; row['cat'] = 'CANE' if nv >= MIN_VOX else 'MISS'
        elif body.sum() == 0:
            row['cat'] = 'NEG'
        rows.append(row)
    c = Counter((r.get('cat', 'mid'), 'walk' if r['speed'] >= .3 else 'still') for r in rows)
    c_near = Counter(r['cat'] for r in rows if r.get('d_body', 9) <= 1.5 and r['speed'] >= .3)
    print('frames', len(rows), 'cam_h %.2f' % cam_h, 'min_body', min_body, dict(c), 'walking near', dict(c_near))
    Path(out).write_text(json.dumps(dict(bag=Path(bag).name, cam_h=cam_h, min_body=min_body, frames=rows,
                                         counts={'|'.join(k): v for k, v in c.items()}), indent=1))
    if sheet:
        miss = [r for r in rows if r.get('cat') == 'MISS' and r['speed'] >= .3]
        rng = np.random.default_rng(0); tiles = []
        for r in [miss[i] for i in rng.choice(len(miss), min(8, len(miss)), replace=False)] if miss else []:
            im = imgs[r['i']]
            if im is None:
                continue
            im = cv2.resize(im, (384, 216)); cv2.putText(im, f"{r['d_body']:.2f} t{r['t']:.0f}", (5, 20), 0, .6, (0, 0, 255), 2); tiles.append(im)
        if tiles:
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(sheet, np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)], 0))


if __name__ == '__main__':
    main(*sys.argv[1:4])
