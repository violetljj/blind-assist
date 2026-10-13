"""Per-frame blob dump of the frozen v6 pipeline on own head-worn captures (EXPLORE; consumed Development data).

Runs exactly the v6 per-frame path (ToF grades + pose gate, DAV2-S, anchor chain incl. robust replacement, boxes) but
stores, per ready ToF frame: ToF time, pitch/roll, anchor source, ToF strong flags, every 8-connected in-box blob
(>= 10 px) as [npx, z50, cx, cy, cz] in the corridor frame, and the corridor-frame occupancy grid of predicted points
(HEAD/BODY bands x 30 lateral x 50 depth bins of 0.1 m, every 2nd pixel) -- the same layout as the ADT dump -- so
motion-consistency rules can be replayed offline with IMU-only ego-motion. The raw IMU stream (time, accel, gyro,
gravity) is copied per session. v6 alerts are recomputable from the blobs (sum npx of blobs with z50 <= 1.8 >= 150).
Usage: own_blob_dump_dev_20261013.py --raw <dir of capture-* sessions> --labels <labels.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import fusion_anchor_candidates_dev_20261013 as A
from adt_backbone_blobs_dev_20261013 import blobs, grid


def imu_arrays(d):
    t, acc, gyr, grav = [], [], [], []
    for r in V1.F.lines(d / 'imu.index.jsonl'):
        if r.get('valid_frame') and r.get('accel_g') and r.get('gyro_dps'):
            t.append(r['received_elapsed_ns']); acc.append(r['accel_g']); gyr.append(r['gyro_dps'])
            grav.append(r.get('gravity_unit') or [np.nan] * 3)
    return dict(t=np.array(t, np.int64), acc=np.array(acc), gyr=np.array(gyr), grav=np.array(grav, float))


def main(raw, labels_path, out):
    import cv2
    from atom_imu import load_mounting
    S, F, R, X, PE = V1.S, V1.F, V1.R, V1.X, V1.PE
    labels = json.loads(Path(labels_path).read_text(encoding='utf8'))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    arrays = np.load(V1.P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = V1.P.read('instrument_parameters.json')['noise']
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = V1.AtomCamera(); K = cam.K_upright; rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    sub = np.zeros((h, w), bool); sub[::2, ::2] = True; sub = sub.reshape(-1)
    sessions = {d.name: d for d in Path(raw).iterdir() if d.name in labels}
    height = V1.DEFAULT_HEIGHT
    for name, lab in labels.items():
        if lab['class'] == 'calibration' and name in sessions:
            fit = R.floor_fit(R.load(sessions[name], imu_rot))
            if fit and 1.0 <= fit[0] <= 1.9:
                height = fit[0]
    engine = V1.ContractReplay(str(V1.MANIFEST)); model, _ = S.load_dav2_small()
    rng = np.random.default_rng(20261013)
    try:
        for name in sorted(sessions):
            if (out / f'{name}.json').exists():
                continue
            d = sessions[name]; x = R.load(d, imu_rot)
            grade, ready = V1.tof_grades(engine, d, x, bias, noise)
            gated, pose_ok = PE.apply(grade, x['pitch'], x['roll'])
            tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
            rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
            dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
            last = None; frames = []; grids = []
            for i in np.flatnonzero(ready):
                j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                anchor = None; bl = None; g = np.zeros((2, 30, 50), np.uint16)
                if abs(rgb_t[j] - tof_t[i]) <= 1e8 and pose_ok[i]:
                    up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
                    disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), V1.SIZE)
                    dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                    valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                    inv = np.where(valid, 1 / (dist[i] * zfac), np.nan)
                    fit = S.fit_affine(dz, inv); anchor = 'tof' if fit else None
                    if not A.fit_quality(dz, inv, fit)['accepted']:
                        robust = A.robust_affine(dz, inv)['fit']
                        if robust is not None:
                            fit, anchor = robust, 'robust_tof'
                    if fit is None:
                        fit = S.floor_anchor(disp, dirs, F.rot(x['pitch'][i], x['roll'][i]), Rct, height, rng); anchor = 'floor' if fit else None
                    if fit is None and last is not None:
                        fit, anchor = last, 'last'
                    if fit is not None:
                        last = fit; a, b = fit
                        inv_d = (disp.reshape(-1) - b) / a
                        zz = np.where(inv_d > 1e-3, 1 / np.maximum(inv_d, 1e-3), np.inf)
                        ok = np.isfinite(zz) & (up.reshape(-1, 3).sum(1) > 0)
                        pts = np.full((h * w, 3), np.nan)
                        pts[ok] = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                        bl = [blobs(ok & (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= V1.Z_MAX), pts, h, w)
                              for lo, hi in F.BOXES.values()]
                        g = grid(pts, ok, sub)
                frames.append(dict(tof_index=int(i), t_ns=int(tof_t[i]), rgb_t_ns=int(rgb_t[j]), pitch=float(x['pitch'][i]), roll=float(x['roll'][i]),
                                   pose_ok=bool(pose_ok[i]), anchor=anchor, tof_strong=[bool(gated[i, q] == 2) for q in range(2)], blobs=bl))
                grids.append(g)
            np.savez_compressed(out / f'{name}.npz', grid=np.array(grids) if grids else np.zeros((0, 2, 30, 50), np.uint16), **{f'imu_{k}': v for k, v in imu_arrays(d).items()})
            (out / f'{name}.json').write_text(json.dumps(dict(label=labels[name], height=height, frames=frames)), encoding='utf8')
            print(name[17:23], labels[name]['class'], len(frames), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); main(a.raw, a.labels, a.out)
