"""How much does the head-aligned warning box disagree with a travel-aligned box? GT-only upper bound on ADT.

For prepared ADT frames while walking (horizontal speed > 0.3 m/s from the GT trajectory), GT points are evaluated in
the frozen HEAD/BODY boxes (|x| <= 0.3, z 0.3-2.0 m) in the head-aligned corridor frame (current design) and in a frame
rotated to the GT travel direction. Frames whose travel direction is > 20 deg off the camera heading are counted
separately (the travel box then leaves the camera FOV, so the camera cannot even observe it).
Usage (projectaria venv): adt_heading_box_dev_20261013.py <adt data dir> <prep dir>
"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider
from projectaria_tools.core.stream_id import StreamId

BOX = dict(HEAD=(-.20, .42), BODY=(.42, .90))


def qmat(q):
    x, y, z, w = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1), np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], 1)


def occ(p):
    return [int(((np.abs(p[:, 0]) <= .3) & (p[:, 1] >= lo) & (p[:, 1] <= hi) & (p[:, 2] >= .3) & (p[:, 2] <= 2.)).sum()) >= 38 for lo, hi in BOX.values()]


data, prep = Path(sys.argv[1]), Path(sys.argv[2]); tot = dict(walk=0, offfov=0, inview=0, disagree=0, head_only=0, travel_only=0)
for seq in sorted(p for p in prep.iterdir() if (p / 'frames.npz').exists() and 'sample' not in p.name):
    dp = AriaDigitalTwinDataProvider(AriaDigitalTwinDataPathsProvider(str(data / seq.name)).get_datapaths())
    cz = dp.get_aria_camera_calibration(StreamId('214-1')).get_transform_device_camera().to_matrix()[:3, 2]
    rows = list(csv.DictReader(open(data / seq.name / 'aria_trajectory.csv')))
    tt = np.array([float(r['tracking_timestamp_us']) for r in rows]) * 1e3
    P = np.array([[float(r[k]) for k in ('tx_world_device', 'ty_world_device', 'tz_world_device')] for r in rows])
    Q = np.array([[float(r[k]) for k in ('qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device')] for r in rows])
    meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    vv, uu = np.mgrid[0:h:2, 0:w:2]; dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
    with np.load(seq / 'frames.npz') as Z:
        dep, R, ts = Z['depth'], Z['R_cc'], Z['t_ns']
    for i, t in enumerate(ts):
        j = int(np.searchsorted(tt, t)); a, b = np.searchsorted(tt, t - 5e8), np.searchsorted(tt, t + 5e8)
        if a <= 0 or b >= len(tt):
            continue
        v = (P[b] - P[a]) / ((tt[b] - tt[a]) / 1e9); v[1] = 0
        if np.linalg.norm(v) < .3:
            continue
        f = qmat(Q[j:j + 1])[0] @ cz; f[1] = 0; f /= np.linalg.norm(f)
        xc = np.cross(np.array([0., -1., 0.]), f)
        phi = np.arctan2(v @ xc, v @ f); tot['walk'] += 1
        if abs(np.degrees(phi)) > 20:
            tot['offfov'] += 1; continue
        z = dep[i, ::2, ::2].reshape(-1) / 1000.; ok = z > .1
        p = (dirs[ok] * z[ok, None]) @ R[i].T
        pt = p.copy(); pt[:, 2] = p[:, 0] * np.sin(phi) + p[:, 2] * np.cos(phi); pt[:, 0] = p[:, 0] * np.cos(phi) - p[:, 2] * np.sin(phi)
        oh, ot = occ(p), occ(pt); tot['inview'] += 1
        for q in (0, 1):
            if oh[q] != ot[q]:
                tot['disagree'] += 1; tot['head_only' if oh[q] else 'travel_only'] += 1
print(json.dumps(tot))
n = tot['inview'] * 2
print(f"walking frames {tot['walk']}; travel >20 deg off heading (box outside camera view) {tot['offfov']} ({100*tot['offfov']/tot['walk']:.0f}%)")
print(f"in-view query-frames {n}: head-box vs travel-box disagree {tot['disagree']} ({100*tot['disagree']/n:.1f}%): head-only (likely false alarm) {tot['head_only']}, travel-only (likely miss) {tot['travel_only']}")
