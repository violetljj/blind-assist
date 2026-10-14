"""CNH multi-peak vs firmware-like strongest peak on ADT geometry (plan: adt-eval-20261013/PLAN_HIST_THIN.md).

Histograms from the declared CNH forward proxy (dtr-r0/nearfield/cnh_route_sensor, ASSUMED electronics, H3 readout)
with 64 rays per zone ranged from ADT GT depth through the camera<-ToF extrinsic. Per frame, tilt and box: alarm A
(strongest-peak distance in the box), alarm B (any >= 5-sigma bin whose range along the zone centre ray is in the box),
plus the nearest alarm depth of each. Post hoc (added after the registered A/B result, PLAN_HIST_THIN addendum):
B2 = B minus bins whose 0.30 m range interval overlaps the zone's floor-hit range interval (ray-plane intersection of
the zone's rays with the gravity floor at the sequence's camera height, +-0.15 m) -- known on the device from IMU
gravity + calibrated height. (B2 turned out identical to B: the floor never entered the box.)
B3 = B after subtracting the proxy's known residual-crosstalk template (aggregated to H3 bins) before the 5-sigma test,
as the real pipeline subtracts its calibrated per-zone signed bias; added after the B-only false alarms were seen to
sit in the 0.3-0.6 m bin. (B3 also changed nothing: those alarms are box-edge objects, e.g. hands at z < 0.3 m and
things just beyond 2 m, folded into 0.30 m bins.) B4 = B3 counting a bin only if its whole range interval lies inside the
box depth range along the zone centre ray (resolution-aware box edges). Last post hoc variant of this run. Output: <out>/<seq>.t<tilt>.hist.json. Usage: <prep dir> <out dir> <n procs>
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'dtr-r0/nearfield'))
import frozen_fusion_eval_v1 as V1
from adt_tof_tilt_sweep_dev_20261014 import rx, BOXES
import cnh_route_sensor as C

TILTS = (0, 30)
SNR = 5.0


def in_box(p):
    """p [...,3] corridor points -> [..., 2] bool per box."""
    return np.stack([(np.abs(p[..., 0]) <= .3) & (p[..., 1] >= lo) & (p[..., 1] <= hi) & (p[..., 2] >= .3) & (p[..., 2] <= 2.)
                     for lo, hi in BOXES], -1)


def one(args):
    prep, out, si, seq = args
    if all((Path(out) / f'{seq}.t{t}.hist.json').exists() for t in TILTS):
        return seq
    meta = json.loads((Path(prep) / seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    with np.load(Path(prep) / seq / 'frames.npz') as Z:
        depth, R, cam_y = Z['depth'], Z['R_cc'], Z['cam_y']
    vv, uu = np.mgrid[0:h, 0:w]; dpx = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
    hts = []
    for i in range(0, len(cam_y), max(1, len(cam_y) // 20)):
        zz_ = depth[i].reshape(-1) / 1000.; ok_ = zz_ > .1
        hts.append(cam_y[i] - (R[i] @ (dpx[ok_] * zz_[ok_, None]).T).T[:, 1])
    height = float(np.median(cam_y) - np.percentile(np.concatenate(hts), 1)); del dpx
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    dirs0, wts = C.angular_rays(8)                                    # [8,8,64,3] ToF frame, X right Y down Z fwd
    params = C.SensorParameters(); cfg = C.H3
    xb = int(np.floor((params.crosstalk_range_m - params.range_zero_m) / C.RAW_BIN_M))
    X = params.signal_counts * params.crosstalk_fraction * C._pulse_matrix(params)[xb] * params.output_gain
    X = X[cfg.start_bin:cfg.start_bin + cfg.bins * cfg.sub_sample].reshape(cfg.bins, cfg.sub_sample).sum(-1)   # [16]
    for t in TILTS:
        rc = (dirs0 @ rx(-t).T) @ Rct.T                               # camera frame
        u = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); v = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
        inside = (rc[..., 2] > .05) & (u >= 0) & (u < w) & (v >= 0) & (v < h); u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        ctr = rc.mean(2); ctr /= np.linalg.norm(ctr, axis=-1, keepdims=True)   # zone centre rays, camera frame [8,8,3]
        rows = []
        for i in range(len(depth)):
            z = depth[i][v, u] / 1000.
            radial = np.where(inside & (z > .1), z / np.maximum(rc[..., 2], 1e-6), np.nan)
            resp = C.synthesize_response(radial, .5, 1., wts, params=params, seed=20261014 * 1000 + (si * 100000 + i) * 64 + t)   # int seed per (sequence, frame, tilt)
            rd = C.derive_readout(resp, cfg)
            H = rd['histogram']; amb = rd['ambient']; centers = rd['bin_centers_m']
            cz = ctr @ R[i].T                                          # zone centre rays, corridor frame
            # A: strongest peak
            pA = cz * np.nan_to_num(rd['distance_m'], nan=-1.)[..., None]
            okA = rd['valid'][..., None] & in_box(pA)                    # [8,8,2]
            # B: every significant bin
            sig = np.sqrt(np.maximum(H, 0) + 2 * amb[..., None] * cfg.sub_sample)
            det = H >= SNR * np.maximum(sig, 1e-9)                      # [8,8,16]
            pB = cz[..., None, :] * centers[None, None, :, None]        # [8,8,16,3]
            okB = det[..., None] & in_box(pB)                           # [8,8,16,2]
            ry = ((rc / np.linalg.norm(rc, axis=-1, keepdims=True)) @ R[i].T)[..., 1]   # unit-ray down component [8,8,64]
            fr = np.where(ry > .02, height / np.maximum(ry, .02), np.nan)            # floor-hit range per ray
            lo_f = np.nanmin(np.where(np.isfinite(fr), fr, np.inf), -1) - .15; hi_f = np.nanmax(np.where(np.isfinite(fr), fr, -np.inf), -1) + .15
            half = cfg.sub_sample * C.RAW_BIN_M / 2
            floor_bin = (centers[None, None, :] + half >= lo_f[..., None]) & (centers[None, None, :] - half <= hi_f[..., None])
            okB2 = okB & ~floor_bin[..., None]
            okB3 = ((H - X) >= SNR * np.maximum(sig, 1e-9))[..., None] & in_box(pB)
            zlo = (cz[..., None, :] * (centers - half)[None, None, :, None])[..., 2]; zhi = (cz[..., None, :] * (centers + half)[None, None, :, None])[..., 2]
            okB4 = okB3 & ((zlo >= .3) & (zhi <= 2.))[..., None]
            row = []
            for q in (0, 1):
                zA = pA[..., 2][okA[..., q]]; zB = pB[..., 2][okB[..., q]]
                row.append([bool(okA[..., q].any()), None if zA.size == 0 else round(float(zA.min()), 3),
                            bool(okB[..., q].any()), None if zB.size == 0 else round(float(zB.min()), 3), bool(okB2[..., q].any()), bool(okB3[..., q].any()), bool(okB4[..., q].any())])
            rows.append(row)
        (Path(out) / f'{seq}.t{t}.hist.json').write_text(json.dumps(rows), encoding='utf8')
    return seq


def main(prep, out, n):
    Path(out).mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in Path(prep).iterdir() if (p / 'frames.npz').exists())
    with Pool(int(n)) as pool:
        for s in pool.imap_unordered(one, [(prep, out, si, s) for si, s in enumerate(seqs)]):
            print('done', s, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
