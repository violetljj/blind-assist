"""Estimate the Atom camera <- ToF rotation and UniDepth metric scale from ToF zone ranges (EXPLORE).

Uses the calibrated, undistorted upright Atom image (atom_camera.AtomCamera) and UniDepthV2 with
known intrinsics. Each valid ToF zone (firmware status 5/6/9/10, 0.3-4 m) is sampled as 5x5
sub-rays over its 5.625 deg footprint; R is grid-searched (yaw/pitch/roll) to minimise the median
absolute log ratio between RGB range (after a fitted global scale) and ToF range. Translation
(a few cm between boards) is ignored. Frames: the 2026-10-12 hall batch (already consumed).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import AtomCamera, ROOT
import real_rgb_query_feasibility_dev_20261012 as F

OUT = ROOT / 'artifacts.local/work/atom-camera-calibration-20261012'
EDGE = math.tan(math.pi / 8)
SESSIONS = ('072853', '072939', '073020', '073138', '073245', '073554', '073740', '073821')
STRIDE = 6


def zone_rays(sub=5):
    """[64, sub*sub, 3] unit rays in the ToF model frame (row0 top, col0 left; x right, y down)."""
    off = (np.arange(sub) + .5) / sub
    rays = []
    for r in range(8):
        for c in range(8):
            xs = -EDGE + (c + off) * 2 * EDGE / 8; ys = -EDGE + (r + off) * 2 * EDGE / 8
            g = np.stack(list(np.meshgrid(xs, ys, indexing='xy')) + [np.ones((sub, sub))], -1).reshape(-1, 3)
            rays.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    return np.array(rays)


def euler(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def collect():
    import torch
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    rotation = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    cam = AtomCamera(); K = cam.K_upright
    model = F.load_model()
    samples = []
    for d in sorted(F.RAW.iterdir()):
        if d.name[17:23] not in SESSIONS:
            continue
        x = R.load(d, rotation)
        tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
        rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
        dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
        for i in range(0, len(tof_t), STRIDE):
            j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
            if abs(rgb_t[j] - tof_t[i]) > 6e7:
                continue
            valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
            if valid.sum() < 6:
                continue
            raw = cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1]
            up = cam.undistort_upright(raw)
            tensor = torch.from_numpy(up.transpose(2, 0, 1).copy()).cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                depth = model.infer(tensor, torch.from_numpy(K).float().cuda())['depth'][0, 0].float().cpu().numpy()
            samples.append(dict(session=d.name[17:23], tof_index=i, depth=depth, tof=dist[i], valid=valid))
        print('collected', d.name[17:23], len(samples), flush=True)
    return samples, K


def score(samples, K, R, rays):
    """Per-zone RGB range (median over sub-rays) for all samples; returns log ratios list."""
    logs = []
    rc = rays @ R.T                                            # [64,S,3] camera frame
    ok = rc[..., 2] > .05
    u = K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]; v = K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]
    for s in samples:
        h, w = s['depth'].shape
        inside = ok & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        ui = np.clip(u.round().astype(int), 0, w - 1); vi = np.clip(v.round().astype(int), 0, h - 1)
        z = s['depth'][vi, ui]
        rng = np.where(inside, z * np.linalg.norm(rc, axis=-1) / np.maximum(rc[..., 2], 1e-6), np.nan)
        zone = np.nanmedian(np.where(inside.sum(1, keepdims=True) >= 13, rng, np.nan), 1)
        m = s['valid'] & np.isfinite(zone)
        logs.append(np.log(zone[m] / s['tof'][m]))
    return np.concatenate(logs) if logs else np.array([])


def main():
    cache = OUT / 'extrinsic_samples.npz'
    if cache.exists():
        c = np.load(cache, allow_pickle=True); samples, K = list(c['samples']), c['K']
    else:
        samples, K = collect()
        np.savez_compressed(cache, samples=np.array(samples, dtype=object), K=K)
    rays = zone_rays()
    best = None
    for yaw in range(-20, 21, 2):
        for pitch in range(-24, 25, 2):
            for roll in range(-12, 13, 4):
                lr = score(samples, K, euler(yaw, pitch, roll), rays)
                if len(lr) < 200:
                    continue
                c = np.median(lr); cost = float(np.median(np.abs(lr - c)))
                if best is None or cost < best[0]:
                    best = (cost, yaw, pitch, roll, float(c), len(lr))
    _, y0, p0, r0, _, _ = best
    for yaw in np.arange(y0 - 2, y0 + 2.01, .5):           # refine
        for pitch in np.arange(p0 - 2, p0 + 2.01, .5):
            for roll in np.arange(r0 - 4, r0 + 4.01, 1):
                lr = score(samples, K, euler(yaw, pitch, roll), rays)
                c = np.median(lr); cost = float(np.median(np.abs(lr - c)))
                if cost < best[0]:
                    best = (cost, float(yaw), float(pitch), float(roll), float(c), len(lr))
    cost, yaw, pitch, roll, logscale, n = best
    ident = score(samples, K, np.eye(3), rays); ic = np.median(ident)
    result = dict(camera_from_tof_euler_deg=dict(yaw=yaw, pitch=pitch, roll=roll), rotation=euler(yaw, pitch, roll).tolist(),
                  unidepth_scale_rgb_over_tof=float(math.exp(logscale)), median_abs_log_residual=cost, zones=n, frames=len(samples),
                  identity_median_abs_log_residual=float(np.median(np.abs(ident - ic))), identity_scale=float(math.exp(ic)),
                  per_session_scale={s: float(math.exp(np.median(score([x for x in samples if x['session'] == s], K, euler(yaw, pitch, roll), rays))))
                                     for s in sorted({x['session'] for x in samples})},
                  note='Rotation-only fit; translation ignored; samples from consumed hall batch; UniDepthV2 with calibrated K.')
    (OUT / 'camera_tof_extrinsic.json').write_text(json.dumps(result, indent=1), encoding='utf8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
