"""Phone-sized RGB depth for the RGB-or-ToF fusion: DAV2-Small relative depth anchored by ToF (EXPLORE).

DAV2-Small (24.8M params; frozen weights embedded in the DEPTHOR-ZJU-Small checkpoint, 239 keys)
gives affine-invariant inverse depth. Each frame is anchored to metric depth with the valid ToF
zones: robust fit disparity = a / z + b over zone footprints (z = optical depth along the camera
ray, via the fitted camera<-ToF rotation); frames with < 6 usable zones reuse the segment's last
fit. The query readout, MIN_PX, pose gate and ToF cache are exactly those of
real_rgb_tof_fusion_dev_20261012 (UniDepthV2 ViT-L reference). Input sizes 518 and 308 px.
Development description on already-viewed captures; labels are Claude's reading.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
import real_rgb_tof_fusion_dev_20261012 as U
import pose_envelope as PE

HEIGHT = 1.35
ONLY_ONLINE = '--online-only' in sys.argv

CHECKPOINT = ROOT / 'artifacts.local/work/mz140-depthor-20260915/depthor-zju-small.pt'  # depth_anything.* identical to official DAV2-Small (Apache-2.0)
UPSTREAM = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/Depth-Anything-V2'
OUT = U.OUT
SIZES = (518, 308)
HEAD_HEIGHTS = (1.35, 1.25, 1.45)   # assumed sensor height above floor for ground-plane anchoring


def load_dav2_small():
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from depth_anything_v2.dpt import DepthAnythingV2
    weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = {k[len('depth_anything.'):]: v for k, v in weights.items() if k.startswith('depth_anything.')}
    if len(state) != 239:
        raise ValueError('Unexpected DAV2-Small key count')
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(state, strict=True)
    return model.cuda().eval(), sum(p.numel() for p in model.parameters())


def zone_disparity(disp, K, Rct, rays):
    """Median disparity and camera optical depth factor per ToF zone (NaN when off-image)."""
    rc = rays @ Rct.T
    u = K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]; v = K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]
    h, w = disp.shape
    inside = (rc[..., 2] > .05) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    d = disp[np.clip(v.round().astype(int), 0, h - 1), np.clip(u.round().astype(int), 0, w - 1)]
    dz = np.nanmedian(np.where(inside, d, np.nan), 1)
    zfac = np.nanmedian(np.where(inside, rc[..., 2] / np.linalg.norm(rc, axis=-1), np.nan), 1)   # z = range * zfac
    dz[inside.sum(1) < 13] = np.nan
    return dz, zfac


def fit_affine(dz, inv_z):
    """Robust d = a * inv_z + b (two trimming passes); None if ill-posed."""
    m = np.isfinite(dz) & np.isfinite(inv_z)
    if m.sum() < 6:
        return None
    x, y = inv_z[m], dz[m]
    for _ in range(3):
        A = np.stack([x, np.ones_like(x)], 1)
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - (a * x + b); s = np.median(np.abs(r)) * 1.4826 + 1e-9
        keep = np.abs(r) < 2.5 * s
        if keep.sum() < 6 or keep.all():
            break
        x, y = x[keep], y[keep]
    return (float(a), float(b)) if a > 0 else None


def floor_height(pts):
    """Floor depth below the sensor from metric corridor points: mode of y for ground-like points."""
    c = pts[(pts[:, 2] > .8) & (pts[:, 2] < 4.) & (np.abs(pts[:, 0]) < 1.5) & (pts[:, 1] > .8) & (pts[:, 1] < 2.2)]
    if len(c) < 2000:
        return None
    hist, edges = np.histogram(c[:, 1], bins=np.arange(.8, 2.22, .02))
    k = int(np.argmax(hist))
    if hist[k] < .05 * len(c):
        return None
    sel = c[(c[:, 1] >= edges[max(k - 2, 0)]) & (c[:, 1] < edges[min(k + 3, len(edges) - 1)]), 1]
    return float(np.median(sel))


def floor_anchor(disp, dirs, Rpose, Rct, height, rng):
    """Fit disparity = a * inv_z + b on pixels whose ray meets the floor 0.8-6 m ahead (RANSAC)."""
    r = (Rpose @ (Rct.T @ dirs.T)).T
    inv_z = r[:, 1] / height                     # camera optical depth t = height / r_y (dirs have z = 1)
    cand = np.flatnonzero((r[:, 1] > .1) & (inv_z > 1 / 6.) & (inv_z < 1 / .8))
    if len(cand) < 500:
        return None
    sel = rng.choice(cand, min(4000, len(cand)), replace=False)
    x, y = inv_z[sel], disp.reshape(-1)[sel]
    best = None
    for _ in range(64):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-3:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j]); b = y[i] - a * x[i]
        if a <= 0:
            continue
        inl = np.abs(y - (a * x + b)) < .03 * (np.percentile(y, 95) - np.percentile(y, 5) + 1e-6)
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best is None or best[0] < .3 * len(x):
        return None
    inl = best[1]
    a, b = np.polyfit(x[inl], y[inl], 1)
    return (float(a), float(b)) if a > 0 else None


def main():
    import torch
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    ext = json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text(encoding='utf8'))
    Rct = np.array(ext['rotation'])
    cam = AtomCamera(); K = cam.K_upright
    model, params = load_dav2_small(); rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    reference = json.loads((OUT / 'rgb_tof_fusion_readout.json').read_text(encoding='utf8'))
    report = dict(model='DAV2-Small relative (DEPTHOR embedded)', parameters=int(params), sizes={})
    global HEIGHT
    rng = np.random.default_rng(20261012)
    configs = [(518, 'online')] if ONLY_ONLINE else [(308, 1.35), (308, 1.25), (308, 1.45), (518, 1.35), (518, 'online')]
    for size, HEIGHT in configs:
        online = HEIGHT == 'online'
        latencies = []; segs = {}
        for d in sorted(F.RAW.iterdir()):
            key = d.name[17:23]
            if key not in U.SCENES:
                continue
            x = R.load(d, imu_rot)
            tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
            rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
            dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
            cached = np.load(U.NEW / 'joint_replay' / f'{d.name}_imu.npz')
            gated, valid_pose = PE.apply(cached['S_grade'], x['pitch'], x['roll']); ready = cached['ready'].astype(bool)
            last = None; rows = []; heights = []
            for i in range(0, len(tof_t), U.STRIDE):
                if not ready[i]:
                    continue
                j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                if abs(rgb_t[j] - tof_t[i]) > 1e8:
                    continue
                up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
                torch.cuda.synchronize(); t0 = time.perf_counter()
                disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), size)          # expects BGR
                torch.cuda.synchronize(); latencies.append(time.perf_counter() - t0)
                dz, zfac = zone_disparity(disp, K, Rct, rays)
                valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                inv_z = np.where(valid, 1 / (dist[i] * zfac), np.nan)
                fit = fit_affine(dz, inv_z); source = 'tof' if fit else None
                if fit is None:
                    h_use = (float(np.median(heights)) if len(heights) >= 3 else 1.35) if online else HEIGHT
                    fit = floor_anchor(disp, dirs, F.rot(x['pitch'][i], x['roll'][i]), Rct, h_use, rng); source = 'floor' if fit else None
                fresh = fit is not None
                fit = fit or last
                row = dict(tof_index=i, anchored=fresh, anchor=source, pose_ok=bool(valid_pose[i]), tof_grade=[int(v) for v in gated[i]])
                if fit is None:
                    for name in F.BOXES: row[name] = dict(px=0, rgb=False)
                    rows.append(row); continue
                last = fit; a, b = fit
                inv = (disp.reshape(-1) - b) / a
                z = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)
                okpx = np.isfinite(z) & (up.reshape(-1, 3).sum(1) > 0)
                pts = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ (dirs[okpx] * z[okpx, None]).T)).T
                if source == 'tof':
                    fh = floor_height(pts)
                    if fh is not None: heights.append(fh)
                for name, (lo, hi) in F.BOXES.items():
                    inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= U.Z_MAX)
                    zz = np.sort(pts[inside, 2])
                    row[name] = dict(px=int(inside.sum()), z_near=float(zz[min(15, len(zz) - 1)]) if len(zz) else None,
                                     rgb=bool(inside.sum() >= U.MIN_PX and valid_pose[i]))
                rows.append(row)
            segs[key] = dict(frames=rows, floor_height_estimates=heights)
            n = max(len(rows), 1); ref = reference[key]['frames']
            hs = f"h~{np.median(heights):.2f}({len(heights)})" if heights else 'h=1.35(0)'
            def frac(rs, fn): return sum(fn(r) for r in rs) / max(len(rs), 1)
            print(f"[{size} H{'online' if online else HEIGHT}] {key} {U.SCENES[key][:30]:30s} {hs:12s} anchor tof/floor {frac(rows, lambda r: r['anchor'] == 'tof'):4.0%}/{frac(rows, lambda r: r['anchor'] == 'floor'):4.0%} | RGB-S H/B "
                  f"{frac(rows, lambda r: r['HEAD']['rgb']):4.0%}/{frac(rows, lambda r: r['BODY']['rgb']):4.0%}  UniDepth-L H/B "
                  f"{frac(ref, lambda r: r['HEAD']['rgb']):4.0%}/{frac(ref, lambda r: r['BODY']['rgb']):4.0%} | OR-S H/B "
                  f"{frac(rows, lambda r: r['tof_grade'][0] == 2 or r['HEAD']['rgb']):4.0%}/{frac(rows, lambda r: r['tof_grade'][1] == 2 or r['BODY']['rgb']):4.0%}", flush=True)
        report['sizes'][f"{size}_H{'online' if online else HEIGHT}"] = dict(latency_ms_median=float(np.median(latencies) * 1000), latency_ms_p95=float(np.percentile(latencies, 95) * 1000), segments=segs)
        print(f'[{size}] DAV2-S latency median {np.median(latencies)*1000:.1f} ms, p95 {np.percentile(latencies, 95)*1000:.1f} ms (laptop GPU, incl. resize)', flush=True)
    (OUT / ('rgb_small_model_online_height.json' if ONLY_ONLINE else 'rgb_small_model_readout.json')).write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
