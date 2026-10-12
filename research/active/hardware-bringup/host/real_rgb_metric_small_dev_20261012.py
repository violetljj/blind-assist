"""DAV2-Small metric (Hypersim, 24.8M params) in the RGB-or-ToF fusion readout (EXPLORE).

Same calibrated camera, camera<-ToF rotation, query boxes, MIN_PX, pose gate and ToF cache as
real_rgb_tof_fusion_dev_20261012 (UniDepthV2 ViT-L reference). Metric depth is rescaled per frame by
the median ToF/RGB zone range ratio when >= 6 valid zones (clipped to SCALE_RANGE), otherwise used
as predicted (scale 1). Input sizes 518 and 308 px. Development description on viewed captures.
"""
from __future__ import annotations

import json
import math
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

WEIGHT = ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/depth_anything_v2_metric_hypersim_vits.pth'
UPSTREAM = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/Depth-Anything-V2/metric_depth'
SCALE_RANGE = (.6, 1.4)


def load():
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
    model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    return model.cuda().eval(), sum(p.numel() for p in model.parameters())


def main():
    import torch
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    Rct = np.array(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text(encoding='utf8'))['rotation'])
    cam = AtomCamera(); K = cam.K_upright
    model, params = load(); rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    reference = json.loads((U.OUT / 'rgb_tof_fusion_readout.json').read_text(encoding='utf8'))
    report = dict(model='DAV2-Small metric Hypersim', parameters=int(params), sizes={})
    for size in (518, 308):
        lat, segs, scales = [], {}, []
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
            rows = []
            for i in range(0, len(tof_t), U.STRIDE):
                if not ready[i]:
                    continue
                j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                if abs(rgb_t[j] - tof_t[i]) > 1e8:
                    continue
                up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
                torch.cuda.synchronize(); t0 = time.perf_counter()
                depth = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), size)
                torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
                sample = dict(depth=depth, tof=dist[i], valid=np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.))
                lr = X.score([sample], K, Rct, rays)
                scale = float(np.clip(math.exp(-np.median(lr)), *SCALE_RANGE)) if len(lr) >= 6 else 1.
                if len(lr) >= 6: scales.append(math.exp(-np.median(lr)))
                okpx = (up.reshape(-1, 3).sum(1) > 0) & (depth.reshape(-1) > 0)
                pts = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ (dirs[okpx] * depth.reshape(-1)[okpx, None] * scale).T)).T
                row = dict(tof_index=i, scale=scale, anchored=len(lr) >= 6, tof_grade=[int(v) for v in gated[i]])
                for name, (lo, hi) in F.BOXES.items():
                    inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= U.Z_MAX)
                    zz = np.sort(pts[inside, 2])
                    row[name] = dict(px=int(inside.sum()), z_near=float(zz[min(15, len(zz) - 1)]) if len(zz) else None,
                                     rgb=bool(inside.sum() >= U.MIN_PX and valid_pose[i]))
                rows.append(row)
            segs[key] = rows; ref = reference[key]['frames']
            def frac(rs, fn): return sum(fn(r) for r in rs) / max(len(rs), 1)
            print(f"[{size}] {key} {U.SCENES[key][:34]:34s} anchored {frac(rows, lambda r: r['anchored']):4.0%} | RGB-Sm H/B "
                  f"{frac(rows, lambda r: r['HEAD']['rgb']):4.0%}/{frac(rows, lambda r: r['BODY']['rgb']):4.0%}  UniDepth-L H/B "
                  f"{frac(ref, lambda r: r['HEAD']['rgb']):4.0%}/{frac(ref, lambda r: r['BODY']['rgb']):4.0%} | OR H/B "
                  f"{frac(rows, lambda r: r['tof_grade'][0] == 2 or r['HEAD']['rgb']):4.0%}/{frac(rows, lambda r: r['tof_grade'][1] == 2 or r['BODY']['rgb']):4.0%}", flush=True)
        report['sizes'][str(size)] = dict(latency_ms_median=float(np.median(lat) * 1000), raw_tof_scale_median=float(np.median(scales)) if scales else None, segments=segs)
        print(f'[{size}] latency median {np.median(lat)*1000:.1f} ms; raw ToF/RGB scale median {np.median(scales):.2f}', flush=True)
    (U.OUT / 'rgb_metric_small_readout.json').write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
