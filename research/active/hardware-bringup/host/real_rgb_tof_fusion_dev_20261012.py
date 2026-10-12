"""Calibrated Atom RGB query readout and a simple RGB OR ToF fusion on the 2026-10-12 captures (EXPLORE).

RGB: calibrated undistorted upright image -> UniDepthV2 (known K) -> points -> ToF frame via the
fitted camera<-ToF rotation (atom_tof_extrinsic_dev_20261012) -> per-frame metric scale anchored to
valid ToF zones (median ToF/RGB range ratio; global fit if < 6 zones) -> corridor frame with the
frame's IMU pitch/roll -> pixels inside HEAD/BODY boxes (x +/-0.30, y -0.20..0.42 / 0.42..0.90,
z 0.30..2.0 m: the timely window). RGB alert: >= MIN_PX pixels in a box. ToF: cached frozen joint
stack grades (Codex REAL_PHONE_NEW_BATCH), pose-envelope gated. Fusion = ToF strong OR RGB alert.
All hall/office segments were already viewed; MIN_PX fixed a priori (not tuned). Labels: Claude's
reading of RGB. Development description only.
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
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
import pose_envelope as PE

NEW = ROOT / 'artifacts.local/work/phone-new-capture-20261012'
OUT = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012'
MIN_PX = 150          # ~ a 2.5 cm pole spanning half a box band at 2 m; fixed before running
Z_MAX = 2.0
STRIDE = 2
SCALE_RANGE = (.60, 1.05)   # per-session medians 0.73-0.93, global fit 0.79
SCENES = dict(F.SCENES, **{
    '065950': 'office chair close', '070046': 'monitor and desk', '070523': 'desk with bottle, desk edge',
    '070601': 'room toward TV wall', '070630': 'room looking down, side chair', '070657': 'room looking up',
    '070849': 'two seated people, chairs', '070942': 'extreme up / head tilted', '071014': 'looking down at desk and knees'})


def main():
    import torch
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    ext = json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text(encoding='utf8'))
    Rct = np.array(ext['rotation']); global_scale = 1 / ext['unidepth_scale_rgb_over_tof']
    cam = AtomCamera(); K = cam.K_upright
    model = F.load_model(); rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    report = {}
    for d in sorted(F.RAW.iterdir()):
        key = d.name[17:23]
        if key not in SCENES:
            continue
        x = R.load(d, imu_rot)
        tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
        rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
        dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
        cached = np.load(NEW / 'joint_replay' / f'{d.name}_imu.npz')
        grade, ready = cached['S_grade'], cached['ready'].astype(bool)
        if len(grade) != len(tof_t):
            raise ValueError(f'{key}: cached grades {len(grade)} vs ToF frames {len(tof_t)}')
        gated, valid_pose = PE.apply(grade, x['pitch'], x['roll'])
        rows = []
        for i in range(0, len(tof_t), STRIDE):
            if not ready[i]:
                continue
            j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
            if abs(rgb_t[j] - tof_t[i]) > 1e8:
                continue
            up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
            tensor = torch.from_numpy(up.transpose(2, 0, 1).copy()).cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                depth = model.infer(tensor, torch.from_numpy(K).float().cuda())['depth'][0, 0].float().cpu().numpy()
            sample = dict(depth=depth, tof=dist[i], valid=np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.))
            lr = X.score([sample], K, Rct, rays)
            scale = float(math.exp(-np.median(lr))) if len(lr) >= 6 else global_scale
            # Thin near returns can drag the zone median; bound the per-frame scale around the fit.
            scale = float(np.clip(scale, SCALE_RANGE[0], SCALE_RANGE[1]))
            pts_cam = dirs * depth.reshape(-1, 1) * scale
            valid_px = (up.reshape(-1, 3).sum(1) > 0) & (depth.reshape(-1) > 0)          # drop undistortion border
            pts = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ pts_cam[valid_px].T)).T
            row = dict(tof_index=i, scale=scale, scale_zones=int(len(lr)), pitch=float(x['pitch'][i]),
                       pose_ok=bool(valid_pose[i]), tof_grade=[int(v) for v in gated[i]])
            for q, (name, (lo, hi)) in enumerate(F.BOXES.items()):
                inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= Z_MAX)
                z = np.sort(pts[inside, 2])
                cols = (np.flatnonzero(valid_px)[inside] % w)
                span = float(np.percentile(cols, 95) - np.percentile(cols, 5)) if len(cols) else None
                row[name] = dict(px=int(inside.sum()), z_near=float(z[min(15, len(z) - 1)]) if len(z) else None, u_span_px=span,
                                 rgb=bool(inside.sum() >= MIN_PX and valid_pose[i]))
            rows.append(row)
        report[key] = dict(scene=SCENES[key], session=d.name, frames=rows)
        n = len(rows)
        def frac(fn): return sum(fn(r) for r in rows) / max(n, 1)
        line = f"{key} {SCENES[key][:40]:40s} n={n:3d} scale~{np.median([r['scale'] for r in rows]) if rows else float('nan'):.2f} |"
        for q, name in enumerate(('HEAD', 'BODY')):
            line += (f" {name}: ToF strong {frac(lambda r: r['tof_grade'][q] == 2):4.0%} light {frac(lambda r: r['tof_grade'][q] == 1):4.0%}"
                     f" RGB {frac(lambda r: r[name]['rgb']):4.0%} OR {frac(lambda r: r['tof_grade'][q] == 2 or r[name]['rgb']):4.0%} |")
        print(line, flush=True)
    (OUT / 'rgb_tof_fusion_readout.json').write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--weights', help='alternative UniDepthV2 checkpoint dir (e.g. unidepth-v2-vits14)')
    a = ap.parse_args()
    if a.weights:
        F.WEIGHTS = Path(a.weights); OUT = OUT / Path(a.weights).name.split('--')[-1].replace('unidepth-v2-', 'unidepth_'); OUT.mkdir(exist_ok=True)
    main()
