"""Offline multi-view reconstruction + ToF scale as automatic distance truth? (EXPLORE, 2026-10-14)

Question: can a feed-forward multi-view model (MapAnything, Apache weights) run on a whole recorded Atom RGB
segment give metric geometry good enough to label our own recordings without manual work?
Per session: RGB frames matched (<=60 ms receipt gap) to valid ToF frames are undistorted (Atom fisheye calibration),
fed with the calibrated intrinsics, and reconstructed jointly. ToF zone rays are mapped into the camera with the
pitch-0 camera<-ToF extrinsic and the reconstruction range is sampled per zone (5x5 sub-rays).
Checks (consumed Development data, no tape-measured truth):
  1. raw metric agreement: median ToF / reconstruction range on locally flat zones (scale MapAnything got by itself);
  2. held-out consistency: one scale fitted on even views, relative error on odd views;
  3. corridor nearest-obstacle series (HEAD/BODY boxes as the frozen pipeline), single view vs the whole-segment map
     (later, closer views labelling earlier ones).
Usage: recon_pseudo_gt_dev_20261014.py --raw <dir> --sessions 054159 054405 --out <dir> [--max-views 40]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import atom_tof_extrinsic_dev_20261012 as X
import real_phone_imu_replay as R
import real_rgb_query_feasibility_dev_20261012 as F
from atom_camera import AtomCamera, ROOT
from atom_imu import load_mounting

EXTRINSIC = ROOT / 'artifacts.local/work/hindsight-truth-dev-20261014/pitchfix/camera_tof_extrinsic_pitch0.json'
HF_HOME = ROOT / 'artifacts.local/cache/hf-mapanything'
VALID = (5, 6, 9, 10)
FLAT = 0.08            # zone counted as locally flat when reconstruction range IQR / median < this
BOX_X, Z_MIN, Z_MAX, MIN_PTS = .30, .30, 3.0, 150


def session_inputs(d, cam, imu_rot, max_views):
    x = R.load(d, imu_rot)
    tof_t = np.array([r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')])
    dist = np.where(np.isin(x['S'], VALID), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
    rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
    pairs = []
    for i, t in enumerate(tof_t):
        j = int(np.argmin(np.abs(rgb_t - t)))
        if abs(rgb_t[j] - t) <= 6e7 and np.isfinite(x['pitch'][i]):
            pairs.append((i, j))
    if len(pairs) > max_views:
        pairs = [pairs[k] for k in np.linspace(0, len(pairs) - 1, max_views).round().astype(int)]
    imgs = [cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1]) for _, j in pairs]
    return dict(pairs=pairs, imgs=imgs, dist=np.array([dist[i] for i, _ in pairs]),
                pitch=np.array([x['pitch'][i] for i, _ in pairs]), roll=np.array([x['roll'][i] for i, _ in pairs]),
                t=np.array([tof_t[i] for i, _ in pairs]))


def reconstruct(model, imgs, K):
    import torch
    from mapanything.utils.image import preprocess_inputs
    views = [dict(img=torch.from_numpy(im), intrinsics=torch.tensor(K, dtype=torch.float32)) for im in imgs]
    pred = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True, amp_dtype='bf16',
                       apply_mask=True, mask_edges=True)
    out = []
    for p in pred:
        out.append(dict(pts_cam=p['pts3d_cam'][0].float().cpu().numpy(), pts_world=p['pts3d'][0].float().cpu().numpy(),
                        mask=p['mask'][0, ..., 0].cpu().numpy().astype(bool), conf=p['conf'][0].float().cpu().numpy(),
                        K=p['intrinsics'][0].float().cpu().numpy(), pose=p['camera_poses'][0].float().cpu().numpy()))
    return out


def zone_ranges(view, rays_cam):
    """Per zone median reconstruction range and IQR/median over the 5x5 sub-rays that land on valid pixels."""
    rng = np.linalg.norm(view['pts_cam'], axis=-1); h, w = rng.shape; K = view['K']
    u = K[0, 0] * rays_cam[..., 0] / rays_cam[..., 2] + K[0, 2]; v = K[1, 1] * rays_cam[..., 1] / rays_cam[..., 2] + K[1, 2]
    inside = (rays_cam[..., 2] > .05) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    ui, vi = np.clip(u.round().astype(int), 0, w - 1), np.clip(v.round().astype(int), 0, h - 1)
    r = np.where(inside & view['mask'][vi, ui], rng[vi, ui], np.nan)
    ok = np.isfinite(r).sum(1) >= 13
    with np.errstate(all='ignore'):
        med = np.where(ok, np.nanmedian(r, 1), np.nan)
        spread = np.where(ok, (np.nanpercentile(r, 75, 1) - np.nanpercentile(r, 25, 1)) / med, np.nan)
    return med, spread


def nearest_in_boxes(pts_cam, pitch, roll, Rct):
    """Nearest forward distance (m) of >= MIN_PTS reconstruction points in the HEAD/BODY corridor boxes, else inf."""
    p = (F.rot(pitch, roll) @ (Rct.T @ pts_cam.reshape(-1, 3).T)).T
    res = {}
    for name, (lo, hi) in F.BOXES.items():
        z = p[(np.abs(p[:, 0]) <= BOX_X) & (p[:, 1] >= lo) & (p[:, 1] <= hi) & (p[:, 2] >= Z_MIN) & (p[:, 2] <= Z_MAX), 2]
        res[name] = float(np.sort(z)[MIN_PTS - 1]) if len(z) >= MIN_PTS else float('inf')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True); ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--max-views', type=int, default=40)
    a = ap.parse_args()
    os.environ.setdefault('HF_HOME', str(HF_HOME)); os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    from mapanything.models import MapAnything
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cam = AtomCamera(); K = cam.K_upright
    Rct = np.array(json.loads(EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    rays_cam = X.zone_rays() @ Rct.T                                  # [64, 25, 3] in the upright camera frame
    model = MapAnything.from_pretrained('facebook/map-anything-apache').to('cuda').eval()
    report = dict(protocol='RECON_PSEUDO_GT_DEV_20261014', model='facebook/map-anything-apache', extrinsic=str(EXTRINSIC.relative_to(ROOT)),
                  flat_iqr_ratio=FLAT, sessions={})
    for d in sorted(Path(a.raw).iterdir()):
        if d.name[17:23] not in a.sessions:
            continue
        s = session_inputs(d, cam, imu_rot, a.max_views)
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        views = reconstruct(model, s['imgs'], K)
        secs, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
        n = len(views); rec = np.full((n, 64), np.nan); spread = np.full((n, 64), np.nan)
        for k, v in enumerate(views):
            rec[k], spread[k] = zone_ranges(v, rays_cam)
        tof = s['dist']; use = np.isfinite(tof) & (tof > .3) & (tof < 4.) & np.isfinite(rec) & (spread < FLAT)
        ratio = tof / rec
        even, odd = use & (np.arange(n)[:, None] % 2 == 0), use & (np.arange(n)[:, None] % 2 == 1)
        scale = float(np.median(ratio[even])) if even.any() else float('nan')
        rel = np.abs(scale * rec[odd] - tof[odd]) / tof[odd]
        # Corridor series: this view alone vs the whole-segment map (all views, world frame -> this camera).
        world = np.concatenate([v['pts_world'][v['mask'] & (v['conf'] >= np.percentile(v['conf'][v['mask']], 30))] for v in views])
        world = world.astype(np.float32) * scale   # all points: a random subsample hid thin poles under MIN_PTS
        np.savez_compressed(out / f'{d.name[17:23]}_views.npz', pts_cam=np.stack([v['pts_cam'] for v in views]).astype(np.float16),
                            mask=np.stack([v['mask'] for v in views]), pose=np.stack([v['pose'] for v in views]), K=np.stack([v['K'] for v in views]),
                            pitch=s['pitch'], roll=s['roll'])
        series = []
        for k, v in enumerate(views):
            single = nearest_in_boxes(v['pts_cam'][v['mask']] * scale, s['pitch'][k], s['roll'][k], Rct)
            P = v['pose'].copy(); P[:3, 3] *= scale; Pinv = np.linalg.inv(P)
            in_cam = (Pinv[:3, :3] @ world.T).T + Pinv[:3, 3]
            fused = nearest_in_boxes(in_cam, s['pitch'][k], s['roll'][k], Rct)
            tof_min = float(np.min(np.where(np.isfinite(tof[k]) & (tof[k] > .3), tof[k], np.inf)))
            series.append(dict(t_s=round(float((s['t'][k] - s['t'][0]) / 1e9), 3), single=single, map=fused, tof_min_m=tof_min))
        res = dict(views=n, recon_seconds=round(secs, 1), peak_gpu_gb=round(peak, 2), flat_zone_pairs=int(use.sum()),
                   metric_ratio_tof_over_recon=dict(median=float(np.median(ratio[use])), p10=float(np.percentile(ratio[use], 10)),
                                                    p90=float(np.percentile(ratio[use], 90))) if use.any() else None,
                   heldout=dict(scale_even=scale, n=int(odd.sum()), median_rel=float(np.median(rel)), p90_rel=float(np.percentile(rel, 90)),
                                within_10pct=float(np.mean(rel <= .10))) if odd.any() else None,
                   series=series)
        report['sessions'][d.name] = res
        np.savez_compressed(out / f'{d.name[17:23]}_zones.npz', rec=rec, spread=spread, tof=tof, scale=scale)
        cv2.imwrite(str(out / f'{d.name[17:23]}_first_last.jpg'),
                    np.concatenate([s['imgs'][0], s['imgs'][-1]], 1)[:, :, ::-1])
        print(d.name[17:23], json.dumps({k: res[k] for k in res if k != 'series'}), flush=True)
        del views; torch.cuda.empty_cache()
    (out / 'report.json').write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
