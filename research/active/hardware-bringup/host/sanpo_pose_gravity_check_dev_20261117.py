"""Can SANPO-Real device poses serve as gravity, and does outdoor floor anchoring hold with that gravity and eye-height
error? (SANPO_POSE_GRAVITY_CHECK_DEV_20261117.md). Local 10 sessions x 5 frames, CREStereo pseudo-ground-truth.
Usage: sanpo_pose_gravity_check_dev_20261117.py <out json>
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanpo_scale_without_tof_dev_20261117 as P
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine

W, H = P.W, P.H
CV_FROM_GL = np.diag([1., -1., -1.])
UPS = {'+y': np.array([0, 1., 0]), '-y': np.array([0, -1., 0]), '+z': np.array([0, 0, 1.]), '-z': np.array([0, 0, -1.])}
H_ERR = (-.10, -.05, 0., .05, .10)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def up_in_cam(R, inv, conv, up):
    Rwc = R.T if inv else R                      # maps pose-camera axes to world
    u = Rwc.T @ UPS[up]                          # world up in pose-camera axes
    return (CV_FROM_GL @ u) if conv == 'gl' else u


def main(out_json):
    import torch
    sys.path.insert(0, str(P.DAV2_SRC))
    from depth_anything_v2.dpt import DepthAnythingV2
    rel = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    rel.load_state_dict(torch.load(P.A / 'models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
    rel = rel.cuda().eval()
    S = P.V1.S; rays = P.V1.X.zone_rays(); I3 = np.eye(3)
    rng = np.random.default_rng(20261117)
    frames = []
    for sess in sorted(os.listdir(P.DATA)):
        d = P.DATA / sess; cams = [c for c in os.listdir(d) if c.startswith('camera')]
        if not cams:
            continue
        cam = cams[0]; base = d / cam / 'left'
        desc = json.loads((d / 'description.json').read_text())['session_camera_details'][0]['left_camera_params']
        sc = W / desc['image_width']
        K = np.array([[desc['fx'] * sc, 0, desc['cx'] * sc], [0, desc['fy'] * sc, desc['cy'] * sc], [0, 0, 1.]])
        poses = list(csv.DictReader(open(d / cam / 'fixed_camera_poses.csv')))
        vv, uu = np.mgrid[0:H, 0:W]
        dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3)
        for fr in sorted(p.name.split('.')[0] for p in (base / 'depth_maps').glob('*.float16.gz')):
            k = int(fr); r = poses[k] if k < len(poses) else None
            g = P.read_depth(base / 'depth_maps' / f'{fr}.float16.gz').reshape(-1)
            seg = cv2.resize(cv2.imread(str(base / 'segmentation_masks' / f'{fr}.png'), cv2.IMREAD_UNCHANGED)[:, :, 2], (W, H),
                             interpolation=cv2.INTER_NEAREST).reshape(-1)
            valid = np.isfinite(g) & (g > .3) & (g < 10) & (seg != P.SKY)
            pts = dirs * np.where(valid, g, 0)[:, None]
            plane = P.fit_ground(pts[valid & np.isin(seg, P.WALK)], rng)
            R = None
            if r is not None and r['tracking_state'].endswith('READY') and not r['tracking_state'].endswith('NOT_READY'):
                R = Rotation.from_quat([float(r[c]) for c in ('q_x', 'q_y', 'q_z', 'q_w')]).as_matrix()
            frames.append(dict(sess=sess, cam=cam, fr=fr, K=K, dirs=dirs, g=g, seg=seg, valid=valid, pts=pts, plane=plane, R=R, base=base))
    # 1. convention check
    combos = list(itertools.product((False, True), ('cv', 'gl'), UPS))
    usable = [f for f in frames if f['plane'] is not None and f['R'] is not None]
    errs = {c: [angle(up_in_cam(f['R'], *c), f['plane'][0]) for f in usable] for c in combos}
    best = min(combos, key=lambda c: np.median(errs[c]))
    e = np.array(errs[best])
    conv = dict(frames_usable=len(usable), frames_total=len(frames), best=dict(inv=best[0], cam_axes=best[1], world_up=best[2]),
                median_deg=float(np.median(e)), p95_deg=float(np.percentile(e, 95)), max_deg=float(e.max()),
                runner_up_median_deg=sorted(float(np.median(v)) for v in errs.values())[1],
                per_session_median_deg={s: float(np.median([x for x, f in zip(e, usable) if f['sess'] == s])) for s in sorted({f['sess'] for f in usable})})
    conv['pass'] = conv['median_deg'] <= 3 and conv['p95_deg'] <= 6
    print('统一约定', conv['best'], '夹角中位 %.2f° 95分位 %.2f° 最大 %.2f°（次优约定中位 %.2f°）可用帧 %d/%d 判据1通过=%s' % (
        conv['median_deg'], conv['p95_deg'], conv['max_deg'], conv['runner_up_median_deg'], len(usable), len(frames), conv['pass']), flush=True)
    # 2. floor anchoring with device gravity and eye-height error
    hs = {}
    for f in frames:
        if f['plane'] is not None:
            hs.setdefault(f['sess'], []).append(f['plane'][1])
    h_sess = {s: float(np.median(v)) for s, v in hs.items()}
    arms = ['A', 'B'] + [f'Bzed{dh:+.2f}' for dh in H_ERR]
    rows, last_tof = [], {}
    for f in frames:
        g, valid, dirs, K = f['g'], f['valid'], f['dirs'], f['K']
        bgr = cv2.resize(cv2.imread(str(f['base'] / 'video_frames' / f"{f['fr']}.png")), (W, H), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            disp = rel.infer_image(bgr, 518).astype(np.float32).reshape(-1)
        zgt = g.reshape(H, W); rc = rays
        zu = np.round(K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]).astype(int); zv = np.round(K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]).astype(int)
        zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < W) & (zv >= 0) & (zv < H); zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
        dist = np.full(len(rays), np.nan)
        for zi in range(len(rays)):
            s = zin[zi]
            if s.sum() < 13:
                continue
            zz = zgt[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz) & (zz > .1)
            if ok.sum() >= 13:
                rr = float(np.median(zz[ok] / zcos[zi][s][ok]))
                if .3 < rr < 4.:
                    dist[zi] = rr
        dz, zfac = S.zone_disparity(disp.reshape(H, W), K, I3, rays)
        inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
        fitA = S.fit_affine(dz, inv)
        if not fit_quality(dz, inv, fitA)['accepted']:
            rb = robust_affine(dz, inv)['fit']
            if rb is not None:
                fitA = rb
        fitA = fitA if fitA is not None else last_tof.get(f['sess'])
        if fitA is not None:
            last_tof[f['sess']] = fitA
        Z = {'A': P.to_depth(disp, fitA)}
        h0 = h_sess.get(f['sess'])
        Z['B'] = P.to_depth(disp, P.floor_anchor(disp, -(dirs @ f['plane'][0]), h0, rng)) if (f['plane'] is not None and h0) else None
        up_zed = up_in_cam(f['R'], *best) if f['R'] is not None else None
        for dh in H_ERR:
            Z[f'Bzed{dh:+.2f}'] = (P.to_depth(disp, P.floor_anchor(disp, -(dirs @ (up_zed / np.linalg.norm(up_zed))), h0 + dh, rng))
                                   if (up_zed is not None and h0) else None)
        near = valid & (g >= .5) & (g <= 3.); near_ng = near & ~np.isin(f['seg'], P.WALK)
        row = dict(session=f['sess'], frame=f['fr'], fit={a: Z[a] is not None for a in arms})
        for a in arms:
            row[a] = dict(near=P.metrics(Z[a], g, near), near_nonground=P.metrics(Z[a], g, near_ng) if near_ng.sum() >= 200 else None)
        rows.append(row)
    summ = {}
    for region in ('near', 'near_nonground'):
        for a in arms:
            per = []
            for s in sorted({r['session'] for r in rows}):
                v = [r[a][region]['absrel'] for r in rows if r['session'] == s and r[a][region] and r[a][region]['absrel'] is not None]
                d1 = [r[a][region]['delta1'] for r in rows if r['session'] == s and r[a][region]]
                if v:
                    per.append((s, float(np.median(v)), float(np.median(d1)), len(v)))
            summ[f'{region}|{a}'] = dict(sessions=len(per), frames=sum(p[3] for p in per),
                                          absrel=float(np.median([p[1] for p in per])) if per else None,
                                          delta1=float(np.median([p[2] for p in per])) if per else None)
    fit_rate = {a: [sum(r['fit'][a] for r in rows), len(rows)] for a in arms}
    b, bz = summ['near_nonground|B']['absrel'], summ['near_nonground|Bzed+0.00']['absrel']
    crit2 = dict(ratio=bz / b if (b and bz) else None, fit_rate=fit_rate['Bzed+0.00'])
    crit2['pass'] = bool(crit2['ratio'] is not None and crit2['ratio'] <= 1.25 and fit_rate['Bzed+0.00'][0] >= .9 * fit_rate['Bzed+0.00'][1])
    res = dict(convention=conv, session_height=h_sess, fit_rate=fit_rate, summary=summ, criterion2=crit2, rows=rows)
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf8')
    print('锚定成功率', fit_rate)
    for k, v in summ.items():
        print(f"{k:28s} 段 {v['sessions']:2d} 帧 {v['frames']:3d}  AbsRel {v['absrel'] if v['absrel'] is None else round(v['absrel'], 3)}  δ1 {v['delta1'] if v['delta1'] is None else round(v['delta1'], 3)}")
    print('判据2：Bzed/B（近场非地面）=', None if crit2['ratio'] is None else round(crit2['ratio'], 3), '成功率', crit2['fit_rate'], '通过=', crit2['pass'])


if __name__ == '__main__':
    main(sys.argv[1])
