"""Outdoor scale without ToF on local SANPO-Real (SANPO_SCALE_WITHOUT_TOF_DEV_20261117.md). EXPLORE screening.

Arms per frame (1280x720, CREStereo depth as pseudo-ground-truth):
  A  DAV2-S disparity + emulated coaxial 45 deg 8x8 ToF zones (GT zone ranges), affine -> robust -> last
  B  DAV2-S disparity + floor anchor (GT-fitted ground normal as perfect IMU, per-session median camera height)
  C  DAV2-Metric-Hypersim-Small raw metric depth
  D  UniDepth-V2-S raw metric depth with the known K
  O  DAV2-S per-frame least-squares affine to GT (shape ceiling)
Metrics: AbsRel median and delta1 on corridor obstacle pixels (main) and near-field pixels (0.5-3 m).
Usage: sanpo_scale_without_tof_dev_20261117.py <out json>
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_fusion_eval_v1 as V1
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine

ROOT = V1.ROOT; A = ROOT / 'artifacts.local'
DATA = A / 'datasets/sanpo-real-coverage-20260927'
DAV2_SRC = A / 'work/corridor-depth-e1-20260917/upstream'
UD_SRC = next(p for p in (A / 'downloads/unidepth-src-8d8cfe4c-20261117').glob('UniDepth-*') if p.is_dir())
W, H = 1280, 720
WALK, SKY = [1, 2, 3, 5, 6, 17, 30], 27
X_HALF, Z_NEAR, Z_FAR, HGT = .30, .30, 2.0, (.15, 1.90)


def load_models():
    import torch
    sys.path.insert(0, str(DAV2_SRC))
    from depth_anything_v2.dpt import DepthAnythingV2
    rel = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    rel.load_state_dict(torch.load(A / 'models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
    for k in [k for k in sys.modules if k.startswith('depth_anything_v2')]:
        del sys.modules[k]
    sys.path.remove(str(DAV2_SRC)); sys.path.insert(0, str(DAV2_SRC / 'metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2 as MetricDAV2
    met = MetricDAV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
    met.load_state_dict(torch.load(A / 'models/depth-anything--Depth-Anything-V2-Metric-Hypersim-Small/depth_anything_v2_metric_hypersim_vits.pth',
                                   map_location='cpu', weights_only=True))
    sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))
    from rgb_depth_backbone_adapters import load_model
    ud, ud_infer, _ = load_model('unidepth', UD_SRC, A / 'models/lpiccinelli--unidepth-v2-vits14')
    return rel.cuda().eval(), met.cuda().eval(), ud.cuda(), ud_infer


def read_depth(path):
    a = np.frombuffer(gzip.decompress(Path(path).read_bytes()), '<f2')
    h, w = int(a[0]), int(a[1])
    return a[2:].astype(np.float32).reshape(h, w)


def fit_ground(pts, rng):
    """RANSAC plane on walkable GT points; returns (unit normal pointing from ground to camera side, camera height)."""
    if len(pts) < 2000:
        return None
    sel = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]
    best = None
    for _ in range(200):
        p = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn; d = -n @ p[0]
        inl = np.abs(sel @ n + d) < .03
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), n, d)
    if best is None or best[0] < .5 * len(sel):
        return None
    _, n, d = best
    if d < 0:                       # camera (origin) on the positive side: n . 0 + d = height > 0
        n, d = -n, -d
    return n, float(d)


def floor_anchor(disp, r_down, height, rng):
    """Project rule: disparity = a * inv_z + b on pixels whose ray meets the floor 0.8-6 m ahead (RANSAC)."""
    inv_z = r_down / height
    cand = np.flatnonzero((r_down > .1) & (inv_z > 1 / 6.) & (inv_z < 1 / .8) & np.isfinite(disp))
    if len(cand) < 500:
        return None
    sel = rng.choice(cand, min(4000, len(cand)), replace=False)
    x, y = inv_z[sel], disp[sel]
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
    a, b = np.polyfit(x[best[1]], y[best[1]], 1)
    return (float(a), float(b)) if a > 0 else None


def to_depth(disp, fit):
    if fit is None:
        return None
    iz = (disp - fit[1]) / fit[0]
    return np.where(iz > 1e-3, 1 / np.maximum(iz, 1e-3), np.inf).astype(np.float32)


def metrics(pred, gt, mask):
    if pred is None or mask.sum() == 0:
        return None
    p, g = pred[mask], gt[mask]
    ok = np.isfinite(p) & (p > 0)
    if ok.mean() < .5:
        return dict(absrel=None, delta1=float(0), valid=float(ok.mean()))
    r = np.where(ok, p / g, np.inf)
    return dict(absrel=float(np.median(np.abs(np.where(ok, p, 1e9) - g) / g)), delta1=float(np.mean(np.maximum(r, 1 / r) < 1.25)),
                valid=float(ok.mean()))


def main(out_json):
    import torch
    from PIL import Image
    rel, met, ud, ud_infer = load_models()
    S = V1.S; rays = V1.X.zone_rays(); I3 = np.eye(3)
    rng = np.random.default_rng(20261117)
    rows = []
    for sess in sorted(os.listdir(DATA)):
        d = DATA / sess
        cams = [c for c in os.listdir(d) if c.startswith('camera')]
        if not cams:
            continue
        cam = cams[0]; base = d / cam / 'left'
        desc = json.loads((d / 'description.json').read_text())['session_camera_details'][0]['left_camera_params']
        sc = W / desc['image_width']
        K = np.array([[desc['fx'] * sc, 0, desc['cx'] * sc], [0, desc['fy'] * sc, desc['cy'] * sc], [0, 0, 1.]])
        vv, uu = np.mgrid[0:H, 0:W]
        dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3)
        frames = sorted(p.name.split('.')[0] for p in (base / 'depth_maps').glob('*.float16.gz'))
        F = []
        for fr in frames:
            gt = read_depth(base / 'depth_maps' / f'{fr}.float16.gz')
            assert gt.shape == (H, W), gt.shape
            seg = cv2.resize(cv2.imread(str(base / 'segmentation_masks' / f'{fr}.png'), cv2.IMREAD_UNCHANGED)[:, :, 2], (W, H),
                             interpolation=cv2.INTER_NEAREST).reshape(-1)
            g = gt.reshape(-1); valid = np.isfinite(g) & (g > .3) & (g < 10) & (seg != SKY)
            pts = dirs * np.where(valid, g, 0)[:, None]
            plane = fit_ground(pts[valid & np.isin(seg, WALK)], rng)
            F.append(dict(fr=fr, gt=g, seg=seg, valid=valid, pts=pts, plane=plane))
        hs = [f['plane'][1] for f in F if f['plane'] is not None]
        h_sess = float(np.median(hs)) if hs else None
        last_tof = None
        for f in F:
            g, valid, pts = f['gt'], f['valid'], f['pts']
            bgr = cv2.resize(cv2.imread(str(base / 'video_frames' / f"{f['fr']}.png")), (W, H), interpolation=cv2.INTER_AREA)
            with torch.inference_mode():
                disp = rel.infer_image(bgr, 518).astype(np.float32).reshape(-1)
                zC = met.infer_image(bgr, 518).astype(np.float32).reshape(-1)
            zD = ud_infer(Image.fromarray(bgr[:, :, ::-1]), K).reshape(-1)
            # A: emulated coaxial ToF zones from GT
            zgt = g.reshape(H, W); rc = rays; zu = np.round(K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]).astype(int)
            zv = np.round(K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]).astype(int)
            zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < W) & (zv >= 0) & (zv < H); zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
            dist = np.full(len(rays), np.nan)
            for zi in range(len(rays)):
                s = zin[zi]
                if s.sum() < 13:
                    continue
                zz = zgt[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz) & (zz > .1)
                if ok.sum() >= 13:
                    r = float(np.median(zz[ok] / zcos[zi][s][ok]))
                    if .3 < r < 4.:
                        dist[zi] = r
            dz, zfac = S.zone_disparity(disp.reshape(H, W), K, I3, rays)
            inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
            fitA = S.fit_affine(dz, inv)
            if not fit_quality(dz, inv, fitA)['accepted']:
                rb = robust_affine(dz, inv)['fit']
                if rb is not None:          # as the project chain: otherwise keep the first fit
                    fitA = rb
            fitA = fitA if fitA is not None else last_tof
            last_tof = fitA if fitA is not None else last_tof
            zA = to_depth(disp, fitA)
            # B: floor anchor with GT normal (perfect IMU) and session-median height
            fitB = None
            if f['plane'] is not None and h_sess is not None:
                r_down = -(dirs @ f['plane'][0])            # component of each ray toward the ground
                fitB = floor_anchor(disp, r_down, h_sess, rng)
            zB = to_depth(disp, fitB)
            # O: oracle affine
            m = valid & np.isfinite(disp)
            a_, b_ = np.polyfit(1 / g[m], disp[m], 1)
            zO = to_depth(disp, (a_, b_) if a_ > 0 else None)
            # regions
            near = valid & (g >= .5) & (g <= 3.)
            near_ng = near & ~np.isin(f['seg'], WALK)   # added after the run (descriptive): near field without walkable ground
            corr = np.zeros_like(valid)
            if f['plane'] is not None:
                n, h = f['plane']; down = -n
                fwd = np.array([0, 0, 1.]) - down * down[2]; fwd /= np.linalg.norm(fwd); right = np.cross(down, fwd)
                hgt = h - pts @ down; z = pts @ fwd; x = pts @ right
                corr = valid & (np.abs(x) <= X_HALF) & (z >= Z_NEAR) & (z <= Z_FAR) & (hgt >= HGT[0]) & (hgt <= HGT[1])
            row = dict(session=sess, cam=cam, frame=f['fr'], ground_plane=f['plane'] is not None, floor_fit=fitB is not None,
                       tof_fit=fitA is not None, corridor_px=int(corr.sum()), near_px=int(near.sum()))
            for arm, z in (('A', zA), ('B', zB), ('C', zC), ('D', zD), ('O', zO)):
                row[arm] = dict(corridor=metrics(z, g, corr) if corr.sum() >= 200 else None, near=metrics(z, g, near),
                                near_nonground=metrics(z, g, near_ng) if near_ng.sum() >= 200 else None)
            rows.append(row)
        print(sess, cam, len(F), 'h_sess', None if h_sess is None else round(h_sess, 3), flush=True)
    summ = {}
    sessions = sorted({r['session'] for r in rows})
    for region in ('corridor', 'near', 'near_nonground'):
        for arm in 'ABCDO':
            per = []
            for s in sessions:
                v = [r[arm][region]['absrel'] for r in rows if r['session'] == s and r[arm][region] and r[arm][region]['absrel'] is not None]
                d1 = [r[arm][region]['delta1'] for r in rows if r['session'] == s and r[arm][region]]
                if v:
                    per.append((s, float(np.median(v)), float(np.median(d1)), len(v)))
            summ[f'{region}|{arm}'] = dict(sessions=len(per), frames=sum(p[3] for p in per),
                                            absrel_session_median=float(np.median([p[1] for p in per])) if per else None,
                                            delta1_session_median=float(np.median([p[2] for p in per])) if per else None,
                                            per_session=per)
    res = dict(frames=len(rows), floor_fit_rate=[sum(r['floor_fit'] for r in rows), len(rows)],
               ground_plane_rate=[sum(r['ground_plane'] for r in rows), len(rows)], tof_fit_rate=[sum(r['tof_fit'] for r in rows), len(rows)],
               corridor_frames=sum(r['corridor_px'] >= 200 for r in rows), summary=summ, rows=rows)
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf8')
    print('帧', len(rows), '| 地面平面', res['ground_plane_rate'], '| 地面锚定成功', res['floor_fit_rate'], '| ToF 拟合', res['tof_fit_rate'],
          '| 有效走廊帧', res['corridor_frames'])
    for k, v in summ.items():
        print(f"{k:12s} 段数 {v['sessions']:2d} 帧 {v['frames']:3d}  AbsRel(段中位) {v['absrel_session_median'] if v['absrel_session_median'] is None else round(v['absrel_session_median'], 3)}"
              f"  δ1 {v['delta1_session_median'] if v['delta1_session_median'] is None else round(v['delta1_session_median'], 3)}")


if __name__ == '__main__':
    main(sys.argv[1])
