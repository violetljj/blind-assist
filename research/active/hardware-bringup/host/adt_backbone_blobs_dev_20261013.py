"""Depth-backbone ceiling + per-blob dump on prepared ADT frames (plan: adt-eval-20261013/PLAN_BACKBONE_MOTION.md).

Same pipeline as adt_eval_dev_20261013 (GT truth per HEAD/BODY query, ToF zones emulated from GT, anchor chain
fit_affine -> robust -> floor anchor -> last fit, oracle affine to GT), with the depth backbone swapped:
  dav2s         DAV2-Small relative disparity (the v6 baseline)
  dav2l_metric  DAV2-Large Metric-Hypersim (inverse depth used as the disparity for the chain; raw metric arm too)
  unidepth_l    UniDepth-V2 ViT-L with the known pinhole K (inverse depth for the chain; raw metric arm)
  moge2_l       MoGe-2 ViT-L metric (inverse depth for the chain; raw metric arm)
Instead of booleans, every 8-connected in-box blob (>= 10 px) is stored as [npx, z50, cx, cy, cz] (corridor frame) plus
the total in-box pixel count, so readout rules (v3/v6 and motion-consistency candidates) run offline. Per frame and arm a
corridor-frame occupancy grid of predicted points (HEAD/BODY bands x 30 lateral x 50 depth bins of 0.1 m, every 2nd
pixel) is stored for world-consistency look-ups; the same grid from GT points is stored once ('gt').
Env TOF_TILT=<deg>: zone rays pitched further down (as adt_tof_tilt_sim_dev_20261013), for the anchor.
Usage: adt_backbone_blobs_dev_20261013.py <prep dir> <out dir> <backbone> [sequence substring]
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine

ROOT = V1.ROOT
BOXES = list(V1.F.BOXES.values())
GX, GZ = np.linspace(-1.5, 1.5, 31), np.linspace(.3, 5.3, 51)
UP = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream'
MODELS = ROOT / 'artifacts.local/models'


def load_backbone(name):
    """Return infer(rgb uint8 HxWx3 RGB, K) -> (disparity-like map, metric z map or None), both HxW float32."""
    import torch
    import torch.nn.functional as Fn
    if name in ('dav2s', 'dav2l_metric'):
        if name == 'dav2s':
            model, _ = V1.S.load_dav2_small()
        else:
            sys.path.insert(0, str(UP / 'Depth-Anything-V2/metric_depth'))
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], max_depth=20)
            model.load_state_dict(torch.load(MODELS / 'depth-anything--Depth-Anything-V2-Metric-Hypersim-Large/depth_anything_v2_metric_hypersim_vitl.pth',
                                             map_location='cpu', weights_only=True), strict=True)
            model = model.cuda().eval()
        def infer(rgb, K):
            t, _ = model.image2tensor(np.ascontiguousarray(rgb[:, :, ::-1]), V1.SIZE)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            out = Fn.interpolate(dd[:, None].float(), rgb.shape[:2], mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            if name == 'dav2s':
                return out, None
            return np.where(out > .05, 1 / np.maximum(out, .05), 0.).astype(np.float32), out
        return infer
    sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))
    from rgb_depth_backbone_adapters import load_model
    from PIL import Image
    if name == 'unidepth_l':
        model, inf, _ = load_model('unidepth', UP / 'UniDepth', MODELS / 'lpiccinelli--unidepth-v2-vitl14')
    else:
        model, inf, _ = load_model('moge2', UP / 'MoGe', MODELS / 'Ruicheng--moge-2-vitl-normal')
    model.cuda()
    def infer(rgb, K):
        z = inf(Image.fromarray(rgb), K)
        ok = np.isfinite(z) & (z > .05)
        return np.where(ok, 1 / np.where(ok, z, 1.), 0.).astype(np.float32), np.where(ok, z, np.inf).astype(np.float32)
    return infer


def blobs(mask, pts, h, w):
    import cv2
    total = int(mask.sum()); out = []
    if total:
        n, cc = cv2.connectedComponents(mask.reshape(h, w).astype(np.uint8), connectivity=8)
        cc = cc.reshape(-1)
        idx = np.nonzero(mask)[0]; lab = cc[idx]
        order = np.argsort(lab, kind='stable'); idx, lab = idx[order], lab[order]
        cuts = np.flatnonzero(np.diff(lab)) + 1
        for g in np.split(idx, cuts):
            if len(g) >= 10:
                p = pts[g]
                out.append([len(g), round(float(np.median(p[:, 2])), 3)] + [round(float(v), 3) for v in p.mean(0)])
    return total, out


def grid(pts, ok, sub):
    m = ok & sub & np.isfinite(pts[:, 2])
    g = np.zeros((2, 30, 50), np.uint16)
    for k, (lo, hi) in enumerate(BOXES):
        s = m & (pts[:, 1] >= lo) & (pts[:, 1] <= hi)
        g[k] = np.minimum(np.histogram2d(pts[s, 0], pts[s, 2], bins=(GX, GZ))[0], 65535)
    return g


def box_masks(pts, ok):
    return [ok & (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= V1.Z_MAX)
            for lo, hi in BOXES]


def main(prep, out, name, only=''):
    S, X = V1.S, V1.X
    out = Path(out) / name; out.mkdir(parents=True, exist_ok=True)
    infer = load_backbone(name)
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays = X.zone_rays()
    tilt = float(os.environ.get('TOF_TILT', '0'))                  # emulate the ToF re-aimed t deg further down
    if tilt:
        a = np.radians(tilt); rays = rays @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    pool = ThreadPoolExecutor(6); start = time.perf_counter()
    seqs = sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists() and only in p.name)
    for seq in (seqs[::-1] if os.environ.get('REVERSE') == '1' else seqs):          # REVERSE=1: second worker from the end
        if (out / f'{seq.name}.json').exists():
            continue
        meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        sub = np.zeros((h, w), bool); sub[::2, ::2] = True; sub = sub.reshape(-1)
        rc = rays @ Rct.T
        zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
        zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h)
        zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
        with np.load(seq / 'frames.npz') as Z:
            D = {k: Z[k] for k in Z.files}
        R_cc, cam_y = D['R_cc'], D['cam_y']; N = len(cam_y)
        hts = []
        for i in range(0, N, max(1, N // 20)):
            z = D['depth'][i].reshape(-1) / 1000.; ok = z > .1
            hts.append(cam_y[i] - (R_cc[i] @ (dirs[ok] * z[ok, None]).T).T[:, 1])
        height = float(np.median(cam_y) - np.percentile(np.concatenate(hts), 1))
        rng = np.random.default_rng(20261013); state = dict(last=None)

        def fit_chain(i, disp):
            zgt = D['depth'][i] / 1000.
            dist = np.full(64, np.nan)
            for zi in range(64):
                s = zin[zi]
                if s.sum() < 13:
                    continue
                zz = zgt[zv[zi][s], zu[zi][s]]; good = zz > .1
                if good.sum() >= 13:
                    r = float(np.median(zz[good] / zcos[zi][s][good]))
                    if .3 < r < 4.:
                        dist[zi] = r
            dz, zfac = S.zone_disparity(disp, K, Rct, rays)
            inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
            fit = S.fit_affine(dz, inv); src = 'tof' if fit else None
            if not fit_quality(dz, inv, fit)['accepted']:
                rb = robust_affine(dz, inv)['fit']
                if rb is not None:
                    fit, src = rb, 'robust'
            if fit is None:
                fit = S.floor_anchor(disp, dirs, R_cc[i] @ Rct, Rct, height, rng); src = 'floor' if fit else None
            if fit is None and state['last'] is not None:
                fit, src = state['last'], 'last'
            if fit is not None:
                state['last'] = fit
            return fit, src, int(np.isfinite(dist).sum())

        def readout(i, disp, zmetric, fit, src, nz):
            zgt = D['depth'][i].reshape(-1) / 1000.; gok = zgt > .1
            gpts = np.full((h * w, 3), np.nan); gpts[gok] = (R_cc[i] @ (dirs[gok] * zgt[gok, None]).T).T
            truth = []
            for m in box_masks(gpts, gok):
                n = int(m.sum()); truth.append(dict(n=n, z50=float(np.median(gpts[m, 2])) if n else None))
            m = gok & (zgt > .3) & (zgt < 10)
            a, b = np.polyfit(1 / zgt[m], disp.reshape(-1)[m], 1) if m.sum() > 1000 else (np.nan, np.nan)
            arms, grids = {}, {'gt': grid(gpts, gok, sub)}
            zsets = [('chain', fit), ('oracle', (a, b) if a > 0 else None)]
            if zmetric is not None:
                zsets.append(('raw', 'raw'))
            for arm, f in zsets:
                if f is None:
                    arms[arm] = None; grids[arm] = np.zeros((2, 30, 50), np.uint16); continue
                if f == 'raw':
                    zz = zmetric.reshape(-1).astype(np.float64); zz = np.where(np.isfinite(zz) & (zz > .05), zz, np.inf)
                else:
                    invd = (disp.reshape(-1) - f[1]) / f[0]
                    zz = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf)
                ok = np.isfinite(zz)
                pts = np.full((h * w, 3), np.nan); pts[ok] = (R_cc[i] @ (dirs[ok] * zz[ok, None]).T).T
                arms[arm] = [blobs(mk, pts, h, w) for mk in box_masks(pts, ok)]
                grids[arm] = grid(pts, ok, sub)
            return dict(truth=truth, src=src, zones=nz, arms=arms), grids

        futs = []
        for i in range(N):
            disp, zmetric = infer(D['rgb'][i], K)
            fit, src, nz = fit_chain(i, disp)                 # sequential: the chain carries the last fit
            futs.append(pool.submit(readout, i, disp, zmetric, fit, src, nz))
        rows, G = [], {}
        for f in futs:
            r, g = f.result(); rows.append(r)
            for k, v in g.items():
                G.setdefault(k, []).append(v)
        np.savez_compressed(out / f'{seq.name}.npz', **{k: np.array(v) for k, v in G.items()})
        (out / f'{seq.name}.json').write_text(json.dumps(dict(height=height, frames=rows)), encoding='utf8')
        print(name, seq.name, N, f'{time.perf_counter()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
