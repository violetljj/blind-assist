"""Geometry features for a learned occupancy readout head on prepared ADT frames (plan: adt-eval-20261013/PLAN_HEAD.md).

Per frame and per variant (a: emulated-ToF anchor chain; b: ToF dropped -> floor anchor -> last fit): predicted points
binned in the gravity frame (4 height bands x 18 lateral x 30 depth bins, log1p counts on every 2nd pixel), scalars
(anchor source one-hot [tof, robust, floor, last, none], valid zones, pitch, height), and the v3/v6 readouts of
variant b (variant a readouts already sit in adt_eval.json). Labels are reused from adt_eval.json by frame index.
Batched fp16 DAV2-S with CPU preprocessing and bounded prefetch; run one process per sequence subset.
Usage: adt_head_features_dev_20261013.py <prep dir> <out dir> <worker> <n_workers>
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
from adt_eval_dev_20261013 import box_masks, readouts

XE, ZE = np.linspace(-.9, .9, 19), np.linspace(.3, 3.3, 31)
YB = [(-.6, -.2), (-.2, .42), (.42, .9), (.9, 1.9)]
SRC = ['tof', 'robust', 'floor', 'last', None]
MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)


def hist(pts):
    out = np.zeros((4, 18, 30), np.float32)
    for k, (lo, hi) in enumerate(YB):
        m = (pts[:, 1] >= lo) & (pts[:, 1] < hi)
        out[k] = np.histogram2d(pts[m, 0], pts[m, 2], bins=(XE, ZE))[0]
    return np.log1p(out)


def main(prep, out, worker, n_workers):
    import cv2
    import torch
    import torch.nn.functional as Fn
    S, X = V1.S, V1.X
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model, _ = S.load_dav2_small()
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays = X.zone_rays()
    seqs = sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists())[worker::n_workers]
    pool = ThreadPoolExecutor(4); start = time.perf_counter()
    for seq in seqs:
        if (out / f'{seq.name}.npz').exists():
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
        R_cc, cam_y, N = D['R_cc'], D['cam_y'], len(D['cam_y'])
        hts = []
        for i in range(0, N, max(1, N // 20)):
            z = D['depth'][i].reshape(-1) / 1000.; ok = z > .1
            hts.append(cam_y[i] - ((dirs[ok] * z[ok, None]) @ R_cc[i].T)[:, 1])
        height = float(np.median(cam_y) - np.percentile(np.concatenate(hts), 1))
        nh, nw = int(round(h * 518 / min(h, w) / 14)) * 14, int(round(w * 518 / min(h, w) / 14)) * 14
        def prep_batch(idx):
            ims = [((cv2.resize(D['rgb'][i].astype(np.float32) / 255, (nw, nh), interpolation=cv2.INTER_CUBIC) - MEAN) / STD).transpose(2, 0, 1) for i in idx]
            return np.ascontiguousarray(np.stack(ims))
        B = 8; batches = [list(range(k, min(k + B, N))) for k in range(0, N, B)]
        pending = [pool.submit(prep_batch, b) for b in batches[:3]]
        feats = {v: dict(grid=[], scal=[]) for v in 'ab'}; ro_b = []; last = {'a': None, 'b': None}
        rng = {'a': np.random.default_rng(20261013), 'b': np.random.default_rng(20261013)}
        for bi, idx in enumerate(batches):
            x = torch.from_numpy(pending.pop(0).result()).cuda()
            if bi + 3 < len(batches):
                pending.append(pool.submit(prep_batch, batches[bi + 3]))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(x)
            disps = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[:, 0].cpu().numpy()
            for i, disp in zip(idx, disps):
                zgt = D['depth'][i] / 1000.
                dist = np.full(64, np.nan)
                for zi in range(64):
                    s = zin[zi]
                    if s.sum() >= 13:
                        zz = zgt[zv[zi][s], zu[zi][s]]; g = zz > .1
                        if g.sum() >= 13:
                            r = float(np.median(zz[g] / zcos[zi][s][g]))
                            if .3 < r < 4.:
                                dist[zi] = r
                dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                Rp = R_cc[i] @ Rct
                pitch = float(np.degrees(np.arcsin(np.clip(R_cc[i][1, 2], -1, 1))))
                for v in 'ab':
                    fit, src = None, None
                    if v == 'a':
                        inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
                        fit = S.fit_affine(dz, inv); src = 'tof' if fit else None
                        if not fit_quality(dz, inv, fit)['accepted']:
                            rb = robust_affine(dz, inv)['fit']
                            if rb is not None:
                                fit, src = rb, 'robust'
                    if fit is None:
                        fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng[v]); src = 'floor' if fit else None
                    if fit is None and last[v] is not None:
                        fit, src = last[v], 'last'
                    if fit is not None:
                        last[v] = fit
                        invd = (disp.reshape(-1) - fit[1]) / fit[0]
                        zz = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf); ok = np.isfinite(zz)
                        pts = np.full((h * w, 3), np.nan); pts[ok] = (Rp @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                        grid = hist(pts[ok & sub])
                        if v == 'b':
                            ro_b.append([list(readouts(mk, pts[:, 2], h, w)) for mk in box_masks(pts, ok)])
                    else:
                        grid = np.zeros((4, 18, 30), np.float32)
                        if v == 'b':
                            ro_b.append([[False, False], [False, False]])
                    onehot = [float(src == s_) for s_ in SRC]
                    feats[v]['grid'].append(grid.astype(np.float16))
                    feats[v]['scal'].append(onehot + [float(np.isfinite(dist).sum()) / 64 if v == 'a' else 0., pitch / 30, height])
        np.savez_compressed(out / f'{seq.name}.npz', grid_a=np.array(feats['a']['grid']), scal_a=np.array(feats['a']['scal'], np.float32),
                            grid_b=np.array(feats['b']['grid']), scal_b=np.array(feats['b']['scal'], np.float32), readout_b=np.array(ro_b, bool))
        print(worker, seq.name, N, f'{time.perf_counter()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
