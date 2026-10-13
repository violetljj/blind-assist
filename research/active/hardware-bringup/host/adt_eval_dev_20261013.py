"""Ground-truth evaluation of the RGB branch (v3 vs v6 readout) on prepared ADT frames (plan: adt-eval-20261013/PLAN.md).

Per frame: fp16 DAV2-S on the upright pinhole RGB; ToF zones emulated from GT depth through our zone rays and camera<-ToF
extrinsic (median GT range per zone, valid 0.3-4 m); v3 anchor chain (fit_affine -> robust -> floor anchor with the GT
camera height -> last fit); oracle scale = affine fit of disparity to GT inverse depth. Readouts v3 (all in-box pixels)
and v6 (blobs with median z <= 1.8 m). Truth per query: GT in-box pixels >= 150 positive, 0 negative, else excluded.
Usage: adt_eval_dev_20261013.py <prep dir> <out json>
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

BOXES = list(V1.F.BOXES.values())
BLOB_Z, MIN_BLOB = 1.8, 10


def box_masks(pts, ok):
    out = []
    for lo, hi in BOXES:
        out.append(ok & (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= V1.Z_MAX))
    return out


def readouts(mask, z, h, w):
    import cv2
    v3 = int(mask.sum()) >= V1.MIN_PX
    if not mask.any():
        return v3, False
    n, cc = cv2.connectedComponents(mask.reshape(h, w).astype(np.uint8), connectivity=8)
    cc = cc.reshape(-1); counted = 0
    for c in range(1, n):
        m = cc == c
        if m.sum() >= MIN_BLOB and np.median(z[m]) <= BLOB_Z:
            counted += int(m.sum())
    return v3, counted >= V1.MIN_PX


def main(prep, out):
    import torch
    import torch.nn.functional as Fn
    S, X = V1.S, V1.X
    model, _ = S.load_dav2_small()
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays = X.zone_rays()
    res = json.loads(Path(out).read_text(encoding='utf8')) if Path(out).exists() else {}   # resume: skip done sequences
    start = time.perf_counter(); pool = ThreadPoolExecutor(2)
    for seq in sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists() and p.name not in res):
        meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        rc = rays @ Rct.T
        zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
        zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h)
        zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
        with np.load(seq / 'frames.npz') as Z:
            D = {k: Z[k] for k in Z.files}                    # decompress once (NpzFile re-reads per access)
        R_cc, cam_y = D['R_cc'], D['cam_y']; N = len(cam_y)
        # floor level: low percentile of GT point heights over a frame subset
        hts = []
        for i in range(0, N, max(1, N // 20)):
            z = D['depth'][i].reshape(-1) / 1000.; ok = z > .1
            p = (R_cc[i] @ (dirs[ok] * z[ok, None]).T).T
            hts.append(cam_y[i] - p[:, 1])
        floor = float(np.percentile(np.concatenate(hts), 1)); height = float(np.median(cam_y) - floor)
        rng = np.random.default_rng(20261013); last = None; rows = []
        def dav(i):
            t, _ = model.image2tensor(np.ascontiguousarray(D['rgb'][i][:, :, ::-1]), V1.SIZE); return t
        tens = pool.map(dav, range(N))
        for i, t in enumerate(tens):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            zgt = D['depth'][i].reshape(-1) / 1000.; gok = zgt > .1
            Rp = R_cc[i] @ Rct
            gpts = np.full((h * w, 3), np.nan); gpts[gok] = (R_cc[i] @ (dirs[gok] * zgt[gok, None]).T).T
            truth = []
            for m in box_masks(gpts, gok):
                n = int(m.sum()); truth.append(dict(n=n, z50=float(np.median(gpts[m, 2])) if n else None))
            # emulated ToF zones
            dist = np.full(64, np.nan)
            for zi in range(64):
                s = zin[zi]
                if s.sum() < 13:
                    continue
                zz = zgt.reshape(h, w)[zv[zi][s], zu[zi][s]]; good = zz > .1
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
                fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng); src = 'floor' if fit else None
            if fit is None and last is not None:
                fit, src = last, 'last'
            if fit is not None:
                last = fit
            m = gok & (zgt > .3) & (zgt < 10)
            a, b = np.polyfit(1 / zgt[m], disp.reshape(-1)[m], 1) if m.sum() > 1000 else (np.nan, np.nan)
            arms = {}
            for name, f in (('chain', fit), ('oracle', (a, b) if a > 0 else None)):
                if f is None:
                    arms[name] = [[False, False], [False, False]]; continue
                invd = (disp.reshape(-1) - f[1]) / f[0]
                zz = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf); ok = np.isfinite(zz)
                pts = np.full((h * w, 3), np.nan); pts[ok] = (Rp @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                arms[name] = [list(readouts(mk, pts[:, 2], h, w)) for mk in box_masks(pts, ok)]   # [query][v3, v6]
            rows.append(dict(truth=truth, src=src, arms=arms, zones=int(np.isfinite(dist).sum())))
        res[seq.name] = dict(height=height, floor=floor, frames=rows)
        print(seq.name, N, 'height %.2f' % height, f'{time.perf_counter()-start:.0f}s', flush=True)
        Path(out).write_text(json.dumps(res), encoding='utf8')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
