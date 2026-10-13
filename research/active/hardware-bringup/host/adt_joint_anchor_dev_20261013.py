"""Joint ToF + floor anchor vs the v6 chain on prepared ADT frames (plan: adt-eval-20261013/PLAN_JOINT_ANCHOR.md).

Motivation (bb_analysis_dav2s): oracle global affine lifts recall 81 -> 93 % and FA at matched recall is lower, while
92 % of false-alarm blob pixels lie outside the +-22 deg ToF field (the chain extrapolates a centre-only fit).
Arms per frame (DAV2-S, same truth/ToF emulation as adt_eval_dev_20261013):
  chain  v6 chain (fit_affine -> robust -> floor -> last)
  joint  one affine fit on ToF zone pairs + floor RANSAC inliers, each source carrying half the total weight,
         three trimming passes (2.5 robust sigma); ToF-only/floor-only when the other source is missing; then last fit
  floor  floor RANSAC fit only (then last fit)
  oracle affine to GT (diagnostic)
Blobs are stored like adt_backbone_blobs_dev_20261013 (no grids). Usage: <prep dir> <out dir>
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
from adt_backbone_blobs_dev_20261013 import blobs, box_masks


def floor_pairs(disp, dirs, Rpose, Rct, height, rng):
    """S.floor_anchor RANSAC, returning the inlier pairs (inv_z, disp) or None."""
    r = (Rpose @ (Rct.T @ dirs.T)).T
    inv_z = r[:, 1] / height
    cand = np.flatnonzero((r[:, 1] > .1) & (inv_z > 1 / 6.) & (inv_z < 1 / .8))
    if len(cand) < 500:
        return None
    sel = rng.choice(cand, min(4000, len(cand)), replace=False)
    x, y = inv_z[sel], disp.reshape(-1)[sel]; best = None
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
    return x[best[1]], y[best[1]]


def wfit(x, y, wt):
    for _ in range(3):
        A = np.stack([x, np.ones_like(x)], 1) * np.sqrt(wt)[:, None]
        (a, b), *_ = np.linalg.lstsq(A, y * np.sqrt(wt), rcond=None)
        r = y - (a * x + b); s = np.median(np.abs(r)) * 1.4826 + 1e-9; keep = np.abs(r) < 2.5 * s
        if keep.sum() < 6 or keep.all():
            break
        x, y, wt = x[keep], y[keep], wt[keep]
    return (float(a), float(b)) if a > 0 else None


def main(prep, out):
    S, X = V1.S, V1.X
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model, _ = S.load_dav2_small()
    import torch
    import torch.nn.functional as Fn
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays = X.zone_rays()
    pool = ThreadPoolExecutor(6); start = time.perf_counter()
    for seq in sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists()):
        if (out / f'{seq.name}.json').exists():
            continue
        meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
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
        rng_c, rng_j = np.random.default_rng(20261013), np.random.default_rng(20261013)
        last = dict(chain=None, joint=None, floor=None)

        def readout(i, disp, fits):
            zgt = D['depth'][i].reshape(-1) / 1000.; gok = zgt > .1
            gpts = np.full((h * w, 3), np.nan); gpts[gok] = (R_cc[i] @ (dirs[gok] * zgt[gok, None]).T).T
            truth = [dict(n=int(m.sum()), z50=float(np.median(gpts[m, 2])) if m.any() else None) for m in box_masks(gpts, gok)]
            m = gok & (zgt > .3) & (zgt < 10)
            a, b = np.polyfit(1 / zgt[m], disp.reshape(-1)[m], 1) if m.sum() > 1000 else (np.nan, np.nan)
            fits = dict(fits, oracle=(a, b) if a > 0 else None); arms = {}
            for arm, f in fits.items():
                if f is None:
                    arms[arm] = None; continue
                invd = (disp.reshape(-1) - f[1]) / f[0]
                zz = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf); ok = np.isfinite(zz)
                pts = np.full((h * w, 3), np.nan); pts[ok] = (R_cc[i] @ (dirs[ok] * zz[ok, None]).T).T
                arms[arm] = [blobs(mk, pts, h, w) for mk in box_masks(pts, ok)]
            return dict(truth=truth, arms=arms)

        futs = []
        for i in range(N):
            t, _ = model.image2tensor(np.ascontiguousarray(D['rgb'][i][:, :, ::-1]), V1.SIZE)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            zgt = D['depth'][i] / 1000.; dist = np.full(64, np.nan)
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
            Rp = R_cc[i] @ Rct
            # chain (identical to v6)
            fit = S.fit_affine(dz, inv)
            if not fit_quality(dz, inv, fit)['accepted']:
                rb = robust_affine(dz, inv)['fit']
                if rb is not None:
                    fit = rb
            if fit is None:
                fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng_c)
            fits = dict(chain=fit or last['chain'])
            # joint and floor-only (own RNG stream so the chain stays identical to v6)
            fp = floor_pairs(disp, dirs, Rp, Rct, height, rng_j)
            tm = np.isfinite(dz) & np.isfinite(inv); fj = None
            if fp is not None and tm.sum() >= 6:
                x = np.concatenate([inv[tm], fp[0]]); y = np.concatenate([dz[tm], fp[1]])
                wt = np.concatenate([np.full(tm.sum(), .5 / tm.sum()), np.full(len(fp[0]), .5 / len(fp[0]))])
                fj = wfit(x, y, wt)
            elif tm.sum() >= 6:
                fj = S.fit_affine(dz, inv)
            elif fp is not None:
                fj = wfit(fp[0], fp[1], np.ones(len(fp[0])))
            ff = wfit(fp[0], fp[1], np.ones(len(fp[0]))) if fp is not None else None
            fits.update(joint=fj or last['joint'], floor=ff or last['floor'])
            for k in last:
                if fits[k] is not None:
                    last[k] = fits[k]
            futs.append(pool.submit(readout, i, disp, fits))
        rows = [f.result() for f in futs]
        (out / f'{seq.name}.json').write_text(json.dumps(dict(height=height, frames=rows)), encoding='utf8')
        print(seq.name, N, f'{time.perf_counter()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
