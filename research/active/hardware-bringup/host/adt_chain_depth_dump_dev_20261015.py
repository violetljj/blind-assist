"""Dump the device-side metric depth of the frozen v3/v6 anchor chain on the prepared ADT frames (EXPLORE, 2026-10-15).

Same model, emulated ToF, anchor order and extrinsic as adt_eval_dev_20261013.py (fp16 DAV2-S; ToF fit -> robust ->
floor anchor -> last fit), but instead of box readouts it stores the anchored camera z-depth at stride 2 (float16,
inf where invalid) for the cane-reach short-term-memory study. Output: <prep>/<sequence>/chain_depth_s2.npz.
Usage: adt_chain_depth_dump_dev_20261015.py <prep dir> <adt_eval.json>
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

STEP = 2


def main(prep, evaluation):
    import torch
    import torch.nn.functional as Fn
    S, X = V1.S, V1.X
    R = json.loads(Path(evaluation).read_text(encoding='utf8'))
    model, _ = S.load_dav2_small()
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays = X.zone_rays()
    pool = ThreadPoolExecutor(2); start = time.perf_counter()
    for name in R:
        seq = Path(prep) / name
        if (seq / 'chain_depth_s2.npz').exists():
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
            D = {k: Z[k] for k in ('rgb', 'depth', 'R_cc', 'cam_y')}
        R_cc, height = D['R_cc'], R[name]['height']; N = len(R_cc)
        rng = np.random.default_rng(20261013); last = None; out, srcs = [], []
        tens = pool.map(lambda i: model.image2tensor(np.ascontiguousarray(D['rgb'][i][:, :, ::-1]), V1.SIZE)[0], range(N))
        for i, t in enumerate(tens):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            zgt = D['depth'][i].reshape(-1) / 1000.
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
                fit = S.floor_anchor(disp, dirs, R_cc[i] @ Rct, Rct, height, rng); src = 'floor' if fit else None
            if fit is None and last is not None:
                fit, src = last, 'last'
            if fit is not None:
                last = fit
            if fit is None:
                out.append(np.full((h // STEP, w // STEP), np.inf, np.float16)); srcs.append('none'); continue
            invd = (disp[::STEP, ::STEP] - fit[1]) / fit[0]
            out.append(np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf).astype(np.float16)); srcs.append(src)
        np.savez_compressed(seq / 'chain_depth_s2.npz', z=np.stack(out), src=np.array(srcs))
        print(name, N, f'{time.perf_counter() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
