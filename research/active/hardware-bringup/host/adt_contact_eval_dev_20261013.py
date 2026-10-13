"""ADT GT evaluation of the floor-contact veto (plan PLAN_CONTACT.md): same frames/chain as adt_eval_dev, readouts v6 and
v7-candidate (v6 + contact veto) for the emulated-ToF chain and the oracle scale.
Usage: adt_contact_eval_dev_20261013.py <prep dir> <out json>
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
from adt_eval_dev_20261013 import box_masks
from floor_contact_dev_20261013 import contact_readout
BOXES = list(V1.F.BOXES.values())


def main(prep, out):
    import torch, torch.nn.functional as Fn
    S, X = V1.S, V1.X
    model, _ = S.load_dav2_small(); Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays = X.zone_rays()
    res = json.loads(Path(out).read_text()) if Path(out).exists() else {}; pool = ThreadPoolExecutor(2); start = time.perf_counter()
    for seq in sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists() and p.name not in res):
        meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]); vv, uu = np.mgrid[0:h, 0:w]
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        rc = rays @ Rct.T; zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
        zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h); zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
        with np.load(seq / 'frames.npz') as Z:
            D = {k: Z[k] for k in Z.files}
        R_cc, cam_y, N = D['R_cc'], D['cam_y'], len(D['cam_y'])
        hts = []
        for i in range(0, N, max(1, N // 20)):
            z = D['depth'][i].reshape(-1) / 1000.; ok = z > .1; hts.append(cam_y[i] - ((dirs[ok] * z[ok, None]) @ R_cc[i].T)[:, 1])
        height = float(np.median(cam_y) - np.percentile(np.concatenate(hts), 1))
        rng = np.random.default_rng(20261013); last = None; rows = []
        tens = pool.map(lambda i: model.image2tensor(np.ascontiguousarray(D['rgb'][i][:, :, ::-1]), V1.SIZE)[0], range(N))
        for i, t in enumerate(tens):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t)
            disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            zgt = D['depth'][i].reshape(-1) / 1000.; gok = zgt > .1; Rp = R_cc[i] @ Rct
            gpts = np.full((h * w, 3), np.nan); gpts[gok] = (R_cc[i] @ (dirs[gok] * zgt[gok, None]).T).T
            truth = [dict(n=int(m.sum()), z50=float(np.median(gpts[m, 2])) if m.sum() else None) for m in box_masks(gpts, gok)]
            dist = np.full(64, np.nan)
            for zi in range(64):
                s = zin[zi]
                if s.sum() >= 13:
                    zz = zgt.reshape(h, w)[zv[zi][s], zu[zi][s]]; g = zz > .1
                    if g.sum() >= 13:
                        r = float(np.median(zz[g] / zcos[zi][s][g]))
                        if .3 < r < 4.: dist[zi] = r
            dz, zfac = S.zone_disparity(disp, K, Rct, rays); inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
            fit = S.fit_affine(dz, inv)
            if not fit_quality(dz, inv, fit)['accepted']:
                rb = robust_affine(dz, inv)['fit']; fit = rb if rb is not None else fit
            if fit is None: fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng)
            if fit is None: fit = last
            if fit is not None: last = fit
            m = gok & (zgt > .3) & (zgt < 10)
            a, b = np.polyfit(1 / zgt[m], disp.reshape(-1)[m], 1) if m.sum() > 1000 else (np.nan, np.nan)
            rays_corr = (Rp @ (Rct.T @ dirs.T)).T; arms = {}
            for name, f in (('chain', fit), ('oracle', (a, b) if a > 0 else None)):
                if f is None:
                    arms[name] = [[False, False, 0, 0]] * 2; continue
                invd = (disp.reshape(-1) - f[1]) / f[0]; zz = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf); ok = np.isfinite(zz)
                pts = np.full((h * w, 3), np.nan); pts[ok] = (Rp @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                arms[name] = [list(x) for x in contact_readout(None, pts, ok, rays_corr, height, h, w, BOXES)]
            rows.append(dict(truth=truth, arms=arms))
        res[seq.name] = dict(height=height, frames=rows); Path(out).write_text(json.dumps(res))
        print(seq.name, N, f'{time.perf_counter()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
