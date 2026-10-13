"""ToF down-tilt sweep with RGB re-anchoring (plan: adt-eval-20261013/PLAN_TILT_SWEEP.md).

Two stages (RAM is ~15 GB, so the GPU runtime is loaded once and CPU workers stay small):
  cache <prep> <cache dir>              one GPU process: DAV2-S disparity per frame (float32) and GT depth as .npy
  run   <prep> <cache dir> <out dir> N  N CPU processes, one task per (sequence, ToF variant, tilt): the sequential
                                        v6 anchor chain on the tilted zones, then the readout of every frame
For every extra down-tilt t in TILTS and ToF variant
  'i' ideal   : zone samples ranged from GT depth, as adt_eval / adt_tof_tilt_sim (anchor zone median valid 0.3-4 m)
  'd' degraded: each zone dropped with p = 0.25 (unusable: no anchor, alarm or veto), no return beyond 3.0 m,
                per-zone multiplicative range error N(0, 0.04), seeded by (sequence index, frame, tilt); a kept zone
                without a return near the blob range counts as free space for the veto
the chain (fit_affine -> robust -> floor -> last) is fitted on the tilted zones, every in-box blob (>= 10 px) is stored
as [npx, z50, veto] where veto = blob centre inside a usable zone's cone (<= 4 deg) and < 10 % of that zone's GT-valid
samples return within +-0.4 m of the blob range; ToF alarm per box = a usable zone with >= 13 GT-valid samples of which
>= 30 % return inside the box. Truth and the current-mount ToF axis pitch are stored once per frame (task 'truth').
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
from adt_backbone_blobs_dev_20261013 import box_masks

TILTS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
VARIANTS = ('i', 'd')
P_DROP, CAP_D, SIG_D = .25, 3.0, .04
BOXES = list(V1.F.BOXES.values())


def rx(d):
    a = np.radians(d); return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def seqs_of(prep):
    return sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists())


def cache(prep, cdir):
    import torch
    import torch.nn.functional as Fn
    from concurrent.futures import ThreadPoolExecutor
    cdir = Path(cdir); cdir.mkdir(parents=True, exist_ok=True)
    model, _ = V1.S.load_dav2_small(); pool = ThreadPoolExecutor(2); start = time.perf_counter()
    for seq in seqs_of(prep):
        if (cdir / f'{seq.name}.disp.npy').exists():
            continue
        meta = json.loads((seq / 'meta.json').read_text()); w, h = meta['image_size']
        with np.load(seq / 'frames.npz') as Z:
            rgb = Z['rgb']; np.save(cdir / f'{seq.name}.depth.npy', Z['depth']); np.save(cdir / f'{seq.name}.R_cc.npy', Z['R_cc'])
            np.save(cdir / f'{seq.name}.cam_y.npy', Z['cam_y'])
        N = len(rgb); tmp = cdir / f'{seq.name}.disp.tmp.npy'
        out = np.lib.format.open_memmap(tmp, 'w+', np.float32, (N, h, w))
        def prep_t(i):
            t_, _ = model.image2tensor(np.ascontiguousarray(rgb[i][:, :, ::-1]), V1.SIZE); return t_
        pend = [pool.submit(prep_t, i) for i in range(min(3, N))]
        for i in range(N):
            x = pend.pop(0).result()
            if i + 3 < N:
                pend.append(pool.submit(prep_t, i + 3))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(x)
            out[i] = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
        out.flush(); del out, rgb; tmp.replace(cdir / f'{seq.name}.disp.npy')
        print('cache', seq.name, N, f'{time.perf_counter()-start:.0f}s', flush=True)


class Seq:
    """Per-sequence geometry shared by the tasks of one process."""
    def __init__(self, prep, cdir, seq):
        meta = json.loads((Path(prep) / seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; self.w, self.h = w, h = meta['image_size']
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        vv, uu = np.mgrid[0:h, 0:w]
        self.dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        c = Path(cdir)
        self.disp = np.load(c / f'{seq}.disp.npy', mmap_mode='r'); self.depth = np.load(c / f'{seq}.depth.npy', mmap_mode='r')
        self.R = np.load(c / f'{seq}.R_cc.npy'); cam_y = np.load(c / f'{seq}.cam_y.npy'); N = len(cam_y)
        hts = []
        for i in range(0, N, max(1, N // 20)):
            z = self.depth[i].reshape(-1) / 1000.; ok = z > .1
            hts.append(cam_y[i] - (self.R[i] @ (self.dirs[ok] * z[ok, None]).T).T[:, 1])
        self.height = float(np.median(cam_y) - np.percentile(np.concatenate(hts), 1)); self.N = N
        self.Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
        self.rays0 = V1.X.zone_rays()
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy


_SEQ = {}


def get_seq(prep, cdir, seq):
    if seq not in _SEQ:
        _SEQ.clear(); _SEQ[seq] = Seq(prep, cdir, seq)
    return _SEQ[seq]


def blob_list(mask, pts, h, w):
    import cv2
    out = []
    if mask.any():
        n, cc = cv2.connectedComponents(mask.reshape(h, w).astype(np.uint8), connectivity=8)
        cc = cc.reshape(-1); idx = np.nonzero(mask)[0]; lab = cc[idx]
        order = np.argsort(lab, kind='stable'); idx, lab = idx[order], lab[order]
        for g in np.split(idx, np.flatnonzero(np.diff(lab)) + 1):
            if len(g) >= 10:
                p = pts[g]; out.append((len(g), float(np.median(p[:, 2])), p.mean(0)))
    return out


def task(args):
    prep, cdir, out, si, seq, key = args
    dst = Path(out) / f'{seq}.{key}.json'
    if dst.exists():
        return seq, key, 0.
    t0 = time.perf_counter(); S = get_seq(prep, cdir, seq); h, w, dirs, R = S.h, S.w, S.dirs, S.R
    if key == 'truth':
        rows = []
        for i in range(S.N):
            z = S.depth[i].reshape(-1) / 1000.; gok = z > .1
            gpts = np.full((h * w, 3), np.nan); gpts[gok] = (R[i] @ (dirs[gok] * z[gok, None]).T).T
            ax = R[i] @ S.Rct @ np.array([0, 0, 1.])
            rows.append(dict(truth=[dict(n=int(m.sum()), z50=float(np.median(gpts[m, 2])) if m.any() else None) for m in box_masks(gpts, gok)],
                             pitch=round(float(np.degrees(np.arctan2(ax[1], ax[2]))), 2)))
        dst.write_text(json.dumps(rows), encoding='utf8'); return seq, key, time.perf_counter() - t0
    v, t = key[0], int(key[1:])
    rays = S.rays0 @ rx(-t).T                                         # ToF pitched down by t (as adt_tof_tilt_sim)
    rc = rays @ S.Rct.T; un = rc / np.linalg.norm(rc, axis=-1, keepdims=True)
    zu = np.round(S.fx * rc[..., 0] / rc[..., 2] + S.cx).astype(int); zv = np.round(S.fy * rc[..., 1] / rc[..., 2] + S.cy).astype(int)
    zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h); zu, zv = np.clip(zu, 0, w - 1), np.clip(zv, 0, h - 1)
    ctr = un.mean(1); ctr /= np.linalg.norm(ctr, axis=1, keepdims=True)
    rng = np.random.default_rng(20261013); last = None; rows = []
    for i in range(S.N):
        zgt = S.depth[i] / 1000.; disp = np.asarray(S.disp[i])
        zz = zgt[zv, zu]; ok0 = zin & (zz > .1)
        rr = np.where(ok0, zz / np.maximum(un[..., 2], 1e-6), np.nan)
        if v == 'i':
            use = np.ones(64, bool); ret = ok0; cap = 4.
        else:
            g = np.random.default_rng([20261014, si, i, t])
            use = g.random(64) >= P_DROP; rr = rr * (1 + SIG_D * g.standard_normal(64))[:, None]; cap = CAP_D
            ret = ok0 & use[:, None] & (np.nan_to_num(rr, nan=99.) < cap)
        nv = ok0.sum(1)
        dist = np.full(64, np.nan)
        for zi in range(64):
            if use[zi] and ret[zi].sum() >= 13 and nv[zi] >= 13:
                m_ = float(np.median(rr[zi][ret[zi]]))
                if .3 < m_ < cap:
                    dist[zi] = m_
        dz, zfac = V1.S.zone_disparity(disp, S.K, S.Rct, rays)
        inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
        fit = V1.S.fit_affine(dz, inv); src = 'tof' if fit else None
        if not fit_quality(dz, inv, fit)['accepted']:
            rb = robust_affine(dz, inv)['fit']
            if rb is not None:
                fit, src = rb, 'robust'
        if fit is None:
            fit = V1.S.floor_anchor(disp, dirs, R[i] @ S.Rct, S.Rct, S.height, rng); src = 'floor' if fit else None
        if fit is None and last is not None:
            fit, src = last, 'last'
        if fit is not None:
            last = fit
        sp = (un * np.nan_to_num(rr)[..., None]) @ R[i].T
        tof = []
        for lo, hi in BOXES:
            inb = ret & (np.abs(sp[..., 0]) <= .3) & (sp[..., 1] >= lo) & (sp[..., 1] <= hi) & (sp[..., 2] >= .3) & (sp[..., 2] <= 2.)
            tof.append(bool((use & (nv >= 13) & (inb.sum(1) >= .3 * np.maximum(nv, 1))).any()))
        bl = [[], []]
        if fit is not None:
            invd = (disp.reshape(-1) - fit[1]) / fit[0]
            zp = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf); ok = np.isfinite(zp)
            pts = np.full((h * w, 3), np.nan); pts[ok] = (R[i] @ (dirs[ok] * zp[ok, None]).T).T
            for q, mk in enumerate(box_masks(pts, ok)):
                for npx, z50, c in blob_list(mk, pts, h, w):
                    veto = False
                    if z50 <= 2.0:
                        pc = R[i].T @ c; r = np.linalg.norm(pc); d = pc / r; k = int(np.argmax(ctr @ d))
                        if np.degrees(np.arccos(np.clip(ctr[k] @ d, -1, 1))) <= 4. and use[k] and nv[k] >= 13:
                            veto = bool((ret[k] & (np.abs(np.nan_to_num(rr[k]) - r) <= .4)).sum() < .1 * nv[k])
                    bl[q].append([npx, round(z50, 3), int(veto)])
        rows.append(dict(src=src, tof=tof, blobs=bl))
    dst.write_text(json.dumps(rows), encoding='utf8')
    return seq, key, time.perf_counter() - t0


def run(prep, cdir, out, n):
    Path(out).mkdir(parents=True, exist_ok=True)
    seqs = [p.name for p in seqs_of(prep)]
    keys = ['truth'] + [f'{v}{t}' for v in VARIANTS for t in TILTS]
    jobs = [(prep, cdir, out, si, s, k) for si, s in enumerate(seqs) for k in keys]   # sequence-major: few Seq loads
    start = time.perf_counter(); done = 0
    with Pool(int(n), maxtasksperchild=40) as pool:
        for seq, key, dt in pool.imap_unordered(task, jobs, chunksize=1):
            done += 1
            if done % 21 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} tasks, last {seq[18:40]} {key} {dt:.0f}s, wall {time.perf_counter()-start:.0f}s', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'cache':
        cache(sys.argv[2], sys.argv[3])
    else:
        run(*sys.argv[2:6])
