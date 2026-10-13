"""ToF-zone <-> RGB-disparity anchor correspondence candidates on consumed captures (EXPLORE, v4 candidate work).

Frozen v1/v2/v3 untouched. The v1/v3 anchor fits firmware zone distance against the MEDIAN of 25 sampled
disparities per zone footprint. Candidates change only that per-zone summary:
  base      : 5x5 samples, median (exactly the v3 chain)
  dense50   : 15x15 samples, median
  near70    : 15x15 samples, 70th disparity percentile (nearer-biased)
  far30     : 15x15 samples, 30th disparity percentile (farther-biased)
  mode      : 15x15 samples, peak of a 16-bin per-zone disparity histogram
  strength  : 15x15 samples; initial dense50 chain fit converts samples to metric inverse depth, then the
              peak of an inverse-depth histogram weighted by inv_z^2 (crude strongest-return proxy)
Each candidate then runs the v3 chain (fit -> robust replacement -> floor anchor -> last fit).
Measured per candidate: quality-contract pass rate, a floor-depth check on ToF-anchored frames (median fitted /
geometric depth over pixels whose ray meets the calibrated floor plane 0.8-4 m ahead; obstacles bias it), and the v1-criteria session verdicts
with 2-of-3 confirmation.
Usage:
  anchor_correspondence_dev_20261013.py cache-morning   (GPU: DAV2 disparity for morning ready frames)
  anchor_correspondence_dev_20261013.py evaluate        (CPU)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_small_model_dev_20261012 as S
import real_rgb_query_feasibility_dev_20261012 as F
import fusion_v3_dev_20261013 as D
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine

WORK = ROOT / 'artifacts.local/work/fusion-v3-dev-20261013'
OUT = ROOT / 'artifacts.local/work/anchor-correspondence-dev-20261013'
AFTERNOON = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
MORNING_RAW = ROOT / 'artifacts.local/work/phone-imu-capture-20261012/raw'
CANDIDATES = ('base', 'dense50', 'near70', 'far30', 'mode', 'strength')


def cache_morning():
    import cv2
    import torch
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((WORK / 'cache_morning.json').read_text(encoding='utf8'))
    cam = AtomCamera(); model, _ = S.load_dav2_small()
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    start = time.perf_counter()
    for name, s in rep['sessions'].items():
        d = MORNING_RAW / name
        x = R.load(d, imu_rot)
        dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
        rgb = list(F.lines(d / 'rgb.jsonl'))
        folder = OUT / 'cache' / name[17:23]; folder.mkdir(parents=True, exist_ok=True)
        for f in s['frames']:
            if not f['rgb_index']:
                continue
            up = cam.undistort_upright(cv2.imread(str(d / rgb[f['rgb_index'][0]]['file']))[:, :, ::-1])
            with torch.inference_mode():
                disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), 518)
            np.savez_compressed(folder / f"{f['tof_index']:04d}.npz", disp=disp.astype(np.float32), valid=up.sum(axis=2) > 0,
                                dist=dist[f['tof_index']])
        print(name[17:23], round(time.perf_counter() - start), 's', flush=True)


class Geometry:
    def __init__(self):
        cam = AtomCamera(); self.K = cam.K_upright; w, h = cam.upright_size; self.h, self.w = h, w
        self.Rct = np.array(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
        vv, uu = np.mgrid[0:h, 0:w]
        self.dirs = np.stack([(uu - self.K[0, 2]) / self.K[0, 0], (vv - self.K[1, 2]) / self.K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
        self.rays5 = X.zone_rays(); self.idx15, self.zfac15 = self._pixels(X.zone_rays(15))

    def _pixels(self, rays):
        rc = rays @ self.Rct.T
        u = self.K[0, 0] * rc[..., 0] / rc[..., 2] + self.K[0, 2]; v = self.K[1, 1] * rc[..., 1] / rc[..., 2] + self.K[1, 2]
        inside = (rc[..., 2] > .05) & (u >= 0) & (u < self.w - 1) & (v >= 0) & (v < self.h - 1)
        idx = np.where(inside, np.clip(v.round().astype(int), 0, self.h - 1) * self.w + np.clip(u.round().astype(int), 0, self.w - 1), -1)
        zfac = np.nanmedian(np.where(inside, rc[..., 2] / np.linalg.norm(rc, axis=-1), np.nan), 1)
        return idx, zfac


def summaries(disp, G, dist, valid_zone):
    """Per-candidate per-zone disparity summary (NaN where < half the samples land on the image)."""
    out = {}
    out['base'], zfac5 = S.zone_disparity(disp, G.K, G.Rct, G.rays5)
    flat = disp.reshape(-1)
    samp = np.where(G.idx15 >= 0, flat[np.maximum(G.idx15, 0)], np.nan)        # [64, 225]
    enough = np.isfinite(samp).sum(1) >= samp.shape[1] // 2
    def per(fn):
        v = np.full(64, np.nan)
        for z in np.flatnonzero(enough):
            v[z] = fn(samp[z][np.isfinite(samp[z])])
        return v
    out['dense50'] = per(np.median)
    out['near70'] = per(lambda s: np.percentile(s, 70))
    out['far30'] = per(lambda s: np.percentile(s, 30))
    def mode(s):
        hist, edges = np.histogram(s, bins=16)
        k = int(np.argmax(hist)); return float(np.median(s[(s >= edges[k]) & (s <= edges[k + 1])]))
    out['mode'] = per(mode)
    return out, zfac5, samp, enough


def chain(dz, inv, disp, valid_px, Rp, G, height, rng, last):
    fit = S.fit_affine(dz, inv); src = 'tof' if fit else None
    q = fit_quality(dz, inv, fit); accepted = q['accepted']
    if not accepted:
        rb = robust_affine(dz, inv)['fit']
        if rb is not None:
            fit, src, accepted = rb, 'robust_tof', True
    if fit is None:
        fit = S.floor_anchor(disp, G.dirs, Rp, G.Rct, height, rng); src = 'floor' if fit else None
    if fit is None and last is not None:
        fit, src = last, 'last'
    return fit, src, accepted


G_height = [1.35]


def readout(disp, valid_px, fit, Rp, G):
    a, b = fit
    inv = (disp.reshape(-1) - b) / a
    zz = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)
    ok = np.isfinite(zz) & valid_px
    pts = (Rp @ (G.Rct.T @ (G.dirs[ok] * zz[ok, None]).T)).T
    px = []
    for lo, hi in F.BOXES.values():
        inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= 2.)
        px.append(int(inside.sum()))
    # Floor check: pixels whose ray meets the calibrated floor plane 0.8-4 m ahead; median fitted/geometric depth.
    r = (Rp @ (G.Rct.T @ G.dirs.T)).T
    zg = np.where(r[:, 1] > .05, G_height[0] / np.maximum(r[:, 1], 1e-6), np.inf)
    cand = ok & (zg > .8) & (zg < 4.)
    cand_full = np.zeros(len(zz), bool); cand_full[:] = cand
    ratio = float(np.median(zz[cand_full] / zg[cand_full])) if cand_full.sum() >= 500 else None
    return px, ratio


def frames_of(batch):
    """Yield (session, cls, height, list of frame dicts with disp path, dist, pose, tof_strong)."""
    rep = json.loads((WORK / f'cache_{batch}.json').read_text(encoding='utf8'))
    height = rep['head_height_m']
    if batch == 'afternoon':
        man = json.loads((AFTERNOON / 'cache_manifest.json').read_text())
        bykey = {s['session']: {f['i']: f for f in s['frames']} for s in man['sessions'].values()}
    for name in sorted(rep['sessions']):
        s = rep['sessions'][name]; rows = []
        for f in s['frames']:
            r = dict(tof_index=f['tof_index'], pitch=f['pitch'], roll=f['roll'], tof_strong=f['tof_strong'], path=None, dist=None)
            if f['rgb_index']:
                if batch == 'morning':
                    r['path'] = OUT / 'cache' / name[17:23] / f"{f['tof_index']:04d}.npz"
                else:
                    m = bykey[name].get(f['tof_index'])
                    if m and 'cache' in m:
                        r['path'] = AFTERNOON / m['cache']; r['dist'] = np.array(m['dist'], float)
            rows.append(r)
        yield name, s['cls'], s['scene'], height, rows


def evaluate():
    G = Geometry(); start = time.perf_counter()
    result = dict(evidence='consumed Development', candidates=CANDIDATES, sessions={})
    for batch in ('morning', 'afternoon'):
        for name, cls, scene, height, rows in frames_of(batch):
            G_height[0] = height
            rng = {c: np.random.default_rng(20261013) for c in CANDIDATES}
            last = {c: None for c in CANDIDATES}
            per = {c: dict(rgb=[], accepted=0, src={}, floor_err=[]) for c in CANDIDATES}
            tof = np.array([r['tof_strong'] for r in rows], bool).reshape(-1, 2)
            n_rgb = 0
            for r in rows:
                if r['path'] is None or not Path(r['path']).exists():
                    for c in CANDIDATES:
                        per[c]['rgb'].append([False, False])
                    continue
                n_rgb += 1
                with np.load(r['path']) as p:
                    disp, valid = p['disp'], p['valid'].reshape(-1)
                    dist = p['dist'] if 'dist' in p else r['dist']
                vz = np.isfinite(dist) & (dist > .3) & (dist < 4.)
                summ, zfac, samp, enough = summaries(disp, G, dist, vz)
                inv = np.where(vz, 1 / (dist * zfac), np.nan)
                Rp = F.rot(r['pitch'], r['roll'])
                for c in CANDIDATES:
                    if c == 'strength':
                        f0, _, _ = chain(summ['dense50'], inv, disp, valid, Rp, G, height, np.random.default_rng(1), None)
                        dz = summ['dense50'].copy()
                        if f0 is not None:
                            a0, b0 = f0
                            for z in np.flatnonzero(enough):
                                s = samp[z][np.isfinite(samp[z])]; iz = (s - b0) / a0
                                good = iz > 1e-3
                                if good.sum() < 10:
                                    continue
                                hist, edges = np.histogram(iz[good], bins=16, weights=iz[good] ** 2)
                                k = int(np.argmax(hist)); sel = s[good][(iz[good] >= edges[k]) & (iz[good] <= edges[k + 1])]
                                dz[z] = float(np.median(sel))
                    else:
                        dz = summ[c]
                    fit, src, acc = chain(dz, inv, disp, valid, Rp, G, height, rng[c], last[c])
                    if fit is not None:
                        last[c] = fit
                    per[c]['accepted'] += int(acc)
                    per[c]['src'][str(src)] = per[c]['src'].get(str(src), 0) + 1
                    if fit is None:
                        per[c]['rgb'].append([False, False]); continue
                    px, fh = readout(disp, valid, fit, Rp, G)
                    per[c]['rgb'].append([v >= 150 for v in px])
                    if src in ('tof', 'robust_tof') and fh is not None:
                        per[c]['floor_err'].append(fh)
            row = dict(cls=cls, scene=scene, batch=batch, ready=len(rows), rgb_frames=n_rgb, height=height, candidates={})
            for c in CANDIDATES:
                rgb = np.array(per[c]['rgb'], bool).reshape(-1, 2)
                a = tof | rgb
                a = np.stack([D.V2.confirm(a[:, 0]), D.V2.confirm(a[:, 1])], 1) if len(a) else a
                sc = D.session_score(cls, a)
                row['candidates'][c] = dict(accepted=per[c]['accepted'], src=per[c]['src'], floor_err=per[c]['floor_err'], **sc)
            result['sessions'][name] = row
            print(batch[:2], name[17:23], cls[:7], n_rgb, ' '.join(f"{c}:{row['candidates'][c]['accepted']}/{row['candidates'][c].get('fused_any_alert', 0) or 0:.2f}" for c in CANDIDATES),
                  f'{time.perf_counter()-start:.0f}s', flush=True)
    result['wall_s'] = time.perf_counter() - start
    (OUT / 'evaluate.json').write_text(json.dumps(result), encoding='utf8')


if __name__ == '__main__':
    {'cache-morning': cache_morning, 'evaluate': evaluate}[sys.argv[1]]()
