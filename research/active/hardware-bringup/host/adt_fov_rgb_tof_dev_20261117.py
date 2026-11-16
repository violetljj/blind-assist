"""ADT offline comparison: RGB without ToF, full Atom view vs ToF field of view (ADT_RGB_VS_TOF_FOV_DEV_20261117.md).

The prepared RGB / GT-depth frames were removed in the 2026-10-25 storage cleanup; what remains per sequence is the
ToF-anchored DAV2-S metric depth at stride 2 (chain_depth_s2.npz, from adt_chain_depth_dump_dev_20261015.py), the camera
poses (poses.npz T_sc, same rotated camera as adt_prep), and the blob dumps with cached GT truth (bb/<backbone>/*.json).
Because each frame's anchor is an affine map of DAV2-S disparity, 1 / chain depth is an affine image of the original
disparity, so re-anchoring it on the floor (known camera height) is equivalent to a floor-only DAV2-S chain, up to
stride-2 sampling and float16.
Arms:
  R0s2  ToF-anchored chain rebuilt from chain_depth_s2 (reproduction check against the cached full-res R0)
  R0    cached dav2s/chain blobs (full resolution)
  R1    DAV2-S floor anchor only (no ToF), from chain_depth_s2
  R2    cached unidepth_l/raw blobs (metric, known K, no ToF)
Views: full Atom image; ToF FoV = blobs whose centroid direction lies inside the angular extent of the emulated zones.
v6 readout (blob >= 10 px, z50 <= 1.8 m counted, >= 150 px alert), 2-of-3 confirmation, FA on GT-negative, recall on
GT-positive with z50 <= 1.5 m. Aggregate numbers only (ADT license).
Usage: adt_fov_rgb_tof_dev_20261117.py <adt-eval dir> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import frozen_fusion_eval_v2 as V2

BOXES = list(V1.F.BOXES.values()); Z_MAX = V1.Z_MAX; STEP = 2


def corridor_from_scene(R_sc):   # as adt_prep_dev_20261013
    y = np.array([0., -1., 0.])
    fwd = R_sc[:, 2] - y * (R_sc[:, 2] @ y); fwd /= np.linalg.norm(fwd)
    return np.stack([np.cross(y, fwd), y, fwd])


def floor_anchor(disp, dirs, R, height, rng):
    """As real_rgb_small_model_dev_20261012.floor_anchor (RANSAC disparity = a * inv_z + b on floor-ray pixels 0.8-6 m),
    with non-finite disparity pixels excluded from the candidates."""
    r = (R @ dirs.T).T
    inv_z = r[:, 1] / height
    cand = np.flatnonzero((r[:, 1] > .1) & (inv_z > 1 / 6.) & (inv_z < 1 / .8) & np.isfinite(disp))
    if len(cand) < 500 // (STEP * STEP):
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


def blobs_s2(pts, ok, shape):
    out = []
    for lo, hi in BOXES:
        m = ok & (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= Z_MAX)
        bl = []
        if m.any():
            n, cc = cv2.connectedComponents(m.reshape(shape).astype(np.uint8), connectivity=8)
            lab = cc.reshape(-1)
            for k in range(1, n):
                s = lab == k
                if s.sum() * STEP * STEP >= 10:
                    p = pts[s]; bl.append([int(s.sum()) * STEP * STEP, float(np.median(p[:, 2]))] + p.mean(0).tolist())
        out.append([None, bl])
    return out


def tof_extent():
    rays = V1.X.zone_rays().reshape(-1, 3)
    return float(np.abs(np.arctan2(rays[:, 0], rays[:, 2])).max()), float(np.abs(np.arctan2(rays[:, 1], rays[:, 2])).max())


def readout(arms_q, R_cc, Rct, ext, view):
    alerts = np.zeros(2, bool)
    if arms_q is None:
        return alerts
    for q in (0, 1):
        cnt = 0
        for b in arms_q[q][1]:
            if b[1] > 1.8:
                continue
            if view == 'tof_fov':
                p = Rct.T @ (R_cc.T @ np.array(b[2:5]))
                if p[2] <= 0 or abs(np.arctan2(p[0], p[2])) > ext[0] or abs(np.arctan2(p[1], p[2])) > ext[1]:
                    continue
            cnt += b[0]
        alerts[q] = cnt >= 150
    return alerts


def main(root, out_json):
    root = Path(root); prep, bb = root / 'prep', root / 'bb'
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); ext = tof_extent()
    arms = ('R0s2', 'R0', 'R1', 'R2'); views = ('full', 'tof_fov', 'tof_fov_px')   # tof_fov_px added after the run (descriptive)
    cnt = {(a, v): {k: [0, 0] for k in ('fa', 'rec', 'rec_head', 'rec_body')} for a in arms for v in views}
    agree = [0, 0]; r1_fallback = [0, 0]; seqs = sorted(p.stem for p in (bb / 'dav2s').glob('*.json'))
    rng = np.random.default_rng(20261117)
    for seq in seqs:
        J0 = json.loads((bb / 'dav2s' / f'{seq}.json').read_text(encoding='utf8'))
        J2 = json.loads((bb / 'unidepth_l' / f'{seq}.json').read_text(encoding='utf8'))['frames']
        frames, height = J0['frames'], J0['height']
        meta = json.loads((prep / seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        Z = np.load(prep / seq / 'chain_depth_s2.npz'); zc = Z['z']; T = np.load(prep / seq / 'poses.npz')['T_sc']
        assert len(zc) == len(frames) == len(T) == len(J2), seq
        vv, uu = np.mgrid[0:h:STEP, 0:w:STEP]; shape = vv.shape
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        dt = dirs @ Rct   # pixel direction in the ToF frame (Rct.T @ d)
        fov_px = (dt[:, 2] > 0) & (np.abs(np.arctan2(dt[:, 0], dt[:, 2])) <= ext[0]) & (np.abs(np.arctan2(dt[:, 1], dt[:, 2])) <= ext[1])
        raw = {(a, v): np.zeros((len(frames), 2), bool) for a in arms for v in views}
        last = None
        for i, f in enumerate(frames):
            R_sc = T[i][:3, :3]; R_cc = corridor_from_scene(R_sc) @ R_sc
            z0 = zc[i].reshape(-1).astype(np.float64); ok0 = np.isfinite(z0) & (z0 > .05)
            pts0 = np.full((len(z0), 3), np.nan); pts0[ok0] = (R_cc @ (dirs[ok0] * z0[ok0, None]).T).T
            A = {'R0s2': blobs_s2(pts0, ok0, shape) if ok0.any() else None, 'R0': f['arms'].get('chain'), 'R2': J2[i]['arms'].get('raw')}
            PX = {'R0s2': blobs_s2(pts0, ok0 & fov_px, shape) if ok0.any() else None}
            inv = np.where(ok0, 1 / np.where(ok0, z0, 1.), np.nan)
            fit = floor_anchor(inv, dirs, R_cc, height, rng) if ok0.any() else None
            r1_fallback[1] += 1
            if fit is None:
                r1_fallback[0] += 1; fit = last
            else:
                last = fit
            A['R1'] = None
            if fit is not None and ok0.any():
                iz = (inv - fit[1]) / fit[0]; z1 = np.where(np.isfinite(iz) & (iz > 1e-3), 1 / np.maximum(np.nan_to_num(iz, nan=1e-3), 1e-3), np.inf)
                ok1 = np.isfinite(z1); pts1 = np.full((len(z1), 3), np.nan); pts1[ok1] = (R_cc @ (dirs[ok1] * z1[ok1, None]).T).T
                A['R1'] = blobs_s2(pts1, ok1, shape); PX['R1'] = blobs_s2(pts1, ok1 & fov_px, shape)
            for a in arms:
                for v in views:
                    if v == 'tof_fov_px':   # pixel-level crop: only arms with per-pixel depth (R0s2, R1)
                        raw[(a, v)][i] = readout(PX.get(a), R_cc, Rct, ext, 'full')
                    else:
                        raw[(a, v)][i] = readout(A[a], R_cc, Rct, ext, v)
        conf = {k: np.stack([V2.confirm(x[:, 0]), V2.confirm(x[:, 1])], 1) for k, x in raw.items()}
        for i, f in enumerate(frames):
            for q in (0, 1):
                tr = f['truth'][q]
                if tr['n'] == 0 or (tr['n'] >= 150 and tr['z50'] <= 1.5):
                    agree[0] += int(conf[('R0s2', 'full')][i, q] == conf[('R0', 'full')][i, q]); agree[1] += 1
                for key, c in conf.items():
                    if tr['n'] == 0:
                        cnt[key]['fa'][0] += int(c[i, q]); cnt[key]['fa'][1] += 1
                    elif tr['n'] >= 150 and tr['z50'] <= 1.5:
                        for k in ('rec', 'rec_head' if q == 0 else 'rec_body'):
                            cnt[key][k][0] += int(c[i, q]); cnt[key][k][1] += 1
        print(seq, len(frames), flush=True)
    res = dict(sequences=len(seqs), tof_half_extent_deg=[round(np.degrees(e), 2) for e in ext],
               reproduction_R0s2_vs_R0_full=dict(agree=agree, rate=round(agree[0] / max(agree[1], 1), 4)),
               R1_frames_without_new_floor_fit=r1_fallback,
               arms={f'{a}|{v}': {k: dict(n=x, rate=round(x[0] / max(x[1], 1), 4)) for k, x in cnt[(a, v)].items()} for a in arms for v in views})
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf8')
    print('ToF 区射线半角（水平, 垂直）:', res['tof_half_extent_deg'], '| R0 重建与原缓存的确认报警一致率', res['reproduction_R0s2_vs_R0_full'],
          '| R1 无新地面拟合帧', r1_fallback)
    for k, r in res['arms'].items():
        print(f'{k:14s} 误报 {100*r["fa"]["rate"]:5.2f}% ({r["fa"]["n"][0]}/{r["fa"]["n"][1]})  检出 {100*r["rec"]["rate"]:5.1f}% '
              f'({r["rec"]["n"][0]}/{r["rec"]["n"][1]})  头部 {100*r["rec_head"]["rate"]:5.1f}%  身体 {100*r["rec_body"]["rate"]:5.1f}%')


if __name__ == '__main__':
    main(*sys.argv[1:3])
