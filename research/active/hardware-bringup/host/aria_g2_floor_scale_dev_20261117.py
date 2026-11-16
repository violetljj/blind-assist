"""Outdoor glasses view: floor anchoring with VIO gravity + known eye height vs emulated ToF scale (ARIA_G2_FLOOR_SCALE_DEV_20261117.md).
Aria Gen 2 Pilot walk_0 / walk_1 (CC BY-NC 4.0): FoundationStereo depth (pseudo-GT, mm along ray) on the rectified 512x512
front-left CV camera, the matching rectified grayscale image as model input, and the gravity-aligned MPS world pose.
Usage: aria_g2_floor_scale_dev_20261117.py <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanpo_scale_without_tof_dev_20261117 as P
from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine

DATA = P.A / 'downloads/aria-gen2-pilot-20261117'
SEQS = ('walk_0', 'walk_1'); STEP, BLOCK_S = 15, 30.
X_HALF, Z_NEAR, Z_FAR, HGT = .30, .30, 2.0, (.15, 1.90)
H_ERR = (-.10, -.05, 0., .05, .10)


def ground_plane(pts, down, rng):
    """RANSAC plane below the camera with normal within 15 deg of gravity; returns (unit up normal, camera height) or None."""
    below = pts[(pts @ down) > .5]
    if len(below) < 2000:
        return None
    sel = below[rng.choice(len(below), min(30000, len(below)), replace=False)]
    best = None
    for _ in range(300):
        p = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        if n @ down > 0:
            n = -n                                   # up normal
        if np.degrees(np.arccos(np.clip(n @ -down, -1, 1))) > 15:
            continue
        d = -n @ p[0]                                # camera height = d (origin side)
        if d <= .3:
            continue
        inl = np.abs(sel @ n + d) < .03
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), n, float(d))
    if best is None:
        return None
    _, n, d = best
    inl_all = np.abs(below @ n + d) < .03
    return (n, d) if inl_all.sum() >= 2000 else None


def main(out_json):
    import torch
    from PIL import Image
    rel, met, ud, ud_infer = P.load_models()
    S = P.V1.S; rays = P.V1.X.zone_rays(); I3 = np.eye(3)
    rng = np.random.default_rng(20261117)
    arms = ['A', 'B_ref'] + [f'B_vio{dh:+.2f}' for dh in H_ERR] + ['C', 'D', 'O']
    rows = []
    for seq in SEQS:
        base = DATA / seq / 'x/depth'
        params = json.loads((base / 'pinhole_camera_parameters.json').read_text())
        t0 = params[0]['frameTimestampNs']; W = H = 512
        vv, uu = np.mgrid[0:H, 0:W]

        def load(p):   # intrinsics differ slightly per frame (per-frame rectified pinhole), so build rays per frame
            fu, fv, u0, v0 = p['camera']['Parameters']
            K = np.array([[fu, 0, u0], [0, fv, v0], [0, 0, 1.]])
            dirs = np.stack([(uu - u0) / fu, (vv - v0) / fv, np.ones((H, W))], -1).reshape(-1, 3)
            i = p['index']
            r = cv2.imread(str(base / 'depth' / f'depth_{i:08d}.png'), cv2.IMREAD_UNCHANGED).reshape(-1).astype(np.float64) / 1000.
            z = r / np.linalg.norm(dirs, axis=1)                    # optical depth
            valid = (r > 0) & (z > .3) & (z < 10)
            down = Rotation.from_quat(p['T_world_camera']['QuaternionXYZW']).as_matrix().T @ np.array([0, 0, -1.])
            return K, dirs, z, valid, dirs * np.where(valid, z, 0)[:, None], down
        F = []
        for p in params[::STEP]:   # pass 1: reference ground plane only (no point clouds kept in memory)
            K, dirs, z, valid, pts, down = load(p)
            F.append(dict(p=p, i=p['index'], t=(p['frameTimestampNs'] - t0) / 1e9, gp=ground_plane(pts[valid], down, rng)))
        hs = [f['gp'][1] for f in F if f['gp'] is not None]
        h_seq = float(np.median(hs)) if hs else None
        last_tof = None
        for f in F:                # pass 2
            K, dirs, z, valid, pts, f['down'] = load(f['p'])
            fu, fv, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            gray = cv2.imread(str(base / 'rectified_images' / f"image_{f['i']:08d}.png"), cv2.IMREAD_UNCHANGED)
            bgr = np.repeat(gray[:, :, None], 3, 2)
            with torch.inference_mode():
                disp = rel.infer_image(bgr, 518).astype(np.float32).reshape(-1)
                zC = met.infer_image(bgr, 518).astype(np.float32).reshape(-1)
            zD = ud_infer(Image.fromarray(bgr), K).reshape(-1)
            # A: coaxial 45 deg ToF zones from the reference depth
            zmap = np.where(valid, z, np.nan).reshape(H, W); rc = rays
            zu = np.round(fu * rc[..., 0] / rc[..., 2] + u0).astype(int); zv = np.round(fv * rc[..., 1] / rc[..., 2] + v0).astype(int)
            zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < W) & (zv >= 0) & (zv < H); zcos = rc[..., 2] / np.linalg.norm(rc, axis=-1)
            dist = np.full(len(rays), np.nan)
            for zi in range(len(rays)):
                s = zin[zi]
                if s.sum() < 13:
                    continue
                zz = zmap[zv[zi][s], zu[zi][s]]; ok = np.isfinite(zz)
                if ok.sum() >= 13:
                    rr = float(np.median(zz[ok] / zcos[zi][s][ok]))
                    if .3 < rr < 4.:
                        dist[zi] = rr
            dz, zfac = S.zone_disparity(disp.reshape(H, W), K, I3, rays)
            inv = np.where(np.isfinite(dist), 1 / (dist * zfac), np.nan)
            fitA = S.fit_affine(dz, inv)
            if not fit_quality(dz, inv, fitA)['accepted']:
                rb = robust_affine(dz, inv)['fit']
                if rb is not None:
                    fitA = rb
            fitA = fitA if fitA is not None else last_tof
            last_tof = fitA if fitA is not None else last_tof
            Z = {'A': P.to_depth(disp, fitA), 'C': zC, 'D': zD}
            Z['B_ref'] = P.to_depth(disp, P.floor_anchor(disp, -(dirs @ f['gp'][0]), h_seq, rng)) if (f['gp'] is not None and h_seq) else None
            for dh in H_ERR:
                Z[f'B_vio{dh:+.2f}'] = P.to_depth(disp, P.floor_anchor(disp, dirs @ f['down'], h_seq + dh, rng)) if h_seq else None
            m = valid & np.isfinite(disp)
            a_, b_ = np.polyfit(1 / z[m], disp[m], 1)
            Z['O'] = P.to_depth(disp, (a_, b_) if a_ > 0 else None)
            near = valid & (z >= .5) & (z <= 3.)
            near_ng = corr = np.zeros_like(valid)
            gang = None
            if f['gp'] is not None:
                n, h = f['gp']; dw = -n
                hgt = h - pts @ dw
                near_ng = near & (hgt > .15)
                fwd = np.array([0, 0, 1.]) - dw * dw[2]; fwd /= np.linalg.norm(fwd); right = np.cross(dw, fwd)
                corr = valid & (np.abs(pts @ right) <= X_HALF) & (pts @ fwd >= Z_NEAR) & (pts @ fwd <= Z_FAR) & (hgt >= HGT[0]) & (hgt <= HGT[1])
                gang = float(np.degrees(np.arccos(np.clip(n @ -f['down'], -1, 1))))
            row = dict(seq=seq, index=f['i'], block=f"{seq}:{int(f['t'] // BLOCK_S)}", gravity_angle_deg=gang, ground=f['gp'] is not None,
                       fit={a: Z[a] is not None for a in arms})
            for a in arms:
                row[a] = dict(near_nonground=P.metrics(Z[a], z, near_ng) if near_ng.sum() >= 200 else None,
                              near=P.metrics(Z[a], z, near), corridor=P.metrics(Z[a], z, corr) if corr.sum() >= 200 else None)
            rows.append(row)
        spread = np.ptp([f['p']['camera']['Parameters'][0] for f in F])
        print(seq, len(F), 'frames, focal spread %.2f px' % spread, 'h_seq', None if h_seq is None else round(h_seq, 3), flush=True)
    blocks = sorted({r['block'] for r in rows}); summ = {}
    for region in ('near_nonground', 'near', 'corridor'):
        for a in arms:
            per = []
            for b in blocks:
                v = [r[a][region]['absrel'] for r in rows if r['block'] == b and r[a][region] and r[a][region]['absrel'] is not None]
                d1 = [r[a][region]['delta1'] for r in rows if r['block'] == b and r[a][region]]
                if v:
                    per.append((b, float(np.median(v)), float(np.median(d1)), len(v)))
            summ[f'{region}|{a}'] = dict(blocks=len(per), frames=sum(p[3] for p in per),
                                          absrel=float(np.median([p[1] for p in per])) if per else None,
                                          delta1=float(np.median([p[2] for p in per])) if per else None, per_block=per)
    fit_rate = {a: [sum(r['fit'][a] for r in rows), len(rows)] for a in arms}
    ang = np.array([r['gravity_angle_deg'] for r in rows if r['gravity_angle_deg'] is not None])
    A_, Bv = summ['near_nonground|A']['absrel'], summ['near_nonground|B_vio+0.00']['absrel']
    ratio = Bv / A_ if (A_ and Bv) else None; rate = fit_rate['B_vio+0.00'][0] / fit_rate['B_vio+0.00'][1]
    verdict = ('HOLDS' if ratio is not None and ratio <= 1.25 and rate >= .9 else
               'DOES_NOT_HOLD' if (ratio is None or ratio > 2 or rate < .7) else 'UNCERTAIN')
    res = dict(frames=len(rows), blocks=len(blocks), ground_rate=[sum(r['ground'] for r in rows), len(rows)], fit_rate=fit_rate,
               gravity_vs_ground_deg=dict(median=float(np.median(ang)), p95=float(np.percentile(ang, 95)), n=len(ang)) if len(ang) else None,
               primary=dict(ratio_Bvio_over_A=ratio, Bvio_fit_rate=rate, verdict=verdict), summary=summ, rows=rows)
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf8')
    print('帧', len(rows), '块', len(blocks), '| 参考地面', res['ground_rate'], '| VIO 重力与参考地面夹角', res['gravity_vs_ground_deg'])
    print('锚定成功率', {a: fit_rate[a] for a in ('A', 'B_ref', 'B_vio+0.00')})
    for k, v in summ.items():
        print(f"{k:28s} 块 {v['blocks']:2d} 帧 {v['frames']:4d}  AbsRel {v['absrel'] if v['absrel'] is None else round(v['absrel'], 3)}  δ1 {v['delta1'] if v['delta1'] is None else round(v['delta1'], 3)}")
    print('主判据：B_vio/A =', None if ratio is None else round(ratio, 3), '成功率 %.2f' % rate, '→', verdict)


if __name__ == '__main__':
    main(sys.argv[1])
