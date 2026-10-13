"""System-level effect of re-aiming the ToF further down, on prepared ADT frames (plan: adt-eval-20261013/PLAN_TOF_TILT.md).

ToF zones are emulated from GT depth (zone sample rays of our sensor model, camera<-ToF extrinsic, optional extra
down-tilt t about the ToF x axis); optimistic: no noise, crosstalk or dark-surface dropouts. RGB = the DAV2-S v6 chain
blobs from adt_backbone_blobs_dev_20261013 (the RGB anchor still uses the current ToF; re-anchoring is not simulated).
  ToF alarm (query q): some zone has >= 30 % of its sample points inside the box (first-return-like coverage).
  ToF veto of an RGB blob: the blob centre lies inside a zone's cone (nearest zone centre within 4 deg) and < 10 % of
  that zone's sample ranges fall within +-0.4 m of the blob range (histogram has no return there) -> blob not counted.
Arms per tilt t in {0, 25, 35}: rgb (v6), tof, fused = tof OR rgb, fused_veto = tof OR (rgb with veto). 2-of-3 confirm.
Usage: adt_tof_tilt_sim_dev_20261013.py <prep dir> <bb dav2s dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import frozen_fusion_eval_v2 as V2

TILTS = tuple(int(x) for x in __import__("os").environ.get("TILTS", "0,25,35").split(","))
Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
BOXES = list(V1.F.BOXES.values())


def rx(d):
    a = np.radians(d); return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def main(prep, bbdir, out):
    rays0 = V1.X.zone_rays()                                          # [64, ns, 3] in the ToF frame
    cnt = {(t, arm): dict(fa=[0, 0], rec=[0, 0], rec_head=[0, 0], rec_body=[0, 0]) for t in TILTS for arm in ('rgb', 'tof', 'fused', 'fused_veto')}
    vstat = {t: dict(fa_blobs=[0, 0], tp_blobs=[0, 0]) for t in TILTS}
    for seq in sorted(p for p in Path(prep).iterdir() if (p / 'frames.npz').exists()):
        meta = json.loads((seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        frames = json.loads((Path(bbdir) / f'{seq.name}.json').read_text(encoding='utf8'))['frames']
        with np.load(seq / 'frames.npz') as Z:
            dep, R = Z['depth'], Z['R_cc']
        for t in TILTS:
            rc = (rays0 @ rx(-t).T) @ Rct.T                           # zone sample rays in the camera frame
            # NOTE rays0 @ rx(-t).T == rays0 rotated by rx(-t): ToF pitched down by t (sign as in adt_tof_coverage)
            un = rc / np.linalg.norm(rc, axis=-1, keepdims=True)
            zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
            zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h)
            ctr = un.mean(1); ctr /= np.linalg.norm(ctr, axis=1, keepdims=True)
            A = {a: np.zeros((len(frames), 2), bool) for a in ('rgb', 'tof', 'fused', 'fused_veto')}
            for i, f in enumerate(frames):
                zgt = dep[i] / 1000.
                rng_ = np.full(zin.shape, np.nan)
                zz = zgt[np.clip(zv, 0, h - 1), np.clip(zu, 0, w - 1)]
                ok = zin & (zz > .1)
                rng_[ok] = zz[ok] / un[..., 2][ok]                    # range along each sample ray
                pts = (un * rng_[..., None]) @ R[i].T                 # corridor frame [64, ns, 3]
                for q, (lo, hi) in enumerate(BOXES):
                    inb = ok & (np.abs(pts[..., 0]) <= .3) & (pts[..., 1] >= lo) & (pts[..., 1] <= hi) & (pts[..., 2] >= .3) & (pts[..., 2] <= 2.)
                    tof = bool(((inb.sum(1) >= .3 * np.maximum(ok.sum(1), 1)) & (ok.sum(1) >= 13)).any())
                    arm = f['arms']['chain']; c_all = c_kept = 0
                    lab = f['truth'][q]; kind = 'fa' if lab['n'] == 0 else 'tp' if (lab['n'] >= 150 and lab['z50'] <= 1.5) else None
                    if arm is not None:
                        for b in arm[q][1]:
                            if b[1] > 1.8:
                                continue
                            c_all += b[0]
                            pc = R[i].T @ np.array(b[2:5]); r = np.linalg.norm(pc); d = pc / r
                            k = int(np.argmax(ctr @ d)); veto = False
                            if np.degrees(np.arccos(np.clip(ctr[k] @ d, -1, 1))) <= 4. and ok[k].sum() >= 13:
                                veto = np.mean(np.abs(rng_[k][ok[k]] - r) <= .4) < .1
                            if kind:
                                vstat[t][f'{kind}_blobs'][0] += int(veto) * b[0]; vstat[t][f'{kind}_blobs'][1] += b[0]
                            c_kept += 0 if veto else b[0]
                    A['rgb'][i, q] = c_all >= 150; A['tof'][i, q] = tof
                    A['fused'][i, q] = tof or c_all >= 150; A['fused_veto'][i, q] = tof or c_kept >= 150
            for a, M in A.items():
                M = np.stack([V2.confirm(M[:, 0]), V2.confirm(M[:, 1])], 1); c = cnt[(t, a)]
                for i, f in enumerate(frames):
                    for q in (0, 1):
                        tr = f['truth'][q]
                        if tr['n'] == 0:
                            c['fa'][0] += int(M[i, q]); c['fa'][1] += 1
                        elif tr['n'] >= 150 and tr['z50'] <= 1.5:
                            for k in ('rec', 'rec_head' if q == 0 else 'rec_body'):
                                c[k][0] += int(M[i, q]); c[k][1] += 1
        print(seq.name, flush=True)
    res = {}
    for (t, a), c in cnt.items():
        res[f'tilt{t}/{a}'] = {k: dict(n=v, rate=round(v[0] / max(v[1], 1), 4)) for k, v in c.items()}
        r = res[f'tilt{t}/{a}']
        print(f"tilt {t:2d} {a:10s} FA {r['fa']['rate']:.4f} ({r['fa']['n'][0]}/{r['fa']['n'][1]}) rec {r['rec']['rate']:.4f} HEAD {r['rec_head']['rate']:.3f} BODY {r['rec_body']['rate']:.3f}")
    for t in TILTS:
        v = vstat[t]; res[f'tilt{t}/veto_px'] = v
        print(f"tilt {t:2d} veto: FA-blob px vetoed {v['fa_blobs'][0] / max(v['fa_blobs'][1], 1):.2f}, TP-blob px vetoed {v['tp_blobs'][0] / max(v['tp_blobs'][1], 1):.2f}")
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
