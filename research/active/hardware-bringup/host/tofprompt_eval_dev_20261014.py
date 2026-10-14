"""Evaluate the ToF-prompted net against the DAV2-S chain on held-out frames (plan: adt-eval-20261013/PLAN_TOFPROMPT.md).

For each held-out sequence and key (i0, d0, i30, d30) the ToF zones are emulated exactly as adt_tof_tilt_sweep_dev
(same GT sampling, same degraded seeds (sequence index in its prep dir, frame, tilt)), painted into the prompt, the
net's depth (320x240) is upsampled in log space to 640x480 and read out with the unchanged v6 box/blob rule. Baseline
blobs, ToF alarms and truth come from the sweep outputs of the same frames.
Usage: tofprompt_eval_dev_20261014.py <weights> <cache dir> <out json> (<prep dir> <sweep dir>)...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import tofprompt_train_dev_20261014 as T
from adt_backbone_blobs_dev_20261013 import box_masks
from adt_tof_tilt_sweep_dev_20261014 import blob_list, rx
import adt_tof_tilt_sweep_analysis_dev_20261014 as A

KEYS = ('i0', 'd0', 'i30', 'd30')


def zone_medians(depth_full, K, t, v, si, i, Rct, rays0):
    """Exactly the sweep's per-zone median range (nan if the zone gives no anchor)."""
    h, w = depth_full.shape; fx, fy, cx, cy = K
    rc = (rays0 @ rx(-t).T) @ Rct.T; un = rc / np.linalg.norm(rc, axis=-1, keepdims=True)
    zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
    zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h)
    zz = (depth_full / 1000.)[np.clip(zv, 0, h - 1), np.clip(zu, 0, w - 1)]; ok0 = zin & (zz > .1)
    rr = np.where(ok0, zz / np.maximum(un[..., 2], 1e-6), np.nan)
    if v == 'i':
        use = np.ones(64, bool); ret = ok0; cap = 4.
    else:
        g = np.random.default_rng([20261014, si, i, t])
        use = g.random(64) >= .25; rr = rr * (1 + .04 * g.standard_normal(64))[:, None]; cap = 3.
        ret = ok0 & use[:, None] & (np.nan_to_num(rr, nan=99.) < cap)
    nv = ok0.sum(1); m = np.full(64, np.nan)
    for zi in range(64):
        if use[zi] and ret[zi].sum() >= 13 and nv[zi] >= 13:
            v_ = float(np.median(rr[zi][ret[zi]]))
            if .3 < v_ < cap:
                m[zi] = v_
    return m


def main(weights, cache, out, *pairs):
    import torch
    import torch.nn.functional as Fn
    dev = 'cuda'; cache = Path(cache)
    I = json.loads((cache / 'index.json').read_text()); by = {s['seq']: s for s in I['seqs'] if s['split'] == 'test'}
    mm = {k: np.load(cache / f'{k}.npy', mmap_mode='r') for k in ('rgb', 'disp', 'floor')}
    m = T.net().to(dev); m.load_state_dict(torch.load(weights, map_location=dev)); m.eval()
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation']); rays0 = V1.X.zone_rays()
    res = {}
    for prep, sweep in zip(pairs[0::2], pairs[1::2]):
        names = sorted(p.name for p in Path(prep).iterdir() if (p / 'frames.npz').exists())
        seq_frames = []
        for si, seq in enumerate(names):
            if seq not in by:
                continue
            s = by[seq]; K = s['K']; meta = json.loads((Path(prep) / seq / 'meta.json').read_text()); w, h = meta['image_size']
            vv, uu = np.mgrid[0:h, 0:w]; dirs = np.stack([(uu - K[2]) / K[0], (vv - K[3]) / K[1], np.ones_like(uu, float)], -1).reshape(-1, 3)
            with np.load(Path(prep) / seq / 'frames.npz') as Z:
                depth_full, R = Z['depth'], Z['R_cc']
            frames = json.loads((Path(sweep) / f'{seq}.truth.json').read_text(encoding='utf8'))
            for f in frames:
                f['r'] = {}
            for key in KEYS:
                base = json.loads((Path(sweep) / f'{seq}.{key}.json').read_text(encoding='utf8'))
                v, t = key[0], int(key[1:])
                idx, zin, uz = T.zone_geometry(K, t)
                geo = (torch.tensor(idx, device=dev), torch.tensor(zin, device=dev), torch.tensor(uz, device=dev, dtype=torch.float32))
                for b0 in range(0, s['n'], 32):
                    b1 = min(b0 + 32, s['n']); sl = slice(s['start'] + b0, s['start'] + b1)
                    med = torch.tensor(np.array([zone_medians(depth_full[i], K, t, v, si, i, Rct, rays0) for i in range(b0, b1)]), device=dev, dtype=torch.float32)
                    inv, msk, _ = T.paint(med, geo, b1 - b0, dev)
                    rgb = torch.from_numpy(np.ascontiguousarray(mm['rgb'][sl])).to(dev)
                    disp = torch.from_numpy(np.ascontiguousarray(mm['disp'][sl])).to(dev).float()
                    floor = torch.from_numpy(np.ascontiguousarray(mm['floor'][sl])).to(dev).float()
                    x, lzb = T.inputs(rgb, disp, floor, inv, msk)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        lz = lzb[:, None] + m(x).float()
                    z = torch.exp(Fn.interpolate(lz.float(), (h, w), mode='bilinear', align_corners=False))[:, 0].cpu().numpy()
                    for j, i in enumerate(range(b0, b1)):
                        zz = z[j].reshape(-1).astype(np.float64)
                        pts = (R[i] @ (dirs * zz[:, None]).T).T; ok = np.isfinite(zz)
                        bl = [[[n_, round(z50, 3), 0] for n_, z50, _ in blob_list(mk, pts, h, w)] for mk in box_masks(pts, ok)]
                        frames[i]['r'][f'net_{key}'] = dict(src='net', tof=base[i]['tof'], blobs=bl)
                        frames[i]['r'][key] = base[i]
            seq_frames.append(frames); print('done', seq, flush=True)
        name = Path(prep).name; res[name] = {}
        for key in KEYS:
            for k in (key, f'net_{key}'):
                c = A.count(seq_frames, k, arms=('rgb', 'fused')); sw = A.rgb_sweep(seq_frames, k)
                res[name][k] = dict(count=c, sweep=sw)
                fs = lambda q: '  -  ' if q is None else f'{q[2]*100:4.1f}'
                print(f"{name:13s} {k:8s} RGB FA {c['rgb']['fa']['rate']*100:4.1f} ({c['rgb']['fa']['n'][0]}/{c['rgb']['fa']['n'][1]}) rec {c['rgb']['rec']['rate']*100:4.1f} | "
                      f"FA@rec>=.80/.85 {fs(sw['0.8'])}/{fs(sw['0.85'])} | fused FA {c['fused']['fa']['rate']*100:4.1f} rec {c['fused']['rec']['rate']*100:4.1f}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:])
