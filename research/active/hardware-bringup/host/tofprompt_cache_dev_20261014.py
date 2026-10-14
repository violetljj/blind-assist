"""Half-resolution training cache for the ToF-prompted depth completion net (plan: adt-eval-20261013/PLAN_TOFPROMPT.md).

One GPU pass over the prepared ADT (real) and ASE (synthetic) sequences: DAV2-S disparity at full resolution, then every
2nd pixel (upright 640x480 -> 320x240) of RGB, disparity, GT depth (m) and the floor prior (inverse optical depth of the
gravity floor plane at the sequence's camera height, 0 above the horizon; on the device: IMU pitch/roll + calibrated
height). Writes uncompressed memmaps + index.json (sequence, source, split, frame offsets, intrinsics).
Split: ADT multiuser family and ASE scenes 16-19 held out; the ADT 10 s sample sequence is skipped.
Usage: tofprompt_cache_dev_20261014.py <adt prep> <ase prep> <out dir>
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


def seq_list(adt, ase):
    out = []
    for p in sorted(Path(adt).iterdir()):
        if (p / 'frames.npz').exists() and 'sample' not in p.name:
            out.append((p, 'adt', 'test' if 'multiuser' in p.name else 'train'))
    for p in sorted(Path(ase).iterdir(), key=lambda p: int(p.name.split('_')[1]) if p.name.startswith('ase_') else 99):
        if (p / 'frames.npz').exists():
            out.append((p, 'ase', 'test' if int(p.name.split('_')[1]) >= 16 else 'train'))
    return out


def height_of(depth, R, cam_y, dirs):
    hts = []
    for i in range(0, len(cam_y), max(1, len(cam_y) // 20)):
        z = depth[i].reshape(-1) / 1000.; ok = z > .1
        hts.append(cam_y[i] - (R[i] @ (dirs[ok] * z[ok, None]).T).T[:, 1])
    return float(np.median(cam_y) - np.percentile(np.concatenate(hts), 1))


def main(adt, ase, out):
    import torch
    import torch.nn.functional as Fn
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    seqs = seq_list(adt, ase); counts = [json.loads((p / 'meta.json').read_text())['frames'] for p, _, _ in seqs]
    N = sum(counts); H, W = 320, 240
    mm = {k: np.lib.format.open_memmap(out / f'{k}.npy', 'w+', dt, sh) for k, dt, sh in (
        ('rgb', np.uint8, (N, H, W, 3)), ('disp', np.float16, (N, H, W)), ('depth', np.float16, (N, H, W)), ('floor', np.float16, (N, H, W)))}
    model, _ = V1.S.load_dav2_small(); pool = ThreadPoolExecutor(2); start = time.perf_counter(); index = []; k0 = 0
    for (p, src, split), n_meta in zip(seqs, counts):
        meta = json.loads((p / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
        vv, uu = np.mgrid[0:h, 0:w]; dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        with np.load(p / 'frames.npz') as Z:
            rgb, depth, R, cam_y = Z['rgb'], Z['depth'], Z['R_cc'], Z['cam_y']
        n = len(rgb); height = height_of(depth, R, cam_y, dirs)
        def prep_t(i):
            t_, _ = model.image2tensor(np.ascontiguousarray(rgb[i][:, :, ::-1]), V1.SIZE); return t_
        pend = [pool.submit(prep_t, i) for i in range(min(3, n))]
        for i in range(n):
            x = pend.pop(0).result()
            if i + 3 < n:
                pend.append(pool.submit(prep_t, i + 3))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(x)
            disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            r = (dirs @ R[i].T)[:, 1].reshape(h, w)                              # corridor y (down) per unit-z ray
            mm['rgb'][k0 + i] = rgb[i][::2, ::2]; mm['disp'][k0 + i] = disp[::2, ::2]
            mm['depth'][k0 + i] = (depth[i][::2, ::2] / 1000.).astype(np.float16)
            mm['floor'][k0 + i] = np.where(r > .02, r / height, 0.)[::2, ::2]
        index.append(dict(seq=p.name, src=src, split=split, start=k0, n=n, K=[fx, fy, cx, cy], height=height))
        k0 += n; del rgb, depth
        print(src, p.name, n, f'{time.perf_counter()-start:.0f}s', flush=True)
    for v in mm.values():
        v.flush()
    (out / 'index.json').write_text(json.dumps(dict(N=k0, H=H, W=W, seqs=index), indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
