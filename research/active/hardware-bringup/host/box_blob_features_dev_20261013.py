"""Per-blob 3D shape of RGB in-box pixels on consumed captures (EXPLORE; pixel-level false-alarm rule work).

Reruns only the v3 RGB anchor chain (ToF strong is read from the frozen v3 outputs; the ToF model is not rerun):
fp16 DAV2-S, ToF fit -> robust replacement -> floor anchor -> last fit. For each HEAD/BODY box, the in-box pixel
mask is split into image-connected blobs; per blob we store pixel count, anchor provenance, and 3D extents
(p10-p90 of x, y, z in the corridor frame). Rules are then evaluated offline from blobs.json.
Usage: box_blob_features_dev_20261013.py <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import ROOT

W = ROOT / 'artifacts.local/work'
BATCHES = {
    'm12': ('phone-imu-capture-20261012/raw', 'fusion-v3-dev-20261013/v3_morning/frozen_fusion_eval_v3.json'),
    'a12': ('rgb-real-feasibility-20261012/raw', 'fusion-v3-dev-20261013/v3_afternoon/frozen_fusion_eval_v3.json'),
    'm13': ('frozen-eval-capture-20261013/raw', 'frozen-eval-capture-20261013/eval/frozen_fusion_eval_v3.json'),
    'a13': ('frozen-eval-capture-20261013b/raw', 'frozen-eval-capture-20261013b/eval/frozen_fusion_eval_v3.json'),
    'e13': ('frozen-eval-capture-20261013c/raw', 'frozen-eval-capture-20261013c/eval/frozen_fusion_eval_v3.json')}


def main(out):
    import cv2
    import torch
    import torch.nn.functional as Fn
    from atom_imu import load_mounting
    import frozen_fusion_eval_v1 as V1
    from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
    S, F, R, X = V1.S, V1.F, V1.R, V1.X
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model, _ = S.load_dav2_small(); imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = V1.AtomCamera(); K = cam.K_upright; rays = X.zone_rays(); w, h = cam.upright_size
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    pool = ThreadPoolExecutor(4); res = {}; start = time.perf_counter()
    for bname, (raw, rep) in BATCHES.items():
        rep = json.loads((W / rep).read_text(encoding='utf8')); height = rep['head_height_m']; rng = np.random.default_rng(20261013)
        for name in sorted(rep['sessions']):
            s = rep['sessions'][name]
            if s['cls'] not in ('quiet', 'ahead', 'side', 'head') or not s['frames']:
                continue
            d = W / raw / name; x = R.load(d, imu_rot)
            tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
            rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
            dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
            todo = []
            for k, f in enumerate(s['frames']):
                i = f['tof_index']; j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                if abs(rgb_t[j] - tof_t[i]) <= 1e8 and f['pose_ok']:
                    todo.append((k, i, j))
            ups = pool.map(lambda t: cam.undistort_upright(cv2.imread(str(d / rgb[t[2]]['file']))[:, :, ::-1]), todo)
            last = None; lastsrc = None; frames = []
            for (k, i, j), up in zip(todo, ups):
                t, _ = model.image2tensor(np.ascontiguousarray(up[:, :, ::-1]), V1.SIZE)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    dd = model(t)
                disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
                dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                inv = np.where(valid, 1 / (dist[i] * zfac), np.nan)
                fit = S.fit_affine(dz, inv); src = 'tof' if fit else None
                if not fit_quality(dz, inv, fit)['accepted']:
                    rb = robust_affine(dz, inv)['fit']
                    if rb is not None:
                        fit, src = rb, 'tof'
                Rp = F.rot(x['pitch'][i], x['roll'][i])
                if fit is None:
                    fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng); src = 'floor' if fit else None
                if fit is None and last is not None:
                    fit, src = last, lastsrc
                if fit is None:
                    continue
                last, lastsrc = fit, src
                a, b = fit
                invd = (disp.reshape(-1) - b) / a
                zz = np.where(invd > 1e-3, 1 / np.maximum(invd, 1e-3), np.inf)
                ok = np.isfinite(zz) & (up.reshape(-1, 3).sum(1) > 0)
                pts = np.full((len(zz), 3), np.nan); pts[ok] = (Rp @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                row = dict(k=k, src=src, boxes=[])
                for q, (lo, hi) in enumerate(F.BOXES.values()):
                    ins = ok & (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= V1.Z_MAX)
                    blobs = []
                    if ins.sum():
                        n, lab, stats, _ = cv2.connectedComponentsWithStats(ins.reshape(h, w).astype(np.uint8), connectivity=8)
                        labf = lab.reshape(-1)
                        for c in range(1, n):
                            m = labf == c
                            if m.sum() < 10:
                                continue
                            p = pts[m]; pr = lambda v: float(np.percentile(v, 90) - np.percentile(v, 10))
                            blobs.append(dict(px=int(m.sum()), xr=pr(p[:, 0]), yr=pr(p[:, 1]), zr=pr(p[:, 2]),
                                              z50=float(np.median(p[:, 2])), x50=float(np.median(p[:, 0])),
                                              bbox=[int(v) for v in stats[c, :4]]))
                    row['boxes'].append(dict(total=int(ins.sum()), blobs=blobs))
                frames.append(row)
            res[f'{bname}/{name}'] = dict(cls=s['cls'], n=len(s['frames']), frames=frames,
                                          tof=[f['tof_strong'] for f in s['frames']], v3_rgb=[f['rgb'] for f in s['frames']])
            print(bname, name[17:23], s['cls'], len(frames), f'{time.perf_counter()-start:.0f}s', flush=True)
    (out / 'blobs.json').write_text(json.dumps(res), encoding='utf8')


if __name__ == '__main__':
    main(sys.argv[1])
