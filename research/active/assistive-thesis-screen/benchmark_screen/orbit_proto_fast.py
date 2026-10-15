"""Faster ORBIT CLU-VE prototype pilot (same protocol and arms as orbit_proto_pilot.py, which was CPU-bound):
frames are decoded once in parallel into a uint8 cache shared by all backbones/arms, and every crop (box / grid) is
cut and resized on the GPU with torchvision roi_align in large fp16 batches. EXPLORE, 2026-10-15.
Arms: full (whole frame), box (annotated box x1.2 square - evaluator-only upper bound), grid (14 crops, per object max
cosine; no boxes). Usage: orbit_proto_fast.py <dir with selection.json, frames/> <out json> [backbones] [arms]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orbit_proto_pilot import BACKBONES, load, macs_per_frame  # noqa: E402

GRID = [(0., 0., 1.)] + [(x, y, .5) for y in (0, .25, .5) for x in (0, .25, .5)] + [(x, y, .7) for y in (0, .3) for x in (0, .3)]


def frame_cache(root, names):
    import cv2
    cache = root / 'frames_u8.npy'
    if cache.exists():
        arr = np.load(cache, mmap_mode='r')
        if arr.shape[0] == len(names):
            return np.ascontiguousarray(arr)
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        ims = list(ex.map(lambda f: cv2.imread(str(root / 'frames' / f))[:, :, ::-1], names))
    arr = np.stack(ims); np.save(cache, arr)
    print('decoded', arr.shape, round(time.time() - t0, 1), 's', flush=True)
    return arr


def embed_rois(m, size, mean, std, frames, rois, bs=256):
    """frames: uint8 [N,224,224,3] (CPU); rois: [M,5] float (frame idx, x0, y0, x1, y1 in pixels). Returns [M,D]."""
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    out = []
    for k in range(0, len(rois), bs):
        r = torch.as_tensor(rois[k:k + bs], dtype=torch.float32)
        idx = r[:, 0].long().unique()
        x = torch.from_numpy(frames[idx.numpy()]).cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255.)
        remap = torch.full((int(idx.max()) + 1,), -1, dtype=torch.long); remap[idx] = torch.arange(len(idx))
        rr = r.clone(); rr[:, 0] = remap[r[:, 0].long()].float()
        crops = roi_align(x, rr.cuda(), (size, size), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            f = m(((crops - mean) / std).contiguous(memory_format=torch.channels_last)).float()
        out.append(F.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def box_roi(i, box, W=224, margin=1.2):
    s = W / 1080.; cx, cy = (box['x'] + box['w'] / 2) * s, (box['y'] + box['h'] / 2) * s
    half = max(box['w'], box['h']) * s * margin / 2 + 1
    x0, x1 = max(0., cx - half), min(W, cx + half); y0, y1 = max(0., cy - half), min(W, cy + half)
    return [i, x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)]


def main(root, out, backbones='effb0_in1k,clip_b16,dinov2_s14,dinov2_b14', arms='full,box,grid'):
    import torch
    root = Path(root); sel = json.loads((root / 'selection.json').read_text())
    vids = [v for v in sel['videos'] if not v.get('skipped')]
    names = sorted({f for v in vids for f in v['frames']}); pos = {n: i for i, n in enumerate(names)}
    frames = frame_cache(root, names); W = frames.shape[1]
    arm_list = arms.split(',')
    # ROIs, built once: context frames (full), target frames per arm
    ctx_rows, tgt = [], []                      # tgt: (video, frame idx, box)
    for v in vids:
        for j, f in enumerate(v['frames']):
            if v['type'] == 'clean':
                ctx_rows.append((v['user'], v['obj'], pos[f]))
            else:
                tgt.append((v, pos[f], v['boxes'][j]))
    full_rois = np.array([[i, 0, 0, W, W] for i in range(len(names))], np.float32)
    res = {}
    for name in backbones.split(','):
        t0 = time.time()
        m, size, mean, std = load(name); m = m.to(memory_format=torch.channels_last)
        mpf = macs_per_frame(m, size); npar = sum(p.numel() for p in m.parameters())
        emb_full = embed_rois(m, size, mean, std, frames, full_rois)
        tgt_idx = np.array([t[1] for t in tgt])
        scores = {}
        protos = {}
        for user in sorted({r[0] for r in ctx_rows}):
            objs = sorted({r[1] for r in ctx_rows if r[0] == user})
            P = np.stack([emb_full[[r[2] for r in ctx_rows if r[0] == user and r[1] == o]].mean(0) for o in objs])
            protos[user] = (objs, P / np.linalg.norm(P, axis=1, keepdims=True))
        feats = {}
        if 'full' in arm_list:
            feats['full'] = emb_full[tgt_idx][:, None]
        if 'box' in arm_list:
            rois = np.array([box_roi(t[1], t[2], W) if t[2] else [t[1], 0, 0, W, W] for t in tgt], np.float32)
            feats['box'] = embed_rois(m, size, mean, std, frames, rois)[:, None]
        if 'grid' in arm_list:
            rois = np.array([[t[1], x * W, y * W, (x + s) * W, (y + s) * W] for t in tgt for x, y, s in GRID], np.float32)
            feats['grid'] = embed_rois(m, size, mean, std, frames, rois).reshape(len(tgt), len(GRID), -1)
        r = dict(params_M=round(npar / 1e6, 2), gmacs_per_frame=round(mpf / 1e9, 2), grid_crops=len(GRID),
                 tmacs_to_personalise=round(mpf * np.mean([sum(1 for c in ctx_rows if c[0] == u) for u in protos]) / 1e12, 3))
        for arm, F_ in feats.items():
            per_vid = {}
            for k, (v, _, _) in enumerate(tgt):
                objs, P = protos[v['user']]
                s = (F_[k] @ P.T).max(0)                                      # max over crops (1 crop for full/box)
                per_vid.setdefault(v['video'], (v['user'], []))[1].append(int(s.argmax() == objs.index(v['obj'])))
            accs = np.array([np.mean(x[1]) for x in per_vid.values()])
            users = {}
            for u, x in per_vid.values():
                users.setdefault(u, []).append(np.mean(x))
            r[arm] = dict(frame_acc=round(100 * accs.mean(), 2), ci95=round(100 * 1.96 * accs.std(ddof=1) / np.sqrt(len(accs)), 2),
                          n_videos=len(accs), per_user={u: round(100 * float(np.mean(x)), 1) for u, x in sorted(users.items())})
        r['gpu_wall_s'] = round(time.time() - t0, 1)
        res[name] = r
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'per_user'})
                                for k, v in r.items()}), flush=True)
        del m; torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
