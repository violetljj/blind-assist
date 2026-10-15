"""ORBIT CLU-VE evaluation with the official protocol details, plus an annotation-free personal-localization method.
EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md. Tune on validation users only; run test users once.

Protocol (microsoft/ORBIT-Dataset data/datasets.py, README): each video is capped at its first 1000 frames; context =
every 30th frame of every clean video (uniform, from frame 0); target candidates = clutter frames with the object
present; clutter videos with < 50 such frames are dropped; each of `tasks` tasks draws 200 random candidates per video;
frame accuracy per (task, video), mean and 1.96*std/sqrt(n) over the flattened list.

Arms (frozen backbone, no training):
  full   cosine to mean CLS prototypes of each object (whole frame)
  box    same on the annotated object box (evaluator-only upper bound)
  part   personal part memory: context patch tokens of an object that are much closer to that object's other frames
         than to any other object of the same user (removes shared tables/hands/backgrounds - no annotation), k-means
         into part centroids; frame score = mean of the top-k patch similarities to the object's parts
  loc    localize with the part heat map of the best object, crop (roi_align on GPU), CLS of the crop; score = full + crop
  fuse   full + part (z-scored per frame)
Usage: orbit_eval.py <split dir> <spec: selection.json | plan.json> <out json> [backbone=dinov2_s14] [tasks=50] [cfg json]
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orbit_proto_pilot import load, macs_per_frame  # noqa: E402

CFG = dict(keep=0.3, parts=32, topk=8, delta=0.12, expand=1.6, min_side=4)


def users_from_spec(spec):
    """-> {user: {'objects': [...], 'context': {obj: [frames]}, 'clutter': [(video, obj, [cand frames], {frame: box})]}}"""
    U = {}
    for v in spec['videos']:
        if v.get('skipped'):
            continue
        u = U.setdefault(v['user'], dict(context={}, clutter=[]))
        if v['type'] == 'clean':
            frames = v['context'] if 'context' in v and v['context'] is not None else v['frames']
            allf = v.get('n_frames')
            # official cap: first 1000 frames of the video (frame numbers are 1-based in the names)
            frames = [f for f in frames if int(f.rsplit('-', 1)[1].split('.')[0]) <= 1000] if allf else frames
            u['context'].setdefault(v['obj'], []).extend(frames)
        else:
            if 'valid' in v and v['valid'] is not None:                     # test plan: all frames available
                cand = [f for f in v['valid'] if int(f.rsplit('-', 1)[1].split('.')[0]) <= 1000]
                boxes = v['boxes']
            else:                                                            # validation selection: fixed 200
                cand, boxes = v['frames'], dict(zip(v['frames'], v['boxes']))
            if len(cand) >= 50:
                u['clutter'].append((v['video'], v['obj'], cand, boxes))
    for u in U.values():
        u['objects'] = sorted(u['context'])
    return U


def read_many(frames_dir, names):
    import cv2
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(lambda f: cv2.imread(str(frames_dir / f))[:, :, ::-1], names)))


def forward(m, size, mean, std, x_u8, rois=None, bs=192):
    """x_u8: uint8 [N,H,W,3] CPU. rois: optional [M,5] (frame idx, x0,y0,x1,y1) px. Returns (cls [M,D], patches [M,T,D])."""
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    cls_out = pat_out = None                      # preallocated on the first batch: no second copy at the end
    n = len(rois) if rois is not None else len(x_u8)
    for k in range(0, n, bs):
        if rois is None:
            x = (x_u8[k:k + bs].float().div(255.) if torch.is_tensor(x_u8)          # nvJPEG frames: CUDA NCHW uint8
                 else torch.from_numpy(x_u8[k:k + bs]).cuda().permute(0, 3, 1, 2).float().div_(255.))
            if x.shape[-1] != size:
                x = F.interpolate(x, (size, size), mode='bilinear', align_corners=False, antialias=True)
        else:
            r = torch.as_tensor(rois[k:k + bs], dtype=torch.float32)
            idx = r[:, 0].long().unique(); remap = torch.full((int(idx.max()) + 1,), -1, dtype=torch.long)
            remap[idx] = torch.arange(len(idx)); r[:, 0] = remap[r[:, 0].long()].float()
            src = torch.from_numpy(x_u8[idx.numpy()]).cuda().permute(0, 3, 1, 2).float().div_(255.)
            x = roi_align(src, r.cuda(), (size, size), 1.0, 2, True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            t = m.forward_features(((x - mean) / std).contiguous(memory_format=torch.channels_last))
        t = F.normalize(t.float(), dim=-1)
        npre = getattr(m, 'num_prefix_tokens', 1)
        if cls_out is None:
            cls_out = torch.empty(n, t.shape[-1], device=t.device)
            pat_out = torch.empty(n, t.shape[1] - npre, t.shape[-1], device=t.device, dtype=torch.half)
        cls_out[k:k + len(t)] = t[:, 0]; pat_out[k:k + len(t)] = t[:, npre:].half(); del t
    return cls_out, pat_out


def kmeans(x, k, iters=15, seed=0):
    import torch
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(len(c)):
            s = x[a == j]
            if len(s):
                c[j] = torch.nn.functional.normalize(s.mean(0), dim=0)
    return c


def part_memory(pat_by_obj, cfg):
    """pat_by_obj: list of [n_o, T, D] patch tensors (normalised). Returns list of [parts, D] centroids."""
    import torch
    flat = [p.reshape(-1, p.shape[-1]).float() for p in pat_by_obj]
    mem = []
    for o, P in enumerate(flat):
        others = torch.cat([q for j, q in enumerate(flat) if j != o])
        g = torch.Generator(device=P.device).manual_seed(o)
        others = others[torch.randperm(len(others), generator=g, device=P.device)[:30000]]
        own = P[torch.randperm(len(P), generator=g, device=P.device)[:30000]]
        s_other = torch.cat([(P[i:i + 4096] @ others.T).amax(1) for i in range(0, len(P), 4096)])
        s_own = torch.cat([(P[i:i + 4096] @ own.T).topk(6, dim=1).values[:, 1:].mean(1) for i in range(0, len(P), 4096)])
        d = s_own - s_other
        keep = P[d >= d.quantile(1 - cfg['keep'])]
        mem.append(kmeans(keep, min(cfg['parts'], len(keep))))
    return mem


def evaluate(root, spec_path, out, backbone='dinov2_s14', tasks=50, cfg=None, arms='full,box,part,fuse,loc'):
    """cfg: JSON object, or a JSON list of objects to sweep the part-memory settings on shared features (part/fuse
    arms are reported per config as part@i / fuse@i; full/box/loc use the first config)."""
    import torch
    root = Path(root); tasks = int(tasks); want = arms.split(',')
    cfgs = json.loads(cfg) if cfg else [{}]
    cfgs = [dict(CFG, **c) for c in (cfgs if isinstance(cfgs, list) else [cfgs])]; cfg = cfgs[0]
    U = users_from_spec(json.loads(Path(spec_path).read_text()))
    m, size, mean, std = load(backbone); m = m.to(memory_format=torch.channels_last)
    mpf = macs_per_frame(m, size); grid = size // 14 if 'dinov2' in backbone else None
    arms = [a for a in ['full', 'box', 'loc'] if a in want] + [f'{a}@{j}' for j in range(len(cfgs)) for a in ('part', 'fuse') if a in want]
    per_frame = {}                                   # (video) -> {arm: bool array over candidates}
    ctx_counts, t0 = [], time.time()
    for user, u in sorted(U.items()):
        objs = u['objects']; cs = int(cfg.get('ctx_stride', 1))       # ctx_stride 2 = every 60th frame
        ctx_names = [f for o in objs for f in u['context'][o][::cs]]
        ctx_lab = np.array([i for i, o in enumerate(objs) for _ in u['context'][o][::cs]])
        ctx_counts.append(len(ctx_names))
        xc = read_many(root / 'frames', ctx_names)
        ccls, cpat = forward(m, size, mean, std, xc)
        lab = torch.as_tensor(ctx_lab).cuda()
        protos = torch.nn.functional.normalize(torch.stack([ccls[lab == i].mean(0) for i in range(len(objs))]), dim=1)
        mems = []
        for c in cfgs:
            mem = part_memory([cpat[lab == i] for i in range(len(objs))], c)
            mems.append((torch.cat(mem).half(), torch.cat([torch.full((len(q),), i) for i, q in enumerate(mem)]).cuda()))
        del xc, cpat
        for video, obj, cand, boxes in u['clutter']:
            y = objs.index(obj)
            x = read_many(root / 'frames', cand)
            fcls, fpat = forward(m, size, mean, std, x)
            s_full = fcls @ protos.T                                         # [N, O]
            z = lambda s: (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
            out_ = {'full': s_full}
            for j, (allp, owner) in enumerate(mems):
                sim = (fpat @ allp.T).float()                                # [N, T, C]
                per_obj = torch.stack([sim[:, :, owner == i].amax(2) for i in range(len(objs))], 2)   # [N, T, O]
                s_part = per_obj.topk(cfgs[j]['topk'], dim=1).values.mean(1)  # [N, O]
                out_[f'part@{j}'] = s_part; out_[f'fuse@{j}'] = z(s_full) + z(s_part)
                if j == 0:
                    per_obj0, s_part0 = per_obj, s_part
                del sim
            W = x.shape[1]
            if 'loc' in want:                                                 # localize from the best object's part map
                best = s_part0.argmax(1)
                hm = per_obj0[torch.arange(len(x)), :, best].reshape(len(x), grid, grid)
                rois = []
                for i in range(len(x)):
                    h = hm[i]; thr = h.max() - cfg['delta']; ys, xs = torch.nonzero(h >= thr, as_tuple=True)
                    cy, cx = (ys.float().mean() + .5), (xs.float().mean() + .5)
                    half = max(float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1), cfg['min_side']) * cfg['expand'] / 2
                    px = W / grid
                    rois.append([i, max(0., (float(cx) - half) * px), max(0., (float(cy) - half) * px),
                                 min(W, (float(cx) + half) * px), min(W, (float(cy) + half) * px)])
                lcls, _ = forward(m, size, mean, std, x, np.array(rois, np.float32))
                out_['loc'] = s_full + lcls @ protos.T
            if 'box' in want:                                                 # evaluator-only oracle box
                s_ = W / 1080.; brois = []
                for i, f in enumerate(cand):
                    b = boxes.get(f) if isinstance(boxes, dict) else None
                    if b:
                        cx_, cy_ = (b['x'] + b['w'] / 2) * s_, (b['y'] + b['h'] / 2) * s_; hb = max(b['w'], b['h']) * s_ * 1.2 / 2 + 1
                        brois.append([i, max(0, cx_ - hb), max(0, cy_ - hb), min(W, cx_ + hb), min(W, cy_ + hb)])
                    else:
                        brois.append([i, 0, 0, W, W])
                bcls, _ = forward(m, size, mean, std, x, np.array(brois, np.float32))
                out_['box'] = bcls @ protos.T
            per_frame[video] = (user, {a: (s.argmax(1) == y).cpu().numpy() for a, s in out_.items()})
        print(user, len(objs), 'objects,', len(u['clutter']), 'clutter videos,', round(time.time() - t0), 's', flush=True)
    rng = random.Random(0); res = dict(backbone=backbone, cfgs=cfgs, tasks=tasks, n_videos=len(per_frame),
                                       gmacs_per_frame=round(mpf / 1e9, 2),
                                       tmacs_to_personalise=round(mpf * float(np.mean(ctx_counts)) / 1e12, 3),
                                       extra_forward_per_frame=dict(full=0, part=0, fuse=0, loc=1, box=1))
    samples = {v: [sorted(rng.sample(range(len(next(iter(d.values())))), min(200, len(next(iter(d.values()))))))
                   for _ in range(tasks)] for v, (_, d) in per_frame.items()}
    for a in arms:
        acc = np.array([d[a][idx].mean() for v, (_, d) in per_frame.items() for idx in samples[v]])
        users = {}
        for v, (u, d) in per_frame.items():
            users.setdefault(u, []).extend(d[a][idx].mean() for idx in samples[v])
        res[a] = dict(frame_acc=round(100 * acc.mean(), 2), ci95=round(100 * 1.96 * acc.std() / np.sqrt(len(acc)), 2),
                      per_user={u: round(100 * float(np.mean(x)), 1) for u, x in sorted(users.items())})
    res['wall_s'] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) or 'frame_acc' not in v else {'acc': v['frame_acc'], 'ci': v['ci95']})
                      for k, v in res.items()}), flush=True)


if __name__ == '__main__':
    evaluate(*sys.argv[1:])
