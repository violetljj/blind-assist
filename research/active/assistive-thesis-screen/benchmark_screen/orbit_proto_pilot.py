"""ORBIT CLU-VE pilot on validation users: training-free prototype recognizers with frozen backbones, plus a
ground-truth-box crop arm as the localization headroom signal. EXPLORE, 2026-10-15; plan: benchmark_screen/PLAN.md.

For each user: prototype per object = mean of L2-normalised embeddings of that user's clean-video context frames
(every 30th frame, from fetch_orbit_frames.py select); each selected clutter frame (object present, <= 200 per video)
is classified among all of the user's objects by cosine similarity. Frame accuracy is averaged per video, then over
videos (95% CI = 1.96 sd / sqrt(n)); per-user means are reported too. Arms:
  full   whole 224x224 frame
  box    crop to the annotated object box (x1.2 margin, square, resized to the backbone input) - uses evaluator-only
         boxes, so it is a localization upper bound, not a method.
MACs per frame are counted with torch.utils.flop_counter (FLOPs / 2); "MACs to personalise" = per-frame MACs x the
user's context frames, averaged over users.
Usage: orbit_proto_pilot.py <dir with selection.json and frames/> <out json> [backbones]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BACKBONES = {
    'effb0_in1k': ('efficientnet_b0.ra_in1k', 224),
    'clip_b16': ('vit_base_patch16_clip_224.openai', 224),
    'dinov2_s14': ('vit_small_patch14_dinov2.lvd142m', 224),
    'dinov2_b14': ('vit_base_patch14_dinov2.lvd142m', 224),
    'dinov2_s14_168': ('vit_small_patch14_dinov2.lvd142m', 168),     # efficiency variant: 144 instead of 256 tokens
}


def load(name):
    import timm
    import torch
    arch, size = BACKBONES[name]
    kw = dict(img_size=size) if 'dinov2' in arch else {}
    m = timm.create_model(arch, pretrained=True, num_classes=0, **kw).eval().cuda()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean, std = torch.tensor(cfg['mean']).view(1, 3, 1, 1).cuda(), torch.tensor(cfg['std']).view(1, 3, 1, 1).cuda()
    return m, size, mean, std


def macs_per_frame(m, size):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, 3, size, size).cuda()
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 2


def embed(m, size, mean, std, imgs, bs=64):
    import cv2
    import torch
    import torch.nn.functional as F
    out = []
    for k in range(0, len(imgs), bs):
        batch = [i if i.shape[:2] == (size, size) else
                 cv2.resize(np.ascontiguousarray(i), (size, size), interpolation=cv2.INTER_AREA if i.shape[0] > size else cv2.INTER_LINEAR)
                 for i in imgs[k:k + bs]]
        x = torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).float() / 255.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            f = m((x - mean) / std).float()
        out.append(F.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def crop(img, box, margin=1.2):
    h, w = img.shape[:2]; s = w / 1080.                                  # ORBIT boxes are in 1080x1080 coordinates
    cx, cy = (box['x'] + box['w'] / 2) * s, (box['y'] + box['h'] / 2) * s
    half = max(box['w'], box['h']) * s * margin / 2 + 1
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half)); y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    return img[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]


def grid_scores(m, size, mean, std, imgs, protos, bs=16):
    """Post-hoc arm 'grid' (no boxes): 14 crops per frame (full; 3x3 at 50 % side, stride 25 %; 2x2 at 70 %, stride 30 %),
    per object the max cosine over crops. Crops are cut and resized on the GPU."""
    import torch
    import torch.nn.functional as F
    boxes = [(0., 0., 1.)] + [(x, y, .5) for y in (0, .25, .5) for x in (0, .25, .5)] + [(x, y, .7) for y in (0, .3) for x in (0, .3)]
    P = torch.from_numpy(protos).cuda().float(); out = []
    for k in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[k:k + bs])).cuda().permute(0, 3, 1, 2).float() / 255.
        H = x.shape[-1]
        crops = torch.cat([F.interpolate(x[:, :, int(y * H):int((y + s) * H), int(cx * H):int((cx + s) * H)], (size, size),
                                         mode='bilinear', align_corners=False, antialias=True) for cx, y, s in boxes])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            f = F.normalize(m((crops - mean) / std).float(), dim=1)
        s_ = (f @ P.T).view(len(boxes), x.shape[0], -1).amax(0)
        out.append(s_.cpu().numpy())
    return np.concatenate(out), len(boxes)


def main(root, out, backbones='effb0_in1k,clip_b16,dinov2_s14,dinov2_b14', arms='full,box'):
    import cv2
    import torch
    root = Path(root); sel = json.loads((root / 'selection.json').read_text())
    vids = [v for v in sel['videos'] if not v.get('skipped')]
    read = lambda f: cv2.imread(str(root / 'frames' / f))[:, :, ::-1]
    res = {}
    for name in backbones.split(','):
        m, size, mean, std = load(name)
        mpf = macs_per_frame(m, size); npar = sum(p.numel() for p in m.parameters())
        arm_list = arms.split(',')
        accs = {a: [] for a in arm_list}; per_user = {a: {} for a in arm_list}; ctx_frames = {}; ncrop = 1
        for user in sorted({v['user'] for v in vids}):
            uv = [v for v in vids if v['user'] == user]
            objs = sorted({v['obj'] for v in uv})
            ctx = [v for v in uv if v['type'] == 'clean']
            ctx_frames[user] = sum(len(v['frames']) for v in ctx)
            protos = np.stack([embed(m, size, mean, std, [read(f) for v in ctx if v['obj'] == o for f in v['frames']]).mean(0)
                               for o in objs])
            protos /= np.linalg.norm(protos, axis=1, keepdims=True)
            for v in (v for v in uv if v['type'] == 'clutter'):
                imgs = [read(f) for f in v['frames']]
                y = objs.index(v['obj'])
                for arm in arm_list:
                    if arm == 'grid':
                        sc, ncrop = grid_scores(m, size, mean, std, imgs, protos)
                    else:
                        ims = imgs if arm == 'full' else [crop(i, b) if b else i for i, b in zip(imgs, v['boxes'])]
                        sc = embed(m, size, mean, std, ims) @ protos.T
                    acc = float(np.mean(sc.argmax(1) == y))
                    accs[arm].append(acc); per_user[arm].setdefault(user, []).append(acc)
        r = dict(params_M=round(npar / 1e6, 2), gmacs_per_frame=round(mpf / 1e9, 2),
                 tmacs_to_personalise=round(mpf * float(np.mean(list(ctx_frames.values()))) / 1e12, 3),
                 grid_crops=ncrop, n_videos=len(accs[arm_list[0]]), n_users=len(per_user[arm_list[0]]))
        for arm, a in accs.items():
            a = np.array(a)
            r[arm] = dict(frame_acc=round(100 * a.mean(), 2), ci95=round(100 * 1.96 * a.std(ddof=1) / np.sqrt(len(a)), 2),
                          per_user={u: round(100 * float(np.mean(x)), 1) for u, x in per_user[arm].items()})
        res[name] = r
        print(name, json.dumps(r), flush=True)
        del m; torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
