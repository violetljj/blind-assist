"""Fine-tune DAV2-Small on ADT ground truth, staying relative (plan: adt-eval-20261013/PLAN_FINETUNE.md).

Train the DPT head + last 4 ViT blocks (rest frozen), bf16, scale-shift-invariant L1 on inverse depth (median / mean
absolute deviation normalised per image; GT 0.3-10 m; pixels <= 3 m weighted x2), horizontal flips, 4 epochs.
Train families: clean, decoration, golden, meal, work; held out: multiuser. Data: prepared upright pinhole frames.
Train frames are stored once as uncompressed memmaps (RAM is ~15 GB); each batch is resized/normalised on the GPU.
Usage: adt_finetune_dav2_dev_20261013.py <prep dir> <out .pth>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import real_rgb_small_model_dev_20261012 as S

MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)


def load_frames(prep):
    """Uncompressed memmaps of the train frames (RAM is small); built once, reused."""
    seqs = [p for p in sorted(Path(prep).iterdir()) if (p / 'frames.npz').exists() and 'multiuser' not in p.name and 'sample' not in p.name]
    rp, dp = Path(prep) / 'train_rgb.npy', Path(prep) / 'train_depth.npy'
    if not rp.exists():
        counts = [json.loads((p / 'meta.json').read_text())['frames'] for p in seqs]
        R = np.lib.format.open_memmap(rp, 'w+', np.uint8, (sum(counts), 640, 480, 3))
        D = np.lib.format.open_memmap(dp, 'w+', np.uint16, (sum(counts), 640, 480)); k = 0
        for p in seqs:
            with np.load(p / 'frames.npz') as Z:
                r, d = Z['rgb'], Z['depth']
            R[k:k + len(r)] = r; D[k:k + len(r)] = d; k += len(r); del r, d
            print('memmap', p.name, k, flush=True)
        R.flush(); D.flush(); del R, D
    return np.load(rp, mmap_mode='r'), np.load(dp, mmap_mode='r')


def ssi(pred, gt_mm):
    import torch
    z = gt_mm.float() / 1000.
    m = (z > .3) & (z < 10)
    inv = torch.where(m, 1 / z.clamp(min=.3), torch.zeros_like(z))
    w = torch.where(z <= 3., 2., 1.) * m
    loss = 0.
    for p, g, mm, ww in zip(pred, inv, m, w):
        if mm.sum() < 500:
            continue
        def norm(x):
            v = x[mm]; med = v.median(); return (x - med) / (v - med).abs().mean().clamp(min=1e-6)
        loss = loss + ((norm(p) - norm(g)).abs() * ww).sum() / ww.sum()
    return loss / len(pred)


def main(prep, out):
    import torch
    start = time.perf_counter()
    import torch.nn.functional as Fn
    X, Y = load_frames(prep); n = len(X)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).cuda(); std = torch.tensor(STD).view(1, 3, 1, 1).cuda()
    nh, nw = int(round(640 * 518 / 480 / 14)) * 14, 518
    model, _ = S.load_dav2_small(); model.train()
    for p in model.parameters():
        p.requires_grad = False
    blocks = list(model.pretrained.blocks)[-4:]
    for p in model.depth_head.parameters():
        p.requires_grad = True
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW([{'params': model.depth_head.parameters(), 'lr': 2e-5},
                             {'params': [p for b in blocks for p in b.parameters()], 'lr': 5e-6}], weight_decay=1e-4)
    g = torch.Generator().manual_seed(20261013); log = []
    for ep in range(4):
        perm = torch.randperm(n, generator=g); tot = 0.
        for k in range(0, n, 4):
            i = np.sort(perm[k:k + 4].numpy())
            x = torch.from_numpy(np.ascontiguousarray(X[i])).cuda().permute(0, 3, 1, 2).float() / 255
            x = (Fn.interpolate(x, (nh, nw), mode='bicubic', align_corners=False) - mean) / std
            y = Fn.interpolate(torch.from_numpy(Y[i].astype(np.float32)).cuda()[:, None], (nh, nw), mode='nearest')[:, 0]
            if ep % 2:
                x, y = x.flip(-1), y.flip(-1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
            loss = ssi(pred.float(), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(i)
        log.append(dict(epoch=ep, loss=tot / n, wall=round(time.perf_counter() - start, 1))); print(log[-1], flush=True)
    model.eval()
    torch.save(model.state_dict(), out)
    Path(out).with_suffix('.json').write_text(json.dumps(dict(log=log, frames=n, plan='PLAN_FINETUNE.md'), indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
