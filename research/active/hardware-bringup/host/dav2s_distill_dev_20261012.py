"""Distil UniDepthV2-L into DAV2-Small on Atom frames (EXPLORE).

Stage 'labels': calibrated, undistorted upright Atom frames (every 4th RGB frame) from the morning
batch (27 sessions) and the afternoon office batch (9 sessions); the afternoon hall H0-H8 sessions
are held out for development evaluation and the 2026-10-13 frozen-test captures are never used.
UniDepthV2-L (known K) depth -> inverse-depth pseudo labels at 518 x 686.
Stage 'train': fine-tune official DAV2-Small (relative) with the scale-and-shift-invariant loss in
disparity space (MiDaS-style, per image median/MAD alignment) + multiscale gradient matching;
AdamW lr 2e-6 encoder / 2e-5 decoder, 3 epochs, batch 2 (batch 4 spilled 8 GB VRAM), bf16, horizontal flips.
Output: artifacts.local/work/dav2s-distill-20261012/dav2s_distilled.pth (plain DepthAnythingV2 state).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import AtomCamera, ROOT

OUT = ROOT / 'artifacts.local/work/dav2s-distill-20261012'
SOURCES = [ROOT / 'artifacts.local/work/phone-imu-capture-20261012/raw', ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012/raw']
HELD_OUT = ('072853', '072939', '073020', '073138', '073245', '073554', '073635', '073740', '073821')   # afternoon hall, UTC
H, W, STEP = 686, 518, 4
UPSTREAM = ROOT / 'artifacts.local/work/rgb-depth-backbone-compare-dev-20261010/upstream/Depth-Anything-V2'
WEIGHT = ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth'
MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)


def frames():
    for root in SOURCES:
        for d in sorted(root.iterdir()):
            if d.name[17:23] in HELD_OUT or not (d / 'rgb').is_dir():
                continue
            for f in sorted((d / 'rgb').glob('*.jpg'))[::STEP]:
                yield d.name, f


def labels():
    import torch
    import real_rgb_query_feasibility_dev_20261012 as F
    cam = AtomCamera(); K = torch.from_numpy(cam.K_upright).float().cuda()
    model = F.load_model()
    items = list(frames()); OUT.mkdir(parents=True, exist_ok=True)
    img = np.lib.format.open_memmap(OUT / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(items), H, W, 3))
    tgt = np.lib.format.open_memmap(OUT / 'inv_depth.npy', mode='w+', dtype=np.float16, shape=(len(items), H, W))
    began = time.monotonic()
    for k, (sess, f) in enumerate(items):
        up = cam.undistort_upright(cv2.imread(str(f))[:, :, ::-1])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            depth = model.infer(torch.from_numpy(up.transpose(2, 0, 1).copy()).cuda(), K)['depth'][0, 0].float().cpu().numpy()
        valid = (up.sum(-1) > 0) & (depth > 0)
        inv = np.where(valid, 1 / np.maximum(depth, 1e-3), 0)                       # 0 = invalid (undistortion border)
        img[k] = cv2.resize(up, (W, H), interpolation=cv2.INTER_AREA)
        tgt[k] = cv2.resize(inv, (W, H), interpolation=cv2.INTER_NEAREST)
        if k % 500 == 0:
            print('LABELS', k, len(items), round(time.monotonic() - began, 1), flush=True)
    img.flush(); tgt.flush()
    (OUT / 'labels_index.json').write_text(json.dumps([dict(session=s, file=f.name) for s, f in items]), encoding='utf8')
    return len(items)


def ssi(pred, target, mask):
    """Scale-and-shift-invariant L1 after per-image median/MAD normalisation (MiDaS)."""
    def norm(x):
        out = []
        for i in range(len(x)):
            v = x[i][mask[i]]
            t = v.median(); s = (v - t).abs().mean().clamp_min(1e-6)
            out.append((x[i] - t) / s)
        return torch.stack(out)
    import torch
    p, t = norm(pred), norm(target)
    l1 = ((p - t).abs() * mask).sum() / mask.sum().clamp_min(1)
    grad = 0.
    for s in range(4):
        step = 2 ** s
        d = (p - t)[:, ::step, ::step]; m = mask[:, ::step, ::step].float()
        gx = (d[:, :, 1:] - d[:, :, :-1]).abs() * m[:, :, 1:] * m[:, :, :-1]
        gy = (d[:, 1:, :] - d[:, :-1, :]).abs() * m[:, 1:, :] * m[:, :-1, :]
        grad = grad + (gx.sum() + gy.sum()) / m.sum().clamp_min(1)
    return l1 + .5 * grad


def train(epochs=3, batch=2):
    global torch
    import torch
    sys.path.insert(0, str(UPSTREAM))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True)); model = model.cuda().train()
    enc = [p for n, p in model.named_parameters() if n.startswith('pretrained')]
    dec = [p for n, p in model.named_parameters() if not n.startswith('pretrained')]
    opt = torch.optim.AdamW([dict(params=enc, lr=2e-6), dict(params=dec, lr=2e-5)], weight_decay=1e-4)
    img = np.load(OUT / 'images.npy', mmap_mode='r'); tgt = np.load(OUT / 'inv_depth.npy', mmap_mode='r')
    n = len(img); rng = np.random.default_rng(20261012); history = []
    steps = epochs * (n // batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-6, 2e-5], total_steps=steps, pct_start=.1)
    mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1); std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    began = time.monotonic(); step = 0
    for epoch in range(epochs):
        order = rng.permutation(n); total = 0.
        for b in range(0, n - batch + 1, batch):
            ix = np.sort(order[b:b + batch])
            x = torch.from_numpy(np.array(img[ix])).cuda().permute(0, 3, 1, 2).float() / 255
            y = torch.from_numpy(np.array(tgt[ix]).astype(np.float32)).cuda()
            if rng.random() < .5:
                x = x.flip(-1); y = y.flip(-1)
            x = (x - mean) / std; mask = y > 0
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
            loss = ssi(pred.float(), y, mask)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step(); step += 1; total += float(loss.detach())
            if step % 200 == 0:
                print('STEP', step, steps, round(total / (b // batch + 1), 4), round((time.monotonic() - began) / step, 3), 's/step', flush=True)
        history.append(dict(epoch=epoch + 1, loss=total / (n // batch), seconds=time.monotonic() - began))
        print('TRAIN', history[-1], flush=True)
    torch.save(model.state_dict(), OUT / 'dav2s_distilled.pth')
    (OUT / 'train_receipt.json').write_text(json.dumps(dict(images=n, epochs=epochs, batch=batch, steps=step, history=history,
                                                            held_out_sessions_utc=HELD_OUT), indent=1), encoding='utf8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('labels', 'train'))
    a = ap.parse_args()
    print(labels() if a.stage == 'labels' else train())
