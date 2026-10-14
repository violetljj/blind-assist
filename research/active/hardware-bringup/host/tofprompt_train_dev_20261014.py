"""Train the ToF-prompted depth completion net (plan: adt-eval-20261013/PLAN_TOFPROMPT.md).

Inputs at 320x240 (upright): RGB, DAV2-S disparity (per-image robust normalisation), the ToF prompt (each valid zone's
median range painted at its 25 sample pixels as inverse optical depth, plus a mask), the floor prior, and log z_base,
where z_base is the per-image least-squares affine of disparity to the painted ToF inverse depth (constant 2 m when
fewer than 3 zones are valid or the slope is not positive). Output: log z = log z_base + residual (the residual is computed at 160x120 and upsampled). Every 2nd training frame
lives on the GPU (the full split does not fit in 8 GB); ToF prompts are emulated per batch (extra down-tilt uniform over 0-35 deg in 5 deg steps;
half of the samples degraded: 25 % zones dropped, no return beyond 3 m, 4 % per-zone range error). Loss: L1 on log
depth for GT 0.3-10 m, weight 2 for GT <= 3 m. Augmentation: horizontal mirror of the scene, mild brightness/contrast.
Usage: tofprompt_train_dev_20261014.py <cache dir> <out .pth> [epochs]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1

TILTS = (0, 5, 10, 15, 20, 25, 30, 35)
H, W = 320, 240


def rx(d):
    a = np.radians(d); return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def zone_geometry(K, tilt, full_hw=(640, 480)):
    """Half-resolution flat pixel index, in-image mask and optical-z cosine of the 64x25 zone sample rays."""
    fx, fy, cx, cy = K; h, w = full_hw
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    rc = (V1.X.zone_rays() @ rx(-tilt).T) @ Rct.T; un = rc / np.linalg.norm(rc, axis=-1, keepdims=True)
    zu = np.round(fx * rc[..., 0] / rc[..., 2] + cx).astype(int); zv = np.round(fy * rc[..., 1] / rc[..., 2] + cy).astype(int)
    zin = (rc[..., 2] > .05) & (zu >= 0) & (zu < w) & (zv >= 0) & (zv < h)
    idx = (np.clip(zv, 0, h - 1) // 2) * (w // 2) + np.clip(zu, 0, w - 1) // 2
    return idx, zin, un[..., 2]


def net():
    import torch
    import torch.nn as nn

    def block(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())

    class UNet(nn.Module):
        def __init__(self, cin=8, ch=(32, 64, 96, 128, 160)):
            super().__init__()
            self.enc = nn.ModuleList([block(cin, ch[0], 2)] + [block(ch[k], ch[k + 1], 2) for k in range(len(ch) - 1)])   # stem at 1/2 res
            self.glob = nn.Sequential(nn.Linear(ch[-1], ch[-1]), nn.SiLU())
            self.dec = nn.ModuleList([block(ch[k + 1] + ch[k], ch[k]) for k in reversed(range(len(ch) - 1))])
            self.head = nn.Conv2d(ch[0], 1, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, x):
            feats = []
            for e in self.enc:
                x = e(x); feats.append(x)
            x = x + self.glob(x.mean((2, 3)))[:, :, None, None]           # global context (scale) at the bottleneck
            for d, f in zip(self.dec, reversed(feats[:-1])):
                x = d(torch.cat([nn.functional.interpolate(x, size=f.shape[-2:], mode='bilinear', align_corners=False), f], 1))
            return nn.functional.interpolate(1.5 * torch.tanh(self.head(x)), scale_factor=2, mode='bilinear', align_corners=False)
    return UNet()


def prompt(depth, geo, degrade, gen):
    """depth [B,H,W] (m) on GPU; geo = (idx [64,25] long, zin [64,25] bool, uz [64,25]) tensors; degrade [B] bool.
    Returns tof inverse-depth map [B,1,H,W], mask [B,1,H,W], zone medians [B,64] (nan if invalid)."""
    import torch
    idx, zin, uz = geo; B = depth.shape[0]
    z = depth.reshape(B, -1)[:, idx.reshape(-1)].reshape(B, 64, -1)
    valid = zin[None] & (z > .1)
    r = z / uz[None]
    drop = (torch.rand(B, 64, generator=gen, device=depth.device) < .25) & degrade[:, None]
    noise = 1 + .04 * torch.randn(B, 64, generator=gen, device=depth.device) * degrade[:, None]
    r = r * noise[..., None]
    cap = torch.where(degrade, 3., 4.)[:, None, None]
    ret = valid & ~drop[..., None] & (r < cap)
    m = torch.nanmedian(torch.where(ret, r, torch.nan), dim=2).values
    ok = (ret.sum(2) >= 13) & (valid.sum(2) >= 13) & ~drop & (m > .3) & (m < cap[:, :, 0])
    m = torch.where(ok, m, torch.nan)
    return paint(m, geo, B, depth.device)


def paint(m, geo, B, device):
    import torch
    idx, zin, uz = geo
    val = 1 / (m[..., None] * uz[None])                                  # inverse optical depth per sample pixel
    use = torch.isfinite(val) & zin[None]
    inv = torch.zeros(B, H * W, device=device); msk = torch.zeros(B, H * W, device=device)
    bi = torch.arange(B, device=device)[:, None, None].expand_as(val)
    pi = idx[None].expand_as(val)
    inv[bi[use], pi[use]] = val[use].float(); msk[bi[use], pi[use]] = 1.
    return inv.view(B, 1, H, W), msk.view(B, 1, H, W), m


def inputs(rgb, disp, floor, inv, msk):
    """Assemble the 8-channel input and log z_base. rgb uint8 [B,H,W,3]; disp/floor [B,H,W]."""
    import torch
    B = rgb.shape[0]
    d = disp.float(); med = d.reshape(B, -1).median(1).values[:, None, None]
    mad = (d - med).abs().reshape(B, -1).mean(1)[:, None, None].clamp(min=1e-3); dn = ((d - med) / mad).clamp(-8, 8)
    # per-image least squares disp = a * inv + b on the painted ToF pixels
    w_ = msk[:, 0]; x = inv[:, 0]; n = w_.sum((1, 2)).clamp(min=1)
    mx = (w_ * x).sum((1, 2)) / n; my = (w_ * d).sum((1, 2)) / n
    a = (w_ * (x - mx[:, None, None]) * (d - my[:, None, None])).sum((1, 2)) / (w_ * (x - mx[:, None, None]) ** 2).sum((1, 2)).clamp(min=1e-6)
    b = my - a * mx
    nz = (msk[:, 0].reshape(B, -1).sum(1) / 25)                         # approx. number of valid zones
    good = (a > 0) & (nz >= 3)
    invz = ((d - b[:, None, None]) / a[:, None, None].clamp(min=1e-6)).clamp(1 / 20., 1 / .15)
    zb = torch.where(good[:, None, None], 1 / invz, torch.full_like(d, 2.))
    lzb = torch.log(zb)
    x_in = torch.cat([rgb.permute(0, 3, 1, 2).float() / 255 - .5, dn[:, None], inv, msk, floor.float()[:, None], lzb[:, None]], 1)
    return x_in, lzb


def main(cache, out, epochs='15'):
    import torch
    epochs = int(epochs); dev = 'cuda'; cache = Path(cache)
    I = json.loads((cache / 'index.json').read_text()); seqs = I['seqs']
    tr = [s for s in seqs if s['split'] == 'train']
    mm = {k: np.load(cache / f'{k}.npy', mmap_mode='r') for k in ('rgb', 'disp', 'depth', 'floor')}
    G = {k: [] for k in mm}
    for s in tr:                                                         # contiguous reads, straight to the GPU
        for k in mm:
            G[k].append(torch.from_numpy(np.ascontiguousarray(mm[k][s['start']:s['start'] + s['n']:2])).to(dev))   # every 2nd frame (5 Hz is redundant; fits 8 GB)
    G = {k: torch.cat(v) for k, v in G.items()}; N = len(G['rgb'])
    K = tr[0]['K']
    geos = []
    for t in TILTS:
        idx, zin, uz = zone_geometry(K, t)
        geos.append((torch.tensor(idx, device=dev), torch.tensor(zin, device=dev), torch.tensor(uz, device=dev, dtype=torch.float32)))
    torch.manual_seed(20261014); gen = torch.Generator(device=dev).manual_seed(20261014)
    m = net().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * (N // 16); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=.1)
    start = time.perf_counter(); log = []; step = 0
    print('train frames', N, 'params', sum(p.numel() for p in m.parameters()), flush=True)
    for ep in range(epochs):
        perm = torch.randperm(N, device=dev); tot = 0.
        for b in range(0, N - 15, 16):
            i = perm[b:b + 16]; B = len(i)
            rgb, disp, depth, floor = G['rgb'][i], G['disp'][i].float(), G['depth'][i].float(), G['floor'][i].float()
            flip = torch.rand(B, device=dev) < .5
            if flip.any():
                rgb[flip] = rgb[flip].flip(2); disp[flip] = disp[flip].flip(2); depth[flip] = depth[flip].flip(2); floor[flip] = floor[flip].flip(2)
            jit = (1 + .2 * (torch.rand(B, 1, 1, 1, device=dev) - .5))
            rgb = (rgb.float() * jit + 20 * (torch.rand(B, 1, 1, 1, device=dev) - .5)).clamp(0, 255)
            ti = int(torch.randint(len(TILTS), (1,), generator=gen, device=dev))
            inv, msk, _ = prompt(depth, geos[ti], torch.rand(B, device=dev, generator=gen) < .5, gen)
            x, lzb = inputs(rgb, disp, floor, inv, msk)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                res = m(x)
            lz = lzb[:, None] + res.float()
            valid = (depth > .3) & (depth < 10); wt = torch.where(depth <= 3, 2., 1.) * valid
            loss = ((lz[:, 0] - torch.log(depth.clamp(min=.1))).abs() * wt).sum() / wt.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * B; step += 1
        log.append(dict(epoch=ep, loss=round(tot / N, 4), wall=round(time.perf_counter() - start))); print(log[-1], flush=True)
    torch.save(m.state_dict(), out)
    Path(out).with_suffix('.json').write_text(json.dumps(dict(log=log, train_frames=N, tilts=TILTS, plan='PLAN_TOFPROMPT.md'), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
