"""Learned RGB scale (anchor) pilot on Hypersim TRAIN; real check on consumed captures (EXPLORE).

Plan and stop rules: artifacts.local/work/learned-anchor-pilot-20261013/PLAN.md (written before training).
Stages:
  cache   : DAV2-S (518 short side, as deployed) on Hypersim train RGB -> disparity at 192x256 (float16)
  train   : tiny CNN -> global (s, t); metric inverse depth = s * d_norm + t, d_norm = (disp - median) / IQR
  hypeval : held-out train families, learned vs fit_affine(+robust) chain on the same emulated zones
  real    : v3 chain vs arm A (learned replaces the floor anchor) vs arm B (learned always) on four consumed batches,
            one DAV2 pass per frame; v1 criteria with the v2 2-of-3 confirmation
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import ROOT

OUT = ROOT / 'artifacts.local/work/learned-anchor-pilot-20261013'
BANFO = ROOT / 'artifacts.local/work/ba-nfo-20260919'
HYP = ROOT / 'artifacts.local/datasets/hypersim-ba-nfo'
H, W = 192, 256
HOLDOUT_FAMILIES = 6


def train_rows():
    rows = [r for r in json.loads((BANFO / 'prepared-manifest.json').read_text()) if r['source'] == 'hypersim' and r['split'] == 'train']
    fams = sorted({r['scene'][:6] for r in rows})
    held = set(fams[-HOLDOUT_FAMILIES:])
    for r in rows:
        r['holdout'] = r['scene'][:6] in held
    return rows


def norm_disp(d):
    med = np.nanmedian(d); iqr = np.nanpercentile(d, 75) - np.nanpercentile(d, 25)
    return (d - med) / max(iqr, 1e-6), med, iqr


def raster(values, boxes, h=H, w=W):
    """values: metric optical-Z per zone (NaN invalid); boxes [y0,x0,y1,x1] on (h,w)."""
    inv = np.zeros((h, w), np.float32); mask = np.zeros((h, w), np.float32)
    for v, (y0, x0, y1, x1) in zip(values, boxes):
        if np.isfinite(v) and v > .3:
            inv[y0:y1, x0:x1] = 1 / v; mask[y0:y1, x0:x1] = 1
    return inv, mask


def cache():
    """Batched fp16 DAV2-S (same 518 preprocessing as infer_image) with threaded decode; one packed array."""
    import cv2
    import torch
    import torch.nn.functional as Fn
    from concurrent.futures import ThreadPoolExecutor
    import real_rgb_small_model_dev_20261012 as S
    OUT.mkdir(parents=True, exist_ok=True)
    rows = train_rows(); model, _ = S.load_dav2_small(); start = time.perf_counter()
    disp = np.lib.format.open_memmap(OUT / 'hyp_disp192.npy', 'w+', np.float16, (len(rows), H, W))
    mean, std = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
    def prep(r):
        # infer_image preprocessing on CPU: BGR->RGB/255, cubic resize with short side 518 (multiple of 14), normalise.
        img = cv2.cvtColor(cv2.imread(str(HYP / r['rgb'])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        h0, w0 = img.shape[:2]; sc = 518 / min(h0, w0)
        nh, nw = int(round(h0 * sc / 14)) * 14, int(round(w0 * sc / 14)) * 14
        img = (cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC) - mean) / std
        return np.ascontiguousarray(img.transpose(2, 0, 1))
    B = 8
    batches = [rows[k:k + B] for k in range(0, len(rows), B)]
    with ThreadPoolExecutor(8) as pool:
        pending = [pool.submit(lambda b: np.stack([prep(r) for r in b]), b) for b in batches[:4]]   # bounded prefetch
        for k in range(len(batches)):
            x = torch.from_numpy(pending.pop(0).result()).cuda(non_blocking=True)
            if k + 4 < len(batches):
                pending.append(pool.submit(lambda b: np.stack([prep(r) for r in b]), batches[k + 4]))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                d = model(x)
            d = Fn.interpolate(d[:, None].float(), (H, W), mode='bilinear', align_corners=True)[:, 0]
            disp[k * B:k * B + len(x)] = d.cpu().numpy().astype(np.float16)
            if k % 100 == 0:
                print(k * B, round(time.perf_counter() - start), 's', flush=True)
    disp.flush()
    (OUT / 'hyp_rows.json').write_text(json.dumps([dict(id=r['id'], prepared=r['prepared'], holdout=r['holdout']) for r in rows]))
    print('cached', len(rows), round(time.perf_counter() - start), 's')


def pack():
    """Training tensors at 96x128 (the net outputs a global scale): normalised disparity, GT inverse depth, zones."""
    import cv2
    rows = json.loads((OUT / 'hyp_rows.json').read_text()); disp = np.load(OUT / 'hyp_disp192.npy', mmap_mode='r')
    n = len(rows); dn = np.zeros((n, 96, 128), np.float16); gt = np.full((n, 96, 128), np.nan, np.float16); vals = np.zeros((n, 64), np.float32)
    for i, r in enumerate(rows):
        with np.load(BANFO / r['prepared']) as p:
            depth, vals[i] = p['depth'], p['values']
            if i == 0:
                boxes = p['boxes']
        z, _, _ = norm_disp(disp[i].astype(np.float32))
        dn[i] = cv2.resize(z, (128, 96), interpolation=cv2.INTER_AREA)
        inv = np.where(np.isfinite(depth) & (depth > .3) & (depth < 10), 1 / np.maximum(depth, 1e-3), np.nan).astype(np.float32)
        gt[i] = inv[::2, ::2]
    zidx = np.full((96, 128), 64, np.int64)
    for k, (y0, x0, y1, x1) in enumerate(boxes // 2):
        zidx[y0:y1, x0:x1] = k
    np.savez(OUT / 'pack96.npz', dn=dn, gt=gt, vals=vals, zidx=zidx, holdout=np.array([r['holdout'] for r in rows]), boxes=boxes)
    print('packed', n)


def build_net():
    import torch.nn as nn
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    return nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 64), blk(64, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 2))


def train():
    """All data on GPU; vectorised zone dropout/raster; bf16 autocast."""
    import torch
    P = np.load(OUT / 'pack96.npz'); tr = ~P['holdout']
    dev = 'cuda'; torch.manual_seed(20261013); g = torch.Generator(device=dev).manual_seed(20261013)
    dn = torch.tensor(P['dn'][tr]).to(dev); gt = torch.tensor(P['gt'][tr]).to(dev); vals = torch.tensor(P['vals'][tr]).to(dev)
    zidx = torch.tensor(P['zidx']).to(dev); n = len(dn)
    net = build_net().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    start = time.perf_counter(); log = []
    for epoch in range(12):
        order = torch.randperm(n, device=dev, generator=g); tot = 0.
        for k in range(0, n, 64):
            idx = order[k:k + 64]; v = vals[idx].clone(); b = len(idx)
            v[v > 4.] = float('nan')
            keep = torch.rand(b, 64, device=dev, generator=g) >= torch.rand(b, 1, device=dev, generator=g) * .9
            keep &= (torch.rand(b, 1, device=dev, generator=g) >= .15)
            v = torch.where(keep, v * (1 + .03 * torch.randn(b, 64, device=dev, generator=g)), torch.full_like(v, float('nan')))
            ok = torch.isfinite(v) & (v > .3)
            inv = torch.cat([torch.where(ok, 1 / v.clamp_min(.3), torch.zeros_like(v)), torch.zeros(b, 1, device=dev)], 1)[:, zidx]
            msk = torch.cat([ok.float(), torch.zeros(b, 1, device=dev)], 1)[:, zidx]
            d = dn[idx].float(); y = gt[idx].float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                st = net(torch.stack([d, inv, msk], 1))
            st = st.float(); pred = st[:, 0, None, None] * d + st[:, 1, None, None]
            m = torch.isfinite(y)
            loss = (torch.abs(pred - torch.nan_to_num(y)) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * b
        log.append(dict(epoch=epoch, loss=tot / n, wall=round(time.perf_counter() - start, 1)))
        print(log[-1], flush=True)
    torch.save(net.state_dict(), OUT / 'anchor_net.pt')
    (OUT / 'train_log.json').write_text(json.dumps(log, indent=1))


def load_net():
    import torch
    net = build_net(); net.load_state_dict(torch.load(OUT / 'anchor_net.pt', map_location='cpu')); return net.eval()


def predict(net, disp, values, boxes):
    import torch
    dn, med, iqr = norm_disp(disp)
    inv, mask = raster(values, boxes, *disp.shape)
    with torch.no_grad():
        s, t = net(torch.tensor(np.stack([dn, inv, mask])[None].astype(np.float32)))[0].tolist()
    # metric inverse depth = s * (disp - med) / iqr + t  ->  disp = a * inv + b
    a = iqr / s if abs(s) > 1e-6 else np.nan
    b = med - t * iqr / s if abs(s) > 1e-6 else np.nan
    return (float(a), float(b)) if np.isfinite(a) and a > 0 else None


def hypeval():
    import real_rgb_small_model_dev_20261012 as S
    from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
    net = load_net(); rng = np.random.default_rng(7); res = {'learned': [], 'chain': [], 'chain_none': 0, 'n': 0}
    rows = json.loads((OUT / 'hyp_rows.json').read_text()); D = np.load(OUT / 'hyp_disp192.npy', mmap_mode='r')
    for i, r in enumerate(rows):
        if not r['holdout']:
            continue
        disp = D[i].astype(np.float32)
        with np.load(BANFO / r['prepared']) as p:
            depth, values, boxes = p['depth'], p['values'].copy(), p['boxes']
        values[values > 4.] = np.nan
        values[rng.random(64) < .3] = np.nan                       # mild, fixed-seed dropout, same for both arms
        near = np.isfinite(depth) & (depth > .3) & (depth < 3.)
        if near.sum() < 200:
            continue
        res['n'] += 1; gt = 1 / depth[near]
        fl = predict(net, disp, values, boxes)
        res['learned'].append(float(np.median(np.abs((disp[near] - fl[1]) / fl[0] - gt) / gt)) if fl else np.nan)
        dz = np.array([np.median(disp[y0:y1, x0:x1]) for y0, x0, y1, x1 in boxes])
        inv = np.where(np.isfinite(values), 1 / values, np.nan)
        fit = S.fit_affine(dz, inv)
        if not fit_quality(dz, inv, fit)['accepted']:
            fit = robust_affine(dz, inv)['fit'] or fit
        if fit is None:
            res['chain_none'] += 1; res['chain'].append(np.nan)
        else:
            res['chain'].append(float(np.median(np.abs((disp[near] - fit[1]) / fit[0] - gt) / gt)))
    L, C = np.array(res['learned']), np.array(res['chain'])
    both = np.isfinite(L) & np.isfinite(C)
    summary = dict(frames=res['n'], chain_no_fit=res['chain_none'],
                   learned_median_rel_err=float(np.nanmedian(L)), chain_median_rel_err_where_fit=float(np.nanmedian(C)),
                   paired_frames=int(both.sum()), paired_learned=float(np.median(L[both])), paired_chain=float(np.median(C[both])),
                   learned_on_chain_failures=float(np.nanmedian(L[~np.isfinite(C)])) if (~np.isfinite(C)).any() else None)
    (OUT / 'hypeval.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    {'cache': cache, 'pack': pack, 'train': train, 'hypeval': hypeval}[sys.argv[1]]()
