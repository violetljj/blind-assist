"""M3 retrained with real-like near-bin crosstalk tail and DC nuisance (EXPLORE).

Reuses the no-bin0 DR training histories (cnh-no-bin0-dr-retrain-dev-20261011/inputs/train)
without rerendering. Per M3 row, nuisance is drawn once and held over the eight history
frames: with probability .25 none, otherwise tail t1~U(0,3) normalized units on bin1 with
bins 2/3 at .43/.30 of it and a per-zone factor U(.5,1.5), plus DC~U(-.2,.5) on bins 1..15.
Amplitudes come from the consumed dorm pilot (REAL_CONTRACT_V5_ORIENTATION_DEV_20261011).
bin0 stays dropped. Original M3 recipe (BCE, AdamW .002/wd .0001, batch 256, cosine 20).
Ordinary/S are not retrained here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_no_bin0_dr_train_20261011 as T

ROOT = T.ROOT
SRC = T.OUT
OUT = ROOT / 'artifacts.local/work/cnh-nuisance-dr-m3-dev-20261011'
NUISANCE_SEED = 2026101191
P_CLEAN = .25


def nuisance(rng, n):
    clean = rng.random(n) < P_CLEAN
    t1 = np.where(clean, 0., rng.uniform(0, 3, n))[:, None, None]
    zone = rng.uniform(.5, 1.5, (n, 8, 8))
    dc = np.where(clean, 0., rng.uniform(-.2, .5, n))
    tail = np.zeros((n, 8, 8, 16), np.float32)
    for b, f in ((1, 1.), (2, .43), (3, .30)):
        tail[..., b] = t1 * f * zone
    tail[..., 1:] += dc[:, None, None, None]
    return tail, clean


def features():
    from cnh_cvr_projection import Projector
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'm3.npy'
    if target.exists():
        raise FileExistsError(target)
    z = np.load(SRC / 'inputs/train/histories.npy', mmap_mode='r')
    t = np.load(SRC / 'inputs/train/transforms.npy', mmap_mode='r')
    le = np.load(SRC / 'inputs/train/length.npy', mmap_mode='r')
    ids = np.load(SRC / 'inputs/train/m3_indices.npy')
    rng = np.random.default_rng(NUISANCE_SEED)
    vox = np.lib.format.open_memmap(target, mode='w+', dtype=np.float16, shape=(len(ids), 3, 24, 17, 33))
    projector = Projector()
    began = time.monotonic()
    clean_count = 0
    for begin in range(0, len(ids), 8):
        ix = ids[begin:begin + 8]
        zz = np.array(z[ix]).astype(np.float32)
        add, clean = nuisance(rng, len(ix))
        valid = np.arange(8)[None, :] >= (8 - np.array(le[ix]))[:, None]
        zz += add[:, None] * valid[:, :, None, None, None]
        zz[..., 0] = 0
        clean_count += int(clean.sum())
        vox[begin:begin + len(ix)] = T.project_native(zz.astype(np.float16), np.array(t[ix]), np.array(le[ix]), projector).half().cpu().numpy()
        if begin % 4000 == 0:
            print('FEATURES', begin, '/', len(ids), round(time.monotonic() - began, 1), flush=True)
    vox.flush()
    receipt = dict(stage='features', rows=int(len(ids)), clean_rows=clean_count, seconds=time.monotonic() - began,
                   source=str(SRC / 'inputs/train'), nuisance_seed=NUISANCE_SEED, p_clean=P_CLEAN,
                   code_sha256=T.sha(Path(__file__)))
    (OUT / 'features_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    return receipt


def train(seeds, amp=False):
    """amp=True runs forward/loss under bf16 autocast (Tensor Core throughput); default keeps FP32."""
    from cnh_cvr_pilot import CVR
    from cnh_cvr_projection import query_masks
    T.configure()
    x = T.load_gpu(OUT / 'm3.npy')
    y = torch.as_tensor(np.load(SRC / 'inputs/train/m3_labels.npy'), device='cuda')
    masks = torch.as_tensor(query_masks(), device='cuda')
    for seed in seeds:
        target = OUT / 'models/M3' / f'model_seed{seed}.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed); net = CVR().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); history = []
        for epoch in range(20):
            t0 = time.monotonic(); order = rng.permutation(len(x)); total = 0
            for begin in range(0, len(x), 256):
                ix = torch.as_tensor(np.sort(order[begin:begin + 256]), device='cuda')
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    logits = net(T.prep_cvr(x[ix], masks))
                loss = F.binary_cross_entropy_with_logits(logits.float(), y[ix])
                loss.backward(); opt.step(); total += float(loss.detach()) * len(ix)
            scheduler.step()
            history.append(dict(epoch=epoch + 1, loss=total / len(x), seconds=time.monotonic() - t0))
            print('TRAIN_M3', seed, epoch + 1, round(history[-1]['loss'], 6), round(history[-1]['seconds'], 1), flush=True)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, target)
        (target.with_suffix('.json')).write_text(json.dumps(dict(seed=seed, epochs=20, rows=len(x),
            steps=20 * math.ceil(len(x) / 256), amp_bf16=amp, sha256=T.sha(target), history=history), indent=1), encoding='utf8')
        del net, opt, scheduler
    torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('features', 'train'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    a = p.parse_args()
    print(features() if a.stage == 'features' else train(a.seeds))
