"""M3 retrained with nuisance DR plus assumed-pitch error DR (EXPLORE).

Extends cnh_nuisance_dr_m3_dev_20261011. The consumed dorm replay showed the M3-only chain is
highly sensitive to the assumed head pitch (+/-5 deg flips it between plausible and wrong), and
the floor is rarely within range to estimate pitch from ToF. Histories stay rendered at the
true pose; projection uses an assumed pose rotated by delta about the sensor x axis, held over
the history (corridor_from_sensor @ Rx(delta)), delta~U(-12,12) deg for 80% of rows and 0
otherwise. Nuisance as before except DC widened to U(-.3,.8) (dark residual). Labels unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_no_bin0_dr_train_20261011 as T
import cnh_nuisance_dr_m3_dev_20261011 as N
from cnh_cvr_pilot import rotation

ROOT = T.ROOT
SRC = T.OUT
OUT = ROOT / 'artifacts.local/work/cnh-nuisance-pitch-dr-m3-dev-20261012'
SEED = 2026101203
P_CLEAN, P_EXACT, PITCH = .25, .20, 12.
BATCH = 64


def nuisance(rng, n):
    clean = rng.random(n) < P_CLEAN
    t1 = np.where(clean, 0., rng.uniform(0, 3, n))[:, None, None]
    zone = rng.uniform(.5, 1.5, (n, 8, 8))
    dc = np.where(clean, 0., rng.uniform(-.3, .8, n))
    tail = np.zeros((n, 8, 8, 16), np.float32)
    for b, f in ((1, 1.), (2, .43), (3, .30)):
        tail[..., b] = t1 * f * zone
    tail[..., 1:] += dc[:, None, None, None]
    return tail, clean


def pitch_error(rng, n):
    delta = np.where(rng.random(n) < P_EXACT, 0., rng.uniform(-PITCH, PITCH, n))
    mats = np.repeat(np.eye(4)[None], n, 0)
    for i, d in enumerate(delta):
        mats[i, :3, :3] = rotation(d, 'x')
    return mats, delta


def fast_project(histories, transforms, lengths, projector):
    """float32 batched copy of T.project_native (laptop GPUs run float64 at ~1/64 rate)."""
    import torch
    from cnh_cvr_projection import EDGE, WIDTH, SUB, SHAPE
    points = projector.points.float(); volumes = projector.volumes.float()
    b = len(histories)
    t = torch.as_tensor(transforms, device='cuda', dtype=torch.float32)
    z = torch.as_tensor(histories, device='cuda', dtype=torch.float32)
    le = torch.as_tensor(lengths, device='cuda')
    total = torch.zeros((b, *SHAPE), device='cuda'); count = torch.zeros_like(total); current = None
    for j in range(8):
        p = torch.bmm(points[None] - t[:, j, None, :3, 3], t[:, j, :3, :3])
        radius = torch.linalg.vector_norm(p, dim=2)
        xy = p[:, :, :2] / p[:, :, 2:3].clamp_min(1e-30)
        ij = torch.floor((xy + EDGE) / (2 * EDGE) * 8).long()
        bins = torch.floor(radius / WIDTH).long()
        valid = (p[:, :, 2] > 0) & (ij >= 0).all(2) & (ij < 8).all(2) & (bins >= 0) & (bins < 16)
        valid &= (j >= 8 - le)[:, None]
        index = (ij[:, :, 1].clamp(0, 7) * 8 + ij[:, :, 0].clamp(0, 7)) * 16 + bins.clamp(0, 15)
        weight = valid * projector.voxel_volume / (SUB ** 3) / volumes[index]
        mass = torch.gather(z[:, j].reshape(b, -1), 1, index) * weight
        evidence = mass.reshape(b, -1, SUB ** 3).sum(2).reshape(b, *SHAPE)
        coverage = valid.reshape(b, -1, SUB ** 3).float().mean(2).reshape(b, *SHAPE)
        total += evidence; count += coverage; current = evidence
    return torch.stack((total, count, current), 1)


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
    rng = np.random.default_rng(SEED)
    vox = np.lib.format.open_memmap(target, mode='w+', dtype=np.float16, shape=(len(ids), 3, 24, 17, 33))
    projector = Projector(); began = time.monotonic(); deltas = []
    for begin in range(0, len(ids), BATCH):
        ix = ids[begin:begin + BATCH]
        zz = np.array(z[ix]).astype(np.float32)
        add, _ = nuisance(rng, len(ix))
        valid = np.arange(8)[None, :] >= (8 - np.array(le[ix]))[:, None]
        zz += add[:, None] * valid[:, :, None, None, None]
        zz[..., 0] = 0
        mats, delta = pitch_error(rng, len(ix)); deltas.extend(delta.tolist())
        tt = np.array(t[ix]) @ mats[:, None]
        vox[begin:begin + len(ix)] = fast_project(zz.astype(np.float16), tt, np.array(le[ix]), projector).half().cpu().numpy()
        if begin % (BATCH * 64) == 0:
            print('FEATURES', begin, '/', len(ids), round(time.monotonic() - began, 1), flush=True)
    vox.flush()
    np.save(OUT / 'pitch_delta_deg.npy', np.asarray(deltas, np.float32))
    receipt = dict(stage='features', rows=int(len(ids)), seconds=time.monotonic() - began, seed=SEED,
                   p_clean=P_CLEAN, p_exact_pitch=P_EXACT, pitch_range_deg=PITCH, dc_range=[-.3, .8],
                   code_sha256=T.sha(Path(__file__)), projection='float32 batched fast_project, batch 64')
    (OUT / 'features_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('features', 'train'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    a = p.parse_args()
    if a.stage == 'features':
        print(features())
    else:
        N.OUT = OUT          # reuse the identical M3 training loop on this task's features
        N.train(a.seeds, amp=True)
