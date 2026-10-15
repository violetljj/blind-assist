"""Swiss DINO (Paramonov et al., IROS 2024, arXiv 2407.07541) identification core, re-implemented on shared frozen
DINOv2 patch tokens for ORBIT (the official code only covers PerSeg / iCubWorld). EXPLORE, 2026-10-15.

Support (Sec. IV-B), per object over all its context frames:
  mask 'full'   : every patch is the object (no annotation; no negatives, threshold = positive 95th-pct distance)
  mask 'center' : a central box (no annotation) turned into a segmentation with the paper's bbox step: k-means (k=5) on
                  the box + 1-patch border ring, drop clusters that touch the ring
  proto = mean of segment patches; distances d = 1 - cos; tr = min(pct95(d_pos), pct5(d_neg)) after the paper's 5 %
  outlier trimming (equivalent to the plain percentiles used here).
Query (Sec. IV-C2/3): patches with d < tr (or the single closest patch), split into 4-connected components; class score
= max over components of cos(mean component feature, proto). All batched on the GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _kmeans_batch(x, k=5, iters=10):
    """x: [B, M, D] normalised -> labels [B, M]. Deterministic init: evenly spaced points."""
    B, M, _ = x.shape
    c = x[:, torch.linspace(0, M - 1, k, device=x.device).long()].clone()
    for _ in range(iters):
        a = torch.einsum('bmd,bkd->bmk', x, c).argmax(2)
        oh = F.one_hot(a, k).float()
        c = F.normalize(torch.einsum('bmk,bmd->bkd', oh, x) + 1e-6 * c, dim=2)
    return torch.einsum('bmd,bkd->bmk', x, c).argmax(2)


def support(pat, grid, mode='center', box=None):
    """pat: [n, T, D] context patch tokens of one object (normalised). Returns (proto [D], tr scalar)."""
    pat = pat.float(); n, T, D = pat.shape
    if mode == 'full':
        seg = torch.ones(n, T, dtype=torch.bool, device=pat.device)
    else:
        a, b = box or (round(grid * 3 / 16), round(grid * 13 / 16))
        r = torch.arange(grid, device=pat.device)
        inb = ((r[:, None] >= a) & (r[:, None] < b) & (r[None] >= a) & (r[None] < b)).reshape(-1)
        ring = ((r[:, None] >= a - 1) & (r[:, None] < b + 1) & (r[None] >= a - 1) & (r[None] < b + 1)).reshape(-1) & ~inb
        idx = torch.nonzero(inb | ring).squeeze(1); is_ring = ring[idx]
        lab = _kmeans_batch(pat[:, idx], 5)                                   # [n, M]
        bad = torch.zeros(n, 5, dtype=torch.bool, device=pat.device)
        bad.scatter_(1, lab[:, is_ring], True)                               # clusters touching the ring
        keep = ~bad.gather(1, lab) & ~is_ring[None]
        keep = torch.where(keep.any(1, keepdim=True), keep, ~is_ring[None].expand_as(keep))   # fallback: the box
        seg = torch.zeros(n, T, dtype=torch.bool, device=pat.device); seg[:, idx] = keep
    proto = F.normalize(pat[seg].mean(0), dim=0)
    d = 1 - pat @ proto
    ptr = torch.quantile(d[seg], .95) if seg.sum() < 2 ** 24 else d[seg].float().kthvalue(int(.95 * seg.sum())).values
    tr = ptr if (~seg).sum() == 0 else torch.minimum(ptr, torch.quantile(d[~seg][:2 ** 24], .05))
    return proto, tr


def _components(active, grid, iters=40):
    """active: [B, T] bool -> component id per patch ([B, T] long, -1 inactive) by max-label propagation."""
    B, T = active.shape
    lab = torch.where(active, torch.arange(T, device=active.device).expand(B, T), torch.full_like(active, -1, dtype=torch.long))
    lab = lab.view(B, 1, grid, grid).float()
    act = active.view(B, 1, grid, grid)
    for _ in range(iters):
        nb = F.max_pool2d(lab, 3, 1, 1)
        # 4-connectivity: max over the cross only
        cross = torch.stack([lab, F.pad(lab, (1, 0, 0, 0), value=-1)[..., :-1], F.pad(lab, (0, 1, 0, 0), value=-1)[..., 1:],
                             F.pad(lab, (0, 0, 1, 0), value=-1)[..., :-1, :], F.pad(lab, (0, 0, 0, 1), value=-1)[..., 1:, :]]).amax(0)
        new = torch.where(act, cross, lab)
        if torch.equal(new, lab):
            break
        lab = new
        del nb
    return lab.view(B, T).long()


def query_scores(fpat, protos, trs, grid, chunk=16, prec='fp32'):
    """fpat: [N, T, D] query patch tokens; protos: [O, D]; trs: [O]. Returns [N, O] Swiss DINO class scores.
    prec 'fp16' keeps the patch features, component sums and dot products in half precision (tensor cores)."""
    if prec == 'fp16':
        chunk *= 2
    return torch.cat([_query(fpat[i:i + chunk], protos, trs, grid, prec) for i in range(0, len(fpat), chunk)])


def _query(fpat, protos, trs, grid, prec='fp32'):
    N, T, D = fpat.shape; O = len(protos)
    dt = torch.half if prec == 'fp16' else torch.float
    fp = fpat.to(dt); protos = protos.to(dt)
    d = 1 - (fp @ protos.T).float()                                          # [N, T, O]
    active = (d < trs[None, None]).permute(0, 2, 1).reshape(N * O, T)         # [N*O, T]
    nearest = d.permute(0, 2, 1).reshape(N * O, T).argmin(1)
    active[torch.arange(N * O, device=fp.device), nearest] |= ~active.any(1)
    comp = _components(active, grid)                                         # [N*O, T]
    # component means: scatter patch features by (row, comp id)
    feats = fp.unsqueeze(1).expand(N, O, T, D).reshape(N * O, T, D)
    valid = comp >= 0; cid = comp.clamp(min=0)
    sums = torch.zeros(N * O, T, D, device=fp.device, dtype=dt).scatter_add_(1, cid.unsqueeze(2).expand(-1, -1, D), feats * valid.unsqueeze(2).to(dt))
    cnt = torch.zeros(N * O, T, device=fp.device).scatter_add_(1, cid, valid.float())
    means = F.normalize(sums.float() / cnt.clamp(min=1).unsqueeze(2), dim=2).to(dt)
    pr = protos.unsqueeze(0).expand(N, O, D).reshape(N * O, D)
    s = torch.einsum('btd,bd->bt', means, pr).float().masked_fill(cnt == 0, -2.)
    return s.amax(1).view(N, O)
