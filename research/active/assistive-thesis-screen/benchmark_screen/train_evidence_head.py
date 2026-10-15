"""Train / cross-validate the evidence head (evidence_head.py) on ORBIT train users. EXPLORE, 2026-10-16.

  pack <dump dir> <prefix>                      per-video patch samples (orbit_dump patch_k) -> <prefix>.X.npy (fp16
                                                memmap [N,T,D]) + <prefix>.meta.npz (S, full, valid padded to O_max)
  cv   <prefix> <variants json> <out json> [folds=4] [seed=0]
                                                user-disjoint folds; per-frame accuracy of each variant vs the
                                                training-free rules (top-8 mean, top-8 + z(full)) on the held-out users
  fit  <prefix> <variant json> <out.pt> [seed=0]   train on all users of <prefix>

Variant keys: objectness, contrast, rank (EvidenceHead args; rank > 0 = adapter on the memory matching), box (BCE
weight of the objectness on train boxes), patchce (weight of in-box patch classification in adapter space), epochs, lr.
Everything runs on the GPU (X ~2 GB fp16 for 11 k frames) under the same memory guard as orbit_dump.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def pack(dump, prefix):
    files = sorted(f for f in Path(dump, 'patch').glob('*.npz') if not f.name.startswith('_mem_'))
    heads = [np.load(f) for f in files]
    N = sum(len(h['idx']) for h in heads); T, D = heads[0]['X'].shape[1:]; O = max(h['S'].shape[2] for h in heads)
    X = np.lib.format.open_memmap(f'{prefix}.X.npy', 'w+', np.float16, (N, T, D))
    S = np.zeros((N, T, O), np.float16); full = np.zeros((N, O), np.float32); valid = np.zeros((N, O), bool)
    mask = np.zeros((N, T), bool); y = np.zeros(N, int); user = []; video = []; k = 0
    for f, h in zip(files, heads):
        n, o = len(h['idx']), h['S'].shape[2]
        X[k:k + n] = h['X']; S[k:k + n, :, :o] = h['S']; full[k:k + n, :o] = h['full']; valid[k:k + n, :o] = True
        mask[k:k + n] = h['mask']; y[k:k + n] = int(h['y']); user += [str(h['user'])] * n; video += [f.stem] * n; k += n
    X.flush(); del X
    users = sorted(set(user)); mems = [np.load(Path(dump, 'patch', f'_mem_{u}.npz')) for u in users]
    K = max(np.bincount(mm['owner']).max() for mm in mems)
    P = np.zeros((len(users), O, K, D), np.float32); pvalid = np.zeros((len(users), O), bool)
    for i, mm in enumerate(mems):
        for o in range(int(mm['owner'].max()) + 1):
            q = mm['parts'][mm['owner'] == o]; P[i, o] = q[np.arange(K) % len(q)]; pvalid[i, o] = True   # pad by repeating
    np.savez(f'{prefix}.meta.npz', S=S, full=full, valid=valid, mask=mask, y=y, user=np.array(user), video=np.array(video),
             P=P, pvalid=pvalid, uidx=np.array([users.index(u) for u in user]))
    print('packed', N, 'frames', len(set(user)), 'users', len(files), 'videos, O_max', O, flush=True)


def load(prefix):
    import torch
    from orbit_dump import guard
    guard()
    m = np.load(f'{prefix}.meta.npz')
    Xm = np.load(f'{prefix}.X.npy', mmap_mode='r')
    X = torch.empty(Xm.shape, dtype=torch.half, device='cuda')
    for i in range(0, len(Xm), 1024):                               # chunked: no full host copy
        X[i:i + 1024] = torch.from_numpy(np.ascontiguousarray(Xm[i:i + 1024])).cuda()
    d = {k: torch.from_numpy(m[k]).cuda() for k in ('S', 'full', 'valid', 'mask', 'y', 'P', 'pvalid', 'uidx')}
    d['X'] = X; d['user'] = m['user']; d['video'] = m['video']
    return d


def rules(d, idx):
    """Training-free baselines on frames idx: top-8 mean of S, and z(top-8) + z(full)."""
    import torch
    S = d['S'][idx].float(); v = d['valid'][idx]
    top = S.topk(8, dim=1).values.mean(1)

    def z(s):
        s = s.masked_fill(~v, float('nan')); mu = torch.nanmean(s, 1, keepdim=True)
        sd = ((s - mu) ** 2).nan_to_num(0).sum(1, keepdim=True).div(v.sum(1, keepdim=True)).sqrt()
        return ((s - mu) / (sd + 1e-6)).nan_to_num(-1e4)
    return dict(top8=top.masked_fill(~v, -1e4), top8_fuse=z(top) + z(d['full'][idx]))


def train(d, idx, var, seed=0, log=None):
    import torch
    import torch.nn.functional as F
    from evidence_head import EvidenceHead
    torch.manual_seed(seed)
    kw = dict(objectness=var.get('objectness', True), contrast=var.get('contrast', True), rank=var.get('rank', 0))
    m = EvidenceHead(D=d['X'].shape[2], T=d['X'].shape[1], **kw).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=var.get('lr', 1e-3), weight_decay=1e-4)
    ep, bs = var.get('epochs', 15), 64
    if ep == 0:
        return m.eval(), kw
    steps = ep * ((len(idx) + bs - 1) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, var.get('lr', 1e-3), total_steps=steps)
    g = torch.Generator(device='cuda').manual_seed(seed)
    for e in range(ep):
        perm = idx[torch.randperm(len(idx), generator=g, device='cuda')]; tot = 0.
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            S = smaps(m, d, j)
            logits, a = m(d['X'][j], S, d['full'][j], d['valid'][j])
            loss = F.cross_entropy(logits, d['y'][j])
            if var.get('patchce', 0) and m.rank:                       # in-box patches should match their object
                inb = d['mask'][j]; pl = (S / 0.05).masked_fill(~d['valid'][j][:, None, :], -1e4)[inb]
                yt = d['y'][j][:, None].expand_as(inb)[inb]
                loss = loss + var['patchce'] * F.cross_entropy(pl, yt)
            if var.get('box', 0) and a is not None:
                tgt = d['mask'][j].float(); pw = ((1 - tgt).sum() / tgt.sum().clamp(min=1)).clamp(max=20)
                loss = loss + var['box'] * F.binary_cross_entropy_with_logits(a, tgt, pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach()) * len(j)
        if log is not None:
            log.append(round(tot / len(idx), 4))
    return m.eval(), kw


def smaps(m, d, j):
    """Similarity maps for frames j: stored training-free S, or recomputed in the adapter space."""
    if not m.rank:
        return d['S'][j]
    u = d['uidx'][j]
    return m.sim_maps(d['X'][j], d['P'][u], d['pvalid'][u])


def predict(m, d, idx):
    import torch
    out = []
    with torch.inference_mode():
        for b in range(0, len(idx), 256):
            j = idx[b:b + 256]
            out.append(m(d['X'][j], smaps(m, d, j), d['full'][j], d['valid'][j])[0])
    return torch.cat(out)


def cv(prefix, variants, out, folds=4, seed=0):
    import torch
    d = load(prefix); variants = json.loads(variants); folds, seed = int(folds), int(seed)
    users = sorted(set(d['user'])); rs = np.random.default_rng(seed); rs.shuffle(users)
    fold_of = {u: i % folds for i, u in enumerate(users)}
    fo = torch.as_tensor(np.array([fold_of[u] for u in d['user']]), device='cuda')
    res = {k: [] for k in ['top8', 'top8_fuse'] + [v['name'] for v in variants]}; per_user = {k: {} for k in res}
    t0 = time.time()

    def score(name, logits, te):
        ok = (logits.argmax(1) == d['y'][te]).float().cpu().numpy(); res[name].append(float(ok.mean()))
        for u in np.unique(d['user'][te.cpu().numpy()]):
            per_user[name][u] = float(ok[d['user'][te.cpu().numpy()] == u].mean())
    for f in range(folds):
        tr, te = torch.nonzero(fo != f).squeeze(1), torch.nonzero(fo == f).squeeze(1)
        for k, v in rules(d, te).items():
            score(k, v, te)
        for var in variants:
            m, _ = train(d, tr, var, seed); score(var['name'], predict(m, d, te), te); del m
        print('fold', f, {k: round(100 * v[-1], 2) for k, v in res.items()}, round(time.time() - t0), 's', flush=True)
    summ = {k: dict(fold_mean=round(100 * float(np.mean(v)), 2),
                    user_mean=round(100 * float(np.mean(list(per_user[k].values()))), 2)) for k, v in res.items()}
    Path(out).write_text(json.dumps(dict(prefix=prefix, folds=folds, seed=seed, variants=variants, summary=summ,
                                         per_fold=res, per_user=per_user), indent=1))
    for k, s in sorted(summ.items(), key=lambda kv: -kv[1]['user_mean']):
        print(f'{k:24s} user-mean {s["user_mean"]:6.2f}  fold-mean {s["fold_mean"]:6.2f}', flush=True)


def fit(prefix, variant, out, seed=0):
    import torch
    d = load(prefix); var = json.loads(variant); log = []
    idx = torch.arange(len(d['y']), device='cuda')
    m, kw = train(d, idx, var, int(seed), log)
    kw.update(D=d['X'].shape[2], T=d['X'].shape[1])
    torch.save(dict(kw=kw, state=m.state_dict(), variant=var, prefix=prefix, loss=log,
                    params=sum(p.numel() for p in m.parameters())), out)
    print('saved', out, 'params', sum(p.numel() for p in m.parameters()), 'loss', log[-3:], flush=True)


if __name__ == '__main__':
    dict(pack=pack, cv=cv, fit=fit)[sys.argv[1]](*sys.argv[2:])
