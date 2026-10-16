"""Learn where to look again (PLAN addendum 5, variant B; EXPLORE, 2026-10-16).

A tiny per-patch localizer predicts "this patch shows the object the user is filming" from the frozen patch token, the
part-memory evidence (best similarity over the user's objects, and its margin over the second best object) and a learned
position prior. It is trained on ORBIT *train* users with the train-split boxes as patch targets; at test time it needs
no annotation. The zoom box is a square around the smoothed peak of its map.

  cv  <prefix> <out json> [folds=4] [seed=0]  user-disjoint folds on packed patch samples (train_evidence_head pack):
                                              share of frames whose chosen peak lies inside the annotated box, for the
                                              learned map vs the raw evidence peak vs the frame centre
  fit <prefix> <out dir> [folds=4] [seed=0]   one model on all users (loc_all.pt) plus one per fold (loc_f<k>.pt) and
                                              loc_folds.json (user -> held-out fold model), for honest dev dumps
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


class Localizer(nn.Module):
    def __init__(self, D=384, T=256, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + 2, hid), nn.GELU(), nn.Linear(hid, 1))
        self.pos = nn.Parameter(torch.zeros(T))

    @staticmethod
    def evidence(S, valid=None):
        """S [N,T,O] part similarities -> [N,T,2]: best object similarity and its margin over the second best."""
        S = S.float()
        if valid is not None:
            S = S.masked_fill(~valid[:, None, :], -2.)
        top = S.topk(min(2, S.shape[2]), dim=2).values
        second = top[..., 1] if top.shape[2] > 1 else torch.zeros_like(top[..., 0])
        return torch.stack([top[..., 0], top[..., 0] - second], -1)

    def forward(self, X, S, valid=None):
        return self.net(torch.cat([X.float(), self.evidence(S, valid)], -1)).squeeze(-1) + self.pos


def peak(score, grid):
    """[N,T] map -> index of the 3x3-smoothed maximum."""
    return F.avg_pool2d(score.view(-1, 1, grid, grid).float(), 3, 1, 1, count_include_pad=False).flatten(1).argmax(1)


def train_loc(d, idx, seed=0, epochs=10, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    m = Localizer(d['X'].shape[2], d['X'].shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(idx) + bs - 1) // bs))
    g = torch.Generator(device='cuda').manual_seed(seed)
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), generator=g, device='cuda')]
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]; tgt = d['mask'][j].float()
            pw = ((1 - tgt).sum() / tgt.sum().clamp(min=1)).clamp(max=20)
            loss = F.binary_cross_entropy_with_logits(m(d['X'][j], d['S'][j], d['valid'][j]), tgt, pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m.eval()


def hits(d, idx, score, grid):
    p = peak(score, grid)
    return d['mask'][idx].gather(1, p[:, None]).squeeze(1).float()


def cv(prefix, out, folds=4, seed=0):
    from train_evidence_head import load
    d = load(prefix); folds, seed = int(folds), int(seed); T = d['X'].shape[1]; grid = int(round(T ** .5))
    users = sorted(set(d['user'])); rs = np.random.default_rng(seed); rs.shuffle(users)
    fold_of = {u: i % folds for i, u in enumerate(users)}
    fo = torch.as_tensor(np.array([fold_of[u] for u in d['user']]), device='cuda')
    has_box = d['mask'].any(1)
    res = {k: [] for k in ('evidence_peak', 'centre', 'learned')}
    for f in range(folds):
        tr = torch.nonzero((fo != f) & has_box).squeeze(1); te = torch.nonzero((fo == f) & has_box).squeeze(1)
        ev = Localizer.evidence(d['S'][te], d['valid'][te])[..., 0]
        res['evidence_peak'].append(hits(d, te, ev, grid))
        c = torch.zeros(len(te), T, device='cuda'); c[:, (grid // 2) * grid + grid // 2] = 1
        res['centre'].append(hits(d, te, c, grid))
        m = train_loc(d, tr, seed)
        with torch.inference_mode():
            lg = torch.cat([m(d['X'][te[b:b + 512]], d['S'][te[b:b + 512]], d['valid'][te[b:b + 512]]) for b in range(0, len(te), 512)])
        res['learned'].append(hits(d, te, lg, grid))
        print('fold', f, {k: round(100 * float(v[-1].mean()), 1) for k, v in res.items()}, flush=True)
    summ = {k: round(100 * float(torch.cat(v).mean()), 2) for k, v in res.items()}
    Path(out).write_text(json.dumps(dict(prefix=prefix, folds=folds, seed=seed, peak_inside_box=summ), indent=1))
    print('peak inside annotated box (%):', summ, flush=True)


def fit(prefix, out, folds=4, seed=0):
    from train_evidence_head import load
    d = load(prefix); folds, seed = int(folds), int(seed); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    users = sorted(set(d['user'])); rs = np.random.default_rng(seed); rs.shuffle(users)
    fold_of = {u: i % folds for i, u in enumerate(users)}
    fo = torch.as_tensor(np.array([fold_of[u] for u in d['user']]), device='cuda'); has_box = d['mask'].any(1)
    kw = dict(D=d['X'].shape[2], T=d['X'].shape[1])
    for f in range(folds):
        m = train_loc(d, torch.nonzero((fo != f) & has_box).squeeze(1), seed)
        torch.save(dict(kw=kw, state=m.state_dict()), out / f'loc_f{f}.pt')
    m = train_loc(d, torch.nonzero(has_box).squeeze(1), seed)
    torch.save(dict(kw=kw, state=m.state_dict()), out / 'loc_all.pt')
    (out / 'loc_folds.json').write_text(json.dumps({u: str(out / f'loc_f{k}.pt') for u, k in fold_of.items()}, indent=1))
    print('saved fold and all-user localizers to', out, 'params', sum(p.numel() for p in m.parameters()), flush=True)


def load_loc(path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    m = Localizer(**ck['kw']).cuda().eval(); m.load_state_dict(ck['state'])
    return m


if __name__ == '__main__':
    dict(cv=cv, fit=fit)[sys.argv[1]](*sys.argv[2:])
