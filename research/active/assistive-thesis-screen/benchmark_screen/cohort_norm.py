"""Background-cohort score normalisation for the part memory (EXPLORE, 2026-10-16; plan: PLAN.md addendum 4).

Failure analysis showed "hub" objects: one object's memory scores high on every clutter frame and absorbs the errors
(e.g. five markers that differ only in colour all predicted as one of them). Speaker verification handles the same
problem with cohort score normalisation (Z-norm): each enrolled model's score is standardised by its score
distribution on impostor recordings. Here the impostors are clutter frames of *other people* (users outside the
evaluated split, so no labels of the evaluated user are touched):

  s'_o = (s_o - mu_o) / sd_o,  mu_o / sd_o = mean / std of object o's top-k part score over the cohort frames

  eval <prefix> <cohort prefix | 'folds'> <out json> [n_cohort=1500]
      offline check on packed patch samples (train_evidence_head.py pack); 'folds' takes the cohort from other users
      of the same packed split (user-disjoint), otherwise from another packed split
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def topk_scores(X, parts, n_obj, topk=8, chunk=128):
    """X [N,T,D] half; parts [n_obj*K,D] half grouped by object (equal K). Returns [N,n_obj] top-k mean scores."""
    import torch
    K = len(parts) // n_obj; out = []
    for i in range(0, len(X), chunk):
        s = (X[i:i + chunk].half() @ parts.half().T).view(len(X[i:i + chunk]), X.shape[1], n_obj, K).amax(3).float()
        out.append(s.topk(topk, dim=1).values.mean(1))
    return torch.cat(out)


def cohort_stats(Xc, parts, n_obj, topk=8):
    s = topk_scores(Xc, parts, n_obj, topk)
    return s.mean(0), s.std(0)


def evaluate(prefix, cohort, out, n_cohort=1500):
    import torch
    from train_evidence_head import load, rules
    d = load(prefix); n_cohort = int(n_cohort)
    users = sorted(set(d['user'])); ua = np.array(d['user'])
    if cohort != 'folds':
        dc = load(cohort); Xc_all = dc['X']
    g = torch.Generator().manual_seed(0)
    res = {k: [] for k in ('top8', 'zm', 'zn', 'top8_fuse', 'zn_fuse')}; hub = []
    for ui, u in enumerate(users):
        idx = torch.as_tensor(np.flatnonzero(ua == u), device='cuda')
        if cohort == 'folds':
            pool = np.flatnonzero(ua != u)
            pick = pool[torch.randperm(len(pool), generator=g)[:n_cohort].numpy()]
            Xc = d['X'][torch.as_tensor(pick, device='cuda')]
        else:
            Xc = Xc_all[torch.randperm(len(Xc_all), generator=g)[:n_cohort].cuda()]
        o = int(d['pvalid'][d['uidx'][idx[0]]].sum()); P = d['P'][d['uidx'][idx[0]], :o]
        parts = P.reshape(-1, P.shape[-1])
        mu, sd = cohort_stats(Xc, parts, o)
        base = rules(d, idx)['top8'][:, :o]; full = d['full'][idx][:, :o]; y = d['y'][idx]

        def z(s):
            return (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
        arms = dict(top8=base, zm=base - mu, zn=(base - mu) / (sd + 1e-6), top8_fuse=z(base) + z(full),
                    zn_fuse=z((base - mu) / (sd + 1e-6)) + z(full))
        for k, s in arms.items():
            res[k].append(float((s.argmax(1) == y).float().mean()))
        wrong = base.argmax(1)[base.argmax(1) != y]
        hub.append(float(torch.bincount(wrong, minlength=o).max() / max(len(wrong), 1)) if len(wrong) else 0.)
    summ = {k: round(100 * float(np.mean(v)), 2) for k, v in res.items()}
    Path(out).write_text(json.dumps(dict(prefix=prefix, cohort=cohort, n_cohort=n_cohort, user_mean=summ,
                                         per_user={k: dict(zip(users, v)) for k, v in res.items()},
                                         hub_share=dict(zip(users, hub))), indent=1))
    print('user-mean acc', summ, '| mean share of errors absorbed by one object', round(float(np.mean(hub)), 3), flush=True)


if __name__ == '__main__':
    dict(eval=evaluate)[sys.argv[1]](*sys.argv[2:])
