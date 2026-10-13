"""Train/evaluate the learned occupancy readout head with leave-family-out CV (plan: adt-eval-20261013/PLAN_HEAD.md).

Small CNN on the 4x18x30 gravity-frame grid + scalars -> HEAD/BODY logits. Trained on variants a and b together
(x-flip augmentation). Per held-out fold: threshold chosen on the training folds so that the confirmed recall
(GT z50 <= 1.5 m) of variant a matches v6's on those folds; then held-out FA/recall vs v6 (variants a and b).
Usage: adt_head_train_dev_20261013.py <adt_eval.json> <features dir> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2

FAMILIES = [('clean',), ('decoration', 'golden'), ('meal', 'work'), ('multiuser',)]


def fam(seq):
    for k, keys in enumerate(FAMILIES):
        if any(f'release_{x}' in seq for x in keys):
            return k
    raise ValueError(seq)


def load(eval_json, feat_dir):
    E = json.loads(Path(eval_json).read_text()); data = {}
    for seq, v in E.items():
        f = Path(feat_dir) / f'{seq}.npz'
        if not f.exists():
            continue
        Z = np.load(f); fr = v['frames']
        lab = np.full((len(fr), 2), -1, np.int8); z50 = np.full((len(fr), 2), np.nan)
        for i, x in enumerate(fr):
            for q in (0, 1):
                t = x['truth'][q]
                if t['n'] >= 150: lab[i, q] = 1; z50[i, q] = t['z50']
                elif t['n'] == 0: lab[i, q] = 0
        ro_a = np.array([[f_['arms']['chain'][q] for q in (0, 1)] for f_ in fr], bool)     # [frame, q, (v3, v6)]
        data[seq] = dict(fam=fam(seq), lab=lab, z50=z50, grid={'a': Z['grid_a'].astype(np.float32), 'b': Z['grid_b'].astype(np.float32)},
                         scal={'a': Z['scal_a'], 'b': Z['scal_b']}, v6={'a': ro_a[:, :, 1], 'b': Z['readout_b'][:, :, 1]})
    return data


def net():
    import torch.nn as nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten())
            self.f = nn.Sequential(nn.Linear(32 * 15 + 8, 64), nn.ReLU(), nn.Linear(64, 2))
        def forward(self, g, s):
            import torch
            return self.f(torch.cat([self.c(g), s], 1))
    return Head()


def metrics(data, seqs, alerts_fn):
    fa = [0, 0]; rc = [0, 0]
    for s in seqs:
        a = alerts_fn(s); d = data[s]
        a = np.stack([V2.confirm(a[:, 0]), V2.confirm(a[:, 1])], 1)
        neg = d['lab'] == 0; pos = (d['lab'] == 1) & (d['z50'] <= 1.5)
        fa[0] += int(a[neg].sum()); fa[1] += int(neg.sum()); rc[0] += int(a[pos].sum()); rc[1] += int(pos.sum())
    return fa[0] / max(fa[1], 1), rc[0] / max(rc[1], 1), fa, rc


def main(eval_json, feat_dir, out):
    import torch
    data = load(eval_json, feat_dir); dev = 'cuda'; res = dict(folds=[])
    probs = {}
    for k in range(len(FAMILIES)):
        tr = [s for s in data if data[s]['fam'] != k]; te = [s for s in data if data[s]['fam'] == k]
        G, Sc, Y = [], [], []
        for s in tr:
            for v in 'ab':
                G.append(data[s]['grid'][v]); Sc.append(data[s]['scal'][v]); Y.append(data[s]['lab'])
        G, Sc, Y = (torch.tensor(np.concatenate(a)).to(dev) for a in (G, Sc, Y))
        torch.manual_seed(k); m = net().to(dev); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-3)
        for ep in range(40):
            perm = torch.randperm(len(G), device=dev)
            for b in range(0, len(G), 256):
                i = perm[b:b + 256]; g = G[i]
                if ep % 2:
                    g = torch.flip(g, dims=[2])                    # lateral mirror
                y = Y[i].float(); msk = (Y[i] >= 0).float()
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(m(g, Sc[i]), y.clamp(min=0), reduction='none') * msk).sum() / msk.sum()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            for s in data:                                     # training folds (threshold) and held-out fold
                for v in 'ab':
                    p = torch.sigmoid(m(torch.tensor(data[s]['grid'][v]).to(dev), torch.tensor(data[s]['scal'][v]).to(dev))).cpu().numpy()
                    probs[(k, s, v)] = p
        # threshold on training folds: match v6 recall (variant a)
        _, v6_rc_tr, _, _ = metrics(data, tr, lambda s: data[s]['v6']['a'])
        best = None
        for thr in 1 / (1 + np.exp(-np.linspace(-3, 10, 131))):   # logit grid up to ~0.99995 (first run capped at 0.95)
            _, rc_, _, _ = metrics(data, tr, lambda s: probs[(k, s, 'a')] >= thr)
            if rc_ >= v6_rc_tr:
                best = thr
        thr = best if best is not None else .05
        row = dict(fold=k, test=te, thr=float(thr), v6_recall_train=v6_rc_tr)
        for v in 'ab':
            row[f'head_{v}'] = metrics(data, te, lambda s: probs[(k, s, v)] >= thr)
            row[f'v6_{v}'] = metrics(data, te, lambda s: data[s]['v6'][v])
        res['folds'].append(row)
        print(k, 'thr %.2f' % thr, ' | '.join(f"{n} FA {row[n][0]:.3f} rec {row[n][1]:.3f}" for n in ('head_a', 'v6_a', 'head_b', 'v6_b')), flush=True)
    for v in 'ab':
        for n in ('head', 'v6'):
            fa = [sum(r[f'{n}_{v}'][2][i] for r in res['folds']) for i in (0, 1)]; rc = [sum(r[f'{n}_{v}'][3][i] for r in res['folds']) for i in (0, 1)]
            res[f'pooled_{n}_{v}'] = dict(fa=fa[0] / fa[1], fa_n=fa, recall=rc[0] / rc[1], rc_n=rc)
            print(f'POOLED {n} variant {v}: FA {fa[0]}/{fa[1]}={fa[0]/fa[1]:.4f}  recall(z<=1.5) {rc[0]}/{rc[1]}={rc[0]/rc[1]:.4f}')
    Path(out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main(*sys.argv[1:4])
