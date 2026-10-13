"""Train the learned readout head on ASE synthetic scenes, test on real ADT (plan: adt-eval-20261013/PLAN_ASE_HEAD.md).

Same features/head as adt_head_train_dev_20261013. Threshold chosen on ASE (variant a recall matched to v6 on ASE);
evaluated on all 13 real ADT sequences against v6, variants a and b.
Usage: ase_head_train_dev_20261013.py <ase_eval.json> <ase feature dir> <adt_eval.json> <adt feature dir> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adt_head_train_dev_20261013 import net, metrics


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
        ro_a = np.array([[f_['arms']['chain'][q] for q in (0, 1)] for f_ in fr], bool)
        data[seq] = dict(lab=lab, z50=z50, grid={'a': Z['grid_a'].astype(np.float32), 'b': Z['grid_b'].astype(np.float32)},
                         scal={'a': Z['scal_a'], 'b': Z['scal_b']}, v6={'a': ro_a[:, :, 1], 'b': Z['readout_b'][:, :, 1]})
    return data


def main(ase_eval, ase_feat, adt_eval, adt_feat, out):
    import torch
    tr, te = load(ase_eval, ase_feat), load(adt_eval, adt_feat); dev = 'cuda'
    G, Sc, Y = [], [], []
    for s in tr:
        for v in 'ab':
            G.append(tr[s]['grid'][v]); Sc.append(tr[s]['scal'][v]); Y.append(tr[s]['lab'])
    G, Sc, Y = (torch.tensor(np.concatenate(a)).to(dev) for a in (G, Sc, Y))
    torch.manual_seed(20261013); m = net().to(dev); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-3)
    for ep in range(40):
        perm = torch.randperm(len(G), device=dev)
        for b in range(0, len(G), 256):
            i = perm[b:b + 256]; g = G[i]
            if ep % 2:
                g = torch.flip(g, dims=[2])
            y = Y[i].float(); msk = (Y[i] >= 0).float()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(m(g, Sc[i]), y.clamp(min=0), reduction='none') * msk).sum() / msk.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); probs = {}
    with torch.no_grad():
        for name, D in (('ase', tr), ('adt', te)):
            for s in D:
                for v in 'ab':
                    probs[(name, s, v)] = torch.sigmoid(m(torch.tensor(D[s]['grid'][v]).to(dev), torch.tensor(D[s]['scal'][v]).to(dev))).cpu().numpy()
    _, v6_rc, _, _ = metrics(tr, list(tr), lambda s: tr[s]['v6']['a'])
    thr = .05
    for t in 1 / (1 + np.exp(-np.linspace(-3, 10, 131))):
        if metrics(tr, list(tr), lambda s: probs[('ase', s, 'a')] >= t)[1] >= v6_rc:
            thr = t
    res = dict(threshold=float(thr), v6_recall_ase=v6_rc, ase_frames=sum(len(d['lab']) for d in tr.values()))
    for v in 'ab':
        for n, fn in (('head', lambda s, v=v: probs[('adt', s, v)] >= thr), ('v6', lambda s, v=v: te[s]['v6'][v])):
            fa, rc, fan, rcn = metrics(te, list(te), fn)
            res[f'adt_{n}_{v}'] = dict(fa=fa, recall=rc, fa_n=fan, rc_n=rcn)
            print(f'ADT {n} variant {v}: FA {fan[0]}/{fan[1]}={fa:.4f}  recall(z<=1.5) {rcn[0]}/{rcn[1]}={rc:.4f}')
    print('threshold', round(thr, 4), 'v6 recall on ASE', round(v6_rc, 4))
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
