"""Post-hoc descriptive: pooled held-out curves (B1 has no fitting; P1 per-fold model, common theta)."""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__file__))
import numpy as np, walk_alert_policy_sanpo as w
w.V2 = True; w.BW_SPEEDS.update(w.bw_speeds())
work = sys.argv[1]; data = w.load(work); out = {}
for kind in ('blind', 'sighted'):
    for arm in (0, 1):
        eps_by = {s: w.episodes(s, r, kind) for s, r in data.items()}
        b1 = w.score_table(eps_by, 'B1', arm); B1 = sum(b1.values())
        P = np.zeros((len(w.GRIDS['P1']), 5)); folds = {s: w.fold_of(s) for s in data}
        for f in range(w.CV_K):
            tr = [s for s in data if folds[s] != f]; ho = [s for s in data if folds[s] == f]
            m, _ = w.fit_model([e for s in tr for e in eps_by[s]], arm)
            t = w.score_table({s: eps_by[s] for s in ho}, 'P1', arm, m); P += sum(t.values())
        def curve(T, g): return [(float(x), round(a[0] / a[1], 3), round(a[3] / a[4], 2)) for x, a in zip(g, T)]
        out[f'{kind}/{arm}'] = dict(B1=curve(B1, w.GRIDS['B1']), P1=curve(P, w.GRIDS['P1']))
        print(kind, arm, flush=True)
json.dump(out, open(f'{work}/cv_v3_posthoc_curves.json', 'w'), indent=0)
