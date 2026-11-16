"""MEMORY_DEGRADE_PROTOCOL_20261117.md evaluation. Arms in cache_mem_degrade: 0 single, 1 memory, 2 oracle no-dynamic,
3 recall 1.0 + erosion, 4 recall 0.9 + erosion, 5 recall 0.8 + erosion (primary), 6 recall 0.6 + erosion.
R_k = (FA1 - FA_k) / (FA1 - FA2) at tau = 5, blind walkers; session bootstrap."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, walk_alert_policy_sanpo as w
w.V2 = True; w.CACHE_DIR = 'cache_mem_degrade'; w.BW_SPEEDS.update(w.bw_speeds())
NA = 7; LABEL = ['single', 'memory', 'oracle', 'r1.0+erode', 'r0.9+erode', 'r0.8+erode', 'r0.6+erode']
work = sys.argv[1]; data = w.load(work); gi = len(w.GRIDS['B1']) - 1; assert w.GRIDS['B1'][gi] == 5.0
report = dict(protocol='MEMORY_DEGRADE_PROTOCOL_20261117.md', sessions=len(data), arms=LABEL)
for kind in ('blind', 'sighted'):
    eps_by = {s: w.episodes(s, r, kind) for s, r in data.items()}
    tabs = {a: w.score_table(eps_by, 'B1', a) for a in range(NA)}; tot = {a: sum(t.values()) for a, t in tabs.items()}
    cap = tot[0][gi][0] / tot[0][gi][1] + w.CAP_MARGIN_V2; op = {a: w._pick(tot[a], cap) for a in tabs}
    fa5 = {a: float(tot[a][gi][3] / tot[a][gi][4]) for a in tabs}
    S = list(data); A = np.array([np.r_[[tabs[a][s][gi] for a in range(NA)]].ravel() for s in S])
    rng = np.random.default_rng(20261116); Rb = {a: [] for a in range(3, NA)}
    for _ in range(2000):
        x = A[rng.integers(len(S), size=len(S))].sum(0); f = [x[5 * a + 3] / x[5 * a + 4] for a in range(NA)]
        if f[1] - f[2] > 0:
            for a in Rb:
                Rb[a].append((f[1] - f[a]) / (f[1] - f[2]))
    pct = lambda z: [float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))] if len(z) else None
    res = dict(cap=cap, fa_per_km_tau5=fa5, floor={a: w._summ(tot[a][gi]) for a in tabs},
               recovery={a: (fa5[1] - fa5[a]) / (fa5[1] - fa5[2]) for a in Rb}, recovery_ci95={a: pct(Rb[a]) for a in Rb},
               operating={a: (dict(tau=float(w.GRIDS['B1'][i]), **w._summ(tot[a][i])) if i is not None else None) for a, i in op.items()},
               curves={a: [(float(t), *map(float, w._rates(tot[a][k]))) for k, t in enumerate(w.GRIDS['B1'])] for a in tabs})
    report[kind] = res
    for a in range(NA):
        o = res['operating'][a]
        print(kind, a, LABEL[a], 'floor', res['floor'][a]['late'], round(res['floor'][a]['late_rate'], 4), 'FA@5', round(fa5[a], 1),
              'R', (round(res['recovery'][a], 3), [round(v, 3) for v in res['recovery_ci95'][a]]) if a >= 3 else '',
              'op', (o['tau'], o['late'], round(o['fa_per_km'], 2)) if o else None, flush=True)
json.dump(report, open(os.path.join(work, 'report_mem_degrade.json'), 'w'), indent=0, default=float)
