"""MEMORY_CARVE_PROTOCOL_20261116.md evaluation. Arms in cache_mem_carve: 0 single, 1 memory, 2 memory + free-space carving,
3 memory + oracle no-dynamic (reference). Primary: recovery R = (FA1 - FA2) / (FA1 - FA3) at tau = 5 (blind walkers)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, walk_alert_policy_sanpo as w
w.V2 = True; w.CACHE_DIR = 'cache_mem_carve'; w.BW_SPEEDS.update(w.bw_speeds())
work = sys.argv[1]; data = w.load(work); gi = len(w.GRIDS['B1']) - 1; assert w.GRIDS['B1'][gi] == 5.0
report = dict(protocol='MEMORY_CARVE_PROTOCOL_20261116.md', sessions=len(data))
for kind in ('blind', 'sighted'):
    eps_by = {s: w.episodes(s, r, kind) for s, r in data.items()}
    tabs = {a: w.score_table(eps_by, 'B1', a) for a in range(4)}; tot = {a: sum(t.values()) for a, t in tabs.items()}
    cap = tot[0][gi][0] / tot[0][gi][1] + w.CAP_MARGIN_V2; op = {a: w._pick(tot[a], cap) for a in tabs}
    fa5 = {a: float(tot[a][gi][3] / tot[a][gi][4]) for a in tabs}
    S = list(data); A = np.array([np.r_[[tabs[a][s][gi] for a in range(4)]].ravel() for s in S])   # 4 arms x 5
    o0, o2 = op[0], (gi if op[2] is None else op[2])
    B = np.array([np.r_[tabs[0][s][o0], tabs[2][s][o2]] for s in S])
    rng = np.random.default_rng(20261116); Rb, Fl, Rf = [], [], []
    for _ in range(2000):
        k = rng.integers(len(S), size=len(S)); x = A[k].sum(0); y = B[k].sum(0)
        f = [x[5 * a + 3] / x[5 * a + 4] for a in range(4)]
        if f[1] - f[3] > 0:
            Rb.append((f[1] - f[2]) / (f[1] - f[3]))
        Fl.append((x[0] - x[10]) / x[1])
        if y[3] > 0 and op[2] is not None:
            Rf.append((y[8] / y[9]) / (y[3] / y[4]))
    pct = lambda z: [float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))] if len(z) else None
    res = dict(cap=cap, fa_per_km_tau5=fa5,
               floor={a: w._summ(tot[a][gi]) for a in tabs},
               operating={a: (dict(tau=float(w.GRIDS['B1'][i]), **w._summ(tot[a][i])) if i is not None else None) for a, i in op.items()},
               recovery=(fa5[1] - fa5[2]) / (fa5[1] - fa5[3]) if fa5[1] != fa5[3] else None, recovery_ci95=pct(Rb),
               secondary=dict(floor_reduction=float((tot[0][gi][0] - tot[2][gi][0]) / tot[0][gi][1]), floor_reduction_ci95=pct(Fl),
                              fa_ratio_at_operating=(w._rates(tot[2][op[2]])[1] / w._rates(tot[0][op[0]])[1]) if op[2] is not None else None,
                              fa_ratio_ci95=pct(Rf)),
               curves={a: [(float(t), *map(float, w._rates(tot[a][k]))) for k, t in enumerate(w.GRIDS['B1'])] for a in tabs})
    report[kind] = res
    print(kind, json.dumps(dict(floor={a: (f['late'], round(f['late_rate'], 4)) for a, f in res['floor'].items()},
                                fa5={a: round(v, 1) for a, v in fa5.items()}, recovery=res['recovery'], recovery_ci95=res['recovery_ci95'],
                                operating={a: (o['tau'], o['late'], round(o['fa_per_km'], 2)) if o else None for a, o in res['operating'].items()},
                                secondary=res['secondary']), default=float), flush=True)
json.dump(report, open(os.path.join(work, 'report_mem_carve.json'), 'w'), indent=0, default=float)
