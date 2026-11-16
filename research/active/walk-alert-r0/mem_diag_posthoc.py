"""Post-hoc oracle diagnostics for the memory result (descriptive only, not a gate).
Arms in cache_mem_diag: 0 single frame, 1 memory, 2 memory without past-frame GT dynamic points, 3 memory with GT ground."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, walk_alert_policy_sanpo as w
w.V2 = True; w.CACHE_DIR = 'cache_mem_diag'; w.BW_SPEEDS.update(w.bw_speeds())
work = sys.argv[1]; data = w.load(work); gi = len(w.GRIDS['B1']) - 1; out = {}
for kind in ('blind', 'sighted'):
    eps_by = {s: w.episodes(s, r, kind) for s, r in data.items()}
    tot = {a: sum(w.score_table(eps_by, 'B1', a).values()) for a in range(4)}
    cap = tot[0][gi][0] / tot[0][gi][1] + w.CAP_MARGIN_V2
    res = {}
    for a, T in tot.items():
        i = w._pick(T, cap)
        res[a] = dict(floor=w._summ(T[gi]), fa_per_km_at_tau5=float(T[gi][3] / T[gi][4]),
                      operating=dict(tau=float(w.GRIDS['B1'][i]), **w._summ(T[i])) if i is not None else None,
                      curve=[(float(t), *map(float, w._rates(T[k]))) for k, t in enumerate(w.GRIDS['B1'])])
        o = res[a]['operating']
        print(kind, a, 'floor', res[a]['floor']['late'], '/', res[a]['floor']['avoidable'], 'FA/km@5', round(res[a]['fa_per_km_at_tau5'], 1),
              'op', (o['tau'], o['late'], round(o['fa_per_km'], 2)) if o else None, flush=True)
    out[kind] = dict(cap=cap, arms=res)
json.dump(out, open(os.path.join(work, 'mem_diag_posthoc.json'), 'w'), indent=0)
