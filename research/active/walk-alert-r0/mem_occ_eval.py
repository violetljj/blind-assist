"""MEMORY_OCC_PROTOCOL_20261117.md evaluation. Arms in cache_mem_occ: 0 single, 1 raw memory, 2 oracle no-dynamic memory,
3 log-odds grid l>=2.0 (primary), 4 grid l>=1.5, 5 grid l>=3.0. Blind walkers, B1.
Success (arm 3 vs arm 0): floor reduction >= 10 pp with bootstrap CI lower bound > 0, and both FA episodes/km and
FA frames/km at tau = 5 <= 1.3 x single frame. Otherwise the route is closed."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, walk_alert_policy_sanpo as w
w.V2 = True; w.CACHE_DIR = 'cache_mem_occ'; w.BW_SPEEDS.update(w.bw_speeds())
NA = 6; LABEL = ['single', 'memory', 'oracle', 'occ l2.0 (primary)', 'occ l1.5', 'occ l3.0']
work = sys.argv[1]; data = w.load(work); gi = len(w.GRIDS['B1']) - 1; assert w.GRIDS['B1'][gi] == 5.0
report = dict(protocol='MEMORY_OCC_PROTOCOL_20261117.md', sessions=len(data), arms=LABEL)


def fa_frames(ep, arm):
    al = w.policy_alerts('B1', 5.0, ep, arm) & ep['valid']; r = ep['rows']
    clean = ~(r[:, 0] <= w.H) & ~(r[:, 2] > .5) & ~w.weak_mask(ep)
    return int((al & clean).sum())


for kind in ('blind', 'sighted'):
    eps_by = {s: w.episodes(s, r, kind) for s, r in data.items()}
    tabs = {a: w.score_table(eps_by, 'B1', a) for a in range(NA)}; tot = {a: sum(t.values()) for a, t in tabs.items()}
    S = list(data)
    frm = {a: np.array([sum(fa_frames(e, a) for e in eps_by[s]) for s in S], float) for a in range(NA)}
    km = np.array([tabs[0][s][gi][4] for s in S])
    cap = tot[0][gi][0] / tot[0][gi][1] + w.CAP_MARGIN_V2; op = {a: w._pick(tot[a], cap) for a in tabs}
    fa5 = {a: float(tot[a][gi][3] / tot[a][gi][4]) for a in tabs}
    fr5 = {a: float(frm[a].sum() / km.sum()) for a in tabs}
    A = np.array([np.r_[[tabs[a][s][gi] for a in range(NA)]].ravel() for s in S])
    F = np.stack([frm[a] for a in range(NA)], 1)
    rng = np.random.default_rng(20261116); boot = {a: dict(fl=[], ep=[], fr=[]) for a in range(1, NA)}
    for _ in range(2000):
        k = rng.integers(len(S), size=len(S)); x = A[k].sum(0); f = F[k].sum(0); kk = km[k].sum()
        for a in boot:
            boot[a]['fl'].append((x[0] - x[5 * a]) / x[1])
            if x[3] > 0:
                boot[a]['ep'].append((x[5 * a + 3] / x[5 * a + 4]) / (x[3] / x[4]))
            if f[0] > 0:
                boot[a]['fr'].append(f[a] / f[0])
    pct = lambda z: [float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))] if len(z) else None
    res = dict(cap=cap, fa_episodes_per_km_tau5=fa5, fa_frames_per_km_tau5=fr5,
               floor={a: w._summ(tot[a][gi]) for a in tabs},
               vs_single={a: dict(floor_reduction=float((tot[0][gi][0] - tot[a][gi][0]) / tot[0][gi][1]), floor_reduction_ci95=pct(boot[a]['fl']),
                                  episode_ratio=fa5[a] / fa5[0], episode_ratio_ci95=pct(boot[a]['ep']),
                                  frame_ratio=fr5[a] / fr5[0], frame_ratio_ci95=pct(boot[a]['fr'])) for a in boot},
               operating={a: (dict(tau=float(w.GRIDS['B1'][i]), **w._summ(tot[a][i])) if i is not None else None) for a, i in op.items()},
               curves={a: [(float(t), *map(float, w._rates(tot[a][k]))) for k, t in enumerate(w.GRIDS['B1'])] for a in tabs})
    v = res['vs_single'][3]
    res['decision_primary'] = ('SUCCESS' if v['floor_reduction'] >= .10 and v['floor_reduction_ci95'][0] > 0
                               and v['episode_ratio'] <= 1.3 and v['frame_ratio'] <= 1.3 else 'NOT_SUCCESS')
    report[kind] = res
    for a in range(NA):
        o = res['operating'][a]; vs = res['vs_single'].get(a)
        print(kind, a, LABEL[a], 'floor %d/%d %.4f' % (res['floor'][a]['late'], res['floor'][a]['avoidable'], res['floor'][a]['late_rate']),
              'FAep@5 %.1f FAfr@5 %.1f' % (fa5[a], fr5[a]),
              ('ratios ep %.2f %s fr %.2f %s floor-red %.3f %s' % (vs['episode_ratio'], [round(x, 2) for x in vs['episode_ratio_ci95']],
                                                                   vs['frame_ratio'], [round(x, 2) for x in vs['frame_ratio_ci95']],
                                                                   vs['floor_reduction'], [round(x, 3) for x in vs['floor_reduction_ci95']])) if vs else '',
              'op', (o['tau'], o['late'], round(o['fa_per_km'], 2)) if o else None, flush=True)
    print(kind, 'decision (primary arm 3):', res['decision_primary'], flush=True)
json.dump(report, open(os.path.join(work, 'report_mem_occ.json'), 'w'), indent=0, default=float)
