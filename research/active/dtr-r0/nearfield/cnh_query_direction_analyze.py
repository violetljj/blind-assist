"""Evaluate query-direction conditions on the frozen exact-0.9 m event ledger (descriptive)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_tristate_event_dev as E  # noqa: E402
import cnh_tristate_dev as R  # noqa: E402

OUT = R.WORK/'cnh-query-direction-dev-20261006'
R3 = R.WORK/'cnh-tristate-dev-r3-20261006'
CONDS = ('exact', 'est', 'bias+10', 'bias-10')


def main():
    rows = R.read(R3/'rows.json')
    with np.load(R3/'online.npz') as z:
        r3score, r3gate, thresholds = z['score'], z['gate'][:, :, 1, :], z['thresholds']
    units = {int(f.stem[4:]): f for f in (OUT/'units').glob('unit*.npz')}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]; sub = [rows[i] for i in keep]; n = len(sub)
    score = {(a, c): np.zeros((n, 13)) for a in ('single', 'dual') for c in CONDS}
    err = {c: np.zeros((n, 16)) for c in CONDS[1:]}
    cache = {}
    for j, r in enumerate(sub):
        u, cfg = r['unit'], r['config']
        if u not in cache:
            cache.clear(); cache[u] = dict(np.load(units[u]))
        z = cache[u]
        for c in CONDS[1:]:
            sm = R.smooth(z[f'{c}_raw'][:, cfg]).max(-1)
            score['single', c][j] = sm[0]; score['dual', c][j] = sm[1:].max(0); err[c][j] = z[f'{c}_err'][cfg]
    score['single', 'exact'] = r3score[keep, :, 0]; score['dual', 'exact'] = r3score[keep, :, 1]
    gate = {'single': r3gate[keep, :, 0], 'dual': r3gate[keep, :, 1]}
    g = {k: v[keep] for k, v in E.load_geometry(rows).items()}
    modes = np.array([r['mode'] for r in sub]); uid = np.array([r['unit'] for r in sub])
    uu, inv = np.unique(uid, return_inverse=True)
    strat = {r['unit']: f"{r['batch']}/{r['mode']}/{r['turn']}" for r in sub}; srow = np.array([strat[u] for u in uu])
    rng = np.random.default_rng(2026100671); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(srow):
        ids = np.flatnonzero(srow == s); draw = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(draw[k], minlength=len(ids))
    agg = lambda x, m: np.bincount(inv, weights=np.where(m, x, 0.), minlength=len(uu))
    contact, control = g['contact'], g['control']
    res = dict(sequences=n, units=len(uu), query_error={}, groups={}, diffs={})
    for c in CONDS[1:]:
        e = np.abs(err[c][:, 3:])
        res['query_error'][c] = {f'mode{k}': dict(mean_abs=float(e[modes == k].mean()), p95_abs=float(np.quantile(e[modes == k], .95))) for k in range(3) if (modes == k).any()}
    groups = {'all': np.ones(n, bool)} | {f'mode{k}': modes == k for k in range(3)}
    states = {}
    for gname, mask in groups.items():
        G = res['groups'][gname] = dict(events=int((mask & contact).sum()), controls=int((mask & control).sum()), arms={})
        for a in ('single', 'dual'):
            for c in CONDS:
                curve = []
                for j, tau in enumerate(thresholds):
                    st, clear, unknown, alarm = E.partition(score[a, c], gate[a], tau, g['fraction'], contact)
                    if gname == 'all':
                        states[a, c, j] = (st, unknown, alarm)
                    b = E.burden(unknown); nn, dd = agg(b['seconds'], mask & control), agg(b['total_seconds'], mask & control)
                    fa = alarm[:, 2:12].sum(1)*.2
                    curve.append(dict(index=j, timely=int(((st == 0) & mask).sum()), unknown_miss=int(((st == 1) & mask).sum()),
                                      silent=int(((st == 2) & mask).sum()), unknown_time=float(nn.sum()/dd.sum()) if dd.sum() else None,
                                      control_obstacle_time=float(agg(fa, mask & control).sum()/dd.sum()) if dd.sum() else None))
                G['arms'][f'{a}/{c}'] = curve
    j = len(thresholds)-1
    for a in ('single', 'dual'):
        s0, u0, al0 = states[a, 'exact', j]
        for c in CONDS[1:]:
            s1, u1, al1 = states[a, c, j]
            for name, code in (('timely', 0), ('silent', 2)):
                d = agg((s1 == code).astype(float), contact)-agg((s0 == code).astype(float), contact)
                res['diffs'][f'{a} {c}-exact {name}'] = dict(diff=float(d.sum()), ci95=np.quantile(boot@d, [.025, .975]).tolist())
            fa1, fa0 = agg(al1[:, 2:12].sum(1)*.2, control), agg(al0[:, 2:12].sum(1)*.2, control); dd = agg(np.full(n, 2.), control)
            dist = (boot@(fa1-fa0))/(boot@dd)
            res['diffs'][f'{a} {c}-exact control_obstacle_time'] = dict(diff_pp=100*float((fa1-fa0).sum()/dd.sum()), ci95_pp=(100*np.quantile(dist, [.025, .975])).tolist())
    # post-hoc lag-compensated estimate on mode2 units only (amendment2)
    comp = {f: OUT/'units_comp'/f.name for f in units.values() if (OUT/'units_comp'/f.name).exists()}
    if comp:
        m2 = modes == 2; res['mode2_est_comp'] = {}
        cs = {'single': np.zeros((n, 13)), 'dual': np.zeros((n, 13))}; ce = np.zeros((n, 16)); cc = {}
        for jj, r in enumerate(sub):
            if not m2[jj]:
                continue
            u, cfg = r['unit'], r['config']
            if u not in cc:
                cc.clear(); cc[u] = dict(np.load(OUT/'units_comp'/f'unit{u}.npz'))
            sm = R.smooth(cc[u]['est_comp_raw'][:, cfg]).max(-1); cs['single'][jj] = sm[0]; cs['dual'][jj] = sm[1:].max(0); ce[jj] = cc[u]['est_comp_err'][cfg]
        e = np.abs(ce[m2][:, 3:]); res['query_error']['est_comp'] = {'mode2': dict(mean_abs=float(e.mean()), p95_abs=float(np.quantile(e, .95)))}
        for a in ('single', 'dual'):
            st, _, unknown, alarm = E.partition(cs[a], gate[a], thresholds[j], g['fraction'], contact)
            s0, _, _ = states[a, 'exact', j]; s1, _, _ = states[a, 'est', j]
            b = E.burden(unknown); ctl = m2 & control
            out = dict(timely=int(((st == 0) & m2).sum()), unknown_miss=int(((st == 1) & m2).sum()), silent=int(((st == 2) & m2).sum()),
                       unknown_time=float(b['seconds'][ctl].sum()/b['total_seconds'][ctl].sum()),
                       control_obstacle_time=float((alarm[:, 2:12].sum(1)*.2)[ctl].sum()/(2.*ctl.sum())))
            for ref, sr in (('exact', s0), ('est', s1)):
                d = agg(((st == 0) & m2).astype(float), contact)-agg(((sr == 0) & m2).astype(float), contact)
                out[f'timely_minus_{ref}'] = dict(diff=float(d.sum()), ci95=np.quantile(boot@d, [.025, .975]).tolist())
            res['mode2_est_comp'][a] = out
        print('mode2 est_comp', json.dumps(res['mode2_est_comp'], ensure_ascii=False), res['query_error']['est_comp'])
    (OUT/'result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)+'\n', encoding='utf8')
    print('query error', json.dumps(res['query_error']))
    for gname, G in res['groups'].items():
        print(f"== {gname} events {G['events']} controls {G['controls']}")
        for arm, curve in G['arms'].items():
            p = curve[j]
            print(f"  {arm:14s} T/U/S {p['timely']}/{p['unknown_miss']}/{p['silent']} unknown {100*p['unknown_time']:.2f}% ctrl-obstacle {100*p['control_obstacle_time']:.2f}%")
    for k, v in res['diffs'].items():
        print(' ', k, v)


if __name__ == '__main__':
    main()
