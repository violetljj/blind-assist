"""Evaluate heading-uncertainty conditions and mitigations on the frozen exact-0.9 m ledger."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_tristate_event_dev as E  # noqa: E402
import cnh_tristate_dev as R  # noqa: E402

OUT = R.WORK/'cnh-heading-uncertainty-dev-20261007'
R3 = R.WORK/'cnh-tristate-dev-r3-20261006'
SIGMAS = (2, 4, 8)


def arm_scores(raw):
    """raw[3,13,2] -> single, dual smoothed max scores [13]."""
    sm = R.smooth(raw).max(-1)
    return sm[0], sm[1:].max(0)


def main():
    rows = R.read(R3/'rows.json')
    with np.load(R3/'online.npz') as z:
        r3gate, thresholds = z['gate'][:, :, 1, :], z['thresholds']
    units = {int(f.stem[4:]): f for f in (OUT/'units').glob('unit*.npz') if not f.name.endswith('.tmp.npz')}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]; sub = [rows[i] for i in keep]; n = len(sub)
    names = ['exact']+[f's{s}{t}' for s in SIGMAS for t in 'cpm']
    sc = {(a, nm): np.zeros((n, 13)) for a in ('single', 'dual') for nm in names}
    err = {s: np.zeros((n, 16)) for s in SIGMAS}; parity = []
    cache = {}
    for j, r in enumerate(sub):
        u, cfg = r['unit'], r['config']
        if u not in cache:
            cache.clear(); cache[u] = dict(np.load(units[u])); parity.append(float(cache[u]['exact_vs_stored_max_abs']))
        z = cache[u]
        for nm in names:
            sc['single', nm][j], sc['dual', nm][j] = arm_scores(z[f'{nm}_raw'][:, cfg])
        for s in SIGMAS:
            err[s][j] = z[f's{s}_err'][cfg]
    gate = {'single': r3gate[keep, :, 0], 'dual': r3gate[keep, :, 1]}
    g = {k: v[keep] for k, v in E.load_geometry(rows).items()}
    contact, control = g['contact'], g['control']
    modes = np.array([r['mode'] for r in sub]); uid = np.array([r['unit'] for r in sub])
    uu, inv = np.unique(uid, return_inverse=True)
    strat = {r['unit']: f"{r['batch']}/{r['mode']}/{r['turn']}" for r in sub}; srow = np.array([strat[u] for u in uu])
    rng = np.random.default_rng(2026100782); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(srow):
        ids = np.flatnonzero(srow == s); draw = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(draw[k], minlength=len(ids))
    agg = lambda x, m: np.bincount(inv, weights=np.where(m, x, 0.), minlength=len(uu))

    def evaluate(alarm_score, clear_score, gt, tau):
        alarm = alarm_score >= R.THRESHOLD; clear = ~alarm & gt & (clear_score <= tau)
        ix = E.causal_index(g['fraction']); rr = np.flatnonzero(contact)
        timely = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]
        st = np.full(n, -1, np.int8); st[rr] = np.where(timely, 0, np.where(clear[rr, ix[rr]], 2, 1))
        unknown = ~alarm & ~clear
        return st, unknown[:, 2:12].sum(1)*.2, alarm[:, 2:12].sum(1)*.2

    variants = {'exact': ('exact', 'exact', 'exact')}
    for s in SIGMAS:
        c, p, m = f's{s}c', f's{s}p', f's{s}m'
        variants[f'sigma{s}/none'] = (c, c, None)
        variants[f'sigma{s}/union-alarm'] = ('U', 'U', (c, p, m))
        variants[f'sigma{s}/union-clear'] = (c, 'U', (c, p, m))
    res = dict(units=len(uu), sequences=n, events=int(contact.sum()), controls=int(control.sum()),
               exact_vs_stored_max_abs=max(parity), error_rms={str(s): float(np.sqrt((err[s][:, 3:]**2).mean())) for s in SIGMAS},
               groups={}, diffs={})
    groups = {'all': np.ones(n, bool)} | {f'mode{k}': modes == k for k in range(3)}
    j19 = len(thresholds)-1; keep_states = {}
    for a in ('single', 'dual'):
        for vname, (al, cl, triple) in variants.items():
            if vname == 'exact':
                A_ = C_ = sc[a, 'exact']
            else:
                U = np.maximum.reduce([sc[a, t] for t in triple]) if triple else None
                A_ = U if al == 'U' else sc[a, al]; C_ = U if cl == 'U' else sc[a, cl]
            for gname, mask in groups.items():
                G = res['groups'].setdefault(gname, dict(events=int((mask & contact).sum()), controls=int((mask & control).sum()), arms={}))
                curve = []
                for jj, tau in enumerate(thresholds):
                    st, useconds, aseconds = evaluate(A_, C_, gate[a], tau)
                    if gname == 'all' and jj == j19:
                        keep_states[a, vname] = (st, useconds, aseconds)
                    ctl = mask & control
                    curve.append(dict(index=jj, timely=int(((st == 0) & mask).sum()), unknown_miss=int(((st == 1) & mask).sum()),
                                      silent=int(((st == 2) & mask).sum()),
                                      unknown_time=float(useconds[ctl].sum()/(2.*ctl.sum())),
                                      obstacle_time=float(aseconds[ctl].sum()/(2.*ctl.sum()))))
                G['arms'][f'{a}/{vname}'] = curve
    for a in ('single', 'dual'):
        s0, u0, a0 = keep_states[a, 'exact']
        for vname in variants:
            if vname == 'exact':
                continue
            s1, u1, a1 = keep_states[a, vname]; d = {}
            for name, code in (('timely', 0), ('silent', 2)):
                x = agg((s1 == code).astype(float), contact)-agg((s0 == code).astype(float), contact)
                d[name] = dict(diff=float(x.sum()), ci95=np.quantile(boot@x, [.025, .975]).tolist())
            for name, (v1, v0) in (('unknown_time_pp', (u1, u0)), ('obstacle_time_pp', (a1, a0))):
                x = agg(v1-v0, control); dd = agg(np.full(n, 2.), control)
                d[name] = dict(diff=100*float(x.sum()/dd.sum()), ci95=(100*np.quantile((boot@x)/(boot@dd), [.025, .975])).tolist())
            res['diffs'][f'{a}/{vname}'] = d
    (OUT/'result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)+'\n', encoding='utf8')
    print('units', res['units'], 'events', res['events'], 'controls', res['controls'], 'exact-vs-stored max', round(res['exact_vs_stored_max_abs'], 4), 'rms', res['error_rms'])
    for gname, G in res['groups'].items():
        print(f"== {gname} events {G['events']} controls {G['controls']}")
        for arm, curve in G['arms'].items():
            p = curve[j19]
            print(f"  {arm:24s} T/U/S {p['timely']}/{p['unknown_miss']}/{p['silent']}  unknown {100*p['unknown_time']:.2f}%  obstacle {100*p['obstacle_time']:.2f}%")
    for k, v in res['diffs'].items():
        print(f"  DIFF {k}: timely {v['timely']['diff']:+.0f} {np.round(v['timely']['ci95'],1)} silent {v['silent']['diff']:+.0f} unknown {v['unknown_time_pp']['diff']:+.2f}pp obstacle {v['obstacle_time_pp']['diff']:+.2f}pp {np.round(v['obstacle_time_pp']['ci95'],2)}")


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def frontier():
    """Timely alarms at matched control obstacle (false-alarm) time, sweeping the alarm threshold.
    Separates 'widening helps' from 'any lower threshold helps'. Descriptive."""
    rows = R.read(R3/'rows.json')
    units = {int(f.stem[4:]): f for f in (OUT/'units').glob('unit*.npz') if not f.name.endswith('.tmp.npz')}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]; sub = [rows[i] for i in keep]; n = len(sub)
    g = {k: v[keep] for k, v in E.load_geometry(rows).items()}
    contact, control = g['contact'], g['control']; ix = E.causal_index(g['fraction']); rr = np.flatnonzero(contact)
    names = ['exact']+[f's{s}{t}' for s in SIGMAS for t in 'cpm']
    sc = {(a, nm): np.zeros((n, 13)) for a in ('single', 'dual') for nm in names}; cache = {}
    for j, r in enumerate(sub):
        u, cfg = r['unit'], r['config']
        if u not in cache:
            cache.clear(); cache[u] = dict(np.load(units[u]))
        for nm in names:
            sc['single', nm][j], sc['dual', nm][j] = arm_scores(cache[u][f'{nm}_raw'][:, cfg])
    targets = (.025, .05, .10, .15, .20); out = {}
    for a in ('single', 'dual'):
        conds = {'exact': sc[a, 'exact']}
        for s in SIGMAS:
            conds[f'sigma{s}/none'] = sc[a, f's{s}c']
            conds[f'sigma{s}/union'] = np.maximum.reduce([sc[a, f's{s}{t}'] for t in 'cpm'])
        for name, S in conds.items():
            ctl = S[control][:, 2:12]; peak = np.maximum.accumulate(S, axis=1)[rr, ix[rr]]
            thetas = np.quantile(ctl, np.linspace(.5, .999, 400))
            fa = np.array([(ctl >= t).mean() for t in thetas]); tim = np.array([(peak >= t).sum() for t in thetas])
            row = {}
            for T in targets:
                ok = fa <= T
                row[f'{T:.3f}'] = int(tim[ok].max()) if ok.any() else None
            out[f'{a}/{name}'] = row
    (OUT/'frontier.json').write_text(json.dumps(out, indent=1)+'\n', encoding='utf8')
    print('timely /', int(contact.sum()), 'at control obstacle-time <= target (alarm threshold swept)')
    print(f"{'':22s}"+''.join(f'{T:>8.1%}' for T in targets))
    for k, row in out.items():
        print(f'{k:22s}'+''.join(f"{(v if v is not None else '-'):>8}" for v in row.values()))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'frontier':
    frontier()
