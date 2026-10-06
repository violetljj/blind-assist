"""Frozen analysis for CNH_RHC_20261007 (CNH_REAL_HEAD_CONFIRM_PROTOCOL_20261007.md)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_real_head_confirm as RC  # noqa: E402
import cnh_tristate_dev as R  # noqa: E402
import cnh_tristate_event_dev as E  # noqa: E402

R3 = R.WORK/'cnh-tristate-dev-r3-20261006'
TARGETS = (.025, .05)
CONDS = [('SYN', 'exact'), ('NAT', 'exact'), ('NAT', 'E1'), ('NAT', 'EMA'), ('ALN', 'exact'), ('ALN', 'E1')]


def arm_scores(raw):
    sm = R.smooth(raw).max(-1)  # [3,C,13]
    return sm[0], sm[1:].max(0)


def load(units):
    rows = dict(unit=[], mirror=[]); sc = {}; gate = {}; geo = {}
    keys = [(a, q) for a, qs in RC.QUERIES.items() for q in qs]
    for u in units:
        z = np.load(RC.OUT/'units'/f'unit{u}.npz'); C = z['SYN_exact_raw'].shape[1]
        rows['unit'] += [u]*C; rows['mirror'] += [bool(z['mirror'])]*C
        for a, q in keys:
            s, d = arm_scores(z[f'{a}_{q}_raw']); sc.setdefault((a, q, 'single'), []).append(s); sc.setdefault((a, q, 'dual'), []).append(d)
        for a in RC.QUERIES:
            g = z[f'{a}_gate']; gate.setdefault((a, 'single'), []).append(g[..., 0]); gate.setdefault((a, 'dual'), []).append(g[..., 1])
        for a in ('SYN', 'NAT'):
            for k in ('category', 'clear_all', 'covered', 'fraction'):
                geo.setdefault((a, k), []).append(z[f'{a}_{k}'])
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    sc, gate, geo = cat(sc), cat(gate), cat(geo)
    truth = {}
    for a in ('SYN', 'NAT'):
        cq = geo[a, 'covered'] & np.isin(geo[a, 'category'], E.CATS)
        depth = np.full(len(cq), -1)
        for k, c in enumerate(E.CATS):
            depth[(cq & (geo[a, 'category'] == c)).any(1)] = k
        truth[a] = dict(contact=cq.any(1), control=geo[a, 'clear_all'].all(1), ix=E.causal_index(geo[a, 'fraction']), depth=depth)
    truth['ALN'] = truth['NAT']
    return dict(unit=np.array(rows['unit']), mirror=np.array(rows['mirror']), sc=sc, gate=gate, truth=truth)


def states(alarm_s, clear_s, gt, T, theta, tau):
    alarm = alarm_s >= theta; clear = ~alarm & gt & (clear_s <= tau)
    rr = np.flatnonzero(T['contact']); ix = T['ix']
    timely = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]
    st = np.full(len(alarm), -1, np.int8); st[rr] = np.where(timely, 0, np.where(clear[rr, ix[rr]], 2, 1))
    return st, (~alarm & ~clear)[:, 2:12].sum(1)*.2, alarm[:, 2:12].sum(1)*.2


def fa_threshold(S, control, target):
    v = np.sort(S[control][:, 2:12].ravel()); n = len(v)
    # smallest theta with mean(S >= theta) <= target: theta just above the value at rank n*(1-target)
    k = int(np.ceil(n*(1-target)))
    return float(np.nextafter(v[min(k, n-1)], np.inf)) if k < n else float(v[-1])+1e-6


def main():
    tau = np.load(R3/'online.npz')['thresholds']; tmax = tau[-1]
    cal, ev = load(RC.CAL), load(RC.EVAL)
    uu, inv = np.unique(ev['unit'], return_inverse=True)
    strat = np.array([f"{int(ev['mirror'][np.flatnonzero(ev['unit'] == u)[0]])}/{u % 2}" for u in uu])
    rng = np.random.default_rng(2026100796); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(strat):
        ids = np.flatnonzero(strat == s); dr = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(dr[k], minlength=len(ids))
    agg = lambda x, m: np.bincount(inv, weights=np.where(m, x, 0.), minlength=len(uu))
    ci = lambda b: np.quantile(b, [.025, .975]).tolist()
    res = dict(units=dict(calibration=len(RC.CAL), evaluation=len(uu)), sequences=int(len(ev['unit'])))
    for a in ('SYN', 'NAT'):
        T = ev['truth'][a]; res[f'{a}_events'] = int(T['contact'].sum()); res[f'{a}_controls'] = int(T['control'].sum())
        res[f'{a}_depth'] = [int((T['depth'] == k).sum()) for k in range(3)]
    # three-state at frozen theta, max tau
    tri = {}; keep = {}
    variants = [(a, q, q) for a, q in CONDS]+[('NAT', 'E1', 'E1-union')]
    for a, q, name in variants:
        T = ev['truth'][a]
        for cfg in ('single', 'dual'):
            S = ev['sc'][a, q, cfg]
            Cl = np.maximum.reduce([ev['sc'][a, k, cfg] for k in ('E1', 'E1p', 'E1m')]) if name == 'E1-union' else S
            curve = []
            for j, t in enumerate(tau):
                st, us, as_ = states(S, Cl, ev['gate'][a, cfg], T, R.THRESHOLD, t)
                curve.append(dict(timely=int((st == 0).sum()), unknown_miss=int((st == 1).sum()), silent=int((st == 2).sum()),
                                  unknown_time=float(us[T['control']].sum()/(2*T['control'].sum())),
                                  obstacle_time=float(as_[T['control']].sum()/(2*T['control'].sum()))))
                if j == len(tau)-1:
                    keep[a, name, cfg] = (st, us, as_)
            tri[f'{a}/{name}/{cfg}'] = dict(max_tau=curve[-1], curve=curve)
    res['three_state'] = tri
    # matched false-alarm thresholds from calibration units
    mf = {}; peaks = {}
    for a, q in CONDS:
        for cfg in ('single', 'dual'):
            Sc, Se = cal['sc'][a, q, cfg], ev['sc'][a, q, cfg]; Tc, Te = cal['truth'][a], ev['truth'][a]
            rr = np.flatnonzero(Te['contact']); peak = np.maximum.accumulate(Se, axis=1)[rr, Te['ix'][rr]]
            for tg in TARGETS:
                th = fa_threshold(Sc, Tc['control'], tg)
                timely = np.zeros(len(Se), bool); timely[rr] = peak >= th; peaks[a, q, cfg, tg] = timely
                mf[f'{a}/{q}/{cfg}/{tg}'] = dict(threshold=th, timely=int(timely.sum()),
                                                 realized_eval_fa=float((Se[Te['control']][:, 2:12] >= th).mean()),
                                                 calibration_fa=float((Sc[Tc['control']][:, 2:12] >= th).mean()))
    res['matched_fa'] = mf

    def paired(x1, x0, mask):
        d = agg(x1.astype(float), mask)-agg(x0.astype(float), mask)
        return dict(diff=float(d.sum()), ci95=ci(boot@d))

    def ratio_diff(a1, a0, m1, m0):
        n1, n0 = agg(a1, m1), agg(a0, m0); d1, d0 = agg(np.full(len(a1), 2.), m1), agg(np.full(len(a0), 2.), m0)
        b = (boot@n1)/(boot@d1)-(boot@n0)/(boot@d0)
        return dict(diff_pp=100*float(n1.sum()/d1.sum()-n0.sum()/d0.sum()), ci95_pp=(100*np.quantile(b, [.025, .975])).tolist())

    NE = res['NAT_events']; contact = ev['truth']['NAT']['contact']
    P = {}
    P['P1_single_unknown_time_NAT_minus_SYN'] = ratio_diff(keep['NAT', 'exact', 'single'][1], keep['SYN', 'exact', 'single'][1],
                                                          ev['truth']['NAT']['control'], ev['truth']['SYN']['control'])
    P['P1_single_unknown_time_NAT_minus_SYN']['supported'] = P['P1_single_unknown_time_NAT_minus_SYN']['ci95_pp'][1] < -10
    p2 = paired(peaks['NAT', 'E1', 'dual', .025], peaks['NAT', 'E1', 'single', .025], contact)
    p2.update(events=NE, margin_events=.02*NE, dual_advantage=p2['ci95'][0] > 0, advantage_below_2pct=p2['ci95'][1] < .02*NE)
    P['P2_NAT_E1_dual_minus_single_timely_fa2.5'] = p2
    for cfg in ('single', 'dual'):
        p3 = paired(peaks['NAT', 'E1', cfg, .025], peaks['NAT', 'exact', cfg, .025], contact); p3['confirmed_loss'] = p3['ci95'][1] < 0
        P[f'P3_NAT_{cfg}_E1_minus_exact_timely_fa2.5'] = p3
    res['primary'] = P
    # secondary (descriptive)
    Sx = {}
    for cfg in ('single', 'dual'):
        for tg in TARGETS:
            Sx[f'EMA_minus_E1/{cfg}/{tg}'] = paired(peaks['NAT', 'EMA', cfg, tg], peaks['NAT', 'E1', cfg, tg], contact)
            Sx[f'ALN_minus_NAT_exact/{cfg}/{tg}'] = paired(peaks['ALN', 'exact', cfg, tg], peaks['NAT', 'exact', cfg, tg], contact)
            Sx[f'ALN_minus_NAT_E1/{cfg}/{tg}'] = paired(peaks['ALN', 'E1', cfg, tg], peaks['NAT', 'E1', cfg, tg], contact)
        s1, s0 = keep['NAT', 'E1-union', cfg][0], keep['NAT', 'E1', cfg][0]
        Sx[f'union_minus_none_silent/{cfg}'] = paired(s1 == 2, s0 == 2, contact)
        Sx[f'union_minus_none_unknown_time/{cfg}'] = ratio_diff(keep['NAT', 'E1-union', cfg][1], keep['NAT', 'E1', cfg][1],
                                                               ev['truth']['NAT']['control'], ev['truth']['NAT']['control'])
    Sx['P2_dual_minus_single/fa5'] = paired(peaks['NAT', 'E1', 'dual', .05], peaks['NAT', 'E1', 'single', .05], contact)
    for h in (0, 1):
        m = contact & (ev['unit'] % 2 == h)
        Sx[f'half{h}/P2'] = dict(events=int(m.sum()), diff=int(peaks['NAT', 'E1', 'dual', .025][m].sum()-peaks['NAT', 'E1', 'single', .025][m].sum()))
        for cfg in ('single', 'dual'):
            Sx[f'half{h}/P3/{cfg}'] = int(peaks['NAT', 'E1', cfg, .025][m].sum()-peaks['NAT', 'exact', cfg, .025][m].sum())
    res['secondary'] = Sx
    (RC.OUT/'result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)+'\n', encoding='utf8')
    show(res)


def show(res):
    print('units', res['units'], 'events SYN/NAT', res['SYN_events'], res['NAT_events'], 'controls', res['SYN_controls'], res['NAT_controls'])
    for k, v in res['three_state'].items():
        p = v['max_tau']
        print(f"  {k:24s} T/U/S {p['timely']}/{p['unknown_miss']}/{p['silent']} unknown {100*p['unknown_time']:.2f}% obstacle {100*p['obstacle_time']:.2f}%")
    for k, v in res['matched_fa'].items():
        print(f"  MF {k:28s} timely {v['timely']} eval-FA {100*v['realized_eval_fa']:.2f}%")
    for k, v in res['primary'].items():
        print('PRIMARY', k, json.dumps(v))
    for k, v in res['secondary'].items():
        print('  SEC', k, json.dumps(v))


if __name__ == '__main__':
    main()
