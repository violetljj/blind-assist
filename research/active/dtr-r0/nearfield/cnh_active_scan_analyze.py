"""Evaluator for cnh_active_scan_dev units on the frozen exact-0.9 m event ledger (descriptive)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_tristate_event_dev as E  # noqa: E402
import cnh_tristate_dev as R  # noqa: E402

OUT = R.WORK/'cnh-active-scan-dev-20261006'
R3 = R.WORK/'cnh-tristate-dev-r3-20261006'
ARMS = ('single/passive', 'dual/passive', 'single/aligned', 'dual/aligned', 'single/prompt', 'single/prompt_d1')
D1 = OUT/'units_delay1'


def load():
    rows = R.read(R3/'rows.json')
    with np.load(R3/'online.npz') as z:
        r3score, thresholds = z['score'], z['thresholds']
    files = sorted((OUT/'units').glob('unit*.npz'))
    units = {int(f.stem[4:]): f for f in files}
    keep = [i for i, r in enumerate(rows) if r['unit'] in units]
    sub = [rows[i] for i in keep]
    n = len(sub)
    score = {a: np.zeros((n, 13)) for a in ARMS}; gate = {a: np.zeros((n, 13), bool) for a in ARMS}
    prompts = np.zeros((n, 16), bool); prompts_d1 = np.zeros((n, 16), bool); head = np.zeros((n, 16)); fresh = np.zeros((n, 16), bool)
    cache = {}
    for j, r in enumerate(sub):
        u, c = r['unit'], r['config']
        if u not in cache:
            cache.clear(); cache[u] = dict(np.load(units[u]))
            if (D1/f'unit{u}.npz').exists():
                cache[u]['d1'] = dict(np.load(D1/f'unit{u}.npz'))
        z = cache[u]
        for kind in ('passive', 'aligned'):
            sm = R.smooth(z[f'{kind}_raw'][:, c]).max(-1)  # [3,13]
            score[f'single/{kind}'][j] = sm[0]; score[f'dual/{kind}'][j] = sm[1:].max(0)
            gate[f'single/{kind}'][j] = z[f'{kind}_gate'][c, :, 0]; gate[f'dual/{kind}'][j] = z[f'{kind}_gate'][c, :, 1]
        score['single/prompt'][j] = R.smooth(z['prompt_raw'][c]).max(-1); gate['single/prompt'][j] = z['prompt_gate'][c]
        prompts[j] = z['prompts'][c]; head[j] = z['head'][c]; fresh[j] = z['fresh'][c]
        if 'd1' in z:
            score['single/prompt_d1'][j] = R.smooth(z['d1']['prompt_raw'][c]).max(-1); gate['single/prompt_d1'][j] = z['d1']['prompt_gate'][c]
            prompts_d1[j] = z['d1']['prompts'][c]
    # parity with the frozen r3 online scores for the stored passive arms
    np.testing.assert_allclose(score['single/passive'], r3score[keep, :, 0], atol=1e-5)
    np.testing.assert_allclose(score['dual/passive'], r3score[keep, :, 1], atol=1e-5)
    g = E.load_geometry(rows)
    g = {k: v[keep] for k, v in g.items()}
    has_d1 = all((D1/f'unit{u}.npz').exists() for u in units)
    arms = ARMS if has_d1 else ARMS[:-1]
    return sub, score, gate, prompts, prompts_d1, head, fresh, thresholds, g, arms


def analyze():
    sub, score, gate, prompts, prompts_d1, head, fresh, thresholds, g, arms = load()
    n = len(sub)
    units = np.array([r['unit'] for r in sub]); modes = np.array([r['mode'] for r in sub])
    uu, inv = np.unique(units, return_inverse=True)
    strat = {}
    for r in sub:
        strat[r['unit']] = f"{r['batch']}/{r['mode']}/{r['turn']}"
    srow = np.array([strat[u] for u in uu])
    rng = np.random.default_rng(2026100662); boot = np.zeros((1000, len(uu)), np.int16)
    for s in np.unique(srow):
        ids = np.flatnonzero(srow == s); draw = rng.integers(0, len(ids), size=(1000, len(ids)))
        for k in range(1000):
            boot[k, ids] = np.bincount(draw[k], minlength=len(ids))

    def agg(x, mask):
        return np.bincount(inv, weights=np.where(mask, x, 0.), minlength=len(uu))

    def ratio(num, den, mask):
        nn, dd = agg(num, mask), agg(den, mask); bn, bd = boot@nn, boot@dd; ok = bd > 0
        return dict(num=float(nn.sum()), den=float(dd.sum()), value=float(nn.sum()/dd.sum()) if dd.sum() else None,
                    ci95=np.quantile(bn[ok]/bd[ok], [.025, .975]).tolist() if ok.any() else None), (nn, dd)

    groups = {'all': np.ones(n, bool)} | {f'mode{k}': modes == k for k in range(3)}
    control = g['control']; contact = g['contact']
    # prompt cost on the control burden window: prompts issued in frames 5..14 per 2.0 s
    pr_ctrl = prompts[:, 5:15].sum(1).astype(float)
    res = dict(sequences=n, units=len(uu), groups={})
    for gname, mask in groups.items():
        G = res['groups'][gname] = dict(events=int((mask & contact).sum()), controls=int((mask & control).sum()), arms={})
        G['prompts_per_min_controls'] = ratio(pr_ctrl, np.full(n, 2./60), mask & control)[0]
        G['prompts_per_min_all_sequences'] = ratio(prompts.sum(1).astype(float), np.full(n, 3.2/60), mask)[0]
        G['controls_with_prompt'] = ratio(prompts[:, 5:15].any(1).astype(float), np.ones(n), mask & control)[0]
        G['prompts_per_min_controls_d1'] = ratio(prompts_d1[:, 5:15].sum(1).astype(float), np.full(n, 2./60), mask & control)[0]
        for arm in arms:
            curve = []
            for j, tau in enumerate(thresholds):
                state, clear, unknown, alarm = E.partition(score[arm], gate[arm], tau, g['fraction'], contact)
                b = E.burden(unknown)
                curve.append(dict(index=j, tau=float(tau),
                    timely=int(((state == 0) & mask).sum()), unknown_miss=int(((state == 1) & mask).sum()), silent=int(((state == 2) & mask).sum()),
                    unknown_time=ratio(b['seconds'], b['total_seconds'], mask & control)[0],
                    unknown_time_late=ratio(unknown[:, 7:12].sum(1)*.2, np.full(n, 1.), mask & control)[0]))
            G['arms'][arm] = curve
    # paired differences at max tau (index 19) for the control unknown time
    j = len(thresholds)-1
    diffs = {}
    pairs = [('single/prompt', 'single/passive'), ('single/aligned', 'single/passive'), ('dual/aligned', 'dual/passive'),
                  ('single/prompt', 'dual/passive'), ('single/aligned', 'dual/passive')]
    if 'single/prompt_d1' in arms:
        pairs += [('single/prompt_d1', 'single/passive'), ('single/prompt_d1', 'dual/passive')]
    for a, b_ in pairs:
        for gname, mask in groups.items():
            out = []
            for arm in (a, b_):
                _, _, unknown, _ = E.partition(score[arm], gate[arm], thresholds[j], g['fraction'], contact)
                bb = E.burden(unknown); nn, dd = agg(bb['seconds'], mask & control), agg(bb['total_seconds'], mask & control)
                out.append((nn, dd))
            (n1, d1), (n2, d2) = out; ok = (boot@d1 > 0)&(boot@d2 > 0)
            if not d1.sum() or not ok.any():
                continue
            dist = (boot@n1)[ok]/(boot@d1)[ok]-(boot@n2)[ok]/(boot@d2)[ok]
            diffs[f'{a} - {b_} | {gname}'] = dict(pp=100*float(n1.sum()/d1.sum()-n2.sum()/d2.sum()), ci95_pp=(100*np.quantile(dist, [.025, .975])).tolist())
    res['max_tau_unknown_time_diffs'] = diffs
    # paired whole-unit bootstrap of timely / silent count differences at max tau (all events)
    ev = {}
    for a, b_ in pairs:
        sa, _, _, _ = E.partition(score[a], gate[a], thresholds[j], g['fraction'], contact)
        sb, _, _, _ = E.partition(score[b_], gate[b_], thresholds[j], g['fraction'], contact)
        for name, code in (('timely', 0), ('silent', 2)):
            da = agg((sa == code).astype(float), contact)-agg((sb == code).astype(float), contact)
            dist = boot@da
            ev[f'{a} - {b_} | {name}'] = dict(diff=float(da.sum()), ci95=np.quantile(dist, [.025, .975]).tolist())
    res['max_tau_event_count_diffs'] = ev
    # empirical silent-budget readings (descriptive, no reselection bootstrap)
    budgets = {}
    for arm in arms:
        curve = res['groups']['all']['arms'][arm]
        for b in (0, 2, 5, 15):
            ok = [p for p in curve if p['silent'] <= b]
            best = min(ok, key=lambda p: p['unknown_time']['value']) if ok else None
            budgets[f'{arm} | silent<={b}'] = None if best is None else dict(index=best['index'], timely=best['timely'], unknown_miss=best['unknown_miss'], silent=best['silent'], unknown_time=best['unknown_time']['value'], unknown_time_late=best['unknown_time_late']['value'])
    res['silent_budget_readings'] = budgets
    # head behaviour summary
    res['head'] = dict(fresh_frames=int(fresh.sum()), sequences_diverged=int(fresh.any(1).sum()),
                       prompts_total=int(prompts.sum()), first_prompt_frame_hist=np.bincount(np.where(prompts.any(1), prompts.argmax(1), 0)[prompts.any(1)], minlength=16).tolist())
    (OUT/'result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)+'\n', encoding='utf8')
    return res


def summary(res):
    j = 19
    for gname, G in res['groups'].items():
        if not G['controls']:
            continue
        print(f"== {gname}: events {G['events']} controls {G['controls']} prompts/min(controls) {G['prompts_per_min_controls']['value']:.2f} ctrl-with-prompt {G['controls_with_prompt']['value']:.3f}")
        for arm, curve in G['arms'].items():
            p = curve[j]; u = p['unknown_time']
            if u['value'] is None:
                continue
            ul = p['unknown_time_late']
            print(f"  {arm:15s} maxtau T/U/S {p['timely']}/{p['unknown_miss']}/{p['silent']}  unknown {100*u['value']:.2f}% {np.round(100*np.array(u['ci95']), 2)}  late(f10-14) {100*ul['value']:.2f}%")
    for k, v in res['max_tau_event_count_diffs'].items():
        print(f"  EVENTS {k}: {v['diff']:+.0f} {np.round(v['ci95'], 1)}")
    for k, v in res['silent_budget_readings'].items():
        if v: print(f"  BUDGET {k}: idx {v['index']} T/U/S {v['timely']}/{v['unknown_miss']}/{v['silent']} unknown {100*v['unknown_time']:.2f}% late {100*v['unknown_time_late']:.2f}%")
    for k, v in res['max_tau_unknown_time_diffs'].items():
        print(f"  {k}: {v['pp']:+.2f} pp {np.round(v['ci95_pp'], 2)}")


if __name__ == '__main__':
    summary(analyze())
