"""CPU-only directional confidence readout on consumed adaptive-query Development.

No model inference. Mirrored errors imply symmetric side weights, not left/right
intention inference. Weighted M3 logits are scores, not collision probabilities.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import cnh_adaptive_query_dev as Q
import cnh_heading_uncertainty_analyze as H

R, E = H.R, H.E
OUT = R.WORK / 'cnh-probability-query-dev-20261007'
METHODS = ('center', 'max', 'mean', 'fixed_fit', 'marginal', 'conditional')
SEED = 2026100711


def symmetric_weights(errors, width):
    """Nearest of -k,0,+k, with mirrored empirical errors; keep tail mass."""
    center = float(np.mean(np.abs(errors) <= width / 2))
    return np.array([center, (1-center)/2, (1-center)/2])


def combine(scores, weights):
    return np.sum(scores * weights, axis=-1)


def threshold_at_cap(control_scores, cap=.025):
    x = np.sort(np.asarray(control_scores, dtype=float).ravel())[::-1]
    if not len(x) or not np.isfinite(x).all() or not 0 <= cap < 1:
        raise ValueError('finite nonempty controls and cap in [0,1) required')
    return float(np.nextafter(x[int(np.floor(cap * len(x)))], np.inf))


def split_units(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['batch'], r['mode'], r['turn']), set()).add(r['unit'])
    rng = np.random.default_rng(SEED)
    cal = set()
    for key in sorted(groups):
        ids = np.array(sorted(groups[key]))
        if len(ids) < 2:
            raise ValueError('stratum cannot support disjoint unit groups')
        cal.update(map(int, rng.permutation(ids)[:max(1, len(ids)//3)]))
    return cal


def save(name, value):
    p = OUT / name
    with p.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    unit_files = sorted((Q.OUT/'units').glob('unit*.npz'))
    assert len(unit_files) == 144
    rows = [r for r in R.read(H.R3/'rows.json') if r['unit'] in Q.A.UNITS]
    cal = split_units(rows)
    paths = [Path(__file__), Path(Q.__file__), Path(H.__file__), Path(E.__file__),
             Path(R.__file__), Path(Q.B.__file__), Q.OUT/'meta.json',
             H.R3/'rows.json', H.R3/'online.npz', *E.GEOMETRY,
             *unit_files, *sorted(Q.B.SRC.glob('*.npy'))]
    save('PLAN.json', dict(phase='EXPLORE; consumed Development', budget_cpu_analysis_wall_seconds=600,
        goal='Does direction confidence weighting recover timely alarms without a false-alarm or silent-miss penalty?',
        scope='CPU saved scores only; frozen M3, geometry, event deadline, r3 gate and clear tau; no GPU/new data',
        methods=list(METHODS), sources=list(Q.ESTS), primary='head/single/conditional vs center and marginal',
        fit='Odd participants only; same old valid-frame selection and six causal proxy bins; symmetrize errors to match replay',
        weights='Nearest-center 3-point quadrature at +/-k/2. Tail mass is assigned to outer nodes, never discarded. Not calibrated collision probability.',
        controls='mean=1/3 each; fixed_fit=global normalized-error weights; marginal=global error distribution evaluated at actual bin k; conditional=within-bin error distribution at k',
        order='Per-direction existing causal smoothing and query/sensor max, then current f=3..15 weights; same history for all methods',
        calibration_units=sorted(cal), evaluation_units=sorted(set(Q.A.UNITS)-cal),
        calibration='Stratify unit by batch/mode/turn, seeded 1/3 cal; threshold only from cal controls, alarm >=theta, exact ties handled',
        evaluation='Report disjoint-unit actual FA; secondary evaluation-control-matched 2.5% and 5% frontiers are descriptive, not independent threshold validation',
        clear='All compared methods use adaptive max <= original max tau and original r3 gate; alarm takes precedence',
        inference='1000 stratified paired eval-unit bootstrap, thresholds and weight fits held fixed; not participant or calibration uncertainty',
        decision='Promising only if primary conditional improves center and marginal on matched 2.5% timely; conditional-center unit CI lower>0; calibrated timely improves center with actual FA increase<=0.5pp and no extra silent miss. Otherwise no GPU escalation from this check.',
        decision_check='Prior all-unit head/single center 207 vs exact349 of369 gives headroom; new split denominators reported; minimum difference one event; missing controls invalidates comparison, not success.',
        limitations=['Old reused data, not fresh confirmation', 'Mirrored errors and unsigned proxies cannot test left/right intention',
                    'True errors used only to fit odd-participant distributions and verify replay, never to set evaluation weights',
                    'Real motion error windows independent of simulated scenes; no environment-intention claim',
                    '3 directions approximate error tails at finite outer queries; tail mass reported, no safety claim'],
        hashes={str(p.relative_to(R.ROOT)): R.sha(p) for p in paths}))
    print('PLAN saved; calibration units', len(cal), 'evaluation units', 144-len(cal), flush=True)


def distributions():
    pools = ({}, {})
    for f in sorted(Q.B.SRC.glob('*.npy')):
        d = Q.series(np.load(f)); d['ok'][-1] = False
        dest = pools[0 if int(f.stem[1:3]) % 2 else 1]
        for k, v in d.items():
            dest.setdefault(k, []).append(v)
    cal, ev = [{k: np.concatenate(v) for k, v in p.items()} for p in pools]
    meta = R.read(Q.OUT/'meta.json'); tables = {}; fit_meta = {}
    for e in Q.ESTS:
        mask = cal['ok'] & np.isfinite(cal[e+'_err'])
        bins = Q.bins(cal['speed'], cal[e+'_turn'])
        widths = np.array(meta['table_deg'][e])
        err = cal[e+'_err'][mask]; b = bins[mask]
        np.testing.assert_allclose(widths, [np.percentile(np.abs(err[b == j]), 75) for j in range(6)])
        fixed = symmetric_weights(err/widths[b], 1.)
        conditional = np.array([symmetric_weights(err[b == j], widths[j]) for j in range(6)])
        marginal = np.array([symmetric_weights(err, k) for k in widths])
        tables[e] = dict(fixed_fit=np.tile(fixed, (6, 1)), conditional=conditional, marginal=marginal)
        eb = Q.bins(ev['speed'], ev[e+'_turn'])
        ev[e+'_bin'] = eb; ev[e+'_k'] = widths[eb]
        fit_meta[e] = dict(widths=widths.tolist(), weights={k: v.tolist() for k, v in tables[e].items()},
            frames=int(mask.sum()), bin_frames=[int((b == j).sum()) for j in range(6)],
            training_tail_mass=[float(np.mean(np.abs(err[b == j]) > widths[j])) for j in range(6)])
    ok = ev['ok'] & np.isfinite(ev['head_err']) & np.isfinite(ev['torso_err'])
    c = np.r_[0, np.cumsum(ok)]
    starts = np.flatnonzero(c[Q.WIN:] - c[:-Q.WIN] == Q.WIN)
    assert len(starts) == meta['eval_windows']
    return ev, starts, tables, fit_meta


def run():
    tick = time.monotonic(); plan = R.read(OUT/'PLAN.json')
    if (OUT/'result.json').exists():
        raise FileExistsError('preserve completed run')

    def check():
        if time.monotonic()-tick > plan['budget_cpu_analysis_wall_seconds']:
            raise TimeoutError('CPU analysis budget exhausted')

    for path, sha in plan['hashes'].items():
        assert R.sha(R.ROOT/path) == sha, path
    ev, starts, tables, fits = distributions(); check()
    rows_all = R.read(H.R3/'rows.json')
    keep = [i for i, r in enumerate(rows_all) if r['unit'] in Q.A.UNITS]
    rows = [rows_all[i] for i in keep]; n = len(rows)
    g = {k: v[keep] for k, v in E.load_geometry(rows_all).items()}
    with np.load(H.R3/'online.npz') as z:
        gate = z['gate'][keep, :, 1, :]; tau = float(z['thresholds'][-1])
    scores = {}; bins_saved = {}; tail_saved = {}; cache = {}; parity = 0.
    for j, row in enumerate(rows):
        check(); u, cfg = row['unit'], row['config']
        if u not in cache:
            with np.load(Q.OUT/'units'/f'unit{u}.npz') as z:
                cache = {u: dict(z)}
        z = cache[u]; rng = np.random.default_rng([Q.SEED, u, cfg])
        start = starts[rng.integers(len(starts))]; sg = 1. if rng.random() < .5 else -1.
        exact = H.arm_scores(z['exact_raw'][:, cfg])
        for a, arm in enumerate(('single', 'dual')):
            scores.setdefault(arm+'/exact', np.zeros((n, 13)))[j] = exact[a]
        for e in Q.ESTS:
            sl = slice(start, start+Q.WIN, 6)
            # True errors are restricted to replay integrity and tail diagnostics.
            expected = sg*ev[e+'_err'][sl]
            delta = np.max(np.abs(expected-z[e+'_c_err'][cfg])); parity = max(parity, float(delta))
            np.testing.assert_allclose(expected, z[e+'_c_err'][cfg], atol=2e-5, rtol=1e-6)
            np.testing.assert_allclose(ev[e+'_k'][sl], z[e+'_ap_k'][cfg], atol=2e-5, rtol=1e-6)
            b = ev[e+'_bin'][sl][3:]
            bins_saved.setdefault(e, np.zeros((n, 13), np.int8))[j] = b
            tail_saved.setdefault(e, np.zeros((n, 13), bool))[j] = np.abs(expected[3:]) > ev[e+'_k'][sl][3:]
            directions = [H.arm_scores(z[e+'_'+t+'_raw'][:, cfg]) for t in ('c', 'ap', 'am')]
            for a, arm in enumerate(('single', 'dual')):
                d = np.stack([v[a] for v in directions], axis=-1)
                values = dict(center=d[:, 0], max=d.max(-1), mean=d.mean(-1))
                values.update({m: combine(d, w[b]) for m, w in tables[e].items()})
                for m, value in values.items():
                    scores.setdefault(arm+'/'+e+'/'+m, np.zeros((n, 13)))[j] = value
    uid = np.array([r['unit'] for r in rows]); iscal = np.isin(uid, plan['calibration_units']); iseval = ~iscal
    ix = E.causal_index(g['fraction']); rr = np.flatnonzero(g['contact'])
    assert np.all((ix[rr] >= 0) & (ix[rr] < 13))
    counts = {name: dict(units=len(np.unique(uid[mask])), sequences=int(mask.sum()),
              events=int((mask & g['contact']).sum()), controls=int((mask & g['control']).sum()),
              control_intervals=int((mask & g['control']).sum())*10)
              for name, mask in [('calibration', iscal), ('evaluation', iseval), ('all', np.ones(n, bool))]}
    eunits, inv = np.unique(uid[iseval], return_inverse=True)
    strata = {r['unit']: (r['batch'], r['mode'], r['turn']) for r in rows}
    rng = np.random.default_rng(SEED+1); boot = np.zeros((1000, len(eunits)), int)
    for key in sorted(set(strata.values())):
        ids = np.array([i for i, u in enumerate(eunits) if strata[u] == key])
        if not len(ids):
            continue
        for k, draw in enumerate(rng.integers(0, len(ids), (1000, len(ids)))):
            boot[k, ids] = np.bincount(draw, minlength=len(ids))

    def evaluate(key, theta, mask):
        arm = key.split('/')[0]; a = ('single', 'dual').index(arm)
        clear_score = scores[arm+'/exact'] if key.endswith('/exact') else scores['/'.join(key.split('/')[:2])+'/max']
        alarm = scores[key] >= theta
        clear = ~alarm & gate[:, :, a] & (clear_score <= tau)
        timely = np.zeros(n, bool)
        timely[rr] = np.maximum.accumulate(alarm, axis=1)[rr, ix[rr]]
        silent = np.zeros(n, bool); silent[rr] = ~timely[rr] & clear[rr, ix[rr]]
        ec = mask & g['contact']; ctl = mask & g['control']
        metrics = dict(threshold=float(theta), timely=int((timely & ec).sum()), silent=int((silent & ec).sum()),
                       unknown_miss=int((ec & ~timely & ~silent).sum()),
                       false_alarm=float(alarm[ctl, 2:12].mean()), unknown_time=float((~alarm & ~clear)[ctl, 2:12].mean()))
        return metrics, timely

    records = {}; decisions = {}; timely_save = {}; diffs = {}
    for target in (.025, .05):
        for mode, fitmask in [('calibrated', iscal), ('matched_eval_descriptive', iseval)]:
            tt = {}; prefix = f'{mode}/{target:.3f}'
            for key, s in scores.items():
                theta = threshold_at_cap(s[fitmask & g['control'], 2:12], target)
                metrics, timely = evaluate(key, theta, iseval)
                metrics['threshold_fit_false_alarm'] = float((s[fitmask & g['control'], 2:12] >= theta).mean())
                assert metrics['threshold_fit_false_alarm'] <= target+1e-12
                records[prefix+'/'+key] = metrics; tt[key] = timely
                timely_save[prefix+'/'+key] = timely
            for arm in ('single', 'dual'):
                for e in Q.ESTS:
                    root = arm+'/'+e+'/'
                    for ref in ('center', 'max', 'mean', 'fixed_fit', 'marginal'):
                        delta = (tt[root+'conditional'].astype(int)-tt[root+ref].astype(int))[iseval]
                        sums = np.bincount(inv, weights=delta, minlength=len(eunits))
                        diffs[prefix+'/'+root+'conditional-'+ref] = dict(diff=int(sums.sum()),
                            ci95=np.quantile(boot@sums, [.025, .975]).tolist())
    root = 'head'  # Main decision is fixed to single/head; all other arms are secondary.
    matched = 'matched_eval_descriptive/0.025/single/head/'
    calkey = 'calibrated/0.025/single/head/'
    co, ce = records[calkey+'conditional'], records[calkey+'center']
    decisions['primary_promising'] = bool(diffs[matched+'conditional-center']['ci95'][0] > 0 and
        diffs[matched+'conditional-marginal']['diff'] > 0 and co['timely'] > ce['timely'] and
        co['false_alarm'] <= ce['false_alarm']+.005 and co['silent'] <= ce['silent'])
    save('weights.json', fits)
    np.savez_compressed(OUT/'ledger.npz', unit=uid, config=np.array([r['config'] for r in rows]), calibration=iscal,
        contact=g['contact'], control=g['control'], deadline_index=ix, gate=gate,
        **{'score/'+k: v for k, v in scores.items()}, **{'timely/'+k: v for k, v in timely_save.items()},
        **{'bin/'+k: v for k, v in bins_saved.items()}, **{'tail/'+k: v for k, v in tail_saved.items()})
    check()
    result = dict(counts=counts, records=records, diffs=diffs, decision=decisions, clear_tau=tau,
        replay_error_parity_max_deg=parity,
        evaluation_tail_fraction={e: float(v[iseval].mean()) for e, v in tail_saved.items()},
        seconds=time.monotonic()-tick, plan_sha256=R.sha(OUT/'PLAN.json'), ledger_sha256=R.sha(OUT/'ledger.npz'))
    save('result.json', result)
    print(json.dumps({k: result[k] for k in ('counts', 'decision', 'seconds', 'evaluation_tail_fraction')}, indent=2), flush=True)
    for key, value in records.items():
        if '/0.025/' in key:
            print(key, value['timely'], value['silent'], round(100*value['false_alarm'], 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.stage == 'freeze' else run()
