"""Exact eval-truth oracle for HEAD>=h OR BODY>=b; never a deployable fit."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
SOURCE = WORK/'cnh-querywise-calibration-dev-20261007'
OUT = WORK/'cnh-query-threshold-oracle-dev-20261007'
CAPS = (.025, .05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readable(value):
    if isinstance(value, np.ndarray):
        return readable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else '+inf' if value > 0 else '-inf'
    if isinstance(value, dict):
        return {k: readable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [readable(v) for v in value]
    return value


def save(name, value):
    with (OUT/name).open('x', encoding='utf8') as f:
        json.dump(readable(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), SOURCE/'PLAN.json', SOURCE/'result.json', SOURCE/'ledger.npz',
             Path(__file__).with_name('cnh_querywise_calibration_dev.py')]
    save('PLAN.json', dict(phase='POSTHOC EVAL-TRUTH ORACLE; consumed Development',
        budget_cpu_wall_seconds=120, no_gpu=True, no_training=True,
        question='Exact maximum timely excluding warmup in constant two-raw-threshold OR family, independently per original outer evaluation fold',
        input='querywise ledger query_score/original_center [N,13,2], unit/config/fold/contact/control/deadline only',
        evaluation_truth_access='Explicitly permits selecting h,b using the same evaluation contact labels. Not train/calibration, validation, or deployment evidence.',
        caps=list(CAPS), folds=[0, 1, 2], primary='Event per-query maximum over outputs2..deadline inclusive; OR. Controls outputs2:12, integer FA budget floor(cap*Nintervals).',
        family='HEAD>=h OR BODY>=b, real thresholds including never-alarm +inf, exact >= ties. No HB inputs, candidate selection, new weights or isotonic changes.',
        enumeration='Enumerate nextafter(each unique HEAD control value,+inf), plus +inf; retain H_FA<=K. Given H control union, take nextafter((K-H_FA)th largest remaining BODY control,+inf). This is the lowest feasible B.',
        completeness='For each achievable HEAD control-alarm set, its lowest real floating threshold weakly enlarges HEAD event coverage without extra FA. For fixed H set, lowest feasible B weakly maximizes event coverage. Their enumeration therefore attains global maximum timely over this family. Equal control scores cannot be split.',
        tie_break='After finding lowest feasible B, raise B to smallest BODY peak among events not already alarmed by H but rescued by B; if none, B=+inf. This preserves timely and minimizes extra FA. Across H candidates choose highest timely, lowest FA, then ascending H candidate index.',
        baseline='Control-only matched-evaluation common raw threshold on max(HEAD,BODY); same cap and no-warmup event rule. No contact-label threshold choice for baseline.',
        validation='Small exhaustive threshold breakpoint search on seeded discrete arrays with ties; compare exact best timely and minimum FA. No random grid on real data.',
        limitations=['Family-specific oracle, not a physical upper bound or deployable candidate',
                     'Different thresholds selected per evaluated fold; aggregated gains remain optimistic',
                     'No proof of information absence if this family has little headroom'],
        hashes={str(p.relative_to(ROOT)): sha(p) for p in paths}))
    (OUT/'source').mkdir(exist_ok=True)
    for p in (Path(__file__), Path(__file__).with_name('cnh_querywise_calibration_dev.py')):
        (OUT/'source'/p.name).write_bytes(p.read_bytes())
    print('FROZEN exact two-threshold eval-truth oracle, CPU120s', flush=True)


def lowest_feasible(values, budget):
    """Minimum floating cutoff for >= with <=budget alarms, preserving ties."""
    values = np.asarray(values, float)
    if budget >= len(values):
        return -np.inf
    if budget < 0:
        raise ValueError('Negative residual budget')
    boundary = np.sort(values)[::-1][budget]
    return float(np.nextafter(boundary, np.inf))


def exact_search(control, peaks, eligible, budget):
    control, peaks = np.asarray(control, float), np.asarray(peaks, float)
    eligible = np.asarray(eligible, bool)
    if control.ndim != 2 or control.shape[1] != 2 or peaks.shape != (len(eligible), 2):
        raise ValueError('Two score columns required')
    candidates = np.r_[np.nextafter(np.unique(control[:, 0]), np.inf), np.inf]
    if budget >= len(control):
        candidates = np.r_[-np.inf, candidates]
    best, evaluated, trace = None, 0, []
    for index, h in enumerate(candidates):
        hm = control[:, 0] >= h; hfa = int(hm.sum())
        if hfa > budget:
            continue
        evaluated += 1
        bmin = lowest_feasible(control[~hm, 1], budget-hfa)
        eh = eligible & (peaks[:, 0] >= h)
        eb = eligible & (peaks[:, 1] >= bmin)
        body_only = eb & ~eh
        b = float(peaks[body_only, 1].min()) if body_only.any() else np.inf
        flags = eligible & ((peaks[:, 0] >= h) | (peaks[:, 1] >= b))
        assert np.array_equal(flags, eh | eb)
        fa = int((hm | (control[:, 1] >= b)).sum()); timely = int(flags.sum())
        assert fa <= budget
        key = (-timely, fa, index)
        entry = dict(head_threshold=float(h), body_threshold=float(b), body_min_feasible=bmin,
                     timely=timely, false_alarm_intervals=fa, candidate_index=index,
                     head_false_alarm_intervals=hfa)
        trace.append(entry)
        if best is None or key < best[0]:
            best = key, entry, flags
    return dict(**best[1], candidate_thresholds_total=len(candidates),
                candidate_thresholds_feasible=evaluated,
                unique_head_control_scores=len(np.unique(control[:, 0])),
                flags=best[2], candidates=trace)


def brute_force_check():
    rng = np.random.default_rng(2026100723); cases = 0; checked_pairs = 0
    for _ in range(40):
        nc, ne = int(rng.integers(1, 8)), int(rng.integers(1, 7))
        control = rng.integers(-2, 3, size=(nc, 2)).astype(float)
        peaks = rng.integers(-3, 4, size=(ne, 2)).astype(float)
        eligible = rng.random(ne) > .2
        axis = []
        for q in (0, 1):
            u = np.unique(np.r_[control[:, q], peaks[:, q]])
            axis.append(np.unique(np.r_[-np.inf, u, np.nextafter(u, np.inf), np.inf]))
        for budget in range(nc+1):
            best = None
            for h in axis[0]:
                for b in axis[1]:
                    checked_pairs += 1
                    fa = int(((control[:, 0] >= h) | (control[:, 1] >= b)).sum())
                    if fa > budget:
                        continue
                    count = int((eligible & ((peaks[:, 0] >= h) | (peaks[:, 1] >= b))).sum())
                    key = (-count, fa)
                    if best is None or key < best:
                        best = key
            actual = exact_search(control, peaks, eligible, budget)
            assert (-actual['timely'], actual['false_alarm_intervals']) == best
            cases += 1
    return dict(status='PASS', cases=cases, exhaustive_pairs=checked_pairs,
                tied_discrete_scores=True, includes_zero_and_full_budget=True,
                verifies=['maximum_no_warmup_timely', 'minimum_FA_at_equal_timely'])


def run():
    start = time.monotonic(); plan = json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    if (OUT/'result.json').exists():
        raise FileExistsError('Preserve completed oracle')
    def check():
        if time.monotonic()-start > plan['budget_cpu_wall_seconds']:
            raise TimeoutError('120s oracle analysis budget')
    for p, h in plan['hashes'].items():
        assert sha(ROOT/p) == h, p
    verification = brute_force_check(); check()
    fields = ('unit', 'config', 'fold', 'contact', 'control', 'deadline', 'query_score/original_center')
    with np.load(SOURCE/'ledger.npz', allow_pickle=False) as source:
        z = {key: source[key] for key in fields}
    scores = z['query_score/original_center']; assert scores.shape == (len(z['unit']), 13, 2)
    assert np.isfinite(scores).all() and not (z['contact'] & z['control']).any()
    peaks = np.full((len(scores), 2), -np.inf); eligible = z['contact'] & (z['deadline'] >= 2)
    for row in np.flatnonzero(eligible):
        assert z['deadline'][row] < 13
        peaks[row] = scores[row, 2:z['deadline'][row]+1].max(0)
    records, arrays = [], {}
    for cap in CAPS:
        flags = np.zeros(len(scores), bool); baseline_flags = flags.copy()
        for fold in (0, 1, 2):
            check(); mask = z['fold'] == fold
            ids = np.flatnonzero(mask & z['contact']); controls = scores[mask & z['control'], 2:12].reshape(-1, 2)
            budget = int(np.floor(cap*len(controls)+1e-12))
            answer = exact_search(controls, peaks[ids], eligible[ids], budget)
            flags[ids] = answer.pop('flags')
            theta = lowest_feasible(controls.max(-1), budget)
            baseline_flags[ids] = eligible[ids] & (peaks[ids].max(-1) >= theta)
            baseline_fa = int((controls.max(-1) >= theta).sum())
            assert baseline_fa <= budget
            records.append(dict(cap=cap, fold=fold, events=len(ids),
                controls=int((mask & z['control']).sum()), control_intervals=len(controls),
                false_alarm_budget_intervals=budget, oracle=answer,
                matched_max_baseline=dict(threshold=theta, timely=int(baseline_flags[ids].sum()),
                                          false_alarm_intervals=baseline_fa),
                oracle_minus_baseline=int(flags[ids].sum()-baseline_flags[ids].sum())))
        arrays[f'oracle/{cap:.3f}'] = flags
        arrays[f'matched_max/{cap:.3f}'] = baseline_flags
    aggregate = {}
    for cap in CAPS:
        rs = [r for r in records if r['cap'] == cap]
        aggregate[f'{cap:.3f}'] = dict(events=sum(r['events'] for r in rs),
            controls=sum(r['controls'] for r in rs), control_intervals=sum(r['control_intervals'] for r in rs),
            oracle_timely=sum(r['oracle']['timely'] for r in rs),
            baseline_timely=sum(r['matched_max_baseline']['timely'] for r in rs),
            oracle_false_alarm_intervals=sum(r['oracle']['false_alarm_intervals'] for r in rs),
            baseline_false_alarm_intervals=sum(r['matched_max_baseline']['false_alarm_intervals'] for r in rs),
            exact_candidates_evaluated=sum(r['oracle']['candidate_thresholds_feasible'] for r in rs))
    check()
    np.savez_compressed(OUT/'ledger.npz', **{k:z[k] for k in fields if k != 'query_score/original_center'},
                        event_query_max_no_warmup=peaks, eligible_no_warmup=eligible, **arrays)
    save('result.json', dict(status='COMPLETE_EVAL_TRUTH_ORACLE', aggregate=aggregate,
        fold_records=records, brute_force_validation=verification, seconds=time.monotonic()-start,
        source_hashes=plan['hashes'], ledger_sha256=sha(OUT/'ledger.npz'),
        plan_sha256=sha(OUT/'PLAN.json'), interpretation=plan['limitations']))
    print(json.dumps(dict(aggregate=aggregate, verification=verification,
                         seconds=time.monotonic()-start), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'run'])
    args = parser.parse_args(); freeze() if args.stage == 'freeze' else run()
