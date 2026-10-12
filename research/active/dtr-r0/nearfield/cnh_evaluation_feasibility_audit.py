"""Read existing sealed scores; audit ceiling, pairing and conditional power.

No model imports, inference, calibration, new scenes or old-result mutation.
Exact calculations below are planning diagnostics, NOT replacement verdicts.
Run with the project NumPy/SciPy Python environment.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, binomtest, norm

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-replay-evaluation-audit-20261003/evaluation'
UNITS = list(range(96000, 96096))
RUNS = {'RAY': 'cnh-ray-surface-20261002', 'QMASS': 'cnh-query-mass-20261003',
        'CCON': 'cnh-boundary-contrast-20261003'}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def smooth(raw):
    out = np.empty_like(raw, dtype=np.float64)
    weights = np.array([1, 2, 4, 8, 16], dtype=np.float64)
    for t in range(raw.shape[2]):
        w = weights[-min(5, t + 1):]
        out[:, :, t] = np.tensordot(raw[:, :, t + 1 - len(w):t + 1], w / w.sum(), axes=([2], [0]))
    return out.transpose(0, 1, 3, 2).reshape(-1, 13)


def exact_improvement_power(n, delta, discordance, alpha=.05):
    """Independent paired events: D~Bin(n,q), rescue|D~Bin(D,(q+d)/2q).

    Improvement rejection uses positive tail of two-sided exact McNemar
    (alpha/2). It is intentionally different from the old percentile bootstrap.
    No random simulation; sum all possible discordant counts exactly.
    """
    if not 0 < delta <= discordance <= 1:
        raise ValueError('Need 0 < delta <= discordance <= 1')
    d = np.arange(n + 1)
    critical = binom.isf(alpha / 2, d, .5).astype(int) + 1
    conditional = binom.sf(critical - 1, d, (discordance + delta) / (2 * discordance))
    return float(binom.pmf(d, n, discordance) @ conditional)


def required_independent_pairs(delta, q, power=.8):
    lo, hi = 1, 64
    while exact_improvement_power(hi, delta, q) < power:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if exact_improvement_power(mid, delta, q) >= power:
            hi = mid
        else:
            lo = mid + 1
    # These scenario powers are smooth after marginalizing discordant counts;
    # explicitly check the boundary instead of assuming a rounded normal result.
    assert exact_improvement_power(lo, delta, q) >= power
    assert lo == 1 or exact_improvement_power(lo - 1, delta, q) < power
    return lo


def audit():
    hashes = {}
    results = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Sealed input hash mismatch: ' + str(path))
        hashes[str(path)] = digest
        return path
    for arm, name in RUNS.items():
        results[arm] = read(bind(WORK / name / 'result.json'))
        if results[arm]['status'] != 'COMPLETE':
            raise ValueError('Prior result incomplete: ' + arm)
        plan_path = WORK / name / 'PLAN.json'
        bind(plan_path, results[arm]['provenance']['input_sha256'][str(plan_path)])
    reference = results['RAY']['provenance']['input_sha256']
    def old_input(name, run='cnh-observed-sequence-20261002'):
        path = WORK / run / name
        return bind(path, reference[str(path)])
    with np.load(old_input('geometry.npz'), allow_pickle=False) as cache:
        g = {k: cache[k] for k in cache.files}
    rows = read(old_input('rows.json'))
    keys = [(r['split'], r['unit'], r['config'], r['query']) for r in rows]
    assert keys == list(zip(g['split'].tolist(), g['unit'].tolist(), g['config'].tolist(), g['query'].tolist()))
    ev = g['split'] == 'evaluation'
    assert g['unit'][ev].tolist() == [u for u in UNITS for _ in range(80)]
    covered = g['covered'][ev]
    ranges = g['frame_ranges'][ev]
    clear = g['clear_all'][ev]
    ref = g['ref_category'][ev]
    eunit = g['unit'][ev]
    ui = eunit - UNITS[0]
    shallow = covered & (ref == 'contact0-2cm')
    deep = covered & (ref == 'contact>5cm')
    group = np.array([r['target_group'] for r in rows])[ev]
    offset = np.array([r['target_off'] for r in rows])[ev]
    outside = clear & (g['query'][ev] == group) & (offset >= -.20) & (offset < -.10)
    scores = {}
    old = 'cnh-margin-confirm-20261002'
    with np.load(old_input('frame_scores_M3_early.npz', old), allow_pickle=False) as early, \
         np.load(old_input('frame_scores_M3.npz', old), allow_pickle=False) as late:
        scores['M3'] = smooth(np.stack([np.concatenate((early[str(u)], late[str(u)]), axis=1) for u in UNITS]))
    thresholds = {'M3': results['RAY']['thresholds']['M3']}
    for arm, run in RUNS.items():
        path = WORK / run / f'frame_scores_{arm}.npz'
        bind(path, results[arm]['provenance']['input_sha256'][str(path)])
        with np.load(path, allow_pickle=False) as cache:
            scores[arm] = smooth(np.stack([cache[str(u)] for u in UNITS]))
        thresholds[arm] = results[arm]['thresholds'][arm]
        assert results[arm]['thresholds']['M3'] == thresholds['M3']
    flags = {}
    for arm, score in scores.items():
        assert score.shape == ranges.shape and np.isfinite(score).all()
        alarms = score >= thresholds[arm]
        stopped = alarms.any(1)
        first = alarms.argmax(1)
        timely = stopped & (ranges[np.arange(len(ranges)), first] >= .9)
        flags[arm] = dict(stopped=stopped, timely=timely)
    base = flags['M3']
    count_per_unit = np.bincount(ui[shallow], minlength=96)
    miss = np.flatnonzero(shallow & ~base['timely'])
    clusters = []
    for u in UNITS:
        keep = shallow & (eunit == u)
        if keep.any():
            clusters.append(dict(unit=u, episodes=int(keep.sum()), M3_timely=int((keep & base['timely']).sum())))
    comparisons = {}
    for arm in RUNS:
        f = flags[arm]
        rescue = shallow & f['timely'] & ~base['timely']
        loss = shallow & ~f['timely'] & base['timely']
        change = np.bincount(ui, weights=(rescue.astype(int) - loss), minlength=96)
        cell = results[arm]['cells'][arm]
        changes = results[arm][f'comparisons_{arm}_minus_control']['M3']
        assert int((shallow & f['timely']).sum()) == cell['metrics']['contact0-2cm']['timely_stops']
        assert int(rescue.sum()) == changes['contact0-2cm']['rescues']
        assert int(loss.sum()) == changes['contact0-2cm']['losses']
        assert int((deep & f['timely']).sum()) == cell['metrics']['contact>5cm']['timely_stops']
        assert int((clear & f['stopped']).sum()) == cell['metrics']['clear']['stops']
        assert int((outside & f['stopped']).sum()) == cell['clear_subgroups']['same_height_nominal_outside10_20cm']['first_stops']
        comparisons[arm] = dict(shallow_timely=int((shallow & f['timely']).sum()), rescues=int(rescue.sum()),
            losses=int(loss.sum()), affected_units=[dict(unit=UNITS[i], net_events=int(change[i])) for i in np.flatnonzero(change)],
            old_paired_cluster_ci95=changes['contact0-2cm']['paired_unit_ci95'],
            old_outside_clear=changes['primary_outside_clear'], old_overall_clear=changes['clear'],
            old_deep=changes['contact>5cm'],
            prior_verdict=results[arm]['verdict'],
            rule_role='Frozen RAY decision' if arm == 'RAY' else 'Descriptive reference only; not inherited RAY hard stop',
            old_checks=results[arm].get('per_control_decision_checks', results[arm].get('descriptive_reference_checks')))
    # Empirical ceiling of the ORIGINAL bootstrap gate, fixing all observed rows
    # and denominators. Abstract repair of existing misses is not a new reader.
    repair_audit = {}
    for arm in RUNS:
        seed = results[arm]['bootstrap']['seed']
        rng = np.random.default_rng(seed)
        boot = np.asarray([np.bincount(rng.integers(96, size=96), minlength=96) for _ in range(1000)])
        denominator = boot @ count_per_unit
        assert (denominator > 0).all()
        outcomes = []
        for k in range(1, len(miss) + 1):
            candidate = []
            for selected in itertools.combinations(miss.tolist(), k):
                gains = np.bincount(ui[list(selected)], minlength=96)
                sampled = (boot @ gains) / denominator
                lower = float(np.quantile(sampled, .025))
                candidate.append(dict(repaired_events=k, repaired_units=len(set(eunit[list(selected)].tolist())),
                    bootstrap_lower=lower, bootstrap_zero_draws=int((sampled == 0).sum()),
                    original_shallow_gate_pass=lower > 0))
            outcomes.append(dict(events=k, min_lower=min(x['bootstrap_lower'] for x in candidate),
                max_lower=max(x['bootstrap_lower'] for x in candidate), passing_subsets=sum(x['original_shallow_gate_pass'] for x in candidate),
                total_subsets=len(candidate)))
        repair_audit[arm] = dict(seed=seed, outcomes=outcomes,
            role='Attainability diagnostic of old percentile gate; not model result, new standard or complete joint gate')
    planning = []
    deff_max = float(count_per_unit @ count_per_unit / count_per_unit.sum())
    for delta in (.03, .05):
        for q in (delta, .10, .20):
            n = required_independent_pairs(delta, q)
            normal_n = math.ceil(((norm.ppf(.975) * math.sqrt(q) + norm.ppf(.8) * math.sqrt(q - delta ** 2)) / delta) ** 2)
            planning.append(dict(target_delta_pp=100 * delta, assumed_discordance=q,
                assumed_rescue_probability=(q + delta) / 2, assumed_loss_probability=(q - delta) / 2,
                exact_required_independent_pairs=n, power_at_n=exact_improvement_power(n, delta, q),
                power_at_n_minus_1=exact_improvement_power(n-1, delta, q),
                normal_approx_pairs=normal_n, independent_benchmark_power_n31=exact_improvement_power(31, delta, q),
                independent_benchmark_power_n26=exact_improvement_power(26, delta, q),
                one_eligible_episode_per_independent_unit_required_units=n,
                illustrative_cluster_event_n_rho1=math.ceil(n * deff_max),
                cluster_scaling_role='Variance sensitivity using existing cluster size mix, not exact clustered power or verified future ICC'))
    # Existing ZJU gate defect remains a defect; no outcome is re-adjudicated.
    zpath = bind(WORK / 'cnh-zju-inverse-affine-20261003/result.json')
    zju = read(zpath)
    zgates = []
    for budget, entry in zju['budgets'].items():
        principal = entry['policies']['pooled']
        for arm in ('depthor', principal['chosen_tof']):
            rate = principal['eval'][arm]['main_recall']
            zgates.append(dict(budget=budget, comparator=arm, baseline=rate, required=rate+.03, reachable=rate+.03<=1))
    bind(Path(__file__))
    result = dict(status='COMPLETE', role='Consumed Development statistical feasibility audit; no new evaluation standard',
        counts=dict(evaluation_units=96, calibration_units=48, episodes=int(ev.sum()), covered=int(covered.sum()),
            right_censored=int((~covered).sum()), shallow=int(shallow.sum()), shallow_units=len(clusters),
            M3_timely=int((shallow & base['timely']).sum()), M3_remaining_misses=len(miss),
            max_possible_gain_pp=100*len(miss)/shallow.sum(), event_resolution_pp=100/shallow.sum(),
            deep=int(deep.sum()), clear=int(clear.sum()), clear_proxy_minutes=clear.sum()*2.6/60,
            outside_clear=int(outside.sum()), outside_proxy_minutes=outside.sum()*2.6/60),
        clusters=clusters, M3_miss_rows=[dict(unit=int(eunit[i]), config=int(g['config'][ev][i]), query=int(g['query'][ev][i])) for i in miss],
        comparisons=comparisons, original_bootstrap_gate_attainability=repair_audit,
        iid_exact_ceiling=dict(best_case_rescues=len(miss), losses=0,
            two_sided_p=float(binomtest(len(miss), len(miss), .5).pvalue), one_sided_p=2.**(-len(miss)),
            role='Independent-event idealization, not the frozen bootstrap rule; clustered dependence cannot be ignored'),
        planning_assumptions=dict(alpha=.05, tails='two-sided exact paired McNemar, positive-tail improvement power', power=.8,
            estimand='paired timely-rate difference conditional on covered shallow-contact episodes',
            discordance='P(candidate differs from baseline); assumed scenarios, not estimated from small selected losses',
            independence='One prespecified eligible event per independently generated/source unit for listed unit counts',
            cluster_deff_formula='1 + rho*(sum(cluster_sizes^2)/sum(cluster_sizes)-1)', deff_rho0=1., deff_rho1=deff_max,
            exclusions='No calibration, retraining, model-seed, simulator-domain or multiplicity uncertainty in this calculation',
            no_single_mde='n31/26units alone does not identify MDE; discordance, unit correlation, test and joint guardrails matter'),
        conditional_sample_planning=planning, old_zju_gain_gate=zgates,
        calibration_and_cost=dict(thresholds=thresholds, no_recalibration=True, per_query_exposure_s=2.6,
            caution='Exposure includes time after first stop; parallel queries and scenes are correlated; proxy minutes are not actual walking minutes',
            future_design='Keep separately frozen calibration, independent evaluation units, outside-clear negatives and overall burden guardrails. Power table is shallow marginal test only, not joint trial power.'),
        censoring='6542 unreached deadline episodes are not misses; 1138/7680 coverage conditions estimand. No unobserved outcomes, extrapolation or survivor correction inferred.',
        original_results_preserved=True, input_sha256=hashes)
    assert result['counts']['shallow'] == 31 and result['counts']['shallow_units'] == 26
    assert result['counts']['M3_timely'] == 26 and result['counts']['right_censored'] == 6542
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(counts=result['counts'], planning=planning, zju=zgates), ensure_ascii=False))
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
