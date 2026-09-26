"""Soft candidate priors on repaired-v2 consumed Development; no formal claims.

One policy's eight score matrices are loaded at a time. Joint selection uses
exact threshold intervals for the FIRST positive-frame hit, including
nonmonotonic witness trajectories; every fixed threshold is evaluated.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cnh_position_prior_analysis as metrics

CONDITIONS = ('G1', 'FA1', 'FA3', 'BGFA1', 'BGFA3', 'DROP40', 'COMBO_MODERATE', 'COMBO_SEVERE')
DELTAS = (0., .25, .5, .75, 1., 1.5, 2., 3.)
AUCS = (80, 90, 95, 99)
GRID = np.r_[np.linspace(1., 12., 441), np.inf]


def specification():
    return dict(scope='Repaired v2 consumed Development; privileged synthetic candidates and confidence, not camera results',
        cohorts=dict(calib=list(range(96, 128)), eval=[u for u in range(128, 192) if u != 143]),
        conditions=list(CONDITIONS), confidence_auc=[a/100 for a in AUCS], deltas=list(DELTAS),
        G1s='max(G0,hard_local+delta); local threshold = global threshold-delta',
        G1c='max(G0,max_admitted_window(z+delta*max_intersecting_candidate_confidence))',
        confidence='c=Phi(latent); false N(0,1); true N(sqrt(2)*Phi^-1(AUC),1); fixed shared iid normal draws across AUC',
        threshold_grid='linspace(1,12,441)+infinity; all thresholds evaluated for all deltas',
        objective='per HEAD/BODY group, maximize all-size calib near-event first-positive-hit timely count; ties fewer false alerts, smaller delta, larger global threshold',
        baseline='Use original saved repaired-v2 G0/hardG1/G1s locals; G0 and hard G1 keep old lowest-feasible selection; retention denominator always original hard G1 minus G0. New score parity tolerance1e-5 with identical -inf masks.',
        fallback='Confidence delta0 validated exact new recomputed G0, then replaced by original saved G0 for exact fallback; included in joint search. Guarantees no-worse calib objective, not audit performance.',
        gate='Moderate combo retains>=50% original hard G1 gain at AUC<=0.90, separately by group; actual FA>G0+2pp flagged separately, not incorporated into this gate',
        gate_budget='10%; also report retention and flags at all three budgets',
        evidence_limits='Finite consumed cohort, synthetic iid per-frame confidence/perturbations; no universal camera or architecture claim')


def threshold_counts(p, score, grid=GRID):
    """Return all-grid false-pair and near-timely counts without a 3D tensor."""
    if score.shape != p['y'].shape or np.isnan(score).any() or np.isposinf(score).any():
        raise ValueError('Invalid score matrix')
    empty_max = np.sort(score[p['empty']].max(1))
    false = len(empty_max)-np.searchsorted(empty_max, grid, side='left')
    y, w, s = p['y'][p['near']], p['w'][p['near']], score[p['near']]
    previous = np.full(len(s), -np.inf)
    change = np.zeros(len(grid)+1, np.int64)
    for t in range(s.shape[1]):
        current = np.where(y[:, t], s[:, t], -np.inf)
        good = y[:, t] & (w[:, t] >= 1.) & (current > previous)
        low = np.searchsorted(grid, previous[good], side='right')
        high = np.searchsorted(grid, current[good], side='right')
        change += np.bincount(low, minlength=len(change))
        change -= np.bincount(high, minlength=len(change))
        previous = np.maximum(previous, current)
    return false, np.cumsum(change)[:-1]


def select_joint(p, scores):
    if len(scores) != len(DELTAS) or not p['empty'].any() or not p['near'].any():
        raise ValueError('Joint selection requires eight deltas, empty pairs and near events')
    best = {b: None for b in metrics.BUDGETS}
    for index, (delta, score) in enumerate(zip(DELTAS, scores)):
        false, timely = threshold_counts(p, score)
        for budget in metrics.BUDGETS:
            for j in np.flatnonzero(false/p['empty'].sum() <= budget):
                key = (int(timely[j]), -int(false[j]), -delta, float(GRID[j]))
                if best[budget] is None or key > best[budget][0]:
                    best[budget] = (key, index)
    out = {}
    for budget, found in best.items():
        if found is None:
            raise ValueError('No feasible joint threshold including disabled fallback')
        key, index = found
        out[budget] = dict(delta_index=index, delta=DELTAS[index], threshold=key[3],
                           calib_near_timely=key[0], calib_false_alerts=-key[1])
    return out


def _equal(a, b):
    if np.issubdtype(a.dtype, np.inexact):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def score_parity(reference, observed):
    if reference.shape != observed.shape or not np.array_equal(np.isneginf(reference), np.isneginf(observed)):
        raise ValueError('Score shape or negative-infinity mask mismatch')
    finite = np.isfinite(reference)
    if np.isnan(observed).any() or np.isposinf(observed).any():
        raise ValueError('Nonfinite score')
    error = np.abs(reference[finite]-observed[finite])
    relative = error/np.maximum(1., np.abs(reference[finite]))
    if relative.size and relative.max() > 1e-5:
        raise ValueError('Recomputed score parity exceeds1e-5')
    return dict(max_abs=float(error.max()) if error.size else 0., max_rel=float(relative.max()) if relative.size else 0.)


def load_metadata(source, new):
    """Open only common metadata/baselines and establish exact source alignment."""
    source, new = Path(source), Path(new)
    expected = specification()['cohorts']
    source_ids = sorted(int(p.stem[4:]) for p in source.glob('unit*.npz'))
    new_ids = sorted(int(p.stem[4:]) for p in new.glob('unit*.npz'))
    if source_ids != expected['calib']+expected['eval'] or new_ids != source_ids:
        raise ValueError('Require exact95 repaired-v2 cohort; original INCOMPLETE143 excluded')
    seqs, layouts = {'calib': [], 'eval': []}, {}
    common = ('config', 'frame', 'labels', 'witness', 'strata', 'split')
    parity = {}
    for unit in source_ids:
        with np.load(source/f'unit{unit:02d}.npz', allow_pickle=False) as ref, np.load(new/f'unit{unit:02d}.npz', allow_pickle=False) as fresh:
            data = {k: ref[k] for k in common}
            for key in common:
                if not _equal(data[key], fresh[key]):
                    raise ValueError(f'unit{unit} source metadata/baseline mismatch {key}')
            parity[str(unit)] = {key: score_parity(ref[key], fresh[key]) for key in ('G0', *CONDITIONS)}
            data.update(G0=ref['G0'], G1=ref['G1'])
            split = {'calib': 'calib', 'audit': 'eval'}[str(data['split'])]
            if unit not in expected[split]:
                raise ValueError(f'unit{unit} split mismatch')
            records = []
            for cfg in np.unique(data['config']):
                ix = np.flatnonzero((data['config'] == cfg) & (data['frame'] >= 3))
                ix = ix[np.argsort(data['frame'][ix])]
                if not np.array_equal(data['frame'][ix], np.arange(3, 12)):
                    raise ValueError('Incomplete/duplicate frames')
                records.append((len(seqs[split]), ix))
                seqs[split].append(dict(unit=unit, y=data['labels'][ix], w=data['witness'][ix],
                    strata=data['strata'][ix].astype(str), s={a: data['G0'][ix] for a in metrics.ARMS}, hard=data['G1'][ix]))
            layouts[unit] = (split, records)
    packed = {}
    for split in seqs:
        packed[split] = {}
        for group, boxes in metrics.GROUPS.items():
            p = metrics.pack(seqs[split], boxes)
            p['s'] = {'G0': p['s']['G0'], 'G1': np.concatenate([s['hard'][:, boxes].T for s in seqs[split]])}
            packed[split][group] = p
    return packed, layouts, parity


def load_policy_scores(root, keys, packed, layouts):
    """At most eight deltas in memory, independent of the number of policies."""
    out = {s: {g: np.full((len(keys), *p['y'].shape), -np.inf) for g, p in groups.items()} for s, groups in packed.items()}
    for unit, (split, records) in layouts.items():
        with np.load(Path(root)/f'unit{unit:02d}.npz', allow_pickle=False) as f:
            new_g0 = f['G0'] if keys[0].startswith('C__') else None
            for d, key in enumerate(keys):
                value = f[key]
                if value.ndim != 2 or value.shape[1] != 6 or np.isnan(value).any() or np.isposinf(value).any():
                    raise ValueError(f'unit{unit}/{key}: invalid score')
                if key.startswith('C__') and key.endswith('__D0') and not np.array_equal(value, new_g0):
                    raise ValueError(f'unit{unit}/{key}: confidence D0 not exact recomputed G0')
                for seq_index, ix in records:
                    for group, boxes in metrics.GROUPS.items():
                        out[split][group][d, seq_index*3:(seq_index+1)*3] = value[ix][:, boxes].T
    return out


def evaluate(p, score, threshold):
    return metrics.evaluate(dict(p, s={'selected': score}), 'selected', (threshold,))


def result_checks(results):
    retention = {}
    for key, row in results.items():
        group, policy, budget = key.split('/')
        base = results[f'{group}/G0/{budget}']['eval']
        hard = results[f'{group}/G1/{budget}']['eval']
        now = row['eval']
        b, h, v = [r['tiny']['timely']['rate'] for r in (base, hard, now)]
        gain = h-b if h is not None and b is not None else None
        delta = v-b if v is not None and b is not None else None
        keep = delta/gain if delta is not None and gain is not None and gain > 0 else None
        bfa, vfa = [r['all']['false_alert']['rate'] for r in (base, now)]
        fa_delta = vfa-bfa if vfa is not None and bfa is not None else None
        row['tiny_gain_over_G0'] = delta
        row['fixed_hard_G1_gain_over_G0'] = gain
        row['retention_of_fixed_hard_G1_gain'] = keep
        row['eval_false_alert_delta_over_G0'] = fa_delta
        row['eval_false_alert_exceeds_G0_by_2pp'] = bool(fa_delta > .02) if fa_delta is not None else None
        retention[key] = dict(retention=keep, gain=delta, false_alert_delta=fa_delta,
                              false_alert_exceeds_2pp=row['eval_false_alert_exceeds_G0_by_2pp'])
    gates = {}
    for group in metrics.GROUPS:
        tested = []
        for auc in (80, 90):
            row = results[f'{group}/G1c__COMBO_MODERATE__A{auc}/0.10']
            keep = row['retention_of_fixed_hard_G1_gain']
            tested.append(dict(auc=auc/100, retention=keep,
                retains_half=bool(keep >= .5) if keep is not None else None,
                false_alert_exceeds_G0_by_2pp=row['eval_false_alert_exceeds_G0_by_2pp']))
        gates[group] = dict(tested=tested, passes_any_auc_at_most_090=any(r['retains_half'] is True for r in tested),
                            note='FA flag separate; this gate alone does not mandate an architecture')
    return retention, gates


def analyze(scores, reference):
    packed, layouts, parity = load_metadata(reference, scores)
    out = dict(specification=specification(), recomputed_score_parity=parity,
               baseline_provenance='Original saved repaired-v2 scores for G0, hard G1 and G1s locals; confidence D0 verified exact recomputed G0 then uses original saved G0 for selection/evaluation', results={})
    for group in metrics.GROUPS:
        ca, ev = packed['calib'][group], packed['eval'][group]
        for arm in ('G0', 'G1'):
            for budget in metrics.BUDGETS:
                threshold = metrics.select(ca, arm, budget)[0]
                out['results'][f'{group}/{arm}/{budget:.2f}'] = dict(selection='old lowest feasible threshold',
                    threshold=metrics.threshold_json((threshold,))[0], delta=None,
                    calib=evaluate(ca, ca['s'][arm], threshold), eval=evaluate(ev, ev['s'][arm], threshold))
    for condition in CONDITIONS:
        variants = [(f'G1s__{condition}', None)]+[(f'G1c__{condition}__A{auc}', auc) for auc in AUCS]
        for policy, auc in variants:
            keys = [condition] if auc is None else [f'C__{condition}__A{auc}__D{i}' for i in range(len(DELTAS))]
            arrays = load_policy_scores(reference if auc is None else scores, keys, packed, layouts)
            for split in packed:
                for group in metrics.GROUPS:
                    base = packed[split][group]['s']['G0']
                    if auc is None:
                        local = arrays[split][group][0]
                        arrays[split][group] = np.stack([np.maximum(base, local+d) for d in DELTAS])
                    else:
                        # Scoring validates D0 against its own recomputed G0;
                        # preserve the exact fixed historical baseline here.
                        arrays[split][group][0] = base
                    if not np.array_equal(arrays[split][group][0], base):
                        raise ValueError(f'{policy}/{split}/{group}: delta0 must be exact G0')
            for group in metrics.GROUPS:
                ca, ev = packed['calib'][group], packed['eval'][group]
                selected = select_joint(ca, arrays['calib'][group])
                for budget, selection in selected.items():
                    index, threshold = selection['delta_index'], selection['threshold']
                    baseline = out['results'][f'{group}/G0/{budget:.2f}']['calib']['all']['timely']['numerator']
                    if selection['calib_near_timely'] < baseline:
                        raise AssertionError('delta0 fallback failed calib no-worse objective')
                    out['results'][f'{group}/{policy}/{budget:.2f}'] = dict(
                        condition=condition, confidence_auc=None if auc is None else auc/100,
                        delta_index=index, delta=selection['delta'], threshold=metrics.threshold_json((threshold,))[0],
                        local_threshold=metrics.threshold_json((threshold-selection['delta'],))[0] if auc is None else None,
                        calib_objective_near_timely=selection['calib_near_timely'], calib_false_alerts=selection['calib_false_alerts'],
                        calib_objective_no_worse_than_G0=True,
                        calib=evaluate(ca, arrays['calib'][group][index], threshold),
                        eval=evaluate(ev, arrays['eval'][group][index], threshold))
            del arrays
    out['retention_all_budgets'], out['moderate_combo_gate_at_10pct'] = result_checks(out['results'])
    out['retention_at_10pct'] = {k: v for k, v in out['retention_all_budgets'].items() if k.endswith('/0.10')}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', nargs='?')
    parser.add_argument('reference_scores', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--write-spec', type=Path)
    args = parser.parse_args()
    if args.write_spec:
        args.write_spec.write_text(json.dumps(specification(), indent=2, allow_nan=False), encoding='utf-8')
        return
    if not all((args.scores, args.reference_scores, args.output)):
        parser.error('scores reference_scores output required')
    result = analyze(args.scores, args.reference_scores)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['moderate_combo_gate_at_10pct'], indent=2))


if __name__ == '__main__':
    main()
