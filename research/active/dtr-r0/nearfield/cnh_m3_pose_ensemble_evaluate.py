"""Evaluation-only frozen M3 pose-ensemble pilot; never constructs network inputs.

Raw five-seed mean query logits enter this module after hypothesis aggregation.
The original causal five-logit smoother is applied once, independently per episode.
Evaluation-batch cost matching is kept separate from natural calibration.
"""
import argparse
from pathlib import Path
import time

import numpy as np

import cnh_unknown_target_reference_evaluate as U
import cnh_pose_marginal_evaluate as P
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE
import cnh_temporal_readout_evaluate as TE

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-m3-pose-ensemble-20261004'
FRAMES = np.arange(3, 16)
BASE = P.BASE
PRIMARY = 'main1.2-2.1m'
N_BOOT = 1000
BOOT_SEED = 2026100431


def boot_weights(n, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return np.array([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(N_BOOT)])


def ordered(raw, rows, *, natural=False):
    """Complete episode identity, not serialization order, defines smoothing."""
    raw = np.asarray(raw, np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError('Finite raw logits [row, HEAD/BODY] required')
    fields = ('unit', 'config') if natural else ('unit', 'variant', 'replica')
    for key in (*fields, 'frame'):
        if key not in rows or rows[key].shape != (len(raw),):
            raise ValueError('Missing or unaligned prediction identity: '+key)
    key_rows = list(zip(*(rows[k].astype(int).tolist() for k in fields)))
    keys = sorted(set(key_rows))
    lookup = {key: [] for key in keys}
    for i, key in enumerate(key_rows):
        lookup[key].append(i)
    episodes = []
    for key in keys:
        ids = np.asarray(lookup[key])
        ids = ids[np.argsort(rows['frame'][ids], kind='stable')]
        if not np.array_equal(rows['frame'][ids], FRAMES):
            raise ValueError('Exactly one retained decision per frame3..15 required: '+str(key))
        episodes.append(raw[ids])
    smooth = SE.smooth(np.stack(episodes)[:, None])[:, 0]
    return smooth, keys


def controlled(raw, rows, scenes):
    value, keys = ordered(raw, rows)
    expected = [(s['unit'], v, k) for s in scenes for v in range(7) for k in range(4)]
    if keys != expected:
        raise ValueError('Controlled predictions must contain all48 x7 xK4 episodes')
    value = value.reshape(48, 7, 4, 13, 2)
    return np.stack([value[i, ..., s['group']] for i, s in enumerate(scenes)])


def groups_for(scenes):
    high = np.array([s['visibility']['fov_in'] for s in scenes])
    groups = dict(all=np.ones(48, bool), FOV_IN=high, FOV_OUT=~high)
    for q, label in ((0, 'HEAD'), (1, 'BODY')):
        groups[label] = np.array([s['group'] == q for s in scenes])
        for context in ('none', 'panel'):
            groups[label+'/'+context] = np.array([s['group'] == q and s['context'] == context for s in scenes])
    return groups


def fresh_geometry():
    source = TE.OUT
    scenes, visible, paths = TE.fresh_geometry(source)
    categories = []
    for scene in scenes:
        truth = U.read(source/'truth/evaluation'/f"unit{scene['unit']}.json")
        label = np.asarray(truth['categories'])
        if label.shape != (7, 16, 2):
            raise ValueError('Fresh actual all-object category axes differ')
        categories.append(label[:, FRAMES, scene['group']])
        scene['visibility'] = dict(vf=scene.pop('visibility_fraction'), fov_in=scene.pop('fov_in'))
    cats = np.stack(categories)
    if not np.isin(cats, ['clear', 'pass0-10cm', 'contact0-2cm', 'contact2-5cm', 'contact>5cm']).all():
        raise ValueError('UNKNOWN category cannot be counted as clear')
    return scenes, cats, visible, paths


def load_raw(out, split, arm, hashes):
    path = out/'predictions'/f'{arm}_{split}.npz'
    identity_path = out/'inputs'/split/'rows.npz'
    receipt_path = path.with_suffix('.json')
    receipt = U.read(receipt_path)
    if receipt.get('status') != 'COMPLETE' or receipt.get('score_sha256') != U.sha(path):
        raise ValueError('Sealed prediction receipt mismatch: '+str(path))
    if receipt.get('plan_sha256') != U.sha(out/'PLAN.json'):
        raise ValueError('Prediction belongs to another plan')
    with np.load(path, allow_pickle=False) as z:
        raw = z['raw'].copy()
    with np.load(identity_path, allow_pickle=False) as z:
        rows = {key: z[key] for key in z.files}
    for p in (path, receipt_path, identity_path):
        hashes[str(p)] = U.sha(p)
    return raw, rows


def natural(out, arms, hashes):
    geometry, baseline_scores, provenance = OE.load_inputs()
    old_path = OE.OUT/'result.json'
    old = U.read(old_path)
    hashes[str(old_path)] = U.sha(old_path)
    cal = geometry['split'] == 'calib'
    ev = geometry['split'] == 'evaluation'
    scores = {BASE: baseline_scores['M3'][ev]}
    thresholds = {BASE: dict(threshold=P.BASE_THRESHOLD, operator='>=')}
    calibration = {BASE: old['cells']['M3']['calibration']}
    for arm in arms:
        for split, selected in (('calibration', cal), ('evaluation', ev)):
            raw, rows = load_raw(out, split, arm, hashes)
            value, keys = ordered(raw, rows, natural=True)
            expected = sorted(set(zip(geometry['unit'][selected].astype(int), geometry['config'][selected].astype(int))))
            if keys != expected:
                raise ValueError('Natural predictions differ from original geometry identity')
            query_scores = value.transpose(0, 2, 1).reshape(-1, 13)
            if split == 'calibration':
                threshold, cal_receipt = SE.calibrate(query_scores, geometry['clear_all'][cal], 1.)
                thresholds[arm] = dict(threshold=threshold, operator='>=')
                calibration[arm] = cal_receipt
            else:
                scores[arm] = query_scores
    g = {key: geometry[key][ev] for key in ('unit', 'query', 'frame_ranges', 'frame_category', 'ref_category', 'clear_all', 'covered', 'last_category')}
    units, ui = np.unique(g['unit'], return_inverse=True)
    boot = boot_weights(len(units))
    weights = {category: (g['covered'] & (g['ref_category'] == category)).astype(float) for category in OE.CONTACTS+('pass0-10cm',)}
    weights['clear'] = g['clear_all'].astype(float)
    target_rows = U.read(OE.OUT/'rows.json')
    target_group = np.array([r['target_group'] for r, keep in zip(target_rows, ev) if keep])
    cells, samples, flags = {}, {}, {}
    for arm, value in scores.items():
        stopped, timely, lead = SE.first_stops(value, thresholds[arm]['threshold'], g['frame_ranges'])
        metrics, draws = SE.summarize(weights, g['covered'], stopped, timely, lead, ui, boot)
        for metric in metrics.values():
            metric.pop('episode_draw_pairs', None)
        clear_num = SE.unit_totals(weights['clear']*stopped, ui, len(units))
        clear_den = SE.unit_totals(weights['clear'], ui, len(units))*2.6/60
        _, draws['clear'] = SE.rates(clear_num, clear_den, boot)
        other = g['clear_all'] & (g['query'] != target_group)
        cells[arm] = dict(threshold=thresholds[arm]['threshold'], calibration=calibration[arm], metrics=metrics,
            other_height_clear=dict(stops=int((other & stopped).sum()), n=int(other.sum()), proxy_minutes=float(other.sum()*2.6/60)),
            censored=dict(n=int((~g['covered']).sum()), already_alarm=int((~g['covered'] & stopped).sum()), no_alarm_right_censored=int((~g['covered'] & ~stopped).sum())),
            timing_diagnostics={category: OE.extra_counts(weights[category] > 0, stopped, timely, value, thresholds[arm]['threshold']) for category in OE.CONTACTS+('pass0-10cm',)})
        samples[arm] = draws
        flags[arm] = dict(stopped=stopped, timely=timely)
    comparisons = {}
    for arm in arms:
        comparison = {}
        for category, metric in (('contact0-2cm', 'timely_rate'), ('contact>5cm', 'timely_rate'), ('clear', 'false_stops_per_min')):
            a = cells[arm]['metrics'][category][metric]['value']
            b = cells[BASE]['metrics'][category][metric]['value']
            comparison[category] = dict(delta=a-b, paired_unit_ci95=SE.interval(samples[arm][category]-samples[BASE][category]))
            if category != 'clear':
                selected = weights[category] > 0
                a_flag, b_flag = flags[arm]['timely'], flags[BASE]['timely']
                comparison[category].update(rescues=int((selected & a_flag & ~b_flag).sum()), losses=int((selected & b_flag & ~a_flag).sum()))
        comparison['guardrail_pass'] = comparison['clear']['delta'] <= .1+1e-12 and comparison['contact>5cm']['delta'] >= -.02-1e-12
        comparison['guardrail_role'] = 'Secondary inherited V/Tpilot tolerance: clear<=M3+.1/min and deep>=M3-.02; does not imply equal evaluation clear burden'
        comparisons[arm] = comparison
    for category, expected in [('contact0-2cm', (26., 31.)), ('contact>5cm', (162., 164.))]:
        metric = cells[BASE]['metrics'][category]
        if (metric['expected_timely_stops'], metric['expected_episodes']) != expected:
            raise ValueError('Natural frozen M3 baseline parity failed')
    clear = cells[BASE]['metrics']['clear']
    if clear['expected_stops'] != 205 or not np.isclose(clear['clear_minutes'], 194.35):
        raise ValueError('Natural frozen M3 clear-cost parity failed')
    return dict(cells=cells, comparisons_minus_M3=comparisons, source_provenance=provenance,
        scope='Natural95000 calibration once <=1 first stop/proxy minute;96000 evaluation covered contacts only; unalarmed rightcensoring is not a missed deadline'), thresholds


def cohort_summary(scores, scenes, categories, visible, thresholds=None, *, original=False):
    if any(value.shape != (48, 7, 4, 13) or not np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Complete finite controlled score tensors required')
    boot = boot_weights(48)
    groups = groups_for(scenes)
    ranges = np.array([s['front_range_m'] for s in scenes])
    auc = P.auc_summary(scores, ranges, groups, boot)
    # Add all frozen distance bins without modifying the reused helper's globals.
    for lo, hi in U.BINS:
        name = f'{lo:g}-{hi:g}m'
        mask = (ranges >= lo) & (ranges < hi)
        for group, keep in groups.items():
            auc['cells'][group][name] = {}
            auc['changes_minus_M3'][group][name] = {}
            base_draws = None
            for arm, value in scores.items():
                scene_auc = np.array([U.binary_auc(value[i, [0, 1]][..., m], value[i, [4, 5, 6]][..., m]) for i, m in enumerate(mask)], float)
                metric, draws = U.macro_summary(scene_auc, keep, boot)
                metric.update(positive_n=int(mask[keep].sum()*8), negative_n=int(mask[keep].sum()*12))
                auc['cells'][group][name][arm] = dict(macro=metric)
                if arm == BASE:
                    base_draws, base_value = draws, metric['value']
                else:
                    auc['changes_minus_M3'][group][name][arm] = dict(macro=dict(delta=metric['value']-base_value if metric['value'] is not None and base_value is not None else None, paired_scene_ci95=U.interval(draws-base_draws)))
    result = dict(auc=auc, scenes=scenes,
        n=dict(scenes=48, FOV_IN=int(groups['FOV_IN'].sum()), FOV_OUT=int(groups['FOV_OUT'].sum())),
        provenance='Original consumed displacement Development' if original else 'Consumed V/Tpilot random scenes; reused Development, not protected confirmation')
    ledgers = {}
    if thresholds is not None:
        seq, ledgers = P.sequence_summary(scores, thresholds, scenes, categories, visible, groups, boot)
        result['natural_calibrated_sequences'] = seq
    else:
        result['natural_calibrated_sequences'] = dict(status='NOT_RUN', reason='Original-only prospective screen; no candidate natural calibration has been computed')
    if original:
        P.baseline_check_data(scores[BASE], scenes, categories, visible)
        matched = P.episode_thresholds(scores, categories)
        matched_seq, matched_ledgers = P.sequence_summary(scores, matched, scenes, categories, visible, groups, boot)
        if thresholds is None:
            ledgers = matched_ledgers
        result['descriptive_original18_384_cost'] = dict(thresholds=matched, sequences=matched_seq,
            interpretation='One merged whole48scene evaluation threshold per candidate; strict whole ties <=18/384 actualclear firststops. Not deployment calibration; same threshold retained in every stratum.')
    return result, ledgers


def zero_parity(score, baseline, ranges):
    """Control scores and events are compared at the frozen original threshold."""
    difference = np.abs(score-baseline)
    a = P.event_arrays(score, P.BASE_THRESHOLD, '>=', ranges)
    b = P.event_arrays(baseline, P.BASE_THRESHOLD, '>=', ranges)
    equality = {name: bool(np.array_equal(a[name], b[name])) for name in ('alarm', 'stopped', 'first_index', 'timely')}
    # Ranking reversals matter only for untied original score pairs; rank vectors
    # also disclose newly split ties, without claiming identical order from eps.
    from scipy.stats import rankdata
    old_rank = rankdata(baseline.ravel(), method='average')
    new_rank = rankdata(score.ravel(), method='average')
    return dict(status='PASS' if float(difference.max()) <= 1e-4 and all(equality.values()) else 'FAIL',
        max_abs_smoothed_logit_difference=float(difference.max()), tolerance=1e-4,
        event_equality_at_frozen_M3_threshold=equality,
        global_rank_positions_changed=int(np.count_nonzero(old_rank != new_rank)),
        ranking_note='Includes tie splits; a tiny logit discrepancy alone does not certify exact score ordering')


def screen_decision(sequence, parity):
    change = sequence['changes_minus_M3']
    low = change['FOV_OUT']['P24']['timely']
    high = change['FOV_IN']['P24']['timely']
    signal = low['paired_scene_ci95'][0] is not None and low['paired_scene_ci95'][0] > 0
    guard = high['delta_rate'] >= -.02-1e-12
    passed = signal and guard and parity['status'] == 'PASS'
    branch = 'ORIGINAL_SIGNAL_NATURAL_REQUIRED' if passed else 'NOT_ESTABLISHED_NATURAL_NOT_RUN'
    if parity['status'] != 'PASS':
        branch = 'IMPLEMENTATION_PARITY_FAILED'
    return dict(branch=branch, screen_pass=passed, FOV_OUT=low, FOV_IN=high,
        Z_parity=parity['status'], FOV_OUT_paired_ci_lower_positive=signal, FOV_IN_decline_le2pp=guard,
        rule='Prospective amendment: FOVout P24-M3 timely pairedscene95%lower>0 AND FOVin pointdecline<=.02 AND Zparity; original18/384 diagnostic frontier only. Passing requires complete natural calibration/evaluation before practical benefit recommendation.')


def matched_attribution(scores, ledgers, scenes, categories):
    """Same-cost P24-minus-Z isolates pose averaging from threshold selection.

    This is supplemental attribution; it cannot rewrite the frozen comparison
    of the candidate against M3 at M3's original threshold.
    """
    boot = boot_weights(48)
    groups = groups_for(scenes)
    groups = {name: groups[name] for name in ('all', 'FOV_IN', 'FOV_OUT')}
    ranges = np.array([s['front_range_m'] for s in scenes])
    sequence = {}
    for group, keep in groups.items():
        sequence[group] = {}
        for field in ('timely', 'stopped'):
            a, b = ledgers['P24'][field][:, :2], ledgers['Z'][field][:, :2]
            covered = np.broadcast_to(ledgers['Z']['covered'][:, None, None] if field == 'timely' else True, a.shape)
            am, ad = P.count_metric(a & covered, covered, keep, boot)
            bm, bd = P.count_metric(b & covered, covered, keep, boot)
            sequence[group][field] = dict(delta_rate=am['rate']-bm['rate'], paired_scene_ci95=U.interval(ad-bd),
                P24=am, Z=bm, rescues=int((a & ~b & covered)[keep].sum()), losses=int((b & ~a & covered)[keep].sum()), n=am['n'])
    auc = {}
    for domain, (lo, hi) in {'main1.2-2.1m': (1.2, 2.1), 'far2.1-2.6m': (2.1, 2.6)}.items():
        mask = (ranges >= lo) & (ranges < hi)
        values = {arm: np.array([U.binary_auc(scores[arm][i, [0, 1]][..., m], scores[arm][i, [4, 5, 6]][..., m]) for i, m in enumerate(mask)], float) for arm in ('P24', 'Z')}
        auc[domain] = {}
        for group, keep in groups.items():
            macro = U.macro_summary(values['P24']-values['Z'], keep, boot)[0]
            pa, pd = P.weighted_pooled_auc(scores['P24'], mask, keep, boot)
            za, zd = P.weighted_pooled_auc(scores['Z'], mask, keep, boot)
            auc[domain][group] = dict(macro=dict(delta=macro['value'], paired_scene_ci95=macro['ci95'], scenes=macro['scenes']),
                pooled=dict(delta=pa['value']-za['value'] if pa['value'] is not None and za['value'] is not None else None,
                            paired_scene_ci95=U.interval(pd-zd), positive_n=pa['positive_n'], negative_n=pa['negative_n']))
    actual_clear = np.all(categories[:, [5, 6]] == 'clear', -1)
    actual_clear = np.broadcast_to(actual_clear[..., None], (48, 2, 4))
    costs = {arm: dict(stops=int((ledgers[arm]['stopped'][:, [5, 6]] & actual_clear).sum()), n=int(actual_clear.sum())) for arm in ('P24', 'Z')}
    cost_valid = all(value['n'] == 384 and value['stops'] <= 18 for value in costs.values())
    low = sequence['FOV_OUT']['timely']
    high = sequence['FOV_IN']['timely']
    supported = cost_valid and low['paired_scene_ci95'][0] is not None and low['paired_scene_ci95'][0] > 0 and high['delta_rate'] >= -.02-1e-12
    return dict(status='SUPPORTED' if supported else 'NOT_ESTABLISHED', mechanism_supported=supported,
        sequence_P24_minus_matched_Z=sequence, auc_P24_minus_Z=auc, actual_clear_costs=costs, cost_valid=cost_valid,
        rule='Supplemental attribution/routing only: lowP24-Ztimely pairedsceneCIlo>0, highpointdecline<=.02, both actualclear costs<=18/384. Preserves primaryP24-frozenM3 gate.',
        interpretation='Both candidates use their own wholebatch18/384 matched thresholds. Zscoreparity is separately checked at frozenM3 threshold. A P24-frozenM3 gain alone can include threshold interval effects.')


def original_screen(out=OUT):
    target = out/'original_screen.json'
    ledger_path = out/'event_ledgers_original_screen.npz'
    if target.exists() or ledger_path.exists():
        raise FileExistsError('Original-only completed screen evidence remains immutable')
    started = time.monotonic()
    plan_path, amendment_path = out/'PLAN.json', out/'PLAN_AMENDMENT.json'
    note_path = out/'IMPLEMENTATION_NOTE.json'
    plan, amendment, note = U.read(plan_path), U.read(amendment_path), U.read(note_path)
    if plan.get('run') != 'CNH_M3_POSE_ENSEMBLE_20261004' or plan['evaluation_arms'] != ['Z', 'P24']:
        raise ValueError('Wrong fixed pilot PLAN or candidate arms')
    if amendment.get('original_plan_sha256') != U.sha(plan_path) or amendment.get('status') != 'FIXED_BEFORE_SCIENTIFIC_CANDIDATE_INFERENCE':
        raise ValueError('Prospective cost-routing amendment does not match PLAN')
    if amendment['changes'].get('execution_order') != ['original', 'natural_only_if_original_signal']:
        raise ValueError('Unknown original-screen execution routing')
    if note.get('original_plan_sha256') != U.sha(plan_path) or note.get('status') != 'FIXED_BEFORE_ANY_FULL_CANDIDATE_EVALUATION':
        raise ValueError('Supplemental attribution note does not match original PLAN')
    hashes = {str(p): U.sha(p) for p in (plan_path, amendment_path, note_path)}
    baseline, scenes, cats, visible, paths = P.geometry()
    if plan['original_units'] != [s['unit'] for s in scenes] or plan['frames'] != FRAMES.tolist() or plan['history'] != 8:
        raise ValueError('Screen identities/history differ from frozen scientific PLAN')
    hashes.update({str(p): U.sha(p) for p in paths})
    scores = {BASE: baseline}
    for arm in plan['evaluation_arms']:
        raw, rows = load_raw(out, 'original', arm, hashes)
        scores[arm] = controlled(raw, rows, scenes)
    ranges = np.array([s['front_range_m'] for s in scenes])
    parity = zero_parity(scores['Z'], baseline, ranges)
    baseline_check = P.baseline_check_data(baseline, scenes, cats, visible)
    original, ledgers = cohort_summary(scores, scenes, cats, visible, original=True)
    decision = screen_decision(original['descriptive_original18_384_cost']['sequences'], parity)
    attribution = matched_attribution(scores, ledgers, scenes, cats)
    decision['natural_followup_supported'] = decision['screen_pass'] and attribution['mechanism_supported']
    decision['attribution_status'] = attribution['status']
    if any(U.sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Original screen input changed during evaluation')
    natural = dict(status='NOT_RUN', reason='Original signal and attribution require complete natural follow-up' if decision['natural_followup_supported'] else 'NOT_RUN_COST_JUSTIFIED',
        calibration_units_planned=plan['natural_calibration_units'], evaluation_units_planned=plan['natural_evaluation_units'],
        candidate_calibration_episodes_evaluated=0, candidate_evaluation_episodes_evaluated=0,
        interpretation='No natural candidate performance/cost benefit claimed; planned units are not evaluated denominators')
    if decision['branch'] == 'IMPLEMENTATION_PARITY_FAILED':
        natural['reason'] = 'Engineering parity must be diagnosed before any scientific recipe verdict'
    result = dict(status='COMPLETE_ORIGINAL_SCREEN', decision=decision, cohort=original,
        Z_parity=parity, matched_Z_attribution=attribution, baseline_check=baseline_check, natural=natural,
        bootstrap=dict(n=N_BOOT, seed=BOOT_SEED, unit='Whole48scenes; fixed score tensors and original evaluation-selected cost thresholds'),
        elapsed_s=time.monotonic()-started,
        provenance=dict(input_sha256=hashes, evaluator_sha256=U.sha(__file__),
                        helper_sha256={str(Path(m.__file__)): U.sha(m.__file__) for m in (P, U, SE)}))
    np.savez_compressed(ledger_path, **{f'{arm}__{field}': value for arm, fields in ledgers.items() for field, value in fields.items()})
    result['provenance']['event_ledger_sha256'] = U.sha(ledger_path)
    P.save(target, result)
    print('ORIGINAL SCREEN', decision['branch'], result['elapsed_s'], flush=True)
    return result


def evaluate(out=OUT):
    target = out/'result.json'
    if target.exists():
        raise FileExistsError('Completed result remains immutable')
    started = time.monotonic()
    plan = U.read(out/'PLAN.json')
    arms = plan['evaluation_arms']
    if not arms or BASE in arms or len(set(arms)) != len(arms):
        raise ValueError('Explicit unique candidate arms required in PLAN')
    hashes = {str(out/'PLAN.json'): U.sha(out/'PLAN.json')}
    natural_result, thresholds = natural(out, arms, hashes)
    original_base, original_scenes, original_cats, original_vis, original_paths = P.geometry()
    splits = plan.get('evaluation_splits', ['original'])
    if splits not in (['original'], ['original', 'fresh']):
        raise ValueError('PLAN must declare original with optional fresh secondary cohort')
    cohort_inputs = [('original', original_base, original_scenes, original_cats, original_vis)]
    hashes.update({str(p): U.sha(p) for p in original_paths})
    if 'fresh' in splits:
        fresh_scenes, fresh_cats, fresh_vis, fresh_paths = fresh_geometry()
        hashes.update({str(p): U.sha(p) for p in fresh_paths})
        cohort_inputs.append(('fresh', None, fresh_scenes, fresh_cats, fresh_vis))
    cohorts, ledgers = {}, {}
    for split, baseline, scenes, cats, vis in cohort_inputs:
        if baseline is None:
            old_path = TE.OUT/'predictions/M3_fresh_evaluation.npz'
            old_rows = TE.OUT/'inputs/fresh_evaluation/rows.npz'
            with np.load(old_path, allow_pickle=False) as z:
                raw = z['raw']
            with np.load(old_rows, allow_pickle=False) as z:
                rows = {k: z[k] for k in z.files}
            baseline = controlled(raw, rows, scenes)
            hashes.update({str(p): U.sha(p) for p in (old_path, old_rows)})
        scores = {BASE: baseline}
        for arm in arms:
            raw, rows = load_raw(out, split, arm, hashes)
            scores[arm] = controlled(raw, rows, scenes)
        cohorts[split], ledgers[split] = cohort_summary(scores, scenes, cats, vis, thresholds, original=split=='original')
    if any(U.sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Input changed during evaluation')
    result = dict(status='COMPLETE', natural=natural_result, cohorts=cohorts, evaluation_arms=arms,
        interpretation='Observable-only frozenM3 candidates; no training/baseline promotion; reused synthetic Development; cost-matched original frontier is descriptive',
        bootstrap=dict(n=N_BOOT, seed=BOOT_SEED, unit='Whole48controlled scenes or96natural units; fixed scores and thresholds'),
        elapsed_s=time.monotonic()-started,
        provenance=dict(input_sha256=hashes, evaluator_sha256=U.sha(__file__)))
    P.save(target, result)
    np.savez_compressed(out/'event_ledgers.npz', **{f'{split}__{arm}__{field}': value for split, arms_dict in ledgers.items() for arm, fields in arms_dict.items() for field, value in fields.items()})
    print('POSE ENSEMBLE evaluation COMPLETE', result['elapsed_s'], flush=True)
    return result


def check(out=OUT):
    # Deliberately shuffled rows prove episode membership and causal smoothing.
    rows = dict(unit=np.repeat([1, 2], 13), config=np.zeros(26, int), frame=np.tile(FRAMES, 2))
    raw = np.arange(52, dtype=float).reshape(26, 2)
    shuffle = np.random.default_rng(43).permutation(26)
    result, keys = ordered(raw[shuffle], {k: v[shuffle] for k, v in rows.items()}, natural=True)
    assert keys == [(1, 0), (2, 0)]
    np.testing.assert_array_equal(result[:, 0], raw.reshape(2, 13, 2)[:, 0])
    expected = np.tensordot(raw.reshape(2, 13, 2)[:, -5:], np.array([1, 2, 4, 8, 16])/31, axes=([1], [0]))
    np.testing.assert_allclose(result[:, -1], expected)
    changed = raw.copy(); changed[-1] += 1000
    later, _ = ordered(changed, rows, natural=True)
    np.testing.assert_array_equal(later[:, :-1], result[:, :-1])
    bad = {k: v.copy() for k, v in rows.items()}; bad['frame'][0] = 4
    try:
        ordered(raw, bad, natural=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate/missing decision silently accepted')
    baseline, scenes, cats, visible, _ = P.geometry()
    baseline_check = P.baseline_check_data(baseline, scenes, cats, visible)
    all_raw, row_fields = [], {k: [] for k in ('unit', 'variant', 'replica', 'frame')}
    for scene in scenes:
        unit = scene['unit']
        with np.load(P.OLD/'scores'/f'unit{unit}.npz', allow_pickle=False) as z:
            all_raw.append(z['raw'].reshape(-1, 2))
        for variant in range(7):
            for replica in range(4):
                row_fields['unit'].extend([unit]*13)
                row_fields['variant'].extend([variant]*13)
                row_fields['replica'].extend([replica]*13)
                row_fields['frame'].extend(FRAMES)
    adapted = controlled(np.concatenate(all_raw), {k: np.array(v) for k, v in row_fields.items()}, scenes)
    np.testing.assert_allclose(adapted, baseline, rtol=0, atol=1e-12)
    g, old, _ = OE.load_inputs()
    cal, ev = g['split'] == 'calib', g['split'] == 'evaluation'
    threshold, calibration = SE.calibrate(old['M3'], cal & g['clear_all'], 1.)
    stopped, timely, _ = SE.first_stops(old['M3'][ev], threshold, g['frame_ranges'][ev])
    contacts = {}
    for name, expected in [('contact0-2cm', (26, 31)), ('contact>5cm', (162, 164))]:
        selected = g['covered'][ev] & (g['ref_category'][ev] == name)
        actual = (int((timely & selected).sum()), int(selected.sum()))
        assert actual == expected
        contacts[name] = dict(timely=actual[0], n=actual[1])
    clear = g['clear_all'][ev]
    assert int((clear & stopped).sum()) == 205 and int(clear.sum()) == 4485
    assert threshold == P.BASE_THRESHOLD and calibration['stops'] == 95
    # Check prospectively fixed screen boundaries independently of candidate outputs.
    seq = {'changes_minus_M3': {'FOV_OUT': {'P24': dict(timely=dict(delta_rate=.1, paired_scene_ci95=[.001, .2]))},
                                'FOV_IN': {'P24': dict(timely=dict(delta_rate=-.02, paired_scene_ci95=[-.1, .1]))}}}
    assert screen_decision(seq, dict(status='PASS'))['screen_pass']
    assert not screen_decision(seq, dict(status='FAIL'))['screen_pass']
    seq['changes_minus_M3']['FOV_OUT']['P24']['timely']['paired_scene_ci95'][0] = 0
    assert not screen_decision(seq, dict(status='PASS'))['screen_pass']
    # Exercise the complete original screen statistics on identical immutable
    # baseline clones; these are engineering controls, never candidate scores.
    integration_scores = {BASE: baseline, 'Z': baseline, 'P24': baseline}
    integration, null_ledgers = cohort_summary(integration_scores, scenes, cats, visible, original=True)
    null_decision = screen_decision(integration['descriptive_original18_384_cost']['sequences'],
        zero_parity(baseline, baseline, np.array([s['front_range_m'] for s in scenes])))
    assert not null_decision['screen_pass'] and null_decision['FOV_OUT']['delta_rate'] == 0
    null_attribution = matched_attribution(integration_scores, null_ledgers, scenes, cats)
    assert not null_attribution['mechanism_supported']
    assert null_attribution['sequence_P24_minus_matched_Z']['FOV_OUT']['timely']['delta_rate'] == 0
    receipt = dict(status='PASS', causal_smoothing=True, shuffled_identity=True, duplicate_missing_frames_rejected=True,
        original_baseline_check=baseline_check, raw_adapter_max_abs=float(np.abs(adapted-baseline).max()),
        natural_baseline=dict(threshold=threshold, calibration=calibration, contacts=contacts,
            clear_stops=205, clear_n=4485, clear_proxy_minutes=194.35, censored=int((~g['covered'][ev]).sum())),
        prospective_screen_boundaries=True, identical_baseline_screen=null_decision,
        identical_baseline_attribution=null_attribution,
        original_screen_helper_integration=True, candidate_outputs_read=False, source_sha256=U.sha(__file__))
    out.mkdir(parents=True, exist_ok=True)
    P.save(out/'evaluator_checks.json', receipt)
    print('POSE ENSEMBLE causal identity/smoothing check PASS', flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check', 'evaluate', 'original-screen'], required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == 'check':
        check(args.out)
    elif args.stage == 'original-screen':
        original_screen(args.out)
    else:
        evaluate(args.out)
