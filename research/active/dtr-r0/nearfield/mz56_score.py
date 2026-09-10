"""Independent saved-output CPU score for the registered MZ56 anchor arms."""
import argparse
from pathlib import Path
import time

import numpy as np

from mz15_train import cutoff_zero_added
from mz40_evaluate import metrics, paired
from mz47_score import scalar_check
from mz54_score import read, sha, write, arrays, static_geometry

ARMS = ('LOCAL_ONLY', 'GLOBAL_ANCHOR')
MODES = ARMS + ('GLOBAL_SUPPRESSED',)
KEYS = tuple('MZ56/' + arm for arm in MODES)
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
UNIONS = ('MZ50/UNION', 'NEW_NEG/UNION', 'OLD_NEG/UNION')
PRIMARY_DEFINITION = ('Pre-fit operational clarification: count nonfit BODY_NEAR true events added over '
    'MZ37 by this arm passing its existing cutoff, with a native winning cell outside the measured '
    '45-degree field. Compare GLOBAL_ANCHOR > LOCAL_ONLY, with no higher nonfit or legacy noncal FP. '
    'Inherited MZ37 positives do not count as new-path evidence; report all final TP separately.')


def run(task, run_name='run-v1'):
    started = time.perf_counter()
    task = task.resolve(strict=True)
    assert run_name in ('run-v1', 'run-v2')
    source, out = task / run_name, task / 'score-v1'
    report_path = Path(__file__).with_name('MZ56_GLOBAL_ANCHOR_RESULTS_20260911.md')
    assert not out.exists() and not report_path.exists(), 'Never overwrite a sealed score'
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        assert expected is None or digest == expected, f'Input mismatch: {path}'
        inputs[str(path)] = digest
        return path
    receipt = read(bind(source / 'receipt.json'))
    assert receipt['status'] == 'PASS' and receipt['run_name'] == run_name
    def bound(suffix):
        matches = [(p, h) for p, h in receipt['inputs'].items() if p.replace('\\', '/').endswith(suffix)]
        assert len(matches) == 1, suffix
        return bind(*matches[0])
    assert receipt['primary_operational_definition'] == PRIMARY_DEFINITION
    assert receipt['steps_per_arm'] == 600 and receipt['total_steps'] == 1200
    assert set(receipt['fits']) == set(ARMS) and all(receipt['fits'][a]['steps'] == 600 for a in ARMS)
    assert receipt['exact_initial_weights'] and receipt['trainable_parameters'] == 11020
    assert set(receipt['initial_parameter_sha256']) == set(ARMS)
    assert len(set(receipt['initial_parameter_sha256'].values())) == 1
    assert receipt['new_cutoffs'] == 2 and receipt['threshold_searches'] == 0
    assert receipt['source_unknown_preserved'] and receipt['native_depth_reads'] == 0
    assert receipt['original_baseline_cohort_inferences'] == receipt['new_training_feature_extractions'] == 0
    assert receipt['training_cache_mode'] == 'r' and not receipt['new_physical_dense_copy']
    assert receipt['cache_sha256_unchanged'] and receipt['cache_identity_before'] == receipt['cache_identity_after']
    assert len(receipt['cache_identity_after']['links']) == 1
    assert receipt['feature_counts'] == dict(training_extracted_frames=0, full_extracted_frames=5734, full_cache_eval_hits=1250)
    assert receipt['rgb_loads'] == 5734
    if run_name == 'run-v2':
        failure = read(bound('mz56-global-anchor-20260911/preflight-failure-v1/receipt.json'))
        assert failure['completed_fit_steps'] == failure['training_feature_extractions'] == failure['evaluation_feature_extractions'] == 0
    for name in ('MZ56_GLOBAL_ANCHOR_20260911.md', 'mz56_global_anchor.py', 'mz56_global_anchor_model.py',
                 'mz54_full_rgb.py', 'mz54_full_rgb_model.py', 'mz54_full_rgb_source.py'):
        bound('nearfield/' + name)
    handoff = read(bound('mz56-global-anchor-20260911/cache-handoff.json'))
    assert handoff['status'] == 'PASS' and handoff['owner'] == task.name
    # Bind the producer's cache identity and digest; the scorer does not read 4.4 GB of features.
    cache_hashes = [h for p, h in receipt['inputs'].items() if p.replace('\\', '/').endswith('mz56-global-anchor-20260911/scratch-v1/training-full.npy')]
    assert cache_hashes == [handoff['sha256']]
    needed = ('predictions.npz', 'groups.json', 'schedule.npz', 'feature-cache-index.npz',
              'LOCAL_ONLY-cutoff.npy', 'GLOBAL_ANCHOR-cutoff.npy', 'initialization-parity.json',
              'learned-all-missing.json', 'start.json')
    for name in needed:
        bind(source / name, receipt['outputs'][name])
    assert {n for n in receipt['outputs'] if n.endswith('-cutoff.npy')} == {'LOCAL_ONLY-cutoff.npy', 'GLOBAL_ANCHOR-cutoff.npy'}
    assert read(source / 'start.json')['primary_operational_definition'] == PRIMARY_DEFINITION
    parity = read(source / 'initialization-parity.json')
    assert parity == receipt['initial_logits_parity'] and parity['status'] == 'PASS'
    assert parity['unique_frames'] == 16 and parity['profiles'] == 3 and parity['two_initial_arms_within_tolerance']
    assert parity['atol'] == 2e-5 and parity['rtol'] == 1e-6
    learned = read(source / 'learned-all-missing.json')
    assert learned['status'] == 'PASS' and learned['arms'] == receipt['learned_all_missing']
    for value in (parity['initial_all_missing'], learned['arms']):
        assert set(value) == set(ARMS)
        assert all(v['frames'] == 16 and v['zero_values'] == 128 and v['all_missing_anchor_exactly_zero'] for v in value.values())
    p = arrays(source / 'predictions.npz')
    grouping = read(source / 'groups.json')
    records = grouping['records']
    groups = {k: np.asarray(v, dtype=np.int64) for k, v in grouping['groups'].items()}
    assert {k: len(v) for k, v in groups.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(2560))
    np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
    assert len({r['frame_id'] for r in records}) == 2560
    owners, pair_rows = {}, {}
    for group, ids in groups.items():
        for i in ids:
            assert owners.setdefault(records[i]['pair_id'], group) == group
            pair_rows.setdefault(records[i]['pair_id'], []).append(int(i))
    pairs = np.array(list(pair_rows.values()))
    assert pairs.shape == (1280, 2)
    np.testing.assert_array_equal(p['mz48/truth'][pairs[:, 0]], p['mz48/truth'][pairs[:, 1]])

    prior_root = task.parent / 'mz54-full-rgb-20260911'
    prior_run = read(bound('mz54-full-rgb-20260911/run-v1/receipt.json'))
    prior_seal = read(bound('mz54-full-rgb-20260911/score-v1/receipt.json'))
    assert prior_run['status'] == prior_seal['status'] == 'PASS'
    prior_score = read(bind(prior_root / 'score-v1/result.json', prior_seal['outputs']['result.json']))
    prior_audit = read(bind(prior_root / 'score-v1/audit.json', prior_seal['outputs']['audit.json']))
    assert prior_audit['prior_MZ37_positive_retention'] and prior_audit['exact_MZ51_schedule_arrays'] == 6
    assert sha(prior_root / 'run-v1/receipt.json') == handoff['source_receipt_sha256']
    previous = arrays(bound('mz54-full-rgb-20260911/run-v1/predictions.npz'))
    for key, value in previous.items():
        np.testing.assert_array_equal(p[key], value, err_msg=key)
    preserved = len(previous)
    assert preserved == receipt['preserved_MZ54_arrays']
    del previous
    assert grouping == read(bind(prior_root / 'run-v1/groups.json', prior_run['outputs']['groups.json']))
    schedule = arrays(source / 'schedule.npz')
    prior_schedule = arrays(bind(prior_root / 'run-v1/schedule.npz', prior_run['outputs']['schedule.npz']))
    assert set(schedule) == set(prior_schedule)
    for key in schedule:
        np.testing.assert_array_equal(schedule[key], prior_schedule[key])
    assert schedule['shared'].shape == schedule['OLD_NEG'].shape == schedule['query'].shape == (600, 8)
    assert np.isin(schedule['shared'], groups['fit']).all() and np.isin(schedule['OLD_NEG'], schedule['train_ids']).all()
    assert len(schedule['train_ids']) == 7562
    np.testing.assert_array_equal(schedule['fit_ids'], groups['fit'])
    np.testing.assert_array_equal(schedule['query'], np.tile(np.repeat(np.arange(4), 2), (600, 1)))
    cache_index = arrays(source / 'feature-cache-index.npz')
    old_index = arrays(bind(handoff['source_index_path'], handoff['source_index_sha256']))
    assert set(cache_index) == set(old_index)
    for key in cache_index:
        np.testing.assert_array_equal(cache_index[key], old_index[key])
    np.testing.assert_array_equal(cache_index['new_ids'], np.unique(schedule['shared']))
    np.testing.assert_array_equal(cache_index['old_ids'], np.unique(schedule['OLD_NEG']))
    np.testing.assert_array_equal(parity['train_ids'], cache_index['old_ids'][:16])
    assert (len(cache_index['new_ids']), len(cache_index['old_ids'])) == (1250, 3533)
    r51 = read(bound('mz51-training-coverage-20260911/score-v1/receipt.json'))
    audit51 = read(bind(task.parent / 'mz51-training-coverage-20260911/score-v1/audit.json', r51['outputs']['audit.json']))
    assert audit51['schedule']['schedule_exact'] and audit51['schedule']['old_train_disjoint_from_three_DEV_cohorts']
    assert audit51['schedule']['arms']['OLD_NEG']['negative_query_truth_verified']
    r53 = read(bound('mz53-dual-readout-union-20260911/score-v1/receipt.json'))
    score53 = read(bind(task.parent / 'mz53-dual-readout-union-20260911/score-v1/result.json', r53['outputs']['result.json']))
    groups['nonfit'] = np.sort(np.concatenate([groups['heldout_site'], groups['nonfit_family']]))
    crop, coverage, geometry = static_geometry(p)
    row, col = np.indices((8, 8), dtype=np.float32)
    angles = np.stack([-22.5 + (col + .5) * 5.625, 22.5 - (row + .5) * 5.625], -1).reshape(64, 2) / 22.5
    np.testing.assert_array_equal(p['mz56/zone_angles'], angles)
    for name in ('receipt.json', 'source-index.json'):
        bound('mz52-full-frame-supervision-20260911/' + name)
    r52 = read(bound('mz52-full-frame-supervision-20260911/verified-v1/receipt.json'))
    label_path = bound('mz52-full-frame-supervision-20260911/verified-v1/fullframe-cells.npz')
    assert sha(label_path) == r52['outputs']['fullframe-cells.npz'] and r52['status'] == 'PASS'
    labels = arrays(label_path)
    np.testing.assert_array_equal(labels['frame_ids'], p['mz48/frame_ids'])
    counts, valid_counts = labels['fullframe_event_counts'], labels['valid_counts']
    assert counts.shape == (2560, 45, 80, 4) and valid_counts.shape == (2560, 45, 80)
    assert counts.dtype == valid_counts.dtype == np.uint8 and (counts <= valid_counts[..., None]).all()
    np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, p['mz48/truth'])
    native, local_known = (counts > 0).reshape(2560, 3600, 4), (valid_counts > 0).reshape(2560, 3600)
    del labels, counts, valid_counts
    cuts = {}
    for arm in ARMS:
        def collect(key):
            return np.concatenate([p['DEV/DROP_CLOSE/' + key], p['mz48/DROP_CLOSE/' + key][groups['calibration']]])
        truth = np.concatenate([p['DEV/truth'], p['mz48/truth'][groups['calibration']]])
        known = np.concatenate([p['DEV/known'], p['mz48/known'][groups['calibration']]])
        assert truth.shape == (1256, 4) and known.all()
        key = 'MZ56/' + arm + '/'
        saved = np.load(source / (arm + '-cutoff.npy'), allow_pickle=False)
        np.testing.assert_array_equal(saved, cutoff_zero_added(collect(key + 'raw'), collect(key + 'support'), collect('MZ37'), truth))
        np.testing.assert_array_equal(saved, receipt['fits'][arm]['cutoff'])
        assert not ((collect(key + 'candidate') >= 0) & (collect('MZ37') < 0) & ~truth).any()
        cuts[arm] = saved
    cuts['GLOBAL_SUPPRESSED'] = cuts['GLOBAL_ANCHOR']
    references = tuple(prior_score['methods'])
    methods = references + tuple(k + '/candidate' for k in KEYS)
    scores, comparisons, attribution, cases, contexts, cached = {}, {}, {}, {}, {}, {}
    scalar_bits = reference_rows = native_values = anchor_values = 0
    missing_raw_max_abs, missing_raw_rows = 0., 0
    def compare(a, truth, known):
        result = {key: {ref: paired(a[key + '/candidate'], a[ref], truth, known)
                       for ref in methods if ref != key + '/candidate'} for key in KEYS}
        for key in KEYS:
            assert not any(result[key]['MZ37']['tp_lost']) and not any(result[key]['MZ37']['fp_removed'])
        return result
    for cohort in COHORTS:
        truth, known = p[cohort + '/truth'], p[cohort + '/known']
        if cohort == 'mz36':
            truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
            admitted = p['mz36/admitted_index']
            assert truth.shape == known.shape == (400, 4) and int((~known).sum()) == 80
            assert len(admitted) == len(np.unique(admitted)) == 380
            np.testing.assert_array_equal(truth[admitted], p['mz36/truth'])
        cohort_groups = {'all': np.arange(len(truth))}
        if cohort == 'mz48':
            cohort_groups.update(groups)
            for field in ('family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                for value in sorted({str(r[field]) for r in records}):
                    cohort_groups[field + '/' + value] = np.array([i for i, r in enumerate(records) if str(r[field]) == value])
        scores[cohort], comparisons[cohort] = {}, {}
        for profile in PROFILES:
            prefix = cohort + '/' + profile + '/'
            a = {name: p[prefix + name] for name in references if name not in UNIONS}
            for family in ('MZ50', 'NEW_NEG', 'OLD_NEG'):
                lead = '' if family == 'MZ50' else family + '/'
                a[family + '/UNION'] = np.maximum(a[lead + 'OPEN/candidate'], a[lead + 'GATED/candidate'])
            for mode, key in zip(MODES, KEYS):
                raw, support, winner, candidate = (p[prefix + key + '/' + name] for name in ('raw', 'support', 'winner', 'candidate'))
                assert raw.shape == support.shape == winner.shape == candidate.shape == p[cohort + '/truth'].shape
                assert support.dtype == np.bool_ and support.all() and np.isfinite(raw).all()
                assert winner.dtype.kind in 'iu' and winner.min() >= 0 and winner.max() < 3600
                margin = raw.astype(float) - cuts[mode]
                np.testing.assert_array_equal(candidate, np.where((a['MZ37'] < 0) & support & (margin >= 0), margin, a['MZ37']))
                a[key + '/candidate'] = candidate
                available, vector = p[prefix + key + '/anchor_available'], p[prefix + key + '/anchor_vector']
                assert available.shape == (len(raw),) and available.dtype == np.bool_
                assert vector.shape == (len(raw), 8) and np.isfinite(vector).all()
                assert np.count_nonzero(vector[~available]) == 0
                if mode != 'GLOBAL_ANCHOR':
                    assert np.count_nonzero(vector) == 0
                np.testing.assert_array_equal(available, p[prefix + 'MZ56/GLOBAL_ANCHOR/anchor_available'])
                anchor_values += vector.size
            unavailable = ~p[prefix + 'MZ56/GLOBAL_ANCHOR/anchor_available']
            missing_global = p[prefix + 'MZ56/GLOBAL_ANCHOR/raw'][unavailable]
            missing_suppressed = p[prefix + 'MZ56/GLOBAL_SUPPRESSED/raw'][unavailable]
            missing_raw_rows += len(missing_global)
            np.testing.assert_allclose(missing_global, missing_suppressed, atol=2e-5, rtol=1e-6)
            if missing_global.size:
                missing_raw_max_abs = max(missing_raw_max_abs, float(np.abs(missing_global-missing_suppressed).max()))
            if cohort == 'mz36':
                expanded = {}
                for key, value in a.items():
                    expanded[key] = np.full((400, 4), np.nan)
                    expanded[key][admitted] = value
                a = expanded
            scores[cohort][profile], comparisons[cohort][profile] = {}, {}
            for group, ids in cohort_groups.items():
                aa = {k: v[ids] for k, v in a.items()}
                rows = {k: metrics(v, truth[ids], known[ids]) for k, v in aa.items()}
                for key in references:
                    ref = score53 if key in UNIONS else prior_score
                    assert rows[key] == ref['conditions'][cohort][profile][group][key], (cohort, profile, group, key)
                    reference_rows += 1
                scores[cohort][profile][group] = rows
                comparisons[cohort][profile][group] = compare(aa, truth[ids], known[ids])
            for key in KEYS:
                scalar_bits += scalar_check(a[key + '/candidate'], truth, known, scores[cohort][profile]['all'][key + '/candidate'])
            cached.setdefault(profile, {})[cohort] = (a, truth, known)
            if cohort == 'mz48':
                winning, winning_known, winners = {}, {}, {}
                for key in KEYS:
                    winner = p[prefix + key + '/winner']
                    expected = np.take_along_axis(native, winner[:, None, :], 1)[:, 0] & p[prefix + key + '/support']
                    np.testing.assert_array_equal(expected, p[prefix + key + '/winning_native'])
                    native_values += expected.size
                    winning[key], winners[key] = expected, winner
                    winning_known[key] = np.take_along_axis(local_known, winner, 1)
                attribution[profile] = {}
                for group, ids in groups.items():
                    attribution[profile][group] = {}
                    for key in KEYS:
                        winner = winners[key]
                        final_tp = (a[key + '/candidate'] >= 0) & truth & known
                        added = final_tp & (a['MZ37'] < 0)
                        masks = dict(added_tp=added, native_winner_tp=added & winning[key],
                            native_outside_sensor_tp=added & winning[key] & ~coverage[winner],
                            known_winner_tp=added & winning_known[key], outside_sensor_tp=added & ~coverage[winner],
                            native_below_original_crop_tp=added & winning[key] & (winner // 80 >= 37),
                            final_tp=final_tp, final_native_winner_tp=final_tp & winning[key],
                            final_native_outside_sensor_tp=final_tp & winning[key] & ~coverage[winner])
                        attribution[profile][group][key] = {name: mask[ids].sum(0).tolist() for name, mask in masks.items()}
                    global_key = 'MZ56/GLOBAL_ANCHOR'
                    for ref in ('MZ56/LOCAL_ONLY/candidate', 'MZ56/GLOBAL_SUPPRESSED/candidate', 'MZ54/FULL_RASTER/candidate', 'OLD_NEG/UNION'):
                        recovered = (a[global_key + '/candidate'] >= 0) & (a[ref] < 0) & truth & known
                        masks = dict(tp_gained=recovered, winning_native=recovered & winning[global_key],
                            native_outside_sensor=recovered & winning[global_key] & ~coverage[winners[global_key]])
                        attribution[profile][group]['GLOBAL_vs_' + ref] = {name: mask[ids].sum(0).tolist() for name, mask in masks.items()}
                contexts[profile] = {name: ((v[pairs[:, 0]] >= 0) != (v[pairs[:, 1]] >= 0)).sum(0).tolist() for name, v in a.items()}
                ids = groups['nonfit']
                changed = ((a['MZ56/GLOBAL_ANCHOR/candidate'] >= 0) != (a['MZ56/LOCAL_ONLY/candidate'] >= 0)) & known
                cases[profile] = []
                for j, q in zip(*np.where(changed[ids])):
                    i = int(ids[j])
                    case = dict(index=i, frame_id=records[i]['frame_id'], query=QUERIES[q], truth=bool(truth[i, q]),
                        MZ37_positive=bool(a['MZ37'][i, q] >= 0),
                        **{k: records[i][k] for k in ('family', 'site_id', 'range', 'support_context', 'pair_id')})
                    for key in KEYS:
                        winner = int(winners[key][i, q])
                        case[key] = dict(positive=bool(a[key + '/candidate'][i, q] >= 0), winner=winner,
                            native=bool(winning[key][i, q]), known=bool(winning_known[key][i, q]),
                            sensor_covered=bool(coverage[winner]), raw=float(p[prefix + key + '/raw'][i, q]))
                    cases[profile].append(case)
    aggregates, aggregate_comparisons = {}, {}
    for profile in PROFILES:
        data = cached[profile]
        legacy = [(data[name], np.arange(len(data[name][1]))) for name in ('relation10000', 'distance5000', 'rich', 'mz36')]
        selections = dict(legacy_noncal=legacy, mz48_fit=[(data['mz48'], groups['fit'])], mz48_nonfit=[(data['mz48'], groups['nonfit'])])
        selections['all_noncal'] = legacy + selections['mz48_fit'] + selections['mz48_nonfit']
        aggregates[profile], aggregate_comparisons[profile] = {}, {}
        for group, selected in selections.items():
            truth = np.concatenate([v[ids] for (_, v, _), ids in selected])
            known = np.concatenate([v[ids] for (_, _, v), ids in selected])
            a = {key: np.concatenate([values[key][ids] for (values, _, _), ids in selected]) for key in methods}
            rows = {key: metrics(v, truth, known) for key, v in a.items()}
            for key in references:
                ref = score53 if key in UNIONS else prior_score
                assert rows[key] == ref['aggregates'][profile][group][key]
                reference_rows += 1
            aggregates[profile][group] = rows
            aggregate_comparisons[profile][group] = compare(a, truth, known)
    drop, attr = aggregates['DROP_CLOSE'], attribution['DROP_CLOSE']['nonfit']
    local, glob = (drop['mz48_nonfit']['MZ56/' + arm + '/candidate'] for arm in ARMS)
    legacy_local, legacy_global = (drop['legacy_noncal']['MZ56/' + arm + '/candidate'] for arm in ARMS)
    local_new, global_new = (attr['MZ56/' + arm]['native_outside_sensor_tp'][0] for arm in ARMS)
    gate = dict(local_added_native_outside_body_near_tp=local_new, global_added_native_outside_body_near_tp=global_new,
        local_final_native_outside_body_near_tp=attr['MZ56/LOCAL_ONLY']['final_native_outside_sensor_tp'][0],
        global_final_native_outside_body_near_tp=attr['MZ56/GLOBAL_ANCHOR']['final_native_outside_sensor_tp'][0],
        local_body_near_tp=local['tp'][0], global_body_near_tp=glob['tp'][0],
        local_nonfit_fp=sum(local['fp']), global_nonfit_fp=sum(glob['fp']),
        local_legacy_fp=sum(legacy_local['fp']), global_legacy_fp=sum(legacy_global['fp']),
        added_native_outside_body_near_improved=global_new > local_new,
        nonfit_fp_not_higher=sum(glob['fp']) <= sum(local['fp']),
        legacy_fp_not_higher=sum(legacy_global['fp']) <= sum(legacy_local['fp']))
    gate['passes_primary'] = all(gate[k] for k in ('added_native_outside_body_near_improved', 'nonfit_fp_not_higher', 'legacy_fp_not_higher'))
    result = dict(status='PASS', source_run=run_name, primary_condition='DROP_CLOSE', primary_operational_definition=PRIMARY_DEFINITION,
        primary_gate=gate, event_order=QUERIES, methods=methods, conditions=scores, comparisons=comparisons,
        aggregates=aggregates, aggregate_comparisons=aggregate_comparisons, attribution=attribution,
        nonfit_global_local_disagreements=cases, context_pair_changed_decisions=contexts,
        existing_cutoffs={arm: cuts[arm].tolist() for arm in ARMS},
        GLOBAL_SUPPRESSED_cutoff_source='GLOBAL_ANCHOR-cutoff.npy; same learned GLOBAL weights, zero global vector',
        geometry=geometry, aggregation_scope=prior_score['aggregation_scope'],
        limitation='Matched continuations on consumed Development. Global context is observed in-field evidence, not outside-object ranging. Failure rejects this fixed design, not all metric context. No hardware, natural-scene, clearance or default-App claim.')
    audit = dict(status='PASS', preserved_MZ54_arrays=preserved, sealed_reference_count_rows_equal=reference_rows,
        scalar_known_bits=scalar_bits, scalar_new_readouts=3, profiles=3, mz36_attempts=400, mz36_unknown_bits=80,
        source_frames=2560, pairs_intact=1280, exact_prior_schedule_arrays=len(schedule),
        shared_unique_fit_frames=1250, old_negative_unique_train_frames=3533, schedule_membership=True,
        negative_query_truth_authority='Sealed MZ51 independent audit plus exact schedule/value preservation through MZ54',
        initial_logits_parity_receipt_verified=True, all_missing_initial_and_learned_zero_receipts_verified=True,
        saved_anchor_values_checked=anchor_values, suppressed_and_local_vectors_exactly_zero=True,
        missing_global_and_suppressed_raw_within_preexisting_tolerance=True,
        missing_global_suppressed_raw_max_abs=missing_raw_max_abs, parity_atol=2e-5, parity_rtol=1e-6,
        missing_global_suppressed_raw_rows=missing_raw_rows,
        missing_global_suppressed_raw_exactly_equal=bool(missing_raw_max_abs == 0) if missing_raw_rows else None,
        matched_steps=[600, 600], trainable_parameters=11020,
        cutoffs_recomputed_for_parity=2, new_cutoffs=0, GLOBAL_SUPPRESSED_reuses_GLOBAL_cutoff=True,
        prior_MZ37_positive_retention=True, winning_native_independently_checked_values=native_values,
        native_winner_authority='Existing compact MZ52 counts indexed independently, no new native-depth access',
        training_cache_producer_hash_and_identity_receipt_verified=True, full_feature_cache_reads=0,
        all_noncal_includes_fit1280_once=True, new_fits=0, new_inference_frames=0,
        native_depth_reads=0, dense_feature_reads=0, RGB_reads=0, checkpoint_loads=0)
    for name in ('mz56_score.py', 'mz54_score.py', 'mz15_train.py', 'mz40_evaluate.py', 'mz47_score.py'):
        bind(Path(__file__).with_name(name))
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest, path
    out.mkdir(exist_ok=False)
    write(out / 'result.json', result)
    write(out / 'audit.json', audit)
    report(report_path, task, result, audit)
    final = dict(status='PASS', inputs=inputs, code_sha256=sha(Path(__file__)),
        outputs={name: sha(out / name) for name in ('result.json', 'audit.json')},
        report=dict(path=str(report_path), sha256=sha(report_path)), seconds=time.perf_counter()-started,
        backend='CPU saved outputs and existing compact MZ52 labels; TASK_NOT_GPU_SUITABLE',
        new_fits=0, new_inference_frames=0, new_cutoffs=0, native_depth_reads=0,
        scientific_binding='FINAL; report, result, audit and code sealed before this receipt')
    write(out / 'receipt.json', final)
    print('PRIMARY_GATE', gate)
    print('DROP_TOTALS', {group: {key: [sum(row[k]) for k in ('tp', 'fp', 'fn')] for key, row in values.items()
        if key.startswith('MZ56/') or key in ('MZ54/FULL_RASTER/candidate', 'OLD_NEG/UNION')} for group, values in drop.items()})
    print('SEALED', dict(seconds=final['seconds'], scalar_known_bits=scalar_bits, native_values=native_values,
                         code_sha256=final['code_sha256'], report_sha256=final['report']['sha256'], outputs=final['outputs']))


def report(path, task, result, audit):
    gate = result['primary_gate']
    names = tuple(k + '/candidate' for k in KEYS) + ('MZ54/FULL_RASTER/candidate', 'OLD_NEG/UNION')
    lines = ['# MZ56: global measured-context continuation', '',
        f'Primary DROP_CLOSE gate: **{"PASS" if gate["passes_primary"] else "FAIL"}**. '
        f'Added nonfit BODY_NEAR TP with a native winner outside 45 degrees: GLOBAL {gate["global_added_native_outside_body_near_tp"]}, '
        f'LOCAL {gate["local_added_native_outside_body_near_tp"]}. '
        f'Nonfit FP GLOBAL/LOCAL: {gate["global_nonfit_fp"]}/{gate["local_nonfit_fp"]}; '
        f'old noncal FP: {gate["global_legacy_fp"]}/{gate["local_legacy_fp"]}.', '',
        PRIMARY_DEFINITION, '',
        '| DROP_CLOSE group | LOCAL TP/FP/FN | GLOBAL TP/FP/FN | SUPPRESSED TP/FP/FN | MZ54 FULL TP/FP/FN | fixed OLD_NEG union TP/FP/FN |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group, values in result['aggregates']['DROP_CLOSE'].items():
        lines.append('| ' + group + ' | ' + ' | '.join('/'.join(str(sum(values[name][k])) for k in ('tp', 'fp', 'fn')) for name in names) + ' |')
    lines += ['', 'Old noncalibration excludes DEV. Nonfit is 640 heldout-site + 384 withheld-family frames; all_noncal includes fit 1280 once and is not wholly held-out evidence.', '',
        '| nonfit DROP query | LOCAL TP/FP/FN | GLOBAL TP/FP/FN | SUPPRESSED TP/FP/FN | MZ54 FULL TP/FP/FN | OLD_NEG union TP/FP/FN |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    values = result['aggregates']['DROP_CLOSE']['mz48_nonfit']
    for q, query in enumerate(QUERIES):
        lines.append('| ' + query + ' | ' + ' | '.join('/'.join(str(values[name][k][q]) for k in ('tp', 'fp', 'fn')) for name in names) + ' |')
    lines += ['', '| profile nonfit | LOCAL TP/FP | GLOBAL TP/FP | SUPPRESSED TP/FP | MZ54 FULL TP/FP | OLD_NEG union TP/FP |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for profile in PROFILES:
        values = result['aggregates'][profile]['mz48_nonfit']
        lines.append('| ' + profile + ' | ' + ' | '.join('/'.join(str(sum(values[name][k])) for k in ('tp', 'fp')) for name in names) + ' |')
    lines += ['', '| DROP GLOBAL versus | group | TP gained/lost | FP added/removed |', '| --- | --- | ---: | ---: |']
    for group in ('legacy_noncal', 'mz48_nonfit', 'mz48_fit'):
        for ref in (names[0], names[2], names[3], names[4]):
            row = result['aggregate_comparisons']['DROP_CLOSE'][group]['MZ56/GLOBAL_ANCHOR'][ref]
            lines.append(f'| {ref} | {group} | {sum(row["tp_gained"])}/{sum(row["tp_lost"])} | {sum(row["fp_added"])}/{sum(row["fp_removed"])} |')
    lines += ['', '| nonfit BODY_NEAR attribution | added TP over MZ37 | added native winner | added native outside 45 | all final native outside 45 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key in KEYS:
        value = result['attribution']['DROP_CLOSE']['nonfit'][key]
        lines.append('| ' + key + ' | ' + ' | '.join(str(value[k][0]) for k in ('added_tp', 'native_winner_tp', 'native_outside_sensor_tp', 'final_native_outside_sensor_tp')) + ' |')
    lines += ['', 'GLOBAL_SUPPRESSED uses the same learned GLOBAL weights and cutoff with its global vector zeroed. '
        'Its changes isolate inference use of that path; GLOBAL versus LOCAL also includes different learned responses. '
        'Global context is an average of actual valid in-field zone tokens, not a measured range for an outside object. '
        'The native label and sensor coverage are evaluator-only attribution. Other query/profile regressions remain costs.', '',
        f'Validation: {audit["preserved_MZ54_arrays"]} MZ54 arrays preserved exactly; '
        f'{audit["sealed_reference_count_rows_equal"]} previous individual/union count rows agree; '
        f'{audit["scalar_known_bits"]:,} scalar known-bit checks and {audit["winning_native_independently_checked_values"]:,} native-winning lookups. '
        'Two existing cutoffs recomputed only for parity. The exact 600-step schedules, 1280 pairs, '
        '2560-frame partition, all MZ37 positives, and MZ36 400 attempts/80 UNKNOWN remain. '
        'The run separately records 16 actual TRAIN-feature initialization checks against completed MZ54, '
        'initial and learned all-missing zeros, read-only cache identity, and 5734 uncached full RGB extractions.', '',
        f'Scored source: {result["source_run"]}. The earlier run-v1 stopped before fit or feature extraction because an extra '
        'GPU bitwise-equality check was stricter than the registered numerical tolerance. Its directory, traceback, '
        'original runner bytes and handle-release receipt remain preserved in preflight-failure-v1. '
        'The actual run uses the same predeclared atol 2e-5 / rtol 1e-6 and retains exact initial parameter equality.', '',
        'This is one fixed controlled Development continuation on consumed sources. '
        'It cannot establish hardware calibration, natural-scene generalization, clearance or default-App suitability. '
        'A failure rejects this design and budget, not every possible use of metric context.', '',
        f'Evidence: `{task}/score-v1/result.json`, `audit.json`, and final `receipt.json`. '
        'The receipt seals this report, scorer, brief, run outputs, and source/prior-score bindings.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--run-name', choices=('run-v1', 'run-v2'), default='run-v1')
    args = parser.parse_args()
    run(args.task, args.run_name)
