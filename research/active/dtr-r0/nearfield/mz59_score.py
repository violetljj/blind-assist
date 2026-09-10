"""MZ59 independent saved-output scoring; no model/native/RGB inference.

--self-test accesses synthetic arrays only. --task requires a sealed run and
root score GO; original cohorts and all MZ55 roles remain Development data.
"""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np

from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired

ARMS = ('CONTROL', 'DIVERSE')
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
BASE = 'OLD_NEG/UNION'
KEYS = tuple('MZ59/' + arm for arm in ARMS)
NEW = tuple(key + '/candidate' for key in KEYS) + tuple(key + '/UNION' for key in KEYS)
GATE_DEFINITION = ('DIVERSE versus CONTROL: more native-winning BODY_NEAR TP added over OLD_NEG/UNION '
                   'on MZ55 heldout new-family480 DROP frames; no higher final OLD_NEG/UNION OR candidate '
                   'FP on all MZ55 heldout640 or original legacy noncalibration. Candidate FP is reported separately.')
FROZEN_DEFINITION = ('Native BODY_NEAR TP newly recovered over OLD_NEG/UNION on480 MZ55 heldout new-family frames: DIVERSE > CONTROL. '
                     'TotalFP compared using final OLD_NEG/UNION OR eachcandidate on all640 MZ55 heldout and original legacy noncalibration; '
                     'DIVERSE <= CONTROL in both. Report candidate costs separately; do not offset OR false alerts with candidate-only gains.')


def canonical_metric(row):
    """Rename the legacy denominator only; never coerce or drop other counts."""
    result = dict(row)
    assert 'frames' in result or 'attempted_frames' in result, 'Missing frame denominator'
    if 'attempted_frames' in result:
        if 'frames' in result:
            assert result['frames'] == result['attempted_frames'], 'Conflicting frame denominators'
        result['frames'] = result.pop('attempted_frames')
    return result


def assert_metric_equal(actual, previous):
    assert canonical_metric(actual) == canonical_metric(previous), 'Metric count/denominator mismatch'


def scalar_cutoff(raw, support, baseline, truth, known):
    """Exact original rule, independently enumerating eligible negative rows."""
    assert raw.shape == support.shape == baseline.shape == truth.shape == known.shape == (1256, 4)
    assert known.all(), 'Original calibration was fully known; never reinterpret UNKNOWN'
    result, authority = [], []
    for q in range(4):
        eligible, negative = [], []
        for i in range(1256):
            if bool(support[i, q]) and float(baseline[i, q]) < 0:
                eligible.append(i)
                if not bool(truth[i, q]):
                    negative.append(i)
        if negative:
            maximum = max(float(raw[i, q]) for i in negative)
            value = np.nextafter(np.float64(maximum), np.inf)
            winners = [i for i in negative if float(raw[i, q]) == maximum]
        else:
            value, winners = (-np.inf if eligible else np.inf), []
        result.append(value)
        authority.append(dict(eligible=len(eligible), negative_rows=len(negative), maximum_rows=winners))
    return np.array(result), authority


def validate_schedules(schedule, original, mz55_train):
    for key in ('shared', 'OLD_NEG', 'query'):
        np.testing.assert_array_equal(schedule[key], original[key])
        assert schedule[key].shape == (600, 8)
    expected = np.random.default_rng(159).choice(mz55_train, (600, 4), replace=True)
    np.testing.assert_array_equal(schedule['mz55_diverse'], expected)
    np.testing.assert_array_equal(schedule['query'], np.tile(np.repeat(np.arange(4), 2), (600, 1)))
    assert len(np.unique(mz55_train)) == 1600
    return dict(shared_rows_exact=600, original_negative_rows_exact=600, query_rows_exact=600,
                mz55_diverse_draws_exact=2400, mz55_unique_sampled=int(len(np.unique(expected))),
                control_native_presentations=4800, diverse_mz48_presentations=2400,
                diverse_mz55_presentations=2400, old_negative_presentations_per_arm=4800)


def validate_candidate(raw, support, winner, vector, available, candidate, baseline, cutoff):
    n = len(baseline)
    assert raw.shape == support.shape == winner.shape == candidate.shape == baseline.shape == (n, 4)
    assert support.dtype == np.bool_ and support.all() and np.isfinite(raw).all()
    assert np.issubdtype(winner.dtype, np.integer) and ((winner >= 0) & (winner < 3600)).all()
    assert vector.shape == (n, 8) and np.isfinite(vector).all() and (vector >= 0).all()
    assert available.shape == (n,) and available.dtype == np.bool_ and (vector[~available] == 0).all()
    expected = np.empty((n, 4), dtype=float)
    for i in range(n):
        for q in range(4):
            margin = float(raw[i, q]) - float(cutoff[q])
            expected[i, q] = margin if baseline[i, q] < 0 and support[i, q] and margin >= 0 else baseline[i, q]
    np.testing.assert_array_equal(candidate, expected)
    assert ((candidate >= 0) | (baseline < 0)).all()


def reconstruct(p, prefix, reference_methods):
    a = {name: p[prefix + name] for name in reference_methods if not name.endswith('/UNION')}
    for family in ('MZ50', 'NEW_NEG', 'OLD_NEG'):
        key = family + '/UNION'
        if key in reference_methods:
            lead = '' if family == 'MZ50' else family + '/'
            a[key] = np.maximum(a[lead + 'OPEN/candidate'], a[lead + 'GATED/candidate'])
    for mode in ('LOCAL_ONLY', 'GLOBAL_ANCHOR', 'GLOBAL_SUPPRESSED'):
        key = 'MZ57/' + mode + '/UNION'
        if key in reference_methods:
            a[key] = np.maximum(a[BASE], a['MZ56/' + mode + '/candidate'])
            if prefix + key in p:
                np.testing.assert_array_equal(a[key], p[prefix + key])
    for key in KEYS:
        a[key + '/candidate'] = p[prefix + key + '/candidate']
        a[key + '/UNION'] = np.maximum(a[BASE], a[key + '/candidate'])
        np.testing.assert_array_equal(a[key + '/UNION'], p[prefix + key + '/UNION'])
        for i in range(len(a[BASE])):
            for q in range(4):
                assert bool(a[key + '/UNION'][i, q] >= 0) == (bool(a[BASE][i, q] >= 0) or bool(a[key + '/candidate'][i, q] >= 0))
    return a


def compare(a, truth, known):
    refs = (BASE, 'MZ37', 'MZ56/GLOBAL_ANCHOR/candidate', 'MZ57/GLOBAL_ANCHOR/UNION') + NEW
    return {key: {ref: paired(a[key], a[ref], truth, known) for ref in refs if ref != key} for key in NEW}


def native_masks(p, prefix, key, counts, valid_counts):
    n = len(counts)
    w = p[prefix + key + '/winner'].astype(np.int64)
    native = (counts > 0).reshape(n, 3600, 4)
    local_known = (valid_counts > 0).reshape(n, 3600)
    hit = native[np.arange(n)[:, None], w, np.arange(4)[None]]
    known = local_known[np.arange(n)[:, None], w]
    for i in range(n):
        for q in range(4):
            assert hit[i, q] == native[i, int(w[i, q]), q]
    assert not (hit & ~known).any()
    return w, hit, known


def primary_gate(native_added, heldout, legacy):
    c, d = (key + '/UNION' for key in KEYS)
    control, diverse = (native_added[key][0] for key in KEYS)
    return dict(definition=GATE_DEFINITION, control_native_added_body_near=int(control), diverse_native_added_body_near=int(diverse),
                control_heldout_fp=sum(heldout[c]['fp']), diverse_heldout_fp=sum(heldout[d]['fp']),
                control_legacy_fp=sum(legacy[c]['fp']), diverse_legacy_fp=sum(legacy[d]['fp']),
                passes_primary=bool(diverse > control and sum(heldout[d]['fp']) <= sum(heldout[c]['fp'])
                                    and sum(legacy[d]['fp']) <= sum(legacy[c]['fp'])))


def run(task):
    task = task.resolve(strict=True)
    out, run_dir, work = task / 'score-v1', task / 'run-v1', task.parent
    assert not out.exists(), 'Never overwrite scoring or failure evidence'
    out.mkdir()
    started, inputs = time.perf_counter(), {}
    try:
        def bind(path, expected=None):
            path = Path(path).resolve(strict=True)
            digest = sha(path)
            assert expected is None or digest == expected, f'Hash mismatch: {path}'
            inputs[str(path)] = digest
            return path

        bind(Path(__file__))
        bind(Path(__file__).with_name('mz58_score.py'))
        brief = Path(__file__).with_name('MZ59_TRAINING_DIVERSITY_20260911.md')
        rr = read(bind(run_dir / 'receipt.json'))
        assert rr['status'] == 'PASS'
        def inherited(path):
            path = Path(path).resolve(strict=True)
            matches = [h for p, h in rr['inputs'].items() if Path(p).resolve() == path]
            assert len(matches) == 1, f'Producer receipt does not bind {path}'
            return bind(path, matches[0])
        inherited(brief)
        # Receipt schema is finalized by the runner before fitting. Validate the
        # scientifically operative facts, not an incidental progress-file layout.
        assert rr['steps_per_arm'] == 600 and rr['total_steps'] == 1200
        assert rr['trainable_parameters'] == 11020 and rr['new_cutoffs'] == 2 and rr['threshold_searches'] == 0
        definition = read(inherited(task / 'primary-definition.json'))
        assert definition == rr['primary_definition']
        assert definition['status'] == 'FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        assert definition['definition'] == FROZEN_DEFINITION
        assert len(set(rr['initial_parameter_sha256'].values())) == 1 and set(rr['fits']) == set(ARMS)
        assert all(rr['fits'][arm]['steps'] == 600 for arm in ARMS)
        assert rr['exact_initial_weights'] and rr['original_baseline_cohort_inferences'] == 0
        assert rr['mz55_calibration_rows_used'] == rr['native_depth_reads'] == 0 and rr['source_unknown_preserved']
        assert rr['original_calibration_frames'] == 1256 and rr['initial_train_parity_unique_frames'] == 16
        for name in ('initialization-parity.json', 'initial-missing.json', 'learned-missing.json'):
            obj = read(bind(run_dir / name, rr['outputs'][name]))
            assert obj['status'] == 'PASS'
            if name == 'initialization-parity.json':
                assert obj['unique_train_frames'] == 16 and obj['atol'] == 2e-5 and obj['rtol'] == 1e-6
                assert len(obj['comparisons']) == 6
            else:
                assert obj['frames_per_arm'] == 8 and obj['all_missing_context_exactly_zero']
        for path, digest in rr['inputs'].items():
            if path.endswith('.py'):
                bind(path, digest)
        p = load(bind(run_dir / 'predictions.npz', rr['outputs']['predictions.npz']))
        old_run = work / 'mz56-global-anchor-20260911/run-v2'
        r56 = read(inherited(old_run / 'receipt.json'))
        inherited(old_run / 'GLOBAL_ANCHOR.pt')
        for arm in ARMS:
            bind(run_dir / (arm + '.pt'), rr['outputs'][arm + '.pt'])
        old = load(bind(old_run / 'predictions.npz', r56['outputs']['predictions.npz']))
        for key, value in old.items():
            np.testing.assert_array_equal(p[key], value, err_msg=key)
        old_preserved = len(old)
        del old
        transfer_run = work / 'mz58-diverse-transfer-20260911/run-v1'
        r58 = read(inherited(transfer_run / 'receipt.json'))
        transfer = load(bind(transfer_run / 'predictions.npz', r58['outputs']['predictions.npz']))
        for key, value in transfer.items():
            np.testing.assert_array_equal(p['mz55/' + key], value, err_msg=key)
        transfer_preserved = len(transfer)
        del transfer
        score57_dir = work / 'mz57-complementary-scale-union-20260911/score-v1'
        seal57 = read(inherited(score57_dir / 'receipt.json'))
        s57 = read(bind(score57_dir / 'result.json', seal57['outputs']['result.json']))
        score58_dir = work / 'mz58-diverse-transfer-20260911/score-v1'
        seal58 = read(inherited(score58_dir / 'receipt.json'))
        s58 = read(bind(score58_dir / 'result.json', seal58['outputs']['result.json']))
        grouping = read(bind(old_run / 'groups.json', r56['outputs']['groups.json']))
        actual_groups = read(bind(run_dir / 'groups.json', rr['outputs']['groups.json']))
        assert actual_groups['records'] == grouping['records'] and actual_groups['groups'] == grouping['groups']
        groups48 = {k: np.array(v, int) for k, v in grouping['groups'].items()}
        records48 = grouping['records']
        assert {k: len(v) for k, v in groups48.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
        np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records48])
        source55 = work / 'mz55-diverse-mesh-source-20260911'
        index55 = read(inherited(source55 / 'source-index.json'))
        assert index55['status'] == 'COMPLETE' and index55['frames'] == 2560
        def source_member(name):
            ref = index55['combined'][name]
            return bind(source55 / ref['path'], ref['sha256'])
        meta55, labels55, eval55 = read(source_member('metadata.json')), load(source_member('fullframe-cells.npz')), load(source_member('evaluator.npz'))
        records55 = meta55['records']
        assert actual_groups['mz55_records'] == records55
        for data in (labels55, eval55):
            np.testing.assert_array_equal(data['frame_ids'], p['mz55/frame_ids'])
        for name in ('truth', 'known'):
            np.testing.assert_array_equal(eval55[name], p['mz55/' + name])
        np.testing.assert_array_equal(p['mz55/frame_ids'], [r['frame_id'] for r in records55])
        groups55 = {role: np.array([i for i, r in enumerate(records55) if r['role'] == role], int)
                    for role in ('TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_SITE')}
        assert {k: len(v) for k, v in groups55.items()} == dict(TRAIN_CANDIDATE=1600, CALIBRATION=320, HELDOUT_SITE=640)
        split_alias = dict(train='TRAIN_CANDIDATE', fit='TRAIN_CANDIDATE', calibration='CALIBRATION', heldout_site='HELDOUT_SITE')
        for name, ii in actual_groups['mz55_groups'].items():
            role = split_alias.get(name, name)
            if role in groups55:
                np.testing.assert_array_equal(ii, groups55[role])
        groups55['heldout_new_family'] = np.array([i for i in groups55['HELDOUT_SITE'] if records55[i]['family'] != 'retained_rod'])
        assert len(groups55['heldout_new_family']) == 480
        np.testing.assert_array_equal(actual_groups['mz55_groups']['heldout_new_forms'], groups55['heldout_new_family'])
        np.testing.assert_array_equal(actual_groups['mz55_groups']['new_forms'], [i for i, r in enumerate(records55) if r['family'] != 'retained_rod'])
        groups55['nonfit'] = np.sort(np.r_[groups55['CALIBRATION'], groups55['HELDOUT_SITE']])
        groups48['nonfit'] = np.sort(np.r_[groups48['heldout_site'], groups48['nonfit_family']])
        schedule = load(bind(run_dir / 'schedule.npz', rr['outputs']['schedule.npz']))
        old_schedule = load(bind(old_run / 'schedule.npz', r56['outputs']['schedule.npz']))
        schedule_audit = validate_schedules(schedule, old_schedule, groups55['TRAIN_CANDIDATE'])
        for key in old_schedule:
            np.testing.assert_array_equal(schedule[key], old_schedule[key])
        init_parity = read(run_dir / 'initialization-parity.json')
        np.testing.assert_array_equal(init_parity['ids'], np.unique(schedule['shared'])[:16])
        source52 = work / 'mz52-full-frame-supervision-20260911'
        index52 = read(inherited(source52 / 'source-index.json'))
        ref52 = index52['array']
        labels48 = load(bind(source52 / ref52['path'], ref52['sha256']))
        np.testing.assert_array_equal(labels48['frame_ids'], p['mz48/frame_ids'])
        labels = dict(mz48=labels48, mz55=labels55)
        for name in ('crop_mask', 'sensor_coverage', 'rays'):
            np.testing.assert_array_equal(p['mz59/' + name], p['mz54/' + name])
            np.testing.assert_array_equal(p['mz55/geometry/' + name], p['mz54/' + name])
        for cohort, lab in labels.items():
            assert lab['fullframe_event_counts'].shape == (2560, 45, 80, 4)
            assert lab['valid_counts'].shape == (2560, 45, 80)
            assert lab['fullframe_event_counts'].dtype == lab['valid_counts'].dtype == np.uint8
            assert (lab['fullframe_event_counts'] <= lab['valid_counts'][..., None]).all()
            np.testing.assert_array_equal(lab['fullframe_event_counts'].sum((1, 2)) >= 3, p[cohort + '/truth'])
        for cohort, records, pairs in (('mz48', records48, None), ('mz55', records55, meta55['pairs'])):
            by_pair = {}
            for i, row in enumerate(records):
                by_pair.setdefault(row['pair_id'], []).append(i)
            assert len(by_pair) == 1280 and all(len(ii) == 2 for ii in by_pair.values())
            if pairs is not None:
                assert pairs == by_pair
            for ii in by_pair.values():
                assert records[ii[0]]['role'] == records[ii[1]]['role']
                np.testing.assert_array_equal(p[cohort + '/truth'][ii[0]], p[cohort + '/truth'][ii[1]])
        cuts, authorities = {}, {}
        cal = groups48['calibration']
        def calibration(name):
            return np.concatenate([p['DEV/DROP_CLOSE/' + name], p['mz48/DROP_CLOSE/' + name][cal]])
        tcal = np.concatenate([p['DEV/truth'], p['mz48/truth'][cal]])
        kcal = np.concatenate([p['DEV/known'], p['mz48/known'][cal]])
        for arm, key in zip(ARMS, KEYS):
            cuts[arm] = np.load(bind(run_dir / (arm + '-cutoff.npy'), rr['outputs'][arm + '-cutoff.npy']), allow_pickle=False)
            rebuilt, authorities[arm] = scalar_cutoff(calibration(key + '/raw'), calibration(key + '/support'), calibration('MZ37'), tcal, kcal)
            np.testing.assert_array_equal(cuts[arm], rebuilt)
            assert np.isfinite(cuts[arm]).all(), 'Unexpected nonfinite cutoff: preserve evidence before JSON reporting'
            added = (calibration(key + '/raw').astype(float) >= cuts[arm]) & calibration(key + '/support') & (calibration('MZ37') < 0)
            assert not (added & ~tcal).any()
        scores, changes, attributes, cases, context, cached = {}, {}, {}, {}, {}, {}
        scalar_bits = reference_rows = native_checks = missing_rows = 0
        for cohort in COHORTS:
            truth, known = p[cohort + '/truth'], p[cohort + '/known']
            refs = tuple(s58['methods']) if cohort == 'mz55' else tuple(s57['methods'])
            records = records55 if cohort == 'mz55' else records48 if cohort == 'mz48' else None
            if cohort == 'mz36':
                truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
                admitted = p['mz36/admitted_index']
                assert truth.shape == known.shape == (400, 4) and int((~known).sum()) == 80 and len(admitted) == 380
            groups = {'all': np.arange(len(truth))}
            if records is not None:
                groups.update(groups55 if cohort == 'mz55' else groups48)
                for field in ('role', 'family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                    for value in sorted({str(r[field]) for r in records}):
                        groups[field + '/' + value] = np.array([i for i, r in enumerate(records) if str(r[field]) == value])
            scores[cohort], changes[cohort], context[cohort] = {}, {}, {}
            for profile in PROFILES:
                prefix = cohort + '/' + profile + '/'
                for arm, key in zip(ARMS, KEYS):
                    values = [p[prefix + key + '/' + name] for name in ('raw', 'support', 'winner', 'anchor_vector', 'anchor_available', 'candidate')]
                    validate_candidate(*values, p[prefix + 'MZ37'], cuts[arm])
                    np.testing.assert_array_equal(values[4], p[prefix + 'MZ56/GLOBAL_ANCHOR/anchor_available'])
                    missing_rows += int((~values[4]).sum())
                a = reconstruct(p, prefix, refs)
                if cohort == 'mz36':
                    expanded = {}
                    for key, value in a.items():
                        expanded[key] = np.full((400, 4), np.nan)
                        expanded[key][admitted] = value
                    a = expanded
                scores[cohort][profile], changes[cohort][profile] = {}, {}
                for group, ids in groups.items():
                    aa = {key: value[ids] for key, value in a.items()}
                    rows = {key: metrics(value, truth[ids], known[ids]) for key, value in aa.items()}
                    scores[cohort][profile][group] = rows
                    changes[cohort][profile][group] = compare(aa, truth[ids], known[ids])
                    prior_groups = s58['profiles'][profile] if cohort == 'mz55' else s57['conditions'][cohort][profile]
                    if group in prior_groups:
                        previous = prior_groups[group]['methods'] if cohort == 'mz55' else prior_groups[group]
                        for key in refs:
                            assert_metric_equal(rows[key], previous[key])
                            reference_rows += 1
                for key in NEW:
                    assert scalar_metrics(a[key], truth, known) == scores[cohort][profile]['all'][key]
                    scalar_bits += int(known.sum())
                cached.setdefault(profile, {})[cohort] = (a, truth, known)
                if records is None:
                    continue
                attributes.setdefault(cohort, {})[profile] = {}
                cases.setdefault(cohort, {})[profile] = []
                lab = labels[cohort]
                crop, sensor = p['mz54/crop_mask'].flatten(), p['mz54/sensor_coverage'].flatten()
                for key in KEYS:
                    winner, hit, wk = native_masks(p, prefix, key, lab['fullframe_event_counts'], lab['valid_counts'])
                    native_checks += hit.size
                    added = (a[key + '/candidate'] >= 0) & (a[BASE] < 0) & truth & known
                    masks = dict(added_tp=added, native_added_tp=added & hit, unknown_winner_added_tp=added & ~wk,
                                 known_non_native_added_tp=added & wk & ~hit, native_outside_crop_added_tp=added & hit & ~crop[winner],
                                 native_outside_sensor_added_tp=added & hit & ~sensor[winner])
                    for group, ids in groups.items():
                        attributes[cohort][profile].setdefault(group, {})[key] = {name: mask[ids].sum(0).tolist() for name, mask in masks.items()}
                    for i, q in zip(*np.where(added)):
                        cases[cohort][profile].append(dict(model=key, index=int(i), frame_id=records[i]['frame_id'], query=QUERIES[q],
                            **{k: records[i][k] for k in ('role', 'family', 'site_id', 'relation', 'support_context', 'pair_id')},
                            native=bool(hit[i, q]), known=bool(wk[i, q]), winner=int(winner[i, q]),
                            outside_sensor=bool(~sensor[winner[i, q]]), raw=float(p[prefix + key + '/raw'][i, q]), cutoff=float(cuts[key.split('/')[-1]][q])))
                pairrows = {}
                for i, row in enumerate(records):
                    pairrows.setdefault(row['pair_id'], []).append(i)
                pi = np.array(list(pairrows.values()))
                context[cohort][profile] = {key: dict(changed_query_bits=((value[pi[:, 0]] >= 0) != (value[pi[:, 1]] >= 0)).sum(0).tolist(),
                    changed_pairs=int(((value[pi[:, 0]] >= 0) != (value[pi[:, 1]] >= 0)).any(1).sum()), total_pairs=1280)
                    for key, value in a.items() if key in (BASE,) + NEW}
        aggregates, aggregate_changes = {}, {}
        aggregate_methods = tuple(s57['methods']) + NEW
        for profile in PROFILES:
            data = cached[profile]
            legacy = [(cohort, np.arange(len(data[cohort][1]))) for cohort in ('relation10000', 'distance5000', 'rich', 'mz36')]
            selections = dict(legacy_noncal=legacy, mz48_fit=[('mz48', groups48['fit'])], mz48_nonfit=[('mz48', groups48['nonfit'])])
            selections['all_old_noncal'] = legacy + selections['mz48_fit'] + selections['mz48_nonfit']
            aggregates[profile], aggregate_changes[profile] = {}, {}
            for group, selected in selections.items():
                tt = np.concatenate([data[c][1][ii] for c, ii in selected]); kk = np.concatenate([data[c][2][ii] for c, ii in selected])
                aa = {key: np.concatenate([data[c][0][key][ii] for c, ii in selected]) for key in aggregate_methods}
                aggregates[profile][group] = {key: metrics(v, tt, kk) for key, v in aa.items()}
                aggregate_changes[profile][group] = compare(aa, tt, kk)
                prior_group = 'all_noncal' if group == 'all_old_noncal' else group
                for key in s57['methods']:
                    assert_metric_equal(aggregates[profile][group][key], s57['aggregates'][profile][prior_group][key])
                    reference_rows += 1
        native_added = {key: attributes['mz55']['DROP_CLOSE']['heldout_new_family'][key]['native_added_tp'] for key in KEYS}
        gate = primary_gate(native_added, scores['mz55']['DROP_CLOSE']['HELDOUT_SITE'], aggregates['DROP_CLOSE']['legacy_noncal'])
        result = dict(status='PASS', primary_gate=gate, event_order=QUERIES, new_methods=NEW,
            conditions=scores, comparisons=changes, aggregates=aggregates, aggregate_comparisons=aggregate_changes,
            attribution=attributes, support_context_pairs=context, original_rule_cutoffs={arm: cut.tolist() for arm, cut in cuts.items()},
            cutoff_authority=authorities, source_role='CONSUMED_DEVELOPMENT', all_mz55_roles_descriptive=True,
            limitation='Matched coverage continuation. Both candidates and final fixed ORs reported; no selected winner or cutoff change. Native winning cell is localization evidence, not proof of causal use or sensor detectability. No hardware/natural-scene/clearance/safety claim.')
        audit = dict(status='PASS', preserved_MZ56_arrays=old_preserved, preserved_MZ58_arrays=transfer_preserved,
            prior_metric_rows_exact=reference_rows, scalar_known_decisions=scalar_bits, native_winner_lookups=native_checks,
            schedule=schedule_audit, exact_rebuilt_cutoffs=2, calibration_rows=1256, mz55_calibration_rows_used=0,
            all_missing_anchor_rows_checked=missing_rows, original_mz36_attempts=400, original_mz36_UNKNOWN_bits=80,
            mz55_drop_all_missing_frames=int((~p['mz55/DROP_CLOSE/MZ56/GLOBAL_ANCHOR/anchor_available']).sum()),
            local_UNKNOWN_cells={cohort: int((lab['valid_counts'] == 0).sum()) for cohort, lab in labels.items()},
            query_UNKNOWN_bits={cohort: int((~p[cohort + '/known']).sum()) for cohort in ('mz48', 'mz55')},
            mz55_heldout_new_family_frames=480, mz55_heldout_all_frames=640, paired_groups_preserved=2560,
            new_fits=0, new_inference_frames=0, RGB_reads=0, raw_depth_reads=0, checkpoint_loads=0)
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(out / 'result.json', result); write(out / 'audit.json', audit); write(out / 'native-added-events.json', cases)
        report(out / 'report.md', result)
        write(out / 'receipt.json', dict(status='PASS', inputs=inputs, outputs={name: sha(out / name) for name in
            ('result.json', 'audit.json', 'native-added-events.json', 'report.md')}, code_sha256=sha(__file__),
            seconds=time.perf_counter()-started, backend='FROZEN_PROTOCOL_CPU_ONLY saved arrays',
            primary_operational_definition=GATE_DEFINITION, new_fits=0, new_inference_frames=0, new_cutoffs=0))
        print('PASS', gate)
    except BaseException:
        write(out / 'failure.json', dict(status='FAIL', inputs=inputs, error=traceback.format_exc()))
        raise


def report(path, r):
    gate = r['primary_gate']
    lines = ['# MZ59 matched training coverage', '', f'Primary gate: **{"PASS" if gate["passes_primary"] else "FAIL"}**.', '',
             GATE_DEFINITION, '', f'Native new-family heldout BODY_NEAR gains: CONTROL {gate["control_native_added_body_near"]}, DIVERSE {gate["diverse_native_added_body_near"]}.', '',
             '| DROP group | Method | TP | FP | FN | Exact |', '| --- | --- | ---: | ---: | ---: | ---: |']
    groups = [('MZ55 all', r['conditions']['mz55']['DROP_CLOSE']['all']),
              ('MZ55 heldout640', r['conditions']['mz55']['DROP_CLOSE']['HELDOUT_SITE']),
              ('MZ55 new-family heldout480', r['conditions']['mz55']['DROP_CLOSE']['heldout_new_family']),
              ('Old noncal', r['aggregates']['DROP_CLOSE']['legacy_noncal']),
              ('MZ48 nonfit', r['aggregates']['DROP_CLOSE']['mz48_nonfit'])]
    for group, rows in groups:
        for key in (BASE, 'MZ57/GLOBAL_ANCHOR/UNION') + NEW:
            row = rows[key]
            lines.append(f'| {group} | {key} | ' + ' | '.join(str(sum(row[k])) for k in ('tp', 'fp', 'fn')) + f' | {row["exact_frames"]} |')
    lines += ['', 'All three profiles, per-query/group counts, paired gains/losses, source support pairs and native winner events are retained in result.json/native-added-events.json. MZ55 roles remain descriptive and its calibration-named rows did not set cutoffs; MZ36 retains all400 attempts and80 UNKNOWN bits.', '', r['limitation']]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test():
    current = dict(frames=2, tp=[1, 0, 0, 0], fp=[0, 0, 0, 0])
    legacy = dict(attempted_frames=2, tp=[1, 0, 0, 0], fp=[0, 0, 0, 0])
    assert_metric_equal(current, legacy)
    assert_metric_equal(current, current)
    assert_metric_equal(current, dict(legacy, frames=2))
    for incorrect in (dict(legacy, frames=3), dict(legacy, attempted_frames=3),
                      dict(legacy, tp=[0, 0, 0, 0]), dict(legacy, extra_count=0)):
        try:
            assert_metric_equal(current, incorrect)
        except AssertionError:
            pass
        else:
            raise AssertionError('Metric schema correction concealed a count/schema mismatch')
    raw = np.full((1256, 4), -.5, np.float32)
    support = np.ones_like(raw, bool); baseline = np.full_like(raw, -1); truth = np.zeros_like(raw, bool); known = np.ones_like(raw, bool)
    raw[42] = np.array([1., 2., 3., 4.], np.float32)
    raw[99] = 99; truth[99] = True
    cut, refs = scalar_cutoff(raw, support, baseline, truth, known)
    np.testing.assert_array_equal(cut, np.nextafter(np.array([1., 2., 3., 4.]), np.inf))
    assert all(x['maximum_rows'] == [42] for x in refs)
    keys = [k + '/UNION' for k in KEYS]
    rows = {keys[0]: {'fp': [1, 0, 0, 0]}, keys[1]: {'fp': [1, 0, 0, 0]}}
    native = {KEYS[0]: [1, 0, 0, 0], KEYS[1]: [2, 0, 0, 0]}
    assert primary_gate(native, rows, rows)['passes_primary']
    bad = {keys[0]: {'fp': [1, 0, 0, 0]}, keys[1]: {'fp': [2, 0, 0, 0]}}
    assert not primary_gate(native, bad, rows)['passes_primary']
    baseline = np.array([[1., -1., -1., -1.]])
    scores = np.array([[.2, .5, -.1, .2]], np.float32)
    cutoff = np.array([.5, .5, 0., .1])
    expected = np.where((baseline < 0) & (scores.astype(float) >= cutoff), scores.astype(float) - cutoff, baseline)
    validate_candidate(scores, np.ones((1, 4), bool), np.zeros((1, 4), np.int16),
                       np.zeros((1, 8)), np.zeros(1, bool), expected, baseline, cutoff)
    corrupted = expected.copy(); corrupted[0, 1] = -1
    try:
        validate_candidate(scores, np.ones((1, 4), bool), np.zeros((1, 4), np.int16),
                           np.zeros((1, 8)), np.zeros(1, bool), corrupted, baseline, cutoff)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong zero-boundary candidate accepted')
    original = dict(shared=np.tile(np.arange(8), (600, 1)), OLD_NEG=np.tile(np.arange(8), (600, 1)),
                    query=np.tile(np.repeat(np.arange(4), 2), (600, 1)))
    schedule = dict(original, mz55_diverse=np.random.default_rng(159).choice(np.arange(1600), (600, 4), replace=True))
    assert validate_schedules(schedule, original, np.arange(1600))['mz55_diverse_draws_exact'] == 2400
    # Unknown prediction must not count as a negative or an exact frame.
    values = np.array([[1, -1, 0, np.nan]])
    t = np.array([[1, 0, 0, 1]], bool); k = np.array([[1, 1, 1, 0]], bool)
    assert metrics(values, t, k) == scalar_metrics(values, t, k)
    assert metrics(values, t, k)['exact_frames'] == 0
    print('PASS synthetic frame-schema equivalence/conflict/count mismatch rejection; cutoff maximum/nextafter/positive exclusion; candidate equality/corruption; seeded matched schedule; final-OR FP gate; UNKNOWN scalar counts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert args.task is None
        self_test()
    else:
        assert args.task is not None
        run(args.task)
