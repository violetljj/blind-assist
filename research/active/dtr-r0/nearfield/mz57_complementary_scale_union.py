"""Post hoc saved-decision ORs; no model, source image or native-depth reads."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from mz40_evaluate import metrics, paired
from mz53_dual_readout_union import scalar_or


COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
MODES = ('LOCAL_ONLY', 'GLOBAL_ANCHOR', 'GLOBAL_SUPPRESSED')
BASE = 'OLD_NEG/UNION'
UNIONS = tuple('MZ57/' + mode + '/UNION' for mode in MODES)
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def run(task):
    started = time.perf_counter()
    task = task.resolve()
    out = task / 'score-v1'
    assert not out.exists(), 'Never overwrite a previous MZ57 score or failure'
    out.mkdir(parents=True)
    inputs = {}
    try:
        def bind(path, expected=None):
            path = Path(path).resolve(strict=True)
            digest = sha(path)
            assert expected is None or digest == expected, str(path)
            inputs[str(path)] = digest
            return path

        bind(Path(__file__))
        bind(Path(__file__).with_name('MZ57_COMPLEMENTARY_SCALE_UNION_20260911.md'))
        for name in ('mz40_evaluate.py', 'mz53_dual_readout_union.py'):
            bind(Path(__file__).with_name(name))
        prior_task = task.parent / 'mz56-global-anchor-20260911'
        sr = read(bind(prior_task / 'score-v1/receipt.json'))
        assert sr['status'] == 'PASS'

        def scored_input(suffix):
            rows = [(p, h) for p, h in sr['inputs'].items() if p.replace('\\', '/').endswith(suffix)]
            assert len(rows) == 1, suffix
            return bind(*rows[0])

        source = scored_input('mz56-global-anchor-20260911/run-v2/receipt.json').parent
        rr = read(source / 'receipt.json')
        assert rr['status'] == 'PASS' and rr['run_name'] == 'run-v2'
        model_sources = {}
        for name in ('mz56_global_anchor_model.py', 'mz54_full_rgb_model.py'):
            rows = [(p0, h) for p0, h in rr['inputs'].items() if p0.replace('\\', '/').endswith('nearfield/' + name)]
            assert len(rows) == 1
            model_sources[name] = dict(path=str(bind(*rows[0])), sha256=rows[0][1])
        score = read(bind(prior_task / 'score-v1/result.json', sr['outputs']['result.json']))
        prior_audit = read(bind(prior_task / 'score-v1/audit.json', sr['outputs']['audit.json']))
        assert prior_audit['status'] == 'PASS'
        prediction_path = bind(source / 'predictions.npz', rr['outputs']['predictions.npz'])
        group_path = bind(source / 'groups.json', rr['outputs']['groups.json'])
        for path in (prediction_path, group_path):
            assert inputs[str(path)] == sr['inputs'][str(path)]
        p = load(prediction_path)
        grouping = read(group_path)
        records = grouping['records']
        groups = {k: np.array(v, dtype=np.int64) for k, v in grouping['groups'].items()}
        assert {k: len(v) for k, v in groups.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
        assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(2560))
        np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
        assert len({r['frame_id'] for r in records}) == 2560
        owners, pair_rows = {}, {}
        for group, ids in groups.items():
            for i in ids:
                pair = records[i]['pair_id']
                assert owners.setdefault(pair, group) == group
                pair_rows.setdefault(pair, []).append(int(i))
        pairs = np.array(list(pair_rows.values()))
        assert pairs.shape == (1280, 2)
        np.testing.assert_array_equal(p['mz48/truth'][pairs[:, 0]], p['mz48/truth'][pairs[:, 1]])
        groups['nonfit'] = np.sort(np.r_[groups['heldout_site'], groups['nonfit_family']])

        prior54 = task.parent / 'mz54-full-rgb-20260911/run-v1'
        r54 = read(scored_input('mz54-full-rgb-20260911/run-v1/receipt.json'))
        old_arrays = load(bind(prior54 / 'predictions.npz', r54['outputs']['predictions.npz']))
        for key, value in old_arrays.items():
            np.testing.assert_array_equal(p[key], value, err_msg=key)
        preserved_arrays = len(old_arrays)
        del old_arrays
        prior53 = task.parent / 'mz53-dual-readout-union-20260911/score-v1'
        r53 = read(bind(prior53 / 'receipt.json'))
        assert r53['status'] == 'PASS'
        score53 = read(bind(prior53 / 'result.json', r53['outputs']['result.json']))
        old51 = task.parent / 'mz51-training-coverage-20260911/run-v1/receipt.json'
        expected51 = [(p0, h) for p0, h in r53['inputs'].items()
                      if p0.replace('\\', '/').endswith('mz51-training-coverage-20260911/run-v1/receipt.json')]
        assert len(expected51) == 1
        r51 = read(bind(old51, expected51[0][1]))
        cuts = {mode: np.load(bind(source / (mode + '-cutoff.npy'), rr['outputs'][mode + '-cutoff.npy'])).tolist()
                for mode in MODES[:2]}
        cuts['GLOBAL_SUPPRESSED'] = cuts['GLOBAL_ANCHOR']
        retained_cuts = {name: r51['fits']['OLD_NEG'][name + '_cutoff'] for name in ('OPEN', 'GATED')}

        label_root = task.parent / 'mz52-full-frame-supervision-20260911'
        label_index = read(bind(label_root / 'source-index.json'))
        assert label_index['status'] == 'PASS'
        label_path = bind(label_root / label_index['array']['path'], label_index['array']['sha256'])
        labels = load(label_path)
        np.testing.assert_array_equal(labels['frame_ids'], p['mz48/frame_ids'])
        np.testing.assert_array_equal(labels['global_indices'], np.arange(2560))
        counts, valid_count = labels['fullframe_event_counts'], labels['valid_counts']
        assert counts.shape == (2560, 45, 80, 4) and valid_count.shape == (2560, 45, 80)
        assert counts.dtype == valid_count.dtype == np.uint8 and (counts <= valid_count[..., None]).all()
        np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, p['mz48/truth'])
        native = (counts > 0).reshape(2560, 3600, 4)
        local_known = (valid_count > 0).reshape(2560, 3600)
        coverage = p['mz54/sensor_coverage'].flatten()
        crop = p['mz54/crop_mask'].flatten()
        assert coverage.shape == crop.shape == (3600,)
        del labels, counts, valid_count

        references = tuple(score['methods'])
        assert BASE in references and all('MZ56/' + mode + '/candidate' in references for mode in MODES)
        methods = references + UNIONS
        scores, changes, attribution, cases, contexts, cached, decisions = {}, {}, {}, {}, {}, {}, {}
        scalar_bits = reference_rows = native_checks = 0

        def compare(values, truth, known):
            compare_to = (BASE, 'MZ37') + UNIONS
            result = {key: {ref: paired(values[key], values[ref], truth, known)
                            for ref in compare_to if ref != key} for key in UNIONS}
            for key in UNIONS:
                assert not any(result[key][BASE]['tp_lost']) and not any(result[key][BASE]['fp_removed'])
            return result

        for cohort in COHORTS:
            truth, known = p[cohort + '/truth'], p[cohort + '/known']
            if cohort == 'mz36':
                truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
                admitted = p['mz36/admitted_index']
                assert truth.shape == known.shape == (400, 4) and int((~known).sum()) == 80
                assert len(admitted) == len(np.unique(admitted)) == 380
                np.testing.assert_array_equal(truth[admitted], p['mz36/truth'])
                decisions['mz36/admitted_index'] = admitted
            decisions[cohort + '/truth'], decisions[cohort + '/known'] = truth, known
            decisions[cohort + '/frame_ids'] = p[cohort + '/frame_ids']
            cohort_groups = {'all': np.arange(len(truth))}
            if cohort == 'mz48':
                cohort_groups.update(groups)
                for field in ('family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                    for value in sorted({str(r[field]) for r in records}):
                        cohort_groups[field + '/' + value] = np.array([i for i, r in enumerate(records) if str(r[field]) == value])
            scores[cohort], changes[cohort] = {}, {}
            for profile in PROFILES:
                prefix = cohort + '/' + profile + '/'
                a = {name: p[prefix + name] for name in references if name not in ('MZ50/UNION', 'NEW_NEG/UNION', BASE)}
                for family in ('MZ50', 'NEW_NEG', 'OLD_NEG'):
                    lead = '' if family == 'MZ50' else family + '/'
                    a[family + '/UNION'] = np.maximum(a[lead + 'OPEN/candidate'], a[lead + 'GATED/candidate'])
                if cohort == 'mz36':
                    expanded = {}
                    for name, value in a.items():
                        expanded[name] = np.full((400, 4), np.nan)
                        expanded[name][admitted] = value
                    a = expanded
                for mode, key in zip(MODES, UNIONS):
                    constituent = 'MZ56/' + mode + '/candidate'
                    # Maximum carries the existing zero-threshold OR only.
                    # Its magnitude is not a probability or a calibrated score.
                    a[key] = np.maximum(a[BASE], a[constituent])
                    assert not (((a[BASE] >= 0) | (a[constituent] >= 0)) & (a[key] < 0)).any()
                    decisions[prefix + key] = a[key]
                scores[cohort][profile], changes[cohort][profile] = {}, {}
                for group, ids in cohort_groups.items():
                    aa = {key: value[ids] for key, value in a.items()}
                    evaluated = {name: metrics(value, truth[ids], known[ids]) for name, value in aa.items()}
                    for name in references:
                        assert evaluated[name] == score['conditions'][cohort][profile][group][name], (cohort, profile, group, name)
                        if name == BASE:
                            assert evaluated[name] == score53['conditions'][cohort][profile][group][name]
                        reference_rows += 1
                    scores[cohort][profile][group] = evaluated
                    changes[cohort][profile][group] = compare(aa, truth[ids], known[ids])
                for mode, key in zip(MODES, UNIONS):
                    scalar_bits += scalar_or(a[BASE], a['MZ56/' + mode + '/candidate'], a[key], truth, known,
                                             scores[cohort][profile]['all'][key])
                cached.setdefault(profile, {})[cohort] = (a, truth, known)
                if cohort == 'mz48':
                    attribution[profile], cases[profile] = {}, []
                    native_winner, winner_known, winners = {}, {}, {}
                    for mode in MODES:
                        key = prefix + 'MZ56/' + mode + '/'
                        win = p[key + 'winner']
                        assert win.shape == (2560, 4) and win.min() >= 0 and win.max() < 3600
                        actual = np.take_along_axis(native, win[:, None, :], 1)[:, 0] & p[key + 'support']
                        np.testing.assert_array_equal(actual, p[key + 'winning_native'])
                        native_checks += actual.size
                        native_winner[mode], winners[mode] = actual, win
                        winner_known[mode] = np.take_along_axis(local_known, win, 1)
                    for group, ids in groups.items():
                        attribution[profile][group] = {}
                        for mode, key in zip(MODES, UNIONS):
                            added = (a[key] >= 0) & (a[BASE] < 0) & truth & known
                            accepted = a['MZ56/' + mode + '/candidate'] >= 0
                            assert not (added & ~accepted).any()
                            win, wn, wk = winners[mode], native_winner[mode], winner_known[mode]
                            masks = dict(added_tp=added, added_native_winner=added & wn,
                                added_known_winner=added & wk, added_unknown_winner=added & ~wk,
                                added_native_outside_sensor=added & wn & ~coverage[win],
                                added_native_outside_crop=added & wn & ~crop[win],
                                added_native_below_original_crop=added & wn & (win // 80 >= 37),
                                added_native_bottom_boundary=added & wn & (win // 80 == 36),
                                added_fp=(a[key] >= 0) & (a[BASE] < 0) & ~truth & known)
                            attribution[profile][group][mode] = {k: v[ids].sum(0).tolist() for k, v in masks.items()}
                        global_added = (a[UNIONS[1]] >= 0) & (a[UNIONS[2]] < 0) & truth & known
                        global_win = winners['GLOBAL_ANCHOR']
                        global_native = native_winner['GLOBAL_ANCHOR']
                        attribution[profile][group]['global_path_increment'] = dict(
                            tp_over_suppressed=global_added[ids].sum(0).tolist(),
                            native_tp_over_suppressed=(global_added & global_native)[ids].sum(0).tolist(),
                            native_outside_tp_over_suppressed=(global_added & global_native & ~coverage[global_win])[ids].sum(0).tolist())
                    for i in groups['nonfit']:
                        for q, query in enumerate(QUERIES):
                            if known[i, q] and a[BASE][i, q] < 0 and any(a[k][i, q] >= 0 for k in UNIONS):
                                row = dict(index=int(i), frame_id=records[i]['frame_id'], query=query, truth=bool(truth[i, q]),
                                    **{name: records[i][name] for name in ('family', 'site_id', 'range', 'support_context', 'pair_id')})
                                for mode, union in zip(MODES, UNIONS):
                                    win = int(winners[mode][i, q])
                                    row[mode] = dict(accepted=bool(a[union][i, q] >= 0),
                                        native_winner=bool(native_winner[mode][i, q]), known_winner=bool(winner_known[mode][i, q]),
                                        winner=win, row=win // 80, col=win % 80, sensor_coverage=bool(coverage[win]))
                                cases[profile].append(row)
                    contexts[profile] = {key: ((a[key][pairs[:, 0]] >= 0) != (a[key][pairs[:, 1]] >= 0)).sum(0).tolist()
                                         for key in (BASE,) + UNIONS}

        aggregates, aggregate_changes = {}, {}
        for profile in PROFILES:
            data = cached[profile]
            legacy = [(data[name], np.arange(len(data[name][1]))) for name in ('relation10000', 'distance5000', 'rich', 'mz36')]
            pieces = dict(legacy_noncal=legacy, mz48_fit=[(data['mz48'], groups['fit'])],
                          mz48_nonfit=[(data['mz48'], groups['nonfit'])])
            pieces['all_noncal'] = legacy + pieces['mz48_fit'] + pieces['mz48_nonfit']
            aggregates[profile], aggregate_changes[profile] = {}, {}
            for group, selected in pieces.items():
                truth = np.concatenate([y[ids] for (_, y, _), ids in selected])
                known = np.concatenate([k[ids] for (_, _, k), ids in selected])
                a = {name: np.concatenate([values[name][ids] for (values, _, _), ids in selected]) for name in methods}
                evaluated = {name: metrics(value, truth, known) for name, value in a.items()}
                for name in references:
                    assert evaluated[name] == score['aggregates'][profile][group][name], (profile, group, name)
                    reference_rows += 1
                aggregates[profile][group], aggregate_changes[profile][group] = evaluated, compare(a, truth, known)

        drop = aggregates['DROP_CLOSE']
        baseline = drop['mz48_nonfit'][BASE]
        gates = {}
        # Use all legacy noncal groups for the documented historical reference.
        historical_fp = 0
        for cohort in ('relation10000', 'distance5000', 'rich', 'mz36'):
            a, truth, known = cached['DROP_CLOSE'][cohort]
            historical_fp += sum(paired(a['GATED/candidate'], a['MZ37'], truth, known)['fp_added'])
        assert historical_fp == 19
        for mode, key in zip(MODES, UNIONS):
            target = drop['mz48_nonfit'][key]
            legacy_cost = aggregate_changes['DROP_CLOSE']['legacy_noncal'][key]
            gates[mode] = dict(nonfit_tp=sum(target['tp']), nonfit_fp=sum(target['fp']),
                tp_gain_over_retained=sum(target['tp']) - sum(baseline['tp']),
                fp_added_over_retained=sum(aggregate_changes['DROP_CLOSE']['mz48_nonfit'][key][BASE]['fp_added']),
                old_noncal_fp_added_over_retained=sum(legacy_cost[BASE]['fp_added']),
                old_noncal_fp_added_over_MZ37=sum(legacy_cost['MZ37']['fp_added']),
                historical_MZ50_GATED_added_fp=historical_fp,
                passes_descriptive_utility=sum(target['tp']) > sum(baseline['tp']) and sum(target['fp']) <= sum(baseline['fp']))
        result = dict(status='PASS', primary_condition='DROP_CLOSE', event_order=QUERIES, methods=methods,
            conditions=scores, comparisons=changes, aggregates=aggregates, aggregate_comparisons=aggregate_changes,
            attribution=attribution, nonfit_added_cases=cases, context_pair_changed_decisions=contexts,
            primary_utility=gates, unchanged_retained_baseline=BASE,
            existing_cutoffs=dict(OLD_NEG=retained_cuts, MZ56=cuts), new_cutoffs=0,
            historical_reference_is_not_a_new_gate=True,
            known_before_registration=dict(GLOBAL_vs_retained_nonfit_tp_gained=24, tp_lost=77,
                old_noncal_fp_added=3, old_noncal_fp_removed=16),
            scope='Explicitly post hoc consumed controlled Development; all three fixed unions retained, no selection',
            composition='Boolean OR at existing zero decision boundaries; maximum is only a score carrier',
            aggregation_scope=score['aggregation_scope'],
            bindings=dict(model_sources=model_sources, MZ56_prediction_sha256=rr['outputs']['predictions.npz'],
                MZ56_checkpoint_hashes_from_sealed_receipt={mode: rr['outputs'][mode + '.pt'] for mode in MODES[:2]},
                MZ52_compact_label_sha256=inputs[str(label_path)]),
            deployment_cost='Saved-output CPU composition needs no new inference here. Deployment combines retained crop/ensemble and full-RGB anchor paths; Android latency/memory and removal of duplicate encoder work are unmeasured.',
            native_interpretation='MZ52 native-positive8x8 winning cell of newly accepted MZ56 constituent, not direct outside measurement')
        audit = dict(status='PASS', scalar_or_known_bits=scalar_bits, scalar_or_checks=54,
            reference_metric_rows_equal=reference_rows, preserved_MZ54_arrays=preserved_arrays,
            native_winner_lookups=native_checks, source_frames=2560, intact_pairs=1280,
            mz36_attempts=400, mz36_admitted=380, mz36_unknown_bits=80,
            all_constituent_positive_decisions_retained=True, fits=0, new_cutoffs=0,
            model_inference_frames=0, native_depth_reads=0, rgb_reads=0, threshold_searches=0)
        np.savez_compressed(out / 'decisions.npz', **decisions)
        write(out / 'result.json', result)
        write(out / 'audit.json', audit)
        report(out / 'report.md', result, audit)
        final = dict(status='PASS', inputs=inputs, seconds=time.perf_counter() - started,
            backend='NumPy CPU saved-output composition',
            outputs={name: sha(out / name) for name in ('result.json', 'audit.json', 'decisions.npz', 'report.md')})
        write(out / 'receipt.json', final)
        print(json.dumps(dict(status='PASS', utility=gates, audit=audit, seconds=final['seconds']), indent=2))
    except BaseException:
        write(out / 'failure.json', dict(status='FAIL', inputs=inputs, traceback=traceback.format_exc(),
                                        seconds=time.perf_counter() - started))
        raise


def report(path, result, audit):
    drop = result['aggregates']['DROP_CLOSE']
    lines = ['# MZ57: fixed complementary-scale union result', '',
        'POSTHOC consumed Development: GLOBAL24TP gained/77lost against the retained union, '
        'and old3FP added/16removed, were known before registration. This is not fresh confirmation. '
        'All three unions below use unchanged existing decisions; no fit, new cutoff or selection.', '',
        '| DROP group | retained OLD_NEG union TP/FP/FN | +LOCAL | +GLOBAL | +GLOBAL_SUPPRESSED |',
        '|---|---:|---:|---:|---:|']
    for group, values in drop.items():
        columns = ['/'.join(str(sum(values[name][k])) for k in ('tp', 'fp', 'fn')) for name in (BASE,) + UNIONS]
        lines.append('| ' + group + ' | ' + ' | '.join(columns) + ' |')
    lines += ['', 'legacy_noncal excludes old DEV; nonfit=640heldout-site+384withheld-family. '
        'all_noncal includes1280fit frames once and is not wholly heldout.', '',
        '| DROP nonfit query | retained TP/FP/FN | +LOCAL | +GLOBAL | +SUPPRESSED |',
        '|---|---:|---:|---:|---:|']
    for q, query in enumerate(QUERIES):
        columns = ['/'.join(str(drop['mz48_nonfit'][name][k][q]) for k in ('tp', 'fp', 'fn')) for name in (BASE,) + UNIONS]
        lines.append('| ' + query + ' | ' + ' | '.join(columns) + ' |')
    lines += ['', '| Union | descriptive utility | nonfit addedTP/FP | old addedFP vs retained / MZ37 | native newTP / outside45 |',
              '|---|---|---:|---:|---:|']
    for mode in MODES:
        gate = result['primary_utility'][mode]
        loc = result['attribution']['DROP_CLOSE']['nonfit'][mode]
        lines.append(f"| {mode} | {'PASS' if gate['passes_descriptive_utility'] else 'FAIL'} | "
            f"{gate['tp_gain_over_retained']}/{gate['fp_added_over_retained']} | "
            f"{gate['old_noncal_fp_added_over_retained']}/{gate['old_noncal_fp_added_over_MZ37']} | "
            f"{sum(loc['added_native_winner'])}/{sum(loc['added_native_outside_sensor'])} |")
    lines += ['', 'Historical MZ50 GATED adds19oldnoncalFP over MZ37; this remains a reference, '
        'not a newly tuned gate or budget. Report all old costs even if nonfit descriptive utility passes.', '',
        '| Nonfit profile | retained TP/FP | +LOCAL | +GLOBAL | +SUPPRESSED |',
        '|---|---:|---:|---:|---:|']
    for profile in PROFILES:
        values = result['aggregates'][profile]['mz48_nonfit']
        columns = ['/'.join(str(sum(values[name][k])) for k in ('tp', 'fp')) for name in (BASE,) + UNIONS]
        lines.append('| ' + profile + ' | ' + ' | '.join(columns) + ' |')
    lines += ['', 'GLOBAL versus SUPPRESSED unions uses the same learned GLOBAL weights/cutoff, '
        'with only the global vector suppressed. GLOBAL versus LOCAL also includes learned-weight differences. '
        'All fixed comparisons and query-wise paired changes are in result.json.', '',
        result['deployment_cost'] + ' CPU composition time is not deployment inference time.', '',
        f"Validation: {audit['scalar_or_known_bits']} scalar known-bit OR/count checks; "
        f"{audit['reference_metric_rows_equal']} prior metric rows equal; "
        f"{audit['preserved_MZ54_arrays']} original arrays preserved; "
        f"{audit['native_winner_lookups']} independently checked MZ52 winners. "
        'MZ36 remains400attempts/20excluded/80UNKNOWN. No native depth, RGB, model or cutoff search was used.', '',
        'Native attribution applies only to newly accepted MZ56 constituents beyond the retained union. '
        'An8x8 cell containing a native query point is not direct ToF observation outside its field. '
        'No hardware, natural-scene, calibrated-uncertainty, clearance or default-App claim follows.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args()
    run(args.task)
