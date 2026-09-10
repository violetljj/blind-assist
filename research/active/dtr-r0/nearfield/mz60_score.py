"""Independent saved-output score for the single MZ60 WITNESS fit.

Use --self-test without experiment reads; --task only after sealed-run SCORE GO.
"""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np

from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired
from mz59_score import scalar_cutoff, validate_candidate, reconstruct, native_masks, assert_metric_equal

COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55')
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
KEY = 'MZ60/WITNESS'
NEW = (KEY + '/candidate', KEY + '/UNION')
BASE = 'OLD_NEG/UNION'
REFERENCES = (BASE, 'MZ37', 'MZ57/GLOBAL_ANCHOR/UNION',
              'MZ59/CONTROL/candidate', 'MZ59/DIVERSE/candidate', 'MZ59/CONTROL/UNION', 'MZ59/DIVERSE/UNION')


def retaining_gate(native_body_near, heldout, legacy, nonfit):
    values = dict(native_new_family_heldout_body_near=int(native_body_near), heldout_tp=sum(heldout['tp']), heldout_fp=sum(heldout['fp']),
                  legacy_tp=sum(legacy['tp']), legacy_fp=sum(legacy['fp']), nonfit_mz48_tp=sum(nonfit['tp']), nonfit_mz48_fp=sum(nonfit['fp']))
    clauses = dict(native_body_near_exceeds_one=values['native_new_family_heldout_body_near'] > 1,
                   heldout_tp_at_least_454=values['heldout_tp'] >= 454, heldout_fp_at_most_79=values['heldout_fp'] <= 79,
                   legacy_tp_at_least_2722=values['legacy_tp'] >= 2722, legacy_fp_at_most_45=values['legacy_fp'] <= 45,
                   mz48_nonfit_tp_at_least_514=values['nonfit_mz48_tp'] >= 514, mz48_nonfit_fp_at_most_23=values['nonfit_mz48_fp'] <= 23)
    return dict(values=values, clauses=clauses, passes_primary=all(clauses.values()),
                cost_surface='Final unchanged OLD_NEG union OR WITNESS candidate; native gains exclude inherited OLD_NEG positives')


def compare(a, truth, known):
    return {key: {ref: paired(a[key], a[ref], truth, known) for ref in REFERENCES} for key in NEW}


def run(task):
    task = task.resolve(strict=True)
    out, rd, work = task / 'score-v1', task / 'run-v1', task.parent
    assert not out.exists(), 'Never overwrite score/failure evidence'
    out.mkdir()
    started, inputs = time.perf_counter(), {}
    try:
        def bind(path, expected=None):
            path = Path(path).resolve(strict=True)
            h = sha(path)
            assert expected is None or h == expected, f'Input hash mismatch: {path}'
            inputs[str(path)] = h
            return path

        for name in ('mz60_score.py', 'mz59_score.py', 'mz58_score.py'):
            bind(Path(__file__).with_name(name))
        rr = read(bind(rd / 'receipt.json'))
        assert rr['status'] == 'PASS' and rr['steps_per_arm'] == rr['total_steps'] == 600
        assert rr['trainable_parameters'] == 11020 and rr['positive_pool'] == 'known_native_witness'
        assert rr['original_baseline_cohort_inferences'] == rr['mz55_calibration_rows_used'] == 0
        assert rr['initial_train_parity_unique_frames'] == 16 and rr['original_calibration_frames'] == 1256
        assert rr['new_cutoffs'] == 1 and rr['threshold_searches'] == 0
        assert rr['exact_initial_weights'] and rr['source_unknown_preserved'] and rr['native_depth_reads'] == 0
        assert set(rr['fits']) == {'WITNESS'} and rr['fits']['WITNESS']['steps'] == 600
        def producer(path):
            path = Path(path).resolve(strict=True)
            found = [h for p, h in rr['inputs'].items() if Path(p).resolve() == path]
            assert len(found) == 1, f'Producer did not bind {path}'
            return bind(path, found[0])

        producer(Path(__file__).with_name('MZ60_NATIVE_QUERY_20260911.md'))
        definition = read(producer(task / 'primary-definition.json'))
        assert definition == rr['primary_definition']
        assert definition['status'] == 'FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        for path, h in rr['inputs'].items():
            if path.endswith('.py'):
                bind(path, h)
        for name in ('predictions.npz', 'schedule.npz', 'groups.json', 'WITNESS-cutoff.npy', 'WITNESS.pt',
                     'initialization-parity.json', 'initial-missing.json', 'learned-missing.json'):
            bind(rd / name, rr['outputs'][name])
        for name in ('initial-missing.json', 'learned-missing.json'):
            record = read(rd / name)
            assert record['status'] == 'PASS' and record['frames_per_arm'] == 8 and record['all_missing_context_exactly_zero']
        parity = read(rd / 'initialization-parity.json')
        assert parity['status'] == 'PASS' and parity['unique_train_frames'] == 16 and parity['atol'] == 2e-5 and parity['rtol'] == 1e-6
        p = load(rd / 'predictions.npz')
        previous = work / 'mz59-training-diversity-20260911'
        r59 = read(producer(previous / 'run-v1/receipt.json'))
        s59seal = read(producer(previous / 'score-v1/receipt.json'))
        assert r59['status'] == s59seal['status'] == 'PASS'
        assert rr['initial_parameter_sha256']['WITNESS'] == r59['initial_parameter_sha256']['DIVERSE'] == r59['initial_parameter_sha256']['CONTROL']
        producer(work / 'mz56-global-anchor-20260911/run-v2/GLOBAL_ANCHOR.pt')
        prior = read(bind(previous / 'score-v1/result.json', s59seal['outputs']['result.json']))
        old = load(bind(previous / 'run-v1/predictions.npz', r59['outputs']['predictions.npz']))
        for key, value in old.items():
            np.testing.assert_array_equal(p[key], value, err_msg=key)
        preserved = len(old)
        assert set(rr['baseline_arrays_preserved']) == set(old) if isinstance(rr['baseline_arrays_preserved'], list) else rr['baseline_arrays_preserved'] == preserved
        del old
        schedule = load(rd / 'schedule.npz')
        schedule59 = load(bind(previous / 'run-v1/schedule.npz', r59['outputs']['schedule.npz']))
        assert set(schedule) == set(schedule59)
        for key in schedule:
            np.testing.assert_array_equal(schedule[key], schedule59[key])
        np.testing.assert_array_equal(parity['ids'], np.unique(schedule['shared'])[:16])
        grouping = read(rd / 'groups.json')
        assert grouping == read(bind(previous / 'run-v1/groups.json', r59['outputs']['groups.json']))
        groups48 = {k: np.array(v, int) for k, v in grouping['groups'].items()}
        assert {k: len(v) for k, v in groups48.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
        groups48['nonfit'] = np.sort(np.r_[groups48['heldout_site'], groups48['nonfit_family']])
        records = dict(mz48=grouping['records'], mz55=grouping['mz55_records'])
        groups55 = {role: np.array([i for i, r in enumerate(records['mz55']) if r['role'] == role])
                    for role in ('TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_SITE')}
        assert {k: len(v) for k, v in groups55.items()} == dict(TRAIN_CANDIDATE=1600, CALIBRATION=320, HELDOUT_SITE=640)
        groups55['heldout_new_family'] = np.array([i for i in groups55['HELDOUT_SITE'] if records['mz55'][i]['family'] != 'retained_rod'])
        groups55['nonfit'] = np.sort(np.r_[groups55['CALIBRATION'], groups55['HELDOUT_SITE']])
        assert len(groups55['heldout_new_family']) == 480
        assert np.isin(schedule['mz55_diverse'], groups55['TRAIN_CANDIDATE']).all()
        def prior_input(suffix):
            found = [(p, h) for p, h in s59seal['inputs'].items() if p.replace('\\', '/').endswith(suffix)]
            assert len(found) == 1, suffix
            return bind(*found[0])
        labels = dict(mz48=load(prior_input('mz52-full-frame-supervision-20260911/verified-v1/fullframe-cells.npz')),
                      mz55=load(prior_input('mz55-diverse-mesh-source-20260911/source-v1/fullframe-cells.npz')))
        for cohort, lab in labels.items():
            np.testing.assert_array_equal(lab['frame_ids'], p[cohort + '/frame_ids'])
            np.testing.assert_array_equal(lab['frame_ids'], [r['frame_id'] for r in records[cohort]])
            assert lab['fullframe_event_counts'].shape == (2560, 45, 80, 4) and lab['valid_counts'].shape == (2560, 45, 80)
            assert lab['fullframe_event_counts'].dtype == lab['valid_counts'].dtype == np.uint8
            assert (lab['fullframe_event_counts'] <= lab['valid_counts'][..., None]).all()
            np.testing.assert_array_equal(lab['fullframe_event_counts'].sum((1, 2)) >= 3, p[cohort + '/truth'])
        for name in ('crop_mask', 'sensor_coverage', 'rays'):
            np.testing.assert_array_equal(p['mz60/' + name], p['mz59/' + name])
        cut = np.load(rd / 'WITNESS-cutoff.npy', allow_pickle=False)
        cal = groups48['calibration']
        def calibration(name):
            return np.concatenate([p['DEV/DROP_CLOSE/' + name], p['mz48/DROP_CLOSE/' + name][cal]])
        truth_cal = np.concatenate([p['DEV/truth'], p['mz48/truth'][cal]])
        known_cal = np.concatenate([p['DEV/known'], p['mz48/known'][cal]])
        rebuilt, cutoff_authority = scalar_cutoff(calibration(KEY + '/raw'), calibration(KEY + '/support'), calibration('MZ37'), truth_cal, known_cal)
        np.testing.assert_array_equal(cut, rebuilt)
        assert cut.shape == (4,) and np.isfinite(cut).all()
        scores, comparisons, attribution, cases, contexts, cached = {}, {}, {}, {}, {}, {}
        scalar_bits = reference_rows = native_checks = missing_rows = 0
        for cohort in COHORTS:
            truth, known = p[cohort + '/truth'], p[cohort + '/known']
            if cohort == 'mz36':
                truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
                admitted = p['mz36/admitted_index']
                assert truth.shape == known.shape == (400, 4) and int((~known).sum()) == 80 and len(admitted) == 380
            groups = {'all': np.arange(len(truth))}
            if cohort in records:
                groups.update(groups55 if cohort == 'mz55' else groups48)
                for field in ('role', 'family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                    for value in sorted({str(r[field]) for r in records[cohort]}):
                        groups[field + '/' + value] = np.array([i for i, r in enumerate(records[cohort]) if str(r[field]) == value])
            scores[cohort], comparisons[cohort] = {}, {}
            for profile in PROFILES:
                prefix = cohort + '/' + profile + '/'
                reference_methods = tuple(prior['conditions'][cohort][profile]['all'])
                a = reconstruct(p, prefix, reference_methods)
                values = [p[prefix + KEY + '/' + name] for name in ('raw', 'support', 'winner', 'anchor_vector', 'anchor_available', 'candidate')]
                validate_candidate(*values, p[prefix + 'MZ37'], cut)
                np.testing.assert_array_equal(values[4], p[prefix + 'MZ59/DIVERSE/anchor_available'])
                missing_rows += int((~values[4]).sum())
                a[NEW[0]] = values[-1]
                a[NEW[1]] = np.maximum(a[BASE], a[NEW[0]])
                np.testing.assert_array_equal(a[NEW[1]], p[prefix + NEW[1]])
                for i in range(len(a[BASE])):
                    for q in range(4):
                        assert bool(a[NEW[1]][i, q] >= 0) == (bool(a[BASE][i, q] >= 0) or bool(a[NEW[0]][i, q] >= 0))
                if cohort == 'mz36':
                    expanded = {}
                    for key, value in a.items():
                        expanded[key] = np.full((400, 4), np.nan); expanded[key][admitted] = value
                    a = expanded
                scores[cohort][profile], comparisons[cohort][profile] = {}, {}
                for group, ids in groups.items():
                    aa = {key: value[ids] for key, value in a.items()}
                    rows = {key: metrics(value, truth[ids], known[ids]) for key, value in aa.items()}
                    scores[cohort][profile][group] = rows
                    comparisons[cohort][profile][group] = compare(aa, truth[ids], known[ids])
                    for key in reference_methods:
                        assert_metric_equal(rows[key], prior['conditions'][cohort][profile][group][key])
                        reference_rows += 1
                for key in NEW:
                    assert scalar_metrics(a[key], truth, known) == scores[cohort][profile]['all'][key]
                    scalar_bits += int(known.sum())
                cached.setdefault(profile, {})[cohort] = (a, truth, known)
                if cohort not in labels:
                    continue
                lab = labels[cohort]
                winner, native, wk = native_masks(p, prefix, KEY, lab['fullframe_event_counts'], lab['valid_counts'])
                native_checks += native.size
                crop, sensor = p['mz60/crop_mask'].flatten(), p['mz60/sensor_coverage'].flatten()
                added = (a[NEW[0]] >= 0) & (a[BASE] < 0) & truth & known
                missed = (a[NEW[1]] < 0) & truth & known
                masks = dict(added_tp=added, native_added_tp=added & native, known_non_native_added_tp=added & wk & ~native,
                             unknown_winner_added_tp=added & ~wk, native_outside_crop_added_tp=added & native & ~crop[winner],
                             native_outside_sensor_added_tp=added & native & ~sensor[winner],
                             missed_native_winner=missed & native, missed_known_non_native_winner=missed & wk & ~native,
                             missed_UNKNOWN_winner=missed & ~wk, final_native_winner_tp=(a[NEW[1]] >= 0) & truth & known & native)
                attribution.setdefault(cohort, {})[profile] = {
                    group: {name: mask[ids].sum(0).tolist() for name, mask in masks.items()} for group, ids in groups.items()}
                events = []
                for i, q in zip(*np.where(added)):
                    events.append(dict(index=int(i), frame_id=records[cohort][i]['frame_id'], query=QUERIES[q],
                        **{k: records[cohort][i][k] for k in ('role', 'family', 'site_id', 'relation', 'range', 'setting', 'support_context', 'pair_id')},
                        winner=int(winner[i, q]), native=bool(native[i, q]), local_known=bool(wk[i, q]),
                        inside_crop=bool(crop[winner[i, q]]), inside_sensor=bool(sensor[winner[i, q]]),
                        raw=float(p[prefix + KEY + '/raw'][i, q]), cutoff=float(cut[q])))
                cases.setdefault(cohort, {})[profile] = events
                pairs = {}
                for i, r in enumerate(records[cohort]):
                    pairs.setdefault(r['pair_id'], []).append(i)
                assert len(pairs) == 1280 and all(len(ii) == 2 for ii in pairs.values())
                pi = np.array(list(pairs.values()))
                np.testing.assert_array_equal(truth[pi[:, 0]], truth[pi[:, 1]])
                contexts.setdefault(cohort, {})[profile] = {key: dict(pairs=1280,
                    changed_queries=((a[key][pi[:, 0]] >= 0) != (a[key][pi[:, 1]] >= 0)).sum(0).tolist(),
                    changed_pairs=int(((a[key][pi[:, 0]] >= 0) != (a[key][pi[:, 1]] >= 0)).any(1).sum())) for key in (BASE,) + NEW}
        aggregates, aggregate_comparisons = {}, {}
        for profile in PROFILES:
            data = cached[profile]
            legacy = [(c, np.arange(len(data[c][1]))) for c in ('relation10000', 'distance5000', 'rich', 'mz36')]
            selection = dict(legacy_noncal=legacy, mz48_fit=[('mz48', groups48['fit'])], mz48_nonfit=[('mz48', groups48['nonfit'])])
            selection['all_old_noncal'] = legacy + selection['mz48_fit'] + selection['mz48_nonfit']
            aggregates[profile], aggregate_comparisons[profile] = {}, {}
            for group, selected in selection.items():
                tt = np.concatenate([data[c][1][ii] for c, ii in selected]); kk = np.concatenate([data[c][2][ii] for c, ii in selected])
                methods = tuple(prior['aggregates'][profile][group]) + NEW
                aa = {key: np.concatenate([data[c][0][key][ii] for c, ii in selected]) for key in methods}
                aggregates[profile][group] = {key: metrics(v, tt, kk) for key, v in aa.items()}
                aggregate_comparisons[profile][group] = compare(aa, tt, kk)
                for key in prior['aggregates'][profile][group]:
                    assert_metric_equal(aggregates[profile][group][key], prior['aggregates'][profile][group][key])
                    reference_rows += 1
        gate = retaining_gate(attribution['mz55']['DROP_CLOSE']['heldout_new_family']['native_added_tp'][0],
            scores['mz55']['DROP_CLOSE']['HELDOUT_SITE'][NEW[1]], aggregates['DROP_CLOSE']['legacy_noncal'][NEW[1]],
            aggregates['DROP_CLOSE']['mz48_nonfit'][NEW[1]])
        result = dict(status='PASS', primary_gate=gate, queries=QUERIES, new_methods=NEW, conditions=scores,
            comparisons=comparisons, aggregates=aggregates, aggregate_comparisons=aggregate_comparisons,
            attribution=attribution, support_context_pairs=contexts, cutoff=cut.tolist(), cutoff_authority=cutoff_authority,
            source_role='CONSUMED_DEVELOPMENT', limitation='Single positive-query-pooling change with identical source/schedule/model/initialization. No threshold selection. Native winners are supporting localization evidence, not causal proof or hardware observability. Failed retaining clauses remain failures; no default-App or safety claim.')
        audit = dict(status='PASS', exact_preserved_mz59_arrays=preserved, exact_schedule_arrays=len(schedule), exact_groups=True,
            prior_metric_rows_exact=reference_rows, scalar_known_decisions=scalar_bits, native_winner_lookups=native_checks,
            original_calibration_rows=1256, cutoff_vectors_recomputed=1, mz55_calibration_rows_used=0,
            mz36_attempts=400, mz36_UNKNOWN_bits=80, mz55_all_missing_DROP_frames=int((~p['mz55/DROP_CLOSE/' + KEY + '/anchor_available']).sum()),
            all_missing_anchor_rows_checked=missing_rows, local_UNKNOWN_cells={c: int((v['valid_counts'] == 0).sum()) for c, v in labels.items()},
            new_fits=0, new_inference_frames=0, new_cutoffs=0, RGB_reads=0, native_depth_reads=0, checkpoint_loads=0)
        for path, h in inputs.items():
            assert sha(path) == h, path
        write(out / 'result.json', result); write(out / 'audit.json', audit); write(out / 'native-added-events.json', cases)
        report(out / 'report.md', result)
        write(out / 'receipt.json', dict(status='PASS', inputs=inputs, outputs={name: sha(out / name) for name in
            ('result.json', 'audit.json', 'native-added-events.json', 'report.md')}, code_sha256=sha(__file__),
            seconds=time.perf_counter()-started, backend='FROZEN_PROTOCOL_CPU_ONLY saved-output score', new_inference_frames=0, new_fits=0, new_cutoffs=0))
        print('PASS', gate)
    except BaseException:
        write(out / 'failure.json', dict(status='FAIL', inputs=inputs, error=traceback.format_exc()))
        raise


def report(path, r):
    gate = r['primary_gate']
    lines = ['# MZ60 native positive-query objective', '', f'Primary retaining check: **{"PASS" if gate["passes_primary"] else "FAIL"}**.', '',
             '| Clause | Result |', '| --- | --- |']
    for key, passed in gate['clauses'].items():
        lines.append(f'| {key} | {"PASS" if passed else "FAIL"} |')
    lines += ['', 'Values: ' + str(gate['values']), '', '| DROP group | Method | TP | FP | FN | Exact |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for group, rows in [('MZ55 all', r['conditions']['mz55']['DROP_CLOSE']['all']),
        ('MZ55 heldout640', r['conditions']['mz55']['DROP_CLOSE']['HELDOUT_SITE']),
        ('Legacy noncal', r['aggregates']['DROP_CLOSE']['legacy_noncal']), ('MZ48 nonfit', r['aggregates']['DROP_CLOSE']['mz48_nonfit'])]:
        for key in (BASE, 'MZ59/CONTROL/UNION', 'MZ59/DIVERSE/UNION') + NEW:
            row = rows[key]
            lines.append(f'| {group} | {key} | ' + ' | '.join(str(sum(row[k])) for k in ('tp', 'fp', 'fn')) + f' | {row["exact_frames"]} |')
    lines += ['', 'All original arrays, groups, UNKNOWN attempts and fixed comparisons remain. Full per-query/profile/source-role/family/site/relation/context metrics, paired losses and native winner counts are in result.json.', '', r['limitation']]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test():
    held, legacy, nonfit = dict(tp=[454, 0, 0, 0], fp=[79, 0, 0, 0]), dict(tp=[2722, 0, 0, 0], fp=[45, 0, 0, 0]), dict(tp=[514, 0, 0, 0], fp=[23, 0, 0, 0])
    assert retaining_gate(2, held, legacy, nonfit)['passes_primary']
    assert not retaining_gate(1, held, legacy, nonfit)['passes_primary']
    for group, field, value in ((0, 'tp', 453), (0, 'fp', 80), (1, 'tp', 2721), (1, 'fp', 46), (2, 'tp', 513), (2, 'fp', 24)):
        values = [dict(held), dict(legacy), dict(nonfit)]; values[group][field] = [value, 0, 0, 0]
        result = retaining_gate(2, *values)
        assert not result['passes_primary'] and sum(not passed for passed in result['clauses'].values()) == 1
    # Existing independent cutoff and candidate checks are exercised on synthetic
    # arrays here as well; equality uses >=0 while maximum negative uses nextafter.
    raw = np.zeros((1256, 4), np.float32); raw[3] = 1.
    support = np.ones_like(raw, bool); base = np.full_like(raw, -1.); truth = np.zeros_like(raw, bool)
    cut, _ = scalar_cutoff(raw, support, base, truth, np.ones_like(truth))
    np.testing.assert_array_equal(cut, np.nextafter(np.ones(4), np.inf))
    candidate = np.full((1, 4), -1.)
    validate_candidate(raw[3:4], support[:1], np.zeros((1, 4), np.int16), np.zeros((1, 8)),
                       np.zeros(1, bool), candidate, base[:1], cut)
    print('PASS seven independent gate boundaries; fixed cutoff nextafter; rejected max-negative candidate and all-missing vector')


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
