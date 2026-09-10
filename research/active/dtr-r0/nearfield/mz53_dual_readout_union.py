"""One CPU composition pass over sealed decisions: OPEN OR GATED, uniformly."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mz40_evaluate import metrics, paired


COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
FAMILIES = ('MZ50', 'NEW_NEG', 'OLD_NEG')
QUERY_ORDER = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def readout(family, mode):
    return mode if family == 'MZ50' else family + '/' + mode


def scalar_or(opened, gated, union, truth, known, expected):
    """Independent scalar OR truth table and truth/count audit, including UNKNOWN."""
    values = {key: [0] * 4 for key in ('tp', 'fp', 'fn', 'tn', 'known', 'unknown')}
    exact = 0
    for i in range(len(truth)):
        correct = True
        for q in range(4):
            o, g = bool(opened[i, q] >= 0), bool(gated[i, q] >= 0)
            decision = o or g
            assert bool(union[i, q] >= 0) == decision
            if not known[i, q]:
                values['unknown'][q] += 1
                correct = False
                continue
            values['known'][q] += 1
            target = bool(truth[i, q])
            key = ('tp' if target else 'fp') if decision else ('fn' if target else 'tn')
            values[key][q] += 1
            correct = correct and decision == target
        exact += int(correct)
    assert all(values[key] == expected[key] for key in values)
    assert exact == expected['exact_frames']
    return int(known.sum())


def run(task):
    started = time.perf_counter()
    task = task.resolve()
    out = task / 'score-v1'
    report_path = Path(__file__).with_name('MZ53_DUAL_READOUT_UNION_RESULTS_20260911.md')
    assert not out.exists() and not report_path.exists(), 'Keep any previously sealed composition intact'
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        assert expected is None or digest == expected, f'Hash mismatch: {path}'
        inputs[str(path)] = digest
        return path

    brief = bind(Path(__file__).with_name('MZ53_DUAL_READOUT_UNION_20260911.md'))
    original = task.parent / 'mz51-training-coverage-20260911'
    run_receipt = read(bind(original / 'run-v1/receipt.json'))
    score_receipt = read(bind(original / 'score-v1/receipt.json'))
    assert run_receipt['status'] == score_receipt['status'] == 'PASS'
    score = read(bind(original / 'score-v1/result.json', score_receipt['outputs']['result.json']))
    old_audit = read(bind(original / 'score-v1/audit.json', score_receipt['outputs']['audit.json']))
    assert old_audit['status'] == 'PASS' and old_audit['prior_MZ37_positive_retention']
    assert old_audit['mz36_unknown_bits'] == 80 and not score['primary_gate']['passes_primary']
    prediction_path = bind(original / 'run-v1/predictions.npz', run_receipt['outputs']['predictions.npz'])
    group_path = bind(original / 'run-v1/groups.json', run_receipt['outputs']['groups.json'])
    scored_inputs = {str(Path(path).resolve()): value for path, value in score_receipt['inputs'].items()}
    for path in (prediction_path, group_path, (original / 'run-v1/receipt.json').resolve()):
        assert inputs[str(path)] == scored_inputs[str(path)]
    for name in ('mz51_score.py', 'mz47_score.py', 'mz40_evaluate.py'):
        path = Path(__file__).with_name(name).resolve()
        bind(path, scored_inputs[str(path)])
    with np.load(prediction_path, allow_pickle=False) as archive:
        p = {key: archive[key] for key in archive.files}
    groups_file = read(group_path)
    records = groups_file['records']
    groups = {key: np.array(value, dtype=np.int64) for key, value in groups_file['groups'].items()}
    assert {key: len(value) for key, value in groups.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(2560))
    assert len({r['frame_id'] for r in records}) == 2560
    np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
    owners = {}
    for group, ids in groups.items():
        for i in ids:
            assert owners.setdefault(records[i]['pair_id'], group) == group
    assert len(owners) == 1280
    groups['nonfit'] = np.sort(np.concatenate([groups['heldout_site'], groups['nonfit_family']]))

    # Original MZ50 arrays and source index remain the exact saved comparators.
    for suffix in ('mz50-echo-independent-local-20260911/run-v1/predictions.npz',
                   'mz48-rich-kilotier-20260911/source-index.json',
                   'mz48-rich-kilotier-20260911/source-total-receipt.json',
                   'nearfield/mz51_training_coverage.py'):
        rows = [(Path(path), digest) for path, digest in run_receipt['inputs'].items()
                if path.replace('\\', '/').endswith(suffix)]
        assert len(rows) == 1
        path = bind(*rows[0])
        if suffix.endswith('predictions.npz'):
            with np.load(path, allow_pickle=False) as prior:
                for key in prior.files:
                    np.testing.assert_array_equal(p[key], prior[key], err_msg=key)
                frozen_arrays = len(prior.files)
    assert frozen_arrays == old_audit['frozen_MZ50_arrays_equal']

    individual = tuple(score['methods'])
    unions = tuple(family + '/UNION' for family in FAMILIES)
    methods = individual + unions
    scores, comparisons, local, context, cached, hashes = {}, {}, {}, {}, {}, {}
    scalar_bits, individual_parity_groups = 0, 0

    def paired_unions(values, truth, known):
        result = {union: {name: paired(values[union], values[name], truth, known)
                          for name in methods if name != union} for union in unions}
        for union in unions:
            assert not any(result[union]['MZ37']['tp_lost']) and not any(result[union]['MZ37']['fp_removed'])
        return result

    for cohort in COHORTS:
        truth, known = p[cohort + '/truth'], p[cohort + '/known']
        if cohort == 'mz36':
            truth, known = p['mz36/attempted_truth'], p['mz36/attempted_known']
            admitted = p['mz36/admitted_index']
            assert truth.shape == known.shape == (400, 4) and int((~known).sum()) == 80
            assert len(admitted) == len(np.unique(admitted)) == 380
            np.testing.assert_array_equal(p['mz36/attempted_frame_ids'][admitted], p['mz36/frame_ids'])
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
            a = {name: p[prefix + name] for name in individual}
            if cohort == 'mz36':
                expanded = {}
                for name, value in a.items():
                    expanded[name] = np.full((400, 4), np.nan)
                    expanded[name][admitted] = value
                a = expanded
            for family in FAMILIES:
                opened, gated = (a[readout(family, mode) + '/candidate'] for mode in ('OPEN', 'GATED'))
                # Maximum is only a score carrier for exact OR at the already
                # fixed zero boundary. No raw logit or cutoff is recalibrated.
                a[family + '/UNION'] = np.maximum(opened, gated)
            scores[cohort][profile], comparisons[cohort][profile] = {}, {}
            for group, ids in cohort_groups.items():
                aa = {key: value[ids] for key, value in a.items()}
                evaluated = {name: metrics(value, truth[ids], known[ids]) for name, value in aa.items()}
                for name in individual:
                    assert evaluated[name] == score['conditions'][cohort][profile][group][name], (cohort, profile, group, name)
                    individual_parity_groups += 1
                scores[cohort][profile][group] = evaluated
                comparisons[cohort][profile][group] = paired_unions(aa, truth[ids], known[ids])
            for family in FAMILIES:
                union = family + '/UNION'
                scalar_bits += scalar_or(a[readout(family, 'OPEN') + '/candidate'],
                    a[readout(family, 'GATED') + '/candidate'], a[union], truth, known,
                    scores[cohort][profile]['all'][union])
                hashes[cohort + '/' + profile + '/' + union] = hashlib.sha256(np.packbits(a[union] >= 0).tobytes()).hexdigest()
            cached.setdefault(profile, {})[cohort] = (a, truth, known)
            if cohort == 'mz48':
                local[profile] = {}
                for group, ids in groups.items():
                    local[profile][group] = {}
                    for family in FAMILIES:
                        ok, gk = (readout(family, mode) for mode in ('OPEN', 'GATED'))
                        op, gp = a[ok + '/candidate'] >= 0, a[gk + '/candidate'] >= 0
                        o_native, g_native = p[prefix + ok + '/winning_native'], p[prefix + gk + '/winning_native']
                        assert o_native.dtype == g_native.dtype == np.bool_
                        assert not (g_native & ~p[prefix + gk + '/support']).any()
                        positive = truth & known & (a['MZ37'] < 0)
                        negative = ~truth & known & (a['MZ37'] < 0)
                        union_tp = (op | gp) & positive
                        masks = dict(union_added_tp=union_tp, OPEN_only_tp=op & ~gp & positive,
                            GATED_only_tp=gp & ~op & positive, both_tp=op & gp & positive,
                            OPEN_only_tp_winning_native=op & ~gp & positive & o_native,
                            GATED_only_tp_winning_native=gp & ~op & positive & g_native,
                            union_added_tp_with_accepted_native_winner=union_tp & ((op & o_native) | (gp & g_native)),
                            union_added_tp_without_gated_candidate=union_tp & ~p[prefix + gk + '/support'],
                            union_added_tp_without_gated_candidate_native=union_tp & ~p[prefix + gk + '/support'] & o_native,
                            union_added_fp=(op | gp) & negative, OPEN_only_fp=op & ~gp & negative,
                            GATED_only_fp=gp & ~op & negative, both_fp=op & gp & negative)
                        values = {key: mask[ids].sum(0).tolist() for key, mask in masks.items()}
                        assert np.array_equal(values['union_added_tp'], np.array(values['OPEN_only_tp']) + values['GATED_only_tp'] + values['both_tp'])
                        local[profile][group][family] = values
                pairs = {}
                for i, row in enumerate(records):
                    pairs.setdefault(row['pair_id'], []).append(i)
                pair_ids = np.array(list(pairs.values()))
                assert pair_ids.shape == (1280, 2)
                np.testing.assert_array_equal(truth[pair_ids[:, 0]], truth[pair_ids[:, 1]])
                context[profile] = {name: ((value[pair_ids[:, 0]] >= 0) != (value[pair_ids[:, 1]] >= 0)).sum(0).tolist()
                                    for name, value in a.items()}

    aggregates, aggregate_comparisons = {}, {}
    for profile in PROFILES:
        data = cached[profile]
        legacy = [(data[name], np.arange(len(data[name][1]))) for name in ('relation10000', 'distance5000', 'rich', 'mz36')]
        pieces = dict(legacy_noncal=legacy, mz48_fit=[(data['mz48'], groups['fit'])],
                      mz48_nonfit=[(data['mz48'], groups['nonfit'])])
        pieces['all_noncal'] = legacy + pieces['mz48_fit'] + pieces['mz48_nonfit']
        aggregates[profile], aggregate_comparisons[profile] = {}, {}
        for group, selected in pieces.items():
            truth = np.concatenate([y[ids] for (_, y, _), ids in selected])
            known = np.concatenate([k[ids] for (_, _, k), ids in selected])
            a = {name: np.concatenate([values[name][ids] for (values, _, _), ids in selected]) for name in methods}
            evaluated = {name: metrics(value, truth, known) for name, value in a.items()}
            for name in individual:
                assert evaluated[name] == score['aggregates'][profile][group][name]
                individual_parity_groups += 1
            aggregates[profile][group] = evaluated
            aggregate_comparisons[profile][group] = paired_unions(a, truth, known)
    drop = aggregates['DROP_CLOSE']
    target = drop['mz48_nonfit']['OLD_NEG/UNION']
    legacy_cost = aggregate_comparisons['DROP_CLOSE']['legacy_noncal']['OLD_NEG/UNION']['MZ37']
    gate = dict(nonfit_tp=sum(target['tp']), nonfit_fp=sum(target['fp']),
                legacy_added_fp=sum(legacy_cost['fp_added']), nonfit_tp_at_least_453=sum(target['tp']) >= 453,
                nonfit_fp_at_most_21=sum(target['fp']) <= 21, legacy_added_fp_at_most_19=sum(legacy_cost['fp_added']) <= 19)
    gate['passes_primary'] = gate['nonfit_tp_at_least_453'] and gate['nonfit_fp_at_most_21'] and gate['legacy_added_fp_at_most_19']
    result = dict(status='PASS', primary_condition='DROP_CLOSE', event_order=list(QUERY_ORDER),
        methods=list(methods), conditions=scores, comparisons=comparisons, aggregates=aggregates,
        aggregate_comparisons=aggregate_comparisons, complementarity=local,
        context_pair_changed_decisions=context, primary_gate=gate, prior_MZ51_gate=score['primary_gate'],
        composition='Identical per-frame/query OR of existing OPEN and GATED decisions for each of MZ50, NEW_NEG, OLD_NEG',
        scope='Post hoc controlled Development composition on consumed sources; MZ51 gate failure remains unchanged',
        native_evidence='Saved winning_native flags only; no independent native check. Only an accepted readout winning flag can support a correct union addition.',
        aggregation_scope=score['aggregation_scope'])
    audit = dict(status='PASS', scalar_or_known_bits=scalar_bits, scalar_or_checks=3 * len(COHORTS) * len(PROFILES),
        sealed_individual_count_rows_equal=individual_parity_groups, frozen_MZ50_arrays_equal=frozen_arrays,
        saved_MZ51_scalar_known_bits=old_audit['scalar_known_bits'], prior_MZ37_positive_retention=True,
        pair_groups_intact=1280, source_frames=2560, mz36_attempts=400, mz36_unknown_bits=80,
        prior_MZ51_gate_unchanged=True, all_noncal_fit_frames=1280, aggregate_groups_counted_once=True,
        decision_bit_sha256=hashes, decision_hash_note='Packed >=0 decisions; frame identity and UNKNOWN masks remain bound by original arrays',
        training_steps=0, inference_frames=0, new_cutoffs=0, threshold_searches=0,
        native_depth_reads=0, native_label_array_reads=0, dense_feature_reads=0, checkpoint_loads=0)
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest, path
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'result.json', result)
    write(out / 'audit.json', audit)
    report(report_path, task, result, audit)
    code_path = Path(__file__).resolve()
    # This receipt is written once, after the result and tracked report are final.
    final = dict(status='PASS', inputs=inputs, code_sha256=sha(code_path), brief_sha256=sha(brief),
        report=dict(path=str(report_path), sha256=sha(report_path)),
        outputs={name: sha(out / name) for name in ('result.json', 'audit.json')},
        seconds=time.perf_counter() - started, backend='CPU Boolean saved-output bookkeeping; TASK_NOT_GPU_SUITABLE',
        training_steps=0, inference_frames=0, new_cutoffs=0, native_access=0,
        scientific_binding='FINAL; code/report/result/audit/receipt are sealed for this one pass')
    write(out / 'receipt.json', final)
    print('PRIMARY_GATE', gate)
    print('DROP_UNIONS', {group: {name: [sum(drop[group][name][key]) for key in ('tp', 'fp', 'fn')]
        for name in unions} for group in ('legacy_noncal', 'mz48_fit', 'mz48_nonfit', 'all_noncal')})
    print('SEALED', dict(seconds=final['seconds'], scalar_or_known_bits=scalar_bits,
                         individual_parity_rows=individual_parity_groups, code_sha256=final['code_sha256'],
                         report_sha256=final['report']['sha256'], outputs=final['outputs']))


def report(path, task, result, audit):
    gate = result['primary_gate']
    lines = ['# MZ53: fixed dual-readout union result', '',
        '2026-09-11. One registered CPU-only, explicitly post hoc Development composition pass. '
        'Every union uses the exact existing OPEN OR GATED decisions, uniformly across frames, queries and profiles. '
        'No cutoff, fit, model inference or native access was added.', '',
        f'Primary practical gate: **{"PASS" if gate["passes_primary"] else "FAIL"}**. '
        f'OLD_NEG union has {gate["nonfit_tp"]} nonfit true events and {gate["nonfit_fp"]} false events; '
        f'it adds {gate["legacy_added_fp"]} old noncalibration false events over MZ37. '
        'The declared bounds are TP >=453, FP <=21, and old added FP <=19. '
        'This is a new composition result; MZ51\'s individual OLD_NEG GATED gate remains failed.', '',
        '| DROP_CLOSE group | fixed MZ50 union TP/FP/FN | NEW_NEG union TP/FP/FN | OLD_NEG union TP/FP/FN |',
        '| --- | ---: | ---: | ---: |']
    for group, names in result['aggregates']['DROP_CLOSE'].items():
        cells = ['/'.join(str(sum(names[family + '/UNION'][key])) for key in ('tp', 'fp', 'fn')) for family in FAMILIES]
        lines.append(f'| {group} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'legacy_noncal excludes original DEV and includes relation2000, distance1000, older rich44 and all400 MZ36 attempts. '
        'MZ48 nonfit is heldout-site640 plus withheld-family384. all_noncal includes fit1280 once and is not wholly held-out evidence.', '',
        '| DROP_CLOSE nonfit | OPEN-only TP (native winner) | GATED-only TP (native winner) | both TP | union added TP with accepted native winner | no-gated-candidate TP |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for family, row in result['complementarity']['DROP_CLOSE']['nonfit'].items():
        total = lambda key: sum(row[key])
        lines.append(f'| {family} | {total("OPEN_only_tp")} ({total("OPEN_only_tp_winning_native")}) | '
            f'{total("GATED_only_tp")} ({total("GATED_only_tp_winning_native")}) | {total("both_tp")} | '
            f'{total("union_added_tp_with_accepted_native_winner")}/{total("union_added_tp")} | '
            f'{total("union_added_tp_without_gated_candidate")} |')
    lines += ['', 'Native columns reuse source-bound saved winning flags; they are not new native-depth validation. '
        'A correct event is not automatically a grounded local witness. Detailed per-query exclusive counts are retained below.', '',
        '| OLD_NEG nonfit query | OPEN-only TP / FP | GATED-only TP / FP | shared new TP / FP |',
        '| --- | ---: | ---: | ---: |']
    row = result['complementarity']['DROP_CLOSE']['nonfit']['OLD_NEG']
    for q, query in enumerate(QUERY_ORDER):
        lines.append(f'| {query} | {row["OPEN_only_tp"][q]}/{row["OPEN_only_fp"][q]} | '
            f'{row["GATED_only_tp"][q]}/{row["GATED_only_fp"][q]} | {row["both_tp"][q]}/{row["both_fp"][q]} |')
    lines += ['', '| DROP_CLOSE OLD_NEG union versus | group | TP gained/lost | FP added/removed | exact frames gained/lost |',
        '| --- | --- | ---: | ---: | ---: |']
    for group in ('legacy_noncal', 'mz48_nonfit', 'mz48_fit'):
        for other in ('MZ50/UNION', 'NEW_NEG/UNION', 'OLD_NEG/GATED/candidate'):
            row = result['aggregate_comparisons']['DROP_CLOSE'][group]['OLD_NEG/UNION'][other]
            lines.append(f'| {other} | {group} | {sum(row["tp_gained"])}/{sum(row["tp_lost"])} | '
                f'{sum(row["fp_added"])}/{sum(row["fp_removed"])} | {row["exact_gained"]}/{row["exact_lost"]} |')
    lines += ['', '| profile | fixed MZ50 union nonfit TP/FP | NEW_NEG union nonfit TP/FP | OLD_NEG union nonfit TP/FP |',
        '| --- | ---: | ---: | ---: |']
    for profile in PROFILES:
        names = result['aggregates'][profile]['mz48_nonfit']
        lines.append(f'| {profile} | ' + ' | '.join('/'.join(str(sum(names[family + '/UNION'][key])) for key in ('tp', 'fp'))
                                                       for family in FAMILIES) + ' |')
    lines += ['', f'Independent scalar OR truth-table audit passed for {audit["scalar_or_known_bits"]:,} known event checks across '
        f'{audit["scalar_or_checks"]} cohort/profile/union combinations. '
        f'{audit["sealed_individual_count_rows_equal"]:,} individual count rows exactly reproduce sealed MZ51 scoring; '
        f'{audit["frozen_MZ50_arrays_equal"]} original MZ50 arrays match. '
        'All MZ37 positives, 1280 source pairs, MZ36\'s400 attempts and80 UNKNOWN bits remain intact.', '',
        'Interpretation: the matched unions measure composition and negative-source coverage separately from a generic OR benefit. '
        'Any retained gain is a challenger result on these consumed controlled sources. A remaining false-alert cost prevents a no-cost or default-promotion claim. '
        'The experiment supplies no fresh-confirmation, natural-scene, calibrated-hardware or safety evidence.', '',
        f'Evidence: `{task}/score-v1/result.json`, `audit.json`, and the final `receipt.json`. '
        'The receipt binds this report, the brief, implementation, source receipts and saved outputs. '
        'No source data, previous score, cutoff or MZ51 gate was changed.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    run(parser.parse_args().task)
