"""Independent CPU scoring for matched MZ54 raster readouts and fixed references.

Run only after the registered runner has a PASS receipt and execution is
authorized. Reads saved predictions and MZ52's compact count labels; never
opens native depth, RGB, dense features or a model checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mz15_train import cutoff_zero_added
from mz40_evaluate import metrics, paired
from mz47_score import scalar_check


COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
ARMS = ('CROP_RASTER', 'FULL_RASTER', 'FULL_CROP_ONLY')
KEYS = tuple('MZ54/' + arm for arm in ARMS)
UNIONS = ('MZ50/UNION', 'NEW_NEG/UNION', 'OLD_NEG/UNION')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


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


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def static_geometry(p):
    crop, coverage, rays = (p['mz54/' + key] for key in ('crop_mask', 'sensor_coverage', 'rays'))
    assert crop.shape == coverage.shape == (45, 80) and crop.dtype == coverage.dtype == np.bool_
    assert rays.shape == (45, 80, 3) and rays.dtype == np.float32 and np.isfinite(rays).all()
    row, col = np.indices((45, 80), dtype=np.float64)
    expected_crop = (row >= 9) & (row <= 35) & (col >= 26) & (col <= 53)
    np.testing.assert_array_equal(crop, expected_crop)
    assert int(crop.sum()) == 756
    u, v = col * 8 + 3.5, row * 8 + 3.5
    focal = 320. / np.tan(np.deg2rad(50.))
    right, up = (u - 319.5) / focal, (179.5 - v) / focal
    expected_coverage = (np.abs(np.rad2deg(np.arctan(right))) <= 22.5) & (np.abs(np.rad2deg(np.arctan(up))) <= 22.5)
    np.testing.assert_array_equal(coverage, expected_coverage)
    expected_rays = np.stack([np.ones_like(right), right, up], -1)
    expected_rays /= np.linalg.norm(expected_rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(rays, expected_rays.astype(np.float32), rtol=0, atol=1e-7)
    return crop.flatten(), coverage.flatten(), dict(crop_cells=756, full_cells=3600,
        sensor_covered_centers=int(coverage.sum()), ray_max_abs=float(np.abs(rays - expected_rays).max()),
        ray_check_atol=1e-7, raster_shape=[45, 80], flatten_order='row-major',
        below_original_crop='winner center y>=292, equivalently raster row>=37',
        partial_bottom_boundary='row36 spans y288..295 and is excluded from matched crop; do not call it wholly below the original crop',
        legacy_control_difference='756 contained raster blocks are not the legacy 3136 angular samples')


def run(task):
    started = time.perf_counter()
    task = task.resolve(strict=True)
    source, out = task / 'run-v1', task / 'score-v1'
    report_path = Path(__file__).with_name('MZ54_FULL_RGB_RESULTS_20260911.md')
    assert not out.exists() and not report_path.exists(), 'Do not overwrite a sealed score/report'
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        assert expected is None or digest == expected, f'Input hash mismatch: {path}'
        inputs[str(path)] = digest
        return path

    receipt = read(bind(source / 'receipt.json'))
    assert receipt['status'] == 'PASS'
    assert receipt['steps_per_arm'] == 600 and receipt['total_steps'] == 1200
    assert set(receipt['fits']) == set(ARMS[:2])
    assert all(receipt['fits'][arm]['steps'] == 600 for arm in ARMS[:2])
    assert receipt['exact_initial_weights'] and receipt['trainable_parameters'] == 10708
    assert set(receipt['initial_parameter_sha256']) == set(ARMS[:2])
    assert len(set(receipt['initial_parameter_sha256'].values())) == 1
    assert receipt['new_cutoffs'] == 2 and receipt['threshold_searches'] == 0
    assert receipt['new_baseline_inferences'] == receipt['native_depth_reads'] == 0
    assert receipt['source_unknown_preserved'] and not receipt['permanent_dense_cache']
    assert receipt['crop_candidate_cells'] == 756 and receipt['full_candidate_cells'] == 3600

    def bound_input(suffix):
        matches = [(Path(path), digest) for path, digest in receipt['inputs'].items()
                   if path.replace('\\', '/').endswith(suffix)]
        assert len(matches) == 1, f'Expected one bound input: {suffix}'
        return bind(*matches[0])

    for name in ('MZ54_FULL_RGB_20260911.md', 'mz54_full_rgb.py',
                 'mz54_full_rgb_model.py', 'mz54_full_rgb_source.py'):
        bound_input('nearfield/' + name)
    for name in ('receipt.json', 'source-index.json'):
        bound_input('mz52-full-frame-supervision-20260911/' + name)

    required = ('predictions.npz', 'groups.json', 'schedule.npz',
                'CROP_RASTER-cutoff.npy', 'FULL_RASTER-cutoff.npy')
    for name in required:
        bind(source / name, receipt['outputs'][name])
    assert set(name for name in receipt['outputs'] if name.endswith('-cutoff.npy')) == set(required[-2:])
    p = arrays(source / 'predictions.npz')
    grouping = read(source / 'groups.json')
    records = grouping['records']
    groups = {key: np.array(value, dtype=np.int64) for key, value in grouping['groups'].items()}
    assert {key: len(value) for key, value in groups.items()} == dict(fit=1280, calibration=256, heldout_site=640, nonfit_family=384)
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(2560))
    assert len({r['frame_id'] for r in records}) == 2560
    np.testing.assert_array_equal(p['mz48/frame_ids'], [r['frame_id'] for r in records])
    pair_owner = {}
    for group, ids in groups.items():
        for i in ids:
            assert pair_owner.setdefault(records[i]['pair_id'], group) == group
    assert len(pair_owner) == 1280

    prior_root = task.parent / 'mz51-training-coverage-20260911'
    prior_run = read(bound_input('mz51-training-coverage-20260911/run-v1/receipt.json'))
    prior_scoring = read(bound_input('mz51-training-coverage-20260911/score-v1/receipt.json'))
    assert prior_run['status'] == prior_scoring['status'] == 'PASS'
    prior_score = read(bind(prior_root / 'score-v1/result.json', prior_scoring['outputs']['result.json']))
    prior_audit = read(bind(prior_root / 'score-v1/audit.json', prior_scoring['outputs']['audit.json']))
    assert prior_audit['schedule']['schedule_exact'] and prior_audit['schedule']['old_train_disjoint_from_three_DEV_cohorts']
    for arm in ('NEW_NEG', 'OLD_NEG'):
        assert prior_audit['schedule']['arms'][arm]['negative_query_truth_verified']
    prior_prediction = bound_input('mz51-training-coverage-20260911/run-v1/predictions.npz')
    assert sha(prior_prediction) == prior_run['outputs']['predictions.npz']
    prior = arrays(prior_prediction)
    for key, value in prior.items():
        np.testing.assert_array_equal(p[key], value, err_msg='Original MZ51 array changed: ' + key)
    preserved_arrays = len(prior)
    del prior
    prior_groups = read(bind(prior_root / 'run-v1/groups.json', prior_run['outputs']['groups.json']))
    assert grouping == prior_groups
    old_schedule = arrays(bind(prior_root / 'run-v1/schedule.npz', prior_run['outputs']['schedule.npz']))
    schedule = arrays(source / 'schedule.npz')
    assert set(schedule) == set(old_schedule)
    for key in schedule:
        np.testing.assert_array_equal(schedule[key], old_schedule[key], err_msg=key)
    assert schedule['shared'].shape == schedule['OLD_NEG'].shape == schedule['query'].shape == (600, 8)
    assert np.isin(schedule['shared'], groups['fit']).all()
    assert np.isin(schedule['OLD_NEG'], schedule['train_ids']).all()
    np.testing.assert_array_equal(schedule['fit_ids'], groups['fit'])
    assert len(schedule['train_ids']) == 7562 and len(np.unique(schedule['shared'])) == 1250
    groups['nonfit'] = np.sort(np.concatenate([groups['heldout_site'], groups['nonfit_family']]))

    union_root = task.parent / 'mz53-dual-readout-union-20260911/score-v1'
    union_receipt = read(bound_input('mz53-dual-readout-union-20260911/score-v1/receipt.json'))
    assert union_receipt['status'] == 'PASS'
    union_score = read(bind(union_root / 'result.json', union_receipt['outputs']['result.json']))
    crop, coverage, geometry_audit = static_geometry(p)
    label_receipt_path = bound_input('mz52-full-frame-supervision-20260911/verified-v1/receipt.json')
    label_receipt = read(label_receipt_path)
    assert label_receipt['status'] == 'PASS'
    label_path = bound_input('mz52-full-frame-supervision-20260911/verified-v1/fullframe-cells.npz')
    assert sha(label_path) == label_receipt['outputs']['fullframe-cells.npz']
    labels = arrays(label_path)
    np.testing.assert_array_equal(labels['frame_ids'], p['mz48/frame_ids'])
    np.testing.assert_array_equal(labels['global_indices'], np.arange(2560))
    count, valid_count = labels['fullframe_event_counts'], labels['valid_counts']
    assert count.shape == (2560, 45, 80, 4) and valid_count.shape == (2560, 45, 80)
    assert count.dtype == valid_count.dtype == np.uint8 and count.max() <= 64 and valid_count.max() <= 64
    assert (count <= valid_count[..., None]).all()
    np.testing.assert_array_equal(count.sum((1, 2)) >= 3, p['mz48/truth'])
    native = (count > 0).reshape(2560, 3600, 4)
    local_known = (valid_count > 0).reshape(2560, 3600)
    del labels, count, valid_count

    cuts = {}
    for arm in ARMS[:2]:
        def collect(key):
            return np.concatenate([p['DEV/DROP_CLOSE/' + key], p['mz48/DROP_CLOSE/' + key][groups['calibration']]])
        truth = np.concatenate([p['DEV/truth'], p['mz48/truth'][groups['calibration']]])
        known = np.concatenate([p['DEV/known'], p['mz48/known'][groups['calibration']]])
        assert truth.shape == (1256, 4) and known.all()
        key = 'MZ54/' + arm
        saved = np.load(source / (arm + '-cutoff.npy'), allow_pickle=False)
        recomputed = cutoff_zero_added(collect(key + '/raw'), collect(key + '/support'), collect('MZ37'), truth)
        np.testing.assert_array_equal(saved, recomputed)
        np.testing.assert_array_equal(saved, receipt['fits'][arm]['cutoff'])
        assert not (((collect(key + '/candidate') >= 0) & (collect('MZ37') < 0)) & ~truth).any()
        cuts[arm] = saved
    cuts['FULL_CROP_ONLY'] = cuts['FULL_RASTER']
    individual = tuple(prior_score['methods'])
    references = individual + UNIONS
    methods = references + tuple(key + '/candidate' for key in KEYS)
    scores, changes, attribution, contexts, cases, cached = {}, {}, {}, {}, {}, {}
    scalar_bits, reference_rows, native_values = 0, 0, 0

    def compare(a, truth, known):
        result = {key: {reference: paired(a[key + '/candidate'], a[reference], truth, known)
                       for reference in methods if reference != key + '/candidate'} for key in KEYS}
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
            np.testing.assert_array_equal(p['mz36/attempted_frame_ids'][admitted], p['mz36/frame_ids'])
            np.testing.assert_array_equal(truth[admitted], p['mz36/truth'])
        cohort_groups = {'all': np.arange(len(truth))}
        if cohort == 'mz48':
            cohort_groups.update(groups)
            for field in ('family', 'site_id', 'relation', 'range', 'setting', 'support_context'):
                for value in sorted({str(r[field]) for r in records}):
                    cohort_groups[field + '/' + value] = np.array([i for i, r in enumerate(records) if str(r[field]) == value])
        scores[cohort], changes[cohort] = {}, {}
        for profile in PROFILES:
            prefix = cohort + '/' + profile + '/'
            a = {name: p[prefix + name] for name in individual}
            for family in ('MZ50', 'NEW_NEG', 'OLD_NEG'):
                lead = '' if family == 'MZ50' else family + '/'
                a[family + '/UNION'] = np.maximum(a[lead + 'OPEN/candidate'], a[lead + 'GATED/candidate'])
            for arm, key in zip(ARMS, KEYS):
                raw, support, winner, candidate = (p[prefix + key + '/' + name] for name in ('raw', 'support', 'winner', 'candidate'))
                assert raw.shape == support.shape == winner.shape == candidate.shape == p[cohort + '/truth'].shape
                assert support.dtype == np.bool_ and support.all() and np.isfinite(raw).all()
                assert winner.dtype.kind in 'iu' and winner.min() >= 0 and winner.max() < 3600
                if arm != 'FULL_RASTER':
                    assert crop[winner].all()
                margin = raw.astype(float) - cuts[arm]
                accepted = (a['MZ37'] < 0) & support & (margin >= 0)
                np.testing.assert_array_equal(candidate, np.where(accepted, margin, a['MZ37']))
                a[key + '/candidate'] = candidate
            full = p[prefix + 'MZ54/FULL_RASTER/raw']
            limited = p[prefix + 'MZ54/FULL_CROP_ONLY/raw']
            assert (full >= limited).all()
            assert not ((a['MZ54/FULL_CROP_ONLY/candidate'] >= 0) & (a['MZ54/FULL_RASTER/candidate'] < 0)).any()
            if cohort == 'mz36':
                expanded = {}
                for name, value in a.items():
                    expanded[name] = np.full((400, 4), np.nan)
                    expanded[name][admitted] = value
                a = expanded
            scores[cohort][profile], changes[cohort][profile] = {}, {}
            for group, ids in cohort_groups.items():
                aa = {name: value[ids] for name, value in a.items()}
                evaluated = {name: metrics(value, truth[ids], known[ids]) for name, value in aa.items()}
                for name in references:
                    reference = prior_score if name in individual else union_score
                    assert evaluated[name] == reference['conditions'][cohort][profile][group][name], (cohort, profile, group, name)
                    reference_rows += 1
                scores[cohort][profile][group] = evaluated
                changes[cohort][profile][group] = compare(aa, truth[ids], known[ids])
            for key in KEYS:
                scalar_bits += scalar_check(a[key + '/candidate'], truth, known, scores[cohort][profile]['all'][key + '/candidate'])
            cached.setdefault(profile, {})[cohort] = (a, truth, known)
            if cohort == 'mz48':
                winning, winning_known, winner_indices = {}, {}, {}
                for key in KEYS:
                    winner = p[prefix + key + '/winner']
                    expected = np.take_along_axis(native, winner[:, None, :], 1)[:, 0]
                    expected &= p[prefix + key + '/support']
                    np.testing.assert_array_equal(expected, p[prefix + key + '/winning_native'])
                    native_values += expected.size
                    winning[key], winner_indices[key] = expected, winner
                    winning_known[key] = np.take_along_axis(local_known, winner, 1)
                attribution[profile] = {}
                for group, ids in groups.items():
                    attribution[profile][group] = {}
                    for key in KEYS:
                        winner = winner_indices[key]
                        row = winner // 80
                        added = (a[key + '/candidate'] >= 0) & (a['MZ37'] < 0) & truth & known
                        masks = dict(added_tp=added, native_winner_tp=added & winning[key],
                            known_winner_tp=added & winning_known[key],
                            outside_sensor_tp=added & ~coverage[winner],
                            native_outside_sensor_tp=added & ~coverage[winner] & winning[key],
                            outside_crop_tp=added & ~crop[winner],
                            native_outside_crop_tp=added & ~crop[winner] & winning[key],
                            below_original_crop_tp=added & (row >= 37),
                            native_below_original_crop_tp=added & (row >= 37) & winning[key],
                            bottom_boundary_block_tp=added & (row == 36))
                        attribution[profile][group][key] = {name: mask[ids].sum(0).tolist() for name, mask in masks.items()}
                    full_key = 'MZ54/FULL_RASTER'
                    full_winner = winner_indices[full_key]
                    for reference in ('MZ54/CROP_RASTER/candidate', 'MZ54/FULL_CROP_ONLY/candidate', 'OLD_NEG/UNION'):
                        recovered = (a[full_key + '/candidate'] >= 0) & (a[reference] < 0) & truth & known
                        masks = dict(tp_gained=recovered, winning_native=recovered & winning[full_key],
                            outside_sensor=recovered & ~coverage[full_winner],
                            native_outside_sensor=recovered & ~coverage[full_winner] & winning[full_key],
                            outside_crop=recovered & ~crop[full_winner],
                            below_original_crop=recovered & (full_winner // 80 >= 37),
                            native_below_original_crop=recovered & (full_winner // 80 >= 37) & winning[full_key],
                            bottom_boundary_block=recovered & (full_winner // 80 == 36))
                        attribution[profile][group]['FULL_vs_' + reference] = {name: mask[ids].sum(0).tolist() for name, mask in masks.items()}
                pair_map = {}
                for i, record in enumerate(records):
                    pair_map.setdefault(record['pair_id'], []).append(i)
                pair_ids = np.array(list(pair_map.values()))
                assert pair_ids.shape == (1280, 2)
                np.testing.assert_array_equal(truth[pair_ids[:, 0]], truth[pair_ids[:, 1]])
                contexts[profile] = {name: ((value[pair_ids[:, 0]] >= 0) != (value[pair_ids[:, 1]] >= 0)).sum(0).tolist()
                                     for name, value in a.items()}
                ids = groups['nonfit']
                changed = ((a['MZ54/FULL_RASTER/candidate'] >= 0) != (a['MZ54/CROP_RASTER/candidate'] >= 0)) & known
                cases[profile] = []
                for local_i, q in zip(*np.where(changed[ids])):
                    i = int(ids[local_i]); winner = int(winner_indices['MZ54/FULL_RASTER'][i, q])
                    cases[profile].append(dict(index=i, frame_id=records[i]['frame_id'], query=QUERIES[q],
                        truth=bool(truth[i, q]), full_positive=bool(a['MZ54/FULL_RASTER/candidate'][i, q] >= 0),
                        crop_positive=bool(a['MZ54/CROP_RASTER/candidate'][i, q] >= 0),
                        full_winner=winner, winner_row=winner // 80, winner_col=winner % 80,
                        winner_native=bool(winning['MZ54/FULL_RASTER'][i, q]), winner_known=bool(winning_known['MZ54/FULL_RASTER'][i, q]),
                        outside_sensor=bool(not coverage[winner]), outside_crop=bool(not crop[winner]),
                        below_original_crop=bool(winner // 80 >= 37),
                        **{name: records[i][name] for name in ('family', 'range', 'site_id', 'support_context', 'pair_id')}))

    aggregate, aggregate_changes = {}, {}
    for profile in PROFILES:
        data = cached[profile]
        legacy = [(data[name], np.arange(len(data[name][1]))) for name in ('relation10000', 'distance5000', 'rich', 'mz36')]
        selections = dict(legacy_noncal=legacy, mz48_fit=[(data['mz48'], groups['fit'])], mz48_nonfit=[(data['mz48'], groups['nonfit'])])
        selections['all_noncal'] = legacy + selections['mz48_fit'] + selections['mz48_nonfit']
        aggregate[profile], aggregate_changes[profile] = {}, {}
        for group, selected in selections.items():
            truth = np.concatenate([y[ids] for (_, y, _), ids in selected])
            known = np.concatenate([k[ids] for (_, _, k), ids in selected])
            a = {name: np.concatenate([values[name][ids] for (values, _, _), ids in selected]) for name in methods}
            evaluated = {name: metrics(value, truth, known) for name, value in a.items()}
            for name in references:
                reference = prior_score if name in individual else union_score
                assert evaluated[name] == reference['aggregates'][profile][group][name]
                reference_rows += 1
            aggregate[profile][group] = evaluated
            aggregate_changes[profile][group] = compare(a, truth, known)
    drop = aggregate['DROP_CLOSE']
    f, c = (drop['mz48_nonfit']['MZ54/' + arm + '/candidate'] for arm in ('FULL_RASTER', 'CROP_RASTER'))
    lf, lc = (drop['legacy_noncal']['MZ54/' + arm + '/candidate'] for arm in ('FULL_RASTER', 'CROP_RASTER'))
    gate = dict(full_nonfit_body_near_tp=f['tp'][0], crop_nonfit_body_near_tp=c['tp'][0],
        full_nonfit_fp=sum(f['fp']), crop_nonfit_fp=sum(c['fp']), full_legacy_fp=sum(lf['fp']), crop_legacy_fp=sum(lc['fp']),
        body_near_tp_improved=f['tp'][0] > c['tp'][0], nonfit_fp_not_higher=sum(f['fp']) <= sum(c['fp']),
        legacy_fp_not_higher=sum(lf['fp']) <= sum(lc['fp']))
    gate['passes_primary'] = all(gate[name] for name in ('body_near_tp_improved', 'nonfit_fp_not_higher', 'legacy_fp_not_higher'))
    result = dict(status='PASS', primary_condition='DROP_CLOSE', event_order=list(QUERIES), methods=list(methods),
        conditions=scores, comparisons=changes, aggregates=aggregate, aggregate_comparisons=aggregate_changes,
        attribution=attribution, nonfit_full_crop_disagreements=cases, context_pair_changed_decisions=contexts,
        primary_gate=gate, existing_cutoffs={name: cuts[name].tolist() for name in ARMS[:2]},
        FULL_CROP_ONLY_cutoff_source='FULL_RASTER-cutoff.npy; same fitted full field, restricted pool only',
        geometry=geometry_audit, aggregation_scope=prior_score['aggregation_scope'],
        interpretation='Matched full-field RGB plus full-field supervision is a combined intervention. The 756 raster crop blocks are not legacy 3136 angular samples. FULL_CROP_ONLY retains full input/context/weights and is not an identical-input crop ablation.',
        scope='Controlled Development on consumed sources. A BODY_NEAR gate success is not a claim of gains for other queries, profiles, natural scenes or calibrated hardware.')
    audit = dict(status='PASS', preserved_MZ51_arrays=preserved_arrays, sealed_reference_count_rows_equal=reference_rows,
        scalar_known_bits=scalar_bits, scalar_new_readouts=3, profiles=3, mz36_attempts=400, mz36_unknown_bits=80,
        source_frames=2560, pairs_intact=1280, exact_MZ51_schedule_arrays=len(schedule), shared_unique_frames=1250,
        matched_fit_steps_verified=[600, 600], equal_initial_parameter_hashes=True, trainable_parameters=10708,
        schedule_membership=True, negative_query_truth_authority='Sealed MZ51 scalar audit plus byte-value identical OLD_NEG/query/train_ids schedules',
        cutoffs_recomputed_for_parity=2, new_cutoffs=0, FULL_CROP_ONLY_reuses_FULL_cutoff=True,
        FULL_CROP_ONLY_positive_subset_of_FULL=True, prior_MZ37_positive_retention=True,
        winning_native_independently_checked_values=native_values, fullframe_label_file_bytes=label_path.stat().st_size,
        native_winner_authority='Independent indexed lookup in existing MZ52 count arrays; no new native-depth derivation',
        geometry=geometry_audit, all_noncal_includes_fit1280_once=True,
        new_fits=0, new_inference_frames=0, native_depth_reads=0, dense_feature_reads=0, RGB_reads=0, checkpoint_loads=0)
    for name in ('MZ54_FULL_RGB_20260911.md', 'mz54_score.py', 'mz15_train.py', 'mz40_evaluate.py', 'mz47_score.py'):
        bind(Path(__file__).with_name(name))
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest, path
    out.mkdir(exist_ok=False)
    write(out / 'result.json', result)
    write(out / 'audit.json', audit)
    report(report_path, task, result, audit)
    final = dict(status='PASS', inputs=inputs, code_sha256=sha(Path(__file__)),
        outputs={name: sha(out / name) for name in ('result.json', 'audit.json')},
        report=dict(path=str(report_path), sha256=sha(report_path)), seconds=time.perf_counter() - started,
        backend='CPU saved scores and compact MZ52 label lookup; TASK_NOT_GPU_SUITABLE',
        new_fits=0, new_inference_frames=0, new_cutoffs=0, native_depth_reads=0,
        scientific_binding='FINAL; report, results, audit and code sealed before receipt')
    write(out / 'receipt.json', final)
    print('PRIMARY_GATE', gate)
    print('DROP_TOTALS', {group: {name: [sum(values[key]) for key in ('tp', 'fp', 'fn')]
        for name, values in names.items() if name.startswith('MZ54/') or name == 'OLD_NEG/UNION'} for group, names in drop.items()})
    print('SEALED', dict(seconds=final['seconds'], scalar_known_bits=scalar_bits, winning_native_values=native_values,
                         code_sha256=final['code_sha256'], report_sha256=final['report']['sha256'], outputs=final['outputs']))


def report(path, task, result, audit):
    gate = result['primary_gate']
    arms = tuple('MZ54/' + arm + '/candidate' for arm in ARMS)
    lines = ['# MZ54: matched full-RGB raster result', '',
        f'Primary DROP_CLOSE gate: **{"PASS" if gate["passes_primary"] else "FAIL"}**. '
        f'Nonfit BODY_NEAR TP is FULL {gate["full_nonfit_body_near_tp"]} versus matched CROP {gate["crop_nonfit_body_near_tp"]}; '
        f'nonfit total FP is {gate["full_nonfit_fp"]}/{gate["crop_nonfit_fp"]}, '
        f'old noncalibration FP is {gate["full_legacy_fp"]}/{gate["crop_legacy_fp"]}. '
        'The gate tests this query and FP totals; it does not imply improvements in every query or profile.', '',
        '| DROP_CLOSE group | matched CROP TP/FP/FN | FULL TP/FP/FN | FULL_CROP_ONLY TP/FP/FN | fixed MZ53 OLD_NEG union TP/FP/FN |',
        '| --- | ---: | ---: | ---: | ---: |']
    for group, values in result['aggregates']['DROP_CLOSE'].items():
        names = arms + ('OLD_NEG/UNION',)
        cells = ['/'.join(str(sum(values[name][key])) for key in ('tp', 'fp', 'fn')) for name in names]
        lines.append(f'| {group} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Old noncalibration excludes DEV. New nonfit is 640 heldout-site + 384 withheld-family. '
        'all_noncal includes fit 1280 once and is not wholly held-out evidence.', '',
        '| nonfit DROP_CLOSE query | CROP TP/FP/FN | FULL TP/FP/FN | FULL_CROP_ONLY TP/FP/FN | fixed OLD_NEG union TP/FP/FN |',
        '| --- | ---: | ---: | ---: | ---: |']
    values = result['aggregates']['DROP_CLOSE']['mz48_nonfit']
    for q, query in enumerate(QUERIES):
        lines.append(f'| {query} | ' + ' | '.join('/'.join(str(values[name][key][q]) for key in ('tp', 'fp', 'fn'))
                                                for name in arms + ('OLD_NEG/UNION',)) + ' |')
    lines += ['', '| profile nonfit | CROP total TP/FP | FULL total TP/FP | FULL_CROP_ONLY total TP/FP | fixed OLD_NEG union total TP/FP |',
        '| --- | ---: | ---: | ---: | ---: |']
    for profile in PROFILES:
        values = result['aggregates'][profile]['mz48_nonfit']
        lines.append(f'| {profile} | ' + ' | '.join('/'.join(str(sum(values[name][key])) for key in ('tp', 'fp'))
                                                 for name in arms + ('OLD_NEG/UNION',)) + ' |')
    lines += ['', '| DROP_CLOSE FULL versus | group | TP gained/lost | FP added/removed | exact frames gained/lost |',
        '| --- | --- | ---: | ---: | ---: |']
    for group in ('legacy_noncal', 'mz48_nonfit', 'mz48_fit'):
        for reference in ('MZ54/CROP_RASTER/candidate', 'MZ54/FULL_CROP_ONLY/candidate', 'OLD_NEG/UNION'):
            row = result['aggregate_comparisons']['DROP_CLOSE'][group]['MZ54/FULL_RASTER'][reference]
            lines.append(f'| {reference} | {group} | {sum(row["tp_gained"])}/{sum(row["tp_lost"])} | '
                         f'{sum(row["fp_added"])}/{sum(row["fp_removed"])} | {row["exact_gained"]}/{row["exact_lost"]} |')
    attribution = result['attribution']['DROP_CLOSE']['nonfit']['FULL_vs_MZ54/CROP_RASTER/candidate']
    lines += ['', '| FULL recovered versus matched CROP | TP | native winner | native outside 45 degrees | native below original crop | bottom-boundary block |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for q, query in enumerate(QUERIES):
        lines.append(f'| {query} | ' + ' | '.join(str(attribution[key][q]) for key in
            ('tp_gained', 'winning_native', 'native_outside_sensor', 'native_below_original_crop', 'bottom_boundary_block')) + ' |')
    lines += ['', 'Below the original crop means winning center y>=292 (raster row>=37). '
        'Row 36 overlaps the crop boundary and is reported separately. Saved native-winning flags were independently '
        'looked up in the existing MZ52 uint8 count arrays; no native depth was opened. Correct events without a native winning flag are not automatically local evidence.', '',
        f'Validation: {audit["preserved_MZ51_arrays"]} prior MZ51 arrays exactly equal; '
        f'{audit["sealed_reference_count_rows_equal"]} individual/union count rows match sealed MZ51/MZ53 results; '
        f'{audit["scalar_known_bits"]:,} independent scalar known-bit checks; '
        f'{audit["winning_native_independently_checked_values"]:,} native-winning lookups. '
        'The exact prior schedule, train membership, 1250 shared unique fit frames, 1280 pairs and 2560-frame partition are preserved. '
        'All MZ37 positives and MZ36\'s 400 attempts/80 UNKNOWN remain. Only the two existing cutoffs were recomputed for parity; FULL_CROP_ONLY reuses FULL cutoff.', '',
        'Attribution limit: FULL changes local RGB field and local-supervision field together. '
        'The matched control has 756 contained raster blocks, not the legacy 3136 angular samples. '
        'FULL_CROP_ONLY keeps full image features, context and fitted weights; it diagnoses pooling support and is not an identical-input crop ablation. '
        'Query/profile losses remain costs even if the narrow BODY_NEAR gate passes. No natural-scene, physical VL53L8CX calibration, clearance or default-App claim follows.', '',
        f'Evidence: `{task}/score-v1/result.json`, `audit.json` and final `receipt.json`. '
        'The final receipt binds this report, implementation, brief, source receipts and saved outputs.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    run(parser.parse_args().task)
