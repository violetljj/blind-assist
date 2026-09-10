"""Independent CPU scoring of sealed MZ58 outputs; no fit or source decoding.

Run only after root GO: python mz58_score.py --task <MZ58 task>
Synthetic checks (no experiment input access): python mz58_score.py --self-test
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
MODES = ('LOCAL_ONLY', 'GLOBAL_ANCHOR', 'GLOBAL_SUPPRESSED')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
BASE = 'OLD_NEG/UNION'
LOCAL = ('OLD_NEG/OPEN', 'OLD_NEG/GATED') + tuple('MZ56/' + m for m in MODES)
UNIONS = tuple('MZ57/' + m + '/UNION' for m in MODES)
METHODS = ('MZ37', 'OLD_NEG/OPEN/candidate', 'OLD_NEG/GATED/candidate', BASE) + tuple(
    'MZ56/' + m + '/candidate' for m in MODES) + UNIONS
FIELDS = ('role', 'site_id', 'family', 'relation', 'range', 'setting', 'support_context')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as stream:
        return {key: stream[key] for key in stream.files}


def metrics(values, truth, known):
    pred = values >= 0
    complete = known.all(1)
    return dict(frames=len(truth), known=known.sum(0).tolist(), unknown=(~known).sum(0).tolist(),
                tp=(pred & truth & known).sum(0).tolist(), fp=(pred & ~truth & known).sum(0).tolist(),
                fn=(~pred & truth & known).sum(0).tolist(), tn=(~pred & ~truth & known).sum(0).tolist(),
                complete_frames=int(complete.sum()), exact_frames=int(((pred == truth).all(1) & complete).sum()))


def scalar_metrics(values, truth, known):
    counts = {key: [0] * 4 for key in ('known', 'unknown', 'tp', 'fp', 'fn', 'tn')}
    complete = exact = 0
    for i in range(len(truth)):
        row_known, row_exact = True, True
        for q in range(4):
            if not bool(known[i, q]):
                counts['unknown'][q] += 1
                row_known = False
                continue
            counts['known'][q] += 1
            decision, actual = float(values[i, q]) >= 0, bool(truth[i, q])
            counts['tp' if decision and actual else 'fp' if decision else 'fn' if actual else 'tn'][q] += 1
            row_exact = row_exact and decision == actual
        complete += int(row_known)
        exact += int(row_known and row_exact)
    return dict(frames=len(truth), **counts, complete_frames=complete, exact_frames=exact)


def paired(after, before, truth, known):
    a, b = after >= 0, before >= 0
    complete = known.all(1)
    ae, be = (a == truth).all(1) & complete, (b == truth).all(1) & complete
    return dict(tp_gained=(a & ~b & truth & known).sum(0).tolist(),
                tp_lost=(~a & b & truth & known).sum(0).tolist(),
                fp_added=(a & ~b & ~truth & known).sum(0).tolist(),
                fp_removed=(~a & b & ~truth & known).sum(0).tolist(),
                exact_gained=int((ae & ~be).sum()), exact_lost=int((~ae & be).sum()))


def check_packets(original, saved, profile):
    ranges, valid = original['ranges'], original['valid']
    rr, vv = ranges.copy(), valid.copy()
    close = 0
    for i in range(len(rr)):
        for z in range(64):
            if bool(valid[i, z, 0]) and bool(valid[i, z, 1]) and float(ranges[i, z, 1]) - float(ranges[i, z, 0]) < .6:
                close += 1
                if profile == 'MERGE_CLOSE':
                    rr[i, z, 0] = np.float32((float(ranges[i, z, 0]) + float(ranges[i, z, 1])) / 2)
                    rr[i, z, 1], vv[i, z, 1] = 0, False
                elif profile == 'DROP_CLOSE':
                    rr[i, z], vv[i, z] = 0, False
    np.testing.assert_array_equal(saved['ranges'], rr)
    np.testing.assert_array_equal(saved['valid'], vv)
    assert rr.dtype == np.float32 and vv.dtype == np.bool_
    assert rr.shape == vv.shape == (len(rr), 64, 2) and np.isfinite(rr).all()
    assert (rr[~vv] == 0).all() and ((rr[vv] > 0) & (rr[vv] <= 4)).all()
    counts = vv.sum(2)
    return dict(original_unresolved_zones=close, valid_slots=int(vv.sum()),
                zone_return_counts=[int((counts == n).sum()) for n in range(3)],
                all_tof_missing_frames=int((~vv.any((1, 2))).sum()))


def check_geometry(p):
    crop, coverage = (p['geometry/' + k] for k in ('crop_mask', 'sensor_coverage'))
    row, col = np.indices((45, 80), dtype=float)
    expected_crop = (row >= 9) & (row <= 35) & (col >= 26) & (col <= 53)
    np.testing.assert_array_equal(crop, expected_crop)
    focal = 320 / np.tan(np.deg2rad(50))
    right, up = (col * 8 + 3.5 - 319.5) / focal, (179.5 - row * 8 - 3.5) / focal
    expected_coverage = (np.abs(np.rad2deg(np.arctan(right))) <= 22.5) & (np.abs(np.rad2deg(np.arctan(up))) <= 22.5)
    np.testing.assert_array_equal(coverage, expected_coverage)
    rays = np.stack([np.ones_like(right), right, up], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(p['geometry/rays'], rays, atol=1e-7, rtol=0)
    zr, zc = np.indices((8, 8), dtype=np.float32)
    angles = np.stack([-22.5 + (zc + .5) * 5.625, 22.5 - (zr + .5) * 5.625], -1).reshape(64, 2) / 22.5
    np.testing.assert_array_equal(p['geometry/zone_angles'], angles)
    cr, cc = np.indices((7, 7), dtype=float)
    az = np.deg2rad(-22.5 + (zc.reshape(64, 1) + ((cc + .5) / 7).reshape(1, 49)) * 5.625)
    el = np.deg2rad(22.5 - (zr.reshape(64, 1) + ((cr + .5) / 7).reshape(1, 49)) * 5.625)
    angular = np.stack([np.ones_like(az), np.tan(az), np.tan(el)], -1)
    angular /= np.linalg.norm(angular, axis=-1, keepdims=True)
    np.testing.assert_allclose(p['geometry/angular_rays'], angular, atol=2e-7, rtol=0)
    assert crop.dtype == coverage.dtype == np.bool_ and int(crop.sum()) == 756
    return crop.reshape(-1), coverage.reshape(-1)


def check_candidates(saved, cuts, truth, known):
    n = len(truth)
    baseline = saved['MZ37']
    assert baseline.shape == (n, 4) and np.isfinite(baseline).all()
    for key in LOCAL:
        raw, support, candidate, winner = (saved[key + '/' + k] for k in ('raw', 'support', 'candidate', 'winner'))
        assert raw.shape == support.shape == candidate.shape == winner.shape == (n, 4)
        assert support.dtype == np.bool_ and np.issubdtype(winner.dtype, np.integer)
        assert np.isfinite(raw).all() and np.isfinite(candidate).all()
        assert ((winner >= 0) & (winner < (3136 if key.startswith('OLD_NEG') else 3600))).all()
        expected = np.empty((n, 4), dtype=float)
        for i in range(n):
            for q in range(4):
                margin = float(raw[i, q]) - float(cuts[key][q])
                expected[i, q] = margin if float(baseline[i, q]) < 0 and bool(support[i, q]) and margin >= 0 else baseline[i, q]
        np.testing.assert_array_equal(candidate, expected)
        assert ((candidate >= 0) | (baseline < 0)).all()
    members = {BASE: ('OLD_NEG/OPEN/candidate', 'OLD_NEG/GATED/candidate')}
    members.update({union: (BASE, 'MZ56/' + mode + '/candidate') for mode, union in zip(MODES, UNIONS)})
    for key, (left, right) in members.items():
        np.testing.assert_array_equal(saved[key], np.maximum(saved[left], saved[right]))
        for i in range(n):
            for q in range(4):
                assert bool(saved[key][i, q] >= 0) == (bool(saved[left][i, q] >= 0) or bool(saved[right][i, q] >= 0))
    for method in METHODS:
        assert scalar_metrics(saved[method], truth, known) == metrics(saved[method], truth, known)
    available = saved['valid'].any((1, 2))
    for mode in MODES:
        key = 'MZ56/' + mode
        np.testing.assert_array_equal(saved[key + '/anchor_available'], available)
        vector = saved[key + '/anchor_vector']
        assert vector.shape == (n, 8) and np.isfinite(vector).all()
        assert (vector[~available] == 0).all()
        if mode != 'GLOBAL_ANCHOR':
            assert (vector == 0).all()
    if (~available).any():
        np.testing.assert_allclose(saved['MZ56/GLOBAL_ANCHOR/raw'][~available],
                                   saved['MZ56/GLOBAL_SUPPRESSED/raw'][~available], atol=2e-5, rtol=1e-6)
    return int(known.sum()) * len(METHODS)


def native_winners(saved, full, full_known, angular, angular_known):
    result = {}
    for key in LOCAL:
        presence, known = (angular, angular_known) if key.startswith('OLD_NEG') else (full, full_known)
        winner = saved[key + '/winner'].astype(np.int64)
        n = len(winner)
        native = presence[np.arange(n)[:, None], winner, np.arange(4)[None, :]]
        local_known = known[np.arange(n)[:, None], winner]
        for i in range(n):
            for q in range(4):
                w = int(winner[i, q])
                assert bool(native[i, q]) == bool(presence[i, w, q])
                assert bool(local_known[i, q]) == bool(known[i, w])
        assert not (native & ~local_known).any()
        result[key] = dict(native=native, known=local_known)
    return result


def aggregate(saved, truth, known, ids):
    values = {key: metrics(saved[key][ids], truth[ids], known[ids]) for key in METHODS}
    changes = {key: paired(saved[key][ids], saved[BASE][ids], truth[ids], known[ids]) for key in UNIONS}
    for row in changes.values():
        assert not any(row['tp_lost']) and not any(row['fp_removed'])
    return dict(methods=values, paired_vs_old_union=changes)


def score_arrays(p, packets, truth, known, records, full, full_known, angular, angular_known, cuts, pairs):
    crop, coverage = check_geometry(p)
    n = len(truth)
    groups = {'all': np.arange(n)}
    for field in FIELDS:
        for value in sorted({str(row[field]) for row in records}):
            groups[field + '/' + value] = np.array([i for i, row in enumerate(records) if str(row[field]) == value])
    results, attribution, cases, contexts, packet_stats = {}, {}, {}, {}, {}
    scalar_bits = 0
    for profile in PROFILES:
        lead = profile + '/'
        saved = {key[len(lead):]: value for key, value in p.items() if key.startswith(lead)}
        packet_stats[profile] = check_packets(packets, saved, profile)
        scalar_bits += check_candidates(saved, cuts, truth, known)
        winners = native_winners(saved, full, full_known, angular, angular_known)
        results[profile] = {group: aggregate(saved, truth, known, ids) for group, ids in groups.items()}
        attribution[profile], cases[profile], contexts[profile] = {}, {}, {}
        for key in LOCAL:
            added = (saved[key + '/candidate'] >= 0) & (saved['MZ37'] < 0) & truth & known
            hit = added & winners[key]['native'] & saved[key + '/support']
            attribution[profile][key] = dict(added_tp_over_mz37=added.sum(0).tolist(),
                added_tp_native_winner=hit.sum(0).tolist(), added_tp_unknown_winner=(added & ~winners[key]['known']).sum(0).tolist(),
                added_tp_known_non_native_winner=(added & winners[key]['known'] & ~winners[key]['native']).sum(0).tolist())
        for mode, union in zip(MODES, UNIONS):
            key = 'MZ56/' + mode
            win = saved[key + '/winner'].astype(np.int64)
            added = (saved[union] >= 0) & (saved[BASE] < 0) & known
            gained = added & truth
            native = winners[key]['native']
            assert (saved[key + '/support'][added]).all()
            assert (saved[key + '/candidate'][added] >= 0).all() and (saved['MZ37'][added] < 0).all()
            attribution[profile][union] = dict(added_tp=gained.sum(0).tolist(),
                native_winner_tp=(gained & native).sum(0).tolist(),
                native_outside_sensor_tp=(gained & native & ~coverage[win]).sum(0).tolist(),
                native_outside_crop_tp=(gained & native & ~crop[win]).sum(0).tolist(),
                native_below_original_crop_tp=(gained & native & (win // 80 >= 37)).sum(0).tolist(),
                unknown_winner_tp=(gained & ~winners[key]['known']).sum(0).tolist(),
                known_non_native_winner_tp=(gained & winners[key]['known'] & ~native).sum(0).tolist(),
                tp_without_any_gated_support=(gained & ~saved['OLD_NEG/GATED/support']).sum(0).tolist(),
                tp_without_positive_gated_candidate=(gained & (saved['OLD_NEG/GATED/candidate'] < 0)).sum(0).tolist())
            changes = []
            for i, q in zip(*np.where(added)):
                w = int(win[i, q])
                changes.append(dict(index=int(i), frame_id=str(p['frame_ids'][i]), query=QUERIES[q],
                    change='TP_GAIN' if truth[i, q] else 'FP_ADDED',
                    **{field: records[i][field] for field in FIELDS}, pair_id=records[i]['pair_id'],
                    winner=w, raster_row=w // 80, raster_col=w % 80,
                    native=bool(native[i, q]), local_known=bool(winners[key]['known'][i, q]),
                    in_sensor=bool(coverage[w]), in_crop=bool(crop[w]),
                    gated_support=bool(saved['OLD_NEG/GATED/support'][i, q]),
                    gated_candidate_positive=bool(saved['OLD_NEG/GATED/candidate'][i, q] >= 0),
                    raw=float(saved[key + '/raw'][i, q]), cutoff=float(cuts[key][q]),
                    baseline_score=float(saved[BASE][i, q]), candidate_score=float(saved[key + '/candidate'][i, q])))
            cases[profile][union] = changes
        results[profile]['global_vs_suppressed'] = paired(saved[UNIONS[1]], saved[UNIONS[2]], truth, known)
        for method in METHODS:
            rows = []
            for pair_id, ii in pairs.items():
                a, b = ii
                if not np.array_equal(saved[method][a] >= 0, saved[method][b] >= 0):
                    rows.append(dict(pair_id=pair_id, frame_ids=[str(p['frame_ids'][i]) for i in ii],
                                     contexts=[records[i]['support_context'] for i in ii],
                                     predictions=[(saved[method][i] >= 0).tolist() for i in ii],
                                     truth=truth[a].tolist(), known=[known[i].tolist() for i in ii]))
            contexts[profile][method] = dict(changed_pairs=len(rows), total_pairs=len(pairs), rows=rows)
    gates = {}
    for union in UNIONS:
        change = results['DROP_CLOSE']['all']['paired_vs_old_union'][union]
        gates[union] = dict(tp_gained=sum(change['tp_gained']), fp_added=sum(change['fp_added']),
                            passes_primary=sum(change['tp_gained']) > 0 and sum(change['fp_added']) == 0)
    return (dict(profiles=results, attribution=attribution, primary_gate=gates, packet_coverage=packet_stats),
            dict(union_added_events=cases, support_context_pairs=contexts), scalar_bits)


def run(task):
    task = task.resolve(strict=True)
    out, source_run = task / 'score-v1', task / 'run-v1'
    assert not out.exists(), 'Do not overwrite a sealed score or failed evidence'
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
        bind(Path(__file__).with_name('MZ58_DIVERSE_TRANSFER_20260911.md'))
        rr = read(bind(source_run / 'receipt.json'))
        assert rr['status'] == 'PASS' and rr['frames'] == 2560 and rr['fixed_checkpoints']
        assert tuple(rr['methods']) == METHODS and tuple(rr['profiles']) == PROFILES
        assert rr['training_steps'] == rr['new_cutoffs'] == rr['source_calibration_rows_used'] == 0
        assert not rr['evaluator_labels_read'] and rr['native_depth_reads'] == rr['old_cohort_replay_frames'] == 0
        assert rr['source_role'] == 'CONSUMED_DEVELOPMENT' and rr['existing_cutoff_vectors'] == 4
        p = load(bind(source_run / 'predictions.npz', rr['outputs']['predictions.npz']))
        source = Path(rr['source_task']).resolve(strict=True)
        index = read(bind(source / 'source-index.json', rr['source_index_sha256']))
        assert index['status'] == 'COMPLETE' and index['schema'] == 'mz55-diverse-source-v1'
        assert index['frames'] == 2560 and len(index['shards']) == 10 and index['source_role'] == 'CONSUMED_DEVELOPMENT'

        def combined(name):
            ref = index['combined'][name]
            path = (source / ref['path']).resolve(strict=True)
            assert path.is_relative_to(source), path
            return bind(path, ref['sha256'])

        sr = read(combined('receipt.json'))
        assert sr['status'] == 'PASS' and sr['original_sources_unchanged']
        for name in ('packets.npz', 'evaluator.npz', 'metadata.json', 'fullframe-cells.npz', 'angular-auxiliary.npz', 'result.json'):
            assert sr['outputs'][name] == index['combined'][name]['sha256']
        packets, evaluator = load(combined('packets.npz')), load(combined('evaluator.npz'))
        labels, aux = load(combined('fullframe-cells.npz')), load(combined('angular-auxiliary.npz'))
        meta, source_result = read(combined('metadata.json')), read(combined('result.json'))
        records, pairs = meta['records'], meta['pairs']
        assert len(records) == len(set(p['frame_ids'])) == len(p['frame_ids']) == 2560
        np.testing.assert_array_equal(p['frame_ids'], [row['frame_id'] for row in records])
        for obj in (packets, evaluator, labels, aux):
            np.testing.assert_array_equal(obj['frame_ids'], p['frame_ids'])
        np.testing.assert_array_equal(labels['global_indices'], np.arange(2560))
        np.testing.assert_array_equal([row['index'] for row in records], np.arange(2560))
        truth, known = evaluator['truth'], evaluator['known']
        assert truth.shape == known.shape == (2560, 4) and truth.dtype == known.dtype == np.bool_
        np.testing.assert_array_equal(known, np.repeat(np.array([row['source_valid'] for row in records], dtype=bool)[:, None], 4, 1))
        counts, valid_counts = labels['fullframe_event_counts'], labels['valid_counts']
        assert counts.shape == (2560, 45, 80, 4) and valid_counts.shape == (2560, 45, 80)
        assert counts.dtype == valid_counts.dtype == np.uint8 and (counts <= valid_counts[..., None]).all() and (valid_counts <= 64).all()
        np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, truth)
        assert int((~known).sum()) == source_result['unknown_query_bits']
        assert int((valid_counts == 0).sum()) == source_result['unknown_fullframe_cells']
        full, full_known = (counts > 0).reshape(2560, 3600, 4), (valid_counts > 0).reshape(2560, 3600)
        angular, angular_known = aux['cell_event_presence'], aux['cell_known']
        assert angular.shape == (2560, 64, 49, 4) and angular_known.shape == (2560, 64, 49)
        assert angular.dtype == angular_known.dtype == np.bool_ and not (angular & ~angular_known[..., None]).any()
        angular, angular_known = angular.reshape(2560, 3136, 4), angular_known.reshape(2560, 3136)
        group_counts = {field: dict(Counter(str(row[field]) for row in records)) for field in FIELDS}
        assert group_counts['role'] == dict(TRAIN_CANDIDATE=1600, CALIBRATION=320, HELDOUT_SITE=640)
        for field, kinds in (('site_id', 8), ('family', 4), ('relation', 4), ('range', 2), ('setting', 5), ('support_context', 2)):
            assert len(group_counts[field]) == kinds and set(group_counts[field].values()) == {2560 // kinds}
        combinations = [tuple(str(row[field]) for field in FIELDS[1:]) for row in records]
        assert len(set(combinations)) == 2560
        assert len(pairs) == 1280 and sorted(i for ii in pairs.values() for i in ii) == list(range(2560))
        for pair_id, ii in pairs.items():
            assert len(ii) == 2 and {records[i]['pair_id'] for i in ii} == {pair_id}
            assert len({records[i]['role'] for i in ii}) == 1 and len({records[i]['support_context'] for i in ii}) == 2
            np.testing.assert_array_equal(truth[ii[0]], truth[ii[1]])
            np.testing.assert_array_equal(counts[ii[0]], counts[ii[1]])
        cuts = {}
        for key in LOCAL:
            ref = rr['frozen']['cutoffs'][key]
            path = bind(source_run / ref['filename'], rr['outputs'][ref['filename']])
            assert sha(path) == ref['sha256']
            bind(ref['path'], ref['sha256'])
            cuts[key] = np.load(path, allow_pickle=False)
            assert cuts[key].shape == (4,) and np.isfinite(cuts[key]).all()
            np.testing.assert_array_equal(cuts[key], ref['values'])
        np.testing.assert_array_equal(cuts['MZ56/GLOBAL_ANCHOR'], cuts['MZ56/GLOBAL_SUPPRESSED'])
        for ref in rr['frozen']['checkpoints'].values():
            bind(ref['path'], ref['sha256'])
        model_sources = {}
        for path, digest in rr['inputs'].items():
            if path.endswith('.py') or path.endswith('MZ58_DIVERSE_TRANSFER_20260911.md'):
                model_sources[str(bind(path, digest))] = digest
        scored, events, scalar_bits = score_arrays(p, packets, truth, known, records, full, full_known,
                                                  angular, angular_known, cuts, pairs)
        result = dict(status='PASS', methods=METHODS, queries=QUERIES, frames=2560,
            source_role='CONSUMED_DEVELOPMENT', all_roles_descriptive=True, source_groups=group_counts,
            known_query_bits=known.sum(0).tolist(), unknown_query_bits=(~known).sum(0).tolist(),
            unknown_fullframe_cells=int((~full_known).sum()), unknown_angular_cells=int((~angular_known).sum()),
            source_intent_matching_frames=sum(bool(row['intent_matches']) for row in records),
            original_cutoffs={key: value.tolist() for key, value in cuts.items()},
            gate_definition='Each fixed union independently: DROP_CLOSE all 2560 TP gain > 0 and added FP = 0 versus OLD_NEG/UNION. No winner selection.',
            **scored, limitation='Consumed controlled Development, no MZ55 fitting/calibration or fresh confirmation. Native winners are indexed saved argmax outputs; complete fields are not saved to re-evaluate argmax. Coverage does not establish outside-object ranging. No natural-scene, hardware, clearance or safety claim.',
            deployment_cost='CPU saved-output composition adds no inference here. Deployment retains crop and full RGB feature/head paths; duplicate encoder work and Android latency/memory remain unmeasured.')
        audit = dict(status='PASS', scalar_known_decisions=scalar_bits, scalar_candidate_checks=2560 * 4 * len(LOCAL) * 3,
            scalar_union_checks=2560 * 4 * 4 * 3, independent_native_winner_lookups=2560 * 4 * len(LOCAL) * 3,
            unique_frames=2560, intact_source_pairs=1280, all_ten_methods_all_three_profiles=True,
            candidate_values_exact=True, original_cutoffs_exact=True, retained_MZ37_positives=True,
            packet_transform_scalar_parity=True, query_and_local_UNKNOWN_separate=True,
            winner_flatten_conventions=dict(OLD_NEG=[64, 49], MZ56=[45, 80]),
            no_dense_argmax_recomputation=True, native_labels_evaluator_only=True,
            training_steps=0, threshold_searches=0, new_cutoffs=0, model_inference_frames=0,
            RGB_reads=0, native_depth_reads=0, checkpoint_loads=0,
            all_missing_anchor_comparison_frames={profile: scored['packet_coverage'][profile]['all_tof_missing_frames'] for profile in PROFILES})
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(out / 'result.json', result)
        write(out / 'audit.json', audit)
        write(out / 'paired-events.json', events)
        report(out / 'report.md', result)
        write(out / 'receipt.json', dict(status='PASS', inputs=inputs, source_index_sha256=rr['source_index_sha256'],
            frozen=rr['frozen'], model_sources=model_sources, code_sha256=sha(__file__),
            outputs={name: sha(out / name) for name in ('result.json', 'audit.json', 'paired-events.json', 'report.md')},
            seconds=time.perf_counter() - started, backend='FROZEN_PROTOCOL_CPU_ONLY: NumPy/stdlib saved-output audit',
            source_arrays_only=True, new_inference_frames=0, new_fits=0, new_cutoffs=0,
            original_source_unchanged=True, scientific_binding='FINAL; do not edit sealed outputs'))
        print(json.dumps(dict(status='PASS', primary_gate=result['primary_gate'], seconds=time.perf_counter() - started)))
    except BaseException:
        write(out / 'failure.json', dict(status='FAIL', inputs=inputs, error=traceback.format_exc()))
        raise


def report(path, result):
    lines = ['# MZ58 fixed diverse-shape transfer', '',
             'All 2560 MZ55 frames, 10 methods, 3 fixed profiles. All source roles are descriptive; no new training or calibration.', '',
             '| DROP_CLOSE method | TP | FP | FN | Exact frames |', '| --- | ---: | ---: | ---: | ---: |']
    for key, row in result['profiles']['DROP_CLOSE']['all']['methods'].items():
        lines.append('| ' + key + ' | ' + ' | '.join(str(sum(row[k])) for k in ('tp', 'fp', 'fn')) + f' | {row["exact_frames"]} |')
    lines += ['', '| Fixed union versus OLD_NEG/UNION | TP gained | FP added | Gate |', '| --- | ---: | ---: | --- |']
    for key, gate in result['primary_gate'].items():
        lines.append(f'| {key} | {gate["tp_gained"]} | {gate["fp_added"]} | {"PASS" if gate["passes_primary"] else "FAIL"} |')
    lines += ['', result['gate_definition'], '',
        f'Query UNKNOWN bits: {sum(result["unknown_query_bits"])}; fullframe UNKNOWN cells: {result["unknown_fullframe_cells"]:,}; angular UNKNOWN cells: {result["unknown_angular_cells"]:,}. Query completeness does not imply local completeness.', '',
        'The result retains all roles, sites, families, relations, ranges, settings, support contexts and per-query counts. Actual native positives, including extra far events and intent mismatches, remain in their original denominators. Each changed union TP/FP has an exact frame/query, existing cutoff, native winner and sensor/crop coverage record in paired-events.json.', '',
        'GLOBAL versus SUPPRESSED uses the same learned GLOBAL weights and cutoff. A native winning cell is supporting localization evidence, not proof that the network used that surface causally. Inherited MZ37 positives are excluded from new-path gain attribution.', '',
        result['limitation'], '', result['deployment_cost']]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test():
    # Exercise thresholds at equality, UNKNOWN, inherited positives, no-support
    # rejection, close-return semantics and native winning-cell coordinates.
    truth = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=bool)
    known = np.ones_like(truth)
    known[1, 2] = False
    saved = {'MZ37': np.array([[1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]], dtype=float)}
    cuts = {key: np.array([.2, -.1, .3, .4]) for key in LOCAL}
    for j, key in enumerate(LOCAL):
        raw = np.array([[.0, -.2, .3, .5], [.5, .2, .7, -.1], [.1, .8, -.8, .2]]) + j * .01
        support = np.ones_like(truth)
        support[0, 3] = False
        saved[key + '/raw'], saved[key + '/support'] = raw, support
        saved[key + '/winner'] = np.full_like(truth, j, dtype=np.int16)
        margin = raw - cuts[key]
        saved[key + '/candidate'] = np.where((saved['MZ37'] < 0) & support & (margin >= 0), margin, saved['MZ37'])
    saved[BASE] = np.maximum(saved[LOCAL[0] + '/candidate'], saved[LOCAL[1] + '/candidate'])
    for mode, union in zip(MODES, UNIONS):
        saved[union] = np.maximum(saved[BASE], saved['MZ56/' + mode + '/candidate'])
    rr, vv = np.zeros((3, 64, 2), np.float32), np.zeros((3, 64, 2), bool)
    rr[:, 0] = [1, 1.5]
    vv[:, 0] = True
    rr[:, 1] = [1, 2]
    vv[:, 1] = True
    packets = dict(ranges=rr, valid=vv)
    saved.update(packets)
    for mode in MODES:
        saved['MZ56/' + mode + '/anchor_available'] = np.ones(3, bool)
        saved['MZ56/' + mode + '/anchor_vector'] = np.zeros((3, 8))
    assert check_candidates(saved, cuts, truth, known) == 110
    assert saved['OLD_NEG/OPEN/candidate'][0, 2] == 0
    assert saved['OLD_NEG/OPEN/candidate'][0, 3] == -1
    assert metrics(saved[BASE], truth, known) == scalar_metrics(saved[BASE], truth, known)
    for profile in PROFILES:
        ranges, valid = rr.copy(), vv.copy()
        if profile == 'MERGE_CLOSE':
            ranges[:, 0] = [1.25, 0]
            valid[:, 0] = [True, False]
        elif profile == 'DROP_CLOSE':
            ranges[:, 0], valid[:, 0] = 0, False
        assert check_packets(packets, dict(ranges=ranges, valid=valid), profile)['original_unresolved_zones'] == 3
    full, angular = np.zeros((3, 3600, 4), bool), np.zeros((3, 3136, 4), bool)
    full[:, 2, 0], angular[:, 1, 1] = True, True
    winners = native_winners(saved, full, np.ones((3, 3600), bool), angular, np.ones((3, 3136), bool))
    assert winners['MZ56/LOCAL_ONLY']['native'][:, 0].all()
    assert winners['OLD_NEG/GATED']['native'][:, 1].all()
    assert not winners['OLD_NEG/OPEN']['native'].any()
    damaged = dict(saved)
    damaged[BASE] = saved[BASE].copy()
    damaged[BASE][0, 0] = -1
    try:
        check_candidates(damaged, cuts, truth, known)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt union was accepted')
    print('PASS synthetic: scalar counts/UNKNOWN, zero-boundary candidates, support rejection, OR corruption rejection, three packet profiles, independent native winners')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert args.task is None, 'Synthetic checks never read an experiment'
        self_test()
    else:
        assert args.task is not None
        run(args.task)
