"""Independent v3 audit: no fitting, new data acquisition or eval-label reopen.

Protocol/calibration and post-eval cache stages have separate entry points.
Statistics and threshold selection are recomputed without importing the run's
training, selection or aggregation implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ARMS = ('native_perturbed', 'faro_rho015_ambient1', 'faro_rho060_ambient1',
        'faro_rho030_ambient3', 'faro_rho030_ambient1', 'faro_rho030_ambient10')
BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
GROUPS = ('both', 'tof_only', 'rgb_only', 'neither')
FEATURES = ('tof_margin16_m', 'tof_margin16_missing', 'tof_support_pixels',
            'tof_unknown_fraction', 'tof_before_fraction', 'tof_inside_fraction',
            'tof_after_fraction', 'tof_median_entry_offset_m',
            'tof_median_exit_offset_m', 'tof_valid_K_count', 'dav_margin16_m',
            'dav_margin16_missing', 'dav_finite_fraction', 'uni_margin16_m',
            'uni_margin16_missing', 'uni_finite_fraction')
HGB = dict(max_iter=100, learning_rate=.1, max_leaf_nodes=7, max_depth=3,
           min_samples_leaf=50, l2_regularization=1., early_stopping=False)
METHODS = ('D_prime', 'B_prime', 'audit_b', 'tof', 'rgb', 'or', 'and')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cut(cell):
    return np.inf if cell['threshold_kind'] == 'positive_infinity' else cell['threshold']


def exhaustive(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    order = np.flatnonzero(np.isfinite(scores))
    order = order[np.argsort(-scores[order], kind='stable')]
    values = scores[order]
    boundaries = np.r_[np.flatnonzero(np.diff(values)) + 1, len(order)] if len(order) else []
    counts = np.cumsum(np.array([labels[order] == y for y in (1, 0, -1)]).T, axis=0)
    table = [(np.inf, 0, 0, 0)]
    table.extend((float(values[i-1]), *map(int, counts[i-1])) for i in boundaries)
    return table


def select(scores, labels, target):
    labels = np.asarray(labels, int)
    table = exhaustive(scores, labels)
    nf, np_ = int(sum(labels == 0)), int(sum(labels == 1))
    if not nf:
        return table[0], 'NOT_CALIBRATABLE'
    if not np_:
        return table[0], 'NO_CAL_POS'
    return max((t for t in table if t[2] <= int(np.floor(target * nf))),
               key=lambda t: (t[1], -t[2], t[0])), 'ADOPTED'


def interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(lower=float(np.quantile(values, .025)) if len(values) else None,
                upper=float(np.quantile(values, .975)) if len(values) else None,
                valid_replicates=len(values), total_replicates=2000)


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan, float), where=np.asarray(b) > 0)


def close(actual, expected, path=''):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, (path, key)
            close(actual[key], value, path + '/' + key)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), (path, len(actual), len(expected))
        for i, value in enumerate(expected):
            close(actual[i], value, path + '/' + str(i))
    elif isinstance(expected, (float, np.floating)):
        assert actual is not None and np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def evidence(tof, rgb):
    t = np.isfinite(tof) & (tof >= 0)
    r = np.isfinite(rgb) & (rgb >= 0)
    return t, r, np.where(t, np.where(r, 'both', 'tof_only'), np.where(r, 'rgb_only', 'neither'))


def label_map(path):
    result = {}
    for row in load(path)['rows']:
        for query in row['queries']:
            key = (str(row.get('source_id', row.get('frame_id'))), str(query.get('name', query.get('query_id'))))
            assert key not in result
            result[key] = {'POSITIVE': 1, 'FREE_ON_SAMPLED_RAYS': 0, 'UNKNOWN': -1}[query['state']]
    return result


def protocol_review(root, path):
    p = load(path)
    assert p['task'] == 'SYNC_FUSION_CONFIRM_V3_DEV_20261011'
    assert p['phase'] == 'PRE_REGISTERED_DEVELOPMENT_CONFIRM'
    assert tuple(p['features']['names']) == FEATURES
    assert p['features']['baseline_scores_not_clipped']
    assert p['features']['model_missing_floor_m'] == -10
    assert p['features']['model_metre_features_clip'] == [-10, 10]
    assert 'Remove band_index and all query_x/y/z_index' in p['features']['query_indices']
    assert 'No evidence -> score -inf before cal ties/eval prediction' in p['features']['evidence_gate']
    assert p['models']['seeds'] == [955, 956, 957]
    assert not p['models']['per_source']
    for key, value in HGB.items():
        assert p['models']['HGB'][key] == value
    assert p['models']['HGB']['class_weight'] is None
    lr = p['models']['Logistic']
    assert (lr['C'], lr['penalty'], lr['solver'], lr['max_iter'], lr['class_weight']) == (1, 'l2', 'lbfgs', 1000, None)
    assert p['scores']['methods'] == ['D_prime', 'B_prime', 'audit_b', 'tof', 'rgb', 'or', 'and']
    assert p['calibration']['by'] == ['method', 'band']
    assert p['calibration']['band_targets'] == dict(zip(BANDS, (.02, .05, .1)))
    assert 'Nknown/(6*nknown_arm)' in p['training']['recipe']
    assert 'audit_b only native known rows weight1/unweighted scaler' in p['training']['recipe']
    n = p['new_data']
    assert (n['target_visits'], n['cal_visits'], n['eval_visits'], n['frames_per_visit'], n['total_frames']) == (18, 6, 12, 32, 576)
    assert n['gate']['near_POS_min'] == 16 and n['gate']['midfar_strict_FREE_min'] == 32
    assert 'failure of reference gate consumes whole visit' in n['capture']
    assert all(a in p['ToF']['synthesis'] for a in ARMS)
    assert p['eval']['bootstrap']['replicates'] == 2000 and p['eval']['bootstrap']['seed'] == 20261011
    c = p['confirmation']
    assert (c['primary_method'], c['primary_source'], c['primary_band']) == ('D_prime', ARMS[0], BANDS[1])
    assert 'F/FREE<=.075' in c['criterion'] and 'bootstrap95% deltaW/POS lower>0' in c['criterion']
    assert 'D_prime W-rgb W>=0 AND F/FREE<=.04' in c['secondary']
    assert c['pass_claim'] == '去先验、证据门控的融合在中带得到确认（真实 RGB + 半合成 ToF）'
    assert 'Far report only, no claim' in c['report']
    assert (p['budget']['CPU_command_wall_s'], p['budget']['GPU_wall_s'], p['budget']['download_bytes']) == (2400, 600, 4000000000)
    if (root / 'PLAN.json').exists():
        assert sha(root / 'PLAN.json') == sha(path)
    return dict(status='PASS', protocol_sha256=sha(path), features=16, shared_ensembles=3,
                calibration_cells=21, arms=6, new_eval_reference_reads=0, new_candidate_reads=0,
                checks=['six-arm weighted fixed HGB and LR plus native-only LR',
                        'all four explicit query/band indices removed',
                        'unclipped original sensor minimum evidence gate',
                        'new chronological cal6/eval12 visits and fixed windows',
                        'all-ties mixed calibration and shared thresholds',
                        'mid primary positive paired gain/CI and FREE7.5%',
                        'FARO and ambient10 near secondary gain>=0/FREE4%',
                        'full12visit paired bootstrap and missing-denominator rejection',
                        'far descriptive only; protected480/test excluded',
                        'strict command-wall/GPU/download budget and once-only eval'])


def cluster_counts(visits, values, roster):
    return np.asarray([np.asarray(values)[visits == visit].sum() for visit in roster], int)


def recompute_metric(labels, predictions, visits, roster, weights):
    den = np.asarray([cluster_counts(visits, labels == label, roster) for label in (1, 0, -1)]).T
    count = np.asarray([cluster_counts(visits, (labels == label) & predictions, roster) for label in (1, 0, -1)]).T
    d, n = den.sum(0), count.sum(0)
    return dict(W=int(n[0]), POS=int(d[0]), F=int(n[1]), FREE=int(d[1]),
                U=int(n[2]), UNKNOWN=int(d[2]),
                W_rate=float(n[0] / d[0]) if d[0] else None,
                F_rate=float(n[1] / d[1]) if d[1] else None,
                W_rate_CI=interval(ratio(weights @ count[:, 0], weights @ den[:, 0])),
                F_rate_CI=interval(ratio(weights @ count[:, 1], weights @ den[:, 1])))


def recompute_pair(labels, predictions, comparator, visits, roster, weights):
    added, removed = predictions & ~comparator, ~predictions & comparator
    rescue, loss = added & (labels == 1), removed & (labels == 1)
    net = cluster_counts(visits, rescue, roster) - cluster_counts(visits, loss, roster)
    pos = cluster_counts(visits, labels == 1, roster)
    return dict(rescue=int(rescue.sum()), loss=int(loss.sum()), delta_W=int(net.sum()),
                POS=int(pos.sum()), delta_W_rate=float(net.sum() / pos.sum()) if pos.sum() else None,
                delta_W_CI=interval(weights @ net),
                delta_W_rate_CI=interval(ratio(weights @ net, weights @ pos)),
                FREE_added=int((added & (labels == 0)).sum()),
                FREE_removed=int((removed & (labels == 0)).sum()),
                UNKNOWN_added=int((added & (labels == -1)).sum()),
                UNKNOWN_removed=int((removed & (labels == -1)).sum()))


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def verify_bindings(root):
    import subprocess
    p = load(root / 'PLAN.json')
    receipt = load(root / 'protocol_commit.json')
    assert receipt['pushed_before_data'] and receipt['plan_sha256'] == sha(root / 'PLAN.json')
    repo = Path(__file__).resolve().parents[4]
    rel = 'research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V3_PROTOCOL_DEV_20261011.json'
    committed = subprocess.check_output(['git', 'show', receipt['commit'] + ':' + rel], cwd=repo)
    raw = (root / 'PLAN.json').read_bytes()
    # Git autocrlf changes byte identity only; verify exact normalized text and JSON.
    assert committed.replace(b'\r\n', b'\n') == raw.replace(b'\r\n', b'\n')
    assert json.loads(committed) == p
    receipt = dict(receipt, git_blob_sha256=hashlib.sha256(committed).hexdigest(),
                   protocol_binding='exact text after CRLF to LF normalization; decoded JSON equal')
    return p, receipt


def verify_hashes(entries):
    for entry in entries:
        assert sha(entry['path']) == entry['sha256'], entry['path']


def trainreview(root):
    import joblib
    assert not (root / 'eval/eval_open.json').exists()
    p, receipt = verify_bindings(root)
    seal = load(root / 'train/train_seal.json')
    assert seal['plan_sha256'] == receipt['plan_sha256'] and tuple(seal['feature_names']) == FEATURES
    verify_hashes(seal['dependencies']); verify_hashes(seal['model_entries'])
    assert sha(seal['training_features_path']) == seal['training_features_sha256'] == p['inherited']['training_features_sha256']
    d = npz(seal['training_features_path'])
    assert tuple(d['feature_names'][:16]) == FEATURES and d['X'].shape[1] == 20
    assert len(set(d['visit_id'])) == 30 and sorted(set(d['visit_id'])) == seal['visit_ids']
    known = d['y'] >= 0
    assert set(d['y'][known]) == {0, 1} and set(d['arm']) == set(ARMS)
    weights = np.zeros(len(known))
    for arm, record in zip(ARMS, seal['arm_counts']):
        take = known & (d['arm'] == arm)
        weights[take] = known.sum() / (6 * take.sum())
        close(record, dict(arm=arm, known=int(take.sum()), weight_each=float(weights[take][0]),
                           weight_sum=float(weights[take].sum()), POS=int(((d['arm'] == arm) & (d['y'] == 1)).sum()),
                           FREE=int(((d['arm'] == arm) & (d['y'] == 0)).sum())))
    assert len(seal['model_entries']) == 9
    for method in METHODS[:3]:
        entries = [e for e in seal['model_entries'] if e['method'] == method]
        assert sorted(e['seed'] for e in entries) == [955, 956, 957]
        take = known if method != 'audit_b' else known & (d['arm'] == ARMS[0])
        w = weights[take] if method != 'audit_b' else np.ones(int(take.sum()))
        X = d['X'][take, :16]
        for entry in entries:
            model = joblib.load(entry['path'])
            assert model.classes_.tolist() == [0, 1] and model.n_features_in_ == 16
            assert entry['known_rows'] == int(take.sum())
            if method == 'D_prime':
                for key, value in HGB.items():
                    assert model.get_params()[key] == value
                assert model.random_state == entry['seed']
            else:
                scaler, lr = list(model.named_steps.values())
                mean = np.average(X, axis=0, weights=w)
                variance = np.average((X - mean) ** 2, axis=0, weights=w)
                np.testing.assert_allclose(scaler.mean_, mean, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(scaler.var_, variance, rtol=1e-10, atol=1e-10)
                assert np.isclose(scaler.n_samples_seen_, w.sum())
                assert (lr.C, lr.penalty, lr.solver, lr.max_iter, lr.class_weight, lr.random_state) == (1, 'l2', 'lbfgs', 1000, None, entry['seed'])
    return dict(status='PASS', train_seal_sha256=sha(root / 'train/train_seal.json'),
                training_feature_sha256=seal['training_features_sha256'], models_verified=9,
                weighted_arm_sums_verified=6, training_visits=30, feature_columns=16,
                protocol_commit=receipt['commit'], protocol_plan_sha256=receipt['plan_sha256'],
                protocol_git_blob_sha256=receipt['git_blob_sha256'], protocol_binding=receipt['protocol_binding'],
                new_eval_reads=0, model_inference_runs=0, model_refits=0)


def sensor_samples(root, cal_features):
    """Two preselected calibration endpoints; frozen pure functions, no v3 builder."""
    import csv
    from sync_rgb_tof_dataset_v1 import sample, peak_readout
    from sync_fusion_features_v1 import features_for_depth
    from rgb_body_query_reference_eval import rays, ray_interval
    public = load(root / 'public_roster.json')
    frames = [r for r in public['rows'] if r['role'] == 'cal']
    selected = [frames[0], frames[-1]]
    synth = {r['source_id']: r for r in load(root / 'synthesis_manifest.json')['frames']}
    seal = load(root / 'frozen_readout_seal.json')
    with (root / 'input_coverage_gate.csv').open(newline='', encoding='utf8') as stream:
        gates = {(r['frame_id'], r['arm'], int(r['K'])): int(r['joint_pass_zones']) for r in csv.DictReader(stream)}
    rgb_predictions = {}
    for model in ('dav2', 'unidepth'):
        rgb_predictions[model] = {r['source_id']: r for r in load(root / 'rgb_inference' / model / 'predictions.json')['rows']}
    feature_index = {(str(a), str(f), str(q)): i for i, (a, f, q) in enumerate(zip(
        cal_features['arm'], cal_features['frame_id'], cal_features['query_id']))}
    pressure_arrays = query_rows = gate_frames = 0
    gate_results = []
    for public_row in selected:
        fid = public_row['source_id']; frame = synth[fid]
        assert frame['role'] == 'cal'
        K, shape = np.asarray(public_row['depth_K']), tuple(public_row['depth_shape'])
        rx, ry = rays(K, shape)
        intervals = [ray_interval(rx, ry, q) for q in public['queries']]
        rgb = {}
        for model in rgb_predictions:
            prediction = rgb_predictions[model][fid]
            assert sha(prediction['path']) == prediction['sha256']
            depth = npz(prediction['path'])['depth']
            rgb[model] = [features_for_depth(depth, *v) for v in intervals]
        base = {}
        for a in (ARMS[0], 'faro_rho030_ambient1'):
            for k in (0, 1):
                entry = next(e for e in frame['arms'] if e['arm'] == a and e['repeat'] == k)
                assert sha(entry['path']) == entry['sha256']
                base[a, k] = npz(entry['path'])
        faro = base['faro_rho030_ambient1', 0]
        peak = faro['hist'].argmax(-1)
        height = np.take_along_axis(faro['hist'], peak[..., None], axis=-1)[..., 0]
        background = np.take_along_axis(faro['background'], peak[..., None], axis=-1)[..., 0]
        snr = height / np.sqrt(np.maximum(height, 0) + 2 * background + 1)
        joint = int(((faro['coverage'] >= .75) & (snr >= 3)).sum())
        assert joint == gates[frame['frame_id'], 'faro_rho030_ambient1', 0]
        accepted = joint >= 52
        gate_frames += 1
        gate_results.append(dict(frame_id=fid, joint_pass_zones=joint, admitted=accepted))
        for arm in ARMS:
            reconstructed = []
            for k in (0, 1):
                original = base[arm if arm in (ARMS[0], 'faro_rho030_ambient1') else 'faro_rho030_ambient1', k]
                if arm not in (ARMS[0], 'faro_rho030_ambient1'):
                    signal_scale = .5 if arm == 'faro_rho015_ambient1' else 2 if arm == 'faro_rho060_ambient1' else 1
                    ambient_scale = 3 if arm == 'faro_rho030_ambient3' else 10 if arm == 'faro_rho030_ambient10' else 1
                    hist, counts, bg = sample(original['expectation'][None] * signal_scale,
                                              original['ambient'][None] * ambient_scale, int(original['seed']))
                    stored = npz(root / 'simulated' / fid / (arm + f'_k{k}.npz'))
                    for name, value in [('hist', hist[0]), ('counts', counts[0]), ('background', bg[0]),
                                        ('expectation', original['expectation'] * signal_scale),
                                        ('ambient', original['ambient'] * ambient_scale)]:
                        np.testing.assert_array_equal(stored[name], value)
                        pressure_arrays += 1
                    assert int(stored['seed']) == int(original['seed'])
                    observation = stored
                else:
                    observation = original
                if np.isfinite(observation['grid_pose']).all():
                    depth = peak_readout(observation['hist'], observation['background'], observation['coverage'],
                                         observation['grid_K'], shape, observation['grid_pose'], K, shape,
                                         observation['rgb_pose'], seal)[0]
                else:
                    depth = np.full(shape, np.inf)
                reconstructed.append([features_for_depth(depth, *v) for v in intervals])
            for j, query in enumerate(public['queries']):
                key = arm, fid, query['name']
                if arm == 'faro_rho030_ambient1' and not accepted:
                    assert key not in feature_index
                    continue
                index = feature_index[key]
                s0, v0 = reconstructed[0][j]; s1, v1 = reconstructed[1][j]
                valid = np.isfinite(s0) and np.isfinite(s1)
                raw = (s0 + s1) / 2 if valid else -np.inf
                values = (v0 + v1) / 2
                # Explicit assembly from frozen base-field semantics.
                expected = np.asarray([raw if valid else -10., int(not valid), values[4], values[3],
                                       values[6], values[5], values[7], values[8], values[9],
                                       int(np.isfinite(s0)) + int(np.isfinite(s1)),
                                       *rgb['dav2'][j][1][:3], *rgb['unidepth'][j][1][:3]], float)
                for column, name in enumerate(FEATURES):
                    if name.endswith('_m'):
                        expected[column] = np.clip(expected[column], -10, 10)
                np.testing.assert_array_equal(cal_features['X'][index], expected)
                assert cal_features['tof_score'][index] == raw
                band = 0 if query['low'][2] == .3 else 1 if query['low'][2] == .8 else 2
                rgb_raw = rgb['unidepth' if band == 0 else 'dav2'][j][0]
                assert cal_features['rgb_score'][index] == rgb_raw and cal_features['band'][index] == band
                query_rows += 1
    return dict(status='PASS', selection='first and last public calibration frames, independent of outcomes',
                calibration_frame_ids=[r['source_id'] for r in selected], pressure_arrays_exact=pressure_arrays,
                six_arm_query_rows_exact=query_rows, FARO_K0_gate_frames=gate_frames,
                FARO_K0_gates=gate_results, new_eval_reference_reads=0, new_eval_model_score_reads=0)


def preeval(root):
    import joblib
    from threadpoolctl import threadpool_limits
    assert not (root / 'eval/eval_open.json').exists(), 'Preeval audit must precede OPEN'
    p, receipt = verify_bindings(root)
    tr = load(root / 'train/train_seal.json')
    ca = load(root / 'cal/cal_seal.json')
    assert tr['plan_sha256'] == ca['plan_sha256'] == receipt['plan_sha256']
    assert ca['train_seal_sha256'] == sha(root / 'train/train_seal.json')
    assert ca['model_entries'] == tr['model_entries']
    prior_train_review = load(root / 'independent_train_audit.json')
    assert prior_train_review['status'] == 'PASS' and prior_train_review['train_seal_sha256'] == sha(root / 'train/train_seal.json')
    assert tuple(tr['feature_names']) == FEATURES
    verify_hashes(tr['dependencies']); verify_hashes(ca['dependencies']); verify_hashes(tr['model_entries'])
    assert sha(tr['training_features_path']) == tr['training_features_sha256'] == p['inherited']['training_features_sha256']
    # Reuse the successful independent training audit through immutable hashes.
    visits = tr['visit_ids']
    assert len(visits) == 30
    assert sha(ca['feature_path']) == ca['feature_sha256']
    assert sha(ca['reference_path']) == ca['reference_sha256']
    assert sha(ca['cache_path']) == ca['cache_sha256']
    refs = load(ca['reference_path'])['rows']
    assert all(r.get('role', r.get('split')) == 'cal' for r in refs), 'No eval labels in calibration input'
    for row in refs:
        for query in row['queries']:
            if query['state'] == 'FREE_ON_SAMPLED_RAYS':
                assert query['unknown_pixels'] == query['positive_pixels'] == 0
                assert query['free_ray_pixels'] == query['domain_pixels']
    yc = label_map(ca['reference_path'])
    f = npz(ca['feature_path']); cache = npz(ca['cache_path'])
    assert tuple(f['feature_names']) == FEATURES and f['X'].shape == (len(f['arm']), 16)
    calvisits = sorted(set(map(str, f['visit_id'])))
    assert len(calvisits) == 6 and calvisits == ca['cal_visit_ids'] and not set(calvisits) & set(visits)
    y = np.asarray([yc[str(fid), str(qid)] for fid, qid in zip(f['frame_id'], f['query_id'])], int)
    np.testing.assert_array_equal(y, cache['label'])
    for key in ('arm', 'frame_id', 'query_id', 'visit_id', 'band'):
        np.testing.assert_array_equal(f[key], cache[key])
    t, r, groups = evidence(f['tof_score'], f['rgb_score'])
    np.testing.assert_array_equal(cache['tof_evidence'], t)
    np.testing.assert_array_equal(cache['rgb_evidence'], r)
    np.testing.assert_array_equal(cache['evidence_group'], groups)
    models = {}; scores = dict(tof=f['tof_score'], rgb=f['rgb_score'])
    scores['or'] = np.maximum(scores['tof'], scores['rgb']); scores['and'] = np.minimum(scores['tof'], scores['rgb'])
    with threadpool_limits(limits=1):
        for method in METHODS[:3]:
            entries = [e for e in tr['model_entries'] if e['method'] == method]
            assert sorted(e['seed'] for e in entries) == [955, 956, 957]
            prediction = []
            for entry in entries:
                model = joblib.load(entry['path'])
                prediction.append(model.predict_proba(f['X'])[:, 1])
            scores[method] = np.where(t | r, np.mean(prediction, axis=0), -np.inf)
    for method in METHODS:
        np.testing.assert_allclose(scores[method], cache[method + '_score'], rtol=1e-12, atol=1e-12)
    ties_checked = 0
    assert len(ca['cells']) == 21
    for cell in ca['cells']:
        method, band = cell['method'], cell['band']
        take = f['band'] == band
        target = p['calibration']['band_targets'][BANDS[band]]
        chosen, status = select(scores[method][take], y[take], target)
        assert cut(cell) == chosen[0] and cell['status'] == status
        assert (cell['W'], cell['F'], cell['U']) == chosen[1:]
        assert (cell['POS_denominator'], cell['FREE_denominator'], cell['UNKNOWN_denominator']) == tuple(int((y[take] == k).sum()) for k in (1, 0, -1))
        assert sha(cell['all_ties_path']) == cell['all_ties_sha256']
        all_ties = load(cell['all_ties_path'])['rows']
        expected = exhaustive(scores[method][take], y[take])
        assert len(all_ties) == len(expected)
        for row, expected_row in zip(all_ties, expected):
            assert (cut(row), row['W'], row['F'], row['U']) == expected_row
        ties_checked += len(expected)
    for method in METHODS:
        expected = np.zeros(len(y), bool)
        for cell in ca['cells']:
            if cell['method'] == method and cell['status'] == 'ADOPTED':
                take = f['band'] == cell['band']
                expected[take] = scores[method][take] >= cut(cell)
        np.testing.assert_array_equal(expected, cache[method + '_pred'])
    fm = load(root / 'feature_manifest.json')
    assert not fm['reference_read'] and tuple(fm['feature_names']) == FEATURES
    verify_hashes(fm['inputs']); verify_hashes(fm['outputs']); verify_hashes(fm['simulations'])
    from audit_sync_fusion_confirm_v2 import v2_data_integrity
    data_review = v2_data_integrity(root)
    assert data_review['cohort_complete'] and data_review['admitted_visits'] == 18 and data_review['frames'] == 576
    assert data_review['split_counts'] == {'cal': 6, 'eval': 12}
    eval_entry = next(e for e in fm['outputs'] if e['role'] == 'eval')
    # Lazy NPZ reads inspect IDs only, not labels/features/sensor model scores.
    with np.load(eval_entry['path'], allow_pickle=False) as z:
        ev = {key: z[key] for key in ('arm', 'visit_id', 'frame_id', 'query_id', 'band')}
    evalvisits = sorted(set(map(str, ev['visit_id'])))
    assert len(evalvisits) == 12 and not set(evalvisits) & (set(calvisits) | set(visits))
    for role, data, rolevisits in (('cal', f, calvisits), ('eval', ev, evalvisits)):
        assert len(set(zip(data['arm'], data['frame_id'], data['query_id']))) == len(data['arm'])
        for visit in rolevisits:
            take = (data['visit_id'] == visit) & (data['arm'] == ARMS[0])
            assert int(take.sum()) == 32 * 27 and len(set(data['frame_id'][take])) == 32
    sensors = sensor_samples(root, f)
    return dict(status='PASS', models_verified=9, weighted_arm_sums_verified=6,
                training_visits=30, calibration_visits=6, calibration_rows=len(y),
                exhaustive_cells=21, exhaustive_ties_checked=ties_checked,
                data_integrity=data_review, sensor_samples=sensors,
                prior_train_audit_sha256=sha(root / 'independent_train_audit.json'),
                cal_seal_sha256=sha(root / 'cal/cal_seal.json'),
                new_eval_reference_reads=0, new_eval_model_score_reads=0, training_model_refits=0)


def posteval(root):
    import csv
    p, receipt = verify_bindings(root)
    opened = load(root / 'eval/eval_open.json')
    terminal = load(root / 'eval/eval_terminal.json')
    seal = load(root / 'eval/cache_seal.json')
    cal = load(root / 'cal/cal_seal.json')
    summary = load(root / 'eval/summary.json')
    assert terminal['status'] == 'COMPLETE_CACHE_FROZEN' and not terminal['reopen_permitted']
    assert opened['one_shot'] and opened['plan_sha256'] == receipt['plan_sha256']
    assert opened['cal_seal_sha256'] == seal['cal_seal_sha256'] == sha(root / 'cal/cal_seal.json')
    assert seal['eval_open_sha256'] == sha(root / 'eval/eval_open.json')
    assert sha(root / 'eval/cache.npz') == seal['cache_sha256'] == summary['cache_sha256']
    assert sha(root / 'eval/per_query.csv') == seal['per_query_sha256']
    verify_hashes(opened['dependencies']); verify_hashes(cal['dependencies']); verify_hashes(cal['model_entries'])
    assert sha(cal['train_seal_path']) == cal['train_seal_sha256']
    verify_hashes([dict(path=c['all_ties_path'], sha256=c['all_ties_sha256']) for c in cal['cells']])
    cache = npz(root / 'eval/cache.npz')
    n = len(cache['label'])
    assert all(len(v) == n for v in cache.values()) and seal['rows'] == terminal['rows_cached'] == n
    assert len(set(zip(cache['arm'], cache['frame_id'], cache['query_id']))) == n
    roster = list(map(str, opened['visit_ids']))
    assert len(roster) == len(set(roster)) == 12 and roster == seal['visit_ids']
    train = load(cal['train_seal_path'])
    assert not set(roster) & (set(cal['cal_visit_ids']) | set(train['visit_ids']))
    assert set(cache['visit_id']) <= set(roster) and set(cache['label']) <= {-1, 0, 1}
    assert set(cache['arm']) <= set(ARMS) and set(cache['band']) <= {0, 1, 2}
    for visit in roster:
        take = (cache['arm'] == ARMS[0]) & (cache['visit_id'] == visit)
        assert int(take.sum()) == 32 * 27 and len(set(cache['frame_id'][take])) == 32
    t, r, group = evidence(cache['tof_score'], cache['rgb_score'])
    for key, expected in [('tof_evidence', t), ('rgb_evidence', r), ('evidence_group', group)]:
        np.testing.assert_array_equal(cache[key], expected)
    np.testing.assert_array_equal(cache['or_score'], np.maximum(cache['tof_score'], cache['rgb_score']))
    np.testing.assert_array_equal(cache['and_score'], np.minimum(cache['tof_score'], cache['rgb_score']))
    for method in METHODS[:3]:
        score = cache[method + '_score']
        assert np.all(np.isneginf(score[~(t | r)]))
        assert np.all(np.isfinite(score[t | r]) & (score[t | r] >= 0) & (score[t | r] <= 1))
    decisions = 0
    for method in METHODS:
        expected = np.zeros(n, bool)
        for cell in cal['cells']:
            if cell['method'] == method and cell['status'] == 'ADOPTED':
                take = cache['band'] == cell['band']
                expected[take] = cache[method + '_score'][take] >= cut(cell)
        np.testing.assert_array_equal(cache[method + '_pred'], expected)
        decisions += n
    # The text table must represent the immutable binary cache row for row.
    csv_count = 0
    with (root / 'eval/per_query.csv').open(newline='', encoding='utf8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(cache)
        for i, row in enumerate(reader):
            for key, values in cache.items():
                value = values[i].item()
                if isinstance(value, bool):
                    assert row[key] == str(value)
                elif isinstance(value, (float, int)):
                    assert float(row[key]) == value
                else:
                    assert row[key] == value
            csv_count += 1
    assert csv_count == n
    # Count multiplicities, instead of using the run's indexed-sum implementation.
    draw = np.random.default_rng(20261011).integers(0, 12, (2000, 12))
    weights = np.column_stack([(draw == i).sum(1) for i in range(12)])
    metrics = {}; best = {}; pairs = {}; decompositions = []
    for row in summary['metrics']:
        arm, band, method = row['arm'], row['band'], row['method']
        take = (cache['arm'] == arm) & (cache['band'] == band)
        y, v, pred = cache['label'][take], cache['visit_id'][take], cache[method + '_pred'][take]
        expected = recompute_metric(y, pred, v, roster, weights)
        close(row, expected)
        assert row['contributing_visits'] == len(set(v))
        c = next(c for c in cal['cells'] if (c['method'], c['band']) == (method, band))
        assert row['cal_status'] == c['status'] and row['band_name'] == BANDS[band]
        key = arm, band, method
        assert key not in metrics
        metrics[key] = expected | dict(cal_status=c['status'], contributing_visits=len(set(v)))
    assert len(metrics) == 126
    for row in summary['best_single']:
        arm, band = row['arm'], row['band']
        chosen = max(('tof', 'rgb'), key=lambda m: (metrics[arm, band, m]['W'], -metrics[arm, band, m]['F'], m == 'rgb'))
        assert row['method'] == chosen
        best[arm, band] = chosen
    assert len(best) == 18
    for row in summary['paired']:
        arm, band, method, role = row['arm'], row['band'], row['method'], row['comparator_role']
        comp = best[arm, band] if role == 'best_single' else 'or'
        assert row['comparator'] == comp
        take = (cache['arm'] == arm) & (cache['band'] == band)
        y, v = cache['label'][take], cache['visit_id'][take]
        pred, comparator = cache[method + '_pred'][take], cache[comp + '_pred'][take]
        expected = recompute_pair(y, pred, comparator, v, roster, weights)
        close(row, expected)
        pairs[arm, band, method, role] = expected
        groups = cache['evidence_group'][take]
        for g in GROUPS:
            keep = groups == g
            decomposition = recompute_pair(y[keep], pred[keep], comparator[keep], v[keep], roster, weights)
            decompositions.append(dict(arm=arm, band=band, method=method, comparator=comp,
                                       comparator_role=role, group=g, **decomposition))
        parts = decompositions[-4:]
        for key in ('rescue', 'loss', 'delta_W', 'POS', 'FREE_added', 'FREE_removed', 'UNKNOWN_added', 'UNKNOWN_removed'):
            assert sum(part[key] for part in parts) == expected[key]
    assert len(pairs) == 108 and len(decompositions) == 432
    evidence_checked = 0
    for row in summary['evidence_four']:
        arm, band, method, g = row['arm'], row['band'], row['method'], row['group']
        take = (cache['arm'] == arm) & (cache['band'] == band) & (cache['evidence_group'] == g)
        y, pred = cache['label'][take], cache[method + '_pred'][take]
        expected = dict(POS=int((y == 1).sum()), FREE=int((y == 0).sum()), UNKNOWN=int((y == -1).sum()),
                        W=int(((y == 1) & pred).sum()), F=int(((y == 0) & pred).sum()), U=int(((y == -1) & pred).sum()),
                        supported=int(pred.sum()))
        close(row, expected)
        if method in METHODS[:3] and g == 'neither':
            assert expected['supported'] == 0
        evidence_checked += 1
    assert evidence_checked == 504 and summary['evidence_gate_neither_supported'] == 0
    primary = {}
    for row in summary['model_primary_comparison']:
        method = row['method']; m = metrics[ARMS[0], 1, method]; pair = pairs[ARMS[0], 1, method, 'best_single']
        valid = m['cal_status'] == 'ADOPTED' and m['POS'] > 0 and m['FREE'] > 0
        criteria = dict(calibrated_and_denominators=valid, positive_difference=pair['delta_W'] > 0,
                        interval_lower_positive=pair['delta_W_rate_CI']['lower'] is not None and pair['delta_W_rate_CI']['lower'] > 0,
                        FREE_tolerance_pass=m['F_rate'] is not None and m['F_rate'] <= .075)
        close(row, pair)
        close(row, dict(FREE_rate=m['F_rate'], FREE_tolerance=.075, criteria=criteria,
                        status='PASS' if all(criteria.values()) else 'FAIL' if valid else 'NOT_EVALUABLE', pass_all=all(criteria.values())))
        primary[method] = row
    assert len(primary) == 3 and summary['primary'] == primary['D_prime']
    secondary = []
    for row in summary['secondary']:
        arm, method = row['arm'], row['method']; m = metrics[arm, 0, method]; rgb = metrics[arm, 0, 'rgb']
        valid = m['cal_status'] == 'ADOPTED' and m['POS'] > 0 and m['FREE'] > 0 and m['contributing_visits'] > 0
        criteria = dict(calibrated_and_denominators=valid, nonnegative_net=m['W'] - rgb['W'] >= 0,
                        FREE_tolerance_pass=m['F_rate'] is not None and m['F_rate'] <= .04)
        close(row, dict(delta_W_vs_RGB=m['W'] - rgb['W'], FREE_rate=m['F_rate'], FREE_tolerance=.04,
                        criteria=criteria, status='PASS' if all(criteria.values()) else 'FAIL' if valid else 'NOT_EVALUABLE',
                        pass_all=all(criteria.values())))
        secondary.append(row)
    assert len(secondary) == 6
    assert summary['secondary_D_prime_pass'] == all(r['pass_all'] for r in secondary if r['method'] == 'D_prime')
    assert summary['result'] == (p['confirmation']['pass_claim'] if primary['D_prime']['pass_all'] else '主中带确认未通过')
    for row in summary['faro_RGB_saturation']:
        m = metrics['faro_rho030_ambient1', row['band'], 'rgb']
        close(row, dict(RGB_W=m['W'], POS=m['POS'], RGB_W_rate=m['W_rate'], remaining_POS=m['POS'] - m['W'],
                        saturated_descriptor=m['W_rate'] >= .95 if m['W_rate'] is not None else None))
    decomp_path = root / 'eval/independent_evidence_decomposition.json'
    with decomp_path.open('x', encoding='utf8') as stream:
        json.dump(decompositions, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(status='PASS', cached_rows=n, query_decisions_checked=decisions,
                CSV_rows_verified=csv_count, metrics_checked=126, paired_rows_checked=108,
                evidence_rows_checked=504, four_group_decompositions=432,
                bootstrap_replicates=2000, visit_clusters=12,
                cache_sha256=seal['cache_sha256'], summary_sha256=sha(root / 'eval/summary.json'),
                evidence_decomposition_path=str(decomp_path.resolve()), evidence_decomposition_sha256=sha(decomp_path),
                primary=primary['D_prime'], secondary=[r for r in secondary if r['method'] == 'D_prime'],
                new_eval_reference_reopens=0, model_inference_reruns=0, result=summary['result'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('protocol', 'train', 'preeval', 'posteval'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.stage == 'protocol':
        assert args.protocol is not None
        result = protocol_review(args.root, args.protocol)
    else:
        result = {'train': trainreview, 'preeval': preeval, 'posteval': posteval}[args.stage](args.root)
    result.update(CPU_command_wall_s=time.perf_counter() - started, GPU_s=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
