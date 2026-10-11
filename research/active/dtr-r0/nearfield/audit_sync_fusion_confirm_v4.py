"""Independent v4 checks; no imports from v4 selection/metric implementation.

The pre-eval stage never reads eval labels or model scores. The post-eval stage
uses only the sealed once-opened cache; it cannot reopen evaluator references.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

ARMS = ('native_perturbed', 'faro_rho015_ambient1', 'faro_rho060_ambient1',
        'faro_rho030_ambient3', 'faro_rho030_ambient1', 'faro_rho030_ambient10')
METHODS = ('D_prime', 'rgb', 'tof', 'or', 'and')
BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
FEATURES = ('tof_margin16_m', 'tof_margin16_missing', 'tof_support_pixels',
            'tof_unknown_fraction', 'tof_before_fraction', 'tof_inside_fraction',
            'tof_after_fraction', 'tof_median_entry_offset_m',
            'tof_median_exit_offset_m', 'tof_valid_K_count', 'dav_margin16_m',
            'dav_margin16_missing', 'dav_finite_fraction', 'uni_margin16_m',
            'uni_margin16_missing', 'uni_finite_fraction')
HGB = dict(max_iter=100, learning_rate=.1, max_leaf_nodes=7, max_depth=3,
           min_samples_leaf=50, l2_regularization=1., early_stopping=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def hashes(entries):
    for e in entries:
        assert sha(e['path']) == e['sha256'], e['path']


def close(a, b, name=''):
    if b is None:
        assert a is None, (name, a, b)
    elif isinstance(b, (float, int, np.number)):
        assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (name, a, b)
    elif isinstance(b, dict):
        for k, v in b.items():
            assert k in a, (name, k)
            close(a[k], v, name + '/' + k)
    elif isinstance(b, (list, tuple)):
        assert len(a) == len(b), name
        for i, v in enumerate(b):
            close(a[i], v, name + '/' + str(i))
    else:
        assert a == b, (name, a, b)


def threshold(row):
    return np.inf if row['threshold_kind'] == 'positive_infinity' else row['threshold']


def ties(scores, labels, visits=None, roster=None):
    """Whole finite-score ties in threshold order, including no-support +inf."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    ix = np.flatnonzero(np.isfinite(scores))
    ix = ix[np.argsort(-scores[ix], kind='stable')]
    values, y = scores[ix], labels[ix]
    ends = np.r_[np.flatnonzero(np.diff(values)) + 1, len(ix)] if len(ix) else np.array([], int)
    t = np.r_[np.inf, values[ends - 1]] if len(ix) else np.array([np.inf])
    counts = np.zeros((len(t), 3), int)
    for j, value in enumerate((1, 0, -1)):
        if len(ix):
            counts[1:, j] = np.cumsum(y == value)[ends - 1]
    clusters = None
    if visits is not None:
        v = np.asarray(visits)[ix]
        clusters = np.zeros((len(t), len(roster), 2), int)
        for k, visit in enumerate(roster):
            for j, value in enumerate((1, 0)):
                if len(ix):
                    clusters[1:, k, j] = np.cumsum((v == visit) & (y == value))[ends - 1]
    return t, counts, clusters


def choose(scores, y, target):
    t, c, _ = ties(scores, y)
    nf, npos = int((y == 0).sum()), int((y == 1).sum())
    if not nf or not npos:
        return 0, t, c, 'NOT_CALIBRATABLE' if not nf else 'NO_CAL_POS'
    eligible = np.flatnonzero(c[:, 1] <= int(np.floor(target * nf)))
    i = max(eligible, key=lambda j: (c[j, 0], -c[j, 1], t[j]))
    return i, t, c, 'ADOPTED'


def area(counts, npos, nfree, width=.05):
    """Integrate budget envelope; no interpolation, no tie splitting.

    Threshold-order F and W are monotone. Consecutive equal-F knots contribute
    zero width; the last such knot supplies maximum W on the next interval.
    """
    if not npos or not nfree:
        return None
    x = np.clip(counts[:, 1] / nfree, 0, width)
    y = counts[:, 0] / npos
    return float((np.diff(x) @ y[:-1] + (width - x[-1]) * y[-1]) / width)


def bootstrap(clusters, multiplicities, pd, fd, width=.05):
    values = np.full(len(multiplicities), np.nan)
    # Float64 BLAS products remain exact for these small integer multiplicities
    # and counts; the integer range is bounded well below 2**53.
    assert int(clusters.max()) * int(multiplicities.sum(axis=1).max()) < 2**52
    cw, cf = clusters[:, :, 0].astype(float), clusters[:, :, 1].astype(float)
    for start in range(0, len(values), 64):
        mult = multiplicities[start:start + 64].T.astype(float)
        pden, fden = pd @ mult, fd @ mult
        w, f = cw @ mult, cf @ mult
        good = (pden > 0) & (fden > 0)
        x = np.clip(f[:, good] / fden[good], 0, width)
        y = w[:, good] / pden[good]
        result = (np.sum(np.diff(x, axis=0) * y[:-1], axis=0) + (width - x[-1]) * y[-1]) / width
        batch = values[start:start + mult.shape[1]]
        batch[good] = result
    return values


def interval(values):
    good = np.asarray(values)[np.isfinite(values)]
    return dict(lower=float(np.quantile(good, .025)) if len(good) else None,
                upper=float(np.quantile(good, .975)) if len(good) else None,
                valid_replicates=len(good), total_replicates=2000)


def synthetic():
    """Known exact areas exercise F0, unsplit overshoot, and tail plateau."""
    c = np.array([[0, 0, 0], [1, 0, 0], [3, 1, 0], [5, 8, 0]])
    close(area(c, 10, 100), .26, 'F0_then_1pct_then_overshoot')
    c = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [8, 8, 0]])
    close(area(c, 10, 100), .4, 'same_F0_maxW_no_overshoot_split')
    close(area(np.array([[0, 0, 0], [4, 0, 0]]), 10, 100), .4, 'final_flat_to_5pct')
    assert area(c, 0, 100) is None and area(c, 10, 0) is None
    scores = np.array([.9, .8, .8, .7, -np.inf])
    y = np.array([1, 1, 0, 1, 0]); v = np.array(['a', 'a', 'b', 'b', 'b'])
    _, counts, clusters = ties(scores, y, v, ['a', 'b'])
    assert counts.tolist() == [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]]
    mult = np.array([[1, 1], [2, 0], [0, 2]])
    expected = []
    for m in mult:
        repeated = np.repeat(np.arange(len(y)), [m[0] if z == 'a' else m[1] for z in v])
        _, weighted, _ = ties(scores[repeated], y[repeated])
        expected.append(area(weighted, int((y[repeated] == 1).sum()), int((y[repeated] == 0).sum())))
    actual = bootstrap(clusters, mult, np.array([2, 1]), np.array([0, 2]))
    np.testing.assert_allclose(actual, np.asarray([np.nan if x is None else x for x in expected]), equal_nan=True)
    return dict(status='PASS', cases=6, areas=[.26, .4, .4], whole_tie_plateaus=True,
                zero_denominator_rejection=True, weighted_vs_explicit_repeat=True)


def binding(root):
    p = load(root / 'PLAN.json'); rec = load(root / 'protocol_commit.json')
    assert rec['pushed_before_data'] and rec['plan_sha256'] == sha(root / 'PLAN.json')
    rel = 'research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V4_PROTOCOL_DEV_20261011.json'
    committed = subprocess.check_output(['git', 'show', rec['commit'] + ':' + rel], cwd=Path(__file__).resolve().parents[4])
    assert committed.replace(b'\r\n', b'\n') == (root / 'PLAN.json').read_bytes().replace(b'\r\n', b'\n')
    assert json.loads(committed) == p
    return p, rec


def public_data(root):
    """Independent official sequence, isolation, gate receipts and windows."""
    manifest, records = load(root / 'dataset_manifest.json'), load(root / 'candidate_gate_records.json')
    with (root / 'official/metadata.csv').open(newline='') as stream:
        metadata = list(csv.DictReader(stream))
    entries = manifest['entries']; consumed = set(map(str, load(root / 'consumed_inventory.json')['visit_ids']))
    p = load(root / 'PLAN.json')
    assert sha(root / 'official/metadata.csv') == p['data']['metadata_sha256']
    previous_root = Path(p['inherited']['v3_train_seal_path']).parent.parent
    previous_rows = load(previous_root / 'public_roster.json')['rows']
    previous_failed = {str(r['visit_id']) for r in load(previous_root / 'candidate_gate_records.json') if r['status'] == 'SKIP_REFERENCE_GATE'}
    assert (set(p['inherited']['training_visits']) | {str(r['visit_id']) for r in previous_rows} | previous_failed) <= consumed
    assert len(entries) == 20 and len(set(e['visit_id'] for e in entries)) == 20
    assert not consumed & set(e['visit_id'] for e in entries)
    recs = {r['capture']: r for r in records}; indices = [recs[e['capture']]['metadata_index'] for e in entries]
    assert indices == sorted(indices)
    attempted = [r['metadata_index'] for r in records]
    assert attempted == sorted(attempted)
    rejected = {r['visit_id'] for r in records if r['status'] == 'SKIP_REFERENCE_GATE'}
    assert not rejected & set(e['visit_id'] for e in entries)
    for ordinal, e in enumerate(entries):
        r = recs[e['capture']]; m = metadata[r['metadata_index']]
        role = 'cal' if ordinal < 4 else 'eval'
        assert m['video_id'] == e['capture'] and m['visit_id'] == e['visit_id']
        assert m['is_in_upsampling'] == 'True' and m['visit_id'] != 'NA'
        assert e['role'] == r['role'] == role and e['accepted_ordinal'] == ordinal
        assert r['status'] == 'ACCEPTED' and r['near_POS'] >= 16 and r['midfar_strict_FREE'] >= 32
        s_path = root / 'source' / e['capture'] / 'selection.json'; s = load(s_path)
        for earlier in records:
            if earlier['visit_id'] != e['visit_id'] or earlier['metadata_index'] >= r['metadata_index']:
                continue
            earlier_path = root / 'source' / earlier['capture'] / 'selection.json'
            assert not earlier_path.exists() or len(load(earlier_path)['rows']) < 32
        assert sha(s_path) == r['selection_sha256'] and s['selected_before_labels']
        assert s['plan_sha256'] == sha(root / 'PLAN.json') and s['FARO_not_selection_input']
        times = np.asarray(s['native_common_timestamps_s']); grid = times[0] + np.arange(int((times[-1] - times[0]) / .2) + 1) * .2
        right = np.searchsorted(times, grid).clip(0, len(times) - 1); left = np.maximum(right - 1, 0)
        nearest = np.where(abs(times[left] - grid) <= abs(times[right] - grid), left, right)
        good = abs(times[nearest] - grid) <= .10000001
        windows, used, start = [], set(), 0
        while start + 16 <= len(grid) and len(windows) < 2:
            ix = nearest[start:start + 16]
            if good[start:start + 16].all() and len(set(ix)) == 16 and not set(ix) & used:
                windows.append(list(zip(grid[start:start + 16], times[ix]))); used.update(ix); start += 16
            else:
                start += 1
        assert len(windows) == 2 and len(e['frames']) == len(s['rows']) == 32
        for fr, selected in zip(e['frames'], s['rows']):
            stamp = windows[fr['window_id']][fr['window_frame']]
            assert fr['role'] == role and (fr['grid_timestamp_s'], fr['rgb_timestamp_s']) == stamp
            assert (selected['grid_timestamp_s'], selected['rgb_timestamp_s']) == stamp
    return dict(status='PASS', admitted_visits=20, calibration_visits=4, evaluation_visits=16,
                frames=640, preselected_windows_verified=40, historical_visits_excluded=len(consumed),
                reference_gate_receipts_verified=20, new_eval_reference_reads=0)


def protocol_review(root, protocol):
    p = load(protocol)
    assert p['task'] == 'SYNC_FUSION_CONFIRM_V4_DEV_20261011'
    assert p['phase'] == 'PRE_REGISTERED_DEVELOPMENT_CONFIRM'
    assert tuple(p['features']['names']) == FEATURES
    assert (p['data']['new_visits'], p['data']['cal_visits'], p['data']['eval_visits'], p['data']['frames_per_visit']) == (20, 4, 16, 32)
    assert 'near POS>=16 and midfar strict FREE>=32' in p['data']['admission_gate']
    assert p['budget']['CPU_command_wall_s'] == 2400 and p['budget']['GPU_wall_s'] == 600 and p['budget']['download_bytes'] == 4000000000
    assert p['calibration']['band_targets'] == dict(zip(BANDS, (.02, .05, .1)))
    assert tuple(p['scores']['methods']) == METHODS
    assert '7dc9bbba' in json.dumps(p) and 'b9f227e5' in json.dumps(p)
    assert p['criterion_origin']['source'] == 'Selected after viewing v3 matched-FREE posthoc analysis'
    assert p['criterion_origin']['no_arbitrator']
    assert p['models']['seeds'] == [955, 956, 957]
    for k, v in HGB.items():
        assert p['models']['HGB'][k] == v
    assert p['models']['HGB']['class_weight'] is None
    assert 'Bytewise identical v3' in p['models']['final']
    hashes(p['models']['model_entries'])
    v3 = load(p['inherited']['v3_train_seal_path'])
    assert sha(p['inherited']['v3_train_seal_path']) == p['inherited']['v3_train_seal_sha256']
    assert p['models']['model_entries'] == [e for e in v3['model_entries'] if e['method'] == 'D_prime']
    cf = p['crossfit']
    assert (cf['old_visits'], cf['folds'], cf['seed']) == (30, 5, 20261011)
    assert 'array_split' in cf['fold_assignment'] and 'n_known/(6*n_arm_known)' in cf['fit']
    assert 'maximize query W on POS, then minimize F, then choose highest threshold' in p['calibration']['objective']
    c = p['confirmation']
    assert (c['primary_method'], c['primary_source'], c['primary_band']) == ('D_prime', ARMS[0], BANDS[1])
    assert c['pauc']['range'] == [0, .05] and 'Right-continuous' in c['pauc']['curve']
    assert 'nointerp nor ties splitting' in c['pauc']['ties'] and 'Not trapezoidal ROC' in c['pauc']['curve']
    assert 'bothD_prime-minus-rgb andD_prime-minus-or' in c['criterion'] and 'all2000primary drawsvalid' in c['criterion']
    assert (p['eval']['bootstrap']['replicates'], p['eval']['bootstrap']['seed']) == (2000, 20261011)
    if (root / 'PLAN.json').exists():
        assert sha(root / 'PLAN.json') == sha(protocol)
    return dict(status='PASS', protocol_sha256=sha(protocol), new_candidate_reads=0, new_eval_reference_reads=0,
                exact_step_toys=synthetic())


def preeval(root):
    import joblib
    from threadpoolctl import threadpool_limits
    assert not (root / 'eval/eval_open.json').exists(), 'PreOPEN audit cannot follow OPEN'
    p, rec = binding(root)
    os = load(root / 'oof/oof_seal.json'); ca = load(root / 'cal/cal_seal.json')
    assert os['plan_sha256'] == ca['plan_sha256'] == rec['plan_sha256']
    assert ca['oof_seal_sha256'] == sha(root / 'oof/oof_seal.json')
    assert os['frozen_model_entries'] == ca['model_entries'] == p['models']['model_entries']
    for seal in (os, ca):
        hashes(seal['dependencies']); hashes(seal['model_entries'])
        assert sha(seal['cache_path']) == seal['cache_sha256']
    hashes(os['frozen_model_entries'])
    assert sha(os['training_features_path']) == os['training_features_sha256'] == p['inherited']['training_features_sha256']
    d, old, cal = npz(os['training_features_path']), npz(os['cache_path']), npz(ca['cache_path'])
    assert tuple(d['feature_names'][:16]) == FEATURES and d['X'].shape[1] == 20
    X, y = d['X'][:, :16], d['y']
    assert np.isfinite(X).all()
    visits = sorted(set(map(str, d['visit_id'])))
    assert len(visits) == 30 and visits == os['visit_ids'] == ca['old_visit_ids'] == p['inherited']['training_visits']
    folds = [list(map(str, v)) for v in np.array_split(np.random.default_rng(20261011).permutation(visits), 5)]
    assert len(os['model_entries']) == 15 and len(os['folds']) == 5
    expected_oof = np.full(len(y), np.nan)
    gate = (d['tof_score'] >= 0) | (d['rgb_score'] >= 0)
    with threadpool_limits(limits=1):
        for fi, held in enumerate(folds):
            fr = os['folds'][fi]; hold = np.isin(d['visit_id'], held); train = (y >= 0) & ~hold
            assert fr['fold'] == fi and fr['held_visit_ids'] == held
            assert fr['train_visit_ids'] == [v for v in visits if v not in held]
            assert fr['train_known_rows'] == int(train.sum()) and fr['hold_rows'] == int(hold.sum())
            assert not set(fr['held_visit_ids']) & set(fr['train_visit_ids'])
            for a, arm_row in zip(ARMS, fr['arm_counts']):
                na = int((train & (d['arm'] == a)).sum())
                close(arm_row, dict(arm=a, known=na, weight_each=train.sum() / (6 * na), weight_sum=train.sum() / 6))
            entries = sorted([e for e in os['model_entries'] if e['fold'] == fi], key=lambda e: e['seed'])
            assert [e['seed'] for e in entries] == [955, 956, 957]
            predictions = []
            for e in entries:
                model = joblib.load(e['path'])
                assert model.n_features_in_ == 16 and model.classes_.tolist() == [0, 1]
                for k, value in HGB.items():
                    assert model.get_params()[k] == value
                assert model.random_state == e['seed'] and model.class_weight is None
                predictions.append(model.predict_proba(X[hold])[:, 1])
            expected_oof[hold] = np.where(gate[hold], np.mean(predictions, axis=0), -np.inf)
        frozen = np.mean([joblib.load(e['path']).predict_proba(X)[:, 1] for e in os['frozen_model_entries']], axis=0)
    np.testing.assert_allclose(old['D_prime_score'], expected_oof, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(old['D_prime_frozen_score'], np.where(gate, frozen, -np.inf), rtol=1e-12, atol=1e-12)
    for k in ('arm', 'visit_id', 'frame_id', 'query_id', 'band'):
        np.testing.assert_array_equal(old[k], d[k])
    np.testing.assert_array_equal(old['label'], y)
    assert set(old['cal_origin']) == {'old30_oof'}
    for m, expected in [('rgb', d['rgb_score']), ('tof', d['tof_score']),
                        ('or', np.maximum(d['tof_score'], d['rgb_score'])), ('and', np.minimum(d['tof_score'], d['rgb_score']))]:
        np.testing.assert_array_equal(old[m + '_score'], expected)
    f = npz(ca['feature_path'])
    assert sha(ca['feature_path']) == ca['feature_sha256'] and sha(ca['reference_path']) == ca['reference_sha256']
    assert tuple(f['feature_names']) == FEATURES and f['X'].shape == (len(f['arm']), 16)
    refs = {}; new_visits = set()
    for row in load(ca['reference_path'])['rows']:
        assert row.get('role', row.get('split')) == 'cal'
        new_visits.add(str(row['visit_id']))
        fid = str(row.get('source_id', row.get('frame_id')))
        for q in row['queries']:
            if q['state'] == 'FREE_ON_SAMPLED_RAYS':
                assert q['unknown_pixels'] == q['positive_pixels'] == 0 and q['free_ray_pixels'] == q['domain_pixels']
            key = fid, str(q.get('name', q.get('query_id')))
            assert key not in refs
            refs[key] = {'POSITIVE': 1, 'FREE_ON_SAMPLED_RAYS': 0, 'UNKNOWN': -1}[q['state']]
    assert len(new_visits) == 4 and sorted(new_visits) == ca['cal_visit_ids'] and not new_visits & set(visits)
    ny = np.array([refs[str(fr), str(q)] for fr, q in zip(f['frame_id'], f['query_id'])])
    ngate = (f['tof_score'] >= 0) | (f['rgb_score'] >= 0)
    with threadpool_limits(limits=1):
        ns = np.where(ngate, np.mean([joblib.load(e['path']).predict_proba(f['X'])[:, 1] for e in ca['model_entries']], axis=0), -np.inf)
    new = {k: f[k] for k in ('arm', 'visit_id', 'frame_id', 'query_id', 'band')}
    new.update(label=ny, cal_origin=np.array(['new4_frozen'] * len(ny)), D_prime_score=ns,
               rgb_score=f['rgb_score'], tof_score=f['tof_score'],
               or_score=np.maximum(f['tof_score'], f['rgb_score']), and_score=np.minimum(f['tof_score'], f['rgb_score']))
    for k in new:
        expected = np.concatenate((old[k], new[k]))
        if np.issubdtype(expected.dtype, np.number):
            np.testing.assert_allclose(cal[k], expected, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(cal[k], expected)
    checked = 0
    assert len(ca['cells']) == 15
    for cell in ca['cells']:
        m, b = cell['method'], cell['band']; take = cal['band'] == b
        i, t, c, status = choose(cal[m + '_score'][take], cal['label'][take], p['calibration']['band_targets'][BANDS[b]])
        assert threshold(cell) == t[i] and cell['status'] == status
        assert (cell['W'], cell['F'], cell['U']) == tuple(c[i])
        assert (cell['POS_denominator'], cell['FREE_denominator'], cell['UNKNOWN_denominator']) == tuple(int((cal['label'][take] == j).sum()) for j in (1, 0, -1))
        assert sha(cell['all_ties_path']) == cell['all_ties_sha256']
        rows = load(cell['all_ties_path'])['rows']; assert len(rows) == len(t)
        for j, row in enumerate(rows):
            assert threshold(row) == t[j] and (row['W'], row['F'], row['U']) == tuple(c[j])
        checked += len(t)
    decisions = check_predictions(cal, ca['cells'])
    fm = load(root / 'feature_manifest.json')
    assert not fm['reference_read'] and tuple(fm['feature_names']) == FEATURES
    hashes(fm['inputs']); hashes(fm['outputs']); hashes(fm['simulations'])
    data = public_data(root)
    evalentry = next(e for e in fm['outputs'] if e['role'] == 'eval')
    with np.load(evalentry['path'], allow_pickle=False) as z:
        ev = {k: z[k] for k in ('arm', 'visit_id', 'frame_id', 'query_id', 'band')}
    evvisits = set(map(str, ev['visit_id']))
    assert len(evvisits) == 16 and not evvisits & (set(visits) | new_visits)
    for v in evvisits:
        take = (ev['visit_id'] == v) & (ev['arm'] == ARMS[0])
        assert take.sum() == 32 * 27 and len(set(ev['frame_id'][take])) == 32
    faro_gate = input_gate(root, f, ev)
    distribution_check(root / 'cal/oof_vs_frozen_distribution.csv', old)
    from audit_sync_fusion_confirm_v3 import sensor_samples
    sensors = sensor_samples(root, f)
    return dict(status='PASS', protocol_commit=rec['commit'], oof_models_verified=15,
                frozen_models_byte_verified=3, fold_isolation_verified=5, six_arm_weighted_training_sets_verified=5,
                oof_predictions_recomputed=len(y), calibration_rows=len(cal['label']),
                calibration_visits=34, new_calibration_visits=4, exhaustive_cells=15, exhaustive_ties_checked=checked,
                cal_decisions_checked=decisions, data_integrity=data, sensor_samples=sensors,
                input_FARO_gate=faro_gate,
                cal_seal_sha256=sha(root / 'cal/cal_seal.json'), oof_seal_sha256=sha(root / 'oof/oof_seal.json'),
                distribution_quantiles_recomputed=True, new_eval_reference_reads=0, new_eval_model_score_reads=0)


def input_gate(root, cal, ev):
    """Recompute all FARO K0 joint quality gates from observations only."""
    rows = csvrows(root / 'input_coverage_gate.csv')
    reported = {(r['frame_id'], r['arm'], int(r['K'])): int(r['joint_pass_zones']) for r in rows}
    frames = load(root / 'synthesis_manifest.json')['frames']
    assert len(frames) == 640
    admitted = {'cal': set(), 'eval': set()}
    for frame in frames:
        entry = next(e for e in frame['arms'] if e['arm'] == ARMS[4] and e['repeat'] == 0)
        with np.load(entry['path'], allow_pickle=False) as z:
            hist, bg, coverage = z['hist'], z['background'], z['coverage']
        peak = hist.argmax(-1)
        height = np.take_along_axis(hist, peak[..., None], axis=-1)[..., 0]
        background = np.take_along_axis(bg, peak[..., None], axis=-1)[..., 0]
        snr = height / np.sqrt(np.maximum(height, 0) + 2 * background + 1)
        n = int(((coverage >= .75) & (snr >= 3)).sum())
        assert n == reported[frame['frame_id'], ARMS[4], 0]
        if n >= 52: admitted[frame['role']].add(frame['source_id'])
    for role, data in (('cal', cal), ('eval', ev)):
        assert set(data['frame_id'][data['arm'] == ARMS[4]]) == admitted[role]
    return dict(status='PASS', K0_gate_frames_recomputed=640,
                calibration_pass_frames=len(admitted['cal']), evaluation_pass_frames=len(admitted['eval']),
                labels_or_model_scores_read=False)


def check_predictions(cache, cells):
    for m in METHODS:
        expected = np.zeros(len(cache['label']), bool)
        for cell in cells:
            if cell['method'] == m and cell['status'] == 'ADOPTED':
                take = cache['band'] == cell['band']
                expected[take] = cache[m + '_score'][take] >= threshold(cell)
        np.testing.assert_array_equal(cache[m + '_pred'], expected)
    return len(cache['label']) * len(METHODS)


def distribution_check(path, old):
    with path.open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7 * 3 * 4 * 2
    for r in rows:
        take = old['band'] == int(r['band'])
        if r['arm'] != 'POOLED_SIX_ARMS': take &= old['arm'] == r['arm']
        if r['label'] != 'ALL': take &= old['label'] == int(r['label'])
        score = old['D_prime_score' if r['score_source'] == 'OOF' else 'D_prime_frozen_score'][take]
        finite = score[np.isfinite(score)]
        assert int(r['rows']) == len(score) and int(r['finite_rows']) == len(finite) and int(r['disabled_rows']) == np.isneginf(score).sum()
        for q in (0, .05, .25, .5, .75, .95, 1):
            key = f'q{round(q * 100):02}'
            if len(finite): close(float(r[key]), float(np.quantile(finite, q)), key)
            else: assert r[key] == ''
        valid = take & np.isfinite(old['D_prime_score']) & np.isfinite(old['D_prime_frozen_score'])
        delta = old['D_prime_score'][valid] - old['D_prime_frozen_score'][valid]
        if len(delta): close(float(r['paired_OOF_minus_frozen_mean']), float(delta.mean()))
        else: assert r['paired_OOF_minus_frozen_mean'] == ''


def posteval(root):
    p, rec = binding(root)
    opened, terminal, seal = [load(root / 'eval' / name) for name in ('eval_open.json', 'eval_terminal.json', 'cache_seal.json')]
    ca = load(root / 'cal/cal_seal.json'); summary = load(root / 'summary/summary.json')
    pre = load(root / 'independent_preeval_audit.json')
    assert pre['status'] == 'PASS' and pre['cal_seal_sha256'] == sha(root / 'cal/cal_seal.json')
    assert opened['one_shot'] and opened['plan_sha256'] == rec['plan_sha256']
    assert terminal['status'] == 'COMPLETE_CACHE_FROZEN' and not terminal['reopen_permitted']
    assert opened['cal_seal_sha256'] == seal['cal_seal_sha256'] == sha(root / 'cal/cal_seal.json')
    assert seal['eval_open_sha256'] == sha(root / 'eval/eval_open.json')
    assert sha(seal['cache_path']) == seal['cache_sha256'] == summary['eval_cache_sha256']
    assert sha(ca['cache_path']) == ca['cache_sha256'] == summary['cal_cache_sha256']
    assert sha(root / 'eval/per_query.csv') == seal['per_query_sha256']
    hashes(ca['dependencies']); hashes(opened['dependencies']); hashes(ca['model_entries'])
    for c in ca['cells']: assert sha(c['all_ties_path']) == c['all_ties_sha256']
    cache, cal = npz(seal['cache_path']), npz(ca['cache_path'])
    n = len(cache['label']); assert seal['rows'] == n and all(len(a) == n for a in cache.values())
    assert len(set(zip(cache['arm'], cache['frame_id'], cache['query_id']))) == n
    roster = list(map(str, opened['visit_ids']))
    assert len(roster) == len(set(roster)) == 16 and roster == seal['visit_ids']
    assert set(cache['visit_id']) == set(roster) and not set(roster) & (set(ca['old_visit_ids']) | set(ca['cal_visit_ids']))
    for v in roster:
        take = (cache['visit_id'] == v) & (cache['arm'] == ARMS[0])
        assert int(take.sum()) == 32 * 27 and len(set(cache['frame_id'][take])) == 32
        near = take & (cache['band'] == 0); midfar = take & (cache['band'] != 0)
        assert int((cache['label'][near] == 1).sum()) >= 16 and int((cache['label'][midfar] == 0).sum()) >= 32
    assert set(cache['label']) <= {-1, 0, 1}
    np.testing.assert_array_equal(cache['or_score'], np.maximum(cache['tof_score'], cache['rgb_score']))
    np.testing.assert_array_equal(cache['and_score'], np.minimum(cache['tof_score'], cache['rgb_score']))
    gate = (cache['tof_score'] >= 0) | (cache['rgb_score'] >= 0)
    assert np.isneginf(cache['D_prime_score'][~gate]).all()
    assert np.isfinite(cache['D_prime_score'][gate]).all() and ((cache['D_prime_score'][gate] >= 0) & (cache['D_prime_score'][gate] <= 1)).all()
    decisions = check_predictions(cache, ca['cells'])
    v3ca = load(p['inherited']['v3_cal_seal_path'])
    assert sha(p['inherited']['v3_cal_seal_path']) == p['inherited']['v3_cal_seal_sha256']
    for m in METHODS:
        expected = np.zeros(n, bool)
        for c in v3ca['cells']:
            if c['method'] == m and c['status'] == 'ADOPTED':
                take = cache['band'] == c['band']; expected[take] = cache[m + '_score'][take] >= threshold(c)
        np.testing.assert_array_equal(cache[m + '_v3_pred'], expected)
    with (root / 'eval/per_query.csv').open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream); assert reader.fieldnames == list(cache)
        count = 0
        for i, r in enumerate(reader):
            for k, a in cache.items():
                value = a[i].item()
                if isinstance(value, (int, float)) and not isinstance(value, bool): assert float(r[k]) == value
                else: assert r[k] == str(value)
            count += 1
        assert count == n
    draws = np.random.default_rng(20261011).integers(0, 16, (2000, 16))
    mult = np.column_stack([(draws == k).sum(1) for k in range(16)])
    saved_boot = npz(root / 'summary/bootstrap_replicates.npz')
    np.testing.assert_array_equal(saved_boot['draws'], draws)
    metrics = {(r['arm'], r['band'], r['method']): r for r in summary['pAUC']}
    deployment = {(r['arm'], r['band'], r['method']): r for r in summary['deployment']}
    pairs = {(r['arm'], r['band'], r['comparator']): r for r in summary['paired']}
    assert len(metrics) == len(deployment) == 90 and len(pairs) == 72
    rows_curve = csvrows(root / 'summary/curves.csv')
    rows_boot = csvrows(root / 'summary/bootstrap.csv')
    rows_visit = csvrows(root / 'summary/visit_pauc.csv')
    expected_curves, expected_boots, expected_visits = [], [], []
    for a in ARMS:
        for b in range(3):
            take = (cache['arm'] == a) & (cache['band'] == b)
            y, v = cache['label'][take], cache['visit_id'][take]
            P, N = int((y == 1).sum()), int((y == 0).sum())
            pd = np.array([int(((v == visit) & (y == 1)).sum()) for visit in roster])
            fd = np.array([int(((v == visit) & (y == 0)).sum()) for visit in roster])
            scalar, sampled = {}, {}
            for m in METHODS:
                t, c, cl = ties(cache[m + '_score'][take], y, v, roster)
                value = area(c, P, N); scalar[m] = value
                close(metrics[a, b, m], dict(POS=P, FREE=N, normalized_pAUC=value,
                                            status='EVALUABLE' if P and N else 'NOT_EVALUABLE'))
                sampled[m] = bootstrap(cl, mult, pd, fd)
                np.testing.assert_allclose(saved_boot[f'{a}__band{b}__{m}'], sampled[m], rtol=1e-11, atol=1e-12, equal_nan=True)
                # Last tie at each exact F gives max W with no interpolation.
                last = np.r_[np.flatnonzero(np.diff(c[:, 1])), len(c) - 1]
                for j in last:
                    expected_curves.append(dict(arm=a, band=b, method=m, POS=P, FREE=N,
                        W=int(c[j, 0]), F=int(c[j, 1]), U=int(c[j, 2]),
                        W_rate=float(c[j, 0] / P) if P else None, F_rate=float(c[j, 1] / N) if N else None,
                        threshold=float(t[j]) if np.isfinite(t[j]) else None))
                ds = stats(cache, take, m)
                close(deployment[a, b, m], ds)
                for vi, visit in enumerate(roster):
                    cc = np.c_[cl[:, vi, :], np.zeros(len(t), int)]
                    expected_visits.append(dict(arm=a, band=b, method=m, visit_id=visit,
                        POS=int(pd[vi]), FREE=int(fd[vi]), normalized_pAUC=area(cc, pd[vi], fd[vi])))
            for m in ('rgb', 'tof', 'or', 'and'):
                d, z = cache['D_prime_pred'][take], cache[m + '_pred'][take]
                ds, ms = deployment[a, b, 'D_prime'], deployment[a, b, m]
                close(pairs[a, b, m], dict(rescue=int(((y == 1) & d & ~z).sum()), loss=int(((y == 1) & ~d & z).sum()),
                    delta_W=ds['W'] - ms['W'], delta_F=ds['F'] - ms['F'],
                    FREE_added=int(((y == 0) & d & ~z).sum()), FREE_removed=int(((y == 0) & ~d & z).sum()),
                    deployment_bidirectional_dominance=bool(P and N and ds['W'] >= ms['W'] and ds['F'] <= ms['F'])))
            for m in ('rgb', 'or'):
                delta = sampled['D_prime'] - sampled[m]; ci = interval(delta)
                expected_boots.append(dict(arm=a, band=b, comparator=m, POS=P, FREE=N,
                    delta_pAUC=scalar['D_prime'] - scalar[m] if scalar['D_prime'] is not None else None,
                    **ci, pass_lower_gt_zero=bool(ci['lower'] is not None and ci['lower'] > 0),
                    all_2000_valid=ci['valid_replicates'] == 2000))
            print('independent', a, b, flush=True)
    csvcheck(rows_curve, expected_curves)
    csvcheck(rows_boot, expected_boots)
    csvcheck(rows_visit, expected_visits)
    primary = [r for r in expected_boots if r['arm'] == ARMS[0] and r['band'] == 1]
    for actual, expected in zip(summary['primary'], primary): close(actual, expected)
    evaluable = len(primary) == 2 and all(r['valid_replicates'] == 2000 for r in primary)
    passed = evaluable and all(r['pass_lower_gt_zero'] for r in primary)
    assert summary['status'] == ('PASS' if passed else 'FAIL' if evaluable else 'NOT_EVALUABLE')
    if passed: assert summary['conclusion'] == p['confirmation']['pass_claim']
    verify_drift(root, p, cache, cal, v3ca)
    return dict(status='PASS', eval_cache_sha256=seal['cache_sha256'], summary_sha256=sha(root / 'summary/summary.json'),
                cached_rows=n, per_query_CSV_rows_verified=n, eval_reference_gate_recomputed_visits=16,
                deployed_decisions_checked=decisions, v3_cut_decisions_checked=decisions,
                pAUC_cells_verified=90, bootstrap_curves_verified=90, bootstrap_replicates_each=2000,
                exact_F_curve_rows_verified=len(expected_curves), bootstrap_comparisons_verified=36,
                visit_pAUC_rows_verified=len(expected_visits), deployment_pairs_verified=72,
                drift_all_roles_and_visits_verified=True, primary=primary, confirmation_status=summary['status'],
                conclusion=summary['conclusion'], new_eval_reference_reads=0, model_inference_reruns=0)


def stats(cache, take, m, suffix='_pred'):
    y, pred = cache['label'][take], cache[m + suffix][take]
    W, F, U = [int((pred & (y == j)).sum()) for j in (1, 0, -1)]
    P, N, Q = [int((y == j).sum()) for j in (1, 0, -1)]
    return dict(W=W, F=F, U=U, POS=P, FREE=N, UNKNOWN=Q,
                W_rate=float(W / P) if P else None, F_rate=float(F / N) if N else None)


def csvrows(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def csvcheck(actual, expected):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for r, e in zip(actual, expected):
        for k, value in e.items():
            if value is None: assert r[k] == '', (k, r[k])
            elif isinstance(value, bool): assert r[k] == str(value), (k, r[k], value)
            elif isinstance(value, (float, int, np.number)): close(float(r[k]), value, k)
            else: assert r[k] == str(value), (k, r[k], value)


def verify_drift(root, p, cache, cal, v3ca):
    v3cal, v3eval = npz(v3ca['cache_path']), npz(Path(v3ca['cache_path']).parent.parent / 'eval/cache.npz')
    expected, visits = [], []
    for a in ARMS:
        for b in range(3):
            for m in METHODS:
                for role, cc in (('v4_pooled_cal', cal), ('v4_old30_oof', cal), ('v4_new4_frozen', cal),
                                 ('v4_eval16', cache), ('v3_cal6', v3cal), ('v3_eval12', v3eval), ('v3_cuts_on_v4_eval16', cache)):
                    take = (cc['arm'] == a) & (cc['band'] == b)
                    if role == 'v4_old30_oof': take &= cc['cal_origin'] == 'old30_oof'
                    if role == 'v4_new4_frozen': take &= cc['cal_origin'] == 'new4_frozen'
                    suffix = '_v3_pred' if role == 'v3_cuts_on_v4_eval16' else '_pred'
                    expected.append(dict(arm=a, band=b, method=m, role=role, **stats(cc, take, m, suffix)))
                    for v in sorted(set(cc['visit_id'][take])):
                        vm = take & (cc['visit_id'] == v)
                        visits.append(dict(arm=a, band=b, method=m, role=role, visit_id=str(v), **stats(cc, vm, m, suffix)))
    for b in range(3):
        for m in METHODS:
            for role, cc in (('v4_pooled_cal', cal), ('v4_old30_oof', cal), ('v4_new4_frozen', cal),
                             ('v4_eval16', cache), ('v3_cal6', v3cal), ('v3_eval12', v3eval), ('v3_cuts_on_v4_eval16', cache)):
                take = cc['band'] == b
                if role == 'v4_old30_oof': take &= cc['cal_origin'] == 'old30_oof'
                if role == 'v4_new4_frozen': take &= cc['cal_origin'] == 'new4_frozen'
                suffix = '_v3_pred' if role == 'v3_cuts_on_v4_eval16' else '_pred'
                expected.append(dict(arm='POOLED_SIX_ARMS', band=b, method=m, role=role, **stats(cc, take, m, suffix)))
    csvcheck(csvrows(root / 'summary/drift.csv'), expected)
    csvcheck(csvrows(root / 'summary/drift_visit.csv'), visits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('protocol', 'preeval', 'posteval'))
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); started = time.perf_counter()
    result = protocol_review(args.root, args.protocol) if args.stage == 'protocol' else globals()[args.stage](args.root)
    result.update(CPU_command_wall_s=time.perf_counter() - started, GPU_s=0, protected_480_test_reads=0,
                  audit_source_sha256=sha(__file__),
                  independent_implementation=True, eval_reference_reopens=0, model_refits=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
