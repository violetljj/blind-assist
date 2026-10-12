"""Independent consumed-Development CV audit; never fits or opens protected data.

The audit recomputes operating points, paired events and visit-cluster intervals
from retained observation/calibration/OOF payloads. It deliberately does not
import the run's fitting, threshold-selection or aggregation implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cutoff(row):
    return np.inf if row['threshold_kind'] == 'positive_infinity' else row['threshold']


def independent_ties(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    finite = np.flatnonzero(np.isfinite(scores))
    order = finite[np.argsort(-scores[finite], kind='stable')]
    values = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(values) != 0) + 1, len(order)] if len(order) else []
    counts = np.cumsum(np.stack([labels[order] == i for i in (1, 0, -1)], axis=1), axis=0)
    output = [(np.inf, 0, 0, 0)]
    output.extend((float(values[e-1]), *map(int, counts[e-1])) for e in ends)
    return output


def independent_select(scores, labels, target):
    labels = np.asarray(labels, int)
    ties = independent_ties(scores, labels)
    nf, np_ = int(sum(labels == 0)), int(sum(labels == 1))
    if not nf:
        return ties[0], 'NOT_CALIBRATABLE'
    if not np_:
        return ties[0], 'NO_CAL_POS'
    legal = [r for r in ties if r[2] <= int(np.floor(target * nf))]
    return max(legal, key=lambda r: (r[1], -r[2], r[0])), 'ADOPTED'


def interval(values, total):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(lower=float(np.quantile(values, .025)) if len(values) else None,
                upper=float(np.quantile(values, .975)) if len(values) else None,
                valid_replicates=len(values), total_replicates=total)


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=np.asarray(b) > 0)


def assert_close(actual, expected, path=''):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, (path, 'missing', key)
            assert_close(actual[key], value, path + '/' + key)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (path, len(actual), len(expected))
        for index, value in enumerate(expected):
            assert_close(actual[index], value, path + '/' + str(index))
    elif isinstance(expected, (float, np.floating)):
        assert actual is not None and np.isclose(actual, expected, atol=1e-10, rtol=1e-10), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def recompute_cell(labels, predictions, visits, roster, draw):
    labels = np.asarray(labels, int)
    predictions = np.asarray(predictions, bool)
    visits = np.asarray(visits, str)
    denominator = np.array([[sum((visits == v) & (labels == label)) for label in (1, 0, -1)]
                            for v in roster], int)
    numerator = np.array([[sum((visits == v) & (labels == label) & predictions) for label in (1, 0, -1)]
                          for v in roster], int)
    den, count = denominator.sum(0), numerator.sum(0)
    bd, bn = denominator[draw].sum(1), numerator[draw].sum(1)
    return dict(W=int(count[0]), POS=int(den[0]), F=int(count[1]), FREE=int(den[1]),
                U=int(count[2]), UNKNOWN=int(den[2]),
                W_rate=float(count[0] / den[0]) if den[0] else None,
                F_rate=float(count[1] / den[1]) if den[1] else None,
                W_rate_CI=interval(ratio(bn[:, 0], bd[:, 0]), len(draw)),
                F_rate_CI=interval(ratio(bn[:, 1], bd[:, 1]), len(draw)))


def recompute_paired(labels, predictions, comparator, visits, roster, draw):
    labels = np.asarray(labels, int)
    predictions, comparator = np.asarray(predictions, bool), np.asarray(comparator, bool)
    visits = np.asarray(visits, str)
    rescue = (labels == 1) & predictions & ~comparator
    loss = (labels == 1) & ~predictions & comparator
    delta = int(sum(rescue) - sum(loss))
    den = int(sum(labels == 1))
    byvisit = np.array([sum(rescue & (visits == v)) - sum(loss & (visits == v)) for v in roster])
    bd = np.array([sum((labels == 1) & (visits == v)) for v in roster])[draw].sum(1)
    diff = byvisit[draw].sum(1)
    result = dict(rescue=int(sum(rescue)), loss=int(sum(loss)), delta_W=delta, POS=den,
                  delta_W_rate=float(delta / den) if den else None,
                  delta_W_CI=interval(diff, len(draw)),
                  delta_W_rate_CI=interval(ratio(diff, bd), len(draw)))
    for label, name in ((0, 'FREE'), (-1, 'UNKNOWN')):
        result[name + '_added'] = int(sum((labels == label) & predictions & ~comparator))
        result[name + '_removed'] = int(sum((labels == label) & ~predictions & comparator))
    return result


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def compact_ci(values):
    a = interval(values, len(values))
    return dict(lo=a['lower'], hi=a['upper'], valid=a['valid_replicates'])


def run(root, plan_path, budget_s):
    import joblib
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    def check():
        if time.monotonic() - started > budget_s - 3:
            raise TimeoutError('Independent audit command-wall allocation')

    plan = load(plan_path)
    assert sha(root / 'PLAN.json') == sha(plan_path)
    current_source = Path(__file__).with_name('sync_fusion_robust_dev.py')
    source_sha256 = sha(current_source)
    assert sha(root/'executed_source.py') == source_sha256
    for stage in ('build', 'fit', 'summarize'):
        terminal = load(root/(stage+'_terminal.json'))
        assert terminal['status'] == 'COMPLETE' and terminal['GPU_s'] == 0
        assert terminal['source_sha256'] == source_sha256
    assert plan['budget']['CPU_command_wall_s'] == 1800
    assert plan['budget']['GPU_wall_s'] == 300
    assert plan['data']['protected_access'] is False
    roster = plan['data']['visits']
    assert len(roster) == len(set(roster)) == 30
    folds = plan['data']['folds']
    assert len(folds) == 5
    all_eval = []
    for fold in folds:
        tr, ca, ev = [set(fold[k]) for k in ('train', 'cal', 'eval')]
        assert (len(tr), len(ca), len(ev)) == (19, 5, 6)
        assert not tr & ca and not tr & ev and not ca & ev
        assert tr | ca | ev == set(roster)
        all_eval.extend(fold['eval'])
    assert sorted(all_eval) == sorted(roster)
    for binding in plan['data']['bindings']:
        assert sha(binding['path']) == binding['sha256']
    d = read_npz(root / 'features.npz')
    o = read_npz(root / 'oof.npz')
    manifest = load(root / 'feature_manifest.json')
    summary = load(root / 'summary.json')
    assert sha(root / 'features.npz') == manifest['features_sha256'] == summary['features_sha256']
    assert sha(root / 'oof.npz') == summary['oof_sha256']
    assert sha(plan_path) == manifest['PLAN_sha256'] == summary['PLAN_sha256']
    names, models, arms = tuple(d['feature_names']), tuple(o['models']), tuple(plan['arms'])
    assert names == tuple(plan['features']['base']) and len(names) == 20
    assert models == ('A', 'B', 'C', 'D', 'rgb', 'tof')
    assert len(arms) == 6
    n = len(d['y'])
    assert d['X'].shape == (n, 20) and np.isfinite(d['X']).all()
    assert set(d['y']) == {-1, 0, 1}
    assert not np.isnan(o['scores']).any() and not np.isposinf(o['scores']).any()
    assert o['scores'].shape == o['pred'].shape == (n, 6)
    np.testing.assert_array_equal(o['indices'], np.arange(n))
    assert set(d['visit_id']) == set(roster)
    assert set(d['band']) == {0, 1, 2}
    np.testing.assert_array_equal(d['band'], d['X'][:, 16])
    assert len(set(zip(d['arm'], d['frame_id'], d['query_id']))) == n
    native_ix = np.flatnonzero(d['arm'] == arms[0])
    assert len(native_ix) == 25920
    native_keys = {(d['frame_id'][i], d['query_id'][i]): i for i in native_ix}
    native_frames = set(d['frame_id'][native_ix])
    assert len(native_frames) == 960
    for v in roster:
        assert sum(d['visit_id'][native_ix] == v) == 864
    for arm in arms:
        ix = np.flatnonzero(d['arm'] == arm)
        keys = {(d['frame_id'][i], d['query_id'][i]) for i in ix}
        if arm != 'faro_rho030_ambient1':
            assert keys == set(native_keys)
        else:
            assert keys.issubset(native_keys)
        ni = np.array([native_keys[(d['frame_id'][i], d['query_id'][i])] for i in ix])
        for field in ('rgb_score', 'visit_id', 'band', 'y'):
            np.testing.assert_array_equal(d[field][ix], d[field][ni])
        np.testing.assert_array_equal(d['X'][ix, 10:], d['X'][ni, 10:])
    check()

    # FARO gate membership is recomputed from the inherited input gate, not labels.
    expected_faro = set()
    for binding in plan['data']['bindings']:
        if Path(binding['path']).name != 'input_coverage_gate.csv':
            continue
        parent = Path(binding['path']).parent
        sensor_to_rgb = {f['frame_id']: f['source_id'] for f in load(parent / 'synthesis_manifest.json')['frames']}
        with Path(binding['path']).open(newline='', encoding='utf8') as stream:
            for row in csv.DictReader(stream):
                if row['arm'] == 'faro_rho030_ambient1' and row['K'] == '0' and int(row['joint_pass_zones']) >= 52:
                    expected_faro.add(sensor_to_rgb[row['frame_id']])
    assert set(d['frame_id'][d['arm'] == 'faro_rho030_ambient1']) == expected_faro

    # Full frozen-feature parity for native/FARO arms, including old cal2.
    base = root.parent
    parity = 0
    for source_root, roles in ((base / 'sync-fusion-v1-dev-20261011', ('train', 'cal', 'eval')),
                               (base / 'sync-fusion-confirm-v2-dev-20261011', ('cal', 'eval'))):
        for arm in (arms[0], 'faro_rho030_ambient1'):
            ix = np.flatnonzero(d['arm'] == arm)
            lookup = {(d['frame_id'][i], d['query_id'][i]): i for i in ix}
            for role in roles:
                old = read_npz(source_root / 'features' / arm / (role + '.npz'))
                take = np.array([lookup[key] for key in zip(old['frame_id'], old['query_id'])])
                for field in ('X', 'rgb_score', 'tof_score'):
                    np.testing.assert_array_equal(d[field][take], old[field])
                parity += len(take)
    check()

    # Every regenerated arm's scale declaration, plus deterministic Poisson
    # reconstruction on a fixed sample; no feature/evaluation selection.
    specifications = {'faro_rho015_ambient1': (.5, 1), 'faro_rho060_ambient1': (2, 1),
                      'faro_rho030_ambient3': (1, 3), 'faro_rho030_ambient10': (1, 10)}
    samples = {}
    for row in manifest['simulations']:
        expected = specifications[row['arm']]
        assert (row['signal_scale'], row['ambient_scale']) == expected, row
        samples.setdefault(row['arm'], []).append(row)
    simulation_tests = 0
    for arm, rows in samples.items():
        for j in np.unique(np.linspace(0, len(rows)-1, min(6, len(rows)), dtype=int)):
            check(); row = rows[j]
            source, rebuilt = read_npz(row['source']), read_npz(row['path'])
            assert sha(row['source']) == row['source_sha256']
            assert sha(row['path']) == row['sha256']
            expectation = source['expectation'] * row['signal_scale']
            ambient = source['ambient'] * row['ambient_scale']
            np.testing.assert_array_equal(rebuilt['expectation'], expectation)
            np.testing.assert_array_equal(rebuilt['ambient'], ambient)
            rng = np.random.default_rng(row['seed'])
            bgmean = np.broadcast_to(8 * ambient[None, ..., None], expectation[None].shape)
            background = rng.poisson(bgmean)[0]
            counts = rng.poisson(expectation[None] + bgmean)[0]
            for field, value in (('hist', counts-background), ('counts', counts), ('background', background)):
                np.testing.assert_array_equal(rebuilt[field], value)
            for field in ('coverage', 'grid_K', 'grid_pose', 'rgb_pose', 'rgb_K'):
                np.testing.assert_array_equal(rebuilt[field], source[field])
            simulation_tests += 1
    print('audit features/simulation PASS', n, parity, simulation_tests, flush=True)

    # Independently reconstruct generated-arm query features on the same fixed
    # frames. Frozen observation geometry helpers are shared, but feature
    # reduction/missing encoding and SNR arithmetic are recomputed here.
    from sync_rgb_tof_dataset_v1 import peak_readout
    from rgb_body_query_reference_eval import rays, ray_interval
    public = {}
    queries = None
    readout_seal = None
    for binding in plan['data']['bindings']:
        path = Path(binding['path'])
        if path.name == 'public_roster.json':
            loaded = load(path)
            public.update({r['source_id']: r for r in loaded['rows']})
            if queries is None:
                queries = loaded['queries']
            else:
                assert queries == loaded['queries']
        if path.name == 'frozen_readout_seal.json':
            readout_seal = load(path)
    query_feature_tests = 0
    feature_lookup = {(d['arm'][i], d['frame_id'][i], d['query_id'][i]): i for i in range(n)}
    for arm, rows in samples.items():
        chosen_frames = sorted({Path(rows[j]['path']).parent.name for j in np.unique(np.linspace(0, len(rows)-1, min(6, len(rows)), dtype=int))})
        for frame in chosen_frames:
            check(); pub = public[frame]; K = np.asarray(pub['depth_K']); shape = tuple(pub['depth_shape'])
            rx, ry = rays(K, shape)
            depths, snrs, valid_fraction = [], [], []
            for repeat in (0, 1):
                observed = read_npz(root/'simulated'/frame/(arm+f'_k{repeat}.npz'))
                hist, bg = observed['hist'], observed['background']
                peak = np.argmax(hist, axis=-1)
                height = np.take_along_axis(hist, peak[..., None], -1)[..., 0]
                background = np.take_along_axis(bg, peak[..., None], -1)[..., 0]
                snr = height/np.sqrt(np.maximum(height, 0)+2*background+1)
                snrs.append(float(np.mean(snr)))
                if np.isfinite(observed['grid_pose']).all():
                    depths.append(peak_readout(hist, bg, observed['coverage'], observed['grid_K'], shape,
                                               observed['grid_pose'], K, shape, observed['rgb_pose'], readout_seal)[0])
                    valid_fraction.append(float(np.mean((observed['coverage']>=.75)&(snr>=3))))
                else:
                    depths.append(np.full(shape, np.inf)); valid_fraction.append(0.)
            for query in queries:
                entry, exit_, domain = ray_interval(rx, ry, query)
                values, scores = [], []
                for z in depths:
                    finite = domain & np.isfinite(z) & (z > 0)
                    nd, nf = int(sum(domain.ravel())), int(sum(finite.ravel()))
                    margin = np.minimum(z[finite]-entry[finite], exit_[finite]-z[finite])
                    score = float(np.sort(margin)[-16]) if len(margin)>=16 else -np.inf
                    scores.append(score)
                    support = int(sum(margin>=0))
                    values.append([score if np.isfinite(score) else -10., float(not np.isfinite(score)),
                                   float(support), 1-nf/nd if nd else 1.,
                                   float(sum(z[finite]<entry[finite]))/nd if nd else 0.,
                                   support/nd if nd else 0.,
                                   float(sum(z[finite]>exit_[finite]))/nd if nd else 0.,
                                   float(np.median(z[finite]-entry[finite])) if nf else -10.,
                                   float(np.median(z[finite]-exit_[finite])) if nf else -10.])
                valid = np.isfinite(scores)
                tof_score = float(np.mean(scores)) if all(valid) else -np.inf
                vector = np.mean(values, axis=0)
                vector[0] = tof_score if all(valid) else -10.
                vector[1] = float(not all(valid))
                vector = np.r_[vector, sum(valid)]
                for j, name in enumerate(names[:10]):
                    if name.endswith('_m'):
                        vector[j] = np.clip(vector[j], -10, 10)
                i = feature_lookup[(arm, frame, query['name'])]
                np.testing.assert_allclose(d['X'][i, :10], vector, atol=1e-12)
                assert d['tof_score'][i] == tof_score
                assert_close(d['peak_quality'][i], float(np.mean(snrs)))
                assert_close(d['accepted_zone_fraction'][i], float(np.mean(valid_fraction)))
                query_feature_tests += 1
    print('audit sampled query features PASS', query_feature_tests, flush=True)

    cuts = {(r['fold'], r['model'], r['band']): r for r in load(root / 'cuts.json')}
    entries = load(root / 'models_manifest.json')
    assert len(entries) == 60 and len(cuts) == 90
    coefficients = load(root / 'coefficients.json')
    coef_lookup = {(r['fold'], r['model'], r['seed'], r['feature']): r for r in coefficients}
    tie_count, prediction_count, weight_tests = 0, 0, 0
    scaler_numerics = []
    with threadpool_limits(limits=2):
        for fold in folds:
            f = fold['fold']; folder = root / ('fold' + str(f))
            ev = np.flatnonzero(np.isin(d['visit_id'], fold['eval']))
            ca = np.flatnonzero(np.isin(d['visit_id'], fold['cal']))
            tr = np.flatnonzero(np.isin(d['visit_id'], fold['train']) & (d['y'] >= 0))
            np.testing.assert_array_equal(o['fold'][ev], np.full(len(ev), f))
            cal = read_npz(folder / 'cal_scores.npz')
            np.testing.assert_array_equal(cal['indices'], ca)
            assert tuple(cal['models']) == models
            for mi, model in enumerate(models):
                check()
                if model in ('rgb', 'tof'):
                    np.testing.assert_array_equal(o['scores'][ev, mi], d[model+'_score'][ev])
                    np.testing.assert_array_equal(cal['scores'][:, mi], d[model+'_score'][ca])
                else:
                    use = tr[d['arm'][tr] == arms[0]] if model == 'A' else tr
                    retained = read_npz(folder / (model + '_training.npz'))
                    np.testing.assert_array_equal(retained['indices'], use)
                    weights = np.ones(len(use))
                    if model != 'A':
                        for arm in arms:
                            select = d['arm'][use] == arm
                            weights[select] = len(use)/(6*sum(select))
                    np.testing.assert_allclose(retained['weights'], weights, rtol=1e-13)
                    X = d['X']
                    if model == 'C':
                        X = np.column_stack((X, X[:, 0]*X[:, 3], X[:, 0]*X[:, 2], X[:, 0]*d['peak_quality']))
                    model_entries = [r for r in entries if r['fold'] == f and r['model'] == model]
                    assert sorted(r['seed'] for r in model_entries) == [955, 956, 957]
                    scored_cal, scored_eval = [], []
                    for entry in model_entries:
                        assert sha(entry['path']) == entry['sha256']
                        artifact = joblib.load(entry['path']); estimator, scaler = artifact['estimator'], artifact['scaler']
                        assert artifact['features'] == list(names) + (plan['features']['C_extra'] if model == 'C' else [])
                        assert entry['known_rows'] == len(use)
                        assert_close(entry['weight_sum'], float(sum(weights)))
                        for arm in arms:
                            assert_close(entry['arm_weight_sums'][arm], float(sum(weights[d['arm'][use] == arm])))
                        if model != 'D':
                            expected_mean = np.average(X[use], axis=0, weights=weights)
                            expected_var = np.average((X[use]-expected_mean)**2, axis=0, weights=weights)
                            np.testing.assert_allclose(scaler.mean_, expected_mean, atol=1e-10)
                            np.testing.assert_allclose(scaler.var_, expected_var, atol=1e-10)
                            assert np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()
                            assert np.isfinite(scaler.transform(X[use])).all()
                            assert np.isfinite(scaler.transform(X[ca])).all()
                            assert np.isfinite(scaler.transform(X[ev])).all()
                            negative = np.flatnonzero(scaler.var_ < 0)
                            for j in negative:
                                assert abs(scaler.var_[j]) < 1e-10 and scaler.scale_[j] == 1
                            if entry['seed'] == 955:
                                scaler_numerics.append(dict(fold=f, model=model, finite_transforms=True,
                                    negative_roundoff_variance=[dict(feature=artifact['features'][j], value=float(scaler.var_[j]), scale=float(scaler.scale_[j])) for j in negative],
                                    min_variance=float(np.min(scaler.var_)), all_scales_finite_positive=True))
                            parameters = estimator.get_params()
                            for key, value in plan['models']['logistic'].items():
                                assert parameters[key] == value
                            # sklearn 1.9 names the default "deprecated"; its
                            # installed fit implementation maps l1_ratio=0 to
                            # L2. This is the same effective v2 regularization.
                            assert (parameters['penalty'] == 'l2' or
                                    (parameters['penalty'] == 'deprecated' and parameters['l1_ratio'] == 0))
                            for j, name in enumerate(artifact['features']):
                                cr = coef_lookup[(f, model, entry['seed'], name)]
                                assert_close(cr['standardized_coefficient'], float(estimator.coef_[0, j]))
                                assert_close(cr['raw_coefficient'], float(estimator.coef_[0, j]/scaler.scale_[j]))
                        else:
                            assert scaler is None
                            parameters = estimator.get_params()
                            for key, value in plan['models']['D']['parameters'].items():
                                assert parameters[key] == value
                        assert parameters['random_state'] == entry['seed']
                        scored_cal.append(estimator.predict_proba(scaler.transform(X[ca]) if scaler else X[ca])[:, 1])
                        scored_eval.append(estimator.predict_proba(scaler.transform(X[ev]) if scaler else X[ev])[:, 1])
                    np.testing.assert_allclose(cal['scores'][:, mi], np.mean(scored_cal, axis=0), atol=1e-12)
                    np.testing.assert_allclose(o['scores'][ev, mi], np.mean(scored_eval, axis=0), atol=1e-12)
                    weight_tests += 1
                for b in range(3):
                    take = d['band'][ca] == b
                    y, scores = d['y'][ca][take], cal['scores'][take, mi]
                    wanted_ties = independent_ties(scores, y)
                    actual_ties = load(folder / f'ties_{model}_{b}.json')
                    assert len(wanted_ties) == len(actual_ties)
                    for row, want in zip(actual_ties, wanted_ties):
                        assert (cutoff(row), row['W'], row['F'], row['U']) == want
                    chosen, status = independent_select(scores, y, plan['calibration']['targets'][b])
                    selected = cuts[(f, model, b)]
                    assert selected['status'] == status
                    assert (cutoff(selected), selected['W'], selected['F'], selected['U']) == chosen
                    for label, field in ((1, 'POS_denominator'), (0, 'FREE_denominator'), (-1, 'UNKNOWN_denominator')):
                        assert selected[field] == sum(y == label)
                    eval_take = ev[d['band'][ev] == b]
                    expected = (o['scores'][eval_take, mi] >= chosen[0]) & (status == 'ADOPTED')
                    np.testing.assert_array_equal(o['pred'][eval_take, mi], expected)
                    prediction_count += len(eval_take)
                    tie_count += len(wanted_ties)
            print('audit fold PASS', f, flush=True)

    check()
    metrics_lookup = {(r['fold'], r['arm'], r['band'], r['model']): r for r in summary['metrics']}
    paired_lookup = {(r['fold'], r['arm'], r['band'], r['model'], r['baseline']): r for r in summary['paired']}
    retention_lookup = {(r['fold'], r['model']): r for r in summary['retention']}
    assert len(metrics_lookup) == 648 and len(paired_lookup) == 864
    candidates = []
    for f in (-1, 0, 1, 2, 3, 4):
        visits = roster if f == -1 else folds[f]['eval']
        draw = np.random.default_rng(20261011).integers(0, len(visits), size=(2000, len(visits)))
        scope = np.ones(n, bool) if f == -1 else o['fold'] == f
        net_boot = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            for arm in arms:
                for b in range(3):
                    ix = np.flatnonzero(scope & (d['arm'] == arm) & (d['band'] == b))
                    y, v, pp = d['y'][ix], d['visit_id'][ix], o['pred'][ix]
                    vd = np.array([[sum((v == visit) & (y == k)) for k in (1, 0, -1)] for visit in visits])
                    bd = vd[draw].sum(1)
                    for mi, model in enumerate(models):
                        check(); pred = pp[:, mi]
                        counts = np.array([sum(pred & (y == k)) for k in (1, 0, -1)])
                        den = vd.sum(0)
                        vn = np.array([[sum((v == visit) & (y == k) & pred) for k in (1, 0, -1)] for visit in visits])
                        bn = vn[draw].sum(1)
                        metric = dict(W=int(counts[0]), POS=int(den[0]), F=int(counts[1]), FREE=int(den[1]), U=int(counts[2]), UNKNOWN=int(den[2]),
                                      W_rate_ci=compact_ci(100*bn[:,0]/bd[:,0]), F_rate_ci=compact_ci(100*bn[:,1]/bd[:,1]))
                        assert_close(metrics_lookup[(f, arm, b, model)], metric)
                        if model in ('rgb', 'tof'):
                            continue
                        for baseline in ('rgb', 'tof'):
                            base_pred = pp[:, models.index(baseline)]
                            rescue = (y == 1) & pred & ~base_pred
                            loss = (y == 1) & ~pred & base_pred
                            vn = np.array([sum(rescue & (v == visit))-sum(loss & (v == visit)) for visit in visits])
                            bn = vn[draw].sum(1)
                            result = dict(rescue=int(sum(rescue)), loss=int(sum(loss)), net=int(sum(rescue)-sum(loss)),
                                          FREE_added=int(sum((y == 0) & pred & ~base_pred)), FREE_removed=int(sum((y == 0) & ~pred & base_pred)),
                                          net_count_ci=compact_ci(bn), net_pp_ci=compact_ci(100*bn/bd[:,0]))
                            assert_close(paired_lookup[(f, arm, b, model, baseline)], result)
                            if arm == arms[0] and b == 1:
                                net_boot[(model, baseline)] = bn
            best = max(('rgb', 'tof'), key=lambda m: (metrics_lookup[(f, arms[0], 1, m)]['W'], -metrics_lookup[(f, arms[0], 1, m)]['F'], m == 'rgb'))
            a = paired_lookup[(f, arms[0], 1, 'A', best)]['net']
            for model in ('A', 'B', 'C', 'D'):
                net = paired_lookup[(f, arms[0], 1, model, best)]['net']
                retention = net/a if a > 0 else None
                sampled_ratio = np.where(net_boot[('A', best)] > 0, net_boot[(model, best)]/net_boot[('A', best)], np.nan)
                expected = dict(best_single=best, A_net=a, model_net=net, retention=retention, retention_ci=compact_ci(sampled_ratio))
                assert_close(retention_lookup[(f, model)], expected)
                if f == -1:
                    faro = paired_lookup[(f, 'faro_rho030_ambient1', 0, model, 'rgb')]['net']
                    stress = paired_lookup[(f, 'faro_rho030_ambient10', 0, model, 'rgb')]['net']
                    candidates.append(dict(model=model, FARO_near_net=faro, stress_near_net=stress, native_middle_net=net,
                                           A_middle_net=a, retention=retention, robust_candidate=model!='A' and faro>=0 and stress>=0 and retention is not None and retention>=.8))
        print('audit summary/bootstrap PASS', f, flush=True)
    assert_close(summary['candidates'], candidates)
    original_count_descriptor_candidates = []
    for row in candidates:
        if row['model'] == 'A':
            continue
        components = dict(FARO_near_nonnegative=row['FARO_near_net'] >= 0,
                          stress_near_nonnegative=row['stress_near_net'] >= 0,
                          native_middle_count_preserved=row['native_middle_net'] >= .8 * row['A_middle_net'])
        original_count_descriptor_candidates.append(dict(model=row['model'], **components,
            original_count_candidate=all(components.values()),
            A_positive_gain=row['A_middle_net'] > 0, retention=row['retention'],
            interpretation='Post-result interpretation clarification: original user count inequality; negative A makes 80% preservation vacuous; fixed PLAN ratio candidate unchanged'))

    check()
    assert sha(root / 'per_query.csv') == summary['per_query_sha256']
    csv_rows = 0
    with (root / 'per_query.csv').open(newline='', encoding='utf8') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            assert i < n
            for key in ('arm', 'visit_id', 'frame_id', 'query_id'):
                assert row[key] == d[key][i]
            assert int(row['fold']) == o['fold'][i]
            assert int(row['y']) == d['y'][i] and int(row['band']) == d['band'][i]
            for j, model in enumerate(models):
                assert float(row[model+'_score']) == o['scores'][i,j]
                assert bool(int(row[model+'_pred'])) == o['pred'][i,j]
            csv_rows += 1
    assert csv_rows == n
    result = dict(status='PASS', PLAN_sha256=sha(plan_path), features_sha256=sha(root/'features.npz'), oof_sha256=sha(root/'oof.npz'),
                  rows=n, frozen_feature_parity_queries=parity, input_simulation_tests=simulation_tests,
                  sampled_query_feature_tests=query_feature_tests,
                  folds=5, train_weight_checks=weight_tests, cal_ties=tie_count, predictions=prediction_count,
                  metrics=648, paired=864, bootstrap_replicates=2000, clusters_pooled=30, clusters_per_fold=6,
                  candidates=candidates, CPU_command_wall_s=time.monotonic()-started, GPU_s=0,
                  original_count_descriptor_candidates=original_count_descriptor_candidates,
                  protected_reads=0, fits_performed=0,
                  audit_source_sha256=sha(__file__),
                  executed_source_sha256=source_sha256,
                  logistic_penalty_semantics='L2; sklearn default deprecated with l1_ratio=0 accepted after inspecting installed LogisticRegression.fit',
                  scaler_numerics=scaler_numerics,
                  boundaries=['Saved model inference and train scaler statistics independently recomputed; no refit',
                              'Bootstrap intervals conditional on fixed OOF fits; original residual fitting predates CV',
                              'Six fixed simulation samples per generated arm; all generated scale declarations checked'])
    destination = root/'independent_audit.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result), flush=True)
    return result


def audit_report(root, report_path):
    """Check displayed table cells against already independently audited stats."""
    started = time.monotonic()
    summary = load(root/'summary.json')
    audit = load(root/'independent_audit.json')
    assert audit['status'] == 'PASS'
    assert audit['features_sha256'] == summary['features_sha256']
    assert audit['oof_sha256'] == summary['oof_sha256']
    text = report_path.read_text(encoding='utf8')
    lines = text.splitlines()
    labels = {'native_perturbed': '半循环', 'faro_rho015_ambient1': 'rho .15',
              'faro_rho060_ambient1': 'rho .6', 'faro_rho030_ambient3': 'ambient ×3',
              'faro_rho030_ambient1': '过门 FARO', 'faro_rho030_ambient10': 'ambient ×10'}
    bands = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
    def display_ci(ci):
        return 'NA' if ci['lo'] is None else f"[{ci['lo']:.2f}, {ci['hi']:.2f}]"
    def display_ratio(value):
        return 'N/E' if value is None else f'{value*100:.1f}%'
    def display_ratio_ci(ci):
        return 'NA' if ci['lo'] is None else f"[{100*ci['lo']:.1f}%, {100*ci['hi']:.1f}%]"
    metrics = {(r['fold'], r['arm'], r['band'], r['model']): r for r in summary['metrics']}
    paired = {(r['fold'], r['arm'], r['band'], r['model'], r['baseline']): r for r in summary['paired']}
    for r in summary['candidates']:
        expected = f"| {r['model']} | {r['FARO_near_net']:+d} | {r['stress_near_net']:+d} | {r['native_middle_net']:+d} | {display_ratio(r['retention'])} | "
        expected += ('是' if r['robust_candidate'] else '否' if r['model'] != 'A' else 'A对照') + ' |'
        assert expected in lines, ('candidate report mismatch', expected)
    near_rows = 0
    for arm, b in (('faro_rho030_ambient1', 0), ('faro_rho030_ambient10', 0), ('native_perturbed', 1)):
        for model in ('A', 'B', 'C', 'D'):
            r, p = metrics[(-1, arm, b, model)], paired[(-1, arm, b, model, 'rgb')]
            expected = f"| {labels[arm]} / {bands[b]} | {model} | {r['W']}/{r['POS']} | {r['F']}/{r['FREE']} | {p['rescue']}/{p['loss']} | {display_ci(p['net_pp_ci'])} |"
            assert expected in lines, ('focus report mismatch', expected)
            near_rows += 1
    full_rows, single_rows = 0, 0
    for f in (-1, 0, 1, 2, 3, 4):
        marker = '### 合计' if f == -1 else '### 折 '+str(f)
        start = lines.index(marker)
        stops = [i for i in range(start+1, len(lines)) if lines[i].startswith('### ') or lines[i].startswith('## ')]
        section = lines[start:min(stops) if stops else len(lines)]
        actual_fusion = [line for line in section if line.startswith('| ') and len(line.split('|')) == 10 and line.split('|')[3].strip() in ('A','B','C','D')]
        assert len(actual_fusion) == 72, (f, len(actual_fusion))
        for arm in labels:
            for b in range(3):
                for model in ('A','B','C','D'):
                    r = metrics[(f, arm, b, model)]
                    pp = [paired[(f, arm, b, model, baseline)] for baseline in ('rgb','tof')]
                    pairs_text = [f"{p['rescue']}/{p['loss']}={p['net']:+d} {display_ci(p['net_pp_ci'])}" for p in pp]
                    free_text = '; '.join(f"{p['FREE_added']}/{p['FREE_removed']}" for p in pp)
                    expected = f"| {labels[arm]} | {bands[b]} | {model} | {r['W']}/{r['POS']} {display_ci(r['W_rate_ci'])} | {r['F']}/{r['FREE']} {display_ci(r['F_rate_ci'])} | {pairs_text[0]} | {pairs_text[1]} | {free_text} |"
                    assert expected in section, ('full report mismatch', f, arm, b, model)
                    full_rows += 1
                source_text = []
                for model in ('rgb','tof'):
                    r = metrics[(f, arm, b, model)]
                    source_text.append(f"{r['W']}/{r['POS']}, {r['F']}/{r['FREE']}")
                expected = f"| {labels[arm]} | {bands[b]} | {' | '.join(source_text)} |"
                assert expected in section
                single_rows += 1
    assert full_rows == 432 and single_rows == 108
    for r in summary['retention']:
        foldname = '合计' if r['fold'] == -1 else str(r['fold'])
        expected = f"| {foldname} | {r['model']} | {r['best_single']} | {r['A_net']} | {r['model_net']} | {display_ratio(r['retention'])} | {display_ratio_ci(r['retention_ci'])} | {r['retention_ci']['valid']} |"
        assert expected in lines
    result = dict(status='PASS', report_sha256=sha(report_path), summary_sha256=sha(root/'summary.json'),
                  candidate_rows=4, focus_rows=near_rows, full_fusion_rows=full_rows, baseline_rows=single_rows,
                  retention_rows=24, CPU_command_wall_s=time.monotonic()-started,
                  check='Every displayed metric/paired/CI cell agrees with independently audited OOF summary')
    (root/'independent_report_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=150)
    parser.add_argument('--report-only', type=Path)
    args = parser.parse_args()
    if args.report_only:
        audit_report(args.root.resolve(), args.report_only.resolve())
    else:
        run(args.root.resolve(), args.plan.resolve(), args.budget_s)


if __name__ == '__main__':
    main()
