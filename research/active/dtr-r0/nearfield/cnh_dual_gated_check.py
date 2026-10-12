"""Independent CPU checks for frozen-M3 coverage-gated dual fusion.

Scalar fusion, alarm and denominator reconstruction deliberately do not use the
new evaluators' helpers. Geometry is checked against quadrature and invariance,
not a second copy of its convex-polyhedron implementation. Writes only checks/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-dual-gated-fusion-20261005'
THRESHOLD = .8557642486787612


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(name, value):
    path = OUT / 'checks' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def scalar_smooth(raw):
    raw = np.asarray(raw, dtype=np.float64)
    result = np.empty_like(raw)
    for index in np.ndindex(raw.shape[:-2]):
        for f in range(raw.shape[-2]):
            h = list(range(max(0, f - 4), f + 1))
            w = [2 ** j for j in range(len(h))]
            for q in range(raw.shape[-1]):
                result[index + (f, q)] = sum(raw[index + (t, q)] * a for t, a in zip(h, w)) / sum(w)
    return result


def scalar_fusion(scores, coverage, rule):
    scores = np.asarray(scores, dtype=float)
    coverage = np.broadcast_to(np.asarray(coverage, dtype=float), scores.shape)
    result = np.full(scores.shape[:-2] + (scores.shape[-1],), -np.inf)
    for row in np.ndindex(scores.shape[:-2]):
        for f in range(scores.shape[-1]):
            pairs = [(float(scores[row + (s, f)]), float(coverage[row + (s, f)])) for s in range(2)]
            if rule.startswith('G1_'):
                chosen = [s for s, c in pairs if c >= float(rule.split('_')[1])]
            elif rule == 'G2':
                maximum = max(c for _, c in pairs)
                chosen = [s for s, c in pairs if c > 0 and c >= maximum - 1e-12]
            elif rule == 'G3':
                total = sum(c for _, c in pairs)
                chosen = [sum(s * c for s, c in pairs) / total] if total > 0 else []
            else:
                raise ValueError(rule)
            if chosen:
                result[row + (f,)] = max(chosen)
    return result


def first_events(scores, threshold, ranges):
    """Each query episode gets at most one first alarm and no future range."""
    scores = np.asarray(scores, float)
    ranges = np.broadcast_to(np.asarray(ranges, float), scores.shape)
    stopped = np.zeros(scores.shape[:-1], bool)
    timely = stopped.copy()
    first = np.full(stopped.shape, -1, dtype=int)
    for row in np.ndindex(stopped.shape):
        frames = [f for f in range(scores.shape[-1]) if scores[row + (f,)] >= threshold]
        if frames:
            first[row] = frames[0]
            stopped[row] = True
            timely[row] = ranges[row + (frames[0],)] >= .9
    return stopped, timely, first


def quadrature_coverage(poses, powers=17):
    """Independent deterministic Sobol volume integral of public boxes."""
    from scipy.stats import qmc
    points = qmc.Sobol(3, scramble=True, seed=2026100541).random_base2(powers)
    result = []
    for q in np.asarray(poses).reshape(-1, 4, 4):
        groups = []
        for low, high in ((-.2, .42), (.42, .9)):
            xyz = points * [.58, high - low, 1.6] + [-.29, low, .9]
            local = (xyz - q[:3, 3]) @ q[:3, :3]
            visible = (local[:, 2] > 0) & (np.abs(local[:, :2]) <= local[:, 2:3] * np.tan(np.pi / 8)).all(1)
            groups.append(float(visible.mean()))
        result.append(groups)
    return np.asarray(result).reshape(np.asarray(poses).shape[:-2] + (2,))


def geometry_check():
    import cnh_dual_gated_geometry as G
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(2026100540)
    poses = np.repeat(np.eye(4)[None], 16, 0)
    poses[:, 0, 3] = np.linspace(-.2, 0, 16) ** 2
    poses[:, 2, 3] = np.linspace(-2.4, 0, 16)
    poses[:, :3, :3] = Rotation.from_euler('yx', np.column_stack((np.linspace(-15, 17, 16), np.full(16, -10))), degrees=True).as_matrix()
    # Prefix invariance is the operative causal contract, including estimate errors.
    original = G.estimated_query_poses(poses)
    causality_checks = 0
    for f in (3, 7, 12):
        changed = poses.copy()
        changed[f + 1:, :3, 3] += rng.normal(size=changed[f + 1:, :3, 3].shape) * 10
        changed[f + 1:, :3, :3] = Rotation.random(len(changed[f + 1:]), random_state=rng).as_matrix()
        np.testing.assert_array_equal(G.estimated_query_poses(changed)[:, :f - 2], original[:, :f - 2])
        causality_checks += 1
    yaw = Rotation.from_euler('y', 37, degrees=True).as_matrix()
    moved = poses.copy()
    moved[:, :3, 3] = poses[:, :3, 3] @ yaw.T + [4, 0, -3]
    moved[:, :3, :3] = yaw @ poses[:, :3, :3]
    np.testing.assert_allclose(G.estimated_query_poses(moved), original, rtol=0, atol=2e-14)
    sample = np.concatenate((np.eye(4)[None], original[:, [0, 6, 12]].reshape(-1, 4, 4)))
    exact = G.query_coverage(sample)
    numeric = quadrature_coverage(sample)
    error = float(np.max(np.abs(exact - numeric)))
    assert error < .0015, (error, exact, numeric)
    stationary = np.repeat(np.eye(4)[None], 16, 0)
    assert np.isnan(G.estimated_query_poses(stationary)).all()
    assert not G.query_coverage(G.estimated_query_poses(stationary)).any()
    scores = rng.normal(size=(7, 2, 13))
    coverage = rng.uniform(size=scores.shape)
    coverage[0] = 0
    coverage[1] = .5
    for rule in ('G1_0.3', 'G1_0.5', 'G1_0.7', 'G2', 'G3'):
        np.testing.assert_allclose(G.fuse(scores, coverage, rule), scalar_fusion(scores, coverage, rule), rtol=0, atol=1e-15)
    return dict(status='PASS', causal_future_mutation_checks=causality_checks,
                yaw_translation_equivariance=True, stationary_UNKNOWN=True,
                exact_vs_independent_sobol_max_abs_error=error, quadrature_points_per_box=2 ** 17,
                compared_poses=len(sample), fusion_rules_checked=5, geometry_sha256=sha(G.__file__))


def provenance_check():
    plan = read(OUT / 'PLAN.json')
    assert plan['budget_seconds'] == 7200 and not plan['training'] and not plan['hardware']
    assert plan['dual_default_threshold'] == THRESHOLD
    cal = set(plan['natural_calibration_units'])
    evaluation = set(plan['natural_evaluation_units'])
    diagnostic = set(plan['diagnostic_units'])
    assert cal.isdisjoint(evaluation) and cal.isdisjoint(diagnostic) and evaluation.isdisjoint(diagnostic)
    hashes = read(OUT / 'INITIAL_SOURCE_HASHES.json')
    models = 0
    for name, digest in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / name
        assert sha(path).lower() == digest.lower(), name
        models += int('/models/' in str(path).replace('\\', '/'))
    assert models == 5
    return dict(status='PASS', hashes_preserved=len(hashes), M3_hashes_preserved=models,
                disjoint_natural_scene_ID_pools=True, plan_sha256=sha(OUT / 'PLAN.json'))


def natural_check(batch):
    target = OUT / f'natural{batch}'
    plan = read(target / 'PLAN.json')
    completed = read(target / 'run.json')
    units, configs = completed['units'], completed['configs']
    assert completed['status'] == 'COMPLETE' and units == plan['units']
    assert configs in read(OUT / 'PLAN.json')['fallback_config_sets']
    assert plan['photon_prefix'] == 2026100531
    assert plan['models_sha256'] and len(plan['models_sha256']) == 5
    assert plan['parent_plan_sha256'] == sha(OUT / 'PLAN.json')
    for path, digest in plan['models_sha256'].items():
        assert sha(path) == digest
    source_contract = read(OUT / 'NATURAL_SOURCE_CONTRACT.json')
    amendment = read(OUT / 'checks/execution_amendment.json')
    source_checks = 0
    for path, digest in source_contract['source_sha256'].items():
        if Path(path).name == 'cnh_dual_gated_natural.py':
            assert sha(path) == amendment['modified_current_sha256']
        else:
            assert sha(path) == digest, path
        source_checks += 1
    receipt = read(target / 'geometry_receipt.json')
    assert receipt['geometry_sha256'] == sha(target / 'geometry.npz')
    with np.load(target / 'geometry.npz', allow_pickle=False) as d:
        geometry = {k: d[k] for k in d.files}
    expected = [(u, c, q) for u in units for c in configs for q in range(2)]
    assert list(zip(geometry['unit'], geometry['config'], geometry['query'])) == expected
    assert np.array_equal(geometry['covered'], (geometry['frame_ranges'][:, 0] >= .9) & (geometry['frame_ranges'] <= .9).any(1))
    assert np.array_equal(geometry['clear_all'], (geometry['frame_category'] == 'clear').all(1))
    assert np.array_equal(~geometry['covered'], geometry['ref_category'] == 'censored')
    assert np.isfinite(geometry['frame_ranges']).all() and (np.diff(geometry['frame_ranges'], axis=1) <= 1e-8).all()
    if batch == 95000:
        with np.load(ROOT / 'artifacts.local/work/cnh-observed-sequence-20261002/geometry.npz') as old:
            keep = (old['split'] == 'calib') & np.isin(old['config'], configs)
            for key in ('unit', 'config', 'query', 'frame_ranges', 'frame_category', 'clear_all', 'ref_category', 'covered', 'censor_reason'):
                np.testing.assert_array_equal(geometry[key], old[key][keep])
    samples = 0
    seeds = set()
    for unit in units:
        path = target / 'scores' / f'unit{unit}.npz'
        proof = read(path.with_suffix('.json'))
        assert proof['unit'] == unit and proof['configs'] == configs
        assert proof['plan_sha256'] == sha(target / 'PLAN.json')
        assert proof['score_sha256'] == sha(path)
        assert proof['observation_sha256'] == sha(target / 'observations' / f'unit{unit}.npz')
        assert proof['scene_sha256'] == sha(target / 'scene_units' / f'unit{unit}.json')
        scenes = read(target / 'scene_units' / f'unit{unit}.json')
        assert [(r['unit'], r['config']) for r in scenes] == [(unit, c) for c in range(40)]
        with np.load(path) as d:
            branches = 2 if batch == 95000 else 3
            assert np.array_equal(d['frames'], np.arange(3, 16))
            assert list(d['sensors']) == (['L', 'R'] if batch == 95000 else ['S', 'L', 'R'])
            assert d['raw'].shape == (branches, len(configs), 13, 2)
            assert np.array_equal(d['configs'], configs) and int(d['unit']) == unit
            assert np.isfinite(d['raw']).all()
        for config in configs:
            for branch in range(branches):
                seed = int(np.random.SeedSequence([plan['photon_prefix'], unit, config, branch]).generate_state(1)[0])
                assert seed not in seeds, 'Accidental photon seed collision'
                seeds.add(seed)
                samples += 1
    return dict(status='PASS', batch=batch, units=len(units), configs=len(configs),
                query_episodes=len(expected), clear_query_episodes=int(geometry['clear_all'].sum()),
                censored_query_episodes=int((~geometry['covered']).sum()),
                photon_seed_uniqueness_checks=samples, whole_unit_receipts_checked=len(units),
                final_scientific_source_hashes_checked=source_checks,
                no_extrapolated_deadline_labels=True, geometry_sha256=sha(target / 'geometry.npz'))


def execution_amendment_check():
    current = Path(__file__).with_name('cnh_dual_gated_natural.py')
    contract = read(OUT / 'NATURAL_SOURCE_CONTRACT.json')
    recovery = read(OUT / 'EXECUTION_DLL_RECOVERY.json')
    assert sha(current) == recovery['script_sha256']
    text = current.read_text(encoding='utf8')
    added = "    receipt = PARENT/'natural95000/engineering/parity.json'\n    if receipt.exists():\n        assert read(receipt)['status'] == 'PASS'\n        return\n"
    assert text.count(added) == 1
    restored = text.replace(added, '').replace(
        "    save(receipt, dict(status='PASS', raw_logit_max_abs_error=error,",
        "    save(PARENT/'natural95000/engineering/parity.json', dict(status='PASS', raw_logit_max_abs_error=error,")
    original = hashlib.sha256(restored.encode('utf8')).hexdigest()
    assert original == contract['script_sha256']
    for path, digest in contract['source_sha256'].items():
        if Path(path) != current:
            assert sha(path) == digest, path
    return dict(status='PASS', modified_current_sha256=sha(current), reconstructed_original_sha256=original,
                only_change='Reuse existing sealed parity PASS receipt; no model, seed, observations, scoring or wall-clock change',
                scientific_dependency_hashes_preserved=len(contract['source_sha256']) - 1)


def metrics(scores, threshold, d):
    stopped, timely, _ = first_events(scores, threshold, d['ranges'])
    result = {}
    for name, flag in (('clear', stopped), ('shallow', timely), ('mid', timely), ('deep', timely)):
        if name not in d:
            continue
        n = int(d[name].sum())
        num = int((d[name] & flag).sum())
        result[name] = dict(stops=num, n=n, rate=num / n if n else None)
    return result


def independent_cutoff(maxima, budget, scores):
    finite = maxima[np.isfinite(maxima)]
    all_finite = scores[np.isfinite(scores)]
    if not len(finite) or budget >= len(finite):
        return float(np.nextafter(all_finite.min(), -np.inf)) if len(all_finite) else 0.
    possible = np.nextafter(np.unique(finite), np.inf)
    for threshold in possible:
        if (finite >= threshold).sum() <= budget:
            return float(threshold)
    raise AssertionError('No finite feasible cutoff')


def calibration_check():
    selection = read(OUT / 'fusion/selection.json')
    plan = read(OUT / 'fusion/PLAN.json')
    assert selection['plan_sha256'] == sha(OUT / 'fusion/PLAN.json')
    assert selection['ledger_sha256'] == sha(OUT / 'fusion/calibration_ledger.npz')
    with np.load(OUT / 'fusion/calibration_ledger.npz') as z:
        ledger = {k: z[k] for k in z.files}
    domains = selection['calibration_domains']
    assert domains == ['pilot2_calibration', 'natural95000']
    data = [{k: v for name, v in ledger.items() if name.startswith(domain + '_') for k in [name[len(domain) + 1:]]} for domain in domains]
    pilot = ROOT / 'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
    pilot_plan = read(pilot / 'PLAN.json')
    assert set(data[0]['unit']) == set(pilot_plan['calibration_units'])
    assert set(data[0]['unit']).isdisjoint(pilot_plan['evaluation_units'])
    assert set(data[1]['unit']) == set(read(OUT / 'PLAN.json')['natural_calibration_units'])
    raw_pilot = np.asarray([np.load(pilot / 'scores' / f'unit{u}.npz')['raw_full'] for u in pilot_plan['calibration_units']])
    expected = scalar_smooth(raw_pilot).transpose(0, 2, 3, 5, 1, 4).reshape(-1, 2, 13)
    np.testing.assert_allclose(data[0]['sensor'], expected, rtol=0, atol=1e-12)
    original_pilot = ROOT / 'artifacts.local/work/cnh-readout-pilot2-20261005'
    with np.load(original_pilot / 'predictions/M3_fresh_evaluation.npz') as z:
        baseline_raw = z['raw']
    with np.load(original_pilot / 'inputs/fresh_evaluation/rows.npz') as z:
        keep = np.isin(z['unit'], pilot_plan['calibration_units'])
        order = np.lexsort((z['frame'][keep], z['replica'][keep], z['variant'][keep], z['unit'][keep]))
    baseline_raw = baseline_raw[keep][order].reshape(len(pilot_plan['calibration_units']), 7, 4, 13, 2)
    expected = scalar_smooth(baseline_raw).transpose(0, 1, 2, 4, 3).reshape(-1, 13)
    np.testing.assert_allclose(data[0]['single'], expected, rtol=0, atol=1e-12)
    configs = read(OUT / 'natural95000/run.json')['configs']
    units = sorted(set(data[1]['unit']))
    raw_natural = np.asarray([np.load(OUT / 'natural95000/scores' / f'unit{u}.npz')['raw'] for u in units])
    expected = scalar_smooth(raw_natural).transpose(0, 2, 4, 1, 3).reshape(-1, 2, 13)
    np.testing.assert_allclose(data[1]['sensor'], expected, rtol=0, atol=1e-12)
    margin = ROOT / 'artifacts.local/work/cnh-margin-confirm-20261002'
    with np.load(margin / 'frame_scores_M3_early.npz') as early, np.load(margin / 'frame_scores_M3.npz') as late:
        baseline_raw = np.asarray([np.concatenate((early[str(u)][configs], late[str(u)][configs]), axis=1) for u in units])
    expected = scalar_smooth(baseline_raw).transpose(0, 1, 3, 2).reshape(-1, 13)
    np.testing.assert_allclose(data[1]['single'], expected, rtol=0, atol=1e-12)
    # Fresh natural calibration truth is also a sealed exact parity of the old truth.
    with np.load(OUT / 'natural95000/geometry.npz') as z:
        for key in ('unit', 'config', 'query', 'covered'):
            np.testing.assert_array_equal(data[1][key], z[key])
        np.testing.assert_array_equal(data[1]['clear'], z['clear_all'])
        np.testing.assert_array_equal(data[1]['ranges'], z['frame_ranges'])
        np.testing.assert_array_equal(data[1]['deep'], z['covered'] & (z['ref_category'] == 'contact>5cm'))
        np.testing.assert_array_equal(data[1]['shallow'], z['covered'] & (z['ref_category'] == 'contact0-2cm'))
        np.testing.assert_array_equal(data[1]['mid'], z['covered'] & (z['ref_category'] == 'contact2-5cm'))
    single = [metrics(d['single'], THRESHOLD, d) for d in data]
    count = sum(m['clear']['stops'] for m in single)
    budget = int(np.floor(count * 1.1 + 1e-12))
    assert count == selection['pooled_single_clear_stops'] and budget == selection['pooled_allowed_clear_stops']
    rebuilt = []
    for ri, candidate in enumerate(selection['candidates']):
        rule = candidate['rule']
        assert rule == plan['rules'][ri]
        fused = [scalar_fusion(d['sensor'], d['coverage'], rule) for d in data]
        for domain, value in zip(domains, fused):
            np.testing.assert_allclose(value, ledger[domain + '_' + rule], rtol=0, atol=1e-12)
        maxima = np.concatenate([v.max(-1)[d['clear']] for v, d in zip(fused, data)])
        threshold = independent_cutoff(maxima, budget, np.concatenate(fused))
        assert np.isfinite(candidate['threshold'])
        np.testing.assert_allclose(candidate['threshold'], threshold, rtol=0, atol=1e-12)
        reconstructed = [metrics(v, candidate['threshold'], d) for v, d in zip(fused, data)]
        for domain, met in zip(domains, reconstructed):
            existing = candidate['domains'][domain]['metrics']
            for name in ('clear', 'shallow', 'mid', 'deep'):
                assert (met[name]['stops'], met[name]['n']) == (existing[name]['stops'], existing[name]['n'])
        timely = sum(m['shallow']['stops'] + m['mid']['stops'] + m['deep']['stops'] for m in reconstructed)
        shallow = sum(m['shallow']['stops'] for m in reconstructed)
        deep = sum(m['deep']['stops'] for m in reconstructed)
        rank = [timely, shallow, deep, -ri]
        assert rank == candidate['rank_key']
        assert (maxima >= candidate['threshold']).sum() <= budget
        for v in fused:
            absent = ~np.isfinite(v)
            assert not (v[absent] >= candidate['threshold']).any()
        rebuilt.append((rank, rule))
    best = max(rebuilt)
    assert selection['selected']['rule'] == best[1]
    return dict(status='PASS', domains=domains, candidates_checked=len(rebuilt),
                selected_rule=best[1], selected_threshold=selection['selected']['threshold'],
                pooled_clear_budget=budget, pooled_single_clear_stops=count,
                raw_single_and_sensor_smoothing_reconstructed=True, finite_threshold_absent_branches_never_alarm=True,
                no_evaluation_scene_in_calibration_arrays=True, selection_sha256=sha(OUT / 'fusion/selection.json'))


def angular_check():
    selected = read(OUT / 'fusion/selection.json')['selected']
    thresholds = dict(single=THRESHOLD, OR=THRESHOLD, gated=selected['threshold'])
    angular = read(OUT / 'fusion/angular_result.json')
    pilot = ROOT / 'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
    envelope = ROOT / 'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
    pp = read(pilot / 'PLAN.json')
    old = read(pilot / 'result.json')
    expected_units = sorted(set(pp['evaluation_units']) & set(read(envelope / 'PLAN.json')['A']['units']))
    assert angular['units'] == expected_units and set(expected_units).isdisjoint(pp['calibration_units'])
    assert angular['ledger_sha256'] == sha(OUT / 'fusion/angular_ledger.npz')
    with np.load(OUT / 'fusion/angular_ledger.npz') as z:
        angular_data = {k: z[k] for k in z.files}
    old_scenes = {s['unit']: s for s in old['scenes']}
    raw_angles = []
    for unit in expected_units:
        with np.load(envelope / 'angular/scores' / f'unit{unit}.npz') as z:
            raw_angles.append(z['raw'])
            angles = z['angles_deg'].tolist()
    score_angles = scalar_smooth(np.asarray(raw_angles))
    passed = []
    angular_counts = 0
    for psi in angular['psi_deg']:
        prefix = f'psi{psi}_'
        stream = score_angles[:, [angles.index(psi - 15), angles.index(psi + 15)]]
        expected_sensor = stream.transpose(0, 2, 3, 5, 1, 4).reshape(-1, 2, 13)
        np.testing.assert_allclose(expected_sensor, angular_data[prefix + 'sensor'], rtol=0, atol=1e-12)
        expected_single = score_angles[:, angles.index(psi)].transpose(0, 1, 2, 4, 3).reshape(-1, 13)
        arms = dict(single=expected_single, OR=expected_sensor.max(1), gated=scalar_fusion(expected_sensor, angular_data[prefix + 'coverage'], selected['rule']))
        for arm, score in arms.items():
            np.testing.assert_allclose(score, angular_data[prefix + arm], rtol=0, atol=1e-12)
            _, timely, _ = first_events(score, thresholds[arm], angular_data['ranges'])
            assert np.array_equal(timely, angular_data[prefix + arm + '_timely'])
            timely = timely.reshape(len(expected_units), 7, 4, 2)
            for group in ('all', 'same', 'opposite'):
                for name, variants in (('shallow', (0, 1)), ('deep', (2,))):
                    num = den = 0
                    for i, unit in enumerate(expected_units):
                        scene = old_scenes[unit]
                        if group != 'all' and ((group == 'same') != scene['fov_in']):
                            continue
                        for d in variants:
                            if angular_data['covered'][i, d]:
                                den += 4
                                num += int(timely[i, d, :, scene['group']].sum())
                    metric = angular['sequence'][str(psi)][arm][group][name]
                    assert (num, den) == (metric['stops'], metric['n'])
                    angular_counts += 2
        if psi <= 25:
            metric = angular['sequence'][str(psi)]['gated']['opposite']['shallow']
            passed.append(metric['n'] > 0 and metric['stops'] / metric['n'] >= .9 - 1e-12)
    tolerance = all(passed)
    assert tolerance == angular['tolerance_psi_le25_pass']
    return dict(status='PASS', angular_integer_comparisons=angular_counts,
                angular_evaluation_scenes=len(expected_units), per_psi_90percent_check=tolerance,
                all_raw_smoothing_and_fusion_reconstructed=True,
                result_sha256=sha(OUT / 'fusion/angular_result.json'))


def evaluation_check():
    result = read(OUT / 'fusion/result.json')
    selection = read(OUT / 'fusion/selection.json')
    selected = selection['selected']
    seal = read(OUT / 'fusion/EXECUTION_SOURCE_SEAL.json')
    assert seal['source_sha256'] == sha(Path(__file__).with_name('cnh_dual_gated_evaluate.py'))
    assert seal['geometry_source_sha256'] == sha(Path(__file__).with_name('cnh_dual_gated_geometry.py'))
    assert result['selection_sha256'] == sha(OUT / 'fusion/selection.json')
    assert result['ledger_sha256'] == sha(OUT / 'fusion/natural97000_ledger.npz')
    with np.load(OUT / 'fusion/natural97000_ledger.npz') as z:
        data = {k: z[k] for k in z.files}
    assert result['units'] == read(OUT / 'natural97000/run.json')['units']
    assert result['configs'] == read(OUT / 'natural97000/run.json')['configs']
    raw = []
    for unit in result['units']:
        with np.load(OUT / 'natural97000/scores' / f'unit{unit}.npz') as z:
            raw.append(z['raw'])
    smoothed = scalar_smooth(np.asarray(raw))
    single = smoothed[:, 0].transpose(0, 1, 3, 2).reshape(-1, 13)
    sensor = smoothed[:, 1:].transpose(0, 2, 4, 1, 3).reshape(-1, 2, 13)
    np.testing.assert_allclose(data['single'], single, rtol=0, atol=1e-12)
    np.testing.assert_allclose(data['sensor'], sensor, rtol=0, atol=1e-12)
    gated = scalar_fusion(sensor, data['coverage'], selected['rule'])
    np.testing.assert_allclose(data['gated'], gated, rtol=0, atol=1e-12)
    with np.load(OUT / 'natural97000/geometry.npz') as z:
        np.testing.assert_array_equal(data['clear'], z['clear_all'])
        np.testing.assert_array_equal(data['covered'], z['covered'])
        np.testing.assert_array_equal(data['ranges'], z['frame_ranges'])
        np.testing.assert_array_equal(data['shallow'], z['covered'] & (z['ref_category'] == 'contact0-2cm'))
        np.testing.assert_array_equal(data['deep'], z['covered'] & (z['ref_category'] == 'contact>5cm'))
    arms = dict(single=single, OR=sensor.max(1), gated=gated)
    thresholds = dict(single=THRESHOLD, OR=THRESHOLD, gated=selected['threshold'])
    counts = 0
    ci_checks = 0
    clusters = sorted(set(data['unit']))
    rng = np.random.default_rng(2026100543)
    bootstrap = np.zeros((1000, len(clusters)), dtype=int)
    for draw in bootstrap:
        for index in rng.integers(len(clusters), size=len(clusters)):
            draw[index] += 1
    base_stopped, base_timely, _ = first_events(single, THRESHOLD, data['ranges'])
    for arm, scores in arms.items():
        stopped, timely, first = first_events(scores, thresholds[arm], data['ranges'])
        np.testing.assert_array_equal(stopped, data[arm + '_stopped'])
        np.testing.assert_array_equal(timely, data[arm + '_timely'])
        np.testing.assert_array_equal(first[stopped], data[arm + '_first_index'][stopped])
        for group in ('all', 'mode0', 'mode1', 'mode2'):
            keep = np.ones(len(stopped), bool) if group == 'all' else data['unit'] % 3 == int(group[-1])
            cell = result['natural97000'][group][arm]
            for category in ('clear', 'shallow', 'contact0_2', 'contact2_5', 'deep'):
                den = keep & data[category]
                flag = stopped if category == 'clear' else timely
                num, n = int((den & flag).sum()), int(den.sum())
                assert (num, n) == (cell[category]['stops'], cell[category]['n'])
                if category == 'clear':
                    assert cell[category]['proxy_minutes'] == n * 2.6 / 60
                    assert cell[category]['stops_per_proxy_minute'] == (num / (n * 2.6 / 60) if n else None)
                baseline = base_stopped if category == 'clear' else base_timely
                assert cell[category]['rescues'] == int((den & flag & ~baseline).sum())
                assert cell[category]['losses'] == int((den & ~flag & baseline).sum())
                group_num = np.asarray([int((den & flag & (data['unit'] == unit)).sum()) for unit in clusters])
                group_base = np.asarray([int((den & baseline & (data['unit'] == unit)).sum()) for unit in clusters])
                group_den = np.asarray([int((den & (data['unit'] == unit)).sum()) for unit in clusters])
                with np.errstate(divide='ignore', invalid='ignore'):
                    draws = (bootstrap @ group_num) / (bootstrap @ group_den)
                    delta = (bootstrap @ (group_num - group_base)) / (bootstrap @ group_den)
                finite = draws[np.isfinite(draws)]
                df = delta[np.isfinite(delta)]
                for field, values in (('ci95', finite), ('paired_delta_ci95', df)):
                    expected_ci = np.quantile(values, [.025, .975]) if len(values) else [None, None]
                    if len(values):
                        np.testing.assert_allclose(cell[category][field], expected_ci, rtol=0, atol=1e-14)
                    else:
                        assert cell[category][field] == expected_ci
                    ci_checks += 1
                if category == 'clear' and len(finite):
                    np.testing.assert_allclose(cell[category]['rate_ci95_per_proxy_minute'], np.quantile(finite, [.025, .975]) * 60 / 2.6, rtol=0, atol=1e-13)
                    ci_checks += 1
                counts += 2
            joint_den = data['clear'].reshape(-1, 2).all(1) & keep.reshape(-1, 2).all(1)
            joint = stopped.reshape(-1, 2).any(1)
            assert (int((joint & joint_den).sum()), int(joint_den.sum())) == (cell['joint_clear']['stops'], cell['joint_clear']['n'])
            assert cell['censored'] == int((keep & ~data['covered']).sum())
            assert cell['abstention_frames'] == int((~np.isfinite(scores) & keep[:, None]).sum())
            assert cell['episodes_with_abstention'] == int((~np.isfinite(scores).all(1) & keep).sum())
            assert cell['episodes_all_frames_abstention'] == int((~np.isfinite(scores).any(1) & keep).sum())
            counts += 3
    angular_recheck = read(OUT / 'checks/angular.json')
    assert angular_recheck['status'] == 'PASS'
    assert angular_recheck['result_sha256'] == sha(OUT / 'fusion/angular_result.json')
    assert result['angular_evaluation'] == read(OUT / 'fusion/angular_result.json')
    tolerance = angular_recheck['per_psi_90percent_check']
    angular_counts = angular_recheck['angular_integer_comparisons']
    single_cell = result['natural97000']['all']['single']
    gated_cell = result['natural97000']['all']['gated']
    baseline_clear, gated_clear = single_cell['clear']['stops'], gated_cell['clear']['stops']
    ratio = gated_clear / baseline_clear if baseline_clear else (1. if not gated_clear else None)
    sh = gated_cell['shallow']['rate'] - single_cell['shallow']['rate'] if single_cell['shallow']['n'] else None
    de = gated_cell['deep']['rate'] - single_cell['deep']['rate'] if single_cell['deep']['n'] else None
    clear_pass = ratio is not None and ratio <= 1.15 + 1e-12
    if ratio is None or sh is None or de is None:
        decision = 'NOT_EVALUABLE'
    elif clear_pass and sh >= -.03 - 1e-12 and de >= -.03 - 1e-12 and tolerance:
        decision = 'GATED_DUAL_SUPPORTED_SIM'
    elif (clear_pass and not tolerance) or (tolerance and ratio > 1.3):
        decision = 'GATING_TRADEOFF'
    else:
        decision = 'GATING_NOT_SUPPORTED'
    assert decision == result['decision']['branch']
    return dict(status='PASS', natural_integer_comparisons=counts, angular_integer_comparisons=angular_counts,
                natural_CI_comparisons=ci_checks, bootstrap_clusters=len(clusters), bootstrap_draws=1000,
                angular_evaluation_scenes=angular_recheck['angular_evaluation_scenes'], per_psi_90percent_check=tolerance,
                decision=decision, selected_rule=selected['rule'], selected_threshold=selected['threshold'],
                all_raw_smoothing_and_fusion_reconstructed=True, result_sha256=sha(OUT / 'fusion/result.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('geometry', 'provenance', 'execution_amendment', 'calibration', 'angular', 'evaluation', 'natural95000', 'natural97000'))
    args = parser.parse_args()
    tick = time.monotonic()
    if args.stage == 'geometry':
        value = geometry_check()
    elif args.stage == 'provenance':
        value = provenance_check()
    elif args.stage == 'execution_amendment':
        value = execution_amendment_check()
    elif args.stage == 'calibration':
        value = calibration_check()
    elif args.stage == 'angular':
        value = angular_check()
    elif args.stage == 'evaluation':
        value = evaluation_check()
    else:
        value = natural_check(int(args.stage.removeprefix('natural')))
    value['elapsed_seconds'] = time.monotonic() - tick
    value['check_source_sha256'] = sha(__file__)
    record(args.stage + '.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
