"""Independent saved-output audit for no-bin0 domain-randomized retraining.

No renderer, trainer or producer evaluator imports. All truth is used only for
recomputation of descriptive evaluation, never for model inference or selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_notifications(grades):
    grades = np.asarray(grades)
    assert grades.ndim == 4 and grades.shape[-2:] == (13, 2)
    assert np.isin(grades, (0, 1, 2)).all()
    emitted = np.zeros_like(grades)
    for scene in range(len(grades)):
        for replica in range(grades.shape[1]):
            for query in range(2):
                level, zeros = 0, 0
                for frame in range(13):
                    current = int(grades[scene, replica, frame, query])
                    if not current:
                        zeros += 1
                        if zeros >= 2:
                            level = 0
                    else:
                        zeros = 0
                        if current > level:
                            emitted[scene, replica, frame, query] = current
                        level = max(level, current)
    return emitted


def masks(rows, category):
    category = np.asarray(category)
    assert category.shape == (len(rows), 2)
    assert np.isin(category, ('contact', 'pass', 'clear')).all()
    negative = ~(category == 'contact').any(1)
    near = (category == 'pass').any(1) & np.array([
        float(row['lateral_gap_m']) <= .10+1e-10 for row in rows])
    return negative, near, negative & ~near


def statistics(grades, reference, category, rows):
    """Independently replay each clip; reference is common original old5."""
    emitted = scalar_notifications(grades)
    base = scalar_notifications(reference)
    negative, near, far_clear = masks(rows, category)
    joint = emitted.max(-1)
    costs = np.where((joint == 1) & near[:, None, None], .25, 1.)
    weighted_cost = float(np.where((joint > 0) & negative[:, None, None], costs, 0).sum())
    timely, base_timely = (emitted[:, :, :11] > 0).any(2), (base[:, :, :11] > 0).any(2)
    result = dict(weighted_cost=weighted_cost,
                  far_clear_notified_clips=int((joint[far_clear] > 0).any(-1).sum()),
                  far_clear_clip_denominator=int(far_clear.sum()*grades.shape[1]),
                  far_clear_notifications=int((joint[far_clear] > 0).sum()), contacts={})
    for q, height in enumerate(('HEAD', 'BODY')):
        positive = np.asarray(category)[:, q] == 'contact'
        a, b = timely[positive, :, q], base_timely[positive, :, q]
        rescue, loss = int((a & ~b).sum()), int((~a & b).sum())
        result['contacts'][height] = dict(denominator=int(a.size), timely=int(a.sum()),
            baseline_timely=int(b.sum()), rescue=rescue, loss=loss, net=rescue-loss)
    return result, emitted


def selected_threshold(fixed, score, category, rows, cap_increment=.03, baseline_fixed=None):
    """Exact canonical all-tie selection, independently using clip maxima.

    Gap1 emits the first positive immediately, hence any-notification clip rate
    equals any-positive grade support. This does not assume cost monotonicity.
    """
    negative, _, far_clear = masks(rows, category)
    assert negative.all(), 'Threshold chooser must contain pass/clear only'
    fixed, score = np.asarray(fixed), np.asarray(score, dtype=np.float64)
    assert fixed.shape == score.shape and fixed.shape[-2:] == (13, 2)
    assert np.isin(fixed, (0, 2)).all() and not np.isnan(score).any()
    eligible = (fixed == 0) & np.isfinite(score)
    ties = np.unique(score[eligible])
    taus = np.r_[-np.inf, np.nextafter(ties, np.inf), np.inf]
    strong = (fixed[far_clear] > 0).any((-2, -1))
    maxima = np.where(eligible[far_clear], score[far_clear], -np.inf).max((-2, -1))
    den = int(strong.size)
    baseline = int(strong.sum()) if baseline_fixed is None else int(
        (np.asarray(baseline_fixed)[far_clear] > 0).any((-2, -1)).sum())
    vals = np.sort(maxima[~strong & np.isfinite(maxima)])
    counts = baseline+len(vals)-np.searchsorted(vals, taus, side='left')
    cap = baseline+cap_increment*den
    valid = np.flatnonzero(counts <= cap)
    index = int(valid[0]) if valid.size else len(taus)-1
    return dict(tau=float(taus[index]), candidate_index=index, count=int(counts[index]),
                denominator=den, baseline=baseline, cap=float(cap), ties=len(ties),
                status='SEALED' if valid.size else 'STRONG_ALREADY_EXCEEDS_POOLED_CAP'), taus, counts


def physical_key(boxes):
    encoded = json.dumps(boxes, sort_keys=True, separators=(',', ':')).encode('utf8')
    return hashlib.sha256(encoded).hexdigest()


def source_disjointness(rows, prior_rows):
    keys, families, backgrounds = set(), set(), set()
    for row in rows:
        key = physical_key(row['boxes'])
        assert key == row['physical_key'] and key not in keys
        keys.add(key)
        families.add(row['background_family'])
        backgrounds.add(physical_key(row['background_boxes']))
    oldkeys = {row['physical_key'] for row in prior_rows}
    oldfamilies = {row['background_family'] for row in prior_rows if 'background_family' in row}
    oldbackgrounds = {physical_key(row['background_boxes']) for row in prior_rows if 'background_boxes' in row}
    assert not keys & oldkeys and not families & oldfamilies and not backgrounds & oldbackgrounds
    return dict(physical_scenes=len(rows), physical_overlap=0, family_overlap=0,
                background_geometry_overlap=0, families=sorted(families),
                prior_physical_scenes=len(oldkeys))


def acceptance(new_stats, old_stats):
    new_gain = sum(v['net'] for v in new_stats['contacts'].values())
    old_gain = sum(v['net'] for v in old_stats['contacts'].values())
    assert new_stats['far_clear_clip_denominator'] == old_stats['far_clear_clip_denominator']
    denominator = old_stats['far_clear_clip_denominator']
    increase_pp = 100*(new_stats['far_clear_notified_clips']-old_stats['far_clear_notified_clips'])/denominator
    return dict(new_gain=new_gain, old_gain=old_gain,
        gain_retention=None if old_gain <= 0 else new_gain/old_gain,
        timely_pass=new_gain >= (.8*old_gain if old_gain > 0 else old_gain),
        far_clear_increase_pp=increase_pp, far_clear_pass=increase_pp <= 1.+1e-12)


def harvest(value, keys, families, backgrounds):
    if isinstance(value, dict):
        if isinstance(value.get('physical_key'), str):
            keys.add(value['physical_key'])
        if isinstance(value.get('background_family'), str):
            families.add(value['background_family'])
        if isinstance(value.get('background_boxes'), list):
            backgrounds.add(physical_key(value['background_boxes']))
        boxes = value.get('boxes')
        if isinstance(boxes, list) and boxes and all(isinstance(b, dict) and set(('lo', 'hi', 'rho')) <= set(b) for b in boxes):
            keys.add(physical_key(boxes))
            if value.get('background_family') and 'scene_id' not in value and 'scene_uid' not in value:
                backgrounds.add(physical_key(boxes))
        for nested in value.values():
            harvest(nested, keys, families, backgrounds)
    elif isinstance(value, list):
        for nested in value:
            harvest(nested, keys, families, backgrounds)


def source_audit(out):
    rows = read(out/'scene_rows.json')['hold']
    inventory = read(out/'inventory.json')
    oldkeys, oldfamilies, oldbackgrounds = set(), set(), set()
    latest_found = False
    for entry in inventory['files']:
        path = ROOT/entry['path']
        assert sha(path) == entry['sha256'], str(path)
        assert not any(part.lower() in ('test', 'protected') or '480' in part
                       for part in Path(entry['path']).parts)
        latest_found |= 'cnh-s-ensemble-tradeoff-dev-20261010/scene_rows.json' in str(path).replace('\\', '/')
        harvest(read(path), oldkeys, oldfamilies, oldbackgrounds)
    assert latest_found, 'Already-consumed tradeoff source must be inventoried'
    keys = set()
    for row in rows:
        key = physical_key(row['boxes'])
        assert key == row['physical_key'] and key not in keys
        keys.add(key)
        assert key not in oldkeys
        assert row['background_family'] not in oldfamilies
        assert physical_key(row['background_boxes']) not in oldbackgrounds
    assert len(rows) == 1536 and len({r['background_family'] for r in rows}) == 4
    assert {p:sum(r['placement'] == p for r in rows) for p in ('contact', 'pass', 'clear')} == {'contact':512, 'pass':512, 'clear':512}
    assert [sum(r['placement'] == 'contact' and r['group'] == q for r in rows) for q in range(2)] == [256, 256]
    assert (out/'sealed_candidates.json').stat().st_mtime_ns < (out/'data/hold/geometry.npz').stat().st_mtime_ns
    return rows, dict(prior_files=len(inventory['files']), prior_keys=len(oldkeys),
        physical_overlap=0, background_overlap=0, family_overlap=0, consumed_tradeoff_included=True)


def identity_audit(out):
    seal = read(out/'sealed_candidates.json')
    assert seal['calibration_sha256'] == sha(out/'sealed_calibration.json')
    assert seal['plan_sha256'] == sha(out/'PLAN.json')
    for path, digest in seal['models_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for arm, digest in seal['real_manifest_hashes'].items():
        manifest = Path(seal['real_manifests'][arm])
        assert sha(manifest) == digest
        parsed = read(manifest)
        assert parsed['drop_bins'] == ([0] if arm == 'main' else [0, 1])
        assert parsed['real_selection'] is False and parsed['hold_selection'] is False
        assert len(parsed['ordinary_paths']) == len(parsed['s_paths']) == 3
        entries = parsed['m3_paths']+parsed['ordinary_paths']+[e for pair in parsed['s_paths'] for e in pair]
        for entry in entries:
            assert sha(ROOT/entry['path']) == entry['sha256']
    draws = read(out/'domain_randomization_draws.json')
    limits = dict(gain=(.42, .73), sigma_m=(.070, .115), tail_fraction=(0., .05),
        range_offset_m=(-.1, .1), ambient=(4., 10.4), noise_a=(0., .25357091365),
        noise_b=(.08671592722, .09326653791), noise_c=(.5807095993, .68960300094))
    assert len(draws)*13 == 59904
    assert len({tuple(d['key']) for d in draws}) == len(draws)
    for i, draw in enumerate(draws):
        assert draw['row_start'] == i*13
        assert draw['tail_scale_m'] == .5
        for name, (lo, hi) in limits.items():
            assert lo <= draw[name] <= hi, (name, draw[name])
    model_files = list((out/'models').glob('*/M3/model_seed*.pt'))+list((out/'models').glob('*/ordinary_seed*.pt'))
    for path in model_files:
        receipt = read(path.with_suffix('.json'))
        assert receipt['sha256'] == sha(path)
        assert receipt['epochs'] == (20 if path.name.startswith('model_seed') else 32)
        if path.name.startswith('ordinary_seed'):
            assert receipt['steps'] == 7488 and receipt['rows'] == 59904
    for arm in ('no_bin0', 'no_bin01'):
        sreceipt = read(out/arm/'S_training_receipt.json')
        assert len(sreceipt['jobs']) == 6
        assert sreceipt['recipe'] == dict(max_iter=100, learning_rate=.1, max_leaf_nodes=7,
            max_depth=3, min_samples_leaf=50, l2_regularization=1., early_stopping=False)
        for job in sreceipt['jobs']:
            assert sha(job['path']) == job['sha256'] and job['iterations'] == 100
    return dict(model_files=len(model_files), domain_randomization_episodes=len(draws),
                bounds_checked=len(draws)*len(limits), model_hashes=len(seal['models_sha256']))


def calibration_audit(out):
    pool_rows = read(out/'pool_rows.json')
    with np.load(out/'pool_metadata.npz', allow_pickle=False) as saved:
        category, original_old5 = saved['category'], saved['old5']
    seal = read(out/'sealed_calibration.json')
    answer = {}
    for arm in ('no_bin0', 'no_bin01'):
        with np.load(out/arm/'pool_light_scores.npz', allow_pickle=False) as saved:
            score, strong = saved['score'], saved['strong']
        recomputed, _, _ = selected_threshold(strong, score, category, pool_rows, baseline_fixed=original_old5)
        record = seal['cuts'][arm]['s_calibration']
        for name, audit_name in (('old5_far_clear_clips', 'baseline'), ('far_clear_denominator', 'denominator'),
                                 ('selected_far_clear_clips', 'count'), ('ties', 'ties')):
            assert record[name] == recomputed[audit_name], (arm, name)
        assert record['status'] == recomputed['status'] and record['negative_only'] is True
        stored_tau = seal['cuts'][arm]['s_tau']
        if np.isposinf(recomputed['tau']):
            assert stored_tau >= 1e300
        else:
            assert stored_tau == recomputed['tau'], (arm, stored_tau, recomputed['tau'])
        answer[arm] = dict(**{k:v for k,v in recomputed.items() if k != 'tau'},
            tau=stored_tau, no_contact_selection=True)
    return answer


def hold_audit(out, rows):
    saved = np.load(out/'hold_grades_notifications.npz', allow_pickle=False)
    keys = [str(k) for k in saved['keys']]
    grades, notifications, category = saved['grades'], saved['notifications'], saved['category']
    reference = grades[keys.index('original_old5')]
    producer = read(out/'simulation_evaluation.json')
    answer = {}
    for i, arm in enumerate(keys):
        result, emitted = statistics(grades[i], reference, category, rows)
        np.testing.assert_array_equal(emitted, notifications[i])
        reported = producer[arm]
        assert result['weighted_cost'] == reported['weighted_cost'], arm
        assert result['far_clear_notified_clips'] == reported['far_clear']['notified_clips'], arm
        assert result['far_clear_clip_denominator'] == reported['far_clear']['denominator'], arm
        for height, data in result['contacts'].items():
            assert data == reported[height], (arm, height)
        answer[arm] = result
    calculated = acceptance(answer['no_bin0_S'], answer['old_default_S'])
    declared = producer['acceptance']
    assert calculated['timely_pass'] == declared['timely_pass']
    assert calculated['far_clear_pass'] == declared['far_clear_pass']
    assert calculated['new_gain'] == declared['new_gain_vs_common_old5']
    assert calculated['old_gain'] == declared['old_gain_vs_common_old5']
    assert abs(calculated['far_clear_increase_pp']-declared['far_clear_increment_pp']) < 1e-10
    return dict(arms=len(keys), notifications_checked=int(notifications.size),
                statistics=answer, acceptance=calculated)


def real_audit(out):
    receipt = read(out/'real_receipt.json')
    assert receipt['model_seal_sha256'] == sha(out/'sealed_candidates.json')
    for entry in receipt['outputs']:
        path = ROOT/entry['path']
        assert sha(path) == entry['sha256']
        assert (out/'sealed_candidates.json').stat().st_mtime_ns < path.stat().st_mtime_ns
    checks = 0
    for row in receipt['tables']:
        arm, input_path, sid = row['arm'], row['path'], row['condition']
        filename = 'bin0_only' if sid == 'BIN0_BACKGROUND_ONLY' else sid
        with np.load(out/'real'/arm/input_path/(filename+'.npz'), allow_pickle=False) as arrays:
            ready = arrays['ready']; q = ('HEAD', 'BODY').index(row['height'])
            assert int(ready.sum()) == row['ready_frames']
            if row['model'] == 'M3':
                strong, light = arrays['m3'][ready, q], np.zeros(int(ready.sum()), bool)
            elif row['model'] == 'old5':
                strong, light = arrays['old5'][ready, q], np.zeros(int(ready.sum()), bool)
            else:
                strong, light = arrays['S_grade'][ready, q] == 2, arrays['S_grade'][ready, q] == 1
            assert int(strong.sum()) == row['strong'] and int(light.sum()) == row['light']
            assert float(strong.mean()) == row['strong_rate'] and float(light.mean()) == row['light_rate']
            checks += 1
    for check in receipt['input_checks']:
        assert all(check['poison_score_and_grade_exact'].values())
        assert check['drop_bins'] == ([0] if check['arm'] == 'main' else [0, 1])
    return dict(table_rows_checked=checks, output_hashes=len(receipt['outputs']),
                poison_invariance_receipts=receipt['input_checks'])


def run(out):
    start = time.monotonic()
    identities = identity_audit(out)
    rows, source = source_audit(out)
    calibration = calibration_audit(out)
    hold = hold_audit(out, rows)
    real = real_audit(out) if (out/'real_receipt.json').exists() else None
    result = dict(status='PASS', seconds=time.monotonic()-start, independent=True,
        protected_access=0, training=0, source_sha256=sha(__file__), identities=identities,
        source=source, calibration=calibration, hold=hold, real=real)
    output = out/('independent_audit_real.json' if real is not None else 'independent_audit_sim.json')
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        calibration=calibration, acceptance=hold['acceptance'], real_rows=None if real is None else real['table_rows_checked']), ensure_ascii=False), flush=True)
    return result


def forward_audit(out):
    """Fresh CUDA process: sealed candidate, first16 pilot frames and poison.

    Prefix windows have exactly the same causal history as saved full episodes.
    Adapter arithmetic is recomputed here without importing the real producer.
    """
    import sys
    sys.path.insert(0, str(ROOT/'research/active/hardware-bringup/host'))
    import real_pilot_calibration as pilot
    from cnh_no_bin0_dr_replay import DesignReplay
    started = time.monotonic()
    data = pilot.data()
    old_parameters = pilot.read('instrument_parameters.json')['noise']
    with np.load(pilot.OUT/'instrument_arrays.npz', allow_pickle=False) as arrays:
        old_bias = arrays['bias'].copy()
    with np.load(ROOT/'artifacts.local/work/real-twin-joint-calibration-dev-20261011/instrument_fit_arrays.npz', allow_pickle=False) as arrays:
        new_bias = arrays['bias'].reshape(64, 16)
    a, b, c = old_parameters['adapter_coefficients_a_b_c']
    manifests = read(out/'sealed_candidates.json')['real_manifests']
    rows = []
    checks = []
    def adapt(observation, path):
        denominator = np.sqrt(np.maximum(a*np.maximum(old_bias, 0)+
            b*np.maximum(observation['A'], 0)[..., None]+c, old_parameters['numerical_variance_floor']))
        return (observation['H']-(old_bias if path == 'v2' else new_bias))/denominator
    for arm, manifest in manifests.items():
        engine = DesignReplay(manifest)
        try:
            for sid in ('4', '8'):
                observation = {key:np.asarray(data[sid][key])[:16] for key in ('H', 'A', 'seq', 'ms')}
                for path in ('v2', 'bias_only'):
                    result = engine.infer(adapt(observation, path).reshape(16, 8, 8, 16), observation['seq'], observation['ms'])
                    with np.load(out/'real'/arm/path/(sid+'.npz'), allow_pickle=False) as original:
                        maximum = 0.
                        for name in ('m3_raw', 'local_raw', 'ordinary_raw', 's_seed_score', 's_score'):
                            left, right = np.asarray(result[name]), np.asarray(original[name])
                            if name in ('ordinary_raw', 's_seed_score'):
                                right = right[:, :16]
                            else:
                                right = right[:16]
                            active = np.isfinite(left) & np.isfinite(right)
                            if active.any():
                                maximum = max(maximum, float(np.abs(left[active]-right[active]).max()))
                            np.testing.assert_allclose(left, right, atol=1e-4, rtol=1e-5, equal_nan=True)
                        for name in ('m3', 'old5', 'S_grade', 'ready', 'history', 'reset'):
                            np.testing.assert_array_equal(result[name], original[name][:16])
                    rows.append(dict(arm=arm, condition=sid, path=path, frames=16, maximum_absolute_score_error=maximum))
            seq = np.arange(16); ms = seq*200
            rng = np.random.default_rng(20261011192)
            base = rng.normal(0, 1, (16, 8, 8, 16)).astype(np.float32)
            poisoned = base.copy()
            poisoned[..., engine.drop_bins] = rng.uniform(-2e4, 2e4, poisoned[..., engine.drop_bins].shape)
            first = engine.infer(base, seq, ms)
            second = engine.infer(poisoned, seq, ms)
            for name in ('m3_raw', 'local_raw', 'ordinary_raw', 's_seed_score', 's_score', 'm3', 'old5', 'S_grade'):
                np.testing.assert_array_equal(first[name], second[name])
            checks.append(dict(arm=arm, dropped_bins=list(engine.drop_bins), exact_poison_invariance=True))
            ambient = data['2']['A'][data['2']['stable']].mean(0)
            observation = dict(H=np.repeat(new_bias[None], 16, 0), A=np.repeat(ambient[None], 16, 0))
            for path in ('v2', 'bias_only'):
                result = engine.infer(adapt(observation, path).reshape(16, 8, 8, 16), seq, ms)
                with np.load(out/'real'/arm/path/'bin0_only.npz', allow_pickle=False) as original:
                    for name in ('m3_raw', 'local_raw', 'ordinary_raw', 's_seed_score', 's_score'):
                        np.testing.assert_allclose(result[name], original[name], atol=1e-4, rtol=1e-5, equal_nan=True)
                    for name in ('m3', 'old5', 'S_grade', 'ready'):
                        np.testing.assert_array_equal(result[name], original[name])
                rows.append(dict(arm=arm, condition='BIN0_BACKGROUND_ONLY', path=path, frames=16, exact_grades=True))
        finally:
            engine.close()
    result = dict(status='PASS', seconds=time.monotonic()-started, backend='Fresh CUDA FP32 engineering and CPU HGB',
        independent_adapter=True, sealed_models=True, windows=rows, poison_checks=checks,
        protected_access=0, training=0)
    with (out/'independent_forward_audit.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', seconds=result['seconds'], windows=len(rows), poison_checks=len(checks))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--forward', action='store_true')
    args = parser.parse_args()
    (forward_audit if args.forward else run)(args.out)
