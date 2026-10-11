"""Independent saved-output audit for no-bin0 domain-randomized retraining.

No renderer, trainer or producer evaluator imports. All truth is used only for
recomputation of descriptive evaluation, never for model inference or selection.
"""
import argparse
import hashlib
import json
import csv
import pickle
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

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


def load_scores(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name:saved[name].copy() for name in saved.files}


def smooth(raw):
    raw = np.asarray(raw, dtype=np.float64)
    output = np.empty_like(raw)
    assert raw.shape[-2:] == (13, 2)
    for frame in range(13):
        start = max(0, frame-4)
        weights = 2.**np.arange(frame-start+1)
        output[..., frame, :] = (raw[..., start:frame+1, :]*weights[:, None]).sum(-2)/weights.sum()
    return output


def strong_grades(scores, cuts):
    return (2*((smooth(scores['m3_raw']) >= cuts['raised_tau']) |
               (smooth(scores['local_raw']) >= cuts['local_tau']))).astype(np.int8)


def features47(scores, single):
    """Independent feature arithmetic, retaining inherited dtype transitions."""
    raw = scores['ordinary_raw']
    smoothed = smooth(raw)
    slope, residual = np.zeros_like(raw, dtype=float), np.zeros_like(raw, dtype=float)
    for frame in range(13):
        y = raw[..., max(0, frame-4):frame+1, :]
        x = np.arange(y.shape[-2], dtype=float)
        x -= x.mean()
        mean = y.mean(-2)
        if len(x) > 1:
            fit = (y*x[:, None]).sum(-2)/(x*x).sum()
            slope[..., frame, :] = fit
            residual[..., frame, :] = np.sqrt(((y-mean[..., None, :]-fit[..., None, :]*x[:, None])**2).mean(-2))
    current = scores['current'].astype(float)
    valid = scores['current_valid'] & np.isfinite(current)
    spatial = np.concatenate((np.where(valid, current, np.nan), (~valid).astype(float)), -1)
    return np.asarray([np.concatenate((np.stack((smoothed[i]-single[i], slope[i], residual[i]), -1),
        spatial), -1) for i in range(3)], np.float32)


def predict_hgb(folder, features):
    output = np.empty(features.shape[:-1], np.float32)
    with threadpool_limits(limits=2):
        for i, seed in enumerate((2026100955, 2026100956, 2026100957)):
            for q, height in enumerate(('HEAD', 'BODY')):
                with (Path(folder)/'models'/f'S_{seed}_{height}.pickle').open('rb') as stream:
                    model = pickle.load(stream)
                values = features[i, ..., q, :]
                output[i, ..., q] = model.decision_function(values.reshape(-1, 47)).reshape(values.shape[:-1])
    return output.mean(0)


def append_light(strong, score, tau):
    return np.where(strong == 2, 2, np.where(np.isfinite(score) & (score >= tau), 1, 0)).astype(np.int8)


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
    # The operating budget references original old5; the candidate curve must
    # start from this arm's own immutable strong clips, which can differ.
    counts = int(strong.sum())+len(vals)-np.searchsorted(vals, taus, side='left')
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
    assert seal['plan_markdown_sha256'] == sha(out/'PLAN.md')
    for mapping in ('scientific_code_sha256', 'old_frozen_dependencies_sha256'):
        assert seal[mapping], mapping
        for path, digest in seal[mapping].items():
            assert sha(ROOT/path) == digest, (mapping, path)
    calibration = read(out/'sealed_calibration.json')['cuts']
    for path, digest in seal['models_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for arm, digest in seal['real_manifest_hashes'].items():
        manifest = Path(seal['real_manifests'][arm])
        assert sha(manifest) == digest
        parsed = read(manifest)
        assert parsed['drop_bins'] == ([0] if arm == 'main' else [0, 1])
        assert parsed['real_selection'] is False and parsed['hold_selection'] is False
        assert parsed['cuts'] == calibration['no_bin0' if arm == 'main' else 'no_bin01']
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
    model_files = [ROOT/path for path in seal['models_sha256'] if path.endswith('.pt')]
    for path in model_files:
        assert '_partial_' not in path.name, 'Incomplete diagnostic checkpoint cannot be a candidate'
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
                bounds_checked=len(draws)*len(limits), model_hashes=len(seal['models_sha256']),
                scientific_code_hashes=len(seal['scientific_code_sha256']),
                old_frozen_hashes=len(seal['old_frozen_dependencies_sha256']), plan_hashes=2)


def calibration_audit(out):
    pool_rows = read(out/'pool_rows.json')
    with np.load(out/'pool_metadata.npz', allow_pickle=False) as saved:
        category, original_old5 = saved['category'], saved['old5']
    seal = read(out/'sealed_calibration.json')
    answer = {}
    for arm in ('no_bin0', 'no_bin01'):
        with np.load(out/arm/'pool_light_scores.npz', allow_pickle=False) as saved:
            score, strong = saved['score'], saved['strong']
        raw = load_scores(out/(arm+'_pool_scores.npz'))
        cuts = seal['cuts'][arm]
        np.testing.assert_array_equal(strong, strong_grades(raw, cuts))
        fresh_score = predict_hgb(out/arm, features47(raw, cuts['single']))
        np.testing.assert_array_equal(score, fresh_score)
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


def supervision_audit(out):
    source = ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010'
    rows = read(source/'scene_rows.json')['train']
    with np.load(source/'data/train/geometry.npz', allow_pickle=False) as geometry:
        category, support = geometry['category'], geometry['target_support']
    cuts = read(out/'sealed_calibration.json')['cuts']
    answer = {}
    for arm in ('no_bin0', 'no_bin01'):
        scores = load_scores(out/'S_train_scores'/(arm+'_scores.npz'))
        strong = strong_grades(scores, cuts[arm])
        visibility = np.broadcast_to(support[:, None], strong.shape) if support.ndim == 3 else support
        visible_before = np.logical_or.accumulate(visibility, axis=2)
        contact = np.broadcast_to((category == 'contact')[:, None, None], strong.shape)
        timely = np.arange(3, 16) <= 13
        positive = contact & visible_before & timely[None, None, :, None]
        mask = (~contact | positive) & (strong == 0)
        near = (category == 'pass').any(1) & np.array([float(r['lateral_gap_m']) <= .10+1e-10 for r in rows])
        weights = np.broadcast_to(np.where(near, .25, 1.)[:, None, None, None], strong.shape).copy()*mask
        counts = {}
        for q, height in enumerate(('HEAD', 'BODY')):
            active_pos = positive[..., q] & mask[..., q]
            active_neg = ~positive[..., q] & mask[..., q]
            negative_weight = weights[..., q][active_neg].sum()
            npos = int(active_pos.sum())
            assert npos and negative_weight > 0
            weights[..., q][active_pos] = negative_weight/npos
            counts[height] = dict(positive_slots=npos, negative_slots=int(active_neg.sum()),
                ignored_slots=int((~mask[..., q]).sum()), positive_weight=float(weights[..., q][active_pos].sum()),
                negative_weight=float(negative_weight))
        labels = load_scores(out/arm/'S_train_labels.npz')
        np.testing.assert_array_equal(labels['labels'], positive.astype(np.float32))
        np.testing.assert_array_equal(labels['mask'], mask)
        np.testing.assert_array_equal(labels['weights'], weights.astype(np.float32))
        assert counts == read(out/arm/'supervision_receipt.json')
        saved_features = np.load(out/arm/'S_train_features.npy', mmap_mode='r')
        recomputed = features47(scores, cuts[arm]['single'])
        np.testing.assert_array_equal(saved_features, recomputed)
        answer[arm] = dict(labels_checked=int(mask.size), features_checked=int(recomputed.size),
                          ignored_new_strong_only=True, counts=counts)
    return answer


def hold_audit(out, rows):
    saved = np.load(out/'hold_grades_notifications.npz', allow_pickle=False)
    keys = [str(k) for k in saved['keys']]
    grades, notifications, category = saved['grades'], saved['notifications'], saved['category']
    reference = grades[keys.index('original_old5')]
    producer = read(out/'simulation_evaluation.json')
    sealed_cuts = read(out/'sealed_calibration.json')['cuts']
    old = load_scores(out/'hold_scores/old_scores.npz')
    old_cuts = dict(raised_tau=.9404184587540165, local_tau=4.625390338985158)
    old_strong = strong_grades(old, old_cuts)
    thresholds = read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    single = [thresholds[str(seed)]['single'] for seed in (2026100955, 2026100956, 2026100957)]
    old_light = predict_hgb(ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010', features47(old, single))
    old_manifest = read(ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json')
    reconstructed = dict(original_old5=old_strong,
        old_default_S=append_light(old_strong, old_light, old_manifest['threshold']['tau']))
    for arm in ('no_bin0', 'no_bin01'):
        scores = load_scores(out/'hold_scores'/(arm+'_scores.npz'))
        cuts = sealed_cuts[arm]
        strong = strong_grades(scores, cuts)
        light = predict_hgb(out/arm, features47(scores, cuts['single']))
        reconstructed[arm+'_old5'] = strong
        reconstructed[arm+'_S'] = append_light(strong, light, cuts['s_tau'])
        reconstructed[arm+'_M3'] = (2*(smooth(scores['m3_raw']) >= cuts['m3_tau'])).astype(np.int8)
    assert set(keys) == set(reconstructed)
    for i, name in enumerate(keys):
        np.testing.assert_array_equal(grades[i], reconstructed[name])
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
        hit, baseline_hit = (emitted[:, :, :11] > 0).any(2), (scalar_notifications(reference)[:, :, :11] > 0).any(2)
        groups = {}
        for name, selected in (('dark_thin', np.array([r['dark_thin'] for r in rows])),
                               ('sign_edge', np.array([r['shape_family'] == 'sign_edge' for r in rows]))):
            groups[name] = []
            for q in range(2):
                mask = selected & (category[:, q] == 'contact')
                groups[name].append(dict(denominator=int(mask.sum()*grades[i].shape[1]),
                    timely=int(hit[mask, :, q].sum()), baseline=int(baseline_hit[mask, :, q].sum())))
            assert groups[name] == reported['groups'][name], (arm, name)
        result['groups'] = groups
        answer[arm] = result
    with (out/'simulation_comparison.csv').open(encoding='utf8', newline='') as stream:
        summary_rows = list(csv.DictReader(stream))
    assert len(summary_rows) == len(keys)*2
    for row in summary_rows:
        result = answer[row['arm']]
        assert all(int(row[name]) == value for name, value in result['contacts'][row['height']].items())
        assert int(row['far_clear_notified_clips']) == result['far_clear_notified_clips']
        assert int(row['far_clear_clip_denominator']) == result['far_clear_clip_denominator']
        assert float(row['weighted_cost']) == result['weighted_cost']
    with (out/'simulation_groups.csv').open(encoding='utf8', newline='') as stream:
        group_rows = list(csv.DictReader(stream))
    assert len(group_rows) == len(keys)*4
    for row in group_rows:
        result = answer[row['arm']]['groups'][row['group']][('HEAD', 'BODY').index(row['height'])]
        assert all(int(row[name]) == value for name, value in result.items())
    calculated = acceptance(answer['no_bin0_S'], answer['old_default_S'])
    declared = producer['acceptance']
    assert calculated['timely_pass'] == declared['timely_pass']
    assert calculated['far_clear_pass'] == declared['far_clear_pass']
    assert calculated['new_gain'] == declared['new_gain_vs_common_old5']
    assert calculated['old_gain'] == declared['old_gain_vs_common_old5']
    assert abs(calculated['far_clear_increase_pp']-declared['far_clear_increment_pp']) < 1e-10
    return dict(arms=len(keys), notifications_checked=int(notifications.size),
                grades_rebuilt_from_scores=True, HGB_models_reloaded=18,
                subgroup_rows_checked=len(group_rows), summary_rows_checked=len(summary_rows),
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
    index = {(r['arm'], r['path'], r['condition'], r['height'], r['model']):r for r in receipt['tables']}
    gates = {}
    for path in ('v2', 'bias_only'):
        empty = [index[('main', path, 'BIN0_BACKGROUND_ONLY', height, model)]
            for height in ('HEAD', 'BODY') for model in ('M3', 'old5', 'S')]
        bin0_pass = all(row['strong'] == row['light'] == 0 for row in empty)
        presence = {}
        for sid in ('2', '3', '5', '6', '8'):
            heights = ('HEAD',) if sid == '8' else ('HEAD', 'BODY')
            presence[sid] = any(index[('main', path, sid, height, 'S')]['strong']+
                               index[('main', path, sid, height, 'S')]['light'] > 0 for height in heights)
        gates[path] = dict(bin0_pass=bin0_pass, reminder_present=presence, walls_rods_pass=all(presence.values()))
    return dict(table_rows_checked=checks, output_hashes=len(receipt['outputs']),
                poison_invariance_receipts=receipt['input_checks'], gate_interpretations=gates)


def run(out):
    start = time.monotonic()
    identities = identity_audit(out)
    rows, source = source_audit(out)
    calibration = calibration_audit(out)
    supervision = supervision_audit(out)
    hold = hold_audit(out, rows)
    real = real_audit(out) if (out/'real_receipt.json').exists() else None
    recommendation = None if real is None else bool(calibration['no_bin0']['status'] == 'SEALED' and
        hold['acceptance']['timely_pass'] and hold['acceptance']['far_clear_pass'] and
        real['gate_interpretations']['v2']['bin0_pass'] and real['gate_interpretations']['v2']['walls_rods_pass'])
    result = dict(status='PASS', seconds=time.monotonic()-start, independent=True,
        protected_access=0, training=0, source_sha256=sha(__file__), identities=identities,
        source=source, calibration=calibration, supervision=supervision, hold=hold, real=real,
        independent_recommendation=recommendation)
    output = out/('independent_audit_real.json' if real is not None else 'independent_audit_sim.json')
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        calibration=calibration, acceptance=hold['acceptance'], real_rows=None if real is None else real['table_rows_checked']), ensure_ascii=False), flush=True)
    return result


def forward_audit(out, seconds=60.):
    """Fresh CUDA process: sealed candidate, first16 pilot frames and poison.

    Prefix windows have exactly the same causal history as saved full episodes.
    Adapter arithmetic is recomputed here without importing the real producer.
    """
    import sys
    sys.path.insert(0, str(ROOT/'research/active/hardware-bringup/host'))
    import real_pilot_calibration as pilot
    from cnh_no_bin0_dr_replay import DesignReplay
    started = time.monotonic()
    if seconds <= 0:
        raise ValueError('Forward audit needs a positive reserved command-stage budget')
    def deadline():
        if time.monotonic()-started >= seconds:
            raise TimeoutError('Independent forward audit reached reserved wall budget')
    deadline()
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
        deadline()
        engine = DesignReplay(manifest)
        try:
            for sid in ('4', '8'):
                observation = {key:np.asarray(data[sid][key])[:16] for key in ('H', 'A', 'seq', 'ms')}
                for path in ('v2', 'bias_only'):
                    deadline()
                    result = engine.infer(adapt(observation, path).reshape(16, 8, 8, 16), observation['seq'], observation['ms'], deadline)
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
            deadline()
            first = engine.infer(base, seq, ms, deadline)
            second = engine.infer(poisoned, seq, ms, deadline)
            for name in ('m3_raw', 'local_raw', 'ordinary_raw', 's_seed_score', 's_score', 'm3', 'old5', 'S_grade'):
                np.testing.assert_array_equal(first[name], second[name])
            checks.append(dict(arm=arm, dropped_bins=list(engine.drop_bins), exact_poison_invariance=True))
            ambient = data['2']['A'][data['2']['stable']].mean(0)
            observation = dict(H=np.repeat(new_bias[None], 16, 0), A=np.repeat(ambient[None], 16, 0))
            for path in ('v2', 'bias_only'):
                deadline()
                result = engine.infer(adapt(observation, path).reshape(16, 8, 8, 16), seq, ms, deadline)
                with np.load(out/'real'/arm/path/'bin0_only.npz', allow_pickle=False) as original:
                    for name in ('m3_raw', 'local_raw', 'ordinary_raw', 's_seed_score', 's_score'):
                        np.testing.assert_allclose(result[name], original[name], atol=1e-4, rtol=1e-5, equal_nan=True)
                    for name in ('m3', 'old5', 'S_grade', 'ready'):
                        np.testing.assert_array_equal(result[name], original[name])
                rows.append(dict(arm=arm, condition='BIN0_BACKGROUND_ONLY', path=path, frames=16, exact_grades=True))
        finally:
            engine.close()
    deadline()
    result = dict(status='PASS', seconds=time.monotonic()-started, backend='Fresh CUDA FP32 engineering and CPU HGB',
        independent_adapter=True, sealed_models=True, windows=rows, poison_checks=checks,
        protected_access=0, training=0)
    with (out/'independent_forward_audit.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', seconds=result['seconds'], windows=len(rows), poison_checks=len(checks))), flush=True)
    return result


def branch_audit(out):
    """Saved-output mechanism check: retained-bin signed bias and LOCAL.

    This does not replay, fit, remove another bin, or alter a frozen candidate.
    It identifies the exact sufficient strong branch in the synthetic empty test.
    """
    started = time.monotonic()
    producer = read(out/'bin0_remaining_branch_diagnostic.json')
    pilot_out = ROOT/'artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot'
    with np.load(pilot_out/'instrument_arrays.npz', allow_pickle=False) as saved:
        bias = saved['bias']
    bins = [dict(bin=b, min=float(bias[:, b].min()), max=float(bias[:, b].max()),
                 negative_cells=int((bias[:, b] < 0).sum()), positive_cells=int((bias[:, b] > 0).sum()))
            for b in range(5)]
    assert bins == producer['v2_bias_by_bin']
    cuts = read(out/'main_real_manifest.json')['cuts']
    with np.load(out/'real/main/v2/bin0_only.npz', allow_pickle=False) as arrays:
        ready = arrays['ready']
        local, m3 = arrays['local_score'][ready], arrays['m3_score'][ready]
        local_crossings = (local >= cuts['local_tau']).sum(0).tolist()
        raised_crossings = (m3 >= cuts['raised_tau']).sum(0).tolist()
        score_ranges = dict(m3_score=[m3.min(0).tolist(), m3.max(0).tolist()],
                            local_score=[local.min(0).tolist(), local.max(0).tolist()])
        assert local_crossings == producer['local_crossings'] == [int(ready.sum())]*2
        assert raised_crossings == producer['raised_crossings'] == [0, 0]
        assert score_ranges == producer['main_empty_v2_score_ranges']
        np.testing.assert_array_equal(arrays['old5'][ready], local >= cuts['local_tau'])
        np.testing.assert_array_equal(arrays['S_grade'][ready], 2*(local >= cuts['local_tau']))
    for name, value in producer['cuts'].items():
        assert cuts[name] == value
    with np.load(out/'real/main/bias_only/bin0_only.npz', allow_pickle=False) as arrays:
        mask = arrays['ready']
        assert not arrays['m3'][mask].any() and not arrays['old5'][mask].any() and not arrays['S_grade'][mask].any()
    result = dict(status='PASS', seconds=time.monotonic()-started, GPU_seconds=0.,
        role='Independent saved-output sufficient-branch diagnosis; no repair',
        v2_bias_by_bin=bins, ready_frames=int(ready.sum()), cuts={k:cuts[k] for k in producer['cuts']},
        score_ranges=score_ranges, local_crossings=local_crossings, raised_crossings=raised_crossings,
        bin0_mask_cannot_remove_retained_signed_bias=True, bias_only_empty_silent=True,
        result='Main v2 bin0-only strong failure is inherited LOCAL support, not retrained M3 raised threshold.',
        producer_diagnostic_sha256=sha(out/'bin0_remaining_branch_diagnostic.json'), source_sha256=sha(__file__))
    with (out/'independent_branch_audit.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', seconds=result['seconds'], local_crossings=local_crossings,
        raised_crossings=raised_crossings, bias_only_empty_silent=True)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--branch', action='store_true')
    parser.add_argument('--seconds', type=float, default=60., help='Reserved forward-audit stage wall seconds; full command wall is booked externally')
    args = parser.parse_args()
    if args.branch:
        branch_audit(args.out)
    elif args.forward:
        forward_audit(args.out, args.seconds)
    else:
        run(args.out)
