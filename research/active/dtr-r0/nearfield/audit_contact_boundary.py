"""Independent saved-output audit; NumPy/stdlib, no producer or model imports.

Geometry/metrics are scalar and small-array work: TASK_NOT_GPU_SUITABLE.
The sealed execution source establishes a procedural boundary, not isolation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] / 'artifacts.local/evidence/ba-contact-boundary-20260923'
ARMS = ('direct', 'geometry')
SPLITS = ('train', 'dev', 'evaluation')
CLASSIFICATIONS = ('seen', 'width', 'horizon', 'both', 'extrapolation')
BANDS = ((.42, .9), (-.2, .42))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, description):
    if not value:
        raise AssertionError(description)


def compare(actual, expected, where):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), where + ': keys')
        for key in expected:
            compare(actual[key], expected[key], where + '/' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), where + ': length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, where + '/' + str(index))
    elif isinstance(expected, float):
        require(np.isclose(actual, expected, rtol=1e-9, atol=1e-10), where + ': numeric')
    else:
        require(actual == expected, where + ': value')


def expected_queries():
    make = lambda w, h: np.array([(a, b, layer) for layer in (0, 1)
                                 for a in w for b in h], np.float32)
    w = [.36, .60, .84, 1.08]
    h = [.6, .9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.]
    nw = [.48, .72, .96]
    nh = [.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85]
    return dict(seen=make(w, h), width=make(nw, h), horizon=make(w, nh),
                both=make(nw, nh), extrapolation=make([.24, 1.2], nh),
                width_curve=make(np.linspace(.2, 1.2, 51), [3.]),
                horizon_curve=make([.6], np.linspace(.3, 3., 55)))


def world_labels(geo, query):
    """Intersect world AABBs with world query prisms; no camera-box conversion."""
    camera = geo['declared_camera']
    require(all(abs(camera[k]) <= 1e-8 for k in ('pitch', 'yaw', 'roll')), 'camera pose')
    require(max(abs(geo['actual_camera_location_m'][i] - camera[k])
                for i, k in enumerate(('x', 'y', 'z'))) < .002, 'camera location')
    require(max(map(abs, geo['actual_camera_rotation'])) < .002, 'rendered camera rotation')
    q = np.asarray(query, np.float64)
    bands = np.array([BANDS[int(k)] for k in q[:, 2]])
    lower = np.column_stack((np.full(len(q), camera['x'] + .3),
                             camera['y'] - q[:, 0] / 2, camera['z'] - bands[:, 1]))
    upper = np.column_stack((camera['x'] + q[:, 1], camera['y'] + q[:, 0] / 2,
                             camera['z'] - bands[:, 0]))
    hit = np.zeros(len(q), bool)
    for obj in geo['objects']:
        c = np.array(obj['render_bounds_center_m'], np.float64)
        e = np.array(obj['render_bounds_extent_m'], np.float64)
        require(np.isfinite(c).all() and np.isfinite(e).all() and (e > 0).all(), 'world bounds')
        hit |= np.all(np.maximum(c - e, lower) <= np.minimum(c + e, upper), axis=1)
    return hit


def world_boundaries(geo):
    camera = geo['declared_camera']
    result = dict(width=[], horizon=[])
    for yl, yh in BANDS:
        widths, horizons = [], []
        for obj in geo['objects']:
            c = np.array(obj['render_bounds_center_m'])
            e = np.array(obj['render_bounds_extent_m'])
            lo, hi = c - e, c + e
            if hi[2] < camera['z'] - yh or lo[2] > camera['z'] - yl or hi[0] < camera['x'] + .3:
                continue
            if lo[0] <= camera['x'] + 3.:
                lateral_gap = max(0., lo[1] - camera['y'], camera['y'] - hi[1])
                widths.append(2 * lateral_gap)
            if hi[1] >= camera['y'] - .3 and lo[1] <= camera['y'] + .3:
                horizons.append(max(.3, lo[0] - camera['x']))
        result['width'].append(min(widths, default=np.inf))
        result['horizon'].append(min(horizons, default=np.inf))
    return result


def metrics(score, truth, threshold):
    s = np.asarray(score, np.float64)
    y = np.asarray(truth, bool)
    predicted = s >= float(threshold)
    tp = int(np.count_nonzero(predicted & y))
    fp = int(np.count_nonzero(predicted & ~y))
    fn = int(np.count_nonzero(~predicted & y))
    tn = int(np.count_nonzero(~predicted & ~y))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None, FPR=fp/(fp+tn) if fp+tn else None,
                Brier=float(np.square(s-y).mean()))


def atomic_threshold(score, truth):
    """Ascending score groups and reverse cumulative counts, including empty set."""
    s, y = np.asarray(score, np.float64).ravel(), np.asarray(truth, bool).ravel()
    values, inverse = np.unique(s, return_inverse=True)
    positive = np.bincount(inverse, weights=y, minlength=len(values)).astype(np.int64)
    negative = np.bincount(inverse, weights=~y, minlength=len(values)).astype(np.int64)
    tp, fp = positive[::-1].cumsum()[::-1], negative[::-1].cumsum()[::-1]
    candidates = [(0, 0, float(np.nextafter(values[-1], np.inf)))]
    candidates.extend((int(tp[i]), -int(fp[i]), float(value)) for i, value in enumerate(values)
                      if fp[i] * 20 <= int((~y).sum()))
    return max(candidates)[2]


def boundary_report(score, threshold, truth, kind):
    axis = np.linspace(.2, 1.2, 51) if kind == 'width' else np.linspace(.3, 3., 55)
    probability = np.asarray(score, np.float64).reshape(-1, 2, len(axis))
    on = probability >= threshold
    first = np.full(truth.shape, np.inf)
    for frame in range(len(first)):
        for layer in range(2):
            crossings = np.flatnonzero(on[frame, layer])
            if len(crossings):
                first[frame, layer] = axis[crossings[0]]
    left = truth <= axis[0] + 1e-6
    right = truth > axis[-1] + 1e-6
    interior = np.isfinite(truth) & ~left & ~right
    resolved = interior & np.isfinite(first)
    errors = np.full(truth.shape, np.inf)
    errors[resolved] = abs(first[resolved] - truth[resolved])
    count = int(interior.sum())
    correct = int(np.count_nonzero(interior & (errors <= .050001)))
    return dict(interior_true_boundaries=count, left_censored_truth=int(left.sum()),
        right_censored_truth=int(right.sum()), predicted_boundary_coverage=float(resolved.sum()/count) if count else None,
        conditional_MAE_m=float(errors[resolved].mean()) if resolved.any() else None,
        within_5cm=correct, joint_within_5cm=correct/count if count else None,
        wrong_crossings_on_right_censored=int(np.count_nonzero(right & np.isfinite(first))),
        probability_monotonic_violations=int(np.count_nonzero(np.diff(probability, axis=2) < -1e-6)),
        binary_reversals=int(np.count_nonzero(on[:, :, :-1] & ~on[:, :, 1:])))


def pair_report(score, truth, metadata, threshold):
    groups = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[item['base_group_id'], item['frame_in_clip']].append(index)
    changed = correct_order = ties = joint = 0
    s = np.asarray(score, np.float64)
    for rows in groups.values():
        require(len(rows) == 3 and len({metadata[i]['layout_relation'] for i in rows}) == 3,
                'lateral triple coverage')
        for a, b in itertools.combinations(rows, 2):
            for q in np.flatnonzero(truth[a] != truth[b]):
                positive, negative = (a, b) if truth[a, q] else (b, a)
                changed += 1
                correct_order += int(s[positive, q] > s[negative, q])
                ties += int(s[positive, q] == s[negative, q])
                joint += int(s[positive, q] >= threshold and s[negative, q] < threshold)
    return dict(changed_pairs=changed, strict_order_correct=correct_order, ties=ties,
                order_accuracy=(correct_order+.5*ties)/changed if changed else None,
                both_decisions_correct=joint, joint_accuracy=joint/changed if changed else None)


def summary(predictions, labels, boundary, metadata, threshold):
    result = {k: metrics(predictions[k], labels[k], threshold) for k in CLASSIFICATIONS}
    result['boundaries'] = {k: boundary_report(predictions[k+'_curve'], threshold, boundary[k], k)
                            for k in ('width', 'horizon')}
    for field in ('type_id', 'layout_relation'):
        result[field] = {}
        for value in sorted({row[field] for row in metadata}):
            rows = np.array([m[field] == value for m in metadata])
            result[field][value] = metrics(predictions['both'][rows], labels['both'][rows], threshold)
    result['lateral_pairs'] = pair_report(predictions['both'], labels['both'], metadata, threshold)
    result['unknown_sensor_frames'] = sum(bool(m['baseline']['unknown']) for m in metadata)
    primary = result['both']
    result['component_gate'] = bool(primary['recall'] >= .8 and primary['FPR'] <= .05 and
        all((b['joint_within_5cm'] or 0) >= .8 for b in result['boundaries'].values()))
    return result


def bootstrap(predictions, labels, metadata, selections):
    names = sorted({m['base_group_id'] for m in metadata})
    per_group = {}
    for arm in ARMS:
        per_group[arm] = []
        for name in names:
            rows = np.array([m['base_group_id'] == name for m in metadata])
            result = metrics(predictions[arm]['both'][rows], labels['both'][rows], selections[arm]['threshold'])
            per_group[arm].append([result[k] for k in ('TP', 'FN', 'FP', 'TN')])
        per_group[arm] = np.array(per_group[arm])
    rng = np.random.default_rng(202609232)
    differences = []
    for _ in range(1000):
        sample = rng.integers(len(names), size=len(names))
        totals = [per_group[arm][sample].sum(0) for arm in ARMS]
        rates = [[a[0]/max(1, a[0]+a[1]), a[2]/max(1, a[2]+a[3])] for a in totals]
        differences.append(np.subtract(rates[1], rates[0]))
    interval = np.percentile(differences, [2.5, 97.5], axis=0)
    return dict(groups=len(names), resamples=1000, direction='geometry-minus-direct',
                recall_difference_95pct=interval[:, 0].tolist(), FPR_difference_95pct=interval[:, 1].tolist())


def audit(root):
    seal, prediction_seal = read(root/'source-seal.json'), read(root/'prediction-seal.json')
    hashes = {}
    for name, value in seal['code'].items():
        require(digest(root/'source-snapshot'/name) == value, 'sealed code snapshot '+name)
        require(digest(HERE/name) == value, 'current code '+name)
    require(seal['protocol_sha256'] == seal['code']['CONTACT_BOUNDARY_PROTOCOL_20260923.md'], 'protocol seal')
    for name, value in seal['input_hashes'].items():
        require(digest(name) == value, 'source input '+name)
    for name, value in prediction_seal['files'].items():
        require(digest(root/name) == value, 'prediction hash '+name)
    for arm in ARMS:
        require(digest(root/(arm+'.pt')) == prediction_seal['checkpoint_hashes'][arm], 'checkpoint hash')
        require(digest(root/(arm+'-selection.json')) == prediction_seal['selection_hashes'][arm], 'selection hash')
    require(prediction_seal['evaluation_labels_generated'] is False and
            prediction_seal['all_predicted_before_evaluator_join'] is True, 'procedural prediction seal')
    prepared = root.with_name('ba-query-occupancy-20260922-prepared')
    capture = root.with_name('ba-query-occupancy-20260922-capture')
    identities = read(prepared/'observations/identities.json')
    geometry = read(capture/'evaluator/geometry.json')
    visibility = read(prepared/'labels/visibility-audit.json')
    require(len(identities) == len(geometry) == len(visibility) == 1728, 'source row count')
    require(all(all(r['query_label_valid']) for r in visibility), 'source visibility validity')
    cohort = read(root/'cohort.json')
    indices = {s: np.array([i for i, m in enumerate(identities) if m['split'] == s]) for s in SPLITS}
    groups = {s: {identities[i]['base_group_id'] for i in indices[s]} for s in SPLITS}
    for s, frames, layouts in zip(SPLITS, (864, 288, 576), (24, 8, 16)):
        require(len(indices[s]) == frames and len(groups[s]) == layouts, 'split sizes')
        require(indices[s].tolist() == cohort['indices'][s], 'cohort order')
        require(sorted(groups[s]) == cohort['groups'][s], 'cohort groups')
    require(all(not groups[a] & groups[b] for a, b in itertools.combinations(SPLITS, 2)), 'layout leakage')
    for i, (g, m) in enumerate(zip(geometry, identities)):
        require(g['id'] == m['id'] and g['sample_index'] == m['index'] == i, 'geometry identity')
    query = expected_queries()
    stored_query = read(root/'queries.json')
    require(set(query) == set(stored_query), 'query keys')
    for name, q in query.items():
        require(np.array_equal(q, stored_query[name]), 'protocol query '+name)
        require((q[:, 0] > 0).all() and (q[:, 0] <= np.float32(1.2)).all() and
                (q[:, 1] >= np.float32(.3)).all() and (q[:, 1] <= 3).all(), 'query visibility subset')
    for a, b in itertools.combinations(CLASSIFICATIONS, 2):
        require(not set(map(tuple, query[a])) & set(map(tuple, query[b])), 'query overlap')
    require(not set(query['seen'][:, 0]) & set(query['both'][:, 0]), 'new widths')
    require(not set(query['seen'][:, 1]) & set(query['both'][:, 1]), 'new horizons')
    labels, boundaries, checked = {}, {}, 0
    for split, filename in (('train', 'training-targets.npz'), ('dev', 'dev-selection-targets.npz'),
                            ('evaluation', 'evaluation-targets.npz')):
        with np.load(root/filename, allow_pickle=False) as archive:
            saved = dict(archive)
        names = ['seen'] if split == 'dev' else list(query)
        labels[split] = {name: np.array([world_labels(geometry[i], query[name]) for i in indices[split]])
                         for name in names}
        for name in names:
            require(np.array_equal(saved[name], labels[split][name]), 'independent geometry '+split+'/'+name)
            checked += labels[split][name].size
        if split != 'dev':
            each = [world_boundaries(geometry[i]) for i in indices[split]]
            boundaries[split] = {k: np.array([r[k] for r in each]) for k in ('width', 'horizon')}
            for k in boundaries[split]:
                require(np.allclose(saved[k+'_boundary'], boundaries[split][k], atol=2e-14, rtol=0), 'exact world boundary')
    feature_receipt = read(root/'feature-receipt.json')
    require(digest(root/'features.npy') == feature_receipt['features_sha256'], 'features hash')
    features = np.load(root/'features.npy', mmap_mode='r', allow_pickle=False)
    require(features.shape == (1728, 4864) and np.isfinite(features).all(), 'feature shape')
    train_features = np.asarray(features[indices['train']], np.float64)
    with np.load(root/'normalization.npz', allow_pickle=False) as norm:
        mean_error = float(np.max(abs(norm['mean'] - train_features.mean(0))))
        std_error = float(np.max(abs(norm['std'] - np.maximum(train_features.std(0), .01))))
        require(np.allclose(norm['mean'], train_features.mean(0), rtol=1e-4, atol=3e-6), 'train-only mean')
        require(np.allclose(norm['std'], np.maximum(train_features.std(0), .01), rtol=1e-4, atol=3e-6), 'train-only std')
    schedule = np.load(root/'batch-schedule.npy', allow_pickle=False)
    rng = np.random.default_rng(202609231)
    require(np.array_equal(schedule, np.array([rng.permutation(864) for _ in range(100)])), 'shared seeded image schedule')
    result = read(root/'result.json')
    predictions, selections, summaries = {}, {}, {}
    shared = (4864+1)*128 + 2*128 + (128+1)*128
    direct = (132+1)*128 + (128+1)*64 + (64+1)
    geometric = (128+1)*1200
    for arm in ARMS:
        selection = selections[arm] = read(root/(arm+'-selection.json'))
        compare(result['selection'][arm], selection, 'recorded selection')
        require(selection['checkpoints_sha256'] == prediction_seal['checkpoint_hashes'][arm], 'selected checkpoint')
        history = selection['history']
        require([r['epoch'] for r in history] == list(range(10, 101, 10)), 'selection epoch schedule')
        best = min(history, key=lambda r: (r['dev_BCE'], r['epoch']))
        require((selection['epoch'], selection['dev_BCE']) == (best['epoch'], best['dev_BCE']), 'best recorded dev epoch')
        require(selection['updates'] == 2700, 'one fit budget')
        active = direct if arm == 'direct' else geometric
        compare(selection['parameters'], dict(carrier=shared+direct+geometric, shared=shared,
                direct_head=direct, geometry_head=geometric, effective=shared+active,
                unused=direct+geometric-active), 'parameter formulas')
        predictions[arm] = {}
        for split in SPLITS:
            with np.load(root/(arm+'-'+split+'-predictions.npz'), allow_pickle=False) as archive:
                pred = predictions[arm][split] = dict(archive)
            require(set(pred) == set(query), 'prediction query keys')
            for name, value in pred.items():
                require(value.shape == (len(indices[split]), len(query[name])) and
                        np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all(), 'prediction probabilities')
        cutoff = atomic_threshold(predictions[arm]['dev']['seen'], labels['dev']['seen'])
        require(cutoff == selection['threshold'], 'dev-seen-only atomic threshold')
        dev = metrics(predictions[arm]['dev']['seen'], labels['dev']['seen'], cutoff)
        compare(dev, selection['dev'], 'dev counts')
        require(dev['FPR'] <= .05, 'selected dev FPR')
        summaries[arm] = {}
        for split in ('evaluation', 'train'):
            meta = [identities[i] for i in indices[split]]
            independent = summary(predictions[arm][split], labels[split], boundaries[split], meta, cutoff)
            recorded = result['arms'][arm]['training_layouts'] if split == 'train' else {
                k: v for k, v in result['arms'][arm].items() if k != 'training_layouts'}
            compare(independent, recorded, arm+'/'+split)
            summaries[arm][split] = independent
    eval_meta = [identities[i] for i in indices['evaluation']]
    boot = bootstrap({a: predictions[a]['evaluation'] for a in ARMS}, labels['evaluation'], eval_meta, selections)
    compare(boot, result['paired_layout_bootstrap'], 'paired layout bootstrap')
    d, g = (summaries[a]['evaluation']['both'] for a in ARMS)
    relative = ((g['recall']-d['recall'] >= .05 and g['FPR']-d['FPR'] <= .02) or
                (d['FPR'] > 0 and g['FPR'] <= .7*d['FPR'] and g['recall']-d['recall'] >= -.03))
    relative &= all(summaries['geometry']['evaluation']['type_id'][f]['recall']-
                    summaries['direct']['evaluation']['type_id'][f]['recall'] >= -.05
                    for f in summaries['direct']['evaluation']['type_id'])
    require(bool(relative) == result['geometry_relative_advantage'], 'relative advantage gate')
    decision = ('GEOMETRY_COMPONENT' if summaries['geometry']['evaluation']['component_gate'] else
                'DIRECT_COMPONENT' if summaries['direct']['evaluation']['component_gate'] else 'NEITHER_CONTACT_PACKAGE_PASSES')
    require(result['status'] == 'PASS' and result['decision'] == decision, 'decision')
    near_closed_contacts = sum(int(world_labels(g, [[.6, .3, 0], [.6, .3, 1]]).sum()) for g in geometry)
    require(near_closed_contacts == 0, 'zero-thickness horizon affects recorded case')
    for path in root.iterdir():
        if path.is_file():
            hashes[path.name] = digest(path)
    return dict(status='PASS', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        backend='TASK_NOT_GPU_SUITABLE', source_and_prediction_hashes_verified=True,
        code_snapshot_and_current_source_match=True, protocol_query_sets_exact=True,
        independently_checked_world_space_queries=checked, all_source_visibility_valid=True,
        visibility_domain_subset='centred width<=1.2m; original BODY/HEAD bands; axial .3..3m',
        source_layout_split={s:dict(frames=len(indices[s]),groups=len(groups[s])) for s in SPLITS},
        normalization=dict(train_rows=864, mean_max_absolute_error=mean_error, std_max_absolute_error=std_error),
        closed_point_three_horizon_contacts=near_closed_contacts,
        dev_seen_thresholds={a:selections[a]['threshold'] for a in ARMS},
        selected_epochs={a:selections[a]['epoch'] for a in ARMS},
        exact_parameter_formulas={a:selections[a]['parameters'] for a in ARMS},
        decision=decision, geometry_relative_advantage=bool(relative),
        evaluation={a:summaries[a]['evaluation'] for a in ARMS},
        paired_layout_bootstrap=boot, artifact_sha256=hashes, audit_source_sha256=digest(__file__),
        limits=['Seals and source establish procedural order, not process isolation; geometry was available before fitting.',
                'No checkpoint inference or training rerun; recorded epoch selection is checked against saved history.',
                'Feature extraction itself is not rerun; feature hash, shape and train-only normalization are checked.',
                'World-AABB truth covers declared controlled objects, not hidden complete scene geometry.',
                'Consumed same-generator Development; pair ordering is not threshold correctness or boundary accuracy.'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root/'independent-audit.json'
    require(not output.exists(), 'Audit output already exists; never overwrite')
    try:
        report = audit(args.root)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        report = dict(status='FAIL', error=str(error), backend='TASK_NOT_GPU_SUITABLE')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(output), error=report.get('error'))))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
