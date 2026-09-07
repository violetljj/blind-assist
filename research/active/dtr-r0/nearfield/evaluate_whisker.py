"""NF-G8 validation-frozen, current-near and approaching Development metrics.

CPU placement: TASK_NOT_GPU_SUITABLE (cached scalar and small mask accounting).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HEADS = ('BODYnear', 'HEADnear', 'BODYapproaching', 'HEADapproaching')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def ratio(a, b):
    return a / b if b else None

def validate_samples(samples):
    ids = [s['sample_id'] for s in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate sample IDs')
    groups, clips = {}, {}
    for s in samples:
        if s['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        for table, key in ((groups, s['group_id']), (clips, s['clip_id'])):
            if key in table and table[key] != s['split']:
                raise ValueError('Group or clip crosses split')
            table[key] = s['split']
    if set(groups.values()) != {'train', 'val', 'test'}:
        raise ValueError('Missing split')

def normalize_predictions(payload, samples):
    if payload.get('complete') is False:
        raise ValueError('Incomplete predictions')
    ids = payload['sample_ids']
    if len(ids) != len(set(ids)) or set(ids) != {s['sample_id'] for s in samples}:
        raise ValueError('Prediction coverage mismatch')
    order = payload.get('target_order', list(HEADS))
    if order != list(HEADS):
        raise ValueError('Target order mismatch: '+str(order))
    result = {}
    for arm, data in payload['arms'].items():
        seeds = data['seeds']
        modes = set(next(iter(seeds.values())))
        for seed, values in seeds.items():
            if set(values) != modes:
                raise ValueError('Seed modes disagree')
            for mode, scores in values.items():
                array = np.asarray(scores, dtype=float)
                if array.shape != (len(ids), 4) or not np.isfinite(array).all() or (((array < 0) & (array != -1)) | (array > 1)).any():
                    raise ValueError('Invalid score shape/range')
                if (array == -1).any() and payload.get('unknown_score') != -1:
                    raise ValueError('Unknown predictions require explicit unknown_score=-1 contract')
                result[arm, str(seed), mode] = {sid: array[i] for i, sid in enumerate(ids)}
        # Recompute ensembles, never trust a differently selected supplied ensemble.
        for mode in modes:
            result[arm, 'ensemble', mode] = {
                sid: np.where(np.any(np.asarray([result[arm, str(seed), mode][sid] for seed in seeds]) < 0, axis=0), -1,
                              np.mean([result[arm, str(seed), mode][sid] for seed in seeds], axis=0))
                for sid in ids}
    return result

def thresholds(rows, samples, targets, max_fpr=.05):
    """Smallest threshold meeting validation-negative FPR; >= defines alert.

    With monotone thresholding this also maximizes validation-positive recall.
    No test score or test label is inspected. No negative support means abstain.
    """
    selected = []
    for h in range(4):
        negatives = sorted(float(rows[s['sample_id']][h]) for s in samples
                           if s['split'] == 'val' and targets[s['sample_id']][h] == 0 and rows[s['sample_id']][h] >= 0)
        if not negatives:
            selected.append(dict(value=float(np.nextafter(1., math.inf)), negatives=0,
                                 allowed_fp=0, actual_fp=0, fpr=None, status='NOT_EVALUABLE'))
            continue
        allowed = math.floor(max_fpr * len(negatives) + 1e-12)
        bound = float(np.nextafter(negatives[len(negatives)-allowed-1], math.inf))
        fp = sum(v >= bound for v in negatives)
        selected.append(dict(value=bound, negatives=len(negatives), allowed_fp=allowed,
                             actual_fp=fp, fpr=fp/len(negatives), status='FROZEN_FROM_VALIDATION'))
    return selected

def confusion(rows, subset, targets, frozen):
    cells = {}
    for h, name in enumerate(HEADS):
        c = dict(TP=0, FP=0, FN=0, TN=0, UNKNOWN=0, truth_UNKNOWN=0,
                 prediction_UNKNOWN=0, unknown_positive=0, unknown_negative=0, total=len(subset))
        for s in subset:
            sid = s['sample_id']; truth = targets[sid][h]
            if truth == -1:
                c['UNKNOWN'] += 1
                c['truth_UNKNOWN'] += 1
                continue
            if rows[sid][h] < 0:
                c['UNKNOWN'] += 1
                c['prediction_UNKNOWN'] += 1
                c['unknown_positive' if truth == 1 else 'unknown_negative'] += 1
                continue
            alert = rows[sid][h] >= frozen[h]['value']
            c[('TP' if alert else 'FN') if truth == 1 else ('FP' if alert else 'TN')] += 1
        c.update(known=c['total']-c['UNKNOWN'], coverage=ratio(c['total']-c['UNKNOWN'], c['total']),
                 recall=ratio(c['TP'], c['TP']+c['FN']), actual_FPR=ratio(c['FP'], c['FP']+c['TN']),
                 precision=ratio(c['TP'], c['TP']+c['FP']))
        c['all_opportunity_recall'] = ratio(c['TP'], c['TP']+c['FN']+c['unknown_positive'])
        c['truth_known_prediction_coverage'] = ratio(c['known'], c['total']-c['truth_UNKNOWN'])
        cells[name] = c
    return cells

def matched_near_retention(rows, samples, targets, frozen, metadata):
    """Evaluator-only matched current geometry; never supplies learner pose inputs."""
    groups = {}
    for s in samples:
        if s['split'] != 'test':
            continue
        sid = s['sample_id']; meta = metadata[sid]
        if 'matched_pair_id' in meta:
            groups.setdefault(meta['matched_pair_id'], {})[meta['matched_pair_role']] = sid
    roles = ('approach', 'stop', 'static')
    verified = {gid: pair for gid, pair in groups.items() if set(pair) == set(roles)
                and all(metadata[sid].get('matched_geometry_verified') for sid in pair.values())}
    result = dict(candidate_groups=len(groups), verified_geometry_groups=len(verified),
        NOT_EVALUABLE_geometry=len(groups)-len(verified),
        scope='Approach t2 versus stop/static t5; exact evaluator camera/wearer/objects equality checked; all three near labels must be 1. UNKNOWN predictions are not retained detections and are excluded from probability deltas.', parts={})
    for h, part in enumerate(('BODY', 'HEAD')):
        pairs = []
        for gid, pair in verified.items():
            if not all(targets[sid][h] == 1 for sid in pair.values()):
                continue
            probabilities = {role: float(rows[pair[role]][h]) for role in roles}
            detections = {role: probabilities[role] >= frozen[h]['value'] for role in roles}
            deltas = {role+'_minus_approach': probabilities[role]-probabilities['approach']
                      if probabilities[role] >= 0 and probabilities['approach'] >= 0 else None
                      for role in ('stop', 'static')}
            pairs.append(dict(pair_id=gid, sample_ids=pair, probabilities=probabilities,
                              detected=detections, probability_deltas=deltas))
        approach_detected = sum(p['detected']['approach'] for p in pairs)
        cell = dict(positive_triplets=len(pairs), truth_ineligible_triplets=len(verified)-len(pairs),
                    approach_detected=approach_detected,
                    all_three_detected=sum(all(p['detected'].values()) for p in pairs), pairs=pairs)
        for role in ('stop', 'static'):
            kept = sum(p['detected']['approach'] and p['detected'][role] for p in pairs)
            values = [p['probability_deltas'][role+'_minus_approach'] for p in pairs
                      if p['probability_deltas'][role+'_minus_approach'] is not None]
            cell[role+'_kept_given_approach_detected'] = kept
            cell[role+'_dropped_given_approach_detected'] = approach_detected-kept
            cell[role+'_conditional_retention'] = ratio(kept, approach_detected)
            cell[role+'_minus_approach_probability_delta'] = dict(evaluable=len(values),
                mean=float(np.mean(values)) if values else None)
        cell['prediction_UNKNOWN'] = {role: sum(p['probabilities'][role] < 0 for p in pairs) for role in roles}
        result['parts'][part] = cell
    return result

def evaluate(rows, samples, targets, frozen, metadata, support=None, maps=None):
    test = [s for s in samples if s['split'] == 'test']
    cells = confusion(rows, test, targets, frozen)
    by_motion = {motion: confusion(rows, [s for s in test if metadata[s['sample_id']]['motion'] == motion], targets, frozen)
                 for motion in sorted({metadata[s['sample_id']]['motion'] for s in test})}
    by_control = {control: confusion(rows, [s for s in test if metadata[s['sample_id']].get('control', 'unspecified') == control], targets, frozen)
                  for control in sorted({metadata[s['sample_id']].get('control', 'unspecified') for s in test})}
    by_shape = {shape: confusion(rows, [s for s in test if metadata[s['sample_id']].get('shape', 'unspecified') == shape], targets, frozen)
                for shape in sorted({metadata[s['sample_id']].get('shape', 'unspecified') for s in test})}
    clips = {}
    for s in test:
        sid = s['sample_id']; c = clips.setdefault(s['clip_id'], dict(false_alert_samples=[], false_alert_heads={}, samples=0))
        c['samples'] += 1
        wrong = [name for h, name in enumerate(HEADS) if targets[sid][h] == 0 and rows[sid][h] >= frozen[h]['value']]
        if wrong:
            c['false_alert_samples'].append(sid); c['false_alert_heads'][sid] = wrong
    persisted = {}
    for h, part in enumerate(('BODY', 'HEAD')):
        ids = [s['sample_id'] for s in test if metadata[s['sample_id']]['motion'] in ('stop', 'static') and targets[s['sample_id']][h] == 1]
        persisted[part] = dict(positive_opportunities=len(ids), recovered=sum(bool(rows[sid][h] >= frozen[h]['value']) for sid in ids),
                               recall=ratio(sum(bool(rows[sid][h] >= frozen[h]['value']) for sid in ids), len(ids)))
    localization = {}
    for h, part in enumerate(('BODY', 'HEAD')):
        positives = [s['sample_id'] for s in test if targets[s['sample_id']][h] == 1]
        observations = []
        for sid in positives:
            if support is None or maps is None or sid not in support or sid not in maps:
                continue
            truth = np.asarray(support[sid][h], bool); predicted = np.asarray(maps[sid][h], float)
            if truth.shape != (18, 32) or predicted.shape != truth.shape or not np.isfinite(predicted).all() or ((predicted < 0) | (predicted > 1)).any():
                raise ValueError('Invalid localization mask')
            if not truth.any():
                continue  # Positive target without visible raster support is NOT_EVALUABLE.
            binary = predicted >= .5
            iou = float((binary & truth).sum() / (binary | truth).sum())
            observations.append(dict(sample_id=sid, IoU=iou, pointing_hit=bool(truth.flat[int(predicted.argmax())])))
        localization[part] = dict(positive_samples=len(positives), evaluable=len(observations),
            NOT_EVALUABLE=len(positives)-len(observations), coverage=ratio(len(observations), len(positives)),
            mask_threshold=.5, mean_IoU=float(np.mean([x['IoU'] for x in observations])) if observations else None,
            pointing_hit_rate=ratio(sum(x['pointing_hit'] for x in observations), len(observations)), samples=observations)
        localization[part]['IoU_at_least_0_5_rate'] = ratio(sum(x['IoU'] >= .5 for x in observations), len(observations))
    false_ids = sorted(cid for cid, c in clips.items() if c['false_alert_samples'])
    first_opportunity = {}
    for cid in clips:
        window = sorted([s for s in test if s['clip_id'] == cid],
                        key=lambda s: metadata[s['sample_id']].get('time_s') or 0)
        first_opportunity[cid] = {}
        for h, part in enumerate(('BODY', 'HEAD')):
            positive = [s['sample_id'] for s in window if targets[s['sample_id']][h] == 1]
            detected = [sid for sid in positive if rows[sid][h] >= frozen[h]['value']]
            first_sid = positive[0] if positive else None
            first_detection = detected[0] if detected else None
            first_time = metadata[first_sid].get('time_s') if first_sid else None
            detection_time = metadata[first_detection].get('time_s') if first_detection else None
            first_opportunity[cid][part] = dict(first_positive_sample=first_sid,
                first_positive_simulated_time_s=first_time, first_correct_sample=first_detection,
                first_correct_simulated_time_s=detection_time,
                simulated_detection_delay_s=detection_time-first_time if detection_time is not None and first_time is not None else None)
    return dict(thresholds=frozen, cells=cells, by_motion=by_motion, by_control=by_control, by_shape=by_shape, persisted_near_stop_static=persisted,
                rotation_false_approach={h: by_motion.get('rotation', {}).get(h) for h in HEADS[2:]},
                false_alert_clips=false_ids, false_alert_clip_count=len(false_ids), test_clips=len(clips),
                false_alert_clip_rate=ratio(len(false_ids), len(clips)), clips=clips, localization=localization,
                first_near_opportunity=first_opportunity,
                matched_geometry_near_retention=matched_near_retention(rows, samples, targets, frozen, metadata),
                timing_scope='First observed positive within the four evaluated history windows; simulated time only, not warning lead or actual camera cadence.')

def sample_metadata(dataset, spec):
    clips = {c['clip_id']: c for c in spec['clips']}
    cases_raw = spec.get('cases', [])
    cases = cases_raw if isinstance(cases_raw, dict) else {c.get('case_id', c.get('sample_index', i)): c for i, c in enumerate(cases_raw)}
    frames = {f.get('sample_index', f.get('frame_index', i)): f for i, f in enumerate(dataset.get('frames', []))}
    result, pairs = {}, {}
    for s in dataset['samples']:
        clip = clips[s['clip_id']]
        if clip['split'] != s['split'] or clip['group_id'] != s['group_id']:
            raise ValueError('Metadata split/group mismatch')
        index = s['frame_indices'][-1]
        frame = frames.get(index, {})
        case = cases.get(index, cases.get(str(index), cases.get(frame.get('case_id'), {})))
        motion = case.get('phase', clip.get('motion'))
        if motion not in ('approach', 'stop', 'static', 'rotation'):
            raise ValueError('Unknown sample phase')
        result[s['sample_id']] = dict(motion=motion, control=clip.get('control'), shape=clip.get('shape', 'unspecified'), clip_id=s['clip_id'], time_s=case.get('time_s', frame.get('time_s')))
        role = clip.get('motion')
        if (role == 'approach' and case.get('frame_in_clip') == 2) or (role in ('stop', 'static') and case.get('frame_in_clip') == 5):
            pair_id, suffix = s['clip_id'].rsplit('_', 1)
            if suffix != role:
                raise ValueError('Matched-pair clip naming mismatch')
            geometry = {key: case.get(key) for key in ('camera', 'wearer', 'objects')}
            members = pairs.setdefault(pair_id, {})
            if role in members:
                raise ValueError('Duplicate matched-pair role')
            members[role] = (s['sample_id'], geometry)
    for pair_id, members in pairs.items():
        geometries = [g for _,g in members.values()]
        verified = (set(members) == {'approach', 'stop', 'static'}
                    and all(all(value is not None for value in g.values()) for g in geometries)
                    and all(g == geometries[0] for g in geometries))
        for role, (sid, _) in members.items():
            result[sid].update(matched_pair_id=pair_id, matched_pair_role=role,
                               matched_geometry_verified=verified)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new evaluation output directory; do not overwrite consumed results')
    learned = args.learned/'predictions.json' if args.learned.is_dir() else args.learned
    paths = {'dataset': args.capture/'model/dataset.json', 'labels': args.capture/'evaluator/labels.json',
             'spec': args.capture/'evaluator/spec.json', 'predictions': learned, 'support': args.capture/'evaluator/support.npz'}
    dataset = read(paths['dataset']); samples = dataset['samples']; validate_samples(samples)
    labels = read(paths['labels']); targets = labels.get('targets', labels)
    if set(targets) != {s['sample_id'] for s in samples} or any(len(v) != 4 or any(x not in (-1, 0, 1) for x in v) for v in targets.values()):
        raise ValueError('Label coverage/value mismatch')
    predictions = normalize_predictions(read(learned), samples)
    baseline_arms = set()
    if args.baseline:
        paths['baseline'] = args.baseline
        extra = normalize_predictions(read(args.baseline), samples)
        if set(extra) & set(predictions):
            raise ValueError('Baseline arm collision')
        predictions.update(extra)
        baseline_arms = {a for a, seed, mode in extra}
    # Construct restricted label mapping before selection; test truth cannot enter threshold function.
    val_targets = {s['sample_id']: targets[s['sample_id']] for s in samples if s['split'] == 'val'}
    frozen = {(a, seed): thresholds(rows, samples, val_targets) for (a, seed, mode), rows in predictions.items() if mode == 'normal'}
    metadata = sample_metadata(dataset, read(paths['spec']))
    support = dict(np.load(paths['support'], allow_pickle=False))
    map_path = learned.parent/'support_predictions.npz'
    map_cache = dict(np.load(map_path, allow_pickle=False)) if map_path.exists() else {}
    if map_cache:
        paths['support_predictions'] = map_path
    if args.baseline:
        baseline_maps_path = args.baseline.parent/'support_predictions.npz'
        if baseline_maps_path.exists():
            baseline_maps = dict(np.load(baseline_maps_path, allow_pickle=False))
            baseline_ids = [str(x) for x in baseline_maps.pop('test_sample_ids')]
            existing_ids = [str(x) for x in map_cache.get('test_sample_ids', baseline_ids)]
            if len(baseline_ids) != len(set(baseline_ids)) or set(existing_ids) != set(baseline_ids):
                raise ValueError('Baseline localization coverage mismatch')
            positions = {sid:i for i,sid in enumerate(baseline_ids)}
            permutation = [positions[sid] for sid in existing_ids]
            for key,value in baseline_maps.items():
                if key in map_cache or value.shape != (len(baseline_ids),2,18,32):
                    raise ValueError('Baseline localization shape/key mismatch')
                map_cache[key] = value[permutation]
            map_cache['test_sample_ids'] = np.asarray(existing_ids)
            paths['baseline_support_predictions'] = baseline_maps_path
    test_ids = [str(x) for x in map_cache.get('test_sample_ids', [])]
    if map_cache and (len(test_ids) != len(set(test_ids)) or set(test_ids) != {s['sample_id'] for s in samples if s['split'] == 'test'}):
        raise ValueError('Localization test sample coverage mismatch')
    evaluations = {}
    for (arm, seed, mode), rows in predictions.items():
        key = f'{arm}__'+('ensemble' if seed == 'ensemble' else 'seed'+seed)+f'__{mode}'
        values = map_cache.get(key)
        if values is not None and values.shape != (len(test_ids), 2, 18, 32):
            raise ValueError('Localization shape mismatch')
        maps = {sid: values[i] for i, sid in enumerate(test_ids)} if values is not None else None
        evaluations[f'{arm}/{seed}/{mode}'] = evaluate(rows, samples, targets, frozen[arm, seed], metadata, support, maps)
        if arm in baseline_arms and mode == 'normal':
            fixed = [dict(value=.5, status='PREDECLARED_BINARY_BASELINE_FIXED_POINT') for _ in HEADS]
            evaluations[f'{arm}/{seed}/fixed_0_5'] = evaluate(rows, samples, targets, fixed, metadata, support, maps)
    result = dict(schema='nf-g8-whisker-evaluation-v1', evaluations=evaluations,
        input_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        threshold_rule='Per head <=5% validation known-negative FPR, smallest feasible >= threshold; recomputed seed average ensemble; repeated_history reuses normal thresholds. Actual test FPR is reported, never matched.',
        unknown_rule='Unknown truth and explicitly declared prediction -1 are separately counted and excluded from known confusion denominators; unknown predictions on positive truth remain misses in all_opportunity_recall. Reported FPR is conditional on known predictions; prediction coverage is explicit. Localization positive without visible support is NOT_EVALUABLE. Label 0 means NO_VISIBLE_TASK_OBSTRUCTION_IN_QUERY, not certified all-scene free space. Visibility beyond partial occlusion is not certified.',
        interpretation='Controlled synthetic Development only. Near is current visible proximity independent of approach. Repeated history tests joint temporal information, not distance proof. Ordinary useful effect may be retained without a bio-inspired extra gain; no test checkpoint or threshold tuning and no automatic deployment claim; component disposition is exploratory.',
        backend='CPU: TASK_NOT_GPU_SUITABLE')
    args.output.mkdir(parents=True)
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# NF-G8 current-near evaluation', '', result['interpretation'], '', '| Arm / seed / mode | Head | TP | FP | FN | UNKNOWN | Recall | Test FPR |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, ev in evaluations.items():
        for head,c in ev['cells'].items():
            lines.append(f"| {name} | {head} | {c['TP']} | {c['FP']} | {c['FN']} | {c['UNKNOWN']} | {c['recall']} | {c['actual_FPR']} |")
    lines.extend(['', result['threshold_rule'], '', 'Thresholds, phase-specific persistence, per-clip false alerts and visible-support localization are in result.json.'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'evaluations': len(evaluations)}))

if __name__ == '__main__':
    main()
