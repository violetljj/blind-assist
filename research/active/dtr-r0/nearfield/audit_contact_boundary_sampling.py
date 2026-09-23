"""Independent saved-output audit; no fitting, model or producer-module imports.

Reuse the previous independent world-coordinate and metric reference, never
the sampling producer's labels, threshold selector, or reporting functions.
Small NumPy/stdlib work: TASK_NOT_GPU_SUITABLE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np

import audit_contact_boundary as reference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] / 'artifacts.local/evidence/ba-contact-sampling-20260923'
require, compare, read, digest = reference.require, reference.compare, reference.read, reference.digest


def change_report(new, baseline):
    result = {}
    for kind in ('width', 'horizon'):
        current, old = new['boundaries'][kind], baseline['boundaries'][kind]
        require(current['right_censored_truth'] == old['right_censored_truth'] > 0, 'right-censored denominator')
        gain = current['joint_within_5cm'] - old['joint_within_5cm']
        false_change = ((current['wrong_crossings_on_right_censored'] - old['wrong_crossings_on_right_censored'])
                        / current['right_censored_truth'])
        result[kind] = dict(within5cm_gain=gain, right_censored_false_rate_change=false_change,
                            improved=gain >= .10 and false_change <= .05)
    result['joint_boundary_improvement'] = all(result[k]['improved'] for k in ('width', 'horizon'))
    result['query_cost_guard'] = (new['both']['FPR'] - baseline['both']['FPR'] <= .02
                                 and new['both']['recall'] - baseline['both']['recall'] >= -.03)
    return result


def boundary_bootstrap(old_pred, new_pred, boundaries, metadata, old_cutoff, new_cutoff):
    """Additional descriptive interval, separate from frozen decisions/gates."""
    groups = sorted({row['base_group_id'] for row in metadata})
    contributions = {}
    for kind in ('width', 'horizon'):
        values = []
        for group in groups:
            rows = np.array([row['base_group_id'] == group for row in metadata])
            old = reference.boundary_report(old_pred[kind+'_curve'][rows], old_cutoff, boundaries[kind][rows], kind)
            new = reference.boundary_report(new_pred[kind+'_curve'][rows], new_cutoff, boundaries[kind][rows], kind)
            require(old['interior_true_boundaries'] == new['interior_true_boundaries'], 'paired boundary denominator')
            values.append([old['within_5cm'], new['within_5cm'], old['interior_true_boundaries']])
        contributions[kind] = np.array(values)
    rng = np.random.default_rng(202609232)
    differences = {kind: [] for kind in contributions}
    for _ in range(1000):
        sample = rng.integers(len(groups), size=len(groups))
        for kind, values in contributions.items():
            old_hits, new_hits, count = values[sample].sum(0)
            require(count > 0, 'bootstrap finite-boundary support')
            differences[kind].append((new_hits-old_hits)/count)
    return dict(groups=len(groups), resamples=1000, direction='new-sampling-minus-old-fixed-query',
        purpose='Post-run descriptive uncertainty; not used to select models or alter gates',
        within5cm_difference_95pct={kind: np.percentile(values, [2.5, 97.5]).tolist() for kind, values in differences.items()})


def audit(root):
    baseline = root.with_name('ba-contact-boundary-20260923')
    result = read(root/'result.json')
    seal, prediction_seal = read(root/'source-seal.json'), read(root/'prediction-seal.json')
    for path, expected in seal['inputs'].items():
        require(digest(path) == expected, 'frozen input ' + path)
    for name, expected in seal['code'].items():
        require(digest(HERE/name) == expected, 'current code '+name)
        require(digest(root/'source-snapshot'/name) == expected, 'snapshot code '+name)
    for name, expected in prediction_seal['files'].items():
        require(digest(root/name) == expected, 'prediction seal '+name)
    require(prediction_seal['evaluation_targets_joined'] is False, 'procedural pre-join seal')
    old_source = read(baseline/'source-seal.json')
    for path, expected in old_source['input_hashes'].items():
        require(digest(path) == expected, 'inherited original input '+path)
    for name in ('contact_boundary_data.py', 'contact_boundary_model.py', 'run_contact_boundary.py'):
        require(seal['code'][name] == old_source['code'][name], 'original source preserved '+name)
    old_audit = read(baseline/'independent-audit.json')
    require(old_audit['status'] == 'PASS', 'prior independent control audit')
    for name, expected in old_audit['artifact_sha256'].items():
        require(digest(baseline/name) == expected, 'audited control unchanged '+name)
    require(seal['positive_weight'] == 50168/12040, 'fixed positive class weight')
    prepared = root.with_name('ba-query-occupancy-20260922-prepared')
    capture = root.with_name('ba-query-occupancy-20260922-capture')
    identities = read(prepared/'observations/identities.json')
    geometry = read(capture/'evaluator/geometry.json')
    require(len(identities) == len(geometry) == 1728, 'source identity count')
    indices = {split: np.array(rows, np.int64) for split, rows in read(baseline/'cohort.json')['indices'].items()}
    groups = {}
    for split, frames, count in zip(('train', 'dev', 'evaluation'), (864, 288, 576), (24, 8, 16)):
        require(indices[split].tolist() == [i for i, row in enumerate(identities) if row['split'] == split], 'split membership')
        groups[split] = {identities[i]['base_group_id'] for i in indices[split]}
        require(len(indices[split]) == frames and len(groups[split]) == count, 'whole-layout split counts')
    require(all(not groups[a] & groups[b] for a, b in itertools.combinations(groups, 2)), 'layout overlap')
    for i, (g, m) in enumerate(zip(geometry, identities)):
        require(g['id'] == m['id'] and g['sample_index'] == m['index'] == i, 'geometry identity')

    sampled = dict(np.load(root/'training-queries.npz', allow_pickle=False))
    q, y = sampled['queries'], sampled['labels']
    require(q.shape == (864, 72, 3) and y.shape == (864, 72), 'sampled query dimensions')
    require(np.array_equal(sampled['indices'], indices['train']), 'sampled train-only indices')
    require(q.dtype == np.float32 and np.isfinite(q).all(), 'query representation')
    require((q[..., 0] >= np.float32(.36)).all() and (q[..., 0] <= np.float32(1.08)).all(), 'sampled width bounds')
    require((q[..., 1] >= np.float32(.6)).all() and (q[..., 1] <= 3).all(), 'sampled horizon bounds')
    receipts = read(root/'sampling-receipts.json')
    require(len(receipts) == len(q), 'receipt count')
    independent_y = np.array([reference.world_labels(geometry[index], query) for index, query in zip(indices['train'], q)])
    require(np.array_equal(independent_y, y), 'all sampled labels versus independent world prisms')
    accepted_total = fallback_total = 0
    for row, (query, receipt, index) in enumerate(zip(q, receipts, indices['train'])):
        require(receipt['index'] == int(index) and receipt['seed'] == 202609232+int(index), 'sampling identity/seed')
        require(len(set(map(tuple, query))) == 72, '72 unique queries per image')
        require(np.count_nonzero(query[:, 2] == 0) == np.count_nonzero(query[:, 2] == 1) == 36, 'layer balance')
        require(np.all(query[:18, 2] == 0) and np.all(query[18:36, 2] == 1), 'global-query slots')
        require(len(receipt['pairs']) == 18, 'pair slot count')
        used, accepted = [], 0
        for pair in receipt['pairs']:
            a, b = pair['rows']; used.extend((a, b))
            require(b == a+1 and a >= 36 and b < 72, 'pair rows')
            require(pair['attempts'] in range(1, 9), 'bounded sampler attempts')
            require(np.all(query[[a, b], 2] == pair['layer']), 'pair layer')
            require(pair['offset_m'] in (.01, .02, .05), 'allowed offset')
            if pair['accepted']:
                accepted += 1
                axis = 0 if pair['kind'] == 'width' else 1
                require(pair['kind'] in ('width', 'horizon'), 'pair kind')
                require(not independent_y[row, a] and independent_y[row, b], 'true-changing boundary pair')
                require(np.allclose(query[[a, b], axis], np.array([pair['critical_value']-pair['offset_m'],
                    pair['critical_value']+pair['offset_m']]), atol=2e-7, rtol=0), 'boundary offsets')
                require(np.all(query[[a, b], 1-axis] == pair['fixed_value']), 'fixed slice')
                require(pair['fallback_reason'] is None, 'accepted pair receipt')
            else:
                require(pair['critical_value'] is None and pair['attempts'] == 8, 'fallback receipt')
        require(sorted(used) == list(range(36, 72)), 'pair slot partition')
        require(receipt['boundary_pairs'] == accepted and receipt['fallback_pairs'] == 18-accepted, 'pair aggregate')
        accepted_total += accepted; fallback_total += 18-accepted

    query_sets = reference.expected_queries()
    labels, boundaries = {}, {}
    checked = int(y.size)
    for split, filename in (('train', 'training-targets.npz'), ('dev', 'dev-selection-targets.npz'), ('evaluation', 'evaluation-targets.npz')):
        saved = dict(np.load(baseline/filename, allow_pickle=False))
        names = ['seen'] if split == 'dev' else list(query_sets)
        labels[split] = {name: np.array([reference.world_labels(geometry[i], query_sets[name]) for i in indices[split]]) for name in names}
        for name in names:
            require(np.array_equal(labels[split][name], saved[name]), 'fixed evaluation labels '+split+'/'+name)
            checked += labels[split][name].size
        if split != 'dev':
            raw = [reference.world_boundaries(geometry[i]) for i in indices[split]]
            boundaries[split] = {kind: np.array([r[kind] for r in raw]) for kind in ('width', 'horizon')}
            for kind in boundaries[split]:
                require(np.allclose(boundaries[split][kind], saved[kind+'_boundary'], rtol=0, atol=2e-14), 'world boundary '+kind)
    schedule = np.load(baseline/'batch-schedule.npy', allow_pickle=False)
    rng = np.random.default_rng(202609231)
    require(np.array_equal(schedule, np.array([rng.permutation(864) for _ in range(100)])), 'frozen schedule')
    old_result = read(baseline/'result.json')
    evaluation = {}
    for mode in ('direct', 'geometry'):
        selection = read(root/(mode+'-selection.json'))
        compare(selection, result['selection'][mode], 'selection '+mode)
        require(digest(root/(mode+'.pt')) == prediction_seal['checkpoint_hashes'][mode] == selection['checkpoint_sha256'], 'checkpoint seal')
        require(digest(root/(mode+'-selection.json')) == prediction_seal['selection_hashes'][mode], 'selection seal')
        require(selection['parameters'] == old_result['selection'][mode]['parameters'], 'unchanged parameter counts')
        require(selection['updates'] == 2700 and [h['epoch'] for h in selection['history']] == list(range(10, 101, 10)), 'fit budget')
        best = min(selection['history'], key=lambda item: (item['dev_BCE'], item['epoch']))
        require((selection['epoch'], selection['dev_BCE']) == (best['epoch'], best['dev_BCE']), 'dev checkpoint choice')
        predictions = {}
        for split in indices:
            predictions[split] = dict(np.load(root/(mode+'-'+split+'-predictions.npz'), allow_pickle=False))
            require(set(predictions[split]) == set(query_sets), 'query key preservation')
            for name, scores in predictions[split].items():
                require(scores.shape == (len(indices[split]), len(query_sets[name])) and np.isfinite(scores).all()
                        and (scores >= 0).all() and (scores <= 1).all(), 'prediction dimensions/probabilities')
        threshold = reference.atomic_threshold(predictions['dev']['seen'], labels['dev']['seen'])
        require(threshold == selection['threshold'], 'seen-dev-only atomic cutoff')
        compare(reference.metrics(predictions['dev']['seen'], labels['dev']['seen'], threshold), selection['dev'], 'dev metrics')
        require(selection['dev']['FPR'] <= .05, 'dev FPR budget')
        for split in ('train', 'evaluation'):
            metadata = [identities[i] for i in indices[split]]
            pred = predictions[split]
            current = reference.summary(pred, labels[split], boundaries[split], metadata, threshold)
            report = result['arms'][mode][split]
            compare(current, report['metrics'], mode+'/'+split+'/metrics')
            cutoff = old_result['selection'][mode]['threshold']
            compare(reference.summary(pred, labels[split], boundaries[split], metadata, cutoff), report['at_old_cutoff'], 'old cutoff diagnostic')
            old_pred = dict(np.load(baseline/(mode+'-'+split+'-predictions.npz'), allow_pickle=False))
            previous = reference.summary(old_pred, labels[split], boundaries[split], metadata, cutoff)
            compare(change_report(current, previous), report['change'], 'change/gates')
            boot = reference.bootstrap({'direct': old_pred, 'geometry': pred}, labels[split], metadata,
                {'direct': old_result['selection'][mode], 'geometry': selection})
            boot['direction'] = 'new-sampling-minus-old-fixed-query'
            compare(boot, report['paired_query_bootstrap'], 'paired bootstrap')
            if split == 'evaluation':
                evaluation[mode] = dict(metrics=current, change=report['change'],
                    descriptive_boundary_bootstrap=boundary_bootstrap(old_pred, pred, boundaries[split], metadata, cutoff, threshold))
    require(result['status'] == 'PASS' and result['epochs_per_arm'] == 100 and result['unique_queries_per_train_image'] == 72, 'run completion')
    return dict(status='PASS', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        backend='TASK_NOT_GPU_SUITABLE', source_and_prediction_seals_verified=True,
        old_control_artifacts_unchanged=True, independently_checked_world_space_queries=checked,
        training_query_checks=dict(images=864, queries=62208, positives=int(y.sum()),
            boundary_pairs=accepted_total, fallback_pairs=fallback_total, unique_per_image=72),
        selected_thresholds={mode: result['selection'][mode]['threshold'] for mode in ('direct', 'geometry')},
        evaluation=evaluation, audit_source_sha256=digest(__file__), independent_reference_sha256=digest(reference.__file__),
        artifacts_sha256={p.name:digest(p) for p in root.iterdir() if p.is_file()},
        limits=['No refitting or checkpoint inference; selected epochs checked against saved history.',
                'Geometry was process-accessible before fitting; sealing establishes procedural order only.',
                'Feature extraction and initialization are inherited, not recomputed; audited control bytes are verified.',
                'Boundary score denominator is finite interior boundaries, with censoring counted separately.',
                'Consumed same-generator Development, not fresh confirmation or complete occupancy identification.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root/'independent-audit.json'
    require(not output.exists(), 'Audit output already exists; never overwrite')
    report = audit(args.root)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(output),
                          checked_queries=report['independently_checked_world_space_queries'])))
