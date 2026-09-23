"""Independent governed saved-output audit; no dense-supervision producer imports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from audit_contact_boundary import (read, digest, compare, expected_queries,
    world_labels, world_boundaries, summary, atomic_threshold, metrics)

HERE = Path(__file__).resolve().parent
ART = HERE.parents[3] / 'artifacts.local/evidence'
OUT = ART / 'ba-boundary-supervision-20260923-v2'
OLD = ART / 'ba-contact-boundary-20260923'
PREPARED = ART / 'ba-query-occupancy-20260922-prepared'
CAPTURE = ART / 'ba-query-occupancy-20260922-capture'


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def independent_schedule():
    query = expected_queries()
    key = lambda row: tuple(round(float(v), 6) for v in row)
    anchor = {key(row) for row in query['seen'][:36]}
    rng = np.random.default_rng(202609278)
    banks = []
    for field in ('seen', 'width_curve', 'horizon_curve'):
        candidates = [(field, j) for j in range(len(query[field]) // 2)
                      if field == 'seen' or key(query[field][j]) not in anchor]
        order = rng.permutation(np.arange(len(candidates)))
        banks.append([candidates[int(j)] for j in order])
    assert list(map(len, banks)) == [36, 47, 46]
    mapping = []
    for epoch in range(100):
        batch = []
        for band in (0, 1):
            for bank in banks:
                for slot in range(12):
                    field, index = bank[(epoch * 12 + slot) % len(bank)]
                    batch.append([field, index + band * (len(query[field]) // 2)])
        mapping.append(batch)
    values = np.array([[query[field][index] for field, index in epoch]
                       for epoch in mapping], np.float32)
    assert values.shape == (100, 72, 3)
    assert all(len({key(row) for row in epoch}) == 72 for epoch in values)
    unique, inverse = np.unique(values.reshape(-1, 3), axis=0, return_inverse=True)
    assert len(unique) == 258
    for band in (0, 1):
        for offset, size in zip((0, 12, 24), (36, 47, 46)):
            counts = Counter(key(row) for epoch in values
                             for row in epoch[36 * band + offset:36 * band + offset + 12])
            assert len(counts) == size and max(counts.values()) - min(counts.values()) <= 1
    assert not {key(row) for row in unique} & {key(row) for row in query['both']}
    return values, mapping, unique, inverse.reshape(100, 72)


def verdict(old, new):
    gains = {split: {kind: new[split]['boundaries'][kind]['joint_within_5cm'] -
                    old[split]['boundaries'][kind]['joint_within_5cm']
                    for kind in ('width', 'horizon')} for split in ('train', 'evaluation')}
    fitted = min(gains['train'].values()) >= .1 - 1e-12
    transferred = min(gains['evaluation'].values()) >= .1 - 1e-12
    before, after = old['evaluation'], new['evaluation']
    costs = dict(recall=after['both']['recall'] - before['both']['recall'] >= -.03 - 1e-12,
                 FPR=after['both']['FPR'] - before['both']['FPR'] <= .01 + 1e-12,
                 false_crossings=max(after['boundaries'][k]['wrong_crossings_on_right_censored'] -
                     before['boundaries'][k]['wrong_crossings_on_right_censored']
                     for k in ('width', 'horizon')) <= 5)
    supported = fitted and transferred and all(costs.values())
    status = ('MECHANISM_SUPPORTED' if supported else 'ACCURACY_COST_TRADEOFF'
              if fitted and transferred else 'FITTING_ONLY' if fitted else 'DENSE_SUPERVISION_INSUFFICIENT')
    return dict(decision=status, train_gain_pass=fitted, transfer_gain_pass=transferred,
                costs=costs, boundary_accuracy_gain=gains, mechanism_supported=supported,
                full_component_pass=after['component_gate'])


def audit():
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue execution'
    started = time.perf_counter(); checks = Counter()
    result = read(OUT / 'result.json'); assert result['status'] == 'PASS'
    failed_launch = ART / 'ba-boundary-supervision-20260923'
    assert failed_launch.exists() and not (failed_launch / 'fit-started.json').exists(), 'Prior launch crossed fit boundary'
    seal = read(OUT / 'input-seal.json'); hashes = dict(seal['files'])
    hashes.update({str(HERE / name): value for name, value in seal['code'].items()})
    old_source = read(OLD / 'source-seal.json')
    for name in ('contact_boundary_model.py', 'contact_boundary_data.py', 'run_contact_boundary.py'):
        assert seal['code'][name] == old_source['code'][name], name
    hashes.update(old_source['input_hashes'])
    previous_predictions = read(OLD / 'prediction-seal.json')
    hashes.update({str(OLD / name): value for name, value in previous_predictions['files'].items()})
    hashes.update({str(OLD / (arm + '-selection.json')): value
                   for arm, value in previous_predictions['selection_hashes'].items()})
    hashes.update({str(OLD / (arm + '.pt')): value
                   for arm, value in previous_predictions['checkpoint_hashes'].items()})
    prediction_seal = read(OUT / 'prediction-seal.json')
    assert prediction_seal['all_predictions_before_evaluation_target_access'] is True
    assert prediction_seal['evaluation_targets_loaded'] is False
    hashes.update({str(OUT / name): value for name, value in prediction_seal['files'].items()})
    hashes[str(OUT / 'dense.pt')] = prediction_seal['checkpoint']
    hashes[str(OUT / 'selection.json')] = prediction_seal['selection']
    hashes[str(OUT / 'dev-checkpoint-logits.npz')] = prediction_seal['dev_checkpoint_logits']
    for path in OUT.iterdir():
        if path.is_file() and str(path) not in hashes:
            hashes[str(path)] = digest(path)
    hashes[str(Path(__file__))] = digest(__file__)
    hashes[str(HERE / 'audit_contact_boundary.py')] = digest(HERE / 'audit_contact_boundary.py')
    support_path = ART / 'ba-boundary-error-20260923-v3/rows.json'
    hashes[str(support_path)] = digest(support_path)
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    query = expected_queries(); qepoch, mapping, unique, expansion = independent_schedule()
    np.testing.assert_array_equal(np.load(OUT / 'query-schedule.npy', allow_pickle=False), qepoch)
    assert read(OUT / 'query-mapping.json') == mapping
    assert read(OUT / 'cohort.json') == read(OLD / 'cohort.json')
    cohort = read(OUT / 'cohort.json')['indices']
    meta = read(PREPARED / 'observations/identities.json')
    geometry = read(CAPTURE / 'evaluator/geometry.json')
    assert len(meta) == len(geometry) == 1728
    assert sorted(i for indices in cohort.values() for i in indices) == list(range(1728))
    assert [len(cohort[s]) for s in ('train', 'dev', 'evaluation')] == [864, 288, 576]
    groups = [{meta[i]['base_group_id'] for i in cohort[s]} for s in ('train', 'dev', 'evaluation')]
    assert list(map(len, groups)) == [24, 8, 16]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all(m['index'] == g['sample_index'] == i and m['id'] == g['id']
               for i, (m, g) in enumerate(zip(meta, geometry)))
    training = np.load(OUT / 'training-targets.npy', mmap_mode='r')
    assert training.shape == (100, 864, 72) and training.dtype == np.bool_
    independent_unique = np.array([world_labels(geometry[i], unique) for i in cohort['train']])
    for epoch in range(100):
        np.testing.assert_array_equal(training[epoch], independent_unique[:, expansion[epoch]])
    assert not independent_unique[:, np.isclose(unique[:, 1], .3)].any()
    checks['independent_unique_training_labels'] = int(independent_unique.size)
    checks['expanded_training_labels'] = int(training.size)
    assert checks['expanded_training_labels'] == 6220800
    permutations = np.load(OLD / 'batch-schedule.npy', allow_pickle=False)
    rng = np.random.default_rng(202609231)
    np.testing.assert_array_equal(permutations, [rng.permutation(864) for _ in range(100)])
    features = np.load(OLD / 'features.npy', mmap_mode='r'); normal = arrays(OLD / 'normalization.npz')
    assert digest(OLD / 'features.npy') == read(OLD / 'feature-receipt.json')['features_sha256']
    np.testing.assert_array_equal(normal['mean'], features[cohort['train']].mean(0))
    np.testing.assert_array_equal(normal['std'], np.maximum(features[cohort['train']].std(0), .01))
    labels, boundaries = {}, {}
    for split in ('train', 'dev', 'evaluation'):
        use = query if split != 'dev' else {'seen': query['seen']}
        labels[split] = {name: np.array([world_labels(geometry[i], queries) for i in cohort[split]])
                         for name, queries in use.items()}
        bounds = [world_boundaries(geometry[i]) for i in cohort[split]]
        boundaries[split] = {kind: np.array([b[kind] for b in bounds]) for kind in ('width', 'horizon')}
        checks['original_world_labels'] += sum(values.size for values in labels[split].values())
    selection = read(OUT / 'selection.json'); compare(result['selection'], selection, 'selection copy')
    positive = int(labels['train']['seen'].sum()); negative = int(labels['train']['seen'].size - positive)
    assert selection['old_positive_weight'] == negative / positive
    feasibility = read(OUT / 'feasibility.json')
    assert feasibility['initialization_exact'] and feasibility['batch_schedule_exact']
    assert feasibility['evaluation_targets_loaded'] is False and feasibility['schedule_geometry_independent']
    assert feasibility['unique_queries_per_image'] == 258 and feasibility['total_training_exposures'] == 6220800
    assert np.isclose(feasibility['new_positive_fraction'], training.mean(), atol=1e-12, rtol=0)
    assert feasibility['old_positive_weight'] == negative / positive
    fit_start = read(OUT / 'fit-started.json')
    assert fit_start['seed'] == 202609231 and fit_start['epochs'] == 100 and fit_start['updates'] == 2700
    assert selection['updates'] == 2700 and selection['checkpoint_sha256'] == prediction_seal['checkpoint']
    history = selection['history']; assert [h['epoch'] for h in history] == list(range(10, 101, 10))
    assert all(np.isfinite(h['dev_BCE']) and np.isfinite(h['train_balanced_BCE']) for h in history)
    best = min(history, key=lambda h: (h['dev_BCE'], h['epoch']))
    assert selection['epoch'] == best['epoch'] and selection['dev_BCE'] == best['dev_BCE']
    dev_checkpoints = arrays(OUT / 'dev-checkpoint-logits.npz')
    np.testing.assert_array_equal(dev_checkpoints['epochs'], np.arange(10, 101, 10))
    logits = dev_checkpoints['logits']
    assert logits.shape == (10, 288, 72) and logits.dtype == np.float32 and np.isfinite(logits).all()
    checkpoint_losses = []
    for index, item in enumerate(history):
        x = logits[index].astype(np.float64); y = labels['dev']['seen']
        loss = float(np.mean(np.maximum(x, 0) - x * y + np.logaddexp(0., -np.abs(x))))
        assert np.isclose(loss, item['dev_BCE'], atol=5e-7, rtol=1e-6), ('checkpoint BCE', item['epoch'])
        checkpoint_losses.append(loss); checks['dev_checkpoint_BCE'] += 1
    predicted = {s: arrays(OUT / (s + '-predictions.npz')) for s in cohort}
    for split, values in predicted.items():
        assert set(values) == set(query)
        assert all(v.shape == (len(cohort[split]), len(query[k])) and np.isfinite(v).all()
                   and ((v >= 0) & (v <= 1)).all() for k, v in values.items())
    threshold = atomic_threshold(predicted['dev']['seen'], labels['dev']['seen'])
    selected_logits = logits[selection['epoch'] // 10 - 1].astype(np.float64)
    selected_probabilities = np.exp(-np.logaddexp(0., -selected_logits))
    np.testing.assert_allclose(predicted['dev']['seen'], selected_probabilities, atol=2e-7, rtol=2e-6)
    assert threshold == selection['threshold']
    compare(metrics(predicted['dev']['seen'], labels['dev']['seen'], threshold), selection['dev'], 'dev selection')
    assert selection['dev']['FPR'] <= .05
    new, old, old_predictions = {}, {}, {}; previous = read(OLD / 'result.json')['arms']['geometry']
    old_threshold = read(OLD / 'geometry-selection.json')['threshold']
    for split in ('train', 'evaluation'):
        identities = [meta[i] for i in cohort[split]]
        new[split] = summary(predicted[split], labels[split], boundaries[split], identities, threshold)
        old_predictions[split] = arrays(OLD / f'geometry-{split}-predictions.npz')
        old[split] = summary(old_predictions[split], labels[split],
                             boundaries[split], identities, old_threshold)
        reference = previous['training_layouts'] if split == 'train' else {
            k: v for k, v in previous.items() if k != 'training_layouts'}
        compare(old[split], reference, 'old/' + split)
        compare(old[split], result['old'][split], 'old copy/' + split)
        compare(new[split], result['new'][split], 'new/' + split)
        checks['full_arm_split_summaries'] += 2
    decision = verdict(old, new)
    for key, value in decision.items():
        compare(result[key], value, 'decision/' + key)
    support_rows = {(r['index'], r['kind'], r['layer']): r for r in read(support_path)
                    if r['arm'] == 'geometry' and r['split'] == 'evaluation'}
    assert len(support_rows) == 576 * 4
    strata, paired = {}, {}
    for kind in ('width', 'horizon'):
        axis = np.linspace(.2, 1.2, 51) if kind == 'width' else np.linspace(.3, 3., 55)
        first, successes = [], []
        truth = boundaries['evaluation'][kind]
        interior = (truth > axis[0] + 1e-6) & (truth <= axis[-1] + 1e-6)
        for scores, cutoff in ((old_predictions['evaluation'], old_threshold), (predicted['evaluation'], threshold)):
            on = scores[kind + '_curve'].reshape(576, 2, len(axis)).astype(float) >= cutoff
            crossing = np.where(on.any(-1), axis[on.argmax(-1)], np.inf)
            first.append(crossing)
            # Subtract only interior truths, avoiding inf-inf for censored rows.
            correct = np.zeros_like(interior)
            correct[interior] = abs(crossing[interior] - truth[interior]) <= .050001
            successes.append(correct)
        categories = {}; pair = Counter()
        for local, index in enumerate(cohort['evaluation']):
            for layer in (0, 1):
                row = support_rows[index, kind, layer]
                stored_truth = np.inf if row['truth'] is None else row['truth']
                stored_prediction = np.inf if row['predicted'] is None else row['predicted']
                assert np.isclose(stored_truth, truth[local, layer], atol=1e-9, rtol=1e-9)
                assert np.isclose(stored_prediction, first[0][local, layer], atol=1e-9, rtol=1e-9)
                a, b = (bool(success[local, layer]) for success in successes)
                assert row['within_5cm'] == a
                if not interior[local, layer]: continue
                category = row['support_class']
                entry = categories.setdefault(category, dict(total=0, old_within_5cm=0, new_within_5cm=0))
                entry['total'] += 1; entry['old_within_5cm'] += int(a); entry['new_within_5cm'] += int(b)
                pair['both' if a and b else 'old_only' if a else 'new_only' if b else 'neither'] += 1
        strata[kind] = categories
        paired[kind] = {name: int(pair[name]) for name in ('both', 'old_only', 'new_only', 'neither')}
        assert sum(pair.values()) == int(interior.sum())
        assert sum(v['old_within_5cm'] for v in categories.values()) == old['evaluation']['boundaries'][kind]['within_5cm']
        assert sum(v['new_within_5cm'] for v in categories.values()) == new['evaluation']['boundaries'][kind]['within_5cm']
    checks['old_support_rows_verified'] = len(support_rows)
    compare(result['budget'], dict(new_fits=1, epochs=100, updates=2700, query_exposures=6220800), 'budget')
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    return dict(status='PASS', decision=decision, checks=dict(checks), elapsed_s=time.perf_counter() - started,
                input_code_prediction_hashes_unchanged=True, schedule_geometry_independent=True,
                independently_recomputed_checkpoint_BCE=checkpoint_losses,
                descriptive_native_support_strata=strata, paired_boundary_successes=paired,
                backend='TASK_NOT_GPU_SUITABLE', audit_sha256=digest(__file__),
                limits=['Saved outputs and source establish procedural order, not isolation.',
                        'No fit or checkpoint inference rerun; initialization use is source and receipt verified.',
                        'All ten dev losses replayed from logits; training trajectory itself is not rerun.',
                        'Consumed layout-disjoint simulation; no feature-information or hardware claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--result', required=True)
    destination = Path(parser.parse_args().result)
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue execution'
    assert not destination.exists(), 'Preserve previous audit receipts'
    try:
        report = audit()
    except Exception as exc:
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(dict(status='FAIL', error=repr(exc)), stream, indent=2)
        raise
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))
