"""Independent saved-probability analysis of the range-untied count readout.

CPU metadata/tensor checks only; no model forward, optimizer or training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from body_query_data import read, sha, truth, write
from body_query_10000_readout_analysis import (
    TRAINED_10K_SHA, confusion, groups, native_range_events, range_event_from_counts,
)
from city_dev_selection import select_threshold


def verify(cache, run):
    protocol, preflight, fit, receipt = [read(run / name) for name in
        ('protocol.json', 'preflight.json', 'fit-complete.json', 'receipt.json')]
    baseline = cache.parent / 'run-v1'
    assert receipt['status'] == preflight['status'] == 'PASS'
    assert protocol['initial_sha256'] == protocol['old_sha256'] == TRAINED_10K_SHA
    assert preflight['initial_sha256'] == preflight['baseline_sha256'] == TRAINED_10K_SHA
    assert sha(baseline / 'NEW-step2000.pt') == TRAINED_10K_SHA
    assert protocol['steps'] == fit['steps'] == receipt['steps'] == 2000
    assert protocol['seed'] == 17 and protocol['batch'] == 32 and receipt['fits'] == 1
    assert protocol['lr'] == 1e-5 and protocol['weight_decay'] == 1e-4
    assert preflight['schedule_identical'] and 0 <= preflight['train_prediction_parity'] < 2e-6
    assert set(preflight['trainable']) == {'query_readout.weight', 'query_readout.bias'}
    assert fit['frozen_parameters_verified'] and fit['buffers_verified']
    schedule = np.load(run / 'schedule.npy', allow_pickle=False)
    assert np.array_equal(schedule, np.random.default_rng(17).integers(0, 5000, (2000, 32)))
    assert np.array_equal(schedule, np.load(baseline / 'schedule.npy', allow_pickle=False))
    assert sha(run / 'schedule.npy') == protocol['schedule_sha256']
    assert sha(cache / 'manifest.json') == protocol['cache_sha256']
    assert sha(run / 'NEW-step2000.pt') == fit['checkpoint_sha256']
    assert sha(run / 'result.json') == receipt['result_sha256']
    assert sha(run / 'selection.json') == receipt['selection_sha256']
    assert sha(Path(__file__).with_name('body_query_10000_range_readout_train.py')) == protocol['source_sha256']
    for name, digest in protocol['dependency_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    old = torch.load(baseline / 'NEW-step2000.pt', map_location='cpu', weights_only=True)
    new = torch.load(run / 'NEW-step2000.pt', map_location='cpu', weights_only=True)
    assert old.keys() == new.keys()
    changed = [name for name in old if not torch.equal(old[name], new[name])]
    assert set(changed) == {'query_readout.weight', 'query_readout.bias'}
    assert tuple(new['query_readout.weight'].shape) == (2, 4, 32)
    assert tuple(new['query_readout.bias'].shape) == (2, 4)
    return dict(initial_sha256=TRAINED_10K_SHA, checkpoint_sha256=fit['checkpoint_sha256'],
                train_prediction_parity=preflight['train_prediction_parity'], changed_tensors=changed,
                frozen_tensor_count=len(old)-2, schedule_sha256=sha(run / 'schedule.npy'))


def score(cache, run, arm):
    selection = read(run / 'selection.json')[arm]
    result = {}
    payload_hashes = {}
    for role in ('train', 'dev', 'eval'):
        record, target = truth(cache, role, training=(role == 'train'))
        labels = target['near'].astype(bool)
        payload = run / f'{arm}-{role}.npz'
        with np.load(payload, allow_pickle=False) as saved:
            count, near = saved['counts'].astype(np.float64), saved['near'].astype(np.float64)
        assert count.shape == (len(labels), 12, 4) and near.shape == labels.shape
        assert np.isfinite(count).all() and np.isfinite(near).all()
        assert ((count >= 0) & (count <= 1)).all() and ((near >= 0) & (near <= 1)).all()
        norm_error = float(np.abs(count.sum(-1)-1).max())
        assert norm_error < 2e-6
        # Preserve stored probabilities: validate normalization rather than alter scores.
        event = range_event_from_counts(count)
        native = native_range_events(target['counts'])
        near_only = native[:, 1, 0] & ~native[:, 1, 1]
        thresholds = np.asarray(selection['thresholds'], dtype=np.float64)
        decisions = near >= thresholds
        if role == 'dev':
            selected = [select_threshold(near[:, h], labels[:, h], min_count=48) for h in range(2)]
            assert np.array_equal([row['threshold'] for row in selected], thresholds)
        query = {}
        events = {}
        for h, head in enumerate(('BODY', 'HEAD')):
            for d, distance in enumerate(('near', 'far')):
                cells = slice(h*6+d*3, h*6+d*3+3)
                key = f'{head}_{distance}'
                query[key] = confusion(1-count[:, cells, 0], target['counts'][:, cells] > 0, .5)
                events[key] = confusion(event[:, h, d], native[:, h, d], .5)
        head_only = np.array([row['condition'] == 'HEAD_ONLY' for row in record['records']])
        assert int(head_only.sum()) == {'train': 1000, 'dev': 400, 'eval': 600}[role]
        result[role] = dict(
            thresholds=thresholds.tolist(), count_normalization_max_error=norm_error,
            heads={head: confusion(near[:, h], labels[:, h], thresholds[h])
                   for h, head in enumerate(('BODY', 'HEAD'))},
            complete_groups=groups(decisions, labels, record['records']),
            query_cells=query, range_events=events,
            native_HEAD_near_only_frames=int(near_only.sum()),
            native_HEAD_near_only_wrong_far=int((event[near_only, 1, 1] >= .5).sum()),
            HEAD_ONLY_BODY_FP=int((decisions[:, 0] & head_only & ~labels[:, 0]).sum()))
        payload_hashes[role] = sha(payload)
    return result, payload_hashes


def run_analysis(cache, run, historical):
    invariant = verify(cache, run)
    historical_protocol = read(historical / 'protocol.json')
    assert historical_protocol['initial_sha256'] == TRAINED_10K_SHA
    assert historical_protocol['cache_sha256'] == sha(cache / 'manifest.json')
    assert np.array_equal(np.load(historical / 'schedule.npy', allow_pickle=False),
                          np.load(run / 'schedule.npy', allow_pickle=False))
    arms, hashes = {}, {}
    for name, source, arm in [('ORIGINAL_B', run, 'OLD'), ('RANGE_UNTIED', run, 'NEW'),
                              ('SHARED_COUNTONLY', historical, 'NEW')]:
        arms[name], hashes[name] = score(cache, source, arm)
    # Agreement check follows independent recomputation, never supplies the metrics.
    trained = read(run / 'result.json')
    for name, arm in [('ORIGINAL_B', 'OLD'), ('RANGE_UNTIED', 'NEW')]:
        for role, actual in arms[name].items():
            expected = trained['arms'][arm][role]
            for head, values in actual['heads'].items():
                assert all(expected['heads'][head]['selected'][k] == v for k, v in values.items())
            for key in ('correct', 'total'):
                assert actual['complete_groups'][key] == expected['selected_groups'][key]
            for key, values in actual['query_cells'].items():
                assert all(expected['query_strata'][key][k] == v for k, v in values.items())
    candidate = arms['RANGE_UNTIED']['eval']
    assert candidate['query_cells']['HEAD_near']['TP'] + candidate['query_cells']['HEAD_near']['FN'] == 1788
    assert candidate['native_HEAD_near_only_frames'] == 600
    gates = dict(HEAD_near_TP_at_least_294=candidate['query_cells']['HEAD_near']['TP'] >= 294,
                 wrong_far_below_564=candidate['native_HEAD_near_only_wrong_far'] < 564,
                 HEAD_ONLY_BODY_FP_below_38=candidate['HEAD_ONLY_BODY_FP'] < 38,
                 groups_at_least_508=candidate['complete_groups']['correct'] >= 508,
                 BODY_TP_at_least_1187=candidate['heads']['BODY']['TP'] >= 1187,
                 BODY_FP_at_most_69=candidate['heads']['BODY']['FP'] <= 69,
                 HEAD_TP_at_least_1150=candidate['heads']['HEAD']['TP'] >= 1150,
                 HEAD_FP_at_most_35=candidate['heads']['HEAD']['FP'] <= 35)
    summary = dict(schema='body-query-range-untied-analysis-v1', status='PASS',
                   scope='Consumed controlled shared-asset Development; saved probabilities only',
                   arms=arms, gates=dict(values=gates, all_pass=all(gates.values())),
                   invariants=invariant, prediction_sha256=hashes,
                   analyzer_sha256=sha(Path(__file__)), historical_run=str(historical))
    write(run / 'range-untied-analysis.json', summary)
    write(run / 'validation.json', dict(status='PASS', fits=0, optimizer_steps=0,
          no_training_or_inference=True, all_acceptance_gates_pass=all(gates.values()),
          checks=['Initial SHA and exact schedule', 'All non-readout tensors exact; range-readout shapes',
                  'Finite normalized probabilities', 'Independent DEV selection, confusion, groups and all query strata',
                  'Native three-cell range convolution', 'Historical shared-countonly saved-probability comparison'],
          analysis_sha256=sha(run / 'range-untied-analysis.json')))
    print(summary['gates'], flush=True)
    print({name: {k: arm['eval'][k] for k in ('query_cells', 'heads', 'native_HEAD_near_only_wrong_far', 'HEAD_ONLY_BODY_FP')}
           for name, arm in arms.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    args = parser.parse_args()
    run_analysis(args.cache, args.run, args.historical)
