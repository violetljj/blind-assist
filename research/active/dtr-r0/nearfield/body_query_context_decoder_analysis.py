"""Saved-probability verification of frozen LOCAL/JOINT/PRIOR count decoders."""
import argparse
from pathlib import Path

import numpy as np
import torch

from body_query_data import read, sha, truth, write
from body_query_range_untied_analysis import score
from body_query_10000_readout_analysis import TRAINED_10K_SHA


def verify(cache, baseline, run):
    protocol, receipt, fits = [read(run / name) for name in ('protocol.json', 'receipt.json', 'fits.json')]
    assert receipt['status'] == 'PASS' and receipt['fits'] == 3 and receipt['steps_per_fit'] == 2000
    assert receipt['baseline_state_unchanged']
    assert protocol['baseline_sha256'] == sha(baseline / 'NEW-step2000.pt') == TRAINED_10K_SHA
    assert protocol['cache_sha256'] == sha(cache / 'manifest.json')
    assert protocol['arms'] == ['LOCAL', 'JOINT', 'PRIOR']
    assert protocol['seed'] == 17 and protocol['steps'] == 2000 and protocol['batch'] == 32
    assert protocol['lr'] == .001 and protocol['weight_decay'] == .0001
    assert protocol['source_sha256'] == sha(Path(__file__).with_name('body_query_context_decoder.py'))
    for name, digest in protocol['dependencies'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for name in ('result', 'selection', 'schedule'):
        suffix = '.npy' if name == 'schedule' else '.json'
        assert sha(run / (name + suffix)) == receipt[name + '_sha256']
    schedule = np.load(run / 'schedule.npy', allow_pickle=False)
    assert np.array_equal(schedule, np.random.default_rng(17).integers(0, 5000, (2000, 32)))
    assert np.array_equal(schedule, np.load(baseline / 'schedule.npy', allow_pickle=False))
    base_state = torch.load(baseline / 'NEW-step2000.pt', map_location='cpu', weights_only=True)
    xyz = (base_state['query_xyz'] * base_state['query_valid'][:, :, None]).sum(1)
    xyz = xyz / base_state['query_valid'].sum(1)[:, None].clamp_min(1)
    checkpoint_hashes = {}
    xyz_errors = {}
    saved_xyz = None
    for arm in protocol['arms']:
        assert fits[arm]['steps'] == 2000 and fits[arm]['sha256'] == sha(run / f'{arm}.pt')
        state = torch.load(run / f'{arm}.pt', map_location='cpu', weights_only=True)
        assert set(state) == {'xyz', 'net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias'}
        # GPU/CPU reduction order differs for symmetric XYZ values near zero.
        xyz_errors[arm] = float((state['xyz']-xyz).abs().max())
        assert torch.allclose(state['xyz'], xyz, rtol=0, atol=1e-7)
        if saved_xyz is None:
            saved_xyz = state['xyz']
        else:
            assert torch.equal(state['xyz'], saved_xyz)
        assert tuple(state['net.0.weight'].shape) == (128, 768 if arm == 'JOINT' else 67)
        assert tuple(state['net.2.weight'].shape) == (48 if arm == 'JOINT' else 4, 128)
        assert sum(value.numel() for key, value in state.items() if key != 'xyz') == fits[arm]['parameters']
        checkpoint_hashes[arm] = sha(run / f'{arm}.pt')
    feature_checks = {}
    for role in ('train', 'dev', 'eval'):
        meta = read(run / f'features-{role}.json')
        record, _ = truth(cache, role, training=role == 'train')
        assert meta['ids'] == record['sample_indices']
        assert meta['rgb_manifest_sha256'] == protocol['cache_sha256']
        assert 0 <= meta['parity'] < 2e-6
        assert meta['feature_sha256'] == sha(run / f'features-{role}.npy')
        raw = np.load(run / f'features-{role}.npy', allow_pickle=False)
        assert raw.shape == (meta['rows'], 12, 64) and len(raw) == len(meta['ids'])
        assert np.isfinite(raw).all()
        if role == 'train':
            with np.load(run / 'normalization.npz', allow_pickle=False) as norm:
                assert np.array_equal(norm['mean'], raw.mean((0, 1), keepdims=True))
                assert np.array_equal(norm['std'], raw.std((0, 1), keepdims=True).clip(.01))
        feature_checks[role] = meta
    return dict(checkpoint_sha256=checkpoint_hashes, feature_metadata=feature_checks, xyz_cpu_reduction_error=xyz_errors,
                normalization_sha256=sha(run / 'normalization.npz'),
                baseline_frozen_evidence='Source and receipt assertion; baseline checkpoint hash and extraction parity verified; no new inference')


def run_analysis(cache, baseline, run):
    invariant = verify(cache, baseline, run)
    arms, hashes = {}, {}
    trained = read(run / 'result.json')
    prior_reference = None
    prior_max_error = 0.
    alert_max_error = 0.
    for arm in ('BASE', 'LOCAL', 'JOINT', 'PRIOR'):
        arms[arm], hashes[arm] = score(cache, run, arm)
        for role, actual in arms[arm].items():
            expected = trained['arms'][arm][role]
            for head, values in actual['heads'].items():
                assert all(expected['heads'][head]['selected'][k] == v for k, v in values.items())
            for key in ('correct', 'total'):
                assert actual['complete_groups'][key] == expected['selected_groups'][key]
            for key, values in actual['query_cells'].items():
                assert expected['query_strata'][key] == values
                assert expected['range_events'][key] == actual['range_events'][key]
            assert actual['native_HEAD_near_only_wrong_far'] == expected['wrong_far_on_HEAD_near_only']['fired']
            assert actual['native_HEAD_near_only_frames'] == expected['wrong_far_on_HEAD_near_only']['total']
            record, target = truth(cache, role, training=role == 'train')
            with np.load(run / f'{arm}-{role}.npz', allow_pickle=False) as pred, np.load(baseline / f'NEW-{role}.npz', allow_pickle=False) as base:
                assert np.array_equal(pred['support'], base['support'])
                if arm == 'BASE':
                    assert all(np.array_equal(pred[key], base[key]) for key in pred.files)
                if arm == 'PRIOR':
                    if prior_reference is None:
                        prior_reference = pred['counts'][0].copy()
                    error = float(np.abs(pred['counts'] - prior_reference).max())
                    prior_max_error = max(prior_max_error, error)
                    assert error < 2e-6, 'PRIOR must be image-independent for each query'
                p = pred['counts'].astype(np.float64).reshape(-1, 2, 6, 4)
                below = np.zeros((len(p), 2, 3), dtype=np.float64)
                below[:, :, 0] = 1
                for cell in range(6):
                    nxt = np.zeros_like(below)
                    for total in range(3):
                        for count in range(total+1):
                            nxt[:, :, total] += below[:, :, total-count] * p[:, :, cell, count]
                    below = nxt
                error = float(np.abs((1-below.sum(-1)) - pred['near']).max())
                alert_max_error = max(alert_max_error, error)
                assert error < 2e-6
                decisions = pred['near'].astype(np.float64) >= np.asarray(actual['thresholds'])
                labels = target['near'].astype(bool)
                controls = {}
                conditions = np.array([row['condition'] for row in record['records']])
                for condition in sorted(set(conditions)):
                    ids = conditions == condition
                    controls[str(condition)] = dict(frames=int(ids.sum()),
                        false_positives=(decisions[ids] & ~labels[ids]).sum(0).tolist(),
                        false_negatives=(~decisions[ids] & labels[ids]).sum(0).tolist())
                actual['controls'] = controls
    prior = arms['PRIOR']['eval']
    gates = {}
    for arm in ('LOCAL', 'JOINT', 'PRIOR'):
        c = arms[arm]['eval']
        assert c['query_cells']['HEAD_near']['TP'] + c['query_cells']['HEAD_near']['FN'] == 1788
        assert c['native_HEAD_near_only_frames'] == 600
        q, pq = c['query_cells']['HEAD_near'], prior['query_cells']['HEAD_near']
        # PRIOR can abstain everywhere; its zero wrong-far is not a useful
        # superiority requirement. Compare balanced query discrimination instead.
        balanced = .5*(q['TP']/(q['TP']+q['FN']) + q['TN']/(q['TN']+q['FP']))
        prior_balanced = .5*(pq['TP']/(pq['TP']+pq['FN']) + pq['TN']/(pq['TN']+pq['FP']))
        values = dict(HEAD_near_at_least_894=c['query_cells']['HEAD_near']['TP'] >= 894,
            wrong_far_at_most_300=c['native_HEAD_near_only_wrong_far'] <= 300,
            improves_over_PRIOR=(q['TP'] > pq['TP'] and balanced > prior_balanced),
            groups_at_least_508=c['complete_groups']['correct'] >= 508,
            BODY_TP_at_least_1187=c['heads']['BODY']['TP'] >= 1187,
            BODY_FP_at_most_69=c['heads']['BODY']['FP'] <= 69,
            HEAD_TP_at_least_1150=c['heads']['HEAD']['TP'] >= 1150,
            HEAD_FP_at_most_35=c['heads']['HEAD']['FP'] <= 35,
            HEAD_ONLY_BODY_FP_below_38=c['HEAD_ONLY_BODY_FP'] < 38)
        gates[arm] = dict(values=values, HEAD_near_balanced_accuracy=balanced, PRIOR_HEAD_near_balanced_accuracy=prior_balanced,
                         strong_spatial_gain=all(list(values.values())[:3]), full_replacement=all(values.values()))
    summary = dict(schema='body-query-context-decoder-independent-analysis-v1', status='PASS',
        scope='Consumed controlled Development; saved probabilities, no fitting or inference',
        arms=arms, gates=gates, invariants=invariant, prediction_sha256=hashes,
        prior_image_independence_max_error=prior_max_error, alert_convolution_max_error=alert_max_error,
        prior_comparison_interpretation='Near TP and near-query balanced accuracy strictly greater; descriptive operationalization because protocol did not specify a numerical PRIOR comparison metric',
        analyzer_sha256=sha(Path(__file__)))
    write(run / 'context-decoder-analysis.json', summary)
    write(run / 'validation.json', dict(status='PASS', fits=0, optimizer_steps=0, no_inference=True,
          analysis_sha256=sha(run / 'context-decoder-analysis.json'),
          checks=['Source, checkpoints, schedule and frozen-feature metadata', 'TRAIN-only normalization',
                  'Independent DEV thresholds, query/range/confusion/group metrics and gates',
                  'Image-independent PRIOR probabilities across all roles', 'Six-cell alert convolution and unchanged support'],
          limits='Consumed Development; frozen runtime assertion cannot be independently replayed without inference'))
    print(gates, flush=True)
    for arm in arms:
        c = arms[arm]['eval']
        print(arm, {key: c[key] for key in ('query_cells', 'heads', 'native_HEAD_near_only_wrong_far', 'HEAD_ONLY_BODY_FP')}, 'groups', c['complete_groups']['correct'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'baseline', 'run'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    run_analysis(args.cache, args.baseline, args.run)
