"""CPU scalar audit of the frozen MZ11 run; no fitting or cutoff selection."""
import argparse
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import load_npz, read, write, sha


def scalar_metrics(logits, truth):
    out = dict(exact=0, tp=[0]*4, fp=[0]*4, positives=[0]*4)
    for row, target in zip(logits, truth):
        exact = True
        for q in range(4):
            positive, actual = bool(row[q] >= 0), bool(target[q])
            out['tp'][q] += int(positive and actual)
            out['fp'][q] += int(positive and not actual)
            out['positives'][q] += int(actual)
            exact &= positive == actual
        out['exact'] += int(exact)
    return out


def main(root, run):
    receipt, result, start = [read(run/name) for name in ['receipt.json', 'result.json', 'start-receipt.json']]
    for name, digest in receipt['outputs'].items():
        assert sha(run/name) == digest, name
    assert sha(Path(__file__).with_name('mz11_selective_addition.py')) == start['source_sha256']
    assert sha(Path(__file__).with_name('MZ11_SELECTIVE_ADDITION_PROTOCOL_20260910.md')) == start['protocol_sha256']
    assert start['seed'] == 111 and start['steps'] == 600 and start['parameters'] == 20
    a = load_npz(run/'predictions.npz')
    cutoff = np.load(run/'threshold.npy', allow_pickle=False)
    weights = torch.load(run/'gate.pt', map_location='cpu', weights_only=True)
    weight, bias = weights['weight'].numpy(), weights['bias'].numpy()
    assert weight.shape == (4, 4) and bias.shape == (4,)
    p9 = load_npz(root/'artifacts.local/work/mz9-source-supervision-20260910/run-v1/predictions.npz')
    p10 = load_npz(root/'artifacts.local/work/mz10-availability-20260910/run-v2/predictions.npz')
    cohorts = ['TRAIN', 'DEV', 'clean', 'stress', 'clean_wrong', 'stress_wrong']
    rejected, summary, score_error = {}, {}, 0.
    for cohort in cohorts:
        d = {key.split('/', 1)[1]: value for key, value in a.items() if key.startswith(cohort+'/')}
        added_tp, added_fp, rejected_tp = [0]*4, [0]*4, [0]*4
        for i in range(len(d['truth'])):
            for q in range(4):
                eligible = d['baseline'][i, q] < 0 and d['source'][i, q] >= 0 and d['available'][i, q]
                assert bool(d['eligible'][i, q]) == eligible
                accepted = eligible and d['score'][i, q] >= cutoff[q]
                assert bool(d['accepted'][i, q]) == accepted
                expected = d['source'][i, q] if accepted else d['baseline'][i, q]
                assert d['candidate'][i, q] == expected
                if d['baseline'][i, q] >= 0:
                    assert d['candidate'][i, q] == d['baseline'][i, q]
                actual = bool(d['truth'][i, q])
                added_tp[q] += int(accepted and actual)
                added_fp[q] += int(accepted and not actual)
                rejected_tp[q] += int(eligible and actual and not accepted)
                features = np.array([d['source'][i,q]/4 if d['available'][i,q] else 0., np.log1p(d['count'][i,q])/6,
                                     d['zones'][i,q]/8, d['returns'][i,q]/8], dtype=np.float32)
                np.testing.assert_array_equal(d['features'][i,q], features)
                replay = float(sum(float(features[j])*float(weight[q,j]) for j in range(4))+float(bias[q]))
                score_error = max(score_error, abs(replay-float(d['score'][i,q])))
        observed = dict(candidate=scalar_metrics(d['candidate'], d['truth']),
                        baseline=scalar_metrics(d['baseline'], d['truth']),
                        added_tp=added_tp, added_fp=added_fp, baseline_positive_lost=0)
        assert observed == result['metrics'][cohort], cohort
        rejected[cohort] = rejected_tp
        summary[cohort] = observed
        if cohort == 'DEV':
            for key in ['baseline', 'source', 'available']:
                np.testing.assert_array_equal(d[key], p10['DEV/'+key])
        elif cohort not in ['TRAIN', 'DEV']:
            condition = cohort.split('_')[0]
            source_key = 'SOURCE_RGB/'+condition+('/wrong' if cohort.endswith('wrong') else '')
            np.testing.assert_array_equal(d['source'], p9[source_key])
            for clip in result['clips'][cohort]:
                mask = a['clip'] == clip
                assert scalar_metrics(d['candidate'][mask], d['truth'][mask]) == result['clips'][cohort][clip]
        if cohort.endswith('wrong'):
            clean = cohort.split('_')[0]
            for key in ['baseline', 'available', 'count', 'zones', 'returns', 'truth']:
                np.testing.assert_array_equal(d[key], a[clean+'/'+key])
    # GPU reduction can differ slightly from scalar64 arithmetic; the saved
    # deployed float32 scores, not approximate CPU scores, define threshold ties.
    assert score_error < 1e-4, score_error
    expected_cutoffs = []
    for q in range(4):
        candidates = [a['DEV/score'][i,q] for i in range(1000) if a['DEV/eligible'][i,q] and not a['DEV/truth'][i,q]]
        expected = np.nextafter(max(candidates), np.float32(np.inf)) if candidates else (-np.inf if a['DEV/eligible'][:,q].any() else np.inf)
        assert cutoff[q] == expected
        expected_cutoffs.append(float(expected))
    thin = {c: sum(result['clips'][c]['thin_pole_background_target']['tp'][1::2]) for c in cohorts[2:]}
    gate = dict(no_fp_increase=all(not any(summary[c]['added_fp']) for c in ['DEV','clean','stress']),
                all_baseline_positives_preserved=True, dev_far_gain=sum(summary['DEV']['added_tp'][1::2]) > 0,
                thin_clean_retained=thin['clean'] >= 48, thin_stress_retained=thin['stress'] >= 48)
    assert result['gate'] == gate
    assert result['confirmation_candidate'] == all(gate.values())
    assert result['wrong_correspondence_reduces_thin'] == (thin['clean_wrong'] < thin['clean'])
    for c in ['TRAIN', 'clean']:
        for key, actual in [('positive', True), ('negative', False)]:
            count = (a[c+'/eligible'] & (a[c+'/truth'] == actual)).sum(0).tolist()
            assert count == start['training_group_counts'][c][key]
    write(run/'audit.json', dict(status='PASS', backend='CPU_SCALAR_NO_FIT',
        scalar_output_bits=sum(len(a[c+'/truth'])*4 for c in cohorts),
        cutoff_exact=expected_cutoffs, max_scalar_gate_score_error=score_error,
        eligible_positive_rejected=rejected, thin_far_tp=thin, gate=gate,
        confirmation_candidate=result['confirmation_candidate'],
        scope='Stored-output composition, scalar metrics, exact cutoffs, frozen source control parity; no new backbone replay',
        source_sha256=sha(Path(__file__)), run_receipt_sha256=sha(run/'receipt.json')))
    print('PASS', dict(rejected=rejected, thin=thin, gate=gate, scalar_score_error=score_error), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.run)
