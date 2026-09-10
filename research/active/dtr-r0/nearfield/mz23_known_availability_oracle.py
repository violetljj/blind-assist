"""Evaluator-only known-availability veto from saved MZ20 local scores.

No inference, fitting or cutoff choice. ALL stored eligible known-valid candidate
scores enter the maximum, regardless of their actual-query contributor label.
"""
import argparse
from pathlib import Path

import numpy as np

from mz5_ensemble_readout import read, sha, write
from mz11_audit import scalar_metrics


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    run = root / 'artifacts.local/work/mz20-rank-objective-20260910/run-v1'
    receipt = read(run / 'receipt.json')
    assert receipt['status'] == 'PASS'
    inputs = {}
    for name in ['predictions.npz', 'local_samples.npz', 'BODY_RANK-cutoff.npy']:
        inputs[str(run / name)] = sha(run / name)
        assert inputs[str(run / name)] == receipt['outputs'][name]
    cutoff = np.load(run / 'BODY_RANK-cutoff.npy')
    cohorts = {}; arrays = {}
    with np.load(run / 'predictions.npz') as saved, np.load(run / 'local_samples.npz') as local:
        for cohort in ['DEV', 'clean', 'stress', 'relation10000', 'distance5000']:
            prefix = f'BODY_RANK/{cohort}/'
            truth, base, candidate = [saved[prefix + k] for k in ['truth', 'baseline', 'candidate']]
            knownmax = np.full(truth.shape, -np.inf)
            knowncount = np.zeros(truth.shape, dtype=int)
            for q in range(4):
                score, frame = [local[prefix + f'{q}/{k}'] for k in ['score', 'frame']]
                assert len(score) == len(frame) and np.isfinite(score).all()
                np.maximum.at(knownmax[:, q], frame, score)
                knowncount[:, q] = np.bincount(frame, minlength=len(truth))
            oracle = np.where(base >= 0, base, knownmax - cutoff)
            supported = knowncount > 0
            np.testing.assert_array_equal(supported, np.isfinite(knownmax))
            assert not ((oracle >= 0) & (candidate < 0)).any()
            np.testing.assert_array_equal(oracle[base >= 0], base[base >= 0])
            gained = (candidate >= 0) & (base < 0) & truth
            fp = (candidate >= 0) & (base < 0) & ~truth
            retained = gained & (oracle >= 0)
            surviving_fp = (oracle >= 0) & (base < 0) & ~truth
            row = dict(frames=len(truth), baseline=scalar_metrics(base, truth),
                mz20=scalar_metrics(candidate, truth), oracle=scalar_metrics(oracle, truth),
                added_tp_before=gained.sum(0).tolist(), added_tp_retained=retained.sum(0).tolist(),
                added_tp_lost=(gained & ~retained).sum(0).tolist(), added_fp_before=fp.sum(0).tolist(),
                added_fp_after=surviving_fp.sum(0).tolist(), no_known_candidates=(~supported).sum(0).tolist())
            if cohort in ['clean', 'stress']:
                row['pole_mz20'] = scalar_metrics(candidate[100:125], truth[100:125])
                row['pole_oracle'] = scalar_metrics(oracle[100:125], truth[100:125])
            cohorts[cohort] = row
            arrays[cohort + '/known_max'] = knownmax
            arrays[cohort + '/known_count'] = knowncount
            arrays[cohort + '/oracle'] = oracle
    placement = ['relation10000', 'distance5000']
    far_before = sum(sum(cohorts[c]['added_tp_before'][1::2]) for c in placement)
    far_after = sum(sum(cohorts[c]['added_tp_retained'][1::2]) for c in placement)
    fp_before = sum(sum(cohorts[c]['added_fp_before']) for c in placement)
    fp_after = sum(sum(cohorts[c]['added_fp_after']) for c in placement)
    pole = {c: sum(cohorts[c]['pole_oracle']['tp'][1::2]) for c in ['clean', 'stress']}
    result = dict(cohorts=cohorts, placement_far_before=far_before, placement_far_retained=far_after,
        placement_added_fp_before=fp_before, placement_added_fp_after=fp_after, pole_retained=pole,
        diagnostic_headroom=dict(far_at_least_68=far_after >= 68, zero_added_fp=fp_after == 0,
            pole_at_least_48=all(v >= 48 for v in pole.values())),
        inputs=inputs, code_sha256=sha(Path(__file__)),
        scope='Evaluator-only oracle on consumed Development; native availability is privileged. Not an inference candidate or learned result.',
        interpretation='No known candidate means unavailable/UNKNOWN local evidence; never physical CLEAR. MZ5 positives preserved.',
        note='Actual-query labels in local_samples are deliberately not read. Existing cutoff and candidate scores remain unchanged.')
    write(output / 'result.json', result)
    np.savez_compressed(output / 'predictions.npz', **arrays)
    write(output / 'receipt.json', dict(status='PASS', inputs=inputs, code_sha256=sha(Path(__file__)),
        outputs={n:sha(output / n) for n in ['result.json', 'predictions.npz']},
        backend='NumPy CPU saved-score reduction', training_steps=0, inference_frames=0))
    print('PASS',dict(far_before=far_before,far_retained=far_after,fp_before=fp_before,fp_after=fp_after,pole=pole))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.output)
