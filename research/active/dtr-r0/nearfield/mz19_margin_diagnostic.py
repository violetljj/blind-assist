"""Read saved consumed-Development scores; no fitting or cutoff selection.

Frame maxima, local known-cell maxima, and pairwise ordering are diagnostics.
They are not independent obstacle observations or a proposed operating point.
"""
import argparse
from pathlib import Path

import numpy as np

from mz5_ensemble_readout import read, sha, write


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), quantiles=dict(zip(
        ['min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'],
        np.quantile(a, [0, .1, .25, .5, .75, .9, 1]).tolist()
    ))) if len(a) else dict(n=0, quantiles=None)


def maxima(score, label, frame, count):
    positive = np.full(count, -np.inf)
    negative = np.full(count, -np.inf)
    np.maximum.at(positive, frame[label], score[label])
    np.maximum.at(negative, frame[~label], score[~label])
    return positive, negative


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    work = root / 'artifacts.local/work'
    inputs, result, stored = {}, {}, {}
    for tag, folder, arm in [
        ('MZ16', work / 'mz16-visual-detail-20260910/run-v1', 'HIGH_DETAIL'),
        ('MZ18', work / 'mz18-body-objective-20260910/run-v1', 'BODY_WITNESS'),
    ]:
        receipt = read(folder / 'receipt.json')
        for name in ['predictions.npz', 'local_samples.npz', arm + '-cutoff.npy']:
            digest = sha(folder / name)
            assert digest == receipt['outputs'][name]
            inputs[str(folder / name)] = digest
        result[tag], stored[tag] = {}, {}
        cutoffs = np.load(folder / (arm + '-cutoff.npy'))
        with np.load(folder / 'predictions.npz') as predictions, np.load(folder / 'local_samples.npz') as local:
            for cohort in ['DEV', 'clean', 'stress', 'relation10000', 'distance5000']:
                result[tag][cohort], stored[tag][cohort] = {}, {}
                prefix = f'{arm}/{cohort}/'
                for q in [0, 1, 3]:
                    truth, support, base, raw, winner = [predictions[prefix + key][:, q] for key in
                        ['truth', 'support', 'baseline', 'raw', 'winning_query']]
                    score, label, frame = [local[prefix + f'{q}/{key}'] for key in ['score', 'label', 'frame']]
                    pos, neg = maxima(score, label, frame, len(truth))
                    hard = support & (base < 0) & ~truth
                    available = truth & np.isfinite(pos)
                    paired = available & np.isfinite(neg)
                    cutoff = float(cutoffs[q])
                    row = dict(cutoff=cutoff, frames=len(truth), true_frames=int(truth.sum()),
                        available_true_witness=int(available.sum()),
                        known_true_max=summary(pos[available]), known_wrong_max_on_true=summary(neg[truth]),
                        true_minus_known_wrong=summary(pos[paired] - neg[paired]),
                        actual_global_winners_on_true=int((winner & truth & support).sum()),
                        true_witness_above_cutoff=int((available & (pos >= cutoff)).sum()),
                        hard_negative_raw_maxima=summary(raw[hard]),
                        hard_negative_known_wrong_maxima=summary(neg[hard]),
                        hard_negative_above_cutoff=int((raw[hard] >= cutoff).sum()),
                        hard_negative_top_frames=sorted([
                            dict(frame=int(i), raw=float(raw[i]), known_wrong_max=float(neg[i]) if np.isfinite(neg[i]) else None,
                                 global_winner_actual_query=bool(winner[i])) for i in np.flatnonzero(hard)
                        ], key=lambda v: v['raw'], reverse=True)[:10])
                    stored[tag][cohort][q] = dict(pos=pos, neg=neg, raw=raw, hard=hard)
                    if cohort in ['clean', 'stress'] and q == 1:
                        ids = np.arange(100, 125)
                        assert truth[ids].all() and np.isfinite(pos[ids]).all()
                        pair = np.isfinite(neg[ids])
                        row['pole'] = dict(frames=ids.tolist(), true_max=pos[ids].tolist(),
                            known_wrong_max=[float(v) if np.isfinite(v) else None for v in neg[ids]],
                            margin_summary=summary(pos[ids] - cutoff),
                            rank_gap_summary=summary(pos[ids][pair] - neg[ids][pair]),
                            actual_global_winner=int(winner[ids].sum()),
                            raw_equals_true_max=bool(np.allclose(raw[ids], pos[ids])))
                    result[tag][cohort][str(q)] = row
        for condition in ['clean', 'stress']:
            pole = stored[tag][condition][1]['pos'][100:125]
            comparison = {}
            for cohort in ['DEV', 'relation10000', 'distance5000']:
                values = stored[tag][cohort][1]
                hard = values['raw'][values['hard']]
                counts = (hard[None, :] >= pole[:, None]).sum(1)
                comparison[cohort] = dict(hard_negative_frames=len(hard),
                    hard_negatives_ranking_above_each_pole=counts.tolist(),
                    count_summary=summary(counts),
                    pairwise_pole_over_negative_fraction=float((pole[:, None] > hard[None, :]).mean()) if len(hard) else None,
                    pole_above_all_hard_negatives=int((counts == 0).sum()))
            result[tag][condition]['1']['pole']['comparison'] = comparison
    write(output / 'result.json', dict(result=result, inputs=inputs,
        code_sha256=sha(Path(__file__)), backend='NumPy CPU: saved score metadata reduction',
        scope='Consumed Development only. No inference, training, threshold sweep/selection, or protected EVAL access.'))
    write(output / 'receipt.json', dict(status='PASS', inputs=inputs, code_sha256=sha(Path(__file__)),
        outputs={'result.json': sha(output / 'result.json')}))
    for tag in result:
        for condition in ['clean', 'stress']:
            print(tag, condition, result[tag][condition]['1']['pole'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
