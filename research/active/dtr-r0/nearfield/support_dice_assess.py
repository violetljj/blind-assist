"""Saved-prediction pixel accounting for the single support-loss comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_counts(probability, target):
    """Known-pixel confusion, including separate positive/empty-map spill."""
    assert probability.shape == target.shape and probability.ndim == 4
    assert np.isfinite(probability).all() and np.isin(target, [-1, 0, 1]).all()
    output = {}
    for j, head in enumerate(('BODY', 'HEAD')):
        p, t = probability[:, j] >= .5, target[:, j]
        positive, background, unknown = t == 1, t == 0, t == -1
        has_positive = positive.reshape(len(t), -1).any(axis=1)
        tp = int((p & positive).sum())
        fp = int((p & background).sum())
        fn = int((~p & positive).sum())
        fp_positive_maps = int((p[has_positive] & background[has_positive]).sum())
        fp_empty_maps = int((p[~has_positive] & background[~has_positive]).sum())
        ious, recalls = [], []
        hit = unknown_peak = known_miss = 0
        for i in np.flatnonzero(has_positive):
            inter = int((p[i] & positive[i]).sum())
            union = int(((p[i] | positive[i]) & ~unknown[i]).sum())
            ious.append(inter / union)
            recalls.append(inter / int(positive[i].sum()))
            at_peak = t[i].ravel()[probability[i, j].argmax()]
            hit += int(at_peak == 1)
            unknown_peak += int(at_peak == -1)
            known_miss += int(at_peak == 0)
        output[head] = dict(
            TP=tp, FP=fp, FN=fn, known_positive_pixels=tp+fn,
            known_background_pixels=int(background.sum()),
            positive_pixel_recall=tp/(tp+fn) if tp+fn else None,
            pixel_precision=tp/(tp+fp) if tp+fp else None,
            positive_map_macro_recall=float(np.mean(recalls)) if recalls else None,
            positive_maps=int(has_positive.sum()),
            positive_map_mean_iou=float(np.mean(ious)) if ious else None,
            FP_on_positive_maps=fp_positive_maps, FP_on_empty_maps=fp_empty_maps,
            empty_map_background_pixels=int(background[~has_positive].sum()),
            unknown_predicted_pixels=int((p & unknown).sum()),
            unknown_pixels=int(unknown.sum()), peak_hits=hit,
            peak_unknown_misses=unknown_peak, peak_known_misses=known_miss)
        assert fp == fp_positive_maps + fp_empty_maps
    return output


def assess(a):
    old, new = read(a.baseline/'result.json'), read(a.run/'result.json')
    for root in (a.baseline, a.run):
        assert sha(root/'result.json') == read(root/'receipt.json')['result_sha256']
    assert sha(a.baseline/'training_indices.npy') == sha(a.run/'training_indices.npy')
    cases = [('old750', a.old_cache, 'train'),
             ('relational384', a.relational_cache, 'train'),
             ('added64', a.train64, 'train'), ('dev', a.dev_cache, 'dev'),
             ('eval', a.eval64, 'eval')]
    report = dict(status='PASS', support_threshold=.5, schedule_identical=True,
                  baseline_result_sha256=sha(a.baseline/'result.json'),
                  result_sha256=sha(a.run/'result.json'), partitions={})
    for name, cache, split in cases:
        prefix = 'supervision' if split == 'train' else 'evaluator'
        labels = read(cache/prefix/f'{split}.json')
        support_path = cache/labels['support']['path']
        assert sha(support_path) == labels['support']['sha256']
        target = np.load(support_path, allow_pickle=False)
        values = {}
        for arm, root, result in [('A', a.baseline, old), ('B', a.run, new)]:
            path = root/f'B-{name}-predictions.npz'
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['sample_indices'], labels['sample_indices'])
                values[arm] = pixel_counts(z['support'], target)
            if split == 'train':
                previous = result['TRAIN'][name]['metrics']['fixed05']
            else:
                previous = result[split.upper()]['B']['fixed05']
            for head in ('BODY', 'HEAD'):
                expected = previous['heads'][head]['support']
                actual = values[arm][head]
                assert np.isclose(actual['positive_map_mean_iou'], expected['positive_frame_mean_iou'])
                assert actual['peak_hits'] == expected['peak_hits']
                assert actual['peak_unknown_misses'] == expected['peak_unknown_misses']
            values[arm+'_prediction_sha256'] = sha(path)
        report['partitions'][name] = values
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def self_check():
    target = np.array([[[[1, 0, -1]], [[1, 1, 0]]],
                       [[[0, 0, -1]], [[0, 0, -1]]]])
    p = np.array([[[[.9, .9, .99]], [[.9, .1, .9]]],
                  [[[.9, .1, .99]], [[.1, .1, .99]]]])
    r = pixel_counts(p, target)
    assert (r['BODY']['TP'], r['BODY']['FP'], r['BODY']['FN']) == (1, 2, 0)
    assert r['BODY']['FP_on_positive_maps'] == r['BODY']['FP_on_empty_maps'] == 1
    assert r['BODY']['unknown_predicted_pixels'] == 2
    assert r['BODY']['peak_unknown_misses'] == 1
    assert r['HEAD']['positive_pixel_recall'] == .5
    assert r['HEAD']['positive_map_mean_iou'] == 1/3
    print('PASS: known TP/FP/FN, positive versus empty maps, UNKNOWN peaks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    for name in ('baseline', 'run', 'old-cache', 'relational-cache', 'train64',
                 'eval64', 'dev-cache', 'output'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    self_check() if args.self_check else assess(args)
