"""Fixed-threshold City pilot metrics; no fitting or threshold selection."""
import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def evaluate(near_prob, support_prob, near_target, support_target, group_ids=None):
    near_prob, support_prob, near_target, support_target = map(
        _array, (near_prob, support_prob, near_target, support_target))
    if near_prob.ndim != 2 or near_prob.shape[1:] != (2,):
        raise ValueError('Expected Nx2 near probabilities')
    n = len(near_prob)
    if near_target.shape != (n, 2) or support_prob.shape != (n, 2, 18, 32) or support_target.shape != support_prob.shape:
        raise ValueError('Probability/target dimensions mismatch')
    for probability in (near_prob, support_prob):
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError('Probabilities must be finite in [0,1]')
    if not np.isin(near_target, [-1, 0, 1]).all() or not np.isin(support_target, [-1, 0, 1]).all():
        raise ValueError('Targets must be -1/0/1')
    decisions = near_prob >= .5
    result = dict(schema='city-pilot-metrics-v1', backend='CPU',
        backend_reason='TASK_NOT_GPU_SUITABLE_SMALL_ARRAY_METRICS',
        thresholds=dict(near=.5, support=.5, comparison='inclusive >='), frames=n, heads={})
    for h, name in enumerate(('BODY', 'HEAD')):
        truth = near_target[:, h]
        known = truth >= 0
        positive, negative = truth == 1, truth == 0
        pred = decisions[:, h]
        tp, fp = int((pred & positive).sum()), int((pred & negative).sum())
        fn, tn = int((~pred & positive).sum()), int((~pred & negative).sum())
        recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
        balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None
        near = dict(TP=tp, FP=fp, FN=fn, TN=tn, recall=recall,
            precision=_ratio(tp, tp + fp), specificity=specificity,
            balanced_accuracy=balanced_accuracy,
            balanced_error=1-balanced_accuracy if balanced_accuracy is not None else None,
            known_frames=int(known.sum()), unknown_frames=int((~known).sum()),
            unknown_fraction=_ratio(int((~known).sum()), n))
        maps, labels = support_prob[:, h], support_target[:, h]
        mask_known, mask_positive = labels >= 0, labels == 1
        mask_pred = maps >= .5
        ious = []
        peak_hits = peak_unknown = peak_known_misses = positive_not_evaluable = 0
        for i in np.flatnonzero(positive):
            if not mask_positive[i].any():
                positive_not_evaluable += 1
                continue
            intersection = int((mask_pred[i] & mask_positive[i]).sum())
            union = int(((mask_pred[i] | mask_positive[i]) & mask_known[i]).sum())
            ious.append(intersection / union)
            # Peak is chosen over the complete predicted map. An unknown peak
            # remains a miss; it is never relocated onto a known pixel.
            peak_truth = labels[i].reshape(-1)[int(maps[i].argmax())]
            peak_hits += int(peak_truth == 1)
            peak_unknown += int(peak_truth == -1)
            peak_known_misses += int(peak_truth == 0)
        negative_known = mask_known[negative]
        negative_false = mask_pred[negative] & (labels[negative] == 0)
        negative_evaluable = negative_known.reshape(int(negative.sum()), 18*32).any(axis=1)
        negative_frame_fp = negative_false.reshape(int(negative.sum()), 18*32).any(axis=1)
        support = dict(positive_frames=int(positive.sum()),
            positive_evaluable_frames=len(ious), positive_not_evaluable_frames=positive_not_evaluable,
            positive_frame_mean_iou=float(np.mean(ious)) if ious else None,
            peak_hits=peak_hits, peak_unknown_misses=peak_unknown, peak_known_misses=peak_known_misses,
            peak_hit_rate=_ratio(peak_hits, len(ious)),
            negative_frames=int(negative.sum()), negative_evaluable_frames=int(negative_evaluable.sum()),
            negative_false_positive_mask_frames=int(negative_frame_fp.sum()),
            negative_false_positive_mask_rate=_ratio(int(negative_frame_fp.sum()), int(negative_evaluable.sum())),
            negative_false_positive_pixels=int(negative_false.sum()),
            negative_known_background_pixels=int((labels[negative] == 0).sum()),
            negative_false_positive_pixel_rate=_ratio(int(negative_false.sum()), int((labels[negative] == 0).sum())),
            unknown_pixels=int((~mask_known).sum()),
            unknown_fraction=_ratio(int((~mask_known).sum()), labels.size))
        result['heads'][name] = dict(near=near, support=support)
    errors = [head['near']['balanced_error'] for head in result['heads'].values()]
    result['near_balanced_error'] = sum(errors)/2 if all(value is not None for value in errors) else None
    if group_ids is not None:
        ids = list(group_ids)
        if len(ids) != n or any(not isinstance(g, str) or not g for g in ids):
            raise ValueError('Expected one nonempty string group ID per frame')
        groups = {}
        for i, group in enumerate(ids):
            groups.setdefault(group, []).append(i)
        correct = evaluable = 0
        for indices in groups.values():
            target = near_target[indices]
            if not (target >= 0).all():
                continue
            evaluable += 1
            correct += int((decisions[indices] == target).all())
        result['group_joint'] = dict(groups=len(groups), evaluable_groups=evaluable,
            unknown_groups=len(groups)-evaluable, all_correct_groups=correct,
            accuracy=_ratio(correct, evaluable), rule='All frames and both heads correct; any unknown excludes group')
    return result
