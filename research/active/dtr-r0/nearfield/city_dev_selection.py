"""DEV-only empirical operating-point and whole-checkpoint selection."""
import numpy as np


def apply_thresholds(scores, thresholds):
    """Inclusive float64 comparison; preserve the nextafter all-negative sentinel.

    Accept a 1D score vector and scalar threshold, or Nx2 scores and two
    thresholds ordered BODY, HEAD. Return a boolean NumPy array.
    """
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    valid_shape = ((scores.ndim == 1 and thresholds.ndim == 0)
                   or (scores.ndim == 2 and scores.shape[1:] == (2,) and thresholds.shape == (2,)))
    if not valid_shape:
        raise ValueError('Expected scores[N]/scalar or scores[N,2]/thresholds[2]')
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Scores must be finite in [0,1]')
    if not np.isfinite(thresholds).all():
        raise ValueError('Thresholds must be finite')
    return scores >= thresholds


def select_threshold(scores, labels, min_count=48):
    """Maximize recall at FPR <= .10, then lower FPR, then higher threshold.

    The caller owns DEV provenance; no TEST data is accepted separately or used.
    Unknown labels are excluded from selection and reported in coverage.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError('Expected equal one-dimensional scores and labels')
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Scores must be finite in [0,1]')
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Labels must be -1/0/1')
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError('min_count must be a positive integer')
    known = labels >= 0
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives < min_count or negatives < min_count:
        raise ValueError(f'Inadequate DEV classes: positive={positives}, negative={negatives}, minimum={min_count}')
    # Include a finite all-negative candidate even when the maximum is 1.
    known_scores = scores[known].astype(np.float64)
    known_labels = labels[known]
    thresholds = np.append(np.unique(known_scores), np.nextafter(float(known_scores.max()), np.inf))
    best = None
    for threshold in thresholds:
        predicted = apply_thresholds(known_scores, threshold)
        tp = int((predicted & (known_labels == 1)).sum())
        fp = int((predicted & (known_labels == 0)).sum())
        # Integer arithmetic makes the exact 10% boundary independent of float roundoff.
        if 10 * fp > negatives:
            continue
        rank = (tp, -fp, float(threshold))
        if best is None or rank > best[0]:
            best = (rank, dict(threshold=float(threshold), TP=tp, FP=fp,
                FN=positives-tp, TN=negatives-fp, recall=tp/positives, FPR=fp/negatives))
    result = best[1]
    result.update(total_count=len(labels), known_count=int(known.sum()),
        unknown_count=int((~known).sum()), known_fraction=float(known.mean()),
        positive_count=positives, negative_count=negatives, min_count=min_count,
        max_FPR=.10, comparison='inclusive >=',
        selection='DEV only: max recall, then lower FPR, then higher threshold')
    return result


def select_checkpoint(dev_arms, labels, min_count=48):
    """Choose one whole A/B/C checkpoint; never splice heads across models.

    dev_arms maps exactly A/B/C to Nx2 probabilities; labels is the same DEV
    Nx2 target array for all arms. Every head must meet class adequacy.
    """
    if set(dev_arms) != {'A', 'B', 'C'}:
        raise ValueError('Expected exactly the fixed arms A, B, C')
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1:] != (2,):
        raise ValueError('Expected Nx2 DEV labels')
    arms = {}
    for arm in ('A', 'B', 'C'):
        scores = np.asarray(dev_arms[arm])
        if scores.shape != labels.shape:
            raise ValueError('Arm/DEV label shape mismatch')
        heads = {name: select_threshold(scores[:, h], labels[:, h], min_count)
                 for h, name in enumerate(('BODY', 'HEAD'))}
        recalls = [row['recall'] for row in heads.values()]
        fprs = [row['FPR'] for row in heads.values()]
        arms[arm] = dict(heads=heads, minimum_head_recall=min(recalls),
            macro_recall=sum(recalls)/2, macro_FPR=sum(fprs)/2)
    winner = max(('A', 'B', 'C'), key=lambda arm: (
        arms[arm]['minimum_head_recall'], arms[arm]['macro_recall'],
        -arms[arm]['macro_FPR'], -('A', 'B', 'C').index(arm)))
    return dict(schema='city-dev-selection-v1', selected_arm=winner, arms=arms,
        selected_thresholds={name: row['threshold'] for name, row in arms[winner]['heads'].items()},
        rule='One whole checkpoint: max minimum-head recall, then macro recall, then lower macro FPR, then A/B/C order',
        data_scope='DEV only; caller must bind provenance and keep TEST unopened')
