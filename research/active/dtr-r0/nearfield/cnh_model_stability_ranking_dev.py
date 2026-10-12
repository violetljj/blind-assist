"""Descriptive ranking of deadline event peaks against clear time intervals.

No fitting, threshold recommendation or deployable operating point is returned.
The low-FA integral uses only whole tied-score groups: it holds the last
attainable sensitivity until the budget can admit the next group. The full AUC
is the conventional pairwise ranking probability, with half credit for ties.
These are different units (event peaks versus control intervals), not a
frame-classification AUC or a claim that repeated frames are independent.
"""
import numpy as np


def _rank_summary(positive, negative, cap):
    if not len(positive) or not len(negative):
        return dict(auc=None, partial_step_area=None,
                    partial_step_area_normalized=None, unique_scores=0)
    values, inverse = np.unique(np.concatenate((positive, negative)), return_inverse=True)
    pc = np.bincount(inverse[:len(positive)], minlength=len(values))
    nc = np.bincount(inverse[len(positive):], minlength=len(values))
    below = np.cumsum(nc) - nc
    auc = float(np.sum(pc * (below + .5 * nc)) / (len(positive) * len(negative)))

    # Descending whole groups. Simultaneous positive/negative ties cannot be
    # fractionally admitted; horizontal spans use the previously attainable TPR.
    false_count = true_count = 0
    previous_fpr = sensitivity = area = 0.0
    for p, n in zip(pc[::-1], nc[::-1]):
        false_count += int(n)
        true_count += int(p)
        fpr = false_count / len(negative)
        if fpr > cap:
            area += (cap - previous_fpr) * sensitivity
            break
        area += (fpr - previous_fpr) * sensitivity
        previous_fpr = fpr
        sensitivity = true_count / len(positive)
    else:
        area += (cap - previous_fpr) * sensitivity
    return dict(auc=auc, partial_step_area=float(area),
                partial_step_area_normalized=float(area / cap),
                unique_scores=len(values))


def ranking_metrics(scores, contact, control, deadline, mask=None, cap=.05):
    """Rank contact max(scores[2:deadline+1]) against clear scores[2:12].

    Scores have shape (N,13). ``mask`` selects the requested valid population.
    Contact deadlines below2 are explicitly unevaluable; negative deadlines
    are additionally distinguished as missing. Their scores never enter the
    ranking. Ignored rows and unused time windows may contain NaNs. A nonfinite
    score inside a used window is an error, never a silently dropped sample.
    No class means undefined metrics (None), with all denominators retained.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 13:
        raise ValueError('scores must have shape (N,13)')
    n = len(scores)
    contact, control = np.asarray(contact, bool), np.asarray(control, bool)
    deadline = np.asarray(deadline)
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    if any(v.shape != (n,) for v in (contact, control, deadline, mask)):
        raise ValueError('labels, deadlines and mask must have shape (N,)')
    if not np.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError('cap must be in (0,1]')
    selected_contact, selected_control = contact & mask, control & mask
    if (selected_contact & selected_control).any():
        raise ValueError('selected contact and control overlap')
    chosen_deadline = deadline[selected_contact]
    if (not np.isfinite(chosen_deadline).all() or
            not np.equal(chosen_deadline, np.floor(chosen_deadline)).all() or
            (chosen_deadline >= 13).any()):
        raise ValueError('contact deadlines must be integral and below13; negative means missing')
    early = selected_contact & (deadline >= 0) & (deadline < 2)
    missing = selected_contact & (deadline < 0)
    evaluable = selected_contact & (deadline >= 2)
    peaks = []
    for i in np.flatnonzero(evaluable):
        window = scores[i, 2:int(deadline[i])+1]
        if not np.isfinite(window).all():
            raise ValueError('nonfinite score within an evaluable contact window')
        peaks.append(float(window.max()))
    negative = np.asarray(scores[selected_control, 2:12], float).ravel()
    if not np.isfinite(negative).all():
        raise ValueError('nonfinite score within a selected control window')
    positive = np.asarray(peaks, float)
    return dict(
        definition='deadline event peak versus clear time interval ranking; descriptive only',
        cap=float(cap), selected_sequences=int(mask.sum()),
        contact_events=int(selected_contact.sum()), positive_events=len(positive),
        unevaluable_contact_events=int((early | missing).sum()),
        early_deadline_events=int(early.sum()), missing_deadline_events=int(missing.sum()),
        control_sequences=int(selected_control.sum()), control_intervals=len(negative),
        **_rank_summary(positive, negative, float(cap)))
