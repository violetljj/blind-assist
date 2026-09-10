"""BODY witness separation across bags; evaluator labels are training-only."""
import torch
from torch.nn import functional as F


def body_rank_query_loss(field, target, known, eligible, truth, logits, support):
    """Preserve HEAD terms; distinguish witness ranking from absolute negatives.

    Each positive BODY query rewards a real witness. Negative bags penalize
    their original task-level inference maximum, as MZ16 already did. Positive
    bags supply only known false locations as additional ranking competitors.
    Unknown local labels remain unknown; negative event supervision does not
    assert local occupancy or CLEAR. Averaging negatives per positive prevents candidate/bag counts from
    silently multiplying this term. Margin=1 and weight=1 are fixed choices.
    """
    original_positive = (target & eligible).flatten(1, 3).any(1)
    mask = support & (~truth | original_positive)
    head = F.binary_cross_entropy_with_logits(
        logits[:, 2:], truth[:, 2:].float(), reduction='none') * mask[:, 2:]
    allowed = known.expand_as(target)[..., :2] & eligible[..., :2]
    positive, negative = allowed & target[..., :2], allowed & ~target[..., :2]
    hp = positive.flatten(1, 3).any(1) & truth[:, :2]
    hn = negative.flatten(1, 3).any(1)
    bp = field[..., :2].masked_fill(~positive, -1e6).flatten(1, 3).amax(1)
    bn = field[..., :2].masked_fill(~negative, -1e6).flatten(1, 3).amax(1)
    negative_bag = mask[:, :2] & ~truth[:, :2]
    # MZ19 found that known-local-only maxima omit the actual negative-bag
    # tail driving the alert cutoff. Retain existing task-level supervision.
    competitor = torch.where(negative_bag, logits[:, :2], bn)
    has_competitor = negative_bag | (hn & truth[:, :2])
    classification = (F.softplus(-bp) * hp + F.softplus(logits[:, :2]) * negative_bag).sum()
    # [positive frame, negative frame, BODY query], including same-frame pairs.
    pairs = hp[:, None, :] & has_competitor[None, :, :]
    losses = F.softplus(1.0 + competitor[None, :, :] - bp[:, None, :]) * pairs
    ranking = (losses.sum(1) / pairs.sum(1).clamp_min(1)).sum()
    return (classification + ranking + head.sum()) / mask.sum().clamp_min(1)
