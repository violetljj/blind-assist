"""Replace BODY query rewards; preserve HEAD's original query-logit gradients."""
from torch.nn import functional as F


def body_witness_query_loss(field, target, known, eligible, truth, logits, support):
    original_positive = (target & eligible).flatten(1, 3).any(1)
    original_mask = support & (~truth | original_positive)
    head_terms = F.binary_cross_entropy_with_logits(logits[:, 2:], truth[:, 2:].float(), reduction='none') * original_mask[:, 2:]
    allowed = known.expand_as(target)[..., :2] & eligible[..., :2]
    positive = allowed & target[..., :2]
    negative = allowed & ~target[..., :2]
    has_positive = positive.flatten(1, 3).any(1) & truth[:, :2]
    has_negative = negative.flatten(1, 3).any(1)
    best_positive = field[..., :2].masked_fill(~positive, -1e6).flatten(1, 3).amax(1)
    best_negative = field[..., :2].masked_fill(~negative, -1e6).flatten(1, 3).amax(1)
    body_terms = F.softplus(-best_positive) * has_positive + F.softplus(best_negative) * has_negative
    # Keep the original denominator: HEAD supervision is unchanged at fixed logits.
    # BODY now has up to two terms and its effective weight intentionally changes.
    return (body_terms.sum() + head_terms.sum()) / original_mask.sum().clamp_min(1)
