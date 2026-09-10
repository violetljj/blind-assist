"""Training-only witness reward and worst known false-candidate penalty."""
import torch
from torch.nn import functional as F


def witness_query_loss(field, target, known, eligible, truth):
    allowed = known.expand_as(target) & eligible
    positive = allowed & target
    negative = allowed & ~target
    has_positive = positive.flatten(1, 3).any(1) & truth
    has_negative = negative.flatten(1, 3).any(1)
    # Finite sentinels and explicit masks keep all-unknown batches differentiable.
    best_positive = field.masked_fill(~positive, -1e6).flatten(1, 3).amax(1)
    best_negative = field.masked_fill(~negative, -1e6).flatten(1, 3).amax(1)
    terms = F.softplus(-best_positive) * has_positive + F.softplus(best_negative) * has_negative
    active = has_positive | has_negative
    return terms.sum() / active.sum().clamp_min(1)
