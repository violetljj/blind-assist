"""Positive image/head masked Dice, supplementing unchanged support BCE."""
import torch


def masked_soft_dice(logits, targets):
    """Mean over image/head maps with known positives; UNKNOWN has no gradient.

    Empty-positive maps contribute no Dice term, including all-UNKNOWN maps.
    Known background remains in each eligible map's predicted-area denominator.
    The existing balanced BCE still supervises every known pixel separately.
    """
    if logits.ndim != 4 or logits.shape != targets.shape:
        raise ValueError('Expected matching N x head x height x width tensors')
    if not torch.all((targets == -1) | (targets == 0) | (targets == 1)):
        raise ValueError('Support targets must be -1/0/1')
    known = targets >= 0
    positive = targets == 1
    probability = torch.sigmoid(logits) * known
    count = positive.sum(dim=(-2, -1))
    eligible = count > 0
    intersection = (probability * positive).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + count
    per_map = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    return (per_map * eligible).sum() / eligible.sum().clamp_min(1)
