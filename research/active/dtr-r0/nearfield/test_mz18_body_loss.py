"""Focused field-gradient contracts for the BODY-only witness intervention."""
import unittest

import torch
from torch.nn import functional as F

from mz15_train import balanced_local
from mz18_body_loss import body_witness_query_loss


def readout(field, eligible):
    support = eligible.flatten(1, 3).any(1)
    pooled = field.masked_fill(~eligible, -torch.inf).flatten(1, 3).amax(1)
    logits = torch.where(support, pooled, pooled.new_full(pooled.shape, -20.))
    return logits, support


def original_query_loss(field, target, eligible, truth):
    logits, support = readout(field, eligible)
    positive = (target & eligible).flatten(1, 3).any(1)
    mask = support & (~truth | positive)
    return (F.binary_cross_entropy_with_logits(logits, truth.float(), reduction='none') * mask).sum() / mask.sum().clamp_min(1)


def hybrid_loss(field, target, known, eligible, truth):
    logits, support = readout(field, eligible)
    return body_witness_query_loss(field, target, known, eligible, truth, logits, support)


class BodyWitnessLossTests(unittest.TestCase):
    def test_wrong_body_max_changes_from_reward_to_suppression(self):
        field = torch.zeros(1, 1, 1, 102, 4, requires_grad=True)
        target = torch.zeros_like(field, dtype=torch.bool)
        target[..., 0, :2] = True
        known = torch.ones_like(target)
        eligible = known.clone()
        truth = torch.tensor([[True, True, False, False]])
        with torch.no_grad():
            field[..., 1, :2] = 1
        local = balanced_local(field, target, known)
        old = local + .25 * original_query_loss(field, target, eligible, truth)
        new = local + .25 * hybrid_loss(field, target, known, eligible, truth)
        go = torch.autograd.grad(old, field, retain_graph=True)[0]
        gn = torch.autograd.grad(new, field)[0]
        self.assertTrue((go[..., 1, :2] < 0).all())
        self.assertTrue((gn[..., 1, :2] > 0).all())
        self.assertTrue((gn[..., 0, :2] < 0).all())

    def test_head_gradients_exact_with_original_global_denominator(self):
        field = torch.tensor([
            [[[[-1., 2., 0., 3.], [2., -1., 0., 3.], [0., 0., 1., -2.]]]],
            [[[[1., -2., 2., 0.], [0., 0., 2., 1.], [-1., 1., -1., 2.]]]],
        ], requires_grad=True)
        target = torch.zeros_like(field, dtype=torch.bool)
        target[0, ..., 0, 0] = True
        target[0, ..., 0, 3] = True
        target[1, ..., 0, 2] = True
        # 5 supervised outputs total: BODY0 and HEAD2/3 in batch0,
        # BODY1 and HEAD2 in batch1. BODY1/0 lack witnesses respectively;
        # batch1 HEAD3 is unsupported. Mixed masks expose renormalization.
        truth = torch.tensor([[True, True, False, True], [True, False, True, False]])
        eligible = torch.ones_like(target)
        eligible[1, ..., 3] = False
        known = torch.ones_like(target)
        known[0, ..., 1, 2:] = False  # HEAD must keep original max semantics.
        go = torch.autograd.grad(original_query_loss(field, target, eligible, truth), field, retain_graph=True)[0]
        gn = torch.autograd.grad(hybrid_loss(field, target, known, eligible, truth), field)[0]
        self.assertTrue(torch.equal(go[..., 2:], gn[..., 2:]))
        self.assertTrue((go[..., 2:] != 0).any())
        # HEAD3's two equal maxima must still receive equal nonzero shares.
        self.assertEqual(gn[0, 0, 0, 0, 3], gn[0, 0, 0, 1, 3])
        self.assertLess(gn[0, 0, 0, 0, 3], 0)

    def test_body_unknown_ineligible_multiple_witnesses_and_empty(self):
        field = torch.tensor([1., 2., 3., 20., 30.]).reshape(1, 1, 1, 5, 1).expand(-1, -1, -1, -1, 4).clone().requires_grad_()
        target = torch.zeros_like(field, dtype=torch.bool)
        target[..., :2, :2] = True
        known = torch.ones_like(target)
        known[..., 3, :2] = False
        eligible = torch.ones_like(target)
        eligible[..., 4, :2] = False
        eligible[..., 2:] = False
        truth = torch.ones((1, 4), dtype=torch.bool)
        grad = torch.autograd.grad(hybrid_loss(field, target, known, eligible, truth), field, retain_graph=True)[0]
        self.assertTrue((grad[..., 0, :2] == 0).all())
        self.assertTrue((grad[..., 1, :2] < 0).all())
        self.assertTrue((grad[..., 2, :2] > 0).all())
        self.assertTrue((grad[..., 3:, :2] == 0).all())
        empty = hybrid_loss(field, target, torch.zeros_like(known), eligible, truth)
        self.assertTrue(torch.isfinite(empty))
        self.assertEqual(empty.item(), 0)
        self.assertTrue(torch.equal(torch.autograd.grad(empty, field)[0], torch.zeros_like(field)))


if __name__ == '__main__':
    unittest.main()
