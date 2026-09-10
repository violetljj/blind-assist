import unittest
import torch
from torch.nn import functional as F
from mz15_train import balanced_local
from mz17_witness_loss import witness_query_loss


class WitnessLossTests(unittest.TestCase):
    def test_wrong_winner_is_suppressed_not_rewarded(self):
        x = torch.zeros(1, 1, 1, 102, 1, requires_grad=True)
        y = torch.zeros_like(x, dtype=torch.bool); y[..., 0, :] = True
        known = torch.ones_like(y); eligible = known.clone(); truth = torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad(): x[..., 1, :] = 1
        old = balanced_local(x, y, known) + .25 * F.softplus(-x.flatten(1, 3).amax(1)).mean()
        new = balanced_local(x, y, known) + .25 * witness_query_loss(x, y, known, eligible, truth)
        go = torch.autograd.grad(old, x, retain_graph=True)[0]
        gn = torch.autograd.grad(new, x)[0]
        self.assertLess(go[..., 1, :].item(), 0)
        self.assertGreater(gn[..., 1, :].item(), 0)
        self.assertLess(gn[..., 0, :].item(), 0)

    def test_unknown_and_ineligible_have_no_gradient(self):
        x = torch.tensor([1., 20., 30.]).reshape(1, 1, 1, 3, 1).requires_grad_()
        y = torch.zeros_like(x, dtype=torch.bool)
        k = torch.tensor([True, False, True]).reshape_as(y)
        e = torch.tensor([True, True, False]).reshape_as(y)
        loss = witness_query_loss(x, y, k, e, torch.zeros(1, 1, dtype=torch.bool))
        g = torch.autograd.grad(loss, x)[0].flatten()
        self.assertGreater(g[0], 0); self.assertEqual(g[1], 0); self.assertEqual(g[2], 0)
        empty = witness_query_loss(x, y, torch.zeros_like(k), e, torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(empty.item(), 0)
        self.assertTrue(torch.equal(torch.autograd.grad(empty, x)[0], torch.zeros_like(x)))

    def test_multiple_witnesses_are_not_false_candidates(self):
        x = torch.tensor([1., 2., 3.]).reshape(1, 1, 1, 3, 1).requires_grad_()
        y = torch.tensor([True, True, False]).reshape_as(x)
        k = torch.ones_like(y)
        loss = witness_query_loss(x, y, k, k, torch.ones(1, 1, dtype=torch.bool))
        g = torch.autograd.grad(loss, x)[0].flatten()
        self.assertEqual(g[0], 0); self.assertLess(g[1], 0); self.assertGreater(g[2], 0)


if __name__ == '__main__': unittest.main()
