"""The new path must retain image evidence when the packet has no candidate."""
import unittest

import torch

from mz15_shared_support import LocalSupportReadout
from mz50_echo_independent import SpatialQuery, loss_for


class EchoIndependentTest(unittest.TestCase):
    def test_missing_returns_do_not_delete_rgb_path(self):
        torch.set_num_threads(1)
        torch.manual_seed(150)
        model = SpatialQuery(LocalSupportReadout(shared=False))
        with torch.no_grad():
            model.head[2].weight.zero_()
            model.head[2].bias.fill_(2.)
        x = torch.randn(2, 64, 28, 28)
        r, v = torch.zeros(2, 64, 2), torch.zeros(2, 64, 2, dtype=torch.bool)
        out = model.inspect(x, r, v)
        self.assertFalse(out['packet_available'].any())
        self.assertTrue(out['OPEN']['support'].all())
        self.assertFalse(out['GATED']['support'].any())
        self.assertTrue(torch.equal(out['OPEN']['raw'], torch.full((2, 4), 2.)))
        self.assertTrue(torch.equal(out['GATED']['raw'], torch.full((2, 4), -20.)))
        target = torch.zeros(2, 64, 49, 4, dtype=torch.bool)
        known = torch.zeros(2, 64, 49, dtype=torch.bool)
        loss, _ = loss_for(out, target, known, torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.equal(model.head[2].bias.grad, torch.zeros(4)))

    def test_shared_logits_and_input_contract(self):
        torch.manual_seed(150)
        rank = LocalSupportReadout(shared=False)
        model = SpatialQuery(rank)
        self.assertTrue(torch.equal(model.head[0].weight, rank.head[0].weight[:, :40]))
        x = torch.randn(2, 64, 28, 28)
        r, v = torch.ones(2, 64, 2), torch.ones(2, 64, 2, dtype=torch.bool)
        out = model.inspect(x, r, v)
        self.assertTrue(out['GATED']['support'].any())
        for q in range(4):
            for i in range(2):
                eligible = out['eligible'][i, :, :, q]
                if eligible.any():
                    self.assertEqual(float(out['GATED']['raw'][i, q].detach()),
                                     float(out['field'][i, :, :, q][eligible].max().detach()))
        self.assertTrue((out['OPEN']['raw'] >= out['GATED']['raw']).all())


if __name__ == '__main__':
    unittest.main()
