"""Small CPU geometry/causal tests; no training or model-quality claims."""
import unittest

import torch

from mz8_attribution import AngularReturnReadout, angular_hypotheses, query_membership


class AttributionTests(unittest.TestCase):
    def test_half_open_near_far_and_region(self):
        points = torch.tensor([[1.679, 0., 1.], [1.68, 0., 1.],
                               [1.629, 0., 1.7], [1.63, 0., 1.7],
                               [3.18, 0., 1.], [3.181, 0., 1.]])
        self.assertEqual(query_membership(points).tolist(),
                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]])

    def test_radial_rays_and_wrong_visual_geometry_separation(self):
        rays, grid = angular_hypotheses()
        wrong_rays, wrong = angular_hypotheses(visual_shift_columns=4)
        torch.testing.assert_close(rays.norm(dim=-1), torch.ones(64, 49))
        torch.testing.assert_close(rays, wrong_rays)
        torch.testing.assert_close(grid[..., 1], wrong[..., 1])
        self.assertTrue(torch.all(grid[..., 0] != wrong[..., 0]))
        self.assertGreater(rays[0, 0, 2].item(), 0.)
        self.assertLess(rays[0, 0, 1].item(), 0.)

    def test_unknown_invalid_ranges_never_become_positive(self):
        model = AngularReturnReadout(1)
        ranges = torch.full((2, 64, 2), float('nan'))
        ranges[0, 0] = torch.tensor([0., 4.01])
        result = model.inspect(torch.zeros(2, 1, 8, 8), ranges, torch.ones_like(ranges, dtype=torch.bool))
        self.assertFalse(result['observation_valid'].any())
        self.assertFalse(result['support'].any())
        self.assertTrue(torch.isfinite(result['logits']).all())
        self.assertTrue((result['logits'] < 0).all())

    def test_visual_correspondence_changes_evidence_not_eligibility(self):
        model = AngularReturnReadout(1, hidden=1)
        with torch.no_grad():
            model.evidence[0].weight[:] = torch.tensor([[1., 0.]])
            model.evidence[0].bias.zero_()
            model.evidence[2].weight.fill_(1.)
            model.evidence[2].bias.fill_(-.5)
        features = torch.linspace(0, 1, 64)[None, None, None].expand(1, 1, 36, 64).clone()
        ranges = torch.full((1, 64, 2), 2.)
        valid = torch.zeros_like(ranges, dtype=torch.bool)
        valid[:, 28, 0] = True
        correct = model.inspect(features, ranges, valid)
        wrong = model.inspect(features, ranges, valid, wrong_zone=True)
        tokens = model.sample_visual(features)
        torch.testing.assert_close(correct['logits'], model.forward_tokens(tokens, ranges, valid))
        torch.testing.assert_close(wrong['logits'], model.forward_tokens(tokens, ranges, valid, wrong_zone=True))
        torch.testing.assert_close(correct['eligible'], wrong['eligible'])
        self.assertTrue(correct['support'][0, 3])
        self.assertGreater(correct['logits'][0, 3].item(), wrong['logits'][0, 3].item())
        # Eligible geometry alone cannot force a positive learned output.
        with torch.no_grad():
            model.evidence[2].bias.fill_(-10.)
        self.assertTrue((model(features, ranges, valid) < 0).all())

    def test_return_order_invariance_and_trainable_gradients(self):
        model = AngularReturnReadout(2)
        features = torch.randn(2, 2, 18, 32)
        ranges = torch.full((2, 64, 2), 2.)
        ranges[..., 1] = 1.
        valid = torch.ones_like(ranges, dtype=torch.bool)
        logits = model(features, ranges, valid)
        torch.testing.assert_close(logits, model(features, ranges.flip(-1), valid.flip(-1)))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(model.evidence[0].weight.grad).all())
        self.assertGreater(model.evidence[0].weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
