"""Focused checks for the matched scalar arm's input isolation."""
import unittest

import torch

from cnh_scalar_alley_control import AzimuthMaskedFrustumFusion, query_azimuth_mask


class ScalarAlleyControlTests(unittest.TestCase):
    def test_scalar_arm_ignores_histogram_but_uses_scalar(self):
        torch.manual_seed(7)
        model = AzimuthMaskedFrustumFusion().eval()
        rgb = torch.zeros(1, 3, 72, 128)
        hist = torch.randn(1, 64, 16)
        ambient = torch.zeros(1, 64)
        scalar = torch.ones(1, 64)
        valid = torch.ones(1, 64, dtype=torch.bool)
        age = torch.zeros(1, 64)
        support = torch.ones(64, 18, 32)
        with torch.no_grad():
            base = model(rgb, hist, ambient, scalar, valid, age, support)
            changed_hist = model(rgb, hist * 100, ambient, scalar, valid, age, support)
            changed_scalar = model(rgb, hist, ambient, scalar * 4, valid, age, support)
        self.assertTrue(torch.equal(base['occupancy_logits'], changed_hist['occupancy_logits']))
        self.assertGreater(torch.max(torch.abs(base['occupancy_logits'] - changed_scalar['occupancy_logits'])).item(), 1e-6)
        self.assertTrue(torch.equal(base['query_zone_weights'][0][~query_azimuth_mask()],
                                    torch.zeros_like(base['query_zone_weights'][0][~query_azimuth_mask()])))


if __name__ == '__main__':
    unittest.main()
