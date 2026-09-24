"""Forward contract and local-support falsifier for the untrained model."""
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch research runtime unavailable')
class FusionInterfaceTest(unittest.TestCase):
    def test_h2_h3_candidate_parameter_gap(self):
        from cnh_rgb_fusion import FrustumFusion
        h2 = FrustumFusion(16, 24)
        h3 = FrustumFusion(64, 16)
        count = lambda model: sum(p.numel() for p in model.parameters())
        self.assertLessEqual(abs(count(h2) - count(h3)) / max(count(h2), count(h3)), .10)

    def test_three_modes_preserve_shapes_and_signed_bins(self):
        from cnh_rgb_fusion import FrustumFusion
        model = FrustumFusion(4, 16, width=16).eval()
        rgb = torch.rand(2, 3, 16, 16)
        hist = torch.randn(2, 4, 16)
        ambient = torch.ones(2, 4)
        scalar = torch.tensor([[float('nan'), 1., 2., 3.]] * 2)
        valid = torch.tensor([[False, True, True, True]] * 2)
        age = torch.zeros(2, 4)
        support = torch.ones(4, 4, 4)
        with torch.no_grad():
            for mode in ('cnh', 'scalar', 'rgb'):
                result = model(rgb, hist, ambient, scalar, valid, age, support, mode)
                self.assertEqual(result['occupancy_logits'].shape, (2, 6))
                self.assertTrue(torch.isfinite(result['occupancy_logits']).all())
                self.assertTrue(torch.isfinite(result['distance_m']).all())
                self.assertTrue((result['distance_m'] >= 0).all())
            changed = model(rgb, -hist, ambient, scalar, valid, age, support, 'cnh')
            original = model(rgb, hist, ambient, scalar, valid, age, support, 'cnh')
            self.assertFalse(torch.equal(changed['occupancy_logits'], original['occupancy_logits']))

    def test_rgb_patch_outside_zone_has_no_local_effect(self):
        from cnh_rgb_fusion import FrustumFusion
        model = FrustumFusion(1, 16, width=16).eval()
        rgb = torch.zeros(1, 3, 16, 16)
        hist = torch.zeros(1, 1, 16)
        ambient = scalar = age = torch.zeros(1, 1)
        valid = torch.zeros(1, 1, dtype=torch.bool)
        support = torch.zeros(1, 4, 4)
        support[0, 0, 0] = 1
        with torch.no_grad():
            first = model(rgb, hist, ambient, scalar, valid, age, support, 'cnh')
            changed = rgb.clone()
            changed[:, :, 12:16, 12:16] = 1
            second = model(changed, hist, ambient, scalar, valid, age, support, 'cnh')
        self.assertTrue(torch.equal(first['occupancy_logits'], second['occupancy_logits']))


if __name__ == '__main__':
    unittest.main()
