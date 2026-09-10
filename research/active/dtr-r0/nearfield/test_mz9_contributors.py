"""CUDA packet parity and actual-source attribution checks."""
import unittest

import torch

from multizone64_observation import geometry, observe
from mz9_contributors import reconstruct
from mz9_source_readout import SourceReadout


class SourceReadoutTests(unittest.TestCase):
    def test_query_axes_and_support_mask(self):
        model = SourceReadout()
        with torch.no_grad():
            model.evidence[0].weight.zero_()
            model.evidence[0].bias.zero_()
            model.evidence[2].weight.zero_()
            model.evidence[2].bias.copy_(torch.tensor([1., 2., 3., 4.]))
        tokens = torch.zeros(2, 64, 49, 64)
        ranges = torch.full((2, 64, 2), 2.)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        valid[1] = False
        result = model.forward_tokens(tokens, ranges, valid, return_details=True)
        self.assertEqual(result['candidate_logits'].shape, (2, 64, 2, 49, 4))
        self.assertEqual(result['eligible'].shape, (2, 64, 2, 49, 4))
        self.assertEqual(result['support'][0].tolist(), [False, True, False, True])
        torch.testing.assert_close(result['logits'][0], torch.tensor([-20., 2., -20., 4.]))
        self.assertFalse(result['observation_valid'][1])
        self.assertTrue((result['logits'][1] == -20).all())

    def test_no_rgb_invariance_and_inactive_input_weights(self):
        torch.manual_seed(109)
        model = SourceReadout(no_rgb=True)
        first = torch.randn(1, 64, 49, 64)
        second = torch.randn_like(first)*100
        ranges = torch.full((1, 64, 2), 2.)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        a = model.forward_tokens(first, ranges, valid, return_details=True)
        b = model.forward_tokens(second, ranges, valid, wrong_zone=True, return_details=True)
        torch.testing.assert_close(a['candidate_logits'], b['candidate_logits'], atol=0, rtol=0)
        torch.testing.assert_close(a['logits'], b['logits'], atol=0, rtol=0)
        (a['candidate_logits'].sum()+a['logits'].sum()).backward()
        self.assertEqual(model.evidence[0].weight.grad[:, :64].abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(model.evidence[0].weight.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA observation contract')
class ContributorTests(unittest.TestCase):
    def setUp(self):
        self.g = geometry(torch.device('cuda'))

    def check_packet(self, depth):
        result = reconstruct(depth)
        expected = observe(depth, readout='multi_surface')
        torch.testing.assert_close(result['valid'], expected['valid'])
        torch.testing.assert_close(result['range_m'], expected['range_m'], atol=1e-11, rtol=1e-11, equal_nan=True)
        torch.testing.assert_close(result['source_counts'].sum(-1), result['selected_bin_counts'])
        self.assertTrue((result['query_counts'] <= result['source_counts'][..., None]).all())
        return result

    def test_two_bins_single_bin_and_unsupported(self):
        depth = torch.zeros(3, 360, 640, dtype=torch.float64, device='cuda')
        pixels = (self.g['zone_ids'] == 35).flatten().nonzero().flatten()
        factor = self.g['radial_factor'].flatten()
        # First and last supported bins chosen; middle supported bin excluded.
        for start, stop, radial in [(0, 4, 1.25), (4, 9, 1.85), (9, 15, 2.35)]:
            depth[0].flatten()[pixels[start:stop]] = radial/factor[pixels[start:stop]]
        depth[1].flatten()[pixels[:5]] = 2.35/factor[pixels[:5]]
        depth[2].flatten()[pixels[:2]] = 1.25/factor[pixels[:2]]
        result = self.check_packet(depth)
        self.assertEqual(result['source_counts'][:, 35].sum(-1).tolist(), [[4, 6], [5, 0], [0, 0]])
        self.assertEqual(result['selected_bins'][0, 35].tolist(), [12, 23])
        self.assertEqual(result['cell_known_counts'][:, 35].sum(-1).tolist(), [15, 5, 2])

    def test_actual_outside_boundary_does_not_gain_query_membership(self):
        depth = torch.zeros(2, 360, 640, dtype=torch.float64, device='cuda')
        rays = self.g['rays']
        # Body-far endpoint is axial3.18m. Choose lateral-center lower rays;
        # sources at3.181/3.2m remain outside although radial bin is quantized.
        mask = (rays[..., 1].abs() < .03) & (rays[..., 2] < -.12) & (rays[..., 2] > -.20)
        depth[0, mask] = 3.181
        depth[1, mask] = 3.2
        result = self.check_packet(depth)
        self.assertGreater(result['source_counts'].sum().item(), 0)
        self.assertEqual(result['query_counts'].sum().item(), 0)

    def test_dense_mixed_surface_actual_query_counts(self):
        depth = torch.zeros(1, 360, 640, dtype=torch.float64, device='cuda')
        rays = self.g['rays']
        mask = self.g['crop'] & (rays[..., 1].abs() < .04) & (rays[..., 2] < -.10) & (rays[..., 2] > -.25)
        depth[0, mask] = 2.
        result = self.check_packet(depth)
        self.assertGreater(result['query_counts'][..., 1].sum().item(), 0)
        self.assertEqual(result['query_counts'][..., 0].sum().item(), 0)
        # Independent source count: every supported pixel in this one-bin-per
        # zone plane belongs to exactly one output subcell/echo.
        expected = 0
        for z in range(64):
            radial = depth[0]*self.g['radial_factor']
            values = radial[mask & (self.g['zone_ids'] == z)]
            ids = torch.bucketize(values, torch.linspace(0, 4, 41, dtype=torch.float64, device='cuda')[1:-1], right=True)
            sizes = torch.bincount(ids, minlength=40)
            supported = (sizes >= 3).nonzero().flatten()
            if len(supported):
                expected += sizes[supported[0]].item()
                if len(supported) > 1:
                    expected += sizes[supported[-1]].item()
        self.assertEqual(result['source_counts'].sum().item(), expected)

    def test_invalid_and_radial_limit(self):
        depth = torch.full((1, 360, 640), float('nan'), dtype=torch.float64, device='cuda')
        depth[0, :90] = 0
        depth[0, 90:180] = -1
        depth[0, 180:270] = 100
        depth[0, 270:] = 4.01
        result = self.check_packet(depth)
        self.assertFalse(result['valid'].any())
        self.assertEqual(result['source_counts'].sum().item(), 0)
        self.assertEqual(result['cell_known_counts'].sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
