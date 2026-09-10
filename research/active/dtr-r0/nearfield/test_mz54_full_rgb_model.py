"""Synthetic CPU checks only: no checkpoint, RGB file or native depth access."""
import math
import unittest

import torch
from torch import nn

from mz54_full_rgb_model import RasterQuery, loss_for


def initial_state():
    torch.manual_seed(54)
    local = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
    head = nn.Sequential(nn.Linear(40, 32), nn.ReLU(), nn.Linear(32, 4))
    return {**{'local.' + k: v for k, v in local.state_dict().items()},
            **{'head.' + k: v for k, v in head.state_dict().items()}}


class RasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, arm='FULL_RASTER'):
        return RasterQuery(initial_state(), arm)

    def test_identical_initial_parameters_and_input_state_preserved(self):
        state = initial_state()
        before = {k: v.clone() for k, v in state.items()}
        a, b = (RasterQuery(state, arm) for arm in ('CROP_RASTER', 'FULL_RASTER'))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 10708)
        for (ka, va), (kb, vb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(ka, kb)
            torch.testing.assert_close(va, vb, rtol=0, atol=0)
        for k in state:
            torch.testing.assert_close(state[k], before[k], rtol=0, atol=0)
        torch.testing.assert_close(a.head[0].weight[:, :40], state['head.0.weight'], rtol=0, atol=0)
        self.assertFalse(a.head[0].weight[:, 40].any())
        with self.assertRaises(ValueError):
            RasterQuery(state, 'SQUARE_RESIZE')

    def test_geometry_against_scalar_native_centers(self):
        a, b = self.model('CROP_RASTER'), self.model()
        focal = 320 / math.tan(math.radians(50))
        for r in range(45):
            for c in range(80):
                u, v = 8 * c + 3.5, 8 * r + 3.5
                az = math.degrees(math.atan((u - 319.5) / focal))
                el = math.degrees(math.atan((179.5 - v) / focal))
                covered = abs(az) <= 22.5 and abs(el) <= 22.5
                self.assertEqual(bool(b.sensor_coverage[r, c]), covered)
                expected_zone = math.floor((22.5 - el) / 5.625) * 8 + math.floor((az + 22.5) / 5.625) if covered else -1
                self.assertEqual(int(b.sensor_zone[r, c]), expected_zone)
                for model, width, height, x0, y0 in [(a, 224, 224, 208, 68), (b, 640, 360, 0, 0)]:
                    gx, gy = model.grid[r, c].tolist()
                    self.assertAlmostEqual((gx + 1) * width / 2 - .5 + x0, u, delta=3e-5)
                    self.assertAlmostEqual((gy + 1) * height / 2 - .5 + y0, v, delta=3e-5)
                for j, angle in enumerate(((az + 22.5) / 5.625, (22.5 - el) / 5.625)):
                    self.assertAlmostEqual(float(b.relative[r, c, j]), 2 * (angle % 1) - 1, delta=1e-6)
                contained = 8*c >= 208 and 8*(c+1) <= 432 and 8*r >= 68 and 8*(r+1) <= 292
                self.assertEqual(bool(a.candidate_mask[r, c]), contained)
        self.assertEqual(int(a.candidate_mask.sum()), 756)
        self.assertEqual(int(b.candidate_mask.sum()), 3600)
        self.assertFalse(a.crop_mask[8].any())
        self.assertFalse(a.crop_mask[36].any())
        torch.testing.assert_close(a.absolute, b.absolute, rtol=0, atol=0)
        torch.testing.assert_close(a.rays, b.rays, rtol=0, atol=0)
        torch.testing.assert_close(b.rays.norm(dim=-1), torch.ones(45, 80), atol=1e-6, rtol=0)
        self.assertTrue((b.absolute.abs() > 1).any())

    def test_invalid_and_outside_packets_do_not_leak(self):
        model = self.model()
        ranges = torch.tensor([0., -1., float('nan'), float('inf'), 4.01, 4., 1., 2.]).repeat(16).reshape(1, 64, 2)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        valid[:, 3::4] = False
        actual, available = model.packet_features(ranges, valid)
        self.assertTrue(available.item())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(actual[:, ~model.sensor_coverage].any())
        for r, c in model.sensor_coverage.nonzero().tolist():
            z = int(model.sensor_zone[r, c])
            expected = []
            flags = []
            for k in range(2):
                x = float(ranges[0, z, k])
                good = bool(valid[0, z, k]) and math.isfinite(x) and 0 < x <= 4
                expected.append(x / 4 if good else 0.)
                flags.append(float(good))
            torch.testing.assert_close(actual[0, r, c], torch.tensor(expected + flags), rtol=0, atol=0)
        changed = torch.where(valid, ranges, torch.full_like(ranges, float('nan')))
        torch.testing.assert_close(actual, model.packet_features(changed, valid)[0], rtol=0, atol=0)

    def test_grid_sample_reconstructs_native_coordinate_plane(self):
        for arm, height, width, x0, y0 in [('FULL_RASTER', 45, 80, 0, 0),
                                           ('CROP_RASTER', 28, 28, 208, 68)]:
            model = self.model(arm)
            yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            plane = (x0 + 8 * xx + 3.5) + 10 * (y0 + 8 * yy + 3.5)
            sampled = torch.nn.functional.grid_sample(plane[None, None].float(),
                                                       model.grid[None], align_corners=False)[0, 0]
            r, c = torch.meshgrid(torch.arange(45), torch.arange(80), indexing='ij')
            expected = (8 * c + 3.5) + 10 * (8 * r + 3.5)
            # Float32 grid_sample can round an edge center slightly into zero
            # padding; its relative error here is below one part per million.
            torch.testing.assert_close(sampled[model.candidate_mask], expected[model.candidate_mask],
                                       atol=5e-4, rtol=1e-6)

    def test_outside_logits_invariant_to_all_packet_changes(self):
        model = self.model()
        x = torch.randn(2, 64, 45, 80)
        r = torch.ones(2, 64, 2)
        a = model.inspect(x, r, torch.ones_like(r, dtype=torch.bool))
        b = model.inspect(x, r * 3, torch.zeros_like(r, dtype=torch.bool))
        outside = ~model.sensor_coverage
        torch.testing.assert_close(a['field'][:, outside], b['field'][:, outside], rtol=0, atol=0)
        self.assertTrue((a['field'][:, model.sensor_coverage] != b['field'][:, model.sensor_coverage]).any())
        self.assertEqual(tuple(a['field'].shape), (2, 45, 80, 4))
        self.assertEqual(tuple(a['OPEN']['raw'].shape), (2, 4))
        self.assertTrue(a['OPEN']['support'].all())

    def test_masked_context_means_and_head_feature_order(self):
        model = self.model('CROP_RASTER')
        x = torch.randn(1, 64, 28, 28)
        r = torch.ones(1, 64, 2)
        captured = []
        hook = model.head[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            model.inspect(x, r, torch.ones_like(r, dtype=torch.bool))
        finally:
            hook.remove()
        h = captured[0]
        self.assertEqual(tuple(h.shape), (1, 45, 80, 41))
        flat_tokens = h[0, :, :, :16].reshape(3600, 16)
        for k in torch.unique(model.context_ids):
            positions = model.context_ids == k
            observed = positions & model.candidate_mask.flatten()
            expected = flat_tokens[observed].mean(0) if observed.any() else torch.zeros(16)
            actual = h[0, :, :, 16:32].reshape(3600, 16)[positions]
            torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(h[0, :, :, 32:34], model.absolute, rtol=0, atol=0)
        torch.testing.assert_close(h[0, :, :, 34:36], model.relative, rtol=0, atol=0)
        torch.testing.assert_close(h[0, :, :, 40], model.sensor_coverage.float(), rtol=0, atol=0)

    def test_crop_mask_blocks_output_and_loss_gradient(self):
        model = self.model('CROP_RASTER')
        x = torch.randn(1, 64, 28, 28, requires_grad=True)
        r = torch.ones(1, 64, 2)
        out = model.inspect(x, r, torch.ones_like(r, dtype=torch.bool))
        out['field'].retain_grad()
        truth = torch.zeros(1, 45, 80, 4, dtype=torch.bool)
        truth[:, ~model.crop_mask] = True
        loss, detail = loss_for(out, truth, torch.ones(1, 45, 80, dtype=torch.bool),
                                torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(detail['skipped_positive'].item(), 4)
        self.assertEqual(detail['query'].item(), 0.)
        loss.backward()
        self.assertFalse(out['field'][:, ~model.crop_mask].any())
        self.assertFalse(out['field'].grad[:, ~model.crop_mask].any())
        self.assertTrue(out['field'].grad[:, model.crop_mask].abs().sum() > 0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(model.local[0].weight.grad.abs().sum() > 0)
        self.assertTrue(model.head[0].weight.grad.abs().sum() > 0)

    def test_unknown_labels_have_no_loss_or_gradient(self):
        field = torch.zeros(1, 45, 80, 4, requires_grad=True)
        model = self.model()
        out = dict(field=field, candidate_mask=model.candidate_mask, OPEN=model.pool(field, model.candidate_mask))
        loss, detail = loss_for(out, torch.ones_like(field, dtype=torch.bool),
                                torch.zeros(1, 45, 80, dtype=torch.bool),
                                torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(detail['skipped_positive'].item(), 4)
        loss.backward()
        self.assertFalse(field.grad.any())

    def test_same_logits_crop_diagnostic_and_empty_support(self):
        model = self.model()
        field = torch.full((1, 45, 80, 4), 2.)
        field[:, ~model.crop_mask] = 100.
        self.assertTrue((model.pool(field, model.candidate_mask)['raw'] == 100).all())
        self.assertTrue((model.pool(field, model.crop_mask)['raw'] == 2).all())
        empty = model.pool(field, torch.zeros_like(model.crop_mask))
        self.assertFalse(empty['support'].any())
        self.assertTrue((empty['raw'] == -20).all())

    def test_known_witness_frame_loss_matches_bce(self):
        field = torch.zeros(1, 45, 80, 4, requires_grad=True)
        model = self.model()
        truth = torch.zeros_like(field, dtype=torch.bool)
        truth[0, 0, 0, 0] = True
        out = dict(field=field, candidate_mask=model.candidate_mask, OPEN=model.pool(field, model.candidate_mask))
        loss, details = loss_for(out, truth, torch.ones(1, 45, 80, dtype=torch.bool),
                                torch.tensor([[True, False, False, False]]), torch.ones(1, 4, dtype=torch.bool))
        self.assertAlmostEqual(details['local'].item(), math.log(2), places=6)
        self.assertAlmostEqual(details['query'].item(), math.log(2), places=6)
        self.assertAlmostEqual(loss.item(), 1.25 * math.log(2), places=6)
        self.assertEqual(details['skipped_positive'].item(), 0)


if __name__ == '__main__':
    unittest.main()
