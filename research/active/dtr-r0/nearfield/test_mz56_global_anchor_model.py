"""Synthetic CPU checks; no real checkpoint, dataset or native-depth access."""
import inspect
import unittest

import torch
from torch import nn

from mz54_full_rgb_model import RasterQuery, loss_for as raster_loss
from mz56_global_anchor_model import AnchorQuery, loss_for


def reference():
    torch.manual_seed(54)
    local = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
    head = nn.Sequential(nn.Linear(40, 32), nn.ReLU(), nn.Linear(32, 4))
    state = {**{'local.' + k: v for k, v in local.state_dict().items()},
             **{'head.' + k: v for k, v in head.state_dict().items()}}
    model = RasterQuery(state, 'FULL_RASTER')
    # Mimic a learned MZ54 coverage column; bridging must not erase it.
    with torch.no_grad():
        model.head[0].weight[:, 40].fill_(.3)
    return model


class AnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self):
        base = reference()
        torch.manual_seed(151)
        local = AnchorQuery(base.state_dict(), 'LOCAL_ONLY')
        torch.manual_seed(151)
        global_model = AnchorQuery(base.state_dict(), 'GLOBAL_ANCHOR')
        return base, local, global_model

    @staticmethod
    def activate_global(model):
        with torch.no_grad():
            model.packet_encoder[0].weight.fill_(.1)
            model.packet_encoder[0].bias.fill_(1.)
            model.head[0].weight[:, 41:].fill_(.125)
            model.head[0].bias.fill_(2.)
            model.head[2].weight.fill_(.2)

    def test_parameter_budget_identical_initialization_and_geometry(self):
        base, local, global_model = self.pair()
        self.assertEqual(sum(p.numel() for p in local.parameters()), 11020)
        self.assertEqual(sum(p.numel() for p in local.packet_encoder.parameters()), 56)
        for (ka, va), (kb, vb) in zip(local.named_parameters(), global_model.named_parameters()):
            self.assertEqual(ka, kb)
            torch.testing.assert_close(va, vb, rtol=0, atol=0)
        torch.testing.assert_close(local.head[0].weight[:, :41], base.head[0].weight, rtol=0, atol=0)
        self.assertFalse(local.head[0].weight[:, 41:].any())
        for name, value in base.named_buffers():
            torch.testing.assert_close(getattr(local, name), value, rtol=0, atol=0)
        self.assertEqual(int(local.candidate_mask.sum()), 3600)
        self.assertEqual(int(local.crop_mask.sum()), 756)
        torch.testing.assert_close(local.zone_angles[0], torch.tensor([-.875, .875]), rtol=0, atol=0)
        torch.testing.assert_close(local.zone_angles[-1], torch.tensor([.875, -.875]), rtol=0, atol=0)

    def test_initial_logits_replay_mz54_including_learned_coverage_column(self):
        base, local, global_model = self.pair()
        features = torch.randn(2, 64, 45, 80)
        ranges = torch.linspace(.1, 4., 256).reshape(2, 64, 2)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        valid[0, ::3] = False
        valid[1] = False
        expected = base.inspect(features, ranges, valid)
        for model in (local, global_model):
            actual = model.inspect(features, ranges, valid)
            torch.testing.assert_close(actual['field'], expected['field'], atol=2e-5, rtol=1e-6)
            for name in ('OPEN', 'CROP'):
                torch.testing.assert_close(actual[name]['raw'], expected[name]['raw'], atol=2e-5, rtol=1e-6)
                torch.testing.assert_close(actual[name]['support'], expected[name]['support'], rtol=0, atol=0)
            torch.testing.assert_close(actual['sensor_coverage'], expected['sensor_coverage'], rtol=0, atol=0)
        self.assertFalse(local.inspect(features, ranges, valid)['anchor_vector'].any())

    def test_masked_mean_uses_valid_zones_not_slot_count(self):
        _, _, model = self.pair()
        ranges = torch.full((1, 64, 2), float('nan'))
        valid = torch.zeros_like(ranges, dtype=torch.bool)
        ranges[0, 0] = torch.tensor([1., 2.]); valid[0, 0] = True
        ranges[0, 63] = torch.tensor([3., 99.]); valid[0, 63] = True
        actual, available = model.encode_anchor(ranges, valid)
        inputs = torch.tensor([[.25, .5, 1., 1., -.875, .875],
                               [.75, 0., 1., 0., .875, -.875]])
        expected = model.packet_encoder(inputs).mean(0)
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-7)
        self.assertTrue(available.item())
        ranges[:, 1:63] = -1000.
        changed, _ = model.encode_anchor(ranges, valid)
        torch.testing.assert_close(actual, changed, rtol=0, atol=0)

    def test_all_missing_is_zero_even_with_learned_bias(self):
        _, _, model = self.pair()
        self.activate_global(model)
        with torch.no_grad():
            model.packet_encoder[0].bias.fill_(7.)
        ranges = torch.tensor([float('nan'), float('inf'), -1., 0., 4.1, 90., -.3, 0.]).repeat(16).reshape(1, 64, 2)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        anchor, available = model.encode_anchor(ranges, valid)
        self.assertFalse(anchor.any()); self.assertFalse(available.any())
        features = torch.randn(1, 64, 45, 80)
        a = model.inspect(features, ranges, valid)
        b = model.inspect(features, ranges, valid, suppress_global=True)
        torch.testing.assert_close(a['field'], b['field'], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a['field']).all())

    def test_real_global_path_changes_outside_logits_without_fake_local_packet(self):
        _, local, model = self.pair()
        self.activate_global(model)
        local.load_state_dict(model.state_dict())
        features = torch.randn(1, 64, 45, 80)
        ranges = torch.ones(1, 64, 2)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        outside = ~model.sensor_coverage
        a = model.inspect(features, ranges, valid)
        b = model.inspect(features, 3 * ranges, valid)
        self.assertTrue((a['field'][:, outside] != b['field'][:, outside]).any())
        x = local.inspect(features, ranges, valid)
        y = local.inspect(features, 3 * ranges, valid)
        torch.testing.assert_close(x['field'][:, outside], y['field'][:, outside], rtol=0, atol=0)
        self.assertFalse(model.packet_features(ranges, valid)[0][:, outside].any())
        self.assertFalse(a['sensor_coverage'][outside].any())
        self.assertTrue(a['anchor_available'].all())
        self.assertTrue(a['anchor_vector'].abs().sum() > 0)
        self.assertFalse(x['anchor_vector'].any())

    def test_suppression_reproduces_local_path_with_identical_learned_weights(self):
        _, local, model = self.pair()
        self.activate_global(model)
        local.load_state_dict(model.state_dict())
        features = torch.randn(1, 64, 45, 80)
        ranges = torch.ones(1, 64, 2)
        valid = torch.ones_like(ranges, dtype=torch.bool)
        suppressed = model.inspect(features, ranges, valid, suppress_global=True)
        reference_out = local.inspect(features, ranges, valid)
        torch.testing.assert_close(suppressed['field'], reference_out['field'], rtol=0, atol=0)
        torch.testing.assert_close(suppressed['observed_anchor'], reference_out['observed_anchor'], rtol=0, atol=0)
        self.assertFalse(suppressed['anchor_vector'].any())

    def test_global_gradients_and_loss_contract(self):
        _, _, model = self.pair()
        self.activate_global(model)
        features = torch.randn(1, 64, 45, 80)
        ranges = torch.ones(1, 64, 2)
        out = model.inspect(features, ranges, torch.ones_like(ranges, dtype=torch.bool))
        target = torch.zeros(1, 45, 80, 4, dtype=torch.bool)
        loss, _ = loss_for(out, target, torch.ones(1, 45, 80, dtype=torch.bool),
                           torch.zeros(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool))
        loss.backward()
        self.assertTrue(model.packet_encoder[0].weight.grad.abs().sum() > 0)
        self.assertTrue(model.head[0].weight.grad[:, 41:].abs().sum() > 0)
        self.assertIs(loss_for, raster_loss)
        self.assertEqual(list(inspect.signature(model.inspect).parameters),
                         ['normalized_features', 'ranges', 'valid', 'suppress_global'])

    def test_reject_changed_geometry_and_invalid_arm(self):
        state = reference().state_dict()
        state['candidate_mask'] = torch.zeros_like(state['candidate_mask'])
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            AnchorQuery(state, 'GLOBAL_ANCHOR')
        with self.assertRaises(ValueError):
            AnchorQuery(reference().state_dict(), 'FAKE_OUTSIDE_RANGE')


if __name__ == '__main__':
    unittest.main()
