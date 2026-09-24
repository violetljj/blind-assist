"""Focused CPU checks before the frozen pretrained alley GPU control."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from cnh_rgb_alley_pretrained_control import FrozenResNet18Stem, PretrainedAzimuthFusion
from cnh_rgb_alley_v2 import query_azimuth_mask


WEIGHTS = Path(r'C:\Users\26442\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth')


class PretrainedControlTest(unittest.TestCase):
    def test_pretrained_stem_is_frozen_eval_and_keeps_frustum_grid(self):
        encoder = FrozenResNet18Stem(WEIGHTS)
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not module.training for module in encoder.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        with torch.no_grad():
            features = encoder(torch.zeros(2, 3, 72, 128))
        self.assertEqual(features.shape, (2, 64, 18, 32))
        self.assertFalse(features.requires_grad)
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)

    def test_adapter_and_query_path_are_trainable_with_fixed_azimuth(self):
        torch.manual_seed(17)
        model = PretrainedAzimuthFusion().train()
        features = torch.randn(2, 64, 18, 32)
        histogram = torch.randn(2, 64, 16)
        ambient = torch.ones(2, 64)
        scalar = torch.ones(2, 64)
        valid = torch.ones(2, 64, dtype=torch.bool)
        age = torch.zeros(2, 64)
        support = torch.ones(64, 18, 32)
        output = model(features, histogram, ambient, scalar, valid, age, support)
        self.assertEqual(output['occupancy_logits'].shape, (2, 6))
        mask = query_azimuth_mask()
        self.assertTrue(torch.equal(output['query_zone_weights'][:, ~mask],
                                    torch.zeros_like(output['query_zone_weights'][:, ~mask])))
        output['occupancy_logits'].sum().backward()
        self.assertIsNotNone(model.rgb[0].weight.grad)
        self.assertTrue(torch.isfinite(model.rgb[0].weight.grad).all())
        self.assertGreater(float(model.rgb[0].weight.grad.abs().sum()), 0)
        self.assertFalse(features.requires_grad)


if __name__ == '__main__':
    unittest.main()
