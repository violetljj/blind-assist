"""Focused CPU checks for the frozen V2 mask, weight source, and output shapes."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from cnh_rgb_alley_v2 import (
    AzimuthMaskedFrustumFusion, decision, masked_attention, query_azimuth_mask,
    train_pos_weight,
)
from cnh_rgb_fusion import FrustumFusion


class AlleyV2Test(unittest.TestCase):
    def test_azimuth_mask_exact_columns(self):
        mask = query_azimuth_mask()
        self.assertEqual(mask.shape, (6, 64))
        self.assertEqual(mask.sum(-1).tolist(), [24, 24, 16, 16, 24, 24])
        for query, allowed in enumerate(({0, 1, 2}, {0, 1, 2}, {3, 4}, {3, 4}, {5, 6, 7}, {5, 6, 7})):
            for zone in range(64):
                self.assertEqual(bool(mask[query, zone]), zone % 8 in allowed)
        self.assertFalse(bool((mask[0] & mask[2]).any()))
        self.assertFalse(bool((mask[2] & mask[4]).any()))

    def test_attention_forbidden_zones_have_zero_weight_and_gradient(self):
        torch.manual_seed(3)
        zones = torch.randn(2, 64, 64, requires_grad=True)
        query = torch.randn(6, 64)
        mask = query_azimuth_mask()
        weights = masked_attention(query, zones, mask)
        self.assertEqual(weights.shape, (2, 6, 64))
        self.assertTrue(torch.equal(weights[:, ~mask], torch.zeros_like(weights[:, ~mask])))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 6)))
        decoded_left = torch.einsum('bz,bzd->bd', weights[:, 0], zones).sum()
        decoded_left.backward()
        self.assertTrue(torch.equal(zones.grad[:, ~mask[0]], torch.zeros_like(zones.grad[:, ~mask[0]])))

    def test_model_forward_shapes_mask_and_parameter_parity(self):
        model = AzimuthMaskedFrustumFusion().eval()
        base = FrustumFusion(64, 16, width=64, queries=6)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in base.parameters()))
        batch = 2
        rgb = torch.rand(batch, 3, 72, 128)
        histogram = torch.randn(batch, 64, 16)
        ambient = torch.rand(batch, 64)
        scalar = torch.full((batch, 64), 2.0)
        valid = torch.ones(batch, 64, dtype=torch.bool)
        age = torch.zeros(batch, 64)
        support = torch.ones(64, 18, 32)
        with torch.no_grad():
            out = model(rgb, histogram, ambient, scalar, valid, age, support)
        self.assertEqual(out['occupancy_logits'].shape, (batch, 6))
        self.assertEqual(out['distance_m'].shape, (batch, 6))
        self.assertEqual(out['query_zone_weights'].shape, (batch, 6, 64))
        self.assertTrue(all(torch.isfinite(value).all() for value in out.values()))
        mask = query_azimuth_mask()
        self.assertTrue(torch.equal(out['query_zone_weights'][:, ~mask],
                                    torch.zeros_like(out['query_zone_weights'][:, ~mask])))

    def test_class_weight_only_uses_known_training_labels(self):
        labels = np.array([[1, 0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0, 1],
                           [0, 0, -1, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1]], dtype=np.int8)
        train = np.array([True, True, True, False])
        weight, counts = train_pos_weight(labels, train)
        self.assertEqual(counts['positive'], [1, 1, 1, 1, 1, 1])
        self.assertEqual(counts['negative'], [2, 2, 1, 2, 2, 2])
        self.assertEqual(counts['unknown'], [0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(weight, [2, 2, 1, 2, 2, 2])
        labels[~train] = -1
        changed, _ = train_pos_weight(labels, train)
        np.testing.assert_array_equal(weight, changed)
        labels[train, 0] = 0
        with self.assertRaisesRegex(ValueError, 'both known train classes'):
            train_pos_weight(labels, train)

    def test_decision_requires_nondegenerate_paired_dominance(self):
        runs = [dict(seed=seed, arm=arm, dev=dict(tp=2 if arm == 'cnh_rgb' else 1,
                                                 fp=1, tn=9))
                for seed in (20260924, 20260925, 20260926)
                for arm in ('tof_only', 'cnh_rgb')]
        self.assertEqual(decision(runs)['rgb_increment'], 'CONSISTENT_DEV_PARETO_SIGNAL_ONLY')
        runs[0]['dev']['tp'] = 0
        self.assertEqual(decision(runs)['mechanism'], 'COLLAPSED_IN_AT_LEAST_ONE_ARM_SEED')
        self.assertEqual(decision(runs)['rgb_increment'], 'NO_CONSISTENT_DEV_PARETO_SIGNAL')


if __name__ == '__main__':
    unittest.main()
