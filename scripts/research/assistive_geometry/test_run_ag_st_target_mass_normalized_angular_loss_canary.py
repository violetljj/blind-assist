#!/usr/bin/env python3

import unittest

import numpy as np

from run_ag_st_target_mass_normalized_angular_loss_canary import gradient_summary


class TargetMassNormalizedAngularLossCanaryTest(unittest.TestCase):
    def test_binary_density_does_not_change_component_mass(self) -> None:
        totals = []
        for positive_pixels in (1, 50):
            target = np.zeros((10, 10), dtype=np.float32)
            target.flat[:positive_pixels] = 1.0
            valid = np.ones_like(target, dtype=np.bool_)
            valid[-1, -1] = False
            tier = np.full_like(target, 3, dtype=np.uint8)
            tier[-1, -1] = 0
            row = gradient_summary(
                target,
                valid,
                tier,
                loss_profile="target_mass_normalized_bce",
            )
            self.assertTrue(row["finite"])
            self.assertEqual(row["unknown_gradient_max_abs"], 0.0)
            self.assertAlmostEqual(row["dual_class_component_balance_ratio"], 1.0, places=6)
            totals.append(row["component_gradient_mass_total"])
        self.assertAlmostEqual(totals[0], 0.5, places=6)
        self.assertAlmostEqual(totals[1], 0.5, places=6)

    def test_empty_positive_class_uses_bounded_single_class_fallback(self) -> None:
        target = np.zeros((4, 5), dtype=np.float32)
        valid = np.ones_like(target, dtype=np.bool_)
        tier = np.full_like(target, 3, dtype=np.uint8)
        row = gradient_summary(
            target,
            valid,
            tier,
            loss_profile="target_mass_normalized_bce",
        )
        self.assertFalse(row["has_positive_target_mass"])
        self.assertTrue(row["has_negative_target_mass"])
        self.assertAlmostEqual(row["component_gradient_mass_total"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
