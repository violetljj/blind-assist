#!/usr/bin/env python3

import unittest

import numpy as np
from evaluate_calibration_head import fit_ridge_head, predict_head
from evaluate_dense_propagation import flow_consistency_mask, propagate_residual


class DensePropagationTest(unittest.TestCase):
    def test_zero_bidirectional_flow_is_consistent(self):
        zero = np.zeros((2, 4, 5), dtype=np.float32)
        valid, map_x, map_y = flow_consistency_mask(zero, zero, 1.5)
        self.assertTrue(np.all(valid))
        self.assertEqual(map_x.shape, (4, 5))
        self.assertEqual(map_y.shape, (4, 5))

    def test_inconsistent_flow_is_rejected(self):
        backward = np.zeros((2, 4, 5), dtype=np.float32)
        forward = np.zeros_like(backward)
        forward[0] = 3.0
        valid, _, _ = flow_consistency_mask(backward, forward, 1.5)
        self.assertFalse(np.any(valid))

    def test_identity_warp_reconstructs_anchor_metric(self):
        fast = np.linspace(0.5, 2.0, 20, dtype=np.float32).reshape(4, 5)
        metric = 1.2 * fast + 0.1
        zero = np.zeros((2, 4, 5), dtype=np.float32)
        output, diagnostics = propagate_residual(
            fast,
            metric,
            fast,
            zero,
            zero,
            {"status": "VALID", "slope": 1.2, "intercept_m": 0.1},
            {
                "forward_backward_consistency_px_max": 1.5,
                "minimum_consistent_residual_coverage": 0.5,
            },
        )
        self.assertEqual(diagnostics["status"], "VALID")
        np.testing.assert_allclose(output, metric, atol=1e-6)


class CalibrationHeadTest(unittest.TestCase):
    def test_multioutput_ridge_is_deterministic_and_predictive(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(80, 6))
        truth_kernel = rng.normal(size=(6, 2))
        targets = features @ truth_kernel + np.asarray([1.0, -0.2])
        first = fit_ridge_head(features, targets, 1e-6)
        second = fit_ridge_head(features, targets, 1e-6)
        np.testing.assert_array_equal(first["kernel"], second["kernel"])
        prediction = predict_head(first, features)
        self.assertLess(float(np.mean(np.abs(prediction - targets))), 1e-5)


if __name__ == "__main__":
    unittest.main()
