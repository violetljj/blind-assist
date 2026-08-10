#!/usr/bin/env python3

import unittest

import numpy as np

from run_ag_st_icl_fresh_depth_boundary_canary import (
    boundary_metrics,
    exact_depth_boundary_target,
)


class IclFreshDepthBoundaryCanaryTest(unittest.TestCase):
    def test_exact_metric_gap_defines_both_boundary_sides(self) -> None:
        depth = np.ones((5, 5), dtype=np.float32)
        depth[:, 3:] = 1.20
        valid = np.ones_like(depth, dtype=np.bool_)
        intrinsics = np.asarray(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        target, evaluable = exact_depth_boundary_target(depth, valid, intrinsics)
        self.assertTrue(np.all(target[:, 2:4]))
        self.assertFalse(np.any(target[:, :2]))
        self.assertFalse(np.any(target[:, 4:]))
        self.assertTrue(np.all(evaluable))

    def test_metric_ignores_predictions_outside_exact_evaluability(self) -> None:
        target = np.zeros((7, 7), dtype=np.bool_)
        target[3, 3] = True
        predicted = np.zeros_like(target)
        predicted[3, 4] = True
        predicted[0, 0] = True
        evaluable = np.zeros_like(target)
        evaluable[1:6, 1:6] = True
        metrics = boundary_metrics(predicted, target, evaluable)
        self.assertEqual(1, metrics["predicted_seed_pixels"])
        self.assertEqual(1.0, metrics["precision_within_2px"])
        self.assertEqual(1.0, metrics["recall_within_2px"])


if __name__ == "__main__":
    unittest.main()
