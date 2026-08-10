#!/usr/bin/env python3

import unittest

import numpy as np

from train_ag_st_soft_boundary_bonn_canary import boundary_metrics, soft_boundary_target


class SoftBoundaryBonnCanaryTest(unittest.TestCase):
    def test_soft_target_expands_core_but_keeps_unknown_zero(self) -> None:
        probability = np.zeros((9, 9), dtype=np.float32)
        probability[4, 4] = 1.0
        valid = np.ones((9, 9), dtype=np.bool_)
        valid[:, 0] = False
        target = soft_boundary_target(probability, valid, sigma_px=2.0)
        self.assertEqual(1.0, float(target[4, 4]))
        self.assertGreater(float(target[4, 5]), 0.5)
        self.assertTrue(np.all(target[:, 0] == 0.0))

    def test_soft_target_without_core_does_not_invent_positive(self) -> None:
        probability = np.full((5, 5), 0.2, dtype=np.float32)
        valid = np.ones((5, 5), dtype=np.bool_)
        target = soft_boundary_target(probability, valid)
        np.testing.assert_allclose(probability, target)

    def test_boundary_metrics_rewards_correct_ranking(self) -> None:
        truth = np.zeros((5, 5), dtype=np.bool_)
        truth[2, 2] = True
        valid = np.ones_like(truth)
        score = np.full((5, 5), 0.01, dtype=np.float32)
        score[2, 2] = 0.99
        metrics = boundary_metrics([(score, truth, valid)], threshold=0.5, tolerance_px=0)
        self.assertEqual(1.0, metrics["student_average_precision"])
        self.assertEqual(1.0, metrics["f1_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
