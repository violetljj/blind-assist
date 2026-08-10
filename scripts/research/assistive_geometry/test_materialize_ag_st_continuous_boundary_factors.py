#!/usr/bin/env python3

import unittest

import numpy as np

from materialize_ag_st_continuous_boundary_factors import continuous_boundary_factors


class ContinuousBoundaryFactorsTest(unittest.TestCase):
    def test_distance_soft_probability_and_unknown(self) -> None:
        probability = np.zeros((7, 7), dtype=np.float32)
        probability[3, 3] = 1.0
        valid = np.ones((7, 7), dtype=np.bool_)
        valid[:, 0] = False
        distance, soft = continuous_boundary_factors(probability, valid, sigma_px=2.0)
        self.assertEqual(0.0, float(distance[3, 3]))
        self.assertEqual(1.0, float(distance[3, 4]))
        self.assertGreater(float(soft[3, 4]), 0.5)
        self.assertTrue(np.isnan(distance[:, 0]).all())
        self.assertTrue(np.all(soft[:, 0] == 0.0))

    def test_empty_core_is_clipped_far_not_positive(self) -> None:
        probability = np.zeros((5, 5), dtype=np.float32)
        valid = np.ones((5, 5), dtype=np.bool_)
        distance, soft = continuous_boundary_factors(probability, valid, max_distance_px=8.0)
        self.assertTrue(np.all(distance == 8.0))
        self.assertTrue(np.all(soft < 0.05))


if __name__ == "__main__":
    unittest.main()
