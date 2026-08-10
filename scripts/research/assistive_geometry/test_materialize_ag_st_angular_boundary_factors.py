#!/usr/bin/env python3

import unittest

import numpy as np

from materialize_ag_st_angular_boundary_factors import add_angular_fields


class AngularBoundaryFactorsTest(unittest.TestCase):
    def test_adds_parallel_fields_without_changing_base(self) -> None:
        base = {
            "boundary_core_probability_hw": np.zeros((7, 7), dtype=np.float32),
            "boundary_truth_valid_hw": np.ones((7, 7), dtype=np.uint8),
            "boundary_quality_tier_hw": np.ones((7, 7), dtype=np.uint8),
        }
        base["boundary_core_probability_hw"][3, 3] = 1.0
        before = {key: value.copy() for key, value in base.items()}
        intrinsics = np.asarray(
            [[20.0, 0.0, 3.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]]
        )
        output = add_angular_fields(base, intrinsics)
        for key, value in before.items():
            np.testing.assert_array_equal(output[key], value)
        self.assertIn("boundary_angular_distance_rad_hw", output)
        self.assertIn("boundary_angular_soft_probability_hw", output)
        self.assertEqual(0.0, float(output["boundary_angular_distance_rad_hw"][3, 3]))

    def test_invalid_pixels_remain_unknown(self) -> None:
        probability = np.zeros((5, 5), dtype=np.float32)
        probability[2, 2] = 1.0
        valid = np.ones((5, 5), dtype=np.uint8)
        valid[:, 0] = 0
        output = add_angular_fields(
            {
                "boundary_core_probability_hw": probability,
                "boundary_truth_valid_hw": valid,
            },
            np.asarray([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]),
        )
        self.assertTrue(
            np.isnan(output["boundary_angular_distance_rad_hw"][:, 0]).all()
        )
        self.assertTrue(
            np.all(output["boundary_angular_soft_probability_hw"][:, 0] == 0.0)
        )


if __name__ == "__main__":
    unittest.main()
