#!/usr/bin/env python3

import unittest

import numpy as np

from run_ag_st_angular_boundary_resize_canary import (
    camera_angular_boundary_factors,
    line_case,
    scaled_intrinsics,
)


class AngularBoundaryResizeCanaryTest(unittest.TestCase):
    def test_scaled_intrinsics_preserve_homogeneous_row(self) -> None:
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
        )
        output = scaled_intrinsics(intrinsics, 2, 3)
        np.testing.assert_array_equal(output[2], intrinsics[2])
        self.assertEqual((200.0, 360.0), (output[0, 0], output[1, 1]))

    def test_angular_factor_preserves_unknown(self) -> None:
        probability = np.zeros((5, 5), dtype=np.float32)
        probability[2, 2] = 1.0
        valid = np.ones((5, 5), dtype=np.bool_)
        valid[0] = False
        k = np.asarray([[5.0, 0.0, 2.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])
        angle, soft = camera_angular_boundary_factors(probability, valid, k)
        self.assertTrue(np.isnan(angle[0]).all())
        self.assertTrue(np.all(soft[0] == 0.0))
        self.assertEqual(0.0, float(angle[2, 2]))

    def test_anisotropic_resize_preserves_ray_angle_not_pixel_distance(self) -> None:
        for orientation in ("vertical", "horizontal"):
            row = line_case(orientation, scale_x=2, scale_y=3)
            self.assertLessEqual(row["max_angular_distance_abs_error_rad"], 1e-6)
            self.assertLessEqual(row["max_angular_soft_abs_error"], 1e-6)
            self.assertFalse(row["raw_pixel_distance_invariant"])


if __name__ == "__main__":
    unittest.main()
