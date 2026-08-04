#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_fresh_tf_r1a_local_validity import classify_cell, pose_matrix, zbuffer_winners


CONFIG = {
    "hard_ttl_ms": 750,
    "minimum_valid_projected_points_per_cell": 32,
    "minimum_occlusion_evidence_points_per_cell": 32,
    "minimum_supported_area_fraction": 0.60,
    "geometry_flow_warp_residual_px_max": 3.0,
}


def classify(**changes):
    values = {
        "age_ms": 100,
        "denominator": 100,
        "out_of_frame": 0,
        "occluded": 0,
        "visible": 80,
        "projected": 80,
        "flow_pass": 70,
        "median_warp_residual_px": 1.0,
        "config": CONFIG,
    }
    values.update(changes)
    return classify_cell(**values)


class FreshTfR1aTest(unittest.TestCase):
    def test_pose_identity(self) -> None:
        self.assertTrue(np.allclose(np.eye(4), pose_matrix([0, 0, 0, 0, 0, 0, 1])))

    def test_zbuffer_selects_nearest_then_stable_first(self) -> None:
        winners = zbuffer_winners(np.array([5, 5, 7, 7]), np.array([2.0, 1.0, 3.0, 3.0]))
        self.assertEqual([1, 2], winners.tolist())

    def test_supported(self) -> None:
        self.assertEqual("SUPPORTED", classify())

    def test_stale_has_first_precedence(self) -> None:
        self.assertEqual("STALE", classify(age_ms=751, out_of_frame=100, occluded=100))

    def test_out_of_frame_precedes_occlusion(self) -> None:
        self.assertEqual("OUT_OF_FRAME", classify(out_of_frame=40, occluded=100))

    def test_occlusion_precedes_newly_exposed(self) -> None:
        self.assertEqual("OCCLUDED", classify(occluded=32, visible=0))

    def test_newly_exposed_on_low_geometric_support(self) -> None:
        self.assertEqual("NEWLY_EXPOSED", classify(visible=59))

    def test_low_flow_support(self) -> None:
        self.assertEqual("LOW_FLOW_SUPPORT", classify(flow_pass=31))

    def test_high_warp_residual(self) -> None:
        self.assertEqual("HIGH_WARP_RESIDUAL", classify(median_warp_residual_px=3.01))


if __name__ == "__main__":
    unittest.main()
