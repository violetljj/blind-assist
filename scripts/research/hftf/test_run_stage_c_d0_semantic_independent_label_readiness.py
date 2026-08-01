from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d0_semantic_independent_label_readiness import (
    _fit_ground_plane,
    _formal_frame_indices,
    _structural_canaries,
)


PROFILE = {
    "direction_degrees": [-30, -15, 0, 15, 30],
    "section_center_distance_m": [1.4, 1.8, 2.2, 2.6, 3.0],
    "cell_forward_half_extent_m": 0.22,
    "cell_lateral_half_extent_m": 0.22,
    "candidate_surface_height_m_inclusive": [-0.5, 0.5],
    "height_histogram_bin_width_m": 0.04,
    "winning_mode_expansion_m_each_side": 0.04,
    "minimum_raw_points_per_section": 12,
    "minimum_mode_points_per_section": 12,
    "minimum_mode_fraction": 0.18,
    "minimum_support_normal_ground_alignment": 0.8660254038,
    "maximum_support_plane_p90_residual_m": 0.025,
    "minimum_known_sections_per_direction": 4,
    "rise_risk_if_adjacent_delta_m_strictly_greater_than": 0.18,
    "drop_risk_if_adjacent_delta_m_strictly_less_than": -0.15,
}

PLANE = {
    "minimum_candidate_inliers": 200,
    "maximum_fit_points": 8000,
    "ransac_iterations": 256,
    "minimum_triplet_cross_product_norm": 0.0001,
    "minimum_camera_y_axis_alignment": 0.6427876097,
    "maximum_camera_height_error_m_for_candidate": 0.35,
    "inlier_distance_m": 0.04,
}


class StageCD0SemanticIndependentLabelReadinessTest(unittest.TestCase):
    def test_formal_frame_selection_includes_last(self) -> None:
        self.assertEqual([0, 5, 10, 11], _formal_frame_indices(12))
        self.assertEqual([0, 5, 10], _formal_frame_indices(11))

    def test_all_structural_canaries_pass(self) -> None:
        result = _structural_canaries(PROFILE)
        self.assertEqual(7, len(result))
        self.assertTrue(all(result.values()), result)

    def test_ground_plane_fit_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.uniform(-3, 3, 3000)
        z = rng.uniform(1, 8, 3000)
        y = 1.3 + rng.normal(0, 0.005, 3000)
        points = np.column_stack([x, y, z])
        first = _fit_ground_plane(points, 1.3, "source", 9, PLANE)
        second = _fit_ground_plane(points, 1.3, "source", 9, PLANE)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        np.testing.assert_array_equal(first["normal"], second["normal"])
        self.assertEqual(first["offset"], second["offset"])
        self.assertGreater(first["inlier_fraction"], 0.99)

    def test_insufficient_points_remain_unknown(self) -> None:
        self.assertIsNone(
            _fit_ground_plane(
                np.zeros((199, 3)), 1.3, "source", 0, PLANE
            )
        )


if __name__ == "__main__":
    unittest.main()
