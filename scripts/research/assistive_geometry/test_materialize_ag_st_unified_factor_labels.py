#!/usr/bin/env python3

import unittest

import numpy as np

from materialize_ag_st_unified_factor_labels import arrays_equal, merge_payload


class UnifiedFactorLabelsTest(unittest.TestCase):
    def test_merge_replaces_old_boundary_and_preserves_other_factor(self) -> None:
        base = {
            "metric_depth_m_hw": np.ones((2, 2), dtype=np.float32),
            "boundary_distance_px_hw": np.full((2, 2), 9.0, dtype=np.float32),
            "boundary_probability_pseudo_hw": np.zeros((2, 2), dtype=np.float32),
            "boundary_uncertainty_proxy_px_hw": np.ones((2, 2), dtype=np.float32),
        }
        boundary = {
            "boundary_core_probability_hw": np.eye(2, dtype=np.float32),
            "boundary_soft_probability_hw": np.eye(2, dtype=np.float32),
            "boundary_distance_px_hw": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            "boundary_uncertainty_px_hw": np.full((2, 2), 0.5, dtype=np.float32),
            "boundary_truth_valid_hw": np.ones((2, 2), dtype=np.uint8),
            "boundary_unknown_hw": np.zeros((2, 2), dtype=np.uint8),
            "boundary_quality_tier_hw": np.ones((2, 2), dtype=np.uint8),
            "boundary_provenance_hw": np.ones((2, 2), dtype=np.uint8),
        }
        merged = merge_payload(base, boundary)
        self.assertTrue(arrays_equal(merged["metric_depth_m_hw"], base["metric_depth_m_hw"]))
        self.assertTrue(arrays_equal(merged["boundary_distance_px_hw"], boundary["boundary_distance_px_hw"]))
        self.assertNotIn("boundary_probability_pseudo_hw", merged)
        self.assertIn("boundary_factor_valid_hw", merged)

    def test_arrays_equal_handles_matching_nan(self) -> None:
        left = np.asarray([np.nan, 1.0], dtype=np.float32)
        right = left.copy()
        self.assertTrue(arrays_equal(left, right))

    def test_arrays_equal_handles_string_metadata(self) -> None:
        left = np.asarray("dav2", dtype="<U4")
        self.assertTrue(arrays_equal(left, left.copy()))


if __name__ == "__main__":
    unittest.main()
