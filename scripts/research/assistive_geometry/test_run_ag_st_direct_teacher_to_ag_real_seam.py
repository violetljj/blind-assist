#!/usr/bin/env python3
"""Focused invariants for the direct SuperTeacher-to-AG seam."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .run_ag_st_direct_teacher_to_ag_real_seam import (
        block_depth,
        factor_identity,
        nearest_completion,
        scaled_intrinsics,
    )
except ImportError:
    from run_ag_st_direct_teacher_to_ag_real_seam import (
        block_depth,
        factor_identity,
        nearest_completion,
        scaled_intrinsics,
    )


class DirectTeacherSeamTests(unittest.TestCase):
    def test_block_depth_retains_source_median_and_marks_empty_block(self) -> None:
        depth = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        valid = np.ones((4, 4), dtype=np.bool_)
        valid[:2, :2] = False
        reduced, available = block_depth(depth, valid, 2)
        self.assertEqual(reduced.shape, (2, 2))
        self.assertFalse(bool(available[0, 0]))
        self.assertAlmostEqual(float(reduced[0, 1]), 5.5)
        self.assertTrue(bool(available[1, 1]))

    def test_nearest_completion_preserves_observed_and_fills_unknown(self) -> None:
        depth = np.asarray([[1.0, np.nan, np.nan], [2.0, np.nan, 4.0]], dtype=np.float32)
        valid = np.isfinite(depth)
        completed, distance = nearest_completion(depth, valid)
        np.testing.assert_array_equal(completed[valid], depth[valid])
        self.assertTrue(bool(np.isfinite(completed).all()))
        self.assertTrue(bool((completed > 0.0).all()))
        self.assertTrue(bool((distance[~valid] > 0.0).all()))

    def test_scaled_intrinsics_preserve_pixel_center_projection(self) -> None:
        intrinsics = np.asarray(
            [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        scaled = scaled_intrinsics(intrinsics, 4)
        point = np.asarray([0.31, -0.17, 2.4], dtype=np.float64)
        source = intrinsics @ point
        source = source[:2] / source[2]
        target = scaled @ point
        target = target[:2] / target[2]
        expected = (source + 0.5) / 4.0 - 0.5
        np.testing.assert_allclose(target, expected, atol=1.0e-12)

    def test_factor_identity_forbids_learned_final_task_head(self) -> None:
        identity = factor_identity("A" * 64)
        self.assertIs(identity["learned_final_task_head"], False)
        self.assertEqual(identity["metric_depth_tier"], "A_SOURCE_NATIVE")
        self.assertEqual(identity["completion_tier"], "C_CONSERVATIVE_GEOMETRY_PSEUDO_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
