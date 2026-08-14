from __future__ import annotations

import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_pose_constrained_candidate_monocular_geometry as subject,
)


class PoseConstrainedCandidateMonocularGeometryTest(unittest.TestCase):
    def test_robust_affine_alignment_recovers_scale_shift_with_outliers(self) -> None:
        candidate = np.linspace(1.0, 5.0, 400, dtype=np.float64).reshape(20, 20)
        reference = 1.2 * candidate - 0.3
        reference.reshape(-1)[::5] += 4.0
        scale, shift, aligned, receipt = subject.robust_affine_alignment(candidate, reference)
        self.assertTrue(receipt["alignment_evaluable"])
        self.assertAlmostEqual(1.2, scale, places=6)
        self.assertAlmostEqual(-0.3, shift, places=6)
        self.assertGreater(receipt["alignment_inlier_fraction"], 0.7)
        np.testing.assert_allclose(aligned, 1.2 * candidate - 0.3, atol=1e-6)

    def test_insufficient_overlap_fails_closed_to_identity_and_zero_confidence_support(self) -> None:
        candidate = np.ones((12, 12), dtype=np.float64)
        reference = np.full((12, 12), np.inf, dtype=np.float64)
        reference.reshape(-1)[: subject.MINIMUM_ALIGNMENT_PIXELS - 1] = 1.0
        scale, shift, aligned, receipt = subject.robust_affine_alignment(candidate, reference)
        self.assertFalse(receipt["alignment_evaluable"])
        self.assertEqual(0, receipt["alignment_inlier_pixel_count"])
        self.assertEqual(1.0, scale)
        self.assertEqual(0.0, shift)
        np.testing.assert_array_equal(candidate, aligned)


if __name__ == "__main__":
    unittest.main()
