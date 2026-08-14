from __future__ import annotations

import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_multi_candidate_reliability_consistency as subject


class MultiCandidateReliabilityConsistencyTest(unittest.TestCase):
    def test_robust_affine_residual_removes_gain_and_bias(self) -> None:
        reference = np.linspace(-0.8, 0.8, 100, dtype=np.float32).reshape(10, 10)
        candidate = 1.25 * reference + 0.10
        coverage = np.ones((10, 10), dtype=bool)
        residual, receipt = subject._robust_affine_residual(reference, candidate, coverage)
        self.assertLess(float(np.max(residual)), 1e-5)
        self.assertAlmostEqual(1.25, receipt["affine_gain"], places=5)
        self.assertAlmostEqual(0.10, receipt["affine_bias"], places=5)
        self.assertEqual(1.0, receipt["affine_inlier_fraction"])

    def test_robust_affine_residual_exposes_local_outlier(self) -> None:
        reference = np.linspace(-0.8, 0.8, 100, dtype=np.float32).reshape(10, 10)
        candidate = reference.copy()
        candidate[5, 5] += 1.0
        coverage = np.ones((10, 10), dtype=bool)
        residual, receipt = subject._robust_affine_residual(reference, candidate, coverage)
        self.assertGreater(float(residual[5, 5]), receipt["affine_residual_threshold"])
        self.assertLess(receipt["affine_inlier_fraction"], 1.0)

    def test_feature_contract_dimensions_are_stable(self) -> None:
        self.assertEqual(72, subject.CELL_COUNT_PER_QUERY)
        self.assertEqual(6, len(subject.CELL_CHANNEL_NAMES))
        self.assertEqual(16, len(subject.GLOBAL_FEATURE_NAMES))
        self.assertEqual(9, len(subject.SET_CONSISTENCY_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
