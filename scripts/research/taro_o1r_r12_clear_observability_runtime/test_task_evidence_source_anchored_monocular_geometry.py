from __future__ import annotations

import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_source_anchored_monocular_geometry as subject,
)


class SourceAnchoredMonocularGeometryTest(unittest.TestCase):
    def test_robust_reference_scale_recovers_scale_with_outliers(self) -> None:
        prediction = np.linspace(0.5, 4.0, 400, dtype=np.float64).reshape(20, 20)
        source = prediction * 1.75
        source.reshape(-1)[:30] *= 3.0
        scale, receipt = subject.robust_reference_scale(
            source,
            prediction,
            np.ones_like(source, dtype=bool),
        )
        self.assertAlmostEqual(1.75, scale, places=6)
        self.assertGreater(receipt["reference_anchor_inlier_fraction"], 0.8)

    def test_robust_reference_scale_rejects_insufficient_support(self) -> None:
        values = np.ones((10, 10), dtype=np.float64)
        with self.assertRaisesRegex(subject.R32Error, "support insufficient"):
            subject.robust_reference_scale(values, values, np.ones_like(values, dtype=bool))


if __name__ == "__main__":
    unittest.main()
