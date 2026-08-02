import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_low_rank_spatial_relation_head import (
    fit_low_rank_logistic,
)


class SanpoLowRankSpatialRelationHeadTest(unittest.TestCase):
    def test_fit_respects_rank_and_parameter_count(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(40, 12))
        labels = (
            features[:, 0]
            - 0.7 * features[:, 4]
            + 0.3 * features[:, 8]
            > 0.0
        ).astype(np.int64)
        weights = np.full(40, 1.0 / 40.0)
        coefficient, intercept, diagnostics = fit_low_rank_logistic(
            features,
            labels,
            weights,
            channels=3,
            cells=4,
            rank=2,
            l2_strength=1.0,
        )
        matrix = coefficient.reshape(3, 4)
        self.assertLessEqual(np.linalg.matrix_rank(matrix), 2)
        self.assertEqual(15, diagnostics["parameter_count"])
        self.assertEqual(13, diagnostics["unconstrained_parameter_count"])
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(diagnostics["low_rank_final_loss"]))
        self.assertLessEqual(
            diagnostics["low_rank_final_loss"],
            diagnostics["truncated_svd_initial_loss"] + 1e-10,
        )

    def test_invalid_feature_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 12 features"):
            fit_low_rank_logistic(
                np.zeros((5, 11)),
                np.zeros(5, dtype=np.int64),
                np.ones(5),
                channels=3,
                cells=4,
                rank=2,
            )


if __name__ == "__main__":
    unittest.main()
