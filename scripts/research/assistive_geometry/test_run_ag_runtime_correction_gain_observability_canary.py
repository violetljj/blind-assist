from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_ag_runtime_correction_gain_observability_canary import (  # noqa: E402
    frame_sufficient_statistics,
    horizontal_flip_intrinsics,
    relative_consistency_gain,
    score_thresholds,
    summarize_gated_sufficient_statistics,
)
from train_ag_st_no_regret_selector import (  # noqa: E402
    SelectorObservation,
    summarize_selector_observations,
)


class RuntimeCorrectionGainObservabilityTest(unittest.TestCase):
    def test_horizontal_flip_intrinsics_mirrors_principal_point(self) -> None:
        matrix = np.asarray(
            [[520.0, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        flipped = horizontal_flip_intrinsics(matrix, 640)
        self.assertAlmostEqual(313.9, float(flipped[0, 2]), places=4)
        self.assertAlmostEqual(325.1, float(matrix[0, 2]), places=4)
        np.testing.assert_allclose(flipped[1:], matrix[1:])

    def test_relative_consistency_gain_requires_both_robust_statistics(self) -> None:
        score, receipt = relative_consistency_gain(
            np.asarray([1.0, 1.0, 1.0, 10.0]),
            np.asarray([0.5, 0.5, 0.5, 12.0]),
            base_floor=1e-4,
        )
        self.assertLess(score, 0.0)
        self.assertGreater(receipt["median_relative_gain"], 0.0)
        self.assertLess(receipt["q90_relative_gain"], 0.0)

    def test_score_thresholds_are_finite_unique_and_sorted(self) -> None:
        self.assertEqual(
            (-1.0, 0.2, 0.5),
            score_thresholds([0.5, float("nan"), 0.2, -1.0, 0.2]),
        )

    def test_sufficient_statistics_match_dense_open_and_closed_summary(self) -> None:
        observations = []
        for parent, offset in (("a", 0.0), ("b", 0.1)):
            truth = np.asarray([[1.0, 1.0]], dtype=np.float32)
            observations.append(
                SelectorObservation(
                    parent_id=parent,
                    domain="TUM_RGBD",
                    truth_depth_m=truth,
                    valid=np.ones_like(truth, dtype=bool),
                    base_depth_m=np.asarray([[1.2, 1.2]], dtype=np.float32),
                    expert_depth_m=np.asarray(
                        [[1.0 + offset, 1.4 + offset]], dtype=np.float32
                    ),
                    selector_probability=np.asarray([[0.9, 0.01]], dtype=np.float32),
                )
            )
        dense_open = summarize_selector_observations(observations, 0.05)
        stats = [frame_sufficient_statistics(row) for row in observations]
        compact_open = summarize_gated_sufficient_statistics(
            stats, np.ones(2, dtype=bool)
        )
        self.assertEqual(dense_open["per_parent"], compact_open["per_parent"])
        dense_closed_rows = [
            SelectorObservation(
                parent_id=row.parent_id,
                domain=row.domain,
                truth_depth_m=row.truth_depth_m,
                valid=row.valid,
                base_depth_m=row.base_depth_m,
                expert_depth_m=row.expert_depth_m,
                selector_probability=np.full_like(row.selector_probability, -1.0),
            )
            for row in observations
        ]
        dense_closed = summarize_selector_observations(dense_closed_rows, 0.05)
        compact_closed = summarize_gated_sufficient_statistics(
            stats, np.zeros(2, dtype=bool)
        )
        self.assertEqual(dense_closed["per_parent"], compact_closed["per_parent"])


if __name__ == "__main__":
    unittest.main()
