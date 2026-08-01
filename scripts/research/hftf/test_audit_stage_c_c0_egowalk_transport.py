from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_c0_egowalk_transport import (
    _sample_indices,
    _surface_metrics,
)


class StageCC0EgoWalkTransportAuditTest(unittest.TestCase):
    def test_sample_indices_are_exact_bounded_and_unique(self) -> None:
        indices = _sample_indices(647)
        self.assertEqual(32, len(indices))
        self.assertEqual(0, indices[0])
        self.assertEqual(646, indices[-1])
        self.assertEqual(indices, sorted(set(indices)))

    def test_surface_metrics_keep_missing_depth_unknown(self) -> None:
        samples = {
            0: np.array([[1.0, np.nan], [2.0, 0.0]], dtype=np.float32),
            1: np.array([[1.2, np.nan], [2.1, 0.0]], dtype=np.float32),
        }
        metrics = _surface_metrics(samples, 0.25, 0.25)
        self.assertEqual(2, metrics["frames_passing_global_depth_fraction"])
        self.assertEqual(
            1, metrics["adjacent_pairs_passing_common_support"]
        )
        self.assertEqual(
            0.5,
            metrics["adjacent_pairs"][0][
                "common_positive_finite_depth_fraction"
            ],
        )

    def test_common_support_fails_when_valid_pixels_do_not_overlap(
        self,
    ) -> None:
        samples = {
            0: np.array([[1.0, np.nan], [1.0, np.nan]], dtype=np.float32),
            1: np.array([[np.nan, 1.0], [np.nan, 1.0]], dtype=np.float32),
        }
        metrics = _surface_metrics(samples, 0.25, 0.25)
        self.assertEqual(
            0, metrics["adjacent_pairs_passing_common_support"]
        )


if __name__ == "__main__":
    unittest.main()
