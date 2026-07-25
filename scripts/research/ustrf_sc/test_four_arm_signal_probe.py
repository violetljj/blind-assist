from __future__ import annotations

import unittest

import numpy as np

from four_arm_signal_probe import (
    bbox_route_score,
    compare_delta,
    dense_route_score,
    summarize_deltas,
)


class FourArmSignalProbeTest(unittest.TestCase):
    def test_dense_route_score_uses_only_route_patch(self) -> None:
        risk = np.zeros((63, 63), dtype=np.float64)
        valid = np.ones((63, 63), dtype=bool)
        risk[28:35, 28:35] = 0.9
        center = dense_route_score(
            risk,
            valid,
            [320.0, 240.0],
            source_width=640,
            source_height=480,
        )
        far = dense_route_score(
            risk,
            valid,
            [40.0, 40.0],
            source_width=640,
            source_height=480,
        )
        self.assertIsNotNone(center)
        self.assertIsNotNone(far)
        self.assertGreater(center, far)

    def test_bbox_score_is_continuous_route_patch_occupancy(self) -> None:
        full = bbox_route_score(
            [{"label": "person", "confidence": 0.8, "box": [280.0, 200.0, 360.0, 280.0]}],
            [320.0, 240.0],
            source_width=640,
            source_height=480,
        )
        partial = bbox_route_score(
            [{"label": "person", "confidence": 0.8, "box": [340.0, 240.0, 400.0, 300.0]}],
            [320.0, 240.0],
            source_width=640,
            source_height=480,
        )
        absent = bbox_route_score(
            [{"label": "chair", "confidence": 0.9, "box": [280.0, 200.0, 360.0, 280.0]}],
            [320.0, 240.0],
            source_width=640,
            source_height=480,
        )
        self.assertGreater(full, partial)
        self.assertGreater(partial, absent)
        self.assertEqual(absent, 0.0)

    def test_pair_summary_and_tie_handling(self) -> None:
        summary = summarize_deltas([1.0] * 12 + [-1.0] * 3)
        self.assertEqual(summary["wins"], 12)
        self.assertEqual(summary["losses"], 3)
        self.assertGreater(summary["win_rate_wilson_95"][0], 0.5)
        self.assertEqual(compare_delta(0.0), "tie")


if __name__ == "__main__":
    unittest.main()
