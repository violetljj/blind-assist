#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency-free test runtime
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_bonn_static_surface_truth_ledger_r0 as subject


@unittest.skipIf(np is None, "numpy absent in dependency-free runtime")
class BonnStaticSurfaceTruthLedgerTest(unittest.TestCase):
    def test_identity_pose_matrix(self) -> None:
        row = ["1.0", "0", "0", "0", "0", "0", "0", "1"]
        np.testing.assert_allclose(subject.pose_matrix(row), np.eye(4))

    def test_map_reduction_is_deterministic(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 1.0], [0.01, 0.01, 1.01], [1.0, 1.0, 1.0]]
        )
        first = subject.reduce_map(points)
        second = subject.reduce_map(points.copy())
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 2)

    def test_static_surface_observation_requires_frozen_support(self) -> None:
        points = np.repeat(
            np.asarray([[0.0, 0.0, 2.0]]),
            subject.MIN_STATIC_MAP_POINTS_IN_ROI,
            axis=0,
        )
        observation = subject.static_surface_observation(points, np.eye(4))
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(
            observation["static_surface_depth_q05_meters"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
