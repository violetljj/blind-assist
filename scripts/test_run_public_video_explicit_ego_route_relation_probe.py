#!/usr/bin/env python3
"""Pure tests for explicit obstacle-to-restored-route geometry."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_video_explicit_ego_route_relation_probe as subject


class ExplicitEgoRouteRelationProbeTest(unittest.TestCase):
    def test_no_obstacle_has_zero_intrusion(self) -> None:
        walkable = np.zeros((64, 64), dtype=np.float32)
        walkable[:, 24:40] = 1.0
        result = subject.explicit_route_relation(walkable, np.zeros((64, 64), dtype=bool))
        self.assertEqual(result["route_intrusion_score"], 0.0)

    def test_central_obstacle_scores_above_lateral_obstacle(self) -> None:
        walkable = np.zeros((64, 64), dtype=np.float32)
        walkable[:, 20:44] = 1.0
        center = np.zeros((64, 64), dtype=bool)
        center[34:48, 28:36] = True
        lateral = np.zeros((64, 64), dtype=bool)
        lateral[34:48, 52:60] = True
        center_score = subject.explicit_route_relation(walkable, center)["route_intrusion_score"]
        lateral_score = subject.explicit_route_relation(walkable, lateral)["route_intrusion_score"]
        self.assertGreater(center_score, lateral_score)

    def test_marker_notch_is_restored_before_path_trace(self) -> None:
        walkable = np.zeros((64, 64), dtype=np.float32)
        walkable[:, 24:40] = 1.0
        obstacle = np.zeros((64, 64), dtype=bool)
        obstacle[34:48, 28:36] = True
        observed = walkable.copy()
        observed[obstacle] = 0.0
        result = subject.explicit_route_relation(observed, obstacle)
        self.assertGreater(result["route_intrusion_score"], 0.8)
        self.assertLess(abs(result["route_center_mean"] - 0.5), 0.12)

    def test_invalid_probability_is_rejected(self) -> None:
        walkable = np.ones((64, 64), dtype=np.float32)
        walkable[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "zero to one"):
            subject.restore_walkable_support(walkable, np.zeros((64, 64), dtype=bool))


if __name__ == "__main__":
    unittest.main()
