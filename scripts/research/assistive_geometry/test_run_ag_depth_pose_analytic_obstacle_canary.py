import unittest

import numpy as np

from scripts.research.assistive_geometry.run_ag_depth_pose_analytic_obstacle_canary import (
    HeightModePolicy,
    horizontal_world_heights,
    persistent_height_modes,
)


class DepthPoseAnalyticObstacleCanaryTest(unittest.TestCase):
    def test_persistent_modes_select_lowest_supported_height(self) -> None:
        policy = HeightModePolicy(
            minimum_frame_points=16,
            minimum_total_points=48,
            sample_stride=1,
        )
        frames = []
        for offset in (0.0, 0.005, -0.005):
            floor = np.linspace(-0.015, 0.015, 128) + offset
            table = np.linspace(0.735, 0.765, 256) + offset
            frames.append(np.concatenate((floor, table)))
        modes = persistent_height_modes(frames, policy)
        self.assertGreaterEqual(len(modes), 2)
        self.assertLess(abs(modes[0]["world_height_m"]), 0.03)
        self.assertGreater(modes[1]["world_height_m"], 0.70)

    def test_horizontal_height_uses_pose_world_z(self) -> None:
        height, width = 32, 40
        depth = np.full((height, width), 1.0, dtype=np.float32)
        intrinsics = np.asarray(
            [[40.0, 0.0, 19.5], [0.0, 40.0, 15.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 0.25
        policy = HeightModePolicy(sample_stride=1)
        values = horizontal_world_heights(depth, intrinsics, pose, policy)
        self.assertGreater(values.size, 100)
        self.assertTrue(np.allclose(values, 1.25, atol=1e-5))

    def test_mode_requires_persistence(self) -> None:
        policy = HeightModePolicy(
            minimum_persistent_frames=2,
            minimum_frame_points=4,
            minimum_total_points=8,
            sample_stride=1,
        )
        with self.assertRaisesRegex(RuntimeError, "too few pose frames"):
            persistent_height_modes([np.zeros(32)], policy)


if __name__ == "__main__":
    unittest.main()
