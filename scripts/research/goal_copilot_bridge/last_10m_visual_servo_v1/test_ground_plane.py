from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ground_plane import depth_points, estimate_ground_plane, ground_mask_from_depth


class GroundPlaneTest(unittest.TestCase):
    def test_recovers_synthetic_horizontal_floor(self) -> None:
        height = width = 120
        focal = width / 2.0
        yy, _ = np.indices((height, width), dtype=np.float32)
        # A horizontal floor at camera-frame y=1.5m satisfies
        # y=(pixel_y-cy)*z/focal, so solve for z below the horizon.
        denominator = yy - (height - 1) / 2.0
        depth = np.full((height, width), np.nan, dtype=np.float32)
        floor = denominator > 12
        depth[floor] = 1.5 * focal / denominator[floor]
        normal, offset = estimate_ground_plane(depth, iterations=128)
        self.assertGreater(float(normal[1]), 0.98)
        self.assertAlmostEqual(abs(offset), 1.5, delta=0.05)
        mask, _ = ground_mask_from_depth(depth)
        self.assertGreater(float(mask[floor].mean()), 0.95)

    def test_depth_points_center_ray(self) -> None:
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        points = depth_points(depth)
        np.testing.assert_allclose(points[1, 1], [0.0, 0.0, 2.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
