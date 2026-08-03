#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_metric3d_clearance_field_a0 import (
    clearance_field,
    depth_to_points,
    fit_ground_plane,
    fit_gravity_guided_ground_plane,
    fixed_world_plane_in_camera,
    fit_normal_guided_ground_plane,
    summarize,
    tum_fixed_world_floor_in_camera,
    tum_depth_metres,
)


class Metric3dClearanceFieldA0Test(unittest.TestCase):
    def test_depth_to_points_preserves_optical_axis(self) -> None:
        depth = np.full((3, 3), 2.0)
        intrinsics = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
        points, _ = depth_to_points(depth, intrinsics, stride=1)
        np.testing.assert_allclose(points[4], [0.0, 0.0, 2.0])

    def test_fit_ground_plane_recovers_camera_height(self) -> None:
        x, z = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(0.5, 5, 30))
        points = np.stack((x.ravel(), np.full(x.size, 1.2), z.ravel()), axis=1)
        pixels = np.stack((np.arange(x.size) % 640, np.full(x.size, 400)), axis=1)
        plane = fit_ground_plane(points, pixels, 480)
        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertAlmostEqual(plane[1], 1.2, places=6)

    def test_normal_guided_plane_rejects_wall_normals(self) -> None:
        x, z = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(0.5, 5, 30))
        points = np.stack((x.ravel(), np.full(x.size, 1.2), z.ravel()), axis=1)
        pixels = np.stack((np.arange(x.size) % 40, np.full(x.size, 30)), axis=1)
        normal_map = np.zeros((40, 40, 4), dtype=np.float64)
        normal_map[:, :, :] = [0.0, -1.0, 0.0, 5.0]
        normal_map[:10, :, :] = [1.0, 0.0, 0.0, 20.0]
        plane = fit_normal_guided_ground_plane(points, pixels, normal_map, 40)
        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertAlmostEqual(plane[1], 1.2, places=6)

    def test_gravity_guided_plane_keeps_given_orientation(self) -> None:
        x, z = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(0.5, 5, 30))
        points = np.stack((x.ravel(), np.full(x.size, 1.2), z.ravel()), axis=1)
        pixels = np.stack((np.arange(x.size) % 40, np.full(x.size, 30)), axis=1)
        plane = fit_gravity_guided_ground_plane(
            points, pixels, np.asarray([0.0, -1.0, 0.0]), 40
        )
        self.assertIsNotNone(plane)
        assert plane is not None
        np.testing.assert_allclose(plane[0], [0.0, -1.0, 0.0])
        self.assertAlmostEqual(plane[1], 1.2, places=6)

    def test_fixed_world_floor_uses_camera_translation(self) -> None:
        poses = (
            np.asarray([10.0]),
            np.asarray([[0.0, 0.0, 1.5]]),
            np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        )
        plane = tum_fixed_world_floor_in_camera(10.0, poses, 0.1)
        self.assertIsNotNone(plane)
        assert plane is not None
        np.testing.assert_allclose(plane[0], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(plane[1], 1.4)

    def test_fixed_world_plane_supports_non_z_aligned_world(self) -> None:
        poses = (
            np.asarray([10.0]),
            np.asarray([[0.0, 1.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        )
        plane = fixed_world_plane_in_camera(
            10.0, poses, np.asarray([0.0, -1.0, 0.0]), 2.0
        )
        self.assertIsNotNone(plane)
        assert plane is not None
        np.testing.assert_allclose(plane[0], [0.0, -1.0, 0.0])
        self.assertAlmostEqual(plane[1], 1.0)

    def test_unknown_when_no_ground_support(self) -> None:
        depth = np.zeros((20, 20))
        intrinsics = np.eye(3)
        self.assertEqual(clearance_field(depth, intrinsics)["status"], "UNKNOWN_GROUND")

    def test_tum_depth_scale(self) -> None:
        actual = tum_depth_metres(np.asarray([[0, 5000, 10000]], dtype=np.uint16))
        np.testing.assert_allclose(actual, [[0.0, 1.0, 2.0]])

    def test_summary_counts_false_clear(self) -> None:
        def field(clearance: float) -> dict:
            return {
                "status": "VALID",
                "camera_height_m": 1.0,
                "bands": {
                    band: {
                        "clearance_m": clearance,
                        "occupied_by_horizon": {str(h): clearance <= h for h in (1.0, 1.5, 2.0)},
                    }
                    for band in ("left", "center", "right")
                },
            }
        report = summarize([{"sequence_root": "s", "sequence_id": "s-000", "latency_ms": 1.0, "sensor": field(0.5), "metric3d": field(3.0)}])
        self.assertEqual(report["false_clear_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
