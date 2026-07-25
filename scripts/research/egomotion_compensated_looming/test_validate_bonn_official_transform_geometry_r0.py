#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import numpy as np
    import PIL  # noqa: F401
except ImportError:  # pragma: no cover - dependency-free test runtime
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_bonn_official_transform_geometry_r0 as subject


@unittest.skipIf(np is None, "numpy/Pillow absent in dependency-free runtime")
class BonnOfficialTransformGeometryTest(unittest.TestCase):
    def test_t_ros_is_self_inverse(self) -> None:
        np.testing.assert_allclose(
            np.linalg.inv(subject.T_ROS), subject.T_ROS
        )

    def test_identity_quaternion(self) -> None:
        rotation = subject.quaternion_xyzw_rotation(
            np.asarray([0.0, 0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(rotation, np.eye(3))

    def test_pose_join_over_hard_cap_abstains(self) -> None:
        poses = [["1.0", "0", "0", "0", "0", "0", "0", "1"]]
        self.assertIsNone(subject.nearest_pose(poses, [1.0], 1.041))

    def test_voxel_support_detects_neighbor_and_rejects_far_point(self) -> None:
        map_points = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        keys, origin, widths = subject.build_voxel_index(map_points)
        near = subject.support_fraction(
            np.asarray([[0.02, 0.01, 0.0]]), keys, origin, widths
        )
        far = subject.support_fraction(
            np.asarray([[5.0, 5.0, 5.0]]), keys, origin, widths
        )
        self.assertEqual(near, 1.0)
        self.assertEqual(far, 0.0)


if __name__ == "__main__":
    unittest.main()
