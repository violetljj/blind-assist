from __future__ import annotations

from decimal import Decimal
import math
import unittest

import numpy as np

from egomotion_compensated_looming.tum_fr2_rpy_geometry_audit.audit import (
    IndexRow,
    PoseRow,
    associate_unique_nearest,
    fixed_windows,
    interpolate_pose,
    relative_geometry,
)


class TumFr2RpyGeometryAuditTests(unittest.TestCase):
    def test_unique_nearest_matches_official_greedy_shape(self) -> None:
        rgb = [
            IndexRow(Decimal("0.000"), "rgb/a.png"),
            IndexRow(Decimal("0.030"), "rgb/b.png"),
        ]
        depth = [
            IndexRow(Decimal("0.010"), "depth/a.png"),
            IndexRow(Decimal("0.040"), "depth/b.png"),
        ]
        self.assertEqual({0: 0, 1: 1}, associate_unique_nearest(rgb, depth))

    def test_fixed_windows_use_integer_anchor_and_complete_tail(self) -> None:
        rgb = [
            IndexRow(Decimal("0.10"), "a"),
            IndexRow(Decimal("31.00"), "b"),
        ]
        depth = [
            IndexRow(Decimal("0.20"), "a"),
            IndexRow(Decimal("31.10"), "b"),
        ]
        poses = [
            PoseRow(
                Decimal("0.00"),
                np.zeros(3),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ),
            PoseRow(
                Decimal("31.20"),
                np.zeros(3),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ),
        ]
        self.assertEqual(
            [
                (Decimal("1"), Decimal("11")),
                (Decimal("11"), Decimal("21")),
                (Decimal("21"), Decimal("31")),
            ],
            fixed_windows(rgb, depth, poses),
        )

    def test_pose_interpolation_and_relative_translation_sign(self) -> None:
        poses = [
            PoseRow(
                Decimal("0.00"),
                np.asarray((0.0, 0.0, 0.0)),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ),
            PoseRow(
                Decimal("0.04"),
                np.asarray((0.04, 0.0, 0.0)),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ),
        ]
        timestamps = [row.timestamp for row in poses]
        previous = interpolate_pose(poses, timestamps, Decimal("0.00"))
        current = interpolate_pose(poses, timestamps, Decimal("0.04"))
        rotation, translation, angular = relative_geometry(
            previous, current, 0.04
        )
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, (-0.04, 0.0, 0.0), atol=1e-12)
        self.assertEqual(0.0, angular)

    def test_relative_rotation_rate(self) -> None:
        half_angle = math.radians(4.0)
        poses = [
            (
                np.zeros(3),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ),
            (
                np.zeros(3),
                np.asarray((0.0, 0.0, math.sin(half_angle), math.cos(half_angle))),
            ),
        ]
        _, translation, angular = relative_geometry(poses[0], poses[1], 1.0)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(math.radians(8.0), angular, places=12)


if __name__ == "__main__":
    unittest.main()
