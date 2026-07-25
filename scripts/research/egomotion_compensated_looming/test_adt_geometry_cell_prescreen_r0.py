#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_adt_geometry_cell_prescreen_r0 as subject  # noqa: E402


def synthetic_series(start_range: float, end_range: float, target_dx: float = 0.0):
    rows = []
    for index in range(301):
        fraction = index / 300
        rows.append(
            {
                "time_ns": int(index * (10_000_000_000 / 301)),
                "range_m": start_range + fraction * (end_range - start_range),
                "target_position": (fraction * target_dx, 0.0, 0.0),
            }
        )
    return rows


class AdtGeometryPrescreenTest(unittest.TestCase):
    def test_identity_oriented_box_distance(self) -> None:
        rotation = subject.quaternion_rotation_wxyz((1.0, 0.0, 0.0, 0.0))
        distance = subject.point_to_oriented_box_distance(
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            rotation,
            (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        )
        self.assertAlmostEqual(distance, 1.0)
        with self.assertRaises(ValueError):
            subject.quaternion_rotation_wxyz((math.nan, 0.0, 0.0, 1.0))

    def test_nearest_index_rejects_tie_and_out_of_tolerance(self) -> None:
        self.assertEqual(subject.nearest_index([1_000, 3_000], 1_400, 1_000), 0)
        self.assertIsNone(subject.nearest_index([1_000, 3_000], 2_000, 1_000))
        self.assertIsNone(subject.nearest_index([1_000], 2_001, 1_000))

    def test_static_approach_cell(self) -> None:
        cells = subject.classify(
            "STATIC_OBJECT",
            synthetic_series(2.0, 1.0),
            [2.0 - index / 19 for index in range(20)],
            [0.08] * 20,
            {
                "endpoint_displacement_m": 1.0,
                "path_length_m": 1.2,
                "angular_speed_median_rad_s": 0.05,
            },
            0,
        )
        self.assertIn("EGO_APPROACH_STATIC_SURFACE", cells)

    def test_active_target_approach_cell(self) -> None:
        cells = subject.classify(
            "TIMESTAMPED_OBJECT",
            synthetic_series(2.0, 1.0, target_dx=1.0),
            [2.0 - index / 19 for index in range(20)],
            [0.08] * 20,
            {
                "endpoint_displacement_m": 0.1,
                "path_length_m": 0.2,
                "angular_speed_median_rad_s": 0.05,
            },
            0,
        )
        self.assertIn("STATIONARY_EGO_ACTIVE_TARGET_APPROACH", cells)

    def test_rotation_and_lateral_rules(self) -> None:
        rotation_cells = subject.classify(
            "STATIC_OBJECT",
            synthetic_series(2.0, 2.0),
            [2.0] * 20,
            [0.0] * 20,
            {
                "endpoint_displacement_m": 0.1,
                "path_length_m": 0.2,
                "angular_speed_median_rad_s": 0.30,
            },
            0,
        )
        self.assertIn("PURE_EGO_ROTATION_NO_CLOSING", rotation_cells)

        ranges = [2.0 - 0.1 * index for index in range(10)]
        ranges += [1.0 + 0.1 * (index + 1) for index in range(10)]
        lateral_cells = subject.classify(
            "STATIC_OBJECT",
            synthetic_series(2.0, 2.0),
            ranges,
            [0.08] * 10 + [-0.08] * 10,
            {
                "endpoint_displacement_m": 1.0,
                "path_length_m": 2.0,
                "angular_speed_median_rad_s": 0.05,
            },
            0,
        )
        self.assertIn("LATERAL_PASS_NO_SUSTAINED_CLOSING", lateral_cells)


if __name__ == "__main__":
    unittest.main()
