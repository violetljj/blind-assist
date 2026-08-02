#!/usr/bin/env python3
"""Tests for D30 box-to-world measurement mechanics."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate_stage_c_d30_thor_magni_box_world_measurement import (
    assign_measurements,
    is_person_body,
    relative_bearing_degrees,
)


class D30MeasurementTests(unittest.TestCase):
    def test_person_body_rule_excludes_robot_and_carried_object(self) -> None:
        self.assertTrue(is_person_body("Helmet_4", "Visitors-Alone"))
        self.assertTrue(is_person_body("Helmet_5", "Carrier-Box"))
        self.assertFalse(
            is_person_body("DARKO_Robot", "Differential-Teleoperated")
        )
        self.assertFalse(is_person_body("LO1", "Carried"))

    def test_bearing_sign_maps_left_to_negative_image_x(self) -> None:
        forward = np.asarray([1.0, 0.0])
        left = np.asarray([1.0, 1.0])
        right = np.asarray([1.0, -1.0])
        self.assertAlmostEqual(
            relative_bearing_degrees(forward, left),
            45.0,
        )
        self.assertAlmostEqual(
            relative_bearing_degrees(forward, right),
            -45.0,
        )
        assignments = assign_measurements(
            np.asarray([-0.9, 0.9]),
            np.asarray([45.0, -45.0]),
        )
        self.assertEqual([row["body_index"] for row in assignments], [0, 1])
        self.assertTrue(all(row["accepted"] for row in assignments))

    def test_hungarian_assignment_preserves_best_global_pairing(self) -> None:
        assignments = assign_measurements(
            np.asarray([-0.8, 0.0, 0.8]),
            np.asarray([40.0, 0.0, -40.0]),
        )
        self.assertEqual(len(assignments), 3)
        self.assertEqual(
            [row["body_index"] for row in assignments],
            [0, 1, 2],
        )
        self.assertTrue(all(row["x_error"] < 1e-9 for row in assignments))


if __name__ == "__main__":
    unittest.main()
