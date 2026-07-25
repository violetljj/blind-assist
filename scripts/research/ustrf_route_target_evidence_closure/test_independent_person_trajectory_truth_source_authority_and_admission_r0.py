from __future__ import annotations

import unittest

from audit_independent_person_trajectory_truth_source_authority_and_admission_r0 import (
    distance_band,
    missing_rigid_body,
)


class IndependentPersonTrajectoryTruthR0Test(unittest.TestCase):
    def setUp(self) -> None:
        self.bands = [
            {"id": "0-5", "lower_inclusive": 0.0, "upper_exclusive": 5.0},
            {"id": "5-10", "lower_inclusive": 5.0, "upper_exclusive": 10.0},
            {"id": "10-20", "lower_inclusive": 10.0, "upper_exclusive": 20.0},
            {"id": "20-40", "lower_inclusive": 20.0, "upper_exclusive": 40.0},
            {"id": "40-plus", "lower_inclusive": 40.0, "upper_exclusive": None},
        ]

    def test_distance_band_boundaries(self) -> None:
        cases = {
            0.0: "0-5",
            4.999: "0-5",
            5.0: "5-10",
            10.0: "10-20",
            20.0: "20-40",
            39.999: "20-40",
            40.0: "40-plus",
            100.0: "40-plus",
        }
        for distance, expected in cases.items():
            with self.subTest(distance=distance):
                self.assertEqual(distance_band(distance, self.bands), expected)

    def test_all_zero_rigid_body_is_missing(self) -> None:
        self.assertTrue(missing_rigid_body([0.0] * 16))

    def test_nonzero_rigid_body_is_observed(self) -> None:
        values = [0.0] * 16
        values[2] = 1.0
        self.assertFalse(missing_rigid_body(values))


if __name__ == "__main__":
    unittest.main()
