#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from evaluate_external_rgb_future_occupancy_capsule import (
    MOTION_CAPSULE,
    OLS_DISK,
    STATIC_DISK,
    calibrate_and_evaluate,
    conformal_radius,
    point_segment_distance,
)


class ExternalRgbFutureOccupancyCapsuleTest(unittest.TestCase):
    def test_point_segment_distance_covers_stop_and_continue(self) -> None:
        start = np.asarray([0.0, 0.0])
        end = np.asarray([2.0, 0.0])
        self.assertEqual(point_segment_distance(start, start, end), 0.0)
        self.assertEqual(point_segment_distance(end, start, end), 0.0)
        self.assertAlmostEqual(
            point_segment_distance(np.asarray([1.0, 0.5]), start, end), 0.5
        )

    def test_split_conformal_uses_finite_sample_rank(self) -> None:
        radius, rank = conformal_radius([float(i) for i in range(1, 11)], 0.1)
        self.assertEqual(rank, 10)
        self.assertEqual(radius, 10.0)

    def test_capsule_can_cover_stop_or_continue_with_smaller_radius(self) -> None:
        calibration = []
        evaluation = []
        for index in range(20):
            stopped = index % 2 == 0
            row = {
                "sequence_id": "s",
                "distances": {
                    STATIC_DISK: 0.0 if stopped else 1.0,
                    OLS_DISK: 1.0 if stopped else 0.0,
                    MOTION_CAPSULE: 0.0,
                },
                "capsule_segment_length_m": 1.0,
            }
            calibration.append(row)
            evaluation.append(row)
        result = calibrate_and_evaluate(calibration, evaluation, 0.1)
        self.assertEqual(result["arms"][MOTION_CAPSULE]["coverage"], 1.0)
        self.assertEqual(result["arms"][MOTION_CAPSULE]["radius_m"], 0.0)
        self.assertEqual(result["arms"][MOTION_CAPSULE]["mean_area_m2"], 0.0)


if __name__ == "__main__":
    unittest.main()
