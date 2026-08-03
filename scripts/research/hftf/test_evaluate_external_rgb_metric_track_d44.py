#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from evaluate_external_rgb_metric_track_d44 import (
    HISTORY_COUNT,
    ols_predict,
    relative_position,
)


class ExternalRgbMetricTrackD44Test(unittest.TestCase):
    def test_maps_camera_depth_to_forward_lateral_vertical(self) -> None:
        row = {
            "torso_roi_xyxy_px": [14, 18, 16, 22],
            "intrinsics_fx_fy_cx_cy": [10, 20, 10, 10],
        }
        np.testing.assert_allclose(
            relative_position(row, 2.0), [2.0, 1.0, -1.0]
        )

    def test_ols_recovers_constant_velocity_at_future_time(self) -> None:
        timestamps = [index * 100_000_000 for index in range(HISTORY_COUNT)]
        positions = [
            np.asarray([2.0 - index * 0.1, index * 0.05, 0.0])
            for index in range(HISTORY_COUNT)
        ]
        prediction = ols_predict(timestamps, positions, 1_600_000_000)
        np.testing.assert_allclose(prediction, [0.4, 0.8, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
