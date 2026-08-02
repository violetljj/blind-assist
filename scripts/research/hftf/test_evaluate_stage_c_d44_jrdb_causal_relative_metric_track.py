import unittest

import numpy as np

from evaluate_stage_c_d44_jrdb_causal_relative_metric_track import (
    NOT_SUPPORTED_STATUS,
    SUPPORTED_STATUS,
    determine_terminal,
    predict_relative_metric_track,
)


class D44RelativeMetricTrackTest(unittest.TestCase):
    def test_linear_relative_motion_extrapolates(self):
        history = []
        for frame in range(7):
            time_s = frame * 0.1
            history.append(
                {
                    "frame_index": frame,
                    "timestamp_ns": int(time_s * 1_000_000_000),
                    "center_base_link_m": [
                        10.0 - 0.5 * time_s,
                        1.0 + 0.2 * time_s,
                        1.0,
                    ],
                }
            )
        prediction = predict_relative_metric_track(
            history,
            1_000_000_000,
        )
        np.testing.assert_allclose(
            prediction,
            [9.5, 1.2, 1.0],
            atol=1e-9,
        )

    def test_exact_supported_status_requires_all_gates(self):
        pooled = {
            "opportunities": 500,
            "distinct_native_identities": 20,
            "candidate_vs_current": {
                "mean_horizontal_error_relative_reduction": 0.30,
                "median_horizontal_error_relative_reduction": 0.30,
                "horizontal_error_better_fraction": 0.70,
                "mean_range_error_relative_reduction": 0.20,
                "mean_bearing_error_relative_reduction": 0.15,
            },
        }
        by_sequence = [
            {
                "opportunities": 100,
                "candidate_vs_current": {
                    "mean_horizontal_error_relative_reduction": 0.10,
                },
            }
            for _ in range(4)
        ]
        _, _, status = determine_terminal(pooled, by_sequence, 480)
        self.assertEqual(status, SUPPORTED_STATUS)
        pooled["candidate_vs_current"][
            "horizontal_error_better_fraction"
        ] = 0.50
        _, _, status = determine_terminal(pooled, by_sequence, 480)
        self.assertEqual(status, NOT_SUPPORTED_STATUS)


if __name__ == "__main__":
    unittest.main()
