import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_stage_c_d42_jrdb_ego_object_metric_teacher import (
    ARM_CURRENT,
    ARM_FULL,
    NOT_SUPPORTED_STATUS,
    SUPPORTED_STATUS,
    determine_terminal,
    predict_arms,
)


class D42MetricTeacherTest(unittest.TestCase):
    def test_ego_and_object_world_motion_are_extrapolated(self):
        history = []
        for frame in range(7):
            time_s = frame * 0.1
            ego = np.asarray([time_s, 0.0, 0.0])
            object_odom = np.asarray([10.0 + 0.5 * time_s, 1.0, 1.0])
            center_base = object_odom - ego
            history.append(
                {
                    "timestamp_ns": int(time_s * 1_000_000_000),
                    "pose_translation": ego,
                    "pose_quaternion": Rotation.from_euler(
                        "z",
                        0.0,
                    ).as_quat(),
                    "center_odom_m": object_odom,
                    "center_base_link_m": center_base,
                }
            )
        predicted = predict_arms(history, 1_000_000_000)
        truth = np.asarray([9.5, 1.0, 1.0])
        np.testing.assert_allclose(predicted[ARM_FULL], truth, atol=1e-9)
        self.assertGreater(
            float(np.linalg.norm(predicted[ARM_CURRENT] - truth)),
            0.1,
        )

    def test_exact_supported_status_requires_all_gates(self):
        pooled = {
            "opportunities": 500,
            "distinct_native_identities": 20,
            "full_vs_current": {
                "mean_horizontal_error_relative_reduction": 0.30,
                "median_horizontal_error_relative_reduction": 0.25,
                "horizontal_error_better_fraction": 0.70,
                "mean_range_error_relative_reduction": 0.20,
                "mean_bearing_error_relative_reduction": 0.15,
            },
        }
        by_sequence = [
            {
                "sequence": str(index),
                "opportunities": 100,
                "full_vs_current": {
                    "mean_horizontal_error_relative_reduction": 0.10,
                },
            }
            for index in range(4)
        ]
        _, _, status = determine_terminal(
            pooled,
            by_sequence,
            480,
            1e-12,
        )
        self.assertEqual(status, SUPPORTED_STATUS)
        pooled["full_vs_current"][
            "mean_bearing_error_relative_reduction"
        ] = 0.0
        _, _, status = determine_terminal(
            pooled,
            by_sequence,
            480,
            1e-12,
        )
        self.assertEqual(status, NOT_SUPPORTED_STATUS)


if __name__ == "__main__":
    unittest.main()
