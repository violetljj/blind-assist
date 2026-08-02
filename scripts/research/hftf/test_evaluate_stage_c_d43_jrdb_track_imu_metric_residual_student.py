import unittest

from evaluate_stage_c_d43_jrdb_track_imu_metric_residual_student import (
    FULL_FEATURE_NAMES,
    NOT_SUPPORTED_STATUS,
    SUPPORTED_STATUS,
    TRACK_FEATURE_NAMES,
    TRACK_IMU,
    TRACK_ONLY,
    determine_terminal,
)


class D43ResidualStudentTest(unittest.TestCase):
    def test_exact_supported_status_requires_all_gates(self):
        pooled = {
            "opportunities": 500,
            "distinct_native_identities": 20,
            "track_imu_vs_zero": {
                "teacher_error_relative_reduction": 0.30,
                "actual_error_relative_reduction": 0.20,
                "actual_error_better_fraction": 0.60,
            },
            "track_imu_vs_track_only": {
                "actual_error_relative_delta": 0.01,
            },
        }
        by_fold = []
        for index in range(4):
            by_fold.append(
                {
                    "opportunities": 100,
                    "training_sequences": ["a", "b", "c"],
                    "track_imu_vs_zero": {
                        "teacher_error_relative_reduction": 0.10,
                        "actual_error_relative_reduction": 0.10,
                    },
                    "model_receipts": {
                        TRACK_ONLY: {
                            "feature_count": len(TRACK_FEATURE_NAMES),
                        },
                        TRACK_IMU: {
                            "feature_count": len(FULL_FEATURE_NAMES),
                        },
                    },
                }
            )
        _, _, status = determine_terminal(
            pooled,
            by_fold,
            480,
            1e-12,
        )
        self.assertEqual(status, SUPPORTED_STATUS)
        pooled["track_imu_vs_zero"][
            "actual_error_better_fraction"
        ] = 0.50
        _, _, status = determine_terminal(
            pooled,
            by_fold,
            480,
            1e-12,
        )
        self.assertEqual(status, NOT_SUPPORTED_STATUS)


if __name__ == "__main__":
    unittest.main()
