import unittest

from evaluate_stage_c_d43_1_jrdb_track_only_metric_residual_student import (
    NOT_SUPPORTED_STATUS,
    SUPPORTED_STATUS,
    determine_terminal,
)
from evaluate_stage_c_d43_jrdb_track_imu_metric_residual_student import (
    TRACK_FEATURE_NAMES,
)


class D431TrackOnlyStudentTest(unittest.TestCase):
    def test_exact_supported_status_requires_all_gates(self):
        pooled = {
            "opportunities": 500,
            "distinct_native_identities": 20,
            "track_only_vs_zero": {
                "teacher_error_relative_reduction": 0.30,
                "actual_error_relative_reduction": 0.20,
                "actual_error_better_fraction": 0.60,
            },
        }
        by_fold = [
            {
                "opportunities": 100,
                "training_sequences": ["a", "b", "c"],
                "track_only_vs_zero": {
                    "teacher_error_relative_reduction": 0.10,
                    "actual_error_relative_reduction": 0.10,
                },
                "model_receipt": {
                    "feature_count": len(TRACK_FEATURE_NAMES),
                },
            }
            for _ in range(4)
        ]
        _, _, status = determine_terminal(
            pooled,
            by_fold,
            480,
            1e-12,
        )
        self.assertEqual(status, SUPPORTED_STATUS)
        pooled["track_only_vs_zero"][
            "teacher_error_relative_reduction"
        ] = 0.0
        _, _, status = determine_terminal(
            pooled,
            by_fold,
            480,
            1e-12,
        )
        self.assertEqual(status, NOT_SUPPORTED_STATUS)


if __name__ == "__main__":
    unittest.main()
