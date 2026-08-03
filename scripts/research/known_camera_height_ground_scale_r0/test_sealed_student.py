import json
import unittest
from pathlib import Path

import numpy as np

from sealed_student import MODEL_ID, SealedScaleStudent, validate_golden


REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT = REPO_ROOT / "configs/hftf/camera_conditioned_scale_student_r0_model.json"
GOLDEN = REPO_ROOT / "configs/hftf/camera_conditioned_scale_student_r0_golden.json"


class SealedScaleStudentTest(unittest.TestCase):
    def test_receipt_and_golden_vectors_are_exact(self):
        validate_golden(RECEIPT, GOLDEN)

    def test_receipt_records_fixed_training_membership_and_exclusions(self):
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(MODEL_ID, receipt["model_id"])
        membership = receipt["training_membership"]
        self.assertEqual(163, membership["sorted_record_count"])
        self.assertEqual(64, len(membership["sorted_membership_sha256"]))
        self.assertEqual(2, len(membership["excluded_records"]))

    def test_invalid_and_extreme_features_fail_closed(self):
        student = SealedScaleStudent.load(RECEIPT)
        self.assertEqual("UNKNOWN", student.predict([float("nan")] * 10)["status"])
        extreme = student.feature_mean.copy()
        extreme[0] += 100.0 * student.feature_standard_deviation[0]
        self.assertEqual("UNKNOWN", student.predict(extreme)["status"])

    def test_feature_order_is_part_of_contract(self):
        student = SealedScaleStudent.load(RECEIPT)
        expected = (
            "log_r0_known_height_scale",
            "log_known_camera_height_m",
            "r0_plane_normal_x",
            "r0_plane_normal_y",
            "r0_plane_normal_z",
            "r0_normalized_plane_residual",
            "log_da_depth_q10",
            "log_da_depth_q50",
            "log_da_depth_q90",
            "log_da_depth_q90_over_q10",
        )
        self.assertEqual(expected, student.feature_names)


if __name__ == "__main__":
    unittest.main()
