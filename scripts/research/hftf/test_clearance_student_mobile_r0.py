import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clearance_student_mobile_r0 import (
    ClearanceStudentMobileR0,
    clearance_student_loss,
    parameter_count,
    require_finite_metrics,
)


class ClearanceStudentMobileR0Test(unittest.TestCase):
    def test_model_has_mobile_size_and_all_heads(self) -> None:
        model = ClearanceStudentMobileR0(pretrained=False)
        self.assertLess(parameter_count(model), 1_600_000)
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 384, 384), (64, 96))
        self.assertEqual(tuple(output["metric_depth"].shape), (2, 64, 96))
        self.assertEqual(tuple(output["confidence"].shape), (2, 64, 96))
        self.assertEqual(tuple(output["clearance"].shape), (2, 3))
        self.assertEqual(tuple(output["ground_plane"].shape), (2, 4))
        self.assertEqual(tuple(output["camera_height"].shape), (2,))

    def test_loss_is_finite_and_has_asymmetric_components(self) -> None:
        model = ClearanceStudentMobileR0(pretrained=False)
        prediction = model(torch.randn(2, 3, 384, 384), (32, 48))
        truth = torch.rand(2, 32, 48) * 4.0 + 0.25
        teacher = truth + 0.01
        total, parts = clearance_student_loss(prediction, truth, teacher)
        self.assertTrue(math.isfinite(float(total)))
        self.assertIn("occupancy_false_clear", parts)
        self.assertIn("clearance", parts)

    def test_nonfinite_metrics_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            require_finite_metrics({"clearance_mae_m": float("nan")})


if __name__ == "__main__":
    unittest.main()
