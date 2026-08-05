#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dav2_rgbd_mobile_student_a4_r0 import masked_gradient_loss, rgbd_teacher_loss


class A4LossTest(unittest.TestCase):
    def test_exact_truth_has_zero_sensor_losses(self) -> None:
        truth = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones_like(truth, dtype=torch.bool)
        config = {
            "depth_clamp_m": [0.1, 20.0],
            "sensor_log_depth_smooth_l1_beta": 0.05,
            "sensor_log_gradient_l1_weight": 0.5,
            "teacher_centered_log_weight": 0.1,
            "teacher_centered_log_smooth_l1_beta": 0.05,
        }
        _total, parts = rgbd_teacher_loss(truth, truth, mask, truth, config)
        self.assertAlmostEqual(parts["sensor_log_depth"], 0.0)
        self.assertAlmostEqual(parts["sensor_log_gradient"], 0.0)

    def test_gradient_ignores_unpaired_invalid_pixels(self) -> None:
        prediction = torch.zeros((1, 2, 2))
        truth = torch.tensor([[[0.0, 100.0], [100.0, 100.0]]])
        mask = torch.tensor([[[True, False], [False, False]]])
        self.assertEqual(float(masked_gradient_loss(prediction, truth, mask)), 0.0)


if __name__ == "__main__":
    unittest.main()
