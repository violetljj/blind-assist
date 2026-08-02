import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stage_c_d5_tartanground_development_student import (
    TemporalStudent,
    binary_metrics,
    decode_labels,
    train_prior_metrics,
)


class TartanGroundDevelopmentStudentTest(unittest.TestCase):
    def test_decode_labels_preserves_soft_risk_and_unknown(self):
        labels = {}
        for horizon in ("current", "near", "far"):
            known = np.zeros((3, 6, 6), dtype=np.uint8)
            risk = np.full((3, 6, 6), None, dtype=object)
            known[0, 1, 2] = 1
            risk[0, 1, 2] = 0.75
            labels[horizon] = {
                "known_target": known.tolist(),
                "risk_score_target_nullable": risk.tolist(),
            }

        risk, known = decode_labels({"labels": labels})

        self.assertEqual(tuple(risk.shape), (3, 3, 6, 6))
        self.assertAlmostEqual(float(risk[1, 0, 1, 2]), 0.75)
        self.assertEqual(float(known[1, 0, 1, 2]), 1.0)
        self.assertEqual(float(known[1, 1, 1, 2]), 0.0)

    def test_binary_metrics_masks_unknown(self):
        probability = np.asarray([0.9, 0.9, 0.1])
        truth = np.asarray([1.0, 0.0, 1.0])
        known = np.asarray([1, 0, 1])

        result = binary_metrics(probability, truth, known)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertAlmostEqual(result["f1"], 2.0 / 3.0)

    def test_temporal_student_output_shape(self):
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 576, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((2, 2)),
        )
        model.temporal_depthwise = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        model.pointwise = torch.nn.Conv2d(576, 128, kernel_size=1)
        model.pool = torch.nn.AdaptiveAvgPool2d(1)
        model.dropout = torch.nn.Identity()
        model.head = torch.nn.Linear(128, 2 * 3 * 3 * 6 * 6)

        risk, known = model(torch.zeros(2, 5, 3, 8, 8))

        self.assertEqual(tuple(risk.shape), (2, 3, 3, 6, 6))
        self.assertEqual(tuple(known.shape), (2, 3, 3, 6, 6))

    def test_train_prior_metrics_use_train_cell_prior_on_dev(self):
        labels = {}
        for horizon in ("current", "near", "far"):
            labels[horizon] = {
                "known_target": np.ones((3, 6, 6), dtype=np.uint8).tolist(),
                "risk_score_target_nullable": np.ones(
                    (3, 6, 6), dtype=np.float32
                ).tolist(),
            }
        record = {"labels": labels}

        result = train_prior_metrics([record], [record])

        self.assertEqual(result["risk_all"]["fn"], 0)
        self.assertEqual(result["risk_all"]["fp"], 0)
        self.assertEqual(result["risk_all"]["f1"], 1.0)
        self.assertEqual(result["future_body_head_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
