import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stage_c_d5_tartanground_development_student import (
    EarlyPairStem,
    TemporalStudent,
    binary_metrics,
    decode_labels,
    known_positive_weights,
    losses,
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

    def test_binary_metrics_include_threshold_free_ranking(self):
        result = binary_metrics(
            np.asarray([0.9, 0.8, 0.2, 0.1]),
            np.asarray([1.0, 0.0, 1.0, 0.0]),
            np.ones(4, dtype=np.uint8),
        )

        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(
            result["average_precision"],
            (1.0 + 2.0 / 3.0) / 2.0,
        )

    def test_balanced_known_loss_increases_positive_known_penalty(self):
        shape = (1, 3, 3, 1, 1)
        risk_logits = torch.zeros(shape)
        known_logits = torch.zeros(shape)
        risk = torch.zeros(shape)
        known = torch.ones(shape)
        risk_weights = torch.ones((3, 3))

        _, _, plain = losses(
            risk_logits,
            known_logits,
            risk,
            known,
            risk_weights,
        )
        _, _, balanced = losses(
            risk_logits,
            known_logits,
            risk,
            known,
            risk_weights,
            torch.full((3, 3), 2.0),
        )

        self.assertGreater(float(balanced), float(plain))

    def test_sqrt_balanced_known_weight_is_log_space_halfway(self):
        labels = {}
        for horizon in ("current", "near", "far"):
            known = np.zeros((3, 6, 6), dtype=np.uint8)
            known[:, :3, :3] = 1
            risk = np.full((3, 6, 6), None, dtype=object)
            risk[known.astype(bool)] = 0.0
            labels[horizon] = {
                "known_target": known.tolist(),
                "risk_score_target_nullable": risk.tolist(),
            }

        balanced = known_positive_weights([{"labels": labels}])
        sqrt_balanced = known_positive_weights(
            [{"labels": labels}],
            power=0.5,
        )

        self.assertTrue(torch.allclose(balanced, torch.full((3, 3), 3.0)))
        self.assertTrue(
            torch.allclose(
                sqrt_balanced,
                torch.full((3, 3), 3.0**0.5),
            )
        )

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

    def test_directional_student_preserves_direction_axis(self):
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.architecture = "directional"
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 576, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((2, 7)),
        )
        model.temporal_depthwise = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        model.pointwise = torch.nn.Conv2d(576, 128, kernel_size=1)
        model.pool = torch.nn.AdaptiveAvgPool2d((1, 6))
        model.dropout = torch.nn.Identity()
        model.head = torch.nn.Conv1d(
            128,
            2 * 3 * 3 * 6,
            kernel_size=1,
        )

        risk, known = model(torch.zeros(2, 5, 3, 8, 14))

        self.assertEqual(tuple(risk.shape), (2, 3, 3, 6, 6))
        self.assertEqual(tuple(known.shape), (2, 3, 3, 6, 6))

    def test_grid_student_preserves_height_and_direction_axes(self):
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.architecture = "grid"
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 576, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((4, 7)),
        )
        model.temporal_depthwise = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        model.pointwise = torch.nn.Conv2d(576, 128, kernel_size=1)
        model.pool = torch.nn.AdaptiveAvgPool2d((3, 6))
        model.dropout = torch.nn.Identity()
        model.head = torch.nn.Conv2d(
            128,
            2 * 3 * 6,
            kernel_size=1,
        )

        risk, known = model(torch.zeros(2, 5, 3, 16, 28))

        self.assertEqual(tuple(risk.shape), (2, 3, 3, 6, 6))
        self.assertEqual(tuple(known.shape), (2, 3, 3, 6, 6))

    def test_current_residual_starts_at_repeated_current_baseline(self):
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.architecture = "directional"
        model.temporal_mode = "current_residual"
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 576, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((2, 7)),
        )
        model.temporal_depthwise = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        model.temporal_residual = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(4, 1, 1),
            groups=576,
            bias=False,
        )
        torch.nn.init.zeros_(model.temporal_residual.weight)
        model.pointwise = torch.nn.Conv2d(576, 128, kernel_size=1)
        model.pool = torch.nn.AdaptiveAvgPool2d((1, 6))
        model.dropout = torch.nn.Identity()
        model.head = torch.nn.Conv1d(
            128,
            2 * 3 * 3 * 6,
            kernel_size=1,
        )
        history = torch.randn(2, 5, 3, 8, 14)
        repeated_current = history[:, -1:].repeat(1, 5, 1, 1, 1)

        history_risk, history_known = model(history)
        current_risk, current_known = model(repeated_current)

        torch.testing.assert_close(history_risk, current_risk)
        torch.testing.assert_close(history_known, current_known)

    def test_spatial_residual_starts_at_repeated_current_baseline(self):
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.architecture = "directional"
        model.temporal_mode = "current_spatial_residual"
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 576, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((4, 7)),
        )
        model.temporal_depthwise = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        model.temporal_residual = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(4, 3, 3),
            padding=(0, 1, 1),
            groups=576,
            bias=False,
        )
        torch.nn.init.zeros_(model.temporal_residual.weight)
        model.pointwise = torch.nn.Conv2d(576, 128, kernel_size=1)
        model.pool = torch.nn.AdaptiveAvgPool2d((1, 6))
        model.dropout = torch.nn.Identity()
        model.head = torch.nn.Conv1d(
            128,
            2 * 3 * 3 * 6,
            kernel_size=1,
        )
        history = torch.randn(2, 5, 3, 16, 28)
        repeated_current = history[:, -1:].repeat(1, 5, 1, 1, 1)

        history_risk, history_known = model(history)
        current_risk, current_known = model(repeated_current)

        torch.testing.assert_close(history_risk, current_risk)
        torch.testing.assert_close(history_known, current_known)

    def test_early_pair_starts_at_repeated_current_baseline(self):
        model = TemporalStudent.__new__(TemporalStudent)
        torch.nn.Module.__init__(model)
        model.architecture = "directional"
        model.temporal_mode = "early_pair"
        model.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 576, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d((4, 7)),
        )
        model.temporal_depthwise = torch.nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        model.pointwise = torch.nn.Conv2d(576, 128, kernel_size=1)
        model.early_pair_stem = EarlyPairStem()
        model.early_pair_output = torch.nn.Conv2d(
            128,
            128,
            kernel_size=1,
            bias=False,
        )
        torch.nn.init.zeros_(model.early_pair_output.weight)
        model.pool = torch.nn.AdaptiveAvgPool2d((1, 6))
        model.dropout = torch.nn.Identity()
        model.head = torch.nn.Conv1d(
            128,
            2 * 3 * 3 * 6,
            kernel_size=1,
        )
        history = torch.randn(2, 5, 3, 32, 56)
        repeated_current = history[:, -1:].repeat(1, 5, 1, 1, 1)

        history_risk, history_known = model(history)
        current_risk, current_known = model(repeated_current)

        torch.testing.assert_close(history_risk, current_risk)
        torch.testing.assert_close(history_known, current_known)

    def test_early_pair_stem_preserves_lightweight_spatial_output(self):
        stem = EarlyPairStem()
        output = stem(torch.zeros(2, 12, 128, 224))

        self.assertEqual((2, 128, 8, 14), tuple(output.shape))
        self.assertLess(
            sum(parameter.numel() for parameter in stem.parameters()),
            20_000,
        )

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
