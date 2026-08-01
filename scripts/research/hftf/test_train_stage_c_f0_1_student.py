from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stage_c_f0_1_student import (
    TemporalStudent,
    _arm_target,
    _class_weights,
    _contract_parent_hashes,
    _decode_label,
    _flip_targets,
    _losses,
    _metric_counts,
    _metrics_from_counts,
    _sample_augmentation,
    _sha256,
    _validate_arm_history_images,
)


def _label(positive: bool) -> dict:
    known = np.ones((2, 6, 6), dtype=int)
    risk = np.zeros((2, 6, 6), dtype=object)
    risk[:] = 1 if positive else 0
    return {
        "known_target": known.tolist(),
        "risk_target_nullable": risk.tolist(),
    }


class StageCF01StudentTrainingTest(unittest.TestCase):
    def test_frozen_arm_inputs_and_targets(self) -> None:
        self.assertEqual(("current", True), _arm_target("SF_CURRENT"))
        self.assertEqual(("future", True), _arm_target("SF_FUTURE"))
        self.assertEqual(("future", False), _arm_target("HIST_FUTURE"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            _arm_target("OTHER")

    def test_nullable_label_decode_and_theta_flip(self) -> None:
        known = np.zeros((2, 6, 6), dtype=int)
        risk = np.full((2, 6, 6), None, dtype=object)
        known[0, 0, 1] = 1
        risk[0, 0, 1] = 1
        risk_tensor, known_tensor = _decode_label(
            {
                "known_target": known.tolist(),
                "risk_target_nullable": risk.tolist(),
            }
        )
        flipped_risk, flipped_known = _flip_targets(
            risk_tensor, known_tensor
        )
        self.assertEqual(1, flipped_risk[0, 5, 1])
        self.assertEqual(1, flipped_known[0, 5, 1])

    def test_augmentation_is_seed_epoch_sample_deterministic(self) -> None:
        first = _sample_augmentation(17, 4, "sample")
        second = _sample_augmentation(17, 4, "sample")
        changed = _sample_augmentation(17, 5, "sample")
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_history_image_preflight_binds_current_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(path)
            item = {
                "image_path": str(path),
                "image_sha256": _sha256(path),
            }
            history = [
                {**item, "relative_time_s": relative_time}
                for relative_time in (-0.8, -0.6, -0.4, -0.2, 0.0)
            ]
            records = [{"history_rgb": history}]
            self.assertEqual(
                1, _validate_arm_history_images(records, "HIST_FUTURE")
            )
            history[-1]["image_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _validate_arm_history_images(records, "SF_CURRENT")

    def test_checkpoint_parent_hashes_are_complete_and_valid(self) -> None:
        names = (
            "f0_protocol",
            "f0_1_protocol",
            "corpus_contract",
            "student_samples",
            "corpus_validation",
        )
        contract = {
            "parents": {
                name: {"sha256": f"{index:x}" * 64}
                for index, name in enumerate(names, start=1)
            }
        }
        self.assertEqual(set(names), set(_contract_parent_hashes(contract)))
        contract["parents"]["f0_protocol"]["sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "parent hash"):
            _contract_parent_hashes(contract)

    def test_train_only_class_weights_are_height_specific(self) -> None:
        records = [
            {"labels": {"future": _label(True)}},
            {"labels": {"future": _label(False)}},
        ]
        self.assertEqual([1.0, 1.0], _class_weights(records, "future"))

    def test_risk_loss_masks_unknown_and_applies_height_weights(self) -> None:
        risk_logits = torch.zeros((1, 2, 1, 2))
        known_logits = torch.zeros((1, 2, 1, 2))
        risk = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
        known = torch.tensor([[[[1.0, 0.0]], [[1.0, 1.0]]]])
        weights = torch.tensor([2.0, 4.0]).view(1, 2, 1, 1)
        total, risk_loss, known_loss = _losses(
            risk_logits, known_logits, risk, known, weights
        )
        expected_risk = (2.0 + 4.0 + 1.0) * np.log(2.0) / 3.0
        self.assertAlmostEqual(expected_risk, float(risk_loss), places=6)
        self.assertAlmostEqual(np.log(2.0), float(known_loss), places=6)
        self.assertAlmostEqual(
            expected_risk + np.log(2.0), float(total), places=6
        )

    def test_micro_metrics(self) -> None:
        probabilities = torch.tensor([0.9, 0.8, 0.2, 0.1])
        risk = torch.tensor([1.0, 0.0, 1.0, 0.0])
        known = torch.ones(4)
        counts = _metric_counts(probabilities, risk, known)
        self.assertEqual({"tp": 1, "fp": 1, "fn": 1, "tn": 1}, counts)
        metrics = _metrics_from_counts(counts)
        self.assertEqual(0.5, metrics["f1"])
        self.assertEqual(0.5, metrics["recall"])
        self.assertEqual(0.5, metrics["false_positive_rate"])

    def test_model_has_same_output_shape_for_any_arm_input(self) -> None:
        torch.manual_seed(1)
        model = TemporalStudent(pretrained_path=None)
        model.eval()
        with torch.no_grad():
            risk, known = model(torch.randn(1, 5, 3, 64, 64))
        self.assertEqual((1, 2, 6, 6), tuple(risk.shape))
        self.assertEqual((1, 2, 6, 6), tuple(known.shape))


if __name__ == "__main__":
    unittest.main()
