from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_p3_temporal_development_screen_r0 import (
    EXPECTED_TRAINING,
    TRAINING_RESULT_SCHEMA,
    TRAINING_RESULT_FIELDS,
    _class_weights,
    _teacher_index,
    _lexical_inside,
)


def _frame(state: str) -> dict:
    return {"geometry_state": [state, state, state], "geometry_target_valid": [True, True, True]}


class DevelopmentScreenTrainerTest(unittest.TestCase):
    def test_fixed_training_contract(self) -> None:
        self.assertEqual(EXPECTED_TRAINING["epochs"], 3)
        self.assertEqual(EXPECTED_TRAINING["batch_size"], 1)
        self.assertEqual(EXPECTED_TRAINING["gradient_accumulation_steps"], 8)
        self.assertEqual(EXPECTED_TRAINING["learning_rate"], 0.00002)

    def test_class_weights_fail_closed_when_transition_support_missing(self) -> None:
        clips = [{"frames": [_frame("CLEAR") for _ in range(4)]}]
        with self.assertRaisesRegex(ValueError, "positive train support"):
            _class_weights(clips)

    def test_class_weights_are_nine_classes_when_all_present(self) -> None:
        states = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
        clips = []
        for left in states:
            for right in states:
                clips.append({"frames": [_frame(left), _frame(right), _frame(left), _frame(right)]})
        weights = _class_weights(clips)
        self.assertEqual(weights.shape, (9,))
        self.assertTrue(torch.isfinite(weights).all())

    def test_teacher_reference_is_exact_npy_index(self) -> None:
        self.assertEqual(_teacher_index("npy-index:7"), 7)
        with self.assertRaisesRegex(ValueError, "teacher depth reference"):
            _teacher_index("7")
        with self.assertRaisesRegex(ValueError, "teacher depth index"):
            _teacher_index("npy-index:-1")

    def test_lexical_input_boundary_accepts_artifacts_junction_path(self) -> None:
        root = Path(r"E:\linnan\linnan")
        path = _lexical_inside(root, "artifacts.local/evidence/value.json")
        self.assertTrue(path.is_relative_to(root))
        with self.assertRaisesRegex(ValueError, "path leaves repository"):
            _lexical_inside(root, "../escape.json")

    def test_output_boundary_is_lexical_and_overwrite_protected(self) -> None:
        root = Path(r"E:\linnan\linnan")
        path = _lexical_inside(root, "artifacts.local/evidence/new-output")
        self.assertTrue(path.is_relative_to(root))

    def test_training_receipt_contract_is_auditable(self) -> None:
        required = {
            "schema", "protocol_sha256", "evidence_limit", "activation_bindings_sha256",
            "train_manifest_sha256", "validation_manifest_sha256", "a2_checkpoint_sha256",
            "teacher_depth_sha256", "seed", "epochs_completed", "best_epoch",
            "best_validation_composite_total", "history", "checkpoint",
            "training_duration_s", "sealed_holdout_opened", "terminal",
        }
        self.assertEqual(TRAINING_RESULT_SCHEMA, "blindassist_p3_temporal_development_screen_r0_training_result")
        self.assertEqual(TRAINING_RESULT_FIELDS, required)


if __name__ == "__main__":
    unittest.main()
