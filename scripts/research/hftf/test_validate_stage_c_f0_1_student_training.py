from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_stage_c_f0_1_student_training import (
    TemporalStudent,
    _assert_finite,
    _assert_tensor_tree_finite,
    _expected_runs,
    _selected_epoch,
    _validate_implementation_path,
    _validate_model_state,
    _validate_optimizer_state,
)


class StageCF01StudentTrainingValidationTest(unittest.TestCase):
    def test_expected_order_is_seed_major(self) -> None:
        runs = _expected_runs()
        self.assertEqual(
            [
                (17, "SF_CURRENT"),
                (17, "SF_FUTURE"),
                (17, "HIST_FUTURE"),
            ],
            runs[:3],
        )
        self.assertEqual((43, "HIST_FUTURE"), runs[-1])
        self.assertEqual(9, len(runs))

    def test_selected_epoch_uses_earliest_exact_f1_tie(self) -> None:
        history = [
            {
                "epoch": epoch,
                "dev": {"risk_micro": {"f1": 0.3 if epoch in (4, 7) else 0.2}},
            }
            for epoch in range(1, 31)
        ]
        self.assertEqual((4, 0.3), _selected_epoch(history))

    def test_recursive_finite_gate_rejects_nan(self) -> None:
        _assert_finite({"finite": [1.0, 2, True, None]})
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            _assert_finite({"bad": [math.nan]})
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            _assert_tensor_tree_finite({"bad": math.inf}, "checkpoint")

    def test_incomplete_optimizer_state_is_rejected(self) -> None:
        model = TemporalStudent(pretrained_path=None)
        incomplete = {
            "state": {},
            "param_groups": [
                {"lr": 3e-5, "weight_decay": 1e-4},
                {"lr": 3e-4, "weight_decay": 1e-4},
            ],
        }
        with self.assertRaisesRegex(ValueError, "group 0"):
            _validate_optimizer_state(model, incomplete, selected_epoch=1)

    def test_model_state_dtype_drift_is_rejected_before_strict_load(self) -> None:
        model = TemporalStudent(pretrained_path=None)
        state = model.state_dict()
        first_key = next(iter(state))
        state[first_key] = state[first_key].double()
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            _validate_model_state(model, state)

    def test_implementation_receipt_must_be_imported_trainer(self) -> None:
        with self.assertRaisesRegex(ValueError, "imported trainer"):
            _validate_implementation_path(Path(__file__))


if __name__ == "__main__":
    unittest.main()
