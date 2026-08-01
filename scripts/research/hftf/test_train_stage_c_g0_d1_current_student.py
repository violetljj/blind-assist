from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_stage_c_g0_d1_current_student as target  # noqa: E402
from train_stage_c_g0_d1_current_student import (  # noqa: E402
    TemporalStudent,
    _decode_targets,
    _corpus_checks_pass,
    _expected_source_maps,
    _loss_parameters,
    _losses,
    _model_state_sha256,
    _parameter_count,
    _seed_everything,
    _selection_key,
    _validate_runtime_contract,
    _validate_source_partition,
)


def _labels(
    risk_value: int,
    clearance_value: float,
) -> dict:
    known = np.ones((2, 6, 6), dtype=int).tolist()
    risk = np.full((2, 6, 6), risk_value, dtype=object).tolist()
    clearance = np.full(
        (2, 6, 6), clearance_value, dtype=object
    ).tolist()
    return {
        "known_target": known,
        "risk_target_nullable": risk,
        "clearance_target_m_nullable": clearance,
    }


def _mixed_labels() -> dict:
    labels = _labels(0, 0.1)
    labels["risk_target_nullable"][0][0][0] = 1
    labels["risk_target_nullable"][1][0][0] = 1
    labels["clearance_target_m_nullable"][0][0][0] = -0.1
    labels["clearance_target_m_nullable"][1][0][0] = -0.1
    return labels


def _records() -> tuple[list[dict], dict[str, str]]:
    roles = {
        **{f"train-{index}": "train" for index in range(6)},
        **{
            f"selection-{index}": "model_selection"
            for index in range(3)
        },
    }
    records = []
    for session_id, role in roles.items():
        for frame in range(25):
            records.append(
                {
                    "sample_id": f"{session_id}-{frame}",
                    "session_id": session_id,
                    "role": role,
                    "source_frame_index": frame,
                    "manifest_id": f"manifest-{session_id}-{frame}",
                    "current_rgb": {
                        "path": f"C:/current/{session_id}-{frame}.png",
                        "sha256": "a" * 64,
                    },
                    "labels": _mixed_labels(),
                }
            )
    return records, roles


class StageCG0D1CurrentStudentTest(unittest.TestCase):
    def test_decode_preserves_null_unknown(self) -> None:
        labels = _labels(0, 0.2)
        labels["known_target"][0][0][0] = 0
        labels["risk_target_nullable"][0][0][0] = None
        labels["clearance_target_m_nullable"][0][0][0] = None
        risk, clearance, known = _decode_targets(labels)
        self.assertEqual(0.0, known[0, 0, 0])
        self.assertEqual(0.0, risk[0, 0, 0])
        self.assertEqual(0.0, clearance[0, 0, 0])

    def test_known_clearance_sign_must_match_risk(self) -> None:
        with self.assertRaisesRegex(ValueError, "sign disagree"):
            _decode_targets(_labels(1, 0.1))

    def test_train_only_clearance_weights_are_normalized(self) -> None:
        records = [
            {"labels": _labels(1, -0.1)},
            {"labels": _labels(0, 0.1)},
        ]
        values = _loss_parameters(records)
        self.assertEqual([36, 36], values["positive"])
        self.assertEqual([36, 36], values["negative"])
        self.assertEqual([1.0, 1.0], values["positive_weight"])
        self.assertEqual([1.0, 1.0], values["risk_base_weight"])
        self.assertEqual([1.0, 1.0], values["safe_base_weight"])
        self.assertEqual(
            [0.5, 0.5],
            values["clearance_weight_normalization"],
        )

    def test_unknown_cells_do_not_change_task_losses(self) -> None:
        task = torch.zeros((1, 2, 6, 6))
        known_logits = torch.zeros_like(task)
        risk = torch.zeros_like(task)
        clearance = torch.zeros_like(task)
        known = torch.ones_like(task)
        known[0, 0, 0, 0] = 0.0
        parameters = {
            key: torch.ones((1, 2, 1, 1))
            for key in (
                "positive_weight",
                "risk_base_weight",
                "safe_base_weight",
                "clearance_weight_normalization",
            )
        }
        baseline = _losses(
            "SIGNED_CLEARANCE_CURRENT",
            task,
            known_logits,
            risk,
            clearance,
            known,
            parameters,
        )
        risk[0, 0, 0, 0] = 1.0
        clearance[0, 0, 0, 0] = -0.5
        changed = _losses(
            "SIGNED_CLEARANCE_CURRENT",
            task,
            known_logits,
            risk,
            clearance,
            known,
            parameters,
        )
        self.assertEqual(float(baseline["task"]), float(changed["task"]))

    def test_selection_is_source_first_then_micro_then_earliest(self) -> None:
        earlier = {
            "risk_source_macro_f1": 0.5,
            "risk_worst_source_f1": 0.4,
            "risk_micro": {"f1": 0.6},
            "clearance_source_macro_mae_m": {"overall": 0.2},
        }
        later = {
            **earlier,
            "risk_micro": {"f1": 0.61},
        }
        self.assertGreater(
            _selection_key("DIRECT_RISK_CURRENT", later, 2),
            _selection_key("DIRECT_RISK_CURRENT", earlier, 1),
        )
        self.assertGreater(
            _selection_key("DIRECT_RISK_CURRENT", earlier, 1),
            _selection_key("DIRECT_RISK_CURRENT", earlier, 2),
        )

    def test_selection_uses_macro_worst_and_clearance_tie_in_order(
        self,
    ) -> None:
        baseline = {
            "risk_source_macro_f1": 0.5,
            "risk_worst_source_f1": 0.4,
            "risk_micro": {"f1": 0.6},
            "clearance_source_macro_mae_m": {"overall": 0.2},
        }
        for field, value in (
            ("risk_source_macro_f1", 0.51),
            ("risk_worst_source_f1", 0.41),
        ):
            better = {
                **baseline,
                field: value,
                "risk_micro": {"f1": 0.0},
            }
            self.assertGreater(
                _selection_key("DIRECT_RISK_CURRENT", better, 30),
                _selection_key("DIRECT_RISK_CURRENT", baseline, 1),
            )
        lower_mae = {
            **baseline,
            "clearance_source_macro_mae_m": {"overall": 0.19},
        }
        self.assertGreater(
            _selection_key("SIGNED_CLEARANCE_CURRENT", lower_mae, 30),
            _selection_key(
                "SIGNED_CLEARANCE_CURRENT", baseline, 1
            ),
        )

    def test_exact_source_partition_rejects_overlap_extra_and_degenerate(
        self,
    ) -> None:
        records, roles = _records()
        _validate_source_partition(records, roles)

        overlap = [dict(record) for record in records]
        overlap[150] = {**overlap[150], "session_id": "train-0"}
        with self.assertRaisesRegex(ValueError, "source|partition"):
            _validate_source_partition(overlap, roles)

        extra = [dict(record) for record in records]
        extra[0] = {**extra[0], "teacher_depth": "forbidden"}
        with self.assertRaisesRegex(ValueError, "schema"):
            _validate_source_partition(extra, roles)

        all_safe = [dict(record) for record in records]
        all_safe = [
            {**record, "labels": _labels(0, 0.1)}
            for record in all_safe
        ]
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            _validate_source_partition(all_safe, roles)

    def test_runtime_contract_is_exact_and_actual_versions_fail_closed(
        self,
    ) -> None:
        design = {
            "runtime_and_model_contract": {
                "runtime": dict(target.FROZEN_RUNTIME)
            }
        }
        with (
            mock.patch.object(
                target.torch,
                "__version__",
                target.FROZEN_RUNTIME["torch_version"],
            ),
            mock.patch.object(
                target.torchvision,
                "__version__",
                target.FROZEN_RUNTIME["torchvision_version"],
            ),
            mock.patch.object(
                target.torch.cuda, "is_available", return_value=True
            ),
        ):
            _validate_runtime_contract(design)
        drifted = {
            "runtime_and_model_contract": {
                "runtime": {
                    **target.FROZEN_RUNTIME,
                    "dataloader_workers": 1,
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "runtime contract"):
            _validate_runtime_contract(drifted)

    def test_source_plan_maps_and_validation_checks_fail_closed(
        self,
    ) -> None:
        repository = Path(target.__file__).resolve().parents[3]
        design_path = (
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_D1_"
            "2026-08-01.json"
        )
        design = target._load_json(design_path)
        roles, frames, forbidden = _expected_source_maps(
            design_path, design
        )
        self.assertEqual(
            list(roles.values()),
            ["train"] * 6 + ["model_selection"] * 3,
        )
        self.assertEqual(len(frames), 9)
        self.assertEqual(len(forbidden), 6)
        self.assertFalse(set(roles) & forbidden)
        self.assertFalse(_corpus_checks_pass({}))
        self.assertFalse(
            _corpus_checks_pass(
                {key: True for key in list(target.CORPUS_VALIDATION_CHECKS)[:-1]}
            )
        )
        self.assertTrue(
            _corpus_checks_pass(
                {
                    key: True
                    for key in target.CORPUS_VALIDATION_CHECKS
                }
            )
        )

    def test_same_seed_produces_same_model_initial_state(self) -> None:
        _seed_everything(17)
        first = TemporalStudent(None)
        torch.nn.init.zeros_(first.head.bias)
        first_hash = _model_state_sha256(first)
        _seed_everything(17)
        second = TemporalStudent(None)
        torch.nn.init.zeros_(second.head.bias)
        self.assertEqual(first_hash, _model_state_sha256(second))
        self.assertEqual(1_022_448, _parameter_count(second))
        self.assertTrue(torch.equal(second.head.bias, torch.zeros(144)))


if __name__ == "__main__":
    unittest.main()
