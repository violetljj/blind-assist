from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.research.assistive_geometry_qsf.run_h1_train_canary import (
    apply_gates,
    missing_scientific_support,
    scientific_support_counts,
    select_parent_frames,
    validate_selected_input_files,
)
from scripts.research.assistive_geometry_qsf.validate_h1_train_canary import (
    PROTOCOL_RELATIVE,
    REPO_ROOT,
    ValidationError,
    find_foreign_gpu_processes,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads((REPO_ROOT / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))


class H1TrainCanaryValidationTest(unittest.TestCase):
    def test_tracked_lock_validates_without_runtime_input_probe(self) -> None:
        report = validate_protocol(_protocol(), verify_inputs=False)
        self.assertEqual("H1_TRAIN_CANARY_PROTOCOL_VALID", report["terminal"])
        self.assertTrue(report["parameter_budget"]["exact_match"])

    def test_development_or_extra_input_cannot_be_smuggled_into_lock(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["inputs"]["b1_development_outcome"] = {
            "kind": "PROTECTED_OUTCOME",
            "data_role": "PROJECT_CONSUMED_DEVELOPMENT",
        }
        with self.assertRaisesRegex(ValidationError, "protected input leak"):
            validate_protocol(protocol, verify_inputs=False)
        protocol = copy.deepcopy(_protocol())
        protocol["inputs"]["target_manifest"]["data_role"] = "PROJECT_CONSUMED_DEVELOPMENT"
        with self.assertRaisesRegex(ValidationError, "target data role"):
            validate_protocol(protocol, verify_inputs=False)

    def test_h2_and_combination_authority_fail_closed(self) -> None:
        for key in ("h2_implementation_or_materialization", "h1_plus_h2_training"):
            protocol = copy.deepcopy(_protocol())
            protocol["execution_authority"][key] = True
            with self.subTest(key=key), self.assertRaises(ValidationError):
                validate_protocol(protocol, verify_inputs=False)

    def test_attempt_03_performance_relock_is_exact(self) -> None:
        protocol = copy.deepcopy(_protocol())
        protocol["resource_budget"]["feature_extraction_batch_size"] = 8
        with self.assertRaisesRegex(ValidationError, "batch relock"):
            validate_protocol(protocol, verify_inputs=False)
        protocol = copy.deepcopy(_protocol())
        protocol["previous_attempts"][1]["terminal"] = "H1_TRAIN_CANARY_PERFORMANCE_QUALIFIED"
        with self.assertRaisesRegex(ValidationError, "prior performance"):
            validate_protocol(protocol, verify_inputs=False)
        protocol = copy.deepcopy(_protocol())
        protocol["performance_estimator"]["fixed_setup_scaled_by_frame_ratio"] = True
        with self.assertRaisesRegex(ValidationError, "estimator"):
            validate_protocol(protocol, verify_inputs=False)

    def test_foreign_gpu_process_detection_is_role_specific(self) -> None:
        report = find_foreign_gpu_processes(
            [
                (10, "python train_b1_a0_formal.py --mode formal"),
                (11, "python run_h1_train_canary.py --mode pilot"),
                (12, "pwsh unrelated.ps1"),
            ]
        )
        self.assertEqual([{"pid": 10, "role": "FOREIGN_FORMAL_TRAIN"}], report)

    def test_evenly_spaced_parent_selection_is_exact(self) -> None:
        frames = [
            {"video_id": parent, "frame_stem": f"{index:03d}"}
            for parent in ("a", "b")
            for index in range(10)
        ]
        selected = select_parent_frames(frames, ("a", "b"), 4)
        self.assertEqual(
            [("a", "000"), ("a", "003"), ("a", "006"), ("a", "009"),
             ("b", "000"), ("b", "003"), ("b", "006"), ("b", "009")],
            [(row["video_id"], row["frame_stem"]) for row in selected],
        )

    def test_selected_input_receipts_are_rechecked_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = root / "frame.png"
            target = root / "frame.npz"
            rgb.write_bytes(b"rgb")
            target.write_bytes(b"target")

            def receipt(path: Path) -> dict:
                return {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                }

            frame = {
                "video_id": "p0",
                "frame_stem": "f0",
                "rgb_source": receipt(rgb),
                "target": receipt(target),
            }
            report = validate_selected_input_files([frame])
            self.assertEqual({"rgb_source": 1, "target": 1}, report["verified_file_counts"])
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "byte-size drift"):
                validate_selected_input_files([frame])

    def test_zero_denominators_route_to_not_evaluable_support(self) -> None:
        counts = scientific_support_counts(
            {
                "targets": {
                    "clearance_m": torch.zeros(1, 3),
                    "clearance_valid": torch.zeros(1, 3, dtype=torch.bool),
                    "occupancy": torch.zeros(1, 3, 3),
                    "occupancy_valid": torch.zeros(1, 3, 3, dtype=torch.bool),
                }
            }
        )
        self.assertEqual(
            [
                "clearance_event_count",
                "event_count",
                "occupied_known_count",
                "right_censor_count",
            ],
            missing_scientific_support(counts),
        )

    def test_gate_application_preserves_coverage_and_monotonicity(self) -> None:
        before_fit = {"survival_nll": 1.0}
        after_fit = {"survival_nll": 0.85}
        before_eval = {
            "survival_nll": 1.0,
            "false_clear_rate": 0.2,
            "clearance_mae_m": 0.4,
            "known_coverage": 0.75,
        }
        after_eval = {
            "survival_nll": 0.95,
            "false_clear_rate": 0.2,
            "clearance_mae_m": 0.4,
            "known_coverage": 0.75,
            "horizon_monotonicity_violations": 0,
        }
        gates = apply_gates(before_fit, after_fit, before_eval, after_eval, _protocol()["gates"])
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
