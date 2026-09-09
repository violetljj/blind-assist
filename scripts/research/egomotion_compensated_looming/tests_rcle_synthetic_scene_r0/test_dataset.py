from __future__ import annotations

import json
from pathlib import Path
import platform
import tempfile
import unittest

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.dataset import (
    build_dataset,
    build_dataset_governed,
    validate_sealed_activation,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.evaluator import (
    evaluate_dataset,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.validator import (
    compare_development_generations,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.sealed_executor import (
    execute_sealed_once,
)


SPEC = (
    Path(__file__).resolve().parents[1]
    / "rcle_synthetic_scene_r0"
    / "dataset_spec.json"
)


def _small_spec(root: Path) -> Path:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["rendering"].update(
        {
            "width": 96,
            "height": 72,
            "fps": 2,
            "duration_seconds": 1,
            "frame_count": 3,
            "pair_count_per_sequence": 2,
        }
    )
    value["rendering"]["intrinsics"] = {
        "fx_pixels": 84.0,
        "fy_pixels": 84.0,
        "cx_pixels": 47.5,
        "cy_pixels": 35.5,
    }
    value["splits"]["development"]["scene_families"] = [
        value["splits"]["development"]["scene_families"][0]
    ]
    value["motions"] = value["motions"][:2]
    path = root / "small_spec.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sealed_activation(
    root: Path, spec_path: Path, output: Path
) -> tuple[Path, Path, Path]:
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    claim_root = root / "claims"
    value["sealed_activation"]["claim_ledger_root"] = str(claim_root)
    spec_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    module_root = SPEC.parent
    claim_id = "RCLE_SYNTHETIC_SCENE_R0_TEST_CLAIM"
    claim_path = claim_root / "test_claim.json"
    governance_path = root / "governance.json"
    write_json(
        governance_path,
        {
            "schema": (
                "rcle.synthetic_scene.sealed_governance_reconciliation.v1"
            ),
            "protocol_id": value["protocol_id"],
            "status": "PASS",
            "sealed_execution_authorized": True,
        },
    )
    schema_path = module_root / "sealed_activation.schema.json"
    review_path = root / "review.json"
    write_json(
        review_path,
        {
            "schema": "rcle.synthetic_scene.sealed_activation_review.v1",
            "protocol_id": value["protocol_id"],
            "claim_id": claim_id,
            "authorizer_id": "unit-test-independent-reviewer",
            "decision": "PASS",
            "execution_authorized": True,
            "sealed_activation_schema_sha256": sha256_file(schema_path),
            "governance_reconciliation_path": str(governance_path.resolve()),
            "governance_reconciliation_sha256": sha256_file(
                governance_path
            ),
            "sealed_executor_sha256": sha256_file(
                module_root / "sealed_executor.py"
            ),
            "cli_sha256": sha256_file(module_root / "cli.py"),
        },
    )
    activation_path = root / "activation.json"
    write_json(
        activation_path,
        {
            "schema": value["sealed_activation"]["receipt_schema"],
            "protocol_id": value["protocol_id"],
            "split": "sealed",
            "claim_id": claim_id,
            "claim_path": str(claim_path.resolve()),
            "authorizer_id": "unit-test-independent-reviewer",
            "current_governance_reconciled": True,
            "execution_authorized": True,
            "dataset_spec_sha256": sha256_file(spec_path),
            "renderer_sha256": sha256_file(module_root / "renderer.py"),
            "dataset_builder_sha256": sha256_file(module_root / "dataset.py"),
            "truth_builder_sha256": sha256_file(module_root / "truth.py"),
            "algorithm_adapter_sha256": sha256_file(
                module_root / "evaluator.py"
            ),
            "validator_sha256": sha256_file(module_root / "validator.py"),
            "dependency_identity": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "canonical_output_root": str(output.resolve()),
            "output_root_preflight_empty": True,
            "independent_review_receipt_path": str(review_path.resolve()),
            "independent_review_receipt_sha256": sha256_file(review_path),
        },
    )
    return activation_path, claim_path, review_path


class DatasetTest(unittest.TestCase):
    def test_generation_is_manifest_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _small_spec(root)
            first = root / "first"
            second = root / "second"
            build_dataset(spec, first)
            build_dataset(spec, second)
            for name in (
                "scene_manifest.jsonl",
                "frame_manifest.jsonl",
                "algorithm_input_manifest.jsonl",
            ):
                self.assertEqual(sha256_file(first / name), sha256_file(second / name))
            first_frames = read_jsonl(first / "frame_manifest.jsonl")
            second_frames = read_jsonl(second / "frame_manifest.jsonl")
            self.assertEqual(
                [
                    (
                        row["rgb_sha256"],
                        row["depth_npy_sha256"],
                        row["surface_id_sha256"],
                    )
                    for row in first_frames
                ],
                [
                    (
                        row["rgb_sha256"],
                        row["depth_npy_sha256"],
                        row["surface_id_sha256"],
                    )
                    for row in second_frames
                ],
            )
            self.assertEqual(
                compare_development_generations(first, second)["status"], "PASS"
            )

    def test_algorithm_manifest_is_strictly_staged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            build_dataset(_small_spec(root), dataset)
            rows = read_jsonl(dataset / "algorithm_input_manifest.jsonl")
            forbidden = {
                "depth_npy_path",
                "surface_id_path",
                "role",
                "motion_id",
                "scene_id",
                "t_world_camera",
                "t_camera_world",
            }
            self.assertTrue(rows)
            for row in rows:
                self.assertFalse(forbidden & row.keys())
                self.assertTrue(row["rgb_path"].startswith("algorithm_staging/"))
                self.assertTrue(
                    row["valid_mask_path"].startswith("algorithm_staging/")
                )

    def test_sealed_generation_fails_without_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                PermissionError, "SEALED_EXECUTION_NOT_AUTHORIZED"
            ):
                build_dataset(
                    _small_spec(root), root / "sealed", split="sealed"
                )
            self.assertFalse((root / "sealed").exists())

    def test_invalid_activation_schema_does_not_consume_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _small_spec(root)
            output = root / "sealed"
            activation, claim, _ = _sealed_activation(root, spec, output)
            value = read_json(activation)
            value["dependency_identity"]["opencv"] = "invalid"
            write_json(activation, value)
            with self.assertRaisesRegex(
                ValueError, "ACTIVATION_SCHEMA_PATTERN"
            ):
                build_dataset_governed(
                    spec, output, split="sealed", activation=activation
                )
            self.assertFalse(claim.exists())
            self.assertFalse(output.exists())

    def test_sealed_preflight_passes_without_consuming_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _small_spec(root)
            output = root / "sealed"
            activation, claim, _ = _sealed_activation(root, spec, output)
            report = validate_sealed_activation(spec, output, activation)
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["claim_consumed"])
            self.assertFalse(claim.exists())

    def test_consumed_sealed_failure_writes_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _small_spec(root)
            value = read_json(spec)
            value["resource_preflight"]["supported_python"] = "0.0.0"
            write_json(spec, value)
            output = root / "sealed"
            activation, claim, _ = _sealed_activation(root, spec, output)
            with self.assertRaisesRegex(RuntimeError, "UNSUPPORTED_PYTHON"):
                build_dataset_governed(
                    spec, output, split="sealed", activation=activation
                )
            self.assertTrue(claim.is_file())
            terminal = claim.with_name(claim.stem + ".terminal.json")
            self.assertEqual(
                read_json(terminal)["status"],
                "INVALID_SEALED_EXECUTION_FAIL_CLOSED",
            )

    def test_evaluator_failure_writes_consumed_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _small_spec(root)
            output = root / "sealed"
            activation, claim, _ = _sealed_activation(root, spec, output)
            with self.assertRaisesRegex(
                ValueError, "FIXED_PAIR_DENOMINATOR_DRIFT"
            ):
                execute_sealed_once(spec, output, activation)
            terminal = claim.with_name(claim.stem + ".terminal.json")
            self.assertEqual(
                read_json(terminal)["status"],
                "INVALID_SEALED_EXECUTION_FAIL_CLOSED",
            )
            with self.assertRaisesRegex(
                PermissionError, "SEALED_CLAIM_ALREADY_CONSUMED"
            ):
                execute_sealed_once(spec, output, activation)

    def test_small_sealed_end_to_end_uses_sealed_outcome_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _small_spec(root)
            value = read_json(spec)
            value["rendering"].update(
                {
                    "fps": 10,
                    "duration_seconds": 10,
                    "frame_count": 101,
                    "pair_count_per_sequence": 100,
                }
            )
            value["splits"]["sealed"]["scene_families"] = [
                value["splits"]["sealed"]["scene_families"][0]
            ]
            value["motions"] = json.loads(
                SPEC.read_text(encoding="utf-8")
            )["motions"][:4]
            write_json(spec, value)
            output = root / "sealed"
            activation, _, _ = _sealed_activation(root, spec, output)
            terminal = execute_sealed_once(spec, output, activation)
            self.assertTrue(
                terminal["scientific_outcome"].startswith(
                    "SYNTHETIC_MECHANISM_SEALED_"
                )
            )
            self.assertEqual(
                terminal["status"],
                "SYNTHETIC_SEALED_EXECUTION_COMPLETE",
            )
            with self.assertRaisesRegex(
                PermissionError,
                "SEALED_EVALUATION_REQUIRES_GOVERNED_EXECUTOR",
            ):
                evaluate_dataset(output)
            with self.assertRaisesRegex(
                PermissionError, "SEALED_CLAIM_ALREADY_CONSUMED"
            ):
                execute_sealed_once(spec, output, activation)

    def test_receipt_records_candidate_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            build_dataset(_small_spec(root), dataset)
            receipt = read_json(dataset / "receipt.json")
            self.assertEqual(
                receipt["status"],
                "CANDIDATE_DATASET_GENERATED_VALIDATION_REQUIRED",
            )
            self.assertEqual(
                receipt["authority"]["maximum_claim"],
                "CONTROLLED_SYNTHETIC_CAUSAL_MECHANISM_EVIDENCE_ONLY",
            )


if __name__ == "__main__":
    unittest.main()
