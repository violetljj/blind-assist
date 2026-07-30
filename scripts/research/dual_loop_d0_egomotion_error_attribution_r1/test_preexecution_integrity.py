from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.research.dual_loop_d0_egomotion_error_attribution_r1.validate_implementation_lock import (
    ACTIVATION_SCHEMA,
    LOCK_SCHEMA,
    PROTOCOL_ID,
    canonical_json_bytes,
    canonical_sha256,
    create_formal_start_marker,
    sha256_file,
    validate,
    validate_activation_identity,
)


class PreexecutionIntegrityTest(unittest.TestCase):
    def test_canonical_json_is_sorted_compact_utf8_lf(self) -> None:
        value = {"z": [2, 1], "中文": "值", "a": {"b": True}}
        expected = '{"a":{"b":true},"z":[2,1],"中文":"值"}'.encode("utf-8")
        self.assertEqual(canonical_json_bytes(value), expected)
        self.assertEqual(
            canonical_sha256(value),
            hashlib.sha256(expected).hexdigest(),
        )
        self.assertFalse(canonical_json_bytes(value).startswith(b"\xef\xbb\xbf"))

    def test_canonical_json_rejects_nonfinite_numbers(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    canonical_json_bytes({"value": value})

    def test_formal_start_marker_is_exclusive_and_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "formal_start.json"
            payload = {
                "protocol_id": PROTOCOL_ID,
                "execution_state": "STARTED_CONSUMED",
            }
            create_formal_start_marker(marker, payload)
            first = marker.read_bytes()
            with self.assertRaises(FileExistsError):
                create_formal_start_marker(marker, payload)
            self.assertEqual(marker.read_bytes(), first)
            self.assertEqual(first, canonical_json_bytes(payload) + b"\n")

    def test_formal_start_marker_does_not_create_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "absent" / "formal_start.json"
            with self.assertRaises(FileNotFoundError):
                create_formal_start_marker(marker, {"status": "STARTED"})
            self.assertFalse(marker.parent.exists())

    def test_valid_synthetic_lock_and_activation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, protocol_path, protocol_hash = self.build_fixture(root)
            result = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
            )
            self.assertEqual(result["status"], "VALID", result["failures"])
            self.assertFalse(result["formal_execution_authorized"])
            self.assertFalse(result["vicon_bag_messages_opened"])

            review_path = root / "implementation_review.md"
            repository = {
                "head": "a" * 40,
                "origin_master": "a" * 40,
            }
            review_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": (
                            "blindassist.d0_egomotion_error_attribution."
                            "implementation_review.v1"
                        ),
                        "status": "PASS",
                        "reviewer_role": "INDEPENDENT_READ_ONLY_REVIEW",
                        "protocol": {
                            "protocol_id": PROTOCOL_ID,
                            "sha256": protocol_hash,
                        },
                        "implementation_lock": {
                            "path": lock_path.relative_to(root).as_posix(),
                            "sha256": sha256_file(lock_path),
                        },
                        "repository": repository,
                        "formal_execution_authorized": False,
                        "checks": [{"name": "fixture", "passed": True}],
                    }
                )
                + b"\n"
            )
            activation = {
                "schema_version": ACTIVATION_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": protocol_hash,
                "execution_state": "NOT_RUN",
                "formal_output_root": "artifacts.local/d0/run-r1",
                "repository": repository,
                "implementation_lock": {
                    "path": lock_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(lock_path),
                },
                "implementation_review": {
                    "path": review_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(review_path),
                },
            }
            identity = validate_activation_identity(
                activation,
                lock_path,
                root,
                expected_formal_output_root="artifacts.local/d0/run-r1",
            )
            self.assertEqual(identity["status"], "VALID_IDENTITY")
            self.assertFalse(identity["authorization_evaluated"])

    def test_input_lock_rejects_hash_drift_and_extra_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, protocol_path, protocol_hash = self.build_fixture(root)
            input_path = root / "inputs" / "events.jsonl"
            input_path.write_text(
                '{"event_id":"e1","primary_event_eligible":true}\n'
                '{"event_id":"e2","primary_event_eligible":false}\n',
                encoding="utf-8",
                newline="\n",
            )
            result = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
            )
            self.assertIn("input_natural_events_sha256", result["failures"])

            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["frozen_inputs"]["forbidden_extra"] = {
                "path": "forbidden/old-f1b.json",
                "sha256": "0" * 64,
            }
            lock_path.write_bytes(canonical_json_bytes(lock))
            result = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
            )
            self.assertIn("frozen_input_binding_set", result["failures"])

    def test_input_lock_rejects_repository_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, protocol_path, protocol_hash = self.build_fixture(root)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["frozen_inputs"]["natural_events"]["path"] = "../events.jsonl"
            lock_path.write_bytes(canonical_json_bytes(lock))
            result = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
            )
            self.assertIn("frozen_input_binding_set", result["failures"])

    def test_activation_identity_rejects_lock_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, _, protocol_hash = self.build_fixture(root)
            repository = {
                "head": "a" * 40,
                "origin_master": "a" * 40,
            }
            review_path = root / "implementation_review.md"
            review_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": (
                            "blindassist.d0_egomotion_error_attribution."
                            "implementation_review.v1"
                        ),
                        "status": "PASS",
                        "reviewer_role": "INDEPENDENT_READ_ONLY_REVIEW",
                        "protocol": {
                            "protocol_id": PROTOCOL_ID,
                            "sha256": protocol_hash,
                        },
                        "implementation_lock": {
                            "path": lock_path.relative_to(root).as_posix(),
                            "sha256": sha256_file(lock_path),
                        },
                        "repository": repository,
                        "formal_execution_authorized": False,
                        "checks": [{"name": "fixture", "passed": True}],
                    }
                )
                + b"\n"
            )
            activation = {
                "schema_version": ACTIVATION_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": protocol_hash,
                "execution_state": "NOT_RUN",
                "formal_output_root": "artifacts.local/d0/run-r1",
                "repository": repository,
                "implementation_lock": {
                    "path": lock_path.relative_to(root).as_posix(),
                    "sha256": "0" * 64,
                },
                "implementation_review": {
                    "path": review_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(review_path),
                },
            }
            identity = validate_activation_identity(
                activation,
                lock_path,
                root,
                expected_formal_output_root="artifacts.local/d0/run-r1",
            )
            self.assertEqual(identity["status"], "INVALID_IDENTITY")
            self.assertIn(
                "ACTIVATION_LOCK_SHA256",
                identity["failures"],
            )

    @staticmethod
    def build_fixture(root: Path) -> tuple[Path, Path, str]:
        module_root = "scripts/research/d0"
        module_dir = root / module_root
        module_dir.mkdir(parents=True)
        modules = [
            "contract.py",
            "bindings.py",
            "producer.py",
            "analysis.py",
            "runner.py",
            "validate_implementation_lock.py",
            "validate_execution_independent.py",
        ]
        for name in modules + ["test_integrity.py"]:
            (module_dir / name).write_text(
                f"# synthetic {name}\n",
                encoding="utf-8",
                newline="\n",
            )
        (module_dir / "README.md").write_text(
            "synthetic\n", encoding="utf-8", newline="\n"
        )
        adapter = root / "scripts" / "run_d0.py"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_text(
            "# synthetic adapter\n", encoding="utf-8", newline="\n"
        )

        inputs = root / "inputs"
        inputs.mkdir()
        events = inputs / "events.jsonl"
        events.write_text(
            '{"event_id":"e1","primary_event_eligible":true}\n',
            encoding="utf-8",
            newline="\n",
        )
        dependency = inputs / "dependency.json"
        dependency_payload = {
            "schema_version": "blindassist.d0_dependency_receipt.v1",
            "status": "VALID",
            "primary_event_count": 1,
            "cross_target_overlap_pair_count": 0,
            "same_target_overlap_pair_count": 0,
            "exact_overlap_component_count": 1,
        }
        dependency.write_bytes(canonical_json_bytes(dependency_payload))

        predecessor_dir = root / "predecessor"
        predecessor_dir.mkdir()
        result_path = predecessor_dir / "result.md"
        result_path.write_text(
            "VALID / NO_INCREMENT\n", encoding="utf-8", newline="\n"
        )
        validation_path = predecessor_dir / "validation.json"
        validation_path.write_bytes(
            canonical_json_bytes(
                {
                    "status": "VALID",
                    "truth_opened": False,
                    "frame_count": 1,
                    "trace_row_count": 2,
                    "branch_pair_mismatch_count": 0,
                    "failure_count": 0,
                }
            )
        )
        seal_path = predecessor_dir / "seal.json"
        seal_path.write_bytes(canonical_json_bytes({"status": "SEALED"}))

        frozen_inputs = {
            "dependency_receipt": {
                "path": dependency.relative_to(root).as_posix(),
                "sha256": sha256_file(dependency),
                **dependency_payload,
            },
            "natural_events": {
                "path": events.relative_to(root).as_posix(),
                "rows": 1,
                "primary_rows": 1,
                "sha256": sha256_file(events),
            },
        }
        canonical_contract = {
            "json_object_key_order": "lexical sort",
            "text_encoding": "UTF-8 without BOM",
            "line_endings": "LF",
        }
        protocol = {
            "schema_version": "blindassist.research_protocol.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "CONTRACT_FROZEN",
            "execution_status": "NOT_RUN",
            "execution_authorized": False,
            "predecessor_gate": {
                "result": {
                    "path": result_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(result_path),
                },
                "independent_validation": {
                    "path": validation_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(validation_path),
                    "required_status": "VALID",
                    "required_truth_opened": False,
                    "required_frame_count": 1,
                    "required_trace_row_count": 2,
                    "required_branch_pair_mismatch_count": 0,
                    "required_failure_count": 0,
                },
                "seal": {
                    "path": seal_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(seal_path),
                    "required_status": "SEALED",
                },
            },
            "frozen_inputs": frozen_inputs,
            "planned_implementation": {
                "stable_adapter": adapter.relative_to(root).as_posix(),
                "module_root": module_root,
                "modules": modules,
                "formal_output_root": "artifacts.local/d0/run-r1",
                "canonical_serialization": canonical_contract,
            },
        }
        protocol_path = root / "docs" / "protocol.json"
        protocol_path.parent.mkdir()
        protocol_path.write_bytes(canonical_json_bytes(protocol))
        protocol_hash = sha256_file(protocol_path)

        implementation_paths = {
            adapter.relative_to(root).as_posix(),
            *{
                path.relative_to(root).as_posix()
                for path in module_dir.iterdir()
                if path.is_file()
            },
        }
        lock = {
            "schema_version": LOCK_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "implementation_status": "FROZEN_FOR_INDEPENDENT_REVIEW",
            "execution_state": "NOT_RUN",
            "repository": {
                "head": "a" * 40,
                "origin_master": "a" * 40,
            },
            "authority": {
                "activation_authorized": False,
                "formal_execution_authorized": False,
                "scientific_exit_authorized": False,
            },
            "protocol": {
                "path": protocol_path.relative_to(root).as_posix(),
                "sha256": protocol_hash,
            },
            "predecessor_bindings": protocol["predecessor_gate"],
            "frozen_inputs": frozen_inputs,
            "canonical_serialization_sha256": canonical_sha256(
                canonical_contract
            ),
            "frozen_inputs_sha256": canonical_sha256(frozen_inputs),
            "implementation_file_hashes": {
                relative: sha256_file(root / relative)
                for relative in sorted(implementation_paths)
            },
        }
        lock_path = root / "implementation_lock.json"
        lock_path.write_bytes(canonical_json_bytes(lock) + b"\n")
        return (
            lock_path,
            protocol_path.relative_to(root),
            protocol_hash,
        )


if __name__ == "__main__":
    unittest.main()
