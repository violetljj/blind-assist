from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import validate_implementation_lock as implementation_validator
import create_activation
from validate_implementation_lock import (
    ACTIVATION_SCHEMA,
    LOCK_SCHEMA,
    PROTOCOL_ID,
    R1_BOUND_FILE_KEYS,
    R2_BOUND_FILE_KEYS,
    canonical_json_bytes,
    canonical_sha256,
    create_formal_start_marker,
    r1_failure_gate_integrity_errors,
    r2_failure_gate_integrity_errors,
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
                validate_scientific_inputs=True,
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
                "formal_output_root": "artifacts.local/d0/run-r3",
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
                expected_formal_output_root="artifacts.local/d0/run-r3",
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
                validate_scientific_inputs=True,
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
                validate_scientific_inputs=True,
            )
            self.assertIn("frozen_input_binding_set", result["failures"])

    def test_dependency_primary_count_is_derived_from_event_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, protocol_path, protocol_hash = self.build_fixture(root)
            dependency_path = root / "inputs" / "dependency.json"
            dependency = json.loads(
                dependency_path.read_text(encoding="utf-8")
            )
            dependency["event_bindings"] = []
            dependency_path.write_bytes(canonical_json_bytes(dependency))
            result = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
                validate_scientific_inputs=True,
            )
            self.assertIn(
                "input_dependency_receipt_primary_event_count",
                result["failures"],
            )

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
                validate_scientific_inputs=True,
            )
            self.assertIn("frozen_input_binding_set", result["failures"])

    def test_prestart_envelope_does_not_open_scientific_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path, protocol_path, protocol_hash = self.build_fixture(root)
            protocol = json.loads(
                (root / protocol_path).read_text(encoding="utf-8")
            )
            forbidden = {
                (root / specification["path"]).resolve()
                for specification in (
                    *protocol["predecessor_gate"].values(),
                    *protocol["frozen_inputs"].values(),
                )
            }
            opened: list[Path] = []
            original_sha256_file = implementation_validator.sha256_file

            def record_hash(path: Path) -> str:
                resolved = path.resolve()
                opened.append(resolved)
                if resolved in forbidden:
                    raise AssertionError(
                        f"scientific input opened pre-marker: {resolved}"
                    )
                return original_sha256_file(path)

            with mock.patch.object(
                implementation_validator,
                "sha256_file",
                side_effect=record_hash,
            ):
                result = validate(
                    lock_path,
                    root,
                    expected_protocol_path=protocol_path.as_posix(),
                    expected_protocol_sha256=protocol_hash,
                    validate_scientific_inputs=False,
                )
            self.assertEqual(result["status"], "VALID", result["failures"])
            self.assertFalse(result["scientific_inputs_validated"])
            self.assertTrue(forbidden.isdisjoint(opened))

            events_path = root / protocol["frozen_inputs"]["natural_events"][
                "path"
            ]
            events_path.write_text(
                '{"event_id":"drift","primary_event_eligible":false}\n',
                encoding="utf-8",
                newline="\n",
            )
            prestart = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
                validate_scientific_inputs=False,
            )
            full = validate(
                lock_path,
                root,
                expected_protocol_path=protocol_path.as_posix(),
                expected_protocol_sha256=protocol_hash,
                validate_scientific_inputs=True,
            )
            self.assertEqual(prestart["status"], "VALID")
            self.assertEqual(full["status"], "INVALID")
            self.assertIn("input_natural_events_sha256", full["failures"])

    def test_review_cli_and_activation_force_prestart_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "implementation_lock.json"
            review_path = root / "implementation_review.json"
            activation_path = root / "activation.json"
            observed_scopes: list[bool] = []

            def reject_after_scope(
                *_: object, **kwargs: object
            ) -> dict[str, object]:
                observed_scopes.append(
                    bool(kwargs["validate_scientific_inputs"])
                )
                return {"status": "INVALID", "failures": ["synthetic"]}

            with (
                mock.patch.object(
                    create_activation,
                    "validate",
                    side_effect=reject_after_scope,
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "create_activation.py",
                        "--repository-root",
                        str(root),
                        "--implementation-lock",
                        str(lock_path),
                        "--implementation-review",
                        str(review_path),
                        "--output",
                        str(activation_path),
                    ],
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "implementation lock is not valid"
                ):
                    create_activation.main()

            def pass_after_scope(
                *_: object, **kwargs: object
            ) -> dict[str, object]:
                observed_scopes.append(
                    bool(kwargs["validate_scientific_inputs"])
                )
                return {"status": "VALID"}

            with (
                mock.patch.object(
                    implementation_validator,
                    "validate",
                    side_effect=pass_after_scope,
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "validate_implementation_lock.py",
                        "--implementation-lock",
                        str(lock_path),
                        "--repository-root",
                        str(root),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(implementation_validator.main(), 0)
            self.assertEqual(observed_scopes, [False, False])

    def test_r2_failure_gate_binds_archives_and_exact_file_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_root = (
                root
                / "artifacts.local/evidence/dual-loop/"
                "d0-egomotion-error-attribution-r2"
            )
            implementation_root = evidence_root / "implementation"
            run_root = evidence_root / "run-r2"
            runtime_root = evidence_root / "runtime-freeze"
            implementation_root.mkdir(parents=True)
            run_root.mkdir()
            runtime_root.mkdir()

            paths = {
                "result": root / "docs/result.md",
                "protocol": root / "docs/protocol.json",
                "implementation_lock": implementation_root
                / "implementation_lock.json",
                "implementation_review": implementation_root
                / "implementation_review.json",
                "activation": evidence_root / "activation.json",
                "activation_archive": evidence_root
                / "activation.pre-a29a776.json",
                "implementation_lock_archive": implementation_root
                / "implementation_lock.pre-a29a776.json",
                "implementation_review_archive": implementation_root
                / "implementation_review.pre-a29a776.json",
                "runtime_manifest": runtime_root
                / "runtime_environment_manifest.json",
                "formal_start": run_root / "formal_start.json",
                "progress": run_root / "progress.json",
                "failure_receipt": run_root / "failure_receipt.json",
            }
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"{name}\n", encoding="utf-8", newline="\n"
                )
            gate = {
                name: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            }
            self.assertEqual(set(gate), set(R2_BOUND_FILE_KEYS))
            gate.update(
                {
                    "exact_run_file_set": [
                        "failure_receipt.json",
                        "formal_start.json",
                        "progress.json",
                    ],
                    "exact_evidence_root_entry_set": [
                        "activation.json",
                        "activation.pre-a29a776.json",
                        "implementation",
                        "run-r2",
                        "runtime-freeze",
                    ],
                    "exact_implementation_file_set": [
                        "implementation_lock.json",
                        "implementation_lock.pre-a29a776.json",
                        "implementation_review.json",
                        "implementation_review.pre-a29a776.json",
                    ],
                    "required_absent_outputs": ["event_table.jsonl"],
                }
            )
            self.assertEqual(
                r2_failure_gate_integrity_errors(root, gate), []
            )

            archive = paths["activation_archive"]
            original_archive = archive.read_bytes()
            archive.write_bytes(original_archive + b"tamper\n")
            self.assertIn(
                "activation_archive:SHA256",
                r2_failure_gate_integrity_errors(root, gate),
            )
            archive.write_bytes(original_archive)

            archive.unlink()
            deletion_errors = r2_failure_gate_integrity_errors(root, gate)
            self.assertIn("activation_archive:MISSING", deletion_errors)
            self.assertIn("evidence_root:ENTRY_SET", deletion_errors)
            archive.write_bytes(original_archive)

            extra = implementation_root / "unexpected.json"
            extra.write_text("extra\n", encoding="utf-8", newline="\n")
            self.assertIn(
                "implementation:FILE_SET",
                r2_failure_gate_integrity_errors(root, gate),
            )

    def test_r1_failure_gate_binds_archives_and_exact_file_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_root = (
                root
                / "artifacts.local/evidence/dual-loop/"
                "d0-egomotion-error-attribution-r1"
            )
            implementation_root = evidence_root / "implementation"
            input_freeze_root = evidence_root / "input-freeze"
            run_root = evidence_root / "run-r1"
            implementation_root.mkdir(parents=True)
            input_freeze_root.mkdir()
            run_root.mkdir()
            paths = {
                "result": root / "docs/result.md",
                "implementation_lock": implementation_root
                / "implementation_lock.json",
                "implementation_review": implementation_root
                / "implementation_review.json",
                "activation": implementation_root / "activation.json",
                "invalid_implementation_lock_archive": implementation_root
                / "implementation_lock.invalid-c992829.json",
                "input_freeze_dependency_receipt": input_freeze_root
                / "dependency_receipt.json",
                "formal_start": run_root / "formal_start.json",
                "progress": run_root / "progress.json",
                "failure_receipt": run_root / "failure_receipt.json",
            }
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"{name}\n", encoding="utf-8", newline="\n"
                )
            gate = {
                name: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            }
            self.assertEqual(set(gate), set(R1_BOUND_FILE_KEYS))
            gate.update(
                {
                    "exact_run_file_set": [
                        "failure_receipt.json",
                        "formal_start.json",
                        "progress.json",
                    ],
                    "exact_evidence_root_entry_set": [
                        "implementation",
                        "input-freeze",
                        "run-r1",
                    ],
                    "exact_implementation_file_set": [
                        "activation.json",
                        "implementation_lock.invalid-c992829.json",
                        "implementation_lock.json",
                        "implementation_review.json",
                    ],
                    "exact_input_freeze_file_set": [
                        "dependency_receipt.json"
                    ],
                    "required_absent_outputs": ["event_table.jsonl"],
                }
            )
            self.assertEqual(
                r1_failure_gate_integrity_errors(root, gate), []
            )

            archive = paths["invalid_implementation_lock_archive"]
            original_archive = archive.read_bytes()
            archive.write_bytes(original_archive + b"tamper\n")
            self.assertIn(
                "invalid_implementation_lock_archive:SHA256",
                r1_failure_gate_integrity_errors(root, gate),
            )
            archive.write_bytes(original_archive)

            archive.unlink()
            deletion_errors = r1_failure_gate_integrity_errors(root, gate)
            self.assertIn(
                "invalid_implementation_lock_archive:MISSING",
                deletion_errors,
            )
            self.assertIn("implementation:FILE_SET", deletion_errors)
            archive.write_bytes(original_archive)

            extra = run_root / "unexpected.json"
            extra.write_text("extra\n", encoding="utf-8", newline="\n")
            self.assertIn(
                "run:FILE_SET",
                r1_failure_gate_integrity_errors(root, gate),
            )

    def test_live_r1_and_r2_failure_gate_layouts_are_exact(self) -> None:
        root = implementation_validator.REPO_ROOT
        protocol = json.loads(
            (
                root
                / "docs/research/dual-loop/"
                "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            r1_failure_gate_integrity_errors(
                root, protocol["r1_failure_gate"]
            ),
            [],
        )
        self.assertEqual(
            r2_failure_gate_integrity_errors(
                root, protocol["r2_failure_gate"]
            ),
            [],
        )

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
                "formal_output_root": "artifacts.local/d0/run-r3",
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
                expected_formal_output_root="artifacts.local/d0/run-r3",
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
            "event_bindings": [{"event_id": "e1"}],
            "natural_events": {"primary_event_count": 1},
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
                "schema_version": dependency_payload["schema_version"],
                "status": dependency_payload["status"],
                "primary_event_count": 1,
                "cross_target_overlap_pair_count": 0,
                "same_target_overlap_pair_count": 0,
                "exact_overlap_component_count": 1,
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
                "formal_output_root": "artifacts.local/d0/run-r3",
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
