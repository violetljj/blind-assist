from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime_environment import RuntimeEnvironmentError


MODULE_PATH = Path(__file__).with_name("validate_execution_independent.py")
SPEC = importlib.util.spec_from_file_location("d0_independent_validator", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPENDENCY_PATH = (
    REPO_ROOT
    / "artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r1"
    / "input-freeze/dependency_receipt.json"
)


def synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for block_id in range(6):
        for group in validator.GROUPS:
            for within_block in range(8):
                wrong = group == validator.WRONG
                rows.append(
                    {
                        "event_id": f"event-{index:04d}",
                        "target_id": f"track-{within_block % 2:03d}",
                        "anchor_region": ("LEFT", "RIGHT")[within_block % 2],
                        "truth_state": validator.TRUTH_STATES[within_block % 2],
                        "overlap_component_id": f"component-{index:04d}",
                        "time_block_id_60s": block_id,
                        "primary_error_partition": group,
                        "median_abs_sensor_approach_component_mps": 1.0 if wrong else 0.0,
                        "median_abs_person_approach_component_mps": 0.0,
                        "median_flow_score_mad_per_s": 1.0 if wrong else 0.0,
                        "median_surviving_tracks": 0.0 if wrong else 1.0,
                        "sensor_absolute_share": 0.75 if wrong else 0.25,
                    }
                )
                index += 1
    return rows


class IndependentValidatorTests(unittest.TestCase):
    def test_source_has_no_producer_or_analysis_import(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(
            any(
                name.endswith("producer")
                or ".producer" in name
                or name.endswith("analysis")
                or ".analysis" in name
                or name.endswith("bindings")
                or ".bindings" in name
                for name in imported
            ),
            imported,
        )

    def test_formal_activation_and_lock_entities_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = {"head": "a" * 40, "origin_master": "a" * 40}
            module_root = root / "module"
            module_root.mkdir()
            (module_root / "README.md").write_text("fixture", encoding="utf-8")
            (root / "adapter.py").write_text("pass\n", encoding="utf-8")
            predecessor: dict[str, dict[str, str]] = {}
            for name in ("result", "independent_validation", "seal"):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                predecessor[name] = {
                    "path": path.name,
                    "sha256": validator.sha256_file(path),
                }
            protocol = {
                "predecessor_gate": predecessor,
                "frozen_inputs": {},
                "planned_implementation": {
                    "formal_output_root": "run-r3",
                    "module_root": "module",
                    "stable_adapter": "adapter.py",
                    "modules": [],
                    "canonical_serialization": {},
                },
            }
            implementation_hashes = {
                "adapter.py": validator.sha256_file(root / "adapter.py"),
                "module/README.md": validator.sha256_file(
                    module_root / "README.md"
                ),
            }
            lock_path = root / "implementation_lock.json"
            lock = {
                "schema_version": validator.IMPLEMENTATION_LOCK_SCHEMA,
                "protocol_id": validator.PROTOCOL_ID,
                "implementation_status": "FROZEN_FOR_INDEPENDENT_REVIEW",
                "execution_state": "NOT_RUN",
                "repository": repository,
                "authority": {
                    "activation_authorized": False,
                    "formal_execution_authorized": False,
                    "scientific_exit_authorized": False,
                },
                "protocol": {
                    "path": (
                        "docs/research/dual-loop/"
                        "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
                    ),
                    "sha256": validator.PROTOCOL_SHA256,
                },
                "predecessor_bindings": predecessor,
                "frozen_inputs": {},
                "frozen_inputs_sha256": validator._canonical_sha256({}),
                "canonical_serialization_sha256": validator._canonical_sha256(
                    {}
                ),
                "implementation_file_hashes": implementation_hashes,
            }
            lock_path.write_bytes(validator.canonical_json_bytes(lock))
            review_path = root / "implementation_review.md"
            review_path.write_bytes(
                validator.canonical_json_bytes(
                    {
                        "schema_version": validator.IMPLEMENTATION_REVIEW_SCHEMA,
                        "status": "PASS",
                        "reviewer_role": "INDEPENDENT_READ_ONLY_REVIEW",
                        "protocol": {
                            "protocol_id": validator.PROTOCOL_ID,
                            "sha256": validator.PROTOCOL_SHA256,
                        },
                        "implementation_lock": {
                            "path": "implementation_lock.json",
                            "sha256": validator.sha256_file(lock_path),
                        },
                        "repository": repository,
                        "formal_execution_authorized": False,
                        "checks": [{"name": "fixture", "passed": True}],
                    }
                )
            )
            activation_path = root / "activation.json"
            activation = {
                "schema_version": validator.ACTIVATION_SCHEMA,
                "protocol_id": validator.PROTOCOL_ID,
                "protocol_sha256": validator.PROTOCOL_SHA256,
                "execution_state": "NOT_RUN",
                "formal_output_root": "run-r3",
                "formal_execution_authorized": True,
                "authority": {
                    "formal_execution_authorized": True,
                    "successor_execution_authorized": False,
                    "confirmation_authorized": False,
                    "product_or_safety_authorized": False,
                },
                "repository": repository,
                "implementation_lock": {
                    "path": "implementation_lock.json",
                    "sha256": validator.sha256_file(lock_path),
                },
                "implementation_review": {
                    "path": "implementation_review.md",
                    "sha256": validator.sha256_file(review_path),
                },
            }
            activation_path.write_bytes(validator.canonical_json_bytes(activation))
            formal = {
                "protocol_id": validator.PROTOCOL_ID,
                "protocol_sha256": validator.PROTOCOL_SHA256,
                "repository": repository,
                "activation": {
                    "path": "activation.json",
                    "sha256": validator.sha256_file(activation_path),
                },
                "implementation_lock": {
                    "path": "implementation_lock.json",
                    "sha256": validator.sha256_file(lock_path),
                },
            }
            result = validator.validate_formal_identities(
                formal, protocol=protocol, repo_root=root
            )
            self.assertEqual(result["activation_path"], activation_path)
            mutated = copy.deepcopy(formal)
            mutated["activation"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                validator.ValidationError, "activation hash mismatch"
            ):
                validator.validate_formal_identities(
                    mutated, protocol=protocol, repo_root=root
                )
            activation_without_review = copy.deepcopy(activation)
            del activation_without_review["implementation_review"]
            activation_path.write_bytes(
                validator.canonical_json_bytes(activation_without_review)
            )
            missing_review_formal = copy.deepcopy(formal)
            missing_review_formal["activation"]["sha256"] = (
                validator.sha256_file(activation_path)
            )
            with self.assertRaisesRegex(
                validator.ValidationError, "implementation-review binding missing"
            ):
                validator.validate_formal_identities(
                    missing_review_formal, protocol=protocol, repo_root=root
                )

            bad_lock = copy.deepcopy(lock)
            bad_lock["implementation_file_hashes"]["adapter.py"] = "0" * 64
            lock_path.write_bytes(validator.canonical_json_bytes(bad_lock))
            activation["implementation_lock"]["sha256"] = validator.sha256_file(
                lock_path
            )
            review = validator.load_canonical_json(review_path)
            review["implementation_lock"]["sha256"] = validator.sha256_file(
                lock_path
            )
            review_path.write_bytes(validator.canonical_json_bytes(review))
            activation["implementation_review"]["sha256"] = validator.sha256_file(
                review_path
            )
            activation_path.write_bytes(validator.canonical_json_bytes(activation))
            bad_source_formal = copy.deepcopy(formal)
            bad_source_formal["activation"]["sha256"] = validator.sha256_file(
                activation_path
            )
            bad_source_formal["implementation_lock"]["sha256"] = (
                validator.sha256_file(lock_path)
            )
            with self.assertRaisesRegex(
                validator.ValidationError, "source hash mismatch"
            ):
                validator.validate_formal_identities(
                    bad_source_formal, protocol=protocol, repo_root=root
                )

    def test_dependency_semantics_recompute_matches_frozen_receipt(self) -> None:
        dependency = json.loads(DEPENDENCY_PATH.read_text(encoding="utf-8"))
        result = validator._recompute_dependency(dependency["event_bindings"])
        self.assertEqual(result["cross_target_overlap_pair_count"], 159)
        self.assertEqual(result["same_target_overlap_pair_count"], 0)
        self.assertEqual(result["exact_overlap_component_count"], 310)
        self.assertEqual(result["component_size_counts"], dependency["component_size_counts"])
        self.assertEqual(
            result["overlap_pairs_sha256"], dependency["overlap_pairs_sha256"]
        )
        self.assertEqual(result["components"], dependency["components"])
        self.assertEqual(result["time_block"], dependency["time_block"])
        self.assertEqual(
            result["event_bindings_sha256"],
            dependency["event_bindings_sha256"],
        )
        self.assertTrue(
            all(
                result["component_id_by_event"][binding["event_id"]]
                == binding["overlap_component_id"]
                for binding in dependency["event_bindings"]
            )
        )

    def test_dependency_component_mutation_is_rejected(self) -> None:
        dependency = json.loads(DEPENDENCY_PATH.read_text(encoding="utf-8"))
        rows = []
        for binding in sorted(
            dependency["event_bindings"], key=lambda item: item["event_id"]
        ):
            row = {
                field: None for field in validator.REQUIRED_EVENT_FIELDS
            }
            row.update(binding)
            row.update(
                {
                    "capture_id": "REVEL_DYNAMIC_SINGLE_CAPTURE",
                    "eligible_frame_count": 1,
                    "primary_error_partition": validator.CORRECT,
                    "reference_error_partition": validator.CORRECT,
                    "source_pair_denominator": 1,
                    "finite_source_pair_count": 0,
                    "source_missing_reason_counts": {},
                    "abstained_pair_count": 1,
                }
            )
            rows.append(row)
        validator.validate_event_rows(rows, dependency)
        rows[0]["overlap_component_id"] = "component-mutated"
        with self.assertRaisesRegex(validator.ValidationError, "dependency mismatch"):
            validator.validate_event_rows(rows, dependency)

    def test_noncanonical_and_duplicate_json_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_bytes(b'{"b":1,"a":2}\n')
            with self.assertRaisesRegex(validator.ValidationError, "not canonical"):
                validator.load_canonical_jsonl(path)
            path.write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaisesRegex(validator.ValidationError, "duplicate JSON key"):
                validator.load_canonical_jsonl(path)
            object_path = Path(temp) / "object.json"
            object_path.write_bytes(b'{"a":1}\n')
            self.assertEqual(
                validator.load_canonical_json(object_path),
                {"a": 1},
            )
            for payload in (b'{"a":1}\n\n', b'{"a":1}\r\n', b'{"a":1} '):
                with self.subTest(payload=payload):
                    object_path.write_bytes(payload)
                    with self.assertRaisesRegex(
                        validator.ValidationError, "not canonical"
                    ):
                        validator.load_canonical_json(object_path)

    def test_dependency_identity_follows_r1_scientific_contract(self) -> None:
        protocol_path = (
            REPO_ROOT
            / "docs/research/dual-loop/"
            "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
        )
        protocol = validator.parse_json_bytes(
            protocol_path.read_bytes(),
            label=str(protocol_path),
        )
        dependency = validator.parse_json_bytes(
            DEPENDENCY_PATH.read_bytes(),
            label=str(DEPENDENCY_PATH),
        )
        expected = validator.expected_dependency_protocol_id(protocol)
        self.assertEqual(expected, "D0_EGOMOTION_ERROR_ATTRIBUTION_R1")
        self.assertEqual(dependency["protocol_id"], expected)
        self.assertNotEqual(expected, validator.PROTOCOL_ID)

    def test_r3_formal_marker_and_probe_keysets_are_exact(self) -> None:
        probe = {
            "status": "VALID_INHERITED_R2_OPERATIONAL_PROBE",
            "source_protocol_id": "D0_EGOMOTION_ERROR_ATTRIBUTION_R2",
            "source_formal_start_sha256": "d" * 64,
            "source_runtime_manifest_sha256": "e" * 64,
            "source_runtime_tree_sha256": "f" * 64,
            "topic": "topic",
            "connection_count": 1,
            "ordinal": 0,
            "msgtype": "type",
            "timestamp_ns": 1,
            "raw_bytes": 1,
            "raw_sha256": "a" * 64,
            "deserialized_message_count": 0,
            "inherited_deserialized_message_count": 1,
            "pose_values_retained": False,
            "vicon_bag_messages_opened": False,
            "d0_metrics_computed": False,
        }
        protocol = {
            "runtime_environment": {
                "manifest": {
                    "sha256": "b" * 64,
                    "tree_sha256": "c" * 64,
                },
                "designated_prestart_probe": probe,
            }
        }
        formal = {
            "schema_version": "blindassist.d0_formal_start.v1",
            "protocol_id": validator.PROTOCOL_ID,
            "protocol_sha256": validator.PROTOCOL_SHA256,
            "activation": {},
            "implementation_lock": {},
            "repository": {},
            "state": "FORMAL_STARTED",
            "d0_metric_computation_pending": True,
            "vicon_bag_messages_opened": False,
            "prestart_dependency_smoke": {
                "status": "VALID_SYNTHETIC_RUNTIME_SMOKE",
                "imports": [
                    "numpy",
                    "yaml",
                    "rosbags.rosbag1.Reader",
                    "rosbags.typesys.Stores",
                    "rosbags.typesys.get_typestore",
                ],
                "yaml_safe_load_called": True,
                "synthetic_calibration_shape": [4, 4],
                "synthetic_calibration_finite": True,
                "real_calibration_opened": False,
                "bag_messages_opened": False,
                "truth_opened": False,
                "event_rows_built": False,
                "d0_metrics_computed": False,
                "producer_calibration_parser": {
                    "fixture": (
                        "scripts/research/"
                        "dual_loop_d0_egomotion_error_attribution_r3/"
                        "synthetic_calibration.yaml"
                    ),
                    "shape": [4, 4],
                    "finite": True,
                    "values_retained": False,
                },
                "independent_calibration_parser": {
                    "fixture": (
                        "scripts/research/"
                        "dual_loop_d0_egomotion_error_attribution_r3/"
                        "synthetic_calibration.yaml"
                    ),
                    "shape": [4, 4],
                    "finite": True,
                    "values_retained": False,
                },
            },
            "prestart_operational_probe": {
                **probe,
                "runtime_manifest_sha256": "b" * 64,
                "runtime_tree_sha256": "c" * 64,
            },
        }
        extra_probe = copy.deepcopy(formal)
        extra_probe["prestart_operational_probe"]["unexpected"] = True
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                validator.ValidationError, "runtime probe mismatch"
            ):
                validator.validate_formal_identities(
                    extra_probe,
                    protocol=protocol,
                    repo_root=Path(temp),
                )
            missing_state = copy.deepcopy(formal)
            del missing_state["state"]
            with self.assertRaisesRegex(
                validator.ValidationError, "envelope mismatch"
            ):
                validator.validate_formal_identities(
                    missing_state,
                    protocol=protocol,
                    repo_root=Path(temp),
                )

    def test_atomic_publication_is_exclusive_and_fail_terminal_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            immutable = root / "immutable.json"
            validator._exclusive_write(immutable, {"value": 1})
            first = immutable.read_bytes()
            with self.assertRaises(FileExistsError):
                validator._exclusive_write(immutable, {"value": 2})
            self.assertEqual(immutable.read_bytes(), first)

            run_root = root / "run-r3"
            run_root.mkdir()
            (run_root / "formal_start.json").write_bytes(
                validator.canonical_json_bytes({})
            )
            protocol = (
                REPO_ROOT
                / "docs/research/dual-loop/"
                "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
            )
            with (
                mock.patch.object(
                    validator,
                    "resolve_expected_run_root",
                    return_value=run_root,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = validator.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--protocol",
                        str(protocol),
                        "--dependency-receipt",
                        str(DEPENDENCY_PATH),
                        "--write-results",
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertTrue((run_root / "execution_validation.json").is_file())
            self.assertTrue((run_root / "failure_receipt.json").is_file())
            self.assertFalse((run_root / "execution_receipt.json").exists())
            progress = validator.load_canonical_json(run_root / "progress.json")
            self.assertEqual(progress["state"], "EXECUTION_INVALID")

    def test_wrong_run_root_is_prestart_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "not-the-frozen-root"
            run_root.mkdir()
            protocol = (
                REPO_ROOT
                / "docs/research/dual-loop/"
                "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = validator.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--protocol",
                        str(protocol),
                        "--dependency-receipt",
                        str(DEPENDENCY_PATH),
                        "--write-results",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertEqual(list(run_root.iterdir()), [])

    def test_absent_formal_marker_never_creates_or_writes_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run-r3"
            protocol = (
                REPO_ROOT
                / "docs/research/dual-loop/"
                "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
            )
            with (
                mock.patch.object(
                    validator,
                    "resolve_expected_run_root",
                    return_value=run_root,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = validator.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--protocol",
                        str(protocol),
                        "--dependency-receipt",
                        str(DEPENDENCY_PATH),
                        "--write-results",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertFalse(run_root.exists())

    def test_runtime_validation_error_is_published_as_closed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run-r3"
            run_root.mkdir()
            (run_root / "formal_start.json").write_bytes(
                validator.canonical_json_bytes({})
            )
            for name in (
                "event_table.jsonl",
                "analysis.json",
                "producer_receipt.json",
            ):
                (run_root / name).write_bytes(b"")
            protocol = (
                REPO_ROOT
                / "docs/research/dual-loop/"
                "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
            )
            with (
                mock.patch.object(
                    validator,
                    "resolve_expected_run_root",
                    return_value=run_root,
                ),
                mock.patch.object(
                    validator,
                    "validate_formal_identities",
                    side_effect=RuntimeEnvironmentError(
                        "synthetic runtime drift"
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = validator.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--protocol",
                        str(protocol),
                        "--dependency-receipt",
                        str(DEPENDENCY_PATH),
                        "--write-results",
                    ]
                )
            self.assertEqual(exit_code, 1)
            validation = validator.load_canonical_json(
                run_root / "execution_validation.json"
            )
            self.assertIn("synthetic runtime drift", validation["errors"])
            self.assertTrue((run_root / "failure_receipt.json").is_file())

    def test_valid_receipt_uses_protocol_claim_ceiling(self) -> None:
        expected_ceiling = "R2 exact claim ceiling"
        result = {
            "schema_version": validator.VALIDATION_SCHEMA,
            "protocol_id": validator.PROTOCOL_ID,
            "protocol_sha256": validator.PROTOCOL_SHA256,
            "status": "PASS",
            "execution_valid": True,
            "terminal": "VALID",
            "event_table_sha256": "a" * 64,
            "analysis_sha256": "b" * 64,
            "scientific_exit": "NO_PRIORITY_IDENTIFIED",
            "claim_ceiling": expected_ceiling,
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run-r3"
            run_root.mkdir()
            (run_root / "formal_start.json").write_bytes(
                validator.canonical_json_bytes({})
            )
            with (
                mock.patch.object(
                    validator,
                    "resolve_expected_run_root",
                    return_value=run_root,
                ),
                mock.patch.object(
                    validator,
                    "validate_execution_package",
                    return_value=result,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = validator.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--protocol",
                        str(Path(temp) / "protocol.json"),
                        "--dependency-receipt",
                        str(Path(temp) / "dependency.json"),
                        "--write-results",
                    ]
                )
            self.assertEqual(exit_code, 0)
            receipt = validator.load_canonical_json(
                run_root / "execution_receipt.json"
            )
            self.assertEqual(receipt["claim_ceiling"], expected_ceiling)

    def test_valid_progress_publication_failure_closes_consumed_terminal(
        self,
    ) -> None:
        result = {
            "schema_version": validator.VALIDATION_SCHEMA,
            "protocol_id": validator.PROTOCOL_ID,
            "protocol_sha256": validator.PROTOCOL_SHA256,
            "status": "PASS",
            "execution_valid": True,
            "terminal": "VALID",
            "event_table_sha256": "a" * 64,
            "analysis_sha256": "b" * 64,
            "scientific_exit": "NO_PRIORITY_IDENTIFIED",
            "claim_ceiling": "bounded",
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run-r3"
            run_root.mkdir()
            (run_root / "formal_start.json").write_bytes(
                validator.canonical_json_bytes({})
            )

            def fail_progress(*_: object, **__: object) -> None:
                raise OSError("synthetic progress replace failure")

            with (
                mock.patch.object(
                    validator,
                    "resolve_expected_run_root",
                    return_value=run_root,
                ),
                mock.patch.object(
                    validator,
                    "validate_execution_package",
                    return_value=result,
                ),
                mock.patch.object(
                    validator,
                    "_atomic_replace",
                    side_effect=fail_progress,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic progress replace failure"
                ):
                    validator.main(
                        [
                            "--run-root",
                            str(run_root),
                            "--protocol",
                            str(Path(temp) / "protocol.json"),
                            "--dependency-receipt",
                            str(Path(temp) / "dependency.json"),
                            "--write-results",
                        ]
                    )
            self.assertTrue(
                (run_root / "execution_validation.json").is_file()
            )
            self.assertTrue((run_root / "failure_receipt.json").is_file())
            self.assertFalse((run_root / "execution_receipt.json").exists())

    def test_existing_terminal_receipt_blocks_validation_republication(
        self,
    ) -> None:
        result = {
            "schema_version": validator.VALIDATION_SCHEMA,
            "protocol_id": validator.PROTOCOL_ID,
            "protocol_sha256": validator.PROTOCOL_SHA256,
            "status": "PASS",
            "execution_valid": True,
            "terminal": "VALID",
            "event_table_sha256": "a" * 64,
            "analysis_sha256": "b" * 64,
            "scientific_exit": "NO_PRIORITY_IDENTIFIED",
            "claim_ceiling": "bounded",
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run-r3"
            run_root.mkdir()
            (run_root / "formal_start.json").write_bytes(
                validator.canonical_json_bytes({})
            )
            original_exclusive_write = validator._exclusive_write

            def fail_validation(path: Path, payload: object) -> None:
                if path.name == "execution_validation.json":
                    raise OSError("synthetic validation publication failure")
                original_exclusive_write(path, payload)

            common_args = [
                "--run-root",
                str(run_root),
                "--protocol",
                str(Path(temp) / "protocol.json"),
                "--dependency-receipt",
                str(Path(temp) / "dependency.json"),
                "--write-results",
            ]
            with (
                mock.patch.object(
                    validator,
                    "resolve_expected_run_root",
                    return_value=run_root,
                ),
                mock.patch.object(
                    validator,
                    "validate_execution_package",
                    return_value=result,
                ) as validate_package,
                mock.patch.object(
                    validator,
                    "_exclusive_write",
                    side_effect=fail_validation,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic validation publication failure"
                ):
                    validator.main(common_args)
                validate_package.reset_mock()
                before = {
                    path.name: path.read_bytes()
                    for path in run_root.iterdir()
                    if path.is_file()
                }
                second_exit = validator.main(common_args)
                after = {
                    path.name: path.read_bytes()
                    for path in run_root.iterdir()
                    if path.is_file()
                }
                validate_package.assert_not_called()
            self.assertEqual(second_exit, 2)
            self.assertEqual(before, after)
            self.assertTrue((run_root / "failure_receipt.json").is_file())
            self.assertFalse((run_root / "execution_receipt.json").exists())

    def test_raw_roi_closure_mutation_is_rejected(self) -> None:
        previous = {
            "captured_at_ns": 0,
            "track_epoch": 1,
            "roi_xywh_normalized": [0.1, 0.1, 0.2, 0.2],
        }
        current = {
            "captured_at_ns": 50_000_000,
            "track_epoch": 1,
            "history_reset": False,
            "roi_xywh_normalized": [0.1, 0.1, 0.22, 0.22],
        }
        expected_rate = __import__("math").log(1.21) / 0.05
        bbox = {"abstention_reason": None, "signed_approach_rate_per_s": expected_rate}
        self.assertIsNotNone(validator._independent_roi_pair(previous, current, bbox))
        bbox["signed_approach_rate_per_s"] += 1e-10
        with self.assertRaisesRegex(validator.ValidationError, "closure mismatch"):
            validator._independent_roi_pair(previous, current, bbox)

    def test_independent_source_and_roi_math_match_producer_ieee_exactly(self) -> None:
        import numpy as np
        import producer

        half_angle = 0.01
        person = {
            "timestamps_ns": np.asarray([0, 10_000_000, 20_000_000], dtype=np.int64),
            "positions": np.asarray(
                [[2.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                dtype=np.float64,
            ),
            "quaternions": np.asarray(
                [[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float64
            ),
        }
        sensor = {
            "timestamps_ns": np.asarray(
                [5_000_000, 15_000_000, 25_000_000], dtype=np.int64
            ),
            "positions": np.asarray(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                dtype=np.float64,
            ),
            "quaternions": np.asarray(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, __import__("math").sin(half_angle), __import__("math").cos(half_angle)],
                    [0.0, 0.0, __import__("math").sin(2 * half_angle), __import__("math").cos(2 * half_angle)],
                ],
                dtype=np.float64,
            ),
        }
        calibration = np.eye(4, dtype=np.float64)
        rotation_angle = 0.3
        calibration[:3, :3] = np.asarray(
            [
                [__import__("math").cos(rotation_angle), -__import__("math").sin(rotation_angle), 0.0],
                [__import__("math").sin(rotation_angle), __import__("math").cos(rotation_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        calibration[0, 3] = 0.1
        produced, produced_reason = producer.source_pair_metrics(
            15_000_000, person, sensor, calibration, 0.0
        )
        independent, independent_reason = validator._independent_source_pair(
            15_000_000, person, sensor, calibration, 0.0
        )
        self.assertEqual(produced_reason, independent_reason)
        assert produced is not None and independent is not None
        self.assertEqual(
            produced["person_approach_component_mps"], independent["person"]
        )
        self.assertEqual(
            produced["sensor_approach_component_mps"], independent["sensor"]
        )
        self.assertEqual(produced["sensor_absolute_share"], independent["share"])
        self.assertEqual(
            produced["camera_translation_speed_mps"], independent["translation"]
        )
        self.assertEqual(
            produced["camera_angular_speed_radps"], independent["angular"]
        )

        same_index_sensor = {
            "timestamps_ns": np.asarray([15_000_000], dtype=np.int64),
            "positions": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
            "quaternions": np.asarray(
                [[0.0, 0.0, __import__("math").sin(half_angle), __import__("math").cos(half_angle)]],
                dtype=np.float64,
            ),
        }
        produced_same, produced_same_reason = producer.source_pair_metrics(
            15_000_000, person, same_index_sensor, calibration, 0.0
        )
        independent_same, independent_same_reason = (
            validator._independent_source_pair(
                15_000_000, person, same_index_sensor, calibration, 0.0
            )
        )
        self.assertEqual(produced_same_reason, independent_same_reason)
        assert produced_same is not None and independent_same is not None
        self.assertEqual(
            produced_same["camera_translation_speed_mps"],
            independent_same["translation"],
        )
        self.assertEqual(
            produced_same["camera_angular_speed_radps"],
            independent_same["angular"],
        )

        boundary_person = copy.deepcopy(person)
        boundary_person["positions"] = np.asarray(
            [
                [2.0, 1.0, 1.0],
                [1.999000005, 1.0, 1.0],
                [1.99800001, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        boundary_sensor = {
            "timestamps_ns": np.asarray(
                [0, 10_000_000, 20_000_000], dtype=np.int64
            ),
            "positions": np.asarray(
                [[1.0, 1.0, 1.0]] * 3, dtype=np.float64
            ),
            "quaternions": np.asarray(
                [[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float64
            ),
        }
        with self.assertRaisesRegex(producer.ProducerError, "deadband"):
            producer.source_pair_metrics(
                5_000_000,
                boundary_person,
                boundary_sensor,
                np.eye(4),
                0.1,
            )
        with self.assertRaisesRegex(validator.ValidationError, "deadband"):
            validator._independent_source_pair(
                5_000_000,
                boundary_person,
                boundary_sensor,
                np.eye(4),
                0.1,
            )

        previous = {
            "captured_at_ns": 0,
            "track_epoch": "epoch",
            "roi_xywh_normalized": [0.1, 0.1, 0.2, 0.2],
        }
        current = {
            "captured_at_ns": 50_000_000,
            "track_epoch": "epoch",
            "history_reset": False,
            "roi_xywh_normalized": [0.11, 0.09, 0.22, 0.21],
        }
        rate = __import__("math").log((0.22 * 0.21) / (0.2 * 0.2)) / 0.05
        bbox = {"abstention_reason": None, "signed_approach_rate_per_s": rate}
        produced_roi, _ = producer._roi_pair(previous, current, bbox)
        independent_roi = validator._independent_roi_pair(previous, current, bbox)
        assert produced_roi is not None and independent_roi is not None
        self.assertEqual(produced_roi["log_area_rate"], independent_roi["rate"])
        self.assertEqual(
            produced_roi["center_velocity"], independent_roi["velocity"]
        )

    def test_independent_event_aggregation_matches_producer_full_keyset(self) -> None:
        import producer

        natural = {
            "event_id": "track-000:event-0000",
            "capture_id": "REVEL_DYNAMIC_V1",
            "target_id": "track-000",
            "anchor_region": "LEFT",
            "truth_state": "approaching",
            "start_timestamp_ns": 100,
            "end_timestamp_ns": 400,
            "eligible_frame_count": 4,
            "duration_s": 0.3,
        }
        binding = {
            "overlap_component_id": "component-0000",
            "time_block_id_60s": 0,
        }
        primary_eval = {
            "correct": False,
            "wrong_signed": True,
            "event_score_per_s": -0.2,
        }
        reference_eval = {
            "correct": True,
            "wrong_signed": False,
            "event_score_per_s": 0.4,
        }
        producer_members: list[dict[str, object]] = []
        validator_members: list[dict[str, object]] = []
        for index in range(4):
            source = {
                "person_approach_component_mps": 0.1 + index,
                "sensor_approach_component_mps": 0.2 + index,
                "signed_approach_mps": 0.3 + 2 * index,
                "sensor_absolute_share": 0.4 + index * 0.01,
                "camera_translation_speed_mps": 0.5 + index,
                "camera_angular_speed_radps": 0.6 + index,
            }
            flow = {
                "abstention_reason": None if index < 3 else "LOW_SUPPORT",
                "signed_approach_rate_per_s": 0.03 * (index + 1),
                "quality": {
                    "components": {
                        "score_mad_per_s": 0.1 + index,
                        "detected_features": 10 + index,
                        "surviving_tracks": 5 + index,
                        "occupied_quadrants": 2 + index,
                        "median_fb_error_px": 0.2 + index,
                    }
                },
            }
            roi = {
                "log_area_rate": 0.2 + index,
                "center_velocity": (0.1 + index, 0.2 + index),
            }
            producer_members.append(
                {
                    "source_frame_index": index,
                    "source_pair": source,
                    "source_missing_reason": None,
                    "roi_pair": roi,
                    "flow": flow,
                }
            )
            validator_members.append(
                {
                    "index": index,
                    "source": {
                        "person": source["person_approach_component_mps"],
                        "sensor": source["sensor_approach_component_mps"],
                        "share": source["sensor_absolute_share"],
                        "translation": source["camera_translation_speed_mps"],
                        "angular": source["camera_angular_speed_radps"],
                    },
                    "source_reason": None,
                    "roi": {
                        "rate": roi["log_area_rate"],
                        "velocity": roi["center_velocity"],
                    },
                    "flow": flow,
                }
            )
        produced = producer._aggregate_event(
            natural,
            binding,
            primary_eval,
            reference_eval,
            producer_members,
        )
        recomputed = validator._independent_event(
            natural,
            binding,
            primary_eval,
            reference_eval,
            validator_members,
        )
        self.assertEqual(produced, recomputed)
        self.assertIn("quality_component_support", recomputed)

    def test_recomputed_required_field_mutation_is_rejected(self) -> None:
        expected = [
            {
                field: None for field in validator.REQUIRED_EVENT_FIELDS
            }
        ]
        expected[0]["event_id"] = "event-0000"
        claimed = copy.deepcopy(expected)
        for field in validator.NULLABLE_SUMMARY_FIELDS:
            if claimed[0][field] is None:
                claimed[0][f"{field}_missing_reason"] = "SYNTHETIC_NULL"
        claimed[0]["eligible_frame_count"] = 9
        expected[0]["eligible_frame_count"] = 8
        with self.assertRaisesRegex(validator.ValidationError, "mismatch"):
            validator.compare_recomputed_event_table(claimed, expected)

    def test_recomputed_missing_reason_string_is_exact(self) -> None:
        expected = [{field: None for field in validator.REQUIRED_EVENT_FIELDS}]
        expected[0]["event_id"] = "event-0000"
        for field in validator.NULLABLE_SUMMARY_FIELDS:
            expected[0][f"{field}_missing_reason"] = "EXPECTED_REASON"
        claimed = copy.deepcopy(expected)
        claimed[0][
            "median_abs_sensor_approach_component_mps_missing_reason"
        ] = "ARBITRARY_NONEMPTY_REASON"
        with self.assertRaisesRegex(validator.ValidationError, "mismatch"):
            validator.compare_recomputed_event_table(claimed, expected)

    def test_analysis_metric_mutation_is_detected_exactly(self) -> None:
        rows = synthetic_rows()
        expected = validator.recompute_analysis(
            rows,
            protocol_sha256=validator.PROTOCOL_SHA256,
            event_table_sha256="0" * 64,
        )
        mutated_rows = copy.deepcopy(rows)
        mutated_rows[8]["median_abs_sensor_approach_component_mps"] = -1.0
        actual = validator.recompute_analysis(
            mutated_rows,
            protocol_sha256=validator.PROTOCOL_SHA256,
            event_table_sha256="0" * 64,
        )
        with self.assertRaisesRegex(validator.ValidationError, "mismatch"):
            validator._assert_exact(actual, expected)

    def test_candidate_block_cannot_be_silently_omitted(self) -> None:
        metric = validator.recompute_metric(
            synthetic_rows(),
            validator.EGO_FIELD,
            "IDENTITY",
        )
        self.assertEqual(
            list(metric["missingness"]["blocks"]["cells"]),
            [str(block_id) for block_id in range(6)],
        )
        claimed = copy.deepcopy(metric)
        del claimed["missingness"]["blocks"]["cells"]["5"]
        with self.assertRaisesRegex(validator.ValidationError, "key/order mismatch"):
            validator._assert_exact(claimed, metric)

    def test_person_competitor_blocks_ego_and_truth_table_is_exclusive(self) -> None:
        rows = synthetic_rows()
        for row in rows:
            row["median_abs_person_approach_component_mps"] = (
                2.0
                if row["primary_error_partition"] == validator.WRONG
                else 0.0
            )
        analysis = validator.recompute_analysis(
            rows,
            protocol_sha256=validator.PROTOCOL_SHA256,
            event_table_sha256="0" * 64,
        )
        self.assertTrue(analysis["routing"]["person_competing"])
        self.assertFalse(analysis["routing"]["ego_candidate"])
        self.assertEqual(
            validator.scientific_exit(
                execution_valid=False,
                global_route_evaluable=True,
                ego_candidate=True,
                temporal_candidate=False,
            ),
            None,
        )
        exits = {
            (ego, temporal): validator.scientific_exit(
                execution_valid=True,
                global_route_evaluable=True,
                ego_candidate=ego,
                temporal_candidate=temporal,
            )
            for ego in (False, True)
            for temporal in (False, True)
        }
        self.assertEqual(exits[(True, False)], "EGO_CANARY_PRIORITY")
        self.assertEqual(exits[(False, True)], "TEMPORAL_TREND_PRIORITY")
        self.assertEqual(exits[(False, False)], "NO_PRIORITY_IDENTIFIED")
        self.assertEqual(exits[(True, True)], "NO_PRIORITY_IDENTIFIED")


if __name__ == "__main__":
    unittest.main()
