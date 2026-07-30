from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import runner
from analysis import analyze_event_table, ordinary_cliff
from contract import METRIC_ORDER, canonical_json_bytes, raw_mad, type7_quantile
from producer import (
    ProducerError,
    _roi_pair,
    read_camera_from_marker,
    source_pair_metrics,
)
from validate_execution_independent import (
    ValidationError,
    read_camera_from_marker_independent,
)


class ContractAndProducerTest(unittest.TestCase):
    def test_progress_record_satisfies_guarded_host_contract(self) -> None:
        record = runner._progress_record(
            state="FORMAL_STARTED",
            phase="formal_started",
            completed_event_count=0,
            status="running",
            started_monotonic=__import__("time").monotonic(),
        )
        required = {
            "phase",
            "completed_units",
            "total_units",
            "throughput",
            "eta_seconds",
            "last_progress_at",
            "status",
        }
        self.assertTrue(required.issubset(record))
        self.assertEqual(record["completed_units"], 0)
        self.assertEqual(record["total_units"], runner.EVENT_COUNT)
        self.assertEqual(record["status"], "running")
        self.assertTrue(str(record["last_progress_at"]).endswith("Z"))

    def test_atomic_progress_replace_failure_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            original = runner.canonical_json_bytes({"state": "original"}) + b"\n"
            path.write_bytes(original)
            with mock.patch.object(
                runner.os,
                "replace",
                side_effect=OSError("synthetic replace failure"),
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic replace failure"
                ):
                    runner._atomic_replace_json(path, {"state": "new"})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                list(path.parent.glob(".progress.json.replace-*")), []
            )

    def test_type7_and_raw_mad_match_frozen_math(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertEqual(type7_quantile(values, 0.25), 7.5)
        self.assertEqual(type7_quantile(values, 0.5), 15.0)
        self.assertEqual(type7_quantile(values, 0.9), 27.0)
        self.assertEqual(raw_mad(values), 10.0)
        self.assertEqual(canonical_json_bytes({"β": 1, "a": 2}), b'{"a":2,"\xce\xb2":1}')

    def test_source_pair_decomposes_person_and_sensor_motion(self) -> None:
        person = {
            "timestamps_ns": np.asarray([0, 10_000_000], dtype=np.int64),
            "positions": np.asarray([[2.0, 0, 0], [1.99, 0, 0]], dtype=np.float64),
            "quaternions": np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float64),
        }
        sensor = {
            "timestamps_ns": np.asarray([0, 10_000_000], dtype=np.int64),
            "positions": np.asarray([[0.5, 0, 0], [0.5, 0, 0]], dtype=np.float64),
            "quaternions": np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float64),
        }
        result, reason = source_pair_metrics(
            5_000_000, person, sensor, np.eye(4), 1.0
        )
        self.assertIsNone(reason)
        self.assertAlmostEqual(result["person_approach_component_mps"], 1.0)
        self.assertAlmostEqual(result["sensor_approach_component_mps"], 0.0)
        self.assertAlmostEqual(result["signed_approach_mps"], 1.0)
        self.assertEqual(result["sensor_absolute_share"], 0.0)

    def test_producer_and_independent_calibration_parsers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            for payload in (
                "T_v_c: [[1.0, 0.0], [0.0, 1.0]]\n",
                (
                    "T_v_c:\n"
                    "  - [.nan, 0.0, 0.0, 0.0]\n"
                    "  - [0.0, 1.0, 0.0, 0.0]\n"
                    "  - [0.0, 0.0, 1.0, 0.0]\n"
                    "  - [0.0, 0.0, 0.0, 1.0]\n"
                ),
            ):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ProducerError):
                        read_camera_from_marker(path)
                    with self.assertRaises(ValidationError):
                        read_camera_from_marker_independent(path)

    def test_roi_pair_uses_previous_target_row_and_exact_bbox_closure(self) -> None:
        previous = {
            "captured_at_ns": 0,
            "track_epoch": "epoch-1",
            "roi_xywh_normalized": [0.0, 0.0, 0.2, 0.2],
        }
        current = {
            "captured_at_ns": 50_000_000,
            "track_epoch": "epoch-1",
            "history_reset": False,
            "roi_xywh_normalized": [0.1, 0.0, 0.3, 0.2],
        }
        expected_rate = math.log(1.5) / 0.05
        result, reason = _roi_pair(
            previous,
            current,
            {
                "abstention_reason": None,
                "signed_approach_rate_per_s": expected_rate,
            },
        )
        self.assertIsNone(reason)
        self.assertAlmostEqual(result["log_area_rate"], expected_rate, places=12)
        self.assertAlmostEqual(result["center_velocity"][0], 3.0)

    def test_prestart_parser_failure_does_not_create_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / runner.FORMAL_OUTPUT_RELATIVE_PATH
            output.parent.mkdir(parents=True)
            activation_path = root / "activation.json"
            lock_path = root / "implementation_lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            repository = {
                "head": "a" * 40,
                "origin_master": "a" * 40,
            }
            activation = {
                "formal_execution_authorized": True,
                "authority": {
                    "formal_execution_authorized": True,
                    "successor_execution_authorized": False,
                    "confirmation_authorized": False,
                    "product_or_safety_authorized": False,
                },
                "repository": repository,
            }
            activation_path.write_bytes(
                runner.canonical_json_bytes(activation) + b"\n"
            )
            probe = {
                "status": "VALID_INHERITED_R2_OPERATIONAL_PROBE",
                "vicon_bag_messages_opened": False,
            }
            protocol = {
                "runtime_environment": {
                    "manifest": {
                        "path": "runtime.json",
                        "sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    },
                    "designated_prestart_probe": dict(probe),
                },
                "frozen_inputs": {
                    "revel_dynamic_bag": {"path": "synthetic.bag"}
                },
            }
            with (
                mock.patch.object(
                    runner, "load_protocol", return_value=protocol
                ),
                mock.patch.object(
                    runner,
                    "validate_implementation_lock",
                    return_value={"status": "VALID"},
                ),
                mock.patch.object(runner, "_load_json", return_value=activation),
                mock.patch.object(
                    runner,
                    "validate_activation_identity",
                    return_value={"status": "VALID_IDENTITY", "failures": []},
                ),
                mock.patch.object(
                    runner, "_validate_runtime_invocation", return_value=None
                ),
                mock.patch.object(
                    runner, "_git_clean_and_at_origin", return_value=repository
                ),
                mock.patch.object(
                    runner,
                    "validate_runtime_manifest",
                    return_value={
                        "status": "VALID",
                        "manifest_sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    },
                ),
                mock.patch.object(
                    runner,
                    "probe_designated_vicon_message",
                    return_value=probe,
                ),
                mock.patch.object(
                    runner,
                    "prestart_dependency_smoke",
                    return_value={},
                ),
                mock.patch.object(
                    runner,
                    "_load_science_runtime",
                    return_value={
                        "load_prestart_bundle": lambda *_: object(),
                        "read_camera_from_marker": mock.Mock(
                            side_effect=ProducerError("synthetic parser failure")
                        ),
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    ProducerError, "synthetic parser failure"
                ):
                    runner.run_producer(
                        repo_root=root,
                        activation_path=activation_path,
                        implementation_lock_path=lock_path,
                        output_root=output,
                    )
            self.assertFalse(output.exists())

    def test_all_prestart_smoke_failures_leave_namespace_absent(self) -> None:
        good_probe = {
            "status": "VALID_INHERITED_R2_OPERATIONAL_PROBE",
            "vicon_bag_messages_opened": False,
        }
        changed = np.eye(4)
        changed[0, 0] = 2.0
        nonfinite = np.eye(4)
        nonfinite[0, 0] = math.nan
        cases = (
            {
                "name": "probe_drift",
                "probe": {
                    "status": "DRIFT",
                    "vicon_bag_messages_opened": False,
                },
            },
            {
                "name": "dependency_smoke_exception",
                "smoke_error": RuntimeError("synthetic smoke failure"),
            },
            {
                "name": "producer_parser_exception",
                "producer_error": ProducerError("producer parser failure"),
            },
            {
                "name": "independent_parser_exception",
                "independent_error": ValidationError(
                    "independent parser failure"
                ),
            },
            {
                "name": "shape_mismatch",
                "producer_value": np.eye(3),
            },
            {
                "name": "value_mismatch",
                "independent_value": changed,
            },
            {
                "name": "nonfinite_result",
                "producer_value": nonfinite,
            },
        )
        for case in cases:
            with self.subTest(case=case["name"]), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                output = root / runner.FORMAL_OUTPUT_RELATIVE_PATH
                output.parent.mkdir(parents=True)
                activation_path = root / "activation.json"
                lock_path = root / "implementation_lock.json"
                lock_path.write_text("{}\n", encoding="utf-8")
                repository = {
                    "head": "a" * 40,
                    "origin_master": "a" * 40,
                }
                activation = {
                    "formal_execution_authorized": True,
                    "authority": {
                        "formal_execution_authorized": True,
                        "successor_execution_authorized": False,
                        "confirmation_authorized": False,
                        "product_or_safety_authorized": False,
                    },
                    "repository": repository,
                }
                activation_path.write_bytes(
                    runner.canonical_json_bytes(activation) + b"\n"
                )
                protocol = {
                    "runtime_environment": {
                        "manifest": {
                            "path": "runtime.json",
                            "sha256": "b" * 64,
                            "tree_sha256": "c" * 64,
                        },
                        "designated_prestart_probe": dict(good_probe),
                    },
                    "frozen_inputs": {
                        "revel_dynamic_bag": {"path": "synthetic.bag"}
                    },
                }
                producer_parser = mock.Mock(
                    return_value=case.get("producer_value", np.eye(4)),
                    side_effect=case.get("producer_error"),
                )
                independent_parser = mock.Mock(
                    return_value=case.get("independent_value", np.eye(4)),
                    side_effect=case.get("independent_error"),
                )
                late_stage = mock.Mock()
                bundle_loader = mock.Mock()
                smoke = mock.Mock(
                    return_value={},
                    side_effect=case.get("smoke_error"),
                )
                with (
                    mock.patch.object(
                        runner, "load_protocol", return_value=protocol
                    ),
                    mock.patch.object(
                        runner,
                        "validate_implementation_lock",
                        return_value={"status": "VALID"},
                    ),
                    mock.patch.object(
                        runner, "_load_json", return_value=activation
                    ),
                    mock.patch.object(
                        runner,
                        "validate_activation_identity",
                        return_value={
                            "status": "VALID_IDENTITY",
                            "failures": [],
                        },
                    ),
                    mock.patch.object(
                        runner,
                        "_validate_runtime_invocation",
                        return_value=None,
                    ),
                    mock.patch.object(
                        runner,
                        "_git_clean_and_at_origin",
                        return_value=repository,
                    ),
                    mock.patch.object(
                        runner,
                        "validate_runtime_manifest",
                        return_value={
                            "status": "VALID",
                            "manifest_sha256": "b" * 64,
                            "tree_sha256": "c" * 64,
                        },
                    ),
                    mock.patch.object(
                        runner,
                        "probe_designated_vicon_message",
                        return_value=case.get("probe", good_probe),
                    ),
                    mock.patch.object(
                        runner,
                        "prestart_dependency_smoke",
                        smoke,
                    ),
                    mock.patch.object(
                        runner,
                        "_load_science_runtime",
                        return_value={
                            "load_prestart_bundle": bundle_loader,
                            "read_camera_from_marker": producer_parser,
                            "read_vicon_tracks": late_stage,
                            "build_event_table": late_stage,
                        },
                    ),
                    mock.patch.object(
                        runner,
                        "_load_independent_calibration_parser",
                        return_value=independent_parser,
                    ),
                ):
                    with self.assertRaises(Exception):
                        runner.run_producer(
                            repo_root=root,
                            activation_path=activation_path,
                            implementation_lock_path=lock_path,
                            output_root=output,
                        )
                self.assertFalse(output.exists())
                bundle_loader.assert_not_called()
                late_stage.assert_not_called()

    def test_post_marker_progress_failure_publishes_consumed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / runner.FORMAL_OUTPUT_RELATIVE_PATH
            output.parent.mkdir(parents=True)
            activation_path = root / "activation.json"
            lock_path = root / "implementation_lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            repository = {
                "head": "a" * 40,
                "origin_master": "a" * 40,
            }
            activation = {
                "formal_execution_authorized": True,
                "authority": {
                    "formal_execution_authorized": True,
                    "successor_execution_authorized": False,
                    "confirmation_authorized": False,
                    "product_or_safety_authorized": False,
                },
                "repository": repository,
            }
            activation_path.write_bytes(
                runner.canonical_json_bytes(activation) + b"\n"
            )
            original_atomic_json = runner._atomic_json

            def fail_initial_progress(path: Path, value: object) -> None:
                if path.name == "progress.json":
                    raise OSError("synthetic progress failure")
                original_atomic_json(path, value)

            probe = {
                "status": "VALID_OPERATIONAL_PROBE",
                "topic": "synthetic",
                "deserialized_message_count": 1,
                "pose_values_retained": False,
                "d0_metrics_computed": False,
            }
            protocol = {
                "runtime_environment": {
                    "manifest": {
                        "path": "runtime.json",
                        "sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    },
                    "designated_prestart_probe": {
                        key: value
                        for key, value in probe.items()
                        if key != "status"
                    },
                },
                "frozen_inputs": {
                    "revel_dynamic_bag": {"path": "synthetic.bag"}
                },
            }
            with (
                mock.patch.object(
                    runner, "load_protocol", return_value=protocol
                ),
                mock.patch.object(
                    runner,
                    "validate_implementation_lock",
                    return_value={"status": "VALID"},
                ),
                mock.patch.object(runner, "_load_json", return_value=activation),
                mock.patch.object(
                    runner,
                    "validate_activation_identity",
                    return_value={"status": "VALID_IDENTITY", "failures": []},
                ),
                mock.patch.object(
                    runner, "_validate_runtime_invocation", return_value=None
                ),
                mock.patch.object(
                    runner, "_git_clean_and_at_origin", return_value=repository
                ),
                mock.patch.object(
                    runner,
                    "validate_runtime_manifest",
                    return_value={
                        "status": "VALID",
                        "manifest_sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    },
                ),
                mock.patch.object(
                    runner,
                    "probe_designated_vicon_message",
                    return_value=probe,
                ),
                mock.patch.object(
                    runner,
                    "_load_science_runtime",
                    return_value={
                        "load_prestart_bundle": lambda *_: object(),
                        "read_camera_from_marker": lambda *_: np.eye(4),
                    },
                ),
                mock.patch.object(
                    runner,
                    "_load_independent_calibration_parser",
                    return_value=lambda *_: np.eye(4),
                ),
                mock.patch.object(
                    runner,
                    "prestart_dependency_smoke",
                    return_value={},
                ),
                mock.patch.object(
                    runner, "_atomic_json", side_effect=fail_initial_progress
                ),
            ):
                with self.assertRaisesRegex(OSError, "synthetic progress failure"):
                    runner.run_producer(
                        repo_root=root,
                        activation_path=activation_path,
                        implementation_lock_path=lock_path,
                        output_root=output,
                    )
            self.assertTrue((output / "formal_start.json").is_file())
            failure = __import__("json").loads(
                (output / "failure_receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(failure["consumed"])
            self.assertFalse(failure["rerun_authorized"])

    def test_post_marker_bundle_failure_publishes_consumed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / runner.FORMAL_OUTPUT_RELATIVE_PATH
            output.parent.mkdir(parents=True)
            activation_path = root / "activation.json"
            lock_path = root / "implementation_lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            repository = {
                "head": "a" * 40,
                "origin_master": "a" * 40,
            }
            activation = {
                "formal_execution_authorized": True,
                "authority": {
                    "formal_execution_authorized": True,
                    "successor_execution_authorized": False,
                    "confirmation_authorized": False,
                    "product_or_safety_authorized": False,
                },
                "repository": repository,
            }
            activation_path.write_bytes(
                runner.canonical_json_bytes(activation) + b"\n"
            )
            probe = {
                "status": "VALID_INHERITED_OPERATIONAL_PROBE",
                "topic": "synthetic",
                "deserialized_message_count": 0,
                "vicon_bag_messages_opened": False,
                "pose_values_retained": False,
                "d0_metrics_computed": False,
            }
            protocol = {
                "runtime_environment": {
                    "manifest": {
                        "path": "runtime.json",
                        "sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    },
                    "designated_prestart_probe": {
                        key: value
                        for key, value in probe.items()
                        if key != "status"
                    },
                },
                "frozen_inputs": {
                    "revel_dynamic_bag": {"path": "synthetic.bag"},
                    "revel_calibration": {"path": "synthetic.yaml"},
                },
            }

            validation_scopes: list[bool] = []

            def validate_with_marker_order(
                *_: object, **kwargs: object
            ) -> dict[str, object]:
                scope = bool(kwargs["validate_scientific_inputs"])
                validation_scopes.append(scope)
                if scope:
                    self.assertTrue((output / "formal_start.json").is_file())
                    self.assertTrue((output / "progress.json").is_file())
                else:
                    self.assertFalse(output.exists())
                return {
                    "status": "VALID",
                    "scientific_inputs_validated": scope,
                }

            def fail_bundle(*_: object) -> object:
                self.assertTrue((output / "formal_start.json").is_file())
                self.assertTrue((output / "progress.json").is_file())
                raise RuntimeError("synthetic bundle failure")

            with (
                mock.patch.object(
                    runner, "load_protocol", return_value=protocol
                ),
                mock.patch.object(
                    runner,
                    "validate_implementation_lock",
                    side_effect=validate_with_marker_order,
                ),
                mock.patch.object(runner, "_load_json", return_value=activation),
                mock.patch.object(
                    runner,
                    "validate_activation_identity",
                    return_value={"status": "VALID_IDENTITY", "failures": []},
                ),
                mock.patch.object(
                    runner, "_validate_runtime_invocation", return_value=None
                ),
                mock.patch.object(
                    runner, "_git_clean_and_at_origin", return_value=repository
                ),
                mock.patch.object(
                    runner,
                    "validate_runtime_manifest",
                    return_value={
                        "status": "VALID",
                        "manifest_sha256": "b" * 64,
                        "tree_sha256": "c" * 64,
                    },
                ),
                mock.patch.object(
                    runner,
                    "probe_designated_vicon_message",
                    return_value=probe,
                ),
                mock.patch.object(
                    runner,
                    "_load_science_runtime",
                    return_value={
                        "load_prestart_bundle": fail_bundle,
                        "read_camera_from_marker": lambda *_: np.eye(4),
                    },
                ),
                mock.patch.object(
                    runner,
                    "_load_independent_calibration_parser",
                    return_value=lambda *_: np.eye(4),
                ),
                mock.patch.object(
                    runner,
                    "prestart_dependency_smoke",
                    return_value={},
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "synthetic bundle failure"
                ):
                    runner.run_producer(
                        repo_root=root,
                        activation_path=activation_path,
                        implementation_lock_path=lock_path,
                        output_root=output,
                    )
            failure = __import__("json").loads(
                (output / "failure_receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(failure["consumed"])
            self.assertFalse(failure["rerun_authorized"])
            self.assertIsNone(failure["scientific_exit"])
            self.assertFalse((output / "event_table.jsonl").exists())
            self.assertEqual(validation_scopes, [False, True])


class AnalysisTest(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        diagnostic_fields = METRIC_ORDER[4:]
        for block in range(6):
            for group_index, group in enumerate(("CORRECT", "WRONG_SIGNED")):
                for index in range(8):
                    wrong = group == "WRONG_SIGNED"
                    row: dict[str, object] = {
                        "event_id": f"event-{block}-{group_index}-{index}",
                        "target_id": f"track-00{index % 2}",
                        "anchor_region": ("LEFT", "CENTER")[index % 2],
                        "truth_state": ("approaching", "receding")[index % 2],
                        "overlap_component_id": f"component-{block}-{group_index}-{index}",
                        "time_block_id_60s": block,
                        "primary_error_partition": group,
                        "median_abs_sensor_approach_component_mps": 1.0 if wrong else 0.0,
                        "median_abs_person_approach_component_mps": 1.0 if wrong else 0.0,
                        "median_flow_score_mad_per_s": 1.0 if wrong else 0.0,
                        "median_surviving_tracks": 0.0 if wrong else 1.0,
                        "sensor_absolute_share": 0.6,
                    }
                    for field in diagnostic_fields:
                        row[field] = 1.0 if wrong else 0.0
                    rows.append(row)
        return rows

    def test_cliff_delta_ties_are_exact_zero(self) -> None:
        self.assertEqual(ordinary_cliff([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_person_competitor_blocks_ego_and_temporal_route_remains_unique(self) -> None:
        result = analyze_event_table(self._rows(), "0" * 64)
        self.assertEqual(result["metric_order"], list(METRIC_ORDER))
        self.assertTrue(result["routing"]["global_route_evaluable"])
        self.assertTrue(result["routing"]["person_competing"])
        self.assertFalse(result["routing"]["ego_candidate"])
        self.assertTrue(result["routing"]["temporal_candidate"])
        self.assertEqual(
            result["routing"]["provisional_scientific_exit"],
            "TEMPORAL_TREND_PRIORITY",
        )
        for field in METRIC_ORDER[4:]:
            metric = result["metrics"][field]
            self.assertEqual(metric["role"], "DIAGNOSTIC_ONLY")
            self.assertFalse(metric["robust_support"])
            self.assertFalse(metric["material_contradiction"])


if __name__ == "__main__":
    unittest.main()
