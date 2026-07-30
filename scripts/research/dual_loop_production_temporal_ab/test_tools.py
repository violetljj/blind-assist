from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import evaluate_trace
    import validate_producer
except ModuleNotFoundError:
    from scripts.research.dual_loop_production_temporal_ab import evaluate_trace
    from scripts.research.dual_loop_production_temporal_ab import validate_producer


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def risk_fixture() -> dict[str, object]:
    return {
        "level": "NONE",
        "direction": "NONE",
        "proximity": "FAR",
        "message": "fixture",
        "urgency_score": 0.0,
        "risk_score": 0.0,
        "approach_trend": "UNKNOWN",
        "evidence_state": "NO_SUPPORTED_TARGET_EVIDENCE",
        "fusion_summary": "NONE",
        "approach_score": 0.0,
        "total_score": 0.0,
        "source_detection": None,
    }


def timing_fixture() -> dict[str, float]:
    return {
        "detector_total_ms": 1.0,
        "preprocess_ms": 0.1,
        "inference_ms": 0.8,
        "postprocess_ms": 0.1,
    }


def validator_lock_and_activation(
    root: Path,
    sessions: dict[str, int],
) -> tuple[Path, Path, Path]:
    input_sessions = []
    for session_index, (session_id, count) in enumerate(sessions.items()):
        ledger = root / f"{session_id}.frames.jsonl"
        rows = [
            {
                "frame_id": f"{index:06d}",
                "source_capture_timestamp_ns": 1000 + session_index + index * 10,
            }
            for index in range(count)
        ]
        ledger.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        input_sessions.append(
            {
                "session_id": session_id,
                "frame_count": count,
                "frame_ledger_path": str(ledger),
                "frame_ledger_sha256": sha256(ledger),
            }
        )
    input_receipt = root / "input.json"
    write_json(
        input_receipt,
        {
            "schema_version": "blindassist.dual_loop_input_preflight.v1",
            "protocol_id": validate_producer.PROTOCOL_ID,
            "status": "VALID",
            "outcome_blind": True,
            "truth_opened": False,
            "frame_count": sum(sessions.values()),
            "canonical_rgb_inventory_sha256": "e" * 64,
            "sessions": input_sessions,
        },
    )
    lock = root / "lock.json"
    write_json(
        lock,
        {
            "schema_version": "blindassist.production_temporal_ab_implementation_lock.v1",
            "protocol_id": validate_producer.PROTOCOL_ID,
            "implementation_id": validate_producer.IMPLEMENTATION_ID,
            "status": "LOCKED",
            "repo_root": str(root),
            "source_sha256": {
                validate_producer.MODEL_PATH: "c" * 64,
                validate_producer.LABELS_PATH: "d" * 64,
            },
            "app_apk": {"sha256": "1" * 64},
            "test_apk": {"sha256": "2" * 64},
            "input_receipt": {
                "path": str(input_receipt),
                "sha256": sha256(input_receipt),
            },
            "device_prestart": {
                "qnn_runtime_version": [0, 24, 0],
                "input": {"canonical_rgb_inventory_sha256": "e" * 64},
                "device": {"model": "fixture"},
            },
        },
    )
    activation = root / "activation.json"
    write_json(
        activation,
        {
            "schema_version": "blindassist.production_temporal_ab_activation.v1",
            "protocol_id": validate_producer.PROTOCOL_ID,
            "implementation_id": validate_producer.IMPLEMENTATION_ID,
            "status": "ACTIVATED",
            "formal_execution_authorized": True,
            "implementation_lock_sha256": sha256(lock),
            "installed_app_apk_sha256": "1" * 64,
            "installed_test_apk_sha256": "2" * 64,
        },
    )
    formal_marker = root / "formal_start.json"
    write_json(
        formal_marker,
        {
            "protocol_id": validate_producer.PROTOCOL_ID,
            "implementation_id": validate_producer.IMPLEMENTATION_ID,
            "state": "FORMAL_STARTED",
            "first_frame_inference_pending": True,
        },
    )
    return lock, activation, formal_marker


def bind_producer_authorization(
    producer: Path,
    lock: Path,
    activation: Path,
) -> None:
    payload = json.loads(producer.read_text(encoding="utf-8"))
    payload["implementation_lock_sha256"] = sha256(lock)
    payload["activation_sha256"] = sha256(activation)
    write_json(producer, payload)


def evaluator_fixture(
    root: Path,
    *,
    a_valid_ids: set[str],
    b_valid_ids: set[str],
    b_earlier_ids: set[str] | None = None,
) -> tuple[Path, Path]:
    b_earlier_ids = b_earlier_ids or set()
    sessions = list(evaluate_trace.SESSION_MEDIAN_DELTA_NS)
    positive_counts = {sessions[0]: 5, sessions[1]: 3}
    negative_counts = {sessions[0]: 3, sessions[1]: 4}
    items: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    positive_index = 1
    negative_index = 1
    for session_index, session_id in enumerate(sessions):
        origin = 20_000_000_000_000 + session_index * 100_000_000_000
        for branch in (evaluate_trace.BRANCH_A, evaluate_trace.BRANCH_B):
            rows.append(
                {
                    "session_id": session_id,
                    "source_capture_timestamp_ns": origin,
                    "branch_id": branch,
                    "feedback_triggered": False,
                }
            )
        gain = int(evaluate_trace.SESSION_MEDIAN_DELTA_NS[session_id]) + 1
        for local_index in range(positive_counts[session_id]):
            item_id = f"P{positive_index:03d}"
            positive_index += 1
            start = (local_index + 1) * 1_000_000_000
            end = start + 2 * gain
            items.append(
                {
                    "item_id": item_id,
                    "session_id": session_id,
                    "item_kind": "positive_event",
                    "valid_interval_ns": [start, end],
                    "premature_interval_ns": [start - 500_000_000, start],
                }
            )
            if item_id in a_valid_ids:
                rows.append(
                    {
                        "session_id": session_id,
                        "source_capture_timestamp_ns": origin + start + gain,
                        "branch_id": evaluate_trace.BRANCH_A,
                        "feedback_triggered": True,
                    }
                )
            if item_id in b_valid_ids:
                b_offset = start if item_id in b_earlier_ids else start + gain
                rows.append(
                    {
                        "session_id": session_id,
                        "source_capture_timestamp_ns": origin + b_offset,
                        "branch_id": evaluate_trace.BRANCH_B,
                        "feedback_triggered": True,
                    }
                )
        for local_index in range(negative_counts[session_id]):
            item_id = f"N{negative_index:03d}"
            negative_index += 1
            start = 20_000_000_000 + local_index * 1_000_000_000
            items.append(
                {
                    "item_id": item_id,
                    "session_id": session_id,
                    "item_kind": "negative_window",
                    "interval_ns": [start, start + 100_000_000],
                }
            )
    excluded = [
        {
            "item_id": "P007_ZERO",
            "session_id": sessions[1],
            "item_kind": "positive_event",
            "valid_interval_ns": [0, 1],
            "premature_interval_ns": [0, 0],
        },
        {
            "item_id": "P009_ZERO",
            "session_id": sessions[1],
            "item_kind": "positive_event",
            "valid_interval_ns": [0, 1],
            "premature_interval_ns": [0, 0],
        },
    ]
    trace = root / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    membership = root / "membership.json"
    write_json(
        membership,
        {
            "schema_version": "blindassist.dual_loop_truth_membership_preflight.v1",
            "status": "VALID",
            "protocol_id": evaluate_trace.PROTOCOL_ID,
            "candidate_output_opened": False,
            "raw_truth_item_count": 17,
            "fixed_scored_item_denominator": 15,
            "scoreable_positive_count": 8,
            "scoreable_negative_count": 7,
            "cross_item_or_class_frame_overlap_count": 0,
            "session_scored_denominators": {
                sessions[0]: {"positive": 5, "negative": 3, "total": 8},
                sessions[1]: {"positive": 3, "negative": 4, "total": 7},
            },
            "scoreable_positive_ids": [
                item["item_id"]
                for item in items
                if item["item_kind"] == "positive_event"
            ],
            "temporal_scoring_not_evaluable_positive_ids": [
                "P007_ZERO",
                "P009_ZERO",
            ],
            "item_membership": items + excluded,
        },
    )
    lock = root / "lock.json"
    write_json(
        lock,
        {
            "truth_membership_receipt": {
                "path": str(membership),
                "sha256": sha256(membership),
            }
        },
    )
    producer = root / "producer.json"
    activation = root / "activation.json"
    formal_marker = root / "formal_start.json"
    validation = root / "validation.json"
    write_json(producer, {"status": "COMPLETE"})
    write_json(activation, {"status": "ACTIVATED"})
    write_json(formal_marker, {"state": "FORMAL_STARTED"})
    write_json(
        validation,
        {
            "schema_version": "blindassist.production_temporal_ab_validation.v1",
            "protocol_id": evaluate_trace.PROTOCOL_ID,
            "implementation_id": "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0",
            "status": "VALID",
            "truth_opened": False,
            "trace_sha256": sha256(trace),
            "implementation_lock_sha256": sha256(lock),
            "activation_sha256": sha256(activation),
            "producer_receipt_sha256": sha256(producer),
            "formal_start_marker_sha256": sha256(formal_marker),
        },
    )
    seal = root / "seal.json"
    write_json(
        seal,
        {
            "schema_version": "blindassist.production_temporal_ab_seal.v1",
            "protocol_id": evaluate_trace.PROTOCOL_ID,
            "implementation_id": "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0",
            "status": "SEALED",
            "truth_opened": False,
            "truth_join_authorized": True,
            "trace": {"path": str(trace), "sha256": sha256(trace)},
            "producer_receipt": {
                "path": str(producer),
                "sha256": sha256(producer),
            },
            "implementation_lock": {
                "path": str(lock),
                "sha256": sha256(lock),
            },
            "activation": {
                "path": str(activation),
                "sha256": sha256(activation),
            },
            "formal_start_marker": {
                "path": str(formal_marker),
                "sha256": sha256(formal_marker),
            },
            "validation": {
                "path": str(validation),
                "sha256": sha256(validation),
            },
        },
    )
    return seal, membership


class ProducerValidatorTest(unittest.TestCase):
    def test_rejects_nonfinite_numeric_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_producer.validate_finite_number(
                float("nan"),
                field="timing.inference_ms",
                line_number=7,
            )

    def test_accepts_complete_paired_truth_blind_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            rows = []
            sessions = {"s1": 1, "s2": 1}
            for session_index, session_id in enumerate(sessions):
                for branch in sorted(validate_producer.BRANCHES):
                    rows.append(
                        {
                            "protocol_id": validate_producer.PROTOCOL_ID,
                            "implementation_id": validate_producer.IMPLEMENTATION_ID,
                            "session_id": session_id,
                            "frame_id": "000000",
                            "source_capture_timestamp_ns": 1000 + session_index,
                            "branch_id": branch,
                            "pre_temporal_raw_risk_sha256": "a" * 64,
                            "detector_output_sha256": "b" * 64,
                            "detection_count": 1,
                            "raw_risk": risk_fixture(),
                            "stable_risk": risk_fixture(),
                            "feedback_triggered": False,
                            "feedback_reason": "NO_FEEDBACK_RISK",
                            "risk_event": {},
                            "timing": timing_fixture(),
                            "failure": None,
                        }
                    )
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            producer = root / "producer.json"
            write_json(
                producer,
                {
                    "schema_version": "blindassist.production_temporal_ab_producer_receipt.v1",
                    "protocol_id": validate_producer.PROTOCOL_ID,
                    "implementation_id": validate_producer.IMPLEMENTATION_ID,
                    "status": "COMPLETE",
                    "truth_joined": False,
                    "backend": "qualcomm_qnn_htp",
                    "qnn_maven_version": "2.47.0",
                    "qnn_runtime_version": [0, 24, 0],
                    "session_count": 2,
                    "frame_count": 2,
                    "detector_invocation_count": 2,
                    "trace_row_count": 4,
                    "trace_sha256": sha256(trace),
                    "model_sha256": "c" * 64,
                    "labels_sha256": "d" * 64,
                    "input_inventory_sha256": "e" * 64,
                    "device": {"model": "fixture"},
                    "failure_count": 0,
                    "formal_start_consumed": True,
                },
            )
            lock, activation, formal_marker = (
                validator_lock_and_activation(root, sessions)
            )
            bind_producer_authorization(producer, lock, activation)

            result = validate_producer.validate(
                trace,
                producer,
                lock,
                activation,
                formal_marker,
                expected_frame_count=2,
                expected_trace_rows=4,
                session_counts=sessions,
            )

            self.assertEqual("VALID", result["status"])
            self.assertEqual(2, result["frame_count"])
            self.assertEqual(0, result["branch_pair_mismatch_count"])

    def test_rejects_branch_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            rows = []
            for index, branch in enumerate(sorted(validate_producer.BRANCHES)):
                rows.append(
                    {
                        "protocol_id": validate_producer.PROTOCOL_ID,
                        "implementation_id": validate_producer.IMPLEMENTATION_ID,
                        "session_id": "s1",
                        "frame_id": "000000",
                        "source_capture_timestamp_ns": 1000,
                        "branch_id": branch,
                        "pre_temporal_raw_risk_sha256": "a" * 64,
                        "detector_output_sha256": str(index) * 64,
                        "detection_count": 1,
                        "raw_risk": risk_fixture(),
                        "stable_risk": risk_fixture(),
                        "feedback_triggered": False,
                        "feedback_reason": "NO_FEEDBACK_RISK",
                        "risk_event": {},
                        "timing": timing_fixture(),
                        "failure": None,
                    }
                )
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            producer = root / "producer.json"
            write_json(
                producer,
                {
                    "schema_version": "blindassist.production_temporal_ab_producer_receipt.v1",
                    "protocol_id": validate_producer.PROTOCOL_ID,
                    "implementation_id": validate_producer.IMPLEMENTATION_ID,
                    "status": "COMPLETE",
                    "truth_joined": False,
                    "backend": "qualcomm_qnn_htp",
                    "qnn_maven_version": "2.47.0",
                    "qnn_runtime_version": [0, 24, 0],
                    "session_count": 1,
                    "frame_count": 1,
                    "detector_invocation_count": 1,
                    "trace_row_count": 2,
                    "trace_sha256": sha256(trace),
                    "model_sha256": "c" * 64,
                    "labels_sha256": "d" * 64,
                    "input_inventory_sha256": "e" * 64,
                    "device": {"model": "fixture"},
                    "failure_count": 0,
                    "formal_start_consumed": True,
                },
            )
            lock, activation, formal_marker = (
                validator_lock_and_activation(root, {"s1": 1})
            )
            bind_producer_authorization(producer, lock, activation)
            with self.assertRaisesRegex(ValueError, "branch pair mismatch"):
                validate_producer.validate(
                    trace,
                    producer,
                    lock,
                    activation,
                    formal_marker,
                    expected_frame_count=1,
                    expected_trace_rows=2,
                    session_counts={"s1": 1},
                )

    def test_rejects_timestamp_drift_from_frozen_frame_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            rows = []
            for branch in sorted(validate_producer.BRANCHES):
                rows.append(
                    {
                        "protocol_id": validate_producer.PROTOCOL_ID,
                        "implementation_id": validate_producer.IMPLEMENTATION_ID,
                        "session_id": "s1",
                        "frame_id": "000000",
                        "source_capture_timestamp_ns": 1001,
                        "branch_id": branch,
                        "pre_temporal_raw_risk_sha256": "a" * 64,
                        "detector_output_sha256": "b" * 64,
                        "detection_count": 1,
                        "raw_risk": risk_fixture(),
                        "stable_risk": risk_fixture(),
                        "feedback_triggered": False,
                        "feedback_reason": "NO_FEEDBACK_RISK",
                        "risk_event": {},
                        "timing": timing_fixture(),
                        "failure": None,
                    }
                )
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            producer = root / "producer.json"
            write_json(
                producer,
                {
                    "schema_version": "blindassist.production_temporal_ab_producer_receipt.v1",
                    "protocol_id": validate_producer.PROTOCOL_ID,
                    "implementation_id": validate_producer.IMPLEMENTATION_ID,
                    "status": "COMPLETE",
                    "truth_joined": False,
                    "backend": "qualcomm_qnn_htp",
                    "qnn_maven_version": "2.47.0",
                    "qnn_runtime_version": [0, 24, 0],
                    "session_count": 1,
                    "frame_count": 1,
                    "detector_invocation_count": 1,
                    "trace_row_count": 2,
                    "trace_sha256": sha256(trace),
                    "model_sha256": "c" * 64,
                    "labels_sha256": "d" * 64,
                    "input_inventory_sha256": "e" * 64,
                    "device": {"model": "fixture"},
                    "failure_count": 0,
                    "formal_start_consumed": True,
                },
            )
            lock, activation, formal_marker = (
                validator_lock_and_activation(root, {"s1": 1})
            )
            bind_producer_authorization(producer, lock, activation)
            with self.assertRaisesRegex(ValueError, "frozen ledger"):
                validate_producer.validate(
                    trace,
                    producer,
                    lock,
                    activation,
                    formal_marker,
                    expected_frame_count=1,
                    expected_trace_rows=2,
                    session_counts={"s1": 1},
                )


class EvaluatorTest(unittest.TestCase):
    def test_all_scientific_terminal_routes_are_deterministic(self) -> None:
        all_positive = {f"P{index:03d}" for index in range(1, 9)}
        common_four = {"P001", "P002", "P006", "P007"}
        b_with_two_session_improvements = common_four | {"P003", "P008"}
        cases = [
            (
                "NO_INCREMENT",
                all_positive,
                all_positive,
                set(),
            ),
            (
                "RISK_DISCRIMINATION",
                common_four,
                b_with_two_session_improvements,
                set(),
            ),
            (
                "MULTIPLE_INCREMENT",
                common_four,
                b_with_two_session_improvements,
                common_four,
            ),
            (
                "RISK_DISCRIMINATION_WITH_EARLY_RESPONSE_NOT_EVALUABLE",
                {"P001", "P006"},
                {"P001", "P003", "P006", "P008"},
                set(),
            ),
            (
                "EARLY_RESPONSE_NOT_EVALUABLE_RISK_DISCRIMINATION_NO_INCREMENT",
                set(),
                set(),
                set(),
            ),
        ]
        for terminal, a_valid, b_valid, b_earlier in cases:
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as temporary:
                seal, membership = evaluator_fixture(
                    Path(temporary),
                    a_valid_ids=a_valid,
                    b_valid_ids=b_valid,
                    b_earlier_ids=b_earlier,
                )
                result = evaluate_trace.evaluate(seal, membership)
                self.assertEqual(terminal, result["scientific_terminal"])

    def test_recomputes_early_response_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            membership = root / "membership.json"
            validation = root / "validation.json"
            sessions = list(evaluate_trace.SESSION_MEDIAN_DELTA_NS)
            positive_counts = {sessions[0]: 5, sessions[1]: 3}
            negative_counts = {sessions[0]: 3, sessions[1]: 4}
            items = []
            trace_rows = []

            def row(session: str, timestamp: int, branch: str, triggered: bool) -> dict[str, object]:
                return {
                    "session_id": session,
                    "source_capture_timestamp_ns": timestamp,
                    "branch_id": branch,
                    "feedback_triggered": triggered,
                }

            positive_index = 1
            negative_index = 1
            for session_index, session in enumerate(sessions):
                origin = 10_000_000_000_000 + session_index * 100_000_000_000
                for branch in (evaluate_trace.BRANCH_A, evaluate_trace.BRANCH_B):
                    trace_rows.append(row(session, origin, branch, False))
                for local_index in range(positive_counts[session]):
                    item_id = f"P{positive_index:03d}"
                    positive_index += 1
                    start = (local_index + 1) * 1_000_000_000
                    end = start + 100_000_000
                    items.append(
                        {
                            "item_id": item_id,
                            "session_id": session,
                            "item_kind": "positive_event",
                            "valid_interval_ns": [start, end],
                            "premature_interval_ns": [start - 500_000_000, start],
                        }
                    )
                    trace_rows.extend(
                        [
                            row(session, origin + start, evaluate_trace.BRANCH_A, False),
                            row(session, origin + start, evaluate_trace.BRANCH_B, True),
                            row(session, origin + end, evaluate_trace.BRANCH_A, True),
                            row(session, origin + end, evaluate_trace.BRANCH_B, False),
                        ]
                    )
                for local_index in range(negative_counts[session]):
                    item_id = f"N{negative_index:03d}"
                    negative_index += 1
                    start = 20_000_000_000 + local_index * 1_000_000_000
                    items.append(
                        {
                            "item_id": item_id,
                            "session_id": session,
                            "item_kind": "negative_window",
                            "interval_ns": [start, start + 100_000_000],
                        }
                    )
                    for branch in (evaluate_trace.BRANCH_A, evaluate_trace.BRANCH_B):
                        trace_rows.append(row(session, origin + start, branch, False))
            trace.write_text("".join(json.dumps(value) + "\n" for value in trace_rows), encoding="utf-8")
            write_json(
                membership,
                {
                    "schema_version": "blindassist.dual_loop_truth_membership_preflight.v1",
                    "status": "VALID",
                    "protocol_id": evaluate_trace.PROTOCOL_ID,
                    "candidate_output_opened": False,
                    "raw_truth_item_count": 17,
                    "fixed_scored_item_denominator": 15,
                    "scoreable_positive_count": 8,
                    "scoreable_negative_count": 7,
                    "cross_item_or_class_frame_overlap_count": 0,
                    "session_scored_denominators": {
                        sessions[0]: {"positive": 5, "negative": 3, "total": 8},
                        sessions[1]: {"positive": 3, "negative": 4, "total": 7},
                    },
                    "scoreable_positive_ids": [item["item_id"] for item in items if item["item_kind"] == "positive_event"],
                    "temporal_scoring_not_evaluable_positive_ids": ["P007_ZERO", "P009_ZERO"],
                    "item_membership": items
                    + [
                        {
                            "item_id": "P007_ZERO",
                            "session_id": sessions[1],
                            "item_kind": "positive_event",
                            "valid_interval_ns": [0, 1],
                            "premature_interval_ns": [0, 0],
                        },
                        {
                            "item_id": "P009_ZERO",
                            "session_id": sessions[1],
                            "item_kind": "positive_event",
                            "valid_interval_ns": [0, 1],
                            "premature_interval_ns": [0, 0],
                        },
                    ],
                },
            )
            write_json(
                validation,
                {
                    "schema_version": "blindassist.production_temporal_ab_validation.v1",
                    "status": "VALID",
                    "protocol_id": evaluate_trace.PROTOCOL_ID,
                    "implementation_id": "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0",
                    "truth_opened": False,
                    "trace_sha256": sha256(trace),
                },
            )
            lock = root / "lock.json"
            write_json(
                lock,
                {
                    "truth_membership_receipt": {
                        "path": str(membership),
                        "sha256": sha256(membership),
                    }
                },
            )
            producer = root / "producer.json"
            activation = root / "activation.json"
            formal_marker = root / "formal_start.json"
            write_json(producer, {"status": "COMPLETE"})
            write_json(activation, {"status": "ACTIVATED"})
            write_json(formal_marker, {"state": "FORMAL_STARTED"})
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            validation_payload.update(
                {
                    "implementation_lock_sha256": sha256(lock),
                    "activation_sha256": sha256(activation),
                    "producer_receipt_sha256": sha256(producer),
                    "formal_start_marker_sha256": sha256(formal_marker),
                }
            )
            write_json(validation, validation_payload)
            seal = root / "seal.json"
            write_json(
                seal,
                {
                    "schema_version": "blindassist.production_temporal_ab_seal.v1",
                    "protocol_id": evaluate_trace.PROTOCOL_ID,
                    "implementation_id": "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0",
                    "status": "SEALED",
                    "truth_opened": False,
                    "truth_join_authorized": True,
                    "trace": {"path": str(trace), "sha256": sha256(trace)},
                    "producer_receipt": {
                        "path": str(producer),
                        "sha256": sha256(producer),
                    },
                    "implementation_lock": {
                        "path": str(lock),
                        "sha256": sha256(lock),
                    },
                    "activation": {
                        "path": str(activation),
                        "sha256": sha256(activation),
                    },
                    "formal_start_marker": {
                        "path": str(formal_marker),
                        "sha256": sha256(formal_marker),
                    },
                    "validation": {
                        "path": str(validation),
                        "sha256": sha256(validation),
                    },
                },
            )

            result = evaluate_trace.evaluate(seal, membership)

            self.assertEqual("EARLY_RESPONSE", result["scientific_terminal"])
            self.assertTrue(result["early_response"]["success"])
            self.assertFalse(result["risk_discrimination"]["success"])
            self.assertEqual(8, result["early_response"]["eligible_pair_count"])
            self.assertEqual(17, len(result["truth_item_table"]))
            self.assertEqual(2, len(result["session_guardrails"]))
            self.assertEqual(
                0,
                result["branch_metrics"][evaluate_trace.BRANCH_B][
                    "premature_alert_event_count"
                ],
            )
            membership.write_text(
                membership.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "implementation lock"):
                evaluate_trace.evaluate(seal, membership)


if __name__ == "__main__":
    unittest.main()
