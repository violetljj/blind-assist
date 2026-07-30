#!/usr/bin/env python3
"""Independent truth-blind validator for the production temporal A/B producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
IMPLEMENTATION_ID = "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0"
BRANCHES = {
    "PRODUCTION_SEMANTIC_WITH_OBJECT_DETECTOR_TEMPORAL_GEOMETRY_NEUTRALIZED",
    "CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY",
}
SESSION_COUNTS = {
    "defaced_2021-03-27-11-51-18_filtered_lidar_odom": 2239,
    "defaced_2021-03-27-11-55-00_filtered_lidar_odom": 2183,
}
EXPECTED_FRAME_COUNT = 4422
EXPECTED_TRACE_ROWS = 8844
SESSION_MEDIAN_DELTA_NS = {
    "defaced_2021-03-27-11-51-18_filtered_lidar_odom": Fraction(190814671, 2),
    "defaced_2021-03-27-11-55-00_filtered_lidar_odom": Fraction(181057555, 2),
}
MODEL_PATH = "app/src/main/assets/yolo11n_fp16_320.tflite"
LABELS_PATH = "app/src/main/assets/coco_labels.txt"
REQUIRED_FIELDS = {
    "protocol_id",
    "implementation_id",
    "session_id",
    "frame_id",
    "source_capture_timestamp_ns",
    "branch_id",
    "pre_temporal_raw_risk_sha256",
    "detector_output_sha256",
    "detection_count",
    "raw_risk",
    "stable_risk",
    "feedback_triggered",
    "feedback_reason",
    "risk_event",
    "timing",
    "failure",
}
RISK_REQUIRED_FIELDS = {
    "level",
    "direction",
    "proximity",
    "message",
    "urgency_score",
    "risk_score",
    "approach_trend",
    "evidence_state",
    "fusion_summary",
    "approach_score",
    "total_score",
    "source_detection",
}
RISK_NUMERIC_FIELDS = {
    "urgency_score",
    "risk_score",
    "approach_score",
    "total_score",
}
TIMING_FIELDS = {
    "detector_total_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_finite_number(value: Any, *, field: str, line_number: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric at line {line_number}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} is not finite at line {line_number}")


def validate_risk(value: Any, *, field: str, line_number: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} is not an object at line {line_number}")
    missing = RISK_REQUIRED_FIELDS - value.keys()
    if missing:
        raise ValueError(f"{field} missing fields at line {line_number}: {sorted(missing)}")
    for numeric_field in RISK_NUMERIC_FIELDS:
        validate_finite_number(
            value[numeric_field],
            field=f"{field}.{numeric_field}",
            line_number=line_number,
        )


def median_fraction(values: list[int]) -> Fraction:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[middle], 1)
    return Fraction(ordered[middle - 1] + ordered[middle], 2)


def frozen_frame_contract(
    implementation_lock: dict[str, Any],
    *,
    expected_frame_count: int,
    session_counts: dict[str, int],
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, dict[str, Any]], Path]:
    repo_root = Path(implementation_lock["repo_root"])
    input_binding = implementation_lock["input_receipt"]
    input_path = Path(input_binding["path"])
    if sha256_file(input_path) != input_binding["sha256"]:
        raise ValueError("frozen input receipt drift")
    input_receipt = json.loads(input_path.read_text(encoding="utf-8"))
    if (
        input_receipt.get("schema_version")
        != "blindassist.dual_loop_input_preflight.v1"
        or input_receipt.get("protocol_id") != PROTOCOL_ID
        or input_receipt.get("status") != "VALID"
        or input_receipt.get("outcome_blind") is not True
        or input_receipt.get("truth_opened") is not False
        or input_receipt.get("frame_count") != expected_frame_count
    ):
        raise ValueError("frozen input receipt contract mismatch")
    locked_input = implementation_lock["device_prestart"]["input"]
    if (
        input_receipt.get("canonical_rgb_inventory_sha256")
        != locked_input.get("canonical_rgb_inventory_sha256")
    ):
        raise ValueError("input receipt differs from device prestart inventory")

    receipt_sessions = {
        item["session_id"]: item
        for item in input_receipt["sessions"]
    }
    if set(receipt_sessions) != set(session_counts):
        raise ValueError("frozen input session identity mismatch")
    expected_frames: dict[str, list[tuple[str, int]]] = {}
    session_summary: dict[str, dict[str, Any]] = {}
    for session_id, expected_count in session_counts.items():
        session_receipt = receipt_sessions[session_id]
        ledger_path = Path(session_receipt["frame_ledger_path"])
        if not ledger_path.is_absolute():
            ledger_path = repo_root / ledger_path
        if sha256_file(ledger_path) != session_receipt["frame_ledger_sha256"]:
            raise ValueError(f"frozen frame-ledger drift: {session_id}")
        rows = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != expected_count or session_receipt["frame_count"] != expected_count:
            raise ValueError(f"frozen frame-ledger denominator mismatch: {session_id}")
        frames = [
            (str(row["frame_id"]), int(row["source_capture_timestamp_ns"]))
            for row in rows
        ]
        if len({frame_id for frame_id, _ in frames}) != expected_count:
            raise ValueError(f"duplicate frozen frame ID: {session_id}")
        timestamps = [timestamp for _, timestamp in frames]
        deltas = [
            second - first
            for first, second in zip(timestamps, timestamps[1:])
        ]
        if any(delta <= 0 for delta in deltas):
            raise ValueError(f"frozen frame timestamps are not strictly increasing: {session_id}")
        median_delta = median_fraction(deltas) if deltas else None
        frozen_median = SESSION_MEDIAN_DELTA_NS.get(session_id)
        if frozen_median is not None and median_delta != frozen_median:
            raise ValueError(f"frozen session median delta drift: {session_id}")
        expected_frames[session_id] = frames
        session_summary[session_id] = {
            "frame_count": expected_count,
            "frame_ledger_path": str(ledger_path),
            "frame_ledger_sha256": session_receipt["frame_ledger_sha256"],
            "first_source_capture_timestamp_ns": timestamps[0],
            "last_source_capture_timestamp_ns": timestamps[-1],
            "median_delta_ns": (
                {
                    "numerator": median_delta.numerator,
                    "denominator": median_delta.denominator,
                    "decimal": float(median_delta),
                }
                if median_delta is not None
                else None
            ),
        }
    return expected_frames, session_summary, input_path


def validate(
    trace_path: Path,
    producer_receipt_path: Path,
    implementation_lock_path: Path,
    activation_path: Path,
    formal_start_marker_path: Path,
    *,
    expected_frame_count: int = EXPECTED_FRAME_COUNT,
    expected_trace_rows: int = EXPECTED_TRACE_ROWS,
    session_counts: dict[str, int] = SESSION_COUNTS,
) -> dict[str, Any]:
    receipt = json.loads(producer_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version")
        != "blindassist.production_temporal_ab_producer_receipt.v1"
    ):
        raise ValueError("producer receipt schema mismatch")
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("producer protocol_id mismatch")
    if receipt.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("producer implementation_id mismatch")
    if receipt.get("status") != "COMPLETE" or receipt.get("truth_joined") is not False:
        raise ValueError("producer is not COMPLETE truth-blind output")
    if receipt.get("backend") != "qualcomm_qnn_htp":
        raise ValueError("formal backend is not strict Qualcomm QNN HTP")
    if receipt.get("qnn_maven_version") != "2.47.0":
        raise ValueError("formal QNN Maven version mismatch")
    if not isinstance(receipt.get("qnn_runtime_version"), list):
        raise ValueError("formal QNN runtime version is missing")
    if receipt.get("session_count") != len(session_counts):
        raise ValueError("producer session_count mismatch")
    if receipt.get("frame_count") != expected_frame_count:
        raise ValueError("producer frame_count mismatch")
    if receipt.get("detector_invocation_count") != expected_frame_count:
        raise ValueError("detector invocation denominator mismatch")
    if receipt.get("trace_row_count") != expected_trace_rows:
        raise ValueError("producer trace row denominator mismatch")
    if receipt.get("failure_count") != 0:
        raise ValueError("producer failure_count is not zero")
    if receipt.get("formal_start_consumed") is not True:
        raise ValueError("producer did not bind the formal-start marker")
    trace_sha256 = sha256_file(trace_path)
    if receipt.get("trace_sha256") != trace_sha256:
        raise ValueError("producer trace hash mismatch")

    implementation_lock = json.loads(implementation_lock_path.read_text(encoding="utf-8"))
    if (
        implementation_lock.get("schema_version")
        != "blindassist.production_temporal_ab_implementation_lock.v1"
    ):
        raise ValueError("implementation lock schema mismatch")
    if implementation_lock.get("status") != "LOCKED":
        raise ValueError("implementation lock is not LOCKED")
    if implementation_lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("implementation lock protocol mismatch")
    if implementation_lock.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("implementation lock identity mismatch")
    locked_sources = implementation_lock.get("source_sha256", {})
    if receipt.get("model_sha256") != locked_sources.get(MODEL_PATH):
        raise ValueError("producer model identity mismatch")
    if receipt.get("labels_sha256") != locked_sources.get(LABELS_PATH):
        raise ValueError("producer labels identity mismatch")
    locked_prestart = implementation_lock.get("device_prestart", {})
    locked_input = locked_prestart.get("input", {})
    if receipt.get("input_inventory_sha256") != locked_input.get(
        "canonical_rgb_inventory_sha256"
    ):
        raise ValueError("producer input inventory identity mismatch")
    if receipt.get("qnn_runtime_version") != locked_prestart.get("qnn_runtime_version"):
        raise ValueError("producer QNN runtime drift from prestart")
    if receipt.get("device") != locked_prestart.get("device"):
        raise ValueError("producer device identity drift from prestart")
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    if (
        activation.get("schema_version")
        != "blindassist.production_temporal_ab_activation.v1"
        or activation.get("protocol_id") != PROTOCOL_ID
        or activation.get("implementation_id") != IMPLEMENTATION_ID
        or activation.get("status") != "ACTIVATED"
        or activation.get("formal_execution_authorized") is not True
        or activation.get("implementation_lock_sha256")
        != sha256_file(implementation_lock_path)
        or activation.get("installed_app_apk_sha256")
        != implementation_lock["app_apk"]["sha256"]
        or activation.get("installed_test_apk_sha256")
        != implementation_lock["test_apk"]["sha256"]
    ):
        raise ValueError("activation identity mismatch")
    if receipt.get("implementation_lock_sha256") != sha256_file(
        implementation_lock_path
    ):
        raise ValueError("producer implementation-lock identity mismatch")
    if receipt.get("activation_sha256") != sha256_file(activation_path):
        raise ValueError("producer activation identity mismatch")
    formal_start = json.loads(formal_start_marker_path.read_text(encoding="utf-8"))
    if (
        formal_start.get("protocol_id") != PROTOCOL_ID
        or formal_start.get("implementation_id") != IMPLEMENTATION_ID
        or formal_start.get("state") != "FORMAL_STARTED"
        or formal_start.get("first_frame_inference_pending") is not True
    ):
        raise ValueError("formal-start marker identity mismatch")
    expected_frames, frozen_session_summary, input_receipt_path = frozen_frame_contract(
        implementation_lock,
        expected_frame_count=expected_frame_count,
        session_counts=session_counts,
    )

    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    row_count = 0
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"trace line {line_number} missing fields: {sorted(missing)}")
            if row["protocol_id"] != PROTOCOL_ID or row["implementation_id"] != IMPLEMENTATION_ID:
                raise ValueError(f"trace identity mismatch at line {line_number}")
            if row["session_id"] not in session_counts:
                raise ValueError(f"unexpected session at line {line_number}")
            if row["branch_id"] not in BRANCHES:
                raise ValueError(f"unexpected branch at line {line_number}")
            if row["failure"] is not None:
                raise ValueError(f"branch failure at line {line_number}")
            if not isinstance(row["feedback_triggered"], bool):
                raise ValueError(f"feedback_triggered is not boolean at line {line_number}")
            if not isinstance(row["feedback_reason"], str) or not row["feedback_reason"]:
                raise ValueError(f"feedback_reason is invalid at line {line_number}")
            if isinstance(row["source_capture_timestamp_ns"], bool) or not isinstance(
                row["source_capture_timestamp_ns"], int
            ):
                raise ValueError(f"source timestamp is not an integer at line {line_number}")
            validate_risk(row["raw_risk"], field="raw_risk", line_number=line_number)
            validate_risk(row["stable_risk"], field="stable_risk", line_number=line_number)
            if not isinstance(row["risk_event"], dict):
                raise ValueError(f"risk_event is not an object at line {line_number}")
            if not isinstance(row["timing"], dict) or set(row["timing"]) != TIMING_FIELDS:
                raise ValueError(f"timing fields mismatch at line {line_number}")
            for timing_field in TIMING_FIELDS:
                validate_finite_number(
                    row["timing"][timing_field],
                    field=f"timing.{timing_field}",
                    line_number=line_number,
                )
            if (
                isinstance(row["detection_count"], bool)
                or not isinstance(row["detection_count"], int)
                or row["detection_count"] < 0
            ):
                raise ValueError(f"negative detection_count at line {line_number}")
            for field in ("detector_output_sha256", "pre_temporal_raw_risk_sha256"):
                value = str(row[field])
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError(f"invalid {field} at line {line_number}")
            key = (str(row["session_id"]), str(row["frame_id"]))
            branch_rows = pairs[key]
            if row["branch_id"] in branch_rows:
                raise ValueError(f"duplicate branch row: {key}/{row['branch_id']}")
            branch_rows[row["branch_id"]] = row
            row_count += 1
    if row_count != expected_trace_rows:
        raise ValueError(f"trace rows mismatch: {row_count}")
    if len(pairs) != expected_frame_count:
        raise ValueError(f"paired frame denominator mismatch: {len(pairs)}")

    per_session: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for (session_id, frame_id), branch_rows in pairs.items():
        if set(branch_rows) != BRANCHES:
            raise ValueError(f"incomplete branch pair: {session_id}/{frame_id}")
        first, second = branch_rows.values()
        for field in (
            "source_capture_timestamp_ns",
            "detector_output_sha256",
            "pre_temporal_raw_risk_sha256",
            "detection_count",
        ):
            if first[field] != second[field]:
                raise ValueError(f"branch pair mismatch for {field}: {session_id}/{frame_id}")
        per_session[session_id][frame_id] = branch_rows

    session_summary: dict[str, Any] = {}
    for session_id, expected_count in session_counts.items():
        trace_frames = per_session[session_id]
        frozen_frames = expected_frames[session_id]
        if len(trace_frames) != expected_count:
            raise ValueError(f"session frame denominator mismatch: {session_id}")
        if set(trace_frames) != {frame_id for frame_id, _ in frozen_frames}:
            raise ValueError(f"session frame IDs differ from frozen ledger: {session_id}")
        for frame_id, expected_timestamp in frozen_frames:
            branch_rows = trace_frames[frame_id]
            actual_timestamp = int(
                next(iter(branch_rows.values()))["source_capture_timestamp_ns"]
            )
            if actual_timestamp != expected_timestamp:
                raise ValueError(
                    f"trace timestamp differs from frozen ledger: {session_id}/{frame_id}"
                )
        session_summary[session_id] = frozen_session_summary[session_id]

    return {
        "schema_version": "blindassist.production_temporal_ab_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "VALID",
        "truth_opened": False,
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha256,
        "producer_receipt_path": str(producer_receipt_path),
        "producer_receipt_sha256": sha256_file(producer_receipt_path),
        "implementation_lock_path": str(implementation_lock_path),
        "implementation_lock_sha256": sha256_file(implementation_lock_path),
        "activation_path": str(activation_path),
        "activation_sha256": sha256_file(activation_path),
        "formal_start_marker_path": str(formal_start_marker_path),
        "formal_start_marker_sha256": sha256_file(formal_start_marker_path),
        "input_receipt_path": str(input_receipt_path),
        "input_receipt_sha256": sha256_file(input_receipt_path),
        "frame_count": expected_frame_count,
        "trace_row_count": expected_trace_rows,
        "detector_invocation_count": expected_frame_count,
        "branch_pair_mismatch_count": 0,
        "failure_count": 0,
        "sessions": session_summary,
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--formal-start-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    trace_path = args.trace.resolve()
    producer_path = args.producer_receipt.resolve()
    lock_path = args.implementation_lock.resolve()
    activation_path = args.activation.resolve()
    formal_start_marker_path = args.formal_start_marker.resolve()
    validation_path = args.output.resolve()
    seal_path = args.seal_output.resolve()
    if validation_path.exists() or seal_path.exists():
        raise ValueError("validation/seal output already exists")
    result = validate(
        trace_path,
        producer_path,
        lock_path,
        activation_path,
        formal_start_marker_path,
    )
    atomic_json(validation_path, result)
    seal = {
        "schema_version": "blindassist.production_temporal_ab_seal.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "SEALED",
        "truth_opened": False,
        "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)},
        "producer_receipt": {
            "path": str(producer_path),
            "sha256": sha256_file(producer_path),
        },
        "implementation_lock": {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
        },
        "activation": {
            "path": str(activation_path),
            "sha256": sha256_file(activation_path),
        },
        "formal_start_marker": {
            "path": str(formal_start_marker_path),
            "sha256": sha256_file(formal_start_marker_path),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
        },
        "truth_join_authorized": True,
    }
    atomic_json(seal_path, seal)
    print(
        json.dumps(
            {"status": "SEALED", "validation": str(validation_path), "seal": str(seal_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
