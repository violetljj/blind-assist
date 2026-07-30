#!/usr/bin/env python3
"""Join a completed producer ledger to frozen Development truth by event."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from radial_geometry import (
    ARM_BBOX,
    ARM_FLOW,
    ARMS,
    IMPLEMENTATION_ID,
    PARAMETER_SHA256,
    PROTOCOL_ID,
    TTL_NS,
)


PRIMARY_DEADBAND_PER_S = 0.02
REQUIRED_OUTPUT_FIELDS = {
    "protocol_id",
    "implementation_id",
    "parameter_sha256",
    "arm_id",
    "capture_id",
    "source_frame_id",
    "captured_at_ns",
    "available_at_ns",
    "target_id",
    "track_epoch",
    "region",
    "signed_approach_rate_per_s",
    "quality",
    "ttl_ns",
    "valid_until_ns",
    "abstention_reason",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_output_distinct(output_path: Path, protected_paths: Iterable[Path]) -> None:
    resolved_output = output_path.resolve()
    for protected in protected_paths:
        if resolved_output == protected.resolve():
            raise ValueError("evaluation output collides with an input ledger")


def predicted_state(score: float, deadband: float = PRIMARY_DEADBAND_PER_S) -> str:
    if score > deadband:
        return "approaching"
    if score < -deadband:
        return "receding"
    return "quasi_static"


def wrong_signed(truth_state: str, predicted: str | None) -> bool:
    return (truth_state, predicted) in {
        ("approaching", "receding"),
        ("receding", "approaching"),
    }


def _summarize_event_rows(
    arm_rows: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    denominator = int(event["eligible_frame_count"])
    finite = [
        float(row["signed_approach_rate_per_s"])
        for row in arm_rows
        if row.get("abstention_reason") is None
        and isinstance(row.get("signed_approach_rate_per_s"), (int, float))
        and math.isfinite(float(row["signed_approach_rate_per_s"]))
    ]
    coverage = len(finite) / denominator if denominator else 0.0
    evaluable = len(finite) >= 3 and coverage >= 0.50
    score = float(median(finite)) if evaluable else None
    prediction = predicted_state(score) if score is not None else None
    truth_state = str(event["truth_state"])
    correct = bool(evaluable and prediction == truth_state)
    return {
        "event_id": event["event_id"],
        "target_id": event["target_id"],
        "anchor_region": event["anchor_region"],
        "truth_state": truth_state,
        "denominator_rows": denominator,
        "non_abstained_rows": len(finite),
        "coverage": coverage,
        "evaluable": evaluable,
        "event_score_per_s": score,
        "predicted_state": prediction,
        "correct": correct,
        "wrong_signed": bool(evaluable and wrong_signed(truth_state, prediction)),
    }


def _metric_summary(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(event_rows)
    correct = sum(bool(row["correct"]) for row in event_rows)
    wrong = sum(bool(row["wrong_signed"]) for row in event_rows)
    evaluable = sum(bool(row["evaluable"]) for row in event_rows)
    return {
        "events": total,
        "correct_events": correct,
        "correct_fraction": correct / total if total else 0.0,
        "wrong_signed_events": wrong,
        "wrong_signed_fraction": wrong / total if total else 0.0,
        "evaluable_events": evaluable,
        "evaluable_fraction": evaluable / total if total else 0.0,
        "abstained_events": total - evaluable,
        "median_event_coverage": float(median([row["coverage"] for row in event_rows])) if event_rows else 0.0,
    }


def _group_summary(event_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row[field])].append(row)
    return {key: _metric_summary(rows) for key, rows in sorted(grouped.items())}


def _readiness(metrics: dict[str, Any], by_truth: dict[str, Any]) -> bool:
    return bool(
        metrics["correct_fraction"] >= 0.60
        and metrics["wrong_signed_fraction"] <= 0.20
        and metrics["evaluable_fraction"] >= 0.80
        and all(
            by_truth.get(state, {}).get("correct_fraction", 0.0) >= 0.50
            for state in ("approaching", "quasi_static", "receding")
        )
    )


def validate_output_ledger(
    outputs: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    replay_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in replay_rows:
        key = (str(row["source_frame_id"]), str(row["target_id"]))
        if key in replay_by_key:
            raise ValueError("duplicate replay-input key")
        replay_by_key[key] = row
    output_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in outputs:
        missing = REQUIRED_OUTPUT_FIELDS.difference(row)
        if missing:
            raise ValueError(f"output schema missing {sorted(missing)}")
        if (
            row["protocol_id"] != PROTOCOL_ID
            or row["implementation_id"] != IMPLEMENTATION_ID
            or row["parameter_sha256"] != PARAMETER_SHA256
        ):
            raise ValueError("output implementation identity drift")
        if row["arm_id"] not in ARMS:
            raise ValueError("unexpected arm")
        key = (str(row["source_frame_id"]), str(row["target_id"]), str(row["arm_id"]))
        if key in output_by_key:
            raise ValueError("duplicate output key")
        replay_row = replay_by_key.get(key[:2])
        if replay_row is None:
            raise ValueError("producer output keyset drift: extra output")
        for field in ("source_frame_id", "target_id", "track_epoch", "region", "captured_at_ns"):
            if row[field] != replay_row[field]:
                raise ValueError(f"producer output metadata drift: {field}")
        if row["capture_id"] != "REVEL_DYNAMIC_V1":
            raise ValueError("producer output metadata drift: capture_id")
        if row["available_at_ns"] != row["captured_at_ns"]:
            raise ValueError("producer output available-at drift")
        if row["ttl_ns"] != TTL_NS:
            raise ValueError("TTL value drift")
        if row["valid_until_ns"] != row["captured_at_ns"] + row["ttl_ns"]:
            raise ValueError("TTL formula drift")
        quality = row["quality"]
        if not isinstance(quality, dict) or not isinstance(quality.get("components"), dict):
            raise ValueError("quality schema drift")
        quality_score = quality.get("score")
        if (
            isinstance(quality_score, bool)
            or not isinstance(quality_score, (int, float))
            or not math.isfinite(float(quality_score))
            or not 0.0 <= float(quality_score) <= 1.0
        ):
            raise ValueError("quality score drift")
        if row["abstention_reason"] is None:
            value = row["signed_approach_rate_per_s"]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("non-abstained output lacks finite score")
        elif row["signed_approach_rate_per_s"] is not None:
            raise ValueError("abstained output carries a score")
        output_by_key[key] = row
    expected_output_keys = {
        (source_frame_id, target_id, arm)
        for source_frame_id, target_id in replay_by_key
        for arm in ARMS
    }
    if set(output_by_key) != expected_output_keys:
        missing = len(expected_output_keys.difference(output_by_key))
        extra = len(set(output_by_key).difference(expected_output_keys))
        raise ValueError(f"producer output keyset drift: missing={missing}, extra={extra}")
    return output_by_key


def evaluate_records(
    outputs: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_by_key = validate_output_ledger(outputs, replay_rows)

    truth_by_key = {
        (str(row["source_frame_id"]), str(row["target_id"])): row
        for row in truth_rows
    }
    event_frames: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, row in truth_by_key.items():
        if row.get("event_id") is not None and row.get("primary_event_eligible") is True:
            event_frames[str(row["event_id"])].append(key)
    primary_events = [event for event in events if event.get("primary_event_eligible") is True]
    by_arm_events: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for event in primary_events:
        keys = event_frames.get(str(event["event_id"]), [])
        if len(keys) != int(event["eligible_frame_count"]):
            raise ValueError("event truth denominator drift")
        for arm in ARMS:
            arm_rows = []
            for source_frame_id, target_id in keys:
                output = output_by_key.get((source_frame_id, target_id, arm))
                if output is None:
                    raise ValueError("missing fixed-denominator output row")
                arm_rows.append(output)
            by_arm_events[arm].append(_summarize_event_rows(arm_rows, event))

    arm_summaries: dict[str, Any] = {}
    for arm, rows in by_arm_events.items():
        metrics = _metric_summary(rows)
        by_truth = _group_summary(rows, "truth_state")
        arm_summaries[arm] = {
            "overall": metrics,
            "by_target": _group_summary(rows, "target_id"),
            "by_anchor_region": _group_summary(rows, "anchor_region"),
            "by_truth_state": by_truth,
            "readiness_floor_passed": _readiness(metrics, by_truth),
            "events": rows,
        }

    bbox = arm_summaries[ARM_BBOX]
    flow = arm_summaries[ARM_FLOW]
    correct_gain = flow["overall"]["correct_events"] - bbox["overall"]["correct_events"]
    target_gain = {
        target: (
            flow["by_target"].get(target, {}).get("correct_events", 0)
            - bbox["by_target"].get(target, {}).get("correct_events", 0)
        )
        for target in ("track-000", "track-001")
    }
    region_gain = {
        region: (
            flow["by_anchor_region"].get(region, {}).get("correct_events", 0)
            - bbox["by_anchor_region"].get(region, {}).get("correct_events", 0)
        )
        for region in ("LEFT", "CENTER", "RIGHT")
    }
    flow_gate = bool(
        flow["readiness_floor_passed"]
        and correct_gain >= 2
        and flow["overall"]["wrong_signed_events"] <= bbox["overall"]["wrong_signed_events"]
        and bbox["overall"]["evaluable_events"] - flow["overall"]["evaluable_events"] <= 23
        and all(value > 0 for value in target_gain.values())
        and sum(value > 0 for value in region_gain.values()) >= 2
    )
    if flow_gate:
        terminal = "FLOW_CANDIDATE_READY_FOR_FUTURE_CONFIRMATION_DESIGN"
        governance_claim = "IMPLEMENTATION_READY_FOR_CONFIRMATION"
    elif bbox["readiness_floor_passed"]:
        terminal = "BBOX_BASELINE_PREFERRED_FOR_FUTURE_CONFIRMATION_DESIGN"
        governance_claim = "IMPLEMENTATION_READY_FOR_CONFIRMATION"
    else:
        terminal = "BOTH_NOT_READY_FOR_CONFIRMATION"
        governance_claim = "IMPLEMENTATION_NOT_READY"
    return {
        "protocol_id": PROTOCOL_ID,
        "primary_event_count": len(primary_events),
        "candidate_deadband_per_s": PRIMARY_DEADBAND_PER_S,
        "arm_summaries": arm_summaries,
        "comparison": {
            "flow_correct_event_gain": correct_gain,
            "flow_target_correct_gains": target_gain,
            "flow_anchor_region_correct_gains": region_gain,
            "flow_over_bbox_gate_passed": flow_gate,
        },
        "terminal": terminal,
        "governance_claim": governance_claim,
        "claim_ceiling": "SINGLE_CAPTURE_ORACLE_ROI_CONDITIONED_DEVELOPMENT_ONLY",
        "confirmation_execution_authorized": False,
    }


def evaluate_files(
    implementation_lock: Path,
    producer_output: Path,
    producer_receipt: Path,
    replay_input: Path,
    truth_path: Path,
    events_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _assert_output_distinct(
        output_path,
        (
            implementation_lock,
            producer_output,
            producer_receipt,
            replay_input,
            truth_path,
            events_path,
        ),
    )
    lock = json.loads(implementation_lock.read_text(encoding="utf-8"))
    if lock.get("schema") != "blindassist_dual_loop_radial_geometry_implementation_lock_v1":
        raise ValueError("unexpected implementation lock")
    receipt = json.loads(producer_receipt.read_text(encoding="utf-8"))
    producer_sha256 = sha256_file(producer_output)
    if receipt.get("output_sha256") != producer_sha256 or receipt.get("truth_joined") is not False:
        raise ValueError("producer receipt is not a pre-truth hash lock")
    replay_input_sha256 = sha256_file(replay_input)
    if receipt.get("replay_input_sha256") != replay_input_sha256:
        raise ValueError("producer receipt replay-input identity drift")
    frozen_replay = lock.get("producer_contract", {}).get("input_allowlist", {}).get("replay_input", {})
    if replay_input_sha256 != frozen_replay.get("sha256"):
        raise ValueError("replay input differs from implementation lock")
    outputs = read_jsonl(producer_output)
    replay_rows = read_jsonl(replay_input)
    if len(replay_rows) != frozen_replay.get("rows"):
        raise ValueError("replay input row count differs from implementation lock")
    if receipt.get("output_rows") != lock.get("producer_contract", {}).get("output_rows_expected"):
        raise ValueError("producer output row count differs from implementation lock")
    validate_output_ledger(outputs, replay_rows)

    # Truth and event ledgers are not opened until producer identity and exact
    # replay-input x arm completeness have passed.
    frozen_truth = lock.get("evaluator_contract", {}).get("truth", {})
    frozen_events = lock.get("evaluator_contract", {}).get("natural_events", {})
    if sha256_file(truth_path) != frozen_truth.get("sha256"):
        raise ValueError("truth identity differs from implementation lock")
    if sha256_file(events_path) != frozen_events.get("sha256"):
        raise ValueError("event identity differs from implementation lock")
    truth_rows = read_jsonl(truth_path)
    events = read_jsonl(events_path)
    if len(truth_rows) != frozen_truth.get("rows"):
        raise ValueError("truth row count differs from implementation lock")
    if len(events) != frozen_events.get("rows"):
        raise ValueError("event row count differs from implementation lock")
    result = evaluate_records(outputs, truth_rows, events, replay_rows)
    result["producer_output_sha256"] = producer_sha256
    result["producer_receipt_sha256"] = sha256_file(producer_receipt)
    result["replay_input_sha256"] = replay_input_sha256
    result["implementation_lock_sha256"] = sha256_file(implementation_lock)
    result["truth_sha256"] = sha256_file(truth_path)
    result["events_sha256"] = sha256_file(events_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--producer-output", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_files(
        args.implementation_lock,
        args.producer_output,
        args.producer_receipt,
        args.replay_input,
        args.truth,
        args.events,
        args.output,
    )
    print(json.dumps({"terminal": result["terminal"], "primary_events": result["primary_event_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
