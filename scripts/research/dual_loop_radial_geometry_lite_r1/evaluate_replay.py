#!/usr/bin/env python3
"""Evaluate an R1 producer ledger against frozen REveL Development truth."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from radial_geometry import (
    ARM_BBOX,
    ARM_FLOW,
    ARMS,
    IMPLEMENTATION_ID,
    PARAMETER_SHA256,
    PROTOCOL_ID,
    TTL_NS,
)


FORMAL_REPLAY_ROWS = 13_014
FORMAL_OUTPUT_ROWS = 26_028
FORMAL_PRIMARY_EVENTS = 469
FORMAL_SHAPE_CHANGE_OPPORTUNITIES = 32
FORMAL_SHAPE_CHANGE_ARM_ROWS = 64
SCIENTIFIC_GATE_CONTRACT: dict[str, Any] = {
    "candidate_deadband_per_s": 0.02,
    "event_evaluable": {
        "minimum_finite_rows": 3,
        "minimum_coverage": 0.50,
    },
    "readiness_floor": {
        "minimum_correct_fraction": 0.60,
        "maximum_wrong_signed_fraction": 0.20,
        "minimum_evaluable_fraction": 0.80,
        "minimum_each_truth_state_correct_fraction": 0.50,
    },
    "flow_over_bbox": {
        "minimum_correct_event_gain": 2,
        "wrong_signed_events_must_not_increase": True,
        "maximum_evaluable_event_loss": 23,
        "positive_correct_gain_required_for_targets": [
            "track-000",
            "track-001",
        ],
        "minimum_regions_with_positive_correct_gain": 2,
        "minimum_distinct_events_accounting_for_correct_gain": 2,
    },
    "fixed_primary_event_denominator": FORMAL_PRIMARY_EVENTS,
    "non_evaluable_events_count_as_incorrect": True,
    "wrong_sign_pairs": [
        ["approaching", "receding"],
        ["receding", "approaching"],
    ],
}
SCIENTIFIC_GATE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        SCIENTIFIC_GATE_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _load_r0_evaluator() -> Any:
    r0_dir = (
        Path(__file__).resolve().parents[1]
        / "dual_loop_radial_geometry_lite_r0"
    )
    geometry_path = r0_dir / "radial_geometry.py"
    geometry_spec = importlib.util.spec_from_file_location(
        "dual_loop_radial_geometry_lite_r0_eval_core",
        geometry_path,
    )
    if geometry_spec is None or geometry_spec.loader is None:
        raise ImportError("cannot load immutable R0 geometry")
    geometry = importlib.util.module_from_spec(geometry_spec)
    sys.modules[geometry_spec.name] = geometry
    geometry_spec.loader.exec_module(geometry)

    evaluator_path = r0_dir / "evaluate_replay.py"
    evaluator_spec = importlib.util.spec_from_file_location(
        "dual_loop_radial_geometry_lite_r0_eval_core",
        evaluator_path,
    )
    if evaluator_spec is None or evaluator_spec.loader is None:
        raise ImportError("cannot load immutable R0 evaluator")
    previous = sys.modules.get("radial_geometry")
    sys.modules["radial_geometry"] = geometry
    try:
        evaluator = importlib.util.module_from_spec(evaluator_spec)
        sys.modules[evaluator_spec.name] = evaluator
        evaluator_spec.loader.exec_module(evaluator)
    finally:
        if previous is None:
            sys.modules.pop("radial_geometry", None)
        else:
            sys.modules["radial_geometry"] = previous
    return evaluator


_R0 = _load_r0_evaluator()
_R0.PROTOCOL_ID = PROTOCOL_ID
_R0.IMPLEMENTATION_ID = IMPLEMENTATION_ID
_R0.PARAMETER_SHA256 = PARAMETER_SHA256
_R0.TTL_NS = TTL_NS
_R0.ARM_BBOX = ARM_BBOX
_R0.ARM_FLOW = ARM_FLOW
_R0.ARMS = ARMS
_ORIGINAL_R0_VALIDATE = _R0.validate_output_ledger
_ORIGINAL_R0_EVALUATE = _R0.evaluate_records

PRIMARY_DEADBAND_PER_S = _R0.PRIMARY_DEADBAND_PER_S
REQUIRED_OUTPUT_FIELDS = _R0.REQUIRED_OUTPUT_FIELDS
sha256_file = _R0.sha256_file
read_jsonl = _R0.read_jsonl
predicted_state = _R0.predicted_state
wrong_signed = _R0.wrong_signed
_summarize_event_rows = _R0._summarize_event_rows
_metric_summary = _R0._metric_summary
_group_summary = _R0._group_summary
_readiness = _R0._readiness
_assert_output_distinct = _R0._assert_output_distinct


def validate_output_ledger(
    outputs: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    output_by_key = _ORIGINAL_R0_VALIDATE(outputs, replay_rows)
    shape_rows = [
        row for row in outputs
        if row.get("abstention_reason") == "FRAME_SHAPE_CHANGE"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in shape_rows:
        key = (str(row["source_frame_id"]), str(row["target_id"]))
        grouped.setdefault(key, []).append(row)
        if row["signed_approach_rate_per_s"] is not None:
            raise ValueError("shape-change abstention carries a score")
        quality = row["quality"]
        if float(quality.get("score", -1.0)) != 0.0:
            raise ValueError("shape-change abstention quality must be zero")
        components = quality.get("components", {})
        previous_shape = components.get("previous_frame_shape_hw")
        current_shape = components.get("current_frame_shape_hw")
        if (
            not isinstance(previous_shape, list)
            or not isinstance(current_shape, list)
            or len(previous_shape) != 2
            or len(current_shape) != 2
            or previous_shape == current_shape
        ):
            raise ValueError("shape-change dimensions are missing or equal")
    for rows in grouped.values():
        if {str(row["arm_id"]) for row in rows} != set(ARMS):
            raise ValueError("shape-change opportunity does not abstain both arms")
        if len(rows) != len(ARMS):
            raise ValueError("shape-change opportunity arm multiplicity drift")
    if len(replay_rows) == FORMAL_REPLAY_ROWS:
        if len(outputs) != FORMAL_OUTPUT_ROWS:
            raise ValueError("formal output row denominator drift")
        if (
            len(grouped) != FORMAL_SHAPE_CHANGE_OPPORTUNITIES
            or len(shape_rows) != FORMAL_SHAPE_CHANGE_ARM_ROWS
        ):
            raise ValueError("formal shape-change denominator drift")
    return output_by_key


_R0.validate_output_ledger = validate_output_ledger


def _shape_text_to_hw(value: str) -> list[int]:
    parts = value.split("x")
    if len(parts) != 2:
        raise ValueError("source-audit shape encoding drift")
    return [int(parts[0]), int(parts[1])]


def validate_shape_audit_binding(
    outputs: list[dict[str, Any]],
    source_shape_audit: dict[str, Any],
) -> None:
    if (
        source_shape_audit.get("status") != "SOURCE_SHAPE_AUDIT_COMPLETE"
        or source_shape_audit.get("truth_or_event_accessed") is not False
        or source_shape_audit.get("shape_mismatch_pair_count")
        != FORMAL_SHAPE_CHANGE_OPPORTUNITIES
        or source_shape_audit.get("expected_common_shape_abstention_arm_rows")
        != FORMAL_SHAPE_CHANGE_ARM_ROWS
    ):
        raise ValueError("source-shape audit terminal drift")
    expected = {
        (
            str(item["current_source_frame_id"]),
            str(item["target_id"]),
        ): {
            "previous": _shape_text_to_hw(str(item["previous_shape"])),
            "current": _shape_text_to_hw(str(item["current_shape"])),
        }
        for item in source_shape_audit.get("shape_mismatches", [])
    }
    actual: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in outputs:
        if row.get("abstention_reason") == "FRAME_SHAPE_CHANGE":
            key = (str(row["source_frame_id"]), str(row["target_id"]))
            actual.setdefault(key, []).append(row)
    if set(actual) != set(expected):
        raise ValueError("shape-change opportunity keyset differs from source audit")
    for key, rows in actual.items():
        expected_shapes = expected[key]
        components = [
            row["quality"]["components"]
            for row in rows
        ]
        if len(components) != len(ARMS) or any(
            item != components[0] for item in components[1:]
        ):
            raise ValueError("shape-change arm components differ")
        if (
            components[0].get("previous_frame_shape_hw")
            != expected_shapes["previous"]
            or components[0].get("current_frame_shape_hw")
            != expected_shapes["current"]
        ):
            raise ValueError("shape-change dimensions differ from source audit")


def evaluate_records(
    outputs: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _ORIGINAL_R0_EVALUATE(outputs, truth_rows, events, replay_rows)
    bbox_events = {
        str(row["event_id"]): row
        for row in result["arm_summaries"][ARM_BBOX]["events"]
    }
    improved_event_ids = sorted(
        str(row["event_id"])
        for row in result["arm_summaries"][ARM_FLOW]["events"]
        if bool(row["correct"])
        and not bool(bbox_events[str(row["event_id"])]["correct"])
    )
    dominance_passed = len(improved_event_ids) >= 2
    result["comparison"]["flow_distinct_improved_event_ids"] = (
        improved_event_ids
    )
    result["comparison"]["flow_single_event_dominance_gate_passed"] = (
        dominance_passed
    )
    if result["comparison"]["flow_over_bbox_gate_passed"] and not dominance_passed:
        raise AssertionError("R0 single-event-dominance gate drift")
    if len(replay_rows) == FORMAL_REPLAY_ROWS:
        if result["primary_event_count"] != FORMAL_PRIMARY_EVENTS:
            raise ValueError("formal primary-event denominator drift")
    result["scientific_gate_contract_sha256"] = (
        SCIENTIFIC_GATE_CONTRACT_SHA256
    )
    result["shape_change_contract"] = {
        "opportunities": FORMAL_SHAPE_CHANGE_OPPORTUNITIES,
        "arm_rows": FORMAL_SHAPE_CHANGE_ARM_ROWS,
        "denominator_changed": False,
    }
    return result


_R0.evaluate_records = evaluate_records


def evaluate_files(
    implementation_lock: Path,
    expected_implementation_lock_sha256: str,
    source_shape_audit_path: Path,
    producer_output: Path,
    producer_receipt: Path,
    replay_input: Path,
    truth_path: Path,
    events_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("evaluation output already exists")
    implementation_lock_sha256 = sha256_file(implementation_lock)
    if implementation_lock_sha256 != expected_implementation_lock_sha256:
        raise ValueError("implementation lock SHA-256 differs from activation")
    lock = json.loads(implementation_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol_id") != PROTOCOL_ID
        or lock.get("implementation_id") != IMPLEMENTATION_ID
        or lock.get("parameter_sha256") != PARAMETER_SHA256
        or lock.get("scientific_gate_contract_sha256")
        != SCIENTIFIC_GATE_CONTRACT_SHA256
    ):
        raise ValueError("implementation lock identity drift")
    receipt = json.loads(producer_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PRODUCER_COMPLETE"
        or receipt.get("mode") != "formal"
        or receipt.get("truth_joined") is not False
        or receipt.get("input_rows") != FORMAL_REPLAY_ROWS
        or receipt.get("output_rows") != FORMAL_OUTPUT_ROWS
        or receipt.get("shape_change_opportunities")
        != FORMAL_SHAPE_CHANGE_OPPORTUNITIES
        or receipt.get("shape_change_arm_rows")
        != FORMAL_SHAPE_CHANGE_ARM_ROWS
        or receipt.get("arm_ids") != list(ARMS)
        or receipt.get("implementation_lock_sha256")
        != expected_implementation_lock_sha256
    ):
        raise ValueError("producer receipt is not the frozen formal terminal")
    source_binding = lock.get("bindings", {}).get("source_shape_audit", {})
    if sha256_file(source_shape_audit_path) != source_binding.get("sha256"):
        raise ValueError("source-shape audit identity drift")
    source_shape_audit = json.loads(
        source_shape_audit_path.read_text(encoding="utf-8")
    )
    producer_sha256 = sha256_file(producer_output)
    replay_sha256 = sha256_file(replay_input)
    frozen_replay = (
        lock.get("producer_contract", {})
        .get("input_allowlist", {})
        .get("replay_input", {})
    )
    if (
        receipt.get("output_sha256") != producer_sha256
        or receipt.get("replay_input_sha256") != replay_sha256
        or replay_sha256 != frozen_replay.get("sha256")
        or source_shape_audit.get("replay_input_sha256") != replay_sha256
    ):
        raise ValueError("producer/replay/source-audit identity drift")
    outputs = read_jsonl(producer_output)
    replay_rows = read_jsonl(replay_input)
    if (
        len(outputs) != FORMAL_OUTPUT_ROWS
        or len(replay_rows) != FORMAL_REPLAY_ROWS
    ):
        raise ValueError("formal producer/replay row denominator drift")
    validate_output_ledger(outputs, replay_rows)
    validate_shape_audit_binding(outputs, source_shape_audit)
    temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    try:
        result = _R0.evaluate_files(
            implementation_lock,
            producer_output,
            producer_receipt,
            replay_input,
            truth_path,
            events_path,
            temporary,
        )
        result["scientific_gate_contract"] = SCIENTIFIC_GATE_CONTRACT
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_path)
        return result
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument(
        "--expected-implementation-lock-sha256",
        required=True,
    )
    parser.add_argument("--source-shape-audit", type=Path, required=True)
    parser.add_argument("--producer-output", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_files(
        args.implementation_lock,
        args.expected_implementation_lock_sha256,
        args.source_shape_audit,
        args.producer_output,
        args.producer_receipt,
        args.replay_input,
        args.truth,
        args.events,
        args.output,
    )
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "primary_events": result["primary_event_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
