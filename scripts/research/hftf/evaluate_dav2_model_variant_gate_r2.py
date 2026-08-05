#!/usr/bin/env python3
"""Fail-closed mechanical successor to the frozen DA V2 P1-R1 evaluator."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_dav2_model_variant_gate_r0 as r0
from evaluate_dav2_model_variant_gate_r1 import (
    load_geometry_rows,
    truth_change_summary,
    truth_geometry_summary,
)

SCHEMA = "blindassist_dav2_model_variant_gate_r2_result"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_le(value: Any, ceiling: Any) -> bool:
    return finite_number(value) and finite_number(ceiling) and float(value) <= float(ceiling)


def safe_ge(value: Any, floor: Any) -> bool:
    return finite_number(value) and finite_number(floor) and float(value) >= float(floor)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(
    r2_protocol_path: Path,
    r1_protocol_path: Path,
    r0_protocol_path: Path,
    roster_path: Path,
    source_root: Path,
    baseline_depth_path: Path,
    candidate_depth_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    protocol = json.loads(r2_protocol_path.read_text(encoding="utf-8"))
    r1_protocol = json.loads(r1_protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "blindassist_dav2_model_variant_gate_r2_protocol":
        raise ValueError("R2 protocol schema mismatch")
    if r0.sha256_file(Path(__file__).resolve()) != protocol["evaluator_sha256"]:
        raise ValueError("R2 evaluator source hash mismatch")
    if r0.sha256_file(r1_protocol_path) != protocol["parent_r1_protocol_sha256"]:
        raise ValueError("R2 parent R1 protocol hash mismatch")
    if r0.sha256_file(r0_protocol_path) != r1_protocol["parent_r0_protocol_sha256"]:
        raise ValueError("R2 parent R0 protocol hash mismatch")
    if r0.sha256_file(roster_path) != r1_protocol["roster_sha256"]:
        raise ValueError("R2 roster hash mismatch")
    if r0.sha256_file(baseline_depth_path) != r1_protocol["baseline_depth_sha256"]:
        raise ValueError("R2 baseline depth hash mismatch")

    original_task_summary: Callable[[list[dict[str, Any]], str], dict[str, Any]] = r0._task_summary

    def fail_closed_task_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        summary = original_task_summary(rows, key)
        replacements = {
            "clearance_mae_m": math.inf,
            "collision_agreement": -math.inf,
            "false_clear_rate_all_known_decisions": math.inf,
            "temporal_clearance_delta_mae_m": math.inf,
        }
        for metric, sentinel in replacements.items():
            if not finite_number(summary.get(metric)):
                summary[metric] = sentinel
        return summary

    r0._task_summary = fail_closed_task_summary
    try:
        r0_result = r0.evaluate(
            r0_protocol_path,
            roster_path,
            source_root,
            baseline_depth_path,
            candidate_depth_path,
            candidate_id,
        )
    finally:
        r0._task_summary = original_task_summary

    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    rows = load_geometry_rows(roster, source_root, baseline_depth_path, candidate_depth_path)
    baseline_truth = truth_geometry_summary(rows, "baseline")
    candidate_truth = truth_geometry_summary(rows, "candidate")
    changes = truth_change_summary(rows)
    excluded = set(r1_protocol["r0_gates_replaced"])
    inherited_gates = {
        key: value
        for key, value in r0_result["engineering_noninferiority_gates"].items()
        if key not in excluded
    }
    tolerances = r1_protocol["truth_referenced_tolerances"]
    truth_gates = {
        "truth_status_exact_agreement": safe_ge(
            candidate_truth["truth_status_exact_agreement"],
            baseline_truth["truth_status_exact_agreement"]
            - float(tolerances["truth_status_exact_agreement_drop"]),
        ),
        "truth_geometry_state_exact_agreement": safe_ge(
            candidate_truth["truth_geometry_state_exact_agreement"],
            baseline_truth["truth_geometry_state_exact_agreement"]
            - float(tolerances["truth_geometry_state_exact_agreement_drop"]),
        ),
        "truth_transition_change_agreement": safe_ge(
            candidate_truth["truth_transition_change_agreement"],
            baseline_truth["truth_transition_change_agreement"]
            - float(tolerances["truth_transition_change_agreement_drop"]),
        ),
        "false_block_rate": safe_le(
            candidate_truth["false_block_rate_all_known"],
            baseline_truth["false_block_rate_all_known"]
            + float(tolerances["false_block_rate_all_known_increase"]),
        ),
        "harmful_change_rate": safe_le(
            changes["harmful_change_rate"],
            float(tolerances["maximum_harmful_change_rate"]),
        ),
        "net_truth_decision_change": safe_ge(changes["net_beneficial_changes"], 0),
    }
    all_gates = {**inherited_gates, **truth_gates}
    passed = all(all_gates.values())
    undefined = []
    candidate_task = r0_result["candidate"]["task"]
    for name in (
        "clearance_mae_m",
        "collision_agreement",
        "false_clear_rate_all_known_decisions",
        "temporal_clearance_delta_mae_m",
    ):
        if not finite_number(candidate_task.get(name)):
            undefined.append(name)
    for name in (
        "truth_geometry_state_exact_agreement",
        "truth_transition_change_agreement",
        "false_block_rate_all_known",
    ):
        if not finite_number(candidate_truth.get(name)):
            undefined.append(name)
    if not finite_number(changes.get("harmful_change_rate")):
        undefined.append("harmful_change_rate")
    result = {
        "schema": SCHEMA,
        "protocol_sha256": r0.sha256_file(r2_protocol_path),
        "mechanical_change_from_r1": "undefined or non-finite gated metrics deterministically fail",
        "parent_r0_result": r0_result,
        "baseline_truth_geometry": baseline_truth,
        "candidate_truth_geometry": candidate_truth,
        "candidate_truth_changes_vs_baseline": changes,
        "undefined_candidate_metrics": sorted(set(undefined)),
        "inherited_r0_gates": inherited_gates,
        "truth_referenced_gates": truth_gates,
        "engineering_noninferiority_gates": all_gates,
        "engineering_noninferiority_passed": passed,
        "historical_terminal_relabel_authorized": False,
        "terminal": (
            "MODEL_VARIANT_R2_ENGINEERING_NONINFERIORITY_PASS"
            if passed
            else "MODEL_VARIANT_R2_ENGINEERING_NONINFERIORITY_FAIL"
        ),
    }
    return json_safe(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-protocol", type=Path, required=True)
    parser.add_argument("--r1-protocol", type=Path, required=True)
    parser.add_argument("--r0-protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-depth", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.r2_protocol.resolve(),
        args.r1_protocol.resolve(),
        args.r0_protocol.resolve(),
        args.roster.resolve(),
        args.source_root.resolve(),
        args.baseline_depth.resolve(),
        args.candidate_depth.resolve(),
        args.candidate_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
