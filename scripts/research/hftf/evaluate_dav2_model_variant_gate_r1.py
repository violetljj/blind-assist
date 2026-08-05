#!/usr/bin/env python3
"""Truth-referenced successor to the frozen DA V2 model-variant gate R0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_model_variant_gate_r0 import evaluate as evaluate_r0
from evaluate_dav2_model_variant_gate_r0 import sha256_file
from evaluate_metric3d_clearance_field_a0 import (
    BANDS,
    HORIZONS_M,
    clearance_field,
    intrinsics_matrix,
    tum_depth_metres,
)
from prepare_bonn_rgbd_metric_depth_manifest import normalize_depth_image

SCHEMA = "blindassist_dav2_model_variant_gate_r1_result"


def field_signature(field: dict[str, Any]) -> tuple[Any, ...]:
    status = str(field.get("status", "UNKNOWN_MISSING_STATUS"))
    if status != "VALID":
        return (status,)
    signature: list[Any] = [status]
    for band in BANDS:
        for horizon in HORIZONS_M:
            signature.append(
                field["bands"][band]["occupied_by_horizon"][str(horizon)]
            )
    return tuple(signature)


def truth_geometry_summary(
    rows: list[dict[str, Any]], candidate_key: str
) -> dict[str, Any]:
    status_exact = 0
    state_exact = 0
    state_pairs = 0
    transition_exact = 0
    transition_pairs = 0
    false_clear = 0
    false_block = 0
    occupied_truth = 0
    clear_truth = 0
    known_decisions = 0
    previous: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    for row in rows:
        truth = row["sensor"]
        candidate = row[candidate_key]
        status_exact += int(truth.get("status") == candidate.get("status"))
        if truth.get("status") != "VALID" or candidate.get("status") != "VALID":
            continue
        truth_signature = field_signature(truth)
        candidate_signature = field_signature(candidate)
        state_pairs += 1
        state_exact += int(truth_signature == candidate_signature)
        sequence_id = str(row["sequence_id"])
        if sequence_id in previous:
            previous_truth, previous_candidate = previous[sequence_id]
            transition_exact += int(
                (truth_signature != previous_truth)
                == (candidate_signature != previous_candidate)
            )
            transition_pairs += 1
        previous[sequence_id] = (truth_signature, candidate_signature)
        for band in BANDS:
            for horizon in HORIZONS_M:
                truth_value = truth["bands"][band]["occupied_by_horizon"][
                    str(horizon)
                ]
                candidate_value = candidate["bands"][band][
                    "occupied_by_horizon"
                ][str(horizon)]
                if truth_value is None or candidate_value is None:
                    continue
                known_decisions += 1
                if bool(truth_value):
                    occupied_truth += 1
                    false_clear += int(not bool(candidate_value))
                else:
                    clear_truth += 1
                    false_block += int(bool(candidate_value))
    count = len(rows)
    return {
        "truth_status_exact_agreement": status_exact / count if count else 0.0,
        "truth_geometry_state_exact_agreement": (
            state_exact / state_pairs if state_pairs else None
        ),
        "truth_geometry_state_pairs": state_pairs,
        "truth_transition_change_agreement": (
            transition_exact / transition_pairs if transition_pairs else None
        ),
        "truth_transition_pairs": transition_pairs,
        "known_decisions": known_decisions,
        "false_clear_count": false_clear,
        "false_clear_rate_all_known": (
            false_clear / known_decisions if known_decisions else None
        ),
        "false_clear_rate_given_occupied": (
            false_clear / occupied_truth if occupied_truth else None
        ),
        "false_block_count": false_block,
        "false_block_rate_all_known": (
            false_block / known_decisions if known_decisions else None
        ),
        "false_block_rate_given_clear": (
            false_block / clear_truth if clear_truth else None
        ),
    }


def truth_change_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    harmful = 0
    beneficial = 0
    unchanged_correct = 0
    unchanged_wrong = 0
    known = 0
    for row in rows:
        truth = row["sensor"]
        baseline = row["baseline"]
        candidate = row["candidate"]
        if any(
            field.get("status") != "VALID"
            for field in (truth, baseline, candidate)
        ):
            continue
        for band in BANDS:
            for horizon in HORIZONS_M:
                key = str(horizon)
                truth_value = truth["bands"][band]["occupied_by_horizon"][key]
                baseline_value = baseline["bands"][band]["occupied_by_horizon"][key]
                candidate_value = candidate["bands"][band]["occupied_by_horizon"][key]
                if any(
                    value is None
                    for value in (truth_value, baseline_value, candidate_value)
                ):
                    continue
                known += 1
                baseline_correct = bool(baseline_value) == bool(truth_value)
                candidate_correct = bool(candidate_value) == bool(truth_value)
                if baseline_correct and not candidate_correct:
                    harmful += 1
                elif not baseline_correct and candidate_correct:
                    beneficial += 1
                elif candidate_correct:
                    unchanged_correct += 1
                else:
                    unchanged_wrong += 1
    return {
        "known_decisions": known,
        "harmful_changes": harmful,
        "beneficial_changes": beneficial,
        "net_beneficial_changes": beneficial - harmful,
        "harmful_change_rate": harmful / known if known else None,
        "beneficial_change_rate": beneficial / known if known else None,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def load_geometry_rows(
    roster: dict[str, Any],
    source_root: Path,
    baseline_depth_path: Path,
    candidate_depth_path: Path,
) -> list[dict[str, Any]]:
    baseline_depth = np.load(baseline_depth_path, mmap_mode="r")
    candidate_depth = np.load(candidate_depth_path, mmap_mode="r")
    rows = []
    for index, roster_row in enumerate(roster["rows"]):
        depth_path = (
            source_root
            / str(roster_row["sequence_root"])
            / str(roster_row["depth_path"])
        )
        sensor_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        sensor = tum_depth_metres(normalize_depth_image(sensor_raw, depth_path))
        intrinsics = intrinsics_matrix(roster_row)
        rows.append(
            {
                "sequence_id": roster_row["sequence_id"],
                "sensor": clearance_field(sensor, intrinsics),
                "baseline": clearance_field(
                    np.asarray(baseline_depth[index], dtype=np.float32), intrinsics
                ),
                "candidate": clearance_field(
                    np.asarray(candidate_depth[index], dtype=np.float32), intrinsics
                ),
            }
        )
    return rows


def evaluate(
    r1_protocol_path: Path,
    r0_protocol_path: Path,
    roster_path: Path,
    source_root: Path,
    baseline_depth_path: Path,
    candidate_depth_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    protocol = json.loads(r1_protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "blindassist_dav2_model_variant_gate_r1_protocol":
        raise ValueError("R1 protocol schema mismatch")
    if sha256_file(r0_protocol_path) != protocol["parent_r0_protocol_sha256"]:
        raise ValueError("R1 parent R0 protocol hash mismatch")
    if sha256_file(roster_path) != protocol["roster_sha256"]:
        raise ValueError("R1 roster hash mismatch")
    if sha256_file(baseline_depth_path) != protocol["baseline_depth_sha256"]:
        raise ValueError("R1 baseline depth hash mismatch")
    evaluator_path = Path(__file__).resolve()
    if sha256_file(evaluator_path) != protocol["evaluator_sha256"]:
        raise ValueError("R1 evaluator source hash mismatch")
    r0_result = evaluate_r0(
        r0_protocol_path,
        roster_path,
        source_root,
        baseline_depth_path,
        candidate_depth_path,
        candidate_id,
    )
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    rows = load_geometry_rows(
        roster,
        source_root,
        baseline_depth_path,
        candidate_depth_path,
    )
    baseline_truth = truth_geometry_summary(rows, "baseline")
    candidate_truth = truth_geometry_summary(rows, "candidate")
    changes = truth_change_summary(rows)
    excluded = set(protocol["r0_gates_replaced"])
    inherited_gates = {
        key: value
        for key, value in r0_result["engineering_noninferiority_gates"].items()
        if key not in excluded
    }
    tolerances = protocol["truth_referenced_tolerances"]
    truth_gates = {
        "truth_status_exact_agreement": candidate_truth[
            "truth_status_exact_agreement"
        ]
        >= baseline_truth["truth_status_exact_agreement"]
        - float(tolerances["truth_status_exact_agreement_drop"]),
        "truth_geometry_state_exact_agreement": candidate_truth[
            "truth_geometry_state_exact_agreement"
        ]
        >= baseline_truth["truth_geometry_state_exact_agreement"]
        - float(tolerances["truth_geometry_state_exact_agreement_drop"]),
        "truth_transition_change_agreement": candidate_truth[
            "truth_transition_change_agreement"
        ]
        >= baseline_truth["truth_transition_change_agreement"]
        - float(tolerances["truth_transition_change_agreement_drop"]),
        "false_block_rate": candidate_truth["false_block_rate_all_known"]
        <= baseline_truth["false_block_rate_all_known"]
        + float(tolerances["false_block_rate_all_known_increase"]),
        "harmful_change_rate": changes["harmful_change_rate"]
        <= float(tolerances["maximum_harmful_change_rate"]),
        "net_truth_decision_change": changes["net_beneficial_changes"] >= 0,
    }
    all_gates = {**inherited_gates, **truth_gates}
    passed = all(all_gates.values())
    terminal = (
        "MODEL_VARIANT_R1_ENGINEERING_NONINFERIORITY_PASS"
        if passed
        else "MODEL_VARIANT_R1_ENGINEERING_NONINFERIORITY_FAIL"
    )
    return {
        "schema": SCHEMA,
        "protocol_sha256": sha256_file(r1_protocol_path),
        "parent_r0_result": r0_result,
        "baseline_truth_geometry": baseline_truth,
        "candidate_truth_geometry": candidate_truth,
        "candidate_truth_changes_vs_baseline": changes,
        "inherited_r0_gates": inherited_gates,
        "truth_referenced_gates": truth_gates,
        "engineering_noninferiority_gates": all_gates,
        "engineering_noninferiority_passed": passed,
        "historical_r0_terminal_relabel_authorized": False,
        "terminal": terminal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
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
        args.r1_protocol.resolve(),
        args.r0_protocol.resolve(),
        args.roster.resolve(),
        args.source_root.resolve(),
        args.baseline_depth.resolve(),
        args.candidate_depth.resolve(),
        args.candidate_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
