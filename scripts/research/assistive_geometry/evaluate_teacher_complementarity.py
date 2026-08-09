#!/usr/bin/env python3
"""Evaluate truth-bound complementarity of metric and temporal/geometry teachers."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TEACHERS = ("metric_teacher", "temporal_geometry_teacher")
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
STATES = {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_rows(rows: list[dict[str, Any]]) -> None:
    require(bool(rows), "teacher complementarity observations are empty")
    identities: set[tuple[str, str, int]] = set()
    for row in rows:
        require(row.get("schema") == "blindassist_assistive_geometry_teacher_complementarity_frame_v1", "observation schema drift")
        require(row.get("data_role") == "TEACHER_EVALUATION", "teacher evaluation role drift")
        identity = (row.get("parent_id"), row.get("session_id"), row.get("sequence_index"))
        require(isinstance(identity[0], str) and isinstance(identity[1], str) and isinstance(identity[2], int), "frame identity invalid")
        require(identity not in identities, "duplicate frame identity")
        identities.add(identity)
        bands = row.get("bands")
        require(isinstance(bands, list) and tuple(band.get("band") for band in bands) == BANDS, "band order drift")
        for band in bands:
            truth_valid = band.get("truth_clearance_valid")
            require(isinstance(truth_valid, bool), "truth clearance validity invalid")
            require(_finite(band.get("truth_clearance_m")) if truth_valid else band.get("truth_clearance_m") is None, "truth clearance/value drift")
            predictions = band.get("teachers")
            require(isinstance(predictions, dict) and tuple(predictions) == TEACHERS, "teacher order/set drift")
            for teacher in TEACHERS:
                prediction = predictions[teacher]
                valid = prediction.get("clearance_valid")
                require(isinstance(valid, bool), "teacher clearance validity invalid")
                require(_finite(prediction.get("clearance_m")) if valid else prediction.get("clearance_m") is None, "teacher clearance/value drift")
                states = prediction.get("states")
                require(isinstance(states, list) and len(states) == 3 and all(state in STATES for state in states), "teacher state drift")
            truth_states = band.get("truth_states")
            require(isinstance(truth_states, list) and len(truth_states) == 3 and all(state in STATES for state in truth_states), "truth state drift")


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    truth_known = [pair for pair in pairs if pair[0] != "UNKNOWN"]
    paired = [pair for pair in truth_known if pair[1] != "UNKNOWN"]
    occupied = [pair for pair in paired if pair[0] == "OCCUPIED_OBSERVED"]
    clear = [pair for pair in paired if pair[0] == "CLEAR_OBSERVED"]
    require(bool(truth_known) and bool(paired) and bool(occupied) and bool(clear), "teacher confusion denominator is zero")
    return {
        "known_coverage": len(paired) / len(truth_known),
        "false_clear_all_known": sum(pair == ("OCCUPIED_OBSERVED", "CLEAR_OBSERVED") for pair in paired) / len(paired),
        "false_clear_given_occupied": sum(pair == ("OCCUPIED_OBSERVED", "CLEAR_OBSERVED") for pair in occupied) / len(occupied),
        "false_block_given_clear": sum(pair == ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED") for pair in clear) / len(clear),
        "truth_known_support": len(truth_known),
        "paired_known_support": len(paired),
    }


def compute_metrics(rows: list[dict[str, Any]], gates: dict[str, float]) -> dict[str, Any]:
    validate_rows(rows)
    teacher_pairs: dict[str, list[tuple[str, str]]] = {teacher: [] for teacher in TEACHERS}
    teacher_clearance_errors: dict[str, list[float]] = {teacher: [] for teacher in TEACHERS}
    oracle_pairs: list[tuple[str, str]] = []
    oracle_clearance_errors: list[float] = []
    exclusive_correct_parents: dict[str, set[str]] = {teacher: set() for teacher in TEACHERS}
    agreement_any_error: list[bool] = []
    disagreement_any_error: list[bool] = []
    clearance_series: dict[tuple[str, str, str], list[tuple[int, float, dict[str, float]]]] = defaultdict(list)

    for row in rows:
        parent = row["parent_id"]
        for band in row["bands"]:
            truth_clearance_valid = band["truth_clearance_valid"]
            available_clearance_errors: list[float] = []
            teacher_clearances: dict[str, float] = {}
            for teacher in TEACHERS:
                prediction = band["teachers"][teacher]
                if truth_clearance_valid and prediction["clearance_valid"]:
                    error = abs(float(prediction["clearance_m"]) - float(band["truth_clearance_m"]))
                    teacher_clearance_errors[teacher].append(error)
                    available_clearance_errors.append(error)
                    teacher_clearances[teacher] = float(prediction["clearance_m"])
            if truth_clearance_valid:
                require(bool(available_clearance_errors), "oracle clearance has zero paired support")
                oracle_clearance_errors.append(min(available_clearance_errors))
                if len(teacher_clearances) == len(TEACHERS):
                    clearance_series[(parent, row["session_id"], band["band"])].append(
                        (int(row["sequence_index"]), float(band["truth_clearance_m"]), teacher_clearances)
                    )
            for horizon_index, _ in enumerate(HORIZONS):
                truth = band["truth_states"][horizon_index]
                predictions = {teacher: band["teachers"][teacher]["states"][horizon_index] for teacher in TEACHERS}
                for teacher in TEACHERS:
                    teacher_pairs[teacher].append((truth, predictions[teacher]))
                if truth == "UNKNOWN":
                    oracle_pairs.append((truth, "UNKNOWN"))
                    continue
                correct = {teacher: predictions[teacher] == truth for teacher in TEACHERS}
                for teacher in TEACHERS:
                    other = TEACHERS[1] if teacher == TEACHERS[0] else TEACHERS[0]
                    if correct[teacher] and not correct[other]:
                        exclusive_correct_parents[teacher].add(parent)
                known = [value for value in predictions.values() if value != "UNKNOWN"]
                if any(correct.values()):
                    oracle_pairs.append((truth, truth))
                elif known:
                    oracle_pairs.append((truth, known[0]))
                else:
                    oracle_pairs.append((truth, "UNKNOWN"))
                if all(predictions[teacher] != "UNKNOWN" for teacher in TEACHERS):
                    any_error = any(not correct[teacher] for teacher in TEACHERS)
                    target = disagreement_any_error if predictions[TEACHERS[0]] != predictions[TEACHERS[1]] else agreement_any_error
                    target.append(any_error)

    teacher_results = {}
    for teacher in TEACHERS:
        require(bool(teacher_clearance_errors[teacher]), f"{teacher} clearance support is zero")
        teacher_results[teacher] = {
            "occupancy": _confusion(teacher_pairs[teacher]),
            "clearance_mae_m": float(statistics.fmean(teacher_clearance_errors[teacher])),
            "clearance_support": len(teacher_clearance_errors[teacher]),
            "exclusive_correct_parent_count": len(exclusive_correct_parents[teacher]),
            "exclusive_correct_parents": sorted(exclusive_correct_parents[teacher]),
        }
    oracle = {
        "occupancy": _confusion(oracle_pairs),
        "clearance_mae_m": float(statistics.fmean(oracle_clearance_errors)),
        "clearance_support": len(oracle_clearance_errors),
    }
    temporal_errors: dict[str, list[float]] = {teacher: [] for teacher in TEACHERS}
    for samples in clearance_series.values():
        samples.sort()
        for previous, current in zip(samples, samples[1:]):
            if current[0] != previous[0] + 1:
                continue
            truth_delta = current[1] - previous[1]
            for teacher in TEACHERS:
                temporal_errors[teacher].append(abs((current[2][teacher] - previous[2][teacher]) - truth_delta))
    for teacher in TEACHERS:
        require(bool(temporal_errors[teacher]), f"{teacher} temporal support is zero")
        teacher_results[teacher]["temporal_clearance_delta_mae_m"] = float(statistics.fmean(temporal_errors[teacher]))
        teacher_results[teacher]["temporal_support"] = len(temporal_errors[teacher])
    require(bool(agreement_any_error) and bool(disagreement_any_error), "agreement/disagreement denominator is zero")
    agreement_error_rate = sum(agreement_any_error) / len(agreement_any_error)
    disagreement_error_rate = sum(disagreement_any_error) / len(disagreement_any_error)
    best_clearance = min(teacher_results[teacher]["clearance_mae_m"] for teacher in TEACHERS)
    best_false_clear = min(teacher_results[teacher]["occupancy"]["false_clear_all_known"] for teacher in TEACHERS)
    oracle_clearance_relative_gain = (best_clearance - oracle["clearance_mae_m"]) / best_clearance if best_clearance > 0 else 0.0
    oracle_false_clear_gain = best_false_clear - oracle["occupancy"]["false_clear_all_known"]
    temporal_advantage = (
        teacher_results["metric_teacher"]["temporal_clearance_delta_mae_m"]
        - teacher_results["temporal_geometry_teacher"]["temporal_clearance_delta_mae_m"]
    )
    checks = {
        "oracle_gain": oracle_clearance_relative_gain >= gates["oracle_clearance_relative_gain_min"] or oracle_false_clear_gain >= gates["oracle_false_clear_absolute_gain_min"],
        "bidirectional_parent_support": all(teacher_results[teacher]["exclusive_correct_parent_count"] >= int(gates["exclusive_correct_parent_count_min"]) for teacher in TEACHERS),
        "disagreement_error_concentration": disagreement_error_rate - agreement_error_rate >= gates["disagreement_error_rate_excess_min"],
        "temporal_teacher_advantage": temporal_advantage >= gates["temporal_delta_mae_advantage_min_m"],
    }
    passed = all(checks.values())
    return {
        "teachers": teacher_results,
        "oracle": oracle,
        "complementarity": {
            "oracle_clearance_relative_gain": oracle_clearance_relative_gain,
            "oracle_false_clear_absolute_gain": oracle_false_clear_gain,
            "agreement_any_error_rate": agreement_error_rate,
            "agreement_support": len(agreement_any_error),
            "disagreement_any_error_rate": disagreement_error_rate,
            "disagreement_support": len(disagreement_any_error),
            "disagreement_error_rate_excess": disagreement_error_rate - agreement_error_rate,
            "temporal_teacher_delta_mae_advantage_m": temporal_advantage,
        },
        "gates": checks,
        "status": "PASS" if passed else "FAIL",
        "terminal": "ASSISTIVE_GEOMETRY_C0_TEACHER_COMPLEMENTARITY_PASS" if passed else "ASSISTIVE_GEOMETRY_C0_TEACHER_COMPLEMENTARITY_NOT_SUPPORTED_STOP",
        "c1_training_authorized": passed,
        "claim_ceiling": "Teacher complementarity on a truth-bound evaluation role only; no student improvement, Confirmation, deployment, product or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "teacher complementarity output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "blindassist_assistive_geometry_c0_teacher_complementarity_protocol_v1", "C0 protocol schema drift")
    require(protocol.get("authority", {}).get("teacher_output_evaluation") is True, "C0 teacher evaluation is not activated")
    rows = [json.loads(line) for line in args.observations.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = compute_metrics(rows, protocol["gates"])
    result.update(protocol_id=protocol["protocol_id"], data_role="TEACHER_EVALUATION", confirmation_content_opened=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
