#!/usr/bin/env python3
"""Truth-joining evaluator for a sealed production temporal A/B trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
BRANCH_A = "PRODUCTION_SEMANTIC_WITH_OBJECT_DETECTOR_TEMPORAL_GEOMETRY_NEUTRALIZED"
BRANCH_B = "CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY"
SESSION_MEDIAN_DELTA_NS = {
    "defaced_2021-03-27-11-51-18_filtered_lidar_odom": Fraction(190814671, 2),
    "defaced_2021-03-27-11-55-00_filtered_lidar_odom": Fraction(181057555, 2),
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


def median_fraction(values: list[Fraction]) -> Fraction:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def fraction_json(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def evaluate(
    seal_path: Path,
    truth_membership_path: Path,
) -> dict[str, Any]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("schema_version") != "blindassist.production_temporal_ab_seal.v1"
        or seal.get("protocol_id") != PROTOCOL_ID
        or seal.get("implementation_id")
        != "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0"
        or seal.get("status") != "SEALED"
        or seal.get("truth_opened") is not False
        or seal.get("truth_join_authorized") is not True
    ):
        raise ValueError("producer seal is invalid")
    bound_paths: dict[str, Path] = {}
    for name in (
        "trace",
        "producer_receipt",
        "implementation_lock",
        "activation",
        "formal_start_marker",
        "validation",
    ):
        binding = seal.get(name, {})
        path = Path(binding["path"])
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"sealed {name} identity drift")
        bound_paths[name] = path
    trace_path = bound_paths["trace"]
    validation_path = bound_paths["validation"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("schema_version")
        != "blindassist.production_temporal_ab_validation.v1"
        or validation.get("implementation_id")
        != "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0"
        or validation.get("status") != "VALID"
        or validation.get("truth_opened") is not False
    ):
        raise ValueError("producer validation is not VALID truth-blind")
    if validation.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("producer validation protocol mismatch")
    if validation.get("trace_sha256") != sha256_file(trace_path):
        raise ValueError("sealed trace hash mismatch")
    if (
        validation.get("implementation_lock_sha256")
        != seal["implementation_lock"]["sha256"]
        or validation.get("activation_sha256") != seal["activation"]["sha256"]
        or validation.get("producer_receipt_sha256")
        != seal["producer_receipt"]["sha256"]
        or validation.get("formal_start_marker_sha256")
        != seal["formal_start_marker"]["sha256"]
    ):
        raise ValueError("sealed validation chain mismatch")
    implementation_lock = json.loads(
        bound_paths["implementation_lock"].read_text(encoding="utf-8")
    )
    truth_binding = implementation_lock.get("truth_membership_receipt", {})
    if sha256_file(truth_membership_path) != truth_binding.get("sha256"):
        raise ValueError("truth-membership receipt differs from implementation lock")
    membership = json.loads(truth_membership_path.read_text(encoding="utf-8"))
    if membership.get("status") != "VALID" or membership.get("candidate_output_opened") is not False:
        raise ValueError("truth-membership receipt is invalid")
    if membership.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("truth-membership protocol mismatch")
    if (
        membership.get("schema_version")
        != "blindassist.dual_loop_truth_membership_preflight.v1"
    ):
        raise ValueError("truth-membership schema mismatch")
    if membership.get("raw_truth_item_count") != 17:
        raise ValueError("truth-membership raw denominator mismatch")
    if membership.get("fixed_scored_item_denominator") != 15:
        raise ValueError("truth-membership denominator mismatch")
    if membership.get("scoreable_positive_count") != 8:
        raise ValueError("truth-membership positive denominator mismatch")
    if membership.get("scoreable_negative_count") != 7:
        raise ValueError("truth-membership negative denominator mismatch")
    if membership.get("cross_item_or_class_frame_overlap_count") != 0:
        raise ValueError("truth-membership overlap is not zero")
    expected_session_denominators = {
        session_id: {
            "positive": 5 if index == 0 else 3,
            "negative": 3 if index == 0 else 4,
            "total": 8 if index == 0 else 7,
        }
        for index, session_id in enumerate(SESSION_MEDIAN_DELTA_NS)
    }
    if membership.get("session_scored_denominators") != expected_session_denominators:
        raise ValueError("truth-membership session denominators drift")

    rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    session_origin: dict[str, int] = {}
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            session_id = str(row["session_id"])
            timestamp_ns = int(row["source_capture_timestamp_ns"])
            session_origin[session_id] = min(session_origin.get(session_id, timestamp_ns), timestamp_ns)
            rows[str(row["branch_id"])][session_id].append(row)
    for branch in (BRANCH_A, BRANCH_B):
        for session_id in rows[branch]:
            rows[branch][session_id].sort(key=lambda row: int(row["source_capture_timestamp_ns"]))

    scoreable_ids = set(membership["scoreable_positive_ids"])
    items = membership["item_membership"]
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != 17 or len(set(item_ids)) != 17:
        raise ValueError("truth-membership raw item identity drift")
    positive_items = [
        item for item in items
        if item["item_kind"] == "positive_event" and item["item_id"] in scoreable_ids
    ]
    negative_items = [item for item in items if item["item_kind"] == "negative_window"]
    if len(positive_items) != 8 or len(negative_items) != 7:
        raise ValueError("scored item denominator drift")
    if {item["item_id"] for item in positive_items} != scoreable_ids:
        raise ValueError("scoreable positive identity drift")
    truth_item_table = []
    excluded_ids = set(membership["temporal_scoring_not_evaluable_positive_ids"])
    for item in items:
        truth_item_table.append(
            {
                "item_id": item["item_id"],
                "session_id": item["session_id"],
                "item_kind": item["item_kind"],
                "scoring_status": (
                    "TEMPORAL_SCORING_NOT_EVALUABLE"
                    if item["item_id"] in excluded_ids
                    else "SCORED"
                ),
                "valid_interval_ns": item.get("valid_interval_ns"),
                "premature_interval_ns": item.get("premature_interval_ns"),
                "interval_ns": item.get("interval_ns"),
            }
        )

    branch_metrics: dict[str, Any] = {}
    positive_pairs: list[dict[str, Any]] = []
    negative_pairs: list[dict[str, Any]] = []
    scored_membership_by_session: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    for item in positive_items:
        valid_start, valid_end = item["valid_interval_ns"]
        premature_start, premature_end = item["premature_interval_ns"]
        scored_membership_by_session[item["session_id"]].append((valid_start, valid_end, True))
        scored_membership_by_session[item["session_id"]].append((premature_start, premature_end, False))
    for item in negative_items:
        start, end = item["interval_ns"]
        scored_membership_by_session[item["session_id"]].append((start, end, True))

    for branch in (BRANCH_A, BRANCH_B):
        positive: dict[str, Any] = {}
        negative: dict[str, Any] = {}
        for item in positive_items:
            session_id = item["session_id"]
            origin = session_origin[session_id]
            valid_start, valid_end = item["valid_interval_ns"]
            premature_start, premature_end = item["premature_interval_ns"]
            triggers = [
                row for row in rows[branch][session_id] if row["feedback_triggered"] is True
            ]
            valid = [
                int(row["source_capture_timestamp_ns"])
                for row in triggers
                if valid_start <= int(row["source_capture_timestamp_ns"]) - origin <= valid_end
            ]
            premature = [
                int(row["source_capture_timestamp_ns"])
                for row in triggers
                if premature_start <= int(row["source_capture_timestamp_ns"]) - origin < premature_end
            ]
            positive[item["item_id"]] = {
                "session_id": session_id,
                "valid_alert": bool(valid),
                "first_valid_alert_timestamp_ns": min(valid) if valid else None,
                "valid_trigger_row_count": len(valid),
                "premature_alert": bool(premature),
                "premature_trigger_row_count": len(premature),
            }
        for item in negative_items:
            session_id = item["session_id"]
            origin = session_origin[session_id]
            start, end = item["interval_ns"]
            false_triggers = [
                row for row in rows[branch][session_id]
                if row["feedback_triggered"] is True
                and start <= int(row["source_capture_timestamp_ns"]) - origin <= end
            ]
            negative[item["item_id"]] = {
                "session_id": session_id,
                "false_alert": bool(false_triggers),
                "false_trigger_row_count": len(false_triggers),
            }
        unscored_trigger_rows = 0
        session_metrics: dict[str, Any] = {}
        for session_id, branch_rows in rows[branch].items():
            origin = session_origin[session_id]
            intervals = scored_membership_by_session[session_id]
            session_unscored = sum(
                1
                for row in branch_rows
                if row["feedback_triggered"] is True
                and not any(
                    start <= int(row["source_capture_timestamp_ns"]) - origin
                    and (
                        int(row["source_capture_timestamp_ns"]) - origin <= end
                        if end_inclusive
                        else int(row["source_capture_timestamp_ns"]) - origin < end
                    )
                    for start, end, end_inclusive in intervals
                )
            )
            unscored_trigger_rows += session_unscored
            session_positive = [
                value for value in positive.values() if value["session_id"] == session_id
            ]
            session_negative = [
                value for value in negative.values() if value["session_id"] == session_id
            ]
            session_metrics[session_id] = {
                "positive_denominator": len(session_positive),
                "negative_denominator": len(session_negative),
                "positive_recall": sum(value["valid_alert"] for value in session_positive),
                "missed_positive_count": sum(
                    not value["valid_alert"] for value in session_positive
                ),
                "false_alert_window_count": sum(
                    value["false_alert"] for value in session_negative
                ),
                "premature_alert_event_count": sum(
                    value["premature_alert"] for value in session_positive
                ),
                "unscored_trigger_row_count": session_unscored,
            }
        branch_metrics[branch] = {
            "positive": positive,
            "negative": negative,
            "positive_recall": sum(item["valid_alert"] for item in positive.values()),
            "missed_positive_count": sum(not item["valid_alert"] for item in positive.values()),
            "false_alert_window_count": sum(item["false_alert"] for item in negative.values()),
            "premature_alert_event_count": sum(item["premature_alert"] for item in positive.values()),
            "unscored_trigger_row_count": unscored_trigger_rows,
            "sessions": session_metrics,
        }

    for item in positive_items:
        item_id = item["item_id"]
        a = branch_metrics[BRANCH_A]["positive"][item_id]
        b = branch_metrics[BRANCH_B]["positive"][item_id]
        positive_pairs.append(
            {
                "item_id": item_id,
                "session_id": item["session_id"],
                "a_valid": a["valid_alert"],
                "b_valid": b["valid_alert"],
                "a_first_timestamp_ns": a["first_valid_alert_timestamp_ns"],
                "b_first_timestamp_ns": b["first_valid_alert_timestamp_ns"],
                "gain_ns": (
                    a["first_valid_alert_timestamp_ns"] - b["first_valid_alert_timestamp_ns"]
                    if a["valid_alert"] and b["valid_alert"] else None
                ),
                "a_premature": a["premature_alert"],
                "b_premature": b["premature_alert"],
            }
        )
    for item in negative_items:
        item_id = item["item_id"]
        a = branch_metrics[BRANCH_A]["negative"][item_id]
        b = branch_metrics[BRANCH_B]["negative"][item_id]
        negative_pairs.append(
            {
                "item_id": item_id,
                "session_id": item["session_id"],
                "a_false": a["false_alert"],
                "b_false": b["false_alert"],
            }
        )

    adverse_positive = [item for item in positive_pairs if item["a_valid"] and not item["b_valid"]]
    adverse_negative = [item for item in negative_pairs if not item["a_false"] and item["b_false"]]
    adverse_premature = [item for item in positive_pairs if not item["a_premature"] and item["b_premature"]]
    common_guardrails = {
        "b_recall_not_lower": branch_metrics[BRANCH_B]["positive_recall"] >= branch_metrics[BRANCH_A]["positive_recall"],
        "b_misses_not_higher": branch_metrics[BRANCH_B]["missed_positive_count"] <= branch_metrics[BRANCH_A]["missed_positive_count"],
        "b_false_windows_not_higher": branch_metrics[BRANCH_B]["false_alert_window_count"] <= branch_metrics[BRANCH_A]["false_alert_window_count"],
        "b_premature_events_not_higher": branch_metrics[BRANCH_B]["premature_alert_event_count"] <= branch_metrics[BRANCH_A]["premature_alert_event_count"],
        "b_unscored_trigger_rows_not_higher": branch_metrics[BRANCH_B]["unscored_trigger_row_count"] <= branch_metrics[BRANCH_A]["unscored_trigger_row_count"],
        "zero_a_valid_b_missed": not adverse_positive,
        "zero_a_clean_b_false": not adverse_negative,
        "zero_a_no_premature_b_premature": not adverse_premature,
        "complete_validated_execution": True,
    }
    all_guardrails = all(common_guardrails.values())
    session_guardrails = {}
    for session_id in SESSION_MEDIAN_DELTA_NS:
        a_session = branch_metrics[BRANCH_A]["sessions"][session_id]
        b_session = branch_metrics[BRANCH_B]["sessions"][session_id]
        session_guardrails[session_id] = {
            "b_recall_not_lower": (
                b_session["positive_recall"] >= a_session["positive_recall"]
            ),
            "b_misses_not_higher": (
                b_session["missed_positive_count"]
                <= a_session["missed_positive_count"]
            ),
            "b_false_windows_not_higher": (
                b_session["false_alert_window_count"]
                <= a_session["false_alert_window_count"]
            ),
            "b_premature_events_not_higher": (
                b_session["premature_alert_event_count"]
                <= a_session["premature_alert_event_count"]
            ),
            "b_unscored_trigger_rows_not_higher": (
                b_session["unscored_trigger_row_count"]
                <= a_session["unscored_trigger_row_count"]
            ),
            "zero_a_valid_b_missed": not any(
                item["session_id"] == session_id for item in adverse_positive
            ),
            "zero_a_clean_b_false": not any(
                item["session_id"] == session_id for item in adverse_negative
            ),
            "zero_a_no_premature_b_premature": not any(
                item["session_id"] == session_id for item in adverse_premature
            ),
            "terminal_gate_scope": (
                "DESCRIPTIVE_SESSION_RECOMPUTATION; pooled gates follow frozen protocol"
            ),
        }

    eligible = [item for item in positive_pairs if item["gain_ns"] is not None]
    eligible_per_session = {
        session_id: sum(item["session_id"] == session_id for item in eligible)
        for session_id in SESSION_MEDIAN_DELTA_NS
    }
    early_evaluable = len(eligible) >= 4 and all(count >= 2 for count in eligible_per_session.values())
    normalized = [
        Fraction(int(item["gain_ns"]), 1) / SESSION_MEDIAN_DELTA_NS[item["session_id"]]
        for item in eligible
    ]
    session_median_gain = {
        session_id: median_fraction(
            [
                Fraction(int(item["gain_ns"]), 1)
                for item in eligible
                if item["session_id"] == session_id
            ]
        )
        for session_id in SESSION_MEDIAN_DELTA_NS
        if eligible_per_session[session_id] > 0
    }
    early_predicates = {
        "evaluable": early_evaluable,
        "common_guardrails": all_guardrails,
        "strict_majority_positive_gain": (
            sum(int(item["gain_ns"]) > 0 for item in eligible) > len(eligible) / 2
            if early_evaluable else False
        ),
        "pooled_median_normalized_gain_at_least_one": (
            median_fraction(normalized) >= 1 if early_evaluable else False
        ),
        "positive_session_medians": (
            all(session_median_gain.get(session_id, 0) > 0 for session_id in SESSION_MEDIAN_DELTA_NS)
            if early_evaluable else False
        ),
        "at_least_two_positive_gain_events": (
            sum(int(item["gain_ns"]) > 0 for item in eligible) >= 2 if early_evaluable else False
        ),
    }
    early_success = early_evaluable and all(
        value for key, value in early_predicates.items() if key != "evaluable"
    )

    paired_items: list[dict[str, Any]] = []
    for item in positive_pairs:
        paired_items.append(
            {
                "item_id": item["item_id"],
                "session_id": item["session_id"],
                "kind": "positive_event",
                "a_correct": item["a_valid"],
                "b_correct": item["b_valid"],
            }
        )
    for item in negative_pairs:
        paired_items.append(
            {
                "item_id": item["item_id"],
                "session_id": item["session_id"],
                "kind": "negative_window",
                "a_correct": not item["a_false"],
                "b_correct": not item["b_false"],
            }
        )
    improvements = [item for item in paired_items if item["b_correct"] and not item["a_correct"]]
    harms = [item for item in paired_items if item["a_correct"] and not item["b_correct"]]
    paired_delta = len(improvements) - len(harms)
    improvements_per_session = {
        session_id: sum(item["session_id"] == session_id for item in improvements)
        for session_id in SESSION_MEDIAN_DELTA_NS
    }
    risk_predicates = {
        "common_guardrails": all_guardrails,
        "paired_correctness_delta_at_least_two": paired_delta >= 2,
        "at_least_one_improvement_each_session": all(
            count >= 1 for count in improvements_per_session.values()
        ),
        "zero_a_correct_b_incorrect": not harms,
    }
    risk_success = all(risk_predicates.values())

    if early_success and risk_success:
        terminal = "MULTIPLE_INCREMENT"
    elif early_success:
        terminal = "EARLY_RESPONSE"
    elif risk_success and early_evaluable:
        terminal = "RISK_DISCRIMINATION"
    elif risk_success:
        terminal = "RISK_DISCRIMINATION_WITH_EARLY_RESPONSE_NOT_EVALUABLE"
    elif early_evaluable:
        terminal = "NO_INCREMENT"
    else:
        terminal = "EARLY_RESPONSE_NOT_EVALUABLE_RISK_DISCRIMINATION_NO_INCREMENT"

    return {
        "schema_version": "blindassist.production_temporal_ab_evaluation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "scientific_terminal": terminal,
        "claim_ceiling": "TWO_SESSION_ONE_CONTEXT_DEVELOPMENT_DIRECTIONAL_SCREEN_ONLY",
        "seal_sha256": sha256_file(seal_path),
        "trace_sha256": sha256_file(trace_path),
        "validation_sha256": sha256_file(validation_path),
        "truth_membership_sha256": sha256_file(truth_membership_path),
        "branch_metrics": branch_metrics,
        "truth_item_table": truth_item_table,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "common_guardrails": common_guardrails,
        "session_guardrails": session_guardrails,
        "early_response": {
            "evaluable": early_evaluable,
            "success": early_success,
            "eligible_pair_count": len(eligible),
            "eligible_pairs_per_session": eligible_per_session,
            "predicates": early_predicates,
            "pooled_median_normalized_gain_frames": (
                fraction_json(median_fraction(normalized)) if normalized else None
            ),
            "session_median_gain_ns": {
                session_id: fraction_json(value)
                for session_id, value in session_median_gain.items()
            },
        },
        "risk_discrimination": {
            "evaluable": True,
            "success": risk_success,
            "paired_correctness_delta": paired_delta,
            "b_correct_a_incorrect_count": len(improvements),
            "a_correct_b_incorrect_count": len(harms),
            "improvements_per_session": improvements_per_session,
            "predicates": risk_predicates,
            "paired_items": paired_items,
        },
        "temporal_scoring_not_evaluable_positive_ids": membership[
            "temporal_scoring_not_evaluable_positive_ids"
        ],
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--truth-membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError("evaluation output already exists")
    result = evaluate(
        args.seal.resolve(),
        args.truth_membership.resolve(),
    )
    atomic_json(output_path, result)
    print(json.dumps({"status": "VALID", "terminal": result["scientific_terminal"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
