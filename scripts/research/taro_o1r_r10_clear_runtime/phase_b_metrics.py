#!/usr/bin/env python3
"""Frozen R10 dual-class evaluability and confirmation metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


PASS_TERMINAL = "TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_PASS"
FAIL_TERMINAL = "TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_FAIL"
NOT_EVALUABLE_TERMINAL = "TARO_O1R_R10_FRESH_CLEAR_ENRICHED_NOT_EVALUABLE_DUAL_CLASS_COVERAGE"
EXPECTED_GATES = {
    "minimum_evaluable_parents": 8,
    "minimum_parents_with_definite_occupied": 6,
    "minimum_definite_occupied_queries": 100,
    "minimum_parents_with_definite_clear": 4,
    "minimum_definite_clear_queries": 12,
    "minimum_occupied_precision": 0.9,
    "minimum_one_sided_95_wilson_occupied_precision_lower_bound": 0.8,
    "minimum_occupied_recall": 0.9,
    "minimum_parent_macro_occupancy_coverage_increase_absolute": 0.05,
    "minimum_clear_specificity": 0.9,
    "minimum_one_sided_95_wilson_clear_specificity_lower_bound": 0.8,
    "maximum_clear_outputs": 0,
    "unknown_is_negative": False,
}


class PhaseBMetricsError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PhaseBMetricsError(code, message)


def positive_state(feature: Mapping[str, Any]) -> str:
    if feature.get("r6_state") == "OCCUPIED_OBSERVED":
        return "OCCUPIED_OBSERVED"
    if feature.get("query_receipt") is None:
        return "UNKNOWN"
    hits = feature.get("occupied_hits")
    require(
        isinstance(hits, list)
        and len(hits) > 0
        and isinstance(hits[0], list)
        and len(hits[0]) > 0
        and isinstance(hits[0][0], list)
        and len(hits[0][0]) > 2,
        "R10_PHASE_B_POSITIVE_STATE_INPUT",
        "frozen positive-cell tensor is unavailable",
    )
    return "OCCUPIED_OBSERVED" if bool(hits[0][0][2]) else "UNKNOWN"


def summarize(
    selected_identities: Sequence[tuple[str, str]],
    sources: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(selected_identities) == 8 and len(set(selected_identities)) == 8, "R10_PHASE_B_SELECTED_IDENTITIES", "exactly eight unique selected identities are required")
    require(len(sources) == len(labels) and len(sources) > 0, "R10_PHASE_B_RECORD_COUNT", "source/label record count drift")
    expected_parents = {parent for parent, _video in selected_identities}
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    parent_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parent_occ: dict[str, list[tuple[str, str]]] = defaultdict(list)
    occupied_tp = occupied_fp = occupied_fn = predicted_on_unknown = clear_outputs = 0
    for source, label_record in zip(sources, labels, strict=True):
        require(
            source.get("physical_frame_id") == label_record.get("physical_frame_id")
            and str(source.get("parent_id")) in expected_parents,
            "R10_PHASE_B_FRAME_ALIGNMENT",
            "source/label frame alignment drift",
        )
        features = source.get("query_features")
        query_labels = label_record.get("query_labels")
        require(
            isinstance(features, list)
            and isinstance(query_labels, list)
            and len(features) == len(query_labels) == 9,
            "R10_PHASE_B_QUERY_COUNT",
            "each frame must contain exactly nine aligned queries",
        )
        parent = str(source["parent_id"])
        for feature, label in zip(features, query_labels, strict=True):
            require(
                feature.get("query_id") == label.get("query_id")
                and label.get("state") in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"},
                "R10_PHASE_B_QUERY_ALIGNMENT",
                "source/label query alignment drift",
            )
            prediction = positive_state(feature)
            truth = str(label["state"])
            label_counts[truth] += 1
            prediction_counts[prediction] += 1
            parent_label_counts[parent][truth] += 1
            clear_outputs += prediction == "CLEAR_OBSERVED"
            if truth == "OCCUPIED_OBSERVED":
                parent_occ[parent].append((str(feature.get("r6_state")), prediction))
                occupied_tp += prediction == "OCCUPIED_OBSERVED"
                occupied_fn += prediction != "OCCUPIED_OBSERVED"
            elif truth == "CLEAR_OBSERVED":
                occupied_fp += prediction == "OCCUPIED_OBSERVED"
            else:
                predicted_on_unknown += prediction == "OCCUPIED_OBSERVED"

    definite_occupied = int(label_counts["OCCUPIED_OBSERVED"])
    definite_clear = int(label_counts["CLEAR_OBSERVED"])
    precision_denominator = occupied_tp + occupied_fp
    occupied_precision = occupied_tp / precision_denominator if precision_denominator else 0.0
    occupied_recall = occupied_tp / definite_occupied if definite_occupied else 0.0
    occupied_wilson = r7_canary._wilson_lower(occupied_tp, precision_denominator)
    clear_specific_success = definite_clear - occupied_fp
    clear_specificity = clear_specific_success / definite_clear if definite_clear else 0.0
    clear_wilson = r7_canary._wilson_lower(clear_specific_success, definite_clear)

    parent_improvements = []
    per_parent = {}
    for parent, _video in selected_identities:
        occupied_rows = parent_occ[parent]
        denominator = len(occupied_rows)
        baseline = sum(base == "OCCUPIED_OBSERVED" for base, _ in occupied_rows) / denominator if denominator else None
        candidate = sum(pred == "OCCUPIED_OBSERVED" for _, pred in occupied_rows) / denominator if denominator else None
        improvement = None if denominator == 0 else float(candidate - baseline)
        if improvement is not None:
            parent_improvements.append(improvement)
        per_parent[parent] = {
            "label_state_counts": {
                state: int(parent_label_counts[parent][state])
                for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
            },
            "definite_occupied_query_count": denominator,
            "baseline_definite_occupancy_coverage": baseline,
            "candidate_definite_occupancy_coverage": candidate,
            "coverage_increase_absolute": improvement,
        }
    macro_increase = sum(parent_improvements) / len(parent_improvements) if parent_improvements else 0.0
    evaluability = {
        "evaluable_parent_count": sum(
            parent_label_counts[parent]["CLEAR_OBSERVED"] + parent_label_counts[parent]["OCCUPIED_OBSERVED"] > 0
            for parent, _video in selected_identities
        ),
        "parents_with_definite_occupied_label": sum(
            parent_label_counts[parent]["OCCUPIED_OBSERVED"] > 0 for parent, _video in selected_identities
        ),
        "parents_with_definite_clear_label": sum(
            parent_label_counts[parent]["CLEAR_OBSERVED"] > 0 for parent, _video in selected_identities
        ),
        "definite_occupied_query_count": definite_occupied,
        "definite_clear_query_count": definite_clear,
    }
    scientifically_evaluable = (
        evaluability["evaluable_parent_count"] >= EXPECTED_GATES["minimum_evaluable_parents"]
        and evaluability["parents_with_definite_occupied_label"] >= EXPECTED_GATES["minimum_parents_with_definite_occupied"]
        and definite_occupied >= EXPECTED_GATES["minimum_definite_occupied_queries"]
        and evaluability["parents_with_definite_clear_label"] >= EXPECTED_GATES["minimum_parents_with_definite_clear"]
        and definite_clear >= EXPECTED_GATES["minimum_definite_clear_queries"]
    )
    gates = {
        "occupied_precision_on_definite_labels": {
            "value": occupied_precision,
            "minimum": EXPECTED_GATES["minimum_occupied_precision"],
            "passed": occupied_precision >= EXPECTED_GATES["minimum_occupied_precision"],
        },
        "one_sided_95_wilson_occupied_precision_lower_bound": {
            "value": occupied_wilson,
            "minimum": EXPECTED_GATES["minimum_one_sided_95_wilson_occupied_precision_lower_bound"],
            "passed": occupied_wilson >= EXPECTED_GATES["minimum_one_sided_95_wilson_occupied_precision_lower_bound"],
        },
        "occupied_recall": {
            "value": occupied_recall,
            "minimum": EXPECTED_GATES["minimum_occupied_recall"],
            "passed": occupied_recall >= EXPECTED_GATES["minimum_occupied_recall"],
        },
        "parent_macro_definite_occupancy_coverage_increase_absolute": {
            "value": macro_increase,
            "minimum": EXPECTED_GATES["minimum_parent_macro_occupancy_coverage_increase_absolute"],
            "parent_denominator": len(parent_improvements),
            "passed": macro_increase >= EXPECTED_GATES["minimum_parent_macro_occupancy_coverage_increase_absolute"],
        },
        "clear_specificity_on_definite_clear": {
            "value": clear_specificity,
            "minimum": EXPECTED_GATES["minimum_clear_specificity"],
            "passed": clear_specificity >= EXPECTED_GATES["minimum_clear_specificity"],
        },
        "one_sided_95_wilson_clear_specificity_lower_bound": {
            "value": clear_wilson,
            "minimum": EXPECTED_GATES["minimum_one_sided_95_wilson_clear_specificity_lower_bound"],
            "passed": clear_wilson >= EXPECTED_GATES["minimum_one_sided_95_wilson_clear_specificity_lower_bound"],
        },
        "maximum_clear_outputs": {
            "value": clear_outputs,
            "maximum": EXPECTED_GATES["maximum_clear_outputs"],
            "passed": clear_outputs <= EXPECTED_GATES["maximum_clear_outputs"],
        },
    }
    all_gates = all(row["passed"] for row in gates.values())
    terminal = NOT_EVALUABLE_TERMINAL if not scientifically_evaluable else PASS_TERMINAL if all_gates else FAIL_TERMINAL
    return {
        "terminal": terminal,
        "passed": bool(scientifically_evaluable and all_gates),
        "scientifically_evaluable": bool(scientifically_evaluable),
        "evaluability": evaluability,
        "label_state_counts": {
            state: int(label_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
        },
        "prediction_state_counts": {
            state: int(prediction_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
        },
        "occupied_true_positive": int(occupied_tp),
        "occupied_false_positive_against_definite_clear": int(occupied_fp),
        "occupied_false_negative": int(occupied_fn),
        "occupied_predictions_on_truth_unknown": int(predicted_on_unknown),
        "clear_specific_successes": int(clear_specific_success),
        "unknown_is_negative": False,
        "gates": gates,
        "all_confirmation_gates_passed": bool(all_gates),
        "per_parent": per_parent,
    }


__all__ = [
    "EXPECTED_GATES",
    "FAIL_TERMINAL",
    "NOT_EVALUABLE_TERMINAL",
    "PASS_TERMINAL",
    "PhaseBMetricsError",
    "positive_state",
    "summarize",
]
