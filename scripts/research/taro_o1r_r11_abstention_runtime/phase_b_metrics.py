#!/usr/bin/env python3
"""Frozen R11 dual-class evaluability and confirmation metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


PASS_TERMINAL = "WILD_LAB_RESEARCH_FACTOR_CONFIRMATION_PASS"
FAIL_TERMINAL = "FAIL_FIXED_CONFIRMATION_GATE"
NOT_EVALUABLE_TERMINAL = "NOT_EVALUABLE_DUAL_CLASS_COVERAGE"

EVALUABILITY_GATES = {
    "selected_parent_count": 24,
    "minimum_evaluable_parents": 16,
    "minimum_parents_with_definite_occupied": 12,
    "minimum_definite_occupied_queries": 200,
    "minimum_parents_with_definite_clear": 4,
    "minimum_physical_frames_with_definite_clear": 12,
    "minimum_definite_clear_queries": 20,
}
CONFIRMATION_GATES = {
    "minimum_candidate_occupied_precision": 0.9,
    "minimum_one_sided_95_wilson_candidate_occupied_precision_lower_bound": 0.8,
    "minimum_candidate_occupied_recall": 0.9,
    "minimum_parent_macro_definite_occupied_recall": 0.9,
    "maximum_micro_occupied_recall_loss_vs_r7": 0.01,
    "maximum_parent_macro_occupied_recall_loss_vs_r7": 0.01,
    "candidate_false_positives_must_not_exceed_r7": True,
    "minimum_query_clear_specificity": 0.9,
    "minimum_clear_frame_specificity": 0.9,
    "minimum_one_sided_95_wilson_clear_frame_specificity_lower_bound": 0.8,
    "minimum_parent_macro_clear_frame_specificity": 0.9,
    "maximum_clear_outputs": 0,
    "unknown_is_negative": False,
}
ABSTENTION_EFFECT_GATES = {
    "minimum_abstained_definite_clear_frames_for_effect_claim": 2,
    "minimum_parents_with_abstained_definite_clear_frame_for_effect_claim": 2,
}
STATES = {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}


class R11PhaseBMetricsError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise R11PhaseBMetricsError(code, message)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator / denominator), 12)


def _wilson(successes: int, total: int) -> float:
    return round(float(r7_canary._wilson_lower(successes, total)), 12)


def _state(row: Mapping[str, Any]) -> str:
    state = str(row.get("state"))
    require(state in STATES, "R11_PHASE_B_STATE", "query state is invalid")
    return state


def summarize(
    selected_identities: Sequence[tuple[str, str]],
    r7_baselines: Sequence[Mapping[str, Any]],
    r11_candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce sealed R7/R11 predictions against FARO labels without treating UNKNOWN as negative."""

    require(
        len(selected_identities) == EVALUABILITY_GATES["selected_parent_count"]
        and len(set(selected_identities)) == EVALUABILITY_GATES["selected_parent_count"],
        "R11_PHASE_B_SELECTED_IDENTITIES",
        "exactly 24 unique selected identities are required",
    )
    require(
        len(r7_baselines) == len(r11_candidates) == len(labels) > 0,
        "R11_PHASE_B_RECORD_COUNT",
        "baseline/candidate/label frame counts differ",
    )
    expected = set(selected_identities)
    label_counts: Counter[str] = Counter()
    baseline_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    parent_label_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    parent_occ: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    parent_clear_frames: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    baseline_tp = baseline_fp = candidate_tp = candidate_fp = candidate_fn = 0
    baseline_on_unknown = candidate_on_unknown = clear_outputs = 0
    definite_clear_frames = baseline_clear_frame_successes = candidate_clear_frame_successes = 0
    abstained_clear_frames = 0
    abstained_clear_parents: set[tuple[str, str]] = set()

    for baseline, candidate, label in zip(r7_baselines, r11_candidates, labels, strict=True):
        identity = (str(label.get("parent_id")), str(label.get("video_id")))
        require(
            identity in expected
            and baseline.get("parent_id") == candidate.get("parent_id") == label.get("parent_id")
            and baseline.get("video_id") == candidate.get("video_id") == label.get("video_id")
            and baseline.get("physical_frame_id") == candidate.get("physical_frame_id") == label.get("physical_frame_id"),
            "R11_PHASE_B_FRAME_ALIGNMENT",
            "R7/R11/FARO frame alignment drift",
        )
        baseline_rows = baseline.get("query_results")
        candidate_rows = candidate.get("query_results")
        label_rows = label.get("query_labels")
        require(
            isinstance(baseline_rows, list)
            and isinstance(candidate_rows, list)
            and isinstance(label_rows, list)
            and len(baseline_rows) == len(candidate_rows) == len(label_rows) == 9,
            "R11_PHASE_B_QUERY_COUNT",
            "each frame must contain nine aligned queries",
        )
        frame_has_clear = frame_baseline_clear_success = frame_candidate_clear_success = False
        frame_has_abstained_clear = False
        for index, (base_row, cand_row, truth_row) in enumerate(
            zip(baseline_rows, candidate_rows, label_rows, strict=True)
        ):
            require(
                base_row.get("grid_index") == cand_row.get("grid_index") == truth_row.get("grid_index") == index
                and base_row.get("query_id") == cand_row.get("query_id") == truth_row.get("query_id"),
                "R11_PHASE_B_QUERY_ALIGNMENT",
                "R7/R11/FARO query alignment drift",
            )
            base_state, cand_state, truth = _state(base_row), _state(cand_row), _state(truth_row)
            require(
                cand_state != "OCCUPIED_OBSERVED" or base_state == "OCCUPIED_OBSERVED",
                "R11_PHASE_B_SUBSET",
                "R11 occupied prediction is not an R7 occupied subset",
            )
            label_counts[truth] += 1
            baseline_counts[base_state] += 1
            candidate_counts[cand_state] += 1
            parent_label_counts[identity][truth] += 1
            clear_outputs += cand_state == "CLEAR_OBSERVED"
            if truth == "OCCUPIED_OBSERVED":
                baseline_tp += base_state == "OCCUPIED_OBSERVED"
                candidate_tp += cand_state == "OCCUPIED_OBSERVED"
                candidate_fn += cand_state != "OCCUPIED_OBSERVED"
                parent_occ[identity]["truth"] += 1
                parent_occ[identity]["baseline_tp"] += base_state == "OCCUPIED_OBSERVED"
                parent_occ[identity]["candidate_tp"] += cand_state == "OCCUPIED_OBSERVED"
            elif truth == "CLEAR_OBSERVED":
                frame_has_clear = True
                frame_baseline_clear_success = frame_baseline_clear_success or base_state == "OCCUPIED_OBSERVED"
                frame_candidate_clear_success = frame_candidate_clear_success or cand_state == "OCCUPIED_OBSERVED"
                frame_has_abstained_clear = frame_has_abstained_clear or (
                    base_state == "OCCUPIED_OBSERVED" and cand_state == "UNKNOWN"
                )
                baseline_fp += base_state == "OCCUPIED_OBSERVED"
                candidate_fp += cand_state == "OCCUPIED_OBSERVED"
            else:
                baseline_on_unknown += base_state == "OCCUPIED_OBSERVED"
                candidate_on_unknown += cand_state == "OCCUPIED_OBSERVED"
        if frame_has_clear:
            definite_clear_frames += 1
            baseline_success = not frame_baseline_clear_success
            candidate_success = not frame_candidate_clear_success
            baseline_clear_frame_successes += baseline_success
            candidate_clear_frame_successes += candidate_success
            parent_clear_frames[identity]["total"] += 1
            parent_clear_frames[identity]["baseline_success"] += baseline_success
            parent_clear_frames[identity]["candidate_success"] += candidate_success
            if frame_has_abstained_clear:
                abstained_clear_frames += 1
                abstained_clear_parents.add(identity)

    definite_occupied = int(label_counts["OCCUPIED_OBSERVED"])
    definite_clear = int(label_counts["CLEAR_OBSERVED"])
    baseline_precision_denominator = baseline_tp + baseline_fp
    candidate_precision_denominator = candidate_tp + candidate_fp
    baseline_recall = _ratio(baseline_tp, definite_occupied)
    candidate_recall = _ratio(candidate_tp, definite_occupied)
    candidate_precision = _ratio(candidate_tp, candidate_precision_denominator)
    query_clear_specificity = _ratio(definite_clear - candidate_fp, definite_clear)
    clear_frame_specificity = _ratio(candidate_clear_frame_successes, definite_clear_frames)

    occupied_parent_rows: list[dict[str, Any]] = []
    clear_parent_rows: list[dict[str, Any]] = []
    per_parent: list[dict[str, Any]] = []
    for identity in selected_identities:
        occ = parent_occ[identity]
        clear = parent_clear_frames[identity]
        occ_denominator = int(occ["truth"])
        clear_denominator = int(clear["total"])
        baseline_parent_recall = None if occ_denominator == 0 else _ratio(int(occ["baseline_tp"]), occ_denominator)
        candidate_parent_recall = None if occ_denominator == 0 else _ratio(int(occ["candidate_tp"]), occ_denominator)
        baseline_parent_clear = None if clear_denominator == 0 else _ratio(int(clear["baseline_success"]), clear_denominator)
        candidate_parent_clear = None if clear_denominator == 0 else _ratio(int(clear["candidate_success"]), clear_denominator)
        if occ_denominator:
            occupied_parent_rows.append(
                {"baseline": baseline_parent_recall, "candidate": candidate_parent_recall}
            )
        if clear_denominator:
            clear_parent_rows.append(
                {"baseline": baseline_parent_clear, "candidate": candidate_parent_clear}
            )
        per_parent.append(
            {
                "parent_id": identity[0],
                "video_id": identity[1],
                "label_state_counts": {
                    state: int(parent_label_counts[identity][state])
                    for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
                },
                "definite_occupied_query_count": occ_denominator,
                "baseline_definite_occupied_recall": baseline_parent_recall,
                "candidate_definite_occupied_recall": candidate_parent_recall,
                "definite_clear_frame_count": clear_denominator,
                "baseline_clear_frame_specificity": baseline_parent_clear,
                "candidate_clear_frame_specificity": candidate_parent_clear,
            }
        )

    baseline_macro_occ = _ratio(
        round(sum(float(row["baseline"]) for row in occupied_parent_rows) * 10**12),
        len(occupied_parent_rows) * 10**12,
    ) if occupied_parent_rows else 0.0
    candidate_macro_occ = _ratio(
        round(sum(float(row["candidate"]) for row in occupied_parent_rows) * 10**12),
        len(occupied_parent_rows) * 10**12,
    ) if occupied_parent_rows else 0.0
    candidate_macro_clear = _ratio(
        round(sum(float(row["candidate"]) for row in clear_parent_rows) * 10**12),
        len(clear_parent_rows) * 10**12,
    ) if clear_parent_rows else 0.0
    micro_recall_loss = round(baseline_recall - candidate_recall, 12)
    macro_recall_loss = round(baseline_macro_occ - candidate_macro_occ, 12)

    evaluability = {
        "selected_parent_count": len(selected_identities),
        "evaluable_parent_count": sum(
            parent_label_counts[identity]["CLEAR_OBSERVED"]
            + parent_label_counts[identity]["OCCUPIED_OBSERVED"]
            > 0
            for identity in selected_identities
        ),
        "parents_with_definite_occupied": len(occupied_parent_rows),
        "definite_occupied_query_count": definite_occupied,
        "parents_with_definite_clear": len(clear_parent_rows),
        "physical_frames_with_definite_clear": definite_clear_frames,
        "definite_clear_query_count": definite_clear,
    }
    scientifically_evaluable = (
        evaluability["selected_parent_count"] == EVALUABILITY_GATES["selected_parent_count"]
        and evaluability["evaluable_parent_count"] >= EVALUABILITY_GATES["minimum_evaluable_parents"]
        and evaluability["parents_with_definite_occupied"] >= EVALUABILITY_GATES["minimum_parents_with_definite_occupied"]
        and definite_occupied >= EVALUABILITY_GATES["minimum_definite_occupied_queries"]
        and evaluability["parents_with_definite_clear"] >= EVALUABILITY_GATES["minimum_parents_with_definite_clear"]
        and definite_clear_frames >= EVALUABILITY_GATES["minimum_physical_frames_with_definite_clear"]
        and definite_clear >= EVALUABILITY_GATES["minimum_definite_clear_queries"]
    )
    gates = {
        "candidate_occupied_precision": {
            "value": candidate_precision,
            "minimum": CONFIRMATION_GATES["minimum_candidate_occupied_precision"],
            "denominator": candidate_precision_denominator,
            "passed": candidate_precision >= CONFIRMATION_GATES["minimum_candidate_occupied_precision"],
        },
        "one_sided_95_wilson_candidate_occupied_precision_lower_bound": {
            "value": _wilson(candidate_tp, candidate_precision_denominator),
            "minimum": CONFIRMATION_GATES["minimum_one_sided_95_wilson_candidate_occupied_precision_lower_bound"],
        },
        "candidate_occupied_recall": {
            "value": candidate_recall,
            "minimum": CONFIRMATION_GATES["minimum_candidate_occupied_recall"],
        },
        "parent_macro_definite_occupied_recall": {
            "value": candidate_macro_occ,
            "minimum": CONFIRMATION_GATES["minimum_parent_macro_definite_occupied_recall"],
            "parent_denominator": len(occupied_parent_rows),
        },
        "micro_occupied_recall_loss_vs_r7": {
            "value": micro_recall_loss,
            "maximum": CONFIRMATION_GATES["maximum_micro_occupied_recall_loss_vs_r7"],
        },
        "parent_macro_occupied_recall_loss_vs_r7": {
            "value": macro_recall_loss,
            "maximum": CONFIRMATION_GATES["maximum_parent_macro_occupied_recall_loss_vs_r7"],
            "parent_denominator": len(occupied_parent_rows),
        },
        "candidate_false_positives_not_exceed_r7": {
            "candidate_value": int(candidate_fp),
            "r7_value": int(baseline_fp),
        },
        "query_clear_specificity": {
            "value": query_clear_specificity,
            "minimum": CONFIRMATION_GATES["minimum_query_clear_specificity"],
            "denominator": definite_clear,
        },
        "clear_frame_specificity": {
            "value": clear_frame_specificity,
            "minimum": CONFIRMATION_GATES["minimum_clear_frame_specificity"],
            "denominator": definite_clear_frames,
        },
        "one_sided_95_wilson_clear_frame_specificity_lower_bound": {
            "value": _wilson(candidate_clear_frame_successes, definite_clear_frames),
            "minimum": CONFIRMATION_GATES["minimum_one_sided_95_wilson_clear_frame_specificity_lower_bound"],
        },
        "parent_macro_clear_frame_specificity": {
            "value": candidate_macro_clear,
            "minimum": CONFIRMATION_GATES["minimum_parent_macro_clear_frame_specificity"],
            "parent_denominator": len(clear_parent_rows),
        },
        "maximum_clear_outputs": {
            "value": int(clear_outputs),
            "maximum": CONFIRMATION_GATES["maximum_clear_outputs"],
        },
    }
    for row in gates.values():
        if "passed" not in row:
            if "minimum" in row:
                row["passed"] = row["value"] >= row["minimum"]
            elif "maximum" in row:
                row["passed"] = row["value"] <= row["maximum"]
            else:
                row["passed"] = row["candidate_value"] <= row["r7_value"]
    all_gates_passed = all(bool(row["passed"]) for row in gates.values())
    terminal = (
        NOT_EVALUABLE_TERMINAL
        if not scientifically_evaluable
        else PASS_TERMINAL
        if all_gates_passed
        else FAIL_TERMINAL
    )
    effect_evaluable = (
        abstained_clear_frames >= ABSTENTION_EFFECT_GATES["minimum_abstained_definite_clear_frames_for_effect_claim"]
        and len(abstained_clear_parents)
        >= ABSTENTION_EFFECT_GATES["minimum_parents_with_abstained_definite_clear_frame_for_effect_claim"]
    )
    return {
        "terminal": terminal,
        "passed": bool(scientifically_evaluable and all_gates_passed),
        "scientifically_evaluable": bool(scientifically_evaluable),
        "evaluability": evaluability,
        "label_state_counts": {state: int(label_counts[state]) for state in sorted(STATES)},
        "r7_prediction_state_counts": {state: int(baseline_counts[state]) for state in sorted(STATES)},
        "r11_prediction_state_counts": {state: int(candidate_counts[state]) for state in sorted(STATES)},
        "r7_occupied_true_positive": int(baseline_tp),
        "r7_occupied_false_positive_against_definite_clear": int(baseline_fp),
        "r11_occupied_true_positive": int(candidate_tp),
        "r11_occupied_false_positive_against_definite_clear": int(candidate_fp),
        "r11_occupied_false_negative": int(candidate_fn),
        "r7_occupied_predictions_on_truth_unknown": int(baseline_on_unknown),
        "r11_occupied_predictions_on_truth_unknown": int(candidate_on_unknown),
        "baseline_occupied_recall": baseline_recall,
        "candidate_occupied_recall": candidate_recall,
        "baseline_parent_macro_definite_occupied_recall": baseline_macro_occ,
        "candidate_parent_macro_definite_occupied_recall": candidate_macro_occ,
        "baseline_clear_frame_specificity": _ratio(baseline_clear_frame_successes, definite_clear_frames),
        "candidate_clear_frame_specificity": clear_frame_specificity,
        "abstention_effect": {
            "abstained_definite_clear_frames": abstained_clear_frames,
            "parents_with_abstained_definite_clear_frame": len(abstained_clear_parents),
            "effect_evaluable": effect_evaluable,
            "status": "ABSTENTION_EFFECT_EVALUABLE" if effect_evaluable else "ABSTENTION_EFFECT_NOT_EVALUABLE",
            "effect_claim_required_for_absolute_confirmation": False,
        },
        "unknown_is_negative": False,
        "gates": gates,
        "all_confirmation_gates_passed": bool(all_gates_passed),
        "per_parent": per_parent,
    }


__all__ = [
    "ABSTENTION_EFFECT_GATES",
    "CONFIRMATION_GATES",
    "EVALUABILITY_GATES",
    "FAIL_TERMINAL",
    "NOT_EVALUABLE_TERMINAL",
    "PASS_TERMINAL",
    "R11PhaseBMetricsError",
    "summarize",
]
