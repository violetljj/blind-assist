"""Matched-coverage and negative-control evaluator for SVRF-O0."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


ARM_IDS = ("A0", "A1", "A2", "A3", "N0", "N1", "N2", "N3")
SINGLE_ARM_IDS = ("A0", "A1", "A2")
NEGATIVE_CONTROL_IDS = ("N0", "N1", "N2", "N3")
CLASSES = ("APPROACHING", "STABLE", "RECEDING")
BEST_SINGLE_RULE = (
    "among A0-A2 meeting parent-macro coverage >= 0.70 and false-block <= 0.15, "
    "minimize parent-macro false-clear, then maximize approach Macro-F1, then coverage; "
    "no eligible single arm makes O0 fail-close"
)


@dataclass(frozen=True)
class EvaluationPolicy:
    minimum_parent_count: int = 8
    minimum_source_count: int = 2
    a3_parent_macro_coverage_min: float = 0.70
    a3_worst_parent_coverage_min: float = 0.50
    a3_approach_macro_f1_min: float = 0.55
    a3_parent_macro_spearman_min: float = 0.35
    a3_parent_macro_pairwise_ranking_min: float = 0.60
    a3_parent_macro_false_clear_max: float = 0.10
    a3_parent_macro_false_block_max: float = 0.15
    a3_worst_parent_false_block_max: float = 0.25
    matched_false_clear_absolute_gain_min: float = 0.02
    matched_parent_improvement_count_min: int = 6
    matched_source_improvement_count_min: int = 2
    matched_coverage_delta_min: float = -0.02
    negative_control_macro_f1_degradation_min: float = 0.08


@dataclass(frozen=True)
class CandidateRow:
    arm_id: str
    parent_id: str
    sequence_id: str
    frame_id: str
    region_id: str
    status: str
    approach_score: float | None
    risk_score: float | None

    @property
    def valid(self) -> bool:
        return self.status == "VALID_RELATIVE_RISK"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.parent_id, self.sequence_id, self.frame_id, self.region_id

    def validate(self) -> None:
        if self.arm_id not in ARM_IDS or not all((self.parent_id, self.sequence_id, self.frame_id, self.region_id)):
            raise ValueError("SVRF candidate identity is invalid")
        if self.status != "VALID_RELATIVE_RISK" and not self.status.startswith("UNKNOWN_"):
            raise ValueError("SVRF candidate must be valid or fail-closed UNKNOWN")
        if self.valid:
            if self.approach_score is None or not math.isfinite(self.approach_score) or not -1 <= self.approach_score <= 1:
                raise ValueError("valid SVRF candidate approach score must lie in [-1,1]")
            if self.risk_score is None or not math.isfinite(self.risk_score) or not 0 <= self.risk_score <= 1:
                raise ValueError("valid SVRF candidate risk score must lie in [0,1]")


@dataclass(frozen=True)
class TruthRow:
    source_id: str
    parent_id: str
    sequence_id: str
    frame_id: str
    region_id: str
    approach_class: str
    relative_risk: float
    high_risk: bool
    time_seconds: float = 0.0
    high_risk_onset_seconds: float | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.parent_id, self.sequence_id, self.frame_id, self.region_id

    def validate(self) -> None:
        if not all((self.source_id, self.parent_id, self.sequence_id, self.frame_id, self.region_id)):
            raise ValueError("SVRF truth identity is invalid")
        if self.approach_class not in CLASSES:
            raise ValueError("SVRF truth approach class is invalid")
        if not math.isfinite(self.relative_risk) or not 0 <= self.relative_risk <= 1:
            raise ValueError("SVRF truth relative risk must lie in [0,1]")
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("SVRF truth time must be finite and non-negative")
        if self.high_risk_onset_seconds is not None and (
            not math.isfinite(self.high_risk_onset_seconds) or self.high_risk_onset_seconds < 0
        ):
            raise ValueError("SVRF high-risk onset must be finite and non-negative")


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _predict_class(score: float) -> str:
    if score >= 0.20:
        return "APPROACHING"
    if score <= -0.20:
        return "RECEDING"
    return "STABLE"


def _macro_f1(truth: list[str], prediction: list[str]) -> float | None:
    if not truth:
        return None
    scores = []
    for label in CLASSES:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, prediction))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, prediction))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, prediction))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return _mean(scores)


def _balanced_accuracy(truth: list[str], prediction: list[str]) -> float | None:
    recalls = []
    for label in CLASSES:
        positives = sum(actual == label for actual in truth)
        if positives:
            recalls.append(sum(actual == label and predicted == label for actual, predicted in zip(truth, prediction)) / positives)
    return _mean(recalls)


def _average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _approaching_auroc(scores: list[float], truth: list[str]) -> float | None:
    positive = np.asarray([label == "APPROACHING" for label in truth], dtype=bool)
    positive_count = int(np.count_nonzero(positive))
    negative_count = len(positive) - positive_count
    if not positive_count or not negative_count:
        return None
    ranks = _average_ranks(scores) + 1.0
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def _pairwise_accuracy(rows: list[tuple[CandidateRow, TruthRow]]) -> float | None:
    correct = 0
    total = 0
    by_frame: dict[tuple[str, str], list[tuple[CandidateRow, TruthRow]]] = {}
    for candidate, truth in rows:
        by_frame.setdefault((truth.sequence_id, truth.frame_id), []).append((candidate, truth))
    for values in by_frame.values():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                truth_delta = left[1].relative_risk - right[1].relative_risk
                if abs(truth_delta) < 0.10:
                    continue
                predicted_delta = float(left[0].risk_score) - float(right[0].risk_score)
                correct += int(predicted_delta * truth_delta > 0)
                total += 1
    return correct / total if total else None


def _time_to_detection(rows: list[tuple[CandidateRow, TruthRow]]) -> tuple[float | None, float | None, int]:
    episodes: dict[tuple[str, str, str], list[tuple[CandidateRow, TruthRow]]] = {}
    for candidate, truth in rows:
        if truth.high_risk_onset_seconds is not None:
            episodes.setdefault((truth.parent_id, truth.sequence_id, truth.region_id), []).append((candidate, truth))
    delays: list[float] = []
    for values in episodes.values():
        onset = float(values[0][1].high_risk_onset_seconds)
        consecutive = 0
        for candidate, truth in sorted(values, key=lambda pair: pair[1].time_seconds):
            detected = (
                candidate.valid
                and truth.time_seconds >= onset
                and truth.approach_class == "APPROACHING"
                and float(candidate.approach_score) >= 0.20
            )
            consecutive = consecutive + 1 if detected else 0
            if consecutive >= 2:
                delays.append(truth.time_seconds - onset)
                break
    return _mean(delays), len(delays) / len(episodes) if episodes else None, len(episodes)


def _parent_metrics(rows: list[tuple[CandidateRow, TruthRow]]) -> dict[str, float | int | None]:
    valid = [(candidate, truth) for candidate, truth in rows if candidate.valid]
    actual = [truth.approach_class for _, truth in valid]
    predicted = [_predict_class(float(candidate.approach_score)) for candidate, _ in valid]
    high = [(candidate, truth) for candidate, truth in valid if truth.high_risk]
    low = [(candidate, truth) for candidate, truth in valid if not truth.high_risk]
    ttd_mean, ttd_coverage, ttd_episodes = _time_to_detection(rows)
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "coverage": len(valid) / len(rows) if rows else None,
        "approach_macro_f1": _macro_f1(actual, predicted),
        "approach_balanced_accuracy": _balanced_accuracy(actual, predicted),
        "approaching_auroc": _approaching_auroc([float(candidate.approach_score) for candidate, _ in valid], actual),
        "risk_spearman": _spearman([float(candidate.risk_score) for candidate, _ in valid], [truth.relative_risk for _, truth in valid]),
        "pairwise_ranking_accuracy": _pairwise_accuracy(valid),
        "false_clear_rate": sum(float(candidate.risk_score) < 0.50 for candidate, _ in high) / len(high) if high else None,
        "false_block_rate": sum(float(candidate.risk_score) >= 0.50 for candidate, _ in low) / len(low) if low else None,
        "time_to_detection_mean_seconds": ttd_mean,
        "time_to_detection_coverage": ttd_coverage,
        "time_to_detection_episode_count": ttd_episodes,
    }


def _matched_false_clear(
    left: list[CandidateRow], right: list[CandidateRow], truth_by_id: dict[tuple[str, str, str, str], TruthRow]
) -> tuple[float | None, float | None, int]:
    left_by_id = {row.key: row for row in left if row.valid}
    right_by_id = {row.key: row for row in right if row.valid}
    matched = [key for key in left_by_id.keys() & right_by_id.keys() if truth_by_id[key].high_risk]
    if not matched:
        return None, None, 0
    left_rate = sum(float(left_by_id[key].risk_score) < 0.50 for key in matched) / len(matched)
    right_rate = sum(float(right_by_id[key].risk_score) < 0.50 for key in matched) / len(matched)
    return left_rate, right_rate, len(matched)


def evaluate_o0(candidates: list[CandidateRow], truth: list[TruthRow], policy: EvaluationPolicy | None = None) -> dict[str, Any]:
    policy = policy or EvaluationPolicy()
    truth_by_id = {}
    for row in truth:
        row.validate()
        if row.key in truth_by_id:
            raise ValueError("duplicate SVRF truth identity")
        truth_by_id[row.key] = row
    if not truth_by_id:
        raise ValueError("SVRF evaluator truth is required")
    by_arm: dict[str, list[CandidateRow]] = {arm: [] for arm in ARM_IDS}
    seen = set()
    for row in candidates:
        row.validate()
        identity = (row.arm_id, *row.key)
        if identity in seen:
            raise ValueError("duplicate SVRF candidate identity")
        seen.add(identity)
        by_arm[row.arm_id].append(row)
    truth_keys = set(truth_by_id)
    for arm, rows in by_arm.items():
        if {row.key for row in rows} != truth_keys:
            raise ValueError(f"{arm} ledger does not exactly match evaluator truth identities")
    arm_results = {}
    parent_ids = sorted({row.parent_id for row in truth})
    source_ids = sorted({row.source_id for row in truth})
    for arm, rows in by_arm.items():
        by_parent = {}
        for parent in parent_ids:
            joined = [(row, truth_by_id[row.key]) for row in rows if row.parent_id == parent]
            by_parent[parent] = _parent_metrics(joined)
        by_source = {}
        for source in source_ids:
            joined = [(row, truth_by_id[row.key]) for row in rows if truth_by_id[row.key].source_id == source]
            by_source[source] = _parent_metrics(joined)
        metric_names = ("coverage", "approach_macro_f1", "approach_balanced_accuracy", "approaching_auroc", "risk_spearman", "pairwise_ranking_accuracy", "false_clear_rate", "false_block_rate", "time_to_detection_mean_seconds", "time_to_detection_coverage")
        macro = {name: _mean([float(value[name]) for value in by_parent.values() if value[name] is not None]) for name in metric_names}
        source_macro = {name: _mean([float(value[name]) for value in by_source.values() if value[name] is not None]) for name in metric_names}
        worst = {}
        for name in metric_names:
            values = [float(value[name]) for value in by_parent.values() if value[name] is not None]
            worst[name] = (min(values) if name in {"coverage", "approach_macro_f1", "approach_balanced_accuracy", "approaching_auroc", "risk_spearman", "pairwise_ranking_accuracy", "time_to_detection_coverage"} else max(values)) if values else None
        source_worst = {}
        for name in metric_names:
            values = [float(value[name]) for value in by_source.values() if value[name] is not None]
            source_worst[name] = (min(values) if name in {"coverage", "approach_macro_f1", "approach_balanced_accuracy", "approaching_auroc", "risk_spearman", "pairwise_ranking_accuracy", "time_to_detection_coverage"} else max(values)) if values else None
        arm_results[arm] = {"parent_macro": macro, "source_macro": source_macro, "worst_parent": worst, "worst_source": source_worst, "by_parent": by_parent, "by_source": by_source}
    eligible_single_arms = [
        arm
        for arm in SINGLE_ARM_IDS
        if arm_results[arm]["parent_macro"]["coverage"] is not None
        and arm_results[arm]["parent_macro"]["coverage"] >= policy.a3_parent_macro_coverage_min
        and arm_results[arm]["parent_macro"]["false_block_rate"] is not None
        and arm_results[arm]["parent_macro"]["false_block_rate"] <= policy.a3_parent_macro_false_block_max
    ]
    best_single = min(
        eligible_single_arms or SINGLE_ARM_IDS,
        key=lambda arm: (
            float("inf") if arm_results[arm]["parent_macro"]["false_clear_rate"] is None else arm_results[arm]["parent_macro"]["false_clear_rate"],
            -(arm_results[arm]["parent_macro"]["approach_macro_f1"] or 0.0),
            -(arm_results[arm]["parent_macro"]["coverage"] or 0.0),
        ),
    )
    a3_matched, best_matched, matched_count = _matched_false_clear(by_arm["A3"], by_arm[best_single], truth_by_id)
    improved_parents = 0
    eligible_effect_parents = 0
    for parent in parent_ids:
        left = [row for row in by_arm["A3"] if row.parent_id == parent]
        right = [row for row in by_arm[best_single] if row.parent_id == parent]
        left_rate, right_rate, count = _matched_false_clear(left, right, truth_by_id)
        if count and left_rate is not None and right_rate is not None:
            eligible_effect_parents += 1
            improved_parents += int(left_rate < right_rate)
    improved_sources = 0
    eligible_effect_sources = 0
    for source in source_ids:
        left = [row for row in by_arm["A3"] if truth_by_id[row.key].source_id == source]
        right = [row for row in by_arm[best_single] if truth_by_id[row.key].source_id == source]
        left_rate, right_rate, count = _matched_false_clear(left, right, truth_by_id)
        if count and left_rate is not None and right_rate is not None:
            eligible_effect_sources += 1
            improved_sources += int(left_rate < right_rate)
    a3 = arm_results["A3"]
    negative_degradation = {
        arm: (
            None if a3["parent_macro"]["approach_macro_f1"] is None or arm_results[arm]["parent_macro"]["approach_macro_f1"] is None
            else a3["parent_macro"]["approach_macro_f1"] - arm_results[arm]["parent_macro"]["approach_macro_f1"]
        )
        for arm in NEGATIVE_CONTROL_IDS
    }
    negative_source_degradation = {
        arm: (
            None if a3["source_macro"]["approach_macro_f1"] is None or arm_results[arm]["source_macro"]["approach_macro_f1"] is None
            else a3["source_macro"]["approach_macro_f1"] - arm_results[arm]["source_macro"]["approach_macro_f1"]
        )
        for arm in NEGATIVE_CONTROL_IDS
    }
    available = lambda value: value is not None and math.isfinite(float(value))
    gates = {
        "parent_count_at_least_8": len(parent_ids) >= policy.minimum_parent_count,
        "source_count_at_least_2": len({row.source_id for row in truth}) >= policy.minimum_source_count,
        "best_single_arm_eligible": bool(eligible_single_arms),
        "a3_parent_macro_coverage": available(a3["parent_macro"]["coverage"]) and a3["parent_macro"]["coverage"] >= policy.a3_parent_macro_coverage_min,
        "a3_worst_parent_coverage": available(a3["worst_parent"]["coverage"]) and a3["worst_parent"]["coverage"] >= policy.a3_worst_parent_coverage_min,
        "a3_approach_macro_f1": available(a3["parent_macro"]["approach_macro_f1"]) and a3["parent_macro"]["approach_macro_f1"] >= policy.a3_approach_macro_f1_min,
        "a3_parent_macro_spearman": available(a3["parent_macro"]["risk_spearman"]) and a3["parent_macro"]["risk_spearman"] >= policy.a3_parent_macro_spearman_min,
        "a3_parent_macro_pairwise_ranking": available(a3["parent_macro"]["pairwise_ranking_accuracy"]) and a3["parent_macro"]["pairwise_ranking_accuracy"] >= policy.a3_parent_macro_pairwise_ranking_min,
        "a3_parent_macro_false_clear": available(a3["parent_macro"]["false_clear_rate"]) and a3["parent_macro"]["false_clear_rate"] <= policy.a3_parent_macro_false_clear_max,
        "a3_parent_macro_false_block": available(a3["parent_macro"]["false_block_rate"]) and a3["parent_macro"]["false_block_rate"] <= policy.a3_parent_macro_false_block_max,
        "a3_worst_parent_false_block": available(a3["worst_parent"]["false_block_rate"]) and a3["worst_parent"]["false_block_rate"] <= policy.a3_worst_parent_false_block_max,
        "matched_false_clear_gain": available(a3_matched) and available(best_matched) and best_matched - a3_matched >= policy.matched_false_clear_absolute_gain_min,
        "matched_parent_effect_direction": eligible_effect_parents >= policy.matched_parent_improvement_count_min and improved_parents >= policy.matched_parent_improvement_count_min,
        "matched_source_effect_direction": eligible_effect_sources >= policy.matched_source_improvement_count_min and improved_sources >= policy.matched_source_improvement_count_min,
        "matched_coverage_not_bought_by_unknown": available(a3["parent_macro"]["coverage"]) and available(arm_results[best_single]["parent_macro"]["coverage"]) and a3["parent_macro"]["coverage"] - arm_results[best_single]["parent_macro"]["coverage"] >= policy.matched_coverage_delta_min,
        **{
            f"{arm.lower()}_macro_f1_degrades": available(value) and value >= policy.negative_control_macro_f1_degradation_min
            for arm, value in negative_degradation.items()
        },
        **{
            f"{arm.lower()}_source_macro_f1_degrades": available(value) and value >= policy.negative_control_macro_f1_degradation_min
            for arm, value in negative_source_degradation.items()
        },
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist.svrf_o0.evaluation.v1",
        "status": "SVRF_O0_REPRESENTATION_HEADROOM_PASS" if passed else "SVRF_O0_REPRESENTATION_HEADROOM_FAIL_CLOSE",
        "passed": passed,
        "best_single_arm": best_single,
        "matched_high_risk_rows": matched_count,
        "matched_false_clear": {"A3": a3_matched, "best_single": best_matched},
        "matched_parent_effect": {"eligible": eligible_effect_parents, "improved": improved_parents},
        "matched_source_effect": {"eligible": eligible_effect_sources, "improved": improved_sources},
        "negative_control_macro_f1_degradation": negative_degradation,
        "negative_control_source_macro_f1_degradation": negative_source_degradation,
        "arms": arm_results,
        "gates": gates,
        "causality": {"candidate_inputs": "RGB_DERIVED_ONLY", "truth_role": "EVALUATOR_ONLY"},
        "claim_ceiling": "scale-free relative visual risk representation headroom only; NO_HIGH_RISK_EVIDENCE is not CLEAR and no metric clearance, TTC, Android, product or safety claim is authorized",
    }
