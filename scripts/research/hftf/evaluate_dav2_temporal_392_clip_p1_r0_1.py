#!/usr/bin/env python3
"""Deterministic clip/parent-cluster evaluator for the P3 R0.1 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROTOCOL_SCHEMA = "blindassist_dav2_temporal_392_student_p3_r0_1_protocol"
LEDGER_SCHEMA = "blindassist_dav2_temporal_392_clip_p1_r0_1_opened_ledger"
RESULT_SCHEMA = "blindassist_dav2_temporal_392_clip_p1_r0_1_result"
STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
TRANSITIONS = tuple(f"{left}_TO_{right}" for left in STATES for right in STATES)
SHA256_CHARS = frozenset("0123456789ABCDEF")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_sha(value: Any) -> bool:
    normalized = str(value).upper()
    return len(normalized) == 64 and set(normalized) <= SHA256_CHARS


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _final_state(geometry_state: str, abstain: bool) -> str:
    return "UNKNOWN_GROUND" if abstain else geometry_state


def _f1(
    truth: list[str], prediction: list[str], supported_classes: tuple[str, ...]
) -> float | None:
    if not truth or len(truth) != len(prediction) or not supported_classes:
        return None
    scores = []
    for state in supported_classes:
        true_positive = sum(t == state and p == state for t, p in zip(truth, prediction))
        false_positive = sum(t != state and p == state for t, p in zip(truth, prediction))
        false_negative = sum(t == state and p != state for t, p in zip(truth, prediction))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def validate_opened_ledger(ledger: dict[str, Any]) -> None:
    if set(ledger) != {"schema", "identity_manifest_sha256", "sealed_bundle_sha256", "rows"}:
        raise ValueError("opened ledger top-level fields drifted")
    if ledger["schema"] != LEDGER_SCHEMA:
        raise ValueError("opened ledger schema drifted")
    if not _valid_sha(ledger["identity_manifest_sha256"]) or not _valid_sha(
        ledger["sealed_bundle_sha256"]
    ):
        raise ValueError("opened ledger identity hash drifted")
    rows = ledger["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("opened ledger is empty")
    expected_row = {
        "clip_id",
        "parent_id",
        "frame_index",
        "timestamp_ns",
        "truth",
        "baseline",
        "candidate",
    }
    expected_truth = {
        "clearance_m",
        "clearance_valid",
        "geometry_state",
        "external_abstain_target",
    }
    expected_prediction = {
        "paired_pixel_coverage",
        "metric_abs_rel",
        "scale_aligned_abs_rel",
        "clearance_m",
        "geometry_state",
        "external_abstain",
    }
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if set(row) != expected_row:
            raise ValueError("opened ledger row fields drifted")
        if set(row["truth"]) != expected_truth:
            raise ValueError("truth fields drifted")
        truth = row["truth"]
        if not (
            all(isinstance(truth[name], list) and len(truth[name]) == 3 for name in expected_truth)
            and all(value in STATES for value in truth["geometry_state"])
            and all(isinstance(value, bool) for value in truth["clearance_valid"])
            and all(isinstance(value, bool) for value in truth["external_abstain_target"])
        ):
            raise ValueError("truth band values drifted")
        for arm in ("baseline", "candidate"):
            if set(row[arm]) != expected_prediction:
                raise ValueError(f"{arm} fields drifted")
            if row[arm]["geometry_state"] not in (
                ["CLEAR"] * 3,
                ["OCCUPIED"] * 3,
                ["UNKNOWN_GROUND"] * 3,
            ) and not (
                isinstance(row[arm]["geometry_state"], list)
                and len(row[arm]["geometry_state"]) == 3
                and all(value in STATES for value in row[arm]["geometry_state"])
            ):
                raise ValueError(f"bad {arm} geometry state")
            if not (
                isinstance(row[arm]["clearance_m"], list)
                and len(row[arm]["clearance_m"]) == 3
                and isinstance(row[arm]["external_abstain"], list)
                and len(row[arm]["external_abstain"]) == 3
                and all(isinstance(value, bool) for value in row[arm]["external_abstain"])
            ):
                raise ValueError(f"bad {arm} band values")
            if not (
                finite(row[arm]["paired_pixel_coverage"])
                and 0.0 <= float(row[arm]["paired_pixel_coverage"]) <= 1.0
                and finite(row[arm]["metric_abs_rel"])
                and float(row[arm]["metric_abs_rel"]) >= 0.0
                and finite(row[arm]["scale_aligned_abs_rel"])
                and float(row[arm]["scale_aligned_abs_rel"]) >= 0.0
            ):
                # Non-finite values are preserved for fail-closed evaluation;
                # finite but impossible values are malformed input.
                if all(finite(row[arm][name]) for name in (
                    "paired_pixel_coverage", "metric_abs_rel", "scale_aligned_abs_rel"
                )):
                    raise ValueError(f"impossible {arm} depth metric")
            for value in row[arm]["clearance_m"]:
                if value is not None and (not finite(value) or float(value) < 0.0):
                    raise ValueError(f"bad {arm} clearance")
        clips[str(row["clip_id"])].append(row)
    for clip_id, clip_rows in clips.items():
        ordered = sorted(clip_rows, key=lambda row: int(row["frame_index"]))
        if [int(row["frame_index"]) for row in ordered] != [0, 1, 2, 3]:
            raise ValueError(f"clip is not exactly four frames: {clip_id}")
        if len({str(row["parent_id"]) for row in ordered}) != 1:
            raise ValueError(f"parent drift within clip: {clip_id}")
        timestamps = [int(row["timestamp_ns"]) for row in ordered]
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if not all(0 < gap <= 500_000_000 for gap in gaps):
            raise ValueError(f"invalid clip timestamps: {clip_id}")


def _metric_summary(
    rows: list[dict[str, Any]],
    arm: str,
    supported_transition_classes: tuple[str, ...],
) -> dict[str, float | None]:
    ordered_clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ordered_clips[str(row["clip_id"])].append(row)
    coverage: list[float] = []
    abs_rel: list[float] = []
    scale_abs_rel: list[float] = []
    clearance_errors: list[float] = []
    clearance_delta_errors: list[float] = []
    truth_transitions: list[str] = []
    predicted_transitions: list[str] = []
    transition_exact = 0
    transition_count = 0
    ground_known = 0
    ground_recovered = 0
    false_clear = 0
    truth_known = 0
    invalid_to_known = 0
    invalid_count = 0
    valid_to_unknown = 0
    valid_known_count = 0
    truth_delta_pairs = 0
    candidate_delta_pairs = 0
    truth_clearance_rows = 0
    candidate_clearance_rows = 0

    for clip_rows in ordered_clips.values():
        clip_rows.sort(key=lambda row: int(row["frame_index"]))
        for row in clip_rows:
            prediction = row[arm]
            truth = row["truth"]
            for metric, target in (
                ("paired_pixel_coverage", coverage),
                ("metric_abs_rel", abs_rel),
                ("scale_aligned_abs_rel", scale_abs_rel),
            ):
                if finite(prediction[metric]):
                    target.append(float(prediction[metric]))
            for band in range(3):
                truth_state = truth["geometry_state"][band]
                predicted_geometry = prediction["geometry_state"][band]
                predicted_final = _final_state(
                    predicted_geometry, bool(prediction["external_abstain"][band])
                )
                truth_external_invalid = bool(truth["external_abstain_target"][band])
                if truth_state != "UNKNOWN_GROUND":
                    ground_known += 1
                    ground_recovered += int(predicted_geometry != "UNKNOWN_GROUND")
                if truth_external_invalid:
                    invalid_count += 1
                    invalid_to_known += int(predicted_final != "UNKNOWN_GROUND")
                elif truth_state != "UNKNOWN_GROUND":
                    valid_known_count += 1
                    valid_to_unknown += int(predicted_final == "UNKNOWN_GROUND")
                    truth_known += 1
                    false_clear += int(
                        truth_state == "OCCUPIED" and predicted_final == "CLEAR"
                    )
                if bool(truth["clearance_valid"][band]):
                    truth_clearance_rows += 1
                    truth_clearance = truth["clearance_m"][band]
                    predicted_clearance = prediction["clearance_m"][band]
                    if finite(truth_clearance) and finite(predicted_clearance):
                        candidate_clearance_rows += 1
                        clearance_errors.append(
                            abs(float(predicted_clearance) - float(truth_clearance))
                        )
        for previous, current in zip(clip_rows, clip_rows[1:]):
            for band in range(3):
                truth_left = previous["truth"]["geometry_state"][band]
                truth_right = current["truth"]["geometry_state"][band]
                prediction_left = previous[arm]["geometry_state"][band]
                prediction_right = current[arm]["geometry_state"][band]
                truth_transition = f"{truth_left}_TO_{truth_right}"
                predicted_transition = f"{prediction_left}_TO_{prediction_right}"
                truth_transitions.append(truth_transition)
                predicted_transitions.append(predicted_transition)
                transition_count += 1
                transition_exact += int(truth_transition == predicted_transition)
                truth_pair_valid = bool(previous["truth"]["clearance_valid"][band]) and bool(
                    current["truth"]["clearance_valid"][band]
                )
                if truth_pair_valid:
                    truth_delta_pairs += 1
                    truth_left_clearance = previous["truth"]["clearance_m"][band]
                    truth_right_clearance = current["truth"]["clearance_m"][band]
                    prediction_left_clearance = previous[arm]["clearance_m"][band]
                    prediction_right_clearance = current[arm]["clearance_m"][band]
                    if all(
                        finite(value)
                        for value in (
                            truth_left_clearance,
                            truth_right_clearance,
                            prediction_left_clearance,
                            prediction_right_clearance,
                        )
                    ):
                        candidate_delta_pairs += 1
                        clearance_delta_errors.append(
                            abs(
                                (float(prediction_right_clearance) - float(prediction_left_clearance))
                                - (float(truth_right_clearance) - float(truth_left_clearance))
                            )
                        )
    return {
        "minimum_paired_pixel_coverage": min(coverage) if len(coverage) == len(rows) else None,
        "metric_abs_rel_median": _median(abs_rel) if len(abs_rel) == len(rows) else None,
        "scale_aligned_abs_rel_median": (
            _median(scale_abs_rel) if len(scale_abs_rel) == len(rows) else None
        ),
        "clearance_mae_m": (
            _mean(clearance_errors)
            if truth_clearance_rows > 0 and candidate_clearance_rows == truth_clearance_rows
            else None
        ),
        "clearance_delta_mae_m": _mean(clearance_delta_errors),
        "clearance_delta_pair_coverage": _rate(candidate_delta_pairs, truth_delta_pairs),
        "geometry_transition_exact_agreement": _rate(transition_exact, transition_count),
        "geometry_transition_macro_f1": _f1(
            truth_transitions, predicted_transitions, supported_transition_classes
        ),
        "ground_recovery_rate": _rate(ground_recovered, ground_known),
        "false_clear_rate_all_truth_known": _rate(false_clear, truth_known),
        "invalid_to_known_rate": _rate(invalid_to_known, invalid_count),
        "valid_to_unknown_rate": _rate(valid_to_unknown, valid_known_count),
    }


def _resample_parent_rows(
    rows_by_parent: dict[str, list[dict[str, Any]]], draw: np.ndarray
) -> list[dict[str, Any]]:
    parents = sorted(rows_by_parent)
    sampled: list[dict[str, Any]] = []
    for occurrence, parent_index in enumerate(draw):
        parent = parents[int(parent_index)]
        for row in rows_by_parent[parent]:
            copied = dict(row)
            copied["clip_id"] = f"bootstrap-{occurrence}-{row['clip_id']}"
            copied["parent_id"] = f"bootstrap-{occurrence}-{parent}"
            sampled.append(copied)
    return sampled


def _bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    supported_transition_classes: tuple[str, ...],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int | None]:
    rows_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_parent[str(row["parent_id"])].append(row)
    parent_count = len(rows_by_parent)
    if parent_count == 0:
        return {"finite_draws": 0, "candidate_lower": None, "candidate_upper": None, "difference_lower": None, "difference_upper": None}
    rng = np.random.default_rng(seed)
    candidate_values: list[float] = []
    differences: list[float] = []
    for _ in range(draws):
        draw = rng.integers(0, parent_count, size=parent_count)
        sampled = _resample_parent_rows(rows_by_parent, draw)
        baseline = _metric_summary(sampled, "baseline", supported_transition_classes)[metric]
        candidate = _metric_summary(sampled, "candidate", supported_transition_classes)[metric]
        if finite(baseline) and finite(candidate):
            candidate_values.append(float(candidate))
            differences.append(float(candidate) - float(baseline))
    if len(candidate_values) < math.ceil(0.95 * draws):
        return {"finite_draws": len(candidate_values), "candidate_lower": None, "candidate_upper": None, "difference_lower": None, "difference_upper": None}
    return {
        "finite_draws": len(candidate_values),
        "candidate_lower": float(np.quantile(candidate_values, 0.05)),
        "candidate_upper": float(np.quantile(candidate_values, 0.95)),
        "difference_lower": float(np.quantile(differences, 0.05)),
        "difference_upper": float(np.quantile(differences, 0.95)),
    }


def _safe_gate(value: Any, comparator: Callable[[float], bool]) -> bool:
    return finite(value) and comparator(float(value))


def evaluate(protocol: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("P3 R0.1 protocol schema drift")
    validate_opened_ledger(ledger)
    rows = ledger["rows"]
    if len({row["clip_id"] for row in rows}) < int(protocol["clip_p1"]["minimum_evaluable_clips"]):
        raise ValueError("insufficient evaluable clips")
    if len({row["parent_id"] for row in rows}) < int(protocol["clip_p1"]["minimum_video_parents"]):
        raise ValueError("insufficient video parents")
    supported = TRANSITIONS
    baseline = _metric_summary(rows, "baseline", supported)
    candidate = _metric_summary(rows, "candidate", supported)
    bootstrap_config = protocol["clip_p1"]["bootstrap"]
    draws = int(bootstrap_config["draws"])
    seed = int(bootstrap_config["seed"])
    intervals = {
        metric: _bootstrap(
            rows,
            metric,
            supported,
            draws=draws,
            seed=seed + index,
        )
        for index, metric in enumerate(candidate)
    }
    tolerances = protocol["clip_p1"]["noninferiority_tolerances"]
    absolutes = protocol["clip_p1"]["absolute_gates"]
    gates = {
        "minimum_pixel_coverage": _safe_gate(
            intervals["minimum_paired_pixel_coverage"]["difference_lower"],
            lambda value: value >= -float(tolerances["minimum_pixel_coverage_drop"]),
        ),
        "metric_abs_rel": _safe_gate(
            intervals["metric_abs_rel_median"]["difference_upper"],
            lambda value: value <= float(tolerances["metric_abs_rel_absolute_increase"]),
        ),
        "scale_aligned_abs_rel": _safe_gate(
            intervals["scale_aligned_abs_rel_median"]["difference_upper"],
            lambda value: value <= float(tolerances["scale_aligned_abs_rel_absolute_increase"]),
        ),
        "clearance_mae": _safe_gate(
            intervals["clearance_mae_m"]["difference_upper"],
            lambda value: value <= float(tolerances["clearance_mae_increase_m"]),
        ),
        "clearance_mae_absolute": _safe_gate(
            intervals["clearance_mae_m"]["candidate_upper"],
            lambda value: value <= float(absolutes["maximum_clearance_mae_m"]),
        ),
        "clearance_delta_noninferiority": _safe_gate(
            intervals["clearance_delta_mae_m"]["difference_upper"],
            lambda value: value <= float(tolerances["clearance_delta_mae_increase_m"]),
        ),
        "clearance_delta_absolute": _safe_gate(
            intervals["clearance_delta_mae_m"]["candidate_upper"],
            lambda value: value <= float(absolutes["maximum_clearance_delta_mae_m"]),
        ),
        "clearance_delta_pair_coverage": _safe_gate(
            intervals["clearance_delta_pair_coverage"]["difference_lower"],
            lambda value: value >= -float(tolerances["clearance_delta_pair_coverage_drop"]),
        ),
        "transition_macro_f1": _safe_gate(
            intervals["geometry_transition_macro_f1"]["difference_lower"],
            lambda value: value >= -float(tolerances["transition_macro_f1_drop"]),
        ),
        "transition_exact_agreement": _safe_gate(
            intervals["geometry_transition_exact_agreement"]["difference_lower"],
            lambda value: value >= -float(tolerances["transition_exact_agreement_drop"]),
        ),
        "ground_recovery": _safe_gate(
            intervals["ground_recovery_rate"]["difference_lower"],
            lambda value: value >= -float(tolerances["ground_recovery_rate_drop"]),
        ),
        "false_clear_noninferiority": _safe_gate(
            intervals["false_clear_rate_all_truth_known"]["difference_upper"],
            lambda value: value <= float(tolerances["false_clear_rate_increase"]),
        ),
        "false_clear_absolute": _safe_gate(
            intervals["false_clear_rate_all_truth_known"]["candidate_upper"],
            lambda value: value <= float(absolutes["maximum_false_clear_rate"]),
        ),
        "invalid_to_known_noninferiority": _safe_gate(
            intervals["invalid_to_known_rate"]["difference_upper"],
            lambda value: value <= float(tolerances["invalid_to_known_rate_increase"]),
        ),
        "invalid_to_known_absolute": _safe_gate(
            intervals["invalid_to_known_rate"]["candidate_upper"],
            lambda value: value <= float(absolutes["maximum_invalid_to_known_rate"]),
        ),
        "valid_to_unknown_noninferiority": _safe_gate(
            intervals["valid_to_unknown_rate"]["difference_upper"],
            lambda value: value <= float(tolerances["valid_to_unknown_rate_increase"]),
        ),
        "valid_to_unknown_absolute": _safe_gate(
            intervals["valid_to_unknown_rate"]["candidate_upper"],
            lambda value: value <= float(absolutes["maximum_valid_to_unknown_rate"]),
        ),
    }
    undefined = sorted(
        metric
        for metric, interval in intervals.items()
        if any(interval[key] is None for key in ("candidate_lower", "candidate_upper", "difference_lower", "difference_upper"))
    )
    passed = not undefined and all(gates.values())
    return {
        "schema": RESULT_SCHEMA,
        "data_role": "SINGLE_OPENING_DEVELOPMENT_CLIP_HOLDOUT",
        "baseline": baseline,
        "candidate": candidate,
        "parent_cluster_bootstrap": intervals,
        "undefined_metrics": undefined,
        "gates": gates,
        "passed": passed,
        "terminal": (
            "P3_A2_392_TEMPORAL_STUDENT_QUALITY_NONINFERIORITY_SUPPORTED_A5S_STACK_ELIGIBLE"
            if passed
            else "P3_A2_392_TEMPORAL_STUDENT_QUALITY_NONINFERIORITY_NOT_SUPPORTED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--opened-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(Path(__file__).resolve()) != protocol["implementation"]["clip_p1_evaluator_sha256"]:
        raise ValueError("clip P1 evaluator source hash mismatch")
    result = evaluate(
        protocol,
        json.loads(args.opened_ledger.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
