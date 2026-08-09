#!/usr/bin/env python3
"""Diagnose the frozen B1-A0 Development failure without creating promotion authority."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.evaluate_b1_a0_development import (
    BANDS,
    HORIZONS,
    ROLE,
    STATES,
    validate_observations,
)
from scripts.research.assistive_geometry.evaluate_b1_a0_synthetic import (
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
)


EXPECTED_SEEDS = (17, 29, 43)
PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b1_a0_failure_anatomy_protocol_v1"
PACKAGE_SCHEMA = "blindassist_assistive_geometry_b1_a0_development_evaluation_package_v1"
FRAME_SCHEMA = "blindassist_assistive_geometry_b1_a0_development_frame_v1"
RESULT_SCHEMA = "blindassist_assistive_geometry_b1_a0_failure_anatomy_result_v1"
TERMINAL = "B1_A0_FAILURE_ANATOMY_COMPLETE_NOT_ELIGIBLE_FOR_PROMOTION"
FAILURE_TERMINAL = "B1_A0_FAILURE_ANATOMY_NOT_EVALUABLE_INPUT_DRIFT"


class AnatomyError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise AnatomyError(code, message, **context)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_reference(value: Any, *, base: Path | None = None, code: str = "PATH_INVALID") -> Path:
    require(isinstance(value, str) and bool(value.strip()), code, "bound path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = (base or _repo_root()) / path
    return path.resolve()


def _validate_sha_binding(path: Path, binding: dict[str, Any], code: str) -> str:
    require(path.is_file(), f"{code}_MISSING", "bound file is missing", path=str(path))
    digest = sha256_file(path)
    require(digest == binding.get("sha256"), f"{code}_SHA_DRIFT", "bound file SHA drift", path=str(path))
    return digest


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_DRIFT", "failure-anatomy protocol schema drift")
    require(protocol.get("data_role") == ROLE, "PROTOCOL_ROLE_DRIFT", "only DEVELOPMENT_SELECTION is allowed")
    require(tuple(protocol.get("required_seeds", [])) == EXPECTED_SEEDS, "PROTOCOL_SEED_DRIFT", "required seed order drift")
    require(protocol.get("expected_observations_per_seed") == 1200, "PROTOCOL_COUNT_DRIFT", "observation count drift")
    authority = protocol.get("authority", {})
    require(authority.get("diagnostic_only") is True, "DIAGNOSTIC_AUTHORITY_MISSING", "diagnostic-only authority is required")
    for key in (
        "promotion",
        "training",
        "threshold_change",
        "seed_selection",
        "development_calibration_access",
        "confirmation_access",
        "deployment",
        "teacher_execution",
        "temporal_execution",
        "product",
        "safety",
    ):
        require(authority.get(key) is False, "AUTHORITY_CEILING_DRIFT", "forbidden authority must remain false", authority=key)
    margins = protocol.get("analysis", {}).get("boundary_margin_bins_m")
    require(margins == [0.1, 0.25, 0.5], "BOUNDARY_BINS_DRIFT", "boundary margin bins drift")
    bindings = protocol.get("bindings", {})
    for key in ("observation_package", "governed_development_result", "analysis_script", "analysis_test"):
        binding = bindings.get(key)
        require(isinstance(binding, dict), "PROTOCOL_BINDING_MISSING", "required protocol binding is missing", binding=key)
        target = resolve_reference(binding.get("path"), code="PROTOCOL_BINDING_PATH_INVALID")
        _validate_sha_binding(target, binding, f"PROTOCOL_{key.upper()}")
    return protocol, sha256_file(path)


def _validate_governed_development_result(protocol: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    binding = protocol["bindings"]["governed_development_result"]
    path = resolve_reference(binding["path"])
    digest = _validate_sha_binding(path, binding, "GOVERNED_DEVELOPMENT_RESULT")
    result = load_json(path)
    require(
        result.get("schema") == "blindassist_assistive_geometry_b1_a0_development_evaluation_governed_result_v1",
        "DEVELOPMENT_RESULT_SCHEMA_DRIFT",
        "governed Development result schema drift",
    )
    require(result.get("terminal") == "B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES", "DEVELOPMENT_TERMINAL_DRIFT", "A0 negative terminal is required")
    require(result.get("status") == "FAIL", "DEVELOPMENT_STATUS_DRIFT", "A0 governed result must remain failed")
    require(result.get("decision", {}).get("a0_depth_only_baseline_promoted") is False, "A0_PROMOTION_DRIFT", "A0 must remain unpromoted")
    require(result.get("data_firewall", {}).get("development_calibration_content_opened") is False, "CALIBRATION_FIREWALL_DRIFT", "Development Calibration must remain sealed")
    require(result.get("data_firewall", {}).get("confirmation_content_opened") is False, "CONFIRMATION_FIREWALL_DRIFT", "Confirmation must remain sealed")
    return result, path, digest


def validate_package_metadata(package: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    require(package.get("schema") == PACKAGE_SCHEMA, "PACKAGE_SCHEMA_DRIFT", "observation package schema drift")
    require(package.get("data_role") == ROLE, "PACKAGE_ROLE_DRIFT", "observation package role drift")
    require(package.get("development_content_opened") is True, "DEVELOPMENT_NOT_CONSUMED", "Development Selection must already be consumed")
    require(package.get("development_calibration_content_opened") is False, "CALIBRATION_FIREWALL_DRIFT", "Development Calibration must remain sealed")
    require(package.get("confirmation_content_opened") is False, "CONFIRMATION_FIREWALL_DRIFT", "Confirmation must remain sealed")
    require(
        package.get("evaluation_protocol_sha256") == protocol.get("accepted_observation_protocol_sha256"),
        "OBSERVATION_PROTOCOL_SHA_DRIFT",
        "observation-time protocol SHA drift",
    )
    runs = package.get("seed_runs")
    require(isinstance(runs, list), "SEED_RUNS_INVALID", "seed_runs must be a list")
    seeds = [run.get("seed") for run in runs]
    require(tuple(seeds) == EXPECTED_SEEDS and len(set(seeds)) == 3, "SEED_ORDER_DRIFT", "seed runs must be exactly 17,29,43 in frozen order", seeds=seeds)
    expected_shas = protocol.get("observation_sha256_by_seed", {})
    for run in runs:
        seed = str(run["seed"])
        require(run.get("observation_count") == 1200, "OBSERVATION_COUNT_DRIFT", "observation count must be 1200", seed=seed)
        require(run.get("observations_sha256") == expected_shas.get(seed), "OBSERVATION_SHA_BINDING_DRIFT", "observation SHA binding drift", seed=seed)
    return runs


def _validate_training_receipts(package_runs: list[dict[str, Any]], governed: dict[str, Any]) -> None:
    receipts = governed.get("training_integrity", {}).get("results", {})
    require(governed.get("training_integrity", {}).get("selected_seed") is None, "SELECTED_SEED_FORBIDDEN", "no seed may be selected")
    for run in package_runs:
        seed = str(run["seed"])
        receipt = receipts.get(seed)
        require(isinstance(receipt, dict), "TRAIN_RECEIPT_MISSING", "governed training receipt is missing", seed=seed)
        path = resolve_reference(run.get("train_result_path"), code="TRAIN_RESULT_PATH_INVALID")
        require(path == resolve_reference(receipt.get("path"), code="GOVERNED_TRAIN_RESULT_PATH_INVALID"), "TRAIN_RESULT_PATH_DRIFT", "package/governed train-result path mismatch", seed=seed)
        require(path.is_file() and sha256_file(path) == receipt.get("sha256"), "TRAIN_RESULT_SHA_DRIFT", "formal train-result SHA drift", seed=seed)
        train = load_json(path)
        require(train.get("terminal") == "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE", "TRAIN_RESULT_TERMINAL_DRIFT", "formal train terminal drift", seed=seed)
        require(train.get("completed_optimizer_steps") == 6000, "TRAIN_RESULT_STEP_DRIFT", "formal train step count drift", seed=seed)
        require(train.get("development_or_confirmation_content_opened") is False, "TRAIN_FIREWALL_DRIFT", "training reports forbidden data access", seed=seed)
        require(train.get("teacher_import_or_execution") is False, "TRAIN_TEACHER_DRIFT", "training reports teacher execution", seed=seed)
        require(run.get("final_model_state_sha256") == receipt.get("final_model_state_sha256"), "MODEL_STATE_SHA_DRIFT", "final model-state SHA drift", seed=seed)


def load_observation_rows(path: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), "OBSERVATION_BLANK_LINE", "blank JSONL line is forbidden", seed=seed, line=line_number)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnatomyError("OBSERVATION_JSON_INVALID", str(error), seed=seed, line=line_number) from error
            require(row.get("schema") == FRAME_SCHEMA, "OBSERVATION_SCHEMA_DRIFT", "observation frame schema drift", seed=seed, line=line_number)
            rows.append(row)
    validate_observations(rows, seed)
    require(len(rows) == 1200, "OBSERVATION_COUNT_DRIFT", "each seed must have 1200 observations", seed=seed, count=len(rows))
    return rows


def frame_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("parent_id"),
        row.get("session_id"),
        row.get("sequence_index"),
        row.get("frame_id"),
        row.get("orientation"),
    )


def validate_cross_seed_order(rows_by_seed: dict[int, list[dict[str, Any]]], expected_parents: tuple[str, ...]) -> None:
    require(tuple(rows_by_seed) == EXPECTED_SEEDS, "ROW_SEED_ORDER_DRIFT", "row seed order drift")
    canonical = [frame_identity(row) for row in rows_by_seed[EXPECTED_SEEDS[0]]]
    require(len(canonical) == len(set(canonical)), "FRAME_IDENTITY_DUPLICATE", "canonical frame identities are not unique")
    parent_order = tuple(dict.fromkeys(str(item[0]) for item in canonical))
    require(parent_order == expected_parents, "PARENT_ORDER_DRIFT", "Development parent order drift", parents=parent_order)
    for seed, rows in rows_by_seed.items():
        identities = [frame_identity(row) for row in rows]
        require(identities == canonical, "CROSS_SEED_FRAME_ORDER_DRIFT", "frame identity/order drift across seeds", seed=seed)
        for previous, current in zip(rows, rows[1:]):
            if (previous["parent_id"], previous["session_id"]) == (current["parent_id"], current["session_id"]):
                require(current["sequence_index"] == previous["sequence_index"] + 1, "SEQUENCE_ORDER_DRIFT", "sequence indices must be strictly consecutive", seed=seed)
    for row_index in range(len(canonical)):
        truth_signatures = []
        for seed in EXPECTED_SEEDS:
            row = rows_by_seed[seed][row_index]
            truth_signatures.append(
                (
                    row["truth_ground_valid"],
                    tuple(
                        (
                            band["truth_clearance_valid"],
                            band["truth_clearance_m"],
                            tuple(cell["truth_state"] for cell in band["cells"]),
                        )
                        for band in row["bands"]
                    ),
                )
            )
        require(len(set(truth_signatures)) == 1, "CROSS_SEED_TRUTH_DRIFT", "truth differs across seed observations", row=row_index)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    return {
        "total": total,
        "counts": {state: counts[state] for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "fractions": {state: _safe_ratio(counts[state], total) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
    }


def _summary(values: list[float]) -> dict[str, Any]:
    require(bool(values), "METRIC_DENOMINATOR_ZERO", "required metric denominator is zero")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - index) + ordered[high] * (index - low)

    return {
        "support": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _linear_fit(xs: list[float], ys: list[float]) -> dict[str, Any]:
    require(len(xs) == len(ys) and len(xs) > 1, "LINEAR_FIT_DENOMINATOR_ZERO", "linear fit requires paired values")
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    require(x_ss > 0 and y_ss > 0, "LINEAR_FIT_VARIANCE_ZERO", "linear fit requires nonzero variance")
    slope = covariance / x_ss
    intercept = y_mean - slope * x_mean
    correlation = covariance / math.sqrt(x_ss * y_ss)
    return {"slope_predicted_on_truth": slope, "intercept_m": intercept, "pearson_r": correlation}


def boundary_bin(margin_m: float, bins: tuple[float, float, float]) -> str:
    if margin_m <= bins[0]:
        return "le_0.10m"
    if margin_m <= bins[1]:
        return "gt_0.10_le_0.25m"
    if margin_m <= bins[2]:
        return "gt_0.25_le_0.50m"
    return "gt_0.50m"


def _state_key(row: dict[str, Any], band: str, horizon: float) -> tuple[Any, ...]:
    return (*frame_identity(row), band, horizon)


def _clearance_key(row: dict[str, Any], band: str) -> tuple[Any, ...]:
    return (*frame_identity(row), band)


def compute_seed_anatomy(rows: list[dict[str, Any]], seed: int, bins: tuple[float, float, float], jitter_margin: float) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_states: list[str] = []
    predicted_states: list[str] = []
    paired_known = 0
    truth_clear = 0
    false_blocks = 0
    truth_clear_by_parent: Counter[str] = Counter()
    false_block_by_parent: Counter[str] = Counter()
    decomposition: Counter[str] = Counter()
    boundary_support: Counter[str] = Counter()
    boundary_false_blocks: Counter[str] = Counter()
    residuals: list[float] = []
    absolute_residuals: list[float] = []
    truth_clearances: list[float] = []
    predicted_clearances: list[float] = []
    negative_residuals = 0
    residual_by_key: dict[tuple[Any, ...], float] = {}
    prediction_by_key: dict[tuple[Any, ...], str] = {}
    false_block_by_key: dict[tuple[Any, ...], bool] = {}
    series: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        for band in row["bands"]:
            truth_clearance = band["truth_clearance_m"] if band["truth_clearance_valid"] else None
            predicted_clearance = band["predicted_clearance_m"] if band["predicted_clearance_valid"] else None
            if truth_clearance is not None and predicted_clearance is not None:
                residual = float(predicted_clearance) - float(truth_clearance)
                residuals.append(residual)
                absolute_residuals.append(abs(residual))
                truth_clearances.append(float(truth_clearance))
                predicted_clearances.append(float(predicted_clearance))
                negative_residuals += int(residual < 0)
                residual_by_key[_clearance_key(row, band["band"])] = residual
            for cell in band["cells"]:
                truth = cell["truth_state"]
                predicted = cell["predicted_state"]
                horizon = float(cell["horizon_m"])
                key = _state_key(row, band["band"], horizon)
                truth_states.append(truth)
                predicted_states.append(predicted)
                prediction_by_key[key] = predicted
                if truth != "UNKNOWN" and predicted != "UNKNOWN":
                    paired_known += 1
                is_false_block = truth == "CLEAR_OBSERVED" and predicted == "OCCUPIED_OBSERVED"
                if truth == "CLEAR_OBSERVED" and predicted != "UNKNOWN":
                    truth_clear += 1
                    truth_clear_by_parent[row["parent_id"]] += 1
                    if truth_clearance is not None:
                        margin = float(truth_clearance) - horizon
                        require(margin >= -1e-6, "TRUTH_STATE_CLEARANCE_INCONSISTENT", "truth CLEAR state contradicts truth clearance", seed=seed, key=key)
                        stratum = boundary_bin(max(0.0, margin), bins)
                        boundary_support[stratum] += 1
                        boundary_false_blocks[stratum] += int(is_false_block)
                if is_false_block:
                    false_blocks += 1
                    false_block_by_parent[row["parent_id"]] += 1
                    if not row["predicted_ground_valid"]:
                        source = "predicted_ground_invalid_state_inconsistency"
                    elif truth_clearance is not None and predicted_clearance is not None:
                        truth_implies_clear = float(truth_clearance) + 1e-6 >= horizon
                        prediction_implies_occupied = float(predicted_clearance) + 1e-6 < horizon
                        source = "clearance_threshold_crossing" if truth_implies_clear and prediction_implies_occupied else "state_clearance_aggregation_inconsistency"
                    else:
                        source = "clearance_unavailable_unresolved"
                    decomposition[source] += 1
                if truth == "CLEAR_OBSERVED" and predicted != "UNKNOWN":
                    false_block_by_key[key] = is_false_block
                series[(row["parent_id"], row["session_id"], band["band"], horizon)].append(
                    {
                        "key": key,
                        "sequence": row["sequence_index"],
                        "truth": truth,
                        "predicted": predicted,
                        "truth_clearance": truth_clearance,
                        "predicted_clearance": predicted_clearance,
                    }
                )

    transitions_total = 0
    transitions_agree = 0
    mismatch_categories: Counter[str] = Counter()
    mismatch_threshold_near_truth = 0
    mismatch_threshold_near_either = 0
    transition_failure_by_key: dict[tuple[Any, ...], bool] = {}
    for (parent, session, band, horizon), values in series.items():
        values.sort(key=lambda item: item["sequence"])
        for previous, current in zip(values, values[1:]):
            require(current["sequence"] == previous["sequence"] + 1, "TRANSITION_SEQUENCE_DRIFT", "transition sequence is not consecutive", seed=seed)
            if "UNKNOWN" in (previous["truth"], current["truth"], previous["predicted"], current["predicted"]):
                continue
            transitions_total += 1
            truth_pair = (previous["truth"], current["truth"])
            predicted_pair = (previous["predicted"], current["predicted"])
            agrees = truth_pair == predicted_pair
            transitions_agree += int(agrees)
            event_key = (parent, session, band, horizon, previous["sequence"], current["sequence"])
            transition_failure_by_key[event_key] = not agrees
            if agrees:
                continue
            truth_changes = truth_pair[0] != truth_pair[1]
            prediction_changes = predicted_pair[0] != predicted_pair[1]
            if not truth_changes and not prediction_changes:
                category = (
                    f"stable_truth_{truth_pair[0].lower()}_"
                    f"stable_predicted_{predicted_pair[0].lower()}"
                )
            elif not truth_changes and prediction_changes:
                category = "stable_truth_prediction_jitter"
            elif truth_changes and not prediction_changes:
                category = "truth_transition_missed"
            else:
                category = "truth_and_prediction_transition_disagree"
            mismatch_categories[category] += 1
            truth_margins = [
                abs(float(item["truth_clearance"]) - horizon)
                for item in (previous, current)
                if item["truth_clearance"] is not None
            ]
            predicted_margins = [
                abs(float(item["predicted_clearance"]) - horizon)
                for item in (previous, current)
                if item["predicted_clearance"] is not None
            ]
            threshold_near_truth = bool(truth_margins) and min(truth_margins) <= jitter_margin
            all_margins = truth_margins + predicted_margins
            threshold_near_either = bool(all_margins) and min(all_margins) <= jitter_margin
            mismatch_threshold_near_truth += int(threshold_near_truth)
            mismatch_threshold_near_either += int(threshold_near_either)

    require(truth_clear > 0 and false_blocks > 0, "FALSE_BLOCK_DENOMINATOR_ZERO", "failure anatomy requires observed false blocks", seed=seed)
    require(residuals and transitions_total > 0, "ANATOMY_DENOMINATOR_ZERO", "required anatomy denominator is zero", seed=seed)
    state_result = {
        "truth_all_cells": _distribution(truth_states),
        "prediction_all_cells": _distribution(predicted_states),
        "unknown_policy": "truth UNKNOWN excluded from error denominators; prediction UNKNOWN is abstention, never negative",
        "paired_known_cells": paired_known,
        "truth_clear_paired_prediction_known": truth_clear,
        "false_blocks": false_blocks,
        "false_block_given_clear": false_blocks / truth_clear,
        "truth_clear_support_by_parent": {
            parent: {
                "truth_clear": truth_clear_by_parent[parent],
                "false_blocks": false_block_by_parent[parent],
                "false_block_given_clear": _safe_ratio(false_block_by_parent[parent], truth_clear_by_parent[parent]),
            }
            for parent in sorted(set(truth_clear_by_parent) | set(false_block_by_parent))
        },
    }
    false_block_result = {
        "exclusive_source_decomposition": {
            name: {"count": count, "fraction_of_false_blocks": count / false_blocks}
            for name, count in sorted(decomposition.items())
        },
        "assistive_occupancy_or_task_head": "NOT_APPLICABLE_A0_ASSISTIVE_HEADS_NOT_READ",
        "depth_scale_causality": "NOT_IDENTIFIABLE_FROM_AGGREGATE_OBSERVATIONS; residual slope/bias are diagnostic evidence only",
        "boundary_margin_strata": {
            name: {
                "truth_clear_support": boundary_support[name],
                "false_blocks": boundary_false_blocks[name],
                "false_block_given_clear": _safe_ratio(boundary_false_blocks[name], boundary_support[name]),
            }
            for name in ("le_0.10m", "gt_0.10_le_0.25m", "gt_0.25_le_0.50m", "gt_0.50m")
        },
    }
    clearance_result = {
        "residual_definition": "predicted_clearance_m - truth_clearance_m on independently paired-valid bands",
        "signed_residual_m": _summary(residuals),
        "absolute_residual_m": _summary(absolute_residuals),
        "negative_residual_fraction": negative_residuals / len(residuals),
        "linear_fit": _linear_fit(truth_clearances, predicted_clearances),
    }
    transition_result = {
        "paired_known_consecutive_events": transitions_total,
        "agreement_count": transitions_agree,
        "agreement": transitions_agree / transitions_total,
        "failure_count": transitions_total - transitions_agree,
        "failure_categories": dict(sorted(mismatch_categories.items())),
        "threshold_near_truth_margin_m": jitter_margin,
        "failure_threshold_near_truth": mismatch_threshold_near_truth,
        "failure_threshold_near_either_truth_or_prediction": mismatch_threshold_near_either,
        "failure_threshold_far_from_truth_and_prediction": transitions_total - transitions_agree - mismatch_threshold_near_either,
        "threshold_near_truth_fraction_of_failures": _safe_ratio(mismatch_threshold_near_truth, transitions_total - transitions_agree),
        "threshold_near_either_fraction_of_failures": _safe_ratio(mismatch_threshold_near_either, transitions_total - transitions_agree),
        "stable_truth_clear_stable_predicted_occupied_fraction": _safe_ratio(
            mismatch_categories["stable_truth_clear_observed_stable_predicted_occupied_observed"],
            transitions_total - transitions_agree,
        ),
        "interpretation_rule": "stable truth-clear/stable predicted-occupied and failures far from both truth/prediction thresholds are persistent geometry error, not prediction flip/jitter",
    }
    internal = {
        "prediction_by_key": prediction_by_key,
        "false_block_by_key": false_block_by_key,
        "transition_failure_by_key": transition_failure_by_key,
        "residual_by_key": residual_by_key,
    }
    return {
        "seed": seed,
        "state_distribution": state_result,
        "false_block_anatomy": false_block_result,
        "clearance_anatomy": clearance_result,
        "transition_anatomy": transition_result,
    }, internal


def _binary_similarity(left: dict[Any, bool], right: dict[Any, bool]) -> dict[str, Any]:
    keys = sorted(set(left) & set(right), key=repr)
    require(bool(keys), "SIMILARITY_DENOMINATOR_ZERO", "cross-seed binary similarity denominator is zero")
    exact = sum(left[key] == right[key] for key in keys)
    intersection = sum(left[key] and right[key] for key in keys)
    union = sum(left[key] or right[key] for key in keys)
    return {
        "common_support": len(keys),
        "exact_mask_agreement": exact / len(keys),
        "positive_intersection": intersection,
        "positive_union": union,
        "positive_jaccard": intersection / union if union else None,
    }


def _correlation_by_key(left: dict[Any, float], right: dict[Any, float]) -> dict[str, Any]:
    keys = sorted(set(left) & set(right), key=repr)
    require(len(keys) > 1, "RESIDUAL_CORRELATION_DENOMINATOR_ZERO", "cross-seed residual correlation requires paired support")
    xs = [left[key] for key in keys]
    ys = [right[key] for key in keys]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    require(x_ss > 0 and y_ss > 0, "RESIDUAL_CORRELATION_VARIANCE_ZERO", "cross-seed residual correlation variance is zero")
    return {"common_support": len(keys), "pearson_r": covariance / math.sqrt(x_ss * y_ss)}


def cross_seed_similarity(internals: dict[int, dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for left, right in ((17, 29), (17, 43), (29, 43)):
        left_data = internals[left]
        right_data = internals[right]
        common_state_keys = sorted(set(left_data["prediction_by_key"]) & set(right_data["prediction_by_key"]), key=repr)
        require(bool(common_state_keys), "STATE_SIMILARITY_DENOMINATOR_ZERO", "cross-seed state similarity denominator is zero")
        state_agreement = sum(left_data["prediction_by_key"][key] == right_data["prediction_by_key"][key] for key in common_state_keys) / len(common_state_keys)
        pairs[f"{left}_vs_{right}"] = {
            "prediction_state_exact_agreement": {"support": len(common_state_keys), "value": state_agreement},
            "false_block_mask": _binary_similarity(left_data["false_block_by_key"], right_data["false_block_by_key"]),
            "transition_failure_mask": _binary_similarity(left_data["transition_failure_by_key"], right_data["transition_failure_by_key"]),
            "clearance_residual": _correlation_by_key(left_data["residual_by_key"], right_data["residual_by_key"]),
        }
    common_false_block_keys = set.intersection(*(set(internals[seed]["false_block_by_key"]) for seed in EXPECTED_SEEDS))
    all_three = sum(all(internals[seed]["false_block_by_key"][key] for seed in EXPECTED_SEEDS) for key in common_false_block_keys)
    any_three = sum(any(internals[seed]["false_block_by_key"][key] for seed in EXPECTED_SEEDS) for key in common_false_block_keys)
    return {
        "selected_seed": None,
        "best_seed_selection_forbidden": True,
        "pairwise": pairs,
        "all_three_false_block_overlap": {
            "common_truth_known_support": len(common_false_block_keys),
            "intersection": all_three,
            "union": any_three,
            "jaccard": all_three / any_three if any_three else None,
        },
    }


def analyze(rows_by_seed: dict[int, list[dict[str, Any]]], protocol: dict[str, Any]) -> dict[str, Any]:
    expected_parents = tuple(protocol["expected_parent_order"])
    validate_cross_seed_order(rows_by_seed, expected_parents)
    bins = tuple(float(value) for value in protocol["analysis"]["boundary_margin_bins_m"])
    jitter_margin = float(protocol["analysis"]["transition_jitter_truth_margin_m"])
    seed_results: list[dict[str, Any]] = []
    internals: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        result, internal = compute_seed_anatomy(rows_by_seed[seed], seed, bins, jitter_margin)
        seed_results.append(result)
        internals[seed] = internal
    truth_clear_support = {
        parent: seed_results[0]["state_distribution"]["truth_clear_support_by_parent"].get(
            parent, {"truth_clear": 0, "false_blocks": 0, "false_block_given_clear": None}
        )["truth_clear"]
        for parent in expected_parents
    }
    total_truth_clear = sum(truth_clear_support.values())
    concentrated_parent, concentrated_support = max(truth_clear_support.items(), key=lambda item: item[1])
    return {
        "seed_results": seed_results,
        "support_concentration": {
            "truth_clear_support_by_parent": truth_clear_support,
            "total_truth_clear_support_per_seed": total_truth_clear,
            "largest_support_parent": concentrated_parent,
            "largest_support_fraction": _safe_ratio(concentrated_support, total_truth_clear),
            "claim_boundary": "parent concentration limits scene-general claims even when cross-seed failure patterns agree",
        },
        "cross_seed_similarity": cross_seed_similarity(internals),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# B1-A0 failure anatomy (diagnostic only)",
        "",
        f"Terminal: `{result['terminal']}`",
        "",
        "This post-mortem uses only already-consumed `DEVELOPMENT_SELECTION` observations. It is permanently `NOT_ELIGIBLE_FOR_PROMOTION`; Calibration and Confirmation remain sealed.",
        "",
        "## Seed-preserving findings",
        "",
        "| Seed | False-block | Signed clearance bias | Negative residual | Transition agreement | Transition failures near threshold |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_result in result["analysis"]["seed_results"]:
        state = seed_result["state_distribution"]
        clearance = seed_result["clearance_anatomy"]
        transition = seed_result["transition_anatomy"]
        lines.append(
            f"| {seed_result['seed']} | {state['false_blocks']}/{state['truth_clear_paired_prediction_known']} ({state['false_block_given_clear']:.3%}) | "
            f"{clearance['signed_residual_m']['mean']:+.3f} m | {clearance['negative_residual_fraction']:.3%} | "
            f"{transition['agreement']:.3%} | {transition['failure_threshold_near_either_truth_or_prediction']}/{transition['failure_count']} |"
        )
    support = result["analysis"]["support_concentration"]
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"Truth-clear support is concentrated in parent `{support['largest_support_parent']}` at {support['largest_support_fraction']:.3%} of the per-seed denominator. This supports a repeatable failure on that observed parent, not an all-scene prevalence claim.",
            "",
            "A0 reads dense depth followed by the frozen gravity/geometry reader; it does not read an assistive occupancy/task head. Accordingly, occupancy-head collapse is not an admissible explanation. Signed clearance bias, threshold crossings, boundary strata and cross-seed correlations are diagnostic associations, not causal identification of depth scale or ground estimation.",
            "",
            "UNKNOWN truth is excluded from all error denominators and is never treated as negative. No seed is selected, no threshold is changed, and this result authorizes no training, promotion, deployment, teacher, temporal, product or safety claim.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(protocol_path: Path, package_path: Path, output_root: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(protocol_path)
    package_binding = protocol["bindings"]["observation_package"]
    expected_package_path = resolve_reference(package_binding["path"])
    require(package_path.resolve() == expected_package_path, "PACKAGE_PATH_DRIFT", "CLI package path differs from frozen binding")
    package_sha = _validate_sha_binding(package_path, package_binding, "OBSERVATION_PACKAGE")
    package = load_json(package_path)
    runs = validate_package_metadata(package, protocol)
    governed, governed_path, governed_sha = _validate_governed_development_result(protocol)
    governed_package = governed["execution_receipts"]["observation_package"]
    require(governed_package.get("sha256") == package_sha, "GOVERNED_PACKAGE_SHA_DRIFT", "governed result/package SHA mismatch")
    _validate_training_receipts(runs, governed)
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    observation_receipts: dict[str, Any] = {}
    for run in runs:
        seed = int(run["seed"])
        path = resolve_reference(run.get("observations_path"), base=package_path.parent, code="OBSERVATIONS_PATH_INVALID")
        require(path.is_file(), "OBSERVATIONS_MISSING", "bound observations are missing", seed=seed)
        digest = sha256_file(path)
        require(digest == run.get("observations_sha256"), "OBSERVATIONS_SHA_DRIFT", "observation SHA drift", seed=seed)
        rows_by_seed[seed] = load_observation_rows(path, seed)
        observation_receipts[str(seed)] = {"path": str(path), "sha256": digest, "rows": len(rows_by_seed[seed])}
    analysis = analyze(rows_by_seed, protocol)
    result = {
        "schema": RESULT_SCHEMA,
        "protocol": {"path": str(protocol_path.resolve()), "sha256": protocol_sha},
        "terminal": TERMINAL,
        "status": "DIAGNOSTIC_COMPLETE",
        "eligibility": "NOT_ELIGIBLE_FOR_PROMOTION",
        "data_role": ROLE,
        "input_receipts": {
            "observation_package": {"path": str(package_path.resolve()), "sha256": package_sha},
            "governed_development_result": {"path": str(governed_path), "sha256": governed_sha},
            "observations_by_seed": observation_receipts,
        },
        "analysis": analysis,
        "mechanism_answers": {
            "false_block_source": "Observed false blocks are decomposed into deterministic clearance-threshold crossings versus consistency/unresolved cases; depth/ground causality is not identifiable from these aggregate observations.",
            "occupancy_head_collapse": "NOT_APPLICABLE_A0_ASSISTIVE_HEADS_NOT_READ",
            "clearance_error": "Signed residual, linear fit and task-horizon boundary strata are reported without causal promotion claims.",
            "transition_failure": "Stable wrong geometry, prediction jitter, missed truth transition and other disagreement are separated; threshold-near truth margins are reported.",
            "cross_seed_failure": "All three seeds are retained with pairwise masks/correlations and no winner.",
        },
        "data_firewall": {
            "development_selection_previously_consumed": True,
            "development_calibration_content_opened": False,
            "confirmation_content_opened": False,
        },
        "authority": {
            "diagnostic_only": True,
            "promotion": False,
            "training": False,
            "seed_selection": False,
            "threshold_change": False,
            "deployment": False,
            "teacher_execution": False,
            "temporal_execution": False,
            "product": False,
            "safety": False,
        },
        "claim_ceiling": protocol["claim_ceiling"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "failure_anatomy_result.json", result)
    atomic_write_text(output_root / "report.md", render_report(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.protocol, args.package, args.output_root)
    except Exception as error:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "blindassist_assistive_geometry_b1_a0_failure_anatomy_failure_v1",
            "terminal": FAILURE_TERMINAL,
            "status": "NOT_EVALUABLE",
            "eligibility": "NOT_ELIGIBLE_FOR_PROMOTION",
            "error_type": type(error).__name__,
            "error": str(error),
            "code": getattr(error, "code", "INTERNAL_FAILURE"),
            "context": getattr(error, "context", {}),
            "development_calibration_content_opened": False,
            "confirmation_content_opened": False,
        }
        atomic_write_json(args.output_root / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps({"terminal": TERMINAL, "output_root": str(args.output_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
