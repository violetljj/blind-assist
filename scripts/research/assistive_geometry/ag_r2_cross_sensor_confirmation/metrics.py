"""Frozen factor-only metrics for the AG R2 cross-sensor confirmation.

This module deliberately knows nothing about archives, models, reducers, task
states, or evidence writers.  It accepts already sealed factor and source
arrays, validates their UNKNOWN semantics, and evaluates the 27 frozen gates.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contract import ContractError, canonical_sha256, require

SUMMARY_SCHEMA = "blindassist.ag.r2.cross_sensor_factor_confirmation_summary.v1"
PARENT_IDS = ("plant_scene_2", "motion_1", "mannequin_5")
FAMILIES = ("depth", "support", "boundary")
SPEARMAN_STRATA = 10


def _float64(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    require(array.dtype == np.dtype("float64"), f"F2_{name.upper()}_DTYPE")
    require(array.ndim == 2, f"F2_{name.upper()}_RANK")
    if shape is not None:
        require(array.shape == shape, f"F2_{name.upper()}_SHAPE")
    return array


def _boolean(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    require(array.dtype == np.dtype("bool"), f"F2_{name.upper()}_DTYPE")
    require(array.ndim == 2, f"F2_{name.upper()}_RANK")
    if shape is not None:
        require(array.shape == shape, f"F2_{name.upper()}_SHAPE")
    return array


def _known_float(
    value: Any,
    known: np.ndarray,
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    positive: bool = False,
) -> np.ndarray:
    array = _float64(value, name, known.shape)
    require(bool(np.all(np.isfinite(array[known]))), f"F2_{name.upper()}_KNOWN_NONFINITE")
    require(bool(np.all(np.isnan(array[~known]))), f"F2_{name.upper()}_UNKNOWN_NOT_NAN")
    if positive:
        require(bool(np.all(array[known] > 0.0)), f"F2_{name.upper()}_NOT_POSITIVE")
    if lower is not None:
        require(bool(np.all(array[known] >= lower)), f"F2_{name.upper()}_BELOW_RANGE")
    if upper is not None:
        require(bool(np.all(array[known] <= upper)), f"F2_{name.upper()}_ABOVE_RANGE")
    return array


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    require(set(value) == expected, code)


def validate_frame(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one score frame and return canonical float64/bool arrays."""

    _require_exact_keys(
        row,
        {"parent_id", "frame_id", "fx", "fy", "prediction", "truth"},
        "F2_SCORE_FRAME_KEY_SET",
    )
    parent_id = row["parent_id"]
    frame_id = row["frame_id"]
    require(isinstance(parent_id, str) and parent_id in PARENT_IDS, "F2_SCORE_PARENT_ID")
    require(isinstance(frame_id, str) and frame_id != "", "F2_SCORE_FRAME_ID")
    require(type(row["fx"]) is float and math.isfinite(row["fx"]) and row["fx"] > 0.0, "F2_SCORE_FX")
    require(type(row["fy"]) is float and math.isfinite(row["fy"]) and row["fy"] > 0.0, "F2_SCORE_FY")
    prediction = row["prediction"]
    truth = row["truth"]
    require(isinstance(prediction, Mapping) and isinstance(truth, Mapping), "F2_SCORE_PAYLOAD_TYPE")
    _require_exact_keys(
        prediction,
        {
            "depth_m", "depth_log_sigma", "depth_known",
            "support_probability", "support_residual_sigma_m", "support_known",
            "obstacle_probability", "boundary_distance_px", "boundary_sigma_px",
            "evidence_known",
        },
        "F2_PREDICTION_KEY_SET",
    )
    _require_exact_keys(
        truth,
        {
            "depth_m", "depth_known", "support_probability",
            "support_signed_residual_m", "support_known", "obstacle_probability",
            "boundary_distance_px", "evidence_known",
        },
        "F2_TRUTH_KEY_SET",
    )

    depth_known = _boolean(truth["depth_known"], "truth_depth_known")
    shape = depth_known.shape
    support_known = _boolean(truth["support_known"], "truth_support_known", shape)
    evidence_known = _boolean(truth["evidence_known"], "truth_evidence_known", shape)
    pred_depth_known = _boolean(prediction["depth_known"], "prediction_depth_known", shape)
    pred_support_known = _boolean(prediction["support_known"], "prediction_support_known", shape)
    pred_evidence_known = _boolean(prediction["evidence_known"], "prediction_evidence_known", shape)

    normalized = {
        "parent_id": parent_id,
        "frame_id": frame_id,
        "fx": float(row["fx"]),
        "fy": float(row["fy"]),
        "prediction": {
            "depth_known": pred_depth_known,
            "depth_m": _known_float(prediction["depth_m"], pred_depth_known, "prediction_depth_m", positive=True),
            "depth_log_sigma": _known_float(
                prediction["depth_log_sigma"], pred_depth_known, "prediction_depth_log_sigma", positive=True
            ),
            "support_known": pred_support_known,
            "support_probability": _known_float(
                prediction["support_probability"], pred_support_known, "prediction_support_probability", lower=0.0, upper=1.0
            ),
            "support_residual_sigma_m": _known_float(
                prediction["support_residual_sigma_m"], pred_support_known, "prediction_support_residual_sigma_m", positive=True
            ),
            "evidence_known": pred_evidence_known,
            "obstacle_probability": _known_float(
                prediction["obstacle_probability"], pred_evidence_known, "prediction_obstacle_probability", lower=0.0, upper=1.0
            ),
            "boundary_distance_px": _known_float(
                prediction["boundary_distance_px"], pred_evidence_known, "prediction_boundary_distance_px", lower=0.0, upper=32.0
            ),
            "boundary_sigma_px": _known_float(
                prediction["boundary_sigma_px"], pred_evidence_known, "prediction_boundary_sigma_px", positive=True
            ),
        },
        "truth": {
            "depth_known": depth_known,
            "depth_m": _known_float(truth["depth_m"], depth_known, "truth_depth_m", positive=True),
            "support_known": support_known,
            "support_probability": _known_float(
                truth["support_probability"], support_known, "truth_support_probability", lower=0.0, upper=1.0
            ),
            "support_signed_residual_m": _known_float(
                truth["support_signed_residual_m"], support_known, "truth_support_signed_residual_m"
            ),
            "evidence_known": evidence_known,
            "obstacle_probability": _known_float(
                truth["obstacle_probability"], evidence_known, "truth_obstacle_probability", lower=0.0, upper=1.0
            ),
            "boundary_distance_px": _known_float(
                truth["boundary_distance_px"], evidence_known, "truth_boundary_distance_px", lower=0.0, upper=32.0
            ),
        },
    }
    require(shape[0] > 0 and shape[1] > 0, "F2_SCORE_EMPTY_ARRAY")
    return normalized


def _mean(values: Sequence[float], code: str) -> float:
    require(bool(values), code)
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    require(math.isfinite(result), code)
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def _stratified_spearman(sigma: np.ndarray, residual: np.ndarray, identities: np.ndarray, code: str) -> float:
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    identities = np.asarray(identities).reshape(-1)
    require(sigma.size == residual.size == identities.size and sigma.size >= 2 * SPEARMAN_STRATA, code)
    require(bool(np.all(np.isfinite(sigma))) and bool(np.all(np.isfinite(residual))), code)
    order = np.lexsort((identities.astype(str), sigma))
    groups = np.array_split(order, SPEARMAN_STRATA)
    sigma_mean = np.asarray([np.mean(sigma[group]) for group in groups], dtype=np.float64)
    residual_mean = np.asarray([np.mean(residual[group]) for group in groups], dtype=np.float64)
    sigma_rank = _average_ranks(sigma_mean)
    residual_rank = _average_ranks(residual_mean)
    sigma_centered = sigma_rank - sigma_rank.mean()
    residual_centered = residual_rank - residual_rank.mean()
    denominator = float(np.sqrt(np.sum(sigma_centered**2) * np.sum(residual_centered**2)))
    require(denominator > 0.0 and math.isfinite(denominator), code)
    result = float(np.sum(sigma_centered * residual_centered) / denominator)
    require(math.isfinite(result), code)
    return result


def _gate_specs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    constraints = protocol.get("constraints")
    require(isinstance(constraints, list), "F2_GATE_SPECS_MISSING")
    rows = [row for row in constraints if isinstance(row, Mapping) and row.get("class") == "GATE"]
    require(len(rows) == 27, "F2_GATE_SPECS_COUNT")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        require(
            set(row).issuperset({"id", "metric", "operator", "threshold", "unit"})
            and str(row["id"]).startswith(f"AG-R2-XSR-G{index:02d}-")
            and row["operator"] in {"EQ", "GTE", "LTE", "RANGE_INCLUSIVE"}
            and (
                type(row["threshold"]) in {int, float}
                or (
                    row["operator"] == "RANGE_INCLUSIVE"
                    and isinstance(row["threshold"], list)
                    and len(row["threshold"]) == 2
                    and all(type(value) in {int, float} for value in row["threshold"])
                )
            ),
            "F2_GATE_SPEC_DRIFT",
        )
        result.append({key: row[key] for key in ("id", "metric", "operator", "threshold", "unit")})
    return result


def _evaluate_gate(spec: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    name = str(spec["metric"])
    require(name in metrics, "F2_GATE_METRIC_MISSING", name)
    value = metrics[name]
    if spec["operator"] == "RANGE_INCLUSIVE":
        require(isinstance(value, Mapping) and set(value) == set(PARENT_IDS), "F2_GATE_METRIC_UNDEFINED", name)
        require(all(type(item) is float and math.isfinite(item) for item in value.values()), "F2_GATE_METRIC_UNDEFINED", name)
        lower, upper = (float(item) for item in spec["threshold"])
        passed = all(lower <= item <= upper for item in value.values())
        return {**dict(spec), "value": dict(value), "passed": bool(passed)}
    require(type(value) in {int, float} and math.isfinite(float(value)), "F2_GATE_METRIC_UNDEFINED", name)
    threshold = float(spec["threshold"])
    if spec["operator"] == "EQ":
        passed = float(value) == threshold
    elif spec["operator"] == "GTE":
        passed = float(value) >= threshold
    else:
        passed = float(value) <= threshold
    return {**dict(spec), "value": value, "passed": bool(passed)}


def score(
    protocol: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    score_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the complete frozen summary from sealed frame arrays."""

    _require_exact_keys(source_summary, {"parents"}, "F2_SOURCE_SUMMARY_KEY_SET")
    parents = source_summary["parents"]
    require(isinstance(parents, list) and len(parents) == 3, "F2_SOURCE_PARENT_ROWS")
    source_by_parent: dict[str, Mapping[str, Any]] = {}
    expected_source_keys = {
        "parent_id", "eligible_pair_count", "calibration_count", "score_count",
        "camera_height_m", "camera_height_mad_m", "source_depth_known_coverage",
        "source_support_known_coverage", "source_boundary_known_coverage",
    }
    for row in parents:
        require(isinstance(row, Mapping), "F2_SOURCE_PARENT_TYPE")
        _require_exact_keys(row, expected_source_keys, "F2_SOURCE_PARENT_KEY_SET")
        parent_id = row["parent_id"]
        require(parent_id in PARENT_IDS and parent_id not in source_by_parent, "F2_SOURCE_PARENT_ID")
        for name in ("eligible_pair_count", "calibration_count", "score_count"):
            require(type(row[name]) is int and row[name] >= 0, f"F2_SOURCE_{name.upper()}")
        for name in (
            "camera_height_m", "camera_height_mad_m", "source_depth_known_coverage",
            "source_support_known_coverage", "source_boundary_known_coverage",
        ):
            require(type(row[name]) is float and math.isfinite(row[name]), f"F2_SOURCE_{name.upper()}")
        source_by_parent[parent_id] = row
    require(tuple(source_by_parent) == PARENT_IDS, "F2_SOURCE_PARENT_ORDER")

    frames = [validate_frame(row) for row in score_frames]
    keys = [(row["parent_id"], row["frame_id"]) for row in frames]
    require(len(keys) == len(set(keys)) == 36, "F2_SCORE_FRAME_IDENTITY_COUNT")
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        by_parent[row["parent_id"]].append(row)
    require(all(len(by_parent[parent]) == 12 for parent in PARENT_IDS), "F2_SCORE_PARENT_FRAME_COUNT")

    parent_metrics: dict[str, dict[str, float]] = {}
    uncertainty_detail: dict[str, dict[str, dict[str, float]]] = {}
    for parent_id in PARENT_IDS:
        depth_coverage: list[float] = []
        support_coverage: list[float] = []
        evidence_coverage: list[float] = []
        combined: list[float] = []
        shape_errors: list[float] = []
        scale_errors: list[float] = []
        support_brier: list[float] = []
        obstacle_brier: list[float] = []
        boundary_angle: list[float] = []
        uncertainty: dict[str, dict[str, list[np.ndarray]]] = {
            family: {"sigma": [], "residual": [], "identity": []} for family in FAMILIES
        }
        for frame in sorted(by_parent[parent_id], key=lambda value: value["frame_id"]):
            pred = frame["prediction"]
            truth = frame["truth"]
            masks = {
                "depth": truth["depth_known"] & pred["depth_known"],
                "support": truth["support_known"] & pred["support_known"],
                "boundary": truth["evidence_known"] & pred["evidence_known"],
            }
            for family, pred_known, truth_known, target in (
                ("depth", pred["depth_known"], truth["depth_known"], depth_coverage),
                ("support", pred["support_known"], truth["support_known"], support_coverage),
                ("boundary", pred["evidence_known"], truth["evidence_known"], evidence_coverage),
            ):
                denominator = int(np.sum(truth_known))
                require(denominator > 0, f"F2_{family.upper()}_TRUTH_DENOMINATOR_ZERO")
                target.append(float(np.sum(pred_known & truth_known) / denominator))
                require(bool(np.any(masks[family])), f"F2_{family.upper()}_INTERSECTION_ZERO")

            depth_mask = masks["depth"]
            signed_log = np.log(pred["depth_m"][depth_mask]) - np.log(truth["depth_m"][depth_mask])
            frame_scale = float(np.median(signed_log))
            combined.append(float(np.mean(np.abs(signed_log))))
            shape_errors.append(float(np.mean(np.abs(signed_log - frame_scale))))
            scale_errors.append(abs(frame_scale))

            support_mask = masks["support"]
            support_brier.append(float(np.mean((pred["support_probability"][support_mask] - truth["support_probability"][support_mask]) ** 2)))
            evidence_mask = masks["boundary"]
            obstacle_brier.append(float(np.mean((pred["obstacle_probability"][evidence_mask] - truth["obstacle_probability"][evidence_mask]) ** 2)))
            focal = math.sqrt(frame["fx"] * frame["fy"])
            boundary_residual = np.arctan(
                np.abs(pred["boundary_distance_px"][evidence_mask] - truth["boundary_distance_px"][evidence_mask]) / focal
            )
            boundary_angle.append(float(np.mean(boundary_residual)))

            values = {
                "depth": (pred["depth_log_sigma"][depth_mask], np.abs(signed_log)),
                "support": (
                    pred["support_residual_sigma_m"][support_mask],
                    np.abs(truth["support_signed_residual_m"][support_mask]),
                ),
                "boundary": (
                    np.arctan(pred["boundary_sigma_px"][evidence_mask] / focal),
                    boundary_residual,
                ),
            }
            for family, (sigma, residual) in values.items():
                uncertainty[family]["sigma"].append(np.asarray(sigma, dtype=np.float64))
                uncertainty[family]["residual"].append(np.asarray(residual, dtype=np.float64))
                uncertainty[family]["identity"].append(
                    np.asarray([f'{frame["frame_id"]}:{index:09d}' for index in range(sigma.size)])
                )

        parent_metrics[parent_id] = {
            "metric_prediction_known_coverage": _mean(depth_coverage, "F2_DEPTH_COVERAGE_UNDEFINED"),
            "support_prediction_known_coverage": _mean(support_coverage, "F2_SUPPORT_COVERAGE_UNDEFINED"),
            "obstacle_boundary_prediction_known_coverage": _mean(evidence_coverage, "F2_EVIDENCE_COVERAGE_UNDEFINED"),
            "depth_combined_abs_log_error": _mean(combined, "F2_DEPTH_COMBINED_UNDEFINED"),
            "depth_shape_abs_log_error": _mean(shape_errors, "F2_DEPTH_SHAPE_UNDEFINED"),
            "depth_scale_abs_log_error": _mean(scale_errors, "F2_DEPTH_SCALE_UNDEFINED"),
            "support_brier": _mean(support_brier, "F2_SUPPORT_BRIER_UNDEFINED"),
            "obstacle_brier": _mean(obstacle_brier, "F2_OBSTACLE_BRIER_UNDEFINED"),
            "boundary_camera_angular_error_rad": _mean(boundary_angle, "F2_BOUNDARY_ERROR_UNDEFINED"),
        }
        uncertainty_detail[parent_id] = {}
        for family in FAMILIES:
            sigma = np.concatenate(uncertainty[family]["sigma"])
            residual = np.concatenate(uncertainty[family]["residual"])
            identity = np.concatenate(uncertainty[family]["identity"])
            uncertainty_detail[parent_id][family] = {
                "one_sigma_coverage": float(np.mean(residual <= sigma)),
                "two_sigma_coverage": float(np.mean(residual <= 2.0 * sigma)),
                "spearman_sigma_residual": _stratified_spearman(
                    sigma, residual, identity, f"F2_{parent_id.upper()}_{family.upper()}_SPEARMAN_UNDEFINED"
                ),
                "sample_count": int(sigma.size),
            }

    metrics: dict[str, Any] = {
        "confirmation_parent_count": len(source_by_parent),
        "calibration_identity_count_per_parent": min(int(row["calibration_count"]) for row in parents),
        "score_identity_count_per_parent": min(int(row["score_count"]) for row in parents),
        "minimum_metadata_eligible_pairs_across_parents": min(int(row["eligible_pair_count"]) for row in parents),
        "session_camera_height_m_each_parent": {
            parent: float(source_by_parent[parent]["camera_height_m"]) for parent in PARENT_IDS
        },
        "worst_parent_session_height_mad_m": max(float(row["camera_height_mad_m"]) for row in parents),
        "minimum_parent_source_metric_depth_known_fraction": min(float(row["source_depth_known_coverage"]) for row in parents),
        "minimum_parent_source_support_known_fraction": min(float(row["source_support_known_coverage"]) for row in parents),
        "minimum_parent_source_boundary_evidence_known_fraction": min(float(row["source_boundary_known_coverage"]) for row in parents),
    }
    metric_names = {
        "metric_prediction_known_coverage": ("minimum_parent_metric_prediction_known_coverage", min),
        "support_prediction_known_coverage": ("minimum_parent_support_prediction_known_coverage", min),
        "obstacle_boundary_prediction_known_coverage": ("minimum_parent_min_obstacle_and_boundary_prediction_known_coverage", min),
        "depth_combined_abs_log_error": ("parent_macro_depth_absolute_log_error", np.mean),
        "depth_shape_abs_log_error": ("parent_macro_depth_shape_absolute_log_error", np.mean),
        "depth_scale_abs_log_error": ("parent_macro_depth_scale_absolute_log_error", np.mean),
        "support_brier": ("parent_macro_support_brier", np.mean),
        "obstacle_brier": ("parent_macro_obstacle_evidence_brier", np.mean),
        "boundary_camera_angular_error_rad": ("parent_macro_boundary_camera_angular_error_rad", np.mean),
    }
    for source_name, (target_name, reducer) in metric_names.items():
        values = [parent_metrics[parent][source_name] for parent in PARENT_IDS]
        metrics[target_name] = float(reducer(values))
    for source_name, target_name in (
        ("depth_combined_abs_log_error", "worst_parent_depth_absolute_log_error"),
        ("depth_shape_abs_log_error", "worst_parent_depth_shape_absolute_log_error"),
        ("depth_scale_abs_log_error", "worst_parent_depth_scale_absolute_log_error"),
        ("support_brier", "worst_parent_support_brier"),
        ("obstacle_brier", "worst_parent_obstacle_evidence_brier"),
        ("boundary_camera_angular_error_rad", "worst_parent_boundary_camera_angular_error_rad"),
    ):
        metrics[target_name] = max(parent_metrics[parent][source_name] for parent in PARENT_IDS)

    family_summary: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        family_summary[family] = {
            "parent_macro_one_sigma_coverage": _mean(
                [uncertainty_detail[parent][family]["one_sigma_coverage"] for parent in PARENT_IDS],
                "F2_ONE_SIGMA_UNDEFINED",
            ),
            "parent_macro_two_sigma_coverage": _mean(
                [uncertainty_detail[parent][family]["two_sigma_coverage"] for parent in PARENT_IDS],
                "F2_TWO_SIGMA_UNDEFINED",
            ),
            "parent_macro_spearman_sigma_residual": _mean(
                [uncertainty_detail[parent][family]["spearman_sigma_residual"] for parent in PARENT_IDS],
                "F2_SPEARMAN_UNDEFINED",
            ),
        }
    metrics["maximum_factor_family_abs_empirical_one_sigma_coverage_minus_0_6827"] = max(
        abs(row["parent_macro_one_sigma_coverage"] - 0.6827) for row in family_summary.values()
    )
    metrics["maximum_factor_family_abs_empirical_two_sigma_coverage_minus_0_9545"] = max(
        abs(row["parent_macro_two_sigma_coverage"] - 0.9545) for row in family_summary.values()
    )
    metrics["minimum_factor_family_parent_macro_spearman_sigma_residual"] = min(
        row["parent_macro_spearman_sigma_residual"] for row in family_summary.values()
    )

    gates = [_evaluate_gate(spec, metrics) for spec in _gate_specs(protocol)]
    source_passed = all(row["passed"] for row in gates[:9])
    model_passed = all(row["passed"] for row in gates[9:])
    terminal = "CONFIRM_PASS" if source_passed and model_passed else ("CONFIRM_FAIL" if source_passed else "NOT_EVALUABLE")
    result = {
        "schema": SUMMARY_SCHEMA,
        "terminal": terminal,
        "source_evaluable": source_passed,
        "all_model_gates_passed": model_passed if source_passed else False,
        "spearman_method": {
            "strata": SPEARMAN_STRATA,
            "sort": "sigma_then_frame_id_then_flat_index",
            "ties": "average_rank",
            "undefined": "NOT_EVALUABLE",
        },
        "metrics": metrics,
        "parents": parent_metrics,
        "uncertainty": {"parents": uncertainty_detail, "families": family_summary},
        "gates": gates,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def score_or_not_evaluable(
    protocol: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    score_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert metric undefinedness to the frozen scientific terminal.

    Structural or identity errors remain exceptions and close the evidence
    version as invalid.  Only a required denominator or Spearman undefinedness
    is a scientific NOT_EVALUABLE terminal.
    """

    try:
        return score(protocol, source_summary, score_frames)
    except ContractError as error:
        if any(token in error.code for token in ("DENOMINATOR", "INTERSECTION_ZERO", "SPEARMAN_UNDEFINED")):
            result = {
                "schema": SUMMARY_SCHEMA,
                "terminal": "NOT_EVALUABLE",
                "source_evaluable": False,
                "all_model_gates_passed": False,
                "reason_code": error.code,
                "metrics": {},
                "parents": {},
                "uncertainty": {},
                "gates": [],
            }
            result["content_sha256"] = canonical_sha256(result)
            return result
        raise
