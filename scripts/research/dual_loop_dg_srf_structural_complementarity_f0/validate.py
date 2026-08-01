"""Independent recomputation validator for DG-SRF F0.

This module intentionally does not import the producer, evaluator, structural
operators, or any other DG-SRF implementation module.  It starts from the
frozen source ledgers, raw depth maps, canonical truth, and packed A/B masks,
then independently rebuilds every claim-bearing quantity.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


PROTOCOL_ID = "DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0"
CONFIG_SCHEMA = (
    "blindassist.dg_srf_image_space_structural_complementarity_f0.config.v1"
)
VALIDATION_SCHEMA = (
    "blindassist.dg_srf_image_space_structural_"
    "complementarity_f0.independent_validation.v1"
)
SHAPE = (256, 256)
ARMS = ("D1", "D2", "D3", "D4", "D5")
SINGLE_ARMS = ("D1", "D2", "D3")
EXPECTED_PARAMETER_COUNT = 24_785_089
EXPECTED_STATE_TENSOR_COUNT = 239
FLOAT_ABS_TOLERANCE = 1e-9
FLOAT_REL_TOLERANCE = 1e-9


class ValidationError(RuntimeError):
    """A frozen identity, numerical result, or scientific contract drifted."""


class NotEvaluableError(ValidationError):
    """The frozen estimand cannot be evaluated from the supplied evidence."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json_bytes(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    )
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"expected object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _require_artifact_output(repo_root: Path, output: Path) -> Path:
    artifact_root = (repo_root / "artifacts.local").resolve()
    resolved = output.resolve()
    if resolved == artifact_root or artifact_root not in resolved.parents:
        raise ValidationError(
            f"validation output must stay below artifacts.local: {resolved}"
        )
    return resolved


def _verify_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_bytes: int | None = None,
) -> None:
    if not path.is_file():
        raise ValidationError(f"required file is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValidationError(
            f"byte count mismatch for {path}: "
            f"{path.stat().st_size} != {expected_bytes}"
        )
    actual = _sha256_file(path)
    if actual != str(expected_sha256):
        raise ValidationError(
            f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}"
        )


def _assert_equivalent(
    actual: Any,
    expected: Any,
    label: str,
    *,
    abs_tolerance: float = FLOAT_ABS_TOLERANCE,
    rel_tolerance: float = FLOAT_REL_TOLERANCE,
) -> int:
    """Recursively compare a stored object with an independent recomputation."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValidationError(f"{label}: expected mapping")
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise ValidationError(
                f"{label}: key mismatch; "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        checks = 1
        for key in sorted(expected_keys):
            checks += _assert_equivalent(
                actual[key],
                expected[key],
                f"{label}.{key}",
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
        return checks
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            raise ValidationError(f"{label}: expected sequence")
        if len(actual) != len(expected):
            raise ValidationError(
                f"{label}: length mismatch {len(actual)} != {len(expected)}"
            )
        checks = 1
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            checks += _assert_equivalent(
                actual_item,
                expected_item,
                f"{label}[{index}]",
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
        return checks
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValidationError(f"{label}: {actual!r} != {expected!r}")
        return 1
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, np.integer))
            or int(actual) != int(expected)
        ):
            raise ValidationError(f"{label}: {actual!r} != {expected!r}")
        return 1
    if isinstance(expected, (float, np.floating)):
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{label}: value is not numeric") from error
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            raise ValidationError(f"{label}: non-finite numeric value")
        if not math.isclose(
            actual_float,
            expected_float,
            rel_tol=rel_tolerance,
            abs_tol=abs_tolerance,
        ):
            raise ValidationError(
                f"{label}: {actual_float:.17g} != {expected_float:.17g}"
            )
        return 1
    if actual != expected:
        raise ValidationError(f"{label}: {actual!r} != {expected!r}")
    return 1


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValidationError("unexpected config schema")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValidationError("unexpected protocol id")
    if tuple(int(value) for value in config.get("analysis_shape", [])) != SHAPE:
        raise ValidationError("analysis shape is not 256x256")
    if config.get("stage") != "DEVELOPMENT":
        raise ValidationError("F0 must remain Development")
    if config.get("workflow_profile") != "DEVELOPMENT_STANDARD":
        raise ValidationError("unexpected workflow profile")
    if list(config.get("hazard_truth_ids", [])) != [1, 2]:
        raise ValidationError("hazard truth must remain classes 1 and 2")

    model = config["model_contract"]
    if model["task"] != "relative_depth":
        raise ValidationError("model task drifted from relative depth")
    if model["encoder"] != "vits" or int(model["input_size"]) != 518:
        raise ValidationError("Depth Anything V2 Small interface drifted")
    if bool(model["metric_distance_interpretation"]):
        raise ValidationError("metric distance interpretation is forbidden")
    if bool(model["cross_frame_raw_magnitude_comparison"]):
        raise ValidationError("cross-frame raw magnitude comparison is forbidden")

    direction = config["direction_canary"]
    if direction["frozen_official_output_semantics"] != (
        "AFFINE_INVARIANT_INVERSE_DEPTH"
    ):
        raise ValidationError("official inverse-depth semantics are not frozen")
    if direction["frozen_direction"] != "RAW_LARGER_IS_NEARER":
        raise ValidationError("depth direction is not frozen to larger-is-nearer")
    if bool(direction["canary_may_select_or_flip_direction"]):
        raise ValidationError("direction canary may not select or flip direction")
    if not bool(direction["transform_only_resize_canary_required"]):
        raise ValidationError("transform-only canary is required")

    weights = config["structural_signal"]["D4_weights"]
    if weights != {
        "N": 0.25,
        "E": 0.25,
        "R_plus": 0.25,
        "R_minus": 0.25,
    }:
        raise ValidationError("D4 weights are not frozen 1:1:1:1")
    if bool(config["structural_signal"]["weight_search"]):
        raise ValidationError("D4 weight search is forbidden")
    if bool(config["structural_signal"]["D3_sign_branch_selection"]):
        raise ValidationError("D3 sign branch selection is forbidden")
    proxy = config["proxy_roi_ablation"]
    if float(proxy["lambda"]) != 0.25 or bool(proxy["lambda_search"]):
        raise ValidationError("D5 lambda contract drifted")

    grouped = config["grouped_evaluation"]
    grid = [round(float(value), 10) for value in grouped["threshold_grid"]]
    expected_grid = [round(value / 100.0, 10) for value in range(5, 100, 5)]
    if grid != expected_grid:
        raise ValidationError("threshold grid is not 0.05..0.95 step 0.05")
    if grouped["outer_method"] != "LEAVE_ONE_SOURCE_SESSION_OUT":
        raise ValidationError("outer grouping is not LOSO")
    if grouped["threshold_selection"] != (
        "MAXIMIZE_MINIMUM_NORMALIZED_NINE_GATE_MARGIN"
    ):
        raise ValidationError("threshold selection rule drifted")
    if grouped["average_precision_definition"] != (
        "NON_INTERPOLATED_TIE_GROUP_PRECISION_RECALL_STEP_INTEGRAL"
    ):
        raise ValidationError("average precision definition drifted")
    if grouped["empty_positive_or_negative_group"] != "NOT_EVALUABLE":
        raise ValidationError("empty AP group handling drifted")
    if int(grouped["minimum_D4_positive_advantage_group_count"]) != 8:
        raise ValidationError("D4 group advantage count drifted")
    if not bool(grouped["D4_macro_must_exceed_each_single_signal"]):
        raise ValidationError("D4 macro comparison is not required")
    if grouped["stable_signal_definition"] != (
        "ARM_MACRO_AUPRC_EXCEEDS_B_AND_ARM_GROUP_AUPRC_"
        "EXCEEDS_B_IN_AT_LEAST_8_OF_10_GROUPS"
    ):
        raise ValidationError("stable-signal definition drifted")
    if not bool(
        config["depth_health"][
            "unhealthy_frames_remain_in_metric_denominators_with_zero_scores"
        ]
    ):
        raise ValidationError("unhealthy frames must remain in metric denominators")
    if not bool(config["depth_health"]["semantic_or_visual_abstention_forbidden"]):
        raise ValidationError("semantic or visual abstention must remain forbidden")


def _decode_packed_mask(encoded: str) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValidationError("packed mask is not valid base64") from error
    expected_bytes = (SHAPE[0] * SHAPE[1] + 7) // 8
    if len(raw) != expected_bytes:
        raise ValidationError(
            f"packed mask byte count {len(raw)} != {expected_bytes}"
        )
    unpacked = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8),
        count=SHAPE[0] * SHAPE[1],
        bitorder="big",
    )
    return unpacked.reshape(SHAPE).astype(bool)


def _independent_health_and_proximity(
    raw_depth: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    depth = np.asarray(raw_depth, dtype=np.float64)
    if depth.shape != SHAPE:
        raise ValidationError(f"raw depth shape drifted: {depth.shape}")
    finite_mask = np.isfinite(depth)
    finite_fraction = float(np.count_nonzero(finite_mask) / depth.size)
    if not np.any(finite_mask):
        return {
            "q": 0,
            "finite_fraction": finite_fraction,
            "failure_reasons": ["NO_FINITE_OUTPUT"],
        }, np.zeros(SHAPE, dtype=np.float32)

    finite_values = depth[finite_mask]
    normalization = config["normalization"]
    low_fraction = float(normalization["depth_lower_quantile"])
    high_fraction = float(normalization["depth_upper_quantile"])
    low_value = float(np.quantile(finite_values, low_fraction))
    high_value = float(np.quantile(finite_values, high_fraction))
    median_value = float(np.median(finite_values))
    epsilon = float(normalization["epsilon"])
    robust_span = high_value - low_value
    relative_span = robust_span / max(abs(median_value), epsilon)

    # The official model output is inverse depth; the canary may reject this
    # frozen direction but may never infer or select a different direction.
    if config["direction_canary"]["frozen_direction"] != "RAW_LARGER_IS_NEARER":
        raise ValidationError("unsupported frozen direction")
    normalized = np.zeros(SHAPE, dtype=np.float64)
    if robust_span > epsilon:
        normalized[finite_mask] = np.clip(
            (depth[finite_mask] - low_value) / robust_span,
            float(normalization["clip_min"]),
            float(normalization["clip_max"]),
        )
    normalized_std = float(np.std(normalized[finite_mask]))
    raw_minimum = float(np.min(finite_values))
    raw_maximum = float(np.max(finite_values))
    minimum_plateau = float(np.mean(finite_values == raw_minimum))
    maximum_plateau = float(np.mean(finite_values == raw_maximum))
    extreme_plateau = max(minimum_plateau, maximum_plateau)

    health_contract = config["depth_health"]
    reasons: list[str] = []
    if finite_fraction < float(health_contract["minimum_finite_fraction"]):
        reasons.append("FINITE_FRACTION")
    if relative_span < float(
        health_contract["minimum_relative_robust_span"]
    ):
        reasons.append("ROBUST_SPAN")
    if normalized_std < float(
        health_contract["minimum_normalized_standard_deviation"]
    ):
        reasons.append("NORMALIZED_STD")
    if extreme_plateau > float(
        health_contract["maximum_extreme_plateau_fraction"]
    ):
        reasons.append("EXTREME_PLATEAU")
    q_value = 0 if reasons else 1
    if q_value == 0:
        normalized.fill(0.0)
    return {
        "q": q_value,
        "finite_fraction": finite_fraction,
        "raw_q05": low_value,
        "raw_q95": high_value,
        "relative_robust_span": relative_span,
        "normalized_standard_deviation": normalized_std,
        "extreme_plateau_fraction": extreme_plateau,
        "failure_reasons": reasons,
    }, normalized.astype(np.float32)


def _independent_positive_normalization(
    signal: np.ndarray,
    config: Mapping[str, Any],
    *,
    domain: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    eligible = np.isfinite(values)
    if domain is not None:
        eligible &= np.asarray(domain, dtype=bool)
    samples = values[eligible]
    output = np.zeros(values.shape, dtype=np.float64)
    if samples.size == 0:
        return output.astype(np.float32)
    normalization = config["normalization"]
    lower = float(
        np.quantile(samples, float(normalization["signal_lower_quantile"]))
    )
    upper = float(
        np.quantile(samples, float(normalization["signal_upper_quantile"]))
    )
    denominator = upper - lower
    if denominator <= float(normalization["epsilon"]):
        return output.astype(np.float32)
    output = np.clip(
        (values - lower) / denominator,
        float(normalization["clip_min"]),
        float(normalization["clip_max"]),
    )
    output[~np.isfinite(output)] = 0.0
    return output.astype(np.float32)


def _independent_proxy_mask(config: Mapping[str, Any]) -> np.ndarray:
    height, width = SHAPE
    proxy = config["proxy_roi_ablation"]
    start = int(np.floor(height * float(proxy["start_y_fraction"])))
    center_x = (width - 1) / 2.0
    top_half_width = width * float(proxy["top_half_width_fraction"])
    bottom_half_width = width * float(proxy["bottom_half_width_fraction"])
    output = np.zeros(SHAPE, dtype=bool)
    vertical_span = max(height - 1 - start, 1)
    for row in range(start, height):
        fraction = (row - start) / vertical_span
        half_width = top_half_width + fraction * (
            bottom_half_width - top_half_width
        )
        left = max(0, int(np.ceil(center_x - half_width)))
        right = min(width, int(np.floor(center_x + half_width)) + 1)
        output[row, left:right] = True
    return output


def _independent_structural_scores(
    proximity: np.ndarray,
    q_value: int,
    yolo_mask: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    near = np.asarray(proximity, dtype=np.float32)
    if near.shape != SHAPE:
        raise ValidationError("proximity shape drifted")
    if int(q_value) not in (0, 1):
        raise ValidationError("q must be binary")

    gradient_scales: list[np.ndarray] = []
    for configured_sigma in config["structural_signal"][
        "gradient_gaussian_sigmas_pixels"
    ]:
        sigma = float(configured_sigma)
        blurred = (
            near
            if sigma == 0.0
            else cv2.GaussianBlur(
                near,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
        )
        horizontal = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            1,
            0,
            ksize=int(config["structural_signal"]["gradient_sobel_kernel"]),
        )
        vertical = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            0,
            1,
            ksize=int(config["structural_signal"]["gradient_sobel_kernel"]),
        )
        gradient_scales.append(cv2.magnitude(horizontal, vertical))
    gradient_raw = np.mean(np.stack(gradient_scales, axis=0), axis=0)
    gradient = _independent_positive_normalization(gradient_raw, config)

    trend_contract = config["structural_signal"]["surface_trend"]
    start_row = int(
        np.floor(
            SHAPE[0] * float(trend_contract["lower_image_start_fraction"])
        )
    )
    if start_row < 0 or start_row >= SHAPE[0]:
        raise ValidationError("surface-trend lower-image start is invalid")
    lower_domain = np.zeros(SHAPE, dtype=bool)
    lower_domain[start_row:, :] = True
    row_medians = np.median(near[start_row:, :], axis=1).astype(np.float64)
    degree = int(trend_contract["polynomial_degree"])
    if row_medians.size <= degree or not np.isfinite(row_medians).all():
        raise NotEvaluableError("surface-trend fit input is invalid")
    vertical_coordinate = np.linspace(
        0.0,
        1.0,
        row_medians.size,
        dtype=np.float64,
    )
    design = np.vander(vertical_coordinate, degree + 1)
    if int(np.linalg.matrix_rank(design)) != degree + 1:
        raise NotEvaluableError("surface-trend fit is rank deficient")
    coefficients = np.polyfit(vertical_coordinate, row_medians, degree)
    if not np.isfinite(coefficients).all():
        raise NotEvaluableError("surface-trend fit failed")
    trend_rows = np.polyval(coefficients, vertical_coordinate).astype(
        np.float32
    )
    trend_surface = np.repeat(
        trend_rows[:, None],
        SHAPE[1],
        axis=1,
    )
    dead_zone = float(trend_contract["residual_dead_zone"])
    positive_raw = np.zeros(SHAPE, dtype=np.float32)
    negative_raw = np.zeros(SHAPE, dtype=np.float32)
    lower_near = near[start_row:, :]
    positive_raw[start_row:, :] = np.maximum(
        lower_near - trend_surface - dead_zone,
        0.0,
    )
    negative_raw[start_row:, :] = np.maximum(
        trend_surface - lower_near - dead_zone,
        0.0,
    )
    positive = _independent_positive_normalization(
        positive_raw,
        config,
        domain=lower_domain,
    )
    negative = _independent_positive_normalization(
        negative_raw,
        config,
        domain=lower_domain,
    )

    q_float = float(int(q_value))
    d1 = (q_float * near).astype(np.float32)
    d2 = (q_float * gradient).astype(np.float32)
    d3 = (q_float * 0.5 * (positive + negative)).astype(np.float32)
    d4 = (
        q_float * 0.25 * (near + gradient + positive + negative)
    ).astype(np.float32)
    residual_domain = (~np.asarray(yolo_mask, dtype=bool)).astype(np.float32)
    d1 *= residual_domain
    d2 *= residual_domain
    d3 *= residual_domain
    d4 *= residual_domain
    proxy = _independent_proxy_mask(config).astype(np.float32)
    lambda_value = float(config["proxy_roi_ablation"]["lambda"])
    d5 = (d4 * (lambda_value + (1.0 - lambda_value) * proxy)).astype(
        np.float32
    )
    for arm, score in {
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
        "D5": d5,
    }.items():
        if score.shape != SHAPE or not np.isfinite(score).all():
            raise ValidationError(f"{arm} score is invalid")
        if np.any(score < 0.0) or np.any(score > 1.0):
            raise ValidationError(f"{arm} score escaped [0,1]")
        if np.any(score[np.asarray(yolo_mask, dtype=bool)] != 0.0):
            raise ValidationError(f"{arm} was not residualized after scoring")
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}


def _average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    binary_truth = np.asarray(truth, dtype=np.uint8).reshape(-1)
    numeric_score = np.asarray(score, dtype=np.float64).reshape(-1)
    if binary_truth.shape != numeric_score.shape:
        raise NotEvaluableError("AP truth and score shapes differ")
    if not np.isfinite(numeric_score).all():
        raise NotEvaluableError("AP score is non-finite")
    positive_count = int(np.sum(binary_truth))
    negative_count = int(binary_truth.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise NotEvaluableError("AP group has an empty class")

    descending = np.argsort(-numeric_score, kind="stable")
    ordered_truth = binary_truth[descending]
    ordered_score = numeric_score[descending]
    cumulative_true = np.cumsum(ordered_truth, dtype=np.int64)
    cumulative_false = np.cumsum(1 - ordered_truth, dtype=np.int64)
    tie_group_ends = np.flatnonzero(
        np.concatenate(
            (
                ordered_score[1:] != ordered_score[:-1],
                np.array([True]),
            )
        )
    )
    true_at_threshold = cumulative_true[tie_group_ends].astype(np.float64)
    false_at_threshold = cumulative_false[tie_group_ends].astype(np.float64)
    recall = true_at_threshold / positive_count
    precision = true_at_threshold / np.maximum(
        true_at_threshold + false_at_threshold,
        1.0,
    )
    recall_increments = np.diff(
        np.concatenate((np.array([0.0]), recall))
    )
    return float(np.dot(recall_increments, precision))


def _component_counts(
    candidate: np.ndarray,
    truth: np.ndarray,
) -> tuple[int, int, int]:
    candidate_u8 = np.asarray(candidate, dtype=np.uint8)
    truth_u8 = np.asarray(truth, dtype=np.uint8)
    candidate_label_count, candidate_labels = cv2.connectedComponents(
        candidate_u8,
        connectivity=8,
    )
    truth_label_count, truth_labels = cv2.connectedComponents(
        truth_u8,
        connectivity=8,
    )
    touched_truth_labels = np.unique(
        truth_labels[np.asarray(candidate, dtype=bool)]
    )
    hit_truth_count = int(np.count_nonzero(touched_truth_labels > 0))
    false_candidate_count = 0
    truth_bool = np.asarray(truth, dtype=bool)
    for component_label in range(1, int(candidate_label_count)):
        if not np.any(truth_bool[candidate_labels == component_label]):
            false_candidate_count += 1
    return (
        int(truth_label_count - 1),
        hit_truth_count,
        false_candidate_count,
    )


def _frame_stat(
    context: Mapping[str, Any],
    candidate: np.ndarray,
) -> dict[str, Any]:
    candidate_bool = np.asarray(candidate, dtype=bool)
    truth_residual = np.asarray(context["truth_residual"], dtype=bool)
    baseline = np.asarray(context["baseline_residual"], dtype=bool)
    truth_components, hit_components, false_components = _component_counts(
        candidate_bool,
        truth_residual,
    )
    return {
        "view_row_id": context["view_row_id"],
        "session_id": context["session_id"],
        "source_role": context["source_role"],
        "candidate_tp": int(np.sum(candidate_bool & truth_residual)),
        "candidate_fp": int(
            np.sum(candidate_bool & ~np.asarray(context["truth_full"], dtype=bool))
        ),
        "baseline_tp": int(np.sum(baseline & truth_residual)),
        "baseline_fp": int(
            np.sum(baseline & ~np.asarray(context["truth_full"], dtype=bool))
        ),
        "candidate_boundary_tp": int(
            np.sum(candidate_bool & context["truth_boundary_residual"])
        ),
        "baseline_boundary_tp": int(
            np.sum(baseline & context["truth_boundary_residual"])
        ),
        "candidate_obstacle_tp": int(
            np.sum(candidate_bool & context["truth_obstacle_residual"])
        ),
        "baseline_obstacle_tp": int(
            np.sum(baseline & context["truth_obstacle_residual"])
        ),
        "full_truth_pixels": int(np.sum(context["truth_full"])),
        "residual_truth_pixels": int(np.sum(truth_residual)),
        "truth_component_count": truth_components,
        "hit_truth_component_count": hit_components,
        "false_activation_component_count": false_components,
        "total_pixels": int(candidate_bool.size),
    }


def _safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise NotEvaluableError(f"undefined denominator: {label}")
    return float(numerator / denominator)


def _utility_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not rows:
        raise NotEvaluableError("utility table is empty")
    summed_fields = (
        "candidate_tp",
        "candidate_fp",
        "baseline_tp",
        "baseline_fp",
        "candidate_boundary_tp",
        "baseline_boundary_tp",
        "candidate_obstacle_tp",
        "baseline_obstacle_tp",
        "full_truth_pixels",
        "truth_component_count",
        "hit_truth_component_count",
        "false_activation_component_count",
        "total_pixels",
    )
    sums = {
        field: sum(int(row[field]) for row in rows)
        for field in summed_fields
    }
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["session_id"])].append(row)
    per_session_retention: list[float] = []
    for session_id, session_rows in sorted(by_session.items()):
        candidate_true = sum(
            int(row["candidate_tp"]) for row in session_rows
        )
        baseline_true = sum(int(row["baseline_tp"]) for row in session_rows)
        per_session_retention.append(
            _safe_ratio(
                candidate_true,
                baseline_true,
                f"baseline_tp:{session_id}",
            )
        )
    return {
        "fp_pixel_reduction_vs_B": 1.0
        - _safe_ratio(
            sums["candidate_fp"],
            sums["baseline_fp"],
            "baseline_fp",
        ),
        "overall_residual_recall_retention_vs_B": _safe_ratio(
            sums["candidate_tp"],
            sums["baseline_tp"],
            "baseline_tp",
        ),
        "minimum_group_residual_recall_retention_vs_B": min(
            per_session_retention
        ),
        "boundary_step_curb_recall_retention_vs_B": _safe_ratio(
            sums["candidate_boundary_tp"],
            sums["baseline_boundary_tp"],
            "baseline_boundary_tp",
        ),
        "obstacle_recall_retention_vs_B": _safe_ratio(
            sums["candidate_obstacle_tp"],
            sums["baseline_obstacle_tp"],
            "baseline_obstacle_tp",
        ),
        "delta_recall_C_minus_A": _safe_ratio(
            sums["candidate_tp"],
            sums["full_truth_pixels"],
            "full_truth_pixels",
        ),
        "delta_false_positive_area_fraction_C_minus_A": _safe_ratio(
            sums["candidate_fp"],
            sums["total_pixels"],
            "total_pixels",
        ),
        "residual_truth_component_recall": _safe_ratio(
            sums["hit_truth_component_count"],
            sums["truth_component_count"],
            "truth_component_count",
        ),
        "false_activation_components_per_frame": _safe_ratio(
            sums["false_activation_component_count"],
            len(rows),
            "frame_count",
        ),
    }


GATE_SPECS = (
    ("fp_pixel_reduction_vs_B", "minimum_fp_pixel_reduction_vs_B", "lower"),
    (
        "overall_residual_recall_retention_vs_B",
        "minimum_overall_residual_recall_retention_vs_B",
        "lower",
    ),
    (
        "minimum_group_residual_recall_retention_vs_B",
        "minimum_group_residual_recall_retention_vs_B",
        "lower",
    ),
    (
        "boundary_step_curb_recall_retention_vs_B",
        "minimum_boundary_step_curb_recall_retention_vs_B",
        "lower",
    ),
    (
        "obstacle_recall_retention_vs_B",
        "minimum_obstacle_recall_retention_vs_B",
        "lower",
    ),
    ("delta_recall_C_minus_A", "minimum_delta_recall_C_minus_A", "lower"),
    (
        "delta_false_positive_area_fraction_C_minus_A",
        "maximum_delta_false_positive_area_fraction_C_minus_A",
        "upper",
    ),
    (
        "residual_truth_component_recall",
        "minimum_residual_truth_component_recall",
        "lower",
    ),
    (
        "false_activation_components_per_frame",
        "maximum_false_activation_components_per_frame",
        "upper",
    ),
)


def _gate_report(
    values: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    gates: dict[str, Any] = {}
    margins: list[float] = []
    thresholds = config["utility_gates"]
    for value_name, threshold_name, direction in GATE_SPECS:
        value = float(values[value_name])
        threshold = float(thresholds[threshold_name])
        if threshold == 0.0:
            raise NotEvaluableError("zero gate threshold is undefined")
        if direction == "lower":
            margin = (value - threshold) / abs(threshold)
        else:
            margin = (threshold - value) / abs(threshold)
        gates[value_name] = {
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "normalized_margin": margin,
            "passed": margin >= 0.0,
        }
        margins.append(margin)
    return gates, min(margins)


def _select_threshold(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise NotEvaluableError("threshold search is empty")
    return max(
        candidates,
        key=lambda row: (
            float(row["minimum_normalized_gate_margin"]),
            float(
                row["utility_values"][
                    "minimum_group_residual_recall_retention_vs_B"
                ]
            ),
            float(row["utility_values"]["fp_pixel_reduction_vs_B"]),
            -float(row["threshold"]),
        ),
    )


def _draw_direction_fixture(
    scene_index: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9100 + scene_index)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * (0.31 + 0.025 * scene_index))
    sky_top = np.array([185, 145, 95], dtype=np.float64)
    sky_bottom = np.array([230, 210, 180], dtype=np.float64)
    for row in range(horizon):
        fraction = row / max(horizon - 1, 1)
        image[row, :, :] = (
            (1.0 - fraction) * sky_top + fraction * sky_bottom
        ).astype(np.uint8)
    ground_top = np.array([105, 110, 115], dtype=np.float64)
    ground_bottom = np.array([45, 55, 65], dtype=np.float64)
    for row in range(horizon, height):
        fraction = (row - horizon) / max(height - 1 - horizon, 1)
        color = (1.0 - fraction) * ground_top + fraction * ground_bottom
        texture = rng.normal(
            0.0,
            4.0 + 5.0 * fraction,
            size=(width, 1),
        )
        image[row, :, :] = np.clip(color + texture, 0, 255).astype(
            np.uint8
        )

    vanishing_x = int(width * (0.45 + 0.035 * scene_index))
    for bottom_x in range(-width // 2, width * 3 // 2, width // 8):
        cv2.line(
            image,
            (vanishing_x, horizon),
            (bottom_x, height - 1),
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    for fraction in (0.12, 0.22, 0.36, 0.55, 0.78):
        row = int(horizon + (height - horizon) * fraction)
        cv2.line(image, (0, row), (width - 1, row), (125, 125, 125), 1)

    near_width = int(width * (0.20 + 0.015 * (scene_index % 2)))
    near_height = int(height * (0.45 + 0.02 * (scene_index % 3)))
    near_x = int(width * (0.18 + 0.08 * (scene_index % 3)))
    near_y = height - near_height - int(height * 0.03)
    far_width = int(width * 0.065)
    far_height = int(height * 0.14)
    far_x = int(width * (0.67 - 0.04 * (scene_index % 2)))
    far_y = horizon + int(height * 0.035)
    colors = (
        (35, 65, 210),
        (190, 80, 30),
        (55, 175, 75),
        (155, 50, 160),
    )
    color = colors[scene_index % len(colors)]
    cv2.rectangle(
        image,
        (near_x, near_y),
        (near_x + near_width, near_y + near_height),
        color,
        -1,
    )
    cv2.rectangle(
        image,
        (far_x, far_y),
        (far_x + far_width, far_y + far_height),
        color,
        -1,
    )
    for x, y, width_value, height_value in (
        (near_x, near_y, near_width, near_height),
        (far_x, far_y, far_width, far_height),
    ):
        cv2.rectangle(
            image,
            (x, y),
            (x + width_value, y + height_value),
            (245, 245, 245),
            2,
        )
        cv2.line(
            image,
            (x, y),
            (x + width_value, y + height_value),
            (20, 20, 20),
            2,
        )
        cv2.line(
            image,
            (x + width_value, y),
            (x, y + height_value),
            (20, 20, 20),
            2,
        )
    cv2.ellipse(
        image,
        (near_x + near_width // 2, near_y + near_height),
        (int(near_width * 0.65), int(height * 0.035)),
        0,
        0,
        360,
        (30, 35, 40),
        -1,
    )
    near_mask = np.zeros((height, width), dtype=bool)
    far_mask = np.zeros((height, width), dtype=bool)
    near_inset = max(4, near_width // 8)
    far_inset = max(2, far_width // 8)
    near_mask[
        near_y + near_inset : near_y + near_height - near_inset,
        near_x + near_inset : near_x + near_width - near_inset,
    ] = True
    far_mask[
        far_y + far_inset : far_y + far_height - far_inset,
        far_x + far_inset : far_x + far_width - far_inset,
    ] = True
    return image, near_mask, far_mask


def _independent_transform_canary(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    height = int(config["direction_canary"]["height"])
    width = int(config["direction_canary"]["width"])
    vertical = np.linspace(0.15, 0.65, height, dtype=np.float32)[:, None]
    horizontal = np.linspace(-0.05, 0.05, width, dtype=np.float32)[None, :]
    raw = np.broadcast_to(vertical + horizontal, (height, width)).copy()
    near_mask = np.zeros((height, width), dtype=np.uint8)
    far_mask = np.zeros((height, width), dtype=np.uint8)
    near_mask[
        int(height * 0.55) :,
        int(width * 0.25) : int(width * 0.55),
    ] = 1
    far_mask[
        int(height * 0.18) : int(height * 0.34),
        int(width * 0.62) : int(width * 0.78),
    ] = 1
    raw[near_mask.astype(bool)] += 0.55
    raw[far_mask.astype(bool)] -= 0.10
    resized = cv2.resize(
        raw,
        (SHAPE[1], SHAPE[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized_near = cv2.resize(
        near_mask,
        (SHAPE[1], SHAPE[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized_far = cv2.resize(
        far_mask,
        (SHAPE[1], SHAPE[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    health_a, normalized_a = _independent_health_and_proximity(
        resized,
        config,
    )
    health_b, normalized_b = _independent_health_and_proximity(
        resized * np.float32(3.25) + np.float32(7.0),
        config,
    )
    difference = float(np.max(np.abs(normalized_a - normalized_b)))
    near_median = float(np.median(normalized_a[resized_near]))
    far_median = float(np.median(normalized_a[resized_far]))
    passed = (
        health_a["q"] == 1
        and health_b["q"] == 1
        and near_median > far_median
        and difference <= 2e-6
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "known_inverse_depth_near_median": near_median,
        "known_inverse_depth_far_median": far_median,
        "affine_normalization_max_abs_difference": difference,
        "analysis_shape": list(SHAPE),
    }


def _validate_direction_canary(
    producer_root: Path,
    config: Mapping[str, Any],
    producer_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    canary_path = producer_root / "direction_canary.json"
    canary = _read_json(canary_path)
    if canary.get("protocol_id") != PROTOCOL_ID:
        raise ValidationError("direction canary protocol drifted")
    if canary.get("status") != "PASS":
        raise NotEvaluableError("direction canary did not pass")
    if canary.get("canonical_truth_accessed") is not False:
        raise ValidationError("direction canary accessed canonical truth")
    if canary.get("direction_selected_from_canary") is not False:
        raise ValidationError("direction was selected from canary outcome")

    contract = config["direction_canary"]
    width = int(contract["width"])
    height = int(contract["height"])
    expected_scene_count = int(contract["scene_count"])
    scene_rows = canary.get("scene_rows")
    if not isinstance(scene_rows, list) or len(scene_rows) != expected_scene_count:
        raise ValidationError("direction canary scene count drifted")
    signed_margins: list[float] = []
    checks = 0
    for scene_index, stored in enumerate(scene_rows):
        generated_image, near_mask, far_mask = _draw_direction_fixture(
            scene_index,
            width,
            height,
        )
        image_path = (
            producer_root / "direction_canary" / f"scene_{scene_index}.png"
        )
        raw_path = (
            producer_root
            / "direction_canary"
            / f"raw_depth_{scene_index}.npy"
        )
        if not image_path.is_file() or not raw_path.is_file():
            raise ValidationError("direction canary artifact is missing")
        decoded_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if decoded_image is None or not np.array_equal(
            decoded_image,
            generated_image,
        ):
            raise ValidationError(
                f"direction canary image {scene_index} is not deterministic"
            )
        raw = np.load(raw_path, allow_pickle=False)
        if raw.shape != (height, width) or not np.isfinite(raw).all():
            raise ValidationError(
                f"direction canary raw output {scene_index} is invalid"
            )
        q05, q95 = np.quantile(raw, [0.05, 0.95])
        span = float(q95 - q05)
        margin = (
            0.0
            if span <= 1e-6
            else float(
                (
                    np.median(raw[near_mask])
                    - np.median(raw[far_mask])
                )
                / span
            )
        )
        recomputed = {
            "scene_index": scene_index,
            "image_sha256": _sha256_file(image_path),
            "raw_depth_array_sha256": _sha256_array(raw),
            "near_median": float(np.median(raw[near_mask])),
            "far_median": float(np.median(raw[far_mask])),
            "normalized_near_minus_far_margin": margin,
        }
        checks += _assert_equivalent(
            stored,
            recomputed,
            f"direction_canary.scene_rows[{scene_index}]",
        )
        signed_margins.append(margin)

    positive_count = int(np.sum(np.asarray(signed_margins) > 0.0))
    negative_count = int(np.sum(np.asarray(signed_margins) < 0.0))
    median_margin = float(np.median(np.asarray(signed_margins)))
    decision = {
        "direction": "RAW_LARGER_IS_NEARER",
        "positive_scene_count": positive_count,
        "negative_scene_count": negative_count,
        "consistent_with_frozen_direction_scene_count": positive_count,
        "median_normalized_near_far_margin": median_margin,
        "passed": (
            positive_count
            >= int(contract["minimum_consistent_scene_count"])
            and median_margin
            >= float(contract["minimum_median_normalized_near_far_margin"])
        ),
    }
    checks += _assert_equivalent(
        canary["decision"],
        decision,
        "direction_canary.decision",
    )
    checks += _assert_equivalent(
        producer_receipt["direction_canary"],
        decision,
        "producer_receipt.direction_canary",
    )
    transform_canary = _independent_transform_canary(config)
    checks += _assert_equivalent(
        canary["transform_only_resize_and_normalization_canary"],
        transform_canary,
        "direction_canary.transform_only",
    )
    if not decision["passed"] or transform_canary["status"] != "PASS":
        raise NotEvaluableError("independent direction canary failed")
    return canary, checks


def _validate_model_identity(
    repo_root: Path,
    config: Mapping[str, Any],
    producer_receipt: Mapping[str, Any],
) -> int:
    contract = config["model_contract"]
    archive_path = _resolve_repo_path(
        repo_root,
        contract["source_archive_path"],
    )
    checkpoint_path = _resolve_repo_path(
        repo_root,
        contract["checkpoint_path"],
    )
    _verify_file(archive_path, contract["source_archive_sha256"])
    _verify_file(
        checkpoint_path,
        contract["checkpoint_sha256"],
        expected_bytes=int(contract["checkpoint_bytes"]),
    )
    source_root = _resolve_repo_path(repo_root, contract["source_root"])
    for relative_path, expected_sha in contract["source_file_sha256"].items():
        _verify_file(source_root / relative_path, expected_sha)

    expected_identity = {
        "source_archive_sha256": contract["source_archive_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "source_commit": contract["source_commit"],
        "state_tensor_count": EXPECTED_STATE_TENSOR_COUNT,
        "exact_parameter_count": EXPECTED_PARAMETER_COUNT,
        "official_preprocess": {
            "input_size": 518,
            "keep_aspect_lower_bound_multiple_of_14": True,
            "input_interpolation": "INTER_CUBIC",
            "bgr_to_rgb_divide_255": True,
            "imagenet_normalization": True,
            "original_size_restore": "bilinear_align_corners_true",
            "analysis_resize": "opencv_inter_linear_to_256x256",
        },
    }
    return _assert_equivalent(
        producer_receipt["model_identity"],
        expected_identity,
        "producer_receipt.model_identity",
    )


def _load_independent_contexts(
    *,
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    prepared_root: Path,
    producer_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    checks = 0
    prepare_receipt_path = prepared_root / "prepare_receipt.json"
    inference_manifest_path = prepared_root / "inference_manifest.jsonl"
    prepare_receipt = _read_json(prepare_receipt_path)
    if prepare_receipt.get("protocol_id") != PROTOCOL_ID:
        raise ValidationError("prepare receipt protocol drifted")
    if prepare_receipt.get("status") != "COMPLETE":
        raise NotEvaluableError("prepare did not complete")
    if prepare_receipt.get("config_sha256") != _sha256_file(config_path):
        raise ValidationError("prepare config binding drifted")
    _verify_file(
        inference_manifest_path,
        prepare_receipt["inference_manifest"]["sha256"],
    )
    inference_rows = _read_jsonl(inference_manifest_path)
    if len(inference_rows) != int(
        config["input_contract"]["expected_frame_count"]
    ):
        raise ValidationError("inference manifest frame count drifted")
    if len(inference_rows) != int(
        prepare_receipt["inference_manifest"]["row_count"]
    ):
        raise ValidationError("prepare manifest row count receipt drifted")
    if prepare_receipt.get(
        "scientific_truth_or_packed_masks_emitted_to_inference_manifest"
    ) is not False:
        raise ValidationError("prepare manifest is not truth-minimized")

    producer_receipt_path = producer_root / "producer_receipt.json"
    producer_receipt = _read_json(producer_receipt_path)
    if producer_receipt.get("protocol_id") != PROTOCOL_ID:
        raise ValidationError("producer receipt protocol drifted")
    if producer_receipt.get("status") != "COMPLETE":
        raise NotEvaluableError("producer did not complete")
    if producer_receipt.get("mode") != "full":
        raise NotEvaluableError("validator requires the full producer")
    if producer_receipt.get("config_sha256") != _sha256_file(config_path):
        raise ValidationError("producer config binding drifted")
    if producer_receipt.get("prepare_receipt_sha256") != _sha256_file(
        prepare_receipt_path
    ):
        raise ValidationError("producer prepare-receipt binding drifted")
    if producer_receipt.get("git_head") != prepare_receipt.get("git_head"):
        raise ValidationError("prepare and producer Git identities differ")
    if producer_receipt.get("scientific_truth_accessed_by_producer") is not False:
        raise ValidationError("producer truth-access claim drifted")
    if producer_receipt.get("canonical_mask_or_packed_mask_loaded") is not False:
        raise ValidationError("producer loaded forbidden masks")
    checks += _validate_model_identity(repo_root, config, producer_receipt)
    _, canary_checks = _validate_direction_canary(
        producer_root,
        config,
        producer_receipt,
    )
    checks += canary_checks

    depth_path = producer_root / "depth_maps.npy"
    depth_index_path = producer_root / "depth_index.jsonl"
    _verify_file(depth_path, producer_receipt["depth_map"]["sha256"])
    _verify_file(depth_index_path, producer_receipt["depth_index"]["sha256"])
    depth_index = _read_jsonl(depth_index_path)
    depths = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    expected_depth_shape = (
        int(config["input_contract"]["expected_frame_count"]),
        *SHAPE,
    )
    if tuple(int(value) for value in depths.shape) != expected_depth_shape:
        raise ValidationError(f"depth map shape drifted: {depths.shape}")
    if depths.dtype != np.float32:
        raise ValidationError(f"depth map dtype drifted: {depths.dtype}")
    if list(depths.shape) != list(producer_receipt["depth_map"]["shape"]):
        raise ValidationError("producer depth shape receipt drifted")
    if producer_receipt["depth_map"]["dtype"] != "float32":
        raise ValidationError("producer depth dtype receipt drifted")
    if len(depth_index) != len(inference_rows):
        raise ValidationError("depth index membership drifted")
    if len(depth_index) != int(producer_receipt["depth_index"]["row_count"]):
        raise ValidationError("depth index row-count receipt drifted")

    source_by_view: dict[str, dict[str, Any]] = {}
    expected_source_roles: dict[str, str] = {}
    for source_spec in config["input_contract"]["frame_sources"]:
        source_path = _resolve_repo_path(repo_root, source_spec["path"])
        _verify_file(source_path, source_spec["sha256"])
        source_rows = _read_jsonl(source_path)
        if len(source_rows) != int(source_spec["row_count"]):
            raise ValidationError(f"source ledger row count drifted: {source_path}")
        for source_row_index, source_row in enumerate(source_rows):
            view_row_id = str(source_row["view_row_id"])
            if view_row_id in source_by_view:
                raise ValidationError(f"duplicate source view id: {view_row_id}")
            source_by_view[view_row_id] = {
                "row": source_row,
                "path": source_spec["path"],
                "row_index": source_row_index,
            }
            expected_source_roles[view_row_id] = str(source_spec["role"])

    canonical_spec = config["input_contract"]["canonical_manifest"]
    canonical_manifest_path = _resolve_repo_path(
        repo_root,
        canonical_spec["path"],
    )
    _verify_file(canonical_manifest_path, canonical_spec["sha256"])
    canonical_rows = _read_jsonl(canonical_manifest_path)
    if len(canonical_rows) != int(canonical_spec["row_count"]):
        raise ValidationError("canonical manifest row count drifted")
    canonical_index: dict[tuple[str, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for canonical_row in canonical_rows:
        canonical_index[
            (
                str(canonical_row["session_id"]),
                int(canonical_row["frame_id"]),
                str(canonical_row["image_sha256"]),
            )
        ].append(canonical_row)
    canonical_root = _resolve_repo_path(
        repo_root,
        config["input_contract"]["canonical_view_root"],
    )

    allowed_inference_fields = {
        "schema_version",
        "protocol_id",
        "index",
        "frame_source_path",
        "frame_source_row_index",
        "source_role",
        "view_row_id",
        "session_id",
        "frame_id",
        "image_sha256",
        "image_repo_relative_path",
        "formal_authority",
    }
    contexts: list[dict[str, Any]] = []
    q_count = 0
    for index, (inference_row, depth_row) in enumerate(
        zip(inference_rows, depth_index)
    ):
        if set(inference_row) != allowed_inference_fields:
            raise ValidationError(
                f"inference row {index} is not truth-minimized: "
                f"{sorted(set(inference_row) - allowed_inference_fields)}"
            )
        if inference_row.get("protocol_id") != PROTOCOL_ID:
            raise ValidationError(f"inference row {index} protocol drifted")
        if inference_row.get("formal_authority") is not False:
            raise ValidationError(f"inference row {index} gained authority")
        if int(inference_row["index"]) != index:
            raise ValidationError(f"inference row {index} index drifted")
        view_row_id = str(inference_row["view_row_id"])
        source_binding = source_by_view.get(view_row_id)
        if source_binding is None:
            raise ValidationError(f"unknown inference view id: {view_row_id}")
        if inference_row["frame_source_path"] != source_binding["path"]:
            raise ValidationError(f"frame source path drifted: {view_row_id}")
        if int(inference_row["frame_source_row_index"]) != int(
            source_binding["row_index"]
        ):
            raise ValidationError(f"frame source row drifted: {view_row_id}")
        if inference_row["source_role"] != expected_source_roles[view_row_id]:
            raise ValidationError(f"source role drifted: {view_row_id}")
        source_row = source_binding["row"]
        for field in ("view_row_id", "session_id", "frame_id", "image_sha256"):
            if str(inference_row[field]) != str(source_row[field]):
                raise ValidationError(
                    f"source identity field {field} drifted: {view_row_id}"
                )

        identity_fields = (
            "index",
            "view_row_id",
            "session_id",
            "frame_id",
            "image_sha256",
        )
        if depth_row.get("protocol_id") != PROTOCOL_ID:
            raise ValidationError(f"depth row {index} protocol drifted")
        for field in identity_fields:
            if str(depth_row[field]) != str(inference_row[field]):
                raise ValidationError(
                    f"depth identity field {field} drifted at row {index}"
                )
        image_path = _resolve_repo_path(
            repo_root,
            inference_row["image_repo_relative_path"],
        )
        _verify_file(image_path, inference_row["image_sha256"])

        canonical_key = (
            str(inference_row["session_id"]),
            int(inference_row["frame_id"]),
            str(inference_row["image_sha256"]),
        )
        canonical_matches = canonical_index.get(canonical_key, [])
        if len(canonical_matches) != 1:
            raise ValidationError(
                f"canonical truth mapping count {len(canonical_matches)} "
                f"for row {index}"
            )
        canonical_row = canonical_matches[0]
        mask_path = (
            canonical_root / str(canonical_row["canonical_mask_path"])
        ).resolve()
        if mask_path == canonical_root or canonical_root not in mask_path.parents:
            raise ValidationError("canonical mask escaped canonical root")
        _verify_file(mask_path, canonical_row["canonical_mask_sha256"])
        truth_ids = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if truth_ids is None or truth_ids.shape != SHAPE:
            raise ValidationError(f"canonical mask decode failed at row {index}")
        if not set(int(value) for value in np.unique(truth_ids)).issubset(
            {0, 1, 2, 3}
        ):
            raise ValidationError(f"canonical class id drifted at row {index}")

        packed_masks = source_row.get("packed_masks")
        if not isinstance(packed_masks, Mapping):
            raise ValidationError(f"packed masks missing at row {index}")
        if tuple(int(value) for value in packed_masks.get("shape", [])) != SHAPE:
            raise ValidationError(f"packed mask shape drifted at row {index}")
        for required_name in config["input_contract"]["required_packed_masks"]:
            if required_name not in packed_masks:
                raise ValidationError(
                    f"packed mask {required_name} missing at row {index}"
                )
        yolo_mask = _decode_packed_mask(packed_masks["A"])
        baseline = _decode_packed_mask(packed_masks["B"])
        boundary_candidate = _decode_packed_mask(
            packed_masks["candidate_boundary_step_curb"]
        )
        obstacle_candidate = _decode_packed_mask(
            packed_masks["candidate_obstacle"]
        )
        if not np.array_equal(
            baseline,
            boundary_candidate | obstacle_candidate,
        ):
            raise ValidationError(f"B class union drifted at row {index}")
        if np.any(yolo_mask & baseline):
            raise ValidationError(f"B overlaps A at row {index}")

        raw_depth = np.asarray(depths[index], dtype=np.float32)
        if _sha256_array(raw_depth) != depth_row["raw_depth_array_sha256"]:
            raise ValidationError(f"raw depth array hash drifted at row {index}")
        health, proximity = _independent_health_and_proximity(
            raw_depth,
            config,
        )
        checks += _assert_equivalent(
            depth_row["health"],
            health,
            f"depth_index[{index}].health",
        )
        q_value = int(health["q"])
        q_count += q_value
        scores = _independent_structural_scores(
            proximity,
            q_value,
            yolo_mask,
            config,
        )
        truth_full = np.isin(
            truth_ids,
            np.asarray(config["hazard_truth_ids"], dtype=np.uint8),
        )
        truth_residual = truth_full & ~yolo_mask
        contexts.append(
            {
                "index": index,
                "view_row_id": view_row_id,
                "session_id": str(inference_row["session_id"]),
                "source_role": str(inference_row["source_role"]),
                "a_mask": yolo_mask,
                "baseline_residual": baseline,
                "truth_full": truth_full,
                "truth_residual": truth_residual,
                "truth_boundary_residual": (truth_ids == 1) & ~yolo_mask,
                "truth_obstacle_residual": (truth_ids == 2) & ~yolo_mask,
                "q": q_value,
                "scores": scores,
            }
        )

    session_counts = Counter(
        str(context["session_id"]) for context in contexts
    )
    if len(session_counts) != int(
        config["input_contract"]["expected_source_session_count"]
    ):
        raise ValidationError("source-session count drifted")
    expected_prepare_counts = {
        str(key): int(value)
        for key, value in prepare_receipt["source_session_frame_counts"].items()
    }
    if dict(sorted(session_counts.items())) != dict(
        sorted(expected_prepare_counts.items())
    ):
        raise ValidationError("prepare session membership drifted")
    expected_health_summary = {
        "evaluable_frame_count": q_count,
        "frame_count": len(contexts),
        "coverage": q_count / len(contexts),
    }
    checks += _assert_equivalent(
        producer_receipt["health_summary"],
        expected_health_summary,
        "producer_receipt.health_summary",
    )
    return contexts, producer_receipt, checks


def _group_ap_rows(
    contexts: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for context in contexts:
        by_session[str(context["session_id"])].append(context)
    if not by_session:
        raise NotEvaluableError("AP contexts are empty")
    rows: list[dict[str, Any]] = []
    macro_values: dict[str, list[float]] = {
        arm: [] for arm in ("B", *ARMS)
    }
    for session_id, group in sorted(by_session.items()):
        outside_yolo = np.concatenate(
            [(~context["a_mask"]).reshape(-1) for context in group]
        )
        truth = np.concatenate(
            [context["truth_residual"].reshape(-1) for context in group]
        )[outside_yolo]
        arm_scores: dict[str, np.ndarray] = {
            "B": np.concatenate(
                [
                    context["baseline_residual"]
                    .astype(np.float32)
                    .reshape(-1)
                    for context in group
                ]
            )[outside_yolo]
        }
        for arm in ARMS:
            arm_scores[arm] = np.concatenate(
                [context["scores"][arm].reshape(-1) for context in group]
            )[outside_yolo]
        ap = {
            arm: _average_precision(truth, score)
            for arm, score in arm_scores.items()
        }
        for arm, value in ap.items():
            macro_values[arm].append(value)
        best_single = max(ap[arm] for arm in SINGLE_ARMS)
        d4_delta = ap["D4"] - best_single
        rows.append(
            {
                "session_id": session_id,
                "frame_count": len(group),
                "positive_pixel_count": int(np.sum(truth)),
                "negative_pixel_count": int(truth.size - np.sum(truth)),
                "auprc": ap,
                "D4_minus_best_single": d4_delta,
                "D4_strictly_beats_best_single": d4_delta > epsilon,
                "stable_signal_vs_B": {
                    arm: (ap[arm] - ap["B"]) > epsilon
                    for arm in ("D1", "D2", "D3", "D4")
                },
            }
        )
    macro = {
        arm: float(np.mean(values)) for arm, values in macro_values.items()
    }
    summary: dict[str, Any] = {
        "macro_auprc": macro,
        "D4_positive_advantage_group_count": sum(
            bool(row["D4_strictly_beats_best_single"]) for row in rows
        ),
        "D4_macro_exceeds_each_single_signal": all(
            macro["D4"] - macro[arm] > epsilon for arm in SINGLE_ARMS
        ),
        "stable_signal": {},
    }
    for arm in ("D1", "D2", "D3", "D4"):
        positive_groups = sum(
            bool(row["stable_signal_vs_B"][arm]) for row in rows
        )
        summary["stable_signal"][arm] = {
            "macro_delta_vs_B": macro[arm] - macro["B"],
            "positive_group_count_vs_B": positive_groups,
            "passed": (
                macro[arm] - macro["B"] > epsilon
                and positive_groups >= 8
            ),
        }
    return rows, summary


def _recompute_evaluation(
    contexts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sessions = sorted({str(context["session_id"]) for context in contexts})
    epsilon = float(
        config["grouped_evaluation"]["strict_advantage_epsilon"]
    )
    group_ap_rows, ap_summary = _group_ap_rows(
        contexts,
        epsilon=epsilon,
    )
    thresholds = [
        float(value)
        for value in config["grouped_evaluation"]["threshold_grid"]
    ]
    threshold_stats: dict[float, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        threshold_stats[threshold] = [
            _frame_stat(
                context,
                context["scores"]["D4"] >= threshold,
            )
            for context in contexts
        ]

    search_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_by_session: dict[str, float] = {}
    for held_out_session in sessions:
        training_indices = [
            index
            for index, context in enumerate(contexts)
            if context["session_id"] != held_out_session
        ]
        candidates: list[dict[str, Any]] = []
        for threshold in thresholds:
            training_rows = [
                threshold_stats[threshold][index]
                for index in training_indices
            ]
            values = _utility_values(training_rows)
            gates, minimum_margin = _gate_report(values, config)
            candidate_row = {
                "held_out_session_id": held_out_session,
                "threshold": threshold,
                "utility_values": values,
                "gate_report": gates,
                "minimum_normalized_gate_margin": minimum_margin,
                "all_gates_passed": all(
                    bool(gate["passed"]) for gate in gates.values()
                ),
            }
            search_rows.append(candidate_row)
            candidates.append(candidate_row)
        selected = _select_threshold(candidates)
        selected_threshold = float(selected["threshold"])
        selected_by_session[held_out_session] = selected_threshold
        fold_rows.append(
            {
                "held_out_session_id": held_out_session,
                "training_session_ids": [
                    session
                    for session in sessions
                    if session != held_out_session
                ],
                "selected_threshold": selected_threshold,
                "selection_rule": config["grouped_evaluation"][
                    "threshold_selection"
                ],
                "inner_status": (
                    "ALL_GATE_OPERATING_POINT_AVAILABLE"
                    if any(
                        bool(candidate["all_gates_passed"])
                        for candidate in candidates
                    )
                    else config["grouped_evaluation"][
                        "no_all_gate_marker"
                    ]
                ),
                "selected_minimum_normalized_gate_margin": float(
                    selected["minimum_normalized_gate_margin"]
                ),
            }
        )

    operating_rows: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        selected_threshold = selected_by_session[
            str(context["session_id"])
        ]
        row = dict(threshold_stats[selected_threshold][index])
        row["selected_threshold"] = selected_threshold
        row["q"] = int(context["q"])
        operating_rows.append(row)
    operating_values = _utility_values(operating_rows)
    operating_gates, minimum_margin = _gate_report(
        operating_values,
        config,
    )
    all_gates_passed = all(
        bool(gate["passed"]) for gate in operating_gates.values()
    )

    q_by_session: dict[str, list[int]] = defaultdict(list)
    for context in contexts:
        q_by_session[str(context["session_id"])].append(int(context["q"]))
    group_coverage = {
        session_id: float(np.mean(values))
        for session_id, values in sorted(q_by_session.items())
    }
    overall_coverage = float(
        np.mean([int(context["q"]) for context in contexts])
    )
    health_contract = config["depth_health"]
    coverage_passed = (
        overall_coverage
        >= float(
            health_contract["overall_evaluable_frame_coverage_minimum"]
        )
        and min(group_coverage.values())
        >= float(
            health_contract["minimum_group_evaluable_frame_coverage"]
        )
    )
    composite_passed = (
        bool(ap_summary["D4_macro_exceeds_each_single_signal"])
        and int(ap_summary["D4_positive_advantage_group_count"])
        >= int(
            config["grouped_evaluation"][
                "minimum_D4_positive_advantage_group_count"
            ]
        )
    )
    any_stable_signal = any(
        bool(signal["passed"])
        for signal in ap_summary["stable_signal"].values()
    )
    if not coverage_passed:
        terminal = "NOT_EVALUABLE"
    elif all_gates_passed and composite_passed:
        terminal = "STRUCTURAL_SIGNAL_SUPPORTED_FOR_F1_DESIGN"
    elif any_stable_signal:
        terminal = "SIGNAL_PRESENT_BUT_COMPOSITE_NOT_READY"
    else:
        terminal = "STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP"
    if terminal not in config["terminals"]:
        raise ValidationError(f"recomputed terminal is not allowed: {terminal}")

    role_ap: dict[str, Any] = {}
    source_roles = sorted(
        {str(context["source_role"]) for context in contexts}
    )
    for source_role in source_roles:
        role_contexts = [
            context
            for context in contexts
            if context["source_role"] == source_role
        ]
        try:
            _, role_summary = _group_ap_rows(
                role_contexts,
                epsilon=epsilon,
            )
            role_ap[source_role] = role_summary
        except NotEvaluableError as error:
            role_ap[source_role] = {
                "status": "NOT_EVALUABLE",
                "reason": str(error),
            }
    return {
        "group_auprc": group_ap_rows,
        "threshold_search": search_rows,
        "fold_thresholds": fold_rows,
        "frame_operating_metrics": operating_rows,
        "auprc": ap_summary,
        "selected_thresholds_by_session": selected_by_session,
        "utility_values": operating_values,
        "gate_report": operating_gates,
        "minimum_normalized_gate_margin": minimum_margin,
        "all_utility_gates_passed": all_gates_passed,
        "overall_coverage": overall_coverage,
        "group_coverage": group_coverage,
        "coverage_passed": coverage_passed,
        "composite_advantage_passed": composite_passed,
        "any_stable_signal": any_stable_signal,
        "terminal": terminal,
        "role_ap": role_ap,
    }


def run_validation(
    *,
    repo_root: Path,
    config_path: Path,
    prepared_root: Path,
    producer_root: Path,
    evaluation_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    prepared_root = prepared_root.resolve()
    producer_root = producer_root.resolve()
    evaluation_root = evaluation_root.resolve()
    config = _read_json(config_path)
    _validate_config(config)

    contexts, producer_receipt, check_count = _load_independent_contexts(
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        prepared_root=prepared_root,
        producer_root=producer_root,
    )
    recomputed = _recompute_evaluation(contexts, config)
    result_path = evaluation_root / "result.json"
    stored_result = _read_json(result_path)
    if stored_result.get("schema_version") != (
        "blindassist.dg_srf_image_space_structural_"
        "complementarity_f0.evaluation_result.v1"
    ):
        raise ValidationError("evaluation result schema drifted")
    if stored_result.get("protocol_id") != PROTOCOL_ID:
        raise ValidationError("evaluation result protocol drifted")
    if stored_result.get("status") != (
        "COMPLETE_PENDING_INDEPENDENT_VALIDATION"
    ):
        raise ValidationError("evaluation result status drifted")
    if stored_result.get("validation_status") != "PENDING":
        raise ValidationError("evaluation was modified after validation")
    expected_result_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "stage",
        "workflow_profile",
        "git_head",
        "config_sha256",
        "prepare_receipt_sha256",
        "frame_count",
        "source_session_count",
        "producer_receipt_sha256",
        "depth_map_sha256",
        "auprc",
        "cross_fitted_operating_point",
        "depth_health_coverage",
        "composite_advantage_passed",
        "any_stable_signal",
        "provisional_scientific_terminal",
        "validation_status",
        "detector_identity_stratified_diagnostic_by_confounded_role",
        "limitations",
        "claim_ceiling",
        "runtime_seconds",
        "output_files",
    }
    if set(stored_result) != expected_result_keys:
        raise ValidationError(
            "evaluation result fields drifted; "
            f"missing={sorted(expected_result_keys - set(stored_result))} "
            f"extra={sorted(set(stored_result) - expected_result_keys)}"
        )

    artifact_rows = {
        "group_auprc.jsonl": recomputed["group_auprc"],
        "threshold_search.jsonl": recomputed["threshold_search"],
        "fold_thresholds.jsonl": recomputed["fold_thresholds"],
        "frame_operating_metrics.jsonl": recomputed[
            "frame_operating_metrics"
        ],
    }
    if set(stored_result["output_files"]) != set(artifact_rows):
        raise ValidationError("evaluation output-file binding set drifted")
    for file_name, expected_rows in artifact_rows.items():
        artifact_path = evaluation_root / file_name
        output_binding = stored_result["output_files"].get(file_name)
        if not isinstance(output_binding, Mapping):
            raise ValidationError(f"result lacks output binding: {file_name}")
        _verify_file(
            artifact_path,
            output_binding["sha256"],
            expected_bytes=int(output_binding["bytes"]),
        )
        stored_rows = _read_jsonl(artifact_path)
        check_count += _assert_equivalent(
            stored_rows,
            expected_rows,
            file_name,
        )

    expected_result_fields = {
        "stage": config["stage"],
        "workflow_profile": config["workflow_profile"],
        "git_head": producer_receipt["git_head"],
        "config_sha256": _sha256_file(config_path),
        "prepare_receipt_sha256": _sha256_file(
            prepared_root / "prepare_receipt.json"
        ),
        "frame_count": len(contexts),
        "source_session_count": len(
            {str(context["session_id"]) for context in contexts}
        ),
        "producer_receipt_sha256": _sha256_file(
            producer_root / "producer_receipt.json"
        ),
        "depth_map_sha256": producer_receipt["depth_map"]["sha256"],
        "auprc": recomputed["auprc"],
        "cross_fitted_operating_point": {
            "selected_thresholds_by_session": recomputed[
                "selected_thresholds_by_session"
            ],
            "utility_values": recomputed["utility_values"],
            "gate_report": recomputed["gate_report"],
            "minimum_normalized_gate_margin": recomputed[
                "minimum_normalized_gate_margin"
            ],
            "all_utility_gates_passed": recomputed[
                "all_utility_gates_passed"
            ],
        },
        "depth_health_coverage": {
            "overall": recomputed["overall_coverage"],
            "by_session": recomputed["group_coverage"],
            "passed": recomputed["coverage_passed"],
        },
        "composite_advantage_passed": recomputed[
            "composite_advantage_passed"
        ],
        "any_stable_signal": recomputed["any_stable_signal"],
        "provisional_scientific_terminal": recomputed["terminal"],
        "detector_identity_stratified_diagnostic_by_confounded_role": (
            recomputed["role_ap"]
        ),
        "limitations": [
            "all 10 source-session groups are SANPO-Real",
            "participant route and parent-capture independence are not evaluable",
            "two YOLO detector identities are fully confounded with source role",
            "B is a frozen binary DDRNet residual mask rather than continuous score",
            "no real-time flicker or event effect is evaluated",
            "component hit uses any positive pixel intersection",
        ],
        "claim_ceiling": config["claim_ceiling"],
    }
    for field, expected_value in expected_result_fields.items():
        if field not in stored_result:
            raise ValidationError(f"evaluation result lacks {field}")
        check_count += _assert_equivalent(
            stored_result[field],
            expected_value,
            f"result.{field}",
        )
    if not math.isfinite(float(stored_result.get("runtime_seconds", -1.0))):
        raise ValidationError("evaluation runtime is non-finite")
    if float(stored_result["runtime_seconds"]) < 0.0:
        raise ValidationError("evaluation runtime is negative")

    return {
        "schema_version": VALIDATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "validation_status": "VALID",
        "scientific_terminal": recomputed["terminal"],
        "independent_recomputation": {
            "imports_producer_evaluator_or_operators": False,
            "raw_depth_rehashed": True,
            "depth_health_and_normalization_recomputed": True,
            "D1_D2_D3_D4_D5_recomputed": True,
            "canonical_truth_redecoded": True,
            "packed_A_B_masks_redecoded": True,
            "group_auprc_recomputed": True,
            "nested_threshold_search_recomputed": True,
            "nine_utility_gates_recomputed": True,
            "coverage_and_terminal_recomputed": True,
        },
        "check_count": check_count,
        "frame_count": len(contexts),
        "source_session_count": len(
            {str(context["session_id"]) for context in contexts}
        ),
        "bindings": {
            "config_sha256": _sha256_file(config_path),
            "prepare_receipt_sha256": _sha256_file(
                prepared_root / "prepare_receipt.json"
            ),
            "producer_receipt_sha256": _sha256_file(
                producer_root / "producer_receipt.json"
            ),
            "depth_map_sha256": producer_receipt["depth_map"]["sha256"],
            "evaluation_result_sha256": _sha256_file(result_path),
        },
        "recomputed_summary": {
            "auprc": recomputed["auprc"],
            "cross_fitted_operating_point": {
                "selected_thresholds_by_session": recomputed[
                    "selected_thresholds_by_session"
                ],
                "utility_values": recomputed["utility_values"],
                "gate_report": recomputed["gate_report"],
                "all_utility_gates_passed": recomputed[
                    "all_utility_gates_passed"
                ],
            },
            "depth_health_coverage": {
                "overall": recomputed["overall_coverage"],
                "by_session": recomputed["group_coverage"],
                "passed": recomputed["coverage_passed"],
            },
            "composite_advantage_passed": recomputed[
                "composite_advantage_passed"
            ],
            "any_stable_signal": recomputed["any_stable_signal"],
        },
        "authority_boundary": (
            "Independent evaluation recomputation over consumed Development "
            "depth outputs; it does not rerun all model inference, establish "
            "participant/route/parent-capture independence, or authorize F1 "
            "execution, Android, risk, feedback, product, or safety claims."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = _require_artifact_output(repo_root, args.output)
    if output.exists():
        raise FileExistsError(f"validation output already exists: {output}")
    try:
        result = run_validation(
            repo_root=repo_root,
            config_path=args.config,
            prepared_root=args.prepared_root,
            producer_root=args.producer_root,
            evaluation_root=args.evaluation_root,
        )
    except Exception as error:
        invalid = {
            "schema_version": VALIDATION_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "validation_status": "INVALID",
            "scientific_terminal": "NOT_EVALUABLE",
            "error_type": type(error).__name__,
            "error": str(error),
            "authority_boundary": (
                "A validator failure creates no scientific terminal other "
                "than NOT_EVALUABLE and grants no successor authority."
            ),
        }
        _write_json(output, invalid)
        print(
            f"INVALID {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    _write_json(output, result)
    print(
        f"VALID terminal={result['scientific_terminal']} "
        f"checks={result['check_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
