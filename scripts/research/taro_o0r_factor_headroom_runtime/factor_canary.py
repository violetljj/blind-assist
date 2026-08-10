#!/usr/bin/env python3
"""Pure post-hoc descriptive factor canary for TARO O0R.

This module deliberately performs no source I/O, training, reducer scoring,
thresholding, or pass/fail decision.  It can be applied to already validated
FARO/candidate query-factor frames even when the parent R3 route is
``NOT_EVALUABLE``.  Invalid or insufficient evidence remains explicitly
unevaluable; it is never converted into a negative observation.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.factor_headroom import (
    CANDIDATE_RELATIVE_SCALE_SCHEMA,
    MINIMUM_SCALE_PAIRS,
    FactorHeadroomError,
    derive_candidate_relative_scale,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


FACTOR_CANARY_RECORD_SCHEMA = "blindassist.taro.o0r.descriptive_factor_canary_query.v1"
FACTOR_CANARY_SUMMARY_SCHEMA = "blindassist.taro.o0r.descriptive_factor_canary_summary.v1"
ANALYSIS_KIND = "POST_HOC_DESCRIPTIVE_FACTOR_CANARY"
MINIMUM_BOUNDARY_INTERSECTION_POINTS = 3
CLAIM_CEILING = {
    "scope": "LOCKED_ARKITSCENES_TRAIN_LANDSCAPE_ONLY",
    "use": "POST_HOC_DESCRIPTIVE_FACTOR_DIAGNOSTIC",
    "excluded_claims": [
        "FINAL_TASK_EFFECTIVENESS",
        "DEPLOYMENT",
        "PRODUCT",
        "SAFETY",
    ],
    "threshold_or_pass_fail_decision": False,
}
FACTOR_NAMES = ("SCALE", "SUPPORT", "BOUNDARY")

_SHA256 = re.compile(r"[0-9A-Fa-f]{64}")
_SCALE_RECORD_KEYS = {
    "schema",
    "physical_frame_id",
    "query_id",
    "faro_factor_frame_sha256",
    "candidate_factor_frame_sha256",
    "faro_depth_array_sha256",
    "candidate_depth_array_sha256",
    "common_support_point_ids_sha256",
    "common_support_point_count",
    "valid_pair_count",
    "estimator",
    "value_kind",
    "log_metric_scale",
    "metric_scale",
    "truth_alignment_used_for_candidate_generation",
    "computed_only_after_candidate_output_sealed",
    "content_sha256",
}
_FACTOR_METRIC_FIELDS = {
    "SCALE": ("abs_log_correction", "metric_scale", "valid_pair_count"),
    "SUPPORT": (
        "raw_normal_angular_error_rad",
        "raw_height_abs_error_m",
        "scale_corrected_height_abs_error_m",
    ),
    "BOUNDARY": (
        "truth_point_count",
        "candidate_point_count",
        "point_id_intersection_count",
        "point_id_union_count",
        "point_id_jaccard",
        "raw_xyz_median_error_m",
        "scale_corrected_xyz_median_error_m",
    ),
}


class FactorCanaryError(ValueError):
    """Stable invalid-input or tamper error for the descriptive canary."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.context = context


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FactorCanaryError(code, message, **context)


def _canonical_copy(value: Any) -> Any:
    return json.loads(adapter.canonical_json_bytes(value).decode("utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = _canonical_copy(dict(value))
    _require("content_sha256" not in output, "CANARY_SEAL_COLLISION", "caller supplied a content hash")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return _canonical_copy(output)


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _hash(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "CANARY_HASH_INVALID", "hash is malformed", field=field)
    return value.upper()


def _reason_codes(value: Any, *, factor: str) -> list[str]:
    _require(isinstance(value, list), "CANARY_REASON_CODES_INVALID", "reason_codes must be a list", factor=factor)
    _require(
        all(isinstance(item, str) and bool(item) for item in value) and len(set(value)) == len(value),
        "CANARY_REASON_CODES_INVALID",
        "reason codes must be unique non-empty strings",
        factor=factor,
    )
    return value


def _validate_scale_record(value: Any, lineage: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, dict), "CANARY_SCALE_RECORD_INVALID", "scale record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    _require(
        isinstance(observed, str)
        and bool(_SHA256.fullmatch(observed))
        and adapter.canonical_sha256(record) == observed.upper(),
        "CANARY_SCALE_RECORD_INVALID",
        "scale record seal drift",
    )
    record["content_sha256"] = observed.upper()
    _require(
        set(record) == _SCALE_RECORD_KEYS and record["schema"] == CANDIDATE_RELATIVE_SCALE_SCHEMA,
        "CANARY_SCALE_RECORD_INVALID",
        "scale record schema/key set drift",
    )
    for key in ("physical_frame_id", "query_id", "faro_factor_frame_sha256", "candidate_factor_frame_sha256", "faro_depth_array_sha256", "candidate_depth_array_sha256"):
        _require(record[key] == lineage[key], "CANARY_SCALE_LINEAGE_MISMATCH", "scale record lineage drift", field=key)
    for key in ("common_support_point_count", "valid_pair_count"):
        _require(isinstance(record[key], int) and not isinstance(record[key], bool) and record[key] >= MINIMUM_SCALE_PAIRS, "CANARY_SCALE_RECORD_INVALID", "scale support count is invalid", field=key)
    _require(record["valid_pair_count"] <= record["common_support_point_count"], "CANARY_SCALE_RECORD_INVALID", "valid scale pairs exceed common support")
    log_scale = record["log_metric_scale"]
    metric_scale = record["metric_scale"]
    _require(_finite_number(log_scale) and _finite_number(metric_scale) and float(metric_scale) > 0.0, "CANARY_SCALE_RECORD_INVALID", "scale values are invalid")
    _require(abs(math.exp(float(log_scale)) - float(metric_scale)) <= 1e-10 * max(1.0, float(metric_scale)), "CANARY_SCALE_RECORD_INVALID", "metric/log scale mismatch")
    _require(
        record["truth_alignment_used_for_candidate_generation"] is False
        and record["computed_only_after_candidate_output_sealed"] is True,
        "CANARY_SCALE_RECORD_INVALID",
        "scale record violates post-candidate truth boundary",
    )
    return record


def _validate_depth_array(value: Any, *, unit: str) -> np.ndarray:
    array = np.asarray(value)
    _require(
        array.shape == adapter.HIGHRES_SHAPE_HW
        and array.dtype.kind in "iuf"
        and bool(np.all(np.isfinite(array))),
        "CANARY_DEPTH_INVALID",
        "depth must be a finite numeric 1440x1920 raster",
        unit=unit,
    )
    return array


def _validate_parent_frames(
    parent_id: Any,
    truth_factor_frame: Mapping[str, Any],
    candidate_factor_frame: Mapping[str, Any],
    faro_depth_mm: Any,
    candidate_depth_m: Any,
) -> tuple[str, dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    _require(isinstance(parent_id, str) and bool(parent_id), "CANARY_PARENT_ID_INVALID", "parent_id is required")
    try:
        truth = adapter.validate_query_factor_frame(dict(truth_factor_frame))
        candidate = adapter.validate_query_factor_frame(dict(candidate_factor_frame))
    except (adapter.AdapterError, TypeError, ValueError) as error:
        raise FactorCanaryError("CANARY_FACTOR_FRAME_INVALID", "factor frame validation failed") from error
    _require(truth["factor_identity"]["origin"] == "FARO_TRUTH", "CANARY_TRUTH_ORIGIN_INVALID", "truth frame must have FARO_TRUTH origin")
    _require(candidate["factor_identity"]["origin"] == "CANDIDATE_DEPTH_EXTRACTOR", "CANARY_CANDIDATE_ORIGIN_INVALID", "candidate frame must have CANDIDATE_DEPTH_EXTRACTOR origin")
    for key in ("physical_frame_id", "query_id", "source_frame_receipt_sha256", "query_receipt_sha256", "max_source_timestamp_ns"):
        _require(truth[key] == candidate[key], "CANARY_FRAME_IDENTITY_MISMATCH", "truth/candidate identity drift", field=key)
    _require(
        truth["base_geometry"]["content_sha256"] == candidate["base_geometry"]["content_sha256"],
        "CANARY_BASE_GEOMETRY_MISMATCH",
        "truth/candidate base geometry differs",
    )
    faro = _validate_depth_array(faro_depth_mm, unit="MILLIMETRES")
    candidate_depth = _validate_depth_array(candidate_depth_m, unit="METRES")
    faro_hash = adapter.canonical_sha256(faro)
    candidate_hash = adapter.canonical_sha256(candidate_depth)
    _require(
        faro_hash == truth["factor_identity"]["input_depth_array_sha256"]
        == truth["base_geometry"]["faro_depth_array_sha256"],
        "CANARY_DEPTH_LINEAGE_MISMATCH",
        "FARO depth is not the bound truth input",
    )
    _require(
        candidate_hash == candidate["factor_identity"]["input_depth_array_sha256"],
        "CANARY_DEPTH_LINEAGE_MISMATCH",
        "candidate depth is not the bound extractor input",
    )
    lineage = {
        "physical_frame_id": truth["physical_frame_id"],
        "query_id": truth["query_id"],
        "source_frame_receipt_sha256": truth["source_frame_receipt_sha256"],
        "query_receipt_sha256": truth["query_receipt_sha256"],
        "max_source_timestamp_ns": truth["max_source_timestamp_ns"],
        "base_geometry_sha256": truth["base_geometry"]["content_sha256"],
        "faro_factor_frame_sha256": truth["content_sha256"],
        "candidate_factor_frame_sha256": candidate["content_sha256"],
        "faro_depth_array_sha256": faro_hash,
        "candidate_depth_array_sha256": candidate_hash,
    }
    return parent_id, truth, candidate, faro, candidate_depth, lineage


def _normal_angle(first: Any, second: Any) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))


def _boundary_mapping(value: Mapping[str, Any]) -> dict[tuple[int, int], np.ndarray]:
    ids = np.asarray(value["point_ids_uv"], dtype=np.int64)
    points = np.asarray(value["boundary_points_shape_camera_xyz"], dtype=np.float64)
    return {tuple(int(item) for item in pixel): points[index] for index, pixel in enumerate(ids)}


def build_factor_canary_record(
    parent_id: str,
    truth_factor_frame: Mapping[str, Any],
    candidate_factor_frame: Mapping[str, Any],
    faro_depth_mm: np.ndarray,
    candidate_depth_m: np.ndarray,
) -> dict[str, Any]:
    """Build one sealed, threshold-free descriptive record for one query."""

    parent, truth, candidate, faro, candidate_depth, lineage = _validate_parent_frames(
        parent_id,
        truth_factor_frame,
        candidate_factor_frame,
        faro_depth_mm,
        candidate_depth_m,
    )

    scale_record: dict[str, Any] | None
    scale_reason_codes: list[str]
    try:
        scale_record = derive_candidate_relative_scale(
            faro,
            candidate_depth,
            truth["base_geometry"]["common_point_ids_uv"],
            physical_frame_id=truth["physical_frame_id"],
            query_id=truth["query_id"],
            faro_factor_frame_sha256=truth["content_sha256"],
            candidate_factor_frame_sha256=candidate["content_sha256"],
        )
        scale_record = _validate_scale_record(scale_record, lineage)
        scale_reason_codes = []
    except FactorHeadroomError as error:
        scale_record = None
        scale_reason_codes = [error.code]
    scale_evaluable = scale_record is not None
    scale_factor = {
        "evaluable": scale_evaluable,
        "reason_codes": scale_reason_codes,
        "abs_log_correction": abs(float(scale_record["log_metric_scale"])) if scale_record else None,
        "metric_scale": float(scale_record["metric_scale"]) if scale_record else None,
        "valid_pair_count": int(scale_record["valid_pair_count"]) if scale_record else None,
    }

    truth_support = truth["blocks"]["SUPPORT"]
    candidate_support = candidate["blocks"]["SUPPORT"]
    support_reasons: list[str] = []
    if truth_support["validity"]["valid"] is not True:
        support_reasons.append("TRUTH_SUPPORT_INVALID")
    if candidate_support["validity"]["valid"] is not True:
        support_reasons.append("CANDIDATE_SUPPORT_INVALID")
    if not scale_evaluable:
        support_reasons.append("SCALE_NOT_EVALUABLE")
    support_evaluable = not support_reasons
    if support_evaluable:
        truth_height = float(truth_support["value"]["camera_height_shape_m"])
        candidate_height = float(candidate_support["value"]["camera_height_shape_m"])
        metric_scale = float(scale_record["metric_scale"])
        support_factor = {
            "evaluable": True,
            "reason_codes": [],
            "raw_normal_angular_error_rad": _normal_angle(
                truth_support["value"]["normal_camera_xyz"],
                candidate_support["value"]["normal_camera_xyz"],
            ),
            "raw_height_abs_error_m": abs(truth_height - candidate_height),
            "scale_corrected_height_abs_error_m": abs(truth_height - metric_scale * candidate_height),
        }
    else:
        support_factor = {
            "evaluable": False,
            "reason_codes": support_reasons,
            "raw_normal_angular_error_rad": None,
            "raw_height_abs_error_m": None,
            "scale_corrected_height_abs_error_m": None,
        }

    truth_boundary = truth["blocks"]["BOUNDARY"]
    candidate_boundary = candidate["blocks"]["BOUNDARY"]
    boundary_reasons: list[str] = []
    if truth_boundary["validity"]["valid"] is not True:
        boundary_reasons.append("TRUTH_BOUNDARY_INVALID")
    if candidate_boundary["validity"]["valid"] is not True:
        boundary_reasons.append("CANDIDATE_BOUNDARY_INVALID")
    boundary_metrics: dict[str, Any] = {name: None for name in _FACTOR_METRIC_FIELDS["BOUNDARY"]}
    if not boundary_reasons:
        truth_points = _boundary_mapping(truth_boundary["value"])
        candidate_points = _boundary_mapping(candidate_boundary["value"])
        truth_ids = set(truth_points)
        candidate_ids = set(candidate_points)
        common_ids = sorted(truth_ids & candidate_ids)
        union_count = len(truth_ids | candidate_ids)
        boundary_metrics.update(
            {
                "truth_point_count": len(truth_ids),
                "candidate_point_count": len(candidate_ids),
                "point_id_intersection_count": len(common_ids),
                "point_id_union_count": union_count,
                "point_id_jaccard": (len(common_ids) / union_count) if union_count else None,
            }
        )
        if len(common_ids) < MINIMUM_BOUNDARY_INTERSECTION_POINTS:
            boundary_reasons.append("BOUNDARY_COMMON_POINT_IDS_INSUFFICIENT")
        else:
            truth_xyz = np.stack([truth_points[key] for key in common_ids])
            candidate_xyz = np.stack([candidate_points[key] for key in common_ids])
            raw_errors = np.linalg.norm(truth_xyz - candidate_xyz, axis=1)
            boundary_metrics["raw_xyz_median_error_m"] = float(np.median(raw_errors))
            if scale_evaluable:
                corrected_errors = np.linalg.norm(truth_xyz - float(scale_record["metric_scale"]) * candidate_xyz, axis=1)
                boundary_metrics["scale_corrected_xyz_median_error_m"] = float(np.median(corrected_errors))
            else:
                boundary_reasons.append("SCALE_NOT_EVALUABLE")
    boundary_factor = {
        "evaluable": not boundary_reasons,
        "reason_codes": boundary_reasons,
        **boundary_metrics,
    }

    return _seal(
        {
            "schema": FACTOR_CANARY_RECORD_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "parent_id": parent,
            **lineage,
            "scale_record": scale_record,
            "scale_record_sha256": scale_record["content_sha256"] if scale_record else None,
            "factors": {
                "SCALE": scale_factor,
                "SUPPORT": support_factor,
                "BOUNDARY": boundary_factor,
            },
        }
    )


def _validate_factor_metrics(name: str, value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "CANARY_FACTOR_INVALID", "factor result must be an object", factor=name)
    expected = {"evaluable", "reason_codes", *_FACTOR_METRIC_FIELDS[name]}
    _require(set(value) == expected, "CANARY_FACTOR_INVALID", "factor metric key set drift", factor=name)
    _require(isinstance(value["evaluable"], bool), "CANARY_FACTOR_INVALID", "evaluable must be boolean", factor=name)
    reasons = _reason_codes(value["reason_codes"], factor=name)
    _require(bool(reasons) != value["evaluable"], "CANARY_FACTOR_INVALID", "evaluable/reason code mismatch", factor=name)
    if name == "SCALE":
        if value["evaluable"]:
            _require(_finite_number(value["abs_log_correction"]) and float(value["abs_log_correction"]) >= 0.0, "CANARY_FACTOR_INVALID", "scale correction is invalid")
            _require(_finite_number(value["metric_scale"]) and float(value["metric_scale"]) > 0.0, "CANARY_FACTOR_INVALID", "metric scale is invalid")
            _require(isinstance(value["valid_pair_count"], int) and not isinstance(value["valid_pair_count"], bool) and value["valid_pair_count"] >= MINIMUM_SCALE_PAIRS, "CANARY_FACTOR_INVALID", "scale pair count is invalid")
        else:
            _require(all(value[field] is None for field in _FACTOR_METRIC_FIELDS[name]), "CANARY_FACTOR_INVALID", "unevaluable scale carries metrics")
    elif name == "SUPPORT":
        if value["evaluable"]:
            _require(all(_finite_number(value[field]) and float(value[field]) >= 0.0 for field in _FACTOR_METRIC_FIELDS[name]), "CANARY_FACTOR_INVALID", "support error is invalid")
        else:
            _require(all(value[field] is None for field in _FACTOR_METRIC_FIELDS[name]), "CANARY_FACTOR_INVALID", "unevaluable support carries metrics")
    else:
        count_fields = ("truth_point_count", "candidate_point_count", "point_id_intersection_count", "point_id_union_count")
        has_overlap = all(isinstance(value[field], int) and not isinstance(value[field], bool) and value[field] >= 0 for field in count_fields)
        if value["evaluable"]:
            _require(has_overlap, "CANARY_FACTOR_INVALID", "boundary overlap counts are invalid")
            _require(value["point_id_intersection_count"] >= MINIMUM_BOUNDARY_INTERSECTION_POINTS, "CANARY_FACTOR_INVALID", "boundary intersection is insufficient")
            _require(all(_finite_number(value[field]) and float(value[field]) >= 0.0 for field in ("point_id_jaccard", "raw_xyz_median_error_m", "scale_corrected_xyz_median_error_m")), "CANARY_FACTOR_INVALID", "boundary metric is invalid")
        elif has_overlap:
            _require(value["raw_xyz_median_error_m"] is None or (_finite_number(value["raw_xyz_median_error_m"]) and float(value["raw_xyz_median_error_m"]) >= 0.0), "CANARY_FACTOR_INVALID", "boundary raw error is invalid")
            _require(value["scale_corrected_xyz_median_error_m"] is None, "CANARY_FACTOR_INVALID", "unevaluable boundary carries corrected error")
        else:
            _require(all(value[field] is None for field in _FACTOR_METRIC_FIELDS[name]), "CANARY_FACTOR_INVALID", "invalid boundary carries partial overlap metrics")
        if has_overlap:
            intersection = value["point_id_intersection_count"]
            union = value["point_id_union_count"]
            _require(intersection <= union and intersection <= value["truth_point_count"] and intersection <= value["candidate_point_count"], "CANARY_FACTOR_INVALID", "boundary set counts are inconsistent")
            expected_jaccard = intersection / union if union else None
            _require(
                (expected_jaccard is None and value["point_id_jaccard"] is None)
                or (_finite_number(value["point_id_jaccard"]) and abs(float(value["point_id_jaccard"]) - expected_jaccard) <= 1e-12),
                "CANARY_FACTOR_INVALID",
                "boundary Jaccard is inconsistent",
            )
    return value


def validate_factor_canary_record(value: Any) -> dict[str, Any]:
    """Validate a query record, including its outer and nested scale seals."""

    _require(isinstance(value, dict), "CANARY_RECORD_INVALID", "canary record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    _require(
        isinstance(observed, str)
        and bool(_SHA256.fullmatch(observed))
        and adapter.canonical_sha256(record) == observed.upper(),
        "CANARY_RECORD_SEAL_MISMATCH",
        "canary record seal drift",
    )
    record["content_sha256"] = observed.upper()
    expected = {
        "schema",
        "analysis_kind",
        "claim_ceiling",
        "parent_id",
        "physical_frame_id",
        "query_id",
        "source_frame_receipt_sha256",
        "query_receipt_sha256",
        "max_source_timestamp_ns",
        "base_geometry_sha256",
        "faro_factor_frame_sha256",
        "candidate_factor_frame_sha256",
        "faro_depth_array_sha256",
        "candidate_depth_array_sha256",
        "scale_record",
        "scale_record_sha256",
        "factors",
        "content_sha256",
    }
    _require(set(record) == expected and record["schema"] == FACTOR_CANARY_RECORD_SCHEMA, "CANARY_RECORD_INVALID", "canary record schema/key set drift")
    _require(record["analysis_kind"] == ANALYSIS_KIND and record["claim_ceiling"] == CLAIM_CEILING, "CANARY_CLAIM_CEILING_DRIFT", "descriptive claim ceiling drift")
    for field in ("parent_id", "physical_frame_id", "query_id"):
        _require(isinstance(record[field], str) and bool(record[field]), "CANARY_IDENTITY_INVALID", "identity is required", field=field)
    for field in ("source_frame_receipt_sha256", "query_receipt_sha256", "base_geometry_sha256", "faro_factor_frame_sha256", "candidate_factor_frame_sha256", "faro_depth_array_sha256", "candidate_depth_array_sha256"):
        record[field] = _hash(record[field], field=field)
    _require(isinstance(record["max_source_timestamp_ns"], int) and not isinstance(record["max_source_timestamp_ns"], bool) and record["max_source_timestamp_ns"] >= 0, "CANARY_IDENTITY_INVALID", "watermark is invalid")
    _require(isinstance(record["factors"], dict) and set(record["factors"]) == set(FACTOR_NAMES), "CANARY_FACTOR_SET_INVALID", "factors must be SCALE, SUPPORT, BOUNDARY")
    factors = {name: _validate_factor_metrics(name, record["factors"][name]) for name in FACTOR_NAMES}
    lineage = {key: record[key] for key in ("physical_frame_id", "query_id", "faro_factor_frame_sha256", "candidate_factor_frame_sha256", "faro_depth_array_sha256", "candidate_depth_array_sha256")}
    if factors["SCALE"]["evaluable"]:
        scale = _validate_scale_record(record["scale_record"], lineage)
        _require(record["scale_record_sha256"] == scale["content_sha256"], "CANARY_SCALE_LINEAGE_MISMATCH", "nested scale hash binding drift")
        _require(abs(float(factors["SCALE"]["abs_log_correction"]) - abs(float(scale["log_metric_scale"]))) <= 1e-12, "CANARY_SCALE_LINEAGE_MISMATCH", "scale diagnostic differs from scale record")
        _require(abs(float(factors["SCALE"]["metric_scale"]) - float(scale["metric_scale"])) <= 1e-12 and factors["SCALE"]["valid_pair_count"] == scale["valid_pair_count"], "CANARY_SCALE_LINEAGE_MISMATCH", "scale diagnostics differ from scale record")
    else:
        _require(record["scale_record"] is None and record["scale_record_sha256"] is None, "CANARY_SCALE_LINEAGE_MISMATCH", "unevaluable scale carries a scale record")
    return record


def _metric_parent_macro(records: Sequence[dict[str, Any]], factor: str, metric: str, parents: Sequence[str]) -> dict[str, Any]:
    parent_rows: list[dict[str, Any]] = []
    usable_parent_medians: list[float] = []
    for parent in parents:
        values = [
            float(record["factors"][factor][metric])
            for record in records
            if record["parent_id"] == parent
            and record["factors"][factor]["evaluable"]
            and _finite_number(record["factors"][factor][metric])
        ]
        parent_value = float(np.median(np.asarray(values, dtype=np.float64))) if values else None
        if parent_value is not None:
            usable_parent_medians.append(parent_value)
        parent_rows.append(
            {
                "parent_id": parent,
                "value_count": len(values),
                "median": parent_value,
                "reason_codes": [] if values else ["NO_EVALUABLE_QUERY_METRIC"],
            }
        )
    return {
        "parent_medians": parent_rows,
        "parents_with_metric": len(usable_parent_medians),
        "median_of_parent_medians": float(np.median(np.asarray(usable_parent_medians, dtype=np.float64))) if usable_parent_medians else None,
        "reason_codes": [] if usable_parent_medians else ["NO_EVALUABLE_PARENT_METRIC"],
    }


def summarize_factor_canary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize query records by parent first, without decision thresholds."""

    _require(isinstance(records, Sequence) and not isinstance(records, (str, bytes)) and len(records) > 0, "CANARY_SUMMARY_EMPTY", "at least one canary record is required")
    validated = [validate_factor_canary_record(dict(record)) for record in records]
    identities = [(record["parent_id"], record["physical_frame_id"], record["query_id"]) for record in validated]
    _require(len(set(identities)) == len(identities), "CANARY_SUMMARY_DUPLICATE_QUERY", "a parent/query may appear only once")
    validated.sort(key=lambda record: (record["parent_id"], record["physical_frame_id"], record["query_id"]))
    parents = sorted({record["parent_id"] for record in validated})
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in validated:
        by_parent[record["parent_id"]].append(record)

    factor_summaries: dict[str, Any] = {}
    for factor in FACTOR_NAMES:
        parent_coverage: list[dict[str, Any]] = []
        coverage_values: list[float] = []
        for parent in parents:
            rows = by_parent[parent]
            evaluable = sum(bool(row["factors"][factor]["evaluable"]) for row in rows)
            fraction = evaluable / len(rows)
            coverage_values.append(fraction)
            parent_coverage.append(
                {
                    "parent_id": parent,
                    "query_count": len(rows),
                    "evaluable_query_count": evaluable,
                    "query_coverage": fraction,
                }
            )
        metrics = {
            metric: _metric_parent_macro(validated, factor, metric, parents)
            for metric in _FACTOR_METRIC_FIELDS[factor]
        }
        factor_summaries[factor] = {
            "query_coverage_parent_macro": {
                "parent_values": parent_coverage,
                "median_across_parents": float(np.median(np.asarray(coverage_values, dtype=np.float64))),
                "reason_codes": [],
            },
            "metrics_parent_macro": metrics,
        }

    return _seal(
        {
            "schema": FACTOR_CANARY_SUMMARY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "record_count": len(validated),
            "parent_count": len(parents),
            "factor_names": list(FACTOR_NAMES),
            "aggregation": "QUERY_METRIC_MEDIAN_WITHIN_PARENT_THEN_MEDIAN_ACROSS_PARENTS",
            "threshold_or_pass_fail_decision_applied": False,
            "factors": factor_summaries,
        }
    )


__all__ = [
    "ANALYSIS_KIND",
    "CLAIM_CEILING",
    "FACTOR_CANARY_RECORD_SCHEMA",
    "FACTOR_CANARY_SUMMARY_SCHEMA",
    "FactorCanaryError",
    "MINIMUM_BOUNDARY_INTERSECTION_POINTS",
    "build_factor_canary_record",
    "summarize_factor_canary",
    "validate_factor_canary_record",
]
