#!/usr/bin/env python3
"""Source-anchored TARO factor/query canary mechanics.

The source scale is re-derived from AppleDepth and a sealed DepthART replay;
callers cannot submit an already-scaled raster.  FARO is used only after that
source-only record is immutable, to build descriptive SUPPORT, BOUNDARY and
point-clearance diagnostics.  This module deliberately emits no formal state,
threshold, PASS/FAIL, product or safety claim.
"""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


RELIABILITY_SCHEMA = "blindassist.taro.o0r.apple_scale_reliability.v1"
QUERY_RECORD_SCHEMA = "blindassist.taro.o0r.source_anchored_factor_query_canary.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.source_anchored_factor_canary_summary.v1"
ANALYSIS_KIND = "POST_HOC_SOURCE_ANCHORED_FACTOR_AND_POINT_CLEARANCE_CANARY"
ANCHOR_ID = "DEPTHART_S_PLUS_APPLE_MEDIAN_LOG_SCALE_PRE_EXTRACTION_V1"
RELIABILITY_ID = "APPLE_C2_LOG_RATIO_MAD_Q95_AND_4X4_TILE_IQR_V1"
MINIMUM_TILE_PAIRS = 64
CLAIM_CEILING = {
    "scope": "LOCKED_ARKITSCENES_TRAIN_LANDSCAPE_ONLY",
    "use": "POST_HOC_DESCRIPTIVE_SOURCE_ANCHORED_FACTOR_AND_POINT_CLEARANCE_CANARY",
    "point_estimate_only": True,
    "threshold_or_pass_fail_decision": False,
    "excluded_claims": [
        "FORMAL_O0R_PASS",
        "UNCERTAINTY_CALIBRATED_STATE",
        "FINAL_TASK_EFFECTIVENESS",
        "DEPLOYMENT",
        "PRODUCT",
        "SAFETY",
    ],
}

_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")


class SourceFactorError(RuntimeError):
    """Stable fail-closed error for source-anchored factor mechanics."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise SourceFactorError(code, message, **context)


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _hash(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "SOURCE_FACTOR_HASH_INVALID", "SHA-256 binding is malformed", field=field)
    return value.upper()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    _require("content_sha256" not in output, "SOURCE_FACTOR_SEAL_COLLISION", "payload already contains a content seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "SOURCE_FACTOR_RECORD_INVALID", "sealed record must be an object")
    output = copy.deepcopy(value)
    observed = output.pop("content_sha256", None)
    _require(
        isinstance(observed, str)
        and bool(_SHA256.fullmatch(observed))
        and adapter.canonical_sha256(output) == observed.upper(),
        "SOURCE_FACTOR_SEAL_MISMATCH",
        "record seal drift",
    )
    output["content_sha256"] = observed.upper()
    _require(output.get("schema") == schema, "SOURCE_FACTOR_RECORD_INVALID", "record schema drift")
    return output


def _immutable(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class PreparedSourceCandidate:
    parent_id: str
    physical_frame_id: str
    raw_depth_m: np.ndarray
    anchored_depth_m: np.ndarray
    raw_depth_sha256: str
    anchored_depth_sha256: str
    metric_scale: float
    source_scale_record_sha256: str
    candidate_binding_sha256: str
    apple_source_receipt_sha256: str
    reliability: dict[str, Any]


@dataclass(frozen=True)
class QueryTruthBase:
    physical_frame_id: str
    query_id: str
    query_receipt: dict[str, Any]
    common_point_ids_uv: np.ndarray
    common_point_ids_sha256: str
    local_valid_fraction: float
    truth_normal_camera_xyz: np.ndarray
    truth_camera_height_m: float
    truth_query_support_points: int
    truth_observed_forward_m: float | None
    truth_boundary_point_ids_uv: np.ndarray
    truth_boundary_points_camera_xyz: np.ndarray


@dataclass(frozen=True)
class QueryExtraction:
    depth_sha256: str
    valid_common_point_count: int
    normal_camera_xyz: np.ndarray
    camera_height_m: float
    support_point_count: int
    support_fraction: float
    slope_degrees: float
    median_residual_m: float
    query_support_points: int
    observed_forward_m: float | None
    local_valid_fraction: float
    boundary_point_ids_uv: np.ndarray
    boundary_points_camera_xyz: np.ndarray


def _build_reliability_record(
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    candidate_highres_depth_m: np.ndarray,
    source_scale_record: Mapping[str, Any],
) -> dict[str, Any]:
    record = apple_scale.validate_source_scale_record(dict(source_scale_record))
    sampled = apple_scale.sample_candidate_at_apple_centers(candidate_highres_depth_m)
    apple_m = np.asarray(apple_depth_mm, dtype=np.float64) / 1000.0
    conf = np.asarray(confidence)
    lower, upper = apple_scale.DEPTH_RANGE_M
    valid = (conf == 2) & (apple_m >= lower) & (apple_m <= upper) & (sampled >= lower) & (sampled <= upper)
    _require(int(np.sum(valid)) == int(record["valid_pair_count"]), "SOURCE_SCALE_REDERIVATION_DRIFT", "source-scale valid-pair count drift")
    log_ratio = np.log(apple_m[valid] / sampled[valid])
    center = float(np.median(log_ratio))
    _require(abs(center - float(record["log_metric_scale"])) <= 5e-13, "SOURCE_SCALE_REDERIVATION_DRIFT", "source-scale median drift")
    absolute_deviation = np.abs(log_ratio - center)
    tile_medians: list[float] = []
    height, width = adapter.APPLE_SHAPE_HW
    for tile_row in range(4):
        row_slice = slice(tile_row * height // 4, (tile_row + 1) * height // 4)
        for tile_column in range(4):
            column_slice = slice(tile_column * width // 4, (tile_column + 1) * width // 4)
            tile_valid = valid[row_slice, column_slice]
            if int(np.sum(tile_valid)) >= MINIMUM_TILE_PAIRS:
                tile_log_ratio = np.log(
                    apple_m[row_slice, column_slice][tile_valid]
                    / sampled[row_slice, column_slice][tile_valid]
                )
                tile_medians.append(float(np.median(tile_log_ratio)))
    tile_array = np.asarray(tile_medians, dtype=np.float64)
    tile_iqr = float(np.quantile(tile_array, 0.75, method="linear") - np.quantile(tile_array, 0.25, method="linear")) if len(tile_array) >= 4 else None
    return _seal(
        {
            "schema": RELIABILITY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "reliability_id": RELIABILITY_ID,
            "parent_id": record["parent_id"],
            "physical_frame_id": record["physical_frame_id"],
            "source_scale_record_sha256": record["content_sha256"],
            "candidate_highres_depth_array_sha256": record["candidate_highres_depth_array_sha256"],
            "valid_pair_count": int(record["valid_pair_count"]),
            "valid_pair_fraction": float(np.mean(valid)),
            "log_ratio_mad": float(np.median(absolute_deviation)),
            "log_ratio_q95_abs_deviation": float(np.quantile(absolute_deviation, 0.95, method="linear")),
            "tile_grid": [4, 4],
            "minimum_tile_pairs": MINIMUM_TILE_PAIRS,
            "occupied_tile_count": len(tile_medians),
            "tile_median_log_ratio_iqr": tile_iqr,
            "abstention_threshold_selected": False,
        }
    )


def validate_reliability_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, RELIABILITY_SCHEMA)
    _require(record.get("analysis_kind") == ANALYSIS_KIND and record.get("claim_ceiling") == CLAIM_CEILING, "SOURCE_FACTOR_CLAIM_DRIFT", "reliability claim ceiling drift")
    _require(record.get("reliability_id") == RELIABILITY_ID and record.get("abstention_threshold_selected") is False, "SOURCE_RELIABILITY_METHOD_DRIFT", "reliability method drift")
    for field in ("source_scale_record_sha256", "candidate_highres_depth_array_sha256"):
        _hash(record.get(field), field=field)
    for field in ("valid_pair_fraction", "log_ratio_mad", "log_ratio_q95_abs_deviation"):
        _require(_finite(record.get(field)) and float(record[field]) >= 0.0, "SOURCE_RELIABILITY_VALUE_INVALID", "reliability value is invalid", field=field)
    _require(isinstance(record.get("occupied_tile_count"), int) and 0 <= record["occupied_tile_count"] <= 16, "SOURCE_RELIABILITY_VALUE_INVALID", "occupied tile count is invalid")
    _require(record.get("tile_median_log_ratio_iqr") is None or (_finite(record["tile_median_log_ratio_iqr"]) and float(record["tile_median_log_ratio_iqr"]) >= 0.0), "SOURCE_RELIABILITY_VALUE_INVALID", "tile IQR is invalid")
    return record


def prepare_source_anchored_candidate(
    candidate_highres_depth_m: np.ndarray,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    source_frame_receipt: Mapping[str, Any],
    apple_source_receipt: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    source_scale_record: Mapping[str, Any],
) -> PreparedSourceCandidate:
    """Validate the sealed source estimate and internally create scaled depth."""

    source = adapter._validate_base_receipt(dict(source_frame_receipt))
    apple_receipt = apple_scale.validate_apple_scale_source_receipt(dict(apple_source_receipt), np.asarray(apple_depth_mm), np.asarray(confidence))
    binding = apple_scale.validate_candidate_replay_binding(dict(candidate_binding))
    scale_record = apple_scale.validate_source_scale_record(dict(source_scale_record))
    raw = np.asarray(candidate_highres_depth_m)
    _require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype.kind == "f" and bool(np.all(np.isfinite(raw))), "SOURCE_CANDIDATE_DEPTH_INVALID", "candidate must be finite floating-point 1440x1920 metres")
    raw_hash = adapter.canonical_sha256(raw)
    identity = (str(source["parent_id"]), str(source["session_id"]), str(source["sensor_timestamp"]["decimal_token"]), str(source["physical_frame_id"]))
    expected = (binding["parent_id"], binding["video_id"], binding["timestamp_token"], binding["physical_frame_id"])
    _require(identity == expected, "SOURCE_FACTOR_IDENTITY_MISMATCH", "source frame and candidate replay identities differ")
    _require(identity == (scale_record["parent_id"], scale_record["video_id"], scale_record["timestamp_token"], scale_record["physical_frame_id"]), "SOURCE_FACTOR_IDENTITY_MISMATCH", "source scale and source frame identities differ")
    _require(identity == (apple_receipt["parent_id"], apple_receipt["video_id"], apple_receipt["timestamp_token"], apple_receipt["physical_frame_id"]), "SOURCE_FACTOR_IDENTITY_MISMATCH", "Apple source and source frame identities differ")
    _require(raw_hash == binding["highres_depth_array_sha256"] == scale_record["candidate_highres_depth_array_sha256"], "SOURCE_FACTOR_CANDIDATE_HASH_DRIFT", "raw candidate differs from sealed replay/scale record")
    _require(scale_record["candidate_binding_sha256"] == binding["content_sha256"] and scale_record["source_receipt_sha256"] == apple_receipt["content_sha256"], "SOURCE_FACTOR_LINEAGE_DRIFT", "source scale parent lineage drift")
    _require(scale_record["evaluable"] is True, "SOURCE_SCALE_NOT_EVALUABLE", "source scale is unavailable", reasons=scale_record["reason_codes"])
    rebuilt = apple_scale.build_source_scale_record(
        np.asarray(apple_depth_mm),
        np.asarray(confidence),
        raw,
        apple_receipt,
        binding,
    )
    _require(rebuilt["content_sha256"] == scale_record["content_sha256"], "SOURCE_SCALE_REDERIVATION_DRIFT", "stored source scale is not exactly re-derived from bound source/candidate")
    metric_scale = float(scale_record["metric_scale"])
    anchored = np.ascontiguousarray(raw.astype(np.float64) * metric_scale, dtype=np.float64)
    reliability = validate_reliability_record(_build_reliability_record(np.asarray(apple_depth_mm), np.asarray(confidence), raw, scale_record))
    return PreparedSourceCandidate(
        parent_id=identity[0],
        physical_frame_id=identity[3],
        raw_depth_m=_immutable(raw, raw.dtype),
        anchored_depth_m=_immutable(anchored, np.float64),
        raw_depth_sha256=raw_hash,
        anchored_depth_sha256=adapter.canonical_sha256(anchored),
        metric_scale=metric_scale,
        source_scale_record_sha256=scale_record["content_sha256"],
        candidate_binding_sha256=binding["content_sha256"],
        apple_source_receipt_sha256=apple_receipt["content_sha256"],
        reliability=reliability,
    )


def build_query_truth_base(geometry: adapter.FaroGeometry, query_receipt: Mapping[str, Any]) -> QueryTruthBase:
    """Build one query-local FARO comparison surface after one frame validation."""

    query = adapter._validate_query_receipt(dict(query_receipt))
    _require(query["physical_frame_id"] == geometry.physical_frame_id, "SOURCE_FACTOR_QUERY_IDENTITY_MISMATCH", "query and FARO frame differ")
    surface_points, point_ids, local_fraction = adapter._local_surface(geometry, query)
    _require(len(surface_points) > 0, "QUERY_LOCAL_SURFACE_EMPTY", "query has no local FARO surface")
    truth_query = adapter._query_support_and_boundary(
        surface_points,
        point_ids,
        geometry.support_normal_camera_xyz,
        geometry.camera_height_m,
        query,
    )
    return QueryTruthBase(
        physical_frame_id=geometry.physical_frame_id,
        query_id=query["query_id"],
        query_receipt=query,
        common_point_ids_uv=_immutable(point_ids, np.int32),
        common_point_ids_sha256=adapter.canonical_sha256(point_ids),
        local_valid_fraction=float(local_fraction),
        truth_normal_camera_xyz=_immutable(geometry.support_normal_camera_xyz, np.float64),
        truth_camera_height_m=float(geometry.camera_height_m),
        truth_query_support_points=int(truth_query["query_support_points"]),
        truth_observed_forward_m=None if truth_query["observed_forward_shape_m"] is None else float(truth_query["observed_forward_shape_m"]),
        truth_boundary_point_ids_uv=_immutable(truth_query["boundary_point_ids_uv"], np.int32),
        truth_boundary_points_camera_xyz=_immutable(truth_query["boundary_points_shape_camera_xyz"], np.float64),
    )


def _extract_query(depth_m: np.ndarray, depth_sha256: str, intrinsics: np.ndarray, gravity: np.ndarray, base: QueryTruthBase) -> QueryExtraction:
    point_ids = np.asarray(base.common_point_ids_uv, dtype=np.int64)
    z_all = np.asarray(depth_m, dtype=np.float64)[point_ids[:, 1], point_ids[:, 0]]
    valid = (z_all >= adapter.DEPTH_RANGE_M[0]) & (z_all <= adapter.DEPTH_RANGE_M[1])
    valid_count = int(np.sum(valid))
    _require(valid_count >= adapter.MINIMUM_SUPPORT_POINTS, "CANDIDATE_COMMON_SUPPORT_INSUFFICIENT", "candidate has too few valid points on frozen truth support", valid_count=valid_count)
    valid_ids = point_ids[valid].astype(np.int32, copy=False)
    z = z_all[valid]
    points = np.stack(
        (
            (valid_ids[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (valid_ids[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ),
        axis=1,
    )
    stride = (valid_ids[:, 0] % adapter.SUPPORT_POINT_STRIDE == 0) & (valid_ids[:, 1] % adapter.SUPPORT_POINT_STRIDE == 0)
    support_input = points[stride]
    if len(support_input) < adapter.MINIMUM_SUPPORT_POINTS:
        support_input = points[:: adapter.SUPPORT_POINT_STRIDE]
    try:
        plane = adapter._fit_support_plane(support_input, gravity)
        query_geometry = adapter._query_support_and_boundary(
            points,
            valid_ids,
            plane["normal_camera_xyz"],
            float(plane["camera_height_m"]),
            base.query_receipt,
        )
    except adapter.AdapterError as error:
        raise SourceFactorError(error.code, str(error), **error.context) from error
    local_fraction = float(base.local_valid_fraction) * valid_count / float(len(valid))
    return QueryExtraction(
        depth_sha256=depth_sha256,
        valid_common_point_count=valid_count,
        normal_camera_xyz=_immutable(plane["normal_camera_xyz"], np.float64),
        camera_height_m=float(plane["camera_height_m"]),
        support_point_count=int(plane["support_count"]),
        support_fraction=float(plane["support_fraction"]),
        slope_degrees=float(plane["slope_degrees"]),
        median_residual_m=float(plane["median_residual_m"]),
        query_support_points=int(query_geometry["query_support_points"]),
        observed_forward_m=None if query_geometry["observed_forward_shape_m"] is None else float(query_geometry["observed_forward_shape_m"]),
        local_valid_fraction=local_fraction,
        boundary_point_ids_uv=_immutable(query_geometry["boundary_point_ids_uv"], np.int32),
        boundary_points_camera_xyz=_immutable(query_geometry["boundary_points_shape_camera_xyz"], np.float64),
    )


def _normal_angle(left: np.ndarray, right: np.ndarray) -> float:
    first = np.array(left, dtype=np.float64, copy=True)
    second = np.array(right, dtype=np.float64, copy=True)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def _point_clearance(
    normal_camera_xyz: np.ndarray,
    camera_height_m: float,
    boundary_points_camera_xyz: np.ndarray,
    query_support_points: int,
    observed_forward_m: float | None,
    local_valid_fraction: float,
    query_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        query_support_points < adapter.MINIMUM_QUERY_SUPPORT_POINTS
        or observed_forward_m is None
        or observed_forward_m < adapter.MINIMUM_QUERY_OBSERVED_FORWARD_M
        or local_valid_fraction < adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
    ):
        return {
            "evaluable": False,
            "reason_codes": ["QUERY_KNOWNNESS_GATE_FAILED"],
            "value_m": None,
            "query_support_points": int(query_support_points),
            "observed_forward_m": observed_forward_m,
            "local_valid_fraction": float(local_valid_fraction),
        }
    query = dict(query_receipt)
    normal = np.array(normal_camera_xyz, dtype=np.float64, copy=True)
    normal /= np.linalg.norm(normal)
    _, _, _, receipt_heading = adapter._query_receipt_vectors(query)
    heading = receipt_heading - float(np.dot(receipt_heading, normal)) * normal
    heading /= np.linalg.norm(heading)
    lateral = np.cross(heading, normal)
    lateral /= np.linalg.norm(lateral)
    path_origin = -float(camera_height_m) * normal + float(query["path_lateral_offset_m"]) * lateral
    points = np.asarray(boundary_points_camera_xyz, dtype=np.float64)
    rel = points - path_origin
    along = rel @ heading
    eligible = (along >= adapter.MINIMUM_FORWARD_M) & (along <= adapter.HORIZON_M + adapter.GEOMETRY_ENDPOINT_TOLERANCE_M)
    if bool(np.any(eligible)):
        t = np.clip(along[eligible], adapter.MINIMUM_FORWARD_M, adapter.HORIZON_M)
        closest = path_origin[None, :] + t[:, None] * heading[None, :]
        displacement = points[eligible] - closest
        displacement_ground = displacement - (displacement @ normal)[:, None] * normal[None, :]
        value_m = min(adapter.HORIZON_M, float(np.min(np.linalg.norm(displacement_ground, axis=1))) - adapter.CAPSULE_RADIUS_M)
    else:
        value_m = adapter.HORIZON_M
    return {
        "evaluable": True,
        "reason_codes": [],
        "value_m": value_m,
        "query_support_points": int(query_support_points),
        "observed_forward_m": float(observed_forward_m),
        "local_valid_fraction": float(local_valid_fraction),
    }


def _boundary_metrics(base: QueryTruthBase, extraction: QueryExtraction) -> dict[str, Any]:
    reasons: list[str] = []
    if base.local_valid_fraction < adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION:
        reasons.append("TRUTH_BOUNDARY_INVALID")
    if extraction.local_valid_fraction < adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION:
        reasons.append("CANDIDATE_BOUNDARY_INVALID")
    truth_ids = np.asarray(base.truth_boundary_point_ids_uv, dtype=np.int64)
    candidate_ids = np.asarray(extraction.boundary_point_ids_uv, dtype=np.int64)
    truth_linear = truth_ids[:, 1] * adapter.HIGHRES_SHAPE_HW[1] + truth_ids[:, 0]
    candidate_linear = candidate_ids[:, 1] * adapter.HIGHRES_SHAPE_HW[1] + candidate_ids[:, 0]
    _, truth_index, candidate_index = np.intersect1d(truth_linear, candidate_linear, assume_unique=True, return_indices=True)
    intersection = len(truth_index)
    union = len(truth_linear) + len(candidate_linear) - intersection
    jaccard = intersection / union if union else None
    xyz_error = None
    if intersection < 3:
        reasons.append("BOUNDARY_COMMON_POINT_IDS_INSUFFICIENT")
    else:
        truth_points = np.asarray(base.truth_boundary_points_camera_xyz, dtype=np.float64)[truth_index]
        candidate_points = np.asarray(extraction.boundary_points_camera_xyz, dtype=np.float64)[candidate_index]
        xyz_error = float(np.median(np.linalg.norm(truth_points - candidate_points, axis=1)))
    return {
        "evaluable": not reasons,
        "reason_codes": reasons,
        "truth_point_count": int(len(truth_linear)),
        "candidate_point_count": int(len(candidate_linear)),
        "point_id_intersection_count": int(intersection),
        "point_id_union_count": int(union),
        "point_id_jaccard": jaccard,
        "xyz_median_error_m": xyz_error,
        "local_valid_fraction": float(extraction.local_valid_fraction),
    }


def _mode_result(base: QueryTruthBase, extraction: QueryExtraction) -> dict[str, Any]:
    support = {
        "evaluable": True,
        "reason_codes": [],
        "normal_angular_error_rad": _normal_angle(base.truth_normal_camera_xyz, extraction.normal_camera_xyz),
        "height_abs_error_m": abs(float(base.truth_camera_height_m) - float(extraction.camera_height_m)),
        "camera_height_m": float(extraction.camera_height_m),
        "support_point_count": int(extraction.support_point_count),
        "support_fraction": float(extraction.support_fraction),
        "slope_degrees": float(extraction.slope_degrees),
    }
    boundary = _boundary_metrics(base, extraction)
    query = _point_clearance(
        extraction.normal_camera_xyz,
        extraction.camera_height_m,
        extraction.boundary_points_camera_xyz,
        extraction.query_support_points,
        extraction.observed_forward_m,
        extraction.local_valid_fraction,
        base.query_receipt,
    )
    truth_query = _point_clearance(
        base.truth_normal_camera_xyz,
        base.truth_camera_height_m,
        base.truth_boundary_points_camera_xyz,
        base.truth_query_support_points,
        base.truth_observed_forward_m,
        base.local_valid_fraction,
        base.query_receipt,
    )
    query["truth_value_m"] = truth_query["value_m"]
    query["abs_error_m"] = abs(float(query["value_m"]) - float(truth_query["value_m"])) if query["evaluable"] and truth_query["evaluable"] else None
    if truth_query["evaluable"] is False and "TRUTH_QUERY_NOT_EVALUABLE" not in query["reason_codes"]:
        query["evaluable"] = False
        query["reason_codes"] = [*query["reason_codes"], "TRUTH_QUERY_NOT_EVALUABLE"]
        query["abs_error_m"] = None
    return {
        "extraction_evaluable": True,
        "reason_codes": [],
        "depth_array_sha256": extraction.depth_sha256,
        "valid_common_point_count": int(extraction.valid_common_point_count),
        "support": support,
        "boundary": boundary,
        "query_point_clearance": query,
    }


def _failed_mode(depth_sha256: str, code: str) -> dict[str, Any]:
    return {
        "extraction_evaluable": False,
        "reason_codes": [code],
        "depth_array_sha256": depth_sha256,
        "valid_common_point_count": None,
        "support": {"evaluable": False, "reason_codes": [code], "normal_angular_error_rad": None, "height_abs_error_m": None, "camera_height_m": None, "support_point_count": None, "support_fraction": None, "slope_degrees": None},
        "boundary": {"evaluable": False, "reason_codes": [code], "truth_point_count": None, "candidate_point_count": None, "point_id_intersection_count": None, "point_id_union_count": None, "point_id_jaccard": None, "xyz_median_error_m": None, "local_valid_fraction": None},
        "query_point_clearance": {"evaluable": False, "reason_codes": [code], "value_m": None, "truth_value_m": None, "abs_error_m": None, "query_support_points": 0, "observed_forward_m": None, "local_valid_fraction": None},
    }


def _difference(baseline: Mapping[str, Any], anchored: Mapping[str, Any], factor: str, metric: str, *, higher_is_better: bool = False) -> float | None:
    left = baseline[factor].get(metric)
    right = anchored[factor].get(metric)
    if not (baseline[factor].get("evaluable") and anchored[factor].get("evaluable") and _finite(left) and _finite(right)):
        return None
    return float(right) - float(left) if higher_is_better else float(left) - float(right)


def evaluate_source_anchored_query(
    prepared: PreparedSourceCandidate,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    base: QueryTruthBase,
    *,
    current_faro_geometry_sha256: str,
    compact_truth_record_sha256: str,
    committed_faro_geometry_sha256: str,
    committed_factor_frame_sha256: str,
    committed_base_geometry_sha256: str,
    compact_query_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare raw and internally source-scaled extraction for one query."""

    _require(prepared.physical_frame_id == base.physical_frame_id, "SOURCE_FACTOR_QUERY_IDENTITY_MISMATCH", "prepared candidate and query frame differ")
    matrix = adapter._intrinsics_matrix(intrinsics_highres_3x3)
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "GRAVITY_INVALID")
    modes: dict[str, dict[str, Any]] = {}
    for name, depth, depth_hash in (
        ("baseline", prepared.raw_depth_m, prepared.raw_depth_sha256),
        ("source_anchored", prepared.anchored_depth_m, prepared.anchored_depth_sha256),
    ):
        try:
            modes[name] = _mode_result(base, _extract_query(depth, depth_hash, matrix, gravity, base))
        except SourceFactorError as error:
            modes[name] = _failed_mode(depth_hash, error.code)
    baseline, anchored = modes["baseline"], modes["source_anchored"]
    compact = dict(compact_query_result)
    _require(compact.get("physical_frame_id") == base.physical_frame_id and compact.get("query_id") == base.query_id, "COMPACT_QUERY_RESULT_IDENTITY_DRIFT", "compact truth result identity drift")
    current_truth = _point_clearance(
        base.truth_normal_camera_xyz,
        base.truth_camera_height_m,
        base.truth_boundary_points_camera_xyz,
        base.truth_query_support_points,
        base.truth_observed_forward_m,
        base.local_valid_fraction,
        base.query_receipt,
    )
    compact_value = compact.get("value_m")
    current_vs_compact = abs(float(current_truth["value_m"]) - float(compact_value)) if current_truth["evaluable"] and _finite(compact_value) else None
    record = _seal(
        {
            "schema": QUERY_RECORD_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "anchor_id": ANCHOR_ID,
            "parent_id": prepared.parent_id,
            "physical_frame_id": base.physical_frame_id,
            "query_id": base.query_id,
            "query_receipt_sha256": base.query_receipt["content_sha256"],
            "compact_truth_record_sha256": _hash(compact_truth_record_sha256, field="compact_truth_record_sha256"),
            "committed_faro_geometry_sha256": _hash(committed_faro_geometry_sha256, field="committed_faro_geometry_sha256"),
            "current_faro_geometry_sha256": _hash(current_faro_geometry_sha256, field="current_faro_geometry_sha256"),
            "committed_factor_frame_sha256": _hash(committed_factor_frame_sha256, field="committed_factor_frame_sha256"),
            "committed_base_geometry_sha256": _hash(committed_base_geometry_sha256, field="committed_base_geometry_sha256"),
            "current_common_point_ids_sha256": base.common_point_ids_sha256,
            "runtime_geometry_matches_r3_commitment": current_faro_geometry_sha256.upper() == committed_faro_geometry_sha256.upper(),
            "candidate_binding_sha256": prepared.candidate_binding_sha256,
            "source_scale_record_sha256": prepared.source_scale_record_sha256,
            "apple_source_receipt_sha256": prepared.apple_source_receipt_sha256,
            "reliability_record_sha256": prepared.reliability["content_sha256"],
            "source_metric_scale": prepared.metric_scale,
            "scale_applied_before_depth_range_support_boundary_and_query_extraction": True,
            "scale_factor_block_after_pre_extraction_application": 0.0,
            "truth_alignment_used_for_scale": False,
            "baseline": baseline,
            "source_anchored": anchored,
            "effects": {
                "extraction_recovered": baseline["extraction_evaluable"] is False and anchored["extraction_evaluable"] is True,
                "extraction_lost": baseline["extraction_evaluable"] is True and anchored["extraction_evaluable"] is False,
                "support_normal_error_reduction_rad": _difference(baseline, anchored, "support", "normal_angular_error_rad"),
                "support_height_error_reduction_m": _difference(baseline, anchored, "support", "height_abs_error_m"),
                "boundary_jaccard_increase": _difference(baseline, anchored, "boundary", "point_id_jaccard", higher_is_better=True),
                "boundary_xyz_error_reduction_m": _difference(baseline, anchored, "boundary", "xyz_median_error_m"),
                "query_point_error_reduction_m": _difference(baseline, anchored, "query_point_clearance", "abs_error_m"),
                "boundary_evaluability_recovered": baseline["boundary"]["evaluable"] is False and anchored["boundary"]["evaluable"] is True,
                "query_knownness_recovered": baseline["query_point_clearance"]["evaluable"] is False and anchored["query_point_clearance"]["evaluable"] is True,
            },
            "truth_point_clearance": current_truth,
            "compact_r3_query_result_sha256": adapter.canonical_sha256(compact),
            "compact_r3_value_m": compact_value,
            "current_truth_vs_compact_value_abs_delta_m": current_vs_compact,
            "formal_reducer_executed": False,
            "uncertainty_state_claimed": False,
        }
    )
    return validate_query_record(record)


def validate_query_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, QUERY_RECORD_SCHEMA)
    _require(record.get("analysis_kind") == ANALYSIS_KIND and record.get("claim_ceiling") == CLAIM_CEILING and record.get("anchor_id") == ANCHOR_ID, "SOURCE_FACTOR_CLAIM_DRIFT", "query record claim/method drift")
    _require(record.get("scale_applied_before_depth_range_support_boundary_and_query_extraction") is True and record.get("truth_alignment_used_for_scale") is False, "SOURCE_FACTOR_METHOD_DRIFT", "pre-extraction source-scale rule drift")
    _require(record.get("formal_reducer_executed") is False and record.get("uncertainty_state_claimed") is False, "SOURCE_FACTOR_CLAIM_DRIFT", "point-only canary claims a formal reducer/state")
    for field in (
        "query_receipt_sha256", "compact_truth_record_sha256", "committed_faro_geometry_sha256",
        "current_faro_geometry_sha256", "committed_factor_frame_sha256", "committed_base_geometry_sha256",
        "current_common_point_ids_sha256", "candidate_binding_sha256", "source_scale_record_sha256",
        "apple_source_receipt_sha256", "reliability_record_sha256", "compact_r3_query_result_sha256",
    ):
        _hash(record.get(field), field=field)
    _require(_finite(record.get("source_metric_scale")) and float(record["source_metric_scale"]) > 0.0, "SOURCE_FACTOR_SCALE_INVALID", "source scale is invalid")
    _require(isinstance(record.get("baseline"), dict) and isinstance(record.get("source_anchored"), dict) and isinstance(record.get("effects"), dict), "SOURCE_FACTOR_RECORD_INVALID", "query record mode/effect payload missing")
    return record


def _nested(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = row
    for key in path:
        value = value[key]
    return value


def _parent_macro(records: Sequence[dict[str, Any]], mode: str, factor: str, metric: str) -> dict[str, Any]:
    frame_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        block = row[mode][factor]
        value = block.get(metric)
        if block.get("evaluable") is True and _finite(value):
            frame_values[(row["parent_id"], row["physical_frame_id"])].append(float(value))
    parent_values: dict[str, list[float]] = defaultdict(list)
    for (parent_id, _), values in frame_values.items():
        parent_values[parent_id].append(float(np.median(np.asarray(values, dtype=np.float64))))
    rows: list[dict[str, Any]] = []
    usable: list[float] = []
    for parent_id in sorted({row["parent_id"] for row in records}):
        values = parent_values.get(parent_id, [])
        median = float(np.median(np.asarray(values, dtype=np.float64))) if values else None
        if median is not None:
            usable.append(median)
        rows.append({"parent_id": parent_id, "frame_count": len(values), "median": median})
    return {"parent_medians": rows, "parents_with_metric": len(usable), "median_of_parent_medians": float(np.median(np.asarray(usable, dtype=np.float64))) if usable else None}


def _effect_parent_macro(records: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    frame_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        value = row["effects"].get(metric)
        if _finite(value):
            frame_values[(row["parent_id"], row["physical_frame_id"])].append(float(value))
    parent_values: dict[str, list[float]] = defaultdict(list)
    for (parent_id, _), values in frame_values.items():
        parent_values[parent_id].append(float(np.median(np.asarray(values, dtype=np.float64))))
    usable: list[float] = []
    parent_rows: list[dict[str, Any]] = []
    for parent_id in sorted({row["parent_id"] for row in records}):
        values = parent_values.get(parent_id, [])
        median = float(np.median(np.asarray(values, dtype=np.float64))) if values else None
        if median is not None:
            usable.append(median)
        parent_rows.append({"parent_id": parent_id, "frame_count": len(values), "median": median})
    return {"parent_medians": parent_rows, "parents_with_metric": len(usable), "median_of_parent_medians": float(np.median(np.asarray(usable, dtype=np.float64))) if usable else None}


def _coverage(records: Sequence[dict[str, Any]], mode: str, factor: str) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_parent[row["parent_id"]].append(row)
    parent_rows: list[dict[str, Any]] = []
    fractions: list[float] = []
    for parent_id in sorted(by_parent):
        rows = by_parent[parent_id]
        count = sum(bool(row[mode][factor]["evaluable"]) for row in rows)
        fraction = count / len(rows)
        fractions.append(fraction)
        parent_rows.append({"parent_id": parent_id, "query_count": len(rows), "evaluable_query_count": count, "query_coverage": fraction})
    return {"evaluable_query_count": sum(row["evaluable_query_count"] for row in parent_rows), "parent_values": parent_rows, "median_across_parents": float(np.median(np.asarray(fractions, dtype=np.float64)))}


def summarize_source_anchored_canary(records: Sequence[Mapping[str, Any]], reliability_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validated = [validate_query_record(dict(row)) for row in records]
    reliability = [validate_reliability_record(dict(row)) for row in reliability_records]
    _require(bool(validated) and bool(reliability), "SOURCE_FACTOR_SUMMARY_EMPTY", "query and reliability records are required")
    identities = [(row["physical_frame_id"], row["query_id"]) for row in validated]
    _require(len(set(identities)) == len(identities), "SOURCE_FACTOR_DUPLICATE_QUERY", "query records are duplicated")
    _require(len({row["physical_frame_id"] for row in reliability}) == len(reliability), "SOURCE_FACTOR_DUPLICATE_RELIABILITY", "reliability records are duplicated")
    metric_specs = {
        "support": ("normal_angular_error_rad", "height_abs_error_m"),
        "boundary": ("point_id_jaccard", "xyz_median_error_m"),
        "query_point_clearance": ("abs_error_m",),
    }
    modes: dict[str, Any] = {}
    for mode in ("baseline", "source_anchored"):
        modes[mode] = {
            factor: {
                "coverage": _coverage(validated, mode, factor),
                "metrics_parent_macro": {metric: _parent_macro(validated, mode, factor, metric) for metric in metrics},
            }
            for factor, metrics in metric_specs.items()
        }
    effects = {
        metric: _effect_parent_macro(validated, metric)
        for metric in (
            "support_normal_error_reduction_rad",
            "support_height_error_reduction_m",
            "boundary_jaccard_increase",
            "boundary_xyz_error_reduction_m",
            "query_point_error_reduction_m",
        )
    }
    frame_flags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        frame_flags[row["physical_frame_id"]].append(row)
    recovered_frames = sum(any(row["effects"]["extraction_recovered"] for row in rows) for rows in frame_flags.values())
    lost_frames = sum(any(row["effects"]["extraction_lost"] for row in rows) for rows in frame_flags.values())
    return _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "anchor_id": ANCHOR_ID,
            "aggregation": "QUERY_MEDIAN_WITHIN_FRAME_THEN_FRAME_MEDIAN_WITHIN_PARENT_THEN_MEDIAN_ACROSS_PARENTS",
            "query_record_count": len(validated),
            "physical_frame_count": len(frame_flags),
            "parent_count": len({row["parent_id"] for row in validated}),
            "reliability_frame_count": len(reliability),
            "runtime_geometry_match_query_count": sum(bool(row["runtime_geometry_matches_r3_commitment"]) for row in validated),
            "extraction_recovered_query_count": sum(bool(row["effects"]["extraction_recovered"]) for row in validated),
            "extraction_lost_query_count": sum(bool(row["effects"]["extraction_lost"]) for row in validated),
            "extraction_recovered_frame_count": recovered_frames,
            "extraction_lost_frame_count": lost_frames,
            "boundary_evaluability_recovered_query_count": sum(bool(row["effects"]["boundary_evaluability_recovered"]) for row in validated),
            "query_knownness_recovered_query_count": sum(bool(row["effects"]["query_knownness_recovered"]) for row in validated),
            "modes": modes,
            "effects_parent_macro": effects,
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def summarize_reliability_association(reliability_records: Sequence[Mapping[str, Any]], frame_oracle_abs_log_errors: Mapping[str, float]) -> dict[str, Any]:
    reliability = [validate_reliability_record(dict(row)) for row in reliability_records]
    pairs = [row for row in reliability if row["physical_frame_id"] in frame_oracle_abs_log_errors]
    metrics = ("log_ratio_mad", "log_ratio_q95_abs_deviation", "tile_median_log_ratio_iqr")
    associations: dict[str, Any] = {}
    for metric in metrics:
        rows = [row for row in pairs if _finite(row.get(metric))]
        x = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        y = np.asarray([float(frame_oracle_abs_log_errors[row["physical_frame_id"]]) for row in rows], dtype=np.float64)
        if len(rows) >= 3 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
            rho = float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])
        else:
            rho = None
        associations[metric] = {"paired_frame_count": len(rows), "spearman_rho_with_source_abs_log_error": rho}
    worst = sorted(
        (
            {
                "parent_id": row["parent_id"],
                "physical_frame_id": row["physical_frame_id"],
                "source_abs_log_error": float(frame_oracle_abs_log_errors[row["physical_frame_id"]]),
                "log_ratio_mad": float(row["log_ratio_mad"]),
                "log_ratio_q95_abs_deviation": float(row["log_ratio_q95_abs_deviation"]),
                "tile_median_log_ratio_iqr": row["tile_median_log_ratio_iqr"],
            }
            for row in pairs
        ),
        key=lambda row: (-row["source_abs_log_error"], row["physical_frame_id"]),
    )[:10]
    return _seal(
        {
            "schema": "blindassist.taro.o0r.apple_scale_reliability_association.v1",
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "paired_frame_count": len(pairs),
            "associations": associations,
            "worst_source_scale_frames": worst,
            "post_hoc_only": True,
            "abstention_threshold_selected": False,
        }
    )


__all__ = [
    "ANALYSIS_KIND",
    "ANCHOR_ID",
    "CLAIM_CEILING",
    "PreparedSourceCandidate",
    "QueryTruthBase",
    "SourceFactorError",
    "build_query_truth_base",
    "evaluate_source_anchored_query",
    "prepare_source_anchored_candidate",
    "summarize_reliability_association",
    "summarize_source_anchored_canary",
    "validate_query_record",
    "validate_reliability_record",
]
