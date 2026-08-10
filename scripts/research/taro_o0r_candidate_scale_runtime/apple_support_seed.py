#!/usr/bin/env python3
"""Source-only AppleDepth-seeded support recovery for TARO R1 losses.

The recovery plane is fitted once per physical frame before any FARO/query
payload is joined.  FARO-derived query surfaces are admitted only later for a
descriptive comparison.  Every failed source gate remains UNKNOWN.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


PLANE_SCHEMA = "blindassist.taro.o0r.apple_seeded_candidate_support_plane.v1"
RECOVERY_RECORD_SCHEMA = "blindassist.taro.o0r.apple_seeded_support_recovery_query.v1"
RECOVERY_SUMMARY_SCHEMA = "blindassist.taro.o0r.apple_seeded_support_recovery_summary.v1"
SEED_ID = "R0_C2_PAIR_APPLE_PLANE_TO_SOURCE_SCALED_CANDIDATE_REFIT_V1"
ANALYSIS_KIND = "POST_HOC_APPLE_SEEDED_SUPPORT_RECOVERY_CANARY"
CLAIM_CEILING = {
    "scope": "R1_EXTRACTION_LOST_FRAMES_ONLY_LOCKED_ARKITSCENES_TRAIN_LANDSCAPE",
    "use": "POST_HOC_DESCRIPTIVE_SOURCE_ONLY_SUPPORT_RECOVERY_CANARY",
    "retrospective_cohort": True,
    "threshold_or_pass_fail_decision": False,
    "excluded_claims": ["FORMAL_O0R_PASS", "DEPLOYMENT", "PRODUCT", "SAFETY"],
}
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")


class AppleSupportSeedError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise AppleSupportSeedError(code, message, **context)


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _hash(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "APPLE_SUPPORT_HASH_INVALID", "SHA-256 binding is malformed", field=field)
    return str(value).upper()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    _require("content_sha256" not in output, "APPLE_SUPPORT_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "APPLE_SUPPORT_RECORD_INVALID", "record must be an object")
    output = copy.deepcopy(value)
    observed = output.pop("content_sha256", None)
    _require(
        isinstance(observed, str)
        and bool(_SHA256.fullmatch(observed))
        and adapter.canonical_sha256(output) == observed.upper(),
        "APPLE_SUPPORT_SEAL_MISMATCH",
        "record seal drift",
    )
    output["content_sha256"] = observed.upper()
    _require(output.get("schema") == schema, "APPLE_SUPPORT_RECORD_INVALID", "record schema drift")
    return output


def _immutable(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    array.flags.writeable = False
    return array


def _plane_block(plane: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "normal_camera_xyz": np.asarray(plane["normal_camera_xyz"], dtype=np.float64).tolist(),
        "camera_height_m": float(plane["camera_height_m"]),
        "support_count": int(plane["support_count"]),
        "support_fraction": float(plane["support_fraction"]),
        "slope_degrees": float(plane["slope_degrees"]),
        "median_residual_m": float(plane["median_residual_m"]),
    }


@dataclass(frozen=True)
class AppleSeededCandidatePlane:
    parent_id: str
    physical_frame_id: str
    source_frame_receipt_sha256: str
    source_scale_record_sha256: str
    candidate_binding_sha256: str
    anchored_depth_array_sha256: str
    intrinsics_highres_sha256: str
    gravity_up_camera_xyz_sha256: str
    apple_normal_camera_xyz: np.ndarray
    apple_camera_height_m: float
    normal_camera_xyz: np.ndarray
    camera_height_m: float
    support_count: int
    support_fraction: float
    slope_degrees: float
    median_residual_m: float
    record: dict[str, Any]
    content_sha256: str


def _seeded_candidate_plane(points: np.ndarray, gravity: np.ndarray, apple_plane: Mapping[str, Any]) -> dict[str, Any]:
    apple_normal = np.asarray(apple_plane["normal_camera_xyz"], dtype=np.float64)
    residual = np.abs(points @ apple_normal + float(apple_plane["camera_height_m"]))
    initial = residual <= adapter.SUPPORT_RESIDUAL_TOLERANCE_M
    minimum_count = max(adapter.MINIMUM_SUPPORT_POINTS, int(math.ceil(adapter.MINIMUM_SUPPORT_FRACTION * len(points))))
    _require(int(np.sum(initial)) >= minimum_count, "APPLE_SEEDED_SUPPORT_POINTS_INSUFFICIENT", "too few source-scaled candidate points agree with Apple seed")
    selected = points[initial]
    centered = selected - np.mean(selected, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    normal = adapter._normalize_vector(right[-1], "APPLE_SEEDED_SUPPORT_NORMAL_INVALID")
    if float(np.dot(normal, gravity)) < 0.0:
        normal = -normal
    slope = math.degrees(math.acos(float(np.clip(np.dot(normal, gravity), -1.0, 1.0))))
    _require(slope <= adapter.MAXIMUM_SUPPORT_SLOPE_DEGREES, "APPLE_SEEDED_SUPPORT_SLOPE_EXCEEDED", "candidate plane exceeds frozen slope")
    seed_angle = math.degrees(math.acos(float(np.clip(np.dot(normal, apple_normal), -1.0, 1.0))))
    _require(seed_angle <= adapter.MAXIMUM_SUPPORT_SLOPE_DEGREES, "APPLE_SEEDED_SUPPORT_NORMAL_DISAGREEMENT", "candidate plane disagrees with Apple seed")
    camera_height = -float(np.median(selected @ normal))
    _require(
        camera_height > 0.0 and abs(camera_height - float(apple_plane["camera_height_m"])) <= adapter.SUPPORT_RESIDUAL_TOLERANCE_M,
        "APPLE_SEEDED_SUPPORT_HEIGHT_DISAGREEMENT",
        "candidate height disagrees with Apple seed",
    )
    refined_residual = np.abs(points @ normal + camera_height)
    refined = refined_residual <= adapter.SUPPORT_RESIDUAL_TOLERANCE_M
    support_count = int(np.sum(refined))
    _require(support_count >= minimum_count, "APPLE_SEEDED_SUPPORT_REFINED_GATE_FAILED", "refined support count/fraction failed")
    return {
        "normal_camera_xyz": normal,
        "camera_height_m": camera_height,
        "median_residual_m": float(np.median(refined_residual[refined])),
        "support_count": support_count,
        "support_fraction": support_count / float(len(points)),
        "slope_degrees": slope,
        "initial_mask": initial,
        "refined_mask": refined,
        "minimum_count": minimum_count,
        "apple_candidate_normal_angle_degrees": seed_angle,
        "apple_candidate_height_abs_delta_m": abs(camera_height - float(apple_plane["camera_height_m"])),
    }


def derive_apple_seeded_candidate_plane(
    prepared: source_factor.PreparedSourceCandidate,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    source_frame_receipt: Mapping[str, Any],
    source_scale_record: Mapping[str, Any],
) -> AppleSeededCandidatePlane:
    """Phase A: fit and seal one frame-global plane without FARO/query data."""

    source = adapter._validate_base_receipt(dict(source_frame_receipt))
    scale = apple_scale.validate_source_scale_record(dict(source_scale_record))
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    _require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16, "APPLE_SUPPORT_DEPTH_INVALID", "AppleDepth must be uint16 192x256 millimetres")
    _require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "APPLE_SUPPORT_CONFIDENCE_INVALID", "confidence must be uint8 0..2")
    try:
        adapter._validate_bound_decoded_payload(source, "lowres_depth", apple)
        adapter._validate_bound_decoded_payload(source, "confidence", conf)
        lowres_matrix = adapter._lowres_intrinsics_from_base_receipt(source)
        highres_matrix = adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
        gravity = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    except adapter.AdapterError as error:
        raise AppleSupportSeedError(error.code, str(error), **error.context) from error

    _require(
        source["parent_id"] == prepared.parent_id == scale["parent_id"]
        and source["physical_frame_id"] == prepared.physical_frame_id == scale["physical_frame_id"],
        "APPLE_SUPPORT_IDENTITY_MISMATCH",
        "source/candidate/scale identities differ",
    )
    _require(scale["content_sha256"] == prepared.source_scale_record_sha256, "APPLE_SUPPORT_SCALE_BINDING_DRIFT", "source-scale record differs from prepared candidate")
    _require(scale["candidate_binding_sha256"] == prepared.candidate_binding_sha256 and scale["candidate_highres_depth_array_sha256"] == prepared.raw_depth_sha256, "APPLE_SUPPORT_CANDIDATE_BINDING_DRIFT", "candidate lineage differs from R0 scale")
    _require(scale["evaluable"] is True, "APPLE_SUPPORT_SCALE_NOT_EVALUABLE", "R0 source scale is unavailable")

    sampled_raw = apple_scale.sample_candidate_at_apple_centers(prepared.raw_depth_m)
    apple_m = apple.astype(np.float64) / 1000.0
    lower, upper = apple_scale.DEPTH_RANGE_M
    selected = (conf == 2) & (apple_m >= lower) & (apple_m <= upper) & (sampled_raw >= lower) & (sampled_raw <= upper)
    selected_ids = np.flatnonzero(selected).astype(np.int64)
    _require(int(np.sum(selected)) == int(scale["valid_pair_count"]), "APPLE_SUPPORT_R0_MASK_DRIFT", "R0 valid-pair count drift")
    _require(adapter.canonical_sha256(selected_ids) == scale["selected_pixel_ids_sha256"], "APPLE_SUPPORT_R0_MASK_DRIFT", "R0 selected-pixel identity drift")

    try:
        apple_points, _ = adapter._unproject(apple_m, selected, lowres_matrix, 1)
        apple_plane = adapter._fit_support_plane(apple_points, gravity)
        candidate_valid = (prepared.anchored_depth_m >= adapter.DEPTH_RANGE_M[0]) & (prepared.anchored_depth_m <= adapter.DEPTH_RANGE_M[1])
        candidate_points, candidate_pixels = adapter._unproject(prepared.anchored_depth_m, candidate_valid, highres_matrix, adapter.SUPPORT_POINT_STRIDE)
        candidate_plane = _seeded_candidate_plane(candidate_points, gravity, apple_plane)
    except adapter.AdapterError as error:
        raise AppleSupportSeedError(error.code, str(error), **error.context) from error

    initial_ids = np.flatnonzero(candidate_plane["initial_mask"]).astype(np.int64)
    refined_ids = np.flatnonzero(candidate_plane["refined_mask"]).astype(np.int64)
    record = _seal(
        {
            "schema": PLANE_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "seed_id": SEED_ID,
            "parent_id": source["parent_id"],
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "source_scale_record_sha256": scale["content_sha256"],
            "candidate_binding_sha256": prepared.candidate_binding_sha256,
            "raw_candidate_depth_array_sha256": prepared.raw_depth_sha256,
            "anchored_candidate_depth_array_sha256": prepared.anchored_depth_sha256,
            "apple_depth_array_sha256": adapter.canonical_sha256(apple),
            "confidence_array_sha256": adapter.canonical_sha256(conf),
            "intrinsics_highres_sha256": adapter.canonical_sha256(highres_matrix),
            "intrinsics_lowres_sha256": adapter.canonical_sha256(lowres_matrix),
            "gravity_up_camera_xyz_sha256": adapter.canonical_sha256(gravity),
            "r0_mask": {
                "confidence_selection": "CONFIDENCE_EQ_2",
                "depth_range_m": list(apple_scale.DEPTH_RANGE_M),
                "valid_pair_count": len(selected_ids),
                "selected_pixel_ids_sha256": adapter.canonical_sha256(selected_ids),
            },
            "apple_seed": {
                "point_stride": 1,
                "sampled_point_count": len(apple_points),
                "sampled_points_sha256": adapter.canonical_sha256(apple_points),
                **_plane_block(apple_plane),
            },
            "candidate_refit": {
                "point_stride": adapter.SUPPORT_POINT_STRIDE,
                "sampled_valid_point_count": len(candidate_points),
                "sampled_pixels_sha256": adapter.canonical_sha256(candidate_pixels),
                "sampled_points_sha256": adapter.canonical_sha256(candidate_points),
                "minimum_support_count": int(candidate_plane["minimum_count"]),
                "initial_inlier_count": len(initial_ids),
                "initial_inlier_ids_sha256": adapter.canonical_sha256(initial_ids),
                "refined_inlier_count": len(refined_ids),
                "refined_inlier_ids_sha256": adapter.canonical_sha256(refined_ids),
                "apple_candidate_normal_angle_degrees": float(candidate_plane["apple_candidate_normal_angle_degrees"]),
                "apple_candidate_height_abs_delta_m": float(candidate_plane["apple_candidate_height_abs_delta_m"]),
                **_plane_block(candidate_plane),
            },
            "faro_payload_read": False,
            "query_receipt_read": False,
            "truth_alignment_used": False,
            "computed_before_truth_join": True,
        }
    )
    validate_apple_seeded_candidate_plane_record(record)
    return AppleSeededCandidatePlane(
        parent_id=str(source["parent_id"]),
        physical_frame_id=str(source["physical_frame_id"]),
        source_frame_receipt_sha256=str(source["content_sha256"]),
        source_scale_record_sha256=str(scale["content_sha256"]),
        candidate_binding_sha256=prepared.candidate_binding_sha256,
        anchored_depth_array_sha256=prepared.anchored_depth_sha256,
        intrinsics_highres_sha256=adapter.canonical_sha256(highres_matrix),
        gravity_up_camera_xyz_sha256=adapter.canonical_sha256(gravity),
        apple_normal_camera_xyz=_immutable(apple_plane["normal_camera_xyz"], np.float64),
        apple_camera_height_m=float(apple_plane["camera_height_m"]),
        normal_camera_xyz=_immutable(candidate_plane["normal_camera_xyz"], np.float64),
        camera_height_m=float(candidate_plane["camera_height_m"]),
        support_count=int(candidate_plane["support_count"]),
        support_fraction=float(candidate_plane["support_fraction"]),
        slope_degrees=float(candidate_plane["slope_degrees"]),
        median_residual_m=float(candidate_plane["median_residual_m"]),
        record=record,
        content_sha256=record["content_sha256"],
    )


def validate_apple_seeded_candidate_plane_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, PLANE_SCHEMA)
    _require(record.get("analysis_kind") == ANALYSIS_KIND and record.get("claim_ceiling") == CLAIM_CEILING and record.get("seed_id") == SEED_ID, "APPLE_SUPPORT_METHOD_DRIFT", "support method/claim drift")
    _require(record.get("faro_payload_read") is False and record.get("query_receipt_read") is False and record.get("truth_alignment_used") is False and record.get("computed_before_truth_join") is True, "APPLE_SUPPORT_TRUTH_FIREWALL_BREACH", "support plane crossed the source-only firewall")
    for field in (
        "source_frame_receipt_sha256", "source_scale_record_sha256", "candidate_binding_sha256",
        "raw_candidate_depth_array_sha256", "anchored_candidate_depth_array_sha256", "apple_depth_array_sha256",
        "confidence_array_sha256", "intrinsics_highres_sha256", "intrinsics_lowres_sha256", "gravity_up_camera_xyz_sha256",
    ):
        _hash(record.get(field), field=field)
    r0 = record.get("r0_mask", {})
    _require(r0.get("confidence_selection") == "CONFIDENCE_EQ_2" and r0.get("depth_range_m") == list(apple_scale.DEPTH_RANGE_M), "APPLE_SUPPORT_R0_MASK_DRIFT", "R0 source mask constants drift")
    apple = record.get("apple_seed", {})
    candidate = record.get("candidate_refit", {})
    _require(apple.get("point_stride") == 1 and candidate.get("point_stride") == adapter.SUPPORT_POINT_STRIDE, "APPLE_SUPPORT_METHOD_DRIFT", "support sampling drift")
    _require(isinstance(candidate.get("refined_inlier_count"), int) and candidate["refined_inlier_count"] >= candidate.get("minimum_support_count", math.inf), "APPLE_SUPPORT_GATE_DRIFT", "candidate support gate drift")
    return record


def load_apple_seeded_candidate_plane(value: Any) -> AppleSeededCandidatePlane:
    """Rehydrate only the sealed scalar plane; source arrays remain external."""

    record = validate_apple_seeded_candidate_plane_record(value)
    apple = record["apple_seed"]
    candidate = record["candidate_refit"]
    return AppleSeededCandidatePlane(
        parent_id=str(record["parent_id"]),
        physical_frame_id=str(record["physical_frame_id"]),
        source_frame_receipt_sha256=str(record["source_frame_receipt_sha256"]),
        source_scale_record_sha256=str(record["source_scale_record_sha256"]),
        candidate_binding_sha256=str(record["candidate_binding_sha256"]),
        anchored_depth_array_sha256=str(record["anchored_candidate_depth_array_sha256"]),
        intrinsics_highres_sha256=str(record["intrinsics_highres_sha256"]),
        gravity_up_camera_xyz_sha256=str(record["gravity_up_camera_xyz_sha256"]),
        apple_normal_camera_xyz=_immutable(apple["normal_camera_xyz"], np.float64),
        apple_camera_height_m=float(apple["camera_height_m"]),
        normal_camera_xyz=_immutable(candidate["normal_camera_xyz"], np.float64),
        camera_height_m=float(candidate["camera_height_m"]),
        support_count=int(candidate["support_count"]),
        support_fraction=float(candidate["support_fraction"]),
        slope_degrees=float(candidate["slope_degrees"]),
        median_residual_m=float(candidate["median_residual_m"]),
        record=record,
        content_sha256=str(record["content_sha256"]),
    )


def _posthoc_query_extraction(
    prepared: source_factor.PreparedSourceCandidate,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    base: source_factor.QueryTruthBase,
    plane: AppleSeededCandidatePlane,
) -> source_factor.QueryExtraction:
    """Phase B: score a sealed source-only plane on a FARO-defined query surface."""

    _require(prepared.physical_frame_id == base.physical_frame_id == plane.physical_frame_id, "APPLE_SUPPORT_IDENTITY_MISMATCH", "candidate/query/plane frames differ")
    matrix = adapter._intrinsics_matrix(intrinsics_highres_3x3)
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "GRAVITY_INVALID")
    _require(adapter.canonical_sha256(matrix) == plane.intrinsics_highres_sha256 and adapter.canonical_sha256(gravity) == plane.gravity_up_camera_xyz_sha256, "APPLE_SUPPORT_CAMERA_BINDING_DRIFT", "query camera metadata differs from sealed source plane")
    _require(prepared.anchored_depth_sha256 == plane.anchored_depth_array_sha256 and prepared.source_scale_record_sha256 == plane.source_scale_record_sha256, "APPLE_SUPPORT_CANDIDATE_BINDING_DRIFT", "query candidate differs from sealed source plane")
    point_ids = np.asarray(base.common_point_ids_uv, dtype=np.int64)
    depth = np.asarray(prepared.anchored_depth_m, dtype=np.float64)
    z_all = depth[point_ids[:, 1], point_ids[:, 0]]
    valid = (z_all >= adapter.DEPTH_RANGE_M[0]) & (z_all <= adapter.DEPTH_RANGE_M[1])
    valid_count = int(np.sum(valid))
    _require(valid_count >= adapter.MINIMUM_SUPPORT_POINTS, "POSTHOC_COMMON_SUPPORT_INSUFFICIENT", "too few valid points for post-hoc query comparison")
    valid_ids = point_ids[valid].astype(np.int32, copy=False)
    z = z_all[valid]
    points = np.stack(
        ((valid_ids[:, 0] - matrix[0, 2]) * z / matrix[0, 0], (valid_ids[:, 1] - matrix[1, 2]) * z / matrix[1, 1], z),
        axis=1,
    )
    try:
        query_geometry = adapter._query_support_and_boundary(points, valid_ids, plane.normal_camera_xyz, plane.camera_height_m, base.query_receipt)
    except adapter.AdapterError as error:
        raise AppleSupportSeedError(error.code, str(error), **error.context) from error
    return source_factor.QueryExtraction(
        depth_sha256=prepared.anchored_depth_sha256,
        valid_common_point_count=valid_count,
        normal_camera_xyz=plane.normal_camera_xyz,
        camera_height_m=plane.camera_height_m,
        support_point_count=plane.support_count,
        support_fraction=plane.support_fraction,
        slope_degrees=plane.slope_degrees,
        median_residual_m=plane.median_residual_m,
        query_support_points=int(query_geometry["query_support_points"]),
        observed_forward_m=None if query_geometry["observed_forward_shape_m"] is None else float(query_geometry["observed_forward_shape_m"]),
        local_valid_fraction=float(base.local_valid_fraction) * valid_count / float(len(valid)),
        boundary_point_ids_uv=_immutable(query_geometry["boundary_point_ids_uv"], np.int32),
        boundary_points_camera_xyz=_immutable(query_geometry["boundary_points_shape_camera_xyz"], np.float64),
    )


def evaluate_recovery_query(
    prepared: source_factor.PreparedSourceCandidate,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    base: source_factor.QueryTruthBase,
    plane: AppleSeededCandidatePlane | None,
    r1_query_record: Mapping[str, Any],
    *,
    source_failure_code: str | None = None,
) -> dict[str, Any]:
    r1 = source_factor.validate_query_record(dict(r1_query_record))
    _require(r1["physical_frame_id"] == base.physical_frame_id and r1["query_id"] == base.query_id and r1["effects"]["extraction_lost"] is True, "R1_LOST_QUERY_BINDING_INVALID", "R2 accepts only exact R1 extraction-lost queries")
    _require(r1["candidate_binding_sha256"] == prepared.candidate_binding_sha256 and r1["source_scale_record_sha256"] == prepared.source_scale_record_sha256, "R1_LOST_QUERY_BINDING_INVALID", "R1 candidate/scale lineage drift")
    _require(r1["query_receipt_sha256"] == base.query_receipt["content_sha256"] and r1["current_common_point_ids_sha256"] == base.common_point_ids_sha256, "R1_LOST_QUERY_BINDING_INVALID", "R1 query/common-support lineage drift")
    if plane is None:
        _require(isinstance(source_failure_code, str) and bool(source_failure_code), "APPLE_SUPPORT_SOURCE_FAILURE_CODE_MISSING", "missing source recovery failure code")
        mode = source_factor._failed_mode(prepared.anchored_depth_sha256, source_failure_code)
    else:
        _require(source_failure_code is None, "APPLE_SUPPORT_SOURCE_FAILURE_CODE_UNEXPECTED", "successful source plane carries a failure code")
        try:
            mode = source_factor._mode_result(base, _posthoc_query_extraction(prepared, intrinsics_highres_3x3, gravity_up_camera_xyz, base, plane))
        except AppleSupportSeedError as error:
            mode = source_factor._failed_mode(prepared.anchored_depth_sha256, error.code)
    baseline = r1["baseline"]
    no_regret = bool(
        mode["support"]["evaluable"]
        and float(mode["support"]["height_abs_error_m"]) <= float(baseline["support"]["height_abs_error_m"])
        and float(mode["support"]["normal_angular_error_rad"]) <= float(baseline["support"]["normal_angular_error_rad"])
    )
    return validate_recovery_record(
        _seal(
            {
                "schema": RECOVERY_RECORD_SCHEMA,
                "analysis_kind": ANALYSIS_KIND,
                "claim_ceiling": CLAIM_CEILING,
                "seed_id": SEED_ID,
                "parent_id": prepared.parent_id,
                "physical_frame_id": base.physical_frame_id,
                "query_id": base.query_id,
                "query_receipt_sha256": base.query_receipt["content_sha256"],
                "current_common_point_ids_sha256": base.common_point_ids_sha256,
                "r1_query_record_sha256": r1["content_sha256"],
                "r1_baseline_extraction_evaluable": True,
                "r1_source_anchored_extraction_evaluable": False,
                "r1_source_anchored_failure_codes": list(r1["source_anchored"]["reason_codes"]),
                "apple_seeded_candidate_plane_sha256": plane.content_sha256 if plane is not None else None,
                "candidate_binding_sha256": prepared.candidate_binding_sha256,
                "source_scale_record_sha256": prepared.source_scale_record_sha256,
                "source_frame_support_recovered": plane is not None,
                "posthoc_query_comparison": mode,
                "posthoc_query_comparison_evaluable": bool(mode["extraction_evaluable"]),
                "support_no_regret_vs_r1_baseline": no_regret,
                "height_error_reduction_vs_r1_baseline_m": float(baseline["support"]["height_abs_error_m"]) - float(mode["support"]["height_abs_error_m"]) if mode["support"]["evaluable"] else None,
                "normal_error_reduction_vs_r1_baseline_rad": float(baseline["support"]["normal_angular_error_rad"]) - float(mode["support"]["normal_angular_error_rad"]) if mode["support"]["evaluable"] else None,
                "faro_used_for_source_recovery_decision": False,
                "faro_used_for_post_hoc_scoring": True,
                "threshold_or_pass_fail_decision_applied": False,
            }
        )
    )


def validate_recovery_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, RECOVERY_RECORD_SCHEMA)
    _require(record.get("analysis_kind") == ANALYSIS_KIND and record.get("claim_ceiling") == CLAIM_CEILING and record.get("seed_id") == SEED_ID, "APPLE_SUPPORT_METHOD_DRIFT", "recovery method/claim drift")
    _require(record.get("r1_baseline_extraction_evaluable") is True and record.get("r1_source_anchored_extraction_evaluable") is False, "R1_LOST_QUERY_BINDING_INVALID", "record is not an R1 loss")
    _require(record.get("faro_used_for_source_recovery_decision") is False and record.get("faro_used_for_post_hoc_scoring") is True and record.get("threshold_or_pass_fail_decision_applied") is False, "APPLE_SUPPORT_CLAIM_DRIFT", "truth/decision boundary drift")
    _require(record.get("posthoc_query_comparison_evaluable") == bool(record.get("posthoc_query_comparison", {}).get("extraction_evaluable")), "APPLE_SUPPORT_RECOVERY_DRIFT", "post-hoc comparison flag drift")
    for field in ("query_receipt_sha256", "current_common_point_ids_sha256", "r1_query_record_sha256", "candidate_binding_sha256", "source_scale_record_sha256"):
        _hash(record.get(field), field=field)
    if record.get("source_frame_support_recovered"):
        _hash(record.get("apple_seeded_candidate_plane_sha256"), field="apple_seeded_candidate_plane_sha256")
    else:
        _require(record.get("apple_seeded_candidate_plane_sha256") is None, "APPLE_SUPPORT_RECOVERY_DRIFT", "failed source recovery carries a plane")
    return record


def _parent_macro(records: Sequence[dict[str, Any]], selector) -> dict[str, Any]:
    frame_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        value = selector(row)
        if _finite(value):
            frame_values[(row["parent_id"], row["physical_frame_id"])].append(float(value))
    by_parent: dict[str, list[float]] = defaultdict(list)
    for (parent, _), values in frame_values.items():
        by_parent[parent].append(float(np.median(np.asarray(values, dtype=np.float64))))
    values = [float(np.median(np.asarray(items, dtype=np.float64))) for items in by_parent.values()]
    return {"parents_with_metric": len(values), "median_of_parent_medians": float(np.median(np.asarray(values, dtype=np.float64))) if values else None}


def summarize_recovery(
    records: Sequence[Mapping[str, Any]],
    source_failures: Sequence[Mapping[str, Any]],
    *,
    expected_query_count: int = 112,
    expected_frame_count: int = 14,
) -> dict[str, Any]:
    rows = [validate_recovery_record(dict(row)) for row in records]
    _require(len(rows) == expected_query_count and len({(row["physical_frame_id"], row["query_id"]) for row in rows}) == expected_query_count, "APPLE_SUPPORT_RECOVERY_COHORT_DRIFT", "R2 lost-query cohort drift")
    frame_ids = {row["physical_frame_id"] for row in rows}
    _require(len(frame_ids) == expected_frame_count, "APPLE_SUPPORT_RECOVERY_COHORT_DRIFT", "R2 lost-frame cohort drift")
    source_recovered_frames = {row["physical_frame_id"] for row in rows if row["source_frame_support_recovered"]}
    posthoc = [row for row in rows if row["posthoc_query_comparison_evaluable"]]
    no_regret = [row for row in posthoc if row["support_no_regret_vs_r1_baseline"]]
    posthoc_by_frame = {frame_id: [row for row in rows if row["physical_frame_id"] == frame_id] for frame_id in frame_ids}
    reason_counts = Counter(code for row in rows if not row["posthoc_query_comparison_evaluable"] for code in row["posthoc_query_comparison"]["reason_codes"])
    return _seal(
        {
            "schema": RECOVERY_SUMMARY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "seed_id": SEED_ID,
            "r1_lost_query_count": len(rows),
            "r1_lost_frame_count": len(frame_ids),
            "source_recovered_frame_count": len(source_recovered_frames),
            "source_failure_frame_count": len(source_failures),
            "source_failure_frames": list(source_failures),
            "posthoc_evaluable_query_count": len(posthoc),
            "posthoc_evaluable_frame_count": sum(any(row["posthoc_query_comparison_evaluable"] for row in items) for items in posthoc_by_frame.values()),
            "fully_posthoc_evaluable_frame_count": sum(all(row["posthoc_query_comparison_evaluable"] for row in items) for items in posthoc_by_frame.values()),
            "posthoc_unknown_reason_counts": dict(sorted(reason_counts.items())),
            "support_no_regret_query_count": len(no_regret),
            "support_no_regret_frame_count": len({row["physical_frame_id"] for row in no_regret}),
            "seeded_boundary_evaluable_query_count": sum(bool(row["posthoc_query_comparison"]["boundary"]["evaluable"]) for row in rows),
            "seeded_query_known_count": sum(bool(row["posthoc_query_comparison"]["query_point_clearance"]["evaluable"]) for row in rows),
            "height_error_reduction_parent_macro_m": _parent_macro(rows, lambda row: row["height_error_reduction_vs_r1_baseline_m"]),
            "normal_error_reduction_parent_macro_rad": _parent_macro(rows, lambda row: row["normal_error_reduction_vs_r1_baseline_rad"]),
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


__all__ = [
    "ANALYSIS_KIND", "CLAIM_CEILING", "PLANE_SCHEMA", "RECOVERY_RECORD_SCHEMA", "RECOVERY_SUMMARY_SCHEMA", "SEED_ID",
    "AppleSeededCandidatePlane", "AppleSupportSeedError", "derive_apple_seeded_candidate_plane", "evaluate_recovery_query",
    "load_apple_seeded_candidate_plane",
    "summarize_recovery", "validate_apple_seeded_candidate_plane_record", "validate_recovery_record",
]
