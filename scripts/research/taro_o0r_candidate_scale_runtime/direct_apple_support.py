#!/usr/bin/env python3
"""Direct AppleDepth SUPPORT factor for TARO source fusion.

Unlike R2, this route does not ask the monocular candidate to refit or approve
the metric support plane.  AppleDepth supplies SUPPORT; the source-scaled
candidate remains responsible for dense boundary/query geometry.
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
from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import validate_candidate_input_receipt
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


PLANE_SCHEMA = "blindassist.taro.o0r.direct_apple_support_plane.v1"
SOURCE_RECEIPT_SCHEMA = "blindassist.taro.o0r.direct_apple_source_receipt.v1"
QUERY_SCHEMA = "blindassist.taro.o0r.direct_apple_support_query_canary.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.direct_apple_support_canary_summary.v1"
METHOD_ID = "R0_SCALE_PLUS_C2_APPLE_ONLY_SUPPORT_WITH_SOURCE_SCALED_DENSE_BOUNDARY_V2"
CAMERA_HEIGHT_RANGE_M = (0.45, 2.20)
ANALYSIS_KIND = "POST_HOC_DIRECT_APPLE_SUPPORT_FACTOR_CANARY"
CLAIM_CEILING = {
    "scope": "R1_EXTRACTION_LOST_FRAMES_ONLY_LOCKED_ARKITSCENES_TRAIN_LANDSCAPE",
    "use": "POST_HOC_DESCRIPTIVE_APPLEDEPTH_SUPPORT_FACTOR_CANARY",
    "retrospective_cohort": True,
    "threshold_or_pass_fail_decision": False,
    "excluded_claims": ["RGB_ONLY_CAPABILITY", "FORMAL_O0R_PASS", "DEPLOYMENT", "PRODUCT", "SAFETY"],
}
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")


class DirectAppleSupportError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise DirectAppleSupportError(code, message, **context)


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _hash(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "DIRECT_APPLE_HASH_INVALID", "SHA-256 binding is malformed", field=field)
    return str(value).upper()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    _require("content_sha256" not in output, "DIRECT_APPLE_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "DIRECT_APPLE_RECORD_INVALID", "record must be an object")
    output = copy.deepcopy(value)
    observed = output.pop("content_sha256", None)
    _require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)) and adapter.canonical_sha256(output) == observed.upper(), "DIRECT_APPLE_SEAL_MISMATCH", "record seal drift")
    output["content_sha256"] = observed.upper()
    _require(output.get("schema") == schema, "DIRECT_APPLE_RECORD_INVALID", "record schema drift")
    return output


def _immutable(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class DirectAppleSupportPlane:
    parent_id: str
    physical_frame_id: str
    direct_source_receipt_sha256: str
    source_scale_record_sha256: str
    candidate_binding_sha256: str
    anchored_depth_array_sha256: str
    intrinsics_highres_sha256: str
    gravity_up_camera_xyz_sha256: str
    normal_camera_xyz: np.ndarray
    camera_height_m: float
    support_count: int
    support_fraction: float
    slope_degrees: float
    median_residual_m: float
    record: dict[str, Any]
    content_sha256: str


def build_direct_apple_source_receipt(
    candidate_input_receipt: Mapping[str, Any],
    apple_source_receipt: Mapping[str, Any],
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    lowres_intrinsics: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    intrinsics_member_sha256: str,
    intrinsics_member_crc32: str,
    trajectory_container_sha256: str,
    trajectory_payload_sha256: str,
) -> dict[str, Any]:
    """Build narrow K/pose/Apple metadata without opening compact truth."""

    candidate_input = validate_candidate_input_receipt(dict(candidate_input_receipt))
    apple_source = apple_scale.validate_apple_scale_source_receipt(dict(apple_source_receipt), np.asarray(apple_depth_mm), np.asarray(confidence))
    identity = (candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"], candidate_input["physical_frame_id"])
    _require(identity == (apple_source["parent_id"], apple_source["video_id"], apple_source["timestamp_token"], apple_source["physical_frame_id"]), "DIRECT_APPLE_IDENTITY_MISMATCH", "candidate input and Apple source identities differ")
    lowres = {key: lowres_intrinsics[key] for key in ("width", "height", "fx", "fy", "cx", "cy")}
    try:
        highres = adapter.scale_lowres_intrinsics(lowres)
        transform, pose = adapter.interpolate_camera_to_world_exact(list(trajectory_rows), identity[2])
    except adapter.AdapterError as error:
        raise DirectAppleSupportError(error.code, str(error), **error.context) from error
    _require(adapter.canonical_sha256(lowres) == adapter.canonical_sha256(candidate_input["lowres_intrinsics"]), "DIRECT_APPLE_INTRINSICS_DRIFT", "raw low-res intrinsics differ from sealed candidate input")
    _require(adapter.canonical_sha256(highres) == adapter.canonical_sha256(candidate_input["intrinsics_highres"]), "DIRECT_APPLE_INTRINSICS_DRIFT", "scaled high-res intrinsics differ from sealed candidate input")
    _hash(intrinsics_member_sha256, field="intrinsics_member_sha256")
    _hash(trajectory_container_sha256, field="trajectory_container_sha256")
    _hash(trajectory_payload_sha256, field="trajectory_payload_sha256")
    _require(isinstance(intrinsics_member_crc32, str) and re.fullmatch(r"[0-9A-F]{8}", intrinsics_member_crc32) is not None, "DIRECT_APPLE_INTRINSICS_DRIFT", "intrinsics CRC is malformed")
    intrinsics_binding = candidate_input["intrinsics_member_binding"]
    _require(
        intrinsics_binding["sha256"] == intrinsics_member_sha256.upper()
        and intrinsics_binding["crc32"] == intrinsics_member_crc32,
        "DIRECT_APPLE_INTRINSICS_DRIFT",
        "raw intrinsics bytes differ from the sealed candidate input",
    )
    gravity = adapter._normalize_vector(np.asarray(transform, dtype=np.float64)[2, :3], "GRAVITY_INVALID")
    return validate_direct_apple_source_receipt(
        _seal(
            {
                "schema": SOURCE_RECEIPT_SCHEMA,
                "parent_id": identity[0],
                "video_id": identity[1],
                "timestamp_token": identity[2],
                "physical_frame_id": identity[3],
                "candidate_input_receipt_sha256": candidate_input["content_sha256"],
                "apple_source_receipt_sha256": apple_source["content_sha256"],
                "apple_depth_array_sha256": adapter.canonical_sha256(np.asarray(apple_depth_mm)),
                "confidence_array_sha256": adapter.canonical_sha256(np.asarray(confidence)),
                "lowres_intrinsics": lowres,
                "intrinsics_highres": highres,
                "intrinsics_member_sha256": intrinsics_member_sha256.upper(),
                "intrinsics_member_crc32": intrinsics_member_crc32,
                "trajectory_container_sha256": trajectory_container_sha256.upper(),
                "trajectory_payload_sha256": trajectory_payload_sha256.upper(),
                "trajectory_rows_sha256": adapter.canonical_sha256(list(trajectory_rows)),
                "camera_to_world_4x4": np.asarray(transform, dtype=np.float64).tolist(),
                "gravity_up_camera_xyz": gravity.tolist(),
                "pose_bracket": pose,
                "opened_source_roles": ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"],
                "faro_payload_read": False,
                "rgb_payload_read": False,
                "compact_truth_read": False,
                "query_receipt_read": False,
            }
        ),
        np.asarray(apple_depth_mm),
        np.asarray(confidence),
    )


def validate_direct_apple_source_receipt(
    value: Any,
    apple_depth_mm: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    receipt = _validate_seal(value, SOURCE_RECEIPT_SCHEMA)
    _require(receipt.get("physical_frame_id") == f"{receipt.get('video_id')}:{receipt.get('timestamp_token')}", "DIRECT_APPLE_IDENTITY_MISMATCH", "source receipt identity drift")
    _require(receipt.get("opened_source_roles") == ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"] and receipt.get("faro_payload_read") is False and receipt.get("rgb_payload_read") is False and receipt.get("compact_truth_read") is False and receipt.get("query_receipt_read") is False, "DIRECT_APPLE_SOURCE_FIREWALL_BREACH", "narrow source receipt crossed role firewall")
    for field in ("candidate_input_receipt_sha256", "apple_source_receipt_sha256", "apple_depth_array_sha256", "confidence_array_sha256", "intrinsics_member_sha256", "trajectory_container_sha256", "trajectory_payload_sha256", "trajectory_rows_sha256"):
        _hash(receipt.get(field), field=field)
    try:
        lowres = {key: receipt["lowres_intrinsics"][key] for key in ("width", "height", "fx", "fy", "cx", "cy")}
        highres = adapter.scale_lowres_intrinsics(lowres)
        matrix = np.asarray(receipt["camera_to_world_4x4"], dtype=np.float64)
        gravity = adapter._normalize_vector(receipt["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    except (KeyError, TypeError, adapter.AdapterError) as error:
        if isinstance(error, adapter.AdapterError):
            raise DirectAppleSupportError(error.code, str(error), **error.context) from error
        raise DirectAppleSupportError("DIRECT_APPLE_SOURCE_RECEIPT_INVALID", "source camera metadata is malformed") from error
    _require(adapter.canonical_sha256(highres) == adapter.canonical_sha256(receipt["intrinsics_highres"]), "DIRECT_APPLE_INTRINSICS_DRIFT", "source high-res intrinsics drift")
    _require(matrix.shape == (4, 4) and bool(np.all(np.isfinite(matrix))) and bool(np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)), "DIRECT_APPLE_POSE_DRIFT", "camera-to-world transform is invalid")
    _require(abs(float(np.linalg.det(matrix[:3, :3])) - 1.0) <= 1e-9 and bool(np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0.0, atol=1e-9)), "DIRECT_APPLE_POSE_DRIFT", "camera rotation is not proper orthonormal")
    _require(bool(np.allclose(gravity, matrix[2, :3], rtol=0.0, atol=1e-12)), "DIRECT_APPLE_GRAVITY_DRIFT", "gravity is not derived from camera-to-world")
    if apple_depth_mm is not None or confidence is not None:
        apple = np.asarray(apple_depth_mm)
        conf = np.asarray(confidence)
        _require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16 and adapter.canonical_sha256(apple) == receipt["apple_depth_array_sha256"], "DIRECT_APPLE_DEPTH_INVALID", "AppleDepth array binding drift")
        _require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)) and adapter.canonical_sha256(conf) == receipt["confidence_array_sha256"], "DIRECT_APPLE_CONFIDENCE_INVALID", "confidence array binding drift")
    return receipt


def prepare_direct_source_candidate(
    candidate_highres_depth_m: np.ndarray,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    direct_source_receipt: Mapping[str, Any],
    candidate_input_receipt: Mapping[str, Any],
    apple_source_receipt: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    source_scale_record: Mapping[str, Any],
) -> source_factor.PreparedSourceCandidate:
    """Narrow equivalent of R1 preparation, with no compact truth receipt."""

    source = validate_direct_apple_source_receipt(dict(direct_source_receipt), np.asarray(apple_depth_mm), np.asarray(confidence))
    candidate_input = validate_candidate_input_receipt(dict(candidate_input_receipt))
    apple_receipt = apple_scale.validate_apple_scale_source_receipt(dict(apple_source_receipt), np.asarray(apple_depth_mm), np.asarray(confidence))
    binding = apple_scale.validate_candidate_replay_binding(dict(candidate_binding))
    scale = apple_scale.validate_source_scale_record(dict(source_scale_record))
    raw = np.asarray(candidate_highres_depth_m)
    _require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype.kind == "f" and bool(np.all(np.isfinite(raw))), "DIRECT_APPLE_CANDIDATE_INVALID", "candidate depth is invalid")
    raw_hash = adapter.canonical_sha256(raw)
    identity = (source["parent_id"], source["video_id"], source["timestamp_token"], source["physical_frame_id"])
    for other in (
        (candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"], candidate_input["physical_frame_id"]),
        (apple_receipt["parent_id"], apple_receipt["video_id"], apple_receipt["timestamp_token"], apple_receipt["physical_frame_id"]),
        (binding["parent_id"], binding["video_id"], binding["timestamp_token"], binding["physical_frame_id"]),
        (scale["parent_id"], scale["video_id"], scale["timestamp_token"], scale["physical_frame_id"]),
    ):
        _require(identity == other, "DIRECT_APPLE_IDENTITY_MISMATCH", "source lineage identities differ")
    _require(source["apple_source_receipt_sha256"] == apple_receipt["content_sha256"] and source["candidate_input_receipt_sha256"] == candidate_input["content_sha256"], "DIRECT_APPLE_LINEAGE_DRIFT", "narrow source lineage drift")
    _require(raw_hash == binding["highres_depth_array_sha256"] == scale["candidate_highres_depth_array_sha256"] and scale["candidate_binding_sha256"] == binding["content_sha256"] and scale["source_receipt_sha256"] == apple_receipt["content_sha256"], "DIRECT_APPLE_LINEAGE_DRIFT", "candidate/scale lineage drift")
    rebuilt = apple_scale.build_source_scale_record(np.asarray(apple_depth_mm), np.asarray(confidence), raw, apple_receipt, binding)
    _require(rebuilt["content_sha256"] == scale["content_sha256"] and scale["evaluable"] is True, "DIRECT_APPLE_SCALE_REDERIVATION_DRIFT", "source scale cannot be exactly re-derived")
    anchored = np.ascontiguousarray(raw.astype(np.float64) * float(scale["metric_scale"]), dtype=np.float64)
    reliability = source_factor.validate_reliability_record(source_factor._build_reliability_record(np.asarray(apple_depth_mm), np.asarray(confidence), raw, scale))
    return source_factor.PreparedSourceCandidate(
        parent_id=identity[0],
        physical_frame_id=identity[3],
        raw_depth_m=_immutable(raw, raw.dtype),
        anchored_depth_m=_immutable(anchored, np.float64),
        raw_depth_sha256=raw_hash,
        anchored_depth_sha256=adapter.canonical_sha256(anchored),
        metric_scale=float(scale["metric_scale"]),
        source_scale_record_sha256=scale["content_sha256"],
        candidate_binding_sha256=binding["content_sha256"],
        apple_source_receipt_sha256=apple_receipt["content_sha256"],
        reliability=reliability,
    )


def derive_direct_apple_support_plane(
    prepared: source_factor.PreparedSourceCandidate,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    direct_source_receipt: Mapping[str, Any],
    source_scale_record: Mapping[str, Any],
) -> DirectAppleSupportPlane:
    """Fit and seal a direct metric SUPPORT factor before FARO/query join."""

    source = validate_direct_apple_source_receipt(dict(direct_source_receipt), np.asarray(apple_depth_mm), np.asarray(confidence))
    scale = apple_scale.validate_source_scale_record(dict(source_scale_record))
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    _require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16, "DIRECT_APPLE_DEPTH_INVALID", "AppleDepth must be uint16 192x256")
    _require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "DIRECT_APPLE_CONFIDENCE_INVALID", "confidence must be uint8 0..2")
    try:
        lowres = source["lowres_intrinsics"]
        lowres_matrix = adapter._intrinsics_matrix(
            [[lowres["fx"], 0.0, lowres["cx"]], [0.0, lowres["fy"], lowres["cy"]], [0.0, 0.0, 1.0]],
            adapter.APPLE_SHAPE_HW,
        )
        highres_matrix = adapter._intrinsics_matrix(source["intrinsics_highres"]["matrix_3x3"])
        gravity = adapter._normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    except adapter.AdapterError as error:
        raise DirectAppleSupportError(error.code, str(error), **error.context) from error
    _require(source["parent_id"] == prepared.parent_id == scale["parent_id"] and source["physical_frame_id"] == prepared.physical_frame_id == scale["physical_frame_id"], "DIRECT_APPLE_IDENTITY_MISMATCH", "source/candidate/scale identities differ")
    _require(scale["content_sha256"] == prepared.source_scale_record_sha256 and scale["candidate_binding_sha256"] == prepared.candidate_binding_sha256, "DIRECT_APPLE_LINEAGE_DRIFT", "prepared source-scale lineage drift")
    _require(scale["evaluable"] is True, "DIRECT_APPLE_SCALE_NOT_EVALUABLE", "R0 source scale is unavailable")

    apple_m = apple.astype(np.float64) / 1000.0
    lower, upper = apple_scale.DEPTH_RANGE_M
    sampled_raw = apple_scale.sample_candidate_at_apple_centers(prepared.raw_depth_m)
    r0_pair_mask = (conf == 2) & (apple_m >= lower) & (apple_m <= upper) & (sampled_raw >= lower) & (sampled_raw <= upper)
    r0_pair_ids = np.flatnonzero(r0_pair_mask).astype(np.int64)
    _require(len(r0_pair_ids) == int(scale["valid_pair_count"]) and adapter.canonical_sha256(r0_pair_ids) == scale["selected_pixel_ids_sha256"], "DIRECT_APPLE_R0_MASK_DRIFT", "R0 scale-pair mask drift")
    apple_support_mask = (conf == 2) & (apple_m >= lower) & (apple_m <= upper)
    apple_support_ids = np.flatnonzero(apple_support_mask).astype(np.int64)
    try:
        points, pixels = adapter._unproject(apple_m, apple_support_mask, lowres_matrix, 1)
        plane = adapter._fit_support_plane(points, gravity)
    except adapter.AdapterError as error:
        raise DirectAppleSupportError(error.code, str(error), **error.context) from error
    camera_height = float(plane["camera_height_m"])
    _require(
        CAMERA_HEIGHT_RANGE_M[0] <= camera_height <= CAMERA_HEIGHT_RANGE_M[1],
        "DIRECT_APPLE_SUPPORT_HEIGHT_IMPLAUSIBLE",
        "refined Apple SUPPORT height leaves the frozen physical range",
        camera_height_m=camera_height,
    )

    record = _seal(
        {
            "schema": PLANE_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "method_id": METHOD_ID,
            "parent_id": source["parent_id"],
            "physical_frame_id": source["physical_frame_id"],
            "direct_source_receipt_sha256": source["content_sha256"],
            "source_scale_record_sha256": scale["content_sha256"],
            "candidate_binding_sha256": prepared.candidate_binding_sha256,
            "raw_candidate_depth_array_sha256": prepared.raw_depth_sha256,
            "anchored_candidate_depth_array_sha256": prepared.anchored_depth_sha256,
            "apple_depth_array_sha256": adapter.canonical_sha256(apple),
            "confidence_array_sha256": adapter.canonical_sha256(conf),
            "intrinsics_highres_sha256": adapter.canonical_sha256(highres_matrix),
            "intrinsics_lowres_sha256": adapter.canonical_sha256(lowres_matrix),
            "gravity_up_camera_xyz_sha256": adapter.canonical_sha256(gravity),
            "r0_scale_pair_mask": {
                "confidence_selection": "CONFIDENCE_EQ_2",
                "depth_range_m": list(apple_scale.DEPTH_RANGE_M),
                "valid_pair_count": len(r0_pair_ids),
                "selected_pixel_ids_sha256": adapter.canonical_sha256(r0_pair_ids),
                "purpose": "SOURCE_SCALE_REDERIVATION_ONLY",
            },
            "apple_support_mask": {
                "confidence_selection": "CONFIDENCE_EQ_2",
                "depth_range_m": list(apple_scale.DEPTH_RANGE_M),
                "valid_apple_point_count": len(apple_support_ids),
                "selected_pixel_ids_sha256": adapter.canonical_sha256(apple_support_ids),
                "candidate_depth_used": False,
            },
            "apple_support": {
                "point_stride": 1,
                "sampled_point_count": len(points),
                "sampled_pixels_sha256": adapter.canonical_sha256(pixels),
                "sampled_points_sha256": adapter.canonical_sha256(points),
                "normal_camera_xyz": np.asarray(plane["normal_camera_xyz"], dtype=np.float64).tolist(),
                "camera_height_m": camera_height,
                "support_count": int(plane["support_count"]),
                "support_fraction": float(plane["support_fraction"]),
                "slope_degrees": float(plane["slope_degrees"]),
                "median_residual_m": float(plane["median_residual_m"]),
            },
            "support_factor_source": "REGISTERED_APPLEDEPTH_CONFIDENCE_EQ_2_APPLE_RANGE_ONLY",
            "dense_boundary_source": "SOURCE_SCALED_SEALED_DEPTHART_CANDIDATE",
            "candidate_refit_or_veto_applied": False,
            "faro_payload_read": False,
            "query_receipt_read": False,
            "truth_alignment_used": False,
            "computed_before_truth_join": True,
        }
    )
    validate_direct_apple_support_plane_record(record)
    return load_direct_apple_support_plane(record)


def validate_direct_apple_support_plane_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, PLANE_SCHEMA)
    _require(record.get("analysis_kind") == ANALYSIS_KIND and record.get("claim_ceiling") == CLAIM_CEILING and record.get("method_id") == METHOD_ID, "DIRECT_APPLE_METHOD_DRIFT", "method/claim drift")
    _require(record.get("support_factor_source") == "REGISTERED_APPLEDEPTH_CONFIDENCE_EQ_2_APPLE_RANGE_ONLY" and record.get("dense_boundary_source") == "SOURCE_SCALED_SEALED_DEPTHART_CANDIDATE" and record.get("candidate_refit_or_veto_applied") is False, "DIRECT_APPLE_METHOD_DRIFT", "factor ownership drift")
    _require(record.get("faro_payload_read") is False and record.get("query_receipt_read") is False and record.get("truth_alignment_used") is False and record.get("computed_before_truth_join") is True, "DIRECT_APPLE_TRUTH_FIREWALL_BREACH", "source plane crossed truth firewall")
    for field in ("direct_source_receipt_sha256", "source_scale_record_sha256", "candidate_binding_sha256", "raw_candidate_depth_array_sha256", "anchored_candidate_depth_array_sha256", "apple_depth_array_sha256", "confidence_array_sha256", "intrinsics_highres_sha256", "intrinsics_lowres_sha256", "gravity_up_camera_xyz_sha256"):
        _hash(record.get(field), field=field)
    r0_mask = record.get("r0_scale_pair_mask", {})
    _require(r0_mask.get("confidence_selection") == "CONFIDENCE_EQ_2" and r0_mask.get("depth_range_m") == list(apple_scale.DEPTH_RANGE_M) and r0_mask.get("purpose") == "SOURCE_SCALE_REDERIVATION_ONLY", "DIRECT_APPLE_R0_MASK_DRIFT", "R0 scale-pair mask constants drift")
    apple_mask = record.get("apple_support_mask", {})
    _require(
        apple_mask.get("confidence_selection") == "CONFIDENCE_EQ_2"
        and apple_mask.get("depth_range_m") == list(apple_scale.DEPTH_RANGE_M)
        and isinstance(apple_mask.get("valid_apple_point_count"), int)
        and apple_mask["valid_apple_point_count"] >= adapter.MINIMUM_SUPPORT_POINTS
        and apple_mask.get("candidate_depth_used") is False,
        "DIRECT_APPLE_SUPPORT_MASK_DRIFT",
        "Apple-only SUPPORT mask drift",
    )
    _hash(apple_mask.get("selected_pixel_ids_sha256"), field="apple_support_mask.selected_pixel_ids_sha256")
    support = record.get("apple_support", {})
    normal = np.asarray(support.get("normal_camera_xyz"), dtype=np.float64)
    _require(
        support.get("point_stride") == 1
        and isinstance(support.get("support_count"), int)
        and support["support_count"] >= adapter.MINIMUM_SUPPORT_POINTS
        and _finite(support.get("camera_height_m"))
        and CAMERA_HEIGHT_RANGE_M[0] <= float(support["camera_height_m"]) <= CAMERA_HEIGHT_RANGE_M[1]
        and normal.shape == (3,)
        and bool(np.all(np.isfinite(normal)))
        and abs(float(np.linalg.norm(normal)) - 1.0) <= 1e-9
        and _finite(support.get("support_fraction"))
        and 0.0 < float(support["support_fraction"]) <= 1.0
        and _finite(support.get("slope_degrees"))
        and 0.0 <= float(support["slope_degrees"]) <= adapter.MAXIMUM_SUPPORT_SLOPE_DEGREES
        and _finite(support.get("median_residual_m"))
        and 0.0 <= float(support["median_residual_m"]) <= adapter.SUPPORT_RESIDUAL_TOLERANCE_M,
        "DIRECT_APPLE_SUPPORT_INVALID",
        "Apple SUPPORT block is invalid",
    )
    return record


def load_direct_apple_support_plane(value: Any) -> DirectAppleSupportPlane:
    record = validate_direct_apple_support_plane_record(value)
    support = record["apple_support"]
    return DirectAppleSupportPlane(
        parent_id=str(record["parent_id"]),
        physical_frame_id=str(record["physical_frame_id"]),
        direct_source_receipt_sha256=str(record["direct_source_receipt_sha256"]),
        source_scale_record_sha256=str(record["source_scale_record_sha256"]),
        candidate_binding_sha256=str(record["candidate_binding_sha256"]),
        anchored_depth_array_sha256=str(record["anchored_candidate_depth_array_sha256"]),
        intrinsics_highres_sha256=str(record["intrinsics_highres_sha256"]),
        gravity_up_camera_xyz_sha256=str(record["gravity_up_camera_xyz_sha256"]),
        normal_camera_xyz=_immutable(support["normal_camera_xyz"], np.float64),
        camera_height_m=float(support["camera_height_m"]),
        support_count=int(support["support_count"]),
        support_fraction=float(support["support_fraction"]),
        slope_degrees=float(support["slope_degrees"]),
        median_residual_m=float(support["median_residual_m"]),
        record=record,
        content_sha256=str(record["content_sha256"]),
    )


def _posthoc_extraction(
    prepared: source_factor.PreparedSourceCandidate,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    base: source_factor.QueryTruthBase,
    plane: DirectAppleSupportPlane,
) -> source_factor.QueryExtraction:
    _require(prepared.physical_frame_id == base.physical_frame_id == plane.physical_frame_id, "DIRECT_APPLE_IDENTITY_MISMATCH", "candidate/query/plane frames differ")
    matrix = adapter._intrinsics_matrix(intrinsics_highres_3x3)
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "GRAVITY_INVALID")
    _require(adapter.canonical_sha256(matrix) == plane.intrinsics_highres_sha256 and adapter.canonical_sha256(gravity) == plane.gravity_up_camera_xyz_sha256, "DIRECT_APPLE_CAMERA_BINDING_DRIFT", "query camera differs from source plane")
    _require(prepared.anchored_depth_sha256 == plane.anchored_depth_array_sha256 and prepared.source_scale_record_sha256 == plane.source_scale_record_sha256, "DIRECT_APPLE_LINEAGE_DRIFT", "query candidate differs from source plane")
    ids = np.asarray(base.common_point_ids_uv, dtype=np.int64)
    depth = np.asarray(prepared.anchored_depth_m, dtype=np.float64)
    z_all = depth[ids[:, 1], ids[:, 0]]
    valid = (z_all >= adapter.DEPTH_RANGE_M[0]) & (z_all <= adapter.DEPTH_RANGE_M[1])
    valid_count = int(np.sum(valid))
    _require(valid_count >= adapter.MINIMUM_SUPPORT_POINTS, "DIRECT_APPLE_POSTHOC_COMMON_SUPPORT_INSUFFICIENT", "too few dense candidate points for post-hoc boundary")
    valid_ids = ids[valid].astype(np.int32, copy=False)
    z = z_all[valid]
    points = np.stack(((valid_ids[:, 0] - matrix[0, 2]) * z / matrix[0, 0], (valid_ids[:, 1] - matrix[1, 2]) * z / matrix[1, 1], z), axis=1)
    try:
        query = adapter._query_support_and_boundary(points, valid_ids, plane.normal_camera_xyz, plane.camera_height_m, base.query_receipt)
    except adapter.AdapterError as error:
        raise DirectAppleSupportError(error.code, str(error), **error.context) from error
    return source_factor.QueryExtraction(
        depth_sha256=prepared.anchored_depth_sha256,
        valid_common_point_count=valid_count,
        normal_camera_xyz=plane.normal_camera_xyz,
        camera_height_m=plane.camera_height_m,
        support_point_count=plane.support_count,
        support_fraction=plane.support_fraction,
        slope_degrees=plane.slope_degrees,
        median_residual_m=plane.median_residual_m,
        query_support_points=int(query["query_support_points"]),
        observed_forward_m=None if query["observed_forward_shape_m"] is None else float(query["observed_forward_shape_m"]),
        local_valid_fraction=float(base.local_valid_fraction) * valid_count / float(len(valid)),
        boundary_point_ids_uv=_immutable(query["boundary_point_ids_uv"], np.int32),
        boundary_points_camera_xyz=_immutable(query["boundary_points_shape_camera_xyz"], np.float64),
    )


def evaluate_direct_apple_query(
    prepared: source_factor.PreparedSourceCandidate,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    base: source_factor.QueryTruthBase,
    plane: DirectAppleSupportPlane | None,
    r1_query_record: Mapping[str, Any],
    *,
    source_failure_code: str | None = None,
) -> dict[str, Any]:
    r1 = source_factor.validate_query_record(dict(r1_query_record))
    _require(r1["physical_frame_id"] == base.physical_frame_id and r1["query_id"] == base.query_id and r1["effects"]["extraction_lost"] is True, "DIRECT_APPLE_R1_BINDING_INVALID", "R3 accepts only exact R1 lost queries")
    _require(r1["candidate_binding_sha256"] == prepared.candidate_binding_sha256 and r1["source_scale_record_sha256"] == prepared.source_scale_record_sha256 and r1["query_receipt_sha256"] == base.query_receipt["content_sha256"] and r1["current_common_point_ids_sha256"] == base.common_point_ids_sha256, "DIRECT_APPLE_R1_BINDING_INVALID", "R1 lineage drift")
    if plane is None:
        _require(isinstance(source_failure_code, str) and bool(source_failure_code), "DIRECT_APPLE_FAILURE_CODE_MISSING", "source failure code missing")
        mode = source_factor._failed_mode(prepared.anchored_depth_sha256, source_failure_code)
    else:
        _require(source_failure_code is None, "DIRECT_APPLE_FAILURE_CODE_UNEXPECTED", "successful source plane carries failure code")
        try:
            mode = source_factor._mode_result(base, _posthoc_extraction(prepared, intrinsics_highres_3x3, gravity_up_camera_xyz, base, plane))
        except DirectAppleSupportError as error:
            mode = source_factor._failed_mode(prepared.anchored_depth_sha256, error.code)
    baseline = r1["baseline"]
    no_regret = bool(mode["support"]["evaluable"] and float(mode["support"]["height_abs_error_m"]) <= float(baseline["support"]["height_abs_error_m"]) and float(mode["support"]["normal_angular_error_rad"]) <= float(baseline["support"]["normal_angular_error_rad"]))
    return validate_direct_apple_query_record(
        _seal(
            {
                "schema": QUERY_SCHEMA,
                "analysis_kind": ANALYSIS_KIND,
                "claim_ceiling": CLAIM_CEILING,
                "method_id": METHOD_ID,
                "parent_id": prepared.parent_id,
                "physical_frame_id": base.physical_frame_id,
                "query_id": base.query_id,
                "query_receipt_sha256": base.query_receipt["content_sha256"],
                "current_common_point_ids_sha256": base.common_point_ids_sha256,
                "r1_query_record_sha256": r1["content_sha256"],
                "direct_apple_support_plane_sha256": plane.content_sha256 if plane is not None else None,
                "candidate_binding_sha256": prepared.candidate_binding_sha256,
                "source_scale_record_sha256": prepared.source_scale_record_sha256,
                "source_support_available": plane is not None,
                "posthoc_query_comparison": mode,
                "posthoc_query_comparison_evaluable": bool(mode["extraction_evaluable"]),
                "support_no_regret_vs_r1_baseline": no_regret,
                "height_error_reduction_vs_r1_baseline_m": float(baseline["support"]["height_abs_error_m"]) - float(mode["support"]["height_abs_error_m"]) if mode["support"]["evaluable"] else None,
                "normal_error_reduction_vs_r1_baseline_rad": float(baseline["support"]["normal_angular_error_rad"]) - float(mode["support"]["normal_angular_error_rad"]) if mode["support"]["evaluable"] else None,
                "faro_used_for_source_support": False,
                "faro_used_for_post_hoc_scoring": True,
                "threshold_or_pass_fail_decision_applied": False,
            }
        )
    )


def validate_direct_apple_query_record(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, QUERY_SCHEMA)
    _require(record.get("analysis_kind") == ANALYSIS_KIND and record.get("claim_ceiling") == CLAIM_CEILING and record.get("method_id") == METHOD_ID, "DIRECT_APPLE_METHOD_DRIFT", "query method/claim drift")
    _require(record.get("faro_used_for_source_support") is False and record.get("faro_used_for_post_hoc_scoring") is True and record.get("threshold_or_pass_fail_decision_applied") is False, "DIRECT_APPLE_CLAIM_DRIFT", "truth boundary drift")
    _require(record.get("posthoc_query_comparison_evaluable") == bool(record.get("posthoc_query_comparison", {}).get("extraction_evaluable")), "DIRECT_APPLE_RECORD_INVALID", "comparison flag drift")
    for field in ("query_receipt_sha256", "current_common_point_ids_sha256", "r1_query_record_sha256", "candidate_binding_sha256", "source_scale_record_sha256"):
        _hash(record.get(field), field=field)
    if record.get("source_support_available"):
        _hash(record.get("direct_apple_support_plane_sha256"), field="direct_apple_support_plane_sha256")
    else:
        _require(record.get("direct_apple_support_plane_sha256") is None, "DIRECT_APPLE_RECORD_INVALID", "failed support carries plane hash")
    return record


def _parent_macro(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    by_frame: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if _finite(value):
            by_frame[(row["parent_id"], row["physical_frame_id"])].append(float(value))
    by_parent: dict[str, list[float]] = defaultdict(list)
    for (parent, _), values in by_frame.items():
        by_parent[parent].append(float(np.median(np.asarray(values, dtype=np.float64))))
    values = [float(np.median(np.asarray(items, dtype=np.float64))) for items in by_parent.values()]
    return {"parents_with_metric": len(values), "median_of_parent_medians": float(np.median(np.asarray(values, dtype=np.float64))) if values else None}


def summarize_direct_apple(
    records: Sequence[Mapping[str, Any]],
    source_failures: Sequence[Mapping[str, Any]],
    *,
    expected_query_count: int = 112,
    expected_frame_count: int = 14,
) -> dict[str, Any]:
    rows = [validate_direct_apple_query_record(dict(row)) for row in records]
    _require(len(rows) == expected_query_count and len({(row["physical_frame_id"], row["query_id"]) for row in rows}) == expected_query_count and len({row["physical_frame_id"] for row in rows}) == expected_frame_count, "DIRECT_APPLE_COHORT_DRIFT", "lost cohort drift")
    evaluable = [row for row in rows if row["posthoc_query_comparison_evaluable"]]
    reasons = Counter(code for row in rows if not row["posthoc_query_comparison_evaluable"] for code in row["posthoc_query_comparison"]["reason_codes"])
    return _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "analysis_kind": ANALYSIS_KIND,
            "claim_ceiling": CLAIM_CEILING,
            "method_id": METHOD_ID,
            "r1_lost_frame_count": expected_frame_count,
            "r1_lost_query_count": expected_query_count,
            "source_support_frame_count": len({row["physical_frame_id"] for row in rows if row["source_support_available"]}),
            "source_failure_frame_count": len(source_failures),
            "source_failure_frames": list(source_failures),
            "posthoc_evaluable_query_count": len(evaluable),
            "support_no_regret_query_count": sum(row["support_no_regret_vs_r1_baseline"] for row in rows),
            "support_no_regret_frame_count": len({row["physical_frame_id"] for row in rows if row["support_no_regret_vs_r1_baseline"]}),
            "height_improved_query_count": sum(_finite(row["height_error_reduction_vs_r1_baseline_m"]) and float(row["height_error_reduction_vs_r1_baseline_m"]) > 0.0 for row in rows),
            "normal_improved_query_count": sum(_finite(row["normal_error_reduction_vs_r1_baseline_rad"]) and float(row["normal_error_reduction_vs_r1_baseline_rad"]) > 0.0 for row in rows),
            "boundary_evaluable_query_count": sum(row["posthoc_query_comparison"]["boundary"]["evaluable"] for row in rows),
            "query_known_count": sum(row["posthoc_query_comparison"]["query_point_clearance"]["evaluable"] for row in rows),
            "posthoc_unknown_reason_counts": dict(sorted(reasons.items())),
            "height_error_reduction_parent_macro_m": _parent_macro(rows, "height_error_reduction_vs_r1_baseline_m"),
            "normal_error_reduction_parent_macro_rad": _parent_macro(rows, "normal_error_reduction_vs_r1_baseline_rad"),
            "threshold_or_pass_fail_decision_applied": False,
        }
    )


__all__ = [
    "ANALYSIS_KIND", "CAMERA_HEIGHT_RANGE_M", "CLAIM_CEILING", "METHOD_ID", "DirectAppleSupportError", "DirectAppleSupportPlane",
    "SOURCE_RECEIPT_SCHEMA", "build_direct_apple_source_receipt", "derive_direct_apple_support_plane",
    "evaluate_direct_apple_query", "load_direct_apple_support_plane", "prepare_direct_source_candidate",
    "summarize_direct_apple", "validate_direct_apple_query_record", "validate_direct_apple_source_receipt",
    "validate_direct_apple_support_plane_record",
]
