#!/usr/bin/env python3
"""Source-only prospective TARO factor runtime for the R6-confirmed policy."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_FACTOR_POLICY_ADOPTION_AND_PROSPECTIVE_RUNTIME_PROTOCOL_LOCK_2026-08-11.json"
PROTOCOL_LOCK_SHA256 = "B0A33DAB1532C3E3737BD213BDBB3933ACCF4D1775E9FCE8AB3F211625400296"
PROTOCOL_LOCK_BYTES = 7452
QUERY_FRAME_REPAIR_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_PROSPECTIVE_RUNTIME_QUERY_FRAME_PRE_IMPLEMENTATION_REPAIR_2026-08-11.json"
QUERY_FRAME_REPAIR_SHA256 = "08B56A3C7BD673958DCCE9CAB07608C4251A09514B93659790029A0DF48DDAE2"
QUERY_FRAME_REPAIR_BYTES = 2028
RUNTIME_ID = "FACTOR_OWNER_DEPTH_SOURCE_DEFINED_LOCAL_SURFACE_V1"
POLICY_ID = "R5_SELECTED_SUPPORT_BOUNDARY_PLUS_ALWAYS_R1_QUERY_CLEARANCE_V1"
QUERY_SCHEMA = "blindassist.taro.o0r.r6_source_defined_query_receipt.v1"
QUERY_FACTOR_SCHEMA = "blindassist.taro.o0r.r6_prospective_factor_query.v1"
BUNDLE_SCHEMA = "blindassist.taro.o0r.r6_prospective_factor_bundle.v1"
FORBIDDEN_R6_UNTOUCHED_PARENTS = frozenset({"423306", "435329", "466652", "467175", "467312", "469650", "469830", "470439"})
_SHA256 = re.compile(r"^[0-9A-F]{64}$")
_FORBIDDEN_RUNTIME_KEY_TOKENS = ("faro", "truth", "outcome", "task_metric")


class ProspectiveFactorRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise ProspectiveFactorRuntimeError(code, message, **context)


def _hash(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "R6_RUNTIME_HASH_INVALID", "runtime SHA-256 binding is malformed", field=field)
    return value


def _assert_protocol_binding() -> None:
    require(PROTOCOL_PATH.is_file() and PROTOCOL_PATH.stat().st_size == PROTOCOL_LOCK_BYTES, "R6_RUNTIME_PROTOCOL_BINDING_DRIFT", "prospective runtime protocol file size drift")
    require(hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest().upper() == PROTOCOL_LOCK_SHA256, "R6_RUNTIME_PROTOCOL_BINDING_DRIFT", "prospective runtime protocol file hash drift")
    require(QUERY_FRAME_REPAIR_PATH.is_file() and QUERY_FRAME_REPAIR_PATH.stat().st_size == QUERY_FRAME_REPAIR_BYTES, "R6_RUNTIME_REPAIR_BINDING_DRIFT", "query-frame repair file size drift")
    require(hashlib.sha256(QUERY_FRAME_REPAIR_PATH.read_bytes()).hexdigest().upper() == QUERY_FRAME_REPAIR_SHA256, "R6_RUNTIME_REPAIR_BINDING_DRIFT", "query-frame repair file hash drift")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R6_RUNTIME_SEAL_COLLISION", "caller supplied a runtime content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R6_RUNTIME_RECORD_INVALID", "runtime record must be an object", schema=schema)
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)) and adapter.canonical_sha256(record) == observed, "R6_RUNTIME_SEAL_MISMATCH", "runtime record seal drift", schema=schema)
    record["content_sha256"] = observed
    require(record.get("schema") == schema, "R6_RUNTIME_SCHEMA_DRIFT", "runtime record schema drift", expected=schema)
    return record


def _reject_forbidden_keys(value: Any, path: str = "runtime") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            require(not any(token in normalized for token in _FORBIDDEN_RUNTIME_KEY_TOKENS), "R6_RUNTIME_FORBIDDEN_FIELD", "runtime record contains a forbidden result-side field", path=f"{path}.{key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _unproject_valid(depth_m: np.ndarray, valid: np.ndarray, matrix: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(valid)
    keep = (rows % stride == 0) & (columns % stride == 0)
    rows = rows[keep]
    columns = columns[keep]
    z = depth_m[rows, columns]
    points = np.stack(
        (
            (columns - matrix[0, 2]) * z / matrix[0, 0],
            (rows - matrix[1, 2]) * z / matrix[1, 1],
            z,
        ),
        axis=1,
    ) if len(z) else np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(np.stack((columns, rows), axis=1), dtype=np.int32) if len(z) else np.empty((0, 2), dtype=np.int32)


def _plane_record(plane: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluable": True,
        "reason_codes": [],
        "normal_camera_xyz": np.asarray(plane["normal_camera_xyz"], dtype=np.float64).tolist(),
        "camera_height_m": float(plane["camera_height_m"]),
        "support_point_count": int(plane["support_count"]),
        "sampled_valid_point_count": int(plane["sampled_valid_points"]),
        "support_fraction": float(plane["support_fraction"]),
        "slope_degrees": float(plane["slope_degrees"]),
        "median_residual_m": float(plane["median_residual_m"]),
    }


def _failed_plane(code: str) -> dict[str, Any]:
    return {
        "evaluable": False,
        "reason_codes": [str(code)],
        "normal_camera_xyz": None,
        "camera_height_m": None,
        "support_point_count": 0,
        "sampled_valid_point_count": 0,
        "support_fraction": None,
        "slope_degrees": None,
        "median_residual_m": None,
    }


def _fit_depth_plane(depth_m: np.ndarray, matrix: np.ndarray, gravity: np.ndarray) -> dict[str, Any]:
    valid = (depth_m >= adapter.DEPTH_RANGE_M[0]) & (depth_m <= adapter.DEPTH_RANGE_M[1])
    points, _ = _unproject_valid(depth_m, valid, matrix, adapter.SUPPORT_POINT_STRIDE)
    try:
        plane = adapter._fit_support_plane(points, gravity)
        require(0.45 <= float(plane["camera_height_m"]) <= 2.2, "R6_RUNTIME_BASELINE_HEIGHT_IMPLAUSIBLE", "baseline support height leaves the frozen source-support range")
    except (adapter.AdapterError, ProspectiveFactorRuntimeError) as error:
        return _failed_plane(error.code)
    return _plane_record(plane)


def _fit_direct_plane(apple_depth_mm: np.ndarray, confidence: np.ndarray, matrix: np.ndarray, gravity: np.ndarray) -> dict[str, Any]:
    apple_m = apple_depth_mm.astype(np.float64) / 1000.0
    valid = (confidence == 2) & (apple_m >= adapter.DEPTH_RANGE_M[0]) & (apple_m <= adapter.DEPTH_RANGE_M[1])
    points, _ = _unproject_valid(apple_m, valid, matrix, 1)
    try:
        plane = adapter._fit_support_plane(points, gravity)
        require(0.45 <= float(plane["camera_height_m"]) <= 2.2, "R6_RUNTIME_DIRECT_HEIGHT_IMPLAUSIBLE", "direct Apple support height leaves the frozen range")
    except (adapter.AdapterError, ProspectiveFactorRuntimeError) as error:
        return _failed_plane(str(getattr(error, "code", type(error).__name__)))
    return _plane_record(plane)


@dataclass(frozen=True)
class _Geometry:
    depth_sha256: str
    valid_depth: np.ndarray
    points: np.ndarray
    pixels_uv: np.ndarray


def _build_geometry(depth_m: np.ndarray, depth_sha256: str, matrix: np.ndarray) -> _Geometry:
    valid = np.ascontiguousarray((depth_m >= adapter.DEPTH_RANGE_M[0]) & (depth_m <= adapter.DEPTH_RANGE_M[1]), dtype=np.bool_)
    points, pixels = _unproject_valid(depth_m, valid, matrix, 1)
    valid.flags.writeable = False
    points.flags.writeable = False
    pixels.flags.writeable = False
    return _Geometry(depth_sha256, valid, points, pixels)


def _signed_token(value: float, places: int) -> str:
    rounded = round(float(value), places)
    return f"{0.0 if rounded == 0.0 else rounded:+.{places}f}"


def _build_queries(physical_frame_id: str, source_receipt_sha256: str, max_source_timestamp_ns: int, plane: Mapping[str, Any]) -> list[dict[str, Any]]:
    normal = adapter._normalize_vector(plane["normal_camera_xyz"], "R6_RUNTIME_QUERY_NORMAL_INVALID")
    height = float(plane["camera_height_m"])
    optical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward = adapter._normalize_vector(optical - float(np.dot(optical, normal)) * normal, "R6_RUNTIME_QUERY_FORWARD_INVALID")
    if float(np.dot(forward, optical)) < 0.0:
        forward = -forward
    lateral = adapter._normalize_vector(np.cross(forward, normal), "R6_RUNTIME_QUERY_LATERAL_INVALID")
    origin = -height * normal
    output = []
    for grid_index, (offset, yaw) in enumerate((offset, yaw) for offset in adapter.PATH_LATERAL_OFFSETS_M for yaw in adapter.PATH_YAW_DEGREES):
        radians = math.radians(yaw)
        heading = adapter._normalize_vector(math.cos(radians) * forward + math.sin(radians) * lateral, "R6_RUNTIME_QUERY_HEADING_INVALID")
        path_id = f"lat_{_signed_token(offset, 2)}_yaw_{_signed_token(yaw, 1)}"
        output.append(
            _seal(
                {
                    "schema": QUERY_SCHEMA,
                    "physical_frame_id": physical_frame_id,
                    "source_frame_receipt_sha256": source_receipt_sha256,
                    "query_id": f"{physical_frame_id}:{path_id}",
                    "path_id": path_id,
                    "grid_index": grid_index,
                    "grid_order": "LATERAL_MAJOR_THEN_YAW_ASCENDING",
                    "path_lateral_offset_m": float(offset),
                    "path_yaw_degrees": float(yaw),
                    "minimum_forward_m": adapter.MINIMUM_FORWARD_M,
                    "horizon_m": adapter.HORIZON_M,
                    "capsule_radius_m": adapter.CAPSULE_RADIUS_M,
                    "virtual_query_frame": {
                        "kind": "SOURCE_DEFINED_SELECTED_SUPPORT_PLANE_V1",
                        "origin_camera_xyz": origin.tolist(),
                        "forward_camera_xyz": forward.tolist(),
                        "lateral_camera_xyz": lateral.tolist(),
                        "gravity_up_camera_xyz": normal.tolist(),
                        "path_heading_camera_xyz": heading.tolist(),
                    },
                    "max_source_timestamp_ns": int(max_source_timestamp_ns),
                }
            )
        )
    return output


def _validate_query(value: Any) -> dict[str, Any]:
    query = _validate_seal(value, QUERY_SCHEMA)
    expected = {
        "schema", "physical_frame_id", "source_frame_receipt_sha256", "query_id", "path_id", "grid_index", "grid_order",
        "path_lateral_offset_m", "path_yaw_degrees", "minimum_forward_m", "horizon_m", "capsule_radius_m",
        "virtual_query_frame", "max_source_timestamp_ns", "content_sha256",
    }
    require(set(query) == expected, "R6_RUNTIME_QUERY_FIELD_DRIFT", "source-defined query fields drift")
    index = query["grid_index"]
    require(isinstance(index, int) and 0 <= index < 9, "R6_RUNTIME_QUERY_GRID_DRIFT", "source-defined query index drift")
    require(query["grid_order"] == "LATERAL_MAJOR_THEN_YAW_ASCENDING" and query["path_lateral_offset_m"] == adapter.PATH_LATERAL_OFFSETS_M[index // 3] and query["path_yaw_degrees"] == adapter.PATH_YAW_DEGREES[index % 3], "R6_RUNTIME_QUERY_GRID_DRIFT", "source-defined query grid drift")
    require(query["virtual_query_frame"].get("kind") == "SOURCE_DEFINED_SELECTED_SUPPORT_PLANE_V1", "R6_RUNTIME_QUERY_FRAME_DRIFT", "source-defined query frame kind drift")
    adapter._query_receipt_vectors(query)
    _hash(query["source_frame_receipt_sha256"], "query.source_frame_receipt_sha256")
    return query


def _local_valid_fraction(geometry: _Geometry, matrix: np.ndarray, query: Mapping[str, Any]) -> float:
    origin, _, lateral, heading = adapter._query_receipt_vectors(dict(query))
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "R6_RUNTIME_QUERY_FRAME_INVALID")
    side = adapter._normalize_vector(np.cross(heading, up), "R6_RUNTIME_QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    half_h, half_w = adapter.BOUNDARY_NEIGHBORHOOD_HW[0] // 2, adapter.BOUNDARY_NEIGHBORHOOD_HW[1] // 2
    valid_count = total_count = 0
    for forward_m in np.linspace(adapter.MINIMUM_FORWARD_M, adapter.HORIZON_M, 10):
        for lateral_m in (-adapter.CAPSULE_RADIUS_M, 0.0, adapter.CAPSULE_RADIUS_M):
            point = path_origin + forward_m * heading + lateral_m * side
            if point[2] <= 1e-9:
                total_count += adapter.BOUNDARY_NEIGHBORHOOD_HW[0] * adapter.BOUNDARY_NEIGHBORHOOD_HW[1]
                continue
            column = int(round(float(matrix[0, 0] * point[0] / point[2] + matrix[0, 2])))
            row = int(round(float(matrix[1, 1] * point[1] / point[2] + matrix[1, 2])))
            for dr in range(-half_h, half_h + 1):
                for dc in range(-half_w, half_w + 1):
                    rr, cc = row + dr, column + dc
                    total_count += 1
                    if 0 <= rr < geometry.valid_depth.shape[0] and 0 <= cc < geometry.valid_depth.shape[1] and bool(geometry.valid_depth[rr, cc]):
                        valid_count += 1
    return valid_count / float(total_count) if total_count else 0.0


def _surface(geometry: _Geometry, plane: Mapping[str, Any], query: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    origin, _, lateral, heading = adapter._query_receipt_vectors(dict(query))
    up = adapter._normalize_vector(query["virtual_query_frame"]["gravity_up_camera_xyz"], "R6_RUNTIME_QUERY_FRAME_INVALID")
    side = adapter._normalize_vector(np.cross(heading, up), "R6_RUNTIME_QUERY_FRAME_INVALID")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    rel = geometry.points - path_origin
    along = rel @ heading
    across = rel @ side
    normal = adapter._normalize_vector(plane["normal_camera_xyz"], "R6_RUNTIME_SUPPORT_INVALID")
    height = geometry.points @ normal + float(plane["camera_height_m"])
    keep = (along >= 0.0) & (along <= 2.2) & (np.abs(across) <= 2.3) & (height >= -0.2) & (height <= 2.2)
    return np.ascontiguousarray(geometry.points[keep], dtype=np.float64), np.ascontiguousarray(geometry.pixels_uv[keep], dtype=np.int32)


def _unknown_block(owner: str, depth_sha256: str, code: str) -> dict[str, Any]:
    return {"owner": owner, "depth_array_sha256": depth_sha256, "evaluable": False, "reason_codes": [code], "value": None, "validity": {"known": False}}


def _query_blocks(
    query: dict[str, Any], matrix: np.ndarray, selected_owner: str, selected_geometry: _Geometry | None,
    selected_plane: Mapping[str, Any] | None, baseline_geometry: _Geometry | None, baseline_plane: Mapping[str, Any] | None,
    selected_failure: str, baseline_failure: str,
) -> dict[str, Any]:
    if selected_geometry is None or selected_plane is None:
        support = _unknown_block(selected_owner, "0" * 64, selected_failure)
        boundary = _unknown_block(selected_owner, "0" * 64, selected_failure)
    else:
        selected_points, selected_pixels = _surface(selected_geometry, selected_plane, query)
        selected_local = _local_valid_fraction(selected_geometry, matrix, query)
        selected_query_geometry = adapter._query_support_and_boundary(selected_points, selected_pixels, selected_plane["normal_camera_xyz"], float(selected_plane["camera_height_m"]), query)
        support = {
            "owner": selected_owner,
            "depth_array_sha256": selected_geometry.depth_sha256,
            "evaluable": True,
            "reason_codes": [],
            "value": {"normal_camera_xyz": list(selected_plane["normal_camera_xyz"]), "camera_height_m": float(selected_plane["camera_height_m"])},
            "validity": {
                "known": True,
                "support_point_count": int(selected_plane["support_point_count"]),
                "query_support_points": int(selected_query_geometry["query_support_points"]),
                "observed_forward_m": selected_query_geometry["observed_forward_shape_m"],
                "support_fraction": float(selected_plane["support_fraction"]),
                "slope_degrees": float(selected_plane["slope_degrees"]),
            },
        }
        boundary_valid = selected_local >= adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
        boundary_points = selected_query_geometry["boundary_points_shape_camera_xyz"]
        boundary_ids = selected_query_geometry["boundary_point_ids_uv"]
        boundary = {
            "owner": selected_owner,
            "depth_array_sha256": selected_geometry.depth_sha256,
            "evaluable": boundary_valid,
            "reason_codes": [] if boundary_valid else ["SOURCE_BOUNDARY_LOCAL_VALID_FRACTION_INSUFFICIENT"],
            "value": {
                "point_count": int(len(boundary_ids)),
                "point_ids_uv_sha256": adapter.canonical_sha256(boundary_ids),
                "points_camera_xyz_sha256": adapter.canonical_sha256(boundary_points),
            } if boundary_valid else None,
            "validity": {
                "known": boundary_valid,
                "source_surface_point_count": int(len(selected_pixels)),
                "source_surface_pixel_ids_sha256": adapter.canonical_sha256(selected_pixels),
                "local_valid_fraction": selected_local,
            },
        }
    if baseline_geometry is None or baseline_plane is None:
        clearance = _unknown_block("R1_BASELINE", "0" * 64, baseline_failure)
    else:
        baseline_points, baseline_pixels = _surface(baseline_geometry, baseline_plane, query)
        baseline_local = _local_valid_fraction(baseline_geometry, matrix, query)
        baseline_query_geometry = adapter._query_support_and_boundary(baseline_points, baseline_pixels, baseline_plane["normal_camera_xyz"], float(baseline_plane["camera_height_m"]), query)
        value = source_factor._point_clearance(
            np.asarray(baseline_plane["normal_camera_xyz"], dtype=np.float64),
            float(baseline_plane["camera_height_m"]),
            baseline_query_geometry["boundary_points_shape_camera_xyz"],
            int(baseline_query_geometry["query_support_points"]),
            baseline_query_geometry["observed_forward_shape_m"],
            baseline_local,
            query,
        )
        clearance = {
            "owner": "R1_BASELINE",
            "depth_array_sha256": baseline_geometry.depth_sha256,
            "evaluable": bool(value["evaluable"]),
            "reason_codes": list(value["reason_codes"]),
            "value": {"clearance_m": value["value_m"]} if value["evaluable"] else None,
            "validity": {
                "known": bool(value["evaluable"]),
                "query_support_points": int(value["query_support_points"]),
                "observed_forward_m": value["observed_forward_m"],
                "local_valid_fraction": float(value["local_valid_fraction"]),
                "source_surface_point_count": int(len(baseline_pixels)),
                "source_surface_pixel_ids_sha256": adapter.canonical_sha256(baseline_pixels),
            },
        }
    return {"SUPPORT": support, "BOUNDARY": boundary, "QUERY_CLEARANCE": clearance}


def _placeholder_query(physical_frame_id: str, source_sha256: str, grid_index: int, depth_sha256: str) -> dict[str, Any]:
    offset = adapter.PATH_LATERAL_OFFSETS_M[grid_index // 3]
    yaw = adapter.PATH_YAW_DEGREES[grid_index % 3]
    path_id = f"lat_{_signed_token(offset, 2)}_yaw_{_signed_token(yaw, 1)}"
    blocks = {name: _unknown_block("R1_BASELINE", depth_sha256, "SOURCE_QUERY_FRAME_UNAVAILABLE") for name in ("SUPPORT", "BOUNDARY", "QUERY_CLEARANCE")}
    return _seal(
        {
            "schema": QUERY_FACTOR_SCHEMA,
            "policy_id": POLICY_ID,
            "runtime_id": RUNTIME_ID,
            "physical_frame_id": physical_frame_id,
            "source_frame_receipt_sha256": source_sha256,
            "query_id": f"{physical_frame_id}:{path_id}",
            "grid_index": grid_index,
            "query_receipt": None,
            "query_frame_owner": "UNAVAILABLE",
            "factor_blocks": blocks,
            "final_state": "UNKNOWN",
            "final_state_authorized": False,
        }
    )


def build_prospective_factor_bundle(
    *,
    parent_id: str,
    video_id: str,
    timestamp_token: str,
    source_frame_receipt_sha256: str,
    candidate_frame_record_sha256: str,
    max_source_timestamp_ns: int,
    candidate_highres_depth_m: np.ndarray,
    apple_depth_mm: np.ndarray,
    confidence: np.ndarray,
    intrinsics_highres_3x3: Any,
    intrinsics_apple_3x3: Any,
    gravity_up_camera_xyz: Any,
) -> dict[str, Any]:
    """Build nine source-only factor slots; the signature intentionally has no result-side input."""

    _assert_protocol_binding()
    require(isinstance(parent_id, str) and parent_id and parent_id not in FORBIDDEN_R6_UNTOUCHED_PARENTS, "R6_RUNTIME_PARENT_ROLE_FORBIDDEN", "R6 untouched confirmation parent cannot enter prospective implementation/runtime", parent_id=parent_id)
    require(isinstance(video_id, str) and video_id and isinstance(timestamp_token, str) and timestamp_token, "R6_RUNTIME_IDENTITY_INVALID", "runtime frame identity is incomplete")
    physical_frame_id = f"{video_id}:{timestamp_token}"
    source_sha = _hash(source_frame_receipt_sha256, "source_frame_receipt_sha256")
    candidate_sha = _hash(candidate_frame_record_sha256, "candidate_frame_record_sha256")
    require(isinstance(max_source_timestamp_ns, int) and not isinstance(max_source_timestamp_ns, bool) and max_source_timestamp_ns >= 0, "R6_RUNTIME_WATERMARK_INVALID", "runtime watermark is invalid")
    high_k = adapter._intrinsics_matrix(intrinsics_highres_3x3)
    low_k = adapter._intrinsics_matrix(intrinsics_apple_3x3, adapter.APPLE_SHAPE_HW)
    gravity = adapter._normalize_vector(gravity_up_camera_xyz, "R6_RUNTIME_GRAVITY_INVALID")
    raw = np.asarray(candidate_highres_depth_m)
    apple = np.asarray(apple_depth_mm)
    conf = np.asarray(confidence)
    require(raw.shape == adapter.HIGHRES_SHAPE_HW and raw.dtype.kind == "f" and bool(np.all(np.isfinite(raw))), "R6_RUNTIME_CANDIDATE_INVALID", "candidate depth must be finite floating-point 1440x1920 metres")
    require(apple.shape == adapter.APPLE_SHAPE_HW and apple.dtype == np.uint16, "R6_RUNTIME_APPLE_INVALID", "AppleDepth must be uint16 192x256")
    require(conf.shape == adapter.APPLE_SHAPE_HW and conf.dtype == np.uint8 and bool(np.all(conf <= 2)), "R6_RUNTIME_CONFIDENCE_INVALID", "confidence must be uint8 0..2 at 192x256")
    raw64 = np.ascontiguousarray(raw, dtype=np.float64)
    raw_depth_sha = adapter.canonical_sha256(raw64)
    scale = apple_scale.estimate_source_metric_scale(apple, conf, apple_scale.sample_candidate_at_apple_centers(raw64))
    if scale["evaluable"]:
        anchored = np.ascontiguousarray(raw64 * float(scale["metric_scale"]), dtype=np.float64)
        direct_plane = _fit_direct_plane(apple, conf, low_k, gravity)
    else:
        anchored = np.ascontiguousarray(raw64, dtype=np.float64)
        direct_plane = _failed_plane(str(scale["reason_codes"][0]))
    anchored_depth_sha = adapter.canonical_sha256(anchored)
    baseline_plane = _fit_depth_plane(raw64, high_k, gravity)
    direct_available = bool(scale["evaluable"] and direct_plane["evaluable"])
    baseline_available = bool(baseline_plane["evaluable"])
    selected_owner = "DIRECT_APPLE_SUPPORT" if direct_available else "R1_BASELINE"
    selected_plane = direct_plane if direct_available else baseline_plane if baseline_available else None
    query_frame_owner = selected_owner if selected_plane is not None else "UNAVAILABLE"
    baseline_geometry = _build_geometry(raw64, raw_depth_sha, high_k) if baseline_available else None
    selected_geometry = _build_geometry(anchored, anchored_depth_sha, high_k) if direct_available else baseline_geometry
    if selected_plane is None:
        queries = [_placeholder_query(physical_frame_id, source_sha, index, raw_depth_sha) for index in range(9)]
    else:
        receipts = [_validate_query(row) for row in _build_queries(physical_frame_id, source_sha, max_source_timestamp_ns, selected_plane)]
        queries = []
        for receipt in receipts:
            blocks = _query_blocks(
                receipt, high_k, selected_owner, selected_geometry, selected_plane, baseline_geometry,
                baseline_plane if baseline_available else None,
                str(selected_plane["reason_codes"][0]) if not selected_plane["evaluable"] else "SOURCE_SUPPORT_UNAVAILABLE",
                str(baseline_plane["reason_codes"][0]) if not baseline_available else "SOURCE_BASELINE_UNAVAILABLE",
            )
            queries.append(
                _seal(
                    {
                        "schema": QUERY_FACTOR_SCHEMA,
                        "policy_id": POLICY_ID,
                        "runtime_id": RUNTIME_ID,
                        "physical_frame_id": physical_frame_id,
                        "source_frame_receipt_sha256": source_sha,
                        "query_id": receipt["query_id"],
                        "grid_index": receipt["grid_index"],
                        "query_receipt": receipt,
                        "query_frame_owner": query_frame_owner,
                        "factor_blocks": blocks,
                        "final_state": "UNKNOWN",
                        "final_state_authorized": False,
                    }
                )
            )
    record = _seal(
        {
            "schema": BUNDLE_SCHEMA,
            "protocol_lock_sha256": PROTOCOL_LOCK_SHA256,
            "query_frame_repair_sha256": QUERY_FRAME_REPAIR_SHA256,
            "runtime_id": RUNTIME_ID,
            "policy_id": POLICY_ID,
            "parent_id": parent_id,
            "video_id": video_id,
            "timestamp_token": timestamp_token,
            "physical_frame_id": physical_frame_id,
            "source_frame_receipt_sha256": source_sha,
            "candidate_frame_record_sha256": candidate_sha,
            "max_source_timestamp_ns": max_source_timestamp_ns,
            "input_bindings": {
                "candidate_highres_depth_sha256": raw_depth_sha,
                "anchored_candidate_depth_sha256": anchored_depth_sha,
                "apple_depth_sha256": adapter.canonical_sha256(apple),
                "confidence_sha256": adapter.canonical_sha256(conf),
                "intrinsics_highres_sha256": adapter.canonical_sha256(high_k),
                "intrinsics_apple_sha256": adapter.canonical_sha256(low_k),
                "gravity_up_camera_xyz_sha256": adapter.canonical_sha256(gravity),
            },
            "source_scale": scale,
            "baseline_support": baseline_plane,
            "direct_support": direct_plane,
            "selected_support_boundary_owner": selected_owner,
            "query_clearance_owner": "R1_BASELINE",
            "query_frame_owner": query_frame_owner,
            "query_slots": queries,
            "phase_a_payload_roles": ["APPLE_DEPTH", "CONFIDENCE", "INTRINSICS", "SEALED_CANDIDATE_DEPTH", "TRAJECTORY"],
            "forbidden_payload_reads_confirmed_absent": True,
            "training_steps": 0,
            "network_requests": 0,
            "uncertainty_attached": False,
            "deterministic_reducer_executed": False,
        }
    )
    return validate_prospective_factor_bundle(record, candidate_highres_depth_m=raw64)


def _validate_block(name: str, block: Any, bundle: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(block, dict) and set(block) == {"owner", "depth_array_sha256", "evaluable", "reason_codes", "value", "validity"}, "R6_RUNTIME_FACTOR_BLOCK_DRIFT", "runtime factor block fields drift", factor=name)
    _hash(block["depth_array_sha256"], f"{name}.depth_array_sha256")
    require(isinstance(block["evaluable"], bool) and isinstance(block["reason_codes"], list) and isinstance(block["validity"], dict), "R6_RUNTIME_FACTOR_BLOCK_DRIFT", "runtime factor validity metadata drift", factor=name)
    require((block["evaluable"] and block["reason_codes"] == [] and block["value"] is not None) or (not block["evaluable"] and bool(block["reason_codes"]) and block["value"] is None), "R6_RUNTIME_FACTOR_BLOCK_DRIFT", "runtime factor evaluability/reasons drift", factor=name)
    raw_hash = bundle["input_bindings"]["candidate_highres_depth_sha256"]
    anchored_hash = bundle["input_bindings"]["anchored_candidate_depth_sha256"]
    selected_owner = bundle["selected_support_boundary_owner"]
    if name in ("SUPPORT", "BOUNDARY"):
        require(block["owner"] == selected_owner and block["depth_array_sha256"] == (anchored_hash if selected_owner == "DIRECT_APPLE_SUPPORT" else raw_hash), "R6_RUNTIME_FACTOR_DEPTH_LINEAGE_DRIFT", "support/boundary depth owner drift", factor=name)
    if name == "QUERY_CLEARANCE":
        require(block["owner"] == "R1_BASELINE" and block["depth_array_sha256"] == raw_hash, "R6_RUNTIME_FACTOR_DEPTH_LINEAGE_DRIFT", "query-clearance depth owner drift")
    surface_hash = block["validity"].get("source_surface_pixel_ids_sha256")
    if surface_hash is not None:
        _hash(surface_hash, f"{name}.source_surface_pixel_ids_sha256")
    return block


def validate_prospective_factor_bundle(value: Any, *, candidate_highres_depth_m: np.ndarray | None = None) -> dict[str, Any]:
    _assert_protocol_binding()
    bundle = _validate_seal(value, BUNDLE_SCHEMA)
    expected = {
        "schema", "protocol_lock_sha256", "query_frame_repair_sha256", "runtime_id", "policy_id", "parent_id", "video_id", "timestamp_token", "physical_frame_id",
        "source_frame_receipt_sha256", "candidate_frame_record_sha256", "max_source_timestamp_ns", "input_bindings", "source_scale",
        "baseline_support", "direct_support", "selected_support_boundary_owner", "query_clearance_owner", "query_frame_owner", "query_slots",
        "phase_a_payload_roles", "forbidden_payload_reads_confirmed_absent", "training_steps", "network_requests", "uncertainty_attached",
        "deterministic_reducer_executed", "content_sha256",
    }
    require(set(bundle) == expected and bundle["protocol_lock_sha256"] == PROTOCOL_LOCK_SHA256 and bundle["query_frame_repair_sha256"] == QUERY_FRAME_REPAIR_SHA256 and bundle["runtime_id"] == RUNTIME_ID and bundle["policy_id"] == POLICY_ID, "R6_RUNTIME_BUNDLE_FIELD_DRIFT", "prospective runtime bundle identity/fields drift")
    require(bundle["parent_id"] not in FORBIDDEN_R6_UNTOUCHED_PARENTS and bundle["physical_frame_id"] == f"{bundle['video_id']}:{bundle['timestamp_token']}", "R6_RUNTIME_PARENT_ROLE_FORBIDDEN", "prospective runtime bundle uses forbidden identity")
    for field in ("protocol_lock_sha256", "query_frame_repair_sha256", "source_frame_receipt_sha256", "candidate_frame_record_sha256"):
        _hash(bundle[field], field)
    bindings = bundle["input_bindings"]
    require(isinstance(bindings, dict) and set(bindings) == {"candidate_highres_depth_sha256", "anchored_candidate_depth_sha256", "apple_depth_sha256", "confidence_sha256", "intrinsics_highres_sha256", "intrinsics_apple_sha256", "gravity_up_camera_xyz_sha256"}, "R6_RUNTIME_INPUT_BINDING_DRIFT", "runtime input binding fields drift")
    for field, digest in bindings.items():
        _hash(digest, f"input_bindings.{field}")
    if candidate_highres_depth_m is not None:
        candidate = np.asarray(candidate_highres_depth_m)
        require(candidate.shape == adapter.HIGHRES_SHAPE_HW and adapter.canonical_sha256(np.ascontiguousarray(candidate, dtype=np.float64)) == bindings["candidate_highres_depth_sha256"], "R6_RUNTIME_CANDIDATE_ARRAY_DRIFT", "runtime candidate differs from bundle binding")
    plane_fields = {"evaluable", "reason_codes", "normal_camera_xyz", "camera_height_m", "support_point_count", "sampled_valid_point_count", "support_fraction", "slope_degrees", "median_residual_m"}
    for field in ("baseline_support", "direct_support"):
        plane = bundle[field]
        require(isinstance(plane, dict) and set(plane) == plane_fields and isinstance(plane["evaluable"], bool) and isinstance(plane["reason_codes"], list), "R6_RUNTIME_SUPPORT_RECORD_DRIFT", "runtime support record fields drift", field=field)
        require((plane["evaluable"] and plane["reason_codes"] == [] and plane["normal_camera_xyz"] is not None and plane["camera_height_m"] is not None) or (not plane["evaluable"] and bool(plane["reason_codes"]) and plane["normal_camera_xyz"] is None and plane["camera_height_m"] is None), "R6_RUNTIME_SUPPORT_RECORD_DRIFT", "runtime support record evaluability drift", field=field)
    scale = bundle["source_scale"]
    require(isinstance(scale, dict) and set(scale) == {"evaluable", "reason_codes", "valid_pair_count", "selected_pixel_ids_sha256", "log_metric_scale", "metric_scale"}, "R6_RUNTIME_SCALE_RECORD_DRIFT", "runtime source scale fields drift")
    require((scale["evaluable"] and scale["reason_codes"] == [] and scale["metric_scale"] is not None) or (not scale["evaluable"] and bool(scale["reason_codes"]) and scale["metric_scale"] is None), "R6_RUNTIME_SCALE_RECORD_DRIFT", "runtime source scale evaluability drift")
    direct_available = bool(scale["evaluable"] and bundle["direct_support"]["evaluable"])
    baseline_available = bool(bundle["baseline_support"]["evaluable"])
    expected_owner = "DIRECT_APPLE_SUPPORT" if direct_available else "R1_BASELINE"
    expected_query_owner = expected_owner if direct_available or baseline_available else "UNAVAILABLE"
    require(bundle["selected_support_boundary_owner"] == expected_owner and bundle["query_clearance_owner"] == "R1_BASELINE", "R6_RUNTIME_FACTOR_OWNER_DRIFT", "runtime factor owner drift")
    require(bundle["query_frame_owner"] == expected_query_owner, "R6_RUNTIME_QUERY_FRAME_OWNER_DRIFT", "runtime query frame owner drift")
    slots = bundle["query_slots"]
    require(isinstance(slots, list) and len(slots) == 9, "R6_RUNTIME_QUERY_SLOT_COUNT_DRIFT", "runtime bundle must retain nine query slots")
    for index, slot_value in enumerate(slots):
        slot = _validate_seal(slot_value, QUERY_FACTOR_SCHEMA)
        require(set(slot) == {"schema", "policy_id", "runtime_id", "physical_frame_id", "source_frame_receipt_sha256", "query_id", "grid_index", "query_receipt", "query_frame_owner", "factor_blocks", "final_state", "final_state_authorized", "content_sha256"}, "R6_RUNTIME_QUERY_SLOT_FIELD_DRIFT", "runtime query slot fields drift")
        require(slot["grid_index"] == index and slot["physical_frame_id"] == bundle["physical_frame_id"] and slot["source_frame_receipt_sha256"] == bundle["source_frame_receipt_sha256"], "R6_RUNTIME_QUERY_SLOT_IDENTITY_DRIFT", "runtime query slot identity/order drift")
        require(slot["policy_id"] == POLICY_ID and slot["runtime_id"] == RUNTIME_ID and slot["final_state"] == "UNKNOWN" and slot["final_state_authorized"] is False, "R6_RUNTIME_FINAL_STATE_AUTHORITY_DRIFT", "runtime query slot exceeded state authority")
        if slot["query_receipt"] is None:
            require(bundle["query_frame_owner"] == "UNAVAILABLE", "R6_RUNTIME_QUERY_RECEIPT_DRIFT", "runtime omitted an available query frame")
        else:
            query = _validate_query(slot["query_receipt"])
            require(query["query_id"] == slot["query_id"] and query["grid_index"] == index, "R6_RUNTIME_QUERY_RECEIPT_DRIFT", "runtime query receipt/slot drift")
        blocks = slot["factor_blocks"]
        require(isinstance(blocks, dict) and set(blocks) == {"SUPPORT", "BOUNDARY", "QUERY_CLEARANCE"}, "R6_RUNTIME_FACTOR_SET_DRIFT", "runtime factor set drift")
        for name in ("SUPPORT", "BOUNDARY", "QUERY_CLEARANCE"):
            _validate_block(name, blocks[name], bundle)
    require(bundle["phase_a_payload_roles"] == ["APPLE_DEPTH", "CONFIDENCE", "INTRINSICS", "SEALED_CANDIDATE_DEPTH", "TRAJECTORY"] and bundle["forbidden_payload_reads_confirmed_absent"] is True, "R6_RUNTIME_PHASE_A_FIREWALL_DRIFT", "runtime Phase-A firewall drift")
    require(bundle["training_steps"] == bundle["network_requests"] == 0 and bundle["uncertainty_attached"] is False and bundle["deterministic_reducer_executed"] is False, "R6_RUNTIME_AUTHORITY_DRIFT", "runtime exceeded implementation authority")
    _reject_forbidden_keys(bundle)
    return bundle


__all__ = [
    "BUNDLE_SCHEMA", "FORBIDDEN_R6_UNTOUCHED_PARENTS", "POLICY_ID", "PROTOCOL_LOCK_SHA256", "QUERY_FRAME_REPAIR_SHA256", "ProspectiveFactorRuntimeError",
    "QUERY_FACTOR_SCHEMA", "QUERY_SCHEMA", "RUNTIME_ID", "build_prospective_factor_bundle", "validate_prospective_factor_bundle",
]
