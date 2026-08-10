#!/usr/bin/env python3
"""Pure in-memory TARO O0R ARKitScenes source-adapter mechanics.

This module intentionally has no path, archive, network, model, evaluator, or
artifact-writing interface.  It freezes the deterministic seam between already
loaded source arrays and future separately governed truth-only/O0R runners.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import re
import weakref
from collections.abc import Sequence as RuntimeSequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage


SOURCE_ID = "ARKITSCENES_UPSAMPLING_V1_TRAIN"
BASE_RECEIPT_SCHEMA = "blindassist.taro.o0r.source_frame_receipt.v1"
QUERY_RECEIPT_SCHEMA = "blindassist.taro.o0r.query_receipt.v1"
QUERY_FACTOR_FRAME_SCHEMA = "blindassist.taro.o0r.query_factor_frame.v1"
BASE_GEOMETRY_SCHEMA = "blindassist.taro.o0r.query_base_geometry.v1"
CANDIDATE_OUTPUT_RECEIPT_SCHEMA = "blindassist.taro.o0r.candidate_depth_output_receipt.v1"
QUERY_RESULT_SCHEMA = "blindassist.taro.o0r.query_reducer_result.v1"
QUERY_BUNDLE_RESULT_SCHEMA = "blindassist.taro.o0r.nine_query_bundle_result.v1"
UNCERTAINTY_MODEL_SCHEMA = "blindassist.taro.o0r.uncertainty_model.v1"
REDUCER_VERSION = "taro_query_reducer_p0_contract_v1"
BASELINE_MODEL_ID = "depthart-s-metric-indoor-448-official-fp32"
BASELINE_CHECKPOINT_SHA256 = "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
FLOAT_DECIMALS = 12

HIGHRES_SHAPE_HW = (1440, 1920)
APPLE_SHAPE_HW = (192, 256)
LOWRES_TO_HIGHRES_SCALE_XY = (7.5, 7.5)
DEPTH_RANGE_M = (0.25, 6.0)
MAXIMUM_POSE_GAP_NS = 250_000_000
SUPPORT_POINT_STRIDE = 4
SUPPORT_RESIDUAL_TOLERANCE_M = 0.08
MINIMUM_SUPPORT_POINTS = 256
MINIMUM_SUPPORT_FRACTION = 0.02
MAXIMUM_SUPPORT_SLOPE_DEGREES = 20.0
OBSTACLE_HEIGHT_RANGE_M = (0.08, 1.8)
BOUNDARY_NEIGHBORHOOD_HW = (5, 5)
MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION = 0.8
MINIMUM_QUERY_SUPPORT_POINTS = 128
MINIMUM_QUERY_OBSERVED_FORWARD_M = 2.0
GEOMETRY_ENDPOINT_TOLERANCE_M = 0.001
MINIMUM_FORWARD_M = 0.2
HORIZON_M = 2.0
CAPSULE_RADIUS_M = 0.30
CLEAR_MARGIN_M = 0.05
OCCUPIED_MARGIN_M = 0.0
PATH_LATERAL_OFFSETS_M = (-0.35, 0.0, 0.35)
PATH_YAW_DEGREES = (-10.0, 0.0, 10.0)
FACTOR_NAMES = ("SCALE", "SUPPORT", "BOUNDARY")
ARMS = (
    "NONE",
    "SCALE",
    "SUPPORT",
    "BOUNDARY",
    "SCALE_SUPPORT",
    "SCALE_BOUNDARY",
    "SUPPORT_BOUNDARY",
    "SCALE_SUPPORT_BOUNDARY",
)
ORACLE_MODES = ("VALUE_ONLY_COMMON_SUPPORT", "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY")
RANGE_EDGES_M = (0.0, 1.0, 2.0, 3.0, 6.0)
UNCERTAINTY_TARGETS = (
    "scale_log_abs_residual",
    "support_height_abs_residual_m",
    "support_normal_abs_residual_rad",
    "boundary_localization_abs_residual_m",
)
_SUPPORT_UNOBSERVABLE_CODES = frozenset(
    {
        "SUPPORT_POINTS_INSUFFICIENT",
        "SUPPORT_PLAUSIBLE_INSUFFICIENT",
        "SUPPORT_HISTOGRAM_EMPTY",
        "SUPPORT_GATE_FAILED",
        "SUPPORT_NORMAL_INVALID",
        "SUPPORT_SLOPE_EXCEEDED",
        "SUPPORT_REFINED_GATE_FAILED",
    }
)

ADAPTER_FIT_ROSTER = (
    ("470974", "47332075"),
    ("469216", "47332946"),
    ("423614", "42898071"),
    ("467370", "47333776"),
    ("469460", "47333043"),
    ("438794", "44358241"),
    ("467346", "47333876"),
    ("472473", "47204786"),
)
O0R_EVAL_CANDIDATE_ROSTER = (
    ("466965", "45261294"),
    ("470808", "47430058"),
    ("482587", "47895909"),
    ("468410", "45261689"),
    ("482858", "47670295"),
    ("482984", "47670346"),
    ("469607", "47115143"),
    ("421593", "42445766"),
    ("423474", "42897405"),
    ("470876", "47332015"),
    ("464981", "44796438"),
    ("478016", "47204874"),
    ("422217", "42445723"),
    ("466437", "45260952"),
    ("484003", "48018757"),
    ("421655", "42445698"),
)
FROZEN_ROSTER = {
    "ADAPTER_FIT": ADAPTER_FIT_ROSTER,
    "O0R_EVAL_CANDIDATE": O0R_EVAL_CANDIDATE_ROSTER,
}
DECODED_PAYLOAD_KINDS = {
    "color": "RGB_UINT8_1440X1920X3",
    "highres_depth": "FARO_DEPTH_UINT16_MM_1440X1920",
    "lowres_depth": "APPLE_DEPTH_UINT16_MM_192X256",
    "confidence": "APPLE_CONFIDENCE_UINT8_192X256",
    "intrinsics": "LOWRES_PINCAM_CANONICAL_OBJECT",
    "trajectory": "CAMERA_TRAJECTORY_CANONICAL_ROWS",
}

_DECIMAL_TOKEN = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)$")
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")
_TRUTH_FORBIDDEN_KEYS = {
    "model_output",
    "model_outputs",
    "teacher",
    "teacher_output",
    "task_metric",
    "task_metrics",
    "outcome",
    "outcomes",
    "final_state",
    "clearance_m",
    "occupancy",
    "occupancy_logit",
    "truth_label",
}


class AdapterError(RuntimeError):
    """Fail-closed error with a stable machine code."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise AdapterError(code, message, **context)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def reject_truth_shortcuts(value: Any, path: str = "truth_input") -> None:
    """Reject model, teacher, outcome, or final-task shortcuts recursively."""

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_key(key)
            require(
                normalized not in _TRUTH_FORBIDDEN_KEYS,
                "FORBIDDEN_TRUTH_INPUT_FIELD",
                "truth input contains a forbidden model, teacher, task, or outcome field",
                path=f"{path}.{key}",
            )
            reject_truth_shortcuts(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_truth_shortcuts(child, f"{path}[{index}]")


def _canonical_float(value: float) -> float:
    require(math.isfinite(value), "NONFINITE_CANONICAL_VALUE", "NaN and infinity are forbidden")
    rounded = round(float(value), FLOAT_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _canonical_array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    require(array.dtype.kind in "biuf", "ARRAY_DTYPE_UNSUPPORTED", "only numeric and boolean arrays are canonical")
    if array.dtype.kind == "f":
        require(bool(np.all(np.isfinite(array))), "NONFINITE_CANONICAL_VALUE", "NaN and infinity are forbidden")
        normalized = np.round(array.astype(np.float64), FLOAT_DECIMALS)
        normalized[normalized == 0.0] = 0.0
        normalized = np.ascontiguousarray(normalized.astype("<f8", copy=False))
        dtype = "float64_round12_le"
    elif array.dtype.kind == "b":
        normalized = np.ascontiguousarray(array.astype(np.uint8, copy=False))
        dtype = "bool_u8"
    elif array.dtype.kind == "u":
        normalized = np.ascontiguousarray(array.astype("<u8", copy=False))
        dtype = "uint64_le"
    else:
        normalized = np.ascontiguousarray(array.astype("<i8", copy=False))
        dtype = "int64_le"
    return {
        "schema": "blindassist.canonical.ndarray_receipt.v1",
        "dtype": dtype,
        "shape": [int(item) for item in array.shape],
        "sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest().upper(),
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _canonical_array_receipt(value)
    if isinstance(value, (np.floating, float)):
        return _canonical_float(float(value))
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _canonicalize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(child) for child in value]
    if value is None or isinstance(value, str):
        return value
    raise AdapterError("CANONICAL_TYPE_UNSUPPORTED", f"unsupported canonical type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def frozen_roster_sha256() -> str:
    return canonical_sha256(
        {
            role: [
                {"visit_id": visit_id, "video_id": video_id}
                for visit_id, video_id in pairs
            ]
            for role, pairs in FROZEN_ROSTER.items()
        }
    )


def _validate_roster_identity(source_role: str, visit_id: str, video_id: str) -> None:
    require(source_role in FROZEN_ROSTER, "SOURCE_ROLE_INVALID", "source role is not frozen")
    require((visit_id, video_id) in FROZEN_ROSTER[source_role], "SOURCE_ROSTER_IDENTITY_MISMATCH", "visit/video pair is not in the frozen role roster", source_role=source_role, visit_id=visit_id, video_id=video_id)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(value)
    output.pop("content_sha256", None)
    output["content_sha256"] = canonical_sha256(output)
    return output


def _validate_seal(value: Any, code: str) -> dict[str, Any]:
    require(isinstance(value, dict), code, "sealed value must be an object")
    claimed = value.get("content_sha256")
    require(isinstance(claimed, str) and bool(_SHA256.fullmatch(claimed)), code, "content hash is missing or malformed")
    payload = dict(value)
    payload.pop("content_sha256")
    require(canonical_sha256(payload) == claimed.upper(), code, "content hash mismatch")
    return value


def decimal_timestamp_ns(token: str) -> int:
    """Parse a filename/trajectory decimal timestamp exactly to nanoseconds."""

    require(isinstance(token, str) and bool(_DECIMAL_TOKEN.fullmatch(token)), "TIMESTAMP_TOKEN_INVALID", "timestamp must be a non-negative plain decimal token")
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise AdapterError("TIMESTAMP_TOKEN_INVALID", "timestamp is not a Decimal") from error
    require(value.is_finite() and value >= 0, "TIMESTAMP_TOKEN_INVALID", "timestamp must be finite and non-negative")
    nanoseconds = value * Decimal(1_000_000_000)
    require(nanoseconds == nanoseconds.to_integral_value(), "TIMESTAMP_SUBNANOSECOND", "timestamp is not exactly representable in integer nanoseconds")
    return int(nanoseconds)


def _normalize_vector(value: Any, code: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    require(vector.shape == (3,) and bool(np.all(np.isfinite(vector))), code, "vector must be finite length three")
    norm = float(np.linalg.norm(vector))
    require(norm > 1e-12, code, "vector is degenerate")
    return vector / norm


def _rotation_vector_to_quaternion(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    require(vector.shape == (3,) and bool(np.all(np.isfinite(vector))), "TRAJECTORY_ROTATION_INVALID", "rotation vector must be finite length three")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = vector / angle
    half = angle / 2.0
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def _normalize_quaternion(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    require(quaternion.shape == (4,) and bool(np.all(np.isfinite(quaternion))), "QUATERNION_INVALID", "quaternion must be finite length four")
    norm = float(np.linalg.norm(quaternion))
    require(norm > 1e-12, "QUATERNION_INVALID", "quaternion is degenerate")
    return quaternion / norm


def _quaternion_to_rotation_matrix(value: Any) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(left: Any, right: Any, fraction: float) -> np.ndarray:
    require(math.isfinite(fraction) and 0.0 <= fraction <= 1.0, "POSE_INTERPOLATION_INVALID", "interpolation fraction is invalid")
    first = _normalize_quaternion(left)
    second = _normalize_quaternion(right)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(first + fraction * (second - first))
    angle = math.acos(dot)
    scale = math.sin(angle)
    return _normalize_quaternion(
        math.sin((1.0 - fraction) * angle) / scale * first
        + math.sin(fraction * angle) / scale * second
    )


@dataclass(frozen=True)
class _TrajectorySample:
    timestamp_token: str
    timestamp_ns: int
    rotation_vector: np.ndarray
    translation: np.ndarray


def _trajectory_samples(rows: Sequence[dict[str, Any]]) -> tuple[_TrajectorySample, ...]:
    require(isinstance(rows, Sequence) and len(rows) >= 2, "TRAJECTORY_INVALID", "at least two in-memory trajectory rows are required")
    output: list[_TrajectorySample] = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == {"timestamp_token", "rotation_vector", "translation"}, "TRAJECTORY_ROW_INVALID", "trajectory row key set drift", index=index)
        token = row["timestamp_token"]
        timestamp_ns = decimal_timestamp_ns(token)
        rotation = np.asarray(row["rotation_vector"], dtype=np.float64)
        translation = np.asarray(row["translation"], dtype=np.float64)
        require(rotation.shape == (3,) and translation.shape == (3,), "TRAJECTORY_ROW_INVALID", "trajectory vectors must have length three", index=index)
        require(bool(np.all(np.isfinite(rotation))) and bool(np.all(np.isfinite(translation))), "TRAJECTORY_ROW_INVALID", "trajectory vectors must be finite", index=index)
        output.append(_TrajectorySample(token, timestamp_ns, rotation, translation))
    require(all(left.timestamp_ns < right.timestamp_ns for left, right in zip(output, output[1:])), "TRAJECTORY_ORDER_INVALID", "trajectory timestamps must be strictly increasing")
    return tuple(output)


def _stored_sample_to_camera_world(sample: _TrajectorySample) -> tuple[np.ndarray, np.ndarray]:
    # Matches the official ARKitScenes loader: construct stored world-to-camera
    # angle-axis [R,t], then invert it to camera-to-world.
    world_to_camera = _rotation_vector_to_quaternion(sample.rotation_vector)
    camera_to_world = world_to_camera * np.asarray([1.0, -1.0, -1.0, -1.0])
    rotation = _quaternion_to_rotation_matrix(camera_to_world)
    translation = -(rotation @ sample.translation)
    return camera_to_world, translation


def interpolate_camera_to_world_exact(
    rows: Sequence[dict[str, Any]],
    frame_timestamp_token: str,
    maximum_gap_ns: int = MAXIMUM_POSE_GAP_NS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bound pose interpolation with exact-nanosecond bracket selection."""

    samples = _trajectory_samples(rows)
    frame_ns = decimal_timestamp_ns(frame_timestamp_token)
    require(isinstance(maximum_gap_ns, int) and not isinstance(maximum_gap_ns, bool) and maximum_gap_ns > 0, "POSE_GAP_POLICY_INVALID", "maximum pose gap must be positive integer nanoseconds")
    times = [sample.timestamp_ns for sample in samples]
    right = bisect.bisect_left(times, frame_ns)
    if right < len(samples) and samples[right].timestamp_ns == frame_ns:
        left = right
        fraction = 0.0
        gap_ns = 0
    else:
        require(0 < right < len(samples), "FRAME_OUTSIDE_TRAJECTORY", "frame timestamp lies outside trajectory domain")
        left = right - 1
        gap_ns = samples[right].timestamp_ns - samples[left].timestamp_ns
        require(gap_ns <= maximum_gap_ns, "POSE_BRACKET_GAP_EXCEEDED", "pose bracket exceeds frozen maximum", gap_ns=gap_ns)
        fraction = (frame_ns - samples[left].timestamp_ns) / gap_ns
    q_left, p_left = _stored_sample_to_camera_world(samples[left])
    if left == right:
        quaternion = q_left
        translation = p_left
    else:
        q_right, p_right = _stored_sample_to_camera_world(samples[right])
        quaternion = _slerp(q_left, q_right, fraction)
        translation = p_left + fraction * (p_right - p_left)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_to_rotation_matrix(quaternion)
    transform[:3, 3] = translation
    right_ns = samples[right].timestamp_ns
    return transform, {
        "frame_timestamp_token": frame_timestamp_token,
        "frame_timestamp_ns": frame_ns,
        "left_timestamp_token": samples[left].timestamp_token,
        "left_timestamp_ns": samples[left].timestamp_ns,
        "right_timestamp_token": samples[right].timestamp_token,
        "right_timestamp_ns": right_ns,
        "fraction": float(fraction),
        "bracketing_gap_ns": gap_ns,
        "max_source_timestamp_ns": max(frame_ns, right_ns),
    }


def scale_lowres_intrinsics(lowres: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen pixel-center 256x192 to 1920x1440 transform."""

    required = {"width", "height", "fx", "fy", "cx", "cy"}
    require(isinstance(lowres, dict) and set(lowres) == required, "INTRINSICS_KEY_SET", "lowres intrinsics key set drift")
    require(lowres["width"] == APPLE_SHAPE_HW[1] and lowres["height"] == APPLE_SHAPE_HW[0], "INTRINSICS_SOURCE_SHAPE", "lowres intrinsics must be 256x192")
    require(all(_finite_number(lowres[key]) for key in ("fx", "fy", "cx", "cy")), "INTRINSICS_NONFINITE", "intrinsics must be finite")
    fx, fy, cx, cy = (float(lowres[key]) for key in ("fx", "fy", "cx", "cy"))
    require(fx > 0.0 and fy > 0.0 and 0.0 <= cx < APPLE_SHAPE_HW[1] and 0.0 <= cy < APPLE_SHAPE_HW[0], "INTRINSICS_RANGE", "lowres intrinsics are out of range")
    sx, sy = LOWRES_TO_HIGHRES_SCALE_XY
    output = {
        "width": HIGHRES_SHAPE_HW[1],
        "height": HIGHRES_SHAPE_HW[0],
        "fx": sx * fx,
        "fy": sy * fy,
        "cx": sx * (cx + 0.5) - 0.5,
        "cy": sy * (cy + 0.5) - 0.5,
        "pixel_center_convention": "INTEGER_INDEX_IS_PIXEL_CENTER",
    }
    output["matrix_3x3"] = [
        [output["fx"], 0.0, output["cx"]],
        [0.0, output["fy"], output["cy"]],
        [0.0, 0.0, 1.0],
    ]
    return _canonicalize(output)


def _validate_asset_bindings(value: Any, frame_timestamp_token: str) -> dict[str, Any]:
    keys = {"color", "highres_depth", "lowres_depth", "confidence", "intrinsics", "trajectory"}
    require(isinstance(value, dict) and set(value) == keys, "ASSET_BINDING_KEY_SET", "exact source asset bindings are required")
    for role, receipt in value.items():
        receipt_keys = {"container_id", "member_path", "exact_timestamp_stem", "bytes", "sha256", "crc32"}
        require(isinstance(receipt, dict) and set(receipt) == receipt_keys, "ASSET_BINDING_INVALID", "asset binding key set drift", role=role)
        require(isinstance(receipt["container_id"], str) and bool(receipt["container_id"].strip()), "ASSET_BINDING_INVALID", "container identity is required", role=role)
        member_path = receipt["member_path"]
        require(isinstance(member_path, str) and bool(member_path) and "\\" not in member_path and not member_path.startswith("/"), "ASSET_BINDING_INVALID", "member path must be normalized relative POSIX syntax", role=role)
        require(all(part not in ("", ".", "..") for part in member_path.split("/")), "ASSET_BINDING_INVALID", "member path must be traversal-free and normalized", role=role)
        expected_stem = None if role == "trajectory" else frame_timestamp_token
        require(receipt["exact_timestamp_stem"] == expected_stem, "ASSET_EXACT_STEM_MISMATCH", "asset exact timestamp stem drift", role=role)
        if expected_stem is not None:
            member_name = receipt["member_path"].rsplit("/", 1)[-1]
            member_stem = member_name.rsplit(".", 1)[0]
            require(member_stem == expected_stem, "ASSET_MEMBER_STEM_MISMATCH", "asset member basename is not the exact frame timestamp stem", role=role)
        require(isinstance(receipt["bytes"], int) and not isinstance(receipt["bytes"], bool) and receipt["bytes"] > 0, "ASSET_BINDING_INVALID", "asset bytes must be positive", role=role)
        require(isinstance(receipt["sha256"], str) and bool(_SHA256.fullmatch(receipt["sha256"])), "ASSET_BINDING_INVALID", "asset sha256 is malformed", role=role)
        require(isinstance(receipt["crc32"], str) and bool(re.fullmatch(r"[0-9A-Fa-f]{8}", receipt["crc32"])), "ASSET_BINDING_INVALID", "asset CRC32 is malformed", role=role)
    require(len({receipt["member_path"] for receipt in value.values()}) == len(keys), "ASSET_MEMBER_DUPLICATE", "asset member paths must be unique")
    return value


def _validate_decoded_payload_bindings(
    value: Any,
    assets: dict[str, Any],
    *,
    lowres_intrinsics: dict[str, Any] | None = None,
    trajectory_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == set(DECODED_PAYLOAD_KINDS), "DECODED_PAYLOAD_BINDING_KEY_SET", "decoded payload bindings must cover every frozen asset role")
    for role, binding in value.items():
        keys = {"asset_role", "member_sha256", "member_crc32", "decoded_kind", "decoded_content_sha256"}
        require(isinstance(binding, dict) and set(binding) == keys, "DECODED_PAYLOAD_BINDING_INVALID", "decoded payload binding key set drift", role=role)
        require(binding["asset_role"] == role and binding["member_sha256"] == assets[role]["sha256"] and binding["member_crc32"] == assets[role]["crc32"], "DECODED_PAYLOAD_ASSET_MISMATCH", "decoded payload receipt is not bound to the exact member SHA/CRC", role=role)
        require(binding["decoded_kind"] == DECODED_PAYLOAD_KINDS[role], "DECODED_PAYLOAD_KIND_MISMATCH", "decoded payload kind drift", role=role)
        require(isinstance(binding["decoded_content_sha256"], str) and bool(_SHA256.fullmatch(binding["decoded_content_sha256"])), "DECODED_PAYLOAD_BINDING_INVALID", "decoded content hash is malformed", role=role)
    if lowres_intrinsics is not None:
        require(value["intrinsics"]["decoded_content_sha256"] == canonical_sha256(lowres_intrinsics), "DECODED_PAYLOAD_CONTENT_MISMATCH", "decoded intrinsics hash does not match the supplied canonical object", role="intrinsics")
    if trajectory_rows is not None:
        require(value["trajectory"]["decoded_content_sha256"] == canonical_sha256(trajectory_rows), "DECODED_PAYLOAD_CONTENT_MISMATCH", "decoded trajectory hash does not match the supplied canonical rows", role="trajectory")
    return value


def _validate_bound_decoded_payload(receipt: dict[str, Any], role: str, decoded_value: Any) -> None:
    require(role in DECODED_PAYLOAD_KINDS, "DECODED_PAYLOAD_ROLE_INVALID", "unknown decoded payload role")
    binding = receipt["decoded_payload_bindings"][role]
    require(binding["decoded_content_sha256"] == canonical_sha256(decoded_value), "DECODED_PAYLOAD_CONTENT_MISMATCH", "decoded payload does not match the source receipt", role=role)


def build_source_frame_receipt(
    *,
    source_role: str,
    visit_id: str,
    video_id: str,
    frame_timestamp_token: str,
    lowres_intrinsics: dict[str, Any],
    trajectory_rows: Sequence[dict[str, Any]],
    asset_bindings: dict[str, Any],
    decoded_payload_bindings: dict[str, Any],
) -> dict[str, Any]:
    """Build one immutable source-specific base receipt without source I/O."""

    reject_truth_shortcuts({"lowres_intrinsics": lowres_intrinsics, "trajectory_rows": trajectory_rows, "asset_bindings": asset_bindings, "decoded_payload_bindings": decoded_payload_bindings})
    require(isinstance(visit_id, str) and visit_id.isdigit(), "VISIT_ID_INVALID", "visit_id must be a decimal identity token")
    require(isinstance(video_id, str) and video_id.isdigit(), "VIDEO_ID_INVALID", "video_id must be a decimal identity token")
    _validate_roster_identity(source_role, visit_id, video_id)
    assets = _validate_asset_bindings(asset_bindings, frame_timestamp_token)
    decoded = _validate_decoded_payload_bindings(decoded_payload_bindings, assets, lowres_intrinsics=lowres_intrinsics, trajectory_rows=trajectory_rows)
    transform, pose = interpolate_camera_to_world_exact(trajectory_rows, frame_timestamp_token)
    intrinsics = scale_lowres_intrinsics(lowres_intrinsics)
    gravity = _normalize_vector(transform[2, :3], "GRAVITY_INVALID")
    frame_id = f"{video_id}:{frame_timestamp_token}"
    receipt = {
        "schema": BASE_RECEIPT_SCHEMA,
        "source_id": SOURCE_ID,
        "source_role": source_role,
        "frozen_roster_sha256": frozen_roster_sha256(),
        "site_id": visit_id,
        "parent_id": visit_id,
        "session_id": video_id,
        "capture_id": video_id,
        "physical_frame_id": frame_id,
        "sensor_timestamp": {
            "decimal_token": frame_timestamp_token,
            "nanoseconds": pose["frame_timestamp_ns"],
        },
        "pose_bracket": {
            "left_decimal_token": pose["left_timestamp_token"],
            "left_timestamp_ns": pose["left_timestamp_ns"],
            "right_decimal_token": pose["right_timestamp_token"],
            "right_timestamp_ns": pose["right_timestamp_ns"],
            "fraction": pose["fraction"],
            "bracketing_gap_ns": pose["bracketing_gap_ns"],
        },
        "max_source_timestamp_rule": "MAX_FRAME_AND_RIGHT_POSE_BRACKET",
        "max_source_timestamp_ns": pose["max_source_timestamp_ns"],
        "lowres_intrinsics_source": copy.deepcopy(lowres_intrinsics),
        "intrinsics_highres": intrinsics,
        "camera_to_world_4x4": transform.tolist(),
        "gravity_up_camera_xyz": gravity.tolist(),
        "camera_body_mount": "ARKITSCENES_IPAD_HANDHELD_SOURCE_CHARACTERIZATION_ONLY",
        "metric_reference": "REGISTERED_FARO_SOURCE_TRUTH_APPLEDEPTH_UNCERTAINTY_EVIDENCE_ONLY",
        "p0_frame_receipt_projection": "NOT_AVAILABLE_IN_O0R_SOURCE_CHARACTERIZATION",
        "p0_projection_reason": "SOURCE_NATIVE_POSE_IMU_COVARIANCE_SPARSE_TRACKS_AND_CALIBRATED_CAMERA_BODY_TRANSFORM_ABSENT",
        "asset_bindings": copy.deepcopy(assets),
        "decoded_payload_bindings": copy.deepcopy(decoded),
    }
    return _seal(receipt)


def _validate_base_receipt(value: Any) -> dict[str, Any]:
    receipt = _validate_seal(value, "BASE_RECEIPT_HASH_MISMATCH")
    keys = {
        "schema",
        "source_id",
        "source_role",
        "frozen_roster_sha256",
        "site_id",
        "parent_id",
        "session_id",
        "capture_id",
        "physical_frame_id",
        "sensor_timestamp",
        "pose_bracket",
        "max_source_timestamp_rule",
        "max_source_timestamp_ns",
        "lowres_intrinsics_source",
        "intrinsics_highres",
        "camera_to_world_4x4",
        "gravity_up_camera_xyz",
        "camera_body_mount",
        "metric_reference",
        "p0_frame_receipt_projection",
        "p0_projection_reason",
        "asset_bindings",
        "decoded_payload_bindings",
        "content_sha256",
    }
    require(set(receipt) == keys and receipt["schema"] == BASE_RECEIPT_SCHEMA and receipt["source_id"] == SOURCE_ID, "BASE_RECEIPT_KEY_SET", "base receipt key/schema drift")
    require(receipt["frozen_roster_sha256"] == frozen_roster_sha256(), "SOURCE_ROSTER_DIGEST_MISMATCH", "frozen source roster digest drift")
    _validate_roster_identity(receipt["source_role"], receipt["site_id"], receipt["session_id"])
    require(receipt["site_id"] == receipt["parent_id"] and receipt["session_id"] == receipt["capture_id"], "BASE_RECEIPT_IDENTITY", "source identity aliases drift")
    sensor = receipt["sensor_timestamp"]
    pose = receipt["pose_bracket"]
    require(isinstance(sensor, dict) and set(sensor) == {"decimal_token", "nanoseconds"}, "BASE_RECEIPT_TIMESTAMP", "sensor timestamp key set drift")
    require(decimal_timestamp_ns(sensor["decimal_token"]) == sensor["nanoseconds"], "BASE_RECEIPT_TIMESTAMP", "sensor timestamp token/ns mismatch")
    require(receipt["physical_frame_id"] == f"{receipt['session_id']}:{sensor['decimal_token']}", "BASE_RECEIPT_IDENTITY", "physical frame identity is not bound to session and exact timestamp")
    pose_keys = {"left_decimal_token", "left_timestamp_ns", "right_decimal_token", "right_timestamp_ns", "fraction", "bracketing_gap_ns"}
    require(isinstance(pose, dict) and set(pose) == pose_keys, "BASE_RECEIPT_POSE", "pose bracket key set drift")
    require(decimal_timestamp_ns(pose["left_decimal_token"]) == pose["left_timestamp_ns"] and decimal_timestamp_ns(pose["right_decimal_token"]) == pose["right_timestamp_ns"], "BASE_RECEIPT_POSE", "pose bracket token/ns mismatch")
    require(isinstance(pose["bracketing_gap_ns"], int) and 0 <= pose["bracketing_gap_ns"] <= MAXIMUM_POSE_GAP_NS, "BASE_RECEIPT_POSE", "pose gap exceeds frozen bound")
    require(_finite_number(pose["fraction"]) and 0.0 <= float(pose["fraction"]) <= 1.0, "BASE_RECEIPT_POSE", "pose interpolation fraction is invalid")
    require(pose["left_timestamp_ns"] <= sensor["nanoseconds"] <= pose["right_timestamp_ns"], "BASE_RECEIPT_POSE", "sensor timestamp is outside its pose bracket")
    expected_gap = pose["right_timestamp_ns"] - pose["left_timestamp_ns"]
    require(pose["bracketing_gap_ns"] == expected_gap, "BASE_RECEIPT_POSE", "pose bracket gap is not recomputable from bound timestamps")
    expected_fraction = 0.0 if expected_gap == 0 else (sensor["nanoseconds"] - pose["left_timestamp_ns"]) / expected_gap
    require(abs(float(pose["fraction"]) - expected_fraction) <= 1e-12, "BASE_RECEIPT_POSE", "pose interpolation fraction is not bound to exact timestamps")
    require(receipt["max_source_timestamp_rule"] == "MAX_FRAME_AND_RIGHT_POSE_BRACKET", "BASE_RECEIPT_WATERMARK", "watermark rule drift")
    require(receipt["max_source_timestamp_ns"] == max(sensor["nanoseconds"], pose["right_timestamp_ns"]), "BASE_RECEIPT_WATERMARK", "right pose bracket is absent from watermark")
    lowres = receipt["lowres_intrinsics_source"]
    require(isinstance(lowres, dict), "BASE_RECEIPT_INTRINSICS", "bound lowres intrinsics source is missing")
    recomputed_highres = scale_lowres_intrinsics(lowres)
    intrinsics = receipt["intrinsics_highres"]
    require(isinstance(intrinsics, dict) and set(intrinsics) == {"width", "height", "fx", "fy", "cx", "cy", "pixel_center_convention", "matrix_3x3"}, "BASE_RECEIPT_INTRINSICS", "bound intrinsics key set drift")
    require(intrinsics["width"] == HIGHRES_SHAPE_HW[1] and intrinsics["height"] == HIGHRES_SHAPE_HW[0] and intrinsics["pixel_center_convention"] == "INTEGER_INDEX_IS_PIXEL_CENTER", "BASE_RECEIPT_INTRINSICS", "bound intrinsics geometry drift")
    matrix = _intrinsics_matrix(intrinsics["matrix_3x3"])
    require(all(_finite_number(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")), "BASE_RECEIPT_INTRINSICS", "bound intrinsics scalars must be finite")
    expected_matrix = np.asarray([[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    require(bool(np.allclose(matrix, expected_matrix, rtol=0.0, atol=1e-12)), "BASE_RECEIPT_INTRINSICS", "bound intrinsics matrix/scalars disagree")
    require(canonical_sha256(intrinsics) == canonical_sha256(recomputed_highres), "BASE_RECEIPT_INTRINSICS", "highres intrinsics are not recomputable from the decoded lowres source")
    transform = np.asarray(receipt["camera_to_world_4x4"], dtype=np.float64)
    require(transform.shape == (4, 4) and bool(np.all(np.isfinite(transform))) and bool(np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)), "BASE_RECEIPT_POSE", "camera-to-world transform is invalid")
    rotation = transform[:3, :3]
    require(bool(np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-10)) and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-10, "BASE_RECEIPT_POSE", "camera-to-world rotation is not a proper orthonormal rotation")
    gravity = _normalize_vector(receipt["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    require(bool(np.allclose(gravity, transform[2, :3] / np.linalg.norm(transform[2, :3]), rtol=0.0, atol=1e-12)), "BASE_RECEIPT_GRAVITY", "gravity is not derived from bound pose")
    require(receipt["camera_body_mount"] == "ARKITSCENES_IPAD_HANDHELD_SOURCE_CHARACTERIZATION_ONLY", "BASE_RECEIPT_MOUNT", "source-characterization mount claim drift")
    require(receipt["metric_reference"] == "REGISTERED_FARO_SOURCE_TRUTH_APPLEDEPTH_UNCERTAINTY_EVIDENCE_ONLY", "BASE_RECEIPT_METRIC_REFERENCE", "metric reference claim drift")
    require(receipt["p0_frame_receipt_projection"] == "NOT_AVAILABLE_IN_O0R_SOURCE_CHARACTERIZATION", "P0_RECEIPT_OVERCLAIM", "source receipt may not claim complete P0 projection")
    require(receipt["p0_projection_reason"] == "SOURCE_NATIVE_POSE_IMU_COVARIANCE_SPARSE_TRACKS_AND_CALIBRATED_CAMERA_BODY_TRANSFORM_ABSENT", "P0_RECEIPT_OVERCLAIM", "P0 projection limitation reason drift")
    assets = _validate_asset_bindings(receipt["asset_bindings"], sensor["decimal_token"])
    _validate_decoded_payload_bindings(receipt["decoded_payload_bindings"], assets, lowres_intrinsics=lowres)
    return receipt


def _signed_token(value: float, places: int) -> str:
    rounded = round(float(value), places)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:+.{places}f}"


def _build_geometry_query_receipts(
    *,
    physical_frame_id: str,
    source_frame_receipt_sha256: str,
    max_source_timestamp_ns: int,
    support_normal_camera_xyz: Any,
    camera_height_m: float,
) -> list[dict[str, Any]]:
    normal = _normalize_vector(support_normal_camera_xyz, "SUPPORT_NORMAL_INVALID")
    require(_finite_number(camera_height_m) and float(camera_height_m) > 0.0, "CAMERA_HEIGHT_INVALID", "camera height must be finite and positive")
    height = float(camera_height_m)
    optical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward = optical - float(np.dot(optical, normal)) * normal
    forward = _normalize_vector(forward, "VIRTUAL_FORWARD_INVALID")
    if float(np.dot(forward, optical)) < 0.0:
        forward = -forward
    lateral = _normalize_vector(np.cross(forward, normal), "VIRTUAL_LATERAL_INVALID")
    origin = -height * normal
    frame_id = physical_frame_id
    output: list[dict[str, Any]] = []
    grid_index = 0
    for lateral_offset in PATH_LATERAL_OFFSETS_M:
        for yaw_degrees in PATH_YAW_DEGREES:
            yaw_radians = math.radians(yaw_degrees)
            heading = _normalize_vector(
                math.cos(yaw_radians) * forward + math.sin(yaw_radians) * lateral,
                "QUERY_HEADING_INVALID",
            )
            path_id = f"lat_{_signed_token(lateral_offset, 2)}_yaw_{_signed_token(yaw_degrees, 1)}"
            query_id = f"{frame_id}:{path_id}"
            receipt = {
                "schema": QUERY_RECEIPT_SCHEMA,
                "source_frame_receipt_sha256": source_frame_receipt_sha256,
                "physical_frame_id": frame_id,
                "query_id": query_id,
                "path_id": path_id,
                "grid_index": grid_index,
                "grid_order": "LATERAL_MAJOR_THEN_YAW_ASCENDING",
                "path_lateral_offset_m": lateral_offset,
                "path_yaw_degrees": yaw_degrees,
                "minimum_forward_m": MINIMUM_FORWARD_M,
                "horizon_m": HORIZON_M,
                "capsule_radius_m": CAPSULE_RADIUS_M,
                "virtual_query_frame": {
                    "kind": "FARO_SUPPORT_PLANE_SOURCE_CHARACTERIZATION_ONLY",
                    "origin_camera_xyz": origin.tolist(),
                    "forward_camera_xyz": forward.tolist(),
                    "lateral_camera_xyz": lateral.tolist(),
                    "gravity_up_camera_xyz": normal.tolist(),
                    "path_heading_camera_xyz": heading.tolist(),
                },
                "max_source_timestamp_ns": max_source_timestamp_ns,
            }
            output.append(_seal(receipt))
            grid_index += 1
    require(len(output) == 9 and [item["grid_index"] for item in output] == list(range(9)), "QUERY_RECEIPT_CARDINALITY", "exactly nine ordered query receipts are required")
    return output


def build_query_receipts(base_receipt: dict[str, Any], geometry: FaroGeometry) -> list[dict[str, Any]]:
    """Generate the frozen 3x3 query grid only from bound FARO geometry."""

    base = _validate_base_receipt(base_receipt)
    faro = _validate_faro_geometry(geometry)
    require(base["source_role"] == "O0R_EVAL_CANDIDATE", "QUERY_SOURCE_ROLE_INVALID", "query receipts require a frozen O0R_EVAL_CANDIDATE source")
    require(faro.physical_frame_id == base["physical_frame_id"] and faro.source_frame_receipt_sha256 == base["content_sha256"] and faro.max_source_timestamp_ns == base["max_source_timestamp_ns"], "QUERY_GEOMETRY_BINDING_INVALID", "FARO geometry is not bound to the supplied source receipt")
    return _build_geometry_query_receipts(
        physical_frame_id=faro.physical_frame_id,
        source_frame_receipt_sha256=faro.source_frame_receipt_sha256,
        max_source_timestamp_ns=faro.max_source_timestamp_ns,
        support_normal_camera_xyz=faro.support_normal_camera_xyz,
        camera_height_m=faro.camera_height_m,
    )


def _validate_query_receipt(value: Any) -> dict[str, Any]:
    receipt = _validate_seal(value, "QUERY_RECEIPT_HASH_MISMATCH")
    keys = {
        "schema",
        "source_frame_receipt_sha256",
        "physical_frame_id",
        "query_id",
        "path_id",
        "grid_index",
        "grid_order",
        "path_lateral_offset_m",
        "path_yaw_degrees",
        "minimum_forward_m",
        "horizon_m",
        "capsule_radius_m",
        "virtual_query_frame",
        "max_source_timestamp_ns",
        "content_sha256",
    }
    require(set(receipt) == keys and receipt["schema"] == QUERY_RECEIPT_SCHEMA, "QUERY_RECEIPT_KEY_SET", "query receipt key/schema drift")
    index = receipt["grid_index"]
    require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 9, "QUERY_RECEIPT_GRID", "query grid index is invalid")
    expected_lateral = PATH_LATERAL_OFFSETS_M[index // 3]
    expected_yaw = PATH_YAW_DEGREES[index % 3]
    require(receipt["grid_order"] == "LATERAL_MAJOR_THEN_YAW_ASCENDING" and receipt["path_lateral_offset_m"] == expected_lateral and receipt["path_yaw_degrees"] == expected_yaw, "QUERY_RECEIPT_GRID", "query grid order/value drift")
    expected_path = f"lat_{_signed_token(expected_lateral, 2)}_yaw_{_signed_token(expected_yaw, 1)}"
    require(receipt["path_id"] == expected_path and receipt["query_id"] == f"{receipt['physical_frame_id']}:{expected_path}", "QUERY_RECEIPT_IDENTITY", "query/path identity drift")
    require(receipt["minimum_forward_m"] == MINIMUM_FORWARD_M and receipt["horizon_m"] == HORIZON_M and receipt["capsule_radius_m"] == CAPSULE_RADIUS_M, "QUERY_RECEIPT_GEOMETRY", "query geometry constant drift")
    require(isinstance(receipt["source_frame_receipt_sha256"], str) and bool(_SHA256.fullmatch(receipt["source_frame_receipt_sha256"])), "QUERY_RECEIPT_BASE_HASH", "base receipt hash is malformed")
    require(isinstance(receipt["max_source_timestamp_ns"], int) and not isinstance(receipt["max_source_timestamp_ns"], bool) and receipt["max_source_timestamp_ns"] >= 0, "QUERY_RECEIPT_WATERMARK", "query watermark is invalid")
    frame = receipt["virtual_query_frame"]
    frame_keys = {"kind", "origin_camera_xyz", "forward_camera_xyz", "lateral_camera_xyz", "gravity_up_camera_xyz", "path_heading_camera_xyz"}
    require(isinstance(frame, dict) and set(frame) == frame_keys and frame["kind"] == "FARO_SUPPORT_PLANE_SOURCE_CHARACTERIZATION_ONLY", "QUERY_FRAME_INVALID", "virtual query frame key/kind drift")
    _query_receipt_vectors(receipt)
    return receipt


def _intrinsics_matrix(value: Any, expected_shape_hw: tuple[int, int] = HIGHRES_SHAPE_HW) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (3, 3) and bool(np.all(np.isfinite(matrix))), "INTRINSICS_MATRIX_INVALID", "intrinsics must be finite 3x3")
    require(abs(float(matrix[2, 0])) <= 1e-12 and abs(float(matrix[2, 1])) <= 1e-12 and abs(float(matrix[2, 2]) - 1.0) <= 1e-12, "INTRINSICS_MATRIX_INVALID", "intrinsics bottom row drift")
    height, width = expected_shape_hw
    fx, fy, cx, cy = float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
    require(fx > 0.0 and fy > 0.0 and 0.0 <= cx < width and 0.0 <= cy < height, "INTRINSICS_MATRIX_RANGE", "intrinsics are out of raster range")
    return matrix


def _unproject(depth_m: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    sampled_depth = depth_m[::stride, ::stride]
    sampled_valid = valid[::stride, ::stride]
    z = sampled_depth[sampled_valid]
    u = columns[sampled_valid]
    v = rows[sampled_valid]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    pixels = np.stack((u, v), axis=1).astype(np.int32)
    return points, pixels


def _fit_support_plane(points: np.ndarray, gravity_up: np.ndarray) -> dict[str, Any]:
    require(points.ndim == 2 and points.shape[1] == 3 and bool(np.all(np.isfinite(points))), "SUPPORT_POINTS_INVALID", "support input must be finite Nx3")
    require(len(points) >= MINIMUM_SUPPORT_POINTS, "SUPPORT_POINTS_INSUFFICIENT", "too few sampled FARO points for support")
    up = _normalize_vector(gravity_up, "GRAVITY_INVALID")
    offsets = -(points @ up)
    plausible = np.isfinite(offsets) & (offsets >= 0.45) & (offsets <= 2.20)
    require(int(np.sum(plausible)) >= MINIMUM_SUPPORT_POINTS, "SUPPORT_PLAUSIBLE_INSUFFICIENT", "too few plausible ground-height points")
    edges = np.arange(0.45, 2.20 + 0.04 * 1.001, 0.04)
    counts, edges = np.histogram(offsets[plausible], bins=edges)
    require(len(counts) > 0 and int(np.max(counts)) > 0, "SUPPORT_HISTOGRAM_EMPTY", "support histogram is empty")
    candidates = np.flatnonzero(counts == np.max(counts))
    mode_index = int(candidates[-1])
    mode_center = float((edges[mode_index] + edges[mode_index + 1]) / 2.0)
    initial = plausible & (np.abs(offsets - mode_center) <= SUPPORT_RESIDUAL_TOLERANCE_M)
    minimum_count = max(MINIMUM_SUPPORT_POINTS, int(math.ceil(MINIMUM_SUPPORT_FRACTION * len(points))))
    require(int(np.sum(initial)) >= minimum_count, "SUPPORT_GATE_FAILED", "support count/fraction gate failed")
    selected = points[initial]
    centered = selected - np.mean(selected, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize_vector(right[-1], "SUPPORT_NORMAL_INVALID")
    if float(np.dot(normal, up)) < 0.0:
        normal = -normal
    slope_degrees = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(normal, up))))))
    require(slope_degrees <= MAXIMUM_SUPPORT_SLOPE_DEGREES, "SUPPORT_SLOPE_EXCEEDED", "support plane exceeds frozen slope")
    camera_height = -float(np.median(selected @ normal))
    residuals = np.abs(points @ normal + camera_height)
    refined = residuals <= SUPPORT_RESIDUAL_TOLERANCE_M
    support_count = int(np.sum(refined))
    require(support_count >= minimum_count, "SUPPORT_REFINED_GATE_FAILED", "refined support count/fraction gate failed")
    return {
        "normal_camera_xyz": normal,
        "camera_height_m": camera_height,
        "median_residual_m": float(np.median(residuals[refined])),
        "support_mask": refined,
        "support_points": points[refined],
        "support_count": support_count,
        "sampled_valid_points": int(len(points)),
        "support_fraction": support_count / float(len(points)),
        "slope_degrees": slope_degrees,
    }


@dataclass(frozen=True)
class FaroGeometry:
    """In-memory FARO factor geometry; never an artifact or scientific result."""

    physical_frame_id: str
    source_role: str
    source_frame_receipt_sha256: str
    highres_depth_array_sha256: str
    max_source_timestamp_ns: int
    intrinsics: np.ndarray
    depth_m: np.ndarray
    valid_depth: np.ndarray
    points_camera_xyz: np.ndarray
    pixels_uv: np.ndarray
    support_normal_camera_xyz: np.ndarray
    camera_height_m: float
    support_points_camera_xyz: np.ndarray
    support_count: int
    support_fraction: float
    support_slope_degrees: float
    support_median_residual_m: float
    obstacle_mask_hw: np.ndarray
    boundary_signed_distance_m_hw: np.ndarray
    content_sha256: str


def _immutable_array(value: Any, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)
    require(not immutable.flags.writeable, "IMMUTABLE_ARRAY_FAILURE", "geometry array backing must be immutable")
    return immutable


def _faro_geometry_payload(geometry: FaroGeometry) -> dict[str, Any]:
    return {
        "physical_frame_id": geometry.physical_frame_id,
        "source_role": geometry.source_role,
        "source_frame_receipt_sha256": geometry.source_frame_receipt_sha256,
        "highres_depth_array_sha256": geometry.highres_depth_array_sha256,
        "max_source_timestamp_ns": geometry.max_source_timestamp_ns,
        "intrinsics": geometry.intrinsics,
        "depth_m": geometry.depth_m,
        "valid_depth": geometry.valid_depth,
        "points_camera_xyz": geometry.points_camera_xyz,
        "pixels_uv": geometry.pixels_uv,
        "support_normal_camera_xyz": geometry.support_normal_camera_xyz,
        "camera_height_m": geometry.camera_height_m,
        "support_points_camera_xyz": geometry.support_points_camera_xyz,
        "support_count": geometry.support_count,
        "support_fraction": geometry.support_fraction,
        "support_slope_degrees": geometry.support_slope_degrees,
        "support_median_residual_m": geometry.support_median_residual_m,
        "obstacle_mask_hw": geometry.obstacle_mask_hw,
        "boundary_signed_distance_m_hw": geometry.boundary_signed_distance_m_hw,
    }


def _validate_faro_geometry(geometry: Any) -> FaroGeometry:
    require(isinstance(geometry, FaroGeometry), "FARO_GEOMETRY_TYPE_INVALID", "FARO geometry must come from the controlled extractor")
    require(geometry.source_role == "O0R_EVAL_CANDIDATE", "FARO_TRUTH_SOURCE_ROLE_INVALID", "FARO geometry role drift")
    for name in (
        "intrinsics",
        "depth_m",
        "valid_depth",
        "points_camera_xyz",
        "pixels_uv",
        "support_normal_camera_xyz",
        "support_points_camera_xyz",
        "obstacle_mask_hw",
        "boundary_signed_distance_m_hw",
    ):
        require(not np.asarray(getattr(geometry, name)).flags.writeable, "FARO_GEOMETRY_MUTABLE", "FARO geometry arrays must be deeply read-only", field=name)
    require(isinstance(geometry.content_sha256, str) and bool(_SHA256.fullmatch(geometry.content_sha256)), "FARO_GEOMETRY_HASH_MISMATCH", "FARO geometry hash is malformed")
    require(canonical_sha256(_faro_geometry_payload(geometry)) == geometry.content_sha256, "FARO_GEOMETRY_HASH_MISMATCH", "FARO geometry content drift")
    return geometry


def derive_faro_geometry(
    highres_depth_mm: np.ndarray,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    source_frame_receipt: dict[str, Any],
) -> FaroGeometry:
    """Derive strict 1920x1440 FARO support and continuous boundary geometry."""

    reject_truth_shortcuts({"intrinsics": intrinsics_highres_3x3, "gravity": gravity_up_camera_xyz})
    receipt = _validate_base_receipt(source_frame_receipt)
    raw = np.asarray(highres_depth_mm)
    require(raw.shape == HIGHRES_SHAPE_HW, "FARO_SHAPE_INVALID", "FARO depth must be 1440x1920")
    require(np.issubdtype(raw.dtype, np.integer), "FARO_DTYPE_INVALID", "FARO depth must be an integer millimetre raster")
    require(receipt["source_role"] == "O0R_EVAL_CANDIDATE", "FARO_TRUTH_SOURCE_ROLE_INVALID", "FARO truth geometry requires a frozen O0R_EVAL_CANDIDATE receipt")
    _validate_bound_decoded_payload(receipt, "highres_depth", raw)
    matrix = _intrinsics_matrix(intrinsics_highres_3x3)
    bound_matrix = np.asarray(receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    require(bool(np.allclose(matrix, bound_matrix, rtol=0.0, atol=1e-12)), "RECEIPT_INTRINSICS_MISMATCH", "FARO intrinsics do not match base receipt")
    gravity = _normalize_vector(gravity_up_camera_xyz, "GRAVITY_INVALID")
    bound_gravity = _normalize_vector(receipt["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    require(bool(np.allclose(gravity, bound_gravity, rtol=0.0, atol=1e-12)), "RECEIPT_GRAVITY_MISMATCH", "gravity does not match base receipt")
    depth_m = raw.astype(np.float64) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m >= DEPTH_RANGE_M[0]) & (depth_m <= DEPTH_RANGE_M[1])
    sampled_points, _ = _unproject(depth_m, valid, matrix, SUPPORT_POINT_STRIDE)
    plane = _fit_support_plane(sampled_points, gravity)
    points, pixels = _unproject(depth_m, valid, matrix, 1)
    heights = points @ plane["normal_camera_xyz"] + float(plane["camera_height_m"])
    obstacles = (heights >= OBSTACLE_HEIGHT_RANGE_M[0]) & (heights <= OBSTACLE_HEIGHT_RANGE_M[1])
    obstacle_mask = np.zeros(HIGHRES_SHAPE_HW, dtype=bool)
    obstacle_pixels = pixels[obstacles]
    obstacle_mask[obstacle_pixels[:, 1], obstacle_pixels[:, 0]] = True
    if bool(np.any(obstacle_mask)):
        outside_px = ndimage.distance_transform_edt(~obstacle_mask)
        inside_px = ndimage.distance_transform_edt(obstacle_mask)
        signed_px = np.where(obstacle_mask, -(inside_px - 0.5), outside_px - 0.5)
        metric_per_pixel = np.zeros_like(depth_m, dtype=np.float64)
        metric_per_pixel[valid] = depth_m[valid] / math.sqrt(float(matrix[0, 0] * matrix[1, 1]))
        boundary = signed_px * metric_per_pixel
    else:
        boundary = np.full(HIGHRES_SHAPE_HW, HORIZON_M, dtype=np.float64)
    boundary[~valid] = 0.0
    geometry_values = {
        "physical_frame_id": receipt["physical_frame_id"],
        "source_role": receipt["source_role"],
        "source_frame_receipt_sha256": receipt["content_sha256"],
        "highres_depth_array_sha256": canonical_sha256(raw),
        "max_source_timestamp_ns": int(receipt["max_source_timestamp_ns"]),
        "intrinsics": _immutable_array(matrix, np.float64),
        "depth_m": _immutable_array(depth_m, np.float64),
        "valid_depth": _immutable_array(valid, np.bool_),
        "points_camera_xyz": _immutable_array(points, np.float64),
        "pixels_uv": _immutable_array(pixels, np.int32),
        "support_normal_camera_xyz": _immutable_array(plane["normal_camera_xyz"], np.float64),
        "camera_height_m": float(plane["camera_height_m"]),
        "support_points_camera_xyz": _immutable_array(plane["support_points"], np.float64),
        "support_count": int(plane["support_count"]),
        "support_fraction": float(plane["support_fraction"]),
        "support_slope_degrees": float(plane["slope_degrees"]),
        "support_median_residual_m": float(plane["median_residual_m"]),
        "obstacle_mask_hw": _immutable_array(obstacle_mask, np.bool_),
        "boundary_signed_distance_m_hw": _immutable_array(boundary, np.float64),
    }
    provisional = FaroGeometry(**geometry_values, content_sha256="0" * 64)
    geometry = FaroGeometry(**geometry_values, content_sha256=canonical_sha256(_faro_geometry_payload(provisional)))
    return _validate_faro_geometry(geometry)


def sample_faro_at_apple_centers(highres_faro_depth_mm: np.ndarray) -> np.ndarray:
    """Nearest sample FARO at frozen AppleDepth pixel-center locations."""

    faro = np.asarray(highres_faro_depth_mm)
    require(faro.shape == HIGHRES_SHAPE_HW, "FARO_SHAPE_INVALID", "FARO depth must be 1440x1920")
    rows, columns = np.mgrid[0 : APPLE_SHAPE_HW[0], 0 : APPLE_SHAPE_HW[1]]
    x = np.rint((columns + 0.5) * LOWRES_TO_HIGHRES_SCALE_XY[0] - 0.5).astype(np.int64)
    y = np.rint((rows + 0.5) * LOWRES_TO_HIGHRES_SCALE_XY[1] - 0.5).astype(np.int64)
    require(int(np.min(x)) >= 0 and int(np.max(x)) < HIGHRES_SHAPE_HW[1] and int(np.min(y)) >= 0 and int(np.max(y)) < HIGHRES_SHAPE_HW[0], "PIXEL_CENTER_MAPPING_INVALID", "mapped AppleDepth centers leave FARO raster")
    return np.asarray(faro[y, x])


def _range_bin(value: float) -> int | None:
    if not math.isfinite(value) or value < RANGE_EDGES_M[0] or value > RANGE_EDGES_M[-1]:
        return None
    for index in range(len(RANGE_EDGES_M) - 1):
        lower, upper = RANGE_EDGES_M[index], RANGE_EDGES_M[index + 1]
        if lower <= value < upper or (index == len(RANGE_EDGES_M) - 2 and lower <= value <= upper):
            return index
    return None


def _lowres_intrinsics_from_base_receipt(receipt: dict[str, Any]) -> np.ndarray:
    high = receipt["intrinsics_highres"]
    sx, sy = LOWRES_TO_HIGHRES_SCALE_XY
    fx = float(high["fx"]) / sx
    fy = float(high["fy"]) / sy
    cx = (float(high["cx"]) + 0.5) / sx - 0.5
    cy = (float(high["cy"]) + 0.5) / sy - 0.5
    return _intrinsics_matrix(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        APPLE_SHAPE_HW,
    )


def _signed_distance_pixels(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    require(binary.shape == APPLE_SHAPE_HW, "BOUNDARY_MASK_SHAPE", "boundary mask must be 192x256")
    if not bool(np.any(binary)):
        return np.full(APPLE_SHAPE_HW, float(max(APPLE_SHAPE_HW)), dtype=np.float64)
    if bool(np.all(binary)):
        return np.full(APPLE_SHAPE_HW, -float(max(APPLE_SHAPE_HW)), dtype=np.float64)
    outside = ndimage.distance_transform_edt(~binary)
    inside = ndimage.distance_transform_edt(binary)
    return np.where(binary, -(inside - 0.5), outside - 0.5)


def _derive_adapter_fit_residual_batch(source_frame: dict[str, Any]) -> dict[str, Any]:
    """Derive residual evidence internally from one roster-bound source frame."""

    keys = {"source_frame_receipt", "highres_faro_depth_mm", "apple_depth_mm", "confidence"}
    require(isinstance(source_frame, dict) and set(source_frame) == keys, "ADAPTER_FIT_SOURCE_KEY_SET", "adapter-fit source frame key set drift")
    receipt = _validate_base_receipt(source_frame["source_frame_receipt"])
    require(receipt["source_role"] == "ADAPTER_FIT", "UNCERTAINTY_ROLE_LEAKAGE", "uncertainty fit accepts only roster-bound ADAPTER_FIT receipts")
    faro_raw = np.asarray(source_frame["highres_faro_depth_mm"])
    apple_raw = np.asarray(source_frame["apple_depth_mm"])
    confidence = np.asarray(source_frame["confidence"])
    require(faro_raw.shape == HIGHRES_SHAPE_HW and np.issubdtype(faro_raw.dtype, np.integer), "FARO_SHAPE_INVALID", "FARO residual input must be integer 1440x1920")
    require(apple_raw.shape == APPLE_SHAPE_HW and np.issubdtype(apple_raw.dtype, np.integer), "APPLE_DEPTH_INVALID", "AppleDepth residual input must be integer 192x256")
    require(confidence.shape == APPLE_SHAPE_HW and np.issubdtype(confidence.dtype, np.integer), "CONFIDENCE_DTYPE", "confidence must be integer 192x256")
    require(bool(np.all((confidence >= 0) & (confidence <= 2))), "CONFIDENCE_RANGE", "confidence must be 0..2")
    _validate_bound_decoded_payload(receipt, "highres_depth", faro_raw)
    _validate_bound_decoded_payload(receipt, "lowres_depth", apple_raw)
    _validate_bound_decoded_payload(receipt, "confidence", confidence)
    faro_m = sample_faro_at_apple_centers(faro_raw).astype(np.float64) / 1000.0
    apple_m = apple_raw.astype(np.float64) / 1000.0
    valid = (
        np.isfinite(faro_m)
        & np.isfinite(apple_m)
        & (faro_m >= DEPTH_RANGE_M[0])
        & (faro_m <= DEPTH_RANGE_M[1])
        & (apple_m >= DEPTH_RANGE_M[0])
        & (apple_m <= DEPTH_RANGE_M[1])
    )
    require(int(np.sum(valid)) >= MINIMUM_SUPPORT_POINTS, "ADAPTER_FIT_COMMON_SUPPORT_INSUFFICIENT", "too few FARO/AppleDepth common-support pixels")
    low_k = _lowres_intrinsics_from_base_receipt(receipt)
    gravity = _normalize_vector(receipt["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    support_valid = valid & (confidence >= 1)
    scale_residual = np.zeros(APPLE_SHAPE_HW, dtype=np.float64)
    scale_residual[valid] = np.abs(np.log(faro_m[valid] / apple_m[valid]))
    confidence_counts = np.bincount(confidence[valid].astype(np.int64), minlength=3)
    representative_confidence = int(np.flatnonzero(confidence_counts == np.max(confidence_counts))[-1])
    representative_range = float(np.median(faro_m[valid]))
    batch = {
        "parent_id": receipt["parent_id"],
        "source_frame_receipt_sha256": receipt["content_sha256"],
        "physical_frame_id": receipt["physical_frame_id"],
        "pixel_confidence": np.ascontiguousarray(confidence[valid], dtype=np.int8),
        "pixel_range_m": np.ascontiguousarray(faro_m[valid], dtype=np.float64),
        "scale_log_abs_residual": np.ascontiguousarray(scale_residual[valid], dtype=np.float64),
        "boundary_available": False,
        "support_available": False,
        "support_failure_code": None,
        "support_failure_stage": None,
        "raw_input_receipts": {
            "faro_array_sha256": canonical_sha256(faro_raw),
            "apple_array_sha256": canonical_sha256(apple_raw),
            "confidence_array_sha256": canonical_sha256(confidence),
        },
    }
    faro_support_points, _ = _unproject(faro_m, support_valid, low_k, 2)
    try:
        faro_plane = _fit_support_plane(faro_support_points, gravity)
    except AdapterError as error:
        if error.code not in _SUPPORT_UNOBSERVABLE_CODES:
            raise
        # A physical frame can contain valid metric depth while showing too
        # little floor to identify support. Preserve its scale evidence and
        # record support/boundary as unavailable instead of fabricating a plane
        # or aborting the entire fit cohort.
        batch["support_failure_code"] = error.code
        batch["support_failure_stage"] = "FARO_SUPPORT_PLANE"
        return batch

    faro_points, faro_pixels = _unproject(faro_m, valid, low_k, 1)
    apple_points, apple_pixels = _unproject(apple_m, valid, low_k, 1)
    faro_heights = faro_points @ faro_plane["normal_camera_xyz"] + float(faro_plane["camera_height_m"])
    # Use the FARO support plane for both masks so boundary residual does not
    # re-count the separately calibrated support-plane discrepancy.
    apple_heights_on_faro = apple_points @ faro_plane["normal_camera_xyz"] + float(faro_plane["camera_height_m"])
    faro_obstacle = np.zeros(APPLE_SHAPE_HW, dtype=bool)
    apple_obstacle = np.zeros(APPLE_SHAPE_HW, dtype=bool)
    faro_members = (faro_heights >= OBSTACLE_HEIGHT_RANGE_M[0]) & (faro_heights <= OBSTACLE_HEIGHT_RANGE_M[1])
    apple_members = (apple_heights_on_faro >= OBSTACLE_HEIGHT_RANGE_M[0]) & (apple_heights_on_faro <= OBSTACLE_HEIGHT_RANGE_M[1])
    faro_obstacle[faro_pixels[faro_members, 1], faro_pixels[faro_members, 0]] = True
    apple_obstacle[apple_pixels[apple_members, 1], apple_pixels[apple_members, 0]] = True
    metric_per_pixel = faro_m / math.sqrt(float(low_k[0, 0] * low_k[1, 1]))
    boundary_residual = np.abs(_signed_distance_pixels(faro_obstacle) - _signed_distance_pixels(apple_obstacle)) * metric_per_pixel
    batch.update(
        {
            "boundary_available": True,
            "boundary_localization_abs_residual_m": np.ascontiguousarray(boundary_residual[valid], dtype=np.float64),
        }
    )
    apple_support_points, _ = _unproject(apple_m, support_valid, low_k, 2)
    try:
        apple_plane = _fit_support_plane(apple_support_points, gravity)
    except AdapterError as error:
        if error.code not in _SUPPORT_UNOBSERVABLE_CODES:
            raise
        # FARO support still defines the metric obstacle boundary, so retain
        # boundary residuals even when AppleDepth cannot identify its own plane.
        batch["support_failure_code"] = error.code
        batch["support_failure_stage"] = "APPLE_SUPPORT_PLANE"
        return batch
    support_height_residual = abs(float(faro_plane["camera_height_m"]) - float(apple_plane["camera_height_m"]))
    dot = max(-1.0, min(1.0, float(np.dot(faro_plane["normal_camera_xyz"], apple_plane["normal_camera_xyz"]))))
    batch.update(
        {
            "support_available": True,
            "support_confidence": representative_confidence,
            "support_range_m": representative_range,
            "support_height_abs_residual_m": support_height_residual,
            "support_normal_abs_residual_rad": math.acos(dot),
        }
    )
    return batch


def _quantile_type7(values: Iterable[float], probability: float) -> float:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    else:
        array = np.fromiter((float(value) for value in values), dtype=np.float64)
    require(array.size > 0, "EMPTY_QUANTILE", "quantile input is empty")
    require(bool(np.all(np.isfinite(array))), "NONFINITE_QUANTILE", "quantile input must be finite")
    require(0.0 <= probability <= 1.0, "QUANTILE_PROBABILITY", "quantile probability is out of range")
    return float(np.quantile(array, probability, method="linear"))


@dataclass(frozen=True)
class _UncertaintyCell:
    target: str
    parent_id: str
    confidence: int
    range_bin: int
    values: np.ndarray


class _UncertaintyFactoryToken:
    def __deepcopy__(self, memo: dict[int, Any]) -> _UncertaintyFactoryToken:
        return self


_UNCERTAINTY_FACTORY_TOKEN = _UncertaintyFactoryToken()


def _uncertainty_cell_payload(cell: _UncertaintyCell) -> dict[str, Any]:
    return {
        "target": cell.target,
        "parent_id": cell.parent_id,
        "confidence": cell.confidence,
        "range_bin": cell.range_bin,
        "values": cell.values,
    }


def _uncertainty_model_payload(
    cells: Sequence[_UncertaintyCell],
    fit_parent_ids: Sequence[str],
    source_receipt_sha256s: Sequence[str],
    source_evidence_sha256: str,
    support_frame_observations: int,
    observation_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": UNCERTAINTY_MODEL_SCHEMA,
        "fit_role": "ADAPTER_FIT",
        "frozen_roster_sha256": frozen_roster_sha256(),
        "fit_parent_ids": list(fit_parent_ids),
        "source_receipt_sha256s": list(source_receipt_sha256s),
        "source_evidence_sha256": source_evidence_sha256,
        "source_frame_count": len(source_receipt_sha256s),
        "support_frame_observations": support_frame_observations,
        "observation_counts": dict(observation_counts),
        "parent_aggregation": "PARENT_MACRO_Q95_OF_WITHIN_PARENT_Q95",
        "range_edges_m": list(RANGE_EDGES_M),
        "cells_sha256": canonical_sha256([_uncertainty_cell_payload(cell) for cell in cells]),
    }


@dataclass(frozen=True, eq=False)
class _UncertaintyModel:
    cells: tuple[_UncertaintyCell, ...]
    fit_parent_ids: tuple[str, ...]
    source_receipt_sha256s: tuple[str, ...]
    source_evidence_sha256: str
    support_frame_observations: int
    observation_count_items: tuple[tuple[str, int], ...]
    content_sha256: str
    _factory_token: _UncertaintyFactoryToken

    @property
    def observation_counts(self) -> dict[str, int]:
        return dict(self.observation_count_items)

    def _assert_integrity(self) -> None:
        factory_fingerprint = _UNCERTAINTY_FACTORY_FINGERPRINTS.get(self)
        require(self._factory_token is _UNCERTAINTY_FACTORY_TOKEN and factory_fingerprint is not None, "UNCERTAINTY_MODEL_NOT_FACTORY_BOUND", "uncertainty model must come from fit_uncertainty_model")
        require(factory_fingerprint == self.content_sha256, "UNCERTAINTY_MODEL_FACTORY_FINGERPRINT_MISMATCH", "factory-owned uncertainty model may not be mutated and re-signed")
        expected_parents = tuple(visit_id for visit_id, _ in ADAPTER_FIT_ROSTER)
        require(isinstance(self.cells, tuple) and len(self.cells) > 0, "UNCERTAINTY_MODEL_STRUCTURE_INVALID", "uncertainty cells must be a non-empty immutable tuple")
        require(self.fit_parent_ids == expected_parents, "UNCERTAINTY_MODEL_ROSTER_MISMATCH", "uncertainty model parent roster drift")
        require(
            isinstance(self.source_receipt_sha256s, tuple)
            and len(self.source_receipt_sha256s) == len(set(self.source_receipt_sha256s))
            and len(self.source_receipt_sha256s) > 0
            and all(isinstance(item, str) and bool(_SHA256.fullmatch(item)) for item in self.source_receipt_sha256s),
            "UNCERTAINTY_MODEL_SOURCE_RECEIPTS_INVALID",
            "uncertainty model source receipt identities are malformed or duplicated",
        )
        require(isinstance(self.source_evidence_sha256, str) and bool(_SHA256.fullmatch(self.source_evidence_sha256)), "UNCERTAINTY_MODEL_SOURCE_EVIDENCE_INVALID", "uncertainty source evidence hash is malformed")
        require(isinstance(self.support_frame_observations, int) and not isinstance(self.support_frame_observations, bool) and 0 <= self.support_frame_observations <= len(self.source_receipt_sha256s), "UNCERTAINTY_MODEL_SUPPORT_COUNT_INVALID", "support observations must be a bounded subset of unique source frames")
        observation_counts = self.observation_counts
        require(
            isinstance(self.observation_count_items, tuple)
            and tuple(target for target, _ in self.observation_count_items) == UNCERTAINTY_TARGETS
            and all(isinstance(observation_counts[target], int) and not isinstance(observation_counts[target], bool) and observation_counts[target] >= 0 for target in UNCERTAINTY_TARGETS)
            and observation_counts["scale_log_abs_residual"] > 0,
            "UNCERTAINTY_MODEL_OBSERVATION_COUNTS_INVALID",
            "uncertainty observation counts are malformed",
        )
        seen_keys: set[tuple[str, str, int, int]] = set()
        recomputed_counts = {target: 0 for target in UNCERTAINTY_TARGETS}
        observed_cell_parents: set[str] = set()
        for cell in self.cells:
            require(isinstance(cell, _UncertaintyCell), "UNCERTAINTY_MODEL_CELL_INVALID", "uncertainty cells must be factory-owned immutable records")
            require(cell.target in UNCERTAINTY_TARGETS and cell.parent_id in expected_parents, "UNCERTAINTY_MODEL_CELL_INVALID", "uncertainty cell target/parent drift")
            require(isinstance(cell.confidence, int) and not isinstance(cell.confidence, bool) and cell.confidence in (0, 1, 2), "UNCERTAINTY_MODEL_CELL_INVALID", "uncertainty cell confidence drift")
            require(isinstance(cell.range_bin, int) and not isinstance(cell.range_bin, bool) and 0 <= cell.range_bin < len(RANGE_EDGES_M) - 1, "UNCERTAINTY_MODEL_CELL_INVALID", "uncertainty cell range bin drift")
            key = (cell.target, cell.parent_id, cell.confidence, cell.range_bin)
            require(key not in seen_keys, "UNCERTAINTY_MODEL_CELL_DUPLICATE", "uncertainty cell identity is duplicated")
            seen_keys.add(key)
            values = np.asarray(cell.values)
            require(values.dtype == np.dtype(np.float64) and values.ndim == 1 and values.size > 0 and values.flags.c_contiguous and not values.flags.writeable, "UNCERTAINTY_MODEL_CELL_MUTABLE", "uncertainty cell values must be non-empty read-only contiguous float64")
            require(bool(np.all(np.isfinite(values))) and bool(np.all(values >= 0.0)), "UNCERTAINTY_MODEL_CELL_INVALID", "uncertainty residuals must be finite and non-negative")
            recomputed_counts[cell.target] += int(values.size)
            observed_cell_parents.add(cell.parent_id)
        require(observed_cell_parents == set(expected_parents), "UNCERTAINTY_MODEL_ROSTER_MISMATCH", "uncertainty cells do not cover the exact fit roster")
        require(recomputed_counts == observation_counts, "UNCERTAINTY_MODEL_OBSERVATION_COUNTS_INVALID", "uncertainty observation counts are not recomputable from cells")
        require(
            recomputed_counts["support_height_abs_residual_m"] == self.support_frame_observations
            and recomputed_counts["support_normal_abs_residual_rad"] == self.support_frame_observations,
            "UNCERTAINTY_MODEL_SUPPORT_COUNT_INVALID",
            "support residual targets must contribute exactly once per support-observable source frame",
        )
        require(isinstance(self.content_sha256, str) and bool(_SHA256.fullmatch(self.content_sha256)), "UNCERTAINTY_MODEL_HASH_MISMATCH", "uncertainty model hash is malformed")
        payload = _uncertainty_model_payload(
            self.cells,
            self.fit_parent_ids,
            self.source_receipt_sha256s,
            self.source_evidence_sha256,
            self.support_frame_observations,
            observation_counts,
        )
        require(canonical_sha256(payload) == self.content_sha256, "UNCERTAINTY_MODEL_HASH_MISMATCH", "uncertainty model content drift")

    def resolve(self, confidence: int, range_m: float, target: str) -> dict[str, Any]:
        self._assert_integrity()
        return self._resolve_validated(confidence, range_m, target)

    def _resolve_validated(self, confidence: int, range_m: float, target: str) -> dict[str, Any]:
        require(confidence in (0, 1, 2), "UNCERTAINTY_CONFIDENCE_INVALID", "confidence must be 0, 1, or 2")
        require(target in UNCERTAINTY_TARGETS, "UNCERTAINTY_TARGET_INVALID", "unknown uncertainty target")
        index = _range_bin(float(range_m))
        require(index is not None, "UNCERTAINTY_RANGE_INVALID", "range lies outside frozen bins")

        def evaluate(selected: list[_UncertaintyCell], scope: str) -> dict[str, Any] | None:
            parents = sorted({cell.parent_id for cell in selected})
            samples = sum(int(cell.values.size) for cell in selected)
            if len(parents) < 4 or samples < 128:
                return None
            within: list[float] = []
            for parent in parents:
                arrays = [cell.values for cell in selected if cell.parent_id == parent]
                values = np.concatenate(arrays)
                within.append(_quantile_type7(values, 0.95))
            return {
                "valid": True,
                "value": _quantile_type7(within, 0.95),
                "scope": scope,
                "independent_parents": len(parents),
                "samples": samples,
                "model_sha256": self.content_sha256,
            }

        def select(confidences: set[int], lower: int, upper: int) -> list[_UncertaintyCell]:
            return [
                cell
                for cell in self.cells
                if cell.target == target
                and cell.confidence in confidences
                and lower <= cell.range_bin <= upper
            ]

        result = evaluate(select({confidence}, index, index), "EXACT_CONFIDENCE_EXACT_RANGE")
        if result is not None:
            return result
        for radius in range(1, len(RANGE_EDGES_M) - 1):
            lower, upper = max(0, index - radius), min(len(RANGE_EDGES_M) - 2, index + radius)
            result = evaluate(select({confidence}, lower, upper), f"SAME_CONFIDENCE_SYMMETRIC_CONTIGUOUS_RANGE_EXPANSION_R{radius}_{lower}_{upper}")
            if result is not None:
                return result
        result = evaluate(select({0, 1, 2}, index, index), "ALL_CONFIDENCE_EXACT_RANGE")
        if result is not None:
            return result
        for radius in range(1, len(RANGE_EDGES_M) - 1):
            lower, upper = max(0, index - radius), min(len(RANGE_EDGES_M) - 2, index + radius)
            result = evaluate(select({0, 1, 2}, lower, upper), f"ALL_CONFIDENCE_SYMMETRIC_CONTIGUOUS_RANGE_EXPANSION_R{radius}_{lower}_{upper}")
            if result is not None:
                return result
        result = evaluate([cell for cell in self.cells if cell.target == target], "GLOBAL")
        if result is not None:
            return result
        target_cells = [cell for cell in self.cells if cell.target == target]
        return {
            "valid": False,
            "value": None,
            "scope": "UNCERTAINTY_INVALID_QUERY_UNKNOWN",
            "independent_parents": len({cell.parent_id for cell in target_cells}),
            "samples": sum(int(cell.values.size) for cell in target_cells),
            "model_sha256": self.content_sha256,
        }


_UNCERTAINTY_FACTORY_FINGERPRINTS: weakref.WeakKeyDictionary[_UncertaintyModel, str] = weakref.WeakKeyDictionary()


def _validate_uncertainty_model(value: Any) -> _UncertaintyModel:
    require(isinstance(value, _UncertaintyModel), "UNCERTAINTY_MODEL_NOT_FACTORY_BOUND", "uncertainty model must come from fit_uncertainty_model")
    value._assert_integrity()
    return value


def fit_uncertainty_model(source_frames: Sequence[dict[str, Any]]) -> _UncertaintyModel:
    """Fit residual envelopes only from exact roster-bound source arrays."""

    require(isinstance(source_frames, Sequence) and len(source_frames) > 0, "UNCERTAINTY_SOURCES_EMPTY", "adapter-fit source frames are required")
    expected_parents = tuple(visit_id for visit_id, _ in ADAPTER_FIT_ROSTER)
    batches: list[dict[str, Any]] = []
    receipt_hashes: set[str] = set()
    physical_frames: set[str] = set()
    for index, source_frame in enumerate(source_frames):
        reject_truth_shortcuts(source_frame, f"adapter_fit_source[{index}]")
        batch = _derive_adapter_fit_residual_batch(source_frame)
        require(batch["source_frame_receipt_sha256"] not in receipt_hashes, "UNCERTAINTY_DUPLICATE_SOURCE_RECEIPT", "a source receipt may contribute at most one physical-frame observation", index=index)
        require(batch["physical_frame_id"] not in physical_frames, "UNCERTAINTY_DUPLICATE_PHYSICAL_FRAME", "a physical frame may contribute at most once", index=index)
        receipt_hashes.add(batch["source_frame_receipt_sha256"])
        physical_frames.add(batch["physical_frame_id"])
        batches.append(batch)
    observed_parents = {str(batch["parent_id"]) for batch in batches}
    require(observed_parents == set(expected_parents), "UNCERTAINTY_FIT_ROSTER_INCOMPLETE", "uncertainty fit requires the exact eight-parent ADAPTER_FIT roster", observed=sorted(observed_parents))

    cell_parts: dict[tuple[str, str, int, int], list[np.ndarray]] = {}
    evidence: list[dict[str, Any]] = []
    for batch in batches:
        parent = str(batch["parent_id"])
        confidence = np.asarray(batch["pixel_confidence"], dtype=np.int8)
        ranges = np.asarray(batch["pixel_range_m"], dtype=np.float64)
        require(confidence.ndim == 1 and ranges.shape == confidence.shape, "UNCERTAINTY_DERIVATION_INVALID", "derived pixel confidence/range vectors drift")
        range_bins = np.asarray([_range_bin(float(value)) for value in ranges], dtype=object)
        require(all(value is not None for value in range_bins), "UNCERTAINTY_RANGE_INVALID", "derived pixel range lies outside frozen bins")
        range_bins_i = range_bins.astype(np.int8)
        pixel_targets = ["scale_log_abs_residual"]
        if batch["boundary_available"]:
            pixel_targets.append("boundary_localization_abs_residual_m")
        for target in pixel_targets:
            values = np.asarray(batch[target], dtype=np.float64)
            require(values.shape == confidence.shape and bool(np.all(np.isfinite(values))) and bool(np.all(values >= 0.0)), "UNCERTAINTY_RESIDUAL_INVALID", "derived pixel residuals must be finite and non-negative", target=target)
            for confidence_value in (0, 1, 2):
                for range_bin in range(len(RANGE_EDGES_M) - 1):
                    mask = (confidence == confidence_value) & (range_bins_i == range_bin)
                    if bool(np.any(mask)):
                        key = (target, parent, confidence_value, range_bin)
                        cell_parts.setdefault(key, []).append(np.ascontiguousarray(values[mask], dtype=np.float64))
        if batch["support_available"]:
            support_confidence = int(batch["support_confidence"])
            support_range_bin = _range_bin(float(batch["support_range_m"]))
            require(support_confidence in (0, 1, 2) and support_range_bin is not None, "UNCERTAINTY_DERIVATION_INVALID", "derived support cell is invalid")
            for target in ("support_height_abs_residual_m", "support_normal_abs_residual_rad"):
                residual = float(batch[target])
                require(math.isfinite(residual) and residual >= 0.0, "UNCERTAINTY_RESIDUAL_INVALID", "derived support residual must be finite and non-negative", target=target)
                key = (target, parent, support_confidence, int(support_range_bin))
                cell_parts.setdefault(key, []).append(np.asarray([residual], dtype=np.float64))
        evidence.append(
            {
                "parent_id": parent,
                "physical_frame_id": batch["physical_frame_id"],
                "source_frame_receipt_sha256": batch["source_frame_receipt_sha256"],
                "raw_input_receipts": batch["raw_input_receipts"],
                "target_availability": {
                    "scale_log_abs_residual": True,
                    "support_height_abs_residual_m": bool(batch["support_available"]),
                    "support_normal_abs_residual_rad": bool(batch["support_available"]),
                    "boundary_localization_abs_residual_m": bool(batch["boundary_available"]),
                },
                "support_failure_code": batch["support_failure_code"],
                "support_failure_stage": batch["support_failure_stage"],
            }
        )

    parent_rank = {parent: index for index, parent in enumerate(expected_parents)}
    target_rank = {target: index for index, target in enumerate(UNCERTAINTY_TARGETS)}
    cells: list[_UncertaintyCell] = []
    for (target, parent, confidence_value, range_bin), parts in sorted(
        cell_parts.items(),
        key=lambda item: (target_rank[item[0][0]], parent_rank[item[0][1]], item[0][2], item[0][3]),
    ):
        values = _immutable_array(np.concatenate(parts), np.float64)
        cells.append(_UncertaintyCell(target, parent, confidence_value, range_bin, values))
    source_receipt_sha256s = tuple(sorted(receipt_hashes))
    evidence.sort(key=lambda item: (item["parent_id"], item["physical_frame_id"], item["source_frame_receipt_sha256"]))
    observation_counts = {
        target: sum(int(cell.values.size) for cell in cells if cell.target == target)
        for target in UNCERTAINTY_TARGETS
    }
    support_frame_observations = sum(bool(batch["support_available"]) for batch in batches)
    require(observation_counts["support_height_abs_residual_m"] == support_frame_observations and observation_counts["support_normal_abs_residual_rad"] == support_frame_observations, "UNCERTAINTY_SUPPORT_COUNT_DRIFT", "support residuals must contribute exactly once per support-observable physical frame")
    source_evidence_sha256 = canonical_sha256(evidence)
    payload = _uncertainty_model_payload(
        cells,
        expected_parents,
        source_receipt_sha256s,
        source_evidence_sha256,
        support_frame_observations,
        observation_counts,
    )
    model = _UncertaintyModel(
        tuple(cells),
        expected_parents,
        source_receipt_sha256s,
        source_evidence_sha256,
        support_frame_observations,
        tuple((target, observation_counts[target]) for target in UNCERTAINTY_TARGETS),
        canonical_sha256(payload),
        _UNCERTAINTY_FACTORY_TOKEN,
    )
    _UNCERTAINTY_FACTORY_FINGERPRINTS[model] = model.content_sha256
    return _validate_uncertainty_model(model)


def bootstrap_support_uncertainty(
    support_points_camera_xyz: np.ndarray,
    base_normal_camera_xyz: Any,
    base_camera_height_m: float,
    frame_id: str,
) -> dict[str, Any]:
    """Frozen 256-replicate source-point bootstrap with frame-derived seed."""

    points = np.asarray(support_points_camera_xyz, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) >= MINIMUM_SUPPORT_POINTS, "SUPPORT_BOOTSTRAP_POINTS", "bootstrap requires at least 256 finite support points")
    require(bool(np.all(np.isfinite(points))), "SUPPORT_BOOTSTRAP_POINTS", "bootstrap points must be finite")
    base_normal = _normalize_vector(base_normal_camera_xyz, "SUPPORT_NORMAL_INVALID")
    require(_finite_number(base_camera_height_m) and float(base_camera_height_m) > 0.0, "CAMERA_HEIGHT_INVALID", "base camera height must be positive")
    require(isinstance(frame_id, str) and bool(frame_id), "FRAME_ID_INVALID", "frame_id is required")
    seed = int.from_bytes(hashlib.sha256((frame_id + ":TARO_SUPPORT_BOOTSTRAP_R0").encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    normal_errors: list[float] = []
    height_errors: list[float] = []
    for _ in range(256):
        indices = rng.integers(0, len(points), size=len(points))
        sample = points[indices]
        centered = sample - np.mean(sample, axis=0)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        normal = _normalize_vector(right[-1], "SUPPORT_BOOTSTRAP_DEGENERATE")
        if float(np.dot(normal, base_normal)) < 0.0:
            normal = -normal
        height = -float(np.median(sample @ normal))
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(normal, base_normal)))))
        normal_errors.append(angle)
        height_errors.append(abs(height - float(base_camera_height_m)))
    return {
        "schema": "blindassist.taro.o0r.support_bootstrap.v1",
        "replicates": 256,
        "seed_first_64_bits": seed,
        "normal_q95_rad": _quantile_type7(normal_errors, 0.95),
        "height_q95_m": _quantile_type7(height_errors, 0.95),
    }


def _query_receipt_vectors(receipt: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = receipt.get("virtual_query_frame")
    require(isinstance(frame, dict), "QUERY_FRAME_INVALID", "virtual query frame is missing")
    origin = np.asarray(frame.get("origin_camera_xyz"), dtype=np.float64)
    require(origin.shape == (3,) and bool(np.all(np.isfinite(origin))), "QUERY_FRAME_INVALID", "query origin must be finite length three")
    forward = _normalize_vector(frame.get("forward_camera_xyz"), "QUERY_FRAME_INVALID")
    lateral = _normalize_vector(frame.get("lateral_camera_xyz"), "QUERY_FRAME_INVALID")
    heading = _normalize_vector(frame.get("path_heading_camera_xyz"), "QUERY_FRAME_INVALID")
    require(abs(float(np.dot(forward, lateral))) <= 1e-6, "QUERY_FRAME_INVALID", "query forward/lateral axes are not orthogonal")
    return origin, forward, lateral, heading


def _local_valid_fraction(geometry: FaroGeometry, receipt: dict[str, Any]) -> float:
    origin, _, lateral, heading = _query_receipt_vectors(receipt)
    query_up = _normalize_vector(
        receipt["virtual_query_frame"]["gravity_up_camera_xyz"],
        "QUERY_FRAME_INVALID",
    )
    path_side = _normalize_vector(np.cross(heading, query_up), "QUERY_FRAME_INVALID")
    path_origin = origin + float(receipt["path_lateral_offset_m"]) * lateral
    matrix = geometry.intrinsics
    valid_count = 0
    total_count = 0
    half_h, half_w = BOUNDARY_NEIGHBORHOOD_HW[0] // 2, BOUNDARY_NEIGHBORHOOD_HW[1] // 2
    for forward_m in np.linspace(MINIMUM_FORWARD_M, HORIZON_M, 10):
        for lateral_m in (-CAPSULE_RADIUS_M, 0.0, CAPSULE_RADIUS_M):
            point = path_origin + forward_m * heading + lateral_m * path_side
            if point[2] <= 1e-9:
                total_count += BOUNDARY_NEIGHBORHOOD_HW[0] * BOUNDARY_NEIGHBORHOOD_HW[1]
                continue
            column = int(round(float(matrix[0, 0] * point[0] / point[2] + matrix[0, 2])))
            row = int(round(float(matrix[1, 1] * point[1] / point[2] + matrix[1, 2])))
            for delta_row in range(-half_h, half_h + 1):
                for delta_column in range(-half_w, half_w + 1):
                    rr, cc = row + delta_row, column + delta_column
                    total_count += 1
                    if 0 <= rr < geometry.valid_depth.shape[0] and 0 <= cc < geometry.valid_depth.shape[1] and bool(geometry.valid_depth[rr, cc]):
                        valid_count += 1
    return valid_count / float(total_count) if total_count else 0.0


def _local_surface(geometry: FaroGeometry, receipt: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    origin, _, lateral, heading = _query_receipt_vectors(receipt)
    query_up = _normalize_vector(
        receipt["virtual_query_frame"]["gravity_up_camera_xyz"],
        "QUERY_FRAME_INVALID",
    )
    path_side = _normalize_vector(np.cross(heading, query_up), "QUERY_FRAME_INVALID")
    path_origin = origin + float(receipt["path_lateral_offset_m"]) * lateral
    points = geometry.points_camera_xyz
    rel = points - path_origin
    along = rel @ heading
    across = rel @ path_side
    height = points @ geometry.support_normal_camera_xyz + geometry.camera_height_m
    keep = (
        (along >= MINIMUM_FORWARD_M - 0.20)
        & (along <= HORIZON_M + 0.20)
        # Any point farther than horizon + capsule radius cannot lower the
        # clearance below the positive HORIZON_M cap.
        & (np.abs(across) <= HORIZON_M + CAPSULE_RADIUS_M)
        & (height >= -0.20)
        & (height <= 2.20)
    )
    selected_points = np.ascontiguousarray(points[keep], dtype=np.float64)
    selected_pixels = np.ascontiguousarray(geometry.pixels_uv[keep], dtype=np.int32)
    return selected_points, selected_pixels, _local_valid_fraction(geometry, receipt)


def _query_support_and_boundary(
    points: np.ndarray,
    pixels_uv: np.ndarray,
    support_normal_camera_xyz: Any,
    camera_height_shape_m: float,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    require(points.ndim == 2 and points.shape[1] == 3 and pixels_uv.shape == (len(points), 2), "QUERY_GEOMETRY_INPUT_INVALID", "query geometry points/pixels drift")
    normal = _normalize_vector(support_normal_camera_xyz, "SUPPORT_VALUE_INVALID")
    require(_finite_number(camera_height_shape_m) and float(camera_height_shape_m) > 0.0, "SUPPORT_VALUE_INVALID", "support height must be finite and positive")
    origin, _, lateral, heading = _query_receipt_vectors(receipt)
    query_up = _normalize_vector(receipt["virtual_query_frame"]["gravity_up_camera_xyz"], "QUERY_FRAME_INVALID")
    path_origin = origin + float(receipt["path_lateral_offset_m"]) * lateral
    rel = points - path_origin
    along = rel @ heading
    perpendicular_vector = rel - along[:, None] * heading[None, :]
    perpendicular_ground = perpendicular_vector - (perpendicular_vector @ query_up)[:, None] * query_up[None, :]
    perpendicular = np.linalg.norm(perpendicular_ground, axis=1)
    heights = points @ normal + float(camera_height_shape_m)
    support_members = np.abs(heights) <= SUPPORT_RESIDUAL_TOLERANCE_M
    support_corridor = (
        support_members
        & (along >= MINIMUM_FORWARD_M)
        & (along <= HORIZON_M + GEOMETRY_ENDPOINT_TOLERANCE_M)
        & (perpendicular <= CAPSULE_RADIUS_M)
    )
    query_support_points = int(np.sum(support_corridor))
    observed_forward = min(HORIZON_M, float(np.max(along[support_corridor]))) if query_support_points else None
    boundary_members = (
        (heights >= OBSTACLE_HEIGHT_RANGE_M[0])
        & (heights <= OBSTACLE_HEIGHT_RANGE_M[1])
        & (along >= MINIMUM_FORWARD_M)
        & (along <= HORIZON_M + GEOMETRY_ENDPOINT_TOLERANCE_M)
    )
    return {
        "query_support_points": query_support_points,
        "observed_forward_shape_m": observed_forward,
        "boundary_points_shape_camera_xyz": np.ascontiguousarray(points[boundary_members], dtype=np.float64),
        "boundary_point_ids_uv": np.ascontiguousarray(pixels_uv[boundary_members], dtype=np.int32),
    }


def _build_base_geometry(geometry: FaroGeometry, receipt: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    geometry = _validate_faro_geometry(geometry)
    surface_points, pixels, local_fraction = _local_surface(geometry, receipt)
    require(len(surface_points) > 0, "QUERY_LOCAL_SURFACE_EMPTY", "query has no local FARO surface")
    query_geometry = _query_support_and_boundary(surface_points, pixels, geometry.support_normal_camera_xyz, geometry.camera_height_m, receipt)
    truth_values = {
        "SCALE": {"log_metric_scale": 0.0, "value_kind": "ABSOLUTE_FARO_METRIC_REFERENCE"},
        "SUPPORT": {"normal_camera_xyz": geometry.support_normal_camera_xyz.tolist(), "camera_height_shape_m": geometry.camera_height_m},
        "BOUNDARY": {"point_ids_uv": query_geometry["boundary_point_ids_uv"], "boundary_points_shape_camera_xyz": query_geometry["boundary_points_shape_camera_xyz"]},
    }
    base = _seal(
        {
            "schema": BASE_GEOMETRY_SCHEMA,
            "origin": "FARO_TRUTH_COMMON_SUPPORT",
            "physical_frame_id": geometry.physical_frame_id,
            "query_id": receipt["query_id"],
            "source_frame_receipt_sha256": geometry.source_frame_receipt_sha256,
            "query_receipt_sha256": receipt["content_sha256"],
            "max_source_timestamp_ns": geometry.max_source_timestamp_ns,
            "faro_geometry_sha256": geometry.content_sha256,
            "faro_depth_array_sha256": geometry.highres_depth_array_sha256,
            "faro_factor_value_sha256s": {name: canonical_sha256(truth_values[name]) for name in FACTOR_NAMES},
            "common_point_ids_uv": pixels,
            "local_valid_fraction": local_fraction,
        }
    )
    return base, surface_points, query_geometry


def _validate_base_geometry(value: Any, frame: dict[str, Any] | None = None) -> dict[str, Any]:
    base = _validate_seal(value, "BASE_GEOMETRY_HASH_MISMATCH")
    keys = {
        "schema",
        "origin",
        "physical_frame_id",
        "query_id",
        "source_frame_receipt_sha256",
        "query_receipt_sha256",
        "max_source_timestamp_ns",
        "faro_geometry_sha256",
        "faro_depth_array_sha256",
        "faro_factor_value_sha256s",
        "common_point_ids_uv",
        "local_valid_fraction",
        "content_sha256",
    }
    require(set(base) == keys and base["schema"] == BASE_GEOMETRY_SCHEMA and base["origin"] == "FARO_TRUTH_COMMON_SUPPORT", "BASE_GEOMETRY_KEY_SET", "base geometry key/schema drift")
    for key in ("source_frame_receipt_sha256", "query_receipt_sha256", "faro_geometry_sha256", "faro_depth_array_sha256"):
        require(isinstance(base[key], str) and bool(_SHA256.fullmatch(base[key])), "BASE_GEOMETRY_IDENTITY", "base geometry hash identity is malformed", field=key)
    factor_hashes = base["faro_factor_value_sha256s"]
    require(isinstance(factor_hashes, dict) and tuple(factor_hashes) == FACTOR_NAMES and all(isinstance(factor_hashes[name], str) and bool(_SHA256.fullmatch(factor_hashes[name])) for name in FACTOR_NAMES), "BASE_GEOMETRY_FACTOR_HASHES", "base FARO factor hashes are malformed or reordered")
    raw_ids = np.asarray(base["common_point_ids_uv"])
    require(np.issubdtype(raw_ids.dtype, np.integer), "BASE_GEOMETRY_POINT_IDS", "base geometry point identities must be integer pixels")
    point_ids = raw_ids.astype(np.int64, copy=False)
    require(point_ids.ndim == 2 and point_ids.shape[1] == 2 and len(point_ids) > 0, "BASE_GEOMETRY_POINT_IDS", "base geometry requires non-empty Nx2 point identities")
    require(bool(np.all((point_ids[:, 0] >= 0) & (point_ids[:, 0] < HIGHRES_SHAPE_HW[1]) & (point_ids[:, 1] >= 0) & (point_ids[:, 1] < HIGHRES_SHAPE_HW[0]))), "BASE_GEOMETRY_POINT_IDS", "base geometry point identities leave the highres raster")
    linear = point_ids[:, 1] * HIGHRES_SHAPE_HW[1] + point_ids[:, 0]
    require(len(np.unique(linear)) == len(linear), "BASE_GEOMETRY_POINT_IDS", "base geometry point identities must be unique")
    require(_finite_number(base["local_valid_fraction"]) and 0.0 <= float(base["local_valid_fraction"]) <= 1.0, "BASE_GEOMETRY_COVERAGE", "base local coverage is invalid")
    require(isinstance(base["max_source_timestamp_ns"], int) and not isinstance(base["max_source_timestamp_ns"], bool) and base["max_source_timestamp_ns"] >= 0, "BASE_GEOMETRY_WATERMARK", "base geometry watermark is invalid")
    if frame is not None:
        for key in ("physical_frame_id", "query_id", "source_frame_receipt_sha256", "query_receipt_sha256", "max_source_timestamp_ns"):
            require(base[key] == frame[key], "BASE_GEOMETRY_IDENTITY", "base geometry/frame identity mismatch", field=key)
    return base


def _uncertainty_blocks(
    uncertainty_model: _UncertaintyModel,
    confidence_value: int,
    range_m: float,
    bootstrap: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    uncertainty_model = _validate_uncertainty_model(uncertainty_model)
    scale = uncertainty_model._resolve_validated(confidence_value, range_m, "scale_log_abs_residual")
    support_height = uncertainty_model._resolve_validated(confidence_value, range_m, "support_height_abs_residual_m")
    support_normal = uncertainty_model._resolve_validated(confidence_value, range_m, "support_normal_abs_residual_rad")
    boundary = uncertainty_model._resolve_validated(confidence_value, range_m, "boundary_localization_abs_residual_m")
    scale_uncertainty = {
        "valid": bool(scale["valid"]),
        "q95_log": scale["value"],
        "resolution_scope": scale["scope"],
        "fit_model_sha256": uncertainty_model.content_sha256,
    }
    support_valid = bool(support_height["valid"] and support_normal["valid"])
    support_uncertainty = {
        "valid": support_valid,
        "height_q95_shape_m": max(float(bootstrap["height_q95_m"]), float(support_height["value"])) if support_valid else None,
        "normal_q95_rad": max(float(bootstrap["normal_q95_rad"]), float(support_normal["value"])) if support_valid else None,
        "height_resolution_scope": support_height["scope"],
        "normal_resolution_scope": support_normal["scope"],
        "bootstrap_seed_first_64_bits": bootstrap["seed_first_64_bits"],
        "fit_model_sha256": uncertainty_model.content_sha256,
    }
    boundary_uncertainty = {
        "valid": bool(boundary["valid"]),
        "localization_q95_shape_m": boundary["value"],
        "resolution_scope": boundary["scope"],
        "fit_model_sha256": uncertainty_model.content_sha256,
    }
    complete = bool(scale_uncertainty["valid"] and support_uncertainty["valid"] and boundary_uncertainty["valid"])
    return scale_uncertainty, support_uncertainty, boundary_uncertainty, complete


def _block_common(name: str, frame_identity: dict[str, Any], base_geometry_sha256: str) -> dict[str, Any]:
    return {
        "physical_frame_id": frame_identity["physical_frame_id"],
        "query_id": frame_identity["query_id"],
        "source_frame_receipt_sha256": frame_identity["source_frame_receipt_sha256"],
        "query_receipt_sha256": frame_identity["query_receipt_sha256"],
        "base_geometry_sha256": base_geometry_sha256,
        "factor_name": name,
    }


def _truth_support_bootstrap(geometry: FaroGeometry) -> dict[str, Any]:
    return bootstrap_support_uncertainty(
        geometry.support_points_camera_xyz,
        geometry.support_normal_camera_xyz,
        geometry.camera_height_m,
        geometry.physical_frame_id,
    )


def build_truth_query_factor_frame(
    geometry: FaroGeometry,
    query_receipt: dict[str, Any],
    uncertainty_model: _UncertaintyModel,
    *,
    confidence_value: int,
    range_m: float,
) -> dict[str, Any]:
    """Build one query-bound FARO truth factor frame with fitted uncertainty."""

    uncertainty_model = _validate_uncertainty_model(uncertainty_model)
    geometry = _validate_faro_geometry(geometry)
    return _build_truth_query_factor_frame_with_bootstrap(
        geometry,
        query_receipt,
        uncertainty_model,
        confidence_value=confidence_value,
        range_m=range_m,
        bootstrap=_truth_support_bootstrap(geometry),
    )


def build_truth_query_factor_frames(
    geometry: FaroGeometry,
    query_receipts: Sequence[dict[str, Any]],
    uncertainty_model: _UncertaintyModel,
    *,
    confidence_values: Sequence[int],
    ranges_m: Sequence[float],
) -> list[dict[str, Any]]:
    """Build one physical frame's exact nine truth queries with one shared bootstrap."""

    uncertainty_model = _validate_uncertainty_model(uncertainty_model)
    geometry = _validate_faro_geometry(geometry)
    require(
        isinstance(query_receipts, RuntimeSequence)
        and isinstance(confidence_values, RuntimeSequence)
        and isinstance(ranges_m, RuntimeSequence)
        and len(query_receipts) == len(confidence_values) == len(ranges_m) == 9,
        "QUERY_BUNDLE_CARDINALITY",
        "truth factor batch requires exactly nine query parameter tuples",
    )
    bootstrap = _truth_support_bootstrap(geometry)
    return [
        _build_truth_query_factor_frame_with_bootstrap(
            geometry,
            receipt,
            uncertainty_model,
            confidence_value=confidence_value,
            range_m=range_m,
            bootstrap=bootstrap,
        )
        for receipt, confidence_value, range_m in zip(query_receipts, confidence_values, ranges_m, strict=True)
    ]


def _build_truth_query_factor_frame_with_bootstrap(
    geometry: FaroGeometry,
    query_receipt: dict[str, Any],
    uncertainty_model: _UncertaintyModel,
    *,
    confidence_value: int,
    range_m: float,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    receipt = _validate_query_receipt(query_receipt)
    require(receipt.get("physical_frame_id") == geometry.physical_frame_id, "QUERY_FRAME_ID_MISMATCH", "query and FARO geometry frame identities differ")
    require(receipt.get("source_frame_receipt_sha256") == geometry.source_frame_receipt_sha256, "QUERY_BASE_RECEIPT_MISMATCH", "query and FARO geometry base receipts differ")
    require(int(receipt.get("max_source_timestamp_ns")) == geometry.max_source_timestamp_ns, "QUERY_WATERMARK_MISMATCH", "query watermark drift")
    expected_receipt = _build_geometry_query_receipts(
        physical_frame_id=geometry.physical_frame_id,
        source_frame_receipt_sha256=geometry.source_frame_receipt_sha256,
        max_source_timestamp_ns=geometry.max_source_timestamp_ns,
        support_normal_camera_xyz=geometry.support_normal_camera_xyz,
        camera_height_m=geometry.camera_height_m,
    )[int(receipt["grid_index"])]
    require(receipt["content_sha256"] == expected_receipt["content_sha256"], "QUERY_GEOMETRY_BINDING_INVALID", "query frame is not derived from the bound FARO support plane")
    base_geometry, _, query_geometry = _build_base_geometry(geometry, receipt)
    scale_uncertainty, support_uncertainty, boundary_uncertainty, uncertainties_valid = _uncertainty_blocks(
        uncertainty_model, confidence_value, range_m, bootstrap
    )
    frame_identity = {
        "physical_frame_id": geometry.physical_frame_id,
        "query_id": receipt["query_id"],
        "source_frame_receipt_sha256": geometry.source_frame_receipt_sha256,
        "query_receipt_sha256": receipt["content_sha256"],
        "max_source_timestamp_ns": geometry.max_source_timestamp_ns,
    }
    blocks = {
        "SCALE": {
            **_block_common("SCALE", frame_identity, base_geometry["content_sha256"]),
            "value": {"log_metric_scale": 0.0, "value_kind": "ABSOLUTE_FARO_METRIC_REFERENCE"},
            "validity": {"valid": True, "model_independent": True},
            "uncertainty": scale_uncertainty,
        },
        "SUPPORT": {
            **_block_common("SUPPORT", frame_identity, base_geometry["content_sha256"]),
            "value": {
                "normal_camera_xyz": geometry.support_normal_camera_xyz.tolist(),
                "camera_height_shape_m": geometry.camera_height_m,
            },
            "validity": {
                "valid": geometry.support_count >= MINIMUM_SUPPORT_POINTS
                and geometry.support_fraction >= MINIMUM_SUPPORT_FRACTION
                and geometry.support_slope_degrees <= MAXIMUM_SUPPORT_SLOPE_DEGREES,
                "support_point_count": geometry.support_count,
                "query_support_points": query_geometry["query_support_points"],
                "observed_forward_shape_m": query_geometry["observed_forward_shape_m"],
                "support_fraction": geometry.support_fraction,
                "slope_degrees": geometry.support_slope_degrees,
                "median_residual_m": geometry.support_median_residual_m,
            },
            "uncertainty": support_uncertainty,
        },
        "BOUNDARY": {
            **_block_common("BOUNDARY", frame_identity, base_geometry["content_sha256"]),
            "value": {
                "point_ids_uv": query_geometry["boundary_point_ids_uv"],
                "boundary_points_shape_camera_xyz": query_geometry["boundary_points_shape_camera_xyz"],
            },
            "validity": {
                "valid": float(base_geometry["local_valid_fraction"]) >= MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION,
                "common_support_point_count": int(len(base_geometry["common_point_ids_uv"])),
                "local_valid_fraction": float(base_geometry["local_valid_fraction"]),
            },
            "uncertainty": boundary_uncertainty,
        },
    }
    frame = {
        "schema": QUERY_FACTOR_FRAME_SCHEMA,
        **frame_identity,
        "base_geometry": base_geometry,
        "factor_identity": {
            "origin": "FARO_TRUTH",
            "candidate_id": "ABSOLUTE_FARO_METRIC_REFERENCE",
            "truth_only": True,
            "uncertainty_complete": uncertainties_valid,
            "input_depth_array_sha256": geometry.highres_depth_array_sha256,
            "faro_geometry_sha256": geometry.content_sha256,
        },
        "blocks": blocks,
    }
    output = _seal(frame)
    validate_query_factor_frame(output)
    return output


def build_candidate_depth_output_receipt(
    candidate_depth_m: np.ndarray,
    source_frame_receipt: dict[str, Any],
    *,
    inference_receipt_sha256: str,
) -> dict[str, Any]:
    """Bind one metric baseline depth array to the frozen model/source interface."""

    source = _validate_base_receipt(source_frame_receipt)
    require(source["source_role"] == "O0R_EVAL_CANDIDATE", "CANDIDATE_SOURCE_ROLE_INVALID", "candidate output requires a frozen eval source")
    raw = np.asarray(candidate_depth_m)
    require(raw.shape == HIGHRES_SHAPE_HW and raw.dtype.kind in "iuf" and bool(np.all(np.isfinite(raw))), "CANDIDATE_DEPTH_INVALID", "candidate output must be a finite numeric 1440x1920 metric-depth raster")
    require(isinstance(inference_receipt_sha256, str) and bool(_SHA256.fullmatch(inference_receipt_sha256)), "CANDIDATE_INFERENCE_RECEIPT_INVALID", "candidate inference receipt hash is malformed")
    return _seal(
        {
            "schema": CANDIDATE_OUTPUT_RECEIPT_SCHEMA,
            "model_id": BASELINE_MODEL_ID,
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "inference_receipt_sha256": inference_receipt_sha256.upper(),
            "source_frame_receipt_sha256": source["content_sha256"],
            "physical_frame_id": source["physical_frame_id"],
            "max_source_timestamp_ns": source["max_source_timestamp_ns"],
            "input_color_decoded_content_sha256": source["decoded_payload_bindings"]["color"]["decoded_content_sha256"],
            "effective_intrinsics_sha256": canonical_sha256(source["intrinsics_highres"]),
            "output_kind": "METRIC_DEPTH_M",
            "output_array_sha256": canonical_sha256(raw),
            "baseline_log_metric_scale": 0.0,
            "truth_alignment_used": False,
        }
    )


def _validate_candidate_depth_output_receipt(
    value: Any,
    candidate_depth_m: np.ndarray,
    source: dict[str, Any],
) -> dict[str, Any]:
    receipt = _validate_candidate_depth_output_receipt_envelope(value)
    require(receipt["source_frame_receipt_sha256"] == source["content_sha256"] and receipt["physical_frame_id"] == source["physical_frame_id"] and receipt["max_source_timestamp_ns"] == source["max_source_timestamp_ns"], "CANDIDATE_OUTPUT_SOURCE_MISMATCH", "candidate output receipt source identity drift")
    require(receipt["input_color_decoded_content_sha256"] == source["decoded_payload_bindings"]["color"]["decoded_content_sha256"] and receipt["effective_intrinsics_sha256"] == canonical_sha256(source["intrinsics_highres"]), "CANDIDATE_OUTPUT_INPUT_MISMATCH", "candidate output receipt RGB/K binding drift")
    require(receipt["output_array_sha256"] == canonical_sha256(candidate_depth_m), "CANDIDATE_OUTPUT_ARRAY_MISMATCH", "candidate output array drift")
    return receipt


def _validate_candidate_depth_output_receipt_envelope(value: Any) -> dict[str, Any]:
    """Validate the self-contained fixed-baseline fields retained in a factor frame."""

    receipt = _validate_seal(value, "CANDIDATE_OUTPUT_RECEIPT_HASH_MISMATCH")
    keys = {
        "schema",
        "model_id",
        "checkpoint_sha256",
        "inference_receipt_sha256",
        "source_frame_receipt_sha256",
        "physical_frame_id",
        "max_source_timestamp_ns",
        "input_color_decoded_content_sha256",
        "effective_intrinsics_sha256",
        "output_kind",
        "output_array_sha256",
        "baseline_log_metric_scale",
        "truth_alignment_used",
        "content_sha256",
    }
    require(set(receipt) == keys and receipt["schema"] == CANDIDATE_OUTPUT_RECEIPT_SCHEMA, "CANDIDATE_OUTPUT_RECEIPT_KEY_SET", "candidate output receipt key/schema drift")
    require(receipt["model_id"] == BASELINE_MODEL_ID and receipt["checkpoint_sha256"] == BASELINE_CHECKPOINT_SHA256, "CANDIDATE_MODEL_BINDING_INVALID", "candidate output is not bound to the frozen DepthART baseline")
    for key in (
        "checkpoint_sha256",
        "inference_receipt_sha256",
        "source_frame_receipt_sha256",
        "input_color_decoded_content_sha256",
        "effective_intrinsics_sha256",
        "output_array_sha256",
    ):
        require(isinstance(receipt[key], str) and bool(_SHA256.fullmatch(receipt[key])), "CANDIDATE_OUTPUT_RECEIPT_HASH_INVALID", "candidate output receipt contains a malformed hash", field=key)
    require(isinstance(receipt["physical_frame_id"], str) and bool(receipt["physical_frame_id"]), "CANDIDATE_OUTPUT_RECEIPT_IDENTITY_INVALID", "candidate output physical-frame identity is required")
    require(isinstance(receipt["max_source_timestamp_ns"], int) and not isinstance(receipt["max_source_timestamp_ns"], bool) and receipt["max_source_timestamp_ns"] >= 0, "CANDIDATE_OUTPUT_RECEIPT_IDENTITY_INVALID", "candidate output watermark is invalid")
    require(receipt["output_kind"] == "METRIC_DEPTH_M", "CANDIDATE_OUTPUT_ARRAY_MISMATCH", "candidate output unit drift")
    require(receipt["baseline_log_metric_scale"] == 0.0 and receipt["truth_alignment_used"] is False, "CANDIDATE_SCALE_INVALID", "baseline scale must remain frozen at metric 0 without truth alignment")
    return receipt


def build_candidate_query_factor_frame(
    candidate_depth_m: np.ndarray,
    intrinsics_highres_3x3: Any,
    gravity_up_camera_xyz: Any,
    source_frame_receipt: dict[str, Any],
    query_receipt: dict[str, Any],
    base_geometry: dict[str, Any],
    uncertainty_model: _UncertaintyModel,
    candidate_output_receipt: dict[str, Any],
    *,
    confidence_value: int,
    range_m: float,
) -> dict[str, Any]:
    """Extract baseline S/P/B factors from candidate depth on frozen truth support."""

    uncertainty_model = _validate_uncertainty_model(uncertainty_model)
    source = _validate_base_receipt(source_frame_receipt)
    require(source["source_role"] == "O0R_EVAL_CANDIDATE", "CANDIDATE_SOURCE_ROLE_INVALID", "candidate extraction requires a roster-bound O0R_EVAL_CANDIDATE receipt")
    receipt = _validate_query_receipt(query_receipt)
    require(receipt["source_frame_receipt_sha256"] == source["content_sha256"] and receipt["physical_frame_id"] == source["physical_frame_id"] and receipt["max_source_timestamp_ns"] == source["max_source_timestamp_ns"], "CANDIDATE_QUERY_BINDING_INVALID", "candidate query is not bound to the supplied source receipt")
    frame_identity = {
        "physical_frame_id": receipt["physical_frame_id"],
        "query_id": receipt["query_id"],
        "source_frame_receipt_sha256": receipt["source_frame_receipt_sha256"],
        "query_receipt_sha256": receipt["content_sha256"],
        "max_source_timestamp_ns": receipt["max_source_timestamp_ns"],
    }
    base = _validate_base_geometry(base_geometry, frame_identity)
    matrix = _intrinsics_matrix(intrinsics_highres_3x3)
    bound_matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    require(bool(np.allclose(matrix, bound_matrix, rtol=0.0, atol=1e-12)), "RECEIPT_INTRINSICS_MISMATCH", "candidate intrinsics do not match source receipt")
    gravity = _normalize_vector(gravity_up_camera_xyz, "GRAVITY_INVALID")
    bound_gravity = _normalize_vector(source["gravity_up_camera_xyz"], "GRAVITY_INVALID")
    require(bool(np.allclose(gravity, bound_gravity, rtol=0.0, atol=1e-12)), "RECEIPT_GRAVITY_MISMATCH", "candidate gravity does not match source receipt")
    raw = np.asarray(candidate_depth_m)
    require(raw.shape == HIGHRES_SHAPE_HW and raw.dtype.kind in "iuf" and bool(np.all(np.isfinite(raw))), "CANDIDATE_DEPTH_INVALID", "candidate depth must be a finite numeric 1440x1920 metric-depth raster")
    output_receipt = _validate_candidate_depth_output_receipt(candidate_output_receipt, raw, source)
    depth = raw.astype(np.float64, copy=False)
    point_ids = np.asarray(base["common_point_ids_uv"], dtype=np.int64)
    u, v = point_ids[:, 0], point_ids[:, 1]
    z_all = depth[v, u]
    valid = (z_all >= DEPTH_RANGE_M[0]) & (z_all <= DEPTH_RANGE_M[1])
    require(int(np.sum(valid)) >= MINIMUM_SUPPORT_POINTS, "CANDIDATE_COMMON_SUPPORT_INSUFFICIENT", "candidate has too few valid points on frozen truth support")
    valid_ids = point_ids[valid].astype(np.int32, copy=False)
    z = z_all[valid]
    points = np.stack(((valid_ids[:, 0] - matrix[0, 2]) * z / matrix[0, 0], (valid_ids[:, 1] - matrix[1, 2]) * z / matrix[1, 1], z), axis=1)
    stride_mask = (valid_ids[:, 0] % SUPPORT_POINT_STRIDE == 0) & (valid_ids[:, 1] % SUPPORT_POINT_STRIDE == 0)
    support_input = points[stride_mask]
    if len(support_input) < MINIMUM_SUPPORT_POINTS:
        support_input = points[::SUPPORT_POINT_STRIDE]
    plane = _fit_support_plane(support_input, gravity)
    query_geometry = _query_support_and_boundary(points, valid_ids, plane["normal_camera_xyz"], float(plane["camera_height_m"]), receipt)
    bootstrap = bootstrap_support_uncertainty(plane["support_points"], plane["normal_camera_xyz"], float(plane["camera_height_m"]), f"{receipt['physical_frame_id']}:{BASELINE_MODEL_ID}")
    scale_uncertainty, support_uncertainty, boundary_uncertainty, uncertainties_valid = _uncertainty_blocks(
        uncertainty_model, confidence_value, range_m, bootstrap
    )
    candidate_local_fraction = float(base["local_valid_fraction"]) * (int(np.sum(valid)) / float(len(valid)))
    blocks = {
        "SCALE": {
            **_block_common("SCALE", frame_identity, base["content_sha256"]),
            "value": {"log_metric_scale": 0.0, "value_kind": "FIXED_METRIC_BASELINE_SCALE_NO_TRUTH_ALIGNMENT"},
            "validity": {"valid": True, "model_independent": False},
            "uncertainty": scale_uncertainty,
        },
        "SUPPORT": {
            **_block_common("SUPPORT", frame_identity, base["content_sha256"]),
            "value": {"normal_camera_xyz": np.asarray(plane["normal_camera_xyz"], dtype=np.float64).tolist(), "camera_height_shape_m": float(plane["camera_height_m"])},
            "validity": {
                "valid": int(plane["support_count"]) >= MINIMUM_SUPPORT_POINTS and float(plane["support_fraction"]) >= MINIMUM_SUPPORT_FRACTION and float(plane["slope_degrees"]) <= MAXIMUM_SUPPORT_SLOPE_DEGREES,
                "support_point_count": int(plane["support_count"]),
                "query_support_points": query_geometry["query_support_points"],
                "observed_forward_shape_m": query_geometry["observed_forward_shape_m"],
                "support_fraction": float(plane["support_fraction"]),
                "slope_degrees": float(plane["slope_degrees"]),
                "median_residual_m": float(plane["median_residual_m"]),
            },
            "uncertainty": support_uncertainty,
        },
        "BOUNDARY": {
            **_block_common("BOUNDARY", frame_identity, base["content_sha256"]),
            "value": {"point_ids_uv": query_geometry["boundary_point_ids_uv"], "boundary_points_shape_camera_xyz": query_geometry["boundary_points_shape_camera_xyz"]},
            "validity": {
                "valid": candidate_local_fraction >= MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION,
                "common_support_point_count": int(len(base["common_point_ids_uv"])),
                "local_valid_fraction": candidate_local_fraction,
            },
            "uncertainty": boundary_uncertainty,
        },
    }
    frame = {
        "schema": QUERY_FACTOR_FRAME_SCHEMA,
        **frame_identity,
        "base_geometry": copy.deepcopy(base),
        "factor_identity": {
            "origin": "CANDIDATE_DEPTH_EXTRACTOR",
            "candidate_id": BASELINE_MODEL_ID,
            "truth_only": False,
            "uncertainty_complete": uncertainties_valid,
            "input_depth_array_sha256": output_receipt["output_array_sha256"],
            "candidate_output_receipt": copy.deepcopy(output_receipt),
            "candidate_output_receipt_sha256": output_receipt["content_sha256"],
            "fixed_metric_scale_only": True,
        },
        "blocks": blocks,
    }
    output = _seal(frame)
    validate_query_factor_frame(output)
    return output


def _validate_factor_block(name: str, block: Any, frame: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(block, dict), "FACTOR_BLOCK_INVALID", "factor block must be an object", factor=name)
    common = {
        "physical_frame_id",
        "query_id",
        "source_frame_receipt_sha256",
        "query_receipt_sha256",
        "base_geometry_sha256",
        "factor_name",
        "value",
        "validity",
        "uncertainty",
    }
    require(set(block) == common, "FACTOR_BLOCK_KEY_SET", "factor block key set drift", factor=name)
    require(block["factor_name"] == name, "FACTOR_BLOCK_IDENTITY", "factor name drift", factor=name)
    for key in ("physical_frame_id", "query_id", "source_frame_receipt_sha256", "query_receipt_sha256"):
        require(block[key] == frame[key], "FACTOR_BLOCK_IDENTITY", "factor/frame identity mismatch", factor=name, field=key)
    require(block["base_geometry_sha256"] == frame["base_geometry"]["content_sha256"], "BASE_GEOMETRY_MISMATCH", "factor block is not bound to immutable base geometry", factor=name)
    require(isinstance(block["value"], dict) and isinstance(block["validity"], dict) and isinstance(block["uncertainty"], dict), "FACTOR_BLOCK_INVALID", "factor value/validity/uncertainty must be objects", factor=name)
    return block


def _valid_uncertainty_scope(scope: Any) -> bool:
    if not isinstance(scope, str):
        return False
    if scope in {
        "EXACT_CONFIDENCE_EXACT_RANGE",
        "ALL_CONFIDENCE_EXACT_RANGE",
        "GLOBAL",
        "UNCERTAINTY_INVALID_QUERY_UNKNOWN",
    }:
        return True
    return bool(
        re.fullmatch(
            r"(?:SAME_CONFIDENCE|ALL_CONFIDENCE)_SYMMETRIC_CONTIGUOUS_RANGE_EXPANSION_R[1-3]_[0-3]_[0-3]",
            scope,
        )
    )


def validate_query_factor_frame(frame: Any) -> dict[str, Any]:
    value = _validate_seal(frame, "FACTOR_FRAME_HASH_MISMATCH")
    keys = {
        "schema",
        "physical_frame_id",
        "query_id",
        "source_frame_receipt_sha256",
        "query_receipt_sha256",
        "max_source_timestamp_ns",
        "base_geometry",
        "factor_identity",
        "blocks",
        "content_sha256",
    }
    require(set(value) == keys and value["schema"] == QUERY_FACTOR_FRAME_SCHEMA, "FACTOR_FRAME_KEY_SET", "query factor frame key/schema drift")
    require(isinstance(value["physical_frame_id"], str) and bool(value["physical_frame_id"]) and isinstance(value["query_id"], str) and bool(value["query_id"]), "FACTOR_FRAME_IDENTITY", "factor frame/query identities are required")
    for key in ("source_frame_receipt_sha256", "query_receipt_sha256"):
        require(isinstance(value[key], str) and bool(_SHA256.fullmatch(value[key])), "FACTOR_FRAME_IDENTITY", "factor frame receipt hash is malformed", field=key)
    require(isinstance(value["max_source_timestamp_ns"], int) and not isinstance(value["max_source_timestamp_ns"], bool) and value["max_source_timestamp_ns"] >= 0, "FACTOR_FRAME_IDENTITY", "factor frame watermark is invalid")
    _validate_base_geometry(value["base_geometry"], value)
    require(isinstance(value["blocks"], dict) and tuple(value["blocks"].keys()) == FACTOR_NAMES, "FACTOR_BLOCK_ORDER", "factor blocks must be SCALE, SUPPORT, BOUNDARY")
    blocks = {name: _validate_factor_block(name, value["blocks"][name], value) for name in FACTOR_NAMES}
    scale_value = blocks["SCALE"]["value"]
    require(set(scale_value) == {"log_metric_scale", "value_kind"} and _finite_number(scale_value["log_metric_scale"]), "SCALE_VALUE_INVALID", "scale value is invalid")
    require(isinstance(scale_value["value_kind"], str) and bool(scale_value["value_kind"]), "SCALE_VALUE_INVALID", "scale value kind is required")
    scale_validity = blocks["SCALE"]["validity"]
    require(set(scale_validity) == {"valid", "model_independent"} and all(isinstance(scale_validity[key], bool) for key in scale_validity), "SCALE_VALIDITY_INVALID", "scale validity key/type drift")
    require(scale_validity["valid"] is True, "SCALE_VALIDITY_INVALID", "finite extractor scale must be valid")
    scale_uncertainty = blocks["SCALE"]["uncertainty"]
    require(set(scale_uncertainty) == {"valid", "q95_log", "resolution_scope", "fit_model_sha256"}, "SCALE_UNCERTAINTY_INVALID", "scale uncertainty key set drift")
    require(_valid_uncertainty_scope(scale_uncertainty["resolution_scope"]), "SCALE_UNCERTAINTY_INVALID", "scale uncertainty resolution scope is invalid")
    support_value = blocks["SUPPORT"]["value"]
    require(set(support_value) == {"normal_camera_xyz", "camera_height_shape_m"}, "SUPPORT_VALUE_INVALID", "support value key set drift")
    _normalize_vector(support_value["normal_camera_xyz"], "SUPPORT_VALUE_INVALID")
    require(_finite_number(support_value["camera_height_shape_m"]) and float(support_value["camera_height_shape_m"]) > 0.0, "SUPPORT_VALUE_INVALID", "support height is invalid")
    support_validity = blocks["SUPPORT"]["validity"]
    support_validity_keys = {"valid", "support_point_count", "query_support_points", "observed_forward_shape_m", "support_fraction", "slope_degrees", "median_residual_m"}
    require(set(support_validity) == support_validity_keys and isinstance(support_validity["valid"], bool), "SUPPORT_VALIDITY_INVALID", "support validity key/type drift")
    require(isinstance(support_validity["support_point_count"], int) and not isinstance(support_validity["support_point_count"], bool) and support_validity["support_point_count"] >= 0, "SUPPORT_VALIDITY_INVALID", "support point count is invalid")
    require(isinstance(support_validity["query_support_points"], int) and not isinstance(support_validity["query_support_points"], bool) and support_validity["query_support_points"] >= 0, "SUPPORT_VALIDITY_INVALID", "query support count is invalid")
    require(support_validity["observed_forward_shape_m"] is None or (_finite_number(support_validity["observed_forward_shape_m"]) and float(support_validity["observed_forward_shape_m"]) >= 0.0), "SUPPORT_VALIDITY_INVALID", "observed forward distance is invalid")
    require(_finite_number(support_validity["support_fraction"]) and 0.0 <= float(support_validity["support_fraction"]) <= 1.0 and _finite_number(support_validity["slope_degrees"]) and float(support_validity["slope_degrees"]) >= 0.0 and _finite_number(support_validity["median_residual_m"]) and float(support_validity["median_residual_m"]) >= 0.0, "SUPPORT_VALIDITY_INVALID", "support diagnostic validity values are invalid")
    expected_support_valid = int(support_validity["support_point_count"]) >= MINIMUM_SUPPORT_POINTS and float(support_validity["support_fraction"]) >= MINIMUM_SUPPORT_FRACTION and float(support_validity["slope_degrees"]) <= MAXIMUM_SUPPORT_SLOPE_DEGREES
    require(support_validity["valid"] == expected_support_valid, "SUPPORT_VALIDITY_INVALID", "support validity is not recomputable from frozen fraction/slope gates")
    support_uncertainty = blocks["SUPPORT"]["uncertainty"]
    support_uncertainty_keys = {"valid", "height_q95_shape_m", "normal_q95_rad", "height_resolution_scope", "normal_resolution_scope", "bootstrap_seed_first_64_bits", "fit_model_sha256"}
    require(set(support_uncertainty) == support_uncertainty_keys, "SUPPORT_UNCERTAINTY_INVALID", "support uncertainty key set drift")
    require(_valid_uncertainty_scope(support_uncertainty["height_resolution_scope"]) and _valid_uncertainty_scope(support_uncertainty["normal_resolution_scope"]), "SUPPORT_UNCERTAINTY_INVALID", "support uncertainty resolution scope is invalid")
    require(isinstance(support_uncertainty["bootstrap_seed_first_64_bits"], int) and not isinstance(support_uncertainty["bootstrap_seed_first_64_bits"], bool) and 0 <= support_uncertainty["bootstrap_seed_first_64_bits"] < 2**64, "SUPPORT_UNCERTAINTY_INVALID", "support bootstrap seed is invalid")
    boundary_value = blocks["BOUNDARY"]["value"]
    require(set(boundary_value) == {"point_ids_uv", "boundary_points_shape_camera_xyz"}, "BOUNDARY_VALUE_INVALID", "boundary value key set drift")
    raw_point_ids = np.asarray(boundary_value["point_ids_uv"])
    require(np.issubdtype(raw_point_ids.dtype, np.integer), "BOUNDARY_VALUE_INVALID", "boundary point identities must be integer pixels")
    point_ids = raw_point_ids.astype(np.int64, copy=False)
    points = np.asarray(boundary_value["boundary_points_shape_camera_xyz"], dtype=np.float64)
    require(point_ids.ndim == 2 and point_ids.shape[1] == 2 and points.shape == (len(point_ids), 3) and bool(np.all(np.isfinite(points))), "BOUNDARY_VALUE_INVALID", "boundary point identities/coordinates must be aligned Nx2/Nx3")
    if len(point_ids):
        linear = point_ids[:, 1] * HIGHRES_SHAPE_HW[1] + point_ids[:, 0]
        require(len(np.unique(linear)) == len(linear), "BOUNDARY_VALUE_INVALID", "boundary point identities must be unique")
        base_ids = np.asarray(value["base_geometry"]["common_point_ids_uv"], dtype=np.int64)
        base_linear = set((base_ids[:, 1] * HIGHRES_SHAPE_HW[1] + base_ids[:, 0]).tolist())
        require(all(int(item) in base_linear for item in linear), "BOUNDARY_OUTSIDE_BASE_GEOMETRY", "boundary points must be a subset of immutable base geometry")
    boundary_validity = blocks["BOUNDARY"]["validity"]
    require(set(boundary_validity) == {"valid", "common_support_point_count", "local_valid_fraction"} and isinstance(boundary_validity["valid"], bool), "BOUNDARY_VALIDITY_INVALID", "boundary validity key/type drift")
    require(boundary_validity["common_support_point_count"] == len(value["base_geometry"]["common_point_ids_uv"]), "BOUNDARY_VALIDITY_INVALID", "boundary validity is not bound to the immutable common-support count")
    require(_finite_number(boundary_validity["local_valid_fraction"]) and 0.0 <= float(boundary_validity["local_valid_fraction"]) <= 1.0, "BOUNDARY_VALIDITY_INVALID", "boundary local coverage is invalid")
    require(float(boundary_validity["local_valid_fraction"]) <= float(value["base_geometry"]["local_valid_fraction"]) + 1e-12, "BOUNDARY_VALIDITY_INVALID", "boundary local coverage exceeds immutable FARO base coverage")
    require(boundary_validity["valid"] == (float(boundary_validity["local_valid_fraction"]) >= MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION), "BOUNDARY_VALIDITY_INVALID", "boundary validity is not recomputable from the frozen coverage gate")
    boundary_uncertainty = blocks["BOUNDARY"]["uncertainty"]
    require(set(boundary_uncertainty) == {"valid", "localization_q95_shape_m", "resolution_scope", "fit_model_sha256"}, "BOUNDARY_UNCERTAINTY_INVALID", "boundary uncertainty key set drift")
    require(_valid_uncertainty_scope(boundary_uncertainty["resolution_scope"]), "BOUNDARY_UNCERTAINTY_INVALID", "boundary uncertainty resolution scope is invalid")
    for uncertainty, numeric_keys in ((scale_uncertainty, ("q95_log",)), (support_uncertainty, ("height_q95_shape_m", "normal_q95_rad")), (boundary_uncertainty, ("localization_q95_shape_m",))):
        require(isinstance(uncertainty["valid"], bool) and isinstance(uncertainty["fit_model_sha256"], str) and bool(_SHA256.fullmatch(uncertainty["fit_model_sha256"])), "FACTOR_UNCERTAINTY_INVALID", "factor uncertainty provenance is invalid")
        for key in numeric_keys:
            require((uncertainty["valid"] and _finite_number(uncertainty[key]) and float(uncertainty[key]) >= 0.0) or (not uncertainty["valid"] and uncertainty[key] is None), "FACTOR_UNCERTAINTY_INVALID", "factor uncertainty value/validity mismatch", field=key)
    require(scale_uncertainty["valid"] == (scale_uncertainty["resolution_scope"] != "UNCERTAINTY_INVALID_QUERY_UNKNOWN"), "SCALE_UNCERTAINTY_INVALID", "scale uncertainty validity/scope mismatch")
    require(boundary_uncertainty["valid"] == (boundary_uncertainty["resolution_scope"] != "UNCERTAINTY_INVALID_QUERY_UNKNOWN"), "BOUNDARY_UNCERTAINTY_INVALID", "boundary uncertainty validity/scope mismatch")
    support_scopes_valid = support_uncertainty["height_resolution_scope"] != "UNCERTAINTY_INVALID_QUERY_UNKNOWN" and support_uncertainty["normal_resolution_scope"] != "UNCERTAINTY_INVALID_QUERY_UNKNOWN"
    require(support_uncertainty["valid"] == support_scopes_valid, "SUPPORT_UNCERTAINTY_INVALID", "support uncertainty validity/scope mismatch")
    fit_hashes = {blocks[name]["uncertainty"]["fit_model_sha256"] for name in FACTOR_NAMES}
    require(len(fit_hashes) == 1, "FACTOR_UNCERTAINTY_PROVENANCE_MISMATCH", "factor blocks must use one uncertainty model")
    identity = value["factor_identity"]
    require(isinstance(identity, dict) and isinstance(identity.get("origin"), str), "FACTOR_IDENTITY_INVALID", "factor identity is invalid")
    if identity["origin"] == "FARO_TRUTH":
        expected = {"origin", "candidate_id", "truth_only", "uncertainty_complete", "input_depth_array_sha256", "faro_geometry_sha256"}
        require(set(identity) == expected and identity["candidate_id"] == "ABSOLUTE_FARO_METRIC_REFERENCE" and identity["truth_only"] is True, "FACTOR_IDENTITY_INVALID", "FARO truth identity drift")
        require(scale_value["value_kind"] == "ABSOLUTE_FARO_METRIC_REFERENCE" and float(scale_value["log_metric_scale"]) == 0.0 and scale_validity["model_independent"] is True, "FACTOR_IDENTITY_INVALID", "FARO truth scale block drift")
        require(identity["input_depth_array_sha256"] == value["base_geometry"]["faro_depth_array_sha256"], "FACTOR_IDENTITY_INVALID", "FARO truth depth is not bound to base geometry")
        require(identity["faro_geometry_sha256"] == value["base_geometry"]["faro_geometry_sha256"], "FACTOR_IDENTITY_INVALID", "FARO truth geometry is not bound to base geometry")
        require(all(canonical_sha256(blocks[name]["value"]) == value["base_geometry"]["faro_factor_value_sha256s"][name] for name in FACTOR_NAMES), "FACTOR_IDENTITY_INVALID", "FARO truth factor values do not match the immutable geometry evidence")
    elif identity["origin"] == "CANDIDATE_DEPTH_EXTRACTOR":
        expected = {"origin", "candidate_id", "truth_only", "uncertainty_complete", "input_depth_array_sha256", "candidate_output_receipt", "candidate_output_receipt_sha256", "fixed_metric_scale_only"}
        require(set(identity) == expected and identity["candidate_id"] == BASELINE_MODEL_ID and identity["truth_only"] is False and identity["fixed_metric_scale_only"] is True, "FACTOR_IDENTITY_INVALID", "candidate factor identity drift")
        require(isinstance(identity["candidate_output_receipt_sha256"], str) and bool(_SHA256.fullmatch(identity["candidate_output_receipt_sha256"])), "FACTOR_IDENTITY_INVALID", "candidate output receipt hash is malformed")
        output_receipt = _validate_candidate_depth_output_receipt_envelope(identity["candidate_output_receipt"])
        require(output_receipt["content_sha256"] == identity["candidate_output_receipt_sha256"], "FACTOR_IDENTITY_INVALID", "candidate output receipt hash binding drift")
        require(output_receipt["source_frame_receipt_sha256"] == value["source_frame_receipt_sha256"] and output_receipt["physical_frame_id"] == value["physical_frame_id"] and output_receipt["max_source_timestamp_ns"] == value["max_source_timestamp_ns"], "FACTOR_IDENTITY_INVALID", "candidate output receipt/frame binding drift")
        require(output_receipt["output_array_sha256"] == identity["input_depth_array_sha256"], "FACTOR_IDENTITY_INVALID", "candidate factor input hash is not the bound model output")
        require(scale_value["value_kind"] == "FIXED_METRIC_BASELINE_SCALE_NO_TRUTH_ALIGNMENT" and float(scale_value["log_metric_scale"]) == 0.0 and scale_validity["model_independent"] is False, "FACTOR_IDENTITY_INVALID", "candidate scale must remain frozen at metric zero without truth alignment")
    elif identity["origin"] == "FACTORIAL_INJECTION":
        expected = {"origin", "baseline_candidate_id", "oracle_candidate_id", "baseline_factor_frame_sha256", "oracle_factor_frame_sha256", "arm", "mode", "patched_factors", "block_parent_lineage", "truth_only", "uncertainty_complete"}
        require(set(identity) == expected and identity["truth_only"] is False and identity["arm"] in ARMS and identity["mode"] in ORACLE_MODES and identity["patched_factors"] == list(_arm_factors(identity["arm"])), "FACTOR_IDENTITY_INVALID", "factorial injection identity drift")
        require(identity["baseline_candidate_id"] == BASELINE_MODEL_ID and identity["oracle_candidate_id"] == "ABSOLUTE_FARO_METRIC_REFERENCE", "FACTOR_IDENTITY_INVALID", "factorial parent candidate identities drift")
        for key in ("baseline_factor_frame_sha256", "oracle_factor_frame_sha256"):
            require(isinstance(identity[key], str) and bool(_SHA256.fullmatch(identity[key])), "FACTOR_IDENTITY_INVALID", "factorial parent frame hash is malformed", field=key)
        lineage = identity["block_parent_lineage"]
        require(isinstance(lineage, dict) and tuple(lineage) == FACTOR_NAMES, "FACTOR_LINEAGE_INVALID", "factorial block lineage key/order drift")
        selected = set(identity["patched_factors"])
        for name in FACTOR_NAMES:
            require(isinstance(lineage[name], dict) and set(lineage[name]) == {"value", "validity", "uncertainty"}, "FACTOR_LINEAGE_INVALID", "factorial component lineage key drift", factor=name)
            for component in ("value", "validity", "uncertainty"):
                record = lineage[name][component]
                require(isinstance(record, dict) and set(record) == {"source", "sha256"}, "FACTOR_LINEAGE_INVALID", "factorial component lineage record drift", factor=name, component=component)
                expected_oracle = name in selected and (component == "value" or identity["mode"] == "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY")
                require(record["source"] == ("ORACLE" if expected_oracle else "BASELINE") and record["sha256"] == canonical_sha256(blocks[name][component]), "FACTOR_LINEAGE_INVALID", "factorial component content/lineage mismatch", factor=name, component=component)
    else:
        raise AdapterError("FACTOR_IDENTITY_INVALID", "unknown factor identity origin")
    require(isinstance(identity["uncertainty_complete"], bool) and identity["uncertainty_complete"] == all(blocks[name]["uncertainty"]["valid"] for name in FACTOR_NAMES), "FACTOR_IDENTITY_INVALID", "factor uncertainty completeness drift")
    if "input_depth_array_sha256" in identity:
        require(isinstance(identity["input_depth_array_sha256"], str) and bool(_SHA256.fullmatch(identity["input_depth_array_sha256"])), "FACTOR_IDENTITY_INVALID", "factor input depth hash is malformed")
    return value


def _unknown_query(receipt: dict[str, Any], frame: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "schema": QUERY_RESULT_SCHEMA,
        "reducer_version": REDUCER_VERSION,
        "physical_frame_id": receipt.get("physical_frame_id", "INVALID_FRAME"),
        "query_id": receipt.get("query_id", "INVALID_QUERY"),
        "factor_frame_sha256": frame.get("content_sha256") if isinstance(frame, dict) else None,
        "max_source_timestamp_ns": receipt.get("max_source_timestamp_ns"),
        "value_m": None,
        "uncertainty_m": None,
        "interval_m": {"lower": None, "upper": None},
        "state": "UNKNOWN",
        "knownness": {
            "known": False,
            "support_points": 0,
            "observed_forward_m": None,
            "local_valid_fraction": None,
        },
        "reason_codes": [reason],
    }


def reduce_query_factor_frame(
    frame: dict[str, Any],
    query_receipt: dict[str, Any],
    *,
    factorial_parent_context: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """TARO-specific nine-query interval reducer; never calls the 3-band reducer."""

    receipt = _validate_query_receipt(query_receipt)
    try:
        factor_frame = validate_query_factor_frame(frame)
    except AdapterError as error:
        return _unknown_query(receipt, frame if isinstance(frame, dict) else None, error.code)
    if factor_frame["factor_identity"]["origin"] == "FACTORIAL_INJECTION":
        if not isinstance(factorial_parent_context, tuple) or len(factorial_parent_context) != 2:
            return _unknown_query(receipt, factor_frame, "FACTORIAL_PARENT_CONTEXT_REQUIRED")
        try:
            baseline, oracle = (validate_query_factor_frame(item) for item in factorial_parent_context)
            rebuilt = inject_factor_blocks(
                baseline,
                oracle,
                arm=factor_frame["factor_identity"]["arm"],
                mode=factor_frame["factor_identity"]["mode"],
            )
            require(rebuilt["content_sha256"] == factor_frame["content_sha256"], "FACTORIAL_PARENT_CONTEXT_MISMATCH", "factorial frame is not byte-derived from the supplied parent frames")
        except AdapterError as error:
            return _unknown_query(receipt, factor_frame, error.code)
    elif factorial_parent_context is not None:
        return _unknown_query(receipt, factor_frame, "UNEXPECTED_FACTORIAL_PARENT_CONTEXT")
    identity_matches = (
        factor_frame["physical_frame_id"] == receipt["physical_frame_id"]
        and factor_frame["query_id"] == receipt["query_id"]
        and factor_frame["query_receipt_sha256"] == receipt["content_sha256"]
        and factor_frame["source_frame_receipt_sha256"] == receipt["source_frame_receipt_sha256"]
        and int(factor_frame["max_source_timestamp_ns"]) == int(receipt["max_source_timestamp_ns"])
    )
    if not identity_matches:
        return _unknown_query(receipt, factor_frame, "QUERY_FACTOR_IDENTITY_MISMATCH")
    scale = factor_frame["blocks"]["SCALE"]
    support = factor_frame["blocks"]["SUPPORT"]
    boundary = factor_frame["blocks"]["BOUNDARY"]
    if scale["validity"].get("valid") is not True or support["validity"].get("valid") is not True or boundary["validity"].get("valid") is not True:
        return _unknown_query(receipt, factor_frame, "FACTOR_VALIDITY_INCOMPLETE")
    if scale["uncertainty"].get("valid") is not True or support["uncertainty"].get("valid") is not True or boundary["uncertainty"].get("valid") is not True:
        return _unknown_query(receipt, factor_frame, "UNCERTAINTY_INVALID_QUERY_UNKNOWN")
    numeric_uncertainties = (
        scale["uncertainty"].get("q95_log"),
        support["uncertainty"].get("height_q95_shape_m"),
        support["uncertainty"].get("normal_q95_rad"),
        boundary["uncertainty"].get("localization_q95_shape_m"),
    )
    if not all(_finite_number(item) and float(item) >= 0.0 for item in numeric_uncertainties):
        return _unknown_query(receipt, factor_frame, "UNCERTAINTY_INVALID_QUERY_UNKNOWN")
    log_scale = float(scale["value"]["log_metric_scale"])
    if log_scale < -50.0 or log_scale > 50.0:
        return _unknown_query(receipt, factor_frame, "SCALE_RANGE_INVALID")
    metric_scale = math.exp(log_scale)
    if not math.isfinite(metric_scale) or metric_scale <= 0.0:
        return _unknown_query(receipt, factor_frame, "SCALE_RANGE_INVALID")
    support_normal = _normalize_vector(support["value"]["normal_camera_xyz"], "SUPPORT_VALUE_INVALID")
    camera_height_m = metric_scale * float(support["value"]["camera_height_shape_m"])
    if not math.isfinite(camera_height_m) or camera_height_m <= 0.0:
        return _unknown_query(receipt, factor_frame, "SCALE_RANGE_INVALID")
    points = metric_scale * np.asarray(boundary["value"]["boundary_points_shape_camera_xyz"], dtype=np.float64)
    if not bool(np.all(np.isfinite(points))):
        return _unknown_query(receipt, factor_frame, "SCALE_RANGE_INVALID")
    _, _, _, receipt_heading = _query_receipt_vectors(receipt)
    heading = receipt_heading - float(np.dot(receipt_heading, support_normal)) * support_normal
    heading_norm = float(np.linalg.norm(heading))
    if not math.isfinite(heading_norm) or heading_norm <= 1e-12:
        return _unknown_query(receipt, factor_frame, "SUPPORT_PATH_HEADING_INVALID")
    heading = heading / heading_norm
    lateral_raw = np.cross(heading, support_normal)
    lateral_norm = float(np.linalg.norm(lateral_raw))
    if not math.isfinite(lateral_norm) or lateral_norm <= 1e-12:
        return _unknown_query(receipt, factor_frame, "SUPPORT_PATH_LATERAL_INVALID")
    lateral = lateral_raw / lateral_norm
    path_origin = -camera_height_m * support_normal + float(receipt["path_lateral_offset_m"]) * lateral
    rel = points - path_origin
    along = rel @ heading
    support_count = int(support["validity"]["query_support_points"])
    observed_shape = support["validity"]["observed_forward_shape_m"]
    observed_forward = min(HORIZON_M, metric_scale * float(observed_shape)) if _finite_number(observed_shape) else None
    if observed_forward is not None and not math.isfinite(observed_forward):
        return _unknown_query(receipt, factor_frame, "SCALE_RANGE_INVALID")
    local_fraction = factor_frame["base_geometry"]["local_valid_fraction"]
    known = (
        support_count >= MINIMUM_QUERY_SUPPORT_POINTS
        and observed_forward is not None
        and observed_forward >= MINIMUM_QUERY_OBSERVED_FORWARD_M
        and _finite_number(local_fraction)
        and float(local_fraction) >= MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION
    )
    if not known:
        result = _unknown_query(receipt, factor_frame, "QUERY_KNOWNNESS_GATE_FAILED")
        result["knownness"] = {
            "known": False,
            "support_points": support_count,
            "observed_forward_m": observed_forward,
            "local_valid_fraction": float(local_fraction) if _finite_number(local_fraction) else None,
        }
        return result
    eligible_obstacles = (along >= MINIMUM_FORWARD_M) & (along <= HORIZON_M + GEOMETRY_ENDPOINT_TOLERANCE_M)
    if bool(np.any(eligible_obstacles)):
        t = np.clip(along[eligible_obstacles], MINIMUM_FORWARD_M, HORIZON_M)
        closest = path_origin[None, :] + t[:, None] * heading[None, :]
        displacement = points[eligible_obstacles] - closest
        displacement_ground = displacement - (displacement @ support_normal)[:, None] * support_normal[None, :]
        distance = np.linalg.norm(displacement_ground, axis=1)
        value_m = min(HORIZON_M, float(np.min(distance)) - CAPSULE_RADIUS_M)
    else:
        value_m = HORIZON_M
    scale_q95_log, height_q95_shape, normal_q95, boundary_q95_shape = (float(item) for item in numeric_uncertainties)
    if scale_q95_log > 50.0:
        return _unknown_query(receipt, factor_frame, "UNCERTAINTY_INVALID_QUERY_UNKNOWN")
    scale_uncertainty_m = HORIZON_M * (math.exp(scale_q95_log) - 1.0)
    support_uncertainty_m = metric_scale * height_q95_shape + HORIZON_M * math.tan(min(normal_q95, math.radians(45.0)))
    boundary_uncertainty_m = metric_scale * boundary_q95_shape
    uncertainty_m = scale_uncertainty_m + support_uncertainty_m + boundary_uncertainty_m
    if not all(math.isfinite(item) and item >= 0.0 for item in (scale_uncertainty_m, support_uncertainty_m, boundary_uncertainty_m, uncertainty_m)):
        return _unknown_query(receipt, factor_frame, "UNCERTAINTY_INVALID_QUERY_UNKNOWN")
    lower, upper = value_m - uncertainty_m, value_m + uncertainty_m
    if lower > CLEAR_MARGIN_M:
        state = "CLEAR_OBSERVED"
    elif upper <= OCCUPIED_MARGIN_M:
        state = "OCCUPIED_OBSERVED"
    else:
        state = "UNKNOWN"
    return _canonicalize(
        {
            "schema": QUERY_RESULT_SCHEMA,
            "reducer_version": REDUCER_VERSION,
            "physical_frame_id": receipt["physical_frame_id"],
            "query_id": receipt["query_id"],
            "factor_frame_sha256": factor_frame["content_sha256"],
            "max_source_timestamp_ns": receipt["max_source_timestamp_ns"],
            "value_m": value_m,
            "uncertainty_m": uncertainty_m,
            "interval_m": {"lower": lower, "upper": upper},
            "state": state,
            "knownness": {
                "known": True,
                "support_points": support_count,
                "observed_forward_m": observed_forward,
                "local_valid_fraction": float(local_fraction),
            },
            "reason_codes": [],
        }
    )


def reduce_complete_query_bundle(
    factor_frames: Sequence[dict[str, Any]],
    query_receipts: Sequence[dict[str, Any]],
    source_frame_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Enforce one physical frame's exact 9/9 query cardinality and binding."""

    require(isinstance(factor_frames, Sequence) and len(factor_frames) == 9, "QUERY_BUNDLE_CARDINALITY", "exactly nine query factor frames are required")
    require(isinstance(query_receipts, Sequence) and len(query_receipts) == 9, "QUERY_BUNDLE_CARDINALITY", "exactly nine query receipts are required")
    source = _validate_base_receipt(source_frame_receipt)
    require(source["source_role"] == "O0R_EVAL_CANDIDATE", "QUERY_BUNDLE_SOURCE_ROLE_INVALID", "truth bundle requires a frozen eval source receipt")
    receipts = [_validate_query_receipt(item) for item in query_receipts]
    require([item.get("grid_index") for item in receipts] == list(range(9)), "QUERY_BUNDLE_ORDER", "query receipts must be in frozen grid order 0..8")
    query_ids = [item.get("query_id") for item in receipts]
    require(len(set(query_ids)) == 9, "QUERY_BUNDLE_DUPLICATE", "query receipt identities must be unique")
    for field in ("physical_frame_id", "source_frame_receipt_sha256", "max_source_timestamp_ns"):
        require(len({item.get(field) for item in receipts}) == 1, "QUERY_BUNDLE_BASE_MISMATCH", "query receipts do not share one physical base", field=field)
    require(receipts[0]["physical_frame_id"] == source["physical_frame_id"] and receipts[0]["source_frame_receipt_sha256"] == source["content_sha256"] and receipts[0]["max_source_timestamp_ns"] == source["max_source_timestamp_ns"], "QUERY_BUNDLE_BASE_MISMATCH", "query bundle is not bound to the supplied eval source receipt")
    frames_by_query: dict[str, dict[str, Any]] = {}
    for frame in factor_frames:
        validated_frame = validate_query_factor_frame(frame)
        require(validated_frame["factor_identity"]["origin"] == "FARO_TRUTH" and validated_frame["factor_identity"]["truth_only"] is True, "QUERY_BUNDLE_TRUTH_ORIGIN_REQUIRED", "complete_factor_query_truth accepts only genuine FARO truth frames")
        query_id = validated_frame.get("query_id")
        require(isinstance(query_id, str) and query_id not in frames_by_query, "QUERY_BUNDLE_FACTOR_IDENTITY", "factor frames must have nine unique query identities")
        frames_by_query[query_id] = validated_frame
    require(set(frames_by_query) == set(query_ids), "QUERY_BUNDLE_FACTOR_IDENTITY", "factor-frame and receipt query sets differ")
    results = [reduce_query_factor_frame(frames_by_query[receipt["query_id"]], receipt) for receipt in receipts]
    complete = all(
        result["knownness"]["known"] is True
        and result["value_m"] is not None
        and result["uncertainty_m"] is not None
        for result in results
    )
    state_counts = {
        state: sum(result["state"] == state for result in results)
        for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
    }
    return _seal(
        {
            "schema": QUERY_BUNDLE_RESULT_SCHEMA,
            "physical_frame_id": receipts[0]["physical_frame_id"],
            "source_frame_receipt_sha256": receipts[0]["source_frame_receipt_sha256"],
            "max_source_timestamp_ns": receipts[0]["max_source_timestamp_ns"],
            "query_receipts": 9,
            "factor_frames": 9,
            "complete_factor_query_truth": complete,
            "state_counts": state_counts,
            "results": results,
        }
    )


def _arm_factors(arm: str) -> tuple[str, ...]:
    require(arm in ARMS, "FACTORIAL_ARM_INVALID", "unknown factorial arm")
    if arm == "NONE":
        return ()
    return tuple(name for name in FACTOR_NAMES if name in arm.split("_"))


def inject_factor_blocks(
    baseline_frame: dict[str, Any],
    oracle_frame: dict[str, Any],
    *,
    arm: str,
    mode: str,
) -> dict[str, Any]:
    """Patch only named S/P/B blocks from genuine extractor outputs."""

    baseline = validate_query_factor_frame(baseline_frame)
    oracle = validate_query_factor_frame(oracle_frame)
    require(mode in ORACLE_MODES, "ORACLE_MODE_INVALID", "unknown oracle mode")
    require(baseline["factor_identity"]["origin"] == "CANDIDATE_DEPTH_EXTRACTOR", "FACTOR_BASELINE_ORIGIN_INVALID", "factorial baseline must come from the candidate depth extractor")
    require(oracle["factor_identity"]["origin"] == "FARO_TRUTH", "FACTOR_ORACLE_ORIGIN_INVALID", "factorial oracle must come from the FARO truth extractor")
    for key in ("physical_frame_id", "query_id", "source_frame_receipt_sha256", "query_receipt_sha256", "max_source_timestamp_ns"):
        require(baseline[key] == oracle[key], "FACTOR_IDENTITY_MISMATCH", "baseline/oracle identities differ", field=key)
    require(baseline["base_geometry"]["content_sha256"] == oracle["base_geometry"]["content_sha256"], "BASE_GEOMETRY_MISMATCH", "baseline/oracle immutable base geometry differs")
    patched = copy.deepcopy(baseline)
    selected = _arm_factors(arm)
    for name in selected:
        if mode == "VALUE_ONLY_COMMON_SUPPORT":
            patched["blocks"][name]["value"] = copy.deepcopy(oracle["blocks"][name]["value"])
        else:
            patched["blocks"][name] = copy.deepcopy(oracle["blocks"][name])
    block_parent_lineage: dict[str, Any] = {}
    for name in FACTOR_NAMES:
        component_lineage: dict[str, Any] = {}
        for component in ("value", "validity", "uncertainty"):
            uses_oracle = name in selected and (component == "value" or mode == "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY")
            source = "ORACLE" if uses_oracle else "BASELINE"
            component_lineage[component] = {
                "source": source,
                "sha256": canonical_sha256((oracle if uses_oracle else baseline)["blocks"][name][component]),
            }
        block_parent_lineage[name] = component_lineage
    patched["factor_identity"] = {
        "origin": "FACTORIAL_INJECTION",
        "baseline_candidate_id": baseline["factor_identity"]["candidate_id"],
        "oracle_candidate_id": oracle["factor_identity"]["candidate_id"],
        "baseline_factor_frame_sha256": baseline["content_sha256"],
        "oracle_factor_frame_sha256": oracle["content_sha256"],
        "arm": arm,
        "mode": mode,
        "patched_factors": list(selected),
        "block_parent_lineage": block_parent_lineage,
        "truth_only": False,
        "uncertainty_complete": all(patched["blocks"][name]["uncertainty"]["valid"] for name in FACTOR_NAMES),
    }
    patched.pop("content_sha256", None)
    output = _seal(patched)
    validate_query_factor_frame(output)
    for name in FACTOR_NAMES:
        if name not in selected:
            require(canonical_sha256(output["blocks"][name]) == canonical_sha256(baseline["blocks"][name]), "UNNAMED_FACTOR_CHANGED", "injection changed an unnamed factor", factor=name)
        elif mode == "VALUE_ONLY_COMMON_SUPPORT":
            require(canonical_sha256(output["blocks"][name]["validity"]) == canonical_sha256(baseline["blocks"][name]["validity"]), "VALUE_ONLY_VALIDITY_CHANGED", "VALUE_ONLY changed validity", factor=name)
            require(canonical_sha256(output["blocks"][name]["uncertainty"]) == canonical_sha256(baseline["blocks"][name]["uncertainty"]), "VALUE_ONLY_UNCERTAINTY_CHANGED", "VALUE_ONLY changed uncertainty", factor=name)
    return output


__all__ = [
    "AdapterError",
    "ARMS",
    "APPLE_SHAPE_HW",
    "BASELINE_CHECKPOINT_SHA256",
    "BASELINE_MODEL_ID",
    "BASE_GEOMETRY_SCHEMA",
    "BASE_RECEIPT_SCHEMA",
    "CANDIDATE_OUTPUT_RECEIPT_SCHEMA",
    "FaroGeometry",
    "HIGHRES_SHAPE_HW",
    "ORACLE_MODES",
    "QUERY_FACTOR_FRAME_SCHEMA",
    "QUERY_RECEIPT_SCHEMA",
    "REDUCER_VERSION",
    "bootstrap_support_uncertainty",
    "build_candidate_depth_output_receipt",
    "build_query_receipts",
    "build_candidate_query_factor_frame",
    "build_source_frame_receipt",
    "build_truth_query_factor_frame",
    "build_truth_query_factor_frames",
    "canonical_json_bytes",
    "canonical_sha256",
    "decimal_timestamp_ns",
    "derive_faro_geometry",
    "fit_uncertainty_model",
    "inject_factor_blocks",
    "interpolate_camera_to_world_exact",
    "reduce_query_factor_frame",
    "reduce_complete_query_bundle",
    "reject_truth_shortcuts",
    "sample_faro_at_apple_centers",
    "scale_lowres_intrinsics",
    "validate_query_factor_frame",
]
