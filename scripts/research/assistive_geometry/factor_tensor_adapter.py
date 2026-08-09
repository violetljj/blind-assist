#!/usr/bin/env python3
"""Zero-parameter F1 FactorTensor to byte-frozen F0 factor-frame adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from typing import Any, Iterable


INPUT_SCHEMA = "blindassist_assistive_geometry_r2_factortensor_adapter_input_v1"
PREDICTION_SCHEMA = "blindassist_assistive_geometry_r2_factor_prediction_v1"
GEOMETRY_SCHEMA = "blindassist_assistive_geometry_r2_adapter_geometry_receipt_v1"
CALIBRATION_SCHEMA = "blindassist_assistive_geometry_r2_adapter_calibration_receipt_v1"
OUTPUT_SCHEMA = "blindassist_assistive_geometry_r2_factor_frame_v1"
FACTOR_SCHEMA_SHA256 = "8016430D639EC78199432F55ABB8EBDC847A4073C24F84A17E429A07D1BB5F7E"
PROBABILITY_THRESHOLD = 0.5
FLOAT_DECIMALS = 12

FORBIDDEN_KEYS = {
    "clearance",
    "clearance_m",
    "direct_clearance",
    "occupancy",
    "occupancy_logit",
    "free",
    "blocked",
    "risk",
    "risk_score",
    "task_confidence",
    "final_state",
    "unknown_logit",
    "ttc",
    "future_clearance",
}

TOP_LEVEL_KEYS = {"prediction", "geometry_receipt", "calibration_receipt"}
PREDICTION_KEYS = {
    "schema",
    "sample_id",
    "factor_identity",
    "camera_geometry_receipt_sha256",
    "depth_scale",
    "support_surface",
    "obstacle_boundary_evidence",
}
DEPTH_KEYS = {
    "depth_shape_positive_hw",
    "log_metric_scale_m_scalar",
    "depth_log_sigma_hw",
    "depth_valid_probability_hw",
    "metric_scale_valid",
}
SUPPORT_KEYS = {
    "support_probability_hw",
    "support_plane_normal_camera_xyz",
    "camera_height_m",
    "support_residual_sigma_m",
    "support_valid",
}
EVIDENCE_KEYS = {
    "obstacle_evidence_probability_hw",
    "boundary_probability_hw",
    "boundary_localization_sigma_px_hw",
    "evidence_valid_hw",
}
GEOMETRY_KEYS = {
    "schema",
    "frame_id",
    "sample_id",
    "content_sha256",
    "tensor_hw",
    "orientation",
    "k_display_upright",
    "k_valid",
    "transform_valid",
    "gravity_valid",
    "gravity_up_camera",
}
CALIBRATION_KEYS = {
    "schema",
    "calibration_id",
    "factor_schema_sha256",
    "source_role",
    "task_outcome_used",
    "scale_relative_sigma_floor",
    "scale_relative_sigma_cap",
    "support_normal_sigma_rad",
    "support_height_sigma_m",
    "boundary_sigma_floor_px",
    "evidence_sigma_floor",
}


class AdapterError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise AdapterError(code, message)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _exact_keys(value: Any, expected: set[str], code: str) -> dict[str, Any]:
    require(isinstance(value, dict), code, "value must be an object")
    require(set(value) == expected, code, "object key set drift")
    return value


def _scan_forbidden(value: Any, path: str = "input") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            require(normalized not in FORBIDDEN_KEYS, "FORBIDDEN_FINAL_TASK_FIELD", f"forbidden field at {path}.{key}")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _matrix(value: Any, hw: tuple[int, int], code: str, *, boolean: bool = False, probability: bool = False, positive: bool = False) -> list[list[Any]]:
    height, width = hw
    require(isinstance(value, list) and len(value) == height, code, "matrix height drift")
    output: list[list[Any]] = []
    for row in value:
        require(isinstance(row, list) and len(row) == width, code, "matrix width drift")
        checked: list[Any] = []
        for item in row:
            if boolean:
                require(isinstance(item, bool), code, "boolean matrix required")
                checked.append(item)
            else:
                require(finite(item), code, "finite matrix required")
                number = float(item)
                require(not probability or 0.0 <= number <= 1.0, code, "probability out of range")
                require(not positive or number > 0.0, code, "positive matrix required")
                checked.append(number)
        output.append(checked)
    return output


def quantile_type7(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "EMPTY_QUANTILE", "quantile input is empty")
    require(0.0 <= probability <= 1.0, "QUANTILE_PROBABILITY", "quantile probability out of range")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median_absolute_deviation(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    center = quantile_type7(items, 0.5)
    return quantile_type7((abs(value - center) for value in items), 0.5)


def _components(mask: list[list[bool]]) -> list[list[tuple[int, int]]]:
    height = len(mask)
    width = len(mask[0])
    seen: set[tuple[int, int]] = set()
    output: list[list[tuple[int, int]]] = []
    for row in range(height):
        for column in range(width):
            if not mask[row][column] or (row, column) in seen:
                continue
            queue: deque[tuple[int, int]] = deque([(row, column)])
            seen.add((row, column))
            component: list[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        if delta_row == 0 and delta_column == 0:
                            continue
                        neighbor = (current[0] + delta_row, current[1] + delta_column)
                        if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and mask[neighbor[0]][neighbor[1]] and neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
            output.append(sorted(component))
    return output


def _round_canonical(value: Any) -> Any:
    if isinstance(value, float):
        rounded = round(value, FLOAT_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {key: _round_canonical(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_round_canonical(child) for child in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(_round_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def _invalid_frame(frame_id: str, factor_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA,
        "frame_id": frame_id or "INVALID_FRAME_ID",
        "factor_identity": factor_identity,
        "input_geometry": {
            "k_valid": False,
            "transform_valid": False,
            "gravity_valid": False,
            "gravity_up_camera": [0.0, 1.0, 0.0],
            "orientation": "landscape",
        },
        "depth_scale": {"valid": False, "scale_m": 1.0, "scale_sigma_m": 0.0},
        "support": {
            "valid": False,
            "normal_camera": [0.0, 1.0, 0.0],
            "normal_sigma_rad": 0.0,
            "camera_height_m": 1.0,
            "height_sigma_m": 0.0,
            "residual_sigma_m": 0.0,
        },
        "boundary": {"valid": False, "coverage": 0.0, "obstacles": []},
    }


def adapt_factor_tensor(adapter_input: dict[str, Any]) -> dict[str, Any]:
    """Adapt one complete factor tensor into one F0 factor frame."""

    _scan_forbidden(adapter_input)
    root = _exact_keys(adapter_input, TOP_LEVEL_KEYS, "INPUT_KEY_SET")
    prediction = _exact_keys(root["prediction"], PREDICTION_KEYS, "PREDICTION_KEY_SET")
    geometry = _exact_keys(root["geometry_receipt"], GEOMETRY_KEYS, "GEOMETRY_KEY_SET")
    calibration = _exact_keys(root["calibration_receipt"], CALIBRATION_KEYS, "CALIBRATION_KEY_SET")
    depth = _exact_keys(prediction["depth_scale"], DEPTH_KEYS, "DEPTH_KEY_SET")
    support = _exact_keys(prediction["support_surface"], SUPPORT_KEYS, "SUPPORT_KEY_SET")
    evidence = _exact_keys(prediction["obstacle_boundary_evidence"], EVIDENCE_KEYS, "EVIDENCE_KEY_SET")
    require(prediction.get("schema") == PREDICTION_SCHEMA, "PREDICTION_SCHEMA", "prediction schema drift")
    require(geometry.get("schema") == GEOMETRY_SCHEMA, "GEOMETRY_SCHEMA", "geometry schema drift")
    require(calibration.get("schema") == CALIBRATION_SCHEMA, "CALIBRATION_SCHEMA", "calibration schema drift")
    factor_identity = prediction.get("factor_identity")
    require(isinstance(factor_identity, dict), "FACTOR_IDENTITY", "factor identity required")
    require(factor_identity.get("learned_final_task_head") is False, "FACTOR_IDENTITY", "final task head forbidden")
    frame_id = str(geometry.get("frame_id", ""))

    tensor_hw_raw = geometry.get("tensor_hw")
    receipt_valid = (
        prediction.get("sample_id") == geometry.get("sample_id")
        and _sha256_text(prediction.get("camera_geometry_receipt_sha256"))
        and prediction.get("camera_geometry_receipt_sha256") == geometry.get("content_sha256")
        and calibration.get("factor_schema_sha256") == FACTOR_SCHEMA_SHA256
        and calibration.get("task_outcome_used") is False
        and calibration.get("source_role") in {"SYNTHETIC_CANARY", "FIT_ONLY_CALIBRATION"}
        and isinstance(tensor_hw_raw, list)
        and len(tensor_hw_raw) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in tensor_hw_raw)
        and geometry.get("orientation") in {"portrait", "landscape"}
        and geometry.get("k_valid") is True
        and geometry.get("transform_valid") is True
    )
    if not receipt_valid:
        return _round_canonical(_invalid_frame(frame_id, dict(factor_identity)))
    hw = int(tensor_hw_raw[0]), int(tensor_hw_raw[1])
    k = geometry.get("k_display_upright")
    if not isinstance(k, dict) or set(k) != {"fx", "fy", "cx", "cy"} or not all(finite(k.get(field)) for field in k):
        return _round_canonical(_invalid_frame(frame_id, dict(factor_identity)))
    fx, fy, cx, cy = (float(k[field]) for field in ("fx", "fy", "cx", "cy"))
    if not (fx > 0.0 and fy > 0.0 and 0.0 <= cx < hw[1] and 0.0 <= cy < hw[0]):
        return _round_canonical(_invalid_frame(frame_id, dict(factor_identity)))
    gravity = geometry.get("gravity_up_camera")
    gravity_valid = geometry.get("gravity_valid") is True
    if not isinstance(gravity, list) or len(gravity) != 3 or not all(finite(value) for value in gravity):
        return _round_canonical(_invalid_frame(frame_id, dict(factor_identity)))
    gravity_values = [float(value) for value in gravity]
    if gravity_valid and abs(math.sqrt(sum(value * value for value in gravity_values)) - 1.0) > 1e-6:
        return _round_canonical(_invalid_frame(frame_id, dict(factor_identity)))

    shape = _matrix(depth["depth_shape_positive_hw"], hw, "DEPTH_SHAPE", positive=True)
    log_sigma = _matrix(depth["depth_log_sigma_hw"], hw, "DEPTH_LOG_SIGMA")
    depth_probability = _matrix(depth["depth_valid_probability_hw"], hw, "DEPTH_VALIDITY", probability=True)
    support_probability = _matrix(support["support_probability_hw"], hw, "SUPPORT_PROBABILITY", probability=True)
    obstacle_probability = _matrix(evidence["obstacle_evidence_probability_hw"], hw, "OBSTACLE_PROBABILITY", probability=True)
    boundary_probability = _matrix(evidence["boundary_probability_hw"], hw, "BOUNDARY_PROBABILITY", probability=True)
    boundary_sigma_px = _matrix(evidence["boundary_localization_sigma_px_hw"], hw, "BOUNDARY_SIGMA", positive=True)
    evidence_valid = _matrix(evidence["evidence_valid_hw"], hw, "EVIDENCE_VALID", boolean=True)

    scale_valid = depth.get("metric_scale_valid") is True and finite(depth.get("log_metric_scale_m_scalar"))
    scale_m = math.exp(float(depth["log_metric_scale_m_scalar"])) if scale_valid else 1.0
    scale_valid = scale_valid and math.isfinite(scale_m) and scale_m > 0.0
    calibration_numbers = (
        "scale_relative_sigma_floor",
        "scale_relative_sigma_cap",
        "support_normal_sigma_rad",
        "support_height_sigma_m",
        "boundary_sigma_floor_px",
        "evidence_sigma_floor",
    )
    calibration_valid = all(finite(calibration.get(field)) and float(calibration[field]) >= 0.0 for field in calibration_numbers)
    calibration_valid = calibration_valid and float(calibration["scale_relative_sigma_cap"]) >= float(calibration["scale_relative_sigma_floor"]) and float(calibration["evidence_sigma_floor"]) <= 1.0
    metric_depth = [[shape[row][column] * scale_m for column in range(hw[1])] for row in range(hw[0])]
    relative_sigma_values = [
        math.exp(log_sigma[row][column]) / metric_depth[row][column]
        for row in range(hw[0])
        for column in range(hw[1])
        if depth_probability[row][column] >= PROBABILITY_THRESHOLD
    ]
    if not relative_sigma_values or not calibration_valid or not scale_valid or not all(math.isfinite(value) and value >= 0.0 for value in relative_sigma_values):
        scale_valid = False
        scale_sigma_m = 0.0
    else:
        relative = quantile_type7(relative_sigma_values, 0.5)
        relative = max(float(calibration["scale_relative_sigma_floor"]), min(float(calibration["scale_relative_sigma_cap"]), relative))
        scale_sigma_m = scale_m * relative

    normal = support.get("support_plane_normal_camera_xyz")
    support_valid = support.get("support_valid") is True and calibration_valid and any(value >= PROBABILITY_THRESHOLD for row in support_probability for value in row)
    support_valid = support_valid and isinstance(normal, list) and len(normal) == 3 and all(finite(value) for value in normal)
    normal_values = [float(value) for value in normal] if isinstance(normal, list) and len(normal) == 3 and all(finite(value) for value in normal) else [0.0, 1.0, 0.0]
    support_valid = support_valid and abs(math.sqrt(sum(value * value for value in normal_values)) - 1.0) <= 1e-6
    support_valid = support_valid and finite(support.get("camera_height_m")) and float(support["camera_height_m"]) > 0.0
    support_valid = support_valid and finite(support.get("support_residual_sigma_m")) and float(support["support_residual_sigma_m"]) >= 0.0

    candidate_mask = [
        [bool(evidence_valid[row][column] and obstacle_probability[row][column] >= PROBABILITY_THRESHOLD) for column in range(hw[1])]
        for row in range(hw[0])
    ]
    coverage = sum(1 for row in range(hw[0]) for column in range(hw[1]) if evidence_valid[row][column]) / float(hw[0] * hw[1])
    obstacles_with_keys: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for component in _components(candidate_mask):
        depth_is_valid = scale_valid and all(depth_probability[row][column] >= PROBABILITY_THRESHOLD for row, column in component)
        component_shape = [shape[row][column] for row, column in component]
        component_shape_sigma = [math.exp(log_sigma[row][column]) / scale_m for row, column in component]
        forward = quantile_type7(component_shape, 0.1)
        forward_sigma = quantile_type7(component_shape_sigma, 0.9)
        lateral_values: list[float] = []
        boundary_metric: list[float] = []
        for row, column in component:
            depth_m = metric_depth[row][column]
            lateral_values.extend(((column - 0.5 - cx) * depth_m / fx, (column + 0.5 - cx) * depth_m / fx))
            boundary_metric.append(max(boundary_sigma_px[row][column], float(calibration["boundary_sigma_floor_px"])) * depth_m / fx)
        lateral_lo, lateral_hi = min(lateral_values), max(lateral_values)
        obstacle_values = [obstacle_probability[row][column] for row, column in component]
        boundary_values = [boundary_probability[row][column] for row, column in component]
        probability = min(quantile_type7(obstacle_values, 0.5), quantile_type7(boundary_values, 0.5))
        probability_sigma = max(
            float(calibration["evidence_sigma_floor"]),
            1.4826 * median_absolute_deviation(obstacle_values),
            1.4826 * median_absolute_deviation(boundary_values),
        )
        anchor = min(component)
        obstacle = {
            "kind": "dense_component",
            "depth_valid": bool(depth_is_valid),
            "depth_shape_forward": forward,
            "depth_shape_sigma": forward_sigma,
            "lateral_center_m": (lateral_lo + lateral_hi) / 2.0,
            "lateral_half_width_m": (lateral_hi - lateral_lo) / 2.0,
            "boundary_sigma_m": quantile_type7(boundary_metric, 0.9),
            "evidence_probability": probability,
            "evidence_sigma": min(1.0, probability_sigma),
        }
        key = (not depth_is_valid, forward, obstacle["lateral_center_m"], anchor[0], anchor[1])
        obstacles_with_keys.append((key, obstacle))
    obstacles_with_keys.sort(key=lambda item: item[0])
    require(len({key for key, _ in obstacles_with_keys}) == len(obstacles_with_keys), "NONUNIQUE_COMPONENT_KEY", "canonical component key collision")

    frame = {
        "schema": OUTPUT_SCHEMA,
        "frame_id": frame_id,
        "factor_identity": dict(factor_identity),
        "input_geometry": {
            "k_valid": True,
            "transform_valid": True,
            "gravity_valid": gravity_valid,
            "gravity_up_camera": gravity_values,
            "orientation": geometry["orientation"],
        },
        "depth_scale": {"valid": bool(scale_valid), "scale_m": scale_m, "scale_sigma_m": scale_sigma_m},
        "support": {
            "valid": bool(support_valid),
            "normal_camera": normal_values,
            "normal_sigma_rad": float(calibration["support_normal_sigma_rad"]) if calibration_valid else 0.0,
            "camera_height_m": float(support["camera_height_m"]) if finite(support.get("camera_height_m")) else 1.0,
            "height_sigma_m": float(calibration["support_height_sigma_m"]) if calibration_valid else 0.0,
            "residual_sigma_m": float(support["support_residual_sigma_m"]) if finite(support.get("support_residual_sigma_m")) else 0.0,
        },
        "boundary": {
            "valid": True,
            "coverage": coverage,
            "obstacles": [obstacle for _, obstacle in obstacles_with_keys],
        },
    }
    return _round_canonical(frame)
