"""Fail-closed LILocBench calibration and depth-registration primitives.

All transforms use the explicit ``parent_T_child`` convention: a point in the
child frame is mapped into the parent frame.  The official file named
``extrinsics_depth_to_color.yaml`` declares ``parent=color`` and
``child=depth`` and is therefore consumed as ``color_T_depth`` without an
extra inversion.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contract import validate_pose


def _scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError("expected YAML inline list")
        return parsed
    if (text.startswith("\"") and text.endswith("\"")) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _frame_name(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid LILocBench {field} frame")
    result = value.strip()
    if not result or result.lower() in {"none", "null", "~"}:
        raise ValueError(f"invalid LILocBench {field} frame")
    return result


def parse_intrinsics_yaml(path: Path) -> dict[str, Any]:
    """Parse the small ROS-style intrinsics mapping used by LILocBench."""
    result: dict[str, Any] = {}
    active_list: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            if active_list is None or not isinstance(result.get(active_list), list):
                raise ValueError(f"unexpected LILocBench intrinsics list item: {line}")
            value = _scalar(line[1:])
            if not isinstance(value, (int, float)):
                raise ValueError(f"invalid LILocBench {active_list} list item")
            result[active_list].append(value)
            continue
        if ":" not in line:
            raise ValueError(f"invalid LILocBench intrinsics line: {line}")
        key, value = line.split(":", 1)
        if key in result:
            raise ValueError(f"duplicate LILocBench intrinsics key: {key}")
        parsed = _scalar(value)
        if parsed is None:
            if key not in {"distortion_coefficients", "K", "R", "P"}:
                raise ValueError(f"unexpected empty LILocBench intrinsics field: {key}")
            result[key] = []
            active_list = key
        else:
            result[key] = parsed
            active_list = None
    expected_lengths = {
        "distortion_coefficients": 5,
        "K": 9,
        "R": 9,
        "P": 12,
    }
    for key, length in expected_lengths.items():
        value = result.get(key)
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"invalid LILocBench {key}; expected {length} values")
        if not all(math.isfinite(float(item)) for item in value):
            raise ValueError(f"non-finite LILocBench {key}")
    if int(result.get("width", 0)) <= 0 or int(result.get("height", 0)) <= 0:
        raise ValueError("invalid LILocBench image dimensions")
    if result.get("distortion_model") != "plumb_bob":
        raise ValueError("unsupported LILocBench distortion model")
    camera_matrix = np.asarray(result["K"], dtype=np.float64).reshape(3, 3)
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or not np.isclose(camera_matrix[2, 2], 1.0):
        raise ValueError("invalid LILocBench camera matrix")
    return result


def parse_transformations_yaml(path: Path) -> list[dict[str, Any]]:
    """Parse the official top-level transform list with explicit directions."""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- parent:"):
            if current is not None:
                entries.append(current)
            current = {"parent": _frame_name(_scalar(line.split(":", 1)[1]), "parent"), "translation": {}, "rotation": {}}
            section = None
            continue
        if current is None:
            raise ValueError("LILocBench transformations must be a top-level list")
        if line in {"translation:", "rotation:"}:
            section = line[:-1]
            continue
        if ":" not in line:
            raise ValueError(f"invalid LILocBench transform line: {line}")
        key, value = line.split(":", 1)
        parsed = _scalar(value)
        if section is None:
            if key != "child" or "child" in current:
                raise ValueError(f"unexpected LILocBench transform field: {key}")
            current["child"] = _frame_name(parsed, "child")
        else:
            target = current[section]
            if key in target or key not in {"x", "y", "z", "w"}:
                raise ValueError(f"invalid LILocBench {section} field: {key}")
            target[key] = float(parsed)
    if current is not None:
        entries.append(current)
    if not entries:
        raise ValueError("empty LILocBench transformations")
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.get("parent"), entry.get("child"))
        if not all(isinstance(value, str) and value for value in key) or key in seen:
            raise ValueError("invalid or duplicate LILocBench transform identity")
        seen.add(key)
        if set(entry["translation"]) != {"x", "y", "z"}:
            raise ValueError(f"incomplete translation for {key}")
        if set(entry["rotation"]) != {"x", "y", "z", "w"}:
            raise ValueError(f"incomplete rotation for {key}")
    return entries


def parse_depth_to_color_yaml(path: Path) -> dict[str, Any]:
    """Parse the official single ``color_T_depth`` mapping and pin direction."""
    entry: dict[str, Any] = {"translation": {}, "rotation": {}}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in {"translation:", "rotation:"}:
            section = line[:-1]
            continue
        if ":" not in line:
            raise ValueError(f"invalid LILocBench depth-to-color line: {line}")
        key, value = line.split(":", 1)
        parsed = _scalar(value)
        if section is None:
            if key not in {"parent", "child"} or key in entry:
                raise ValueError(f"unexpected LILocBench depth-to-color field: {key}")
            entry[key] = _frame_name(parsed, key)
        else:
            target = entry[section]
            if key in target or key not in {"x", "y", "z", "w"}:
                raise ValueError(f"invalid LILocBench {section} field: {key}")
            target[key] = float(parsed)
    if entry.get("parent") != "color" or entry.get("child") != "depth":
        raise ValueError("LILocBench depth-to-color direction must be parent=color child=depth")
    if set(entry["translation"]) != {"x", "y", "z"}:
        raise ValueError("incomplete LILocBench depth-to-color translation")
    if set(entry["rotation"]) != {"x", "y", "z", "w"}:
        raise ValueError("incomplete LILocBench depth-to-color rotation")
    transform_matrix(entry)
    return entry


def transform_matrix(entry: dict[str, Any]) -> np.ndarray:
    translation = entry["translation"]
    rotation = entry["rotation"]
    translation_values = np.asarray([float(translation[key]) for key in ("x", "y", "z")], dtype=np.float64)
    if not np.all(np.isfinite(translation_values)):
        raise ValueError("non-finite LILocBench transform translation")
    x, y, z, w = (float(rotation[key]) for key in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError("invalid LILocBench transform quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), translation_values[0]],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), translation_values[1]],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), translation_values[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    validate_pose(matrix.tolist())
    return matrix


def compose_transform_chain(entries: list[dict[str, Any]], frames: list[str]) -> np.ndarray:
    """Compose ``frames[0]_T_frames[-1]`` from direct parent-child edges."""
    if len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError("invalid LILocBench transform chain")
    by_edge: dict[tuple[str, str], np.ndarray] = {}
    for entry in entries:
        edge_key = (_frame_name(entry.get("parent"), "parent"), _frame_name(entry.get("child"), "child"))
        if edge_key in by_edge:
            raise ValueError(f"duplicate LILocBench transform edge: {edge_key[0]}->{edge_key[1]}")
        by_edge[edge_key] = transform_matrix(entry)
    result = np.eye(4, dtype=np.float64)
    for parent, child in zip(frames, frames[1:]):
        edge = by_edge.get((parent, child))
        if edge is None:
            raise ValueError(f"missing LILocBench transform edge: {parent}->{child}")
        result = result @ edge
    validate_pose(result.tolist())
    return result


def validate_front_color_optical(base_from_color_optical: np.ndarray, minimum_forward_x: float = 0.95) -> np.ndarray:
    validate_pose(base_from_color_optical.tolist())
    if not np.all(np.isfinite(base_from_color_optical)) or not math.isfinite(minimum_forward_x):
        raise ValueError("non-finite LILocBench forward-camera contract")
    optical_forward_in_base = base_from_color_optical[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    if (
        float(optical_forward_in_base[0]) < minimum_forward_x
        or abs(float(optical_forward_in_base[1])) >= float(optical_forward_in_base[0])
        or abs(float(optical_forward_in_base[2])) >= float(optical_forward_in_base[0])
    ):
        raise ValueError("selected LILocBench camera is not forward-facing in base_link")
    return optical_forward_in_base


def world_from_color_optical(
    world_from_base: np.ndarray,
    base_from_color_optical: np.ndarray,
) -> np.ndarray:
    """Return ``world_T_color_optical`` from explicitly directed inputs."""
    validate_pose(world_from_base.tolist())
    validate_pose(base_from_color_optical.tolist())
    world_from_camera = world_from_base @ base_from_color_optical
    validate_pose(world_from_camera.tolist())
    return world_from_camera


def calibration_arrays(calibration: dict[str, Any], role: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Validate the official current intrinsics contract used for registration."""
    if role not in {"depth", "color"}:
        raise ValueError("invalid LILocBench calibration role")
    width, height = int(calibration.get("width", 0)), int(calibration.get("height", 0))
    camera_matrix = np.asarray(calibration.get("K"), dtype=np.float64).reshape(3, 3)
    rectification = np.asarray(calibration.get("R"), dtype=np.float64).reshape(3, 3)
    projection = np.asarray(calibration.get("P"), dtype=np.float64).reshape(3, 4)
    distortion = np.asarray(calibration.get("distortion_coefficients"), dtype=np.float64).reshape(-1)
    if (
        width <= 0
        or height <= 0
        or distortion.shape != (5,)
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(rectification))
        or not np.all(np.isfinite(projection))
        or not np.all(np.isfinite(distortion))
    ):
        raise ValueError(f"invalid LILocBench {role} calibration values")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or not np.allclose(camera_matrix[2], [0, 0, 1], atol=1e-12):
        raise ValueError(f"invalid LILocBench {role} camera matrix")
    if not np.allclose(rectification, np.eye(3), atol=1e-9):
        raise ValueError(f"unsupported non-identity LILocBench {role} rectification")
    if not np.allclose(projection[:, :3], camera_matrix, atol=1e-9) or not np.allclose(projection[:, 3], 0.0, atol=1e-12):
        raise ValueError(f"unsupported LILocBench {role} projection drift")
    if int(calibration.get("binning_x", 0)) != 0 or int(calibration.get("binning_y", 0)) != 0:
        raise ValueError(f"unsupported LILocBench {role} binning")
    if role == "depth" and not np.allclose(distortion, 0.0, atol=1e-12):
        raise ValueError("unsupported distorted LILocBench depth raster")
    return camera_matrix, distortion, (height, width)


def register_depth_to_color(
    depth_raw: np.ndarray,
    depth_scale_units_per_meter: float,
    depth_calibration: dict[str, Any],
    color_calibration: dict[str, Any],
    color_from_depth: np.ndarray,
) -> np.ndarray:
    """Project raw depth into the color optical frame using a nearest-z buffer.

    The result is a float32 metric-depth raster in the color image geometry;
    zero means unobserved.  This function does not fill holes or invent depth.
    """
    if (
        depth_raw.ndim != 2
        or not math.isfinite(depth_scale_units_per_meter)
        or depth_scale_units_per_meter <= 0
        or not np.all(np.isfinite(depth_raw))
    ):
        raise ValueError("invalid LILocBench depth raster or scale")
    depth_camera_matrix, _, depth_shape = calibration_arrays(depth_calibration, "depth")
    color_camera_matrix, distortion, color_shape = calibration_arrays(color_calibration, "color")
    if tuple(depth_raw.shape) != depth_shape:
        raise ValueError("LILocBench depth raster shape does not match calibration")
    transform = np.asarray(color_from_depth, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("non-finite LILocBench color-from-depth transform")
    validate_pose(transform.tolist())

    depth_m = np.asarray(depth_raw, dtype=np.float64) / float(depth_scale_units_per_meter)
    if not np.all(np.isfinite(depth_m)):
        raise ValueError("non-finite LILocBench metric depth")
    rows, columns = np.nonzero(depth_m > 0.0)
    if rows.size == 0:
        return np.zeros(color_shape, dtype=np.float32)
    z = depth_m[rows, columns]
    fx_d, fy_d = float(depth_camera_matrix[0, 0]), float(depth_camera_matrix[1, 1])
    cx_d, cy_d = float(depth_camera_matrix[0, 2]), float(depth_camera_matrix[1, 2])
    points_depth = np.column_stack(
        ((columns - cx_d) * z / fx_d, (rows - cy_d) * z / fy_d, z)
    )
    points_color = (transform[:3, :3] @ points_depth.T).T + transform[:3, 3]
    if not np.all(np.isfinite(points_color)):
        raise ValueError("non-finite LILocBench projected 3D point")
    valid_z = points_color[:, 2] > 0.0
    if not np.any(valid_z):
        return np.zeros(color_shape, dtype=np.float32)
    points_color = points_color[valid_z]
    pixels, _ = cv2.projectPoints(
        points_color.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(color_camera_matrix, dtype=np.float64),
        distortion,
    )
    if not np.all(np.isfinite(pixels)):
        raise ValueError("non-finite LILocBench projected color pixel")
    pixels = np.rint(pixels.reshape(-1, 2)).astype(np.int64)
    height, width = color_shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    if not np.any(inside):
        return np.zeros(color_shape, dtype=np.float32)
    pixels = pixels[inside]
    projected_z = points_color[inside, 2]
    flat_indices = pixels[:, 1] * width + pixels[:, 0]
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat_indices, projected_z)
    z_buffer[~np.isfinite(z_buffer)] = 0.0
    return z_buffer.reshape(height, width).astype(np.float32)
