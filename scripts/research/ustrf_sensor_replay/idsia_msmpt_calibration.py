from __future__ import annotations

import math
from typing import Any

import numpy as np

from contract import validate_pose


def calibration_arrays(calibration: dict[str, Any], role: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if role not in {"depth", "color"}:
        raise ValueError("invalid IDSIA calibration role")
    width, height = int(calibration.get("width", 0)), int(calibration.get("height", 0))
    camera_matrix = np.asarray(calibration.get("K"), dtype=np.float64).reshape(3, 3)
    rectification = np.asarray(calibration.get("R"), dtype=np.float64).reshape(3, 3)
    projection = np.asarray(calibration.get("P"), dtype=np.float64).reshape(3, 4)
    distortion = np.asarray(calibration.get("D"), dtype=np.float64).reshape(-1)
    if (
        width <= 0
        or height <= 0
        or distortion.shape != (5,)
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(rectification))
        or not np.all(np.isfinite(projection))
        or not np.all(np.isfinite(distortion))
    ):
        raise ValueError(f"invalid IDSIA {role} calibration values")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or not np.allclose(camera_matrix[2], [0, 0, 1], atol=1e-12):
        raise ValueError(f"invalid IDSIA {role} camera matrix")
    if not np.allclose(rectification, np.eye(3), atol=1e-9):
        raise ValueError(f"unsupported IDSIA {role} rectification")
    if not np.allclose(projection[:, :3], camera_matrix, atol=1e-9) or not np.allclose(projection[:, 3], 0.0, atol=1e-12):
        raise ValueError(f"unsupported IDSIA {role} projection drift")
    if int(calibration.get("binning_x", 0)) != 0 or int(calibration.get("binning_y", 0)) != 0:
        raise ValueError(f"unsupported IDSIA {role} binning")
    if calibration.get("distortion_model") != "plumb_bob":
        raise ValueError(f"unsupported IDSIA {role} distortion model")
    if role == "depth" and not np.allclose(distortion, 0.0, atol=1e-12):
        raise ValueError("unsupported distorted IDSIA depth raster")
    return camera_matrix, distortion, (height, width)


def register_depth_to_color(
    depth_raw: np.ndarray,
    depth_scale_units_per_meter: float,
    depth_calibration: dict[str, Any],
    color_calibration: dict[str, Any],
    color_from_depth: np.ndarray,
) -> np.ndarray:
    """Register rectified depth to the distorted raw color raster without filling holes."""
    if (
        depth_raw.dtype != np.uint16
        or depth_raw.ndim != 2
        or not math.isfinite(depth_scale_units_per_meter)
        or depth_scale_units_per_meter <= 0
    ):
        raise ValueError("invalid IDSIA depth raster or scale")
    depth_camera, _, depth_shape = calibration_arrays(depth_calibration, "depth")
    color_camera, distortion, color_shape = calibration_arrays(color_calibration, "color")
    if tuple(depth_raw.shape) != depth_shape:
        raise ValueError("IDSIA depth raster shape does not match calibration")
    transform = np.asarray(color_from_depth, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("invalid IDSIA color-from-depth transform")
    validate_pose(transform.tolist())

    depth_m = depth_raw.astype(np.float64) / float(depth_scale_units_per_meter)
    rows, columns = np.nonzero(depth_m > 0.0)
    if rows.size == 0:
        return np.zeros(color_shape, dtype=np.float32)
    z = depth_m[rows, columns]
    fx_d, fy_d = float(depth_camera[0, 0]), float(depth_camera[1, 1])
    cx_d, cy_d = float(depth_camera[0, 2]), float(depth_camera[1, 2])
    points_depth = np.column_stack(((columns - cx_d) * z / fx_d, (rows - cy_d) * z / fy_d, z))
    points_color = (transform[:3, :3] @ points_depth.T).T + transform[:3, 3]
    valid_z = np.isfinite(points_color).all(axis=1) & (points_color[:, 2] > 0.0)
    if not np.any(valid_z):
        return np.zeros(color_shape, dtype=np.float32)
    points_color = points_color[valid_z]

    x = points_color[:, 0] / points_color[:, 2]
    y = points_color[:, 1] / points_color[:, 2]
    k1, k2, p1, p2, k3 = distortion
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    pixels_x = np.rint(float(color_camera[0, 0]) * x_distorted + float(color_camera[0, 2])).astype(np.int64)
    pixels_y = np.rint(float(color_camera[1, 1]) * y_distorted + float(color_camera[1, 2])).astype(np.int64)
    height, width = color_shape
    inside = (pixels_x >= 0) & (pixels_x < width) & (pixels_y >= 0) & (pixels_y < height)
    if not np.any(inside):
        return np.zeros(color_shape, dtype=np.float32)
    flat_indices = pixels_y[inside] * width + pixels_x[inside]
    projected_z = points_color[inside, 2]
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat_indices, projected_z)
    z_buffer[~np.isfinite(z_buffer)] = 0.0
    return z_buffer.reshape(height, width).astype(np.float32)
