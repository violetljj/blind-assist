"""Response-blind quality interventions and image metrics for R2 P2.

This module is intentionally independent of every RCLE estimator, response,
trigger, support, feature-collapse, and forward/backward-flow implementation.
It operates only on generator-native RGB, geometry masks, and source-known
calibration edges.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import cv2
import numpy as np

from . import generator_geometry as p1


PLATE_OBJECT_ID = 29
PLATE_ROWS = 4
PLATE_COLUMNS = 8
PLATE_REFERENCE_BOUNDS_PX = (24.0, 336.0, 52.0, 588.0)
PLATE_DEPTH_M = 3.0
EDGE_SAMPLE_OFFSETS_PX = np.arange(-8.0, 8.0001, 0.25, dtype=np.float64)
LAPLACIAN_KERNEL = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
SOBEL_SCALES = (0.0, 1.0, 2.0)


class InvalidQualityMetric(ValueError):
    """Raised when a frozen response-blind metric is not signable."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def average_rank_median(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise InvalidQualityMetric("EMPTY_OR_NONFINITE_MEDIAN_INPUT")
    result = float(np.median(array))
    if not math.isfinite(result):
        raise InvalidQualityMetric("NONFINITE_MEDIAN")
    return result


def srgb_u8_to_linear(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB_MUST_BE_HXWX3_UINT8")
    channel = rgb.astype(np.float64) / 255.0
    linear = np.where(
        channel <= 0.04045,
        channel / 12.92,
        np.power((channel + 0.055) / 1.055, 2.4),
    )
    if not np.all(np.isfinite(linear)):
        raise InvalidQualityMetric("NONFINITE_LINEAR_RGB")
    return linear


def linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError("LINEAR_RGB_MUST_BE_HXWX3")
    if not np.all(np.isfinite(linear)):
        raise InvalidQualityMetric("NONFINITE_LINEAR_RGB")
    clipped = np.clip(linear.astype(np.float64, copy=False), 0.0, 1.0)
    srgb = np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_u8_to_linear(rgb)
    return (
        0.2126 * linear[:, :, 0]
        + 0.7152 * linear[:, :, 1]
        + 0.0722 * linear[:, :, 2]
    )


def apply_blur(rgb: np.ndarray, sigma_px: float) -> np.ndarray:
    sigma = float(sigma_px)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("BLUR_SIGMA_MUST_BE_FINITE_POSITIVE")
    kernel = 2 * math.ceil(3.0 * sigma) + 1
    linear = srgb_u8_to_linear(rgb)
    blurred = cv2.GaussianBlur(
        linear,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return linear_to_srgb_u8(blurred)


def _world_from_reference_pixel(u: float, v: float, z: float) -> list[float]:
    return [
        float((u - p1.K[0, 2]) / p1.K[0, 0] * z),
        float((v - p1.K[1, 2]) / p1.K[1, 1] * z),
        float(z),
    ]


def add_calibration_plate(scene: dict[str, Any]) -> dict[str, Any]:
    """Add the source-known 8x4 single-step plate to a CAL-only scene."""

    if scene.get("namespace") != "CAL":
        raise ValueError("CALIBRATION_PLATE_REQUIRES_CAL_NAMESPACE")
    derived = copy.deepcopy(scene)
    if any(
        int(item["object_id"]) == PLATE_OBJECT_ID
        for item in derived["world"]["objects"]
    ):
        raise ValueError("CALIBRATION_PLATE_OBJECT_ID_COLLISION")
    u0, u1, v0, v1 = PLATE_REFERENCE_BOUNDS_PX
    low = _world_from_reference_pixel(u0, v0, PLATE_DEPTH_M)
    high = _world_from_reference_pixel(u1, v1, PLATE_DEPTH_M)
    plate = {
        "object_id": PLATE_OBJECT_ID,
        "primitive": "rectangle_mesh_2tri",
        "plane_z_m": PLATE_DEPTH_M,
        "bounds_xy_m": [low[0], high[0], low[1], high[1]],
        "vertices_world_m": [
            [low[0], low[1], PLATE_DEPTH_M],
            [high[0], low[1], PLATE_DEPTH_M],
            [high[0], high[1], PLATE_DEPTH_M],
            [low[0], high[1], PLATE_DEPTH_M],
        ],
        "triangles": [[0, 1, 2], [0, 2, 3]],
        "material_id": "MAT_CAL_STEP_PLATE",
        "linear_rgb": [0.5, 0.5, 0.5],
        "texture": {
            "type": "calibration_step_grid_8x4",
            "rows": PLATE_ROWS,
            "columns": PLATE_COLUMNS,
            "low_linear_rgb": [0.15, 0.15, 0.15],
            "high_linear_rgb": [0.85, 0.85, 0.85],
            "mean_linear_rgb": [0.5, 0.5, 0.5],
        },
    }
    derived["world"]["objects"].append(plate)
    identity = {
        "base_scene_geometry_sha256": scene["scene_geometry_sha256"],
        "plate": plate,
    }
    derived["base_scene_geometry_sha256"] = scene["scene_geometry_sha256"]
    derived["calibration_scene_sha256"] = sha256_bytes(canonical_bytes(identity))
    return derived


def _plate_linear_rgb(
    world: np.ndarray,
    plate: dict[str, Any],
    alpha: float,
) -> np.ndarray:
    x0, x1, y0, y1 = (float(value) for value in plate["bounds_xy_m"])
    x_fraction = np.clip((world[:, 0] - x0) / (x1 - x0), 0.0, 1.0 - 1e-15)
    y_fraction = np.clip((world[:, 1] - y0) / (y1 - y0), 0.0, 1.0 - 1e-15)
    columns = np.floor(x_fraction * PLATE_COLUMNS).astype(np.int32)
    rows = np.floor(y_fraction * PLATE_ROWS).astype(np.int32)
    cell_x = x_fraction * PLATE_COLUMNS - columns
    cell_y = y_fraction * PLATE_ROWS - rows
    vertical = ((rows * PLATE_COLUMNS + columns) % 2) == 0
    high_side = np.where(vertical, cell_x >= 0.5, cell_y >= 0.5)
    clean_value = np.where(high_side, 0.85, 0.15)
    contracted = 0.5 + alpha * (clean_value - 0.5)
    return np.repeat(contracted[:, None], 3, axis=1)


def render_calibration_frame(
    scene: dict[str, Any],
    rotation_world_from_camera: np.ndarray,
    translation_world_from_camera: np.ndarray,
    low_texture_alpha: float | None = None,
) -> dict[str, Any]:
    """Render clean or pre-render albedo-contracted CAL RGB."""

    alpha = 1.0 if low_texture_alpha is None else float(low_texture_alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("LOW_TEXTURE_ALPHA_MUST_BE_IN_OPEN_CLOSED_UNIT_INTERVAL")
    if "calibration_scene_sha256" not in scene:
        raise ValueError("CALIBRATION_PLATE_MISSING")

    u, v = np.meshgrid(
        np.arange(p1.WIDTH, dtype=np.float64),
        np.arange(p1.HEIGHT, dtype=np.float64),
    )
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    depth, object_id, world = p1._raycast(  # P1 hash-bound geometry primitive.
        scene,
        np.asarray(rotation_world_from_camera, dtype=np.float64),
        np.asarray(translation_world_from_camera, dtype=np.float64),
        uv,
    )
    valid = np.isfinite(depth)
    linear = np.zeros((len(uv), 3), dtype=np.float64)
    material_mean = np.zeros((len(uv), 3), dtype=np.float64)
    by_id = {int(obj["object_id"]): obj for obj in scene["world"]["objects"]}
    for identifier in np.unique(object_id[valid]):
        selected = object_id == identifier
        obj = by_id[int(identifier)]
        if int(identifier) == PLATE_OBJECT_ID:
            linear[selected] = _plate_linear_rgb(world[selected], obj, alpha)
            material_mean[selected] = 0.5
            continue
        base = np.asarray(obj["linear_rgb"], dtype=np.float64)
        frequency = float(obj["texture"]["cycles_per_m"])
        phase = float(obj["texture"]["phase"])
        checker = (
            np.floor((world[selected, 0] * frequency + phase) % 2.0)
            + np.floor((world[selected, 1] * frequency + phase) % 2.0)
        ) % 2.0
        clean_modulation = 0.65 + 0.35 * checker
        mean_modulation = 0.825
        modulation = mean_modulation + alpha * (
            clean_modulation - mean_modulation
        )
        material_mean[selected] = base[None, :] * mean_modulation
        linear[selected] = np.clip(
            base[None, :] * modulation[:, None],
            0.0,
            1.0,
        )

    rgb = linear_to_srgb_u8(linear.reshape(p1.HEIGHT, p1.WIDTH, 3))
    object_image = object_id.reshape(p1.HEIGHT, p1.WIDTH)
    valid_image = valid.reshape(p1.HEIGHT, p1.WIDTH)
    edges = project_source_known_edges(
        scene,
        np.asarray(rotation_world_from_camera, dtype=np.float64),
        np.asarray(translation_world_from_camera, dtype=np.float64),
    )
    return {
        "rgb": rgb,
        "valid_mask": valid_image,
        "object_id": object_image,
        "edges": edges,
        "_linear_rgb": linear.reshape(p1.HEIGHT, p1.WIDTH, 3),
        "_material_mean_linear_rgb": material_mean.reshape(
            p1.HEIGHT,
            p1.WIDTH,
            3,
        ),
        "geometry_identity": {
            "base_scene_geometry_sha256": scene["base_scene_geometry_sha256"],
            "calibration_scene_sha256": scene["calibration_scene_sha256"],
            "valid_mask_sha256": sha256_bytes(
                valid_image.astype(np.uint8).tobytes()
            ),
            "object_id_sha256": sha256_bytes(
                object_image.astype("<i4").tobytes()
            ),
        },
    }


def apply_low_texture(
    clean_render: dict[str, Any],
    alpha: float,
) -> np.ndarray:
    """Apply the frozen pre-quantization albedo contraction to a clean render."""

    strength = float(alpha)
    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("LOW_TEXTURE_ALPHA_MUST_BE_IN_OPEN_CLOSED_UNIT_INTERVAL")
    clean_linear = clean_render["_linear_rgb"]
    material_mean = clean_render["_material_mean_linear_rgb"]
    contracted = material_mean + strength * (clean_linear - material_mean)
    return linear_to_srgb_u8(contracted)


def _project_world_points(
    points: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    translation_world_from_camera: np.ndarray,
) -> np.ndarray:
    camera = (
        rotation_world_from_camera.T
        @ (points - translation_world_from_camera).T
    ).T
    projected = (p1.K @ camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def project_source_known_edges(
    scene: dict[str, Any],
    rotation_world_from_camera: np.ndarray,
    translation_world_from_camera: np.ndarray,
) -> list[dict[str, Any]]:
    plate = next(
        obj
        for obj in scene["world"]["objects"]
        if int(obj["object_id"]) == PLATE_OBJECT_ID
    )
    x0, x1, y0, y1 = (float(value) for value in plate["bounds_xy_m"])
    cell_width = (x1 - x0) / PLATE_COLUMNS
    cell_height = (y1 - y0) / PLATE_ROWS
    edges: list[dict[str, Any]] = []
    for row in range(PLATE_ROWS):
        for column in range(PLATE_COLUMNS):
            edge_id = row * PLATE_COLUMNS + column
            vertical = edge_id % 2 == 0
            cell_x0 = x0 + column * cell_width
            cell_y0 = y0 + row * cell_height
            if vertical:
                center = np.asarray(
                    [
                        cell_x0 + 0.5 * cell_width,
                        cell_y0 + 0.5 * cell_height,
                        PLATE_DEPTH_M,
                    ],
                    dtype=np.float64,
                )
                tangent_world = np.asarray([0.0, 0.3 * cell_height, 0.0])
                high_world = np.asarray([0.2 * cell_width, 0.0, 0.0])
            else:
                center = np.asarray(
                    [
                        cell_x0 + 0.5 * cell_width,
                        cell_y0 + 0.5 * cell_height,
                        PLATE_DEPTH_M,
                    ],
                    dtype=np.float64,
                )
                tangent_world = np.asarray([0.3 * cell_width, 0.0, 0.0])
                high_world = np.asarray([0.0, 0.2 * cell_height, 0.0])
            projected = _project_world_points(
                np.stack(
                    (
                        center,
                        center - tangent_world,
                        center + tangent_world,
                        center + high_world,
                    )
                ),
                rotation_world_from_camera,
                translation_world_from_camera,
            )
            tangent = projected[2] - projected[1]
            tangent_norm = float(np.linalg.norm(tangent))
            if not math.isfinite(tangent_norm) or tangent_norm <= 0.0:
                raise InvalidQualityMetric("INVALID_EDGE_TANGENT")
            tangent /= tangent_norm
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            high_direction = projected[3] - projected[0]
            if float(np.dot(normal, high_direction)) < 0.0:
                normal *= -1.0
            edges.append(
                {
                    "edge_id": edge_id,
                    "center_uv": projected[0].tolist(),
                    "normal_low_to_high_uv": normal.tolist(),
                }
            )
    return edges


def erode_one_pixel(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    return cv2.erode(
        mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def variance_of_laplacian(
    luminance: np.ndarray,
    paired_valid_mask: np.ndarray,
) -> float:
    mask = erode_one_pixel(paired_valid_mask)
    if not np.any(mask):
        raise InvalidQualityMetric("EMPTY_ERODED_VALID_MASK")
    laplacian = cv2.filter2D(
        luminance,
        cv2.CV_64F,
        LAPLACIAN_KERNEL,
        borderType=cv2.BORDER_REFLECT_101,
    )
    result = float(np.var(laplacian[mask], ddof=0))
    if not math.isfinite(result):
        raise InvalidQualityMetric("NONFINITE_LAPLACIAN_VARIANCE")
    return result


def local_rms_contrast(
    luminance: np.ndarray,
    paired_valid_mask: np.ndarray,
) -> float:
    values: list[float] = []
    for y0 in range(0, luminance.shape[0], 16):
        for x0 in range(0, luminance.shape[1], 16):
            tile = luminance[y0 : y0 + 16, x0 : x0 + 16]
            valid = paired_valid_mask[y0 : y0 + 16, x0 : x0 + 16]
            if valid.size == 0 or float(np.mean(valid)) < 0.75:
                continue
            pixels = tile[valid]
            value = float(np.std(pixels, ddof=0) / (np.mean(pixels) + 1e-12))
            if not math.isfinite(value):
                raise InvalidQualityMetric("NONFINITE_LOCAL_RMS_TILE")
            values.append(value)
    if not values:
        raise InvalidQualityMetric("NO_VALID_LOCAL_RMS_TILE")
    return average_rank_median(values)


def _gaussian_luminance(luminance: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return luminance
    kernel = 2 * math.ceil(3.0 * sigma) + 1
    return cv2.GaussianBlur(
        luminance,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def multiscale_gradient_density_pair(
    clean_luminance: np.ndarray,
    degraded_luminance: np.ndarray,
    paired_valid_mask: np.ndarray,
) -> tuple[float, float]:
    mask = erode_one_pixel(paired_valid_mask)
    if not np.any(mask):
        raise InvalidQualityMetric("EMPTY_GRADIENT_VALID_MASK")
    clean_fractions: list[float] = []
    degraded_fractions: list[float] = []
    for sigma in SOBEL_SCALES:
        clean_filtered = _gaussian_luminance(clean_luminance, sigma)
        degraded_filtered = _gaussian_luminance(degraded_luminance, sigma)
        clean_gx = cv2.Sobel(
            clean_filtered,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        clean_gy = cv2.Sobel(
            clean_filtered,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        degraded_gx = cv2.Sobel(
            degraded_filtered,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        degraded_gy = cv2.Sobel(
            degraded_filtered,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        clean_magnitude = np.hypot(clean_gx, clean_gy)
        degraded_magnitude = np.hypot(degraded_gx, degraded_gy)
        threshold = float(
            np.quantile(clean_magnitude[mask], 0.75, method="linear")
        )
        if not math.isfinite(threshold):
            raise InvalidQualityMetric("MISSING_GRADIENT_THRESHOLD")
        clean_fraction = float(np.mean(clean_magnitude[mask] > threshold))
        degraded_fraction = float(
            np.mean(degraded_magnitude[mask] > threshold)
        )
        if not math.isfinite(clean_fraction) or not math.isfinite(
            degraded_fraction
        ):
            raise InvalidQualityMetric("NONFINITE_GRADIENT_DENSITY")
        clean_fractions.append(clean_fraction)
        degraded_fractions.append(degraded_fraction)
    return float(np.mean(clean_fractions)), float(np.mean(degraded_fractions))


def _bilinear_sample(image: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    map_x = points_uv[:, 0].astype(np.float32).reshape(1, -1)
    map_y = points_uv[:, 1].astype(np.float32).reshape(1, -1)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).reshape(-1)


def _nearest_sample(image: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    x = np.rint(points_uv[:, 0]).astype(np.int64)
    y = np.rint(points_uv[:, 1]).astype(np.int64)
    inside = (
        (x >= 0)
        & (x < image.shape[1])
        & (y >= 0)
        & (y < image.shape[0])
    )
    result = np.zeros(len(points_uv), dtype=np.int32)
    result[inside] = image[y[inside], x[inside]]
    return result


def _first_crossing(
    offsets: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float:
    for index in range(1, len(values)):
        left = float(values[index - 1])
        right = float(values[index])
        if (left <= target <= right) or (right <= target <= left):
            if right == left:
                return float(offsets[index])
            fraction = (target - left) / (right - left)
            return float(
                offsets[index - 1]
                + fraction * (offsets[index] - offsets[index - 1])
            )
    raise InvalidQualityMetric("EDGE_CROSSING_MISSING")


def source_known_edge_spread(
    luminance: np.ndarray,
    object_id: np.ndarray,
    edges: list[dict[str, Any]],
) -> tuple[float, int]:
    widths: list[float] = []
    for edge in edges:
        center = np.asarray(edge["center_uv"], dtype=np.float64)
        normal = np.asarray(
            edge["normal_low_to_high_uv"],
            dtype=np.float64,
        )
        points = center[None, :] + EDGE_SAMPLE_OFFSETS_PX[:, None] * normal
        identifiers = _nearest_sample(object_id, points)
        if not np.all(identifiers == PLATE_OBJECT_ID):
            continue
        values = _bilinear_sample(luminance, points)
        low_plateau = values[
            (EDGE_SAMPLE_OFFSETS_PX >= -8.0)
            & (EDGE_SAMPLE_OFFSETS_PX <= -5.0)
        ]
        high_plateau = values[
            (EDGE_SAMPLE_OFFSETS_PX >= 5.0)
            & (EDGE_SAMPLE_OFFSETS_PX <= 8.0)
        ]
        if low_plateau.size == 0 or high_plateau.size == 0:
            continue
        low = float(np.mean(low_plateau))
        high = float(np.mean(high_plateau))
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            continue
        crossing_10 = _first_crossing(
            EDGE_SAMPLE_OFFSETS_PX,
            values,
            low + 0.10 * (high - low),
        )
        crossing_90 = _first_crossing(
            EDGE_SAMPLE_OFFSETS_PX,
            values,
            low + 0.90 * (high - low),
        )
        width = crossing_90 - crossing_10
        if math.isfinite(width) and width >= 0.0:
            widths.append(width)
    if not widths:
        raise InvalidQualityMetric("NO_VALID_SOURCE_KNOWN_EDGE")
    return average_rank_median(widths), len(widths)


def safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator == 0.0
    ):
        raise InvalidQualityMetric(f"INVALID_RATIO_{label}")
    ratio = float(numerator / denominator)
    if not math.isfinite(ratio):
        raise InvalidQualityMetric(f"NONFINITE_RATIO_{label}")
    return ratio


def prepare_clean_frame_metrics(clean: dict[str, Any]) -> dict[str, Any]:
    mask = clean["valid_mask"].astype(bool)
    eroded = erode_one_pixel(mask)
    if not np.any(eroded):
        raise InvalidQualityMetric("EMPTY_PREPARED_VALID_MASK")
    luminance = linear_luminance(clean["rgb"])
    gradient_thresholds: list[float] = []
    gradient_clean_fractions: list[float] = []
    for sigma in SOBEL_SCALES:
        filtered = _gaussian_luminance(luminance, sigma)
        gx = cv2.Sobel(
            filtered,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        gy = cv2.Sobel(
            filtered,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        magnitude = np.hypot(gx, gy)
        threshold = float(np.quantile(magnitude[eroded], 0.75, method="linear"))
        fraction = float(np.mean(magnitude[eroded] > threshold))
        if not math.isfinite(threshold) or not math.isfinite(fraction):
            raise InvalidQualityMetric("NONFINITE_PREPARED_GRADIENT")
        gradient_thresholds.append(threshold)
        gradient_clean_fractions.append(fraction)
    edge_width, edge_count = source_known_edge_spread(
        luminance,
        clean["object_id"],
        clean["edges"],
    )
    prepared = {
        "luminance": luminance,
        "valid_mask": mask,
        "eroded_mask": eroded,
        "laplacian_variance": variance_of_laplacian(luminance, mask),
        "local_rms_contrast": local_rms_contrast(luminance, mask),
        "gradient_thresholds": gradient_thresholds,
        "gradient_clean_fractions": gradient_clean_fractions,
        "multiscale_gradient_density": float(
            np.mean(gradient_clean_fractions)
        ),
        "edge_spread_px": edge_width,
        "valid_edge_count": edge_count,
    }
    for value in (
        prepared["laplacian_variance"],
        prepared["local_rms_contrast"],
        prepared["multiscale_gradient_density"],
        prepared["edge_spread_px"],
    ):
        if not math.isfinite(float(value)):
            raise InvalidQualityMetric("NONFINITE_PREPARED_METRIC")
    return prepared


def blur_frame_metrics(
    prepared_clean: dict[str, Any],
    degraded_rgb: np.ndarray,
) -> dict[str, float]:
    luminance = linear_luminance(degraded_rgb)
    degraded_laplacian = variance_of_laplacian(
        luminance,
        prepared_clean["valid_mask"],
    )
    degraded_rms = local_rms_contrast(
        luminance,
        prepared_clean["valid_mask"],
    )
    return {
        "clean_laplacian_variance": prepared_clean["laplacian_variance"],
        "degraded_laplacian_variance": degraded_laplacian,
        "laplacian_variance_ratio": safe_ratio(
            degraded_laplacian,
            prepared_clean["laplacian_variance"],
            "LAPLACIAN",
        ),
        "clean_local_rms_contrast": prepared_clean["local_rms_contrast"],
        "degraded_local_rms_contrast": degraded_rms,
        "local_rms_contrast_ratio": safe_ratio(
            degraded_rms,
            prepared_clean["local_rms_contrast"],
            "LOCAL_RMS",
        ),
    }


def low_texture_frame_metrics(
    clean: dict[str, Any],
    prepared_clean: dict[str, Any],
    degraded_rgb: np.ndarray,
) -> dict[str, float | int]:
    luminance = linear_luminance(degraded_rgb)
    degraded_fractions: list[float] = []
    for sigma, threshold in zip(
        SOBEL_SCALES,
        prepared_clean["gradient_thresholds"],
        strict=True,
    ):
        filtered = _gaussian_luminance(luminance, sigma)
        gx = cv2.Sobel(
            filtered,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        gy = cv2.Sobel(
            filtered,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            scale=1.0 / 8.0,
            borderType=cv2.BORDER_REFLECT_101,
        )
        magnitude = np.hypot(gx, gy)
        fraction = float(
            np.mean(magnitude[prepared_clean["eroded_mask"]] > threshold)
        )
        if not math.isfinite(fraction):
            raise InvalidQualityMetric("NONFINITE_DEGRADED_GRADIENT")
        degraded_fractions.append(fraction)
    degraded_density = float(np.mean(degraded_fractions))
    degraded_edge, degraded_edge_count = source_known_edge_spread(
        luminance,
        clean["object_id"],
        clean["edges"],
    )
    return {
        "clean_multiscale_gradient_density": prepared_clean[
            "multiscale_gradient_density"
        ],
        "degraded_multiscale_gradient_density": degraded_density,
        "multiscale_gradient_density_ratio": safe_ratio(
            degraded_density,
            prepared_clean["multiscale_gradient_density"],
            "GRADIENT_DENSITY",
        ),
        "clean_edge_spread_px": prepared_clean["edge_spread_px"],
        "degraded_edge_spread_px": degraded_edge,
        "edge_spread_ratio": safe_ratio(
            degraded_edge,
            prepared_clean["edge_spread_px"],
            "EDGE_SPREAD",
        ),
        "clean_valid_edge_count": prepared_clean["valid_edge_count"],
        "degraded_valid_edge_count": degraded_edge_count,
    }


def frame_metrics(
    clean: dict[str, Any],
    degraded_rgb: np.ndarray,
    include_edge_spread: bool,
) -> dict[str, float | int]:
    paired_mask = clean["valid_mask"].astype(bool)
    clean_luminance = linear_luminance(clean["rgb"])
    degraded_luminance = linear_luminance(degraded_rgb)
    clean_laplacian = variance_of_laplacian(clean_luminance, paired_mask)
    degraded_laplacian = variance_of_laplacian(
        degraded_luminance,
        paired_mask,
    )
    clean_rms = local_rms_contrast(clean_luminance, paired_mask)
    degraded_rms = local_rms_contrast(degraded_luminance, paired_mask)
    clean_gradient, degraded_gradient = multiscale_gradient_density_pair(
        clean_luminance,
        degraded_luminance,
        paired_mask,
    )
    result: dict[str, float | int] = {
        "clean_laplacian_variance": clean_laplacian,
        "degraded_laplacian_variance": degraded_laplacian,
        "laplacian_variance_ratio": safe_ratio(
            degraded_laplacian,
            clean_laplacian,
            "LAPLACIAN",
        ),
        "clean_local_rms_contrast": clean_rms,
        "degraded_local_rms_contrast": degraded_rms,
        "local_rms_contrast_ratio": safe_ratio(
            degraded_rms,
            clean_rms,
            "LOCAL_RMS",
        ),
        "clean_multiscale_gradient_density": clean_gradient,
        "degraded_multiscale_gradient_density": degraded_gradient,
        "multiscale_gradient_density_ratio": safe_ratio(
            degraded_gradient,
            clean_gradient,
            "GRADIENT_DENSITY",
        ),
    }
    if include_edge_spread:
        clean_width, clean_edges = source_known_edge_spread(
            clean_luminance,
            clean["object_id"],
            clean["edges"],
        )
        degraded_width, degraded_edges = source_known_edge_spread(
            degraded_luminance,
            clean["object_id"],
            clean["edges"],
        )
        result.update(
            {
                "clean_edge_spread_px": clean_width,
                "degraded_edge_spread_px": degraded_width,
                "edge_spread_ratio": safe_ratio(
                    degraded_width,
                    clean_width,
                    "EDGE_SPREAD",
                ),
                "clean_valid_edge_count": clean_edges,
                "degraded_valid_edge_count": degraded_edges,
            }
        )
    return result


def analytic_edge_fixture(alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a full-visibility screen-space plate pair for cross-proxy QA."""

    clean_value = np.full((p1.HEIGHT, p1.WIDTH), 0.5, dtype=np.float64)
    degraded_value = np.full_like(clean_value, 0.5)
    object_id = np.zeros((p1.HEIGHT, p1.WIDTH), dtype=np.int32)
    edges: list[dict[str, Any]] = []
    u0, u1, v0, v1 = PLATE_REFERENCE_BOUNDS_PX
    cell_width = (u1 - u0) / PLATE_COLUMNS
    cell_height = (v1 - v0) / PLATE_ROWS
    for row in range(PLATE_ROWS):
        for column in range(PLATE_COLUMNS):
            edge_id = row * PLATE_COLUMNS + column
            x0 = int(math.floor(u0 + column * cell_width))
            x1 = int(math.ceil(u0 + (column + 1) * cell_width))
            y0 = int(math.floor(v0 + row * cell_height))
            y1 = int(math.ceil(v0 + (row + 1) * cell_height))
            object_id[y0:y1, x0:x1] = PLATE_OBJECT_ID
            vertical = edge_id % 2 == 0
            if vertical:
                crossing = u0 + (column + 0.5) * cell_width
                pixel_crossing = int(math.ceil(crossing))
                clean_value[y0:y1, x0:pixel_crossing] = 0.15
                clean_value[y0:y1, pixel_crossing:x1] = 0.85
                degraded_value[y0:y1, x0:pixel_crossing] = (
                    0.5 + alpha * (0.15 - 0.5)
                )
                degraded_value[y0:y1, pixel_crossing:x1] = (
                    0.5 + alpha * (0.85 - 0.5)
                )
                center = [crossing - 0.5, (y0 + y1 - 1) / 2.0]
                normal = [1.0, 0.0]
            else:
                crossing = v0 + (row + 0.5) * cell_height
                pixel_crossing = int(math.ceil(crossing))
                clean_value[y0:pixel_crossing, x0:x1] = 0.15
                clean_value[pixel_crossing:y1, x0:x1] = 0.85
                degraded_value[y0:pixel_crossing, x0:x1] = (
                    0.5 + alpha * (0.15 - 0.5)
                )
                degraded_value[pixel_crossing:y1, x0:x1] = (
                    0.5 + alpha * (0.85 - 0.5)
                )
                center = [(x0 + x1 - 1) / 2.0, crossing - 0.5]
                normal = [0.0, 1.0]
            edges.append(
                {
                    "edge_id": edge_id,
                    "center_uv": center,
                    "normal_low_to_high_uv": normal,
                }
            )
    clean_linear = np.repeat(clean_value[:, :, None], 3, axis=2)
    degraded_linear = np.repeat(degraded_value[:, :, None], 3, axis=2)
    clean = {
        "rgb": linear_to_srgb_u8(clean_linear),
        "object_id": object_id,
        "valid_mask": object_id == PLATE_OBJECT_ID,
        "edges": edges,
    }
    degraded = {
        "rgb": linear_to_srgb_u8(degraded_linear),
        "object_id": object_id.copy(),
        "valid_mask": object_id == PLATE_OBJECT_ID,
        "edges": copy.deepcopy(edges),
    }
    return clean, degraded
