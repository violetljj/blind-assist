#!/usr/bin/env python3
"""ARKitScenes sensor-derived geometry reader for Assistive Geometry B0.

The reader converts source-native millimetre depth, synchronized confidence,
per-frame intrinsics and interpolated camera pose into an upright metric frame.
It then derives gravity-bound ground and three body-swept clearance targets.
These targets are research sensor geometry, not human safety truth.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
ROTATION_LABELS = ("IDENTITY", "CLOCKWISE_90", "ROTATE_180", "COUNTERCLOCKWISE_90")
ROLE_BANDS = {
    "left": 1.0 / 6.0,
    "center": 1.0 / 2.0,
    "right": 5.0 / 6.0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TruthReaderPolicy:
    depth_min_m: float = 0.25
    depth_max_m: float = 6.0
    minimum_sensor_confidence: int = 1
    point_stride: int = 2
    ground_height_min_m: float = 0.45
    ground_height_max_m: float = 2.20
    ground_histogram_bin_m: float = 0.04
    ground_support_tolerance_m: float = 0.08
    minimum_ground_support_points: int = 80
    minimum_ground_support_fraction: float = 0.02
    obstacle_height_min_m: float = 0.08
    obstacle_height_max_m: float = 2.00
    minimum_forward_m: float = 0.20
    maximum_forward_m: float = 6.00
    body_half_width_m: float = 0.32
    lateral_margin_m: float = 0.10
    clearance_quantile: float = 0.02
    support_forward_quantile: float = 0.90
    minimum_band_support_points: int = 20
    minimum_intrusion_points: int = 20
    horizons_m: tuple[float, ...] = (1.0, 1.5, 2.0)
    maximum_pose_bracketing_gap_seconds: float = 0.25

    def validate(self) -> None:
        require(0 < self.depth_min_m < self.depth_max_m, "invalid depth range")
        require(self.minimum_sensor_confidence in (0, 1, 2), "invalid sensor confidence threshold")
        require(self.point_stride > 0, "point stride must be positive")
        require(0 < self.ground_height_min_m < self.ground_height_max_m, "invalid ground height range")
        require(self.ground_histogram_bin_m > 0 and self.ground_support_tolerance_m > 0, "invalid ground support policy")
        require(self.minimum_ground_support_points > 0, "minimum ground support must be positive")
        require(0 < self.minimum_ground_support_fraction < 1, "invalid ground support fraction")
        require(0 <= self.obstacle_height_min_m < self.obstacle_height_max_m, "invalid obstacle height range")
        require(0 < self.minimum_forward_m < self.maximum_forward_m, "invalid forward range")
        require(self.body_half_width_m > 0 and self.lateral_margin_m >= 0, "invalid body profile")
        require(0 < self.clearance_quantile < 0.5, "invalid clearance quantile")
        require(0 < self.support_forward_quantile < 1, "invalid support quantile")
        require(self.minimum_band_support_points > 0 and self.minimum_intrusion_points > 0, "invalid support counts")
        require(tuple(sorted(set(self.horizons_m))) == self.horizons_m, "horizons must be increasing and unique")
        require(all(value > 0 for value in self.horizons_m), "horizons must be positive")
        require(self.maximum_pose_bracketing_gap_seconds > 0, "pose gap must be positive")

    @property
    def total_half_width_m(self) -> float:
        return self.body_half_width_m + self.lateral_margin_m


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    require(norm > 1e-12, "degenerate quaternion")
    return quaternion / norm


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    require(vector.shape == (3,) and np.all(np.isfinite(vector)), "rotation vector must be finite length three")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = vector / angle
    half = angle / 2.0
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_quaternion(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    require(0 <= fraction <= 1 and math.isfinite(fraction), "invalid interpolation fraction")
    first = _normalize_quaternion(left)
    second = _normalize_quaternion(right)
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(first + fraction * (second - first))
    angle = math.acos(dot)
    scale = math.sin(angle)
    return _normalize_quaternion(
        math.sin((1 - fraction) * angle) / scale * first
        + math.sin(fraction * angle) / scale * second
    )


def parse_trajectory(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 7, f"trajectory row {line_number} must have seven fields: {path}")
        values = [float(field) for field in fields]
        require(all(math.isfinite(value) for value in values), f"non-finite trajectory row {line_number}: {path}")
        rows.append(values)
    result = np.asarray(rows, dtype=np.float64)
    require(result.ndim == 2 and result.shape[0] >= 2 and result.shape[1] == 7, f"invalid trajectory: {path}")
    require(np.all(np.diff(result[:, 0]) > 0), f"trajectory timestamps are not strictly increasing: {path}")
    return result


def _stored_row_to_camera_world(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The official ARKitScenes loader forms [R, t] from the stored angle-axis
    # row and then inverts it.  Quaternion conjugation gives R_camera_to_world.
    q_world_to_camera = rotation_vector_to_quaternion(row[1:4])
    q_camera_to_world = q_world_to_camera * np.asarray([1.0, -1.0, -1.0, -1.0])
    rotation = quaternion_to_rotation_matrix(q_camera_to_world)
    translation = -(rotation @ row[4:7])
    return q_camera_to_world, translation


def interpolate_camera_to_world(
    trajectory: np.ndarray,
    timestamp: float,
    maximum_gap_seconds: float,
) -> tuple[np.ndarray, dict[str, float]]:
    require(math.isfinite(timestamp), "frame timestamp must be finite")
    times = trajectory[:, 0]
    right = bisect.bisect_left(times.tolist(), timestamp)
    if right < len(times) and abs(float(times[right]) - timestamp) <= 1e-9:
        left = right
        fraction = 0.0
    else:
        require(0 < right < len(times), f"frame outside trajectory domain: {timestamp}")
        left = right - 1
        gap = float(times[right] - times[left])
        require(gap <= maximum_gap_seconds, f"pose bracket {gap} exceeds {maximum_gap_seconds}")
        fraction = float((timestamp - times[left]) / gap)
    q_left, p_left = _stored_row_to_camera_world(trajectory[left])
    if left == right:
        quaternion, translation = q_left, p_left
        gap = 0.0
    else:
        q_right, p_right = _stored_row_to_camera_world(trajectory[right])
        quaternion = slerp_quaternion(q_left, q_right, fraction)
        translation = p_left + fraction * (p_right - p_left)
        gap = float(times[right] - times[left])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    transform[:3, 3] = translation
    return transform, {
        "left_timestamp": float(times[left]),
        "right_timestamp": float(times[right]),
        "fraction": fraction,
        "bracketing_gap_seconds": gap,
    }


def orientation_index(camera_to_world: np.ndarray) -> int:
    transform = np.asarray(camera_to_world, dtype=np.float64)
    require(transform.shape == (4, 4) and np.all(np.isfinite(transform)), "camera pose must be finite 4x4")
    world_up_in_camera = transform[2, :3]
    canonical = np.asarray(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return int(np.argmax(canonical @ world_up_in_camera))


def upright_to_source_basis(index: int) -> np.ndarray:
    require(index in (0, 1, 2, 3), "rotation index must be 0, 1, 2 or 3")
    return (
        np.eye(3, dtype=np.float64),
        np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.diag([-1.0, -1.0, 1.0]),
        np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )[index]


def rotate_array_upright(array: np.ndarray, index: int) -> np.ndarray:
    require(array.ndim in (2, 3), "image array must be rank two or three")
    return np.ascontiguousarray(np.rot90(array, k=(0, -1, 2, 1)[index]))


def rotate_intrinsics_upright(
    intrinsics: np.ndarray,
    source_width: int,
    source_height: int,
    index: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    require(matrix.shape == (3, 3) and np.all(np.isfinite(matrix)), "intrinsics must be finite 3x3")
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    require(fx > 0 and fy > 0, "focal lengths must be positive")
    if index == 0:
        values = fx, fy, cx, cy
        output_size = source_width, source_height
    elif index == 1:
        values = fy, fx, source_height - 1 - cy, cx
        output_size = source_height, source_width
    elif index == 2:
        values = fx, fy, source_width - 1 - cx, source_height - 1 - cy
        output_size = source_width, source_height
    elif index == 3:
        values = fy, fx, cy, source_width - 1 - cx
        output_size = source_height, source_width
    else:
        raise ValueError("rotation index must be 0, 1, 2 or 3")
    fx_out, fy_out, cx_out, cy_out = values
    output = np.asarray([[fx_out, 0.0, cx_out], [0.0, fy_out, cy_out], [0.0, 0.0, 1.0]])
    return output, output_size


def canonicalize_frame(
    rgb: np.ndarray,
    depth_raw_mm: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> dict[str, Any]:
    require(rgb.ndim == 3 and rgb.shape[2] == 3, "RGB must be HxWx3")
    require(depth_raw_mm.shape == rgb.shape[:2] and confidence.shape == rgb.shape[:2], "registered modality shapes differ")
    index = orientation_index(camera_to_world)
    source_height, source_width = depth_raw_mm.shape
    upright_intrinsics, output_size = rotate_intrinsics_upright(
        intrinsics, source_width, source_height, index
    )
    transform = np.asarray(camera_to_world, dtype=np.float64).copy()
    transform[:3, :3] = transform[:3, :3] @ upright_to_source_basis(index)
    rgb_upright = rotate_array_upright(rgb, index)
    depth_upright = rotate_array_upright(depth_raw_mm, index)
    confidence_upright = rotate_array_upright(confidence, index)
    require((rgb_upright.shape[1], rgb_upright.shape[0]) == output_size, "upright image size drift")
    return {
        "rgb": rgb_upright,
        "depth_raw_mm": depth_upright,
        "confidence": confidence_upright,
        "intrinsics": upright_intrinsics,
        "camera_to_world": transform,
        "rotation_index": index,
        "rotation_label": ROTATION_LABELS[index],
    }


def parse_pincam(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    require(len(values) == 6 and all(math.isfinite(value) for value in values), f"invalid pincam: {path}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height and width > 0 and height > 0, f"invalid pincam dimensions: {path}")
    fx, fy, cx, cy = values[2:]
    require(fx > 0 and fy > 0 and 0 <= cx < width and 0 <= cy < height, f"invalid pincam values: {path}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (width, height)


def depth_mm_to_metres(depth_raw_mm: np.ndarray) -> np.ndarray:
    raw = np.asarray(depth_raw_mm)
    require(raw.ndim == 2 and np.issubdtype(raw.dtype, np.integer), "depth must be an integer raster")
    return raw.astype(np.float32) / 1000.0


def unproject_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    valid: np.ndarray,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    require(depth.ndim == 2 and mask.shape == depth.shape, "depth/valid shape mismatch")
    require(stride > 0, "stride must be positive")
    rows, columns = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sampled_depth = depth[::stride, ::stride]
    sampled_valid = mask[::stride, ::stride]
    fx, fy, cx, cy = (
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )
    z = sampled_depth[sampled_valid]
    u = columns[sampled_valid]
    v = rows[sampled_valid]
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    pixels = np.stack((u, v), axis=1)
    return points, pixels


def fit_gravity_ground_plane(
    points: np.ndarray,
    up_camera: np.ndarray,
    policy: TruthReaderPolicy,
) -> dict[str, Any] | None:
    up = np.asarray(up_camera, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    require(norm > 1e-9 and np.all(np.isfinite(up)), "gravity vector is invalid")
    up /= norm
    offsets = -(points @ up)
    plausible = np.isfinite(offsets) & (offsets >= policy.ground_height_min_m) & (offsets <= policy.ground_height_max_m)
    if int(np.sum(plausible)) < policy.minimum_ground_support_points:
        return None
    edges = np.arange(
        policy.ground_height_min_m,
        policy.ground_height_max_m + policy.ground_histogram_bin_m * 1.001,
        policy.ground_histogram_bin_m,
    )
    counts, edges = np.histogram(offsets[plausible], bins=edges)
    if not len(counts):
        return None
    maximum = int(np.max(counts))
    candidate_bins = np.flatnonzero(counts == maximum)
    mode_index = int(candidate_bins[-1])
    mode_center = float((edges[mode_index] + edges[mode_index + 1]) / 2.0)
    support = plausible & (np.abs(offsets - mode_center) <= policy.ground_support_tolerance_m)
    support_count = int(np.sum(support))
    minimum_count = max(
        policy.minimum_ground_support_points,
        int(math.ceil(policy.minimum_ground_support_fraction * len(points))),
    )
    if support_count < minimum_count:
        return None
    camera_height = float(np.median(offsets[support]))
    residual = float(np.median(np.abs(offsets[support] - camera_height)))
    return {
        "normal_camera": up,
        "camera_height_m": camera_height,
        "median_residual_m": residual,
        "support_points": support_count,
        "sampled_valid_points": int(len(points)),
        "support_fraction": float(support_count / len(points)),
    }


def _band_heading(
    x_fraction: float,
    width: int,
    intrinsics: np.ndarray,
    up: np.ndarray,
) -> np.ndarray | None:
    pixel = np.asarray([x_fraction * width - 0.5, intrinsics[1, 2], 1.0], dtype=np.float64)
    ray = np.linalg.solve(intrinsics, pixel)
    heading = ray - float(np.dot(ray, up)) * up
    norm = float(np.linalg.norm(heading))
    if norm <= 1e-6:
        return None
    heading /= norm
    optical = np.asarray([0.0, 0.0, 1.0])
    optical_ground = optical - float(np.dot(optical, up)) * up
    if float(np.dot(heading, optical_ground)) < 0:
        heading = -heading
    return heading


def derive_assistive_truth(
    depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    up_camera: np.ndarray,
    policy: TruthReaderPolicy = TruthReaderPolicy(),
) -> dict[str, Any]:
    policy.validate()
    depth = np.asarray(depth_m, dtype=np.float64)
    confidence_array = np.asarray(confidence)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    require(depth.ndim == 2 and confidence_array.shape == depth.shape, "depth/confidence shape mismatch")
    require(matrix.shape == (3, 3) and np.all(np.isfinite(matrix)), "intrinsics must be finite 3x3")
    valid = (
        np.isfinite(depth)
        & (depth >= policy.depth_min_m)
        & (depth <= policy.depth_max_m)
        & (confidence_array >= policy.minimum_sensor_confidence)
    )
    valid_fraction = float(np.mean(valid)) if valid.size else 0.0
    sampled_points, _ = unproject_depth(depth, matrix, valid, policy.point_stride)
    if len(sampled_points) < policy.minimum_ground_support_points:
        return {
            "status": "UNKNOWN",
            "unknown_reasons": ["UNKNOWN_INSUFFICIENT_VALID_DEPTH"],
            "depth_valid": valid,
            "ground_valid": np.zeros_like(valid),
            "ground_probability": np.zeros_like(depth, dtype=np.float32),
            "valid_depth_fraction": valid_fraction,
            "ground_plane": None,
            "bands": {},
        }
    plane = fit_gravity_ground_plane(sampled_points, up_camera, policy)
    if plane is None:
        return {
            "status": "UNKNOWN",
            "unknown_reasons": ["UNKNOWN_GROUND_PLANE"],
            "depth_valid": valid,
            "ground_valid": np.zeros_like(valid),
            "ground_probability": np.zeros_like(depth, dtype=np.float32),
            "valid_depth_fraction": valid_fraction,
            "ground_plane": None,
            "bands": {},
        }
    up = plane["normal_camera"]
    all_points, all_pixels = unproject_depth(depth, matrix, valid, 1)
    heights = all_points @ up + float(plane["camera_height_m"])
    ground_support = np.abs(heights) <= policy.ground_support_tolerance_m
    ground_valid = np.zeros_like(valid)
    ground_valid[all_pixels[ground_support, 1], all_pixels[ground_support, 0]] = True
    ground_probability = np.zeros_like(depth, dtype=np.float32)
    ground_probability[all_pixels[ground_support, 1], all_pixels[ground_support, 0]] = 1.0
    obstacle = (
        (heights >= policy.obstacle_height_min_m)
        & (heights <= policy.obstacle_height_max_m)
    )

    bands: dict[str, Any] = {}
    for name, x_fraction in ROLE_BANDS.items():
        heading = _band_heading(x_fraction, depth.shape[1], matrix, up)
        if heading is None:
            bands[name] = {
                "status": "UNKNOWN_GROUND_FORWARD",
                "clearance_m": None,
                "occupied_by_horizon": {str(value): None for value in policy.horizons_m},
            }
            continue
        lateral_axis = np.cross(heading, up)
        lateral_norm = float(np.linalg.norm(lateral_axis))
        if lateral_norm <= 1e-9:
            bands[name] = {
                "status": "UNKNOWN_GROUND_LATERAL",
                "clearance_m": None,
                "occupied_by_horizon": {str(value): None for value in policy.horizons_m},
            }
            continue
        lateral_axis /= lateral_norm
        forward = all_points @ heading
        lateral = all_points @ lateral_axis
        corridor = (
            (forward >= policy.minimum_forward_m)
            & (forward <= policy.maximum_forward_m)
            & (np.abs(lateral) <= policy.total_half_width_m)
        )
        support = forward[corridor]
        support_count = int(len(support))
        observed_forward = (
            float(np.quantile(support, policy.support_forward_quantile))
            if support_count
            else 0.0
        )
        intrusions = forward[corridor & obstacle]
        intrusion_count = int(len(intrusions))
        clearance = (
            float(np.quantile(intrusions, policy.clearance_quantile))
            if intrusion_count >= policy.minimum_intrusion_points
            else None
        )
        occupied: dict[str, bool | None] = {}
        for horizon in policy.horizons_m:
            if clearance is not None and clearance <= horizon:
                occupied[str(horizon)] = True
            elif support_count >= policy.minimum_band_support_points and observed_forward >= horizon:
                occupied[str(horizon)] = False
            else:
                occupied[str(horizon)] = None
        known = support_count >= policy.minimum_band_support_points
        bands[name] = {
            "status": "KNOWN" if known else "UNKNOWN_SUPPORT",
            "x_normalized_interval": (
                [0.0, 1.0 / 3.0]
                if name == "left"
                else [1.0 / 3.0, 2.0 / 3.0]
                if name == "center"
                else [2.0 / 3.0, 1.0]
            ),
            "heading_camera": [float(value) for value in heading],
            "body_total_half_width_m": policy.total_half_width_m,
            "clearance_m": clearance,
            "support_points": support_count,
            "intrusion_points": intrusion_count,
            "observed_forward_m": observed_forward,
            "occupied_by_horizon": occupied,
        }
    status = "VALID" if all(value["status"] == "KNOWN" for value in bands.values()) else "PARTIAL_UNKNOWN"
    return {
        "status": status,
        "unknown_reasons": [] if status == "VALID" else ["UNKNOWN_BAND_SUPPORT"],
        "depth_valid": valid,
        "ground_valid": ground_valid,
        "ground_probability": ground_probability,
        "valid_depth_fraction": valid_fraction,
        "ground_plane": {
            **{key: value for key, value in plane.items() if key != "normal_camera"},
            "normal_camera": [float(value) for value in up],
            "source": "ARKIT_GRAVITY_PLUS_REGISTERED_SENSOR_DEPTH_MODE",
        },
        "bands": bands,
    }


def _entry_stem(entry: dict[str, Any]) -> str:
    return Path(entry["path"]).stem


def load_manifest_frame(
    video: dict[str, Any],
    frame_index: int,
    trajectory: np.ndarray,
    policy: TruthReaderPolicy = TruthReaderPolicy(),
) -> dict[str, Any]:
    selected = [str(value) for value in video["selected_frame_stems"]]
    require(0 <= frame_index < len(selected), "frame index outside selected window")
    stem = selected[frame_index]
    extracted = video["extracted"]
    for modality in ("lowres_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics"):
        require(len(extracted[modality]) == len(selected), f"{modality} mapping count drift")
    require(_entry_stem(extracted["lowres_wide"][frame_index]) == stem, "RGB stem drift")
    require(_entry_stem(extracted["lowres_depth"][frame_index]) == stem, "depth stem drift")
    require(_entry_stem(extracted["confidence"][frame_index]) == stem, "confidence stem drift")
    require(_entry_stem(extracted["lowres_wide_intrinsics"][frame_index]) == stem, "intrinsics stem drift")

    with Image.open(extracted["lowres_wide"][frame_index]["path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(extracted["lowres_depth"][frame_index]["path"]) as image:
        depth_raw = np.asarray(image).copy()
    with Image.open(extracted["confidence"][frame_index]["path"]) as image:
        confidence = np.asarray(image).copy()
    intrinsics, source_size = parse_pincam(Path(extracted["lowres_wide_intrinsics"][frame_index]["path"]))
    require((rgb.shape[1], rgb.shape[0]) == source_size, "RGB/pincam size drift")
    timestamp = float(stem.rsplit("_", 1)[1])
    pose, interpolation = interpolate_camera_to_world(
        trajectory,
        timestamp,
        policy.maximum_pose_bracketing_gap_seconds,
    )
    canonical = canonicalize_frame(rgb, depth_raw, confidence, intrinsics, pose)
    depth_m = depth_mm_to_metres(canonical["depth_raw_mm"])
    up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
    truth = derive_assistive_truth(
        depth_m,
        canonical["confidence"],
        canonical["intrinsics"],
        up_camera,
        policy,
    )
    return {
        "identity": {
            "role": str(video["role"]),
            "visit_id": str(video["visit_id"]),
            "video_id": str(video["video_id"]),
            "frame_index": frame_index,
            "frame_stem": stem,
            "timestamp_seconds": timestamp,
        },
        "source_registration": {
            "rgb_depth_confidence_same_stem": True,
            "intrinsics_same_stem": True,
            "source_size_wh": list(source_size),
            "depth_storage": "uint16_millimetres",
            "confidence_storage": "uint8_0_low_2_high",
        },
        "orientation": {
            "rotation_index": canonical["rotation_index"],
            "rotation_label": canonical["rotation_label"],
            "upright_size_wh": [int(depth_m.shape[1]), int(depth_m.shape[0])],
            "up_camera": [float(value) for value in up_camera],
        },
        "pose_interpolation": interpolation,
        "intrinsics_upright": canonical["intrinsics"],
        "camera_to_world_upright": canonical["camera_to_world"],
        "rgb_upright": canonical["rgb"],
        "depth_m_upright": depth_m,
        "confidence_upright": canonical["confidence"],
        "truth": truth,
        "authority": "SENSOR_DERIVED_RESEARCH_GEOMETRY_NOT_HUMAN_SAFETY_TRUTH",
    }
