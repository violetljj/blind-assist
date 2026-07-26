from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import cv2
import numpy as np

from .protocol import TrialSpec


@dataclass(frozen=True)
class SyntheticSequence:
    frames: tuple[np.ndarray, ...]
    valid_masks: tuple[np.ndarray, ...]
    occlusion_masks: tuple[np.ndarray, ...]
    timestamps_seconds: tuple[float, ...]
    rotation_current_from_previous: np.ndarray
    rotation_homography_previous_to_current: np.ndarray
    pair_homography_previous_to_current: np.ndarray
    scale_factor_per_pair: float
    base_sha256: str
    sequence_sha256: str


def camera_matrix(protocol: dict[str, Any]) -> np.ndarray:
    values = protocol["rendering"]["intrinsics"]
    return np.asarray(
        [
            [values["fx_pixels"], 0.0, values["cx_pixels"]],
            [0.0, values["fy_pixels"], values["cy_pixels"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def axis_rotation(axis: str, angle_radians: float) -> np.ndarray:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    if axis == "yaw":
        return np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    if axis == "pitch":
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    if axis == "roll":
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if axis == "none" and abs(angle_radians) < 1e-15:
        return np.eye(3, dtype=np.float64)
    raise ValueError(f"unsupported rotation axis: {axis}")


def scale_about_principal_point(
    scale: float, cx: float, cy: float
) -> np.ndarray:
    return np.asarray(
        [
            [scale, 0.0, cx * (1.0 - scale)],
            [0.0, scale, cy * (1.0 - scale)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_base_texture(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(
        35, 221, size=(max(8, height // 12), max(8, width // 12)), dtype=np.uint8
    )
    image = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    for row in range(0, height, max(24, height // 12)):
        cv2.line(image, (0, row), (width - 1, row), 40 + row % 160, 1)
    for column in range(0, width, max(24, width // 16)):
        cv2.line(image, (column, 0), (column, height - 1), 210, 1)
    for _ in range(180):
        center = (
            int(rng.integers(4, max(5, width - 4))),
            int(rng.integers(4, max(5, height - 4))),
        )
        radius = int(rng.integers(2, 8))
        color = int(rng.integers(15, 241))
        cv2.circle(image, center, radius, color, -1, lineType=cv2.LINE_AA)
    for _ in range(50):
        point_a = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        point_b = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        cv2.line(
            image,
            point_a,
            point_b,
            int(rng.integers(20, 236)),
            int(rng.integers(1, 3)),
            lineType=cv2.LINE_AA,
        )
    return image


def _apply_degradation(
    image: np.ndarray,
    degradation: str,
    frame_index: int,
    timestamp: float,
    seed: int,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    if degradation == "clean":
        return image.copy(), mask
    if degradation == "gaussian_noise":
        sigma = protocol["degradations"]["gaussian_noise"][
            "sigma_intensity_0_255"
        ]
        rng = np.random.default_rng(seed * 1009 + frame_index * 9176 + 11)
        noise = rng.normal(0.0, sigma, size=image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0.0, 255.0)
        return noisy.astype(np.uint8), mask
    if degradation == "gaussian_blur":
        values = protocol["degradations"]["gaussian_blur"]
        kernel = tuple(int(value) for value in values["kernel"])
        return (
            cv2.GaussianBlur(image, kernel, values["sigma_pixels"]),
            mask,
        )
    if degradation == "partial_occlusion":
        values = protocol["degradations"]["partial_occlusion"]
        fraction = float(values["area_fraction"])
        rect_width = max(8, int(round(width * math.sqrt(fraction))))
        rect_height = max(8, int(round(height * math.sqrt(fraction))))
        rng = np.random.default_rng(seed * 7919 + 23)
        x0 = int(rng.integers(0, max(1, width - rect_width)))
        y0 = int(rng.integers(0, max(1, height - rect_height)))
        direction = -1 if seed % 2 else 1
        shift = int(
            round(
                direction
                * values["horizontal_speed_pixels_per_s"]
                * timestamp
            )
        )
        x0 = int(np.clip(x0 + shift, 0, max(0, width - rect_width)))
        x1 = min(width, x0 + rect_width)
        y1 = min(height, y0 + rect_height)
        patch_rng = np.random.default_rng(seed * 3571 + frame_index * 13)
        patch = patch_rng.integers(
            25, 231, size=(y1 - y0, x1 - x0), dtype=np.uint8
        )
        patch = cv2.GaussianBlur(patch, (3, 3), 0.4)
        result = image.copy()
        result[y0:y1, x0:x1] = patch
        mask[y0:y1, x0:x1] = 255
        return result, mask
    raise ValueError(f"unsupported degradation: {degradation}")


def generate_sequence(
    spec: TrialSpec, protocol: dict[str, Any]
) -> SyntheticSequence:
    rendering = protocol["rendering"]
    width = int(rendering["width"])
    height = int(rendering["height"])
    duration = float(rendering["duration_seconds"])
    pair_count = int(round(duration * spec.fps))
    if pair_count <= 0:
        raise ValueError("NON_POSITIVE_PAIR_COUNT")
    timestamps = tuple(index / spec.fps for index in range(pair_count + 1))
    dt = 1.0 / spec.fps
    base = make_base_texture(width, height, spec.seed)
    base_hash = hashlib.sha256(base.tobytes()).hexdigest()

    rotation = axis_rotation(
        spec.axis,
        math.radians(spec.angular_velocity_deg_per_s) * dt,
    )
    intrinsics = camera_matrix(protocol)
    rotation_h = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    values = rendering["intrinsics"]
    scale_factor = math.exp(spec.scale_rate_per_s * dt)
    scale_h = scale_about_principal_point(
        scale_factor,
        values["cx_pixels"],
        values["cy_pixels"],
    )
    pair_h = rotation_h @ scale_h

    cumulative = np.eye(3, dtype=np.float64)
    frames: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    occlusion_masks: list[np.ndarray] = []
    digest = hashlib.sha256()
    source_valid = np.full((height, width), 255, dtype=np.uint8)
    for frame_index, timestamp in enumerate(timestamps):
        rendered = cv2.warpPerspective(
            base,
            cumulative,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpPerspective(
            source_valid,
            cumulative,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        degraded, occlusion = _apply_degradation(
            rendered,
            spec.degradation,
            frame_index,
            timestamp,
            spec.seed,
            protocol,
        )
        degraded = np.ascontiguousarray(degraded)
        valid = np.ascontiguousarray(valid)
        frames.append(degraded)
        valid_masks.append(valid)
        occlusion_masks.append(occlusion)
        digest.update(degraded.tobytes())
        digest.update(valid.tobytes())
        if frame_index < pair_count:
            cumulative = pair_h @ cumulative

    return SyntheticSequence(
        frames=tuple(frames),
        valid_masks=tuple(valid_masks),
        occlusion_masks=tuple(occlusion_masks),
        timestamps_seconds=timestamps,
        rotation_current_from_previous=rotation,
        rotation_homography_previous_to_current=rotation_h,
        pair_homography_previous_to_current=pair_h,
        scale_factor_per_pair=scale_factor,
        base_sha256=base_hash,
        sequence_sha256=digest.hexdigest(),
    )
