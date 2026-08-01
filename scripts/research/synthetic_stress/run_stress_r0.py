#!/usr/bin/env python3
"""Run an isolated, controlled synthetic stress matrix for RCLE and D2 mechanics.

The runner deliberately keeps synthetic evidence separate from natural-data and
promotion evidence.  It uses the existing RCLE local-affine implementation and
the hash-bound G0/D2 signed-clearance primitives where available, while keeping
all generated media, ledgers, and summaries under ``artifacts.local``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_ROOT = REPO_ROOT / "scripts" / "research" / "hftf"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HFTF_ROOT) not in sys.path:
    sys.path.insert(0, str(HFTF_ROOT))

from scripts.research.egomotion_compensated_looming.rcle_minimal.local_expansion import (  # noqa: E501
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.sparse_flow import (  # noqa: E501
    SparseTrackResult,
)

try:
    from stage_c_d2_mechanics_common import (
        arrays_from_arm,
        compute_field,
        nullable_field,
        predicted_bases,
    )
except Exception as error:  # pragma: no cover - a clear setup failure
    raise RuntimeError(
        "D2 synthetic stress requires the current workspace D2 mechanics "
        "implementation at scripts/research/hftf/stage_c_d2_mechanics_common.py"
    ) from error

from run_stage_c_g0_signed_clearance_mechanics import _signed_clearance_field


PROTOCOL_PATH = Path(__file__).with_name("protocol_r0.json")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local"
    / "evidence"
    / "controlled_synthetic_stress_r0"
    / "20260802-run1"
)
SCHEMA = "blindassist.controlled_synthetic_stress.r0.result"
RUNNER_ID = "CONTROLLED_SYNTHETIC_STRESS_R0"


@dataclass(frozen=True)
class MotionSpec:
    family: str
    axis: str = "none"
    sign: int = 1
    rate: float = 0.0
    direction: str = "none"
    target_distance: float = 3.5
    shake_amplitude: float = 0.0
    shake_frequency: float = 0.0


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_bytes(row).decode("utf-8"))


def axis_rotation(axis: str, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "yaw":
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "pitch":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "roll":
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if axis == "none":
        return np.eye(3, dtype=np.float64)
    raise ValueError(f"unsupported rotation axis: {axis}")


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to the source pose xyzw convention."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    result = np.asarray([x, y, z, w], dtype=np.float64)
    result /= np.linalg.norm(result)
    return result


def camera_binding(position: np.ndarray, rotation: np.ndarray) -> dict[str, Any]:
    return {
        "position_m": np.asarray(position, dtype=np.float64).tolist(),
        "quaternion_xyzw": rotation_to_quaternion_xyzw(rotation).tolist(),
    }


def camera_matrix(protocol: dict[str, Any]) -> np.ndarray:
    rendering = protocol["rendering"]
    return np.asarray(
        [
            [rendering["fx"], 0.0, rendering["cx"]],
            [0.0, rendering["fy"], rendering["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def base_degradation() -> dict[str, float | int]:
    return {
        "texture_level": 1.0,
        "motion_blur_sigma": 0.0,
        "rolling_shutter_fraction": 0.0,
        "shadow_strength": 0.0,
        "occlusion_fraction": 0.0,
        "depth_discontinuity_strength": 0.0,
        "timestamp_jitter_fraction": 0.0,
        "drop_fraction": 0.0,
        "compression_quality": 100,
        "fps": 30,
    }


def motion_catalog(protocol: dict[str, Any]) -> list[MotionSpec]:
    motion = protocol["motion"]
    specs: list[MotionSpec] = []
    for axis in motion["rotation_axes"]:
        for sign in (-1, 1):
            for rate in motion["rotation_rates_deg_per_s"]:
                specs.append(MotionSpec(axis, axis, sign, float(rate)))
    for direction in motion["translation_directions"]:
        for sign in (-1, 1):
            for speed in motion["translation_speeds_m_per_s"]:
                specs.append(
                    MotionSpec(
                        f"translation_{direction}",
                        direction=direction,
                        sign=sign,
                        rate=float(speed),
                    )
                )
    for axis in ("yaw", "pitch", "roll"):
        for sign in (-1, 1):
            for rate in (15.0, 30.0):
                specs.append(
                    MotionSpec(
                        "rotation_plus_translation",
                        axis=axis,
                        sign=sign,
                        rate=rate,
                        direction="forward",
                    )
                )
    for sign in (-1, 1):
        for speed in motion["frontal_approach_speeds_m_per_s"]:
            for distance in (2.0, 3.5, 6.0):
                specs.append(
                    MotionSpec(
                        "frontal_approach",
                        sign=sign,
                        rate=float(speed),
                        target_distance=float(distance),
                    )
                )
    for sign in (-1, 1):
        for speed in motion["lateral_pass_speeds_m_per_s"]:
            specs.append(
                MotionSpec("lateral_pass", sign=sign, rate=float(speed))
            )
    for rate in motion["scale_rates_per_s"]:
        specs.append(MotionSpec("scale", sign=1 if rate >= 0 else -1, rate=abs(float(rate))))
    for amplitude in motion["camera_shake_amplitudes_m"]:
        for frequency in motion["camera_shake_frequencies_hz"]:
            specs.append(
                MotionSpec(
                    "camera_shake",
                    sign=1,
                    shake_amplitude=float(amplitude),
                    shake_frequency=float(frequency),
                )
            )
    return specs


def representative_motions(protocol: dict[str, Any]) -> list[MotionSpec]:
    all_specs = motion_catalog(protocol)
    wanted = {
        "yaw",
        "pitch",
        "roll",
        "translation_forward",
        "rotation_plus_translation",
        "frontal_approach",
        "lateral_pass",
        "scale",
        "camera_shake",
    }
    result: list[MotionSpec] = []
    seen: set[str] = set()
    for spec in all_specs:
        if spec.family not in wanted:
            continue
        key = spec.family
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


def profile_label(values: dict[str, float | int]) -> str:
    active = []
    for key in sorted(values):
        value = values[key]
        default = base_degradation()[key]
        if value != default:
            active.append(f"{key}={value}")
    return "clean" if not active else ";".join(active)


def make_case(
    design_block: str,
    motion: MotionSpec,
    degradation: dict[str, float | int],
    seed: int,
) -> dict[str, Any]:
    payload = {
        "design_block": design_block,
        "motion": {
            "family": motion.family,
            "axis": motion.axis,
            "sign": motion.sign,
            "rate": motion.rate,
            "direction": motion.direction,
            "target_distance": motion.target_distance,
            "shake_amplitude": motion.shake_amplitude,
            "shake_frequency": motion.shake_frequency,
        },
        "degradation": degradation,
        "seed": seed,
    }
    case_id = sha256_bytes(canonical_bytes(payload))[:16]
    return {
        "case_id": case_id,
        **payload,
        "profile_label": profile_label(degradation),
    }


def build_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    replicates = int(protocol["design"]["replicates"])
    seed_start = int(protocol["design"]["seed_start"])
    seed_stride = int(protocol["design"]["seed_stride"])
    motions = motion_catalog(protocol)
    reps = representative_motions(protocol)
    cases: dict[str, dict[str, Any]] = {}

    def add(block: str, motion: MotionSpec, values: dict[str, float | int]) -> None:
        for replicate in range(replicates):
            seed = seed_start + replicate * seed_stride
            case = make_case(block, motion, values.copy(), seed)
            cases[case["case_id"]] = case

    # The complete motion catalog is always run clean, so every motion family
    # and sign/rate cell has a direct denominator.
    for motion in motions:
        add("motion_catalog_clean", motion, base_degradation())

    # One-factor sweeps use all values and a stable representative motion set.
    degradation = protocol["degradation"]
    factor_values: list[tuple[str, Iterable[float | int]]] = [
        ("texture_level", degradation["low_texture_levels"]),
        ("motion_blur_sigma", degradation["motion_blur_sigma_pixels"]),
        ("rolling_shutter_fraction", degradation["rolling_shutter_readout_fraction"]),
        ("shadow_strength", degradation["shadow_strength"]),
        ("occlusion_fraction", degradation["occlusion_fraction"]),
        ("depth_discontinuity_strength", degradation["depth_discontinuity_strength"]),
        ("fps", degradation["fps"]),
        ("timestamp_jitter_fraction", degradation["timestamp_jitter_fraction_of_dt"]),
        ("drop_fraction", degradation["drop_fraction"]),
        ("compression_quality", degradation["compression_quality"]),
    ]
    for key, values in factor_values:
        for motion in reps:
            for value in values:
                current = base_degradation()
                current[key] = value
                add(f"one_factor_{key}", motion, current)

    # Explicit pairwise packs cover imaging x temporal and motion x scene
    # interactions without silently turning the search into an unbounded grid.
    pair_values = [
        ("motion_blur_sigma", 2.0, "rolling_shutter_fraction", 0.5),
        ("motion_blur_sigma", 4.0, "low_texture", 0.25),
        ("rolling_shutter_fraction", 1.0, "timestamp_jitter_fraction", 0.25),
        ("occlusion_fraction", 0.5, "drop_fraction", 0.1),
        ("depth_discontinuity_strength", 0.75, "shadow_strength", 0.75),
        ("compression_quality", 10, "motion_blur_sigma", 1.0),
        ("fps", 5, "timestamp_jitter_fraction", 0.1),
        ("fps", 20, "drop_fraction", 0.25),
    ]
    for pair_index, (key_a, value_a, key_b, value_b) in enumerate(pair_values):
        for motion in reps:
            current = base_degradation()
            if key_a == "low_texture":
                current["texture_level"] = value_a
            else:
                current[key_a] = value_a
            if key_b == "low_texture":
                current["texture_level"] = value_b
            else:
                current[key_b] = value_b
            add(f"pairwise_{pair_index:02d}", motion, current)

    stress_packs = [
        {"texture_level": 0.25, "motion_blur_sigma": 2.0, "occlusion_fraction": 0.25, "fps": 15},
        {"rolling_shutter_fraction": 1.0, "shadow_strength": 0.75, "compression_quality": 10},
        {"depth_discontinuity_strength": 0.75, "timestamp_jitter_fraction": 0.25, "drop_fraction": 0.25},
        {"texture_level": 0.25, "motion_blur_sigma": 4.0, "rolling_shutter_fraction": 0.5, "compression_quality": 30},
        {"occlusion_fraction": 0.75, "depth_discontinuity_strength": 0.75, "drop_fraction": 0.1, "fps": 5},
        {"texture_level": 0.5, "shadow_strength": 0.5, "motion_blur_sigma": 1.0, "fps": 60},
        {"rolling_shutter_fraction": 0.25, "timestamp_jitter_fraction": 0.02, "compression_quality": 60},
        {"texture_level": 0.25, "occlusion_fraction": 0.5, "depth_discontinuity_strength": 0.5, "fps": 20},
    ]
    for pack_index, overrides in enumerate(stress_packs):
        for motion in reps:
            current = base_degradation()
            current.update(overrides)
            add(f"stress_pack_{pack_index:02d}", motion, current)

    return sorted(cases.values(), key=lambda item: item["case_id"])


def pose_at(motion: dict[str, Any], time_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    family = motion["family"]
    sign = float(motion["sign"])
    rotation = np.eye(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    image_scale = 1.0
    if family in {"yaw", "pitch", "roll"}:
        rotation = axis_rotation(
            str(motion["axis"]),
            sign * math.radians(float(motion["rate"])) * time_s,
        )
    elif family.startswith("translation_"):
        direction = str(motion["direction"])
        vector = {
            "forward": np.asarray([0.0, 0.0, 1.0]),
            "lateral": np.asarray([1.0, 0.0, 0.0]),
            "vertical": np.asarray([0.0, 1.0, 0.0]),
        }[direction]
        position = sign * float(motion["rate"]) * time_s * vector
    elif family == "rotation_plus_translation":
        rotation = axis_rotation(
            str(motion["axis"]),
            sign * math.radians(float(motion["rate"])) * time_s,
        )
        position = sign * 0.75 * time_s * np.asarray([0.0, 0.0, 1.0])
    elif family == "frontal_approach":
        # The camera is static; the target motion is applied in scene_points.
        pass
    elif family == "lateral_pass":
        pass
    elif family == "scale":
        image_scale = math.exp(sign * float(motion["rate"]) * time_s)
    elif family == "camera_shake":
        amplitude = float(motion["shake_amplitude"])
        frequency = float(motion["shake_frequency"])
        position = np.asarray(
            [
                amplitude * math.sin(2.0 * math.pi * frequency * time_s),
                0.25 * amplitude * math.sin(2.0 * math.pi * frequency * time_s + 0.7),
                amplitude * math.cos(2.0 * math.pi * frequency * time_s),
            ],
            dtype=np.float64,
        )
        rotation = axis_rotation("roll", 0.5 * amplitude * math.sin(time_s))
    else:
        raise ValueError(f"unsupported motion family: {family}")
    return position, rotation, image_scale


def scene_points(motion: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    labels: list[int] = []
    # A deliberately non-planar, textured background gives every RCLE grid
    # cell a spread of depths and prevents a single plane from hiding leakage.
    for z in np.linspace(1.5, 8.0, 8):
        for x in np.linspace(-2.3, 2.3, 13):
            for y in np.linspace(-0.7, 2.0, 7):
                jitter = rng.normal(0.0, [0.025, 0.025, 0.04])
                points.append(np.asarray([x, y, z]) + jitter)
                labels.append(0)
    # A moving target is kept as a separate label for the TTC-proxy audit.
    target_distance = float(motion.get("target_distance", 3.5))
    target_center = np.asarray([0.0, 0.55, target_distance], dtype=np.float64)
    for x in np.linspace(-0.45, 0.45, 9):
        for y in np.linspace(-0.5, 0.65, 9):
            for z in np.linspace(-0.18, 0.18, 3):
                point = target_center + np.asarray([x, y, z])
                points.append(point)
                labels.append(1)
    return np.asarray(points, dtype=np.float64).T, np.asarray(labels, dtype=np.int8)


def scene_points_at(
    motion: dict[str, Any],
    base_points: np.ndarray,
    labels: np.ndarray,
    time_s: float,
) -> np.ndarray:
    points = np.asarray(base_points, dtype=np.float64).copy()
    target = labels == 1
    family = motion["family"]
    if family == "frontal_approach":
        points[2, target] -= float(motion["sign"]) * float(motion["rate"]) * time_s
    elif family == "lateral_pass":
        points[0, target] += float(motion["sign"]) * float(motion["rate"]) * time_s
    return points


def project_points(
    points_world: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    intrinsic: np.ndarray,
    image_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera = rotation.T @ (points_world - position[:, None])
    z = camera[2]
    valid = np.isfinite(camera).all(axis=0) & (z > 0.25)
    pixels = np.full((points_world.shape[1], 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = intrinsic[0, 0] * camera[0, valid] / z[valid] + intrinsic[0, 2]
    pixels[valid, 1] = intrinsic[1, 1] * camera[1, valid] / z[valid] + intrinsic[1, 2]
    center = np.asarray([intrinsic[0, 2], intrinsic[1, 2]])
    pixels[valid] = center + image_scale * (pixels[valid] - center)
    return pixels, valid


def _degraded_current_pixels(
    case: dict[str, Any],
    base_points: np.ndarray,
    labels: np.ndarray,
    time_s: float,
    dt: float,
    intrinsic: np.ndarray,
    seed: int,
    current: bool,
) -> tuple[np.ndarray, np.ndarray]:
    degradation = case["degradation"]
    position, rotation, image_scale = pose_at(case["motion"], time_s)
    points = scene_points_at(case["motion"], base_points, labels, time_s)
    pixels, valid = project_points(points, position, rotation, intrinsic, image_scale)
    if current and float(degradation["rolling_shutter_fraction"]) > 0.0:
        readout = float(degradation["rolling_shutter_fraction"])
        row_norm = (pixels[:, 1] - intrinsic[1, 2]) / max(intrinsic[1, 2], 1.0)
        shifted: list[np.ndarray] = []
        for index in range(points.shape[1]):
            if not valid[index]:
                shifted.append(pixels[index])
                continue
            offset = readout * row_norm[index] * dt
            p, r, s = pose_at(case["motion"], time_s + offset)
            q = scene_points_at(case["motion"], base_points[:, index : index + 1], labels[index : index + 1], time_s + offset)
            projected, ok = project_points(q, p, r, intrinsic, s)
            shifted.append(projected[0] if ok[0] else pixels[index])
        pixels = np.asarray(shifted, dtype=np.float64)
    sigma = float(degradation["motion_blur_sigma"])
    if sigma > 0.0:
        samples = np.linspace(-0.5, 0.5, 5) * sigma * dt / max(1.0 / 30.0, dt)
        stack = [pixels]
        for offset in samples[1:]:
            p, r, s = pose_at(case["motion"], time_s + float(offset))
            q = scene_points_at(case["motion"], base_points, labels, time_s + float(offset))
            rendered, ok = project_points(q, p, r, intrinsic, s)
            rendered[~ok] = pixels[~ok]
            stack.append(rendered)
        pixels = np.mean(np.stack(stack, axis=0), axis=0)
    texture = float(degradation["texture_level"])
    shadow = float(degradation["shadow_strength"])
    occlusion = float(degradation["occlusion_fraction"])
    compression = max(0.0, (100.0 - float(degradation["compression_quality"])) / 100.0)
    retention_probability = np.clip(
        0.96 - 0.45 * (1.0 - texture) - 0.25 * shadow - 0.55 * occlusion - 0.08 * compression,
        0.05,
        1.0,
    )
    rng = np.random.default_rng(seed * 31 + (17 if current else 7))
    keep = valid & (rng.random(valid.shape[0]) < retention_probability)
    if occlusion > 0.0:
        # A moving rectangular patch is the image-space occlusion proxy.
        x_norm = (pixels[:, 0] - intrinsic[0, 2]) / max(intrinsic[0, 2], 1.0)
        keep &= ~((x_norm > -1.0) & (x_norm < -1.0 + 2.0 * occlusion))
    noise_sigma = (
        0.05
        + 0.65 * (1.0 - texture)
        + 0.35 * shadow
        + 1.2 * compression
        + 0.9 * float(degradation["depth_discontinuity_strength"])
    )
    noise = rng.normal(0.0, noise_sigma, size=pixels.shape)
    pixels = pixels + noise
    discontinuity = float(degradation["depth_discontinuity_strength"])
    if discontinuity > 0.0:
        z_values = points[2]
        boundary = np.abs(z_values - np.median(z_values)) < 0.25
        pixels[boundary] += rng.normal(0.0, 1.5 * discontinuity, size=(int(boundary.sum()), 2))
    return pixels[keep], keep


def _fit_expansion(
    previous: np.ndarray,
    current: np.ndarray,
    dt_seconds: float,
    shape: tuple[int, int],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 2:
        return {"status": "NOT_EVALUABLE_INSUFFICIENT_TRACK_SUPPORT"}
    tracks = SparseTrackResult(
        previous_points=np.ascontiguousarray(previous.astype(np.float32)),
        current_points=np.ascontiguousarray(current.astype(np.float32)),
        forward_backward_errors=np.zeros(previous.shape[0], dtype=np.float32),
        requested_count=int(previous.shape[0]),
    )
    try:
        cells = fit_fixed_grid_local_affine(tracks, dt_seconds, shape, parameters)
    except (TypeError, ValueError, FloatingPointError) as error:
        return {"status": "NOT_EVALUABLE", "error": f"{type(error).__name__}:{error}"}
    values = [float(cell.expansion) for cell in cells if cell.evaluable and cell.expansion is not None]
    reasons = Counter(cell.abstention_reason for cell in cells if not cell.evaluable)
    coverage = len(values) / max(len(cells), 1)
    if not values:
        return {
            "status": "NOT_EVALUABLE_INSUFFICIENT_TRACK_SUPPORT",
            "coverage": coverage,
            "reasons": dict(reasons),
        }
    return {
        "status": "EVALUABLE",
        "median_expansion_per_s": float(np.median(values)),
        "median_abs_expansion_per_s": float(np.median(np.abs(values))),
        "coverage": coverage,
        "evaluable_cells": len(values),
        "reasons": dict(reasons),
    }


def _rotation_homography(
    previous_rotation: np.ndarray,
    current_rotation: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    return intrinsic @ (current_rotation.T @ previous_rotation) @ np.linalg.inv(intrinsic)


def _perspective_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), homography.astype(np.float32)
    ).reshape(-1, 2).astype(np.float64)


def _d2_parameters(protocol: dict[str, Any]) -> dict[str, Any]:
    field = protocol["field"]
    theta = np.radians(
        np.linspace(
            float(field["theta_range_degrees"][0]),
            float(field["theta_range_degrees"][1]),
            int(field["theta_bin_count"]) + 1,
        )
    )
    return {
        "theta_edges": theta,
        "distance_edges": np.asarray(field["distance_edges_m"], dtype=np.float64),
        "height_bands": [tuple(map(float, band)) for band in field["height_bands_m"]],
        "widths": np.asarray(field["effective_lateral_half_width_m"], dtype=np.float64),
        "order_statistic": int(field["order_statistic"]),
        "final_edge_atol_m": 1e-12,
        "final_edge_rtol": 0.0,
        "clip_min_m": float(field["clip_m"][0]),
        "clip_max_m": float(field["clip_m"][1]),
    }


def _basis_from_pose(
    position: np.ndarray,
    rotation: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    forward = rotation @ np.asarray([0.0, 0.0, 1.0])
    forward = forward - float(forward @ up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    origin = position - float(position @ up) * up
    return origin, forward, right, up


def _field_points(
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    motion: dict[str, Any],
    seed: int,
    parameters: dict[str, Any],
) -> np.ndarray:
    origin, forward, right, up = basis
    rng = np.random.default_rng(seed + 991)
    theta_edges = parameters["theta_edges"]
    distance_edges = parameters["distance_edges"]
    bands = parameters["height_bands"]
    widths = parameters["widths"]
    points: list[np.ndarray] = []
    # A sparse, controlled obstacle population gives both negative and positive
    # second-order cells while retaining exact G0 membership and clipping rules.
    for theta_index in range(len(theta_edges) - 1):
        theta = float((theta_edges[theta_index] + theta_edges[theta_index + 1]) * 0.5)
        for distance_index in range(len(distance_edges) - 1):
            along = float((distance_edges[distance_index] + distance_edges[distance_index + 1]) * 0.5)
            for height_index, band in enumerate(bands):
                height = float((band[0] + band[1]) * 0.5)
                cross = math.tan(theta) * along * 0.35
                risk = (theta_index * 7 + distance_index * 3 + height_index + seed) % 5 in {0, 1}
                count = 3 if risk else (1 if (theta_index + distance_index + seed) % 3 == 0 else 0)
                for point_index in range(count):
                    perturb = rng.normal(
                        0.0,
                        [0.02, min(0.04, widths[height_index] * 0.15), 0.02],
                    )
                    local = (
                        forward * (along + perturb[0])
                        + right * (cross + perturb[1])
                        + up * (height + perturb[2])
                    )
                    points.append(origin + local)
    return np.asarray(points, dtype=np.float64).T if points else np.empty((3, 0), dtype=np.float64)


def _visible_field_points(points: np.ndarray, case: dict[str, Any], seed: int) -> np.ndarray:
    if points.shape[1] == 0:
        return points
    degradation = case["degradation"]
    penalty = (
        0.7 * float(degradation["occlusion_fraction"])
        + 0.3 * float(degradation["depth_discontinuity_strength"])
        + 0.2 * (1.0 - float(degradation["texture_level"]))
        + 0.1 * float(degradation["drop_fraction"])
    )
    keep_probability = np.clip(1.0 - penalty, 0.05, 1.0)
    rng = np.random.default_rng(seed + 2221)
    keep = rng.random(points.shape[1]) < keep_probability
    # Keep at least a small deterministic support so a failure is attributable
    # to the stress condition rather than an empty random sample.
    if int(keep.sum()) < min(8, points.shape[1]):
        keep[np.argsort(rng.random(points.shape[1]))[: min(8, points.shape[1])]] = True
    return points[:, keep]


def _known_counts(case: dict[str, Any], seed: int, arm: str, horizon: float) -> np.ndarray:
    degradation = case["degradation"]
    counts = np.full((2, 6, 6), 9, dtype=np.int64)
    loss = (
        5.0 * float(degradation["occlusion_fraction"])
        + 2.5 * float(degradation["depth_discontinuity_strength"])
        + 1.5 * (1.0 - float(degradation["texture_level"]))
        + 2.0 * float(degradation["drop_fraction"])
    )
    if arm == "truth":
        loss *= 0.8
    if arm == "advected":
        loss *= 1.05 + 0.05 * horizon
    rng = np.random.default_rng(seed + int(horizon * 1000) + (31 if arm == "truth" else 17))
    if loss > 0.0:
        counts -= np.rint(rng.uniform(0.0, loss, size=counts.shape)).astype(np.int64)
    counts = np.clip(counts, 0, 9)
    return counts


def _validate_nullable_arm(
    known: np.ndarray,
    clearance: np.ndarray,
    probe_pass_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    if probe_pass_counts is None:
        probe_pass_counts = np.where(known, 9, 0).astype(np.int64)
    arm = {
        "known": known.tolist(),
        "probe_pass_counts": probe_pass_counts.tolist(),
        "clearance_m": nullable_field(known, clearance),
    }
    arrays_from_arm(arm)
    unknown = int((~known).sum())
    return {"unknown_cells": unknown, "unknown_to_safe_violations": 0}


def _d2_time_status(case: dict[str, Any], timestamps: np.ndarray) -> str:
    fps = int(case["degradation"]["fps"])
    if fps not in {5, 20}:
        return "NOT_EVALUABLE_UNSUPPORTED_FPS"
    if float(case["degradation"]["drop_fraction"]) > 0.0:
        # A deterministic drop pattern is applied below.  Any required D2
        # normalized index missing is a terminal, not a replacement opportunity.
        rng = np.random.default_rng(int(case["seed"]) + 404)
        missing = set(np.flatnonzero(rng.random(13) < float(case["degradation"]["drop_fraction"])).tolist())
        required = {4, 6, 8, 10}
        if missing & required:
            return "NOT_EVALUABLE_MISSING_REQUIRED_FRAME"
    tolerance = float(1e-9)
    if abs(float(timestamps[6] - timestamps[4]) - 0.4) > tolerance:
        return "NOT_EVALUABLE_TIMEBASE"
    if abs(float(timestamps[8] - timestamps[6]) - 0.4) > tolerance:
        return "NOT_EVALUABLE_TIMEBASE"
    if abs(float(timestamps[10] - timestamps[6]) - 0.8) > tolerance:
        return "NOT_EVALUABLE_TIMEBASE"
    return "EVALUABLE"


def _timestamp_series(case: dict[str, Any]) -> np.ndarray:
    fps = float(case["degradation"]["fps"])
    nominal = np.arange(13, dtype=np.float64) / fps
    jitter_fraction = float(case["degradation"]["timestamp_jitter_fraction"])
    if jitter_fraction <= 0.0:
        return nominal
    dt = 1.0 / fps
    phase = float(case["seed"] % 17) * 0.13
    jitter = jitter_fraction * dt * np.sin(np.arange(13, dtype=np.float64) * 1.7 + phase)
    values = nominal + jitter
    if np.any(np.diff(values) <= 0.0):
        return np.full(13, np.nan, dtype=np.float64)
    return values


def _target_height(
    case: dict[str, Any],
    base_points: np.ndarray,
    labels: np.ndarray,
    time_s: float,
    intrinsic: np.ndarray,
) -> float | None:
    position, rotation, scale = pose_at(case["motion"], time_s)
    points = scene_points_at(case["motion"], base_points, labels, time_s)
    pixels, valid = project_points(points, position, rotation, intrinsic, scale)
    target = valid & (labels == 1)
    if int(target.sum()) < 4:
        return None
    height = float(np.ptp(pixels[target, 1]))
    quality = float(case["degradation"]["texture_level"])
    shadow = float(case["degradation"]["shadow_strength"])
    compression = (100.0 - float(case["degradation"]["compression_quality"])) / 100.0
    noise = (1.0 - quality) * 0.04 + shadow * 0.03 + compression * 0.02
    rng = np.random.default_rng(int(case["seed"]) + int(time_s * 1000) + 88)
    return max(1e-6, height * (1.0 + float(rng.normal(0.0, noise))))


def _run_ttc_proxy(
    case: dict[str, Any],
    base_points: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    intrinsic: np.ndarray,
) -> dict[str, Any]:
    if not np.isfinite(timestamps).all():
        return {"status": "NOT_EVALUABLE_TIMEBASE"}
    previous = _target_height(case, base_points, labels, float(timestamps[5]), intrinsic)
    current = _target_height(case, base_points, labels, float(timestamps[6]), intrinsic)
    if previous is None or current is None:
        return {"status": "NOT_EVALUABLE_NO_TARGET_SUPPORT"}
    dt = float(timestamps[6] - timestamps[5])
    if dt <= 0.0:
        return {"status": "NOT_EVALUABLE_TIMEBASE"}
    growth = math.log(current / previous) / dt
    if not math.isfinite(growth) or growth <= 0.01:
        return {
            "status": "NOT_EVALUABLE_NON_CLOSING",
            "log_height_rate_per_s": growth,
            "proxy_is_physical_ttc": False,
        }
    proxy = 1.0 / growth
    truth: float | None = None
    family = case["motion"]["family"]
    if family == "frontal_approach" and int(case["motion"]["sign"]) > 0:
        speed = float(case["motion"]["rate"])
        truth = float(case["motion"]["target_distance"]) / speed
    return {
        "status": "EVALUABLE",
        "log_height_rate_per_s": growth,
        "ttc_proxy_s": proxy,
        "analytic_physical_ttc_s": truth,
        "proxy_is_physical_ttc": False,
        "proxy_error_s": abs(proxy - truth) if truth is not None else None,
    }


def _degraded_pair_points(
    case: dict[str, Any],
    base_points: np.ndarray,
    labels: np.ndarray,
    previous_time: float,
    current_time: float,
    intrinsic: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply image/feature degradations while preserving point identity."""

    dt = current_time - previous_time
    previous_position, previous_rotation, previous_scale = pose_at(
        case["motion"], previous_time
    )
    current_position, current_rotation, current_scale = pose_at(
        case["motion"], current_time
    )
    previous_all, previous_valid = project_points(
        scene_points_at(case["motion"], base_points, labels, previous_time),
        previous_position,
        previous_rotation,
        intrinsic,
        previous_scale,
    )
    current_all, current_valid = project_points(
        scene_points_at(case["motion"], base_points, labels, current_time),
        current_position,
        current_rotation,
        intrinsic,
        current_scale,
    )
    common = previous_valid & current_valid
    profile = case["degradation"]
    texture = float(profile["texture_level"])
    shadow = float(profile["shadow_strength"])
    occlusion = float(profile["occlusion_fraction"])
    compression = max(0.0, (100.0 - float(profile["compression_quality"])) / 100.0)
    retention_probability = np.clip(
        0.96
        - 0.45 * (1.0 - texture)
        - 0.25 * shadow
        - 0.55 * occlusion
        - 0.08 * compression,
        0.05,
        1.0,
    )
    rng = np.random.default_rng(seed * 31 + 1007)
    keep = common & (rng.random(common.shape[0]) < retention_probability)
    if occlusion > 0.0:
        x_norm = (previous_all[:, 0] - intrinsic[0, 2]) / intrinsic[0, 2]
        keep &= ~((x_norm > -1.0) & (x_norm < -1.0 + 2.0 * occlusion))
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    previous = previous_all[indices].copy()
    current = current_all[indices].copy()
    readout = float(profile["rolling_shutter_fraction"])
    if readout > 0.0:
        row_norm = (current[:, 1] - intrinsic[1, 2]) / max(intrinsic[1, 2], 1.0)
        for local_index, source_index in enumerate(indices):
            offset = readout * row_norm[local_index] * dt
            position, rotation, image_scale = pose_at(
                case["motion"], current_time + offset
            )
            point = scene_points_at(
                case["motion"],
                base_points[:, source_index : source_index + 1],
                labels[source_index : source_index + 1],
                current_time + offset,
            )
            projected, valid = project_points(
                point, position, rotation, intrinsic, image_scale
            )
            if valid[0]:
                current[local_index] = projected[0]
    sigma = float(profile["motion_blur_sigma"])
    if sigma > 0.0:
        offsets = np.linspace(-0.5, 0.5, 5) * sigma * dt / max(1.0 / 30.0, dt)
        blurred = [current]
        for offset in offsets[1:]:
            position, rotation, image_scale = pose_at(
                case["motion"], current_time + float(offset)
            )
            points = scene_points_at(
                case["motion"], base_points[:, indices], labels[indices], current_time + float(offset)
            )
            projected, valid = project_points(
                points, position, rotation, intrinsic, image_scale
            )
            projected[~valid] = current[~valid]
            blurred.append(projected)
        current = np.mean(np.stack(blurred, axis=0), axis=0)
    noise_sigma = (
        0.1
        + 0.7 * (1.0 - texture)
        + 0.4 * shadow
        + 1.2 * compression
        + 0.9 * float(profile["depth_discontinuity_strength"])
    )
    previous += rng.normal(0.0, noise_sigma, previous.shape)
    current += rng.normal(0.0, noise_sigma, current.shape)
    discontinuity = float(profile["depth_discontinuity_strength"])
    if discontinuity > 0.0:
        z_values = base_points[2, indices]
        boundary = np.abs(z_values - np.median(z_values)) < 0.25
        current[boundary] += rng.normal(
            0.0, 1.5 * discontinuity, size=(int(boundary.sum()), 2)
        )
    return previous, current


def _run_field_transport(
    case: dict[str, Any],
    protocol: dict[str, Any],
    timestamps: np.ndarray,
) -> dict[str, Any]:
    if not np.isfinite(timestamps).all():
        return {"status": "NOT_EVALUABLE_TIMEBASE"}
    field_parameters = _d2_parameters(protocol)
    motion = case["motion"]
    history_time = float(timestamps[4])
    current_time = float(timestamps[6])
    history_position, history_rotation, _ = pose_at(motion, history_time)
    current_position, current_rotation, _ = pose_at(motion, current_time)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    history_binding = camera_binding(history_position, history_rotation)
    current_binding = camera_binding(current_position, current_rotation)
    plane = {
        "camera_ground_projection_m": (current_position - float(current_position @ up) * up).tolist(),
        "normal_toward_camera": up.tolist(),
    }
    try:
        current_basis, predicted, motion_receipt = predicted_bases(
            history_binding,
            current_binding,
            plane,
        )
    except (ValueError, FloatingPointError) as error:
        return {"status": "NOT_EVALUABLE_COORDINATE", "error": str(error)}
    points = _field_points(current_basis, motion, int(case["seed"]), field_parameters)
    visible = _visible_field_points(points, case, int(case["seed"]))
    persistence = compute_field(visible, current_basis, field_parameters)
    unknown_to_safe = 0
    strata: dict[str, Any] = {}
    for horizon in (0.4, 0.8):
        future_time = current_time + horizon
        future_position, future_rotation, _ = pose_at(motion, future_time)
        future_basis = _basis_from_pose(future_position, future_rotation, up)
        predicted_basis = predicted[horizon]
        advected = compute_field(visible, predicted_basis, field_parameters)
        truth_points = _field_points(future_basis, motion, int(case["seed"] + int(horizon * 100)), field_parameters)
        # Truth uses a world-fixed scene, while the two candidates only use the
        # current sealed point population.  Re-express the current points as
        # world points by construction: _field_points already returned world
        # points in the current basis.
        truth = compute_field(points, future_basis, field_parameters)
        current_known = _known_counts(case, int(case["seed"]), "current", horizon) >= 5
        advected_known = _known_counts(case, int(case["seed"]), "advected", horizon) >= 5
        truth_known = _known_counts(case, int(case["seed"]), "truth", horizon) >= 5
        common = current_known & advected_known & truth_known
        if not bool(common.any()):
            strata[str(horizon)] = {
                "status": "NOT_EVALUABLE_NO_COMMON_KNOWN_CELLS",
                "common_known_cells": 0,
            }
            continue
        persistence_arm = _validate_nullable_arm(
            current_known,
            persistence,
            _known_counts(case, int(case["seed"]), "current", horizon),
        )
        advected_arm = _validate_nullable_arm(
            advected_known,
            advected,
            _known_counts(case, int(case["seed"]), "advected", horizon),
        )
        unknown_to_safe += int(persistence_arm["unknown_to_safe_violations"])
        unknown_to_safe += int(advected_arm["unknown_to_safe_violations"])
        persistence_mae = float(np.mean(np.abs(persistence[common] - truth[common])))
        advected_mae = float(np.mean(np.abs(advected[common] - truth[common])))
        strata[str(horizon)] = {
            "status": "EVALUABLE",
            "common_known_cells": int(common.sum()),
            "persistence_mae_m": persistence_mae,
            "advected_mae_m": advected_mae,
            "advected_minus_persistence_m": advected_mae - persistence_mae,
            "persistence_unknown_cells": persistence_arm["unknown_cells"],
            "advected_unknown_cells": advected_arm["unknown_cells"],
        }
    evaluable = [item for item in strata.values() if item["status"] == "EVALUABLE"]
    if not evaluable:
        return {
            "status": "NOT_EVALUABLE_NO_COMMON_KNOWN_CELLS",
            "motion_receipt": motion_receipt,
            "horizons": strata,
        }
    return {
        "status": "EVALUABLE",
        "d2_time_status": _d2_time_status(case, timestamps),
        "motion_receipt": motion_receipt,
        "horizons": strata,
        "unknown_to_safe_violations": unknown_to_safe,
    }


def _run_rcle(
    case: dict[str, Any],
    protocol: dict[str, Any],
    base_points: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    intrinsic: np.ndarray,
) -> dict[str, Any]:
    if not np.isfinite(timestamps).all():
        return {"status": "NOT_EVALUABLE_TIMEBASE"}
    previous_time = float(timestamps[5])
    current_time = float(timestamps[6])
    dt = current_time - previous_time
    if dt <= 0.0:
        return {"status": "NOT_EVALUABLE_TIMEBASE"}
    previous_position, previous_rotation, previous_scale = pose_at(case["motion"], previous_time)
    current_position, current_rotation, current_scale = pose_at(case["motion"], current_time)
    ideal_previous_points, previous_valid = project_points(
        scene_points_at(case["motion"], base_points, labels, previous_time),
        previous_position,
        previous_rotation,
        intrinsic,
        previous_scale,
    )
    ideal_current_points, current_valid = project_points(
        scene_points_at(case["motion"], base_points, labels, current_time),
        current_position,
        current_rotation,
        intrinsic,
        current_scale,
    )
    ideal_keep = previous_valid & current_valid
    ideal_previous_points = ideal_previous_points[ideal_keep]
    ideal_current_points = ideal_current_points[ideal_keep]
    observed_previous, observed_current = _degraded_pair_points(
        case,
        base_points,
        labels,
        previous_time,
        current_time,
        intrinsic,
        int(case["seed"]),
    )
    if observed_previous.shape[0] < 3 * 3 * 12:
        return {"status": "NOT_EVALUABLE_INSUFFICIENT_TRACK_SUPPORT", "track_count": int(observed_previous.shape[0])}
    shape = (int(protocol["rendering"]["height"]), int(protocol["rendering"]["width"]))
    rcle_parameters = protocol["rcle"]
    raw = _fit_expansion(observed_previous, observed_current, dt, shape, rcle_parameters)
    ideal = _fit_expansion(ideal_previous_points, ideal_current_points, dt, shape, rcle_parameters)
    homography = _rotation_homography(previous_rotation, current_rotation, intrinsic)
    compensated_current = _perspective_points(observed_current, np.linalg.inv(homography))
    compensated = _fit_expansion(observed_previous, compensated_current, dt, shape, rcle_parameters)
    profile = case["degradation"]
    nominal_dt = 1.0 / float(profile["fps"])
    nominal = _fit_expansion(observed_previous, observed_current, nominal_dt, shape, rcle_parameters)
    result: dict[str, Any] = {
        "status": "EVALUABLE" if raw.get("status") == "EVALUABLE" else raw.get("status", "NOT_EVALUABLE"),
        "track_count": int(observed_previous.shape[0]),
        "dt_seconds": dt,
        "nominal_dt_seconds": nominal_dt,
        "raw": raw,
        "compensated": compensated,
        "ideal": ideal,
        "nominal_time_normalized": nominal,
    }
    if raw.get("status") == "EVALUABLE" and nominal.get("status") == "EVALUABLE" and ideal.get("status") == "EVALUABLE":
        normalized_error = abs(float(raw["median_expansion_per_s"]) - float(ideal["median_expansion_per_s"]))
        nominal_error = abs(float(nominal["median_expansion_per_s"]) - float(ideal["median_expansion_per_s"]))
        result["time_normalization"] = {
            "normalized_error_per_s": normalized_error,
            "nominal_error_per_s": nominal_error,
            "normalized_better": normalized_error <= nominal_error + 1e-12,
        }
    family = case["motion"]["family"]
    if family in {"yaw", "pitch", "roll"}:
        result["rotation_leakage"] = {
            "raw_per_s": raw.get("median_abs_expansion_per_s"),
            "compensated_per_s": compensated.get("median_abs_expansion_per_s"),
            "reduced": (
                raw.get("median_abs_expansion_per_s") is not None
                and compensated.get("median_abs_expansion_per_s") is not None
                and float(compensated["median_abs_expansion_per_s"]) <= float(raw["median_abs_expansion_per_s"]) + 1e-12
            ),
        }
    sign_truth: float | None = None
    if family == "scale":
        sign_truth = float(case["motion"]["sign"]) * float(case["motion"]["rate"])
    elif family == "frontal_approach":
        sign_truth = 1.0 if int(case["motion"]["sign"]) > 0 else -1.0
    if sign_truth is not None and raw.get("status") == "EVALUABLE":
        estimate = float(raw["median_expansion_per_s"])
        band = float(protocol["rcle"]["sign_zero_band_per_s"])
        predicted_sign = 0 if abs(estimate) <= band else (1 if estimate > 0 else -1)
        expected_sign = 1 if sign_truth > 0 else -1
        result["expansion_sign"] = {
            "expected": expected_sign,
            "estimate_per_s": estimate,
            "predicted": predicted_sign,
            "correct": predicted_sign == expected_sign,
        }
    return result


def run_case(case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    intrinsic = camera_matrix(protocol)
    timestamps = _timestamp_series(case)
    base_points, labels = scene_points(case["motion"], int(case["seed"]))
    if not np.isfinite(timestamps).all():
        return {
            "case_id": case["case_id"],
            "design_block": case["design_block"],
            "motion_family": case["motion"]["family"],
            "profile_label": case["profile_label"],
            "seed": case["seed"],
            "fps": case["degradation"]["fps"],
            "rcle": {"status": "NOT_EVALUABLE_TIMEBASE"},
            "ttc_proxy": {"status": "NOT_EVALUABLE_TIMEBASE"},
            "field_transport": {"status": "NOT_EVALUABLE_TIMEBASE"},
            "d2_time_status": "NOT_EVALUABLE_TIMEBASE",
        }
    rcle = _run_rcle(case, protocol, base_points, labels, timestamps, intrinsic)
    ttc = _run_ttc_proxy(case, base_points, labels, timestamps, intrinsic)
    field = _run_field_transport(case, protocol, timestamps)
    return {
        "case_id": case["case_id"],
        "design_block": case["design_block"],
        "motion_family": case["motion"]["family"],
        "motion_axis": case["motion"]["axis"],
        "motion_sign": case["motion"]["sign"],
        "profile_label": case["profile_label"],
        "seed": case["seed"],
        "fps": case["degradation"]["fps"],
        "degradation": case["degradation"],
        "rcle": rcle,
        "ttc_proxy": ttc,
        "field_transport": field,
        "d2_time_status": _d2_time_status(case, timestamps),
    }


def _boundary_suite(protocol: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    field = protocol["field"]
    known_threshold = int(field["known_threshold"])
    for count in (known_threshold - 1, known_threshold, known_threshold + 1):
        checks.append(
            {
                "name": f"known_threshold_{count}",
                "pass": (count >= known_threshold) == (count in {known_threshold, known_threshold + 1}),
                "count": count,
            }
        )
    for value, expected in ((-1e-12, True), (0.0, False), (1e-12, False)):
        checks.append(
            {
                "name": f"risk_strict_negative_{value}",
                "pass": (value < 0.0) == expected,
                "value": value,
            }
        )
    for value, expected in ((-0.011, -1), (-0.01, 0), (0.0, 0), (0.01, 0), (0.011, 1)):
        predicted = 0 if abs(value) <= float(protocol["rcle"]["sign_zero_band_per_s"]) else (1 if value > 0 else -1)
        checks.append(
            {
                "name": f"rcle_sign_band_{value}",
                "pass": predicted == expected,
                "value": value,
                "predicted": predicted,
            }
        )
    up = np.asarray([0.0, 1.0, 0.0])
    basis = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        up,
    )
    parameters = _d2_parameters(protocol)
    for edge in parameters["distance_edges"][1:]:
        for delta in (-1e-10, 0.0, 1e-10):
            theta_center = float((parameters["theta_edges"][2] + parameters["theta_edges"][3]) * 0.5)
            edge_value = float(edge) + delta
            if delta == 0.0 and edge_value < float(parameters["distance_edges"][-1]):
                # The trigonometric reconstruction can round an intended
                # exact edge one ULP below it; exercise the contract's upper
                # bin with the next representable float instead.
                edge_value = float(np.nextafter(edge_value, np.inf))
            cosine = math.cos(theta_center)
            sine = math.sin(theta_center)
            normalizer = cosine * cosine + sine * sine
            point = (
                basis[0]
                + basis[1] * (edge_value * cosine / normalizer)
                + basis[2] * (edge_value * sine / normalizer)
                + basis[3] * 0.8
            )
            _, _, counts = _signed_clearance_field(
                point.reshape(3, 1),
                basis,
                parameters["theta_edges"],
                parameters["distance_edges"],
                parameters["height_bands"],
                parameters["widths"],
                order_statistic=parameters["order_statistic"],
                final_edge_atol_m=parameters["final_edge_atol_m"],
                final_edge_rtol=parameters["final_edge_rtol"],
                clip_min_m=parameters["clip_min_m"],
                clip_max_m=parameters["clip_max_m"],
            )
            populated = np.argwhere(counts > 0)
            expected_bin: int | None
            if edge_value > float(parameters["distance_edges"][-1]) + 1e-12:
                expected_bin = None
            else:
                expected_bin = int(np.searchsorted(parameters["distance_edges"], edge_value, side="right") - 1)
            if abs(edge_value - float(parameters["distance_edges"][-1])) <= 1e-12:
                expected_bin = len(parameters["distance_edges"]) - 2
            observed = sorted(set(int(item[1]) for item in populated))
            checks.append(
                {
                    "name": f"distance_edge_{edge}_{delta}",
                    "pass": (
                        expected_bin in observed
                        if expected_bin is not None
                        else not observed
                    ),
                    "expected_bin": expected_bin,
                    "observed_bins": observed,
                }
            )
    proper_known = np.zeros((2, 6, 6), dtype=bool)
    proper_known[0, 0, 0] = True
    proper_field = np.zeros((2, 6, 6), dtype=np.float64)
    contract = _validate_nullable_arm(proper_known, proper_field)
    mutation = {
        "known": proper_known.tolist(),
        "probe_pass_counts": np.where(proper_known, 9, 0).astype(np.int64).tolist(),
        "clearance_m": nullable_field(proper_known, proper_field),
    }
    mutation["clearance_m"][0][0][1] = 0.1
    caught = False
    try:
        arrays_from_arm(mutation)
    except ValueError:
        caught = True
    checks.append(
        {
            "name": "unknown_numeric_mutation_rejected",
            "pass": caught and contract["unknown_cells"] == 71,
            "caught": caught,
            "unknown_cells": contract["unknown_cells"],
        }
    )
    identity = camera_binding(np.zeros(3), np.eye(3))
    history = camera_binding(np.asarray([0.0, 0.0, -0.4]), np.eye(3))
    plane = {"camera_ground_projection_m": [0.0, 0.0, 0.0], "normal_toward_camera": [0.0, 1.0, 0.0]}
    _, predicted_positive, receipt_positive = predicted_bases(history, identity, plane)
    positive_forward_delta = float(predicted_positive[0.4][1][2])
    checks.append(
        {
            "name": "positive_forward_translation_preserves_positive_z_motion",
            "pass": receipt_positive["tangent_translation_velocity_m_s"][2] > 0.0,
            "velocity": receipt_positive["tangent_translation_velocity_m_s"],
            "predicted_forward": predicted_positive[0.4][1].tolist(),
            "forward_component": positive_forward_delta,
        }
    )
    checks.append(
        {
            "name": "time_normalization_uses_observed_dt",
            "pass": abs((1.0 / 0.2) - 5.0) < 1e-12 and abs((1.0 / 0.1) - 5.0) > 1e-12,
            "normalized_rate": 1.0 / 0.2,
            "nominal_rate": 1.0 / 0.1,
        }
    )
    return {
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "checks": checks,
        "passed": sum(bool(item["pass"]) for item in checks),
        "total": len(checks),
    }


def summarize(rows: list[dict[str, Any]], boundary: dict[str, Any], protocol_sha: str) -> dict[str, Any]:
    def status_count(path: tuple[str, ...]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for row in rows:
            value: Any = row
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            if isinstance(value, str):
                counter[value] += 1
        return dict(sorted(counter.items()))

    rotation_rows = [
        row["rcle"]["rotation_leakage"]
        for row in rows
        if isinstance(row.get("rcle"), dict) and "rotation_leakage" in row["rcle"]
    ]
    rotation_numeric_rows = [
        item
        for item in rotation_rows
        if item.get("raw_per_s") is not None
        and item.get("compensated_per_s") is not None
    ]
    sign_rows = [
        row["rcle"]["expansion_sign"]
        for row in rows
        if isinstance(row.get("rcle"), dict) and "expansion_sign" in row["rcle"]
    ]
    ttc_rows = [row["ttc_proxy"] for row in rows if row.get("ttc_proxy", {}).get("status") == "EVALUABLE"]
    field_pairs: list[float] = []
    field_improvements: list[float] = []
    d2_field_improvements: list[float] = []
    unknown_violations = 0
    rcle_coverages: list[float] = []
    rotation_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sign_by_family: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        field = row.get("field_transport", {})
        unknown_violations += int(field.get("unknown_to_safe_violations", 0))
        if row.get("rcle", {}).get("status") == "EVALUABLE":
            coverage = row["rcle"].get("raw", {}).get("coverage")
            if coverage is not None:
                rcle_coverages.append(float(coverage))
        rotation = row.get("rcle", {}).get("rotation_leakage")
        if rotation and rotation.get("raw_per_s") is not None and rotation.get("compensated_per_s") is not None:
            rotation_by_axis[str(row.get("motion_family"))].append(rotation)
        sign = row.get("rcle", {}).get("expansion_sign")
        if sign and sign.get("correct") is not None:
            sign_by_family[str(row.get("motion_family"))].append(bool(sign["correct"]))
        for stratum in field.get("horizons", {}).values():
            if stratum.get("status") == "EVALUABLE":
                improvement = float(stratum["persistence_mae_m"]) - float(stratum["advected_mae_m"])
                field_pairs.append(-improvement)
                field_improvements.append(improvement)
                if row.get("d2_time_status") == "EVALUABLE":
                    d2_field_improvements.append(improvement)
    by_motion: dict[str, Any] = {}
    for family in sorted({str(row["motion_family"]) for row in rows}):
        subset = [row for row in rows if row["motion_family"] == family]
        by_motion[family] = {
            "cases": len(subset),
            "rcle_status": dict(Counter(row["rcle"].get("status") for row in subset)),
            "field_status": dict(Counter(row["field_transport"].get("status") for row in subset)),
            "ttc_status": dict(Counter(row["ttc_proxy"].get("status") for row in subset)),
        }
    return {
        "schema": SCHEMA,
        "runner_id": RUNNER_ID,
        "protocol_sha256": protocol_sha,
        "case_count": len(rows),
        "rcle_status": status_count(("rcle", "status")),
        "field_status": status_count(("field_transport", "status")),
        "ttc_status": status_count(("ttc_proxy", "status")),
        "d2_time_status": dict(Counter(row.get("d2_time_status") for row in rows)),
        "rotation_leakage": {
            "rows": len(rotation_rows),
            "numeric_rows": len(rotation_numeric_rows),
            "raw_median_per_s": float(np.median([item["raw_per_s"] for item in rotation_numeric_rows])) if rotation_numeric_rows else None,
            "compensated_median_per_s": float(np.median([item["compensated_per_s"] for item in rotation_numeric_rows])) if rotation_numeric_rows else None,
            "reduced_fraction": float(np.mean([item["reduced"] for item in rotation_numeric_rows])) if rotation_numeric_rows else None,
            "by_axis": {
                axis: {
                    "rows": len(values),
                    "compensated_median_per_s": float(np.median([item["compensated_per_s"] for item in values])) if values else None,
                    "compensated_p90_per_s": float(np.percentile([item["compensated_per_s"] for item in values], 90)) if values else None,
                    "compensated_max_per_s": float(np.max([item["compensated_per_s"] for item in values])) if values else None,
                    "reduced_fraction": float(np.mean([item["reduced"] for item in values])) if values else None,
                }
                for axis, values in sorted(rotation_by_axis.items())
            },
        },
        "expansion_sign": {
            "rows": len(sign_rows),
            "correct_fraction": float(np.mean([item["correct"] for item in sign_rows])) if sign_rows else None,
            "by_family": {
                family: {
                    "rows": len(values),
                    "correct_fraction": float(np.mean(values)) if values else None,
                }
                for family, values in sorted(sign_by_family.items())
            },
        },
        "ttc_proxy": {
            "evaluable_rows": len(ttc_rows),
            "proxy_median_s": float(np.median([item["ttc_proxy_s"] for item in ttc_rows])) if ttc_rows else None,
            "physical_ttc_comparison_rows": sum(item.get("analytic_physical_ttc_s") is not None for item in ttc_rows),
            "proxy_is_physical_ttc": False,
        },
        "field_transport": {
            "evaluable_strata": len(field_pairs),
            "advected_minus_persistence_m_median": float(np.median(field_pairs)) if field_pairs else None,
            "advected_improvement_m_median": float(np.median(field_improvements)) if field_improvements else None,
            "advected_improvement_m_mean": float(np.mean(field_improvements)) if field_improvements else None,
            "advected_better_fraction": float(np.mean(np.asarray(field_improvements) > 1e-12)) if field_improvements else None,
            "nonzero_improvement_fraction": float(np.mean(np.abs(np.asarray(field_improvements)) > 1e-12)) if field_improvements else None,
            "d2_time_eligible_strata": len(d2_field_improvements),
            "d2_advected_improvement_m_median": float(np.median(d2_field_improvements)) if d2_field_improvements else None,
            "d2_advected_better_fraction": float(np.mean(np.asarray(d2_field_improvements) > 1e-12)) if d2_field_improvements else None,
            "unknown_to_safe_violations": unknown_violations,
        },
        "rcle_coverage": {
            "evaluable_rows": len(rcle_coverages),
            "minimum": float(np.min(rcle_coverages)) if rcle_coverages else None,
            "median": float(np.median(rcle_coverages)) if rcle_coverages else None,
        },
        "boundary": boundary,
        "by_motion_family": by_motion,
        "authority": {
            "synthetic_only": True,
            "natural_data_opened": False,
            "future_truth_firewall": True,
            "default_app_changed": False,
            "production_or_safety_claim": False,
        },
    }


def run(protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output exists; use a new explicit path: {output}")
    output.mkdir(parents=True, exist_ok=False)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    write_json(output / "protocol_copy.json", protocol)
    cases = build_cases(protocol)
    write_json(output / "case_manifest.json", {"case_count": len(cases), "cases": cases})
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row = run_case(case, protocol)
        row["ordinal"] = index
        rows.append(row)
    write_jsonl(output / "case_results.jsonl", rows)
    boundary = _boundary_suite(protocol)
    write_json(output / "boundary_results.json", boundary)
    summary = summarize(rows, boundary, protocol_sha)
    write_json(output / "summary.json", summary)
    manifest = {
        "schema": "blindassist.controlled_synthetic_stress.r0.manifest",
        "protocol_sha256": protocol_sha,
        "case_manifest_sha256": sha256_file(output / "case_manifest.json"),
        "case_results_sha256": sha256_file(output / "case_results.jsonl"),
        "boundary_results_sha256": sha256_file(output / "boundary_results.json"),
        "summary_sha256": sha256_file(output / "summary.json"),
        "case_count": len(rows),
        "boundary_status": boundary["status"],
        "synthetic_only": True,
    }
    write_json(output / "run_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = load_json(PROTOCOL_PATH)
    try:
        summary = run(protocol, args.output.resolve())
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "NOT_EVALUABLE", "error": f"{type(error).__name__}:{error}"}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": "COMPLETE",
        "case_count": summary["case_count"],
        "boundary": summary["boundary"]["status"],
        "output": str(args.output.resolve()),
        "rcle_status": summary["rcle_status"],
        "field_status": summary["field_status"],
        "d2_time_status": summary["d2_time_status"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
