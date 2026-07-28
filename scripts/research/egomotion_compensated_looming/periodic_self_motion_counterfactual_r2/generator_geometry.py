"""Deterministic P1-only non-planar 3D geometry generator.

This module never imports or executes RCLE.  It materializes scene, trajectory,
fixture, and bounded replay evidence for independent G01-G14 validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation, Slerp


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
IMPLEMENTATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_R0"
)
SCHEMA = "rcle.periodic_self_motion_counterfactual.p1_geometry_manifest.v1"
WIDTH = 360
HEIGHT = 640
K = np.asarray(
    [[541.2, 0.0, 182.3389], [0.0, 542.2, 321.654], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
K_INV = np.linalg.inv(K)
FRAME_COUNT = 602
PAIR_COUNT = 601
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
ARMS = (
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
)
GUARD_ARMS = ("MONOTONIC_APPROACH", "MONOTONIC_APPROACH_PLUS_PERIODIC")
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r0"
)
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json"
)
GEOMETRY_SPEC_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_VALIDATION_R0_2026-07-28.json"
)
SOURCE_ROOTS = {
    "ADVIO_13": REPO_ROOT
    / "artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0"
    / "sources/advio-13",
    "ADVIO_14": REPO_ROOT
    / "artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0"
    / "sources/advio-14",
    "ADVIO_15": REPO_ROOT
    / "artifacts.local/evidence/public-advio-r792-turn-intent-20260719"
    / "extracted/advio-15",
    "ADVIO_17": REPO_ROOT
    / "artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0"
    / "sources/advio-17",
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def derive_seed(namespace: str, block: str, ordinal: int) -> int:
    token = f"{PROTOCOL_ID}|{namespace}|{block}|{ordinal:02d}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def unit_hash(*parts: object) -> float:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return integer / float(2**64 - 1)


def _load_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [[float(cell) for cell in row] for row in csv.reader(stream)]
    return np.asarray(rows, dtype=np.float64)


def _rotation_log_continuous(rotations: Rotation) -> np.ndarray:
    vectors = rotations.as_rotvec()
    result = np.empty_like(vectors)
    result[0] = vectors[0]
    for index in range(1, len(vectors)):
        current = vectors[index]
        norm = float(np.linalg.norm(current))
        candidates = [current]
        if norm > 1e-12:
            axis = current / norm
            candidates.extend(
                current + axis * (2.0 * math.pi * k)
                for k in (-2, -1, 1, 2)
            )
        result[index] = min(
            candidates,
            key=lambda item: float(np.linalg.norm(item - result[index - 1])),
        )
    return result


def _band_limited_closed(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    uniform_t = np.arange(
        timestamps[0], timestamps[-1] + 1e-12, 1.0 / 60.0, dtype=np.float64
    )
    uniform = np.column_stack(
        [np.interp(uniform_t, timestamps, values[:, axis]) for axis in range(6)]
    )
    design = np.column_stack((np.ones_like(uniform_t), uniform_t))
    detrended = np.empty_like(uniform)
    for axis in range(6):
        beta = np.linalg.lstsq(design, uniform[:, axis], rcond=None)[0]
        detrended[:, axis] = uniform[:, axis] - design @ beta
    frequencies = np.fft.rfftfreq(len(uniform_t), d=1.0 / 60.0)
    keep = (frequencies >= 0.7) & (frequencies <= 3.0)
    filtered = np.empty_like(detrended)
    for axis in range(6):
        spectrum = np.fft.rfft(detrended[:, axis])
        spectrum[~keep] = 0.0
        filtered[:, axis] = np.fft.irfft(spectrum, n=len(uniform_t))
    sampled = np.column_stack(
        [np.interp(timestamps, uniform_t, filtered[:, axis]) for axis in range(6)]
    )
    phase = ((timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0]))[
        :, None
    ]
    sampled = sampled - sampled[0]
    sampled = sampled - phase * sampled[-1]
    sampled[0] = 0.0
    sampled[-1] = 0.0
    return sampled


def build_trajectory(block: str, contract: dict[str, Any]) -> dict[str, Any]:
    source = SOURCE_ROOTS[block]
    frames_path = source / "iphone/frames.csv"
    pose_path = source / "ground-truth/pose.csv"
    lock = contract["trajectory_blocks"][block]
    if sha256_file(frames_path) != lock["frames_csv_sha256"]:
        raise ValueError(f"FRAME_HASH_MISMATCH:{block}")
    if sha256_file(pose_path) != lock["pose_csv_sha256"]:
        raise ValueError(f"POSE_HASH_MISMATCH:{block}")

    frames = _load_csv(frames_path)[:FRAME_COUNT]
    poses = _load_csv(pose_path)
    timestamps = frames[:, 0]
    source_t = poses[:, 0]
    translations = np.column_stack(
        [np.interp(timestamps, source_t, poses[:, axis]) for axis in (1, 2, 3)]
    )
    source_rotations = Rotation.from_quat(poses[:, [5, 6, 7, 4]])
    sampled_rotations = Slerp(source_t, source_rotations)(timestamps)
    r0 = sampled_rotations[0]
    relative_rotation = r0.inv() * sampled_rotations
    relative_translation = r0.inv().apply(translations - translations[0])
    twists = np.column_stack(
        (relative_translation, _rotation_log_continuous(relative_rotation))
    )
    closed = _band_limited_closed(twists, timestamps)
    rotations = Rotation.from_rotvec(closed[:, 3:]).as_matrix()
    translations_closed = closed[:, :3]
    poses_out = [
        {
            "frame_index": index,
            "timestamp_s": float(timestamps[index]),
            "translation_m": translations_closed[index].tolist(),
            "rotation_matrix": rotations[index].tolist(),
        }
        for index in range(FRAME_COUNT)
    ]
    pose_hash = sha256_bytes(canonical_bytes(poses_out))
    return {
        "block": block,
        "frame_count": FRAME_COUNT,
        "pair_count": PAIR_COUNT,
        "source_frames_csv_sha256": lock["frames_csv_sha256"],
        "source_pose_csv_sha256": lock["pose_csv_sha256"],
        "construction": {
            "source_quaternion_order": "wxyz",
            "relative_pose": "R0^-1 applied to world translation; R0^-1*Rt",
            "uniform_rate_hz": 60.0,
            "detrend": "per-axis least-squares affine time trend",
            "band_hz_inclusive": [0.7, 3.0],
            "endpoint_bridge": "linear Lie-algebra bridge; endpoints assigned exact zero",
            "amplitude_scale": 1.0,
            "time_scale": 1.0,
        },
        "periodic_pose_sha256": pose_hash,
        "poses": poses_out,
    }


def _surface(
    object_id: int,
    z: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    seed: int,
) -> dict[str, Any]:
    color = [
        round(0.15 + 0.75 * unit_hash(seed, object_id, channel), 12)
        for channel in ("r", "g", "b")
    ]
    return {
        "object_id": object_id,
        "primitive": "rectangle_mesh_2tri",
        "plane_z_m": round(z, 12),
        "bounds_xy_m": [round(x0, 12), round(x1, 12), round(y0, 12), round(y1, 12)],
        "vertices_world_m": [
            [round(x0, 12), round(y0, 12), round(z, 12)],
            [round(x1, 12), round(y0, 12), round(z, 12)],
            [round(x1, 12), round(y1, 12), round(z, 12)],
            [round(x0, 12), round(y1, 12), round(z, 12)],
        ],
        "triangles": [[0, 1, 2], [0, 2, 3]],
        "material_id": f"MAT_{object_id:02d}",
        "linear_rgb": color,
        "texture": {
            "type": "analytic_checker",
            "cycles_per_m": round(3.0 + 9.0 * unit_hash(seed, object_id, "freq"), 12),
            "phase": round(unit_hash(seed, object_id, "phase"), 12),
        },
    }


def build_scene(block: str, ordinal: int, namespace: str) -> dict[str, Any]:
    seed = derive_seed(namespace, block, ordinal)
    guard = namespace == "GUARD"
    objects: list[dict[str, Any]] = []
    object_id = 1
    for row in range(3):
        v0 = row * HEIGHT / 3.0
        v1 = (row + 1) * HEIGHT / 3.0
        for column in range(3):
            u0 = column * WIDTH / 3.0
            u1 = (column + 1) * WIDTH / 3.0
            for band_index, (sub0, sub1) in enumerate(
                ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0))
            ):
                su0 = u0 + (u1 - u0) * sub0
                su1 = u0 + (u1 - u0) * sub1
                if band_index == 0:
                    z = (1.45 if guard else 1.15) + 0.25 * unit_hash(
                        seed, object_id, "depth"
                    )
                elif band_index == 1:
                    z = 3.6 + 0.8 * unit_hash(seed, object_id, "depth")
                else:
                    z = 8.0 + 4.0 * unit_hash(seed, object_id, "depth")
                margin_px = 10.0
                x0 = ((su0 - margin_px - K[0, 2]) / K[0, 0]) * z
                x1 = ((su1 + margin_px - K[0, 2]) / K[0, 0]) * z
                y0 = ((v0 - margin_px - K[1, 2]) / K[1, 1]) * z
                y1 = ((v1 + margin_px - K[1, 2]) / K[1, 1]) * z
                objects.append(_surface(object_id, z, x0, x1, y0, y1, seed))
                object_id += 1
    z_far = 18.0
    objects.append(
        _surface(
            object_id,
            z_far,
            -12.0,
            12.0,
            -16.0,
            16.0,
            seed,
        )
    )
    scene_core = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "namespace": namespace,
        "block": block,
        "ordinal": ordinal,
        "numeric_seed_uint64": seed,
        "camera": {
            "projection": "pinhole",
            "width_px": WIDTH,
            "height_px": HEIGHT,
            "intrinsic": K.tolist(),
            "near_clip_m": 0.5,
            "far_clip_m": 25.0,
            "distortion": "none",
            "camera_axes": "+x right, +y down, +z optical forward",
        },
        "world": {
            "static": True,
            "moving_objects": False,
            "renderer": "deterministic analytic ray/rectangle z-buffer v1",
            "objects": objects,
        },
    }
    scene_core["scene_geometry_sha256"] = sha256_bytes(canonical_bytes(scene_core))
    return scene_core


def _raycast(
    scene: dict[str, Any],
    rotation_world_from_camera: np.ndarray,
    translation_world_from_camera: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.column_stack((uv, np.ones(len(uv), dtype=np.float64)))
    directions_camera = (K_INV @ pixels.T).T
    directions_world = (rotation_world_from_camera @ directions_camera.T).T
    best_depth = np.full(len(uv), np.inf, dtype=np.float64)
    best_object = np.zeros(len(uv), dtype=np.int32)
    best_world = np.full((len(uv), 3), np.nan, dtype=np.float64)
    for obj in scene["world"]["objects"]:
        z = float(obj["plane_z_m"])
        denom = directions_world[:, 2]
        scale = (z - translation_world_from_camera[2]) / denom
        world = translation_world_from_camera + scale[:, None] * directions_world
        x0, x1, y0, y1 = obj["bounds_xy_m"]
        camera = (
            rotation_world_from_camera.T
            @ (world - translation_world_from_camera).T
        ).T
        depth = camera[:, 2]
        valid = (
            np.isfinite(scale)
            & (scale > 0.0)
            & (world[:, 0] >= x0)
            & (world[:, 0] <= x1)
            & (world[:, 1] >= y0)
            & (world[:, 1] <= y1)
            & (depth >= 0.5)
            & (depth <= 25.0)
            & (depth < best_depth)
        )
        best_depth[valid] = depth[valid]
        best_object[valid] = int(obj["object_id"])
        best_world[valid] = world[valid]
    return best_depth, best_object, best_world


def _project(
    world: np.ndarray, rotation_world_from_camera: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation_world_from_camera.T @ (world - translation).T).T
    projected = (K @ camera.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return uv, camera[:, 2]


def render(scene: dict[str, Any], rotation: np.ndarray, translation: np.ndarray) -> dict[str, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(WIDTH, dtype=np.float64),
        np.arange(HEIGHT, dtype=np.float64),
    )
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    depth, object_id, world = _raycast(scene, rotation, translation, uv)
    valid = np.isfinite(depth)
    linear = np.zeros((len(uv), 3), dtype=np.float64)
    by_id = {int(obj["object_id"]): obj for obj in scene["world"]["objects"]}
    for identifier in np.unique(object_id[valid]):
        selected = object_id == identifier
        obj = by_id[int(identifier)]
        base = np.asarray(obj["linear_rgb"], dtype=np.float64)
        frequency = float(obj["texture"]["cycles_per_m"])
        phase = float(obj["texture"]["phase"])
        checker = (
            np.floor((world[selected, 0] * frequency + phase) % 2.0)
            + np.floor((world[selected, 1] * frequency + phase) % 2.0)
        ) % 2.0
        modulation = 0.65 + 0.35 * checker[:, None]
        linear[selected] = np.clip(base[None, :] * modulation, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    rgb = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return {
        "rgb": rgb.reshape(HEIGHT, WIDTH, 3),
        "depth": depth.reshape(HEIGHT, WIDTH),
        "object_id": object_id.reshape(HEIGHT, WIDTH),
        "world": world.reshape(HEIGHT, WIDTH, 3),
    }


def reference_metrics(scene: dict[str, Any]) -> dict[str, Any]:
    output = render(scene, np.eye(3), np.zeros(3))
    depth = output["depth"]
    valid = np.isfinite(depth)
    fractions = {
        "near": float(np.mean(valid & (depth >= 0.75) & (depth < 2.0))),
        "middle": float(np.mean(valid & (depth >= 2.0) & (depth < 5.0))),
        "far": float(np.mean(valid & (depth >= 5.0) & (depth <= 20.0))),
    }
    diverse = 0
    cells: list[dict[str, Any]] = []
    for row in range(3):
        for column in range(3):
            tile = depth[
                row * HEIGHT // 3 : (row + 1) * HEIGHT // 3,
                column * WIDTH // 3 : (column + 1) * WIDTH // 3,
            ]
            tile_valid = np.isfinite(tile)
            denominator = int(np.count_nonzero(tile_valid))
            band_fractions = {
                "near": float(np.count_nonzero(tile_valid & (tile < 2.0)) / denominator),
                "middle": float(
                    np.count_nonzero(tile_valid & (tile >= 2.0) & (tile < 5.0))
                    / denominator
                ),
                "far": float(np.count_nonzero(tile_valid & (tile >= 5.0)) / denominator),
            }
            populated = sum(value >= 0.05 for value in band_fractions.values())
            diverse += int(populated >= 2)
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "band_fractions": band_fractions,
                    "populated_band_count": populated,
                }
            )
    return {
        "valid_depth_fraction": float(np.mean(valid)),
        "depth_band_fractions": fractions,
        "diverse_grid_cell_count": diverse,
        "grid_cells": cells,
        "reference_depth_sha256": sha256_bytes(depth.astype("<f8").tobytes()),
        "reference_object_id_sha256": sha256_bytes(
            output["object_id"].astype("<i4").tobytes()
        ),
        "reference_visibility_sha256": sha256_bytes(valid.astype(np.uint8).tobytes()),
    }


def _pose_arrays(trajectory: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = trajectory["poses"]
    timestamps = np.asarray([pose["timestamp_s"] for pose in poses], dtype=np.float64)
    translations = np.asarray([pose["translation_m"] for pose in poses], dtype=np.float64)
    rotations = np.asarray([pose["rotation_matrix"] for pose in poses], dtype=np.float64)
    return timestamps, translations, rotations


def _arm_manifest(
    cluster_id: str,
    scene: dict[str, Any],
    block: str,
    arm: str,
    trajectory_hash: str,
) -> dict[str, Any]:
    motion, quality = arm.split("__", 1)
    geometry_identity = {
        "scene_geometry_sha256": scene["scene_geometry_sha256"],
        "camera_intrinsic_sha256": sha256_bytes(K.astype("<f8").tobytes()),
        "timestamp_sha256": trajectory_hash,
        "motion": motion,
    }
    geometry_hash = sha256_bytes(canonical_bytes(geometry_identity))
    return {
        "cluster_id": cluster_id,
        "arm_id": arm,
        "motion": motion,
        "quality": quality,
        "scene_geometry_sha256": scene["scene_geometry_sha256"],
        "trajectory_sha256": trajectory_hash if motion.startswith("PERIODIC") else (
            sha256_bytes(canonical_bytes({"static": FRAME_COUNT}))
        ),
        "geometry_identity_sha256": geometry_hash,
        "depth_sha256": geometry_hash,
        "object_id_sha256": geometry_hash,
        "pose_sha256": geometry_hash,
        "intrinsic_sha256": sha256_bytes(K.astype("<f8").tobytes()),
        "timestamp_sha256": trajectory_hash,
        "visibility_sha256": geometry_hash,
        "quality_operator_status": "NOT_CALIBRATED_P1_IDENTITY_ONLY",
    }


def build_main_records(trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in BLOCKS:
        trajectory = trajectories[block]
        for ordinal in range(20):
            scene = build_scene(block, ordinal, "MAIN")
            metrics = reference_metrics(scene)
            cluster_id = f"{block}__MAIN_{ordinal:02d}"
            records.append(
                {
                    "record_type": "main_cluster",
                    "cluster_id": cluster_id,
                    "block": block,
                    "ordinal": ordinal,
                    "numeric_seed_uint64": scene["numeric_seed_uint64"],
                    "scene": scene,
                    "reference_metrics": metrics,
                    "arms": [
                        _arm_manifest(
                            cluster_id,
                            scene,
                            block,
                            arm,
                            trajectory["periodic_pose_sha256"],
                        )
                        for arm in ARMS
                    ],
                }
            )
    return records


def _guard_trajectory(trajectory: dict[str, Any], plus_periodic: bool) -> dict[str, Any]:
    timestamps, translations, rotations = _pose_arrays(trajectory)
    phase = ((timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0]))[:, None]
    approach = np.column_stack(
        (np.zeros(FRAME_COUNT), np.zeros(FRAME_COUNT), 0.8 * phase[:, 0])
    )
    if plus_periodic:
        translations = translations + approach
    else:
        translations = approach
        rotations = np.repeat(np.eye(3)[None, :, :], FRAME_COUNT, axis=0)
    payload = [
        {
            "frame_index": index,
            "timestamp_s": float(timestamps[index]),
            "translation_m": translations[index].tolist(),
            "rotation_matrix": rotations[index].tolist(),
        }
        for index in range(FRAME_COUNT)
    ]
    return {
        "poses": payload,
        "pose_sha256": sha256_bytes(canonical_bytes(payload)),
    }


def build_guard_records(trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in BLOCKS:
        for ordinal in range(2):
            scene = build_scene(block, ordinal, "GUARD")
            cluster_id = f"{block}__GUARD_{ordinal:02d}"
            arms = []
            for arm in GUARD_ARMS:
                guard_trajectory = _guard_trajectory(
                    trajectories[block], arm.endswith("PLUS_PERIODIC")
                )
                arms.append(
                    {
                        "cluster_id": cluster_id,
                        "arm_id": arm,
                        "quality": "CLEAN",
                        "scene_geometry_sha256": scene["scene_geometry_sha256"],
                        "trajectory_sha256": guard_trajectory["pose_sha256"],
                        "trajectory": guard_trajectory["poses"],
                    }
                )
            records.append(
                {
                    "record_type": "guardrail_cluster",
                    "cluster_id": cluster_id,
                    "block": block,
                    "ordinal": ordinal,
                    "numeric_seed_uint64": scene["numeric_seed_uint64"],
                    "scene": scene,
                    "reference_metrics": reference_metrics(scene),
                    "designated_middle_target_depth_m": 4.0,
                    "approach_translation_m": 0.8,
                    "arms": arms,
                }
            )
    return records


def analytic_fixtures() -> dict[str, Any]:
    fixtures = [
        {
            "id": "PURE_STATIC",
            "camera_poses": [
                {"translation_m": [0.0, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
                {"translation_m": [0.0, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
            ],
            "points_camera0_m": [
                [x * z, y * z, z]
                for z in (1.0, 3.0, 9.0)
                for x, y in ((-0.2, -0.3), (0.2, -0.3), (-0.2, 0.3), (0.2, 0.3))
            ],
        },
        {
            "id": "PURE_TRANSLATION_LATERAL_MULTI_DEPTH",
            "camera_poses": [
                {"translation_m": [0.0, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
                {"translation_m": [0.05, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
            ],
            "points_camera0_m": [
                [x * z, y * z, z]
                for z in (1.0, 3.0, 9.0)
                for x, y in ((-0.2, -0.3), (0.2, -0.3), (-0.2, 0.3), (0.2, 0.3))
            ],
        },
        {
            "id": "PURE_TRANSLATION_OPTICAL_AXIS_MULTI_DEPTH",
            "camera_poses": [
                {"translation_m": [0.0, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
                {"translation_m": [0.0, 0.0, 0.05], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
            ],
            "points_camera0_m": [
                [x * z, y * z, z]
                for z in (1.0, 3.0, 9.0)
                for x, y in ((-0.2, -0.3), (0.2, -0.3), (-0.2, 0.3), (0.2, 0.3))
            ],
        },
        {
            "id": "PURE_ROTATION_SHARED_BEARINGS_MULTI_DEPTH",
            "camera_poses": [
                {"translation_m": [0.0, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
                {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_rotvec_rad": [0.0, math.radians(5.0), 0.0],
                },
            ],
            "points_camera0_m": [
                [x * z, y * z, z]
                for z in (1.0, 3.0, 9.0)
                for x, y in ((-0.2, -0.3), (0.2, -0.3), (-0.2, 0.3), (0.2, 0.3))
            ],
        },
        {
            "id": "OCCLUSION_DISOCCLUSION",
            "camera_poses": [
                {"translation_m": [0.0, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
                {"translation_m": [0.10, 0.0, 0.0], "rotation_rotvec_rad": [0.0, 0.0, 0.0]},
            ],
            "near_box": {"z_m": 1.5, "image_bounds_px": [120, 240, 180, 460]},
            "far_plane": {"z_m": 6.0, "image_bounds_px": [0, 359, 0, 639]},
        },
        {
            "id": "MONOTONIC_APPROACH",
            "frame_count": FRAME_COUNT,
            "pair_count": PAIR_COUNT,
            "target_initial_depth_m": 4.0,
            "camera_forward_translation_m": 0.8,
            "inverse_depth_increase_fraction": 0.25,
        },
    ]
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.analytic_fixtures.v1",
        "protocol_id": PROTOCOL_ID,
        "fixtures": fixtures,
        "ledger_sha256": sha256_bytes(canonical_bytes(fixtures)),
    }


def _render_replay_item(scene: dict[str, Any], rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    output = render(scene, rotation, translation)
    return {
        "rgb_sha256": sha256_bytes(output["rgb"].tobytes()),
        "depth_sha256": sha256_bytes(output["depth"].astype("<f8").tobytes()),
        "object_id_sha256": sha256_bytes(output["object_id"].astype("<i4").tobytes()),
        "visibility_sha256": sha256_bytes(
            np.isfinite(output["depth"]).astype(np.uint8).tobytes()
        ),
    }


def replay_ledger(trajectories: dict[str, Any]) -> dict[str, Any]:
    items = []
    for block in BLOCKS:
        scene = build_scene(block, 0, "CAL")
        timestamps, translations, rotations = _pose_arrays(trajectories[block])
        for frame_index in (0, 601):
            first = _render_replay_item(
                scene, rotations[frame_index], translations[frame_index]
            )
            second = _render_replay_item(
                build_scene(block, 0, "CAL"),
                rotations[frame_index].copy(),
                translations[frame_index].copy(),
            )
            items.append(
                {
                    "kind": "calibration_seed",
                    "block": block,
                    "ordinal": 0,
                    "frame_index": frame_index,
                    "first": first,
                    "second": second,
                    "match": first == second,
                }
            )
    fixture_payload = analytic_fixtures()
    for fixture in fixture_payload["fixtures"]:
        payload = canonical_bytes(fixture)
        items.append(
            {
                "kind": "analytic_fixture_manifest",
                "fixture_id": fixture["id"],
                "first_sha256": sha256_bytes(payload),
                "second_sha256": sha256_bytes(canonical_bytes(dict(fixture))),
                "match": payload == canonical_bytes(dict(fixture)),
            }
        )
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.replay_ledger.v1",
        "items": items,
        "mismatch_count": sum(not item["match"] for item in items),
    }


def projective_sample_ledger(
    main_records: list[dict[str, Any]], trajectories: dict[str, Any]
) -> dict[str, Any]:
    by_block = {
        block: [item for item in main_records if item["block"] == block]
        for block in BLOCKS
    }
    block_ledgers: dict[str, Any] = {}
    grid_u = np.linspace(12.0, WIDTH - 13.0, 8)
    grid_v = np.linspace(12.0, HEIGHT - 13.0, 8)
    uv_grid = np.asarray([(u, v) for v in grid_v for u in grid_u], dtype=np.float64)
    for block in BLOCKS:
        _, translations, rotations = _pose_arrays(trajectories[block])
        candidates: list[tuple[str, dict[str, Any]]] = []
        for record in by_block[block]:
            scene = record["scene"]
            for frame_index in range(0, 600, 50):
                depth0, object0, world = _raycast(
                    scene,
                    rotations[frame_index],
                    translations[frame_index],
                    uv_grid,
                )
                uv1, projected_depth = _project(
                    world,
                    rotations[frame_index + 1],
                    translations[frame_index + 1],
                )
                depth1, object1, _ = _raycast(
                    scene,
                    rotations[frame_index + 1],
                    translations[frame_index + 1],
                    uv1,
                )
                persistent = (
                    np.isfinite(depth0)
                    & np.isfinite(depth1)
                    & np.isfinite(projected_depth)
                    & (object0 == object1)
                    & (np.abs(depth1 - projected_depth) <= 1e-7)
                    & (uv1[:, 0] >= 0.0)
                    & (uv1[:, 0] < WIDTH)
                    & (uv1[:, 1] >= 0.0)
                    & (uv1[:, 1] < HEIGHT)
                )
                for index in np.flatnonzero(persistent):
                    tuple_text = (
                        f"{block},{record['numeric_seed_uint64']},{frame_index},"
                        f"{int(object0[index])},{uv_grid[index,1]:.9f},"
                        f"{uv_grid[index,0]:.9f}"
                    )
                    sample = {
                        "block": block,
                        "scene_seed": record["numeric_seed_uint64"],
                        "scene_ordinal": record["ordinal"],
                        "frame_index": frame_index,
                        "object_id": int(object0[index]),
                        "v": float(uv_grid[index, 1]),
                        "u": float(uv_grid[index, 0]),
                        "renderer_uv_next": uv1[index].tolist(),
                        "world_point_m": world[index].tolist(),
                        "visible_next": True,
                    }
                    candidates.append((sha256_bytes(tuple_text.encode("utf-8")), sample))
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]["scene_seed"],
                item[1]["frame_index"],
                item[1]["object_id"],
                item[1]["v"],
                item[1]["u"],
            )
        )
        if len(candidates) < 10000:
            raise ValueError(f"PROJECTIVE_SAMPLE_SHORTFALL:{block}:{len(candidates)}")
        selected = [sample for _, sample in candidates[:10000]]
        block_ledgers[block] = {
            "sample_count": len(selected),
            "selection": (
                "enumerate persistent fixed-grid tuples; SHA-256 sort then first 10000"
            ),
            "samples": selected,
        }
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.projective_samples.v1",
        "blocks": block_ledgers,
    }


def runtime_manifest() -> dict[str, Any]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260728)
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "renderer": {
            "name": "deterministic analytic ray/rectangle z-buffer",
            "version": "1",
            "backend": "numpy float64 CPU",
            "thread_count": 1,
            "pixel_rounding": "numpy.rint ties-to-even to uint8",
            "visibility": "nearest positive camera-z ray intersection",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": int(cv2.getNumThreads()),
            "opencv_rng_seed": 20260728,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "seed_derivation": {
            "algorithm": "uint64_big_endian(first_8_bytes(SHA256(UTF8 token)))",
            "main_template": f"{PROTOCOL_ID}|MAIN|<block>|<ordinal_two_digits>",
            "cal_template": f"{PROTOCOL_ID}|CAL|<block>|<ordinal_two_digits>",
            "guard_template": f"{PROTOCOL_ID}|GUARD|<block>|<ordinal_two_digits>",
        },
        "camera": {
            "width_px": WIDTH,
            "height_px": HEIGHT,
            "intrinsic": K.tolist(),
            "near_clip_m": 0.5,
            "far_clip_m": 25.0,
            "distortion": "none",
        },
        "trajectory": {
            "frame_count": FRAME_COUNT,
            "pair_count": PAIR_COUNT,
            "blocks": list(BLOCKS),
            "construction": "source-relative SE3 -> 60Hz affine detrend -> inclusive 0.7-3.0Hz FFT -> endpoint bridge",
            "amplitude_or_time_scaling": False,
        },
        "manifest_schema": SCHEMA,
        "formal_execution_authorized": False,
        "quality_calibration_authorized": False,
        "rcle_imported_or_executed": False,
    }


def produce(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("formal_execution_authorized") is not False:
        raise ValueError("FORMAL_EXECUTION_MUST_REMAIN_FALSE")
    output.mkdir(parents=True, exist_ok=True)
    trajectories = {block: build_trajectory(block, contract) for block in BLOCKS}
    main_records = build_main_records(trajectories)
    guard_records = build_guard_records(trajectories)
    fixtures = analytic_fixtures()
    replay = replay_ledger(trajectories)
    projective = projective_sample_ledger(main_records, trajectories)
    runtime = runtime_manifest()

    write_json(output / "runtime_manifest.json", runtime)
    write_json(output / "trajectory_manifest.json", trajectories)
    write_json(output / "analytic_fixture_ledger.json", fixtures)
    write_json(output / "deterministic_replay_ledger.json", replay)
    write_json(output / "projective_sample_ledger.json", projective)
    manifest_path = output / "all_seed_geometry_manifest.jsonl"
    with manifest_path.open("wb") as stream:
        for record in [*main_records, *guard_records]:
            stream.write(canonical_bytes(record))
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "producer_receipt.json"
    }
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.p1_producer_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "P1_GEOMETRY_MATERIALIZED",
        "main_cluster_count": len(main_records),
        "main_arm_identity_count": sum(len(item["arms"]) for item in main_records),
        "guardrail_cluster_count": len(guard_records),
        "guardrail_arm_identity_count": sum(len(item["arms"]) for item in guard_records),
        "trajectory_block_count": len(trajectories),
        "analytic_fixture_count": len(fixtures["fixtures"]),
        "replay_mismatch_count": replay["mismatch_count"],
        "artifact_sha256": artifact_hashes,
        "rcle_output_accessed": False,
        "quality_strength_calibrated": False,
        "performance_preflight_run": False,
        "formal_sequences_run": False,
        "formal_execution_authorized": False,
    }
    write_json(output / "producer_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = produce(args.output.resolve())
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "INTERVENTION_NOT_EVALUABLE",
                    "terminal": "HOLD_P1",
                    "error": f"{type(error).__name__}:{error}",
                    "formal_execution_authorized": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
