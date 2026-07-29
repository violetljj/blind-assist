"""Stage B source-known translation-depth oracle and object-approach audit.

The control command materializes geometry only.  The execute command requires
an independently produced geometry receipt, creates a successor activation,
and only then creates the response root and runs the frozen 8 x 5 design.
R3 is called unchanged; observation-only fit hooks retain its final
rotation-compensated track collection for paired baseline/oracle refits.
"""

from __future__ import annotations

import os

from . import p3_runtime_preflight_r0 as guarded

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Iterable
from unittest import mock

import cv2
import numpy as np
import psutil

from ..rcle_minimal.local_expansion import fit_fixed_grid_local_affine
from ..rcle_minimal.sparse_flow import SparseTrackResult
from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as r3
from . import generator_geometry as geometry
from . import material_residual_contraction_r1 as qms
from . import p3_transport_r0 as transport


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
    "TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
ARMS = (
    "STATIC_SCENE",
    "EGO_ROTATION_STATIC_SCENE",
    "EGO_TRANSLATION_STATIC_SCENE",
    "OBJECT_APPROACH_STATIC_CAMERA",
    "OBJECT_APPROACH_PLUS_EGO_6DOF",
)
FRAME_COUNT = 602
PAIR_COUNT = 601
WORKERS = 4
GIB = 1024**3
LAUNCH_REFILL_BYTES = 6 * GIB
IN_FLIGHT_FLOOR_BYTES = 4 * GIB
THRESHOLD = 0.01
TARGET_ID = 1001

SPEC_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_EXECUTABLE_SPEC_R1_"
    "2026-07-29.json"
)
IDENTITY_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_"
    "IDENTITY_LOCK_2026-07-29.json"
)
MEMORY_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "MEMORY_GATE_6GIB_SUCCESSOR_AMENDMENT_R0_2026-07-29.json"
)
TRAJECTORY_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
PROTOCOL_RELATIVE = transport.PROTOCOL_RELATIVE
DEFAULT_ROOT = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_stage_b_translation_depth_oracle_object_approach_r0"
)
ACTIVATION_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_EXECUTION_ACTIVATION_R1_"
    "2026-07-29.json"
)


class InvalidStageB(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tree_sha256(path: Path) -> str:
    rows = []
    for item in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        rows.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return sha256_value(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_bindings(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path = root / SPEC_RELATIVE
    identity_path = root / IDENTITY_RELATIVE
    spec = load_json(spec_path)
    identity = load_json(identity_path)
    if spec.get("task_id") != TASK_ID or identity.get("task_id") != TASK_ID:
        raise InvalidStageB("TASK_ID")
    for name in (
        "contract",
        "identity_lock",
        "preflight_receipt",
        "historical_hold_decision",
        "memory_gate_amendment",
        "trajectory_manifest",
    ):
        binding = spec["bindings"][name]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise InvalidStageB(f"SPEC_BINDING:{name}")
    if spec["memory_gate"]["launch_and_refill_available_ram_bytes"] != LAUNCH_REFILL_BYTES:
        raise InvalidStageB("MEMORY_6_GIB")
    if spec["memory_gate"]["in_flight_emergency_runtime_floor_bytes"] != IN_FLIGHT_FLOOR_BYTES:
        raise InvalidStageB("MEMORY_4_GIB_FLOOR")
    if len(identity["clusters"]) != 8:
        raise InvalidStageB("IDENTITY_COUNT")
    return spec, identity


def _scene_seed_scene(cluster: dict[str, Any]) -> dict[str, Any]:
    seed = int(cluster["numeric_seed_uint64"])
    objects: list[dict[str, Any]] = []
    object_id = 1
    for row in range(3):
        for column in range(3):
            z = 8.5 + 0.45 * row + 0.7 * column + 0.6 * geometry.unit_hash(
                seed, row, column, "stage_b_depth"
            )
            u0 = column * geometry.WIDTH / 3.0
            u1 = (column + 1) * geometry.WIDTH / 3.0
            v0 = row * geometry.HEIGHT / 3.0
            v1 = (row + 1) * geometry.HEIGHT / 3.0
            margin = 14.0
            x0 = ((u0 - margin - geometry.K[0, 2]) / geometry.K[0, 0]) * z
            x1 = ((u1 + margin - geometry.K[0, 2]) / geometry.K[0, 0]) * z
            y0 = ((v0 - margin - geometry.K[1, 2]) / geometry.K[1, 1]) * z
            y1 = ((v1 + margin - geometry.K[1, 2]) / geometry.K[1, 1]) * z
            objects.append(
                geometry._surface(object_id, z, x0, x1, y0, y1, seed)
            )
            object_id += 1
    objects.append(
        geometry._surface(10, 18.0, -12.0, 12.0, -16.0, 16.0, seed)
    )
    target = geometry._surface(
        TARGET_ID, 6.0, -0.4, 0.8, -0.7, 0.9, seed
    )
    target["linear_rgb"] = [0.92, 0.47, 0.18]
    target["texture"]["cycles_per_m"] = 12.0
    objects.append(target)
    core = {
        "schema": "rcle.stage_b.materialized_scene.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "cluster_id": cluster["cluster_id"],
        "block": cluster["block"],
        "ordinal": cluster["ordinal"],
        "numeric_seed_uint64": seed,
        "camera": {
            "projection": "pinhole",
            "width_px": geometry.WIDTH,
            "height_px": geometry.HEIGHT,
            "intrinsic": geometry.K.tolist(),
            "near_clip_m": 0.5,
            "far_clip_m": 25.0,
            "distortion": "none",
            "camera_axes": "+x right, +y down, +z optical forward",
            "pose": "world_from_camera",
        },
        "world": {
            "static": False,
            "moving_objects": [TARGET_ID],
            "renderer": "deterministic analytic ray/rectangle z-buffer v1",
            "objects": objects,
        },
    }
    core["scene_geometry_sha256"] = sha256_value(core)
    return core


def _target_z(frame_index: int, moving: bool) -> float:
    if not moving:
        return 6.0
    tau = frame_index / PAIR_COUNT
    return 6.0 - 2.0 * (0.5 - 0.5 * math.cos(math.pi * tau))


def _dynamic_scene(base: dict[str, Any], frame_index: int, moving: bool) -> dict[str, Any]:
    scene = json.loads(json.dumps(base))
    target = next(
        item for item in scene["world"]["objects"] if item["object_id"] == TARGET_ID
    )
    z = _target_z(frame_index, moving)
    target["plane_z_m"] = z
    target["bounds_xy_m"] = [-0.4, 0.8, -0.7, 0.9]
    target["vertices_world_m"] = [
        [-0.4, -0.7, z],
        [0.8, -0.7, z],
        [0.8, 0.9, z],
        [-0.4, 0.9, z],
    ]
    scene.pop("scene_geometry_sha256", None)
    scene["frame_index"] = frame_index
    scene["target_motion"] = "APPROACH" if moving else "STATIC"
    scene["scene_geometry_sha256"] = sha256_value(scene)
    return scene


def _arm_pose(trajectory: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    identity = np.eye(3, dtype=np.float64).tolist()
    zero = [0.0, 0.0, 0.0]
    result = []
    for source in trajectory["poses"]:
        rotation = (
            source["rotation_matrix"]
            if arm in {"EGO_ROTATION_STATIC_SCENE", "OBJECT_APPROACH_PLUS_EGO_6DOF"}
            else identity
        )
        translation = (
            source["translation_m"]
            if arm in {"EGO_TRANSLATION_STATIC_SCENE", "OBJECT_APPROACH_PLUS_EGO_6DOF"}
            else zero
        )
        result.append(
            {
                "frame_index": source["frame_index"],
                "timestamp_s": source["timestamp_s"],
                "rotation_matrix": rotation,
                "translation_m": translation,
            }
        )
    return result


def _arm_target_moving(arm: str) -> bool:
    return arm in {
        "OBJECT_APPROACH_STATIC_CAMERA",
        "OBJECT_APPROACH_PLUS_EGO_6DOF",
    }


def _project_world(
    world: np.ndarray, rotation_wc: np.ndarray, translation_wc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation_wc.T @ (world - translation_wc).T).T
    z = camera[:, 2]
    pixel = np.full((len(world), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (z > 0.0)
    pixel[valid, 0] = (
        geometry.K[0, 0] * camera[valid, 0] / z[valid] + geometry.K[0, 2]
    )
    pixel[valid, 1] = (
        geometry.K[1, 1] * camera[valid, 1] / z[valid] + geometry.K[1, 2]
    )
    return pixel, z


def _bilinear_safe(points: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < geometry.WIDTH - 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < geometry.HEIGHT - 1.0)
    )


def _target_material_points(z: float) -> np.ndarray:
    xs = np.linspace(-0.3, 0.7, 5, dtype=np.float64)
    ys = np.linspace(-0.6, 0.8, 5, dtype=np.float64)
    return np.asarray([[x, y, z] for y in ys for x in xs], dtype=np.float64)


def materialize_control(output_root: Path) -> dict[str, Any]:
    root = repo_root()
    spec, identity = _assert_bindings(root)
    response = output_root / "response"
    if response.exists():
        raise InvalidStageB("RESPONSE_EXISTS_BEFORE_GEOMETRY")
    control = output_root / "control"
    if output_root.exists():
        raise InvalidStageB("OUTPUT_ROOT_ALREADY_EXISTS")
    control.mkdir(parents=True, exist_ok=False)
    trajectory_manifest = load_json(root / TRAJECTORY_RELATIVE)
    clusters = []
    for cluster in identity["clusters"]:
        base = _scene_seed_scene(cluster)
        trajectory = trajectory_manifest[cluster["block"]]
        arms = []
        for arm in ARMS:
            poses = _arm_pose(trajectory, arm)
            moving = _arm_target_moving(arm)
            visibility_min = 25
            frame_inputs = []
            for frame_index, pose in enumerate(poses):
                rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
                translation = np.asarray(pose["translation_m"], dtype=np.float64)
                scene = _dynamic_scene(base, frame_index, moving)
                z = _target_z(frame_index, moving)
                material = _target_material_points(z)
                pixels, projected_z = _project_world(material, rotation, translation)
                safe = _bilinear_safe(pixels)
                visible = np.zeros(25, dtype=bool)
                if np.any(safe):
                    depth, object_id, world = geometry._raycast(
                        scene, rotation, translation, pixels[safe]
                    )
                    tolerance = np.maximum(1e-7, 1e-6 * projected_z[safe])
                    visible[safe] = (
                        np.isfinite(depth)
                        & (object_id == TARGET_ID)
                        & (np.abs(depth - projected_z[safe]) <= tolerance)
                        & (
                            np.linalg.norm(world - material[safe], axis=1)
                            <= tolerance
                        )
                    )
                visibility_min = min(visibility_min, int(np.count_nonzero(visible)))
                frame_inputs.append(
                    {
                        "frame_index": frame_index,
                        "scene_geometry_sha256": scene["scene_geometry_sha256"],
                        "rotation_matrix": pose["rotation_matrix"],
                        "translation_m": pose["translation_m"],
                        "target_z_m": z,
                    }
                )
            if visibility_min != 25:
                raise InvalidStageB(
                    f"TARGET_PERSISTENT_VISIBILITY:{cluster['cluster_id']}:{arm}:"
                    f"{visibility_min}"
                )
            arms.append(
                {
                    "arm": arm,
                    "moving_target": moving,
                    "pose_sha256": sha256_value(poses),
                    "render_input_sha256": sha256_value(frame_inputs),
                    "frame_count": len(frame_inputs),
                    "persistent_target_points": 25,
                    "minimum_visible_target_points": visibility_min,
                }
            )
        clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "block": cluster["block"],
                "ordinal": cluster["ordinal"],
                "numeric_seed_uint64": cluster["numeric_seed_uint64"],
                "base_scene": base,
                "base_scene_sha256": base["scene_geometry_sha256"],
                "target_material_identity_sha256": sha256_value(
                    {
                        "object_id": TARGET_ID,
                        "bounds_xy_m": [-0.4, 0.8, -0.7, 0.9],
                        "triangles": [[0, 1, 2], [0, 2, 3]],
                        "material_xy_grid": _target_material_points(6.0)[:, :2],
                    }
                ),
                "arms": arms,
            }
        )
    center_initial = np.asarray([[0.2, 0.1, 6.0]], dtype=np.float64)
    center_final = np.asarray([[0.2, 0.1, 4.0]], dtype=np.float64)
    p0, _ = _project_world(center_initial, np.eye(3), np.zeros(3))
    p1, _ = _project_world(center_final, np.eye(3), np.zeros(3))
    radial0 = float(np.linalg.norm(p0[0] - geometry.K[:2, 2]))
    radial1 = float(np.linalg.norm(p1[0] - geometry.K[:2, 2]))
    manifest = {
        "schema": "rcle.stage_b.geometry_manifest.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "created_utc": utc_now(),
        "spec_sha256": sha256_file(root / SPEC_RELATIVE),
        "identity_lock_sha256": sha256_file(root / IDENTITY_RELATIVE),
        "memory_gate_amendment_sha256": sha256_file(root / MEMORY_RELATIVE),
        "trajectory_manifest_sha256": sha256_file(root / TRAJECTORY_RELATIVE),
        "cluster_count": len(clusters),
        "sequence_count": len(clusters) * len(ARMS),
        "frame_count_per_sequence": FRAME_COUNT,
        "coordinate": "world_from_camera; metre; +x right +y down +z forward",
        "target_center_radial_scale": radial1 / radial0,
        "clusters": clusters,
        "stage_b_response_read": False,
        "stage_b_workload_run": False,
        "formal_480_plus_16_access_count": 0,
        "terminal": "GEOMETRY_MATERIALIZED / INDEPENDENT_VALIDATION_REQUIRED",
    }
    manifest_path = control / "geometry_manifest.json"
    write_exclusive_json(manifest_path, manifest)
    result = {
        "geometry_manifest_path": str(manifest_path),
        "geometry_manifest_sha256": sha256_file(manifest_path),
        "terminal": manifest["terminal"],
    }
    write_exclusive_json(control / "materialization_receipt.json", result)
    return result


def _cell_mask(points: np.ndarray, index: int) -> np.ndarray:
    row, column = divmod(index, 3)
    x0 = int(round(column * geometry.WIDTH / 3))
    x1 = int(round((column + 1) * geometry.WIDTH / 3))
    y0 = int(round(row * geometry.HEIGHT / 3))
    y1 = int(round((row + 1) * geometry.HEIGHT / 3))
    return (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )


def _final_compensated_tracks(
    fit_calls: list[SparseTrackResult], activated: list[int]
) -> SparseTrackResult:
    if not fit_calls:
        return SparseTrackResult(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            0,
        )
    if len(fit_calls) not in (2, 4):
        raise InvalidStageB(f"FIT_CALL_COUNT:{len(fit_calls)}")
    initial = fit_calls[1]
    managed = fit_calls[3] if len(fit_calls) == 4 else initial
    previous_parts = []
    current_parts = []
    error_parts = []
    activated_set = set(activated)
    for index in range(9):
        source = managed if index in activated_set else initial
        selected = _cell_mask(source.previous_points, index)
        previous_parts.append(source.previous_points[selected])
        current_parts.append(source.current_points[selected])
        error_parts.append(source.forward_backward_errors[selected])
    previous = np.ascontiguousarray(np.vstack(previous_parts).astype(np.float32))
    current = np.ascontiguousarray(np.vstack(current_parts).astype(np.float32))
    errors = np.ascontiguousarray(np.concatenate(error_parts).astype(np.float32))
    return SparseTrackResult(previous, current, errors, initial.requested_count)


def _r3_frame(points_current: np.ndarray, homography: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(homography)
    homogeneous = np.column_stack((points_current, np.ones(len(points_current))))
    mapped = (inverse @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2, None]


def _geometry_filter_and_oracle(
    tracks: SparseTrackResult,
    previous_scene: dict[str, Any],
    current_scene: dict[str, Any],
    previous_rotation: np.ndarray,
    current_rotation: np.ndarray,
    previous_translation: np.ndarray,
    current_translation: np.ndarray,
    target_delta_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous_points = tracks.previous_points.astype(np.float64)
    depth, object_id, world = geometry._raycast(
        previous_scene, previous_rotation, previous_translation, previous_points
    )
    actual_world = world.copy()
    target = object_id == TARGET_ID
    actual_world[target, 2] += target_delta_z
    actual_pixel, actual_depth = _project_world(
        actual_world, current_rotation, current_translation
    )
    safe = (
        np.isfinite(depth)
        & (depth >= 0.5)
        & (depth <= 25.0)
        & _bilinear_safe(actual_pixel)
    )
    visible = np.zeros(len(previous_points), dtype=bool)
    if np.any(safe):
        zbuf, current_id, current_world = geometry._raycast(
            current_scene,
            current_rotation,
            current_translation,
            actual_pixel[safe],
        )
        tolerance = np.maximum(1e-7, 1e-6 * actual_depth[safe])
        visible[safe] = (
            np.isfinite(zbuf)
            & (current_id == object_id[safe])
            & (np.abs(zbuf - actual_depth[safe]) <= tolerance)
            & (
                np.linalg.norm(current_world - actual_world[safe], axis=1)
                <= tolerance
            )
        )
    keep = visible & np.isfinite(tracks.current_points).all(axis=1)
    previous_kept = previous_points[keep]
    baseline = tracks.current_points.astype(np.float64)[keep]
    ids = object_id[keep].astype(np.int32)
    if np.array_equal(previous_translation, current_translation):
        displacement = np.zeros_like(previous_kept)
    else:
        rigid_pixel, _ = _project_world(
            world[keep], current_rotation, current_translation
        )
        homography = transport.rotation_homography(
            previous_rotation, current_rotation, geometry.K
        )
        rigid_r3 = _r3_frame(rigid_pixel, homography)
        displacement = rigid_r3 - previous_kept
    oracle = baseline - displacement
    return previous_kept, baseline, oracle, displacement, ids


def _tracks(previous: np.ndarray, current: np.ndarray) -> SparseTrackResult:
    return SparseTrackResult(
        np.ascontiguousarray(previous.astype(np.float32)),
        np.ascontiguousarray(current.astype(np.float32)),
        np.zeros(len(previous), dtype=np.float32),
        len(previous),
    )


def _audit_cell(
    previous: np.ndarray,
    current: np.ndarray,
    dt: float,
    index: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    selected = _cell_mask(previous, index)
    points = previous[selected]
    endpoints = current[selected]
    result: dict[str, Any] = {
        "cell_index": index,
        "support_count": int(len(points)),
        "coefficients": None,
        "median_fit_residual_pixels_per_frame": None,
    }
    if len(points) < int(parameters["minimum_tracks_per_cell"]):
        return result
    row, column = divmod(index, 3)
    x0 = int(round(column * geometry.WIDTH / 3))
    x1 = int(round((column + 1) * geometry.WIDTH / 3))
    y0 = int(round(row * geometry.HEIGHT / 3))
    y1 = int(round((row + 1) * geometry.HEIGHT / 3))
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half_width, half_height = max(0.5 * (x1 - x0), 1.0), max(
        0.5 * (y1 - y0), 1.0
    )
    design = np.column_stack(
        (
            (points[:, 0] - center_x) / half_width,
            (points[:, 1] - center_y) / half_height,
            np.ones(len(points)),
        )
    )
    velocity = (endpoints - points) / dt
    coefficients, _, _, _ = np.linalg.lstsq(design, velocity, rcond=None)
    residual = np.linalg.norm(design @ coefficients - velocity, axis=1)
    result["coefficients"] = coefficients.tolist()
    result["median_fit_residual_pixels_per_frame"] = float(
        np.median(residual) * dt
    )
    result["expansion_from_coefficients_per_s"] = float(
        0.5
        * (
            coefficients[0, 0] / half_width
            + coefficients[1, 1] / half_height
        )
    )
    return result


def _paired_refit(
    previous: np.ndarray,
    baseline: np.ndarray,
    oracle: np.ndarray,
    dt: float,
    parameters: dict[str, Any],
    minimum_common: int,
) -> dict[str, Any]:
    baseline_cells = fit_fixed_grid_local_affine(
        _tracks(previous, baseline),
        dt,
        (geometry.HEIGHT, geometry.WIDTH),
        parameters,
    )
    oracle_cells = fit_fixed_grid_local_affine(
        _tracks(previous, oracle),
        dt,
        (geometry.HEIGHT, geometry.WIDTH),
        parameters,
    )
    common = [
        index
        for index, (left, right) in enumerate(zip(baseline_cells, oracle_cells))
        if left.evaluable
        and right.evaluable
        and left.expansion is not None
        and right.expansion is not None
    ]
    result: dict[str, Any] = {
        "evaluable": len(common) >= minimum_common,
        "common_cell_indices": common,
        "baseline_cells": [asdict(cell) for cell in baseline_cells],
        "oracle_cells": [asdict(cell) for cell in oracle_cells],
        "baseline_cell_audit": [
            _audit_cell(previous, baseline, dt, index, parameters)
            for index in range(9)
        ],
        "oracle_cell_audit": [
            _audit_cell(previous, oracle, dt, index, parameters)
            for index in range(9)
        ],
    }
    if result["evaluable"]:
        left = np.asarray(
            [baseline_cells[index].expansion for index in common], dtype=np.float64
        )
        right = np.asarray(
            [oracle_cells[index].expansion for index in common], dtype=np.float64
        )
        result.update(
            {
                "baseline_signed_per_s": float(np.median(left)),
                "oracle_signed_per_s": float(np.median(right)),
                "baseline_absolute_per_s": float(np.median(np.abs(left))),
                "oracle_absolute_per_s": float(np.median(np.abs(right))),
            }
        )
    else:
        for key in (
            "baseline_signed_per_s",
            "oracle_signed_per_s",
            "baseline_absolute_per_s",
            "oracle_absolute_per_s",
        ):
            result[key] = None
    return result


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values), probability, method="linear"))


def _trigger_count(rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
    streak = 0
    count = 0
    longest = 0
    for row in rows:
        value = row.get(key) if row.get("full_scene", {}).get("evaluable") else None
        if value is None:
            streak = 0
        elif float(value) > THRESHOLD:
            streak += 1
            longest = max(longest, streak)
            if streak >= 3:
                count += 1
        else:
            streak = 0
    return count, longest


def _reduce(rows: list[dict[str, Any]], target_channel: bool) -> dict[str, Any]:
    channel = "target_mask" if target_channel else "full_scene"
    evaluable = [row[channel] for row in rows if row[channel]["evaluable"]]
    baseline_signed = [float(row["baseline_signed_per_s"]) for row in evaluable]
    oracle_signed = [float(row["oracle_signed_per_s"]) for row in evaluable]
    baseline_abs = [float(row["baseline_absolute_per_s"]) for row in evaluable]
    oracle_abs = [float(row["oracle_absolute_per_s"]) for row in evaluable]
    baseline_triggers, baseline_longest = _trigger_count(
        rows, f"{channel}_baseline_signed_per_s"
    )
    oracle_triggers, oracle_longest = _trigger_count(
        rows, f"{channel}_oracle_signed_per_s"
    )
    return {
        "channel": channel,
        "planned_pair_count": PAIR_COUNT,
        "paired_evaluable_pair_count": len(evaluable),
        "paired_evaluable_fraction": len(evaluable) / PAIR_COUNT,
        "baseline_signed_p50_per_s": _quantile(baseline_signed, 0.5),
        "baseline_signed_p90_per_s": _quantile(baseline_signed, 0.9),
        "oracle_signed_p50_per_s": _quantile(oracle_signed, 0.5),
        "oracle_signed_p90_per_s": _quantile(oracle_signed, 0.9),
        "baseline_absolute_p50_per_s": _quantile(baseline_abs, 0.5),
        "baseline_absolute_p90_per_s": _quantile(baseline_abs, 0.9),
        "oracle_absolute_p50_per_s": _quantile(oracle_abs, 0.5),
        "oracle_absolute_p90_per_s": _quantile(oracle_abs, 0.9),
        "baseline_three_pair_trigger_count": baseline_triggers,
        "baseline_three_pair_trigger_density_fixed": baseline_triggers / PAIR_COUNT,
        "baseline_longest_positive_streak": baseline_longest,
        "oracle_three_pair_trigger_count": oracle_triggers,
        "oracle_three_pair_trigger_density_fixed": oracle_triggers / PAIR_COUNT,
        "oracle_longest_positive_streak": oracle_longest,
    }


def _cluster_worker(task: dict[str, Any]) -> dict[str, Any]:
    guarded._initialize_worker()
    root = repo_root()
    cluster = task["cluster"]
    response_root = Path(task["response_root"])
    cluster_id = cluster["cluster_id"]
    staging = response_root / "staging" / f"{cluster_id}.{os.getpid()}.tmp"
    final = response_root / "clusters" / cluster_id
    staging.mkdir(parents=True, exist_ok=False)
    trajectory = load_json(root / TRAJECTORY_RELATIVE)[cluster["block"]]
    protocol = load_json(root / PROTOCOL_RELATIVE)
    affine_parameters = protocol["local_affine"]
    base = _scene_seed_scene(cluster)
    arm_results = []
    started = time.perf_counter()
    minimum_memory = psutil.virtual_memory().available
    for arm in ARMS:
        poses = _arm_pose(trajectory, arm)
        moving = _arm_target_moving(arm)
        state = r3.PairState()
        previous_frame: tuple[
            np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]
        ] | None = None
        rows: list[dict[str, Any]] = []
        offsets = [0]
        all_previous: list[np.ndarray] = []
        all_baseline: list[np.ndarray] = []
        all_oracle: list[np.ndarray] = []
        all_target: list[np.ndarray] = []
        max_oracle_displacement = 0.0
        static_render: tuple[np.ndarray, np.ndarray] | None = None
        for frame_index, pose in enumerate(poses):
            available = psutil.virtual_memory().available
            minimum_memory = min(minimum_memory, available)
            if available < IN_FLIGHT_FLOOR_BYTES:
                raise InvalidStageB("RUN_AVAILABLE_RAM_BELOW_4_GIB")
            scene = _dynamic_scene(base, frame_index, moving)
            rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            translation = np.asarray(pose["translation_m"], dtype=np.float64)
            if arm == "STATIC_SCENE" and static_render is not None:
                rgb, mask = static_render
            else:
                rendered = qms.render_pair(scene, rotation, translation)
                rgb = rendered["rgb_pair"]["clean"]
                mask = rendered["valid_mask"]
                if arm == "STATIC_SCENE":
                    static_render = (rgb, mask)
            if previous_frame is not None:
                previous_rgb, previous_mask, previous_pose, previous_scene = previous_frame
                fit_calls: list[SparseTrackResult] = []
                original_fit = r3.fit_fixed_grid_local_affine

                def capture_fit(
                    tracks: SparseTrackResult,
                    dt_seconds: float,
                    image_shape: tuple[int, int],
                    parameters: dict[str, Any],
                ) -> Any:
                    fit_calls.append(
                        SparseTrackResult(
                            tracks.previous_points.copy(),
                            tracks.current_points.copy(),
                            tracks.forward_backward_errors.copy(),
                            tracks.requested_count,
                        )
                    )
                    return original_fit(tracks, dt_seconds, image_shape, parameters)

                with mock.patch.object(
                    r3, "fit_fixed_grid_local_affine", side_effect=capture_fit
                ):
                    r3_row = transport.evaluate_pair(
                        pair_index=frame_index - 1,
                        previous_rgb=previous_rgb,
                        current_rgb=rgb,
                        previous_valid=previous_mask,
                        current_valid=mask,
                        previous_timestamp_s=previous_pose["timestamp_s"],
                        current_timestamp_s=pose["timestamp_s"],
                        previous_world_from_camera=np.asarray(
                            previous_pose["rotation_matrix"], dtype=np.float64
                        ),
                        current_world_from_camera=rotation,
                        intrinsic=geometry.K,
                        protocol=protocol,
                        state=state,
                    )
                tracks = _final_compensated_tracks(
                    fit_calls,
                    list(r3_row.get("support_manager", {}).get("activated_cell_indices", [])),
                )
                previous_rotation = np.asarray(
                    previous_pose["rotation_matrix"], dtype=np.float64
                )
                previous_translation = np.asarray(
                    previous_pose["translation_m"], dtype=np.float64
                )
                dt = float(pose["timestamp_s"]) - float(
                    previous_pose["timestamp_s"]
                )
                target_delta = _target_z(frame_index, moving) - _target_z(
                    frame_index - 1, moving
                )
                prev, baseline, oracle, displacement, object_ids = (
                    _geometry_filter_and_oracle(
                        tracks,
                        previous_scene,
                        scene,
                        previous_rotation,
                        rotation,
                        previous_translation,
                        translation,
                        target_delta,
                    )
                )
                if displacement.size:
                    max_oracle_displacement = max(
                        max_oracle_displacement,
                        float(np.max(np.abs(displacement))),
                    )
                full = _paired_refit(
                    prev, baseline, oracle, dt, affine_parameters, 5
                )
                target_selected = object_ids == TARGET_ID
                target_fit = _paired_refit(
                    prev[target_selected],
                    baseline[target_selected],
                    oracle[target_selected],
                    dt,
                    affine_parameters,
                    1,
                )
                if not r3_row.get("evaluable", False):
                    full["evaluable"] = False
                    target_fit["evaluable"] = False
                elif not full["evaluable"]:
                    target_fit["evaluable"] = False
                for candidate in (full, target_fit):
                    if not candidate["evaluable"]:
                        for key in (
                            "baseline_signed_per_s",
                            "oracle_signed_per_s",
                            "baseline_absolute_per_s",
                            "oracle_absolute_per_s",
                        ):
                            candidate[key] = None
                row = {
                    "pair_index": frame_index - 1,
                    "dt_s": dt,
                    "r3_pair_evaluable": bool(r3_row.get("evaluable", False)),
                    "r3_pair_reason": r3_row.get("reason"),
                    "r3_common_cell_count": r3_row.get("common_cell_count"),
                    "geometry_valid_track_count": len(prev),
                    "target_track_count": int(np.count_nonzero(target_selected)),
                    "oracle_displacement_max_abs_px": (
                        float(np.max(np.abs(displacement)))
                        if displacement.size
                        else 0.0
                    ),
                    "full_scene": full,
                    "target_mask": target_fit,
                }
                for channel in ("full_scene", "target_mask"):
                    row[f"{channel}_baseline_signed_per_s"] = row[channel].get(
                        "baseline_signed_per_s"
                    )
                    row[f"{channel}_oracle_signed_per_s"] = row[channel].get(
                        "oracle_signed_per_s"
                    )
                rows.append(row)
                all_previous.append(prev.astype(np.float32))
                all_baseline.append(baseline.astype(np.float32))
                all_oracle.append(oracle.astype(np.float32))
                all_target.append(target_selected.astype(np.uint8))
                offsets.append(offsets[-1] + len(prev))
            previous_frame = (rgb, mask, pose, scene)
        arm_dir = staging / arm
        arm_dir.mkdir()
        ledger_path = arm_dir / "pair_ledger.jsonl"
        tracks_path = arm_dir / "paired_tracks.npz"
        metrics_path = arm_dir / "reduced_metrics.json"
        write_jsonl(ledger_path, rows)
        np.savez_compressed(
            tracks_path,
            offsets=np.asarray(offsets, dtype=np.int64),
            previous=np.concatenate(all_previous, axis=0),
            baseline=np.concatenate(all_baseline, axis=0),
            oracle=np.concatenate(all_oracle, axis=0),
            target=np.concatenate(all_target, axis=0),
        )
        metrics = {
            "cluster_id": cluster_id,
            "block": cluster["block"],
            "ordinal": cluster["ordinal"],
            "arm": arm,
            "full_scene": _reduce(rows, False),
            "target_mask": _reduce(rows, True),
            "oracle_displacement_max_abs_px": max_oracle_displacement,
        }
        write_json(metrics_path, metrics)
        receipt = {
            "sequence_id": next(
                sequence for sequence in cluster["sequence_ids"] if f"__{arm}__" in sequence
            ),
            "cluster_id": cluster_id,
            "arm": arm,
            "pair_ledger_sha256": sha256_file(ledger_path),
            "paired_tracks_sha256": sha256_file(tracks_path),
            "reduced_metrics_sha256": sha256_file(metrics_path),
            "r3_source_modified": False,
            "threshold_modified": False,
            "terminal": "STAGE_B_ARM_COMPLETE",
        }
        write_json(arm_dir / "receipt.json", receipt)
        arm_results.append(
            {
                "arm": arm,
                "receipt_sha256": sha256_file(arm_dir / "receipt.json"),
            }
        )
    cluster_receipt = {
        "cluster_id": cluster_id,
        "block": cluster["block"],
        "ordinal": cluster["ordinal"],
        "arm_count": len(arm_results),
        "arms": arm_results,
        "minimum_available_ram_bytes": minimum_memory,
        "wall_seconds": time.perf_counter() - started,
        "terminal": "STAGE_B_CLUSTER_COMPLETE",
    }
    write_json(staging / "cluster_receipt.json", cluster_receipt)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise InvalidStageB("FINAL_CLUSTER_EXISTS")
    os.replace(staging, final)
    return {
        **cluster_receipt,
        "cluster_receipt_sha256": sha256_file(final / "cluster_receipt.json"),
    }


def _validate_geometry_receipt(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "control" / "geometry_independent_receipt.json"
    manifest_path = output_root / "control" / "geometry_manifest.json"
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise InvalidStageB("INDEPENDENT_GEOMETRY_RECEIPT_MISSING")
    receipt = load_json(receipt_path)
    if (
        receipt.get("terminal") != "GEOMETRY_GATE_PASS / VALID"
        or receipt.get("valid") is not True
        or receipt.get("geometry_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise InvalidStageB("INDEPENDENT_GEOMETRY_RECEIPT_INVALID")
    return receipt


def _write_activation(output_root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    root = repo_root()
    activation_path = root / ACTIVATION_RELATIVE
    if activation_path.exists():
        raise InvalidStageB("ACTIVATION_ALREADY_EXISTS")
    activation = {
        "schema": "rcle.stage_b.execution_activation.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "date": "2026-07-29",
        "authority": "USER_DIRECT_ACTIVATION",
        "same_task_geometry_gate": True,
        "spec_path": SPEC_RELATIVE,
        "spec_sha256": sha256_file(root / SPEC_RELATIVE),
        "memory_gate_amendment_path": MEMORY_RELATIVE,
        "memory_gate_amendment_sha256": sha256_file(root / MEMORY_RELATIVE),
        "geometry_manifest_path": (
            output_root / "control" / "geometry_manifest.json"
        ).relative_to(root).as_posix(),
        "geometry_manifest_sha256": receipt["geometry_manifest_sha256"],
        "geometry_independent_receipt_path": (
            output_root / "control" / "geometry_independent_receipt.json"
        ).relative_to(root).as_posix(),
        "geometry_independent_receipt_sha256": sha256_file(
            output_root / "control" / "geometry_independent_receipt.json"
        ),
        "runner_source_path": Path(__file__).resolve().relative_to(root).as_posix(),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "launch_and_refill_available_ram_bytes": LAUNCH_REFILL_BYTES,
        "in_flight_emergency_runtime_floor_bytes": IN_FLIGHT_FLOOR_BYTES,
        "stage_b_execution_authorized": True,
        "stage_b_response_access_authorized": True,
        "formal_480_plus_16_authority_consumed": False,
        "feature_contract_c_or_fusion_d_authorized": False,
        "retry_replacement_or_reseed": False,
        "terminal": "STAGE_B_EXECUTION_ACTIVATED / GEOMETRY_GATE_PASS",
    }
    write_exclusive_json(activation_path, activation)
    return activation


def execute(output_root: Path) -> dict[str, Any]:
    root = repo_root()
    _, identity = _assert_bindings(root)
    response_root = output_root / "response"
    if response_root.exists():
        raise InvalidStageB("RESPONSE_ROOT_ALREADY_EXISTS")
    launch_memory = psutil.virtual_memory().available
    if launch_memory < LAUNCH_REFILL_BYTES:
        raise InvalidStageB("LAUNCH_AVAILABLE_RAM_BELOW_6_GIB")
    predecessor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p4_formal"
    )
    successor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_successor_formal"
    )
    previous_dev = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_four_block_dev_diagnostic_r0"
    )
    if (
        not predecessor_formal.is_dir()
        or successor_formal.exists()
        or not previous_dev.is_dir()
    ):
        raise InvalidStageB("FORMAL_OR_DEV_FIREWALL_PATH")
    firewall_before = {
        "predecessor_formal_tree_sha256": tree_sha256(predecessor_formal),
        "previous_dev_tree_sha256": tree_sha256(previous_dev),
    }
    geometry_receipt = _validate_geometry_receipt(output_root)
    activation = _write_activation(output_root, geometry_receipt)
    response_root.mkdir(parents=True, exist_ok=False)
    (response_root / "staging").mkdir()
    started = time.perf_counter()
    minimum_memory = launch_memory
    results = []
    started_swap = psutil.swap_memory()
    last_swap = started_swap
    paging_streak = 0
    telemetry = []
    worker_pids: list[int] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        remaining = list(identity["clusters"])
        active: dict[Any, dict[str, Any]] = {}
        while remaining or active:
            while remaining and len(active) < WORKERS:
                available = psutil.virtual_memory().available
                minimum_memory = min(minimum_memory, available)
                if available < LAUNCH_REFILL_BYTES:
                    raise InvalidStageB("REFILL_AVAILABLE_RAM_BELOW_6_GIB")
                cluster = remaining.pop(0)
                future = pool.submit(
                    _cluster_worker,
                    {
                        "cluster": cluster,
                        "response_root": str(response_root),
                    },
                )
                active[future] = cluster
                worker_pids = sorted(int(pid) for pid in pool._processes)
            completed, _ = wait(
                active, timeout=5.0, return_when=FIRST_COMPLETED
            )
            available = psutil.virtual_memory().available
            minimum_memory = min(minimum_memory, available)
            current_swap = psutil.swap_memory()
            paging_delta = max(
                0,
                int(current_swap.sin - last_swap.sin)
                + int(current_swap.sout - last_swap.sout),
            )
            paging_streak = paging_streak + 1 if paging_delta else 0
            last_swap = current_swap
            if available < IN_FLIGHT_FLOOR_BYTES:
                raise InvalidStageB("COORDINATOR_AVAILABLE_RAM_BELOW_4_GIB")
            if paging_streak >= 2:
                raise InvalidStageB("SUSTAINED_PAGING")
            for future in completed:
                cluster = active.pop(future)
                results.append(future.result())
                write_json(
                    response_root / "progress.json",
                    {
                        "completed_clusters": len(results),
                        "active_clusters": [
                            item["cluster_id"] for item in active.values()
                        ],
                        "remaining_clusters": len(remaining),
                        "last_completed": cluster["cluster_id"],
                        "updated_utc": utc_now(),
                    },
                )
            sample = {
                "sample_index": len(telemetry),
                "elapsed_seconds": time.perf_counter() - started,
                "available_ram_bytes": available,
                "swap_in_total": int(current_swap.sin),
                "swap_out_total": int(current_swap.sout),
                "paging_streak": paging_streak,
                "completed_clusters": len(results),
                "active_clusters": len(active),
                "remaining_clusters": len(remaining),
                "sampled_utc": utc_now(),
            }
            telemetry.append(sample)
            write_json(
                response_root / "telemetry.json",
                {
                    "task_id": TASK_ID,
                    "samples": telemetry,
                    "scientific_interpretation_present": False,
                },
            )
    residual_workers = [pid for pid in worker_pids if psutil.pid_exists(pid)]
    if residual_workers:
        raise InvalidStageB("RESIDUAL_WORKERS")
    firewall_after = {
        "predecessor_formal_tree_sha256": tree_sha256(predecessor_formal),
        "previous_dev_tree_sha256": tree_sha256(previous_dev),
    }
    if firewall_after != firewall_before or successor_formal.exists():
        raise InvalidStageB("FORMAL_OR_DEV_FIREWALL_DRIFT")
    results.sort(key=lambda item: (item["block"], item["ordinal"]))
    receipt = {
        "schema": "rcle.stage_b.run_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "activation_sha256": sha256_file(root / ACTIVATION_RELATIVE),
        "spec_sha256": sha256_file(root / SPEC_RELATIVE),
        "memory_gate_amendment_sha256": sha256_file(root / MEMORY_RELATIVE),
        "available_ram_at_launch_bytes": launch_memory,
        "minimum_available_ram_bytes": minimum_memory,
        "swap_in_delta": int(last_swap.sin - started_swap.sin),
        "swap_out_delta": int(last_swap.sout - started_swap.sout),
        "launch_and_refill_gate_bytes": LAUNCH_REFILL_BYTES,
        "in_flight_emergency_floor_bytes": IN_FLIGHT_FLOOR_BYTES,
        "workers": WORKERS,
        "cluster_count": len(results),
        "sequence_count": sum(item["arm_count"] for item in results),
        "planned_pair_count": len(results) * len(ARMS) * PAIR_COUNT,
        "clusters": results,
        "wall_seconds": time.perf_counter() - started,
        "formal_480_plus_16_sequences_run": 0,
        "formal_r3_pair_core_calls": 0,
        "formal_firewall_before": firewall_before,
        "formal_firewall_after": firewall_after,
        "successor_formal_path_absent": True,
        "worker_pids": worker_pids,
        "residual_worker_pids": residual_workers,
        "r3_modified": False,
        "threshold_three_pair_pairstate_abstention_modified": False,
        "terminal": "STAGE_B_EXECUTION_COMPLETE / INDEPENDENT_VALIDATION_REQUIRED",
    }
    write_exclusive_json(response_root / "run_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize-control")
    materialize.add_argument("--output-root", type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = repo_root()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (root / DEFAULT_ROOT).resolve()
    )
    if args.command == "materialize-control":
        result = materialize_control(output_root)
    else:
        result = execute(output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
