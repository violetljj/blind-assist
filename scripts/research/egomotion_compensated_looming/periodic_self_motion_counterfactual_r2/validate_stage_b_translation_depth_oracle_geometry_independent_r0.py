"""Independent pre-response geometry gate for Stage B.

This validator does not import the Stage B runner, R3, tracking, evaluation or
local-fit code.  It independently reconstructs the frozen geometry and pose
inputs, audits all persistent target material points, and writes a control-only
receipt.  The response root must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import generator_geometry as geometry


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
    "TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0"
)
ARMS = (
    "STATIC_SCENE",
    "EGO_ROTATION_STATIC_SCENE",
    "EGO_TRANSLATION_STATIC_SCENE",
    "OBJECT_APPROACH_STATIC_CAMERA",
    "OBJECT_APPROACH_PLUS_EGO_6DOF",
)
TARGET_ID = 1001
FRAME_COUNT = 602
PAIR_COUNT = 601
GIB = 1024**3
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
TRAJECTORY_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
DEFAULT_ROOT = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_stage_b_translation_depth_oracle_object_approach_r0"
)


class InvalidGeometry(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item(),
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, label: str) -> None:
    if not condition:
        raise InvalidGeometry(label)


def expected_scene(cluster: dict[str, Any]) -> dict[str, Any]:
    seed = int(cluster["numeric_seed_uint64"])
    objects = []
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
    result = {
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
    result["scene_geometry_sha256"] = sha256_value(result)
    return result


def target_z(frame_index: int, moving: bool) -> float:
    if not moving:
        return 6.0
    tau = frame_index / PAIR_COUNT
    return 6.0 - 2.0 * (0.5 - 0.5 * math.cos(math.pi * tau))


def dynamic_scene(
    base: dict[str, Any], frame_index: int, moving: bool
) -> dict[str, Any]:
    scene = json.loads(json.dumps(base))
    target = next(
        item for item in scene["world"]["objects"] if item["object_id"] == TARGET_ID
    )
    z = target_z(frame_index, moving)
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


def arm_pose(trajectory: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    identity = np.eye(3, dtype=np.float64).tolist()
    zero = [0.0, 0.0, 0.0]
    return [
        {
            "frame_index": source["frame_index"],
            "timestamp_s": source["timestamp_s"],
            "rotation_matrix": (
                source["rotation_matrix"]
                if arm
                in {
                    "EGO_ROTATION_STATIC_SCENE",
                    "OBJECT_APPROACH_PLUS_EGO_6DOF",
                }
                else identity
            ),
            "translation_m": (
                source["translation_m"]
                if arm
                in {
                    "EGO_TRANSLATION_STATIC_SCENE",
                    "OBJECT_APPROACH_PLUS_EGO_6DOF",
                }
                else zero
            ),
        }
        for source in trajectory["poses"]
    ]


def project_world(
    world: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation.T @ (world - translation).T).T
    depth = camera[:, 2]
    pixel = np.full((len(world), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (depth > 0.0)
    pixel[valid, 0] = (
        geometry.K[0, 0] * camera[valid, 0] / depth[valid]
        + geometry.K[0, 2]
    )
    pixel[valid, 1] = (
        geometry.K[1, 1] * camera[valid, 1] / depth[valid]
        + geometry.K[1, 2]
    )
    return pixel, depth


def target_points(z: float) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for y in np.linspace(-0.6, 0.8, 5)
            for x in np.linspace(-0.3, 0.7, 5)
        ],
        dtype=np.float64,
    )


def _collect_scalars(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            found.update(_collect_scalars(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_scalars(child))
    elif isinstance(value, str):
        found.add(value)
    return found


def validate(output_root: Path) -> dict[str, Any]:
    root = repo_root()
    response = output_root / "response"
    require(not response.exists(), "G00_RESPONSE_ROOT_PREEXISTS")
    activation = root / (
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
        "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_EXECUTION_ACTIVATION_R1_"
        "2026-07-29.json"
    )
    require(not activation.exists(), "G00_ACTIVATION_PREEXISTS")
    manifest_path = output_root / "control" / "geometry_manifest.json"
    receipt_path = output_root / "control" / "geometry_independent_receipt.json"
    require(manifest_path.is_file(), "G00_MANIFEST_MISSING")
    require(not receipt_path.exists(), "G00_RECEIPT_PREEXISTS")
    manifest = load_json(manifest_path)
    spec = load_json(root / SPEC_RELATIVE)
    identity = load_json(root / IDENTITY_RELATIVE)
    trajectory = load_json(root / TRAJECTORY_RELATIVE)
    require(manifest["spec_sha256"] == sha256_file(root / SPEC_RELATIVE), "G01_SPEC")
    require(
        manifest["identity_lock_sha256"] == sha256_file(root / IDENTITY_RELATIVE),
        "G01_IDENTITY",
    )
    require(manifest["cluster_count"] == 8, "G01_CLUSTER_COUNT")
    require(manifest["sequence_count"] == 40, "G01_SEQUENCE_COUNT")
    require(spec["memory_gate"]["launch_and_refill_available_ram_bytes"] == 6 * GIB, "G01_RAM6")
    require(spec["memory_gate"]["in_flight_emergency_runtime_floor_bytes"] == 4 * GIB, "G01_RAM4")

    exclusion = load_json(root / identity["exclusion_authority"]["path"])
    external = _collect_scalars(exclusion)
    expected_sequences: set[str] = set()
    seen_tokens: set[str] = set()
    for cluster in identity["clusters"]:
        token = (
            f"{TASK_ID}|SCENE|{cluster['block']}|{int(cluster['ordinal']):02d}"
        )
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        require(cluster["token"] == token, "G02_TOKEN")
        require(cluster["token_sha256"] == digest, "G02_TOKEN_HASH")
        require(
            cluster["numeric_seed_uint64"]
            == int.from_bytes(bytes.fromhex(digest)[:8], "big"),
            "G02_SEED",
        )
        require(token not in seen_tokens and token not in external, "G02_COLLISION")
        seen_tokens.add(token)
        for arm in ARMS:
            sequence = f"{cluster['cluster_id']}__{arm}__CLEAN"
            expected_sequences.add(sequence)
            require(sequence in cluster["sequence_ids"], "G02_SEQUENCE_SET")
            require(sequence not in external, "G02_SEQUENCE_COLLISION")
    require(len(expected_sequences) == 40, "G02_SEQUENCE_UNIQUENESS")

    manifest_by_cluster = {
        item["cluster_id"]: item for item in manifest["clusters"]
    }
    minimum_visible = 25
    arm_checks = 0
    for cluster in identity["clusters"]:
        observed = manifest_by_cluster.get(cluster["cluster_id"])
        require(observed is not None, "G03_CLUSTER_MISSING")
        expected = expected_scene(cluster)
        require(observed["base_scene"] == expected, "G03_BASE_SCENE")
        require(
            observed["base_scene_sha256"] == expected["scene_geometry_sha256"],
            "G03_BASE_HASH",
        )
        target = next(
            item
            for item in expected["world"]["objects"]
            if item["object_id"] == TARGET_ID
        )
        require(target["triangles"] == [[0, 1, 2], [0, 2, 3]], "G03_TRIANGLES")
        require(target["bounds_xy_m"] == [-0.4, 0.8, -0.7, 0.9], "G03_TARGET_BOUNDS")
        traj = trajectory[cluster["block"]]
        observed_arms = {item["arm"]: item for item in observed["arms"]}
        for arm in ARMS:
            arm_checks += 1
            poses = arm_pose(traj, arm)
            moving = arm in {
                "OBJECT_APPROACH_STATIC_CAMERA",
                "OBJECT_APPROACH_PLUS_EGO_6DOF",
            }
            frame_inputs = []
            arm_minimum = 25
            for frame_index, pose in enumerate(poses):
                scene = dynamic_scene(expected, frame_index, moving)
                z = target_z(frame_index, moving)
                material = target_points(z)
                rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
                translation = np.asarray(pose["translation_m"], dtype=np.float64)
                pixels, projected_depth = project_world(
                    material, rotation, translation
                )
                safe = (
                    np.isfinite(pixels).all(axis=1)
                    & (pixels[:, 0] >= 0.0)
                    & (pixels[:, 0] < geometry.WIDTH - 1.0)
                    & (pixels[:, 1] >= 0.0)
                    & (pixels[:, 1] < geometry.HEIGHT - 1.0)
                )
                visible = np.zeros(25, dtype=bool)
                if np.any(safe):
                    depth, object_id, world = geometry._raycast(
                        scene, rotation, translation, pixels[safe]
                    )
                    tolerance = np.maximum(
                        1e-7, 1e-6 * projected_depth[safe]
                    )
                    visible[safe] = (
                        np.isfinite(depth)
                        & (object_id == TARGET_ID)
                        & (np.abs(depth - projected_depth[safe]) <= tolerance)
                        & (
                            np.linalg.norm(world - material[safe], axis=1)
                            <= tolerance
                        )
                    )
                arm_minimum = min(arm_minimum, int(np.count_nonzero(visible)))
                frame_inputs.append(
                    {
                        "frame_index": frame_index,
                        "scene_geometry_sha256": scene["scene_geometry_sha256"],
                        "rotation_matrix": pose["rotation_matrix"],
                        "translation_m": pose["translation_m"],
                        "target_z_m": z,
                    }
                )
            minimum_visible = min(minimum_visible, arm_minimum)
            recorded = observed_arms[arm]
            require(recorded["pose_sha256"] == sha256_value(poses), "G04_POSE")
            require(
                recorded["render_input_sha256"] == sha256_value(frame_inputs),
                "G04_RENDER_INPUT",
            )
            require(recorded["frame_count"] == FRAME_COUNT, "G04_FRAME_COUNT")
            require(arm_minimum == 25, "G05_VISIBILITY")
            require(
                recorded["minimum_visible_target_points"] == arm_minimum,
                "G05_RECORDED_VISIBILITY",
            )
    require(abs(manifest["target_center_radial_scale"] - 1.5) <= 1e-12, "G06_SCALE")
    require(manifest["stage_b_response_read"] is False, "G07_RESPONSE_READ")
    require(manifest["stage_b_workload_run"] is False, "G07_WORKLOAD")
    require(manifest["formal_480_plus_16_access_count"] == 0, "G07_FORMAL")
    gates = {
        "G00_pre_response_firewall": "PASS",
        "G01_bindings_and_6gib_policy": "PASS",
        "G02_all_40_identity_disjointness": "PASS",
        "G03_mesh_material_and_target_identity": "PASS",
        "G04_all_render_input_hashes": "PASS",
        "G05_all_frame_visibility_and_zbuffer": "PASS",
        "G06_endpoint_scale": "PASS",
        "G07_formal_firewall": "PASS",
    }
    receipt = {
        "schema": "rcle.stage_b.geometry_independent_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "valid": True,
        "geometry_manifest_path": manifest_path.relative_to(root).as_posix(),
        "geometry_manifest_sha256": sha256_file(manifest_path),
        "validator_source_path": Path(__file__).resolve().relative_to(root).as_posix(),
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
        "cluster_count": 8,
        "arm_count": arm_checks,
        "frame_inputs_checked": arm_checks * FRAME_COUNT,
        "persistent_material_points_per_frame": 25,
        "minimum_visible_target_points": minimum_visible,
        "gates": gates,
        "stage_b_response_read": False,
        "stage_b_workload_run": False,
        "formal_480_plus_16_access_count": 0,
        "terminal": "GEOMETRY_GATE_PASS / VALID",
    }
    write_exclusive(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = repo_root()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (root / DEFAULT_ROOT).resolve()
    )
    result = validate(output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
