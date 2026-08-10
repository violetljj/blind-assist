#!/usr/bin/env python3
"""Validate lowest-persistent support identity against the ICL-NUIM exact mesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from diagnose_ag_st_tum_support_identity import SupportIdentityPolicy, _mode_candidates
from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-st-icl-support-identity-r0"
DEFAULT_MESH = DEFAULT_SOURCE_DIR / "living-room.obj"
DEFAULT_MESH_ARCHIVE = DEFAULT_SOURCE_DIR / "living_room_obj_mtl.tar.gz"
DEFAULT_GLOBAL_POSES = DEFAULT_SOURCE_DIR / "livingRoom0n.gt.sim"
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-icl-mesh-support-identity-r2/result.json"
)
MESH_URL = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/living_room_obj_mtl.tar.gz"
GLOBAL_POSE_URL = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom0n.gt.sim"
INTRINSICS = np.asarray(
    [[481.2, 0.0, 319.5], [0.0, -480.0, 239.5], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
VERTICAL_AXIS = 1
FLOOR_OBJECT = "room_floor"


def parse_global_pose_text(text: str) -> list[np.ndarray]:
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        values = [float(value) for value in line.split()]
        require(len(values) == 4 and all(math.isfinite(value) for value in values), "ICL pose row invalid")
        rows.append(values)
    require(len(rows) >= 3 and len(rows) % 3 == 0, "ICL global pose row count invalid")
    output: list[np.ndarray] = []
    for start in range(0, len(rows), 3):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = np.asarray(rows[start : start + 3], dtype=np.float64)
        rotation = pose[:3, :3]
        require(
            np.max(np.abs(rotation.T @ rotation - np.eye(3))) <= 2e-3
            and abs(np.linalg.det(rotation) - 1.0) <= 2e-3,
            "ICL global pose rotation invalid",
        )
        output.append(pose)
    return output


def _triangle_samples(vertices: np.ndarray, area: float, area_per_sample: float) -> np.ndarray:
    count = max(1, min(96, int(math.ceil(area / area_per_sample))))
    indices = np.arange(count, dtype=np.float64) + 0.5
    first = np.mod(indices * 0.7548776662466927, 1.0)
    second = np.mod(indices * 0.5698402909980532, 1.0)
    root = np.sqrt(first)
    weights = np.stack((1.0 - root, root * (1.0 - second), root * second), axis=1)
    return weights @ vertices


def load_horizontal_mesh_samples(
    path: Path,
    maximum_tilt_degrees: float = 20.0,
    area_per_sample_m2: float = 0.004,
) -> dict[str, Any]:
    vertices: list[list[float]] = []
    current_object = "unnamed"
    current_surface = current_object
    points: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    objects: list[str] = []
    horizontal_area = 0.0
    horizontal_triangle_count = 0
    cosine_limit = math.cos(math.radians(maximum_tilt_degrees))
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("o "):
            current_object = line[2:].strip()
            current_surface = current_object
        elif line.startswith("usemtl "):
            current_surface = line[7:].strip()
        elif line.startswith("v "):
            values = [float(value) for value in line.split()[1:4]]
            require(len(values) == 3 and all(math.isfinite(value) for value in values), "ICL OBJ vertex invalid")
            vertices.append(values)
        elif line.startswith("f "):
            indices = [int(token.split("/", 1)[0]) for token in line.split()[1:]]
            require(len(indices) >= 3 and all(index > 0 for index in indices), "ICL OBJ face invalid")
            for offset in range(1, len(indices) - 1):
                triangle = np.asarray(
                    [vertices[indices[0] - 1], vertices[indices[offset] - 1], vertices[indices[offset + 1] - 1]],
                    dtype=np.float64,
                )
                cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                twice_area = float(np.linalg.norm(cross))
                if twice_area <= 1e-10:
                    continue
                normal = cross / twice_area
                # Only upward-facing support candidates are admissible.  The
                # exact mesh contains enclosure undersides below room_floor;
                # absolute-normal filtering would make those occluded backsides
                # look like a lower traversable surface.
                if float(normal[VERTICAL_AXIS]) < cosine_limit:
                    continue
                area = twice_area / 2.0
                samples = _triangle_samples(triangle, area, area_per_sample_m2)
                points.append(samples)
                heights.append(samples[:, VERTICAL_AXIS])
                objects.extend([current_surface] * len(samples))
                horizontal_area += area
                horizontal_triangle_count += 1
    require(vertices and points, "ICL OBJ horizontal geometry missing")
    return {
        "points_world": np.concatenate(points, axis=0),
        "heights_world": np.concatenate(heights, axis=0),
        "object_names": np.asarray(objects, dtype=np.str_),
        "vertex_count": len(vertices),
        "horizontal_triangle_count": horizontal_triangle_count,
        "horizontal_area_m2": horizontal_area,
    }


def visible_sample_mask(points_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    points_camera = (points_world - translation) @ rotation
    depth = points_camera[:, 2]
    valid_depth = (depth > 0.10) & (depth <= 5.0)
    safe_depth = np.where(valid_depth, depth, 1.0)
    columns = INTRINSICS[0, 0] * points_camera[:, 0] / safe_depth + INTRINSICS[0, 2]
    rows = INTRINSICS[1, 1] * points_camera[:, 1] / safe_depth + INTRINSICS[1, 2]
    return (
        valid_depth
        & (columns >= 0.0)
        & (columns < IMAGE_WIDTH)
        & (rows >= 0.0)
        & (rows < IMAGE_HEIGHT)
    )


def run(mesh_path: Path, mesh_archive_path: Path, pose_path: Path) -> dict[str, Any]:
    require(mesh_path.is_file() and mesh_archive_path.is_file() and pose_path.is_file(), "ICL exact source missing")
    mesh = load_horizontal_mesh_samples(mesh_path)
    poses = parse_global_pose_text(pose_path.read_text(encoding="utf-8"))
    selected_indices = np.linspace(0, len(poses) - 1, 12, dtype=np.int64).tolist()
    points = mesh["points_world"]
    heights = mesh["heights_world"]
    objects = mesh["object_names"]
    floor_heights = heights[objects == FLOOR_OBJECT]
    require(len(floor_heights) > 0, "ICL room_floor object has no horizontal samples")
    exact_floor_height = float(np.median(floor_heights))
    frame_heights: list[np.ndarray] = []
    frame_receipts: list[dict[str, Any]] = []
    for index in selected_indices:
        pose = poses[index]
        visible = visible_sample_mask(points, pose)
        camera_height = float(pose[VERTICAL_AXIS, 3]) - heights
        plausible = visible & (camera_height >= 0.45) & (camera_height <= 2.20)
        values = heights[plausible]
        evaluable = len(values) >= 96
        if evaluable:
            frame_heights.append(values)
        floor_visible = plausible & (objects == FLOOR_OBJECT)
        frame_receipts.append(
            {
                "pose_index": int(index),
                "camera_world_height_m": float(pose[VERTICAL_AXIS, 3]),
                "visible_horizontal_sample_count": int(np.sum(plausible)),
                "visible_room_floor_sample_count": int(np.sum(floor_visible)),
                "evaluable": evaluable,
            }
        )
    policy = SupportIdentityPolicy(
        sample_stride=1,
        minimum_persistent_frames=3,
        minimum_frame_points=24,
        minimum_frame_fraction=0.001,
        minimum_total_points=192,
        minimum_total_fraction=0.001,
    )
    modes = _mode_candidates(frame_heights, policy)
    require(modes, "ICL support-height mode missing")
    selected_floor_height = float(modes[0]["world_height_m"])
    floor_error = abs(selected_floor_height - exact_floor_height)
    selected_floor_persistence = int(modes[0]["persistent_frame_count"])
    elevated_modes = [
        row
        for row in modes
        if float(row["world_height_m"]) - exact_floor_height >= 0.25
    ]
    camera_heights = np.asarray(
        [row["camera_world_height_m"] - selected_floor_height for row in frame_receipts]
    )
    camera_height_eligible = (camera_heights >= 0.45) & (camera_heights <= 2.20)
    identity_eligible: list[bool] = []
    for row, height, eligible in zip(
        frame_receipts,
        camera_heights,
        camera_height_eligible,
        strict=True,
    ):
        row["identity_camera_height_m"] = float(height)
        row["identity_camera_height_eligible"] = bool(eligible)
        row["support_identity_evaluable"] = bool(row["evaluable"] and eligible)
        identity_eligible.append(bool(row["evaluable"] and eligible))
    gates = {
        "exact_room_floor_recovered_within_0p04m": floor_error <= 0.04,
        "at_least_8_of_12_views_evaluable": len(frame_heights) >= 8,
        "floor_persistent_in_majority_of_evaluable_views": (
            selected_floor_persistence >= math.ceil(len(frame_heights) / 2)
        ),
        "at_least_one_elevated_horizontal_mode_retained": len(elevated_modes) >= 1,
        "support_identity_evaluable_in_at_least_8_views": sum(identity_eligible) >= 8,
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist_ag_st_icl_mesh_support_identity_result_v1",
        "status": (
            "ICL_EXACT_MESH_SUPPORT_IDENTITY_PASS"
            if passed
            else "ICL_EXACT_MESH_SUPPORT_IDENTITY_FAIL"
        ),
        "source": {
            "dataset": "ICL-NUIM living room lr kt0",
            "license": "CC BY 3.0",
            "dataset_url": "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html",
            "mesh_page_url": "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/living_room.html",
            "mesh_archive_url": MESH_URL,
            "mesh_archive_bytes": mesh_archive_path.stat().st_size,
            "mesh_archive_sha256": sha256_file(mesh_archive_path),
            "mesh_path": str(mesh_path.resolve()),
            "mesh_sha256": sha256_file(mesh_path),
            "global_pose_url": GLOBAL_POSE_URL,
            "global_pose_bytes": pose_path.stat().st_size,
            "global_pose_sha256": sha256_file(pose_path),
        },
        "selection": {
            "rule": "12 integer indices from numpy.linspace(0, pose_count-1, 12), before geometry outputs",
            "pose_count": len(poses),
            "selected_pose_indices": selected_indices,
        },
        "mesh": {
            "vertex_count": mesh["vertex_count"],
            "horizontal_triangle_count": mesh["horizontal_triangle_count"],
            "horizontal_area_m2": mesh["horizontal_area_m2"],
            "horizontal_sample_count": len(points),
            "vertical_axis": "+Y",
            "exact_floor_object": FLOOR_OBJECT,
            "exact_floor_world_height_m": exact_floor_height,
        },
        "support_identity": {
            "selected_lowest_persistent_world_height_m": selected_floor_height,
            "absolute_floor_height_error_m": floor_error,
            "selected_floor_persistent_frame_count": selected_floor_persistence,
            "elevated_mode_count": len(elevated_modes),
            "height_modes": modes,
            "camera_height_range_m": [float(np.min(camera_heights)), float(np.max(camera_heights))],
            "camera_height_eligible_frame_count": int(np.sum(camera_height_eligible)),
            "support_identity_evaluable_frame_count": sum(identity_eligible),
        },
        "frames": frame_receipts,
        "gates": gates,
        "decision": {
            "external_synthetic_exact_support_identity_supported": passed,
            "complete_truth_required": False,
            "boundary_external_exact_validated": False,
            "student_training_authorized": passed,
            "student_training_scope": (
                "masked depth/support canary using identity-valid labels only; boundary remains diagnostic"
                if passed
                else None
            ),
        },
        "claim_boundary": (
            "External synthetic exact-mesh validation of the sequence-level lowest-persistent "
            "support-height identity. Frustum centroid/sample visibility is not pixel-exact occlusion; "
            "this is not real-world walkability truth, boundary validation, task utility, product, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--mesh-archive", type=Path, default=DEFAULT_MESH_ARCHIVE)
    parser.add_argument("--global-poses", type=Path, default=DEFAULT_GLOBAL_POSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"ICL mesh result exists: {args.output}")
    result = run(args.mesh, args.mesh_archive, args.global_poses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "support_identity", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
