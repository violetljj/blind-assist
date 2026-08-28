from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scenefun3d_functional_handoff_ceiling import (
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _transform_points,
)


FRAME_STRIDE = 6
POSE_TIME_TOLERANCE_S = 0.06
DEPTH_TIME_TOLERANCE_S = 0.06
MIN_STANDOFF_M = 0.35
MAX_STANDOFF_M = 1.10
MAX_ORIENTATION_ERROR_DEGREES = 25.0
DEPTH_CONSISTENCY_M = 0.25
MIN_VISIBLE_POINT_FRACTION = 0.50
PARENT_PROXY_MIN_AREA_FRACTION = 0.12
PARENT_PROXY_CENTER_TOLERANCE_X = 0.20
PARENT_PROXY_CENTER_TOLERANCE_Y = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_poses(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 7:
            raise ValueError(f"Invalid trajectory row: {line!r}")
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = cv2.Rodrigues(
            np.asarray(values[1:4], dtype=np.float64)
        )[0]
        world_to_camera[:3, 3] = values[4:7]
        timestamps.append(values[0])
        poses.append(np.linalg.inv(world_to_camera))
    order = np.argsort(timestamps)
    return np.asarray(timestamps)[order], [poses[index] for index in order]


def _nearest_index(
    timestamps: np.ndarray, desired: float, tolerance: float
) -> int | None:
    insertion = int(np.searchsorted(timestamps, desired))
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(timestamps)
    ]
    if not candidates:
        return None
    selected = min(candidates, key=lambda index: abs(float(timestamps[index]) - desired))
    if abs(float(timestamps[selected]) - desired) > tolerance:
        return None
    return selected


def _read_intrinsic(path: Path) -> tuple[int, int, np.ndarray]:
    width, height, fx, fy, cx, cy = [
        float(value) for value in path.read_text(encoding="utf-8").split()
    ]
    return int(width), int(height), np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _project_points(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points, np.ones(len(points)))).T
    camera = np.linalg.inv(camera_to_world) @ homogeneous
    positive = camera[2] > 0.05
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if positive.any():
        projected = intrinsic @ camera[:3, positive]
        pixels[positive, 0] = projected[0] / projected[2]
        pixels[positive, 1] = projected[1] / projected[2]
    in_frame = (
        positive
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return pixels, camera[2], in_frame


def _visible_fraction(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    depth_mm: np.ndarray,
) -> float:
    height, width = depth_mm.shape
    pixels, expected_depth, in_frame = _project_points(
        points, camera_to_world, intrinsic, width, height
    )
    indices = np.flatnonzero(in_frame)
    if len(indices) == 0:
        return 0.0
    uv = np.rint(pixels[indices]).astype(np.int64)
    uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)
    observed = depth_mm[uv[:, 1], uv[:, 0]].astype(np.float64) / 1000.0
    valid = observed > 0.0
    consistent = valid & (np.abs(observed - expected_depth[indices]) <= DEPTH_CONSISTENCY_M)
    return float(consistent.sum() / len(points))


def _orientation_error_degrees(
    camera_to_world: np.ndarray, target: np.ndarray
) -> float:
    camera = camera_to_world[:3, 3]
    direction = target - camera
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return 180.0
    direction /= norm
    forward = camera_to_world[:3, 2]
    forward /= np.linalg.norm(forward)
    return math.degrees(math.acos(float(np.clip(np.dot(forward, direction), -1.0, 1.0))))


def _horizontal_distance(camera_to_world: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(camera_to_world[:2, 3] - target[:2]))


def _observational_ready(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    depth_mm: np.ndarray,
) -> tuple[bool, dict[str, float | bool]]:
    center = np.mean(points, axis=0)
    distance = _horizontal_distance(camera_to_world, center)
    orientation_error = _orientation_error_degrees(camera_to_world, center)
    visible_fraction = _visible_fraction(points, camera_to_world, intrinsic, depth_mm)
    position_ready = MIN_STANDOFF_M <= distance <= MAX_STANDOFF_M
    visibility_ready = visible_fraction >= MIN_VISIBLE_POINT_FRACTION
    orientation_ready = orientation_error <= MAX_ORIENTATION_ERROR_DEGREES
    return position_ready and visibility_ready and orientation_ready, {
        "horizontal_standoff_m": round(distance, 6),
        "visible_point_fraction": round(visible_fraction, 6),
        "orientation_error_degrees": round(orientation_error, 6),
        "position_ready": position_ready,
        "visibility_ready": visibility_ready,
        "orientation_ready": orientation_ready,
    }


def _parent_proxy_ready(
    parent: Any,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> bool:
    local = np.asarray(
        [
            (sx, sy, sz)
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.5, 0.5)
        ],
        dtype=np.float64,
    )
    corners = (local * parent.lengths) @ parent.axes + parent.center
    homogeneous = np.column_stack((corners, np.ones(len(corners)))).T
    camera = np.linalg.inv(camera_to_world) @ homogeneous
    valid = camera[2] > 0.05
    if int(valid.sum()) < 4:
        return False
    projected = intrinsic @ camera[:3, valid]
    u = projected[0] / projected[2]
    v = projected[1] / projected[2]
    x1 = max(0.0, float(u.min()))
    y1 = max(0.0, float(v.min()))
    x2 = min(float(width - 1), float(u.max()))
    y2 = min(float(height - 1), float(v.max()))
    if x2 <= x1 or y2 <= y1:
        return False
    area_fraction = ((x2 - x1) * (y2 - y1)) / float(width * height)
    center_x = (x1 + x2) / (2.0 * width)
    center_y = (y1 + y2) / (2.0 * height)
    return bool(
        PARENT_PROXY_MIN_AREA_FRACTION <= area_fraction <= 0.98
        and abs(center_x - 0.5) <= PARENT_PROXY_CENTER_TOLERANCE_X
        and abs(center_y - 0.5) <= PARENT_PROXY_CENTER_TOLERANCE_Y
    )


def _build_task_inputs(sc11_provider: dict[str, Any], sc11_result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        row["candidate_id"]: row
        for rows in sc11_provider["arms"]["multiview"].values()
        for row in rows
    }
    tasks: list[dict[str, Any]] = []
    for row in sc11_result["arms"]["multiview"]["task"]["tasks"]:
        selected = [candidates[candidate_id] for candidate_id in row["selected_candidate_ids"]]
        tasks.append(
            {
                "desc_id": row["desc_id"],
                "description": row["description"],
                "parent_binding_id": row["parent_binding_id"],
                "grounding_state": row["state"],
                "selected_candidate_ids": row["selected_candidate_ids"],
                "predicted_points_xyz_m": [candidate["center_xyz_m"] for candidate in selected],
            }
        )
    return tasks


def build_provider(
    scene_dir: Path,
    video_id: str,
    sc11_provider_path: Path,
    sc11_result_path: Path,
) -> dict[str, Any]:
    video_dir = scene_dir / video_id
    sc11_provider = _load_json(sc11_provider_path)
    sc11_result = _load_json(sc11_result_path)
    tasks = _build_task_inputs(sc11_provider, sc11_result)
    parents = {
        parent.binding_id: parent
        for parent in _load_parent_boxes(video_dir / f"{video_id}_3dod_annotation.json")
    }
    pose_timestamps, poses = _read_poses(video_dir / "lowres_poses.traj")
    depth_paths = sorted((video_dir / "lowres_depth").glob("*.png"))
    depth_timestamps = np.asarray(
        [float(path.stem.split("_")[-1]) for path in depth_paths], dtype=np.float64
    )

    frames: list[dict[str, Any]] = []
    rgb_paths = sorted((video_dir / "lowres_wide").glob("*.png"))[::FRAME_STRIDE]
    for image_path in rgb_paths:
        timestamp = float(image_path.stem.split("_")[-1])
        pose_index = _nearest_index(pose_timestamps, timestamp, POSE_TIME_TOLERANCE_S)
        depth_index = _nearest_index(depth_timestamps, timestamp, DEPTH_TIME_TOLERANCE_S)
        intrinsic_path = video_dir / "lowres_wide_intrinsics" / f"{image_path.stem}.pincam"
        if pose_index is None or depth_index is None or not intrinsic_path.is_file():
            continue
        width, height, intrinsic = _read_intrinsic(intrinsic_path)
        depth_mm = cv2.imread(str(depth_paths[depth_index]), cv2.IMREAD_UNCHANGED)
        if depth_mm is not None:
            depth_mm = np.squeeze(depth_mm)
        if depth_mm is None or depth_mm.shape != (height, width):
            continue
        task_rows: list[dict[str, Any]] = []
        for task in tasks:
            predicted = np.asarray(task["predicted_points_xyz_m"], dtype=np.float64)
            ready, factors = _observational_ready(predicted, poses[pose_index], intrinsic, depth_mm)
            task_rows.append(
                {
                    "desc_id": task["desc_id"],
                    "parent_proxy_ready": _parent_proxy_ready(
                        parents[task["parent_binding_id"]],
                        poses[pose_index],
                        intrinsic,
                        width,
                        height,
                    ),
                    "factorized_observational_ready": ready,
                    "factors": factors,
                }
            )
        frames.append(
            {
                "timestamp": round(timestamp, 6),
                "frame_name": image_path.name,
                "depth_frame_name": depth_paths[depth_index].name,
                "camera_to_world": np.round(poses[pose_index], 8).tolist(),
                "tasks": task_rows,
            }
        )

    return {
        "schema_version": 1,
        "provider": "L10-SC20-FACTORIZED-FUNCTIONAL-ENDPOINT-OBSERVER",
        "source": {
            "visit_id": scene_dir.name,
            "video_id": video_id,
            "sc11_provider_sha256": _sha256(sc11_provider_path),
            "sc11_result_sha256": _sha256(sc11_result_path),
            "trajectory_sha256": _sha256(video_dir / "lowres_poses.traj"),
        },
        "truth_isolation": (
            "The provider reads only SC11 task description, authorized parent binding, belief state, "
            "and selected candidate IDs/coordinates. SC11 evaluator target IDs, legal_commit, "
            "distances, and SceneFun3D functional annotations are not copied or used by the observer."
        ),
        "frozen_contract": {
            "frame_stride": FRAME_STRIDE,
            "minimum_standoff_m": MIN_STANDOFF_M,
            "maximum_standoff_m": MAX_STANDOFF_M,
            "maximum_orientation_error_degrees": MAX_ORIENTATION_ERROR_DEGREES,
            "depth_consistency_m": DEPTH_CONSISTENCY_M,
            "minimum_visible_point_fraction": MIN_VISIBLE_POINT_FRACTION,
            "reachability": "UNKNOWN_NO_FREE_SPACE_OR_HUMAN_BODY_AUTHORITY",
            "handoff_ready": "FORBIDDEN_WITH_UNKNOWN_REACHABILITY",
            "completion_authority": "EXPLICIT_USER_CONFIRMATION_ONLY",
        },
        "tasks": tasks,
        "frames": frames,
    }


def evaluate_provider(
    scene_dir: Path, video_id: str, provider: dict[str, Any], provider_sha256: str
) -> dict[str, Any]:
    visit_id = scene_dir.name
    annotations_path = scene_dir / f"{visit_id}_annotations.json"
    descriptions_path = scene_dir / f"{visit_id}_descriptions.json"
    laser_path = scene_dir / f"{visit_id}_laser_scan.ply"
    transform_path = scene_dir / video_id / f"{video_id}_transform.npy"
    annotations = {
        row["annot_id"]: row for row in _load_json(annotations_path)["annotations"]
    }
    descriptions = {
        row["desc_id"]: row for row in _load_json(descriptions_path)["descriptions"]
    }
    xyz = _load_ply_xyz(laser_path)
    transform = np.load(transform_path)
    truth_points: dict[str, np.ndarray] = {}
    for task in provider["tasks"]:
        description = descriptions[task["desc_id"]]
        parts = [
            _transform_points(
                xyz[np.asarray(annotations[target_id]["indices"], dtype=np.int64)],
                transform,
            )
            for target_id in description["annot_id"]
        ]
        merged = np.concatenate(parts, axis=0)
        if len(merged) > 256:
            sample_indices = np.linspace(0, len(merged) - 1, 256, dtype=np.int64)
            merged = merged[sample_indices]
        truth_points[task["desc_id"]] = merged

    counts = {
        "truth_ready_frames": 0,
        "parent_proxy_ready_frames": 0,
        "parent_proxy_true_ready_frames": 0,
        "parent_proxy_false_ready_frames": 0,
        "factorized_ready_frames": 0,
        "factorized_true_ready_frames": 0,
        "factorized_false_ready_frames": 0,
    }
    task_counts: dict[str, dict[str, int]] = {
        task["desc_id"]: {
            "truth_ready": 0,
            "parent_proxy_ready": 0,
            "parent_proxy_true_ready": 0,
            "factorized_ready": 0,
            "factorized_true_ready": 0,
        }
        for task in provider["tasks"]
    }
    for frame in provider["frames"]:
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)
        intrinsic_path = (
            scene_dir
            / video_id
            / "lowres_wide_intrinsics"
            / Path(frame["frame_name"]).with_suffix(".pincam")
        )
        width, height, intrinsic = _read_intrinsic(intrinsic_path)
        depth_mm = cv2.imread(
            str(scene_dir / video_id / "lowres_depth" / frame["depth_frame_name"]),
            cv2.IMREAD_UNCHANGED,
        )
        if depth_mm is not None:
            depth_mm = np.squeeze(depth_mm)
        if depth_mm is None or depth_mm.shape != (height, width):
            raise ValueError(f"Missing evaluator depth for {frame['frame_name']}")
        for output in frame["tasks"]:
            desc_id = output["desc_id"]
            truth_ready, _ = _observational_ready(
                truth_points[desc_id], camera_to_world, intrinsic, depth_mm
            )
            parent_ready = bool(output["parent_proxy_ready"])
            factorized_ready = bool(output["factorized_observational_ready"])
            task_counts[desc_id]["truth_ready"] += int(truth_ready)
            task_counts[desc_id]["parent_proxy_ready"] += int(parent_ready)
            task_counts[desc_id]["parent_proxy_true_ready"] += int(parent_ready and truth_ready)
            task_counts[desc_id]["factorized_ready"] += int(factorized_ready)
            task_counts[desc_id]["factorized_true_ready"] += int(factorized_ready and truth_ready)
            counts["truth_ready_frames"] += int(truth_ready)
            counts["parent_proxy_ready_frames"] += int(parent_ready)
            counts["parent_proxy_true_ready_frames"] += int(parent_ready and truth_ready)
            counts["parent_proxy_false_ready_frames"] += int(parent_ready and not truth_ready)
            counts["factorized_ready_frames"] += int(factorized_ready)
            counts["factorized_true_ready_frames"] += int(factorized_ready and truth_ready)
            counts["factorized_false_ready_frames"] += int(factorized_ready and not truth_ready)

    def metrics(prefix: str) -> dict[str, float | int]:
        ready = counts[f"{prefix}_ready_frames"]
        true_ready = counts[f"{prefix}_true_ready_frames"]
        truth_ready = counts["truth_ready_frames"]
        precision = true_ready / ready if ready else 0.0
        recall = true_ready / truth_ready if truth_ready else 0.0
        return {
            "ready_frames": ready,
            "true_ready_frames": true_ready,
            "false_ready_frames": counts[f"{prefix}_false_ready_frames"],
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tasks_with_true_ready": sum(
                row[f"{prefix}_true_ready"] > 0 for row in task_counts.values()
            ),
        }

    baseline = metrics("parent_proxy")
    successor = metrics("factorized")
    decision = (
        "SC20_FACTORIZED_ENDPOINT_DEVELOPMENT_SIGNAL"
        if successor["precision"] > baseline["precision"]
        and successor["false_ready_frames"] < baseline["false_ready_frames"]
        and successor["tasks_with_true_ready"] >= baseline["tasks_with_true_ready"]
        else "SC20_FACTORIZED_ENDPOINT_GATE_NOT_MET"
    )
    return {
        "schema_version": 1,
        "experiment": "L10-SC20-FACTORIZED-FUNCTIONAL-ENDPOINT-OBSERVER",
        "decision": decision,
        "claim_layer": "CONSUMED_REAL_RGBD_TRAJECTORY_ENDPOINT_MECHANICS_DIAGNOSTIC",
        "provider_sha256": provider_sha256,
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "tasks": len(provider["tasks"]),
            "frames": len(provider["frames"]),
            "task_frames": len(provider["tasks"]) * len(provider["frames"]),
            "truth_ready_frames": counts["truth_ready_frames"],
        },
        "parent_centered_large_box_proxy": baseline,
        "factorized_functional_endpoint": successor,
        "per_task": [
            {"desc_id": task["desc_id"], "description": task["description"], **task_counts[task["desc_id"]]}
            for task in provider["tasks"]
        ],
        "claim_boundary": (
            "This is a Development diagnostic on the already-consumed SceneFun3D 420683 trajectory. "
            "It tests whether real posed RGB-D plus SC11 functional candidates can reject false observational "
            "arrival from a centered-large-parent proxy. Parent bindings are privileged, task geometry is not "
            "phone-camera identity, and the recorded path was not executed by BlindAssist. Reachability is UNKNOWN, "
            "so this observer cannot emit HANDOFF_READY and explicit user confirmation remains the only completion authority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sc11-provider", type=Path, required=True)
    parser.add_argument("--sc11-result", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provider = build_provider(
        args.scene_dir.resolve(),
        args.video_id,
        args.sc11_provider.resolve(),
        args.sc11_result.resolve(),
    )
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_sha256 = _sha256(args.provider_output)
    result = evaluate_provider(
        args.scene_dir.resolve(), args.video_id, provider, provider_sha256
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "denominators": result["denominators"],
        "parent_proxy": result["parent_centered_large_box_proxy"],
        "factorized": result["factorized_functional_endpoint"],
    }, indent=2))


if __name__ == "__main__":
    main()
