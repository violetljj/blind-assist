from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from functional_part_binding import FunctionalPartCandidate, TaskRelationalFunctionalSelector
from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    ParentBox,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _match_parent,
    _nearest_distance,
    _transform_points,
)


FRAME_STRIDE = 6
MODEL_IMAGE_SIZE = 640
MODEL_BATCH = 8
MODEL_CONFIDENCE = 0.25
MAX_HANDLE_TO_PARENT_AREA_RATIO = 0.20
MIN_PARENT_IMAGE_AREA_FRACTION = 0.015
MAX_PARENT_IMAGE_AREA_FRACTION = 0.98
POSE_TIME_TOLERANCE_S = 0.06
MULTIVIEW_CLUSTER_RADIUS_M = 0.10
MULTIVIEW_MIN_OBSERVATIONS = 3
MULTIVIEW_MIN_TIME_SPAN_S = 0.30
MULTIVIEW_MIN_CAMERA_BASELINE_M = 0.08
MAX_PROPOSALS_PER_PARENT = 12
FUNCTIONAL_PART_MATCH_TOLERANCE_M = 0.12


@dataclass(frozen=True)
class FrameGeometry:
    timestamp: float
    image_path: Path
    intrinsic_path: Path
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    parent_projections: dict[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class HandleObservation:
    parent_binding_id: str
    timestamp: float
    frame_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    point_xyz_m: tuple[float, float, float]
    camera_xyz_m: tuple[float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_poses(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamps: list[float] = []
    camera_to_world: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            values = [float(value) for value in line.split()]
            if len(values) != 7:
                raise ValueError(f"Invalid trajectory row in {path}: {line!r}")
            rotation = cv2.Rodrigues(np.asarray(values[1:4], dtype=np.float64))[0]
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = rotation
            world_to_camera[:3, 3] = values[4:7]
            timestamps.append(values[0])
            camera_to_world.append(np.linalg.inv(world_to_camera))
    order = np.argsort(timestamps)
    return np.asarray(timestamps)[order], [camera_to_world[index] for index in order]


def _read_intrinsic(path: Path) -> tuple[int, int, np.ndarray]:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    if len(values) != 6:
        raise ValueError(f"Invalid pinhole intrinsics file: {path}")
    width, height, fx, fy, cx, cy = values
    intrinsic = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return int(width), int(height), intrinsic


def _box_corners(box: ParentBox) -> np.ndarray:
    local = np.asarray(
        [
            (sx, sy, sz)
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.5, 0.5)
        ],
        dtype=np.float64,
    )
    return (local * box.lengths) @ box.axes + box.center


def _project_parent(
    box: ParentBox,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    corners = _box_corners(box)
    homogeneous = np.column_stack((corners, np.ones(len(corners)))).T
    camera = np.linalg.inv(camera_to_world) @ homogeneous
    valid = camera[2] > 0.05
    if int(valid.sum()) < 4:
        return None
    pixels = intrinsic @ camera[:3, valid]
    u = pixels[0] / pixels[2]
    v = pixels[1] / pixels[2]
    x1 = max(0.0, float(u.min()))
    y1 = max(0.0, float(v.min()))
    x2 = min(float(width - 1), float(u.max()))
    y2 = min(float(height - 1), float(v.max()))
    if x2 <= x1 or y2 <= y1:
        return None
    area_fraction = ((x2 - x1) * (y2 - y1)) / float(width * height)
    if not MIN_PARENT_IMAGE_AREA_FRACTION <= area_fraction <= MAX_PARENT_IMAGE_AREA_FRACTION:
        return None
    return x1, y1, x2, y2


def _nearest_pose_index(timestamps: np.ndarray, timestamp: float) -> int | None:
    insertion = int(np.searchsorted(timestamps, timestamp))
    choices = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    if not choices:
        return None
    selected = min(choices, key=lambda index: abs(float(timestamps[index]) - timestamp))
    if abs(float(timestamps[selected]) - timestamp) > POSE_TIME_TOLERANCE_S:
        return None
    return selected


def _prepare_frames(scene_dir: Path, video_id: str, parents: list[ParentBox]) -> list[FrameGeometry]:
    video_dir = scene_dir / video_id
    image_dir = video_dir / "lowres_wide"
    intrinsic_dir = video_dir / "lowres_wide_intrinsics"
    pose_timestamps, poses = _read_poses(video_dir / "lowres_poses.traj")
    frames: list[FrameGeometry] = []
    for image_path in sorted(image_dir.glob("*.png"))[::FRAME_STRIDE]:
        timestamp = float(image_path.stem.split("_")[-1])
        pose_index = _nearest_pose_index(pose_timestamps, timestamp)
        if pose_index is None:
            continue
        intrinsic_path = intrinsic_dir / f"{image_path.stem}.pincam"
        if not intrinsic_path.is_file():
            continue
        width, height, intrinsic = _read_intrinsic(intrinsic_path)
        camera_to_world = poses[pose_index]
        projections = {
            parent.binding_id: projection
            for parent in parents
            if (
                projection := _project_parent(
                    parent, camera_to_world, intrinsic, width, height
                )
            )
            is not None
        }
        if projections:
            frames.append(
                FrameGeometry(
                    timestamp,
                    image_path,
                    intrinsic_path,
                    camera_to_world,
                    intrinsic,
                    projections,
                )
            )
    return frames


def _ray_from_pixel(
    pixel_xy: tuple[float, float], camera_to_world: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u, v = pixel_xy
    camera_ray = np.asarray(
        [
            (u - intrinsic[0, 2]) / intrinsic[0, 0],
            (v - intrinsic[1, 2]) / intrinsic[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )
    camera_ray /= np.linalg.norm(camera_ray)
    origin = camera_to_world[:3, 3]
    direction = camera_to_world[:3, :3] @ camera_ray
    direction /= np.linalg.norm(direction)
    return origin, direction


def _ray_box_intersection(
    origin: np.ndarray, direction: np.ndarray, box: ParentBox
) -> tuple[float, np.ndarray] | None:
    local_origin = (origin - box.center) @ box.axes.T
    local_direction = direction @ box.axes.T
    half = box.lengths / 2.0
    lower = -np.inf
    upper = np.inf
    for axis in range(3):
        if abs(local_direction[axis]) < 1e-9:
            if not -half[axis] <= local_origin[axis] <= half[axis]:
                return None
            continue
        first = (-half[axis] - local_origin[axis]) / local_direction[axis]
        second = (half[axis] - local_origin[axis]) / local_direction[axis]
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if upper < lower:
            return None
    distance = lower if lower > 0.0 else upper
    if distance <= 0.0:
        return None
    return float(distance), origin + distance * direction


def _detection_observation(
    frame: FrameGeometry,
    box_xyxy: tuple[float, float, float, float],
    confidence: float,
    parents: dict[str, ParentBox],
) -> HandleObservation | None:
    x1, y1, x2, y2 = box_xyxy
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    detection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    origin, direction = _ray_from_pixel(center, frame.camera_to_world, frame.intrinsic)
    matches: list[tuple[float, ParentBox, np.ndarray]] = []
    for binding_id, projection in frame.parent_projections.items():
        px1, py1, px2, py2 = projection
        if not (px1 <= center[0] <= px2 and py1 <= center[1] <= py2):
            continue
        parent_area = (px2 - px1) * (py2 - py1)
        if parent_area <= 0.0 or detection_area / parent_area > MAX_HANDLE_TO_PARENT_AREA_RATIO:
            continue
        intersection = _ray_box_intersection(origin, direction, parents[binding_id])
        if intersection is not None:
            distance, point = intersection
            matches.append((distance, parents[binding_id], point))
    if not matches:
        return None
    _, parent, point = min(matches, key=lambda row: (row[0], row[1].binding_id))
    return HandleObservation(
        parent.binding_id,
        frame.timestamp,
        frame.image_path.name,
        confidence,
        box_xyxy,
        tuple(float(value) for value in point),
        tuple(float(value) for value in origin),
    )


def _camera_baseline(cameras: np.ndarray) -> float:
    if len(cameras) < 2:
        return 0.0
    maximum = 0.0
    for index in range(len(cameras) - 1):
        maximum = max(
            maximum,
            float(np.linalg.norm(cameras[index + 1 :] - cameras[index], axis=1).max()),
        )
    return maximum


def _triangulate_observation_rays(observations: list[HandleObservation]) -> np.ndarray:
    """Least-squares point closest to all current-camera detection rays."""

    normal = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for observation in observations:
        origin = np.asarray(observation.camera_xyz_m, dtype=np.float64)
        point = np.asarray(observation.point_xyz_m, dtype=np.float64)
        direction = point - origin
        direction /= np.linalg.norm(direction)
        projector = identity - np.outer(direction, direction)
        normal += projector
        rhs += projector @ origin
    if np.linalg.cond(normal) > 1e8:
        return np.median(
            np.asarray([observation.point_xyz_m for observation in observations]),
            axis=0,
        )
    return np.linalg.solve(normal, rhs)


def _candidate_row(
    candidate_id: str,
    parent_binding_id: str,
    point: np.ndarray,
    observations: list[HandleObservation],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "parent_binding_id": parent_binding_id,
        "center_xyz_m": [round(float(value), 6) for value in point],
        "support_observations": len(observations),
        "support_frames": len({observation.frame_name for observation in observations}),
        "median_confidence": round(float(np.median([o.confidence for o in observations])), 6),
        "time_span_s": round(
            max(o.timestamp for o in observations) - min(o.timestamp for o in observations), 6
        ),
        "camera_baseline_m": round(
            _camera_baseline(np.asarray([o.camera_xyz_m for o in observations])), 6
        ),
    }


def _single_view_candidates(
    parent_binding_id: str, observations: list[HandleObservation]
) -> list[dict[str, Any]]:
    by_frame: dict[str, list[HandleObservation]] = {}
    for observation in observations:
        by_frame.setdefault(observation.frame_name, []).append(observation)
    if not by_frame:
        return []
    selected = max(
        by_frame.values(),
        key=lambda rows: (
            sum(row.confidence for row in rows),
            len(rows),
            rows[0].frame_name,
        ),
    )
    ordered = sorted(selected, key=lambda row: (row.point_xyz_m, row.box_xyxy))
    return [
        _candidate_row(
            f"sv-{parent_binding_id[:6]}-{index:02d}",
            parent_binding_id,
            np.asarray(observation.point_xyz_m),
            [observation],
        )
        for index, observation in enumerate(ordered)
    ]


def _multiview_candidates(
    parent_binding_id: str, observations: list[HandleObservation]
) -> list[dict[str, Any]]:
    if len(observations) < MULTIVIEW_MIN_OBSERVATIONS:
        return []
    points = np.asarray([observation.point_xyz_m for observation in observations])
    labels = DBSCAN(
        eps=MULTIVIEW_CLUSTER_RADIUS_M,
        min_samples=MULTIVIEW_MIN_OBSERVATIONS,
    ).fit_predict(points)
    clusters: list[tuple[float, dict[str, Any]]] = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        selected = [
            observation
            for observation, assigned in zip(observations, labels)
            if int(assigned) == label
        ]
        time_span = max(o.timestamp for o in selected) - min(o.timestamp for o in selected)
        baseline = _camera_baseline(np.asarray([o.camera_xyz_m for o in selected]))
        if time_span < MULTIVIEW_MIN_TIME_SPAN_S or baseline < MULTIVIEW_MIN_CAMERA_BASELINE_M:
            continue
        center = _triangulate_observation_rays(selected)
        row = _candidate_row(
            f"mv-{parent_binding_id[:6]}-{label:02d}",
            parent_binding_id,
            center,
            selected,
        )
        rank = len(selected) * float(row["median_confidence"])
        clusters.append((rank, row))
    return [row for _, row in sorted(clusters, key=lambda value: value[0], reverse=True)[:MAX_PROPOSALS_PER_PARENT]]


def build_provider(
    scene_dir: Path, video_id: str, model_path: Path
) -> dict[str, Any]:
    object_boxes_path = scene_dir / video_id / f"{video_id}_3dod_annotation.json"
    parents = _load_parent_boxes(object_boxes_path)
    parent_by_id = {parent.binding_id: parent for parent in parents}
    frames = _prepare_frames(scene_dir, video_id, parents)
    if not frames:
        raise RuntimeError("No RGB frames have both a usable pose and projected parent")

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(model_path))
    started = time.perf_counter()
    results = []
    for start in range(0, len(frames), MODEL_BATCH):
        chunk = frames[start : start + MODEL_BATCH]
        results.extend(
            model.predict(
                [str(frame.image_path) for frame in chunk],
                device=device,
                imgsz=MODEL_IMAGE_SIZE,
                conf=MODEL_CONFIDENCE,
                classes=[1],
                batch=MODEL_BATCH,
                verbose=False,
            )
        )
    runtime_seconds = time.perf_counter() - started

    observations: list[HandleObservation] = []
    raw_handle_detections = 0
    for frame, result in zip(frames, results):
        for detected in result.boxes:
            raw_handle_detections += 1
            xyxy = tuple(float(value) for value in detected.xyxy[0].detach().cpu().tolist())
            observation = _detection_observation(
                frame,
                xyxy,
                float(detected.conf.item()),
                parent_by_id,
            )
            if observation is not None:
                observations.append(observation)

    observations_by_parent = {
        parent.binding_id: [
            observation
            for observation in observations
            if observation.parent_binding_id == parent.binding_id
        ]
        for parent in parents
    }
    single_view = {
        parent.binding_id: _single_view_candidates(
            parent.binding_id, observations_by_parent[parent.binding_id]
        )
        for parent in parents
    }
    multiview = {
        parent.binding_id: _multiview_candidates(
            parent.binding_id, observations_by_parent[parent.binding_id]
        )
        for parent in parents
    }
    return {
        "schema_version": 1,
        "provider": "L10-SC9T-PARENT-BOUND-MULTIVIEW-RAY-TRIANGULATED-HANDLE-PROPOSER",
        "truth_isolation": (
            "Functional annotations, description target IDs, affordance labels, and evaluator "
            "part points are not loaded until after this provider payload is written."
        ),
        "inputs": {
            "scene_dir": str(scene_dir),
            "visit_id": scene_dir.name,
            "video_id": video_id,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "object_boxes_sha256": _sha256(object_boxes_path),
        },
        "frozen_parameters": {
            "frame_stride": FRAME_STRIDE,
            "model_image_size": MODEL_IMAGE_SIZE,
            "model_batch": MODEL_BATCH,
            "model_confidence": MODEL_CONFIDENCE,
            "max_handle_to_parent_area_ratio": MAX_HANDLE_TO_PARENT_AREA_RATIO,
            "min_parent_image_area_fraction": MIN_PARENT_IMAGE_AREA_FRACTION,
            "max_parent_image_area_fraction": MAX_PARENT_IMAGE_AREA_FRACTION,
            "pose_time_tolerance_s": POSE_TIME_TOLERANCE_S,
            "multiview_cluster_radius_m": MULTIVIEW_CLUSTER_RADIUS_M,
            "multiview_min_observations": MULTIVIEW_MIN_OBSERVATIONS,
            "multiview_min_time_span_s": MULTIVIEW_MIN_TIME_SPAN_S,
            "multiview_min_camera_baseline_m": MULTIVIEW_MIN_CAMERA_BASELINE_M,
            "multiview_lift": "least_squares_detection_ray_triangulation",
        },
        "runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "runtime_seconds": round(runtime_seconds, 3),
            "frames_inferred": len(frames),
            "raw_handle_detections": raw_handle_detections,
            "parent_bound_observations": len(observations),
        },
        "parent_observation_counts": {
            parent.binding_id: len(observations_by_parent[parent.binding_id])
            for parent in parents
        },
        "arms": {
            "single_view": single_view,
            "multiview": multiview,
        },
    }


def _provider_candidates(
    provider: dict[str, Any], arm: str, parent_binding_id: str
) -> list[FunctionalPartCandidate]:
    return [
        FunctionalPartCandidate(
            row["candidate_id"],
            row["parent_binding_id"],
            tuple(float(value) for value in row["center_xyz_m"]),
        )
        for row in provider["arms"][arm].get(parent_binding_id, [])
    ]


def _candidate_distance(candidate: FunctionalPartCandidate, targets: Iterable[FunctionalProposal]) -> float:
    return _nearest_distance(np.asarray(candidate.center_xyz_m), list(targets))


def _evaluate_arm(
    provider: dict[str, Any],
    arm: str,
    descriptions: list[dict[str, Any]],
    truth: dict[str, FunctionalProposal],
) -> dict[str, Any]:
    selector = TaskRelationalFunctionalSelector()
    task_rows: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    for description in descriptions:
        target_ids = tuple(sorted(description["annot_id"]))
        targets = [truth[target_id] for target_id in target_ids if target_id in truth]
        if len(targets) != len(target_ids):
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "reason": "NOT_EVALUABLE_PARENT_BINDING",
                }
            )
            continue
        parent_ids = {target.parent.binding_id for target in targets}
        if len(parent_ids) != 1:
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "reason": "NOT_EVALUABLE_MULTI_PARENT_TASK",
                }
            )
            continue
        parent_binding_id = next(iter(parent_ids))
        candidates = _provider_candidates(provider, arm, parent_binding_id)
        decision = selector.select(
            description["description"], parent_binding_id, candidates
        )
        selected = [
            candidate
            for candidate in candidates
            if candidate.candidate_id in decision.selected_candidate_ids
        ]
        distances = [
            _candidate_distance(candidate, targets) for candidate in selected
        ]
        wrong_count = sum(
            distance > FUNCTIONAL_PART_MATCH_TOLERANCE_M for distance in distances
        )
        covered_targets = sum(
            bool(selected)
            and min(
                _candidate_distance(candidate, [target]) for candidate in selected
            )
            <= FUNCTIONAL_PART_MATCH_TOLERANCE_M
            for target in targets
        )
        task_rows.append(
            {
                "desc_id": description["desc_id"],
                "description": description["description"],
                "parent_binding_id": parent_binding_id,
                "evaluator_target_ids": list(target_ids),
                "candidate_count": len(candidates),
                "state": decision.state.value,
                "relation": decision.relation,
                "selected_candidate_ids": list(decision.selected_candidate_ids),
                "selected_nearest_target_m": [round(value, 6) for value in distances],
                "legal_commit": bool(selected) and wrong_count == 0,
                "target_set_recall": covered_targets / len(targets),
                "wrong_part_count": wrong_count,
                "action": decision.action,
            }
        )
    denominator = len(task_rows)
    return {
        "tasks_evaluable": denominator,
        "tasks_not_evaluable": len(not_evaluable),
        "legal_commit_count": sum(row["legal_commit"] for row in task_rows),
        "legal_commit_rate": (
            sum(row["legal_commit"] for row in task_rows) / denominator if denominator else 0.0
        ),
        "mean_target_set_recall": (
            float(np.mean([row["target_set_recall"] for row in task_rows]))
            if denominator
            else 0.0
        ),
        "wrong_part_count": sum(row["wrong_part_count"] for row in task_rows),
        "tasks": task_rows,
        "not_evaluable": not_evaluable,
    }


def _proposal_metrics(
    provider: dict[str, Any], arm: str, truth: dict[str, FunctionalProposal]
) -> dict[str, Any]:
    candidates = [
        candidate
        for parent_binding_id in provider["arms"][arm]
        for candidate in _provider_candidates(provider, arm, parent_binding_id)
    ]
    truth_rows = list(truth.values())
    recalled = sum(
        any(
            candidate.parent_binding_id == target.parent.binding_id
            and _candidate_distance(candidate, [target]) <= FUNCTIONAL_PART_MATCH_TOLERANCE_M
            for candidate in candidates
        )
        for target in truth_rows
    )
    correct_candidates = sum(
        any(
            candidate.parent_binding_id == target.parent.binding_id
            and _candidate_distance(candidate, [target]) <= FUNCTIONAL_PART_MATCH_TOLERANCE_M
            for target in truth_rows
        )
        for candidate in candidates
    )
    return {
        "candidate_count": len(candidates),
        "parent_bound_truth_count": len(truth_rows),
        "recalled_truth_count": recalled,
        "proposal_recall": recalled / len(truth_rows) if truth_rows else 0.0,
        "correct_candidate_count": correct_candidates,
        "candidate_precision": correct_candidates / len(candidates) if candidates else 0.0,
    }


def evaluate_provider(
    scene_dir: Path, video_id: str, provider: dict[str, Any], provider_sha256: str
) -> dict[str, Any]:
    visit_id = scene_dir.name
    annotation_path = scene_dir / f"{visit_id}_annotations.json"
    description_path = scene_dir / f"{visit_id}_descriptions.json"
    laser_path = scene_dir / f"{visit_id}_laser_scan.ply"
    transform_path = scene_dir / video_id / f"{video_id}_transform.npy"
    object_boxes_path = scene_dir / video_id / f"{video_id}_3dod_annotation.json"

    annotations = _load_json(annotation_path)["annotations"]
    descriptions = _load_json(description_path)["descriptions"]
    xyz = _load_ply_xyz(laser_path)
    transform = np.load(transform_path)
    parents = _load_parent_boxes(object_boxes_path)
    truth: dict[str, FunctionalProposal] = {}
    unmatched: list[str] = []
    for annotation in annotations:
        if annotation["label"] == "exclude":
            continue
        points = _transform_points(
            xyz[np.asarray(annotation["indices"], dtype=np.int64)], transform
        )
        matched = _match_parent(points, parents)
        if matched is None:
            unmatched.append(annotation["annot_id"])
            continue
        parent, coverage = matched
        truth[annotation["annot_id"]] = FunctionalProposal(
            annotation["annot_id"], points, points.mean(axis=0), parent, coverage
        )

    arms = {
        arm: {
            "proposal": _proposal_metrics(provider, arm, truth),
            "task": _evaluate_arm(provider, arm, descriptions, truth),
        }
        for arm in ("single_view", "multiview")
    }
    single = arms["single_view"]
    multi = arms["multiview"]
    if len(truth) < 8:
        decision = "SC9T_NOT_EVALUABLE_INSUFFICIENT_PARENT_BOUND_FUNCTIONAL_TRUTH"
    elif (
        multi["proposal"]["proposal_recall"] >= 0.50
        and multi["proposal"]["candidate_precision"] >= 0.70
        and multi["task"]["legal_commit_rate"] >= single["task"]["legal_commit_rate"]
        and (
            multi["task"]["legal_commit_rate"]
            > single["task"]["legal_commit_rate"]
            or multi["task"]["mean_target_set_recall"]
            >= single["task"]["mean_target_set_recall"] + 0.20
        )
        and multi["task"]["wrong_part_count"] <= single["task"]["wrong_part_count"]
    ):
        decision = "SC9T_RAY_TRIANGULATED_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL"
    else:
        decision = "SC9T_RAY_TRIANGULATED_FUNCTIONAL_PROPOSAL_GATE_NOT_MET"

    return {
        "schema_version": 1,
        "experiment": "L10-SC9T-SCENEFUN3D-PARENT-BOUND-MULTIVIEW-RAY-TRIANGULATED-FUNCTIONAL-PROPOSER",
        "decision": decision,
        "claim_layer": "PRIVILEGED_PARENT_BOUND_RGB_FUNCTIONAL_PROPOSAL_DEVELOPMENT",
        "provider_sha256": provider_sha256,
        "truth_loaded_after_provider_seal": True,
        "source": {
            "visit_id": visit_id,
            "video_id": video_id,
            "annotations_sha256": _sha256(annotation_path),
            "descriptions_sha256": _sha256(description_path),
            "laser_scan_sha256": _sha256(laser_path),
            "transform_sha256": _sha256(transform_path),
        },
        "frozen_evaluator": {
            "functional_part_match_tolerance_m": FUNCTIONAL_PART_MATCH_TOLERANCE_M,
            "parent_bound_truth_count": len(truth),
            "unmatched_truth_count": len(unmatched),
        },
        "provider_runtime": provider["runtime"],
        "arms": arms,
        "claim_boundary": (
            "This is one already-opened SceneFun3D Development scene. Exact 3D parent boxes "
            "are privileged inputs, and the handle detector covers only its trained door/handle "
            "domain. A positive result would establish a parent-bound multi-view RGB proposal "
            "signal, not open-vocabulary functional grounding, phone-camera transfer, metric "
            "localization, reachability, orientation, arrival, completion, user benefit, or safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    provider = build_provider(args.scene_dir.resolve(), args.video_id, args.model.resolve())
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_sha256 = _sha256(args.provider_output)

    result = evaluate_provider(
        args.scene_dir.resolve(), args.video_id, provider, provider_sha256
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "runtime": result["provider_runtime"],
                "single_view_proposal": result["arms"]["single_view"]["proposal"],
                "multiview_proposal": result["arms"]["multiview"]["proposal"],
                "single_view_task": {
                    key: value
                    for key, value in result["arms"]["single_view"]["task"].items()
                    if key not in {"tasks", "not_evaluable"}
                },
                "multiview_task": {
                    key: value
                    for key, value in result["arms"]["multiview"]["task"].items()
                    if key not in {"tasks", "not_evaluable"}
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
