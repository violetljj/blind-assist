#!/usr/bin/env python3
"""Freeze and replay one posed-reference portal transfer on Hypersim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-hypersim-posed-portal-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-hypersim-posed-portal-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-hypersim-posed-portal-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write(path, payload)


def read_hdf5(path: Path) -> np.ndarray:
    require(path.is_file(), f"MISSING_HDF5:{path}")
    with h5py.File(path, "r") as handle:
        return handle["dataset"][:]


def mask_path(payload_root: Path, scene: str, camera: str, frame: int) -> Path:
    return (
        payload_root
        / scene
        / "images"
        / f"scene_{camera}_geometry_hdf5"
        / f"frame.{frame:04d}.semantic_instance.hdf5"
    )


def position_path(payload_root: Path, scene: str, camera: str, frame: int) -> Path:
    return (
        payload_root
        / scene
        / "images"
        / f"scene_{camera}_geometry_hdf5"
        / f"frame.{frame:04d}.position.hdf5"
    )


def rgb_path(payload_root: Path, scene: str, camera: str, frame: int) -> Path:
    return (
        payload_root
        / scene
        / "images"
        / f"scene_{camera}_final_preview"
        / f"frame.{frame:04d}.color.jpg"
    )


def camera_file(payload_root: Path, scene: str, camera: str, name: str) -> Path:
    return payload_root / scene / "_detail" / camera / name


def relative_to_source(path: Path, source_root: Path) -> str:
    return path.resolve().relative_to(source_root.resolve()).as_posix()


def door_instance_ids(source_root: Path, scene: str, semantic_id: int) -> list[int]:
    mesh = (
        source_root
        / "ml-hypersim"
        / "evermotion_dataset"
        / "scenes"
        / scene
        / "_detail"
        / "mesh"
    )
    semantic = read_hdf5(mesh / "mesh_objects_si.hdf5")[:, 0].astype(np.int32)
    instances = read_hdf5(mesh / "mesh_objects_sii.hdf5")[:, 0].astype(np.int32)
    return sorted(int(value) for value in set(instances[semantic == semantic_id]) if value >= 0)


def mask_stats(mask: np.ndarray, instance_id: int, door_ids: list[int]) -> dict[str, Any] | None:
    ys, xs = np.where(mask == instance_id)
    if xs.size == 0:
        return None
    height, width = mask.shape
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    present = set(int(value) for value in np.unique(mask))
    return {
        "pixels": int(xs.size),
        "image_fraction": float(xs.size / (height * width)),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_width": x1 - x0,
        "bbox_height": y1 - y0,
        "bbox_margin": min(x0, y0, width - x1, height - y1),
        "other_door_instance_ids": [
            value for value in door_ids if value != instance_id and value in present
        ],
    }


def eligible(stats: dict[str, Any], rules: dict[str, Any]) -> bool:
    return (
        stats["pixels"] >= int(rules["minimum_target_pixels"])
        and stats["bbox_width"] >= int(rules["minimum_bbox_width_pixels"])
        and stats["bbox_height"] >= int(rules["minimum_bbox_height_pixels"])
        and stats["bbox_margin"] >= int(rules["minimum_bbox_image_margin_pixels"])
        and stats["image_fraction"] <= float(rules["maximum_target_image_fraction"])
    )


def select_cohort(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    scene_info = protocol["frozen_scene"]
    scene = scene_info["scene_name"]
    payload_root = source_root / "payload"
    scene_payload = payload_root / scene
    forbidden_rgb = sorted(scene_payload.glob("**/frame.*.color.jpg")) + sorted(
        scene_payload.glob("**/frame.*.color.hdf5")
    )
    require(not forbidden_rgb, f"RGB_MATERIALIZED_BEFORE_FREEZE:{len(forbidden_rgb)}")

    door_ids = door_instance_ids(source_root, scene, int(scene_info["semantic_door_id_nyu40"]))
    rules = protocol["pre_rgb_selector"]["eligible_role_frame"]
    records: dict[int, list[dict[str, Any]]] = {value: [] for value in door_ids}
    mask_files = sorted(scene_payload.glob("images/scene_cam_*_geometry_hdf5/frame.*.semantic_instance.hdf5"))
    expected = len(scene_info["camera_trajectories"]) * int(scene_info["frames_per_trajectory"])
    require(len(mask_files) == expected, f"MASK_COUNT:{len(mask_files)}:{expected}")
    total_mask_bytes = 0
    for path in mask_files:
        directory = path.parent.name
        camera = directory[len("scene_") : -len("_geometry_hdf5")]
        frame = int(path.name.split(".")[1])
        mask = read_hdf5(path).astype(np.int32)
        require(mask.shape == (int(scene_info["image_height"]), int(scene_info["image_width"])), f"MASK_SHAPE:{path}:{mask.shape}")
        total_mask_bytes += path.stat().st_size
        for instance_id in door_ids:
            stats = mask_stats(mask, instance_id, door_ids)
            if stats is None:
                continue
            records[instance_id].append(
                {
                    "camera": camera,
                    "frame": frame,
                    "path": relative_to_source(path, source_root),
                    "eligible": eligible(stats, rules),
                    **stats,
                }
            )

    episodes: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    for instance_id in door_ids:
        rows = records[instance_id]
        cameras = sorted(
            camera
            for camera in scene_info["camera_trajectories"]
            if any(row["camera"] == camera and row["eligible"] for row in rows)
        )
        eligible_episode = False
        reason = "FEWER_THAN_TWO_ELIGIBLE_TRAJECTORIES"
        if len(cameras) >= 2:
            reference_camera = cameras[0]
            query_cameras = [
                camera
                for camera in cameras[1:]
                if any(
                    row["camera"] == camera
                    and row["eligible"]
                    and row["other_door_instance_ids"]
                    for row in rows
                )
            ]
            if query_cameras:
                query_camera = query_cameras[0]
                reference = max(
                    (
                        row
                        for row in rows
                        if row["camera"] == reference_camera and row["eligible"]
                    ),
                    key=lambda row: (row["pixels"], -row["frame"]),
                )
                query = max(
                    (
                        row
                        for row in rows
                        if row["camera"] == query_camera
                        and row["eligible"]
                        and row["other_door_instance_ids"]
                    ),
                    key=lambda row: (row["pixels"], -row["frame"]),
                )
                eligible_episode = True
                reason = "ELIGIBLE"
                if len(episodes) < int(protocol["pre_rgb_selector"]["cohort_size"]):
                    episode_id = f"HP{len(episodes) + 1:02d}"
                    for role in (reference, query):
                        role["sha256"] = sha256(source_root / role["path"])
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "scene_name": scene,
                            "target_door_instance_id": instance_id,
                            "reference": reference,
                            "query": query,
                        }
                    )
        selection_audit.append(
            {
                "door_instance_id": instance_id,
                "eligible_trajectories": cameras,
                "episode_eligible": eligible_episode,
                "reason": reason,
            }
        )

    cohort_size = int(protocol["pre_rgb_selector"]["cohort_size"])
    require(len(episodes) == cohort_size, f"COHORT_COVERAGE:{len(episodes)}:{cohort_size}")
    mesh_root = (
        source_root
        / "ml-hypersim"
        / "evermotion_dataset"
        / "scenes"
        / scene
        / "_detail"
        / "mesh"
    )
    parameters = source_root / "ml-hypersim" / "contrib" / "mikeroberts3000" / "metadata_camera_parameters.csv"
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_DEVELOPMENT_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "provider": protocol["provider"],
            "scene_name": scene,
            "semantic_instance_mask_count": len(mask_files),
            "semantic_instance_mask_bytes": total_mask_bytes,
            "door_instance_ids": door_ids,
            "mesh_objects_si_sha256": sha256(mesh_root / "mesh_objects_si.hdf5"),
            "mesh_objects_sii_sha256": sha256(mesh_root / "mesh_objects_sii.hdf5"),
            "camera_parameters_sha256": sha256(parameters),
            "rgb_files_at_freeze": 0,
        },
        "selection": protocol["pre_rgb_selector"],
        "selection_audit": selection_audit,
        "episodes": episodes,
        "materialize_after_freeze": {
            "camera_metadata": sorted(
                {
                    f"payload/{scene}/_detail/{role['camera']}/{name}"
                    for episode in episodes
                    for role in (episode["reference"], episode["query"])
                    for name in (
                        "camera_keyframe_frame_indices.hdf5",
                        "camera_keyframe_orientations.hdf5",
                        "camera_keyframe_positions.hdf5",
                    )
                }
            ),
            "reference_positions": [
                relative_to_source(
                    position_path(payload_root, scene, episode["reference"]["camera"], episode["reference"]["frame"]),
                    source_root,
                )
                for episode in episodes
            ],
            "rgb_previews": [
                relative_to_source(rgb_path(payload_root, scene, role["camera"], role["frame"]), source_root)
                for episode in episodes
                for role in (episode["reference"], episode["query"])
            ],
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output_path, cohort)
    print(json.dumps({"output": str(output_path), "episodes": episodes}, indent=2))


def camera_parameters(source_root: Path, scene: str) -> tuple[int, int, float, np.ndarray]:
    path = source_root / "ml-hypersim" / "contrib" / "mikeroberts3000" / "metadata_camera_parameters.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next((item for item in csv.DictReader(handle) if item["scene_name"] == scene), None)
    require(row is not None, f"CAMERA_PARAMETERS_MISSING:{scene}")
    width = int(float(row["settings_output_img_width"]))
    height = int(float(row["settings_output_img_height"]))
    meters_per_asset_unit = float(row["settings_units_info_meters_scale"])
    projection = np.array(
        [[float(row[f"M_proj_{i}{j}"]) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )
    return width, height, meters_per_asset_unit, projection


def pose(payload_root: Path, scene: str, camera: str, frame: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    index_path = camera_file(payload_root, scene, camera, "camera_keyframe_frame_indices.hdf5")
    orientation_path = camera_file(payload_root, scene, camera, "camera_keyframe_orientations.hdf5")
    position_file = camera_file(payload_root, scene, camera, "camera_keyframe_positions.hdf5")
    indices = read_hdf5(index_path).astype(np.int64).reshape(-1)
    matches = np.flatnonzero(indices == frame)
    require(matches.size == 1, f"FRAME_POSE_MATCH:{scene}:{camera}:{frame}:{matches.size}")
    offset = int(matches[0])
    orientations = read_hdf5(orientation_path).astype(np.float64)
    positions = read_hdf5(position_file).astype(np.float64)
    return positions[offset], orientations[offset], [index_path, orientation_path, position_file]


def project_world(
    points_world: np.ndarray,
    camera_position: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    projection: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    rotation_camera_from_world = rotation_world_from_camera.T
    translation_camera_from_world = -rotation_camera_from_world @ camera_position.reshape(3, 1)
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :3] = rotation_camera_from_world
    camera_from_world[:3, 3:] = translation_camera_from_world
    homogeneous = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1
    ).T
    clip = projection @ (camera_from_world @ homogeneous)
    valid = np.isfinite(clip).all(axis=0) & (clip[3] > 1e-8)
    require(int(valid.sum()) >= 3, f"TOO_FEW_PROJECTED_POINTS:{int(valid.sum())}")
    normalized = clip[:, valid] / clip[3:4, valid]
    x = 0.5 * (width - 1) * normalized[0] + 0.5 * (width - 1)
    y = -0.5 * (height - 1) * normalized[1] + 0.5 * (height - 1)
    screen = np.column_stack([x, y])
    finite = np.isfinite(screen).all(axis=1)
    return screen[finite]


def envelope_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    require(points.shape[0] >= 3, f"ENVELOPE_POINT_COUNT:{points.shape[0]}")
    bounded = points.copy()
    bounded[:, 0] = np.clip(bounded[:, 0], -4 * width, 5 * width)
    bounded[:, 1] = np.clip(bounded[:, 1], -4 * height, 5 * height)
    hull = cv2.convexHull(np.rint(bounded).astype(np.int32).reshape(-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def truth_envelope(mask: np.ndarray, instance_id: int) -> np.ndarray:
    ys, xs = np.where(mask == instance_id)
    require(xs.size >= 3, f"TRUTH_INSTANCE_NOT_VISIBLE:{instance_id}")
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2))
    output = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(output, hull, 1)
    return output.astype(bool)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8))
    require(moments["m00"] > 0.0, "EMPTY_PREDICTED_ENVELOPE")
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.copy()
    color_image = np.zeros_like(output)
    color_image[:] = color
    output[mask] = cv2.addWeighted(output, 1.0 - alpha, color_image, alpha, 0.0)[mask]
    return output


def replay(
    protocol_path: Path,
    cohort_path: Path,
    source_root: Path,
    output_path: Path,
    preview_dir: Path,
) -> None:
    protocol = load_json(protocol_path)
    cohort = load_json(cohort_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort.get("protocol_sha256") == sha256(protocol_path), "PROTOCOL_HASH_MISMATCH")
    scene = protocol["frozen_scene"]["scene_name"]
    payload_root = source_root / "payload"
    door_ids = cohort["source"]["door_instance_ids"]
    width, height, meters_per_asset_unit, projection = camera_parameters(source_root, scene)
    require((width, height) == (1024, 768), f"IMAGE_SIZE:{width}:{height}")
    preview_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    all_input_files: set[Path] = {protocol_path, cohort_path}
    for episode in cohort["episodes"]:
        instance_id = int(episode["target_door_instance_id"])
        reference = episode["reference"]
        query = episode["query"]
        reference_mask_file = source_root / reference["path"]
        query_mask_file = source_root / query["path"]
        require(sha256(reference_mask_file) == reference["sha256"], f"REFERENCE_MASK_HASH:{episode['episode_id']}")
        require(sha256(query_mask_file) == query["sha256"], f"QUERY_MASK_HASH:{episode['episode_id']}")
        reference_mask = read_hdf5(reference_mask_file).astype(np.int32)
        query_mask = read_hdf5(query_mask_file).astype(np.int32)
        reference_position_file = position_path(
            payload_root, scene, reference["camera"], int(reference["frame"])
        )
        reference_positions = read_hdf5(reference_position_file).astype(np.float64)
        require(reference_positions.shape == (height, width, 3), f"POSITION_SHAPE:{reference_positions.shape}")
        binary_reference = (reference_mask == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        require(contours, f"REFERENCE_CONTOUR_MISSING:{episode['episode_id']}")
        contour_pixels = np.concatenate([item.reshape(-1, 2) for item in contours], axis=0)
        points_world = reference_positions[contour_pixels[:, 1], contour_pixels[:, 0]]
        finite_points = np.isfinite(points_world).all(axis=1)
        points_world = points_world[finite_points]

        reference_camera_position, _, reference_pose_files = pose(
            payload_root, scene, reference["camera"], int(reference["frame"])
        )
        query_camera_position, query_orientation, query_pose_files = pose(
            payload_root, scene, query["camera"], int(query["frame"])
        )
        projected = project_world(
            points_world,
            query_camera_position,
            query_orientation,
            projection,
            width,
            height,
        )
        predicted = envelope_from_points(projected, width, height)
        target_truth = truth_envelope(query_mask, instance_id)
        visible_door_ids = [value for value in door_ids if np.any(query_mask == value)]
        instance_ious = {
            str(value): iou(predicted, truth_envelope(query_mask, value)) for value in visible_door_ids
        }
        ranked = sorted(visible_door_ids, key=lambda value: (-instance_ious[str(value)], value))
        selected_instance_id = int(ranked[0])
        target_iou = instance_ious[str(instance_id)]
        intersection = int(np.logical_and(predicted, target_truth).sum())
        prediction_pixels = int(predicted.sum())
        truth_pixels = int(target_truth.sum())
        precision = float(intersection / prediction_pixels) if prediction_pixels else 0.0
        recall = float(intersection / truth_pixels) if truth_pixels else 0.0
        predicted_centroid = centroid(predicted)
        truth_centroid = centroid(target_truth)
        centroid_error = float(np.linalg.norm(np.array(predicted_centroid) - np.array(truth_centroid)))
        cx = int(np.clip(round(predicted_centroid[0]), 0, width - 1))
        cy = int(np.clip(round(predicted_centroid[1]), 0, height - 1))
        centroid_inside = bool(target_truth[cy, cx])
        baseline_m = float(
            np.linalg.norm(reference_camera_position - query_camera_position) * meters_per_asset_unit
        )

        reference_rgb_file = rgb_path(
            payload_root, scene, reference["camera"], int(reference["frame"])
        )
        query_rgb_file = rgb_path(payload_root, scene, query["camera"], int(query["frame"]))
        reference_rgb = cv2.imread(str(reference_rgb_file), cv2.IMREAD_COLOR)
        query_rgb = cv2.imread(str(query_rgb_file), cv2.IMREAD_COLOR)
        require(reference_rgb is not None and query_rgb is not None, f"RGB_MISSING:{episode['episode_id']}")
        reference_view = overlay(reference_rgb, binary_reference.astype(bool), (0, 255, 0), 0.45)
        other_mask = np.isin(query_mask, [value for value in visible_door_ids if value != instance_id])
        query_view = overlay(query_rgb, other_mask, (0, 0, 255), 0.35)
        query_view = overlay(query_view, target_truth, (0, 255, 0), 0.35)
        query_view = overlay(query_view, predicted, (255, 255, 0), 0.35)
        cv2.circle(query_view, (cx, cy), 7, (255, 0, 255), 2)
        label = f"{episode['episode_id']} iid={instance_id} top1={selected_instance_id} IoU={target_iou:.3f}"
        cv2.putText(query_view, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        preview = np.concatenate([reference_view, query_view], axis=1)
        preview_file = preview_dir / f"{episode['episode_id'].lower()}-posed-portal-transfer.jpg"
        require(cv2.imwrite(str(preview_file), preview), f"PREVIEW_WRITE_FAILED:{preview_file}")

        input_files = {
            reference_mask_file,
            query_mask_file,
            reference_position_file,
            reference_rgb_file,
            query_rgb_file,
            *reference_pose_files,
            *query_pose_files,
        }
        all_input_files.update(input_files)
        episode_results.append(
            {
                "episode_id": episode["episode_id"],
                "target_door_instance_id": instance_id,
                "selected_door_instance_id": selected_instance_id,
                "correct_door_instance": selected_instance_id == instance_id,
                "wrong_door_commit": selected_instance_id != instance_id,
                "visible_query_door_instance_ids": visible_door_ids,
                "instance_envelope_iou": {key: round(value, 6) for key, value in instance_ious.items()},
                "target_envelope_iou": round(target_iou, 6),
                "target_precision": round(precision, 6),
                "target_recall": round(recall, 6),
                "centroid_error_pixels": round(centroid_error, 6),
                "centroid_error_image_diagonal_fraction": round(
                    centroid_error / float(np.hypot(width, height)), 6
                ),
                "centroid_inside_target_envelope": centroid_inside,
                "camera_baseline_m": round(baseline_m, 6),
                "reference_contour_points": int(points_world.shape[0]),
                "projected_contour_points": int(projected.shape[0]),
                "predicted_envelope_pixels": prediction_pixels,
                "target_truth_envelope_pixels": truth_pixels,
                "preview_path": str(preview_file),
                "input_sha256": {
                    relative_to_source(path, source_root): sha256(path)
                    for path in sorted(input_files)
                },
            }
        )

    correct = sum(bool(item["correct_door_instance"]) for item in episode_results)
    wrong = sum(bool(item["wrong_door_commit"]) for item in episode_results)
    centroid_inside = sum(bool(item["centroid_inside_target_envelope"]) for item in episode_results)
    median_iou = float(np.median([item["target_envelope_iou"] for item in episode_results]))
    gate_met = (
        correct == 3
        and wrong == 0
        and centroid_inside == 3
        and median_iou >= float(protocol["decision_gate"]["median_envelope_iou_minimum"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "EXPLORE_DEVELOPMENT_SYNTHETIC_MECHANISM_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "conclusion": (
            "L10_HYPERSIM_POSED_PORTAL_TRANSFER_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_HYPERSIM_POSED_PORTAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "summary": {
            "episodes": len(episode_results),
            "correct_door_instance": correct,
            "wrong_door_commit": wrong,
            "centroid_inside_target_envelope": centroid_inside,
            "median_target_envelope_iou": round(median_iou, 6),
            "minimum_target_envelope_iou": round(
                min(item["target_envelope_iou"] for item in episode_results), 6
            ),
            "mean_centroid_error_pixels": round(
                float(np.mean([item["centroid_error_pixels"] for item in episode_results])), 6
            ),
        },
        "episodes": episode_results,
        "input_boundary": protocol["frozen_transfer"],
        "decision_gate": protocol["decision_gate"],
        "claim_boundary": protocol["claim_boundary"],
        "all_input_sha256": {
            str(path): sha256(path) for path in sorted(all_input_files) if path.is_file()
        },
    }
    write_json(output_path, result)
    print(json.dumps({"output": str(output_path), "conclusion": result["conclusion"], "summary": result["summary"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--protocol", type=Path, default=HERE / "l10_hypersim_posed_portal_protocol_v1.json")
    select_parser.add_argument("--source-root", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, default=HERE / "l10_hypersim_posed_portal_cohort_v1.json")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, default=HERE / "l10_hypersim_posed_portal_protocol_v1.json")
    replay_parser.add_argument("--cohort", type=Path, default=HERE / "l10_hypersim_posed_portal_cohort_v1.json")
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, default=HERE / "l10_hypersim_posed_portal_result_v1.json")
    replay_parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_cohort(args.protocol, args.source_root, args.output)
    else:
        replay(args.protocol, args.cohort, args.source_root, args.output, args.preview_dir)


if __name__ == "__main__":
    main()
