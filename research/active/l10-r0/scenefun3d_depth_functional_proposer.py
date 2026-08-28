from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from scenefun3d_functional_handoff_ceiling import _load_parent_boxes
from scenefun3d_multiview_functional_proposer import (
    FUNCTIONAL_PART_MATCH_TOLERANCE_M,
    MAX_PROPOSALS_PER_PARENT,
    MODEL_BATCH,
    MODEL_CONFIDENCE,
    MODEL_IMAGE_SIZE,
    MULTIVIEW_CLUSTER_RADIUS_M,
    MULTIVIEW_MIN_CAMERA_BASELINE_M,
    MULTIVIEW_MIN_OBSERVATIONS,
    MULTIVIEW_MIN_TIME_SPAN_S,
    FrameGeometry,
    HandleObservation,
    ParentBox,
    _candidate_row,
    _camera_baseline,
    _detection_observation,
    _prepare_frames,
    _sha256,
    _single_view_candidates,
    evaluate_provider,
)


DEPTH_PATCH_RADIUS_PX = 2
DEPTH_SCALE = 1000.0
DEPTH_PARENT_MARGIN_M = 0.15


def _distance_outside_parent(point: np.ndarray, parent: ParentBox) -> float:
    local = (point - parent.center) @ parent.axes.T
    outside = np.maximum(np.abs(local) - parent.lengths / 2.0, 0.0)
    return float(np.linalg.norm(outside))


def _depth_point(
    depth: np.ndarray,
    center_xy: tuple[float, float],
    frame: FrameGeometry,
) -> np.ndarray | None:
    u = int(round(center_xy[0]))
    v = int(round(center_xy[1]))
    x1 = max(0, u - DEPTH_PATCH_RADIUS_PX)
    x2 = min(depth.shape[1], u + DEPTH_PATCH_RADIUS_PX + 1)
    y1 = max(0, v - DEPTH_PATCH_RADIUS_PX)
    y2 = min(depth.shape[0], v + DEPTH_PATCH_RADIUS_PX + 1)
    values = depth[y1:y2, x1:x2]
    values = values[values > 0]
    if not len(values):
        return None
    z = float(np.median(values)) / DEPTH_SCALE
    camera_point = np.asarray(
        [
            (center_xy[0] - frame.intrinsic[0, 2]) / frame.intrinsic[0, 0] * z,
            (center_xy[1] - frame.intrinsic[1, 2]) / frame.intrinsic[1, 1] * z,
            z,
            1.0,
        ]
    )
    return (frame.camera_to_world @ camera_point)[:3]


def _depth_observation(
    frame: FrameGeometry,
    box_xyxy: tuple[float, float, float, float],
    confidence: float,
    parents: dict[str, ParentBox],
    depth: np.ndarray,
) -> HandleObservation | None:
    ray_bound = _detection_observation(frame, box_xyxy, confidence, parents)
    if ray_bound is None:
        return None
    x1, y1, x2, y2 = box_xyxy
    point = _depth_point(depth, ((x1 + x2) / 2.0, (y1 + y2) / 2.0), frame)
    if point is None:
        return None
    parent = parents[ray_bound.parent_binding_id]
    if _distance_outside_parent(point, parent) > DEPTH_PARENT_MARGIN_M:
        return None
    return HandleObservation(
        ray_bound.parent_binding_id,
        ray_bound.timestamp,
        ray_bound.frame_name,
        ray_bound.confidence,
        ray_bound.box_xyxy,
        tuple(float(value) for value in point),
        ray_bound.camera_xyz_m,
    )


def _depth_multiview_candidates(
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
        center = np.median(np.asarray([o.point_xyz_m for o in selected]), axis=0)
        row = _candidate_row(
            f"dmv-{parent_binding_id[:6]}-{label:02d}",
            parent_binding_id,
            center,
            selected,
        )
        clusters.append((len(selected) * float(row["median_confidence"]), row))
    return [
        row
        for _, row in sorted(clusters, key=lambda value: value[0], reverse=True)[
            :MAX_PROPOSALS_PER_PARENT
        ]
    ]


def build_depth_provider(
    scene_dir: Path, video_id: str, model_path: Path
) -> dict[str, Any]:
    video_dir = scene_dir / video_id
    object_boxes_path = video_dir / f"{video_id}_3dod_annotation.json"
    depth_dir = video_dir / "lowres_depth"
    parents = _load_parent_boxes(object_boxes_path)
    parent_by_id = {parent.binding_id: parent for parent in parents}
    frames = _prepare_frames(scene_dir, video_id, parents)
    frames = [
        frame for frame in frames if (depth_dir / frame.image_path.name).is_file()
    ]
    if not frames:
        raise RuntimeError("No RGB frame has matching depth, pose, and parent geometry")

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
    depth_valid_detections = 0
    for frame, result in zip(frames, results):
        depth = cv2.imread(
            str(depth_dir / frame.image_path.name), cv2.IMREAD_UNCHANGED
        )
        if depth is None:
            continue
        for detected in result.boxes:
            raw_handle_detections += 1
            xyxy = tuple(float(value) for value in detected.xyxy[0].detach().cpu().tolist())
            observation = _depth_observation(
                frame,
                xyxy,
                float(detected.conf.item()),
                parent_by_id,
                depth,
            )
            if observation is not None:
                depth_valid_detections += 1
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
        parent.binding_id: _depth_multiview_candidates(
            parent.binding_id, observations_by_parent[parent.binding_id]
        )
        for parent in parents
    }
    return {
        "schema_version": 1,
        "provider": "L10-SC10-PARENT-BOUND-RGBD-MULTIVIEW-HANDLE-PROPOSER",
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
            "model_image_size": MODEL_IMAGE_SIZE,
            "model_batch": MODEL_BATCH,
            "model_confidence": MODEL_CONFIDENCE,
            "depth_patch_radius_px": DEPTH_PATCH_RADIUS_PX,
            "depth_scale": DEPTH_SCALE,
            "depth_parent_margin_m": DEPTH_PARENT_MARGIN_M,
            "multiview_cluster_radius_m": MULTIVIEW_CLUSTER_RADIUS_M,
            "multiview_min_observations": MULTIVIEW_MIN_OBSERVATIONS,
            "multiview_min_time_span_s": MULTIVIEW_MIN_TIME_SPAN_S,
            "multiview_min_camera_baseline_m": MULTIVIEW_MIN_CAMERA_BASELINE_M,
            "multiview_lift": "median_native_depth_backprojection",
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
            "depth_valid_parent_bound_observations": depth_valid_detections,
        },
        "parent_observation_counts": {
            parent.binding_id: len(observations_by_parent[parent.binding_id])
            for parent in parents
        },
        "arms": {"single_view": single_view, "multiview": multiview},
    }


def _depth_decision(result: dict[str, Any]) -> str:
    single = result["arms"]["single_view"]
    multi = result["arms"]["multiview"]
    if result["frozen_evaluator"]["parent_bound_truth_count"] < 8:
        return "SC10_NOT_EVALUABLE_INSUFFICIENT_PARENT_BOUND_FUNCTIONAL_TRUTH"
    if (
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
        return "SC10_NATIVE_DEPTH_MULTIVIEW_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL"
    return "SC10_NATIVE_DEPTH_MULTIVIEW_FUNCTIONAL_PROPOSAL_GATE_NOT_MET"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    provider = build_depth_provider(scene_dir, args.video_id, args.model.resolve())
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_sha256 = _sha256(args.provider_output)

    result = evaluate_provider(scene_dir, args.video_id, provider, provider_sha256)
    result["experiment"] = (
        "L10-SC10-SCENEFUN3D-PARENT-BOUND-NATIVE-DEPTH-MULTIVIEW-FUNCTIONAL-PROPOSER"
    )
    result["decision"] = _depth_decision(result)
    result["claim_layer"] = "PRIVILEGED_PARENT_BOUND_RGBD_FUNCTIONAL_PROPOSAL_DEVELOPMENT"
    result["claim_boundary"] = (
        "This is one already-opened SceneFun3D Development scene with privileged exact "
        "3D parent boxes and native ARKit depth. The handle detector covers only its trained "
        "door/handle domain. A positive result would establish a parent-bound RGB-D proposal "
        "signal, not open-vocabulary functionality, phone-camera transfer, reachability, "
        "orientation, arrival, completion, user benefit, or safety."
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
