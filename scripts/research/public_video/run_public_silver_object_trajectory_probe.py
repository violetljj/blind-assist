#!/usr/bin/env python3
"""Probe provisional event labels with frozen object trajectories.

The detector is a fixed COCO proposal model.  It never supplies event truth.
Detections are converted into deterministic object tracks and a fixed episode
vector describing relative scale growth, lateral drift, lower-corridor overlap,
and persistence.  Evaluation is grouped by source_id and uses the same closed-
form class-balanced ridge as the frozen-backbone probes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_silver_object_trajectory_probe_v1"
GROUP_NAMES = ("person", "vehicle", "animal", "furniture", "other")
GROUP_CLASSES = {
    "person": {"person"},
    "vehicle": {"bicycle", "car", "motorcycle", "bus", "truck", "train"},
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
    "furniture": {"bench", "chair", "couch", "bed", "dining table", "toilet", "potted plant"},
}


def coarse_group(class_name: str) -> str:
    for group, names in GROUP_CLASSES.items():
        if class_name in names:
            return group
    return "other"


def corridor_mask(height: int = 96, width: int = 96) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    top = int(round(height * 0.38))
    for y in range(top, height):
        fraction = (y - top) / max(1, height - 1 - top)
        half_width = 0.18 + fraction * (0.42 - 0.18)
        center = 0.5
        x0 = max(0, int(math.floor((center - half_width) * width)))
        x1 = min(width, int(math.ceil((center + half_width) * width)))
        mask[y, x0:x1] = True
    return mask


CORRIDOR_MASK = corridor_mask()


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    union = max(0.0, (ax1 - ax0) * (ay1 - ay0)) + max(0.0, (bx1 - bx0) * (by1 - by0)) - intersection
    return intersection / union if union > 0 else 0.0


def corridor_overlap(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in box]
    height, width = CORRIDOR_MASK.shape
    left = max(0, min(width, int(math.floor(x0 * width))))
    right = max(0, min(width, int(math.ceil(x1 * width))))
    top = max(0, min(height, int(math.floor(y0 * height))))
    bottom = max(0, min(height, int(math.ceil(y1 * height))))
    if right <= left or bottom <= top:
        return 0.0
    return float(CORRIDOR_MASK[top:bottom, left:right].mean())


def normalize_detection(class_name: str, confidence: float, xyxy: Sequence[float], *, width: int, height: int) -> dict[str, Any]:
    x0, y0, x1, y1 = [float(value) for value in xyxy]
    box = [
        float(np.clip(x0 / width, 0.0, 1.0)),
        float(np.clip(y0 / height, 0.0, 1.0)),
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
    ]
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    overlap = corridor_overlap(box)
    bottom = box[3]
    threat = confidence * (0.20 + math.sqrt(area)) * (0.20 + overlap) * (0.20 + bottom)
    return {
        "class_name": class_name,
        "group": coarse_group(class_name),
        "confidence": float(confidence),
        "box": box,
        "area": area,
        "center_x": center_x,
        "center_y": center_y,
        "bottom": bottom,
        "corridor_overlap": overlap,
        "threat": float(threat),
    }


def track_detections(frames: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for frame_index, detections in enumerate(frames):
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            if frame_index - track["last_frame"] > 2:
                continue
            previous = track["detections"][-1]
            for detection_index, detection in enumerate(detections):
                if previous["group"] != detection["group"]:
                    continue
                distance = math.hypot(previous["center_x"] - detection["center_x"], previous["center_y"] - detection["center_y"])
                overlap = bbox_iou(previous["box"], detection["box"])
                area_penalty = abs(math.log((detection["area"] + 1e-6) / (previous["area"] + 1e-6)))
                if overlap < 0.02 and distance > 0.32:
                    continue
                cost = distance + 0.18 * area_penalty - 0.45 * overlap
                candidates.append((cost, track_index, detection_index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _cost, track_index, detection_index in sorted(candidates):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            detection = dict(detections[detection_index])
            detection["frame_index"] = frame_index
            tracks[track_index]["detections"].append(detection)
            tracks[track_index]["last_frame"] = frame_index
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            item = dict(detection)
            item["frame_index"] = frame_index
            tracks.append({"last_frame": frame_index, "detections": [item]})
    return tracks


def slope(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 2 or float(times.max() - times.min()) <= 0:
        return 0.0
    centered = times - times.mean()
    return float(np.dot(centered, values - values.mean()) / max(float(np.dot(centered, centered)), 1e-12))


def track_vector(track: dict[str, Any], *, frame_count: int) -> np.ndarray:
    rows = track["detections"]
    times = np.asarray([row["frame_index"] for row in rows], dtype=np.float64) / max(1, frame_count - 1)
    areas = np.asarray([row["area"] for row in rows], dtype=np.float64)
    bottoms = np.asarray([row["bottom"] for row in rows], dtype=np.float64)
    centers = np.asarray([row["center_x"] for row in rows], dtype=np.float64)
    overlaps = np.asarray([row["corridor_overlap"] for row in rows], dtype=np.float64)
    threats = np.asarray([row["threat"] for row in rows], dtype=np.float64)
    confidences = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    group = rows[0]["group"]
    one_hot = np.asarray([float(group == name) for name in GROUP_NAMES], dtype=np.float64)
    return np.concatenate([
        np.asarray([
            len(rows) / frame_count,
            float(times[0]), float(times[-1]),
            float(confidences.mean()), float(confidences.max()),
            float(areas[0]), float(areas[-1]), float(areas.max()), slope(np.log(areas + 1e-6), times),
            float(bottoms[0]), float(bottoms[-1]), slope(bottoms, times),
            float(centers[0]), float(centers[-1]), slope(centers, times), float(np.ptp(centers)),
            float(overlaps[0]), float(overlaps[-1]), float(overlaps.max()), slope(overlaps, times),
            float(threats[0]), float(threats[-1]), float(threats.max()), slope(threats, times),
        ], dtype=np.float64),
        one_hot,
    ])


def episode_vector(frames: Sequence[Sequence[dict[str, Any]]]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(frames) < 2:
        raise ValueError("trajectory episode needs at least two frames")
    tracks = track_detections(frames)
    track_vectors = np.stack([track_vector(track, frame_count=len(frames)) for track in tracks]) if tracks else np.zeros((0, 29), dtype=np.float64)
    scored = sorted(
        zip(tracks, track_vectors),
        key=lambda item: max(row["threat"] for row in item[0]["detections"]),
        reverse=True,
    )
    top = [vector for _track, vector in scored[:3]]
    while len(top) < 3:
        top.append(np.zeros(29, dtype=np.float64))
    aggregate = np.concatenate([
        track_vectors.mean(axis=0) if len(track_vectors) else np.zeros(29),
        track_vectors.max(axis=0) if len(track_vectors) else np.zeros(29),
    ])
    frame_rows: list[np.ndarray] = []
    for detections in frames:
        if detections:
            frame_rows.append(np.asarray([
                len(detections),
                max(row["threat"] for row in detections),
                max(row["area"] for row in detections),
                max(row["corridor_overlap"] for row in detections),
                sum(row["area"] * row["corridor_overlap"] for row in detections),
                max(row["bottom"] for row in detections),
            ], dtype=np.float64))
        else:
            frame_rows.append(np.zeros(6, dtype=np.float64))
    frame_values = np.stack(frame_rows)
    temporal = np.concatenate([frame_values.mean(axis=0), frame_values.max(axis=0), frame_values[-1] - frame_values[0]])
    vector = np.concatenate([*top, aggregate, temporal])
    return vector, {
        "detection_count": sum(len(items) for items in frames),
        "track_count": len(tracks),
        "persistent_track_count": sum(len(track["detections"]) >= 2 for track in tracks),
        "feature_dimension": int(len(vector)),
    }


def detector_class_ids(names: dict[int, str]) -> list[int]:
    allowed = set().union(*GROUP_CLASSES.values())
    return sorted(index for index, name in names.items() if name in allowed)


def extract_frame_detections(
    model: Any,
    episodes: Sequence[dict[str, Any]],
    *,
    image_size: int,
    confidence: float,
) -> list[list[list[dict[str, Any]]]]:
    """Run the frozen detector once and retain normalized per-frame proposals."""
    class_ids = detector_class_ids(model.names)
    episode_frames: list[list[list[dict[str, Any]]]] = []
    for episode in episodes:
        paths = [frame["path"] for frame in episode["frames"]]
        results = model.predict(
            source=paths,
            imgsz=image_size,
            conf=confidence,
            iou=0.5,
            classes=class_ids,
            agnostic_nms=True,
            device="cpu",
            verbose=False,
        )
        frames: list[list[dict[str, Any]]] = []
        for result in results:
            height, width = result.orig_shape
            detections: list[dict[str, Any]] = []
            for box, class_id, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                detections.append(normalize_detection(
                    model.names[int(class_id)],
                    float(score),
                    [float(value) for value in box],
                    width=width,
                    height=height,
                ))
            frames.append(detections)
        episode_frames.append(frames)
    return episode_frames


def extract(model: Any, episodes: Sequence[dict[str, Any]], *, image_size: int, confidence: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    episode_frames = extract_frame_detections(
        model,
        episodes,
        image_size=image_size,
        confidence=confidence,
    )
    vectors: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for episode, frames in zip(episodes, episode_frames):
        vector, summary = episode_vector(frames)
        vectors.append(vector)
        summaries.append({"episode_id": episode["episode_id"], **summary})
    return np.stack(vectors), summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    weights = args.detector_weights.resolve()
    if not package_root.is_dir() or not weights.is_file():
        raise FileNotFoundError("package root or detector weights are missing")
    episodes, excluded = common.load_episode_specs(package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("trajectory probe requires both classes")

    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    features, summaries = extract(model, episodes, image_size=args.image_size, confidence=args.confidence)
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]
    first = common.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True)
    second = common.leave_one_source_group_out(features, labels, episode_ids, source_ids, ridge=args.ridge, class_balanced=True)
    deterministic = first == second
    metrics = first["metrics"]
    gate = bool(
        deterministic
        and metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
    )
    delta_alignment = common.counterfactual_delta_alignment(features, episodes)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {"candidate_no_alert": int(np.sum(labels == 0)), "candidate_alert": int(np.sum(labels == 1))},
        "feature_source": {
            "model": weights.name,
            "weights": str(weights),
            "weights_sha256": common.sha256_file(weights),
            "image_size": args.image_size,
            "confidence": args.confidence,
            "feature_dimension": int(features.shape[1]),
            "trainable_parameters": 0,
            "role": "frozen COCO object proposals only; detections are not event truth",
            "trajectory_contract": "same coarse group; greedy minimum cost using center distance, IoU, and log-area change; maximum one missing sampled frame",
            "corridor_contract": "fixed trapezoid top=.38, top_half_width=.18, bottom_half_width=.42",
        },
        "evaluation": {
            "split": "leave_one_source_group_out",
            "group_key": "source_id",
            "frame_or_session_leakage": False,
            "ridge": args.ridge,
            "training_fold_class_balance": "inverse_frequency_equal_class_weight",
            **first,
            "repeat_exact": deterministic,
        },
        "counterfactual_delta_alignment": delta_alignment,
        "linear_separability_gate": {
            "passed": gate,
            "thresholds": {"balanced_accuracy_gte": args.minimum_balanced_accuracy, "each_class_recall_gte": args.minimum_class_recall},
        },
        "episode_detection_summaries": summaries,
        "episodes": [{key: value for key, value in row.items() if key != "frames"} | {"frame_count": len(row["frames"])} for row in episodes],
        "excluded_abstentions": excluded,
        "evidence_limit": "Tiny GPT/VLM-labelled provisional set and frozen COCO proposals; diagnostic only, not human accuracy, calibration, blind evaluation, or production promotion.",
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-trajectory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or args.ridge <= 0:
        parser.error("confidence must be in (0,1) and ridge must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "episode_count": report["episode_count"],
        "balanced_accuracy": report["evaluation"]["metrics"]["balanced_accuracy"],
        "linear_separable": report["linear_separability_gate"]["passed"],
        "prototype_direction_aligned": report["counterfactual_delta_alignment"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
