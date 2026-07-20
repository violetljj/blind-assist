#!/usr/bin/env python3
"""Probe obstacle-aware residual route width with frozen segmentation.

The probe combines frozen ADE20K walkable classes with already-frozen
chromatic-marker detections.  It then computes the widest continuous image
path from the lower field of view to the mid-field horizon.  No classifier,
threshold search, or real-video label is fitted.
"""

from __future__ import annotations

import argparse
import heapq
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_free_space_topology_probe as topology
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_obstacle_aware_route_width_probe_v1"
WALKABLE_LABELS = ("floor", "road", "sidewalk", "path")
MAP_SIZE = 128
HORIZON_RATIO = 0.48
START_RATIO = 0.88
START_LEFT_RATIO = 0.12
START_RIGHT_RATIO = 0.88
SAFETY_MARGIN_OBJECT_HEIGHTS = 1.0


def accepted_detection(row: dict[str, Any]) -> bool:
    features = row.get("features") or {}
    return float(features.get("high_saturation_fraction", 0.0)) > float(
        features.get("dark_fraction", 0.0)
    )


def obstacle_mask_from_detections(
    detections: Sequence[dict[str, Any]], shape: tuple[int, int], *, safety_margin_object_heights: float = SAFETY_MARGIN_OBJECT_HEIGHTS
) -> np.ndarray:
    height, width = shape
    if min(height, width) < 8:
        raise ValueError("obstacle map is too small")
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in detections:
        if not accepted_detection(row):
            continue
        features = row["features"]
        center_x = float(features["center_x_norm"])
        box_width = float(features["width_norm"])
        box_height = float(features["height_norm"])
        bottom_y = float(features["bottom_y_norm"])
        if safety_margin_object_heights < 0.0:
            raise ValueError("safety margin must be non-negative")
        margin = safety_margin_object_heights * box_height
        x1 = int(np.floor((center_x - box_width / 2.0 - margin) * width))
        x2 = int(np.ceil((center_x + box_width / 2.0 + margin) * width))
        y1 = int(np.floor((bottom_y - box_height) * height))
        y2 = int(np.ceil(bottom_y * height))
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def obstacle_mask_from_intervention(
    mask: np.ndarray, shape: tuple[int, int], *, safety_margin_object_heights: float = SAFETY_MARGIN_OBJECT_HEIGHTS
) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values.max(axis=2)
    binary = np.asarray(values > 0, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    detections: list[dict[str, Any]] = []
    source_height, source_width = binary.shape
    for component in range(1, count):
        x, y, width, height, area = stats[component].tolist()
        if area <= 0:
            continue
        detections.append({
            "features": {
                "center_x_norm": (x + width / 2.0) / source_width,
                "width_norm": width / source_width,
                "height_norm": height / source_height,
                "bottom_y_norm": (y + height) / source_height,
                "high_saturation_fraction": 1.0,
                "dark_fraction": 0.0,
            }
        })
    return obstacle_mask_from_detections(
        detections, shape, safety_margin_object_heights=safety_margin_object_heights
    )


def widest_route(walkable: np.ndarray, obstacle: np.ndarray) -> dict[str, float | bool]:
    free = np.asarray(walkable, dtype=bool) & ~np.asarray(obstacle, dtype=bool)
    if free.ndim != 2 or free.shape != obstacle.shape or min(free.shape) < 8:
        raise ValueError("walkable and obstacle maps must be matching 2D arrays")
    height, width = free.shape
    distance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    horizon = int(round(height * HORIZON_RATIO))
    start_row = min(height - 1, int(round(height * START_RATIO)))
    left = int(round(width * START_LEFT_RATIO))
    right = int(round(width * START_RIGHT_RATIO))
    capacity = np.full((height, width), -1.0, dtype=np.float32)
    parent = np.full((height, width, 2), -1, dtype=np.int16)
    queue: list[tuple[float, int, int]] = []
    for row in range(start_row, height):
        for column in range(left, right):
            if not free[row, column]:
                continue
            value = float(distance[row, column])
            if value > capacity[row, column]:
                capacity[row, column] = value
                heapq.heappush(queue, (-value, row, column))
    target: tuple[int, int] | None = None
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        negative, row, column = heapq.heappop(queue)
        value = -negative
        if value + 1e-7 < float(capacity[row, column]):
            continue
        if row <= horizon:
            target = (row, column)
            break
        for dr, dc in neighbours:
            nr, nc = row + dr, column + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or not free[nr, nc]:
                continue
            candidate = min(value, float(distance[nr, nc]))
            if candidate > float(capacity[nr, nc]) + 1e-7:
                capacity[nr, nc] = candidate
                parent[nr, nc] = (row, column)
                heapq.heappush(queue, (-candidate, nr, nc))
    if target is None:
        return {
            "path_found": False,
            "route_radius_norm": 0.0,
            "route_width_norm": 0.0,
            "path_offset_mean": 1.0,
            "path_offset_maximum": 1.0,
        }
    points: list[tuple[int, int]] = []
    row, column = target
    while row >= 0 and column >= 0:
        points.append((row, column))
        previous = parent[row, column]
        if previous[0] < 0:
            break
        row, column = int(previous[0]), int(previous[1])
    offsets = [abs(column / max(width - 1, 1) - 0.5) for _, column in points]
    radius = float(capacity[target]) / width
    return {
        "path_found": True,
        "route_radius_norm": radius,
        "route_width_norm": min(1.0, 2.0 * radius),
        "path_offset_mean": float(np.mean(offsets)),
        "path_offset_maximum": float(np.max(offsets)),
    }


def widest_soft_route(walkable_probability: np.ndarray, obstacle: np.ndarray) -> dict[str, float | bool]:
    walkable = np.asarray(walkable_probability, dtype=np.float32)
    blocked = np.asarray(obstacle, dtype=bool)
    if walkable.ndim != 2 or walkable.shape != blocked.shape or min(walkable.shape) < 8:
        raise ValueError("soft walkable and obstacle maps must be matching 2D arrays")
    if not np.isfinite(walkable).all() or float(walkable.min()) < 0.0 or float(walkable.max()) > 1.0:
        raise ValueError("soft walkable probability must be finite and within zero to one")
    height, width = walkable.shape
    free = ~blocked
    free[:, 0] = False
    free[:, -1] = False
    clearance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) / width
    node_capacity = walkable * clearance
    node_capacity[blocked] = 0.0
    horizon = int(round(height * HORIZON_RATIO))
    start_row = min(height - 1, int(round(height * START_RATIO)))
    left = int(round(width * START_LEFT_RATIO))
    right = int(round(width * START_RIGHT_RATIO))
    capacity = np.full((height, width), -1.0, dtype=np.float32)
    parent = np.full((height, width, 2), -1, dtype=np.int16)
    queue: list[tuple[float, int, int]] = []
    for row in range(start_row, height):
        for column in range(left, right):
            value = float(node_capacity[row, column])
            if value <= 0.0:
                continue
            capacity[row, column] = value
            heapq.heappush(queue, (-value, row, column))
    target: tuple[int, int] | None = None
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        negative, row, column = heapq.heappop(queue)
        value = -negative
        if value + 1e-7 < float(capacity[row, column]):
            continue
        if row <= horizon:
            target = (row, column)
            break
        for dr, dc in neighbours:
            nr, nc = row + dr, column + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            local = float(node_capacity[nr, nc])
            if local <= 0.0:
                continue
            candidate = min(value, local)
            if candidate > float(capacity[nr, nc]) + 1e-7:
                capacity[nr, nc] = candidate
                parent[nr, nc] = (row, column)
                heapq.heappush(queue, (-candidate, nr, nc))
    if target is None:
        return {
            "path_found": False,
            "route_radius_norm": 0.0,
            "route_width_norm": 0.0,
            "path_offset_mean": 1.0,
            "path_offset_maximum": 1.0,
        }
    points: list[tuple[int, int]] = []
    row, column = target
    while row >= 0 and column >= 0:
        points.append((row, column))
        previous = parent[row, column]
        if previous[0] < 0:
            break
        row, column = int(previous[0]), int(previous[1])
    offsets = [abs(column / max(width - 1, 1) - 0.5) for _, column in points]
    radius = float(capacity[target])
    return {
        "path_found": True,
        "route_radius_norm": radius,
        "route_width_norm": min(1.0, 2.0 * radius),
        "path_offset_mean": float(np.mean(offsets)),
        "path_offset_maximum": float(np.max(offsets)),
    }


def adaptive_path_obstacle_clearance(
    walkable_probability: np.ndarray, obstacle: np.ndarray
) -> dict[str, float | bool]:
    walkable = np.asarray(walkable_probability, dtype=np.float32)
    blocked = np.asarray(obstacle, dtype=bool)
    if walkable.ndim != 2 or walkable.shape != blocked.shape or min(walkable.shape) < 8:
        raise ValueError("adaptive-path inputs must be matching 2D arrays")
    if not np.isfinite(walkable).all() or float(walkable.min()) < 0.0 or float(walkable.max()) > 1.0:
        raise ValueError("adaptive-path walkable probability must be finite and within zero to one")
    height, width = walkable.shape
    centers, _ = topology.trace_adaptive_path(walkable, horizon_ratio=0.30)
    first_row = height - len(centers)
    rows = np.arange(first_row, height)
    selected = (rows >= int(round(height * HORIZON_RATIO))) & (rows <= int(round(height * 0.94)))
    rows = rows[selected]
    centers = centers[selected]
    if not len(rows):
        raise ValueError("adaptive path has no rows inside the route-width band")
    free = ~blocked
    free[:, 0] = False
    free[:, -1] = False
    clearance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) / width
    values = clearance[rows, centers]
    offsets = np.abs(centers / max(width - 1, 1) - 0.5)
    radius = float(np.quantile(values, 0.10))
    return {
        "path_found": True,
        "route_radius_norm": radius,
        "route_width_norm": min(1.0, 2.0 * radius),
        "path_offset_mean": float(np.mean(offsets)),
        "path_offset_maximum": float(np.max(offsets)),
    }


def route_descriptor(
    walkable: np.ndarray, obstacle: np.ndarray, route_mode: str
) -> dict[str, float | bool]:
    if route_mode == "hard_argmax":
        return widest_route(walkable, obstacle)
    if route_mode == "soft_walkable_margin":
        return widest_soft_route(walkable, obstacle)
    if route_mode == "adaptive_path_distance_field":
        return adaptive_path_obstacle_clearance(walkable, obstacle)
    raise ValueError(f"unsupported route mode: {route_mode}")


class FrozenWalkableTeacher:
    def __init__(self, model_dir: Path) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True).eval()
        labels = {int(key): value.lower() for key, value in self.model.config.id2label.items()}
        by_name = {value: key for key, value in labels.items()}
        missing = [name for name in WALKABLE_LABELS if name not in by_name]
        if missing:
            raise ValueError(f"SegFormer is missing walkable labels: {missing}")
        self.walkable_ids = [by_name[name] for name in WALKABLE_LABELS]

    def maps(self, images: Sequence[np.ndarray], *, batch_size: int) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in batch]
            inputs = self.processor(images=rgb, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                labels = logits.argmax(dim=1).cpu().numpy()
            for label_map in labels:
                resized = cv2.resize(label_map.astype(np.int32), (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_NEAREST)
                rows.append(np.isin(resized, self.walkable_ids))
        return rows

    def probability_maps(self, images: Sequence[np.ndarray], *, batch_size: int) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in batch]
            inputs = self.processor(images=rgb, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)[:, self.walkable_ids].sum(dim=1).cpu().numpy()
            for probability in probabilities:
                rows.append(cv2.resize(probability.astype(np.float32), (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_LINEAR))
        return rows


def decode_at(video: Path, timestamps: Sequence[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    frames: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode {video} at {timestamp}ms")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def window_score(
    teacher: FrozenWalkableTeacher,
    source: dict[str, Any],
    window: tuple[int, int],
    *,
    batch_size: int,
    route_mode: str,
    safety_margin_object_heights: float,
) -> dict[str, Any]:
    samples = [row for row in source["samples"] if window[0] <= int(row["timestamp_ms"]) < window[1]]
    if not samples:
        raise ValueError(f"feature window is empty: {window}")
    timestamps = [int(row["timestamp_ms"]) for row in samples]
    frames = decode_at(Path(source["local_video_path"]), timestamps)
    maps = teacher.maps(frames, batch_size=batch_size) if route_mode == "hard_argmax" else teacher.probability_maps(frames, batch_size=batch_size)
    rows = []
    for walkable, sample in zip(maps, samples):
        obstacle = obstacle_mask_from_detections(
            sample.get("detections", []), walkable.shape,
            safety_margin_object_heights=safety_margin_object_heights,
        )
        route = route_descriptor(walkable, obstacle, route_mode)
        rows.append({"timestamp_ms": int(sample["timestamp_ms"]), **route})
    radii = [float(row["route_radius_norm"]) for row in rows]
    return {
        "window_ms": list(window),
        "frame_count": len(rows),
        "median_route_radius_norm": float(median(radii)),
        "minimum_route_radius_norm": float(min(radii)),
        "path_found_fraction": float(np.mean([bool(row["path_found"]) for row in rows])),
        "frames": rows,
    }


def real_pressure(
    teacher: FrozenWalkableTeacher,
    report: dict[str, Any],
    clear_window: tuple[int, int],
    risk_window: tuple[int, int],
    *,
    batch_size: int,
    route_mode: str,
    safety_margin_object_heights: float,
) -> dict[str, Any]:
    source = report["sources"][0]
    clear = window_score(teacher, source, clear_window, batch_size=batch_size, route_mode=route_mode, safety_margin_object_heights=safety_margin_object_heights)
    risk = window_score(teacher, source, risk_window, batch_size=batch_size, route_mode=route_mode, safety_margin_object_heights=safety_margin_object_heights)
    delta = float(risk["median_route_radius_norm"] - clear["median_route_radius_norm"])
    return {
        "source_id": source["source_id"],
        "video_sha256": source["video_sha256"],
        "clear": clear,
        "risk": risk,
        "risk_minus_clear_route_radius": delta,
        "ordered_as_more_constrained": delta < 0.0,
    }


def synthetic_pressure(
    teacher: FrozenWalkableTeacher, dataset_root: Path, rows: Sequence[dict[str, Any]], *, batch_size: int, route_mode: str, safety_margin_object_heights: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], {})[
            row["attributes"]["risk_state"]
        ] = row
    original: list[dict[str, Any]] = []
    mirrored: list[dict[str, Any]] = []
    for pair_id, members in sorted(grouped.items()):
        images = [cv2.imread(str(dataset_root / members[state]["image_path"]), cv2.IMREAD_COLOR) for state in ("clear", "risk")]
        masks = [cv2.imread(str(dataset_root / "masks" / f"{pair_id}_{state}_mask.png"), cv2.IMREAD_UNCHANGED) for state in ("clear", "risk")]
        if any(value is None for value in images + masks):
            raise ValueError(f"cannot decode synthetic pair: {pair_id}")
        maps = teacher.maps(images, batch_size=batch_size) if route_mode == "hard_argmax" else teacher.probability_maps(images, batch_size=batch_size)
        values = [route_descriptor(walkable, obstacle_mask_from_intervention(mask, walkable.shape, safety_margin_object_heights=safety_margin_object_heights), route_mode) for walkable, mask in zip(maps, masks)]
        delta = float(values[1]["route_radius_norm"] - values[0]["route_radius_norm"])
        original.append({"pair_id": pair_id, "clear": values[0], "risk": values[1], "risk_minus_clear_route_radius": delta, "ordered": delta < 0.0})
        mirror_images = [cv2.flip(image, 1) for image in images]
        mirror_masks = [cv2.flip(mask, 1) for mask in masks]
        mirror_maps = teacher.maps(mirror_images, batch_size=batch_size) if route_mode == "hard_argmax" else teacher.probability_maps(mirror_images, batch_size=batch_size)
        mirror_values = [route_descriptor(walkable, obstacle_mask_from_intervention(mask, walkable.shape, safety_margin_object_heights=safety_margin_object_heights), route_mode) for walkable, mask in zip(mirror_maps, mirror_masks)]
        mirror_delta = float(mirror_values[1]["route_radius_norm"] - mirror_values[0]["route_radius_norm"])
        mirrored.append({"pair_id": pair_id, "risk_minus_clear_route_radius": mirror_delta, "ordered": mirror_delta < 0.0})
    return original, mirrored


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.dataset_root, args.generation_report, args.model_dir, args.japan_features, args.edmonton_features, args.jakarta_features, args.cape_town_features, args.output]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    generation = lifecycle.verify_json_sidecar(args.generation_report)
    manifest_path = args.dataset_root / "manifest.jsonl"
    if common.sha256_file(manifest_path) != generation["manifest"]["sha256"]:
        raise ValueError("manifest differs from generation report")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports = {
        "japan_positive": lifecycle.verify_json_sidecar(args.japan_features),
        "edmonton_left_corridor_positive": lifecycle.verify_json_sidecar(args.edmonton_features),
        "jakarta_dense_boundary_negative": lifecycle.verify_json_sidecar(args.jakarta_features),
        "cape_town_wide_forecourt_negative": lifecycle.verify_json_sidecar(args.cape_town_features),
    }
    teacher = FrozenWalkableTeacher(args.model_dir)
    synthetic, mirrored = synthetic_pressure(teacher, args.dataset_root, manifest, batch_size=args.batch_size, route_mode=args.route_mode, safety_margin_object_heights=args.safety_margin_object_heights)
    real = {
        "japan_positive": real_pressure(teacher, reports["japan_positive"], (17000, 22000), (10000, 14000), batch_size=args.batch_size, route_mode=args.route_mode, safety_margin_object_heights=args.safety_margin_object_heights),
        "edmonton_left_corridor_positive": real_pressure(teacher, reports["edmonton_left_corridor_positive"], (782000, 810000), (671000, 735000), batch_size=args.batch_size, route_mode=args.route_mode, safety_margin_object_heights=args.safety_margin_object_heights),
        "jakarta_dense_boundary_negative": real_pressure(teacher, reports["jakarta_dense_boundary_negative"], (0, 15000), (35000, 49000), batch_size=args.batch_size, route_mode=args.route_mode, safety_margin_object_heights=args.safety_margin_object_heights),
        "cape_town_wide_forecourt_negative": real_pressure(teacher, reports["cape_town_wide_forecourt_negative"], (115000, 125000), (158000, 176000), batch_size=args.batch_size, route_mode=args.route_mode, safety_margin_object_heights=args.safety_margin_object_heights),
    }
    gate = bool(
        all(row["ordered"] for row in synthetic)
        and all(row["ordered"] for row in mirrored)
        and real["japan_positive"]["ordered_as_more_constrained"]
        and real["edmonton_left_corridor_positive"]["ordered_as_more_constrained"]
        and not real["jakarta_dense_boundary_negative"]["ordered_as_more_constrained"]
        and not real["cape_town_wide_forecourt_negative"]["ordered_as_more_constrained"]
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_zero_parameter_obstacle_aware_route_width_diagnostic",
        "inputs": {
            "generation_report_sha256": common.sha256_file(args.generation_report),
            "manifest_sha256": common.sha256_file(manifest_path),
            "segformer_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
            "feature_report_sha256": {key: common.sha256_file(path) for key, path in {
                "japan_positive": args.japan_features,
                "edmonton_left_corridor_positive": args.edmonton_features,
                "jakarta_dense_boundary_negative": args.jakarta_features,
                "cape_town_wide_forecourt_negative": args.cape_town_features,
            }.items()},
        },
        "feature_contract": {
            "walkable_teacher": "nvidia/segformer-b2-finetuned-ade-512-512",
            "route_mode": args.route_mode,
            "walkable_representation": "argmax class membership" if args.route_mode == "hard_argmax" else "summed softmax probability over frozen walkable classes",
            "walkable_labels": list(WALKABLE_LABELS),
            "map_size": MAP_SIZE,
            "horizon_ratio": HORIZON_RATIO,
            "start_ratio": START_RATIO,
            "start_lateral_range": [START_LEFT_RATIO, START_RIGHT_RATIO],
            "marker_safety_margin_object_heights_per_side": args.safety_margin_object_heights,
            "route_measure": {
                "hard_argmax": "maximum bottleneck Euclidean distance through connected walkable pixels after marker exclusion",
                "soft_walkable_margin": "maximum bottleneck of walkable probability times normalized obstacle and side-boundary clearance",
                "adaptive_path_distance_field": "q10 normalized distance from the frozen soft-walkable adaptive centerline to expanded marker obstacles or image side boundaries",
            }[args.route_mode],
            "threshold_fitted": False,
            "trainable_parameters": 0,
            "saved_weights": False,
        },
        "synthetic_pairs": synthetic,
        "horizontal_mirror_pressure": mirrored,
        "real_video_pressure": real,
        "diagnostic_gate": {
            "passed": gate,
            "requirements": [
                "all equal-count synthetic pairs have smaller residual route radius under path intrusion",
                "all mirrored synthetic pairs retain the ordering",
                "Japan and Edmonton positives become more constrained",
                "Jakarta and Cape Town reviewed negatives do not become more constrained",
            ],
        },
        "authorizations": {
            "future_prospective_contract_freeze": gate,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
        "evidence_limit": "Retrospective GPT/VLM silver and train-only synthetic pressure. A pass only permits freezing a new prospective contract; it is not event truth or deployment evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--japan-features", type=Path, required=True)
    parser.add_argument("--edmonton-features", type=Path, required=True)
    parser.add_argument("--jakarta-features", type=Path, required=True)
    parser.add_argument("--cape-town-features", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--route-mode", choices=("hard_argmax", "soft_walkable_margin", "adaptive_path_distance_field"), default="hard_argmax")
    parser.add_argument("--safety-margin-object-heights", type=float, default=SAFETY_MARGIN_OBJECT_HEIGHTS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "gate_passed": value["diagnostic_gate"]["passed"],
        "real_deltas": {key: row["risk_minus_clear_route_radius"] for key, row in value["real_video_pressure"].items()},
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
