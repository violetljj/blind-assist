#!/usr/bin/env python3
"""Run bounded real-video batch inference for candidate-event discovery.

This adapter is intentionally separate from the model-agnostic mining core. It
materializes sampled RGB review frames, runs YOLO and Depth Anything V2, and
converts only normalized discovery signals to the canonical frame JSONL. A
segmentation channel is accepted only from an explicit sidecar or the opt-in
image-space proxy; the proxy is not a segmentation model and is never reported
as one. HFTF is likewise an optional, frame-key-bound sidecar.

The resulting trace is for THESIS_DEVELOPMENT candidate discovery. It does not
create event truth, a benchmark, a safety decision, an Android/App output, or a
production permission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.candidate_event_mining.pipeline import (
    ContractError,
    FRAME_SCHEMA,
    refuse_overwrite,
    read_json,
    sha256_file,
    validate_project_index,
    write_json,
    write_jsonl,
)


ADAPTER_SCHEMA = "blindassist_candidate_event_mining_real_video_adapter_manifest_v1"
ADAPTER_ID = "cem-real-yolo11n-depth-anything-v2-vits-r0"


def _score(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"non-finite adapter score: {value}")
    return max(0.0, min(1.0, value))


def _safe_mean(value: np.ndarray) -> float:
    finite = value[np.isfinite(value)]
    return float(finite.mean()) if finite.size else 0.0


def _clip_norm(value: float, scale: float, offset: float = 0.0) -> float:
    if scale <= 0:
        return 0.0
    return _score((float(value) - offset) / scale)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(1e-9, left_area + right_area - inter)


def _names_get(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ContractError(f"cannot decode sampled frame: {path}")
    return image


def _materialize_frames(
    source: Mapping[str, Any],
    frame_root: Path,
    sample_fps: float,
    max_duration_seconds: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    media_path = Path(str(source["media_path"])).resolve()
    if not media_path.is_file():
        raise ContractError(f"indexed media is missing: {media_path}")
    media_sha = sha256_file(media_path)
    if media_sha != source["content_sha256"]:
        raise ContractError(
            f"indexed media hash mismatch for {source['source_id']}: "
            f"expected={source['content_sha256']} actual={media_sha}"
        )
    if sample_fps <= 0 or not math.isfinite(sample_fps):
        raise ContractError("sample_fps must be positive and finite")

    source_id = str(source["source_id"])
    session_id = str(source["session_id"])
    source_frame_root = frame_root / source_id / session_id
    if source_frame_root.exists() and any(source_frame_root.iterdir()):
        raise ContractError(f"refusing to reuse non-empty frame root: {source_frame_root}")
    source_frame_root.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise ContractError(f"cannot open indexed media: {media_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    stride = max(1, int(round(fps / sample_fps)))
    declared_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame_index = None
    if max_duration_seconds is not None:
        if max_duration_seconds <= 0 or not math.isfinite(max_duration_seconds):
            raise ContractError("max_duration_seconds must be positive when supplied")
        max_frame_index = int(math.floor(max_duration_seconds * fps))

    rows: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frame_index is not None and frame_index >= max_frame_index:
                break
            if frame_index % stride == 0:
                frame_path = source_frame_root / f"frame-{frame_index:010d}.jpg"
                if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise ContractError(f"failed to write review frame: {frame_path}")
                rows.append(
                    {
                        "source_id": source_id,
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "timestamp_ms": int(round(frame_index * 1000.0 / fps)),
                        "frame_path": frame_path.resolve(),
                        "frame_sha256": sha256_file(frame_path),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
            frame_index += 1
    finally:
        capture.release()
    if not rows:
        raise ContractError(f"no sampled frames for {source_id}")
    return rows, {
        "source_id": source_id,
        "session_id": session_id,
        "media_path": str(media_path),
        "media_sha256": media_sha,
        "source_fps": round(fps, 6),
        "decoded_frame_count": frame_index,
        "declared_frame_count": declared_frame_count,
        "sample_stride_frames": stride,
        "sampled_frame_count": len(rows),
        "sampled_duration_seconds": round(rows[-1]["timestamp_ms"] / 1000.0, 6),
        "frame_root": str(source_frame_root.resolve()),
        "window_limit_seconds": max_duration_seconds,
    }


def _compute_motion_and_geometry(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    for row in rows:
        image = _read_image(Path(str(row["frame_path"])))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 60, 150)
        bottom = edges[int(h * 0.55) :, :]
        bottom_central = bottom[:, int(w * 0.15) : int(w * 0.85)]
        edge_density = float(np.mean(bottom_central > 0))

        parallel_score = 0.0
        lines = cv2.HoughLinesP(
            bottom,
            1,
            np.pi / 180.0,
            threshold=max(18, int(w * 0.08)),
            minLineLength=max(24, int(w * 0.18)),
            maxLineGap=max(8, int(w * 0.05)),
        )
        if lines is not None:
            eligible = 0
            lengths: list[float] = []
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [float(value) for value in line]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                angle = min(angle, 180.0 - angle)
                if 4.0 <= angle <= 42.0:
                    eligible += 1
                    lengths.append(math.hypot(x2 - x1, y2 - y1) / max(1.0, w))
            if eligible:
                parallel_score = _score(0.45 * min(1.0, eligible / 5.0) + 0.55 * min(1.0, max(lengths) * 2.4))

        left_vertical = float(np.mean(edges[:, : max(1, int(w * 0.12))] > 0))
        right_vertical = float(np.mean(edges[:, min(w - 1, int(w * 0.88)) :] > 0))
        upper_edges = float(np.mean(edges[: max(1, int(h * 0.42)), :] > 0))
        scene_doorframe = _score((left_vertical + right_vertical) * 5.5)
        scene_tree_branch = _score(upper_edges * 3.5 + edge_density * 1.5)
        step_edge = _score(edge_density * 4.0)

        flow_magnitude = 0.0
        mean_dx = 0.0
        mean_dy = 0.0
        flow_jitter = 0.0
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                small,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            dx = flow[..., 0]
            dy = flow[..., 1]
            magnitude = np.sqrt(dx * dx + dy * dy)
            flow_magnitude = _clip_norm(float(np.percentile(magnitude, 90)), 6.0)
            mean_dx = float(np.median(dx))
            mean_dy = float(np.median(dy))
            flow_jitter = _score(float(np.percentile(magnitude, 90) - np.percentile(magnitude, 50)) / 4.0)
        previous_gray = small

        output.append(
            {
                "flow_magnitude": flow_magnitude,
                "flow_mean_dx": mean_dx,
                "flow_mean_dy": mean_dy,
                "motion_head_turn": _clip_norm(abs(mean_dx), 2.2),
                "motion_jitter": flow_jitter,
                "motion_front_approach": _score(0.55 * flow_magnitude + 0.45 * _clip_norm(max(0.0, mean_dy), 2.0)),
                "edge_density_bottom": _score(edge_density * 4.0),
                "geometry_parallel_curb": parallel_score,
                "geometry_step_edge": step_edge,
                "scene_doorframe": scene_doorframe,
                "scene_tree_branch": scene_tree_branch,
            }
        )
    return output


def _yolo_detections(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    obstacle_names = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "train",
        "truck",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "bench",
        "bird",
        "cat",
        "dog",
        "backpack",
        "umbrella",
        "handbag",
        "suitcase",
        "chair",
        "couch",
        "potted plant",
        "dining table",
    }
    table_names = {"bench", "chair", "couch", "dining table"}
    output: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        images = [_read_image(Path(str(row["frame_path"]))) for row in batch_rows]
        results = model.predict(
            source=images,
            device=(0 if device == "cuda" else device),
            imgsz=640,
            conf=0.15,
            iou=0.7,
            verbose=False,
        )
        if len(results) != len(batch_rows):
            raise ContractError(f"YOLO result count mismatch: {len(results)} != {len(batch_rows)}")
        for result in results:
            height, width = result.orig_shape[:2]
            detections: list[dict[str, Any]] = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confidences = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                for box, confidence, class_id in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)
                    cx = ((x1 + x2) / 2.0) / max(1.0, width)
                    cy = ((y1 + y2) / 2.0) / max(1.0, height)
                    name = _names_get(result.names, int(class_id)).lower()
                    detections.append(
                        {
                            "class_id": int(class_id),
                            "name": name,
                            "confidence": float(confidence),
                            "bbox": [x1, y1, x2, y2],
                            "cx": cx,
                            "cy": cy,
                            "area": area,
                        }
                    )

            coverage = _score(sum(float(item["area"]) for item in detections if item["name"] in obstacle_names) * 6.0)
            front_obstacle = 0.0
            table_corner = 0.0
            for item in detections:
                if item["name"] not in obstacle_names:
                    continue
                centrality = _score(1.0 - abs(float(item["cx"]) - 0.5) / 0.5)
                bottomness = _score(0.35 + 0.65 * float(item["cy"]))
                area_score = _score(float(item["area"]) * 15.0)
                score = _score(float(item["confidence"]) * (0.25 + 0.75 * centrality) * (0.35 + 0.65 * bottomness) * (0.35 + 0.65 * area_score))
                front_obstacle = max(front_obstacle, score)
                if item["name"] in table_names:
                    table_corner = max(table_corner, score)

            crossing = 0.0
            for current in detections:
                matches = [item for item in previous if item["class_id"] == current["class_id"]]
                if not matches:
                    continue
                best = max(matches, key=lambda item: _bbox_iou(item["bbox"], current["bbox"]))
                dx = abs(float(current["cx"]) - float(best["cx"]))
                dy = abs(float(current["cy"]) - float(best["cy"]))
                if dx < 0.025 or dy > 0.30:
                    continue
                crossing = max(crossing, _score(dx * 5.0) * _score((float(current["confidence"]) + float(best["confidence"])) / 2.0 * 1.4))
            person_count = sum(item["name"] == "person" for item in detections)
            dynamic_crowd = _score(min(1.0, person_count / 4.0) * (0.55 + 0.45 * crossing))
            output.append(
                {
                    "coverage": coverage,
                    "front_obstacle": front_obstacle,
                    "table_corner": table_corner,
                    "crossing": crossing,
                    "dynamic_crowd": dynamic_crowd,
                    "person_presence": _score(min(1.0, person_count / 2.0)),
                    "detections": detections,
                }
            )
            previous = detections
    if len(output) != len(rows):
        raise ContractError(f"YOLO output count mismatch after batching: {len(output)} != {len(rows)}")
    return output


def _depth_features(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    input_size: int,
) -> list[dict[str, float]]:
    import torch
    import torch.nn.functional as F

    raw_features: list[dict[str, float]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        images = [_read_image(Path(str(row["frame_path"]))) for row in batch_rows]
        tensors = []
        original_sizes: list[tuple[int, int]] = []
        for image in images:
            tensor, (height, width) = model.image2tensor(image, input_size)
            tensors.append(tensor)
            original_sizes.append((height, width))
        batch = torch.cat(tensors, dim=0)
        with torch.inference_mode():
            depth = model(batch)
            depth = F.interpolate(
                depth[:, None],
                (original_sizes[0][0], original_sizes[0][1]),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        depth_np = depth.detach().float().cpu().numpy()
        for array, (height, width) in zip(depth_np, original_sizes):
            array = np.asarray(array, dtype=np.float32)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise ContractError("Depth Anything returned no finite pixels")
            low, high = np.percentile(finite, [5.0, 95.0])
            normalized = np.clip((array - low) / max(1e-6, high - low), 0.0, 1.0)
            y_mid = int(height * 0.45)
            y_bottom = int(height * 0.68)
            x_left = int(width * 0.22)
            x_right = int(width * 0.78)
            center_left = int(width * 0.35)
            center_right = int(width * 0.65)
            front_raw = _safe_mean(normalized[y_bottom:, x_left:x_right])
            central_raw = _safe_mean(normalized[y_mid:, center_left:center_right])
            mid_raw = _safe_mean(normalized[y_mid:int(height * 0.68), x_left:x_right])
            bottom_raw = _safe_mean(normalized[int(height * 0.68):, x_left:x_right])
            raw_features.append(
                {
                    "front_raw": front_raw,
                    "central_raw": central_raw,
                    "vertical_boundary": _score(abs(bottom_raw - mid_raw) * 2.3),
                }
            )
    if len(raw_features) != len(rows):
        raise ContractError(f"Depth output count mismatch: {len(raw_features)} != {len(rows)}")
    fronts = np.asarray([item["front_raw"] for item in raw_features], dtype=np.float32)
    low, high = np.percentile(fronts, [10.0, 90.0]) if fronts.size > 1 else (0.0, 1.0)
    scale = max(1e-6, float(high - low))
    normalized_fronts = np.clip((fronts - low) / scale, 0.0, 1.0)
    output: list[dict[str, float]] = []
    previous = 0.0
    for index, item in enumerate(raw_features):
        near = float(normalized_fronts[index])
        delta = max(0.0, near - previous) if index else 0.0
        previous = near
        output.append(
            {
                "front_approach": _score(0.30 * near + 0.70 * min(1.0, delta * 4.0)),
                "static_obstacle_approach": _score(0.25 * near + 0.75 * min(1.0, delta * 4.0)),
                "step_drop": _score(0.55 * item["vertical_boundary"] + 0.45 * min(1.0, delta * 2.0)),
                "front_near": near,
            }
        )
    return output


def _load_sidecar(path: Path | None, allowed_prefixes: Sequence[str]) -> dict[tuple[str, str, int], dict[str, float]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ContractError(f"sidecar does not exist: {path}")
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"sidecar line {line_number} must be an object")
            key = (str(value.get("source_id", "")), str(value.get("session_id", "")), int(value.get("frame_index", -1)))
            if key in rows:
                raise ContractError(f"duplicate sidecar frame key: {key}")
            raw_signals = value.get("signals", value)
            if not isinstance(raw_signals, dict):
                raise ContractError(f"sidecar line {line_number} signals must be an object")
            signals: dict[str, float] = {}
            for signal_key, signal_value in raw_signals.items():
                if not isinstance(signal_key, str) or not any(signal_key.startswith(prefix) for prefix in allowed_prefixes):
                    continue
                signals[signal_key] = _score(signal_value)
            if signals:
                rows[key] = signals
    return rows


def _build_trace_rows(
    rows: Sequence[Mapping[str, Any]],
    motion: Sequence[Mapping[str, float]],
    yolo: Sequence[Mapping[str, Any]],
    depth: Sequence[Mapping[str, float]],
    segmentation_sidecar: Mapping[tuple[str, str, int], Mapping[str, float]],
    hftf_sidecar: Mapping[tuple[str, str, int], Mapping[str, float]],
    enable_segmentation_proxy: bool,
    enable_scene_shape_proxy: bool,
    source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, motion_item, yolo_item, depth_item in zip(rows, motion, yolo, depth):
        signals: dict[str, float] = {
            "yolo.coverage": _score(yolo_item["coverage"]),
            "yolo.miss": _score(1.0 - float(yolo_item["coverage"])),
            "yolo.front_obstacle": _score(yolo_item["front_obstacle"]),
            "motion.front_approach": _score(motion_item["motion_front_approach"]),
            "motion.crossing": _score(yolo_item["crossing"]),
            "object.crossing": _score(yolo_item["crossing"]),
            "motion.dynamic_crowd": _score(yolo_item["dynamic_crowd"]),
            "object.dynamic_crowd": _score(yolo_item["dynamic_crowd"]),
            "motion.head_turn": _score(motion_item["motion_head_turn"]),
            "motion.jitter": _score(motion_item["motion_jitter"]),
            "depth.front_approach": _score(depth_item["front_approach"]),
            "depth.static_obstacle_approach": _score(depth_item["static_obstacle_approach"]),
            "geometry.step_drop": _score(max(depth_item["step_drop"], motion_item["geometry_step_edge"])),
            "geometry.parallel_curb": _score(motion_item["geometry_parallel_curb"]),
        }
        if yolo_item["table_corner"] > 0:
            signals["object.table_corner"] = _score(yolo_item["table_corner"])
        else:
            signals["object.table_corner"] = 0.0
        if enable_scene_shape_proxy:
            signals["object.doorframe"] = _score(motion_item["scene_doorframe"])
            signals["object.tree_branch"] = _score(motion_item["scene_tree_branch"])
        else:
            signals["object.doorframe"] = 0.0
            signals["object.tree_branch"] = 0.0

        front_risk = max(float(yolo_item["front_obstacle"]), float(depth_item["front_approach"]))
        signals["context.normal_passage"] = _score(
            (1.0 - front_risk)
            * (1.0 - 0.65 * float(depth_item["front_approach"]))
            * (1.0 - 0.45 * float(motion_item["motion_jitter"]))
        )
        if enable_segmentation_proxy:
            proxy_risk = _score(
                0.45 * float(depth_item["front_approach"])
                + 0.30 * float(depth_item["step_drop"])
                + 0.25 * float(motion_item["edge_density_bottom"])
            )
            signals["segmentation.alert"] = proxy_risk
            signals["segmentation.risk"] = proxy_risk
            signals["segmentation.front_risk"] = _score(max(proxy_risk, float(depth_item["front_approach"])))
            signals["segmentation.boundary_level_change"] = _score(max(float(depth_item["step_drop"]), proxy_risk * 0.7))

        key = (str(row["source_id"]), str(row["session_id"]), int(row["frame_index"]))
        signals.update(segmentation_sidecar.get(key, {}))
        signals.update(hftf_sidecar.get(key, {}))
        output.append(
            {
                "schema": FRAME_SCHEMA,
                "source_id": row["source_id"],
                "session_id": row["session_id"],
                "frame_index": int(row["frame_index"]),
                "timestamp_ms": int(row["timestamp_ms"]),
                "frame_ref": str(row["frame_path"]),
                "frame_sha256": row["frame_sha256"],
                "source_manifest_sha256": source_manifest_sha256,
                "signals": {key: _score(value) for key, value in sorted(signals.items())},
            }
        )
    return output


def _load_models(args: argparse.Namespace) -> tuple[Any, Any, str, dict[str, Any]]:
    import torch
    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("--device cuda requested but CUDA is unavailable")
    yolo_path = args.yolo_model.resolve()
    depth_path = args.depth_checkpoint.resolve()
    if not yolo_path.is_file() or not depth_path.is_file():
        raise ContractError(f"model path missing: yolo={yolo_path} depth={depth_path}")
    yolo = YOLO(str(yolo_path))

    source_root = args.depth_source_root.resolve()
    if not (source_root / "depth_anything_v2" / "dpt.py").is_file():
        raise ContractError(f"Depth Anything source package missing under {source_root}")
    sys.path.insert(0, str(source_root))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    finally:
        sys.path.remove(str(source_root))
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(str(depth_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(torch.device(device)).eval()
    return yolo, model, device, {
        "yolo": {
            "path": str(yolo_path),
            "sha256": sha256_file(yolo_path),
            "runtime": "ultralytics",
        },
        "depth": {
            "path": str(depth_path),
            "sha256": sha256_file(depth_path),
            "runtime": "DepthAnythingV2 PyTorch",
            "encoder": "vits",
            "input_size": int(args.depth_input_size),
        },
        "device": device,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_id.strip():
        raise ContractError("run_id must be non-empty")
    project_index_path = args.project_index.resolve()
    project_index = validate_project_index(read_json(project_index_path))
    selected_ids = set(args.source_id or [source["source_id"] for source in project_index["sources"]])
    sources = [source for source in project_index["sources"] if source["source_id"] in selected_ids]
    if len(sources) != len(selected_ids):
        known = {source["source_id"] for source in project_index["sources"]}
        raise ContractError(f"unknown source_id: {sorted(selected_ids - known)}")
    if not sources:
        raise ContractError("no sources selected")

    output = args.output.resolve()
    manifest_path = output.with_name(output.stem + ".adapter_manifest.json")
    refuse_overwrite(output)
    refuse_overwrite(manifest_path)
    frame_root = (args.frame_root or Path(r"F:\ba-data\blindassist-candidate-event-mining\runs") / args.run_id).resolve()
    if frame_root.exists() and any(frame_root.iterdir()):
        raise ContractError(f"refusing to reuse non-empty frame root: {frame_root}")
    frame_root.mkdir(parents=True, exist_ok=True)

    segmentation_sidecar = _load_sidecar(args.segmentation_sidecar, ("segmentation.",))
    hftf_sidecar = _load_sidecar(args.hftf_sidecar, ("hftf.",))
    yolo, depth_model, device, model_meta = _load_models(args)
    project_index_sha = sha256_file(project_index_path)
    all_trace_rows: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    for source in sources:
        sampled_rows, source_meta = _materialize_frames(
            source,
            frame_root,
            args.sample_fps,
            args.max_duration_seconds,
        )
        motion = _compute_motion_and_geometry(sampled_rows)
        yolo_features = _yolo_detections(yolo, sampled_rows, args.batch_size, device)
        depth_features = _depth_features(depth_model, sampled_rows, args.depth_batch_size, args.depth_input_size)
        trace_rows = _build_trace_rows(
            sampled_rows,
            motion,
            yolo_features,
            depth_features,
            segmentation_sidecar,
            hftf_sidecar,
            args.enable_segmentation_proxy,
            args.enable_scene_shape_proxy,
            project_index_sha,
        )
        all_trace_rows.extend(trace_rows)
        source_runs.append(source_meta)

    write_jsonl(output, all_trace_rows)
    manifest = {
        "schema": ADAPTER_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "run_id": args.run_id,
        "project_index": {"path": str(project_index_path), "sha256": project_index_sha},
        "source_runs": source_runs,
        "frame_trace": {"path": str(output), "sha256": sha256_file(output), "frame_count": len(all_trace_rows)},
        "model_bundle": model_meta,
        "sampling": {
            "sample_fps_requested": args.sample_fps,
            "depth_batch_size": args.depth_batch_size,
            "yolo_batch_size": args.batch_size,
        },
        "optional_channels": {
            "segmentation": (
                "image_space_risk_proxy_not_a_segmentation_model"
                if args.enable_segmentation_proxy
                else ("sidecar" if args.segmentation_sidecar else "not_provided")
            ),
            "hftf": "sidecar" if args.hftf_sidecar else "not_provided",
            "scene_shape": "image_space_shape_proxy" if args.enable_scene_shape_proxy else "not_provided",
        },
        "sidecars": {
            "segmentation": (None if args.segmentation_sidecar is None else {"path": str(args.segmentation_sidecar.resolve()), "sha256": sha256_file(args.segmentation_sidecar.resolve())}),
            "hftf": (None if args.hftf_sidecar is None else {"path": str(args.hftf_sidecar.resolve()), "sha256": sha256_file(args.hftf_sidecar.resolve())}),
        },
        "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
        "execution_boundary": "real_video_batch_inference_to_truth_free_canonical_trace_only",
        "authorization": {
            "event_truth": False,
            "training": False,
            "confirmation": False,
            "production": False,
            "safety": False,
            "default_app": False,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-index", type=Path, required=True)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--depth-batch-size", type=int, default=8)
    parser.add_argument("--depth-input-size", type=int, default=252)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-source-root", type=Path, required=True)
    parser.add_argument("--segmentation-sidecar", type=Path)
    parser.add_argument("--hftf-sidecar", type=Path)
    parser.add_argument("--enable-segmentation-proxy", action="store_true")
    parser.add_argument("--enable-scene-shape-proxy", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        manifest = run(parse_args())
    except (ContractError, OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": manifest["run_id"],
                "frame_count": manifest["frame_trace"]["frame_count"],
                "output": manifest["frame_trace"]["path"],
                "manifest": str(manifest["frame_trace"]["path"]).replace(".jsonl", ".adapter_manifest.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
