#!/usr/bin/env python3
"""Run the bounded Windows-GPU external-RGB metric-track reference sidecar.

This reference emits metric observations and one-second D44 forecasts. It does
not emit alerts or make safety decisions. Camera mode requires calibrated
intrinsics; manifest mode replays already-materialized frames and detections.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from external_camera_calibration import (
    FrameRectifier,
    finite_ratio,
    load_calibration,
    pinhole_calibration,
)
from prepare_external_rgb_metric_depth_manifest import torso_roi_from_person_box
from produce_external_rgb_metric_depth_observations import (
    Metric3DPytorchSource,
    robust_roi_median,
)

HISTORY_COUNT = 7
FUTURE_SECONDS = 1.0


def relative_position(
    roi: list[int] | tuple[int, int, int, int],
    intrinsics: list[float],
    depth_m: float,
) -> np.ndarray:
    left, top, right, bottom = (float(value) for value in roi)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    u, v = (left + right) / 2.0, (top + bottom) / 2.0
    return np.asarray(
        [depth_m, (u - cx) * depth_m / fx, -(v - cy) * depth_m / fy],
        dtype=np.float64,
    )


def validate_intrinsics(intrinsics: list[float], image_shape: tuple[int, ...]) -> None:
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    height, width = image_shape[:2]
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("camera intrinsics must be finite")
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    if not (0 <= cx <= width and 0 <= cy <= height):
        raise ValueError("camera principal point must lie inside the frame")


def d44_predict(history: list[dict[str, Any]], target_timestamp_ns: int) -> np.ndarray:
    if len(history) != HISTORY_COUNT:
        raise ValueError("D44 requires seven observations")
    timestamps = np.asarray(
        [int(row["timestamp_ns"]) for row in history], dtype=np.float64
    ) / 1e9
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("track timestamps must increase")
    positions = np.stack(
        [
            relative_position(
                row["torso_roi_xyxy_px"],
                row["intrinsics_fx_fy_cx_cy"],
                float(row["depth_m"]),
            )
            for row in history
        ]
    )
    centered = timestamps - float(np.mean(timestamps))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise ValueError("degenerate history timestamps")
    velocity = centered @ positions / denominator
    prediction = np.mean(positions, axis=0) + velocity * (
        target_timestamp_ns / 1e9 - float(np.mean(timestamps))
    )
    return prediction


def append_contiguous_history(
    history: deque[dict[str, Any]], observation: dict[str, Any]
) -> None:
    if history and int(observation["frame_index"]) != int(history[-1]["frame_index"]) + 1:
        history.clear()
    history.append(observation)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previous_timestamp_by_sequence: dict[str, int] = {}
    for row in rows:
        frame = Path(row["frame_path"])
        if not frame.is_absolute():
            row["frame_path"] = str((path.parent / frame).resolve())
        sequence_id = str(row.get("sequence_id", "manifest"))
        timestamp_ns = int(row["timestamp_ns"])
        previous = previous_timestamp_by_sequence.get(sequence_id)
        if previous is not None and timestamp_ns <= previous:
            raise ValueError(f"timestamps must increase within sequence {sequence_id}")
        previous_timestamp_by_sequence[sequence_id] = timestamp_ns
    return rows


def yolo_person_detections(
    detector: Any,
    bgr: np.ndarray,
    yolo_device: str,
) -> list[dict[str, Any]]:
    result = detector.track(
        bgr,
        persist=True,
        classes=[0],
        conf=0.25,
        imgsz=640,
        device=yolo_device,
        tracker="bytetrack.yaml",
        verbose=False,
    )[0]
    if result.boxes is None or not len(result.boxes) or result.boxes.id is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    ids = result.boxes.id.detach().cpu().numpy().astype(int)
    detections = []
    for box, score, track_id in zip(boxes, scores, ids, strict=True):
        detections.append(
            {
                "track_id": int(track_id),
                "person_box_xyxy_px": box.tolist(),
                "confidence": float(score),
                "torso_roi_xyxy_px": torso_roi_from_person_box(box.tolist(), bgr.shape),
            }
        )
    return detections


def manifest_frames(
    path: Path,
    detector: Any | None = None,
    yolo_device: str = "0",
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    for row in load_manifest(path):
        bgr = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode {row['frame_path']}")
        frozen_detections = [
            {
                "track_id": 0,
                "person_box_xyxy_px": row["person_box_xyxy_px"],
                "confidence": float(row["person_confidence"]),
                "torso_roi_xyxy_px": row["torso_roi_xyxy_px"],
            }
        ]
        detector_started = time.perf_counter()
        detections = (
            yolo_person_detections(detector, bgr, yolo_device)
            if detector is not None
            else frozen_detections
        )
        detector_latency_ms = (
            (time.perf_counter() - detector_started) * 1000.0
            if detector is not None
            else None
        )
        yield bgr, {
            "frame_index": int(row["frame_index"]),
            "sequence_id": str(row.get("sequence_id", "manifest")),
            "timestamp_ns": int(row["timestamp_ns"]),
            "detections": detections,
            "detector_latency_ms": detector_latency_ms,
            "intrinsics_fx_fy_cx_cy": row["intrinsics_fx_fy_cx_cy"],
            "source": "manifest_redetect" if detector is not None else "manifest_replay",
        }


def camera_frames(
    camera_index: int,
    rectifier: FrameRectifier,
    detector: Any,
    yolo_device: str,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, rectifier.calibration.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.calibration.height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera index {camera_index}")
    frame_index = 0
    try:
        while True:
            ok, raw_bgr = capture.read()
            timestamp_ns = time.monotonic_ns()
            if not ok:
                raise RuntimeError("camera read failed")
            bgr, valid_mask = rectifier.rectify(raw_bgr)
            detector_started = time.perf_counter()
            detections = yolo_person_detections(detector, bgr, yolo_device)
            detector_latency_ms = (time.perf_counter() - detector_started) * 1000.0
            yield bgr, {
                "frame_index": frame_index,
                "sequence_id": f"camera:{camera_index}",
                "timestamp_ns": timestamp_ns,
                "detections": detections,
                "detector_latency_ms": detector_latency_ms,
                "intrinsics_fx_fy_cx_cy": rectifier.calibration.intrinsics,
                "rectification_valid_mask": valid_mask,
                "calibration_source_id": rectifier.calibration.source_id,
                "rectification_applied": rectifier.calibration.rectification_required,
                "source": f"camera:{camera_index}",
            }
            frame_index += 1
    finally:
        capture.release()


def video_frames(
    path: Path,
    rectifier: FrameRectifier,
    detector: Any,
    yolo_device: str,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid FPS: {fps}")
    sequence_id = f"video:{path.resolve()}"
    frame_index = 0
    previous_timestamp_ns = -1
    try:
        while True:
            ok, raw_bgr = capture.read()
            if not ok:
                break
            bgr, valid_mask = rectifier.rectify(raw_bgr)
            position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_ns = round(position_ms * 1e6)
            if timestamp_ns <= previous_timestamp_ns:
                timestamp_ns = round(frame_index / fps * 1e9)
            if timestamp_ns <= previous_timestamp_ns:
                raise RuntimeError("video timestamps are not strictly increasing")
            previous_timestamp_ns = timestamp_ns
            detector_started = time.perf_counter()
            detections = yolo_person_detections(detector, bgr, yolo_device)
            detector_latency_ms = (time.perf_counter() - detector_started) * 1000.0
            yield bgr, {
                "frame_index": frame_index,
                "sequence_id": sequence_id,
                "timestamp_ns": timestamp_ns,
                "detections": detections,
                "detector_latency_ms": detector_latency_ms,
                "intrinsics_fx_fy_cx_cy": rectifier.calibration.intrinsics,
                "rectification_valid_mask": valid_mask,
                "calibration_source_id": rectifier.calibration.source_id,
                "rectification_applied": rectifier.calibration.rectification_required,
                "capture_fps": fps,
                "source": sequence_id,
            }
            frame_index += 1
    finally:
        capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--camera-index", type=int)
    source.add_argument("--video", type=Path)
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument(
        "--calibration-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="pixel dimensions for which the supplied intrinsics were calibrated",
    )
    parser.add_argument("--yolo-weights", type=Path)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument(
        "--redetect-manifest",
        action="store_true",
        help="rerun YOLO+ByteTrack on manifest frames instead of using frozen boxes",
    )
    parser.add_argument("--metric3d-repo", type=Path, required=True)
    parser.add_argument("--metric3d-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--metric3d-precision",
        choices=("fp32", "tf32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_mode = args.camera_index is not None or args.video is not None
    has_cli_calibration = args.intrinsics is not None or args.calibration_size is not None
    if not capture_mode and (args.calibration_json is not None or has_cli_calibration):
        raise ValueError("manifest mode uses per-row calibration and rejects camera calibration flags")
    if args.calibration_json is not None and has_cli_calibration:
        raise ValueError(
            "use either --calibration-json or --intrinsics/--calibration-size, not both"
        )
    if capture_mode and args.calibration_json is None and (
        args.intrinsics is None or args.calibration_size is None
    ):
        raise ValueError(
            "camera/video mode requires --calibration-json or both --intrinsics and --calibration-size"
        )
    if capture_mode and args.yolo_weights is None:
        raise ValueError("camera/video mode requires --yolo-weights")
    if args.redetect_manifest and args.manifest is None:
        raise ValueError("--redetect-manifest requires --manifest")
    if args.redetect_manifest and args.yolo_weights is None:
        raise ValueError("--redetect-manifest requires --yolo-weights")
    detector = None
    if args.camera_index is not None or args.video is not None or args.redetect_manifest:
        from ultralytics import YOLO

        detector = YOLO(str(args.yolo_weights), task="detect")
    rectifier = None
    if capture_mode:
        calibration = (
            load_calibration(args.calibration_json)
            if args.calibration_json is not None
            else pinhole_calibration(list(args.intrinsics), list(args.calibration_size))
        )
        rectifier = FrameRectifier(calibration)
    source = Metric3DPytorchSource(
        args.metric3d_repo,
        args.metric3d_checkpoint,
        args.device,
        args.metric3d_precision,
    )
    if args.manifest is not None:
        frames = manifest_frames(args.manifest, detector, args.yolo_device)
    elif args.video is not None:
        frames = video_frames(
            args.video,
            rectifier,
            detector,
            args.yolo_device,
        )
    else:
        frames = camera_frames(
            args.camera_index,
            rectifier,
            detector,
            args.yolo_device,
        )
    histories: dict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=HISTORY_COUNT)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for bgr, packet in frames:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            height, width = bgr.shape[:2]
            validate_intrinsics(packet["intrinsics_fx_fy_cx_cy"], bgr.shape)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            depth, _metadata = source.infer(
                rgb,
                {"intrinsics_fx_fy_cx_cy": packet["intrinsics_fx_fy_cx_cy"]},
            )
            rectification_valid_mask = packet.get("rectification_valid_mask")
            if rectification_valid_mask is not None:
                if rectification_valid_mask.shape != depth.shape:
                    raise ValueError("rectification mask and depth shape differ")
                depth = np.asarray(depth).copy()
                depth[~rectification_valid_mask] = np.nan
            metric_depth_latency_ms = (time.perf_counter() - started) * 1000.0
            tracks = []
            for detection in packet["detections"]:
                roi = tuple(int(value) for value in detection["torso_roi_xyxy_px"])
                torso_depth, valid_pixels, valid_fraction = robust_roi_median(depth, roi)
                if torso_depth is None:
                    continue
                track_id = int(detection["track_id"])
                history_key = (str(packet["sequence_id"]), track_id)
                observation = {
                    "frame_index": int(packet["frame_index"]),
                    "timestamp_ns": int(packet["timestamp_ns"]),
                    "depth_m": float(torso_depth),
                    "torso_roi_xyxy_px": list(roi),
                    "intrinsics_fx_fy_cx_cy": list(packet["intrinsics_fx_fy_cx_cy"]),
                }
                append_contiguous_history(histories[history_key], observation)
                current = relative_position(
                    observation["torso_roi_xyxy_px"],
                    observation["intrinsics_fx_fy_cx_cy"],
                    observation["depth_m"],
                )
                target_ns = int(packet["timestamp_ns"] + FUTURE_SECONDS * 1e9)
                forecast = (
                    d44_predict(list(histories[history_key]), target_ns)
                    if len(histories[history_key]) == HISTORY_COUNT
                    else None
                )
                tracks.append(
                    {
                        **detection,
                        "depth_m": float(torso_depth),
                        "relative_position_m": current.tolist(),
                        "d44_future_timestamp_ns": target_ns if forecast is not None else None,
                        "d44_future_relative_position_m": forecast.tolist() if forecast is not None else None,
                        "history_count": len(histories[history_key]),
                        "roi_valid_pixels_after_trim": valid_pixels,
                        "roi_valid_fraction_before_trim": valid_fraction,
                    }
                )
            record = {
                "schema": "hftf_external_rgb_metric_track_sidecar_r0",
                "frame_index": int(packet["frame_index"]),
                "sequence_id": str(packet["sequence_id"]),
                "timestamp_ns": int(packet["timestamp_ns"]),
                "frame_width_px": width,
                "frame_height_px": height,
                "source": packet["source"],
                "calibration_source_id": packet.get(
                    "calibration_source_id", "manifest:per-row-pinhole"
                ),
                "rectification_applied": bool(
                    packet.get("rectification_applied", False)
                ),
                "rectification_valid_fraction": (
                    finite_ratio(rectification_valid_mask)
                    if rectification_valid_mask is not None
                    else 1.0
                ),
                "capture_fps": packet.get("capture_fps"),
                "metric_depth_model_id": source.model_id,
                "metric_depth_precision": args.metric3d_precision,
                "detector_latency_ms": packet.get("detector_latency_ms"),
                "metric_depth_latency_ms": metric_depth_latency_ms,
                "tracks": tracks,
                "claim_ceiling": "metric track and D44 research primitive only; no alert or safety decision",
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            processed += 1
            print(
                f"frame={record['frame_index']} tracks={len(tracks)} depth_ms={metric_depth_latency_ms:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
