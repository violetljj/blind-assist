#!/usr/bin/env python3
"""Run candidate-only RGB clearance with causal sparse metric-scale anchors."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_metric3d_clearance_field_a0 import HORIZONS_M, clearance_field
from external_camera_calibration import (
    FrameRectifier,
    finite_ratio,
    load_calibration,
    pinhole_calibration,
)
from metric_scale_anchor import MetricScaleTracker
from produce_external_rgb_metric_depth_observations import DepthAnythingV2MetricSource
from sparse_scale_anchor_io import ScaleAnchorStream, load_scale_anchors

SCHEMA = "hftf_external_rgb_sparse_scale_clearance_sidecar_r0"


def load_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "sequence_id",
                "frame_path",
                "timestamp_ns",
                "intrinsics_fx_fy_cx_cy",
            } - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            frame = Path(str(row["frame_path"]))
            if not frame.is_absolute():
                frame = (path.parent / frame).resolve()
            row["frame_path"] = str(frame)
            unique.setdefault(str(frame), row)
    rows = sorted(
        unique.values(),
        key=lambda row: (str(row["sequence_id"]), int(row["timestamp_ns"])),
    )
    previous: dict[str, int] = {}
    for row in rows:
        sequence = str(row["sequence_id"])
        timestamp = int(row["timestamp_ns"])
        if previous.get(sequence, -1) >= timestamp:
            raise ValueError(f"timestamps must increase within {sequence}")
        previous[sequence] = timestamp
    if not rows:
        raise ValueError("manifests contain no frames")
    return rows


def manifest_frames(paths: list[Path]) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    for index, row in enumerate(load_manifest_rows(paths)):
        bgr = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode {row['frame_path']}")
        yield bgr, {
            "frame_index": int(row.get("frame_index", index)),
            "sequence_id": str(row["sequence_id"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "intrinsics_fx_fy_cx_cy": list(row["intrinsics_fx_fy_cx_cy"]),
            "source": "manifest_replay",
        }


def capture_frames(
    camera_index: int | None,
    video: Path | None,
    sequence_id: str,
    rectifier: FrameRectifier,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    capture = cv2.VideoCapture(str(video) if video is not None else int(camera_index))
    if video is None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, rectifier.calibration.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.calibration.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError("cannot open RGB source")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if video is not None else None
    if video is not None and (not math.isfinite(float(fps)) or float(fps) <= 0):
        capture.release()
        raise RuntimeError("video FPS is invalid")
    previous_timestamp = -1
    frame_index = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                if video is not None:
                    break
                raise RuntimeError("camera read failed")
            timestamp = (
                round(float(capture.get(cv2.CAP_PROP_POS_MSEC)) * 1e6)
                if video is not None
                else time.monotonic_ns()
            )
            if video is not None and timestamp <= previous_timestamp:
                timestamp = round(frame_index / float(fps) * 1e9)
            if timestamp <= previous_timestamp:
                raise RuntimeError("capture timestamps are not strictly increasing")
            previous_timestamp = timestamp
            bgr, valid_mask = rectifier.rectify(raw)
            yield bgr, {
                "frame_index": frame_index,
                "sequence_id": sequence_id,
                "timestamp_ns": timestamp,
                "intrinsics_fx_fy_cx_cy": rectifier.calibration.intrinsics,
                "rectification_valid_mask": valid_mask,
                "source": f"video:{video.resolve()}" if video is not None else f"camera:{camera_index}",
                "capture_fps": fps,
            }
            frame_index += 1
    finally:
        capture.release()


def calibrated_field(
    raw_field: dict[str, Any],
    tracker: MetricScaleTracker,
    timestamp_ns: int,
) -> dict[str, Any]:
    if raw_field.get("status") != "VALID":
        return {"status": "UNKNOWN_RAW_CLEARANCE"}
    clearances = {
        band: raw_field["bands"][band].get("clearance_m")
        for band in ("left", "center", "right")
    }
    scaled = tracker.apply(timestamp_ns, clearances)
    if scaled["status"] != "VALID":
        return scaled
    bands = {}
    for band, value in scaled["bands_m"].items():
        bands[band] = {
            "clearance_m": value,
            "occupied_by_horizon": {
                str(horizon): value is not None and value <= horizon
                for horizon in HORIZONS_M
            },
        }
    return {**scaled, "bands": bands}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, nargs="+")
    source.add_argument("--camera-index", type=int)
    source.add_argument("--video", type=Path)
    parser.add_argument("--sequence-id")
    parser.add_argument("--intrinsics", type=float, nargs=4)
    parser.add_argument("--calibration-size", type=int, nargs=2)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--scale-anchor-jsonl", type=Path, required=True)
    parser.add_argument("--max-anchor-age-ms", type=float, required=True)
    parser.add_argument("--depth-anything-repo", type=Path, required=True)
    parser.add_argument("--depth-anything-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-anything-input-size", type=int, default=392)
    parser.add_argument("--depth-anything-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_mode = args.camera_index is not None or args.video is not None
    if not math.isfinite(args.max_anchor_age_ms) or args.max_anchor_age_ms <= 0:
        raise ValueError("--max-anchor-age-ms must be finite and positive")
    if capture_mode and not args.sequence_id:
        raise ValueError("camera/video mode requires --sequence-id for anchor binding")
    if not capture_mode and (
        args.sequence_id
        or args.intrinsics
        or args.calibration_size
        or args.calibration_json
    ):
        raise ValueError("manifest mode uses per-row sequence and calibration")
    if capture_mode and args.calibration_json is None and (
        args.intrinsics is None or args.calibration_size is None
    ):
        raise ValueError("camera/video mode requires a complete calibration")
    if args.calibration_json is not None and (
        args.intrinsics is not None or args.calibration_size is not None
    ):
        raise ValueError("choose JSON calibration or CLI calibration, not both")

    anchor_stream = ScaleAnchorStream(load_scale_anchors(args.scale_anchor_jsonl))
    trackers: dict[str, MetricScaleTracker] = {}
    max_age_ns = round(args.max_anchor_age_ms * 1e6)
    source = DepthAnythingV2MetricSource(
        args.depth_anything_repo,
        args.depth_anything_checkpoint,
        args.device,
        args.depth_anything_input_size,
        args.depth_anything_precision,
    )
    if args.manifest is not None:
        frames = manifest_frames(args.manifest)
    else:
        calibration = (
            load_calibration(args.calibration_json)
            if args.calibration_json is not None
            else pinhole_calibration(list(args.intrinsics), list(args.calibration_size))
        )
        frames = capture_frames(
            args.camera_index,
            args.video,
            args.sequence_id,
            FrameRectifier(calibration),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for bgr, packet in frames:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            sequence = str(packet["sequence_id"])
            tracker = trackers.setdefault(sequence, MetricScaleTracker(max_age_ns))
            for anchor in anchor_stream.take_available(sequence, int(packet["timestamp_ns"])):
                tracker.update(anchor)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_started = time.perf_counter()
            depth, metadata = source.infer(
                rgb,
                {"intrinsics_fx_fy_cx_cy": packet["intrinsics_fx_fy_cx_cy"]},
            )
            valid_mask = packet.get("rectification_valid_mask")
            if valid_mask is not None:
                depth = np.asarray(depth).copy()
                depth[~valid_mask] = np.nan
            depth_ms = (time.perf_counter() - depth_started) * 1000.0
            geometry_started = time.perf_counter()
            raw_field = clearance_field(
                depth,
                np.asarray(
                    [
                        [packet["intrinsics_fx_fy_cx_cy"][0], 0, packet["intrinsics_fx_fy_cx_cy"][2]],
                        [0, packet["intrinsics_fx_fy_cx_cy"][1], packet["intrinsics_fx_fy_cx_cy"][3]],
                        [0, 0, 1],
                    ],
                    dtype=np.float64,
                ),
            )
            scaled_field = calibrated_field(raw_field, tracker, int(packet["timestamp_ns"]))
            geometry_ms = (time.perf_counter() - geometry_started) * 1000.0
            record = {
                "schema": SCHEMA,
                "sequence_id": sequence,
                "frame_index": int(packet["frame_index"]),
                "timestamp_ns": int(packet["timestamp_ns"]),
                "source": packet["source"],
                "model_id": source.model_id,
                "model_metadata": metadata,
                "depth_latency_ms": depth_ms,
                "geometry_and_scale_latency_ms": geometry_ms,
                "rectification_valid_fraction": finite_ratio(valid_mask) if valid_mask is not None else 1.0,
                "raw_clearance": raw_field,
                "scaled_clearance": scaled_field,
                "claim_ceiling": "candidate-only clearance sidecar; no alert or safety decision",
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            processed += 1
            print(
                f"sequence={sequence} frame={packet['frame_index']} status={scaled_field['status']} depth_ms={depth_ms:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
