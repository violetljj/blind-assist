"""Extract the immutable full-video feature report required by r7.17.

The extractor samples the complete video at the contract's fixed one-second
schedule. It emits only deterministic channel values: frozen COCO corridor
occupancy, registered lower-corridor residual and prompt-free semantic risk
counts. It never sees review windows and never emits an alert/clear verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2

import public_video_dual_evidence_lifecycle_contract as prospective
import run_public_silver_frozen_feature_probe as common
import run_public_silver_motion_compensated_occupancy_probe as motion
import run_public_silver_object_trajectory_probe as trajectory
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_dual_evidence_feature_report_v1"


def sample_timestamps(duration_ms: int, interval_ms: int) -> list[int]:
    if duration_ms <= 0 or interval_ms <= 0:
        raise ValueError("duration and sample interval must be positive")
    return list(range(0, duration_ms, interval_ms))


def semantic_class_ids(names: dict[int, str], selected_groups: dict[str, list[str]]) -> list[int]:
    allowed = {name for values in selected_groups.values() for name in values}
    ids = sorted(index for index, name in names.items() if name in allowed)
    if not ids:
        raise ValueError("semantic model contains none of the frozen risk classes")
    return ids


def dynamic_occupancy_from_result(result: Any, names: dict[int, str]) -> float:
    if result.boxes is None:
        return 0.0
    height, width = result.orig_shape
    allowed = set().union(*trajectory.GROUP_CLASSES.values())
    values: list[float] = []
    for box, class_id, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        class_name = names[int(class_id)]
        if class_name not in allowed:
            continue
        row = trajectory.normalize_detection(
            class_name,
            float(score),
            [float(value) for value in box],
            width=width,
            height=height,
        )
        values.append(float(row["area"]) * float(row["corridor_overlap"]))
    return max(values, default=0.0)


def semantic_count_from_result(
    result: Any,
    names: dict[int, str],
    selected_groups: dict[str, list[str]],
) -> int:
    if result.boxes is None:
        return 0
    allowed = {name for values in selected_groups.values() for name in values}
    return sum(names[int(class_id)] in allowed for class_id in result.boxes.cls)


def video_metadata(path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video frame count or FPS metadata is invalid")
    duration_ms = int(math.ceil(frame_count * 1000.0 / fps))
    return duration_ms, frame_count, fps


def read_scheduled_frames(path: Path, timestamps: Sequence[int]) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {path}")
    frames = []
    try:
        for timestamp_ms in timestamps:
            if not capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms)):
                raise ValueError(f"video seek failed at {timestamp_ms} ms")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"video sample decode failed at {timestamp_ms} ms")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _process_batch(
    *,
    dynamic_model: Any,
    semantic_model: Any,
    frames: Sequence[Any],
    dynamic_contract: dict[str, Any],
    semantic_contract: dict[str, Any],
) -> tuple[list[float], list[int]]:
    dynamic_ids = trajectory.detector_class_ids(dynamic_model.names)
    semantic_ids = semantic_class_ids(semantic_model.names, semantic_contract["selected_groups"])
    dynamic_results = dynamic_model.predict(
        source=list(frames),
        imgsz=int(dynamic_contract["image_size"]),
        conf=float(dynamic_contract["confidence"]),
        iou=0.5,
        classes=dynamic_ids,
        agnostic_nms=True,
        device="cpu",
        verbose=False,
    )
    semantic_results = semantic_model.predict(
        source=list(frames),
        imgsz=int(semantic_contract["image_size"]),
        conf=float(semantic_contract["confidence"]),
        iou=0.5,
        classes=semantic_ids,
        agnostic_nms=True,
        device="cpu",
        verbose=False,
    )
    if len(dynamic_results) != len(frames) or len(semantic_results) != len(frames):
        raise RuntimeError("frozen model result count differs from scheduled frame count")
    occupancies = [
        dynamic_occupancy_from_result(result, dynamic_model.names)
        for result in dynamic_results
    ]
    semantic_counts = [
        semantic_count_from_result(
            result, semantic_model.names, semantic_contract["selected_groups"]
        )
        for result in semantic_results
    ]
    return occupancies, semantic_counts


def extract_samples(
    *,
    video: Path,
    timestamps: Sequence[int],
    dynamic_model: Any,
    semantic_model: Any,
    contract: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    feature_contract = contract["feature_contract"]
    dynamic_contract = feature_contract["dynamic_channel"]
    static_contract = feature_contract["static_channel"]
    semantic_contract = feature_contract["semantic_exit_channel"]
    samples: list[dict[str, Any]] = []
    previous_frame = None
    for start in range(0, len(timestamps), batch_size):
        batch_timestamps = list(timestamps[start:start + batch_size])
        frames = read_scheduled_frames(video, batch_timestamps)
        occupancies, semantic_counts = _process_batch(
            dynamic_model=dynamic_model,
            semantic_model=semantic_model,
            frames=frames,
            dynamic_contract=dynamic_contract,
            semantic_contract=semantic_contract,
        )
        for timestamp_ms, frame, occupancy, semantic_count in zip(
            batch_timestamps, frames, occupancies, semantic_counts
        ):
            residual = None
            reliable = False
            if previous_frame is not None:
                vector, summary = motion.frame_pair_descriptor(
                    previous_frame, frame, size=int(static_contract["motion_size"])
                )
                reliable = bool(summary["homography_success"])
                if reliable:
                    residual = float(vector[13])
            samples.append({
                "timestamp_ms": int(timestamp_ms),
                "dynamic_occupancy": float(occupancy),
                "static_residual": residual,
                "static_residual_reliable": reliable,
                "semantic_risk_count": int(semantic_count),
            })
            previous_frame = frame
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.video, args.contract, args.dynamic_weights,
        args.semantic_weights, args.cache_dir, args.output,
    ):
        mil.reject_independent_direction(path)
    contract, contract_attestation = prospective.load_contract(args.contract)
    video = args.video.resolve()
    dynamic_weights = args.dynamic_weights.resolve()
    semantic_weights = args.semantic_weights.resolve()
    if not video.is_file() or not dynamic_weights.is_file() or not semantic_weights.is_file():
        raise FileNotFoundError("video or frozen model weights are missing")
    video_sha = common.sha256_file(video)
    if args.video_sha256.lower() != video_sha:
        raise ValueError("source video SHA256 differs from the supplied immutable identity")
    feature_contract = contract["feature_contract"]
    if common.sha256_file(dynamic_weights) != feature_contract["dynamic_channel"]["weights_sha256"]:
        raise ValueError("dynamic weights differ from the frozen contract")
    if common.sha256_file(semantic_weights) != feature_contract["semantic_exit_channel"]["weights_sha256"]:
        raise ValueError("semantic weights differ from the frozen contract")
    if not isinstance(args.source_id, str) or not args.source_id.strip():
        raise ValueError("source_id is missing")

    duration_ms, frame_count, fps = video_metadata(video)
    interval_ms = int(feature_contract["sample_interval_ms"])
    timestamps = sample_timestamps(duration_ms, interval_ms)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.cache_dir.resolve())
    from ultralytics import YOLO
    dynamic_model = YOLO(str(dynamic_weights))
    semantic_model = YOLO(str(semantic_weights))
    samples = extract_samples(
        video=video,
        timestamps=timestamps,
        dynamic_model=dynamic_model,
        semantic_model=semantic_model,
        contract=contract,
        batch_size=args.batch_size,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prospective_contract": contract_attestation,
        "feature_generation": {
            "complete_video_processed": len(samples) == len(timestamps),
            "review_windows_known_during_feature_generation": False,
            "feature_values_immutable": True,
            "sample_interval_ms": interval_ms,
            "dynamic_weights_sha256": common.sha256_file(dynamic_weights),
            "semantic_weights_sha256": common.sha256_file(semantic_weights),
            "dynamic_image_size": feature_contract["dynamic_channel"]["image_size"],
            "dynamic_confidence": feature_contract["dynamic_channel"]["confidence"],
            "semantic_image_size": feature_contract["semantic_exit_channel"]["image_size"],
            "semantic_confidence": feature_contract["semantic_exit_channel"]["confidence"],
            "text_prompt_used": False,
            "source_masks_used": False,
            "review_labels_used": False,
        },
        "sources": [{
            "source_id": args.source_id.strip(),
            "video_path": str(video),
            "video_sha256": video_sha,
            "duration_ms": duration_ms,
            "frame_count": frame_count,
            "fps": fps,
            "scheduled_sample_count": len(timestamps),
            "samples": samples,
        }],
        "evidence_limit": "Frozen full-video proposal features only; no alert/clear truth, lifecycle verdict, training authorization, calibration, blind evaluation, Android runtime change, or production evidence.",
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output or sidecar: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-sha256", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dynamic-weights", type=Path, required=True)
    parser.add_argument("--semantic-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    source = report["sources"][0]
    print(json.dumps({
        "ok": True,
        "source_id": source["source_id"],
        "scheduled_sample_count": source["scheduled_sample_count"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
