#!/usr/bin/env python3
"""Materialize D36 seven-frame THOR-MAGNI detector replay input."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d31_thor_magni_full_resolution_measurement import (
    static_slots,
)
from extract_stage_c_d29_thor_magni_object_slots import (
    EXPECTED_YOLO_SHA256,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_SAMPLES,
)


SCHEMA = "blindassist_hftf_stage_c_d36_thor_magni_detector_input_v0"
EXPECTED_SAMPLES = 530
EXPECTED_SESSIONS = 19
HISTORY_FRAMES = 7
TARGET_REPLAY_HZ = 15.0
ANCHOR_PARITY_TOLERANCE = 1e-5
DEFAULT_D31_BOXES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d31-thor-magni-full-resolution-measurement-v0/"
    "full_resolution_boxes.npz"
)
DEFAULT_WEIGHTS = Path("artifacts.local/models/yolo11n.pt")
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d36-thor-magni-production-track-veto-event-v0/"
    "detections.tsv"
)

FIELDNAMES = (
    "sample_id",
    "source_session_id",
    "fold",
    "frame_ordinal",
    "source_scene_frame",
    "captured_at_ns",
    "frame_width",
    "frame_height",
    "detection_index",
    "confidence",
    "left",
    "top",
    "right",
    "bottom",
)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def source_records(
    samples_path: Path,
    d31_sample_ids: np.ndarray,
) -> list[dict[str, Any]]:
    records = load_jsonl(samples_path)
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("D36 D12 sample IDs are not unique")
    requested = [str(value) for value in d31_sample_ids.tolist()]
    if len(requested) != EXPECTED_SAMPLES or len(set(requested)) != len(
        requested
    ):
        raise ValueError("D36 D31 sample roster is invalid")
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise ValueError(f"D36 D31-to-D12 join failed: {missing[:3]}")
    selected = [by_id[sample_id] for sample_id in requested]
    if (
        len({str(record["source_session_id"]) for record in selected})
        != EXPECTED_SESSIONS
    ):
        raise ValueError("D36 source-session census drift")
    return selected


def window_frames(anchor: int, fps: float) -> tuple[int, list[int]]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("D36 invalid video FPS")
    step = max(1, int(round(fps / TARGET_REPLAY_HZ)))
    frames = [anchor - step * offset for offset in range(6, -1, -1)]
    if frames[0] < 1 or frames[-1] != anchor:
        raise ValueError("D36 seven-frame window is outside source video")
    return step, frames


def requested_segments(
    frames: list[int],
    maximum_gap: int = 2,
) -> list[tuple[int, int]]:
    ordered = sorted(set(frames))
    if not ordered:
        return []
    if maximum_gap < 1:
        raise ValueError("D36 segment gap must be positive")
    segments: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for frame in ordered[1:]:
        if frame - end <= maximum_gap:
            end = frame
        else:
            segments.append((start, end))
            start = frame
            end = frame
    segments.append((start, end))
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--d31-boxes", type=Path, default=DEFAULT_D31_BOXES)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("D36 batch size must be positive")
    if sha256(args.weights) != EXPECTED_YOLO_SHA256:
        raise ValueError("D36 YOLO weight SHA mismatch")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("D36 detector input requires CUDA")
    if ultralytics.__version__ != "8.4.102":
        raise ValueError("D36 Ultralytics version mismatch")

    with np.load(args.d31_boxes, allow_pickle=False) as d31:
        d31_ids = d31["sample_ids"].copy()
        d31_slots = d31["slots"].copy()
        d31_mask = d31["mask"].copy()
        d31_raw_count = d31["raw_detection_count"].copy()
    records = source_records(args.samples, d31_ids)
    sample_index_by_id = {
        str(sample_id): index for index, sample_id in enumerate(d31_ids)
    }

    requests: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    video_hashes: dict[str, str] = {}
    video_metadata: dict[str, dict[str, Any]] = {}
    sample_frames: list[list[int]] = [[] for _ in records]
    sample_timestamps: list[list[int]] = [[] for _ in records]
    for sample_index, record in enumerate(records):
        video_path = Path(str(record["video_path"])).resolve()
        video_text = str(video_path)
        expected_hash = str(record["video_sha256"])
        previous_hash = video_hashes.get(video_text)
        if previous_hash is not None and previous_hash != expected_hash:
            raise ValueError("D36 video hash declaration mismatch")
        video_hashes[video_text] = expected_hash
        if video_text not in video_metadata:
            capture = cv2.VideoCapture(video_text)
            if not capture.isOpened():
                raise OSError(f"D36 cannot open video: {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            video_metadata[video_text] = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }
        metadata = video_metadata[video_text]
        anchor = int(record["anchor_scene_frame"])
        step, frames = window_frames(anchor, float(metadata["fps"]))
        if anchor > int(metadata["frame_count"]):
            raise ValueError("D36 anchor exceeds video frame count")
        timestamps = [
            int(
                round(
                    (frame - 1)
                    / float(metadata["fps"])
                    * 1_000_000_000.0
                )
            )
            for frame in frames
        ]
        if any(
            right <= left
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ValueError("D36 source timestamps are not increasing")
        sample_frames[sample_index] = frames
        sample_timestamps[sample_index] = timestamps
        for ordinal, frame in enumerate(frames):
            requests[video_text][frame].append((sample_index, ordinal))
        record["_d36_frame_step"] = step

    model = YOLO(str(args.weights))
    detections: dict[tuple[str, int], np.ndarray] = {}
    anchor_count_mismatches = 0
    anchor_mask_mismatches = 0
    maximum_anchor_slot_error = 0.0
    decoded_receipts: list[dict[str, Any]] = []

    def infer_batch(
        video_text: str,
        frame_numbers: list[int],
        images: list[np.ndarray],
    ) -> None:
        nonlocal anchor_count_mismatches
        nonlocal anchor_mask_mismatches
        nonlocal maximum_anchor_slot_error
        if not images:
            return
        predictions = model.predict(
            images,
            imgsz=640,
            conf=0.10,
            iou=0.50,
            classes=[0],
            max_det=30,
            augment=False,
            device=0,
            batch=args.batch_size,
            verbose=False,
        )
        if len(predictions) != len(images):
            raise RuntimeError("D36 YOLO batch length mismatch")
        for frame_number, prediction in zip(
            frame_numbers,
            predictions,
            strict=True,
        ):
            pixel_boxes = np.empty((0, 5), dtype=np.float32)
            normalized_boxes = np.empty((0, 5), dtype=np.float32)
            if prediction.boxes is not None and len(prediction.boxes):
                confidence = (
                    prediction.boxes.conf.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, 1)
                )
                pixel_boxes = np.concatenate(
                    (
                        prediction.boxes.xyxy.detach().cpu().numpy(),
                        confidence,
                    ),
                    axis=1,
                ).astype(np.float32)
                normalized_boxes = np.concatenate(
                    (
                        prediction.boxes.xyxyn.detach().cpu().numpy(),
                        confidence,
                    ),
                    axis=1,
                ).astype(np.float32)
            detections[(video_text, frame_number)] = pixel_boxes
            slots, mask = static_slots(normalized_boxes)
            for sample_index, ordinal in requests[video_text][frame_number]:
                if ordinal != HISTORY_FRAMES - 1:
                    continue
                d31_index = sample_index_by_id[
                    str(records[sample_index]["sample_id"])
                ]
                if len(pixel_boxes) != int(d31_raw_count[d31_index]):
                    anchor_count_mismatches += 1
                if not np.array_equal(mask, d31_mask[d31_index]):
                    anchor_mask_mismatches += 1
                maximum_anchor_slot_error = max(
                    maximum_anchor_slot_error,
                    float(
                        np.max(
                            np.abs(
                                slots.astype(np.float64)
                                - d31_slots[d31_index].astype(np.float64)
                            )
                        )
                    ),
                )

    for video_index, video_text in enumerate(sorted(requests)):
        video_path = Path(video_text)
        actual_hash = sha256(video_path)
        if actual_hash != video_hashes[video_text]:
            raise ValueError(f"D36 video hash mismatch: {video_path}")
        capture = cv2.VideoCapture(video_text)
        if not capture.isOpened():
            raise OSError(f"D36 cannot open video: {video_path}")
        requested = requests[video_text]
        frame_number = 0
        requested_seen: set[int] = set()
        batch_frames: list[int] = []
        batch_images: list[np.ndarray] = []
        try:
            for segment_start, segment_end in requested_segments(
                list(requested)
            ):
                if not capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    float(segment_start - 1),
                ):
                    raise OSError("D36 video seek failed")
                frame_number = segment_start - 1
                while frame_number < segment_end:
                    ok, frame = capture.read()
                    if not ok:
                        raise OSError("D36 requested source frame decode failed")
                    frame_number += 1
                    if frame_number not in requested:
                        continue
                    if frame_number in requested_seen:
                        raise ValueError("D36 requested frame decoded twice")
                    requested_seen.add(frame_number)
                    batch_frames.append(frame_number)
                    batch_images.append(frame)
                    if len(batch_images) >= args.batch_size:
                        infer_batch(video_text, batch_frames, batch_images)
                        batch_frames.clear()
                        batch_images.clear()
            infer_batch(video_text, batch_frames, batch_images)
        finally:
            capture.release()
        if len(requested_seen) != len(requested):
            raise ValueError("D36 requested source frame missing")
        decoded_receipts.append(
            {
                "video_path": video_text,
                "video_sha256": actual_hash,
                "fps": float(video_metadata[video_text]["fps"]),
                "source_frame_count": int(
                    video_metadata[video_text]["frame_count"]
                ),
                "requested_unique_frames": len(requested_seen),
                "seek_segments": len(
                    requested_segments(list(requested))
                ),
            }
        )
        print(
            json.dumps(
                {
                    "video": video_index + 1,
                    "videos": len(requests),
                    "requested_unique_frames": len(requested_seen),
                }
            ),
            flush=True,
        )

    if anchor_count_mismatches != 0 or anchor_mask_mismatches != 0:
        raise ValueError("D36 anchor detector census drift")
    if maximum_anchor_slot_error > ANCHOR_PARITY_TOLERANCE:
        raise ValueError(
            "D36 anchor selected-box parity drift: "
            f"{maximum_anchor_slot_error}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    row_count = 0
    detection_count = 0
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for sample_index, record in enumerate(records):
            video_text = str(Path(str(record["video_path"])).resolve())
            metadata = video_metadata[video_text]
            for ordinal, (frame, timestamp_ns) in enumerate(
                zip(
                    sample_frames[sample_index],
                    sample_timestamps[sample_index],
                    strict=True,
                )
            ):
                boxes = detections[(video_text, frame)]
                rows = boxes if len(boxes) else [None]
                for detection_index, box in enumerate(rows):
                    output_row: dict[str, Any] = {
                        "sample_id": str(record["sample_id"]),
                        "source_session_id": str(
                            record["source_session_id"]
                        ),
                        "fold": int(record["fold"]),
                        "frame_ordinal": ordinal,
                        "source_scene_frame": frame,
                        "captured_at_ns": timestamp_ns,
                        "frame_width": int(metadata["width"]),
                        "frame_height": int(metadata["height"]),
                        "detection_index": (
                            detection_index if box is not None else -1
                        ),
                        "confidence": "",
                        "left": "",
                        "top": "",
                        "right": "",
                        "bottom": "",
                    }
                    if box is not None:
                        output_row.update(
                            {
                                "confidence": repr(float(box[4])),
                                "left": repr(float(box[0])),
                                "top": repr(float(box[1])),
                                "right": repr(float(box[2])),
                                "bottom": repr(float(box[3])),
                            }
                        )
                        detection_count += 1
                    writer.writerow(output_row)
                    row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, args.output)

    receipt_path = args.output.with_name("producer_receipt.json")
    receipt = {
        "schema": SCHEMA,
        "sample_count": len(records),
        "session_count": len(
            {str(record["source_session_id"]) for record in records}
        ),
        "window_frames": HISTORY_FRAMES,
        "target_replay_hz": TARGET_REPLAY_HZ,
        "tsv_rows": row_count,
        "detection_rows": detection_count,
        "unique_requested_frames": len(detections),
        "anchor_count_mismatches": anchor_count_mismatches,
        "anchor_mask_mismatches": anchor_mask_mismatches,
        "maximum_anchor_slot_error": maximum_anchor_slot_error,
        "anchor_parity_tolerance": ANCHOR_PARITY_TOLERANCE,
        "samples_sha256": sha256(args.samples),
        "d31_boxes_sha256": sha256(args.d31_boxes),
        "weights_sha256": sha256(args.weights),
        "detections_tsv_sha256": sha256(args.output),
        "detector": {
            "ultralytics": ultralytics.__version__,
            "imgsz": 640,
            "confidence": 0.10,
            "nms_iou": 0.50,
            "classes": [0],
            "max_det": 30,
        },
        "source_only": True,
        "future_truth_consumed": False,
        "decoded_videos": decoded_receipts,
    }
    write_text_atomic(
        receipt_path,
        json.dumps(
            receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text_atomic(
        receipt_path.with_suffix(receipt_path.suffix + ".sha256"),
        f"{sha256(receipt_path)}  {receipt_path.name}\n",
    )
    write_text_atomic(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{sha256(args.output)}  {args.output.name}\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
