#!/usr/bin/env python3
"""Produce source-only YOLO11n + ByteTrack tracks for D33 JRDB frames."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_stage_c_d32_jrdb_causal_track_future_range import (
    REPO_ROOT,
    sha256,
)
from extract_stage_c_d29_thor_magni_object_slots import (
    EXPECTED_YOLO_SHA256,
)
from materialize_stage_c_d33_jrdb_stitched_rgb import (
    DEFAULT_RECEIPT as DEFAULT_IMAGE_RECEIPT,
)


SCHEMA = "blindassist_hftf_stage_c_d33_jrdb_detector_tracks_v0"
DEFAULT_WEIGHTS = REPO_ROOT / "artifacts.local/models/yolo11n.pt"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts.local/evidence/hftf/"
    "stage-c-d33-jrdb-detector-track-future-range-v0"
)
DEFAULT_TRACKS = DEFAULT_OUTPUT_ROOT / "tracks.jsonl"
DEFAULT_RECEIPT = DEFAULT_OUTPUT_ROOT / "producer_receipt.json"

IMAGE_WIDTH = 3760
IMAGE_HEIGHT = 480
TILE_WIDTH = 960
TILE_STARTS = (0, 700, 1400, 2100, 2800)
DETECTOR_CONFIDENCE = 0.10
DETECTOR_NMS_IOU = 0.50
DETECTOR_MAX_DET = 50
INFERENCE_SIZE = 640

TRACKER_CONFIG = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.80,
    "fuse_score": True,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nms(
    boxes_xyxy_conf: np.ndarray,
    iou_threshold: float = DETECTOR_NMS_IOU,
) -> np.ndarray:
    if len(boxes_xyxy_conf) == 0:
        return np.empty((0, 5), dtype=np.float32)
    boxes = np.asarray(boxes_xyxy_conf, dtype=np.float32)
    order = np.argsort(-boxes[:, 4], kind="stable")
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        left = np.maximum(boxes[current, 0], boxes[rest, 0])
        top = np.maximum(boxes[current, 1], boxes[rest, 1])
        right = np.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(
            0.0,
            bottom - top,
        )
        current_area = max(
            0.0,
            float(boxes[current, 2] - boxes[current, 0]),
        ) * max(0.0, float(boxes[current, 3] - boxes[current, 1]))
        rest_area = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * (
            np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        )
        union = current_area + rest_area - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= iou_threshold]
    return boxes[np.asarray(kept, dtype=np.int64)]


def tiled_detections(
    predictions: list[Any],
) -> np.ndarray:
    if len(predictions) != len(TILE_STARTS):
        raise ValueError("D33 tile prediction count mismatch")
    detections: list[np.ndarray] = []
    for start, prediction in zip(TILE_STARTS, predictions, strict=True):
        if prediction.boxes is None or not len(prediction.boxes):
            continue
        boxes = prediction.boxes.xyxy.detach().cpu().numpy()
        confidence = (
            prediction.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        )
        boxes[:, 0] += start
        boxes[:, 2] += start
        boxes[:, 0] = np.clip(boxes[:, 0], 0, IMAGE_WIDTH)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, IMAGE_WIDTH)
        detections.append(
            np.concatenate((boxes, confidence), axis=1).astype(
                np.float32
            )
        )
    if not detections:
        return np.empty((0, 5), dtype=np.float32)
    merged = nms(np.concatenate(detections, axis=0))
    return merged[:DETECTOR_MAX_DET]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(partial, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def produce(
    image_receipt_path: Path,
    weights_path: Path,
    batch_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    if not torch.cuda.is_available():
        raise RuntimeError("D33 detector requires CUDA")
    if ultralytics.__version__ != "8.4.102":
        raise ValueError("D33 Ultralytics version mismatch")
    if sha256(weights_path) != EXPECTED_YOLO_SHA256:
        raise ValueError("D33 YOLO weight SHA mismatch")
    image_receipt = load_json(image_receipt_path)
    if (
        image_receipt.get("status") != "COMPLETE"
        or int(image_receipt.get("frame_count", 0)) != 480
    ):
        raise ValueError("D33 image receipt is incomplete")
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_receipt["records"]:
        path = Path(row["path"])
        if not path.exists() or sha256(path) != row["sha256"]:
            raise ValueError(f"D33 source image drift: {path}")
        by_sequence[str(row["sequence"])].append(row)
    model = YOLO(str(weights_path))
    tracker_args = IterableSimpleNamespace(
        tracker_type="bytetrack",
        **TRACKER_CONFIG,
    )

    class ResultsLike:
        def __init__(
            self,
            xywh: np.ndarray,
            confidence: np.ndarray,
        ) -> None:
            self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
            self.conf = np.asarray(confidence, dtype=np.float32)
            self.cls = np.zeros(len(self.conf), dtype=np.float32)

        def __len__(self) -> int:
            return len(self.conf)

        def __getitem__(self, item: Any) -> "ResultsLike":
            return ResultsLike(self.xywh[item], self.conf[item])

    output_rows: list[dict[str, Any]] = []
    sequence_summaries: dict[str, Any] = {}
    for sequence, sequence_rows in sorted(by_sequence.items()):
        ordered = sorted(
            sequence_rows,
            key=lambda row: int(row["frame_stem"]),
        )
        if len(ordered) != 120:
            raise ValueError(f"D33 sequence is not 120 frames: {sequence}")
        tracker = BYTETracker(tracker_args)
        raw_detection_count = 0
        tracked_count = 0
        track_ids: set[int] = set()
        for batch_start in range(0, len(ordered), batch_frames):
            batch = ordered[batch_start : batch_start + batch_frames]
            crops: list[np.ndarray] = []
            for row in batch:
                image = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"D33 cannot decode image: {row['path']}")
                if image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    raise ValueError(
                        f"D33 image shape drift: {row['path']}/"
                        f"{image.shape[:2]}"
                    )
                crops.extend(
                    image[:, start : start + TILE_WIDTH]
                    for start in TILE_STARTS
                )
            predictions = model.predict(
                crops,
                imgsz=INFERENCE_SIZE,
                conf=DETECTOR_CONFIDENCE,
                iou=DETECTOR_NMS_IOU,
                classes=[0],
                max_det=DETECTOR_MAX_DET,
                augment=False,
                device=0,
                batch=len(crops),
                verbose=False,
            )
            if len(predictions) != len(crops):
                raise RuntimeError("D33 YOLO batch length mismatch")
            for index, row in enumerate(batch):
                frame_predictions = predictions[
                    index * len(TILE_STARTS) : (index + 1)
                    * len(TILE_STARTS)
                ]
                detections = tiled_detections(frame_predictions)
                raw_detection_count += len(detections)
                xywh = np.empty((len(detections), 4), dtype=np.float32)
                if len(detections):
                    xywh[:, 0] = (
                        detections[:, 0] + detections[:, 2]
                    ) * 0.5
                    xywh[:, 1] = (
                        detections[:, 1] + detections[:, 3]
                    ) * 0.5
                    xywh[:, 2] = detections[:, 2] - detections[:, 0]
                    xywh[:, 3] = detections[:, 3] - detections[:, 1]
                tracked = tracker.update(
                    ResultsLike(xywh, detections[:, 4])
                )
                for tracked_row in tracked:
                    candidate_index = int(round(float(tracked_row[7])))
                    if not 0 <= candidate_index < len(detections):
                        raise ValueError(
                            "D33 ByteTrack candidate index invalid"
                        )
                    track_id = int(round(float(tracked_row[4])))
                    box = detections[candidate_index]
                    output_rows.append(
                        {
                            "sequence": sequence,
                            "frame_index": int(row["frame_stem"]),
                            "frame_stem": str(row["frame_stem"]),
                            "track_id": track_id,
                            "bbox_xyxy": [
                                float(value) for value in box[:4]
                            ],
                            "confidence": float(box[4]),
                            "image_sha256": str(row["sha256"]),
                        }
                    )
                    tracked_count += 1
                    track_ids.add(track_id)
        sequence_summaries[sequence] = {
            "frame_count": len(ordered),
            "raw_detection_count": raw_detection_count,
            "tracked_occurrence_count": tracked_count,
            "track_count": len(track_ids),
        }
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    **sequence_summaries[sequence],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output_rows.sort(
        key=lambda row: (
            row["sequence"],
            row["frame_index"],
            row["track_id"],
        )
    )
    receipt = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "source_only": True,
        "image_receipt_path": str(image_receipt_path.resolve()),
        "image_receipt_sha256": sha256(image_receipt_path),
        "weights_path": str(weights_path.resolve()),
        "weights_sha256": sha256(weights_path),
        "ultralytics_version": ultralytics.__version__,
        "detector": {
            "image_size": INFERENCE_SIZE,
            "confidence": DETECTOR_CONFIDENCE,
            "nms_iou": DETECTOR_NMS_IOU,
            "max_det": DETECTOR_MAX_DET,
            "tile_width": TILE_WIDTH,
            "tile_starts": list(TILE_STARTS),
        },
        "tracker": TRACKER_CONFIG,
        "frame_count": sum(
            row["frame_count"] for row in sequence_summaries.values()
        ),
        "track_occurrence_count": len(output_rows),
        "sequence_summaries": sequence_summaries,
    }
    return output_rows, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-receipt",
        type=Path,
        default=DEFAULT_IMAGE_RECEIPT,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    rows, receipt = produce(
        args.image_receipt,
        args.weights,
        args.batch_frames,
    )
    write_jsonl(args.tracks, rows)
    receipt["tracks_path"] = str(args.tracks.resolve())
    receipt["tracks_sha256"] = sha256(args.tracks)
    write_json(args.receipt, receipt)
    digest = sha256(args.receipt)
    args.receipt.with_suffix(args.receipt.suffix + ".sha256").write_text(
        f"{digest}  {args.receipt.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "frame_count": receipt["frame_count"],
                "track_occurrence_count": (
                    receipt["track_occurrence_count"]
                ),
                "tracks_sha256": receipt["tracks_sha256"],
                "receipt_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
