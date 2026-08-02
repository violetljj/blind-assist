#!/usr/bin/env python3
"""Replicate D30 box/world measurement from full-resolution source frames."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d24_thor_magni_proximity_event_ablation import (
    infer_scene_column,
)
from evaluate_stage_c_d30_thor_magni_box_world_measurement import (
    ACCEPTED_X_ERROR,
    DISTANCE_CAP_M,
    HALF_FOV_DEGREES,
    MIN_SOURCE_PAIRS,
    assign_measurements,
    is_person_body,
    relative_bearing_degrees,
    summarize,
)
from extract_stage_c_d29_thor_magni_object_slots import (
    EXPECTED_YOLO_SHA256,
    FEATURE_COUNT,
    MAX_SLOTS,
    selected_boxes,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    read_scenario,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_SAMPLES,
)
from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DEFAULT_D8_SAMPLES,
    prepare_records,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d31_thor_magni_"
    "full_resolution_measurement_v0"
)
DEFAULT_WEIGHTS = Path("artifacts.local/models/yolo11n.pt")
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d31-thor-magni-full-resolution-measurement-v0/report.json"
)


def static_slots(
    boxes_xyxy_conf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = selected_boxes(boxes_xyxy_conf, MAX_SLOTS)
    slots = np.zeros((MAX_SLOTS, FEATURE_COUNT), dtype=np.float32)
    mask = np.zeros(MAX_SLOTS, dtype=bool)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2, confidence = (float(value) for value in box)
        width = x2 - x1
        height = y2 - y1
        slots[index, :6] = (
            2.0 * ((x1 + x2) * 0.5) - 1.0,
            y2,
            width,
            height,
            np.sqrt(width * height),
            confidence,
        )
        mask[index] = True
    return slots, mask


def write_npz_atomic(
    path: Path,
    **arrays: np.ndarray,
) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--d8-samples", type=Path, default=DEFAULT_D8_SAMPLES)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    boxes_path = args.output.with_name("full_resolution_boxes.npz")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    boxes_sidecar = boxes_path.with_suffix(boxes_path.suffix + ".sha256")
    if (
        args.output.exists()
        or sidecar.exists()
        or boxes_path.exists()
        or boxes_sidecar.exists()
    ):
        raise FileExistsError("D31 outputs are non-overwriting")
    if sha256(args.weights) != EXPECTED_YOLO_SHA256:
        raise ValueError("D31 YOLO weight SHA mismatch")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("D31 detection requires CUDA")
    if ultralytics.__version__ != "8.4.102":
        raise ValueError("D31 Ultralytics version mismatch")
    d12 = load_jsonl(args.samples)
    d8_records = load_jsonl(args.d8_samples)
    records = prepare_records(d12, d8_records)
    sample_ids = np.asarray(
        [str(record["sample_id"]) for record in records],
        dtype=str,
    )
    slots = np.zeros(
        (len(records), MAX_SLOTS, FEATURE_COUNT),
        dtype=np.float32,
    )
    mask = np.zeros((len(records), MAX_SLOTS), dtype=bool)
    raw_detection_count = np.zeros(len(records), dtype=np.int16)
    requests: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    video_hashes: dict[str, str] = {}
    for index, record in enumerate(records):
        video = str(Path(record["video_path"]).resolve())
        requests[video][int(record["anchor_scene_frame"])].append(index)
        expected = str(record["video_sha256"])
        previous = video_hashes.get(video)
        if previous is not None and previous != expected:
            raise ValueError("D31 video hash declaration mismatch")
        video_hashes[video] = expected
    model = YOLO(str(args.weights))

    def infer_batch(
        images: list[np.ndarray],
        destinations: list[list[int]],
    ) -> None:
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
            raise RuntimeError("D31 YOLO batch length mismatch")
        for prediction, sample_indices in zip(
            predictions,
            destinations,
            strict=True,
        ):
            boxes = np.empty((0, 5), dtype=np.float32)
            if prediction.boxes is not None and len(prediction.boxes):
                boxes = np.concatenate(
                    (
                        prediction.boxes.xyxyn.detach().cpu().numpy(),
                        prediction.boxes.conf.detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 1),
                    ),
                    axis=1,
                ).astype(np.float32)
            value, value_mask = static_slots(boxes)
            for sample_index in sample_indices:
                slots[sample_index] = value
                mask[sample_index] = value_mask
                raw_detection_count[sample_index] = len(boxes)

    decoded_rows = []
    for video_index, video_text in enumerate(sorted(requests)):
        video_path = Path(video_text)
        actual_hash = sha256(video_path)
        if actual_hash != video_hashes[video_text]:
            raise ValueError(f"D31 video hash mismatch: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"D31 cannot open video: {video_path}")
        frame_number = 0
        images: list[np.ndarray] = []
        destinations: list[list[int]] = []
        requested_seen = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                sample_indices = requests[video_text].get(frame_number)
                if not sample_indices:
                    continue
                images.append(frame)
                destinations.append(sample_indices)
                requested_seen += 1
                if len(images) >= args.batch_size:
                    infer_batch(images, destinations)
                    images.clear()
                    destinations.clear()
            infer_batch(images, destinations)
        finally:
            capture.release()
        if requested_seen != len(requests[video_text]):
            raise ValueError("D31 requested source frame missing")
        decoded_rows.append(
            {
                "video_path": video_text,
                "video_sha256": actual_hash,
                "decoded_frames": frame_number,
                "requested_unique_frames": requested_seen,
            }
        )
        print(
            json.dumps(
                {
                    "video": video_index + 1,
                    "videos": len(requests),
                    "anchors_with_person": int(
                        np.sum(raw_detection_count > 0)
                    ),
                }
            ),
            flush=True,
        )
    if not np.all(
        [
            raw_detection_count[index] >= 0
            for index in range(len(records))
        ]
    ):
        raise ValueError("D31 detection census invalid")
    write_npz_atomic(
        boxes_path,
        sample_ids=sample_ids,
        slots=slots,
        mask=mask,
        raw_detection_count=raw_detection_count,
    )
    boxes_hash = sha256(boxes_path)
    boxes_sidecar.write_text(
        f"{boxes_hash}  {boxes_path.name}\n",
        encoding="ascii",
    )

    d8_by_id = {
        str(record["sample_id"]): record for record in d8_records
    }
    trajectory_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for sample_index, record in enumerate(records):
        d8 = d8_by_id[str(record["sample_id"])]
        scenario_path = Path(str(d8["scenario_csv_path"]))
        camera_body = str(d8["camera_body"])
        key = (str(scenario_path.resolve()), camera_body)
        data = trajectory_cache.get(key)
        if data is None:
            data = read_scenario(
                scenario_path,
                camera_body,
                infer_scene_column(scenario_path, camera_body),
            )
            trajectory_cache[key] = data
        matches = np.flatnonzero(
            data["frames"] == int(d8["qtm_frame"])
        )
        if len(matches) != 1:
            raise ValueError("D31 QTM anchor is not unique")
        index = int(matches[0])
        velocity = (
            data["camera"][index + 25, :2]
            - data["camera"][index - 25, :2]
        ) / (data["times"][index + 25] - data["times"][index - 25])
        speed = float(np.linalg.norm(velocity))
        if speed < 0.25 or not np.isfinite(speed):
            raise ValueError("D31 wearer forward invalid")
        forward = velocity / speed
        origin = data["camera"][index, :2]
        body_rows = []
        for body_name, positions in data["others"].items():
            role = str(data["roles"].get(body_name, ""))
            if not is_person_body(str(body_name), role):
                continue
            position = positions[index, :2]
            if not np.isfinite(position).all():
                continue
            relative = position - origin
            distance = float(np.linalg.norm(relative))
            if distance <= 0 or distance > DISTANCE_CAP_M:
                continue
            bearing = relative_bearing_degrees(forward, relative)
            if abs(bearing) > HALF_FOV_DEGREES:
                continue
            body_rows.append(
                {
                    "body_name": str(body_name),
                    "body_role": role,
                    "bearing_degrees": bearing,
                    "distance_m": distance,
                }
            )
        slot_rows = slots[sample_index][mask[sample_index]]
        assignments = assign_measurements(
            slot_rows[:, 0] if len(slot_rows) else np.empty(0),
            np.asarray(
                [row["bearing_degrees"] for row in body_rows],
                dtype=np.float64,
            ),
        )
        accepted_body_indices = {
            int(pair["body_index"])
            for pair in assignments
            if pair["accepted"]
        }
        nearest_body_index = (
            int(np.argmin([row["distance_m"] for row in body_rows]))
            if body_rows
            else None
        )
        for pair in assignments:
            box_index = int(pair["box_index"])
            body_index = int(pair["body_index"])
            pair["box_height"] = float(slot_rows[box_index, 3])
            pair["box_confidence"] = float(slot_rows[box_index, 5])
            pair["body_distance_m"] = float(
                body_rows[body_index]["distance_m"]
            )
            pair["body_name"] = body_rows[body_index]["body_name"]
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "source_session_id": str(record["source_session_id"]),
                "fold": int(record["fold"]),
                "box_count": len(slot_rows),
                "body_count": len(body_rows),
                "body_distances_m": [
                    float(row["distance_m"]) for row in body_rows
                ],
                "assignments": assignments,
                "nearest_body_accepted": (
                    nearest_body_index in accepted_body_indices
                    if nearest_body_index is not None
                    else False
                ),
            }
        )
    pooled = summarize(rows)
    by_source = []
    for source in sorted(
        {str(row["source_session_id"]) for row in rows}
    ):
        summary = summarize(
            [row for row in rows if row["source_session_id"] == source]
        )
        summary["source_session_id"] = source
        by_source.append(summary)
    evaluable_sources = [
        row for row in by_source if row["assigned_pairs"] >= MIN_SOURCE_PAIRS
    ]

    def macro(metric: str) -> float:
        values = [
            float(row[metric])
            for row in evaluable_sources
            if row[metric] is not None
        ]
        if not values:
            raise ValueError(f"D31 no evaluable source metric: {metric}")
        return float(np.mean(values))

    source_macro = {
        "evaluable_sources": len(evaluable_sources),
        "box_x_predicted_x_pearson": macro(
            "box_x_predicted_x_pearson"
        ),
        "bearing_mae_degrees": macro("bearing_mae_degrees"),
        "height_inverse_distance_spearman": macro(
            "height_inverse_distance_spearman"
        ),
        "by_source": by_source,
    }
    by_fold = []
    for fold in range(5):
        summary = summarize(
            [row for row in rows if int(row["fold"]) == fold]
        )
        summary["fold"] = fold
        by_fold.append(summary)
    positive_distance_folds = sum(
        row["height_inverse_distance_spearman"] is not None
        and row["height_inverse_distance_spearman"] > 0
        for row in by_fold
    )
    checks = {
        "anchor_opportunity": pooled["anchors_with_both"] >= 300,
        "accepted_assignment_fraction": (
            pooled["accepted_fraction"] >= 0.60
        ),
        "nearest_body_coverage": (
            pooled["nearest_body_accepted_coverage"] >= 0.60
        ),
        "source_macro_bearing_pearson": (
            source_macro["box_x_predicted_x_pearson"] >= 0.50
        ),
        "source_macro_bearing_mae": (
            source_macro["bearing_mae_degrees"] <= 15.0
        ),
        "source_macro_distance_spearman": (
            source_macro["height_inverse_distance_spearman"] >= 0.30
        ),
        "positive_distance_folds": positive_distance_folds >= 3,
        "evaluable_sources": len(evaluable_sources) >= 15,
    }
    supported = all(checks.values())
    status = (
        "D31_THOR_MAGNI_FULL_RESOLUTION_MEASUREMENT_RELATION_SUPPORTED"
        if supported
        else (
            "D31_THOR_MAGNI_FULL_RESOLUTION_MEASUREMENT_RELATION_"
            "NOT_SUPPORTED"
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development full-resolution current box-to-world "
                "measurement replication"
            ),
            "future_body_positions_read": False,
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "weights_path": str(args.weights.resolve()),
            "weights_sha256": EXPECTED_YOLO_SHA256,
            "full_resolution_boxes_path": str(boxes_path.resolve()),
            "full_resolution_boxes_sha256": boxes_hash,
            "videos": decoded_rows,
        },
        "design": {
            "detector_source": "hash-bound original-resolution video frames",
            "ultralytics": ultralytics.__version__,
            "imgsz": 640,
            "confidence": 0.10,
            "nms_iou": 0.50,
            "max_det": 30,
            "max_slots": MAX_SLOTS,
            "half_fov_degrees": HALF_FOV_DEGREES,
            "distance_cap_m": DISTANCE_CAP_M,
            "accepted_x_error": ACCEPTED_X_ERROR,
            "person_body_rule": (
                "body name Helmet_* and role prefix Visitors- or Carrier-"
            ),
            "future_outcome_read": False,
        },
        "detection": {
            "anchors_with_person": int(
                np.sum(raw_detection_count > 0)
            ),
            "anchor_coverage": float(
                np.mean(raw_detection_count > 0)
            ),
            "raw_detections": int(np.sum(raw_detection_count)),
            "selected_slots": int(np.sum(mask)),
            "saturated_anchors": int(
                np.sum(raw_detection_count > MAX_SLOTS)
            ),
        },
        "pooled": pooled,
        "source_macro": source_macro,
        "by_fold": by_fold,
        "gate": {
            "frozen_thresholds": {
                "anchors_with_both": 300,
                "accepted_fraction": 0.60,
                "nearest_body_accepted_coverage": 0.60,
                "source_macro_box_x_pearson": 0.50,
                "source_macro_bearing_mae_degrees": 15.0,
                "source_macro_distance_spearman": 0.30,
                "positive_distance_folds": 3,
                "evaluable_sources": 15,
            },
            "positive_distance_folds": positive_distance_folds,
            "checks": checks,
            "supported": supported,
        },
        "next_action": (
            "freeze an explicit bearing-distance state filter canary"
            if supported
            else (
                "stop THOR current-box measurement and move to native "
                "2D/3D identity-bound person trajectories"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_hash = sha256(args.output)
    sidecar.write_text(
        f"{report_hash}  {args.output.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": status,
                "detection": report["detection"],
                "pooled": pooled,
                "source_macro": {
                    key: value
                    for key, value in source_macro.items()
                    if key != "by_source"
                },
                "gate": report["gate"],
                "boxes_sha256": boxes_hash,
                "report_sha256": report_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
