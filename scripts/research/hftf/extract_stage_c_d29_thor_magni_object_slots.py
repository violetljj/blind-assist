#!/usr/bin/env python3
"""Extract frozen YOLO-person and within-box backward-flow object slots."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_FLOW,
    DEFAULT_RGB_CACHE,
    DEFAULT_SAMPLES,
    validate_inputs,
)
from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DEFAULT_D8_SAMPLES,
    prepare_records,
)


SCHEMA = "blindassist_hftf_stage_c_d29_thor_magni_object_slots_v0"
EXPECTED_YOLO_SHA256 = (
    "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
)
MAX_SLOTS = 8
STATIC_FEATURE_COUNT = 6
LAG_FEATURE_COUNT = 7
LAG_COUNT = 4
FEATURE_COUNT = STATIC_FEATURE_COUNT + LAG_COUNT * LAG_FEATURE_COUNT
FEATURE_NAMES = [
    "center_x_signed",
    "bottom_y",
    "width",
    "height",
    "sqrt_area",
    "confidence",
]
for lag in range(LAG_COUNT):
    FEATURE_NAMES.extend(
        [
            f"lag_{lag}_raw_flow_x",
            f"lag_{lag}_raw_flow_y",
            f"lag_{lag}_residual_flow_x",
            f"lag_{lag}_residual_flow_y",
            f"lag_{lag}_log_width_ratio",
            f"lag_{lag}_log_height_ratio",
            f"lag_{lag}_valid_fraction",
        ]
    )

DEFAULT_WEIGHTS = Path("artifacts.local/models/yolo11n.pt")
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d29-thor-magni-object-slots-v0/object_slots.npz"
)


def selected_boxes(
    boxes_xyxy_conf: np.ndarray,
    max_slots: int = MAX_SLOTS,
) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy_conf, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("D29 boxes must have shape Nx5")
    if len(boxes) == 0:
        return boxes
    if not np.isfinite(boxes).all():
        raise ValueError("D29 boxes contain non-finite values")
    x1, y1, x2, y2, confidence = boxes.T
    if (
        np.any(x1 < 0)
        or np.any(y1 < 0)
        or np.any(x2 > 1)
        or np.any(y2 > 1)
        or np.any(x2 <= x1)
        or np.any(y2 <= y1)
        or np.any(confidence < 0)
        or np.any(confidence > 1)
    ):
        raise ValueError("D29 normalized box range mismatch")
    score = confidence * np.sqrt((x2 - x1) * (y2 - y1))
    order = np.lexsort((x1, -score))
    return boxes[order[:max_slots]]


def slot_features(
    boxes_xyxy_conf: np.ndarray,
    current_to_history_flow: np.ndarray,
    *,
    max_slots: int = MAX_SLOTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = np.asarray(current_to_history_flow, dtype=np.float32)
    if (
        flow.ndim != 4
        or flow.shape[0] != LAG_COUNT
        or flow.shape[1] != 2
        or not np.isfinite(flow).all()
    ):
        raise ValueError("D29 flow must be finite 4x2xHxW")
    height, width = flow.shape[-2:]
    boxes = selected_boxes(boxes_xyxy_conf, max_slots)
    slots = np.zeros((max_slots, FEATURE_COUNT), dtype=np.float32)
    mask = np.zeros(max_slots, dtype=bool)
    lag_valid = np.zeros((max_slots, LAG_COUNT), dtype=np.float32)
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    global_median = np.median(flow, axis=(2, 3))
    for slot_index, box in enumerate(boxes):
        x1, y1, x2, y2, confidence = (float(value) for value in box)
        box_width = x2 - x1
        box_height = y2 - y1
        slots[slot_index, :STATIC_FEATURE_COUNT] = (
            2.0 * ((x1 + x2) * 0.5) - 1.0,
            y2,
            box_width,
            box_height,
            np.sqrt(box_width * box_height),
            confidence,
        )
        mask[slot_index] = True
        region = (
            (grid_x / width >= x1)
            & (grid_x / width <= x2)
            & (grid_y / height >= y1)
            & (grid_y / height <= y2)
        )
        if not np.any(region):
            continue
        current_region_width = max(box_width * width, 1e-6)
        current_region_height = max(box_height * height, 1e-6)
        for lag in range(LAG_COUNT):
            dx = flow[lag, 0][region]
            dy = flow[lag, 1][region]
            source_x = grid_x[region] + dx
            source_y = grid_y[region] + dy
            valid = (
                (source_x >= 0)
                & (source_x <= width)
                & (source_y >= 0)
                & (source_y <= height)
            )
            valid_fraction = float(np.mean(valid))
            lag_valid[slot_index, lag] = valid_fraction
            if not np.any(valid):
                continue
            raw_x = float(np.median(dx[valid]) / width)
            raw_y = float(np.median(dy[valid]) / height)
            historical_width = float(
                np.quantile(source_x[valid], 0.90)
                - np.quantile(source_x[valid], 0.10)
            )
            historical_height = float(
                np.quantile(source_y[valid], 0.90)
                - np.quantile(source_y[valid], 0.10)
            )
            width_ratio = max(historical_width / current_region_width, 1e-6)
            height_ratio = max(
                historical_height / current_region_height,
                1e-6,
            )
            offset = STATIC_FEATURE_COUNT + lag * LAG_FEATURE_COUNT
            slots[slot_index, offset : offset + LAG_FEATURE_COUNT] = (
                raw_x,
                raw_y,
                raw_x - float(global_median[lag, 0] / width),
                raw_y - float(global_median[lag, 1] / height),
                float(np.clip(np.log(width_ratio), -2.0, 2.0)),
                float(np.clip(np.log(height_ratio), -2.0, 2.0)),
                valid_fraction,
            )
    return slots, mask, lag_valid


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
    parser.add_argument("--rgb-cache", type=Path, default=DEFAULT_RGB_CACHE)
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or report_path.exists() or sidecar.exists():
        raise FileExistsError("D29 object-slot output is non-overwriting")
    if sha256(args.weights) != EXPECTED_YOLO_SHA256:
        raise ValueError("D29 YOLO weight SHA-256 mismatch")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("D29 object-slot extraction requires CUDA")
    if ultralytics.__version__ != "8.4.102":
        raise ValueError("D29 Ultralytics version mismatch")
    d12_records = load_jsonl(args.samples)
    validate_inputs(d12_records, args.samples, args.rgb_cache, args.flow)
    records = prepare_records(d12_records, load_jsonl(args.d8_samples))
    rgb = np.load(args.rgb_cache, mmap_mode="r")
    flow = np.load(args.flow, mmap_mode="r")
    sample_ids = np.asarray(
        [str(record["sample_id"]) for record in records],
        dtype=str,
    )
    slots = np.zeros(
        (len(records), MAX_SLOTS, FEATURE_COUNT),
        dtype=np.float32,
    )
    mask = np.zeros((len(records), MAX_SLOTS), dtype=bool)
    lag_valid = np.zeros(
        (len(records), MAX_SLOTS, LAG_COUNT),
        dtype=np.float32,
    )
    raw_detection_count = np.zeros(len(records), dtype=np.int16)
    model = YOLO(str(args.weights))
    for start in range(0, len(records), args.batch_size):
        stop = min(start + args.batch_size, len(records))
        images = [
            np.array(
                rgb[int(record["_d22_cache_index"]), -1],
                copy=True,
            )
            for record in records[start:stop]
        ]
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
        if len(predictions) != stop - start:
            raise RuntimeError("D29 YOLO batch length mismatch")
        for local_index, prediction in enumerate(predictions):
            record_index = start + local_index
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
            raw_detection_count[record_index] = len(boxes)
            cache_index = int(records[record_index]["_d22_cache_index"])
            (
                slots[record_index],
                mask[record_index],
                lag_valid[record_index],
            ) = slot_features(boxes, flow[cache_index])
        print(
            json.dumps(
                {
                    "complete": stop,
                    "total": len(records),
                    "anchors_with_person": int(
                        np.sum(raw_detection_count[:stop] > 0)
                    ),
                }
            ),
            flush=True,
        )
    if not np.isfinite(slots).all() or not np.isfinite(lag_valid).all():
        raise ValueError("D29 object-slot cache contains non-finite values")
    write_npz_atomic(
        args.output,
        sample_ids=sample_ids,
        slots=slots,
        mask=mask,
        lag_valid=lag_valid,
        raw_detection_count=raw_detection_count,
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
    )
    output_hash = sha256(args.output)
    sidecar.write_text(
        f"{output_hash}  {args.output.name}\n",
        encoding="ascii",
    )
    selected_count = np.sum(mask, axis=1)
    valid_selected = lag_valid[mask]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(),
        "status": "D29_THOR_MAGNI_OBJECT_SLOTS_MATERIALIZED",
        "authority": {
            "role": "Development causal object-slot feature cache",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "rgb_cache_path": str(args.rgb_cache.resolve()),
            "rgb_cache_sha256": sha256(args.rgb_cache),
            "flow_path": str(args.flow.resolve()),
            "flow_sha256": sha256(args.flow),
            "weights_path": str(args.weights.resolve()),
            "weights_sha256": EXPECTED_YOLO_SHA256,
        },
        "design": {
            "detector": "YOLO11n COCO person",
            "ultralytics": ultralytics.__version__,
            "imgsz": 640,
            "confidence": 0.10,
            "nms_iou": 0.50,
            "max_det": 30,
            "max_slots": MAX_SLOTS,
            "feature_names": FEATURE_NAMES,
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "anchors_with_person": int(np.sum(raw_detection_count > 0)),
            "anchor_coverage": float(np.mean(raw_detection_count > 0)),
            "raw_detections": int(np.sum(raw_detection_count)),
            "selected_slots": int(np.sum(mask)),
            "saturated_anchors": int(np.sum(raw_detection_count > MAX_SLOTS)),
            "selected_slots_per_anchor_mean": float(
                np.mean(selected_count)
            ),
            "selected_lag_valid_fraction_mean": (
                float(np.mean(valid_selected)) if len(valid_selected) else 0.0
            ),
            "selected_lag_valid_fraction_min": (
                float(np.min(valid_selected)) if len(valid_selected) else 0.0
            ),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": output_hash,
            "bytes": args.output.stat().st_size,
            "slots_shape": list(slots.shape),
            "mask_shape": list(mask.shape),
            "lag_valid_shape": list(lag_valid.shape),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
