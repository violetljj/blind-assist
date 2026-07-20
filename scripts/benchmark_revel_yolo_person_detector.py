#!/usr/bin/env python3
"""GPU-only offline YOLO11n person-detection benchmark on REveL Dynamic RGB.

The pretrained COCO person class is evaluated against the public green/yellow
helmet boxes after discarding colour identity.  It is a reproducible visual
baseline, not a trained USTRF component, person-identity classifier, or
authorization for device behaviour.

The defaults intentionally favour host stability over throughput.  Full-dataset
runs and batches larger than one must be requested explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


IOU_THRESHOLD = 0.5
SCORE_FLOOR = 0.05
FIXED_SCORE = 0.25
DEFAULT_MEMORY_FRACTION = 0.25
INFERENCE_MODE_FULL = "full_frame"
INFERENCE_MODE_TILED = "full_plus_4_corner_crops"
FIXED_TILE_FRACTION = 0.60
FIXED_MERGE_NMS_IOU = 0.50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou_matrix(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if not len(prediction) or not len(ground_truth):
        return np.zeros((len(prediction), len(ground_truth)), dtype=np.float64)
    top_left = np.maximum(prediction[:, None, :2], ground_truth[None, :, :2])
    bottom_right = np.minimum(prediction[:, None, 2:], ground_truth[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)
    pred_area = np.prod(np.maximum(0.0, prediction[:, 2:] - prediction[:, :2]), axis=1)
    gt_area = np.prod(np.maximum(0.0, ground_truth[:, 2:] - ground_truth[:, :2]), axis=1)
    return intersection / np.maximum(pred_area[:, None] + gt_area[None, :] - intersection, 1e-12)


def _match(prediction: np.ndarray, scores: np.ndarray, ground_truth: np.ndarray, score_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero(scores >= score_threshold)
    selected = selected[np.argsort(-scores[selected], kind="stable")]
    matched_ground = np.zeros(len(ground_truth), dtype=bool); truth = np.zeros(len(selected), dtype=bool)
    iou = _iou_matrix(prediction[selected], ground_truth)
    for order, _ in enumerate(selected):
        if len(ground_truth):
            target = int(np.argmax(iou[order]))
            if iou[order, target] >= IOU_THRESHOLD and not matched_ground[target]:
                matched_ground[target] = True; truth[order] = True
    return truth, matched_ground


def _ap50(records: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if not records or not ground_truth_count:
        return 0.0
    ordered = sorted(records, key=lambda item: -item[0])
    tp = np.cumsum([int(item[1]) for item in ordered], dtype=np.float64)
    fp = np.cumsum([int(not item[1]) for item in ordered], dtype=np.float64)
    recall = tp / ground_truth_count; precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0])); mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def _labels(images_root: Path, labels_root: Path) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    images = sorted(images_root.glob("*.jpg"), key=lambda path: int(path.stem))
    boxes: list[np.ndarray] = []; areas: list[np.ndarray] = []
    for image in images:
        rows = []
        for line in (labels_root / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                _, cx, cy, width, height = line.split(); rows.append(tuple(map(float, (cx, cy, width, height))))
        array = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
        if len(array):
            xyxy = np.column_stack((array[:, 0] - array[:, 2] / 2, array[:, 1] - array[:, 3] / 2, array[:, 0] + array[:, 2] / 2, array[:, 1] + array[:, 3] / 2))
            boxes.append(xyxy); areas.append(array[:, 2] * array[:, 3])
        else:
            boxes.append(np.empty((0, 4), dtype=np.float64)); areas.append(np.empty(0, dtype=np.float64))
    return images, boxes, areas


def _stratum(area: float) -> str:
    return "small" if area < .02 else "medium" if area < .10 else "large"


def _select_indices(total: int, max_frames: int | None, selection: str) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if max_frames is None:
        return list(range(total))
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    count = min(total, max_frames)
    if selection == "head":
        return list(range(count))
    if selection != "uniform":
        raise ValueError(f"unsupported selection: {selection}")
    if count == total:
        return list(range(total))
    return np.linspace(0, total - 1, num=count, dtype=np.int64).tolist()


def _read_selection_contract(path: Path, total: int, expected_count: int | None) -> tuple[list[int], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("format") != "blindassist_revel_crop_tiling_selection_v1":
        raise ValueError("unsupported selection contract format")
    indices = contract.get("selected_indices")
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError("selection contract selected_indices must be an integer list")
    if not indices:
        raise ValueError("selection contract must contain at least one index")
    if indices != sorted(set(indices)):
        raise ValueError("selection contract indices must be unique and strictly increasing")
    if indices[0] < 0 or indices[-1] >= total:
        raise ValueError(f"selection contract contains an out-of-range index for total={total}")
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(f"selection contract count {len(indices)} does not match max_frames={expected_count}")
    if contract.get("sample_role") != "failure_enriched_crop_tiling_upper_bound":
        raise ValueError("selection contract must declare the failure-enriched upper-bound sample role")
    source_receipt = contract.get("source_details_receipt")
    if not isinstance(source_receipt, dict) or not isinstance(source_receipt.get("sha256"), str) or len(source_receipt["sha256"]) != 64:
        raise ValueError("selection contract must bind a source details SHA-256")
    return indices, contract


def _axis_tile_bounds(length: int, tile_fraction: float = FIXED_TILE_FRACTION) -> tuple[tuple[int, int], tuple[int, int]]:
    if length < 2:
        raise ValueError("tile axis length must be at least two pixels")
    if not 0.5 < tile_fraction < 1.0:
        raise ValueError("tile fraction must be in (0.5, 1.0)")
    tile_size = min(length, int(math.ceil(length * tile_fraction)))
    return ((0, tile_size), (length - tile_size, length))


def _tile_bounds(width: int, height: int, tile_fraction: float = FIXED_TILE_FRACTION) -> list[tuple[int, int, int, int]]:
    x_bounds = _axis_tile_bounds(width, tile_fraction)
    y_bounds = _axis_tile_bounds(height, tile_fraction)
    return [(left, top, right, bottom) for top, bottom in y_bounds for left, right in x_bounds]


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be in (0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    remaining = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while len(remaining):
        current = int(remaining[0])
        kept.append(current)
        if len(remaining) == 1:
            break
        rest = remaining[1:]
        overlap = _iou_matrix(boxes[current : current + 1], boxes[rest])[0]
        remaining = rest[overlap <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def _remap_prediction(
    boxes_xyxy: np.ndarray,
    *,
    full_width: int,
    full_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    if not len(boxes_xyxy):
        return np.empty((0, 4), dtype=np.float64)
    remapped = np.asarray(boxes_xyxy, dtype=np.float64).copy()
    remapped[:, (0, 2)] += offset_x
    remapped[:, (1, 3)] += offset_y
    remapped /= np.asarray([full_width, full_height, full_width, full_height], dtype=np.float64)
    return np.clip(remapped, 0.0, 1.0)


def _validate_runtime(
    batch: int,
    imgsz: int,
    memory_fraction: float,
    inter_batch_delay_ms: int,
    inference_mode: str = INFERENCE_MODE_FULL,
    inter_view_delay_ms: int = 0,
) -> None:
    if batch < 1:
        raise ValueError("batch must be positive")
    if imgsz < 64:
        raise ValueError("imgsz must be at least 64")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if inter_batch_delay_ms < 0:
        raise ValueError("inter_batch_delay_ms must be non-negative")
    if inference_mode not in (INFERENCE_MODE_FULL, INFERENCE_MODE_TILED):
        raise ValueError(f"unsupported inference mode: {inference_mode}")
    if inter_view_delay_ms < 0:
        raise ValueError("inter_view_delay_ms must be non-negative")


def _frame_detail_record(
    selected_index: int,
    image: Path,
    prediction: np.ndarray,
    scores: np.ndarray,
    ground_truth: np.ndarray,
    areas: np.ndarray,
    fixed_truth: np.ndarray,
    fixed_ground: np.ndarray,
    inference_mode: str = INFERENCE_MODE_FULL,
    view_ids: np.ndarray | None = None,
    raw_prediction: np.ndarray | None = None,
    raw_scores: np.ndarray | None = None,
    raw_view_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    fused_view_ids = view_ids if view_ids is not None else np.asarray(["full_frame"] * len(prediction), dtype=object)
    raw_boxes = prediction if raw_prediction is None else raw_prediction
    raw_score_values = scores if raw_scores is None else raw_scores
    raw_views = fused_view_ids if raw_view_ids is None else raw_view_ids
    return {
        "selected_index": selected_index,
        "image_name": image.name,
        "source_timestamp_ns": int(image.stem),
        "ground_truth": [
            {
                "xyxy_normalized": [float(value) for value in box],
                "normalized_area": float(area),
                "stratum": _stratum(float(area)),
                "matched_at_fixed_score": bool(matched),
            }
            for box, area, matched in zip(ground_truth, areas, fixed_ground)
        ],
        "predictions_over_score_floor": [
            {"xyxy_normalized": [float(value) for value in box], "score": float(score), "view_id": str(view_id)}
            for box, score, view_id in zip(prediction, scores, fused_view_ids)
        ],
        "raw_predictions_over_score_floor": [
            {"xyxy_normalized": [float(value) for value in box], "score": float(score), "view_id": str(view_id)}
            for box, score, view_id in zip(raw_boxes, raw_score_values, raw_views)
        ],
        "inference": {"mode": inference_mode, "view_count": 1 if inference_mode == INFERENCE_MODE_FULL else 5, "raw_prediction_count": len(raw_boxes), "fused_prediction_count": len(prediction)},
        "fixed_score_counts": {
            "tp": int(fixed_truth.sum()),
            "fp": int(len(fixed_truth) - fixed_truth.sum()),
            "fn": int(len(ground_truth) - fixed_ground.sum()),
        },
    }


def benchmark(
    dataset_root: Path,
    weights: Path,
    batch: int = 1,
    imgsz: int = 256,
    max_frames: int | None = None,
    selection: str = "uniform",
    selection_contract: Path | None = None,
    output: Path | None = None,
    details_output: Path | None = None,
    half: bool = False,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    inter_batch_delay_ms: int = 0,
    inference_mode: str = INFERENCE_MODE_FULL,
    inter_view_delay_ms: int = 0,
) -> dict[str, Any]:
    import torch
    import ultralytics
    from PIL import Image
    from ultralytics import YOLO

    _validate_runtime(
        batch,
        imgsz,
        memory_fraction,
        inter_batch_delay_ms,
        inference_mode,
        inter_view_delay_ms,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for REveL detector benchmark")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weights_sha256 = _sha256(weights)
    all_images, all_ground_truth, all_areas = _labels(dataset_root / "extracted" / "images" / "images", dataset_root / "extracted" / "labels" / "labels")
    if selection_contract is not None:
        selected_indices, contract = _read_selection_contract(selection_contract, len(all_images), max_frames)
        selection_name = "selection_contract"
        selection_receipt = {
            "path": str(selection_contract),
            "sha256": _sha256(selection_contract),
            "count": len(selected_indices),
            "sample_role": contract["sample_role"],
            "source_details_sha256": contract["source_details_receipt"]["sha256"],
        }
        if contract.get("weights_sha256") != weights_sha256:
            raise ValueError("selection contract weights SHA-256 does not match the requested model")
    else:
        selected_indices = _select_indices(len(all_images), max_frames, selection)
        selection_name = selection
        selection_receipt = None
    images = [all_images[index] for index in selected_indices]
    ground_truth = [all_ground_truth[index] for index in selected_indices]
    areas = [all_areas[index] for index in selected_indices]
    if not images:
        raise RuntimeError("REveL benchmark selection is empty")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    model = YOLO(str(weights))
    torch.cuda.synchronize()
    started = time.perf_counter()
    records: list[tuple[float, bool]] = []; fixed = {"tp": 0, "fp": 0, "fn": 0}; strata = {name: {"ground_truth": 0, "matched": 0} for name in ("small", "medium", "large")}
    frame_details: list[dict[str, Any]] = []
    processed = 0
    inference_views = 0

    def predict_arguments(source: Any, *, predict_batch: int) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "source": source,
            "device": 0,
            "imgsz": imgsz,
            "batch": predict_batch,
            "conf": SCORE_FLOOR,
            "classes": [0],
            "iou": .7,
            "max_det": 100,
            "stream": False,
            "verbose": False,
            "workers": 0,
        }
        if half:
            arguments["half"] = True
        return arguments

    def result_prediction(result: Any, *, full_width: int, full_height: int, offset_x: int = 0, offset_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4), dtype=np.float64)
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0, dtype=np.float64)
        return _remap_prediction(boxes, full_width=full_width, full_height=full_height, offset_x=offset_x, offset_y=offset_y), scores

    with torch.inference_mode():
        for start in range(0, len(images), batch):
            stop = min(start + batch, len(images))
            batch_images = images[start:stop]
            frame_predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            if inference_mode == INFERENCE_MODE_FULL:
                results = model.predict(**predict_arguments([str(path) for path in batch_images], predict_batch=batch))
                if len(results) != len(batch_images):
                    raise RuntimeError(f"prediction count mismatch: expected {len(batch_images)}, got {len(results)}")
                for result in results:
                    prediction, scores = result_prediction(result, full_width=result.orig_shape[1], full_height=result.orig_shape[0])
                    view_ids = np.asarray(["full_frame"] * len(prediction), dtype=object)
                    frame_predictions.append((prediction, scores, view_ids, prediction, scores, view_ids))
                inference_views += len(results)
            else:
                # Tiled inference keeps each view independently attributable. ``batch``
                # controls the source-frame group and delay cadence; each full/crop
                # view remains a one-image predict call so differently sized views do
                # not silently change preprocessing or geometry.
                for image in batch_images:
                    with Image.open(image) as opened:
                        rgb = np.asarray(opened.convert("RGB"))
                    height, width = rgb.shape[:2]
                    collected_boxes: list[np.ndarray] = []
                    collected_scores: list[np.ndarray] = []
                    collected_view_ids: list[np.ndarray] = []
                    full_results = model.predict(**predict_arguments([str(image)], predict_batch=1))
                    if len(full_results) != 1:
                        raise RuntimeError(f"full-frame prediction count mismatch: expected 1, got {len(full_results)}")
                    full_boxes, full_scores = result_prediction(full_results[0], full_width=width, full_height=height)
                    collected_boxes.append(full_boxes); collected_scores.append(full_scores); collected_view_ids.append(np.asarray(["full_frame"] * len(full_boxes), dtype=object))
                    bounds = _tile_bounds(width, height)
                    for tile_index, (left, top, right, bottom) in enumerate(bounds):
                        if inter_view_delay_ms:
                            time.sleep(inter_view_delay_ms / 1000.0)
                        tile_bgr = rgb[top:bottom, left:right, ::-1].copy()
                        tile_results = model.predict(**predict_arguments(tile_bgr, predict_batch=1))
                        if len(tile_results) != 1:
                            raise RuntimeError(f"tile prediction count mismatch: expected 1, got {len(tile_results)}")
                        tile_boxes, tile_scores = result_prediction(tile_results[0], full_width=width, full_height=height, offset_x=left, offset_y=top)
                        collected_boxes.append(tile_boxes); collected_scores.append(tile_scores); collected_view_ids.append(np.asarray([f"corner_crop_{tile_index}"] * len(tile_boxes), dtype=object))
                    raw_prediction = np.concatenate(collected_boxes, axis=0)
                    raw_scores = np.concatenate(collected_scores, axis=0)
                    raw_view_ids = np.concatenate(collected_view_ids, axis=0)
                    kept = _nms_indices(raw_prediction, raw_scores, FIXED_MERGE_NMS_IOU)
                    frame_predictions.append((raw_prediction[kept], raw_scores[kept], raw_view_ids[kept], raw_prediction, raw_scores, raw_view_ids))
                    inference_views += 1 + len(bounds)

            for offset, (image, (prediction, scores, view_ids, raw_prediction, raw_scores, raw_view_ids), gt, area) in enumerate(zip(batch_images, frame_predictions, ground_truth[start:stop], areas[start:stop])):
                truth, _ = _match(prediction, scores, gt, SCORE_FLOOR)
                records.extend((float(score), bool(correct)) for score, correct in zip(np.sort(scores)[::-1], truth))
                fixed_truth, fixed_ground = _match(prediction, scores, gt, FIXED_SCORE)
                fixed["tp"] += int(fixed_truth.sum()); fixed["fp"] += int(len(fixed_truth) - fixed_truth.sum()); fixed["fn"] += int(len(gt) - fixed_ground.sum())
                for item_area, matched in zip(area, fixed_ground):
                    bucket = _stratum(float(item_area)); strata[bucket]["ground_truth"] += 1; strata[bucket]["matched"] += int(matched)
                if details_output is not None:
                    frame_details.append(_frame_detail_record(selected_indices[start + offset], image, prediction, scores, gt, area, fixed_truth, fixed_ground, inference_mode, view_ids, raw_prediction, raw_scores, raw_view_ids))
            processed = stop
            print(f"processed={processed}/{len(images)}", file=sys.stderr, flush=True)
            if inter_batch_delay_ms:
                time.sleep(inter_batch_delay_ms / 1000.0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    peak_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
    precision = fixed["tp"] / max(1, fixed["tp"] + fixed["fp"]); recall = fixed["tp"] / max(1, fixed["tp"] + fixed["fn"])
    details_receipt = None
    if details_output is not None:
        details_output.parent.mkdir(parents=True, exist_ok=True)
        details_output.write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in frame_details), encoding="utf-8")
        details_receipt = {"path": str(details_output), "frame_records": len(frame_details), "sha256": _sha256(details_output)}
    report = {
        "format": "blindassist_revel_yolo11n_person_benchmark_v2",
        "dataset": {"total_frames": len(all_images), "evaluated_frames": len(images), "selection": selection_name, "selection_receipt": selection_receipt, "selected_indices": selected_indices, "selected_first_index": selected_indices[0], "selected_last_index": selected_indices[-1], "person_ground_truth_boxes": int(sum(len(item) for item in ground_truth)), "class_handling": "green-helmet and yellow-helmet boxes are merged into one person class"},
        "model": {"name": "YOLO11n", "pretrained_task": "COCO person detection", "weights_sha256": weights_sha256, "ultralytics_version": ultralytics.__version__, "imgsz": imgsz, "batch": batch, "half": half, "score_floor": SCORE_FLOOR, "iou_threshold": IOU_THRESHOLD, "inference_mode": inference_mode, "tiling": None if inference_mode == INFERENCE_MODE_FULL else {"include_full_frame": True, "crop_count": 4, "corner_crop_fraction": FIXED_TILE_FRACTION, "normalized_windows": [[0.0, 0.0, 0.6, 0.6], [0.4, 0.0, 1.0, 0.6], [0.0, 0.4, 0.6, 1.0], [0.4, 0.4, 1.0, 1.0]], "merge_nms_iou": FIXED_MERGE_NMS_IOU, "views_per_frame": 5, "source_group_size": batch, "per_view_predict_batch": 1}},
        "fixed_score_metrics": {"score": FIXED_SCORE, **fixed, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)},
        "ap50_over_score_floor": _ap50(records, int(sum(len(item) for item in ground_truth))),
        "recall_by_normalized_box_area": {name: {**values, "recall": values["matched"] / max(1, values["ground_truth"])} for name, values in strata.items()},
        "throughput": {"elapsed_s": elapsed, "source_frames_per_s": len(images) / elapsed, "inference_views_per_s": inference_views / elapsed, "frames_per_s": len(images) / elapsed},
        "compute_backend": {"name": "ultralytics+torch", "cuda": True, "device": torch.cuda.get_device_name(0), "memory_fraction_limit": memory_fraction, "peak_allocated_mb": peak_allocated_mb, "peak_reserved_mb": peak_reserved_mb, "cudnn_benchmark": False, "tf32": False, "inter_batch_delay_ms": inter_batch_delay_ms, "inter_view_delay_ms": inter_view_delay_ms, "inference_views": inference_views},
        "details_receipt": details_receipt,
        "admission": {"offline_rgb_person_detection_baseline_admitted": True, "not_admitted_for": ["metric distance", "physical TTC", "body-local safe corridor", "assistive event truth", "on-device safety"], "reason": "pretrained COCO detector evaluated offline on source boxes; no device latency, body calibration, or danger-event truth"},
        "production_authority": False,
    }
    output_path = output or dataset_root / "qa" / "revel_yolo11n_person_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1); parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--max-frames", type=int); parser.add_argument("--selection", choices=("uniform", "head"), default="uniform")
    parser.add_argument("--selection-contract", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--details-output", type=Path); parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-fraction", type=float, default=DEFAULT_MEMORY_FRACTION)
    parser.add_argument("--inter-batch-delay-ms", type=int, default=0)
    parser.add_argument("--inference-mode", choices=(INFERENCE_MODE_FULL, INFERENCE_MODE_TILED), default=INFERENCE_MODE_FULL)
    parser.add_argument("--inter-view-delay-ms", type=int, default=0)
    args = parser.parse_args()
    report = benchmark(
        dataset_root=args.dataset_root,
        weights=args.weights,
        batch=args.batch,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        selection=args.selection,
        selection_contract=args.selection_contract,
        output=args.output,
        details_output=args.details_output,
        half=args.half,
        memory_fraction=args.memory_fraction,
        inter_batch_delay_ms=args.inter_batch_delay_ms,
        inference_mode=args.inference_mode,
        inter_view_delay_ms=args.inter_view_delay_ms,
    )
    print(json.dumps({"evaluated_frames": report["dataset"]["evaluated_frames"], "ap50": report["ap50_over_score_floor"], "recall": report["fixed_score_metrics"]["recall"], "fps": report["throughput"]["frames_per_s"], "peak_reserved_mb": report["compute_backend"]["peak_reserved_mb"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
