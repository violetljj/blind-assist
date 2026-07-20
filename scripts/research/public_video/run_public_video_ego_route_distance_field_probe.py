#!/usr/bin/env python3
"""Source-isolated spatial DINO probe for the offline future-route distance field."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_ego_trace_distillation_probe as metrics
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_ego_route_distance_field_probe_v1"


def segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-12:
        return np.linalg.norm(points - start, axis=-1)
    projection = np.clip(((points - start) @ delta) / denominator, 0.0, 1.0)
    closest = start + projection[..., None] * delta
    return np.linalg.norm(points - closest, axis=-1)


def route_distance_field(anchors: list[dict[str, Any]], side: int, sigma: float) -> np.ndarray:
    ordered = [np.asarray([0.5 * side, 0.9 * side], dtype=np.float64)]
    for row in sorted(anchors, key=lambda value: int(value["horizon_ms"])):
        x, y = map(float, row["point_xy_norm"])
        ordered.append(np.asarray([x * side, y * side], dtype=np.float64))
    yy, xx = np.mgrid[0:side, 0:side]
    pixels = np.stack([xx + 0.5, yy + 0.5], axis=-1)
    distances = [segment_distance(pixels, ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
    distance = np.minimum.reduce(distances) if distances else segment_distance(pixels, ordered[0], ordered[0])
    return np.exp(-(distance ** 2) / (2.0 * float(sigma) ** 2))


def obstacle_grid_mask(detections: list[dict[str, Any]], side: int, expansion: float) -> np.ndarray:
    yy, xx = np.mgrid[0:side, 0:side]
    px = (xx + 0.5) / side
    py = (yy + 0.5) / side
    mask = np.zeros((side, side), dtype=bool)
    for detection in detections:
        values = detection["features"]
        center = float(values["center_x_norm"])
        bottom = float(values["bottom_y_norm"])
        width = float(values["width_norm"])
        height = float(values["height_norm"])
        margin = float(expansion) * height
        mask |= ((px >= center - width / 2.0 - margin) & (px <= center + width / 2.0 + margin)
                 & (py >= bottom - height - margin) & (py <= bottom + margin))
    return mask


def fixed_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    return generator.normal(0.0, 1.0 / np.sqrt(output_dim), size=(input_dim, output_dim))


def extract_patch_tokens(model_dir: Path, images: list[np.ndarray], batch_size: int) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    rows = []
    for start in range(0, len(images), batch_size):
        resized = [Image.fromarray(cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)) for image in images[start:start + batch_size]]
        inputs = processor(images=resized, return_tensors="pt", do_resize=False, do_center_crop=False)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, 1:, :].cpu().numpy()
        if hidden.shape[1] != 256:
            raise ValueError(f"expected 16x16 DINO patch grid, got {hidden.shape}")
        rows.append(hidden.reshape(len(hidden), 16, 16, hidden.shape[-1]))
    return np.concatenate(rows, axis=0)


def weighted_ridge(train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    mean = np.sum(weights[:, None] * train_x, axis=0)
    centered = train_x - mean
    scale = np.sqrt(np.sum(weights[:, None] * centered * centered, axis=0))
    scale[scale < 1e-8] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    y_mean = float(np.sum(weights * train_y))
    y_centered = train_y - y_mean
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    coefficients = np.linalg.solve(x_train.T @ (weights[:, None] * x_train) + penalty,
                                   x_train.T @ (weights * y_centered))
    return y_mean + x_test @ coefficients


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.event_set_contract, args.offline_teacher_report, args.causal_feature_report, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    event_contract = json.loads(args.event_set_contract.read_text(encoding="utf-8"))
    bound = contract["bound_inputs"]
    for path, key in ((args.event_set_contract, "event_set_contract_sha256"),
                      (args.offline_teacher_report, "offline_teacher_report_sha256"),
                      (args.causal_feature_report, "causal_feature_report_sha256"),
                      (args.model_dir / "pytorch_model.bin", "dinov2_weights_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    teacher = lifecycle.verify_json_sidecar(args.offline_teacher_report)
    causal = lifecycle.verify_json_sidecar(args.causal_feature_report)
    feature_reports = {}
    for key, binding in event_contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        feature_reports[key] = lifecycle.verify_json_sidecar(path)
    teacher_events = {row["event_id"]: row for row in teacher["events"]}
    causal_events = {row["event_id"]: row for row in causal["events"]}
    side = int(contract["route_target"]["grid_side"])
    sigma = float(contract["route_target"]["sigma_patches"])
    frame_rows = []
    images = []
    for event in event_contract["events"]:
        source_report = feature_reports[event["feature_key"]]
        source = next(row for row in source_report["sources"] if row["source_id"] == event["source_id"])
        samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
        future_frames = {int(row["timestamp_ms"]): row for row in teacher_events[event["event_id"]]["event_diagnostics"]["frames"]}
        past_frames = {int(row["timestamp_ms"]): row for row in causal_events[event["event_id"]]["event_diagnostics"]["frames"]}
        timestamps = sorted(set(future_frames) & set(past_frames) & set(samples))
        decoded = route_width.decode_at(Path(source["local_video_path"]), timestamps)
        for timestamp, image in zip(timestamps, decoded):
            anchors = future_frames[timestamp].get("anchors", [])
            if not anchors:
                continue
            causal_anchors = sorted(past_frames[timestamp].get("anchors", []), key=lambda row: int(row["past_horizon_ms"]))
            causal_xy = []
            for index in range(3):
                causal_xy.extend(list(map(float, causal_anchors[index]["point_xy_norm"])) if index < len(causal_anchors) else [0.5, 0.9])
            frame_rows.append({
                "event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
                "timestamp_ms": timestamp, "target": route_distance_field(anchors, side, sigma),
                "causal_xy": np.asarray(causal_xy, dtype=np.float64), "detections": samples[timestamp].get("detections", []),
            })
            images.append(image)
    tokens = extract_patch_tokens(args.model_dir, images, args.batch_size)
    projection_spec = contract["current_causal_input"]["fixed_random_projection"]
    projection = fixed_projection(int(projection_spec["input_dimension"]), int(projection_spec["output_dimension"]), int(projection_spec["seed"]))
    projected = tokens @ projection
    yy, xx = np.mgrid[0:side, 0:side]
    coordinates = np.stack([(xx + 0.5) / side, (yy + 0.5) / side], axis=-1)
    frame_features = []
    for token_grid, row in zip(projected, frame_rows):
        causal_grid = np.broadcast_to(row["causal_xy"], (side, side, len(row["causal_xy"])))
        frame_features.append(np.concatenate([token_grid, coordinates, causal_grid], axis=-1))
    frame_features = np.stack(frame_features)
    sources = sorted({row["source_id"] for row in frame_rows})
    predictions = np.zeros((len(frame_rows), side, side), dtype=np.float64)
    alpha = float(contract["probe"]["ridge_alpha"])
    folds = []
    for source in sources:
        train_frames = [i for i, row in enumerate(frame_rows) if row["source_id"] != source]
        test_frames = [i for i, row in enumerate(frame_rows) if row["source_id"] == source]
        train_x = frame_features[train_frames].reshape(-1, frame_features.shape[-1])
        train_y = np.stack([frame_rows[i]["target"] for i in train_frames]).reshape(-1)
        test_x = frame_features[test_frames].reshape(-1, frame_features.shape[-1])
        source_counts = {value: sum(frame_rows[i]["source_id"] == value for i in train_frames) for value in sources if value != source}
        frame_source_weight = np.asarray([1.0 / source_counts[frame_rows[i]["source_id"]] for i in train_frames])
        weights = np.repeat(frame_source_weight, side * side) * (1.0 + 4.0 * train_y)
        predictions[test_frames] = weighted_ridge(train_x, train_y, weights, test_x, alpha).reshape(len(test_frames), side, side)
        folds.append({"held_out_source_id": source, "train_frame_count": len(train_frames), "test_frame_count": len(test_frames)})
    targets = np.stack([row["target"] for row in frame_rows])
    route_band = (targets >= np.exp(-0.5)).astype(np.int64)
    pixel_auroc = metrics.binary_auroc(route_band.reshape(-1), predictions.reshape(-1))
    clipped = np.clip(predictions, *map(float, contract["event_readout"]["predicted_route_clipping"]))
    event_scores: dict[str, list[float]] = defaultdict(list)
    expansion = float(contract["event_readout"]["marker_expansion_object_heights"])
    for row, heatmap in zip(frame_rows, clipped):
        mask = obstacle_grid_mask(row["detections"], side, expansion)
        denominator = max(float(heatmap.sum()), 1e-12)
        event_scores[row["event_id"]].append(float(heatmap[mask].sum() / denominator))
    event_predictions = [{"event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
                          "predicted_route_marker_overlap": float(np.mean(event_scores[event["event_id"]]))}
                         for event in event_contract["events"]]
    positive = [row["predicted_route_marker_overlap"] for row in event_predictions if row["label"] == 1]
    negative = [row["predicted_route_marker_overlap"] for row in event_predictions if row["label"] == 0]
    checks = {
        "route_band_pixel_auroc": pixel_auroc >= float(contract["gate"]["route_band_pixel_auroc_at_least"]),
        "strict_event_label_separation": min(positive) > max(negative),
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "event_set_contract_sha256": common.sha256_file(args.event_set_contract),
                   "offline_teacher_report_sha256": common.sha256_file(args.offline_teacher_report),
                   "dinov2_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin")},
        "frame_count": len(frame_rows), "source_count": len(sources), "spatial_feature_dimension": int(frame_features.shape[-1]),
        "folds": folds, "route_band_pixel_auroc": pixel_auroc,
        "distance_field_mean_absolute_error": float(np.mean(np.abs(predictions - targets))),
        "event_predictions": event_predictions, "checks": checks,
        "diagnostic_gate_passed": all(checks.values()), "evidence_limit": contract["evidence_role"],
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--event-set-contract", type=Path, required=True)
    parser.add_argument("--offline-teacher-report", type=Path, required=True)
    parser.add_argument("--causal-feature-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "diagnostic_gate_passed": value["diagnostic_gate_passed"],
                      "route_band_pixel_auroc": value["route_band_pixel_auroc"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
