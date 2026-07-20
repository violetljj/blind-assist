#!/usr/bin/env python3
"""Predict three future-route heatmaps and reuse exact horizon-point marker hits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_ego_route_distance_field_probe as spatial
import run_public_video_ego_trace_distillation_probe as base
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_multi_horizon_route_field_probe_v1"


def horizon_fields(anchors: list[dict[str, Any]], side: int, sigma: float, horizons: list[int]) -> np.ndarray:
    by_horizon = {int(row["horizon_ms"]): row for row in anchors}
    yy, xx = np.mgrid[0:side, 0:side]
    fields = []
    for horizon in horizons:
        row = by_horizon.get(int(horizon))
        if row is None:
            raise ValueError(f"missing future anchor horizon: {horizon}")
        x, y = map(float, row["point_xy_norm"])
        distance2 = (xx + 0.5 - x * side) ** 2 + (yy + 0.5 - y * side) ** 2
        fields.append(np.exp(-distance2 / (2.0 * float(sigma) ** 2)))
    return np.stack(fields)


def argmax_point(field: np.ndarray) -> tuple[float, float]:
    values = np.asarray(field)
    y, x = np.unravel_index(int(np.argmax(values)), values.shape)
    return (x + 0.5) / values.shape[1], (y + 0.5) / values.shape[0]


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
    reports = {}
    for key, binding in event_contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        reports[key] = lifecycle.verify_json_sidecar(path)
    teacher_events = {row["event_id"]: row for row in teacher["events"]}
    causal_events = {row["event_id"]: row for row in causal["events"]}
    side = int(contract["target"]["grid_side"])
    horizons = list(map(int, contract["target"]["channels_ms"]))
    sigma = float(contract["target"]["sigma_patches"])
    rows = []
    images = []
    for event in event_contract["events"]:
        source = next(row for row in reports[event["feature_key"]]["sources"] if row["source_id"] == event["source_id"])
        samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
        future = {int(row["timestamp_ms"]): row for row in teacher_events[event["event_id"]]["event_diagnostics"]["frames"]}
        past = {int(row["timestamp_ms"]): row for row in causal_events[event["event_id"]]["event_diagnostics"]["frames"]}
        timestamps = sorted(set(future) & set(past) & set(samples))
        decoded = route_width.decode_at(Path(source["local_video_path"]), timestamps)
        for timestamp, image in zip(timestamps, decoded):
            future_anchors = future[timestamp].get("anchors", [])
            if len(future_anchors) != len(horizons):
                continue
            past_anchors = sorted(past[timestamp].get("anchors", []), key=lambda row: int(row["past_horizon_ms"]))
            causal_xy = []
            for index in range(3):
                causal_xy.extend(list(map(float, past_anchors[index]["point_xy_norm"])) if index < len(past_anchors) else [0.5, 0.9])
            rows.append({"event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
                         "target": horizon_fields(future_anchors, side, sigma, horizons),
                         "causal_xy": np.asarray(causal_xy), "detections": samples[timestamp].get("detections", [])})
            images.append(image)
    tokens = spatial.extract_patch_tokens(args.model_dir, images, args.batch_size)
    projection = spatial.fixed_projection(tokens.shape[-1], int(contract["input_and_probe"]["fixed_random_projection_dimension"]),
                                          int(contract["input_and_probe"]["fixed_random_projection_seed"]))
    projected = tokens @ projection
    yy, xx = np.mgrid[0:side, 0:side]
    coordinates = np.stack([(xx + 0.5) / side, (yy + 0.5) / side], axis=-1)
    frame_features = np.stack([np.concatenate([
        grid, coordinates, np.broadcast_to(row["causal_xy"], (side, side, len(row["causal_xy"])))
    ], axis=-1) for grid, row in zip(projected, rows)])
    sources = sorted({row["source_id"] for row in rows})
    predictions = np.zeros((len(rows), len(horizons), side, side), dtype=np.float64)
    alpha = float(contract["input_and_probe"]["ridge_alpha"])
    folds = []
    for source in sources:
        train_frames = [i for i, row in enumerate(rows) if row["source_id"] != source]
        test_frames = [i for i, row in enumerate(rows) if row["source_id"] == source]
        train_x = frame_features[train_frames].reshape(-1, frame_features.shape[-1])
        test_x = frame_features[test_frames].reshape(-1, frame_features.shape[-1])
        source_counts = {value: sum(rows[i]["source_id"] == value for i in train_frames) for value in sources if value != source}
        base_source_weight = np.asarray([1.0 / source_counts[rows[i]["source_id"]] for i in train_frames])
        for channel in range(len(horizons)):
            train_y = np.stack([rows[i]["target"][channel] for i in train_frames]).reshape(-1)
            weights = np.repeat(base_source_weight, side * side) * (1.0 + 4.0 * train_y)
            predictions[test_frames, channel] = spatial.weighted_ridge(train_x, train_y, weights, test_x, alpha).reshape(len(test_frames), side, side)
        folds.append({"held_out_source_id": source, "train_frame_count": len(train_frames), "test_frame_count": len(test_frames)})
    targets = np.stack([row["target"] for row in rows])
    route_band = (targets >= np.exp(-0.5)).astype(np.int64)
    pixel_auroc = base.binary_auroc(route_band.reshape(-1), predictions.reshape(-1))
    expansion = float(contract["event_readout"]["marker_expansion_object_heights"])
    event_scores: dict[str, list[float]] = defaultdict(list)
    localization = []
    for row, predicted, target in zip(rows, predictions, targets):
        hits = []
        for channel in range(len(horizons)):
            point = argmax_point(predicted[channel])
            hits.append(base.point_box_distance(point, row["detections"], expansion) <= 1e-12)
            predicted_point = np.asarray(point)
            target_point = np.asarray(argmax_point(target[channel]))
            localization.append(float(np.linalg.norm(predicted_point - target_point)))
        event_scores[row["event_id"]].append(sum(hits) / len(hits))
    event_predictions = [{"event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
                          "predicted_horizon_hit_fraction": float(np.mean(event_scores[event["event_id"]]))}
                         for event in event_contract["events"]]
    positive = [row["predicted_horizon_hit_fraction"] for row in event_predictions if row["label"] == 1]
    negative = [row["predicted_horizon_hit_fraction"] for row in event_predictions if row["label"] == 0]
    checks = {"route_band_pixel_auroc": pixel_auroc >= float(contract["gate"]["route_band_pixel_auroc_at_least"]),
              "strict_event_label_separation": min(positive) > max(negative)}
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "offline_teacher_report_sha256": common.sha256_file(args.offline_teacher_report),
                   "dinov2_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin")},
        "frame_count": len(rows), "source_count": len(sources), "spatial_feature_dimension": int(frame_features.shape[-1]),
        "folds": folds, "route_band_pixel_auroc": pixel_auroc,
        "mean_argmax_localization_error_norm": float(np.mean(localization)),
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
