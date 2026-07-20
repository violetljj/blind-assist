#!/usr/bin/env python3
"""Source-isolated deterministic ridge probe for the r7.55 route auxiliary target."""

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


SCHEMA = "blindassist_public_video_ego_trace_distillation_probe_v1"
FEATURE_NAMES = [
    "causal_score", "anchor1_x", "anchor1_y", "anchor1_marker_distance",
    "anchor2_x", "anchor2_y", "anchor2_marker_distance",
    "anchor3_x", "anchor3_y", "anchor3_marker_distance",
    "detection_count", "accepted_count", "max_bottom", "max_area", "max_width", "max_height",
    "min_center_distance", "left_count", "right_count", "center_marker_distance",
]


def point_box_distance(point: tuple[float, float], detections: list[dict[str, Any]], expansion: float = 0.5) -> float:
    if not detections:
        return 1.0
    distances = []
    for row in detections:
        features = row["features"]
        center_x = float(features["center_x_norm"])
        bottom = float(features["bottom_y_norm"])
        width = float(features["width_norm"])
        height = float(features["height_norm"])
        x1 = center_x - width / 2.0 - expansion * height
        x2 = center_x + width / 2.0 + expansion * height
        y1 = bottom - height - expansion * height
        y2 = bottom + expansion * height
        dx = max(x1 - point[0], 0.0, point[0] - x2)
        dy = max(y1 - point[1], 0.0, point[1] - y2)
        distances.append(float(np.hypot(dx, dy)))
    return min(distances)


def feature_vector(sample: dict[str, Any], causal_frame: dict[str, Any]) -> np.ndarray:
    detections = sample.get("detections", [])
    anchors = sorted(causal_frame.get("anchors", []), key=lambda row: int(row["past_horizon_ms"]))
    values = [float(causal_frame.get("trace_intrusion_score") or 0.0)]
    for index in range(3):
        if index < len(anchors):
            point = tuple(map(float, anchors[index]["point_xy_norm"]))
            values.extend([point[0], point[1], point_box_distance(point, detections)])
        else:
            values.extend([0.5, 0.9, 1.0])
    accepted = [row for row in detections if row.get("features", {}).get("nearfield_corridor_accepted") is True]
    all_features = [row["features"] for row in detections]
    values.extend([
        float(len(detections)),
        float(len(accepted)),
        max((float(row["bottom_y_norm"]) for row in all_features), default=0.0),
        max((float(row["area_ratio"]) for row in all_features), default=0.0),
        max((float(row["width_norm"]) for row in all_features), default=0.0),
        max((float(row["height_norm"]) for row in all_features), default=0.0),
        min((abs(float(row["center_x_norm"]) - 0.5) for row in all_features), default=1.0),
        float(sum(float(row["center_x_norm"]) < 0.5 for row in all_features)),
        float(sum(float(row["center_x_norm"]) >= 0.5 for row in all_features)),
        point_box_distance((0.5, 0.9), detections),
    ])
    return np.asarray(values, dtype=np.float64)


def weighted_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, train_sources: list[str], test_x: np.ndarray, alpha: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    counts = {source: train_sources.count(source) for source in set(train_sources)}
    weights = np.asarray([1.0 / counts[source] for source in train_sources], dtype=np.float64)
    weights /= weights.sum()
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x_train.T @ (weights[:, None] * x_train) + penalty,
                                   x_train.T @ (weights * train_y))
    return x_test @ coefficients


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.0
    comparisons = [(float(p > n) + 0.5 * float(p == n)) for p in positive for n in negative]
    return float(np.mean(comparisons))


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.event_set_contract, args.offline_teacher_report, args.causal_feature_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    event_contract = json.loads(args.event_set_contract.read_text(encoding="utf-8"))
    bound = contract["bound_inputs"]
    for path, key in ((args.event_set_contract, "event_set_contract_sha256"),
                      (args.offline_teacher_report, "offline_teacher_report_sha256"),
                      (args.causal_feature_report, "causal_feature_report_sha256")):
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
    rows = []
    for event in event_contract["events"]:
        source_report = feature_reports[event["feature_key"]]
        source = next(row for row in source_report["sources"] if row["source_id"] == event["source_id"])
        samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
        teacher_frames = {int(row["timestamp_ms"]): row for row in teacher_events[event["event_id"]]["event_diagnostics"]["frames"]}
        causal_frames = {int(row["timestamp_ms"]): row for row in causal_events[event["event_id"]]["event_diagnostics"]["frames"]}
        for timestamp in sorted(set(teacher_frames) & set(causal_frames) & set(samples)):
            target = teacher_frames[timestamp]["trace_intrusion_score"]
            if target is None:
                continue
            rows.append({
                "event_id": event["event_id"], "source_id": event["source_id"], "event_label": int(event["label"]),
                "timestamp_ms": timestamp, "target": float(target),
                "features": feature_vector(samples[timestamp], causal_frames[timestamp]),
            })
    sources = sorted({row["source_id"] for row in rows})
    alpha = float(contract["probe"]["ridge_alpha"])
    predictions = np.zeros(len(rows), dtype=np.float64)
    folds = []
    for source in sources:
        train_indices = [i for i, row in enumerate(rows) if row["source_id"] != source]
        test_indices = [i for i, row in enumerate(rows) if row["source_id"] == source]
        train_x = np.stack([rows[i]["features"] for i in train_indices])
        train_y = np.asarray([rows[i]["target"] for i in train_indices])
        test_x = np.stack([rows[i]["features"] for i in test_indices])
        fold_predictions = weighted_ridge_predict(train_x, train_y, [rows[i]["source_id"] for i in train_indices], test_x, alpha)
        predictions[test_indices] = fold_predictions
        folds.append({"held_out_source_id": source, "train_frame_count": len(train_indices), "test_frame_count": len(test_indices)})
    target = np.asarray([row["target"] for row in rows])
    active = (target > 0.0).astype(np.int64)
    auroc = binary_auroc(active, predictions)
    event_values: dict[str, list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        event_values[row["event_id"]].append(float(prediction))
    event_predictions = []
    for event in event_contract["events"]:
        values = event_values[event["event_id"]]
        event_predictions.append({"event_id": event["event_id"], "source_id": event["source_id"],
                                  "label": int(event["label"]), "predicted_mean": float(np.mean(values))})
    positive = [row["predicted_mean"] for row in event_predictions if row["label"] == 1]
    negative = [row["predicted_mean"] for row in event_predictions if row["label"] == 0]
    checks = {
        "teacher_active_frame_auroc": auroc >= float(contract["gate"]["teacher_active_frame_auroc_at_least"]),
        "strict_event_label_separation": min(positive) > max(negative),
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "event_set_contract_sha256": common.sha256_file(args.event_set_contract),
                   "offline_teacher_report_sha256": common.sha256_file(args.offline_teacher_report),
                   "causal_feature_report_sha256": common.sha256_file(args.causal_feature_report)},
        "feature_names": FEATURE_NAMES,
        "frame_count": len(rows),
        "source_count": len(sources),
        "folds": folds,
        "teacher_active_frame_auroc": auroc,
        "mean_absolute_error": float(np.mean(np.abs(predictions - target))),
        "event_predictions": event_predictions,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "evidence_limit": contract["evidence_role"],
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "diagnostic_gate_passed": value["diagnostic_gate_passed"],
                      "teacher_active_frame_auroc": value["teacher_active_frame_auroc"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
