#!/usr/bin/env python3
"""Frozen DINOv2 source-isolated probe for the r7.55 route auxiliary target."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_dinov2_regional_pair_probe as dino
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_ego_trace_distillation_probe as base
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_ego_trace_dinov2_probe_v1"


def weighted_dual_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, train_sources: list[str], test_x: np.ndarray, alpha: float) -> np.ndarray:
    counts = {source: train_sources.count(source) for source in set(train_sources)}
    weights = np.asarray([1.0 / counts[source] for source in train_sources], dtype=np.float64)
    weights /= weights.sum()
    mean = np.sum(weights[:, None] * train_x, axis=0)
    centered = train_x - mean
    variance = np.sum(weights[:, None] * centered * centered, axis=0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    x_train = centered / scale
    x_test = (test_x - mean) / scale
    y_mean = float(np.sum(weights * train_y))
    y_centered = train_y - y_mean
    sqrt_w = np.sqrt(weights)
    design = sqrt_w[:, None] * x_train
    target = sqrt_w * y_centered
    dual = np.linalg.solve(design @ design.T + float(alpha) * np.eye(len(design)), target)
    coefficients = design.T @ dual
    return y_mean + x_test @ coefficients


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.distillation_contract, args.distillation_report, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    distill_contract = json.loads(args.distillation_contract.read_text(encoding="utf-8"))
    bound = contract["bound_inputs"]
    bindings = {
        args.distillation_contract: bound["distillation_contract_sha256"],
        args.distillation_report: bound["distillation_report_sha256"],
        args.model_dir / "pytorch_model.bin": bound["dinov2_weights_sha256"],
    }
    for path, expected in bindings.items():
        if common.sha256_file(path) != expected:
            raise ValueError(f"bound input hash mismatch: {path}")
    distill_report = lifecycle.verify_json_sidecar(args.distillation_report)
    event_contract_path = Path(distill_contract["bound_inputs"].get("event_set_contract_path", "configs/public_video_future_ego_trace_multisource_contract_r754.json"))
    event_contract = json.loads(event_contract_path.read_text(encoding="utf-8"))
    teacher_path = Path(args.offline_teacher_report)
    causal_path = Path(args.causal_feature_report)
    teacher = lifecycle.verify_json_sidecar(teacher_path)
    causal = lifecycle.verify_json_sidecar(causal_path)
    if common.sha256_file(teacher_path) != distill_contract["bound_inputs"]["offline_teacher_report_sha256"]:
        raise ValueError("offline teacher report hash mismatch")
    if common.sha256_file(causal_path) != distill_contract["bound_inputs"]["causal_feature_report_sha256"]:
        raise ValueError("causal feature report hash mismatch")
    feature_reports = {}
    for key, binding in event_contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        feature_reports[key] = lifecycle.verify_json_sidecar(path)
    teacher_events = {row["event_id"]: row for row in teacher["events"]}
    causal_events = {row["event_id"]: row for row in causal["events"]}
    rows = []
    images = []
    for event in event_contract["events"]:
        source_report = feature_reports[event["feature_key"]]
        source = next(row for row in source_report["sources"] if row["source_id"] == event["source_id"])
        samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
        teacher_frames = {int(row["timestamp_ms"]): row for row in teacher_events[event["event_id"]]["event_diagnostics"]["frames"]}
        causal_frames = {int(row["timestamp_ms"]): row for row in causal_events[event["event_id"]]["event_diagnostics"]["frames"]}
        timestamps = sorted(set(teacher_frames) & set(causal_frames) & set(samples))
        decoded = route_width.decode_at(Path(source["local_video_path"]), timestamps)
        for timestamp, image in zip(timestamps, decoded):
            target = teacher_frames[timestamp]["trace_intrusion_score"]
            if target is None:
                continue
            rows.append({
                "event_id": event["event_id"], "source_id": event["source_id"], "event_label": int(event["label"]),
                "timestamp_ms": timestamp, "target": float(target),
                "causal_features": base.feature_vector(samples[timestamp], causal_frames[timestamp]),
            })
            images.append(image)
    visual_teacher = dino.FrozenDinoV2(args.model_dir, feature_mode="regional_mean")
    visual = visual_teacher.extract(images, batch_size=args.batch_size)
    features = np.concatenate([visual, np.stack([row["causal_features"] for row in rows])], axis=1)
    sources = sorted({row["source_id"] for row in rows})
    predictions = np.zeros(len(rows), dtype=np.float64)
    alpha = float(contract["probe"]["ridge_alpha"])
    folds = []
    for source in sources:
        train_indices = [i for i, row in enumerate(rows) if row["source_id"] != source]
        test_indices = [i for i, row in enumerate(rows) if row["source_id"] == source]
        predictions[test_indices] = weighted_dual_ridge_predict(
            features[train_indices], np.asarray([rows[i]["target"] for i in train_indices]),
            [rows[i]["source_id"] for i in train_indices], features[test_indices], alpha,
        )
        folds.append({"held_out_source_id": source, "train_frame_count": len(train_indices), "test_frame_count": len(test_indices)})
    target = np.asarray([row["target"] for row in rows])
    auroc = base.binary_auroc((target > 0.0).astype(np.int64), predictions)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        grouped[row["event_id"]].append(float(prediction))
    event_predictions = [{"event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
                          "predicted_mean": float(np.mean(grouped[event["event_id"]]))} for event in event_contract["events"]]
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
                   "distillation_report_sha256": common.sha256_file(args.distillation_report),
                   "dinov2_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin")},
        "frame_count": len(rows), "source_count": len(sources), "visual_feature_dimension": int(visual.shape[1]),
        "combined_feature_dimension": int(features.shape[1]), "folds": folds,
        "teacher_active_frame_auroc": auroc,
        "mean_absolute_error": float(np.mean(np.abs(predictions - target))),
        "event_predictions": event_predictions,
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "evidence_limit": contract["evidence_role"], "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--distillation-contract", type=Path, required=True)
    parser.add_argument("--distillation-report", type=Path, required=True)
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
                      "teacher_active_frame_auroc": value["teacher_active_frame_auroc"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
