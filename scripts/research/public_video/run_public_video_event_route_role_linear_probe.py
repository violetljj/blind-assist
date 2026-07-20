#!/usr/bin/env python3
"""Probe frozen causal marker features against event-level route-role silver."""

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
import run_public_video_causal_waypoint_linear_probe as waypoint
import run_public_video_marker_relation_linear_probe as relation


SCHEMA = "blindassist_public_video_event_route_role_linear_probe_v1"


def marker_vectors(grids: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    if len(grids) != len(obstacles):
        raise ValueError("grid and obstacle row counts differ")
    return np.stack([relation.relation_vector(grid, obstacle) for grid, obstacle in zip(grids, obstacles)])


def marker_vectors_with_fallback(
    grids: np.ndarray, obstacles: np.ndarray, sources: np.ndarray, timestamps: np.ndarray,
    detection_index: dict[tuple[str, int], list[dict[str, Any]]], expansion: float
) -> tuple[np.ndarray, np.ndarray, int]:
    masks = np.asarray(obstacles, dtype=bool).copy()
    fallback_count = 0
    eligible: list[int] = []
    for index in range(len(masks)):
        if masks[index].any():
            eligible.append(index)
            continue
        key = (str(sources[index]), int(timestamps[index]))
        detections = detection_index.get(key, [])
        if not detections:
            continue
        masks[index] = relation.marker_grid_mask(detections, masks.shape[-1], expansion)
        if not masks[index].any():
            raise ValueError(f"marker mask remains empty after frozen fallback: {key}")
        fallback_count += 1
        eligible.append(index)
    indices = np.asarray(eligible, dtype=np.int64)
    if not len(indices):
        raise ValueError("no marker-present event rows")
    return marker_vectors(grids[indices], masks[indices]), indices, fallback_count


def class_source_balanced_weights(sources: np.ndarray, labels: np.ndarray) -> np.ndarray:
    source_values = np.asarray(sources).astype(str)
    y = np.asarray(labels).astype(np.int64)
    if source_values.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError("balanced weighting requires aligned binary classes")
    weights = np.zeros(len(y), dtype=np.float64)
    for label in (0, 1):
        class_sources = sorted(set(source_values[y == label].tolist()))
        for source in class_sources:
            indices = np.flatnonzero((y == label) & (source_values == source))
            weights[indices] = len(y) * 0.5 / (len(class_sources) * len(indices))
    if not np.isclose(weights.sum(), len(y)):
        raise ValueError("class-source weights have wrong total mass")
    return weights


def aggregate_events(
    scores: np.ndarray, events: np.ndarray, labels: np.ndarray, sources: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    grouped_sources: dict[str, set[str]] = defaultdict(set)
    for score, event, label, source in zip(scores, events, labels, sources):
        grouped_scores[str(event)].append(float(score))
        grouped_labels[str(event)].add(int(label))
        grouped_sources[str(event)].add(str(source))
    rows = []
    for event in sorted(grouped_scores):
        if len(grouped_labels[event]) != 1 or len(grouped_sources[event]) != 1:
            raise ValueError("event role or source is inconsistent")
        score = float(np.mean(grouped_scores[event]))
        rows.append({"event_id": event, "source_id": next(iter(grouped_sources[event])),
                     "label": next(iter(grouped_labels[event])), "frame_count": len(grouped_scores[event]),
                     "oof_event_score": score, "predicted_label": int(score >= threshold)})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.cache_report, args.cache, args.r779_report, args.r780_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.cache_report, "r764_feature_cache_report_sha256"),
                      (args.cache, "r764_feature_cache_sha256"),
                      (args.r779_report, "r779_report_sha256"),
                      (args.r780_report, "r780_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    cache_report = lifecycle.verify_json_sidecar(args.cache_report)
    if cache_report["cache"]["sha256"] != common.sha256_file(args.cache):
        raise ValueError("cache report does not bind cache")
    cache = np.load(args.cache)
    linear_contract = common.load_json(args.linear_contract)
    source_contract = relation._resolve(linear_contract["bound_inputs"]["r754_source_contract_path"])
    detection_index = relation._load_detection_index(source_contract)
    expansion = float(contract["feature_vector"]["marker_expansion_object_heights"])
    x, eligible, fallback_count = marker_vectors_with_fallback(
        cache["eval_x"], cache["eval_obstacles"], cache["eval_sources"],
        cache["eval_timestamps"], detection_index, expansion)
    y = cache["eval_labels"][eligible].astype(np.int64)
    sources = cache["eval_sources"][eligible].astype(str)
    events = cache["eval_events"][eligible].astype(str)
    missing_events = sorted(set(cache["eval_events"].astype(str).tolist()) - set(events.tolist()))
    unique_sources = sorted(set(sources.tolist()))
    oof = np.zeros(len(x), dtype=np.float64)
    folds = []
    alpha = float(contract["probe"]["alpha"])
    for held_out in unique_sources:
        train = np.flatnonzero(sources != held_out)
        test = np.flatnonzero(sources == held_out)
        classes = sorted(set(y[train].tolist()))
        if classes != [0, 1]:
            raise ValueError(f"training fold lacks a class: {held_out}")
        weights = class_source_balanced_weights(sources[train], y[train])
        model = waypoint.fit_ridge(x[train], y[train, None].astype(np.float64), weights, alpha)
        oof[test] = waypoint.predict(model, x[test])[:, 0]
        folds.append({"held_out_source_id": held_out, "train_count": len(train),
                      "test_count": len(test), "train_classes": classes,
                      "finite": bool(np.isfinite(oof[test]).all())})
    threshold = float(contract["probe"]["decision_threshold"])
    event_rows = aggregate_events(oof, events, y, sources, threshold)
    event_y = np.asarray([row["label"] for row in event_rows])
    event_scores = np.asarray([row["oof_event_score"] for row in event_rows])
    event_pred = np.asarray([row["predicted_label"] for row in event_rows])
    positive_recall = float((event_pred[event_y == 1] == 1).mean())
    negative_recall = float((event_pred[event_y == 0] == 0).mean())
    balanced = (positive_recall + negative_recall) / 2.0
    auroc = relation.roc_auc(event_y, event_scores)
    gate = contract["gate"]
    checks = {
        "event_oof_auroc": auroc >= float(gate["event_oof_auroc_at_least"]),
        "event_balanced_accuracy": balanced >= float(gate["event_balanced_accuracy_at_least"]),
        "positive_event_recall": positive_recall >= float(gate["positive_event_recall_at_least"]),
        "negative_event_recall": negative_recall >= float(gate["negative_event_recall_at_least"]),
        "every_training_fold_contains_both_classes": all(row["train_classes"] == [0, 1] for row in folds),
        "all_folds_finite": all(row["finite"] for row in folds),
        "minimum_retained_positive_events": int((event_y == 1).sum()) >= int(gate.get("minimum_retained_positive_events", 0)),
        "minimum_retained_negative_events": int((event_y == 0).sum()) >= int(gate.get("minimum_retained_negative_events", 0)),
        "minimum_retained_sources": len(unique_sources) >= int(gate.get("minimum_retained_sources", 0)),
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "cache_sha256": common.sha256_file(args.cache)},
        "source_count": len(unique_sources), "event_count": len(event_rows), "frame_count": len(x),
        "feature_dimension": x.shape[1], "nearest_patch_fallback_row_count": fallback_count,
        "excluded_marker_absent_frame_count": int(len(cache["eval_x"]) - len(eligible)),
        "excluded_no_marker_event_ids": missing_events,
        "weights_saved": False,
        "folds": folds, "event_predictions": event_rows,
        "event_oof_auroc": auroc, "event_balanced_accuracy": balanced,
        "positive_event_recall": positive_recall, "negative_event_recall": negative_recall,
        "checks": checks, "diagnostic_gate_passed": all(checks.values()),
        "evidence_limit": "Train-only provisional VLM event-role probe; no independent, human-truth, calibration, Android, blind, or production authority.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--r779-report", type=Path, required=True)
    parser.add_argument("--r780-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, "event_auroc": value["event_oof_auroc"],
                      "balanced_accuracy": value["event_balanced_accuracy"],
                      "passed": value["diagnostic_gate_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
