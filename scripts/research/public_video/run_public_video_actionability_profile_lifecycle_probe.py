#!/usr/bin/env python3
"""Train a frame actionability profile and evaluate frozen lifecycle aggregation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import extract_public_video_temporal_route_features as temporal
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_actionability_linear_probe as event_mean
import run_public_video_causal_waypoint_linear_probe as waypoint
import run_public_video_event_route_role_linear_probe as event_probe
import run_public_video_marker_relation_linear_probe as relation
import run_public_video_ego_route_distance_field_probe as spatial
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_actionability_profile_lifecycle_probe_v1"


def frame_targets(timestamps: list[int], transitions: list[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(transitions, key=lambda row: int(row["timestamp_ms"]))
    result = []
    state = 0
    index = 0
    for timestamp in timestamps:
        while index < len(ordered) and int(ordered[index]["timestamp_ms"]) <= timestamp:
            name = ordered[index]["state"]
            if name == "intervention_needed":
                state = 1
            elif name == "route_clear":
                state = 0
            else:
                raise ValueError(f"unsupported transition: {name}")
            index += 1
        result.append(state)
    return np.asarray(result, dtype=np.int64)


def event_lifecycle_predictions(
    scores: np.ndarray, labels: np.ndarray, event_ids: np.ndarray, sources: np.ndarray,
    timestamps: np.ndarray, threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, float, int, str]]] = defaultdict(list)
    for score, label, event_id, source, timestamp in zip(scores, labels, event_ids, sources, timestamps):
        grouped[str(event_id)].append((int(timestamp), float(score), int(label), str(source)))
    rows = []
    for event_id in sorted(grouped):
        values = sorted(grouped[event_id])
        event_labels = {row[2] for row in values}
        event_sources = {row[3] for row in values}
        if len(event_labels) != 1 or len(event_sources) != 1:
            raise ValueError(f"inconsistent event metadata: {event_id}")
        best_pair_mean = max(row[1] for row in values)
        first_open = None
        for left, right in zip(values, values[1:]):
            if right[0] - left[0] != 1000:
                continue
            best_pair_mean = max(best_pair_mean, (left[1] + right[1]) / 2.0)
            if first_open is None and left[1] >= threshold and right[1] >= threshold:
                first_open = right[0]
        rows.append({
            "event_id": event_id,
            "source_id": next(iter(event_sources)),
            "label": next(iter(event_labels)),
            "frame_count": len(values),
            "oof_event_score": float(best_pair_mean),
            "predicted_label": int(first_open is not None),
            "first_open_timestamp_ms": first_open,
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.r790_contract, args.r790_report, args.manifest,
                 args.model_dir, args.output_cache, args.output_report):
        mil.reject_independent_direction(path)
    if args.output_cache.exists() or args.output_report.exists() or Path(str(args.output_report) + ".sha256").exists():
        raise ValueError("refusing to overwrite profile lifecycle outputs")
    contract = common.load_json(args.contract)
    bindings = {
        args.manifest: "actionability_manifest_sha256",
        args.r790_contract: "r790_contract_sha256",
        args.r790_report: "r790_failure_report_sha256",
    }
    for path, key in bindings.items():
        if common.sha256_file(path) != contract["bound_inputs"][key]:
            raise ValueError(f"bound input mismatch: {path}")
    manifest = common.load_json(args.manifest)
    r790 = common.load_json(args.r790_contract)
    if common.sha256_file(args.model_dir / "pytorch_model.bin") != r790["bound_inputs"]["dinov2_weights_sha256"]:
        raise ValueError("DINOv2 weight hash mismatch")
    sources: dict[str, dict[str, Any]] = {}
    for binding in r790["feature_reports"].values():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {path}")
        report = common.load_json(path)
        for incoming in report["sources"]:
            source_id = incoming["source_id"]
            if source_id in sources:
                event_mean.merge_source(sources[source_id], incoming)
            else:
                sources[source_id] = {**incoming, "samples": list(incoming["samples"])}
    events = event_mean.build_event_rows(manifest, sources)
    items = {row["item_id"]: row for row in manifest["items"]}
    input_spec = r790["input"]
    side = int(input_spec["grid_side"])
    horizons = list(map(int, input_spec["past_flow_horizons_ms"]))
    projection_spec = input_spec["current_dinov2_patch_projection"]
    projection = spatial.fixed_projection(
        int(projection_spec["input_dimension"]), int(projection_spec["output_dimension"]), int(projection_spec["seed"])
    )
    extractor = temporal.PatchExtractor(args.model_dir)
    vectors: dict[tuple[str, int], np.ndarray] = {}
    fallback_count = 0
    for source_id in sorted({event["source_id"] for event in events}):
        source = sources[source_id]
        source_samples = {
            int(sample["timestamp_ms"]): sample
            for event in events if event["source_id"] == source_id for sample in event["samples"]
        }
        timestamps_for_source = sorted(source_samples)
        decode_times = sorted(set(timestamps_for_source + [
            timestamp - horizon for timestamp in timestamps_for_source for horizon in horizons
        ]))
        decoded = route_width.decode_at(Path(source["local_video_path"]), decode_times)
        frame_by_time = dict(zip(decode_times, decoded))
        images = [frame_by_time[timestamp] for timestamp in timestamps_for_source]
        tokens = extractor.extract(images, args.batch_size) @ projection
        for timestamp, image, token_grid in zip(timestamps_for_source, images, tokens):
            past = [frame_by_time[timestamp - horizon] for horizon in horizons]
            flow_grid = temporal.causal_flow_grid(image, past, side)
            grid = temporal.compose_feature_grid(token_grid, image, flow_grid)
            detections = source_samples[timestamp]["detections"]
            marker = spatial.obstacle_grid_mask(detections, side, float(input_spec["marker_expansion_object_heights"]))
            if not marker.any():
                marker = relation.marker_grid_mask(detections, side, float(input_spec["marker_expansion_object_heights"]))
                fallback_count += 1
            if not marker.any():
                raise ValueError(f"marker mask empty: {source_id} {timestamp}")
            vectors[(source_id, timestamp)] = relation.relation_vector(grid, marker)
    frame_x = []
    frame_y = []
    frame_sources = []
    frame_events = []
    frame_timestamps = []
    event_labels: dict[str, int] = {}
    for event in events:
        timestamps_for_event = [int(sample["timestamp_ms"]) for sample in event["samples"]]
        targets = frame_targets(timestamps_for_event, items[event["event_id"]]["transitions"])
        event_labels[event["event_id"]] = event["label"]
        for timestamp, target in zip(timestamps_for_event, targets):
            frame_x.append(vectors[(event["source_id"], timestamp)])
            frame_y.append(int(target))
            frame_sources.append(event["source_id"])
            frame_events.append(event["event_id"])
            frame_timestamps.append(timestamp)
    x = np.stack(frame_x).astype(np.float64)
    y = np.asarray(frame_y, dtype=np.int64)
    source_values = np.asarray(frame_sources).astype(str)
    event_values = np.asarray(frame_events).astype(str)
    timestamp_values = np.asarray(frame_timestamps, dtype=np.int64)
    oof, folds = event_mean.source_loso_probe(x, y, source_values, float(contract["probe"]["alpha"]))
    event_rows = event_lifecycle_predictions(
        oof,
        np.asarray([event_labels[event_id] for event_id in event_values], dtype=np.int64),
        event_values,
        source_values,
        timestamp_values,
        float(contract["probe"]["frame_decision_threshold"]),
    )
    event_y = np.asarray([row["label"] for row in event_rows], dtype=np.int64)
    event_scores = np.asarray([row["oof_event_score"] for row in event_rows], dtype=np.float64)
    event_pred = np.asarray([row["predicted_label"] for row in event_rows], dtype=np.int64)
    positive_recall = float((event_pred[event_y == 1] == 1).mean())
    negative_recall = float((event_pred[event_y == 0] == 0).mean())
    balanced = (positive_recall + negative_recall) / 2.0
    auroc = relation.roc_auc(event_y, event_scores)
    positive_sources = {row["source_id"] for row in event_rows if row["label"] == 1}
    gate = contract["gate"]
    checks = {
        "minimum_intervention_events": int((event_y == 1).sum()) >= int(gate["minimum_intervention_events"]),
        "minimum_context_events": int((event_y == 0).sum()) >= int(gate["minimum_context_events"]),
        "minimum_intervention_sources": len(positive_sources) >= int(gate["minimum_intervention_sources"]),
        "minimum_positive_profile_frames": int((y == 1).sum()) >= int(gate["minimum_positive_profile_frames"]),
        "event_oof_auroc": auroc >= float(gate["event_oof_auroc_at_least"]),
        "event_balanced_accuracy": balanced >= float(gate["event_balanced_accuracy_at_least"]),
        "intervention_event_recall": positive_recall >= float(gate["intervention_event_recall_at_least"]),
        "context_event_recall": negative_recall >= float(gate["context_event_recall_at_least"]),
        "all_training_folds_have_both_frame_classes": all(row["train_classes"] == [0, 1] for row in folds),
        "all_folds_finite": all(row["finite"] for row in folds),
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, frame_x=x.astype(np.float32), frame_y=y,
                        frame_sources=source_values, frame_events=event_values,
                        frame_timestamps=timestamp_values, oof_scores=oof)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "r790_report_sha256": common.sha256_file(args.r790_report)},
        "cache": {"path": str(args.output_cache), "sha256": common.sha256_file(args.output_cache), "shape": list(x.shape)},
        "frame_count": len(y),
        "positive_profile_frame_count": int((y == 1).sum()),
        "event_count": len(event_rows),
        "source_count": len(set(source_values.tolist())),
        "marker_mask_fallback_count": fallback_count,
        "future_frames_in_input": False,
        "trace_intrusion_score_in_input": False,
        "obstacle_hit_in_input": False,
        "weights_saved": False,
        "folds": folds,
        "event_predictions": event_rows,
        "event_oof_auroc": auroc,
        "event_balanced_accuracy": balanced,
        "intervention_event_recall": positive_recall,
        "context_event_recall": negative_recall,
        "checks": checks,
        "profile_lifecycle_probe_passed": all(checks.values()),
        "evidence_limit": "Source-heldout provisional frame-profile/lifecycle OFAT. Passing would authorize only later short-run research, never calibration, blind, Android, or production changes.",
        "authorization": contract["authorization"],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output_report) + ".sha256").write_text(common.sha256_file(args.output_report) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r790-contract", type=Path, required=True)
    parser.add_argument("--r790-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "passed": value["profile_lifecycle_probe_passed"],
        "event_oof_auroc": value["event_oof_auroc"],
        "balanced_accuracy": value["event_balanced_accuracy"],
        "output_sha256": common.sha256_file(parsed.output_report),
    }))
