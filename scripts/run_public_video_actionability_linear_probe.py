#!/usr/bin/env python3
"""Run a source-heldout actionability probe without causal-label leakage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import extract_public_video_temporal_route_features as temporal
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_causal_waypoint_linear_probe as waypoint
import run_public_video_event_route_role_linear_probe as event_probe
import run_public_video_marker_relation_linear_probe as relation
import run_public_video_ego_route_distance_field_probe as spatial
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_actionability_linear_probe_v1"


def merge_source(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for field in ("source_id", "local_video_path", "video_sha256"):
        if target.get(field) != incoming.get(field):
            raise ValueError(f"duplicate source metadata mismatch: {incoming.get('source_id')} {field}")
    by_time = {int(row["timestamp_ms"]): row for row in target["samples"]}
    for sample in incoming["samples"]:
        timestamp = int(sample["timestamp_ms"])
        if timestamp in by_time and by_time[timestamp] != sample:
            raise ValueError(f"duplicate source sample mismatch: {incoming['source_id']} {timestamp}")
        by_time[timestamp] = sample
    target["samples"] = [by_time[key] for key in sorted(by_time)]


def build_event_rows(manifest: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in manifest["items"]:
        source_id = item["parent_source_id"]
        if source_id not in sources:
            raise ValueError(f"missing source features: {source_id}")
        start, end = map(int, item["window_ms"])
        samples = [row for row in sources[source_id]["samples"]
                   if start <= int(row["timestamp_ms"]) < end and row.get("detections")]
        if not samples:
            continue
        rows.append({
            "event_id": item["item_id"],
            "source_id": source_id,
            "label": int(bool(item["intervention_required"])),
            "samples": samples,
        })
    return rows


def source_loso_probe(x: np.ndarray, y: np.ndarray, sources: np.ndarray, alpha: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    oof = np.zeros(len(x), dtype=np.float64)
    folds = []
    for held_out in sorted(set(sources.tolist())):
        train = np.flatnonzero(sources != held_out)
        test = np.flatnonzero(sources == held_out)
        classes = sorted(set(y[train].tolist()))
        if classes != [0, 1]:
            raise ValueError(f"training fold lacks a class: {held_out}")
        weights = event_probe.class_source_balanced_weights(sources[train], y[train])
        model = waypoint.fit_ridge(x[train], y[train, None].astype(np.float64), weights, alpha)
        oof[test] = waypoint.predict(model, x[test])[:, 0]
        folds.append({
            "held_out_source_id": held_out,
            "train_event_count": len(train),
            "test_event_count": len(test),
            "train_classes": classes,
            "finite": bool(np.isfinite(oof[test]).all()),
        })
    return oof, folds


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.manifest, args.model_dir, args.output_cache, args.output_report):
        mil.reject_independent_direction(path)
    if args.output_cache.exists() or args.output_report.exists() or Path(str(args.output_report) + ".sha256").exists():
        raise ValueError("refusing to overwrite actionability probe outputs")
    contract = common.load_json(args.contract)
    if common.sha256_file(args.manifest) != contract["bound_inputs"]["actionability_manifest_sha256"]:
        raise ValueError("actionability manifest hash mismatch")
    if common.sha256_file(args.model_dir / "pytorch_model.bin") != contract["bound_inputs"]["dinov2_weights_sha256"]:
        raise ValueError("DINOv2 weight hash mismatch")
    manifest = common.load_json(args.manifest)
    if manifest.get("deterministic_actionability_probe_ready") is not True:
        raise ValueError("actionability manifest did not pass coverage")
    sources: dict[str, dict[str, Any]] = {}
    for binding in contract["feature_reports"].values():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {path}")
        report = common.load_json(path)
        for incoming in report["sources"]:
            source_id = incoming["source_id"]
            if source_id in sources:
                merge_source(sources[source_id], incoming)
            else:
                sources[source_id] = {**incoming, "samples": list(incoming["samples"])}
    events = build_event_rows(manifest, sources)
    input_spec = contract["input"]
    side = int(input_spec["grid_side"])
    horizons = list(map(int, input_spec["past_flow_horizons_ms"]))
    projection_spec = input_spec["current_dinov2_patch_projection"]
    projection = spatial.fixed_projection(
        int(projection_spec["input_dimension"]), int(projection_spec["output_dimension"]), int(projection_spec["seed"])
    )
    extractor = temporal.PatchExtractor(args.model_dir)
    sample_vectors: dict[tuple[str, int], np.ndarray] = {}
    fallback_count = 0
    for source_id in sorted({event["source_id"] for event in events}):
        source = sources[source_id]
        source_samples = {
            int(sample["timestamp_ms"]): sample
            for event in events if event["source_id"] == source_id for sample in event["samples"]
        }
        timestamps = sorted(source_samples)
        decode_times = sorted(set(timestamps + [timestamp - horizon for timestamp in timestamps for horizon in horizons]))
        if decode_times[0] < 0:
            raise ValueError(f"causal history precedes source start: {source_id}")
        decoded = route_width.decode_at(Path(source["local_video_path"]), decode_times)
        frame_by_time = dict(zip(decode_times, decoded))
        images = [frame_by_time[timestamp] for timestamp in timestamps]
        tokens = extractor.extract(images, args.batch_size) @ projection
        for timestamp, image, token_grid in zip(timestamps, images, tokens):
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
            sample_vectors[(source_id, timestamp)] = relation.relation_vector(grid, marker)
    event_x = []
    event_y = []
    event_sources = []
    event_ids = []
    frame_counts = []
    for event in events:
        vectors = [sample_vectors[(event["source_id"], int(sample["timestamp_ms"]))] for sample in event["samples"]]
        event_x.append(np.mean(np.stack(vectors), axis=0))
        event_y.append(event["label"])
        event_sources.append(event["source_id"])
        event_ids.append(event["event_id"])
        frame_counts.append(len(vectors))
    x = np.stack(event_x).astype(np.float64)
    y = np.asarray(event_y, dtype=np.int64)
    source_values = np.asarray(event_sources).astype(str)
    identifiers = np.asarray(event_ids).astype(str)
    oof, folds = source_loso_probe(x, y, source_values, float(contract["probe"]["alpha"]))
    threshold = float(contract["probe"]["decision_threshold"])
    predicted = (oof >= threshold).astype(np.int64)
    positive_recall = float((predicted[y == 1] == 1).mean())
    negative_recall = float((predicted[y == 0] == 0).mean())
    balanced = (positive_recall + negative_recall) / 2.0
    auroc = relation.roc_auc(y, oof)
    predictions = [{
        "event_id": str(event_id), "source_id": str(source_id), "label": int(label),
        "frame_count": int(frame_count), "oof_score": float(score), "predicted_label": int(prediction),
    } for event_id, source_id, label, frame_count, score, prediction
      in zip(identifiers, source_values, y, frame_counts, oof, predicted)]
    retained_positive_sources = set(source_values[y == 1].tolist())
    retained_negative_sources = set(source_values[y == 0].tolist())
    gate = contract["gate"]
    checks = {
        "minimum_retained_intervention_events": int((y == 1).sum()) >= int(gate["minimum_retained_intervention_events"]),
        "minimum_retained_context_events": int((y == 0).sum()) >= int(gate["minimum_retained_context_events"]),
        "minimum_retained_intervention_sources": len(retained_positive_sources) >= int(gate["minimum_retained_intervention_sources"]),
        "event_oof_auroc": auroc >= float(gate["event_oof_auroc_at_least"]),
        "event_balanced_accuracy": balanced >= float(gate["event_balanced_accuracy_at_least"]),
        "intervention_recall": positive_recall >= float(gate["intervention_recall_at_least"]),
        "context_recall": negative_recall >= float(gate["context_recall_at_least"]),
        "every_training_fold_contains_both_classes": all(row["train_classes"] == [0, 1] for row in folds),
        "all_folds_finite": all(row["finite"] for row in folds),
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, event_x=x.astype(np.float32), event_y=y,
                        event_sources=source_values, event_ids=identifiers, oof_scores=oof)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "manifest_sha256": common.sha256_file(args.manifest)},
        "cache": {"path": str(args.output_cache), "sha256": common.sha256_file(args.output_cache), "shape": list(x.shape)},
        "event_count": len(y),
        "source_count": len(set(source_values.tolist())),
        "intervention_event_count": int((y == 1).sum()),
        "context_event_count": int((y == 0).sum()),
        "intervention_source_count": len(retained_positive_sources),
        "marker_mask_fallback_count": fallback_count,
        "feature_dimension": int(x.shape[1]),
        "future_frames_in_input": False,
        "trace_intrusion_score_in_input": False,
        "obstacle_hit_in_input": False,
        "weights_saved": False,
        "folds": folds,
        "event_predictions": predictions,
        "event_oof_auroc": auroc,
        "event_balanced_accuracy": balanced,
        "intervention_recall": positive_recall,
        "context_recall": negative_recall,
        "checks": checks,
        "deterministic_actionability_probe_passed": all(checks.values()),
        "evidence_limit": "Source-heldout provisional actionability representation probe. It does not authorize optimized training, calibration, blind evaluation, Android changes, or production replacement.",
        "authorization": contract["authorization"],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output_report) + ".sha256").write_text(common.sha256_file(args.output_report) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
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
        "passed": value["deterministic_actionability_probe_passed"],
        "event_oof_auroc": value["event_oof_auroc"],
        "balanced_accuracy": value["event_balanced_accuracy"],
        "output_sha256": common.sha256_file(parsed.output_report),
    }))
