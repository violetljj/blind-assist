#!/usr/bin/env python3
"""Evaluate a source-heldout real-frame route-risk profile with fixed lifecycle decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_synthetic_mask_teacher_probe as teacher_probe
import run_public_video_route_conditioned_real_transfer_probe as r819
import run_public_video_route_conditioned_synthetic_probe as synthetic_probe
import sanpo_depth_anything_linear_probe as depth_probe


SCHEMA = "blindassist_public_video_route_risk_lifecycle_real_probe_v1"


def state_labels(frames: Sequence[dict[str, Any]], transitions: Sequence[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(transitions, key=lambda row: int(row["timestamp_ms"]))
    labels: list[int] = []
    state = 0
    cursor = 0
    for frame in frames:
        timestamp = int(frame["timestamp_ms"])
        while cursor < len(ordered) and int(ordered[cursor]["timestamp_ms"]) <= timestamp:
            value = ordered[cursor]["state"]
            if value not in ("intervention_needed", "route_clear"):
                raise ValueError(f"unsupported lifecycle state: {value}")
            state = int(value == "intervention_needed")
            cursor += 1
        labels.append(state)
    return np.asarray(labels, dtype=np.int64)


def hierarchical_weights(labels: np.ndarray, sources: np.ndarray, events: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    source_values = np.asarray(sources, dtype=str)
    event_values = np.asarray(events, dtype=str)
    if set(y.tolist()) != {0, 1} or not (len(y) == len(source_values) == len(event_values)):
        raise ValueError("hierarchical weights need aligned two-class rows")
    weights = np.zeros(len(y), dtype=np.float64)
    for label in (0, 1):
        label_indices = np.flatnonzero(y == label)
        label_sources = sorted(set(source_values[label_indices].tolist()))
        for source in label_sources:
            source_indices = label_indices[source_values[label_indices] == source]
            source_events = sorted(set(event_values[source_indices].tolist()))
            for event in source_events:
                indices = source_indices[event_values[source_indices] == event]
                weights[indices] = 0.5 / len(label_sources) / len(source_events) / len(indices)
    weights /= weights.mean()
    return weights


def fit_weighted_ridge(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, ridge: float) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(w) or ridge <= 0 or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("weighted ridge inputs are invalid")
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x - mean) ** 2, axis=0, weights=w)
    scale = np.where(np.sqrt(np.maximum(variance, 1e-16)) < 1e-8, 1.0, np.sqrt(np.maximum(variance, 1e-16)))
    z = (x - mean) / scale
    targets = np.eye(2, dtype=np.float64)[y]
    z_mean = np.average(z, axis=0, weights=w)
    target_mean = np.average(targets, axis=0, weights=w)
    root = np.sqrt(w)[:, None]
    centered_x = (z - z_mean) * root
    centered_y = (targets - target_mean) * root
    kernel_z = np.linalg.solve(centered_x.T @ centered_x + ridge * np.eye(x.shape[1]), centered_x.T @ centered_y)
    bias_z = target_mean - z_mean @ kernel_z
    kernel = kernel_z / scale[:, None]
    bias = bias_z - (mean / scale) @ kernel_z
    digest = hashlib.sha256()
    digest.update(np.asarray(kernel, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(bias, dtype="<f8").tobytes(order="C"))
    return {"kernel": kernel, "bias": bias, "coefficient_sha256": digest.hexdigest()}


def first_consecutive(timestamps: Sequence[int], labels: Sequence[int], value: int, count: int, *, after: int | None = None) -> int | None:
    run = 0
    previous: int | None = None
    for timestamp, label in zip(timestamps, labels):
        timestamp = int(timestamp)
        if after is not None and timestamp <= after:
            continue
        if int(label) == value and (previous is None or timestamp - previous == 1000):
            run += 1
        elif int(label) == value:
            run = 1
        else:
            run = 0
        previous = timestamp if int(label) == value else None
        if run >= count:
            return timestamp
    return None


def source_loso(features: np.ndarray, labels: np.ndarray, sources: np.ndarray, events: np.ndarray, ridge: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.full(len(labels), -1, dtype=np.int64)
    folds = []
    for held_out in sorted(set(sources.tolist())):
        train = np.flatnonzero(sources != held_out)
        test = np.flatnonzero(sources == held_out)
        weights = hierarchical_weights(labels[train], sources[train], events[train])
        model = fit_weighted_ridge(features[train], labels[train], weights, ridge)
        logits = features[test] @ model["kernel"] + model["bias"]
        predictions[test] = np.argmax(logits, axis=1)
        folds.append({"held_out_parent_source_id": held_out, "train_frame_count": len(train),
                      "test_frame_count": len(test), "coefficient_sha256": model["coefficient_sha256"]})
    if np.any(predictions < 0):
        raise RuntimeError("frame LOSO left rows unscored")
    return predictions, folds


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.dataset, args.src_root, args.checkpoint, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("refusing to overwrite lifecycle output")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    bound_paths = {
        "synthetic_build_receipt_sha256": args.dataset / "build_receipt.json",
        "synthetic_manual_review_sha256": args.dataset / "qa" / "manual_review.json",
        "checkpoint_sha256": args.checkpoint,
        "route_oracle_sha256": Path(bound["route_oracle_path"]),
        "actionability_manifest_sha256": Path(bound["actionability_manifest_path"]),
        "r790_feature_contract_sha256": Path(bound["r790_feature_contract_path"]),
        "r819_report_sha256": Path(bound["r819_report_path"]),
    }
    for key, path in bound_paths.items():
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    oracle = common.load_json(Path(bound["route_oracle_path"]))
    manifest = common.load_json(Path(bound["actionability_manifest_path"]))
    manifest_by_id = {row["item_id"]: row for row in manifest["items"]}
    feature_contract = common.load_json(Path(bound["r790_feature_contract_path"]))
    sources_by_id, verified_reports = r819.load_sources(feature_contract)
    generation, _route_examples, patch_records = synthetic_probe.load_dataset_records(args.dataset)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    synthetic_paths = sorted({str((args.dataset / row["image_path"]).resolve()) for row in generation})
    with torch.no_grad():
        synthetic_maps = {path: teacher_probe.extract_dino_map(model, path, input_size=args.input_size, layer_index=args.layer_index)
                          for path in synthetic_paths}
        teacher = synthetic_probe.fit_distance_teacher(patch_records, synthetic_maps, ridge=args.teacher_ridge,
                                                       sigma_patches=args.distance_sigma_patches)
        feature_rows: list[np.ndarray] = []
        label_rows: list[int] = []
        source_rows: list[str] = []
        event_rows: list[str] = []
        timestamp_rows: list[int] = []
        for event in oracle["events"]:
            event_id = str(event["item_id"])
            source_id = str(event["parent_source_id"])
            frames = [row for row in event["frames"] if int(row.get("valid_anchor_count", 0)) == 3]
            timestamps = [int(row["timestamp_ms"]) for row in frames]
            source = sources_by_id[source_id]
            video = Path(source["local_video_path"])
            if common.sha256_file(video) != source["video_sha256"]:
                raise ValueError(f"video hash mismatch: {source_id}")
            images = r819.decode_at(video, timestamps)
            labels = state_labels(frames, manifest_by_id[event_id]["transitions"])
            for frame, image, label in zip(frames, images, labels):
                fmap = r819.extract_frame_map(model, image, input_size=args.input_size, layer_index=args.layer_index)
                score_map = synthetic_probe.distance_score_map(fmap, teacher)
                waypoints = [anchor["point_xy_norm"] for anchor in frame["anchors"]]
                feature_rows.append(synthetic_probe.route_conditioned_risk_features(score_map, waypoints))
                label_rows.append(int(label)); source_rows.append(source_id); event_rows.append(event_id)
                timestamp_rows.append(int(frame["timestamp_ms"]))
    x = np.stack(feature_rows)
    y = np.asarray(label_rows, dtype=np.int64)
    source_values = np.asarray(source_rows, dtype=str)
    event_values = np.asarray(event_rows, dtype=str)
    timestamps = np.asarray(timestamp_rows, dtype=np.int64)
    predictions, folds = source_loso(x, y, source_values, event_values, args.head_ridge)
    repeat_predictions, repeat_folds = source_loso(x, y, source_values, event_values, args.head_ridge)
    repeat_exact = predictions.tolist() == repeat_predictions.tolist() and folds == repeat_folds

    lifecycle_rows = []
    event_truth = []
    event_predictions = []
    clear_required = 0
    clear_found = 0
    for event in oracle["events"]:
        event_id = str(event["item_id"])
        indices = np.flatnonzero(event_values == event_id)
        order = indices[np.argsort(timestamps[indices])]
        times = timestamps[order].tolist()
        predicted_states = predictions[order].tolist()
        open_time = first_consecutive(times, predicted_states, 1, int(contract["lifecycle"]["open_consecutive_frames"]))
        clear_time = first_consecutive(times, predicted_states, 0, int(contract["lifecycle"]["clear_consecutive_frames"]), after=open_time) if open_time is not None else None
        item = manifest_by_id[event_id]
        truth = int(bool(item["intervention_required"]))
        event_truth.append(truth); event_predictions.append(int(open_time is not None))
        has_clear = any(row["state"] == "route_clear" for row in item["transitions"])
        clear_required += int(has_clear)
        clear_found += int(has_clear and clear_time is not None)
        lifecycle_rows.append({"event_id": event_id, "parent_source_id": str(event["parent_source_id"]),
                               "reference_intervention_required": bool(truth),
                               "predicted_open_timestamp_ms": open_time, "predicted_clear_timestamp_ms": clear_time,
                               "reference_has_route_clear": has_clear})
    event_metrics = common.binary_metrics(np.asarray(event_truth), np.asarray(event_predictions))
    frame_metrics = common.binary_metrics(y, predictions)
    r819_metrics = common.load_json(Path(bound["r819_report_path"]))["evaluation"]["route_conditioned_readout"]["metrics"]
    gate = contract["gate"]
    checks = {
        "event_balanced_accuracy": event_metrics["balanced_accuracy"] >= float(gate["event_balanced_accuracy_at_least"]),
        "event_intervention_recall": event_metrics["candidate_alert_recall"] >= float(gate["event_intervention_recall_at_least"]),
        "event_context_recall": event_metrics["candidate_no_alert_recall"] >= float(gate["event_context_recall_at_least"]),
        "event_balanced_accuracy_gain_over_r819": event_metrics["balanced_accuracy"] - r819_metrics["balanced_accuracy"] >= float(gate["event_balanced_accuracy_gain_over_r819_at_least"]),
        "clear_lifecycle_recall": clear_found / max(clear_required, 1) >= float(gate["clear_lifecycle_recall_at_least"]),
        "repeat_exact": repeat_exact,
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "verified_feature_reports": verified_reports},
        "teacher": {"coefficient_sha256": teacher["coefficient_sha256"], "real_frames_used_to_fit_teacher": 0},
        "frame_count": len(y), "event_count": len(event_truth), "source_count": len(set(source_rows)),
        "frame_state_counts": {"context_or_clear": int((y == 0).sum()), "intervention_needed": int((y == 1).sum())},
        "frame_metrics": frame_metrics, "event_lifecycle_metrics": event_metrics,
        "clear_lifecycle": {"required_event_count": clear_required, "predicted_clear_event_count": clear_found,
                            "recall": clear_found / max(clear_required, 1)},
        "folds": folds, "events": lifecycle_rows, "repeat_exact": repeat_exact,
        "comparison": {"r819_event_mean_balanced_accuracy": r819_metrics["balanced_accuracy"],
                       "r823_lifecycle_balanced_accuracy": event_metrics["balanced_accuracy"],
                       "gain": event_metrics["balanced_accuracy"] - r819_metrics["balanced_accuracy"]},
        "checks": checks, "risk_lifecycle_gate_passed": bool(all(checks.values())),
        "evidence_limit": "Real provisional source-heldout frame-state/lifecycle diagnosis with future route anchors as offline oracle proxy. No provider, causal runtime, calibration, blind, Android, or production credit.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--encoder", default="vits")
    parser.add_argument("--layer-index", type=int, default=11); parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260719); parser.add_argument("--teacher-ridge", type=float, default=10.0)
    parser.add_argument("--head-ridge", type=float, default=1.0); parser.add_argument("--distance-sigma-patches", type=float, default=1.5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args(); value = run(parsed)
    print(json.dumps({"ok": True, "passed": value["risk_lifecycle_gate_passed"],
                      "comparison": value["comparison"], "event_metrics": value["event_lifecycle_metrics"],
                      "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
