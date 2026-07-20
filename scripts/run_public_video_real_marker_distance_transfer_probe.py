#!/usr/bin/env python3
"""Fit real provisional marker distance teachers inside each source-LOSO training fold."""

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
import run_public_video_route_conditioned_real_transfer_probe as r819
import run_public_video_route_conditioned_synthetic_probe as synthetic_probe
import sanpo_depth_anything_linear_probe as depth_probe


SCHEMA = "blindassist_public_video_real_marker_distance_transfer_probe_v1"


def union_distance_target(detections: Sequence[dict[str, Any]], *, image_width: int, image_height: int,
                          grid_width: int, grid_height: int, sigma_patches: float) -> np.ndarray:
    target = np.zeros((grid_height, grid_width), dtype=np.float64)
    for detection in detections:
        bbox = detection.get("xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("real marker detection lacks xyxy")
        target = np.maximum(target, synthetic_probe.bbox_distance_target(
            bbox, image_width=image_width, image_height=image_height,
            grid_width=grid_width, grid_height=grid_height, sigma_patches=sigma_patches,
        ))
    return target


def patch_weights(targets: np.ndarray, sources: np.ndarray, frames: np.ndarray, boundary: float) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    source_values = np.asarray(sources, dtype=str)
    frame_values = np.asarray(frames, dtype=str)
    near = values >= boundary
    if set(near.tolist()) != {False, True}:
        raise ValueError("distance teacher needs near and far patches")
    weights = np.zeros(len(values), dtype=np.float64)
    for class_value in (False, True):
        class_indices = np.flatnonzero(near == class_value)
        class_sources = sorted(set(source_values[class_indices].tolist()))
        for source in class_sources:
            source_indices = class_indices[source_values[class_indices] == source]
            source_frames = sorted(set(frame_values[source_indices].tolist()))
            for frame in source_frames:
                indices = source_indices[frame_values[source_indices] == frame]
                weights[indices] = 0.5 / len(class_sources) / len(source_frames) / len(indices)
    weights /= weights.mean()
    return weights


def fit_continuous_ridge(features: np.ndarray, targets: np.ndarray, weights: np.ndarray, ridge: float) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64); y = np.asarray(targets, dtype=np.float64); w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(w) or ridge <= 0:
        raise ValueError("continuous ridge inputs are invalid")
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x - mean) ** 2, axis=0, weights=w)
    scale = np.where(np.sqrt(np.maximum(variance, 1e-16)) < 1e-8, 1.0, np.sqrt(np.maximum(variance, 1e-16)))
    z = (x - mean) / scale
    z_mean = np.average(z, axis=0, weights=w); y_mean = float(np.average(y, weights=w))
    root = np.sqrt(w)
    centered_x = (z - z_mean) * root[:, None]; centered_y = (y - y_mean) * root
    kernel_z = np.linalg.solve(centered_x.T @ centered_x + ridge * np.eye(x.shape[1]), centered_x.T @ centered_y)
    bias_z = y_mean - float(z_mean @ kernel_z)
    kernel = kernel_z / scale; bias = bias_z - float((mean / scale) @ kernel_z)
    digest = hashlib.sha256(); digest.update(np.asarray(kernel, dtype="<f8").tobytes()); digest.update(np.asarray([bias], dtype="<f8").tobytes())
    return {"kernel": kernel, "bias": bias, "coefficient_sha256": digest.hexdigest()}


def fit_teacher(frame_records: Sequence[dict[str, Any]], feature_maps: dict[tuple[str, int], np.ndarray], *,
                ridge: float, sigma: float, near_boundary: float) -> dict[str, Any]:
    feature_rows = []; target_rows = []; source_rows = []; frame_rows = []
    for row in frame_records:
        key = (row["source_id"], row["timestamp_ms"]); fmap = feature_maps[key]
        target = union_distance_target(row["detections"], image_width=row["image_width"], image_height=row["image_height"],
                                       grid_width=fmap.shape[1], grid_height=fmap.shape[0], sigma_patches=sigma)
        count = target.size
        feature_rows.append(fmap.reshape(count, -1)); target_rows.append(target.ravel())
        source_rows.extend([row["source_id"]] * count); frame_rows.extend([f"{row['source_id']}:{row['timestamp_ms']}"] * count)
    x = np.concatenate(feature_rows); y = np.concatenate(target_rows)
    weights = patch_weights(y, np.asarray(source_rows), np.asarray(frame_rows), near_boundary)
    result = fit_continuous_ridge(x, y, weights, ridge)
    result.update({"patch_count": len(y), "near_patch_count": int((y >= near_boundary).sum()),
                   "far_patch_count": int((y < near_boundary).sum())})
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.src_root, args.checkpoint, args.output): mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists(): raise ValueError("refusing to overwrite r824 output")
    contract = common.load_json(args.contract); bound = contract["bound_inputs"]
    paths = {"checkpoint_sha256": args.checkpoint, "route_oracle_sha256": Path(bound["route_oracle_path"]),
             "r790_feature_contract_sha256": Path(bound["r790_feature_contract_path"]),
             "r819_report_sha256": Path(bound["r819_report_path"])}
    for key, path in paths.items():
        if common.sha256_file(path) != bound[key]: raise ValueError(f"bound input hash mismatch: {path}")
    oracle = common.load_json(Path(bound["route_oracle_path"])); feature_contract = common.load_json(Path(bound["r790_feature_contract_path"]))
    sources_by_id, verified_reports = r819.load_sources(feature_contract)
    event_ids = [str(row["item_id"]) for row in oracle["events"]]
    event_sources = np.asarray([str(row["parent_source_id"]) for row in oracle["events"]], dtype=str)
    event_labels = np.asarray([int(bool(row["reference_intervention_required"])) for row in oracle["events"]], dtype=np.int64)
    frame_records = []
    for event in oracle["events"]:
        source_id = str(event["parent_source_id"]); source = sources_by_id[source_id]
        samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
        for frame in event["frames"]:
            timestamp = int(frame["timestamp_ms"]); sample = samples[timestamp]
            frame_records.append({"event_id": str(event["item_id"]), "source_id": source_id, "timestamp_ms": timestamp,
                                  "anchors": frame["anchors"], "detections": list(sample.get("detections", []))})
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder); model.eval()
    feature_maps = {}; by_source = {}
    for row in frame_records: by_source.setdefault(row["source_id"], []).append(row)
    with torch.no_grad():
        for source_id, rows in by_source.items():
            source = sources_by_id[source_id]; video = Path(source["local_video_path"])
            if common.sha256_file(video) != source["video_sha256"]: raise ValueError(f"video hash mismatch: {source_id}")
            unique_times = sorted({row["timestamp_ms"] for row in rows}); images = r819.decode_at(video, unique_times)
            image_by_time = dict(zip(unique_times, images))
            for timestamp, image in image_by_time.items():
                feature_maps[(source_id, timestamp)] = r819.extract_frame_map(model, image, input_size=args.input_size, layer_index=args.layer_index)
            for row in rows:
                image = image_by_time[row["timestamp_ms"]]; row["image_height"], row["image_width"] = image.shape[:2]
    global_predictions = np.full(len(event_ids), -1, dtype=np.int64); route_predictions = np.full(len(event_ids), -1, dtype=np.int64); folds = []
    for held_out in sorted(set(event_sources.tolist())):
        eligible_frames = [row for row in frame_records if row["source_id"] != held_out]
        teacher = fit_teacher(eligible_frames, feature_maps, ridge=args.teacher_ridge, sigma=args.distance_sigma_patches,
                              near_boundary=args.near_far_boundary)
        global_x = []; route_x = []
        for event in oracle["events"]:
            source_id = str(event["parent_source_id"]); g_frames = []; r_frames = []
            for frame in event["frames"]:
                key = (source_id, int(frame["timestamp_ms"])); score = synthetic_probe.distance_score_map(feature_maps[key], teacher)
                g_frames.append(r819.global_field_features(score))
                r_frames.append(synthetic_probe.route_conditioned_risk_features(score, [a["point_xy_norm"] for a in frame["anchors"]]))
            global_x.append(np.mean(np.stack(g_frames), axis=0)); route_x.append(np.mean(np.stack(r_frames), axis=0))
        global_x = np.stack(global_x); route_x = np.stack(route_x)
        train = np.flatnonzero(event_sources != held_out); test = np.flatnonzero(event_sources == held_out)
        global_model = common.fit_episode_ridge(global_x[train], event_labels[train], ridge=args.head_ridge, class_balanced=True)
        route_model = common.fit_episode_ridge(route_x[train], event_labels[train], ridge=args.head_ridge, class_balanced=True)
        global_predictions[test] = np.argmax(global_x[test] @ global_model["kernel"] + global_model["bias"], axis=1)
        route_predictions[test] = np.argmax(route_x[test] @ route_model["kernel"] + route_model["bias"], axis=1)
        folds.append({"held_out_parent_source_id": held_out, "teacher_training_source_ids": sorted(set(event_sources.tolist()) - {held_out}),
                      "held_out_detections_used_to_fit_teacher": False, "teacher_coefficient_sha256": teacher["coefficient_sha256"],
                      "teacher_patch_count": teacher["patch_count"], "teacher_near_patch_count": teacher["near_patch_count"],
                      "route_metrics": common.binary_metrics(event_labels[test], route_predictions[test])})
    route_metrics = common.binary_metrics(event_labels, route_predictions); global_metrics = common.binary_metrics(event_labels, global_predictions)
    old = common.load_json(Path(bound["r819_report_path"]))["evaluation"]["route_conditioned_readout"]["metrics"]
    gate = contract["gate"]
    checks = {"route_balanced_accuracy": route_metrics["balanced_accuracy"] >= gate["route_balanced_accuracy_at_least"],
              "intervention_recall": route_metrics["candidate_alert_recall"] >= gate["intervention_recall_at_least"],
              "context_recall": route_metrics["candidate_no_alert_recall"] >= gate["context_recall_at_least"],
              "gain_over_global": route_metrics["balanced_accuracy"] - global_metrics["balanced_accuracy"] >= gate["gain_over_global_at_least"],
              "gain_over_r819": route_metrics["balanced_accuracy"] - old["balanced_accuracy"] >= gate["gain_over_r819_at_least"],
              "all_held_out_detections_excluded": all(not row["held_out_detections_used_to_fit_teacher"] for row in folds)}
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"contract_sha256": common.sha256_file(args.contract), "verified_feature_reports": verified_reports},
              "frame_count": len(frame_records), "zero_detection_frame_count": sum(not row["detections"] for row in frame_records),
              "event_count": len(event_ids), "source_count": len(set(event_sources.tolist())),
              "teacher": contract["teacher"], "folds": folds,
              "evaluation": {"global_metrics": global_metrics, "global_predictions": global_predictions.tolist(),
                             "route_metrics": route_metrics, "route_predictions": route_predictions.tolist()},
              "comparison": {"r819_route_balanced_accuracy": old["balanced_accuracy"],
                             "r824_route_balanced_accuracy": route_metrics["balanced_accuracy"],
                             "gain": route_metrics["balanced_accuracy"] - old["balanced_accuracy"]},
              "checks": checks, "real_marker_distance_gate_passed": bool(all(checks.values())),
              "evidence_limit": "Real provisional marker-box train-fold supervision with complete source holdout. Marker boxes are not human truth; no provider, calibration, blind, Android, or production credit.",
              "authorization": contract["authorization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    Path(str(args.output)+".sha256").write_text(common.sha256_file(args.output)+"\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "src_root", "checkpoint", "output"): parser.add_argument(f"--{name.replace('_','-')}", type=Path, required=True)
    parser.add_argument("--encoder", default="vits"); parser.add_argument("--layer-index", type=int, default=11); parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260719); parser.add_argument("--teacher-ridge", type=float, default=10.0); parser.add_argument("--head-ridge", type=float, default=1.0)
    parser.add_argument("--distance-sigma-patches", type=float, default=1.5); parser.add_argument("--near-far-boundary", type=float, default=0.25)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args(); value = run(parsed)
    print(json.dumps({"ok": True, "passed": value["real_marker_distance_gate_passed"], "comparison": value["comparison"],
                      "metrics": value["evaluation"]["route_metrics"], "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
