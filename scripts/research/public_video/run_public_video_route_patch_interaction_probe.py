#!/usr/bin/env python3
"""Probe a direct route-field by frozen visual-patch interaction on real events."""

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
import sanpo_depth_anything_linear_probe as depth_probe


SCHEMA = "blindassist_public_video_route_patch_interaction_probe_v1"


def fixed_projection(input_dimension: int, output_dimension: int, seed: int) -> np.ndarray:
    if min(input_dimension, output_dimension) <= 0:
        raise ValueError("projection dimensions must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dimension, output_dimension)).astype(np.float64)
    matrix /= np.sqrt(float(input_dimension))
    return matrix


def point_segment_distance(px: np.ndarray, py: np.ndarray, start: Sequence[float], end: Sequence[float]) -> np.ndarray:
    ax, ay = map(float, start); bx, by = map(float, end)
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-16:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / denominator, 0.0, 1.0)
    return np.sqrt((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2)


def route_field(height: int, width: int, waypoints: Sequence[Sequence[float]], sigma_patches: float) -> np.ndarray:
    if min(height, width) <= 0 or len(waypoints) != 3 or sigma_patches <= 0:
        raise ValueError("route field needs a valid grid, three waypoints, and positive sigma")
    yy, xx = np.mgrid[:height, :width]
    px = (xx + 0.5) / width; py = (yy + 0.5) / height
    distances = np.minimum(
        point_segment_distance(px, py, waypoints[0], waypoints[1]),
        point_segment_distance(px, py, waypoints[1], waypoints[2]),
    )
    distance_patches = distances * max(height, width)
    values = np.exp(-distance_patches / sigma_patches)
    if not np.isfinite(values).all() or values.max() <= 0:
        raise ValueError("route field is invalid")
    return values.astype(np.float64)


def interaction_features(projected_map: np.ndarray, field: np.ndarray) -> np.ndarray:
    visual = np.asarray(projected_map, dtype=np.float64)
    weights = np.asarray(field, dtype=np.float64)
    if visual.ndim != 3 or weights.shape != visual.shape[:2] or not np.isfinite(visual).all() or not np.isfinite(weights).all():
        raise ValueError("visual map and route field must align")
    route_total = max(float(weights.sum()), 1e-12)
    off = 1.0 - np.clip(weights, 0.0, 1.0)
    off_total = max(float(off.sum()), 1e-12)
    route_mean = (visual * weights[..., None]).sum(axis=(0, 1)) / route_total
    off_mean = (visual * off[..., None]).sum(axis=(0, 1)) / off_total
    return np.concatenate([route_mean, off_mean, route_mean - off_mean])


def build_event_features(oracle: dict[str, Any], feature_maps: dict[tuple[str, int], np.ndarray],
                         projection: np.ndarray, sigma_patches: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    uniform_rows = []; route_rows = []; labels = []; sources = []; event_ids = []
    for event in oracle["events"]:
        source_id = str(event["parent_source_id"]); uniform_frames = []; route_frames = []
        for frame in event["frames"]:
            fmap = feature_maps[(source_id, int(frame["timestamp_ms"]))]
            projected = np.asarray(fmap, dtype=np.float64) @ projection
            field = route_field(projected.shape[0], projected.shape[1],
                                [anchor["point_xy_norm"] for anchor in frame["anchors"]], sigma_patches)
            route_frames.append(interaction_features(projected, field))
            uniform_frames.append(interaction_features(projected, np.full(field.shape, 0.5, dtype=np.float64)))
        uniform_rows.append(np.mean(np.stack(uniform_frames), axis=0)); route_rows.append(np.mean(np.stack(route_frames), axis=0))
        labels.append(int(bool(event["reference_intervention_required"]))); sources.append(source_id); event_ids.append(str(event["item_id"]))
    return np.stack(uniform_rows), np.stack(route_rows), np.asarray(labels, dtype=np.int64), np.asarray(sources, dtype=str), event_ids


def source_loso(features: np.ndarray, labels: np.ndarray, sources: np.ndarray, ridge: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.full(len(labels), -1, dtype=np.int64); folds = []
    for held_out in sorted(set(sources.tolist())):
        train = np.flatnonzero(sources != held_out); test = np.flatnonzero(sources == held_out)
        model = common.fit_episode_ridge(features[train], labels[train], ridge=ridge, class_balanced=True)
        predictions[test] = np.argmax(features[test] @ model["kernel"] + model["bias"], axis=1)
        folds.append({"held_out_parent_source_id": held_out, "train_event_count": len(train), "test_event_count": len(test),
                      "coefficient_sha256": model["coefficient_sha256"],
                      "metrics": common.binary_metrics(labels[test], predictions[test])})
    if np.any(predictions < 0): raise RuntimeError("LOSO left events unscored")
    return predictions, folds


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.src_root, args.checkpoint, args.output): mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists(): raise ValueError("refusing to overwrite r825 output")
    contract = common.load_json(args.contract); bound = contract["bound_inputs"]
    paths = {"checkpoint_sha256": args.checkpoint, "route_oracle_sha256": Path(bound["route_oracle_path"]),
             "r790_feature_contract_sha256": Path(bound["r790_feature_contract_path"]),
             "r819_report_sha256": Path(bound["r819_report_path"])}
    for key, path in paths.items():
        if common.sha256_file(path) != bound[key]: raise ValueError(f"bound input hash mismatch: {path}")
    oracle = common.load_json(Path(bound["route_oracle_path"])); feature_contract = common.load_json(Path(bound["r790_feature_contract_path"]))
    sources_by_id, verified_reports = r819.load_sources(feature_contract)
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder); model.eval()
    feature_maps = {}
    with torch.no_grad():
        for source_id in sorted({str(event["parent_source_id"]) for event in oracle["events"]}):
            source = sources_by_id[source_id]; video = Path(source["local_video_path"])
            if common.sha256_file(video) != source["video_sha256"]: raise ValueError(f"video hash mismatch: {source_id}")
            timestamps = sorted({int(frame["timestamp_ms"]) for event in oracle["events"] if event["parent_source_id"] == source_id for frame in event["frames"]})
            images = r819.decode_at(video, timestamps)
            for timestamp, image in zip(timestamps, images):
                feature_maps[(source_id, timestamp)] = r819.extract_frame_map(model, image, input_size=args.input_size, layer_index=args.layer_index)
    sample_map = next(iter(feature_maps.values()))
    projection = fixed_projection(sample_map.shape[-1], args.projection_dimension, args.projection_seed)
    uniform_x, route_x, labels, sources, event_ids = build_event_features(oracle, feature_maps, projection, args.route_sigma_patches)
    uniform_predictions, uniform_folds = source_loso(uniform_x, labels, sources, args.head_ridge)
    route_predictions, route_folds = source_loso(route_x, labels, sources, args.head_ridge)
    repeat_uniform, repeat_uniform_folds = source_loso(uniform_x, labels, sources, args.head_ridge)
    repeat_route, repeat_route_folds = source_loso(route_x, labels, sources, args.head_ridge)
    repeat_exact = (uniform_predictions.tolist() == repeat_uniform.tolist() and route_predictions.tolist() == repeat_route.tolist()
                    and uniform_folds == repeat_uniform_folds and route_folds == repeat_route_folds)
    uniform_metrics = common.binary_metrics(labels, uniform_predictions); route_metrics = common.binary_metrics(labels, route_predictions)
    old = common.load_json(Path(bound["r819_report_path"]))["evaluation"]["route_conditioned_readout"]["metrics"]
    gate = contract["gate"]
    checks = {"route_balanced_accuracy": route_metrics["balanced_accuracy"] >= gate["route_balanced_accuracy_at_least"],
              "intervention_recall": route_metrics["candidate_alert_recall"] >= gate["intervention_recall_at_least"],
              "context_recall": route_metrics["candidate_no_alert_recall"] >= gate["context_recall_at_least"],
              "gain_over_uniform": route_metrics["balanced_accuracy"] - uniform_metrics["balanced_accuracy"] >= gate["gain_over_uniform_at_least"],
              "gain_over_r819": route_metrics["balanced_accuracy"] - old["balanced_accuracy"] >= gate["gain_over_r819_at_least"],
              "repeat_exact": repeat_exact}
    projection_digest = hashlib.sha256(np.asarray(projection, dtype="<f8").tobytes(order="C")).hexdigest()
    rows = [{"event_id": event_id, "parent_source_id": str(source), "label": int(label),
             "uniform_prediction": int(up), "route_interaction_prediction": int(rp)}
            for event_id, source, label, up, rp in zip(event_ids, sources, labels, uniform_predictions, route_predictions)]
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"contract_sha256": common.sha256_file(args.contract), "verified_feature_reports": verified_reports},
              "representation": {"backbone": "frozen Depth Anything V2 DINO-S patches", "projection_dimension": args.projection_dimension,
                                 "projection_seed": args.projection_seed, "projection_sha256": projection_digest,
                                 "route_sigma_patches": args.route_sigma_patches,
                                 "feature": "route-weighted mean + off-route-weighted mean + contrast",
                                 "uniform_control": "identical feature function with constant 0.5 field"},
              "event_count": len(labels), "source_count": len(set(sources.tolist())), "feature_dimension": route_x.shape[1],
              "evaluation": {"uniform_metrics": uniform_metrics, "uniform_predictions": uniform_predictions.tolist(), "uniform_folds": uniform_folds,
                             "route_metrics": route_metrics, "route_predictions": route_predictions.tolist(), "route_folds": route_folds,
                             "repeat_exact": repeat_exact}, "event_predictions": rows,
              "comparison": {"uniform_balanced_accuracy": uniform_metrics["balanced_accuracy"],
                             "route_interaction_balanced_accuracy": route_metrics["balanced_accuracy"],
                             "route_gain_over_uniform": route_metrics["balanced_accuracy"] - uniform_metrics["balanced_accuracy"],
                             "r819_balanced_accuracy": old["balanced_accuracy"],
                             "gain_over_r819": route_metrics["balanced_accuracy"] - old["balanced_accuracy"]},
              "checks": checks, "route_patch_interaction_gate_passed": bool(all(checks.values())),
              "evidence_limit": "Real provisional source-heldout interaction diagnosis with future route anchors as offline oracle proxy. No bbox or obstacle-hit inputs, provider credit, runtime claim, calibration, blind, Android, or production authorization.",
              "authorization": contract["authorization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    Path(str(args.output)+".sha256").write_text(common.sha256_file(args.output)+"\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "src_root", "checkpoint", "output"): parser.add_argument(f"--{name.replace('_','-')}", type=Path, required=True)
    parser.add_argument("--encoder", default="vits"); parser.add_argument("--layer-index", type=int, default=11); parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260719); parser.add_argument("--projection-dimension", type=int, default=32)
    parser.add_argument("--projection-seed", type=int, default=0); parser.add_argument("--route-sigma-patches", type=float, default=1.5)
    parser.add_argument("--head-ridge", type=float, default=1.0); return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args(); value = run(parsed)
    print(json.dumps({"ok": True, "passed": value["route_patch_interaction_gate_passed"], "comparison": value["comparison"],
                      "metrics": value["evaluation"]["route_metrics"], "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
