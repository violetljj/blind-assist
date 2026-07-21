#!/usr/bin/env python3
"""Compare global versus route-conditioned readout over a frozen DINO risk map."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_synthetic_mask_teacher_probe as teacher_probe
import sanpo_depth_anything_linear_probe as depth_probe
from build_public_video_route_conditioned_synthetic_dataset import load_json, load_jsonl, reject_independent_direction, sha256_file


SCHEMA = "blindassist_route_conditioned_synthetic_dino_probe_v1"


def ordered_example_ids(route_examples: Sequence[dict[str, Any]]) -> list[str]:
    """Return the exact evaluation order, rejecting identities that cannot bind predictions."""
    example_ids: list[str] = []
    for index, row in enumerate(route_examples):
        value = row.get("example_id")
        if not isinstance(value, str) or not value:
            raise ValueError(f"route example {index} lacks a non-empty example_id")
        example_ids.append(value)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("route example IDs must be unique")
    return example_ids


def global_risk_features(score_map: np.ndarray, route_choice: str) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("risk map must be finite and two-dimensional")
    flat = values.ravel()
    yy, xx = np.mgrid[:values.shape[0], :values.shape[1]]
    weights = np.exp(np.clip(values - values.max(), -30, 0))
    total = max(float(weights.sum()), 1e-12)
    centroid_x = float((weights * xx).sum() / total / max(values.shape[1] - 1, 1))
    centroid_y = float((weights * yy).sum() / total / max(values.shape[0] - 1, 1))
    one_hot = [float(route_choice == choice) for choice in ("LEFT", "STRAIGHT", "RIGHT")]
    return np.asarray([
        float(flat.mean()), float(np.quantile(flat, 0.75)), float(np.quantile(flat, 0.90)),
        float(flat.max()), float(np.mean(flat > 0)), centroid_x, centroid_y, *one_hot,
    ], dtype=np.float64)


def route_conditioned_risk_features(score_map: np.ndarray, waypoints: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or len(waypoints) != 3:
        raise ValueError("route-conditioned risk needs a finite map and three waypoints")
    point_values: list[float] = []
    neighborhood_means: list[float] = []
    neighborhood_maxima: list[float] = []
    height, width = values.shape
    for x_norm, y_norm in waypoints:
        x = min(max(int(round(float(x_norm) * (width - 1))), 0), width - 1)
        y = min(max(int(round(float(y_norm) * (height - 1))), 0), height - 1)
        region = values[max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)]
        point_values.append(float(values[y, x]))
        neighborhood_means.append(float(region.mean()))
        neighborhood_maxima.append(float(region.max()))
    combined = np.asarray(point_values + neighborhood_means + neighborhood_maxima, dtype=np.float64)
    return np.concatenate([combined, [combined.mean(), combined.max(), combined[-3:].mean(), combined[-3:].max()]])


def bbox_distance_target(bbox: Sequence[int], *, image_width: int, image_height: int,
                         grid_width: int, grid_height: int, sigma_patches: float) -> np.ndarray:
    if min(image_width, image_height, grid_width, grid_height) <= 0 or sigma_patches <= 0:
        raise ValueError("distance target dimensions and sigma must be positive")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    yy, xx = np.mgrid[:grid_height, :grid_width]
    px = (xx + 0.5) / grid_width * image_width
    py = (yy + 0.5) / grid_height * image_height
    inside = (px >= x1) & (px < x2) & (py >= y1) & (py < y2)
    if not inside.any():
        center_x = min(max(int(round(((x1 + x2) * 0.5 / image_width) * grid_width - 0.5)), 0), grid_width - 1)
        center_y = min(max(int(round(((y1 + y2) * 0.5 / image_height) * grid_height - 0.5)), 0), grid_height - 1)
        inside[center_y, center_x] = True
    import cv2
    outside = (~inside).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    target = np.exp(-distance.astype(np.float64) / sigma_patches)
    target[inside] = 1.0
    return target


def fit_weighted_ridge_regression(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or not len(x) or ridge <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("distance ridge needs aligned finite samples and positive ridge")
    near = y >= 0.25
    counts = np.bincount(near.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("distance ridge requires near and far samples")
    weights = np.asarray([len(y) / (2.0 * counts[int(value)]) for value in near], dtype=np.float64)
    weights /= weights.mean()
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.where(np.sqrt(np.maximum(variance, 1e-16)) < 1e-8, 1.0, np.sqrt(np.maximum(variance, 1e-16)))
    standardized = (x - mean) / scale
    x_mean = np.average(standardized, axis=0, weights=weights)
    y_mean = float(np.average(y, weights=weights))
    root = np.sqrt(weights)
    centered_x = (standardized - x_mean) * root[:, None]
    centered_y = (y - y_mean) * root
    standardized_kernel = np.linalg.solve(
        centered_x.T @ centered_x + ridge * np.eye(centered_x.shape[1]),
        centered_x.T @ centered_y,
    )
    standardized_bias = y_mean - float(x_mean @ standardized_kernel)
    kernel = standardized_kernel / scale
    bias = standardized_bias - float((mean / scale) @ standardized_kernel)
    digest = hashlib.sha256()
    digest.update(np.asarray(kernel, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray([bias], dtype="<f8").tobytes(order="C"))
    return {"kernel": kernel, "bias": bias, "coefficient_sha256": digest.hexdigest(),
            "sample_count": len(y), "near_sample_count": int(near.sum()), "far_sample_count": int((~near).sum())}


def fit_distance_teacher(records: Sequence[dict[str, Any]], feature_maps: dict[str, np.ndarray], *,
                         ridge: float, sigma_patches: float) -> dict[str, Any]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in records:
        positive = feature_maps[record["positive_path"]]
        negative = feature_maps[record["negative_path"]]
        if positive.shape != negative.shape:
            raise ValueError("positive and clear DINO maps are misaligned")
        target = bbox_distance_target(record["bbox_xyxy"], image_width=int(record["image_width"]),
                                      image_height=int(record["image_height"]), grid_width=positive.shape[1],
                                      grid_height=positive.shape[0], sigma_patches=sigma_patches)
        features.extend([positive.reshape(-1, positive.shape[-1]), negative.reshape(-1, negative.shape[-1])])
        targets.extend([target.ravel(), np.zeros(target.size, dtype=np.float64)])
    return fit_weighted_ridge_regression(np.concatenate(features), np.concatenate(targets), ridge=ridge)


def distance_score_map(feature_map: np.ndarray, teacher: dict[str, Any]) -> np.ndarray:
    return np.asarray(feature_map, dtype=np.float64) @ np.asarray(teacher["kernel"], dtype=np.float64) + float(teacher["bias"])


def load_dataset_records(dataset: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = dataset.resolve()
    reject_independent_direction(dataset)
    review_path = dataset / "qa" / "manual_review.json"
    review = load_json(review_path)
    if review.get("disposition") != "accepted_train_only_auxiliary" or review.get("authorization", {}).get("train_only_auxiliary") is not True:
        raise ValueError("dataset lacks accepted train-only visual review")
    generation = load_jsonl(dataset / "generation_records.jsonl")
    route_examples = load_jsonl(dataset / "route_examples.jsonl")
    by_key = {(row["attributes"]["parent_slug"], int(row["attributes"]["distance_index"]), row["attributes"]["variant"]): row
              for row in generation}
    patch_records = []
    for row in generation:
        attributes = row["attributes"]
        if attributes["variant"] != "static_obstacle_composite":
            continue
        clear = by_key[(attributes["parent_slug"], int(attributes["distance_index"]), "clear_exact_copy")]
        patch_records.append({
            "source_id": row["id"], "parent_source_id": attributes["parent_source_id"],
            "asset_name": attributes["asset_name"],
            "positive_path": str((dataset / row["image_path"]).resolve()),
            "negative_path": str((dataset / clear["image_path"]).resolve()),
            "mask_path": str((dataset / row["objects"][0]["mask_path"]).resolve()),
            "bbox_xyxy": list(row["objects"][0]["bbox_xyxy"]),
            "image_width": int(row["width"]), "image_height": int(row["height"]),
        })
    for row in route_examples:
        row["absolute_image_path"] = str((dataset / row["image_path"]).resolve())
    if (
        len(patch_records) < 27
        or len(route_examples) != len(generation) * 3
        or len({row["parent_source_id"] for row in patch_records}) < 3
    ):
        raise ValueError("route-conditioned dataset is incomplete")
    return generation, route_examples, patch_records


def predict_head(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, ridge: float) -> tuple[np.ndarray, str]:
    fitted = common.fit_episode_ridge(train_features, train_labels, ridge=ridge, class_balanced=True)
    logits = test_features @ fitted["kernel"] + fitted["bias"]
    return np.argmax(logits, axis=1).astype(np.int64), fitted["coefficient_sha256"]


def evaluate(route_examples: Sequence[dict[str, Any]], patch_records: Sequence[dict[str, Any]],
             feature_maps: dict[str, np.ndarray], *, teacher_ridge: float, head_ridge: float,
             teacher_target: str = "binary_patch", distance_sigma_patches: float = 1.5) -> dict[str, Any]:
    example_ids = ordered_example_ids(route_examples)
    labels = np.asarray([int(row["route_blocked"]) for row in route_examples], dtype=np.int64)
    source_ids = np.asarray([row["parent_source_id"] for row in route_examples], dtype=object)
    global_predictions = np.full(len(labels), -1, dtype=np.int64)
    route_predictions = np.full(len(labels), -1, dtype=np.int64)
    exact_predictions = np.full(len(labels), -1, dtype=np.int64)
    exact_features = np.asarray([[float(row["intersection_fraction"])] for row in route_examples], dtype=np.float64)
    folds = []
    for held_source in dict.fromkeys(source_ids.tolist()):
        train = np.flatnonzero(source_ids != held_source)
        test = np.flatnonzero(source_ids == held_source)
        eligible = [row for row in patch_records if row["parent_source_id"] != held_source]
        if teacher_target == "binary_patch":
            teacher = teacher_probe.fit_patch_teacher(eligible, feature_maps, ridge=teacher_ridge)
            score_maps = {path: teacher_probe.teacher_score_map(feature_map, teacher["kernel"], teacher["bias"])
                          for path, feature_map in feature_maps.items()}
        elif teacher_target == "bbox_distance":
            teacher = fit_distance_teacher(eligible, feature_maps, ridge=teacher_ridge, sigma_patches=distance_sigma_patches)
            score_maps = {path: distance_score_map(feature_map, teacher) for path, feature_map in feature_maps.items()}
        else:
            raise ValueError(f"unsupported teacher target: {teacher_target}")
        global_features = np.stack([global_risk_features(score_maps[row["absolute_image_path"]], row["route_choice"])
                                    for row in route_examples])
        route_features = np.stack([route_conditioned_risk_features(score_maps[row["absolute_image_path"]], row["route_waypoints_xy_norm"])
                                   for row in route_examples])
        global_fold, global_sha = predict_head(global_features[train], labels[train], global_features[test], head_ridge)
        route_fold, route_sha = predict_head(route_features[train], labels[train], route_features[test], head_ridge)
        exact_fold, exact_sha = predict_head(exact_features[train], labels[train], exact_features[test], head_ridge)
        global_predictions[test], route_predictions[test] = global_fold, route_fold
        exact_predictions[test] = exact_fold
        folds.append({
            "held_out_parent_source_id": held_source, "held_out_example_count": len(test),
            "eligible_teacher_parent_source_ids": sorted({row["parent_source_id"] for row in eligible}),
            "eligible_teacher_asset_names": sorted({str(row["asset_name"]) for row in eligible}),
            "held_out_parent_descendants_excluded": all(row["parent_source_id"] != held_source for row in eligible),
            "teacher_coefficient_sha256": teacher["coefficient_sha256"],
            "teacher_target": teacher_target,
            "global_head_coefficient_sha256": global_sha, "route_head_coefficient_sha256": route_sha,
            "exact_field_head_coefficient_sha256": exact_sha,
            "global_metrics": common.binary_metrics(labels[test], global_fold),
            "route_conditioned_metrics": common.binary_metrics(labels[test], route_fold),
            "exact_field_head_metrics": common.binary_metrics(labels[test], exact_fold),
        })
    if np.any(global_predictions < 0) or np.any(route_predictions < 0) or np.any(exact_predictions < 0):
        raise RuntimeError("LOSO probe left examples unscored")
    by_choice = {}
    for choice in ("LEFT", "STRAIGHT", "RIGHT"):
        indices = np.asarray([index for index, row in enumerate(route_examples) if row["route_choice"] == choice])
        by_choice[choice] = {
            "example_count": len(indices),
            "global_metrics": common.binary_metrics(labels[indices], global_predictions[indices]),
            "route_conditioned_metrics": common.binary_metrics(labels[indices], route_predictions[indices]),
            "exact_field_head_metrics": common.binary_metrics(labels[indices], exact_predictions[indices]),
        }
    return {
        "global_readout": {"metrics": common.binary_metrics(labels, global_predictions), "predictions": global_predictions.tolist(),
                           "example_ids": example_ids},
        "route_conditioned_readout": {"metrics": common.binary_metrics(labels, route_predictions), "predictions": route_predictions.tolist(),
                                       "example_ids": example_ids},
        "exact_field_linear_head": {"metrics": common.binary_metrics(labels, exact_predictions), "predictions": exact_predictions.tolist(),
                                    "example_ids": example_ids,
                                    "feature": "frozen route-to-exact-composite-bbox intersection fraction only"},
        "by_route_choice": by_choice, "folds": folds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.dataset, args.src_root, args.checkpoint, args.output):
        reject_independent_direction(path)
    if not args.dataset.is_dir() or not args.src_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("dataset, DINO source, or checkpoint is missing")
    generation, route_examples, patch_records = load_dataset_records(args.dataset)
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    paths = sorted({str((args.dataset / row["image_path"]).resolve()) for row in generation})
    feature_maps = {}
    with torch.no_grad():
        for path in paths:
            feature_maps[path] = teacher_probe.extract_dino_map(model, path, input_size=args.input_size, layer_index=args.layer_index)
    first = evaluate(route_examples, patch_records, feature_maps, teacher_ridge=args.teacher_ridge, head_ridge=args.head_ridge,
                     teacher_target=args.teacher_target, distance_sigma_patches=args.distance_sigma_patches)
    second = evaluate(route_examples, patch_records, feature_maps, teacher_ridge=args.teacher_ridge, head_ridge=args.head_ridge,
                      teacher_target=args.teacher_target, distance_sigma_patches=args.distance_sigma_patches)
    repeat_exact = first == second
    global_ba = first["global_readout"]["metrics"]["balanced_accuracy"]
    route_metrics = first["route_conditioned_readout"]["metrics"]
    exact_metrics = first["exact_field_linear_head"]["metrics"]
    gate = bool(
        repeat_exact and route_metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and route_metrics["candidate_no_alert_recall"] >= args.minimum_class_recall
        and route_metrics["candidate_alert_recall"] >= args.minimum_class_recall
        and route_metrics["balanced_accuracy"] - global_ba >= args.minimum_interaction_gain
    )
    exact_head_passed = bool(
        repeat_exact and exact_metrics["balanced_accuracy"] >= 0.95
        and exact_metrics["candidate_no_alert_recall"] >= 0.95
        and exact_metrics["candidate_alert_recall"] >= 0.95
    )
    root_cause = (
        "risk_representation_or_asset_coverage_bottleneck_not_head_optimization"
        if exact_head_passed and not gate else
        "route_head_or_interaction_remains_unresolved" if not exact_head_passed else
        "synthetic_route_interaction_gate_passed"
    )
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()), "dataset_build_receipt_sha256": sha256_file(args.dataset / "build_receipt.json"),
        "dataset_manual_review_sha256": sha256_file(args.dataset / "qa" / "manual_review.json"),
        "example_count": len(route_examples), "parent_source_count": len({row["parent_source_id"] for row in route_examples}),
        "frozen_risk_representation": {"model": "Depth Anything V2 frozen DINO-S patch tokens + train-only auxiliary ridge teacher",
            "checkpoint_sha256": sha256_file(args.checkpoint), "input_size": args.input_size, "layer_index": args.layer_index,
            "teacher_target": args.teacher_target, "distance_sigma_patches": args.distance_sigma_patches,
            "seed": args.seed, "teacher_ridge": args.teacher_ridge, "head_ridge": args.head_ridge,
            "trainable_backbone_parameters": 0, "parent_matched_holdout_exclusion": True},
        "exact_geometry_reference": {"balanced_accuracy": 1.0, "meaning": "Label-construction consistency only; not a learned or real-world score."},
        "evaluation": {**first, "repeat_exact": repeat_exact,
            "split": "leave_one_complete_parent_source_out; all route variants of an image remain together"},
        "route_interaction_gate": {"passed": gate, "thresholds": {
            "route_balanced_accuracy_gte": args.minimum_balanced_accuracy,
            "each_class_recall_gte": args.minimum_class_recall,
            "balanced_accuracy_gain_over_global_gte": args.minimum_interaction_gain}},
        "feature_vs_head_root_cause": {
            "exact_field_linear_head_passed": exact_head_passed,
            "DINO_route_conditioned_head_passed": gate,
            "diagnosis": root_cause,
            "scope": "train-only synthetic diagnosis; not real event or provider evidence"
        },
        "interpretation_if_passed": "Frozen visual risk can support the task only when pooled along the supplied route; global scene readout plus a route token is insufficient.",
        "interpretation_if_failed": "Either frozen DINO does not localize the inserted obstacle robustly across parent sources or the route-conditioned readout is inadequate; do not optimize production heads.",
        "train_only": True, "real_event_truth": False, "provider_evaluation_credit": False,
        "calibration_authorized": False, "blind_authorized": False,
        "android_runtime_change_authorized": False, "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits",), default="vits")
    parser.add_argument("--layer-index", type=int, choices=range(12), default=11)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--teacher-ridge", type=float, default=10.0)
    parser.add_argument("--head-ridge", type=float, default=1.0)
    parser.add_argument("--teacher-target", choices=("binary_patch", "bbox_distance"), default="binary_patch")
    parser.add_argument("--distance-sigma-patches", type=float, default=1.5)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.80)
    parser.add_argument("--minimum-class-recall", type=float, default=0.70)
    parser.add_argument("--minimum-interaction-gain", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.input_size <= 0 or args.input_size % 14 or args.teacher_ridge <= 0 or args.head_ridge <= 0 or args.distance_sigma_patches <= 0:
        parser.error("input size must be a positive multiple of 14 and ridge values must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    evaluation = report["evaluation"]
    print(json.dumps({"ok": True, "global_balanced_accuracy": evaluation["global_readout"]["metrics"]["balanced_accuracy"],
                      "route_balanced_accuracy": evaluation["route_conditioned_readout"]["metrics"]["balanced_accuracy"],
                      "exact_field_balanced_accuracy": evaluation["exact_field_linear_head"]["metrics"]["balanced_accuracy"],
                      "route_interaction_gate": report["route_interaction_gate"]["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
