#!/usr/bin/env python3
"""Run the deterministic r7.67 marker-conditioned future-route linear probe."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_temporal_risk_profile_prospective as prospective
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_ego_route_distance_field_probe as spatial


SCHEMA = "blindassist_public_video_marker_relation_linear_probe_v1"


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0 or labels.shape != scores.shape:
        raise ValueError("AUROC requires aligned positive and negative scores")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def source_class_balanced_weights(sources: np.ndarray, active: np.ndarray) -> np.ndarray:
    sources = np.asarray(sources).astype(str)
    active = np.asarray(active, dtype=bool)
    if sources.shape != active.shape:
        raise ValueError("source and target arrays differ")
    weights = np.zeros(len(sources), dtype=np.float64)
    unique_sources = sorted(set(sources.tolist()))
    for source in unique_sources:
        indices = np.flatnonzero(sources == source)
        positive = indices[active[indices]]
        negative = indices[~active[indices]]
        source_mass = 1.0 / len(unique_sources)
        if len(positive) and len(negative):
            weights[positive] = source_mass * 0.5 / len(positive)
            weights[negative] = source_mass * 0.5 / len(negative)
        else:
            weights[indices] = source_mass / len(indices)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("sample weights do not sum to one")
    return weights


def relation_vector(feature_grid: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    values = np.asarray(feature_grid, dtype=np.float64)
    mask = np.asarray(obstacle, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[1:] or not mask.any():
        raise ValueError("relation vector requires a non-empty aligned obstacle mask")
    selected = values[:, mask]
    masked_mean = selected.mean(axis=1)
    masked_max = selected.max(axis=1)
    mean_delta = masked_mean - values.mean(axis=(1, 2))
    yy, xx = np.nonzero(mask)
    height, width = mask.shape
    geometry = np.asarray([
        float(mask.mean()),
        float(xx.mean() / max(1, width - 1)),
        float(yy.mean() / max(1, height - 1)),
    ])
    return np.concatenate([masked_mean, masked_max, mean_delta, geometry]).astype(np.float64)


def marker_grid_mask(detections: list[dict[str, Any]], side: int, expansion: float) -> np.ndarray:
    """Rasterize every marker, retaining sub-patch boxes deterministically."""
    combined = np.zeros((side, side), dtype=bool)
    for detection in detections:
        single = spatial.obstacle_grid_mask([detection], side, expansion)
        if not single.any():
            values = detection["features"]
            center_x = float(values["center_x_norm"])
            center_y = float(values["bottom_y_norm"]) - float(values["height_norm"]) / 2.0
            x_index = int(np.clip(np.floor(center_x * side), 0, side - 1))
            y_index = int(np.clip(np.floor(center_y * side), 0, side - 1))
            single[y_index, x_index] = True
        combined |= single
    return combined


def fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    weighted_design = design * weights[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ weighted_design + penalty, design.T @ (weights * y))
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(normalized)), normalized])
    return design @ model["coefficients"]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _load_detection_index(source_contract_path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    contract = common.load_json(source_contract_path)
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for binding in contract.get("feature_reports", {}).values():
        path = _resolve(str(binding["path"]))
        if common.sha256_file(path) != str(binding["sha256"]):
            raise ValueError(f"feature report hash mismatch: {path}")
        report = lifecycle.verify_json_sidecar(path)
        for source in report.get("sources", []):
            source_id = str(source["source_id"])
            for sample in source.get("samples", []):
                result[(source_id, int(sample["timestamp_ms"]))] = list(sample.get("detections", []))
    return result


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bangkok_event_vectors(
    report: dict[str, Any],
    features: dict[str, Any],
    training_contract: dict[str, Any],
    model_dir: Path,
    expansion: float,
    batch_size: int,
) -> tuple[np.ndarray, list[int]]:
    source_id = str(report["source_id"])
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == source_id]
    if len(source_rows) != 1:
        raise ValueError("Bangkok source row is missing")
    source = source_rows[0]
    event = report["frozen_radial_event"]
    start = int(event["event_entry_timestamp_ms"])
    end = int(event["last_active_timestamp_ms"])
    timestamps = list(range(start, end + 1, 1000))
    grids, samples = prospective._build_features(source, timestamps, training_contract, model_dir, batch_size)
    vectors = []
    for grid, sample in zip(grids, samples):
        mask = marker_grid_mask(sample.get("detections", []), grid.shape[-1], expansion)
        vectors.append(relation_vector(grid, mask))
    return np.stack(vectors), timestamps


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = (
        args.contract,
        args.training_contract,
        args.bangkok_features,
        args.bangkok_candidates,
        args.negative_result,
        args.positive_result,
        args.model_dir,
        args.output,
    )
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    cache_path = _resolve(bound["r764_feature_cache_path"])
    manifest_path = _resolve(bound["r763_manifest_path"])
    source_contract_path = _resolve(bound["r754_source_contract_path"])
    for path, key in (
        (cache_path, "r764_feature_cache_sha256"),
        (manifest_path, "r763_manifest_sha256"),
        (source_contract_path, "r754_source_contract_sha256"),
        (args.bangkok_features, "bangkok_feature_report_sha256"),
        (args.bangkok_candidates, "bangkok_candidate_report_sha256"),
    ):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    if common.sha256_file(args.training_contract) != bound["r764_training_contract_sha256"]:
        raise ValueError("training contract hash mismatch")
    if common.sha256_file(args.model_dir / "pytorch_model.bin") != bound["dinov2_weights_sha256"]:
        raise ValueError("DINO weights hash mismatch")

    cache = np.load(cache_path, allow_pickle=False)
    manifest = _load_manifest(manifest_path)
    train_sources = cache["train_sources"].astype(str)
    train_timestamps = cache["train_timestamps"].astype(np.int64)
    if len(manifest) != len(train_sources):
        raise ValueError("manifest and feature cache lengths differ")
    for index, row in enumerate(manifest):
        if str(row["source_id"]) != train_sources[index] or int(row["timestamp_ms"]) != train_timestamps[index]:
            raise ValueError("manifest order differs from feature cache")

    detection_index = _load_detection_index(source_contract_path)
    selected_indices = [index for index, row in enumerate(manifest) if int(row["marker_detection_count"]) > 0]
    vectors = []
    targets = []
    sources = []
    grid_side = int(contract["feature_vector"]["grid_side"])
    expansion = float(contract["feature_vector"]["marker_expansion_object_heights"])
    for index in selected_indices:
        row = manifest[index]
        key = (str(row["source_id"]), int(row["timestamp_ms"]))
        detections = detection_index.get(key)
        if detections is None:
            raise ValueError(f"frozen detections missing for {key}")
        mask = marker_grid_mask(detections, grid_side, expansion)
        vectors.append(relation_vector(cache["train_x"][index], mask))
        targets.append(float(row["teacher_marker_hit_fraction_diagnostic_only"]))
        sources.append(str(row["source_id"]))
    x = np.stack(vectors)
    y = np.asarray(targets, dtype=np.float64)
    source_array = np.asarray(sources)
    active = y > 0.0
    if x.shape[1] != int(contract["feature_vector"]["dimension"]):
        raise ValueError("relation feature dimension differs from contract")

    alpha = float(contract["linear_probe"]["alpha"])
    oof = np.zeros(len(y), dtype=np.float64)
    fold_rows = []
    for held_out in sorted(set(sources)):
        train = source_array != held_out
        test = ~train
        weights = source_class_balanced_weights(source_array[train], active[train])
        model = fit_weighted_ridge(x[train], y[train], weights, alpha)
        oof[test] = predict_ridge(model, x[test])
        fold_rows.append({
            "held_out_source_id": held_out,
            "frame_count": int(test.sum()),
            "active_count": int(active[test].sum()),
            "mean_prediction": float(oof[test].mean()),
        })
    oof_auc = roc_auc(active.astype(np.int64), oof)

    all_weights = source_class_balanced_weights(source_array, active)
    final_model = fit_weighted_ridge(x, y, all_weights, alpha)
    training_contract = common.load_json(args.training_contract)
    bangkok_features = lifecycle.verify_json_sidecar(args.bangkok_features)
    negative = lifecycle.verify_json_sidecar(args.negative_result)
    positive = lifecycle.verify_json_sidecar(args.positive_result)
    negative_x, negative_times = _bangkok_event_vectors(
        negative, bangkok_features, training_contract, args.model_dir, expansion, args.batch_size
    )
    positive_x, positive_times = _bangkok_event_vectors(
        positive, bangkok_features, training_contract, args.model_dir, expansion, args.batch_size
    )
    negative_scores = predict_ridge(final_model, negative_x)
    positive_scores = predict_ridge(final_model, positive_x)
    negative_mean = float(negative_scores.mean())
    positive_mean = float(positive_scores.mean())
    gate = contract["diagnostic_gate"]
    checks = {
        "source_held_out_teacher_active_auroc": oof_auc >= float(gate["source_held_out_teacher_active_auroc_at_least"]),
        "bangkok_positive_above_safe_lateral": positive_mean > negative_mean,
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "contract_sha256": common.sha256_file(args.contract),
            "feature_cache_sha256": common.sha256_file(cache_path),
            "manifest_sha256": common.sha256_file(manifest_path),
            "source_contract_sha256": common.sha256_file(source_contract_path),
            "bangkok_feature_report_sha256": common.sha256_file(args.bangkok_features),
            "bangkok_candidate_report_sha256": common.sha256_file(args.bangkok_candidates),
            "negative_result_sha256": common.sha256_file(args.negative_result),
            "positive_result_sha256": common.sha256_file(args.positive_result),
        },
        "training": {
            "frame_count": len(y),
            "active_count": int(active.sum()),
            "source_count": len(set(sources)),
            "feature_dimension": int(x.shape[1]),
            "alpha": alpha,
            "source_held_out_teacher_active_auroc": oof_auc,
            "folds": fold_rows,
        },
        "bangkok_same_source_diagnostic": {
            "negative": {
                "timestamps_ms": negative_times,
                "frame_scores": negative_scores.tolist(),
                "event_mean": negative_mean,
            },
            "positive": {
                "timestamps_ms": positive_times,
                "frame_scores": positive_scores.tolist(),
                "event_mean": positive_mean,
            },
            "margin": positive_mean - negative_mean,
            "prospective_credit": False,
        },
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
        "interpretation": (
            "If the gate passes, marker-conditioned causal features are linearly usable and the r7.66 route-map readout is the immediate bottleneck. "
            "If it fails, the causal representation itself remains insufficient and nonlinear head runs stay closed."
        ),
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--bangkok-features", type=Path, required=True)
    parser.add_argument("--bangkok-candidates", type=Path, required=True)
    parser.add_argument("--negative-result", type=Path, required=True)
    parser.add_argument("--positive-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "oof_auroc": value["training"]["source_held_out_teacher_active_auroc"],
        "bangkok_margin": value["bangkok_same_source_diagnostic"]["margin"],
        "diagnostic_gate_passed": value["diagnostic_gate_passed"],
        "output_sha256": common.sha256_file(parsed.output),
    }))
