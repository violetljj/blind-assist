#!/usr/bin/env python3
"""Run five frozen r7.68 prototype/source-bootstrap relation-head short runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_linear_probe as linear


SCHEMA = "blindassist_public_video_marker_relation_bootstrap_short_runs_v1"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def prototype_initialization(
    x: np.ndarray, active: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(active, dtype=bool)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if not labels.any() or labels.all():
        raise ValueError("prototype initialization requires both target states")
    negative = np.average(values[~labels], axis=0, weights=sample_weights[~labels])
    positive = np.average(values[labels], axis=0, weights=sample_weights[labels])
    direction = positive - negative
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("marker relation prototypes are identical")
    direction /= norm
    midpoint = (positive + negative) / 2.0
    return direction, -float(midpoint @ direction)


def bootstrap_source_class_rows(
    sources: np.ndarray, active: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    values = np.asarray(sources).astype(str)
    labels = np.asarray(active, dtype=bool)
    rng = np.random.default_rng(int(seed))
    indices: list[int] = []
    weights: list[float] = []
    draws_by_class: dict[str, list[str]] = {}
    for class_value, class_name in ((False, "inactive"), (True, "active")):
        class_sources = sorted(set(values[labels == class_value].tolist()))
        if not class_sources:
            raise ValueError(f"bootstrap lacks {class_name} source blocks")
        draws = rng.choice(class_sources, size=len(class_sources), replace=True).tolist()
        draws_by_class[class_name] = draws
        for source in draws:
            rows = np.flatnonzero((values == source) & (labels == class_value)).tolist()
            indices.extend(rows)
            weights.extend([0.5 / len(draws) / len(rows)] * len(rows))
    return np.asarray(indices, dtype=np.int64), np.asarray(weights, dtype=np.float64), draws_by_class


def fit_soft_target_head(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    initial_weight: np.ndarray,
    initial_bias: float,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    values = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    sample_weights /= sample_weights.sum()
    weight = np.asarray(initial_weight, dtype=np.float64).copy()
    bias = float(initial_bias)
    m_w = np.zeros_like(weight)
    v_w = np.zeros_like(weight)
    m_b = v_b = 0.0
    losses: list[float] = []
    for step in range(1, int(steps) + 1):
        probability = sigmoid(values @ weight + bias)
        cross_entropy = -(targets * np.log(np.maximum(probability, 1e-12))
                          + (1.0 - targets) * np.log(np.maximum(1.0 - probability, 1e-12)))
        loss = float(sample_weights @ cross_entropy + 0.5 * weight_decay * (weight @ weight))
        error = sample_weights * (probability - targets)
        gradient_w = values.T @ error + weight_decay * weight
        gradient_b = float(error.sum())
        m_w = 0.9 * m_w + 0.1 * gradient_w
        v_w = 0.999 * v_w + 0.001 * gradient_w * gradient_w
        m_b = 0.9 * m_b + 0.1 * gradient_b
        v_b = 0.999 * v_b + 0.001 * gradient_b * gradient_b
        weight -= learning_rate * (m_w / (1.0 - 0.9**step)) / (np.sqrt(v_w / (1.0 - 0.999**step)) + 1e-8)
        bias -= learning_rate * (m_b / (1.0 - 0.9**step)) / (np.sqrt(v_b / (1.0 - 0.999**step)) + 1e-8)
        if step in {1, int(steps)}:
            losses.append(loss)
    digest = hashlib.sha256(np.asarray([*weight, bias], dtype="<f8").tobytes()).hexdigest()
    return {"weight": weight, "bias": bias, "loss_first_last": losses, "coefficient_sha256": digest}


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return sigmoid(np.asarray(x, dtype=np.float64) @ model["weight"] + float(model["bias"]))


def load_relation_data(contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bound = contract["bound_inputs"]
    cache_path = linear._resolve(bound["r764_feature_cache_path"])
    manifest_path = linear._resolve(bound["r763_manifest_path"])
    source_contract_path = linear._resolve(bound["r754_source_contract_path"])
    for path, key in ((cache_path, "r764_feature_cache_sha256"),
                      (manifest_path, "r763_manifest_sha256"),
                      (source_contract_path, "r754_source_contract_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    cache = np.load(cache_path, allow_pickle=False)
    manifest = linear._load_manifest(manifest_path)
    sources = cache["train_sources"].astype(str)
    timestamps = cache["train_timestamps"].astype(np.int64)
    if len(manifest) != len(sources):
        raise ValueError("manifest and cache lengths differ")
    detections = linear._load_detection_index(source_contract_path)
    vectors: list[np.ndarray] = []
    targets: list[float] = []
    selected_sources: list[str] = []
    side = int(contract["feature_vector"]["grid_side"])
    expansion = float(contract["feature_vector"]["marker_expansion_object_heights"])
    for index, row in enumerate(manifest):
        if str(row["source_id"]) != sources[index] or int(row["timestamp_ms"]) != timestamps[index]:
            raise ValueError("manifest order differs from cache")
        if int(row["marker_detection_count"]) <= 0:
            continue
        key = (str(row["source_id"]), int(row["timestamp_ms"]))
        mask = linear.marker_grid_mask(detections[key], side, expansion)
        vectors.append(linear.relation_vector(cache["train_x"][index], mask))
        targets.append(float(row["teacher_marker_hit_fraction_diagnostic_only"]))
        selected_sources.append(key[0])
    return np.stack(vectors), np.asarray(targets), np.asarray(selected_sources)


def fit_bootstrap_model(
    x: np.ndarray, y: np.ndarray, sources: np.ndarray, seed: int, optimizer: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    active = y > 0.0
    sampled, sampled_weights, draws = bootstrap_source_class_rows(sources, active, seed)
    mean = np.average(x[sampled], axis=0, weights=sampled_weights)
    variance = np.average((x[sampled] - mean) ** 2, axis=0, weights=sampled_weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    initial_weight, initial_bias = prototype_initialization(normalized[sampled], active[sampled], sampled_weights)
    model = fit_soft_target_head(
        normalized[sampled], y[sampled], sampled_weights, initial_weight, initial_bias,
        steps=int(optimizer["steps"]), learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
    )
    model.update({"mean": mean, "scale": scale})
    prototype = {"weight": initial_weight, "bias": initial_bias, "mean": mean, "scale": scale}
    unique_active = len(set(draws["active"]))
    unique_total = len(set(draws["active"] + draws["inactive"]))
    audit = {"sampled_source_ids_by_class": draws, "sampled_unique_active_source_count": unique_active,
             "sampled_unique_total_source_count": unique_total}
    return model, prototype, audit


def model_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return predict(model, (np.asarray(x) - model["mean"]) / model["scale"])


def source_macro_metrics(active: np.ndarray, scores: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    labels = np.asarray(active, dtype=bool)
    predicted = np.asarray(scores, dtype=np.float64) >= 0.5
    source_values = np.asarray(sources).astype(str)
    positive_recalls: list[float] = []
    negative_recalls: list[float] = []
    mixed_aurocs: list[float] = []
    for source in sorted(set(source_values.tolist())):
        selected = source_values == source
        if labels[selected].any():
            positive_recalls.append(float(predicted[selected][labels[selected]].mean()))
        if (~labels[selected]).any():
            negative_recalls.append(float((~predicted[selected][~labels[selected]]).mean()))
        if labels[selected].any() and (~labels[selected]).any():
            mixed_aurocs.append(linear.roc_auc(labels[selected].astype(np.int64), scores[selected]))
    positive = float(np.mean(positive_recalls))
    negative = float(np.mean(negative_recalls))
    return {"source_macro_positive_recall": positive, "source_macro_negative_recall": negative,
            "source_macro_balanced_accuracy": (positive + negative) / 2.0,
            "mixed_class_source_auroc_median": float(np.median(mixed_aurocs))}


def evaluate_seed(
    x: np.ndarray, y: np.ndarray, sources: np.ndarray, negative_x: np.ndarray,
    positive_x: np.ndarray, seed: int, optimizer: dict[str, Any]
) -> dict[str, Any]:
    active = y > 0.0
    oof = np.zeros(len(y), dtype=np.float64)
    prototype_oof = np.zeros(len(y), dtype=np.float64)
    folds = []
    for fold_index, held_source in enumerate(sorted(set(sources.tolist()))):
        test = sources == held_source
        model, prototype, bootstrap = fit_bootstrap_model(x[~test], y[~test], sources[~test], seed + 1009 * fold_index, optimizer)
        oof[test] = model_predict(model, x[test])
        prototype_oof[test] = model_predict(prototype, x[test])
        fold = {"held_out_source_id": held_source, "frame_count": int(test.sum()),
                "active_count": int(active[test].sum()), "mean_score": float(oof[test].mean()), **bootstrap}
        if active[test].any() and (~active[test]).any():
            fold["teacher_active_auroc"] = linear.roc_auc(active[test].astype(np.int64), oof[test])
        folds.append(fold)
    final, final_prototype, bootstrap = fit_bootstrap_model(x, y, sources, seed + 7919, optimizer)
    negative_scores = model_predict(final, negative_x)
    positive_scores = model_predict(final, positive_x)
    return {
        "seed": seed,
        "source_held_out_teacher_active_auroc": linear.roc_auc(active.astype(np.int64), oof),
        "prototype_only_source_macro_metrics": source_macro_metrics(active, prototype_oof, sources),
        "optimized_source_macro_metrics": source_macro_metrics(active, oof, sources),
        "folds": folds,
        "bangkok_same_source_diagnostic": {
            "negative_event_mean": float(negative_scores.mean()),
            "positive_event_mean": float(positive_scores.mean()),
            "margin": float(positive_scores.mean() - negative_scores.mean()),
            "prototype_negative_event_mean": float(model_predict(final_prototype, negative_x).mean()),
            "prototype_positive_event_mean": float(model_predict(final_prototype, positive_x).mean()),
            "prospective_credit": False,
        },
        "final_bootstrap": bootstrap,
        "final_coefficient_sha256": final["coefficient_sha256"],
        "loss_first_last": final["loss_first_last"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.linear_report, args.training_contract,
                 args.bangkok_features, args.negative_result, args.positive_result, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.linear_report, "r767a_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"r7.67a binding mismatch: {path}")
    linear_report = lifecycle.verify_json_sidecar(args.linear_report)
    if linear_report.get("diagnostic_gate_passed") is not True:
        raise ValueError("r7.67a linear gate did not pass")
    linear_contract = common.load_json(args.linear_contract)
    linear_bound = linear_contract["bound_inputs"]
    for path, expected in (
        (args.training_contract, linear_bound["r764_training_contract_sha256"]),
        (args.model_dir / "pytorch_model.bin", linear_bound["dinov2_weights_sha256"]),
        (args.bangkok_features, linear_bound["bangkok_feature_report_sha256"]),
        (args.negative_result, linear_report["inputs"]["negative_result_sha256"]),
        (args.positive_result, linear_report["inputs"]["positive_result_sha256"]),
    ):
        if common.sha256_file(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    x, y, sources = load_relation_data(linear_contract)
    training_contract = common.load_json(args.training_contract)
    features = lifecycle.verify_json_sidecar(args.bangkok_features)
    negative = lifecycle.verify_json_sidecar(args.negative_result)
    positive = lifecycle.verify_json_sidecar(args.positive_result)
    expansion = float(linear_contract["feature_vector"]["marker_expansion_object_heights"])
    negative_x, _ = linear._bangkok_event_vectors(negative, features, training_contract, args.model_dir, expansion, args.batch_size)
    positive_x, _ = linear._bangkok_event_vectors(positive, features, training_contract, args.model_dir, expansion, args.batch_size)
    runs = [evaluate_seed(x, y, sources, negative_x, positive_x, int(seed), contract["optimizer"])
            for seed in contract["seeds"]]
    gate = contract["stability_gate"]
    for row in runs:
        metrics = row["optimized_source_macro_metrics"]
        fold_bootstraps_ok = all(
            fold["sampled_unique_active_source_count"] >= int(gate["each_fold_unique_active_sources_at_least"])
            and fold["sampled_unique_total_source_count"] >= int(gate["each_fold_unique_total_sources_at_least"])
            for fold in row["folds"]
        )
        row["checks"] = {
            "source_macro_balanced_accuracy": metrics["source_macro_balanced_accuracy"] >= float(gate["passing_run_source_macro_balanced_accuracy_at_least"]),
            "source_macro_positive_recall": metrics["source_macro_positive_recall"] >= float(gate["passing_run_each_class_source_macro_recall_at_least"]),
            "source_macro_negative_recall": metrics["source_macro_negative_recall"] >= float(gate["passing_run_each_class_source_macro_recall_at_least"]),
            "mixed_class_source_auroc_median": metrics["mixed_class_source_auroc_median"] >= float(gate["passing_run_mixed_class_source_auroc_median_at_least"]),
            "bootstrap_not_degenerate": fold_bootstraps_ok,
        }
        row["run_gate_passed"] = all(row["checks"].values())
    passing_count = sum(row["run_gate_passed"] for row in runs)
    floor = float(gate["every_run_each_class_source_macro_recall_at_least"])
    every_run_floor = all(
        row["optimized_source_macro_metrics"]["source_macro_positive_recall"] >= floor
        and row["optimized_source_macro_metrics"]["source_macro_negative_recall"] >= floor
        for row in runs
    )
    prototype_median = float(np.median([row["prototype_only_source_macro_metrics"]["source_macro_balanced_accuracy"] for row in runs]))
    optimized_median = float(np.median([row["optimized_source_macro_metrics"]["source_macro_balanced_accuracy"] for row in runs]))
    no_material_optimization_regression = optimized_median >= prototype_median - float(gate["optimized_median_max_drop_from_prototype"])
    passed = passing_count >= int(gate["runs_passing_at_least"]) and every_run_floor and no_material_optimization_regression
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "r767a_contract_sha256": common.sha256_file(args.linear_contract),
                   "r767a_report_sha256": common.sha256_file(args.linear_report)},
        "training": {"frame_count": len(y), "active_count": int((y > 0).sum()),
                     "source_count": len(set(sources.tolist())), "feature_dimension": x.shape[1]},
        "head_contract": contract["head"], "optimizer": contract["optimizer"], "runs": runs,
        "summary": {
            "run_count": len(runs), "runs_passing": passing_count,
            "oof_aurocs": [row["source_held_out_teacher_active_auroc"] for row in runs],
            "bangkok_margins": [row["bangkok_same_source_diagnostic"]["margin"] for row in runs],
            "prototype_source_macro_balanced_accuracy_median": prototype_median,
            "optimized_source_macro_balanced_accuracy_median": optimized_median,
            "every_run_class_recall_floor_passed": every_run_floor,
            "optimized_not_materially_worse_than_prototype": no_material_optimization_regression,
        },
        "bootstrap_stability_gate": {"passed": passed, "requirements": contract["stability_gate"]},
        "authorization": contract["authorization"],
        "evidence_limit": "Train-only future-route teacher supervision plus a post-hoc Bangkok diagnostic; no calibration, blind, Android, or production credit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--linear-report", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--bangkok-features", type=Path, required=True)
    parser.add_argument("--negative-result", type=Path, required=True)
    parser.add_argument("--positive-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"],
                      "stable": result["bootstrap_stability_gate"]["passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
