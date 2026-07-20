#!/usr/bin/env python3
"""Run five source-bootstrap short head runs on frozen trajectory features.

Each leave-one-source-group-out fold standardizes only its training episodes,
initializes a two-class linear head from the class-prototype direction, samples
training sources with replacement within each class, and takes 80 deterministic
full-batch Adam steps.  The detector and trajectory extractor remain frozen.
No model weights are saved and no calibration or production authorization is
created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_object_trajectory_probe as trajectory


SCHEMA = "blindassist_public_silver_prototype_bootstrap_short_runs_v1"
DEFAULT_SEEDS = (2026071601, 2026071602, 2026071603, 2026071604, 2026071605)


def prototype_head(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or set(y.tolist()) != {0, 1}:
        raise ValueError("prototype head needs aligned two-class features")
    negative = x[y == 0].mean(axis=0)
    positive = x[y == 1].mean(axis=0)
    direction = positive - negative
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("class prototypes are identical")
    direction /= norm
    midpoint = (positive + negative) / 2.0
    binary_bias = -float(midpoint @ direction)
    weights = np.stack([-0.5 * direction, 0.5 * direction], axis=1)
    bias = np.asarray([-0.5 * binary_bias, 0.5 * binary_bias], dtype=np.float64)
    return weights, bias


def bootstrap_source_indices(labels: np.ndarray, source_ids: Sequence[str], *, seed: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(source_ids, dtype=object)
    if len(y) != len(sources) or set(y.tolist()) != {0, 1}:
        raise ValueError("source bootstrap needs aligned two-class labels")
    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    for class_id in (0, 1):
        class_sources = sorted({source_ids[index] for index in range(len(y)) if y[index] == class_id})
        if not class_sources:
            raise ValueError(f"class {class_id} has no source groups")
        sampled_sources = rng.choice(class_sources, size=len(class_sources), replace=True)
        for source_id in sampled_sources:
            sampled_indices.extend(index for index in range(len(y)) if sources[index] == source_id and y[index] == class_id)
    return np.asarray(sampled_indices, dtype=np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_short_head(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    bias = np.asarray(initial_bias, dtype=np.float64).copy()
    if steps <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid short-run optimizer settings")
    targets = np.eye(2, dtype=np.float64)[y]
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    loss_history: list[float] = []
    for step in range(1, steps + 1):
        probabilities = softmax(x @ weights + bias)
        cross_entropy = -float(np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean())
        loss = cross_entropy + 0.5 * weight_decay * float(np.sum(weights * weights))
        error = (probabilities - targets) / len(y)
        gradient_w = x.T @ error + weight_decay * weights
        gradient_b = error.sum(axis=0)
        m_w = beta1 * m_w + (1 - beta1) * gradient_w
        v_w = beta2 * v_w + (1 - beta2) * (gradient_w * gradient_w)
        m_b = beta1 * m_b + (1 - beta1) * gradient_b
        v_b = beta2 * v_b + (1 - beta2) * (gradient_b * gradient_b)
        weights -= learning_rate * (m_w / (1 - beta1**step)) / (np.sqrt(v_w / (1 - beta2**step)) + epsilon)
        bias -= learning_rate * (m_b / (1 - beta1**step)) / (np.sqrt(v_b / (1 - beta2**step)) + epsilon)
        if step in {1, steps}:
            loss_history.append(loss)
    digest = hashlib.sha256()
    digest.update(np.asarray(weights, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(bias, dtype="<f8").tobytes(order="C"))
    return {"weights": weights, "bias": bias, "coefficient_sha256": digest.hexdigest(), "loss_first_last": loss_history}


def evaluate_seed(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: Sequence[str],
    source_ids: Sequence[str],
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    source_array = np.asarray(source_ids, dtype=object)
    predictions = np.full(len(y), -1, dtype=np.int64)
    prototype_predictions = np.full(len(y), -1, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for fold_index, held_out_source in enumerate(dict.fromkeys(source_ids)):
        holdout = source_array == held_out_source
        train = ~holdout
        if set(y[train].tolist()) != {0, 1}:
            raise ValueError(f"training fold for {held_out_source} lacks a class")
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        train_x = (x[train] - mean) / scale
        holdout_x = (x[holdout] - mean) / scale
        train_y = y[train]
        train_sources = [source_ids[index] for index in np.flatnonzero(train)]
        initial_w, initial_b = prototype_head(train_x, train_y)
        prototype_predictions[holdout] = np.argmax(holdout_x @ initial_w + initial_b, axis=1)
        sampled = bootstrap_source_indices(train_y, train_sources, seed=seed + fold_index * 1009)
        fitted = fit_short_head(
            train_x[sampled],
            train_y[sampled],
            initial_w,
            initial_b,
            steps=steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        fold_predictions = np.argmax(holdout_x @ fitted["weights"] + fitted["bias"], axis=1).astype(np.int64)
        predictions[holdout] = fold_predictions
        folds.append({
            "held_out_source_id": held_out_source,
            "held_out_episode_ids": [episode_ids[index] for index in np.flatnonzero(holdout)],
            "expected": y[holdout].tolist(),
            "prototype_only_predicted": prototype_predictions[holdout].tolist(),
            "optimized_predicted": fold_predictions.tolist(),
            "bootstrap_sample_count": int(len(sampled)),
            "bootstrap_unique_source_count": len({train_sources[index] for index in sampled}),
            "coefficient_sha256": fitted["coefficient_sha256"],
            "loss_first_last": fitted["loss_first_last"],
        })
    return {
        "seed": seed,
        "prototype_only_metrics": common.binary_metrics(y, prototype_predictions),
        "optimized_metrics": common.binary_metrics(y, predictions),
        "predictions": predictions.tolist(),
        "folds": folds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    weights = args.detector_weights.resolve()
    if not package_root.is_dir() or not weights.is_file():
        raise FileNotFoundError("package root or detector weights are missing")
    episodes, excluded = common.load_episode_specs(package_root)
    labels = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(weights))
    features, detection_summaries = trajectory.extract(detector, episodes, image_size=args.image_size, confidence=args.confidence)
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]
    feature_digest = hashlib.sha256(np.asarray(features, dtype="<f8").tobytes(order="C")).hexdigest()
    runs = [
        evaluate_seed(
            features,
            labels,
            episode_ids,
            source_ids,
            seed=seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for seed in args.seeds
    ]
    optimized_balanced = [run["optimized_metrics"]["balanced_accuracy"] for run in runs]
    prototype_balanced = [run["prototype_only_metrics"]["balanced_accuracy"] for run in runs]
    passing_runs = sum(
        run["optimized_metrics"]["balanced_accuracy"] >= args.minimum_balanced_accuracy
        and run["optimized_metrics"]["candidate_no_alert_recall"] >= args.minimum_class_recall
        and run["optimized_metrics"]["candidate_alert_recall"] >= args.minimum_class_recall
        for run in runs
    )
    stable = bool(
        passing_runs >= 4
        and float(np.median(optimized_balanced)) >= args.minimum_balanced_accuracy
        and min(run["optimized_metrics"]["candidate_no_alert_recall"] for run in runs) >= 0.40
        and min(run["optimized_metrics"]["candidate_alert_recall"] for run in runs) >= 0.40
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "class_counts": {"candidate_no_alert": int(np.sum(labels == 0)), "candidate_alert": int(np.sum(labels == 1))},
        "frozen_feature_source": {
            "extractor": "fixed yolo12n COCO proposals + deterministic object trajectories",
            "weights": str(weights),
            "weights_sha256": common.sha256_file(weights),
            "feature_dimension": int(features.shape[1]),
            "feature_matrix_sha256": feature_digest,
            "trainable_backbone_parameters": 0,
        },
        "head_contract": {
            "type": "two-class linear softmax",
            "initialization": "unit class-prototype difference and midpoint bias within each training fold",
            "bootstrap": "source groups sampled with replacement independently within each class",
            "optimizer": "deterministic full-batch Adam",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "saved_weights": False,
        },
        "evaluation": {"split": "leave_one_source_group_out", "group_key": "source_id", "frame_or_session_leakage": False},
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "prototype_only_balanced_accuracy_median": float(np.median(prototype_balanced)),
            "optimized_balanced_accuracy_values": optimized_balanced,
            "optimized_balanced_accuracy_median": float(np.median(optimized_balanced)),
            "optimized_balanced_accuracy_min": float(min(optimized_balanced)),
            "optimized_balanced_accuracy_max": float(max(optimized_balanced)),
            "runs_passing_linear_gate": passing_runs,
        },
        "head_optimization_stability_gate": {
            "passed": stable,
            "thresholds": {
                "at_least_4_of_5_runs_pass_linear_gate": True,
                "median_balanced_accuracy_gte": args.minimum_balanced_accuracy,
                "every_run_each_class_recall_gte": 0.40,
            },
            "interpretation_if_passed": "Structured trajectory features support a stable tiny head; proceed to a lifecycle-head prototype without changing the production model.",
            "interpretation_if_failed": "The linear probe signal is not bootstrap-stable; collect more matched sources before training a lifecycle head.",
        },
        "episode_detection_summaries": detection_summaries,
        "evidence_limit": "Five tiny source-bootstrap runs on GPT/VLM provisional labels; not calibration, blind evaluation, human accuracy, or production promotion evidence.",
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-trajectory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=5, default=DEFAULT_SEEDS)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--steps", type=int, choices=(80,), default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if len(set(args.seeds)) != 5:
        parser.error("five distinct seeds are required")
    if not 0 < args.confidence < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("invalid fixed detector or optimizer settings")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        **report["summary"],
        "head_optimization_stable": report["head_optimization_stability_gate"]["passed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
