#!/usr/bin/env python3
"""Probe a weakly supervised risk-profile and latent-lifecycle head.

The frozen COCO detector supplies proposals, never event truth.  A tiny linear
multiple-instance head maps deterministic per-frame object/corridor profiles to
an episode alert logit through smooth-max pooling.  Only the hash-bound episode
alert label is optimized.  Per-frame lifecycle states are decoded from the
resulting risk curve for diagnostics and are explicitly not treated as truth.

All evaluation is leave-one-source-group-out.  Five class-stratified source
bootstrap runs are supported, no weights are saved, and the report cannot
authorize calibration, blind evaluation, or production replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import cv2

import run_public_silver_frozen_feature_probe as common
import run_public_silver_motion_compensated_occupancy_probe as motion
import run_public_silver_object_trajectory_probe as trajectory
import run_public_silver_prototype_bootstrap_short_runs as bootstrap


SCHEMA = "blindassist_public_silver_risk_lifecycle_mil_head_v1"
DEFAULT_SEEDS = bootstrap.DEFAULT_SEEDS
LIFECYCLE_STATES = ("non_alert", "approach", "alertable", "post_event")


def _maximum(rows: Sequence[dict[str, Any]], key: str) -> float:
    return max((float(row[key]) for row in rows), default=0.0)


def frame_profile_vector(detections: Sequence[dict[str, Any]]) -> np.ndarray:
    """Build a fixed risk-profile basis without pixel-mask supervision."""
    rows = list(detections)
    values = [
        min(len(rows), 10) / 10.0,
        _maximum(rows, "confidence"),
        _maximum(rows, "area"),
        _maximum(rows, "bottom"),
        _maximum(rows, "corridor_overlap"),
        _maximum(rows, "threat"),
        float(sum(float(row["area"]) * float(row["corridor_overlap"]) for row in rows)),
    ]
    for group in trajectory.GROUP_NAMES:
        grouped = [row for row in rows if row["group"] == group]
        values.extend([
            min(len(grouped), 5) / 5.0,
            _maximum(grouped, "area"),
            _maximum(grouped, "bottom"),
            _maximum(grouped, "corridor_overlap"),
            _maximum(grouped, "threat"),
        ])
    return np.asarray(values, dtype=np.float64)


def causal_profile_sequence(
    frames: Sequence[Sequence[dict[str, Any]]],
    auxiliary_frame_features: np.ndarray | None = None,
) -> np.ndarray:
    if len(frames) < 2:
        raise ValueError("risk/lifecycle sequence needs at least two frames")
    base = np.stack([frame_profile_vector(frame) for frame in frames])
    if auxiliary_frame_features is not None:
        auxiliary = np.asarray(auxiliary_frame_features, dtype=np.float64)
        if auxiliary.ndim != 2 or auxiliary.shape[0] != len(base) or not np.isfinite(auxiliary).all():
            raise ValueError("auxiliary frame features must be finite and aligned with the sequence")
        base = np.concatenate([base, auxiliary], axis=1)
    delta = np.vstack([np.zeros((1, base.shape[1]), dtype=np.float64), np.diff(base, axis=0)])
    return np.concatenate([base, delta], axis=1)


def motion_residual_frame_features(
    paths: Sequence[str],
    *,
    size: int = 320,
    include_baseline_delta: bool = False,
) -> np.ndarray:
    if len(paths) < 2:
        raise ValueError("motion residual channel needs at least two frames")
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        raise ValueError("motion residual channel cannot decode an input frame")
    transitions: list[list[float]] = []
    for previous, current in zip(images, images[1:]):
        vector, summary = motion.frame_pair_descriptor(previous, current, size=size)
        reliable = float(summary["homography_success"])
        residual = float(vector[13]) if reliable else 0.0
        transitions.append([residual, reliable])
    aligned = [transitions[0], *transitions]
    values = np.asarray(aligned, dtype=np.float64)
    if include_baseline_delta:
        reliable_indices = np.flatnonzero(values[:, 1] > 0.5)
        baseline = float(values[reliable_indices[0], 0]) if len(reliable_indices) else 0.0
        relative = np.where(values[:, 1] > 0.5, np.abs(values[:, 0] - baseline), 0.0)
        values = np.concatenate([values, relative[:, None]], axis=1)
    return values


def object_occupancy_baseline_feature(frames: Sequence[Sequence[dict[str, Any]]]) -> np.ndarray:
    if len(frames) < 2:
        raise ValueError("object occupancy baseline needs at least two frames")
    occupancy = np.asarray([
        max((row["area"] * row["corridor_overlap"] for row in detections), default=0.0)
        for detections in frames
    ], dtype=np.float64)
    return np.abs(occupancy - occupancy[0])[:, None]


def corridor_appearance_frame_features(
    paths: Sequence[str],
    *,
    size: int = 320,
    include_baseline_delta: bool = False,
) -> np.ndarray:
    """Compact lower-corridor color/texture signal for non-COCO obstacles."""
    _corridor, lower = motion.corridor_masks(size)
    values: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("corridor appearance channel cannot decode an input frame")
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64) / 255.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        selected_lab = lab[lower]
        selected_gray = gray[lower]
        selected_hsv = hsv[lower]
        edges = cv2.Canny((gray * 255.0).astype(np.uint8), 70, 140)[lower]
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)[lower]
        hue = selected_hsv[:, 0]
        saturation = selected_hsv[:, 1].astype(np.float64) / 255.0
        brightness = selected_hsv[:, 2].astype(np.float64) / 255.0
        tan_fraction = np.mean(
            (hue >= 8) & (hue <= 32) & (saturation >= 0.12) & (brightness >= 0.30)
        )
        red_fraction = np.mean(
            ((hue <= 6) | (hue >= 174)) & (saturation >= 0.45) & (brightness >= 0.35)
        )
        bright_neutral_fraction = np.mean((saturation <= 0.18) & (brightness >= 0.72))
        dark_fraction = np.mean(brightness <= 0.22)
        values.append(np.asarray([
            *selected_lab.mean(axis=0).tolist(),
            *selected_lab.std(axis=0).tolist(),
            float(selected_gray.mean()),
            float(selected_gray.std()),
            float(saturation.mean()),
            float(saturation.std()),
            float(np.mean(edges > 0)),
            float(np.std(laplacian)),
            float(tan_fraction),
            float(red_fraction),
            float(bright_neutral_fraction),
            float(dark_fraction),
        ], dtype=np.float64))
    features = np.stack(values)
    if include_baseline_delta:
        features = np.concatenate([features, np.abs(features - features[0])], axis=1)
    return features


def standardize_sequences(
    train_sequences: Sequence[np.ndarray],
    other_sequences: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    stacked = np.concatenate(train_sequences, axis=0)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (
        [(sequence - mean) / scale for sequence in train_sequences],
        [(sequence - mean) / scale for sequence in other_sequences],
        mean,
        scale,
    )


def prototype_initialization(sequences: Sequence[np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, float]:
    y = np.asarray(labels, dtype=np.int64)
    if len(sequences) != len(y) or set(y.tolist()) != {0, 1}:
        raise ValueError("prototype initialization needs aligned two-class sequences")
    episode_means = np.stack([sequence.mean(axis=0) for sequence in sequences])
    negative = episode_means[y == 0].mean(axis=0)
    positive = episode_means[y == 1].mean(axis=0)
    direction = positive - negative
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("risk-profile prototypes are identical")
    weights = direction / norm
    midpoint = (positive + negative) / 2.0
    return weights, -float(midpoint @ weights)


def smooth_max(sequence_scores: np.ndarray, *, temperature: float) -> tuple[float, np.ndarray]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(sequence_scores, dtype=np.float64) / temperature
    maximum = float(scaled.max())
    exponentials = np.exp(scaled - maximum)
    attention = exponentials / exponentials.sum()
    pooled = temperature * (maximum + math.log(float(exponentials.mean())))
    return float(pooled), attention


def aggregate_sequence_scores(
    sequence_scores: np.ndarray,
    *,
    temperature: float,
    pooling: str,
) -> tuple[float, np.ndarray]:
    """Aggregate a causal frame-risk curve and return its exact score gradient."""
    scores = np.asarray(sequence_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 1:
        raise ValueError("episode pooling needs a non-empty one-dimensional score curve")
    if pooling == "smooth_max":
        return smooth_max(scores, temperature=temperature)
    if pooling == "terminal":
        attention = np.zeros(len(scores), dtype=np.float64)
        attention[-1] = 1.0
        return float(scores[-1]), attention
    raise ValueError(f"unsupported episode pooling: {pooling}")


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    clipped = np.clip(array, -40.0, 40.0)
    result = 1.0 / (1.0 + np.exp(-clipped))
    return float(result) if result.ndim == 0 else result


def loss_and_gradients(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    temperature: float,
    weight_decay: float,
    episode_pooling: str = "smooth_max",
    confidence_weights: np.ndarray | None = None,
    ranking_pairs: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    pairwise_ranking_weight: float = 0.0,
    pairwise_margin: float = 0.0,
) -> tuple[float, np.ndarray, float]:
    y = np.asarray(labels, dtype=np.int64)
    confidence = (
        np.ones(len(y), dtype=np.float64)
        if confidence_weights is None
        else np.asarray(confidence_weights, dtype=np.float64)
    )
    if confidence.shape != y.shape or np.any(~np.isfinite(confidence)) or np.any(confidence <= 0):
        raise ValueError("confidence weights must be positive, finite, and aligned with labels")
    if pairwise_ranking_weight < 0 or pairwise_margin < 0:
        raise ValueError("pairwise ranking weight and margin must be non-negative")
    sample_weights = np.zeros(len(y), dtype=np.float64)
    for class_id in (0, 1):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) == 0:
            raise ValueError("loss needs both classes")
        class_confidence = confidence[class_indices]
        sample_weights[class_indices] = 0.5 * class_confidence / class_confidence.sum()
    loss = 0.5 * weight_decay * float(weights @ weights)
    gradient_w = weight_decay * weights.copy()
    gradient_b = 0.0
    for sequence, label, sample_weight in zip(sequences, y, sample_weights):
        frame_scores = sequence @ weights + bias
        episode_logit, attention = aggregate_sequence_scores(
            frame_scores,
            temperature=temperature,
            pooling=episode_pooling,
        )
        probability = float(sigmoid(episode_logit))
        loss -= sample_weight * (
            label * math.log(max(probability, 1e-12))
            + (1 - label) * math.log(max(1.0 - probability, 1e-12))
        )
        derivative = sample_weight * (probability - float(label))
        gradient_w += derivative * np.sum(sequence * attention[:, None], axis=0)
        gradient_b += derivative
    if pairwise_ranking_weight > 0 and ranking_pairs:
        pair_weight = pairwise_ranking_weight / len(ranking_pairs)
        for negative_sequence, positive_sequence in ranking_pairs:
            negative_scores = negative_sequence @ weights + bias
            positive_scores = positive_sequence @ weights + bias
            negative_logit, negative_attention = aggregate_sequence_scores(
                negative_scores,
                temperature=temperature,
                pooling=episode_pooling,
            )
            positive_logit, positive_attention = aggregate_sequence_scores(
                positive_scores,
                temperature=temperature,
                pooling=episode_pooling,
            )
            gap = positive_logit - negative_logit
            violation = pairwise_margin - gap
            loss += pair_weight * float(np.logaddexp(0.0, violation))
            derivative = pair_weight * float(sigmoid(violation))
            negative_gradient = np.sum(negative_sequence * negative_attention[:, None], axis=0)
            positive_gradient = np.sum(positive_sequence * positive_attention[:, None], axis=0)
            gradient_w += derivative * (negative_gradient - positive_gradient)
    return float(loss), gradient_w, float(gradient_b)


def fit_head(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    episode_pooling: str = "smooth_max",
    confidence_weights: np.ndarray | None = None,
    ranking_pairs: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    pairwise_ranking_weight: float = 0.0,
    pairwise_margin: float = 0.0,
) -> dict[str, Any]:
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    bias = float(initial_bias)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = v_b = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    history: list[float] = []
    for step in range(1, steps + 1):
        loss, gradient_w, gradient_b = loss_and_gradients(
            sequences,
            labels,
            weights,
            bias,
            temperature=temperature,
            weight_decay=weight_decay,
            episode_pooling=episode_pooling,
            confidence_weights=confidence_weights,
            ranking_pairs=ranking_pairs,
            pairwise_ranking_weight=pairwise_ranking_weight,
            pairwise_margin=pairwise_margin,
        )
        m_w = beta1 * m_w + (1 - beta1) * gradient_w
        v_w = beta2 * v_w + (1 - beta2) * (gradient_w * gradient_w)
        m_b = beta1 * m_b + (1 - beta1) * gradient_b
        v_b = beta2 * v_b + (1 - beta2) * (gradient_b * gradient_b)
        weights -= learning_rate * (m_w / (1 - beta1**step)) / (np.sqrt(v_w / (1 - beta2**step)) + epsilon)
        bias -= learning_rate * (m_b / (1 - beta1**step)) / (math.sqrt(v_b / (1 - beta2**step)) + epsilon)
        if step in {1, steps}:
            history.append(loss)
    digest = hashlib.sha256()
    digest.update(np.asarray(weights, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray([bias], dtype="<f8").tobytes(order="C"))
    return {
        "weights": weights,
        "bias": bias,
        "loss_first_last": history,
        "coefficient_sha256": digest.hexdigest(),
        "ranking_pair_count": len(ranking_pairs),
    }


def decode_lifecycle(frame_probabilities: Sequence[float], *, threshold: float = 0.5) -> list[str]:
    probabilities = np.asarray(frame_probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("lifecycle decoding needs a one-dimensional multi-frame risk curve")
    alertable = np.flatnonzero(probabilities >= threshold)
    if len(alertable) == 0:
        return ["non_alert"] * len(probabilities)
    first, last = int(alertable[0]), int(alertable[-1])
    states: list[str] = []
    for index in range(len(probabilities)):
        if first <= index <= last and probabilities[index] >= threshold:
            states.append("alertable")
        elif index < first:
            states.append("approach")
        elif index > last:
            states.append("post_event")
        else:
            states.append("approach")
    return states


def score_sequence(
    sequence: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    temperature: float,
    episode_pooling: str = "smooth_max",
) -> dict[str, Any]:
    frame_logits = sequence @ weights + bias
    episode_logit, attention = aggregate_sequence_scores(
        frame_logits,
        temperature=temperature,
        pooling=episode_pooling,
    )
    frame_probabilities = np.asarray(sigmoid(frame_logits), dtype=np.float64)
    return {
        "episode_probability": float(sigmoid(episode_logit)),
        "frame_probabilities": frame_probabilities.tolist(),
        "temporal_attention": attention.tolist(),
        "latent_lifecycle_states": decode_lifecycle(frame_probabilities),
    }


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this probe's scope: {path}")


def load_train_only_augmentation_specs(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    reject_independent_direction(root)
    episodes, excluded = common.load_episode_specs(root)
    if excluded:
        raise ValueError("train-only synthetic augmentation must not contain abstaining episodes")
    for episode in episodes:
        source = common.load_json(Path(episode["source_path"]))
        contract = source.get("synthetic_counterfactual")
        if not isinstance(contract, dict) or contract.get("train_only") is not True:
            raise ValueError(f"augmentation source is not train-only synthetic: {episode['source_path']}")
        parent_source_id = contract.get("parent_source_id")
        if not isinstance(parent_source_id, str) or not parent_source_id:
            raise ValueError(f"augmentation source has no parent_source_id: {episode['source_path']}")
        episode["augmentation_parent_source_id"] = parent_source_id
    pair_ids = sorted({episode.get("counterfactual_pair_id") for episode in episodes if episode.get("counterfactual_pair_id")})
    for pair_id in pair_ids:
        rows = [episode for episode in episodes if episode.get("counterfactual_pair_id") == pair_id]
        if (
            len(rows) != 2
            or {row["label"] for row in rows} != {0, 1}
            or len({row["augmentation_parent_source_id"] for row in rows}) != 1
        ):
            raise ValueError(f"train-only augmentation pair is incomplete: {pair_id}")
    if len(pair_ids) * 2 != len(episodes):
        raise ValueError("every train-only augmentation episode must belong to one complete pair")
    return episodes


def evaluate_seed(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    episode_ids: Sequence[str],
    source_ids: Sequence[str],
    pair_ids: Sequence[str | None],
    confidence_weights: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    episode_pooling: str = "smooth_max",
    pairwise_ranking_weight: float,
    pairwise_margin: float,
    minimum_pair_confidence: float,
    augmentation_sequences: Sequence[np.ndarray] | None = None,
    augmentation_labels: np.ndarray | None = None,
    augmentation_source_ids: Sequence[str] | None = None,
    augmentation_pair_ids: Sequence[str | None] | None = None,
    augmentation_confidence_weights: np.ndarray | None = None,
    augmentation_parent_source_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    source_array = np.asarray(source_ids, dtype=object)
    aug_sequences = list(augmentation_sequences or [])
    aug_y = np.asarray(
        augmentation_labels if augmentation_labels is not None else [],
        dtype=np.int64,
    )
    aug_sources = list(augmentation_source_ids or [])
    aug_pairs = list(augmentation_pair_ids or [])
    aug_confidence = np.asarray(
        augmentation_confidence_weights if augmentation_confidence_weights is not None else [],
        dtype=np.float64,
    )
    aug_parents = list(augmentation_parent_source_ids or [])
    augmentation_lengths = {
        len(aug_sequences),
        len(aug_y),
        len(aug_sources),
        len(aug_pairs),
        len(aug_confidence),
        len(aug_parents),
    }
    if len(augmentation_lengths) != 1:
        raise ValueError("train-only augmentation inputs must be aligned")
    if len(aug_y) and set(aug_y.tolist()) != {0, 1}:
        raise ValueError("train-only augmentation must contain both classes")
    predictions = np.full(len(y), -1, dtype=np.int64)
    probabilities = np.full(len(y), np.nan, dtype=np.float64)
    profiles: list[dict[str, Any] | None] = [None] * len(y)
    folds: list[dict[str, Any]] = []
    for fold_index, held_out_source in enumerate(dict.fromkeys(source_ids)):
        holdout_indices = np.flatnonzero(source_array == held_out_source)
        train_indices = np.flatnonzero(source_array != held_out_source)
        train_y = y[train_indices]
        if set(train_y.tolist()) != {0, 1}:
            raise ValueError(f"training fold for {held_out_source} lacks a class")
        eligible_augmentation = [
            index for index, parent_source in enumerate(aug_parents)
            if parent_source != held_out_source
        ]
        train_sequences = [sequences[index] for index in train_indices] + [
            aug_sequences[index] for index in eligible_augmentation
        ]
        combined_train_y = np.concatenate([
            train_y,
            aug_y[eligible_augmentation],
        ])
        combined_train_confidences = np.concatenate([
            confidence_weights[train_indices],
            aug_confidence[eligible_augmentation],
        ])
        combined_train_sources = [source_ids[index] for index in train_indices] + [
            aug_sources[index] for index in eligible_augmentation
        ]
        combined_train_pair_ids = [pair_ids[index] for index in train_indices] + [
            aug_pairs[index] for index in eligible_augmentation
        ]
        holdout_sequences = [sequences[index] for index in holdout_indices]
        train_sequences, holdout_sequences, _mean, _scale = standardize_sequences(train_sequences, holdout_sequences)
        initial_w, initial_b = prototype_initialization(train_sequences, combined_train_y)
        ranking_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for pair_id in sorted({value for value in combined_train_pair_ids if value}):
            local_indices = [index for index, value in enumerate(combined_train_pair_ids) if value == pair_id]
            if (
                len(local_indices) == 2
                and set(combined_train_y[local_indices].tolist()) == {0, 1}
                and float(np.min(combined_train_confidences[local_indices])) >= minimum_pair_confidence
            ):
                negative = next(index for index in local_indices if combined_train_y[index] == 0)
                positive = next(index for index in local_indices if combined_train_y[index] == 1)
                ranking_pairs.append((train_sequences[negative], train_sequences[positive]))
        sampled = bootstrap.bootstrap_source_indices(
            combined_train_y,
            combined_train_sources,
            seed=seed + fold_index * 1009,
        )
        fitted = fit_head(
            [train_sequences[index] for index in sampled],
            combined_train_y[sampled],
            initial_w,
            initial_b,
            steps=steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            temperature=temperature,
            episode_pooling=episode_pooling,
            confidence_weights=combined_train_confidences[sampled],
            ranking_pairs=ranking_pairs,
            pairwise_ranking_weight=pairwise_ranking_weight,
            pairwise_margin=pairwise_margin,
        )
        fold_probabilities: list[float] = []
        for local_index, episode_index in enumerate(holdout_indices):
            profile = score_sequence(
                holdout_sequences[local_index],
                fitted["weights"],
                fitted["bias"],
                temperature=temperature,
                episode_pooling=episode_pooling,
            )
            profiles[episode_index] = profile
            probabilities[episode_index] = profile["episode_probability"]
            predictions[episode_index] = int(profile["episode_probability"] >= 0.5)
            fold_probabilities.append(profile["episode_probability"])
        folds.append({
            "held_out_source_id": held_out_source,
            "held_out_episode_ids": [episode_ids[index] for index in holdout_indices],
            "expected": y[holdout_indices].tolist(),
            "predicted": predictions[holdout_indices].tolist(),
            "episode_probabilities": fold_probabilities,
            "coefficient_sha256": fitted["coefficient_sha256"],
            "loss_first_last": fitted["loss_first_last"],
            "ranking_pair_count": fitted["ranking_pair_count"],
            "train_only_augmentation_episode_count": len(eligible_augmentation),
            "train_only_augmentation_source_ids": sorted({
                aug_sources[index] for index in eligible_augmentation
            }),
            "parent_matched_augmentation_excluded_count": len(aug_sequences) - len(eligible_augmentation),
        })
    pair_results: list[dict[str, Any]] = []
    for pair_id in sorted({value for value in pair_ids if value}):
        indices = [index for index, value in enumerate(pair_ids) if value == pair_id]
        if len(indices) == 2 and set(y[indices].tolist()) == {0, 1}:
            negative = next(index for index in indices if y[index] == 0)
            positive = next(index for index in indices if y[index] == 1)
            pair_results.append({
                "counterfactual_pair_id": pair_id,
                "no_alert_episode_id": episode_ids[negative],
                "alert_episode_id": episode_ids[positive],
                "no_alert_probability": float(probabilities[negative]),
                "alert_probability": float(probabilities[positive]),
                "correct_probability_order": bool(probabilities[positive] > probabilities[negative]),
            })
    return {
        "seed": seed,
        "metrics": common.binary_metrics(y, predictions),
        "counterfactual_pairs": pair_results,
        "folds": folds,
        "episode_profiles": [
            {"episode_id": episode_ids[index], "expected": int(y[index]), **(profiles[index] or {})}
            for index in range(len(y))
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    weights_path = args.detector_weights.resolve()
    reject_independent_direction(package_root)
    reject_independent_direction(weights_path)
    if not package_root.is_dir() or not weights_path.is_file():
        raise FileNotFoundError("package root or detector weights are missing")
    episodes, excluded = common.load_episode_specs(package_root)
    augmentation_episodes = (
        load_train_only_augmentation_specs(args.train_only_augmentation_root)
        if args.train_only_augmentation_root is not None
        else []
    )
    if {episode["source_id"] for episode in episodes} & {
        episode["source_id"] for episode in augmentation_episodes
    }:
        raise ValueError("real and train-only synthetic source IDs must be disjoint")
    if {episode["episode_id"] for episode in episodes} & {
        episode["episode_id"] for episode in augmentation_episodes
    }:
        raise ValueError("real and train-only synthetic episode IDs must be disjoint")
    labels = np.asarray([episode["label"] for episode in episodes], dtype=np.int64)
    episode_confidences = np.asarray([episode["confidence"] for episode in episodes], dtype=np.float64)
    confidence_weights = (
        episode_confidences
        if args.confidence_weighting == "linear"
        else np.ones(len(episodes), dtype=np.float64)
    )
    source_ids = [episode["source_id"] for episode in episodes]
    episode_ids = [episode["episode_id"] for episode in episodes]
    pair_ids = [episode.get("counterfactual_pair_id") for episode in episodes]
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(weights_path))
    all_episodes = episodes + augmentation_episodes
    frame_detections = trajectory.extract_frame_detections(
        detector,
        all_episodes,
        image_size=args.image_size,
        confidence=args.confidence,
    )
    all_sequences = []
    for episode, frames in zip(all_episodes, frame_detections):
        auxiliary_parts: list[np.ndarray] = []
        if args.motion_residual_channel:
            auxiliary_parts.append(motion_residual_frame_features(
                [frame["path"] for frame in episode["frames"]],
                size=args.motion_size,
                include_baseline_delta=args.temporal_baseline_channels,
            ))
        if args.temporal_baseline_channels:
            auxiliary_parts.append(object_occupancy_baseline_feature(frames))
        if args.corridor_appearance_channel:
            auxiliary_parts.append(corridor_appearance_frame_features(
                [frame["path"] for frame in episode["frames"]],
                size=args.motion_size,
                include_baseline_delta=args.appearance_baseline_delta,
            ))
        auxiliary = np.concatenate(auxiliary_parts, axis=1) if auxiliary_parts else None
        all_sequences.append(causal_profile_sequence(frames, auxiliary))
    sequences = all_sequences[:len(episodes)]
    augmentation_sequences = all_sequences[len(episodes):]
    augmentation_labels = np.asarray(
        [episode["label"] for episode in augmentation_episodes],
        dtype=np.int64,
    )
    augmentation_confidences = np.asarray(
        [episode["confidence"] for episode in augmentation_episodes],
        dtype=np.float64,
    )
    augmentation_confidence_weights = (
        augmentation_confidences
        if args.confidence_weighting == "linear"
        else np.ones(len(augmentation_episodes), dtype=np.float64)
    )
    augmentation_source_ids = [episode["source_id"] for episode in augmentation_episodes]
    augmentation_pair_ids = [
        episode.get("counterfactual_pair_id") for episode in augmentation_episodes
    ]
    augmentation_parent_source_ids = [
        episode["augmentation_parent_source_id"] for episode in augmentation_episodes
    ]
    sequence_digest = hashlib.sha256()
    for sequence in all_sequences:
        sequence_digest.update(np.asarray(sequence.shape, dtype="<i8").tobytes())
        sequence_digest.update(np.asarray(sequence, dtype="<f8").tobytes(order="C"))
    runs = [
        evaluate_seed(
            sequences,
            labels,
            episode_ids,
            source_ids,
            pair_ids,
            confidence_weights,
            seed=seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            episode_pooling=args.episode_pooling,
            pairwise_ranking_weight=args.pairwise_ranking_weight,
            pairwise_margin=args.pairwise_margin,
            minimum_pair_confidence=args.minimum_pair_confidence,
            augmentation_sequences=augmentation_sequences,
            augmentation_labels=augmentation_labels,
            augmentation_source_ids=augmentation_source_ids,
            augmentation_pair_ids=augmentation_pair_ids,
            augmentation_confidence_weights=augmentation_confidence_weights,
            augmentation_parent_source_ids=augmentation_parent_source_ids,
        )
        for seed in args.seeds
    ]
    balanced = [run["metrics"]["balanced_accuracy"] for run in runs]
    pair_order_rates = [
        float(np.mean([pair["correct_probability_order"] for pair in run["counterfactual_pairs"]]))
        for run in runs
    ]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "train_only_augmentation": {
            "enabled": bool(augmentation_episodes),
            "root": (
                str(args.train_only_augmentation_root.resolve())
                if args.train_only_augmentation_root is not None
                else None
            ),
            "episode_count": len(augmentation_episodes),
            "pair_count": len({
                episode.get("counterfactual_pair_id")
                for episode in augmentation_episodes
                if episode.get("counterfactual_pair_id")
            }),
            "parent_source_ids": sorted(set(augmentation_parent_source_ids)),
            "holdout_exclusion_contract": (
                "augmentation is training-only and excluded whenever its parent real source is held out"
                if augmentation_episodes
                else "disabled"
            ),
            "counted_in_metrics": False,
            "human_event_truth_present": False,
        },
        "frozen_input": {
            "extractor": "fixed yolo12n COCO proposals + deterministic object/corridor frame profiles",
            "weights": str(weights_path),
            "weights_sha256": common.sha256_file(weights_path),
            "frame_feature_dimension": int(sequences[0].shape[1]),
            "sequence_sha256": sequence_digest.hexdigest(),
            "motion_residual_channel": args.motion_residual_channel,
            "temporal_baseline_channels": args.temporal_baseline_channels,
            "corridor_appearance_channel": args.corridor_appearance_channel,
            "appearance_baseline_delta": args.appearance_baseline_delta,
            "motion_residual_contract": (
                "per-frame registered lower-corridor mean grayscale residual + homography reliability; first transition repeated at frame zero"
                if args.motion_residual_channel
                else "disabled"
            ),
            "temporal_baseline_contract": (
                "absolute change from first reliable registered residual plus absolute change from first-frame max object area*corridor overlap"
                if args.temporal_baseline_channels
                else "disabled"
            ),
            "corridor_appearance_contract": (
                "16 deterministic lower-corridor LAB, grayscale, saturation, edge, texture, tan, red, bright-neutral, and dark statistics"
                + (" plus absolute change from the first frame" if args.appearance_baseline_delta else "")
                if args.corridor_appearance_channel
                else "disabled"
            ),
            "pixel_segmentation_role": "auxiliary_only_not_consumed_by_primary_head",
            "trainable_backbone_parameters": 0,
        },
        "head_contract": {
            "type": (
                "linear per-frame risk profile + smooth-max multiple-instance episode head"
                if args.episode_pooling == "smooth_max"
                else "linear per-frame risk profile + terminal current-state episode head"
            ),
            "episode_pooling": args.episode_pooling,
            "episode_pooling_contract": (
                "smooth log-mean-exp over all frame logits"
                if args.episode_pooling == "smooth_max"
                else "episode supervision and pair ranking use only the final causal frame logit"
            ),
            "episode_supervision": "hash-bound provisional candidate_alert/candidate_no_alert only",
            "lifecycle_supervision": "none",
            "lifecycle_output": "latent diagnostic decoded from the learned risk curve",
            "source_bootstrap": "within-class source groups with replacement",
            "loss_confidence_weighting": args.confidence_weighting,
            "pairwise_ranking_weight": args.pairwise_ranking_weight,
            "pairwise_margin": args.pairwise_margin,
            "minimum_pair_confidence": args.minimum_pair_confidence,
            "pairwise_ranking_contract": (
                "full non-held-out source pairs add smooth logistic margin loss; bootstrap remains BCE-only sampling"
                if args.pairwise_ranking_weight > 0
                else "disabled"
            ),
            "confidence_weighting_contract": (
                "within each class, episode loss mass is proportional to the bound silver confidence and renormalized to 0.5"
                if args.confidence_weighting == "linear"
                else "disabled; equal episode loss mass within each class"
            ),
            "saved_weights": False,
        },
        "evaluation": {
            "split": "leave_one_source_group_out",
            "group_key": "source_id",
            "metrics_population": "real provisional public episodes only",
            "train_only_synthetic_counted_in_metrics": False,
        },
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "balanced_accuracy_values": balanced,
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_min": float(min(balanced)),
            "balanced_accuracy_max": float(max(balanced)),
            "counterfactual_pair_order_rate_values": pair_order_rates,
            "counterfactual_pair_order_rate_median": float(np.median(pair_order_rates)),
        },
        "evidence_limit": "Lifecycle states are weak latent diagnostics, not frame truth or lifecycle accuracy. Tiny GPT/VLM provisional set only.",
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
    parser.add_argument("--train-only-augmentation-root", type=Path)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-trajectory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--episode-pooling", choices=("smooth_max", "terminal"), default="smooth_max")
    parser.add_argument("--motion-residual-channel", action="store_true")
    parser.add_argument("--temporal-baseline-channels", action="store_true")
    parser.add_argument("--corridor-appearance-channel", action="store_true")
    parser.add_argument("--appearance-baseline-delta", action="store_true")
    parser.add_argument("--motion-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence-weighting", choices=("none", "linear"), default="none")
    parser.add_argument("--pairwise-ranking-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--minimum-pair-confidence", type=float, default=0.65)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.appearance_baseline_delta and not args.corridor_appearance_channel:
        parser.error("--appearance-baseline-delta requires --corridor-appearance-channel")
    if args.pairwise_ranking_weight < 0 or args.pairwise_margin < 0 or not 0 < args.minimum_pair_confidence <= 1:
        parser.error("pairwise settings must be non-negative and minimum confidence must be in (0,1]")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
