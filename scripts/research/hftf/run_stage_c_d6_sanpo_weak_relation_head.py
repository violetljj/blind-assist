#!/usr/bin/env python3
"""Cross-validate a weak actionability head on fixed HFTF fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf

from diagnose_stage_c_d6_sanpo_direction_profile import (
    POSITIVE_BUCKETS,
    direction_features,
)
from evaluate_stage_c_d5_tartanground_event_proxy import (
    causal_confirmation,
)
from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_BASELINE,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    ManifestFrames,
    aggregate,
    hold_sampled_values,
    infer_manifest_probabilities,
    load_model,
    reference_comparison,
    sampled_indices,
    score_event,
)
from train_stage_c_d5_tartanground_development_student import sha256


FOLD_SEED = "HFTF_D6_SANPO_WEAK_RELATION_HEAD_V0"
FOLD_COUNT = 5
L2_STRENGTH = 0.1
PRIMARY_CONFIRMATION_STEPS = 2


def fold_assignments(
    events: list[dict[str, Any]],
) -> dict[str, int]:
    output = {}
    buckets = sorted({event["bucket"] for event in events})
    for bucket in buckets:
        rows = [
            event for event in events if event["bucket"] == bucket
        ]
        rows.sort(
            key=lambda event: hashlib.sha256(
                (
                    f"{FOLD_SEED}:{event['parent_event_id']}"
                ).encode("utf-8")
            ).hexdigest()
        )
        for index, event in enumerate(rows):
            output[event["parent_event_id"]] = index % FOLD_COUNT
    if len(output) != len(events):
        raise ValueError("Every event must receive exactly one fold")
    return output


def feature_matrix(
    risk: np.ndarray,
    known: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    values = direction_features(risk, known)
    names = []
    columns = []
    for family in sorted(values):
        for direction in range(6):
            names.append(f"{family}/direction-{direction}")
            columns.append(values[family][:, direction])
    return np.stack(columns, axis=1).astype(np.float64), names


def training_phase_labels(
    event: dict[str, Any],
) -> dict[int, int]:
    sample_set = set(sampled_indices(event))
    if event["bucket"] not in POSITIVE_BUCKETS:
        return {index: 0 for index in sorted(sample_set)}
    alert_start, alert_end = map(
        int,
        event["alertable_interval_frames"],
    )
    passed_start, passed_end = map(
        int,
        event["passed_interval_frames"],
    )
    output = {}
    for index in sorted(sample_set):
        if alert_start <= index <= alert_end:
            output[index] = 1
        elif passed_start <= index <= passed_end:
            output[index] = 0
    return output


def event_balanced_weights(
    event_ids: list[str],
    labels: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(len(event_ids), dtype=np.float64)
    for event_id in sorted(set(event_ids)):
        indices = [
            index
            for index, value in enumerate(event_ids)
            if value == event_id
        ]
        weights[indices] = 1.0 / len(indices)
    for label in (0, 1):
        mask = labels == label
        total = float(weights[mask].sum())
        if total <= 0.0:
            raise ValueError(f"Missing class {label} in training fold")
        weights[mask] *= 0.5 / total
    return weights


def weighted_standardize(
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / weights.sum()
    mean = np.sum(features * normalized[:, None], axis=0)
    variance = np.sum(
        np.square(features - mean) * normalized[:, None],
        axis=0,
    )
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return mean, scale


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    x = torch.from_numpy(features).double()
    y = torch.from_numpy(labels.astype(np.float64)).double()
    weights = torch.from_numpy(sample_weights).double()
    coefficient = torch.zeros(
        features.shape[1],
        dtype=torch.float64,
        requires_grad=True,
    )
    intercept = torch.zeros(
        1,
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [coefficient, intercept],
        lr=1.0,
        max_iter=200,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def loss_value() -> torch.Tensor:
        logits = x @ coefficient + intercept
        data_loss = (
            nnf.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none",
            )
            * weights
        ).sum() / weights.sum()
        regularization = (
            0.5 * L2_STRENGTH * coefficient.square().mean()
        )
        return data_loss + regularization

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = float(loss_value().detach())
    return (
        coefficient.detach().numpy(),
        float(intercept.detach()[0]),
        final_loss,
    )


def build_training_rows(
    events: list[dict[str, Any]],
    matrices: list[np.ndarray],
    assignments: dict[str, int],
    heldout_fold: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    event_ids = []
    for event, matrix in zip(events, matrices, strict=True):
        if assignments[event["parent_event_id"]] == heldout_fold:
            continue
        for frame_index, label in training_phase_labels(event).items():
            features.append(matrix[frame_index])
            labels.append(label)
            event_ids.append(event["parent_event_id"])
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        event_ids,
    )


def predict_event(
    event: dict[str, Any],
    matrix: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[list[bool], list[float]]:
    indices = sampled_indices(event)
    standardized = (matrix[indices] - mean) / scale
    logits = standardized @ coefficient + intercept
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    immediate = [bool(value >= 0.5) for value in probabilities]
    confirmed = causal_confirmation(
        immediate,
        PRIMARY_CONFIRMATION_STEPS,
    )
    active = hold_sampled_values(
        confirmed,
        indices,
        len(event["frames"]),
    )
    held_probabilities = [0.0] * len(event["frames"])
    for sample_number, start in enumerate(indices):
        end = (
            indices[sample_number + 1]
            if sample_number + 1 < len(indices)
            else len(event["frames"])
        )
        held_probabilities[start:end] = [
            float(probabilities[sample_number])
        ] * (end - start)
    return active, held_probabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    events = manifest["events"]
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in events) != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    model, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    dataset = ManifestFrames(args.manifest, manifest)
    risks, knowns = infer_manifest_probabilities(
        model,
        dataset,
        manifest,
        args.batch_size,
    )
    matrices = []
    feature_names = None
    for risk, known in zip(risks, knowns, strict=True):
        matrix, names = feature_matrix(risk, known)
        matrices.append(matrix)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature order drift")
    if feature_names is None:
        raise ValueError("No features")

    assignments = fold_assignments(events)
    event_outputs: dict[str, dict[str, Any]] = {}
    fold_outputs = []
    for fold in range(FOLD_COUNT):
        x_train, y_train, training_event_ids = build_training_rows(
            events,
            matrices,
            assignments,
            fold,
        )
        initial_weights = event_balanced_weights(
            training_event_ids,
            y_train,
        )
        mean, scale = weighted_standardize(
            x_train,
            initial_weights,
        )
        standardized = (x_train - mean) / scale
        coefficient, intercept, loss = fit_logistic(
            standardized,
            y_train,
            initial_weights,
        )
        heldout = [
            event
            for event in events
            if assignments[event["parent_event_id"]] == fold
        ]
        for event in heldout:
            event_index = events.index(event)
            active, probabilities = predict_event(
                event,
                matrices[event_index],
                coefficient,
                intercept,
                mean,
                scale,
            )
            score = score_event(event, active)
            score["heldout_fold"] = fold
            score["probability_median"] = float(
                np.median(probabilities)
            )
            score["probability_max"] = float(
                np.max(probabilities)
            )
            event_outputs[event["parent_event_id"]] = score
        fold_outputs.append(
            {
                "fold": fold,
                "train_event_count": len(set(training_event_ids)),
                "train_frame_count": len(y_train),
                "train_positive_frame_count": int(y_train.sum()),
                "test_event_ids": sorted(
                    event["parent_event_id"] for event in heldout
                ),
                "test_bucket_counts": {
                    bucket: sum(
                        event["bucket"] == bucket for event in heldout
                    )
                    for bucket in sorted(
                        {event["bucket"] for event in events}
                    )
                },
                "weighted_regularized_train_loss": loss,
                "intercept": intercept,
                "coefficient_l2_norm": float(
                    np.linalg.norm(coefficient)
                ),
                "coefficients": {
                    name: float(value)
                    for name, value in zip(
                        feature_names,
                        coefficient,
                        strict=True,
                    )
                },
                "standardization": {
                    "mean": {
                        name: float(value)
                        for name, value in zip(
                            feature_names,
                            mean,
                            strict=True,
                        )
                    },
                    "scale": {
                        name: float(value)
                        for name, value in zip(
                            feature_names,
                            scale,
                            strict=True,
                        )
                    },
                },
            }
        )
    if set(event_outputs) != {
        event["parent_event_id"] for event in events
    }:
        raise ValueError("Out-of-fold event coverage mismatch")
    scored_events = [
        event_outputs[event["parent_event_id"]] for event in events
    ]
    metrics = aggregate(scored_events)
    baseline_result = json.loads(
        args.baseline.read_text(encoding="utf-8")
    )
    baseline = baseline_result["event_evaluation"][
        "current_yolo_reference"
    ]
    result = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_"
            "weak_relation_head_cross_validation_v0"
        ),
        "status": "SANPO_WEAK_RELATION_HEAD_CROSS_VALIDATION_COMPLETE",
        "policy": {
            "data_role": "consumed_development",
            "source_session_heldout": True,
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "fixed_backbone": True,
            "feature_family": (
                "five_hftf_direction_profiles_x_six_directions"
            ),
            "feature_count": len(feature_names),
            "event_balanced_training": True,
            "class_balanced_training": True,
            "l2_strength": L2_STRENGTH,
            "probability_threshold": 0.5,
            "causal_confirmation_steps_at_5hz": (
                PRIMARY_CONFIRMATION_STEPS
            ),
            "test_sessions_used_for_standardization_or_fit": False,
            "human_safety_or_app_claim": False,
        },
        "model": {
            "name": args.name,
            "architecture": checkpoint.get("architecture", "pooled"),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "event_count": manifest["event_count"],
            "frame_count": len(dataset),
            "bucket_counts": manifest["bucket_counts"],
        },
        "current_yolo_reference": baseline,
        "metrics": metrics,
        "comparison_to_current_yolo": reference_comparison(
            metrics,
            baseline,
        ),
        "folds": fold_outputs,
        "events": scored_events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
