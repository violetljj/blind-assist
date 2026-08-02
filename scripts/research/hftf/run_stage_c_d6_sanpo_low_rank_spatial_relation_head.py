#!/usr/bin/env python3
"""Cross-validate a rank-constrained head on fixed HFTF spatial features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_BASELINE,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    ManifestFrames,
    aggregate,
    load_model,
    reference_comparison,
    score_event,
)
from run_stage_c_d6_sanpo_spatial_relation_head import (
    SPATIAL_GRID,
    infer_spatial_matrices,
    predict_spatial_event,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    FOLD_COUNT,
    FOLD_SEED,
    PRIMARY_CONFIRMATION_STEPS,
    build_training_rows,
    event_balanced_weights,
    fit_logistic,
    fold_assignments,
    weighted_standardize,
)
from train_stage_c_d5_tartanground_development_student import sha256


FEATURE_CHANNELS = 128
SPATIAL_CELLS = SPATIAL_GRID[0] * SPATIAL_GRID[1]
LOW_RANK = 2
LOW_RANK_L2_STRENGTH = 1.0


def weighted_logistic_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    coefficient: torch.Tensor,
    intercept: torch.Tensor,
    l2_strength: float,
) -> torch.Tensor:
    logits = features @ coefficient.flatten() + intercept
    data_loss = (
        nnf.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        * sample_weights
    ).sum() / sample_weights.sum()
    regularization = (
        0.5 * l2_strength * coefficient.square().mean()
    )
    return data_loss + regularization


def fit_low_rank_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    channels: int = FEATURE_CHANNELS,
    cells: int = SPATIAL_CELLS,
    rank: int = LOW_RANK,
    l2_strength: float = LOW_RANK_L2_STRENGTH,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if features.shape[1] != channels * cells:
        raise ValueError(
            f"Expected {channels * cells} features, "
            f"got {features.shape[1]}"
        )
    if not 0 < rank <= min(channels, cells):
        raise ValueError("Rank must fit both coefficient dimensions")

    full_coefficient, full_intercept, full_loss = fit_logistic(
        features,
        labels,
        sample_weights,
        l2_strength=l2_strength,
    )
    matrix = full_coefficient.reshape(channels, cells)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    root = np.sqrt(singular_values[:rank])
    left = torch.tensor(
        left_vectors[:, :rank] * root[None, :],
        dtype=torch.float64,
        requires_grad=True,
    )
    right = torch.tensor(
        root[:, None] * right_vectors[:rank],
        dtype=torch.float64,
        requires_grad=True,
    )
    intercept = torch.tensor(
        full_intercept,
        dtype=torch.float64,
        requires_grad=True,
    )
    x = torch.from_numpy(features).double()
    y = torch.from_numpy(labels.astype(np.float64)).double()
    weights = torch.from_numpy(sample_weights).double()

    with torch.no_grad():
        initial_loss = float(
            weighted_logistic_loss(
                x,
                y,
                weights,
                left @ right,
                intercept,
                l2_strength,
            )
        )
    optimizer = torch.optim.LBFGS(
        [left, right, intercept],
        lr=1.0,
        max_iter=200,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = weighted_logistic_loss(
            x,
            y,
            weights,
            left @ right,
            intercept,
            l2_strength,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    coefficient = (left @ right).detach()
    final_loss = float(
        weighted_logistic_loss(
            x,
            y,
            weights,
            coefficient,
            intercept.detach(),
            l2_strength,
        )
    )
    return (
        coefficient.numpy().reshape(-1),
        float(intercept.detach()),
        {
            "rank": rank,
            "parameter_count": channels * rank + rank * cells + 1,
            "unconstrained_parameter_count": channels * cells + 1,
            "full_rank_reference_loss": full_loss,
            "truncated_svd_initial_loss": initial_loss,
            "low_rank_final_loss": final_loss,
            "full_rank_singular_values": [
                float(value) for value in singular_values
            ],
            "coefficient_l2_norm": float(
                torch.linalg.vector_norm(coefficient)
            ),
        },
    )


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
    matrices, _ = infer_spatial_matrices(
        model,
        dataset,
        manifest,
        args.batch_size,
    )
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
        weights = event_balanced_weights(
            training_event_ids,
            y_train,
        )
        mean, scale = weighted_standardize(
            x_train,
            weights,
        )
        standardized = (x_train - mean) / scale
        coefficient, intercept, fit = fit_low_rank_logistic(
            standardized,
            y_train,
            weights,
        )
        heldout = [
            event
            for event in events
            if assignments[event["parent_event_id"]] == fold
        ]
        for event in heldout:
            event_index = events.index(event)
            active, probabilities = predict_spatial_event(
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
                **fit,
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
            "low_rank_spatial_relation_head_cv_v0"
        ),
        "status": (
            "SANPO_LOW_RANK_SPATIAL_RELATION_HEAD_"
            "CROSS_VALIDATION_COMPLETE"
        ),
        "policy": {
            "data_role": "consumed_development",
            "source_session_heldout": True,
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "fixed_encoder": True,
            "fixed_backbone": True,
            "spatial_feature_grid": list(SPATIAL_GRID),
            "feature_channels": FEATURE_CHANNELS,
            "spatial_cells": SPATIAL_CELLS,
            "coefficient_rank": LOW_RANK,
            "trainable_parameter_count": (
                FEATURE_CHANNELS * LOW_RANK
                + LOW_RANK * SPATIAL_CELLS
                + 1
            ),
            "event_balanced_training": True,
            "class_balanced_training": True,
            "l2_strength": LOW_RANK_L2_STRENGTH,
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
