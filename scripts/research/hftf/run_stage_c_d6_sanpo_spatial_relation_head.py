#!/usr/bin/env python3
"""Cross-validate a relation head on fixed HFTF spatial features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_BASELINE,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    ManifestFrames,
    aggregate,
    hold_sampled_values,
    load_model,
    reference_comparison,
    sampled_indices,
    score_event,
    single_frame_spatial_features,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    FOLD_COUNT,
    FOLD_SEED,
    PRIMARY_CONFIRMATION_STEPS,
    build_training_rows,
    event_balanced_weights,
    fit_logistic,
    fold_assignments,
    predict_event,
    training_phase_labels,
    weighted_standardize,
)
from train_stage_c_d5_tartanground_development_student import sha256


SPATIAL_GRID = (3, 6)
SPATIAL_L2_STRENGTH = 1.0


def infer_spatial_matrices(
    model: torch.nn.Module,
    dataset: ManifestFrames,
    manifest: dict[str, Any],
    batch_size: int,
) -> tuple[list[np.ndarray], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    matrices = [
        np.zeros(
            (len(event["frames"]), 128 * 3 * 6),
            dtype=np.float32,
        )
        for event in manifest["events"]
    ]
    with torch.inference_mode():
        for frames, event_indices, frame_indices in loader:
            fused = single_frame_spatial_features(model, frames)
            pooled = nnf.adaptive_avg_pool2d(
                fused,
                SPATIAL_GRID,
            )
            values = pooled.flatten(1).cpu().numpy()
            for batch_index in range(len(frames)):
                event_index = int(event_indices[batch_index])
                frame_index = int(frame_indices[batch_index])
                matrices[event_index][frame_index] = values[batch_index]
    names = [
        f"channel-{channel}/row-{row}/direction-{direction}"
        for channel in range(128)
        for row in range(3)
        for direction in range(6)
    ]
    return matrices, names


def predict_spatial_event(
    event: dict[str, Any],
    matrix: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[list[bool], list[float]]:
    return predict_event(
        event,
        matrix,
        coefficient,
        intercept,
        mean,
        scale,
    )


def top_coefficients(
    names: list[str],
    values: np.ndarray,
    count: int = 24,
) -> list[dict[str, Any]]:
    indices = np.argsort(np.abs(values))[::-1][:count]
    return [
        {"feature": names[index], "coefficient": float(values[index])}
        for index in indices
    ]


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
    matrices, feature_names = infer_spatial_matrices(
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
            l2_strength=SPATIAL_L2_STRENGTH,
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
                "weighted_regularized_train_loss": loss,
                "intercept": intercept,
                "coefficient_l2_norm": float(
                    np.linalg.norm(coefficient)
                ),
                "top_absolute_coefficients": top_coefficients(
                    feature_names,
                    coefficient,
                ),
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
            "fixed_encoder_spatial_relation_head_cv_v0"
        ),
        "status": (
            "SANPO_FIXED_ENCODER_SPATIAL_RELATION_HEAD_"
            "CROSS_VALIDATION_COMPLETE"
        ),
        "policy": {
            "data_role": "consumed_development",
            "source_session_heldout": True,
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "fixed_encoder": True,
            "spatial_feature_grid": list(SPATIAL_GRID),
            "feature_channels": 128,
            "feature_count": len(feature_names),
            "event_balanced_training": True,
            "class_balanced_training": True,
            "l2_strength": SPATIAL_L2_STRENGTH,
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
