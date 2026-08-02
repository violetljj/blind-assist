#!/usr/bin/env python3
"""Evaluate explicit RAFT motion increment on THOR future onset."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
    summarize,
)
from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    DEFAULT_FEATURES as DEFAULT_SPATIAL_FEATURES,
    DEFAULT_SAMPLES,
    TARGETS,
    masked_source_weights,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d14_thor_magni_"
    "explicit_raft_motion_future_onset_canary_v0"
)
DEFAULT_MOTION_FEATURES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d14-thor-magni-explicit-motion-features-v0/features.npz"
)
SEEDS = (17, 23, 41)
EPOCHS = 120
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3
AUROC_MEAN_FLOOR = 0.01
AP_MEAN_FLOOR = 0.005


class CurrentMotionOnsetHead(nn.Module):
    """Combine current spatial context with optional explicit motion."""

    def __init__(self) -> None:
        super().__init__()
        self.current_normalization = nn.GroupNorm(32, 576)
        self.current_projection = nn.Conv2d(
            576,
            16,
            kernel_size=1,
        )
        self.motion_normalization = nn.LayerNorm(4 * 8 * 3 * 6)
        self.motion_projection = nn.Linear(4 * 8 * 3 * 6, 64)
        self.head = nn.Linear(16 * 4 * 7 + 64, 2)

    def forward(
        self,
        current: torch.Tensor,
        motion: torch.Tensor,
        *,
        arm: str,
    ) -> torch.Tensor:
        if current.shape[1:] != (576, 4, 7):
            raise ValueError("Expected current [batch,576,4,7]")
        if motion.shape[1:] != (4, 8, 3, 6):
            raise ValueError("Expected motion [batch,4,8,3,6]")
        if arm == "current":
            motion = torch.zeros_like(motion)
        elif arm != "motion":
            raise ValueError(f"Unknown arm: {arm}")
        current_hidden = nnf.hardswish(
            self.current_projection(
                self.current_normalization(current)
            )
        ).flatten(1)
        motion_hidden = nnf.hardswish(
            self.motion_projection(
                self.motion_normalization(motion.flatten(1))
            )
        )
        return self.head(
            torch.cat((current_hidden, motion_hidden), dim=1)
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_features(
    path: Path,
    expected_ids: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str)
        features = payload["features"].astype(np.float32)
    if not np.array_equal(sample_ids, expected_ids):
        raise ValueError(f"Feature sample ordering mismatch: {path}")
    if features.shape != expected_shape:
        raise ValueError(
            f"Feature shape mismatch: {features.shape} != {expected_shape}"
        )
    return features


def normalize_fold(
    spatial: np.ndarray,
    motion: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_mean = np.mean(
        spatial[train_indices],
        axis=(0, 2, 3),
        keepdims=True,
    )
    spatial_scale = np.std(
        spatial[train_indices],
        axis=(0, 2, 3),
        keepdims=True,
    )
    spatial_scale[spatial_scale < 1e-6] = 1.0
    motion_mean = np.mean(
        motion[train_indices],
        axis=(0, 1, 3, 4),
        keepdims=True,
    )
    motion_scale = np.std(
        motion[train_indices],
        axis=(0, 1, 3, 4),
        keepdims=True,
    )
    motion_scale[motion_scale < 1e-6] = 1.0
    return (
        ((spatial - spatial_mean) / spatial_scale).astype(np.float32),
        ((motion - motion_mean) / motion_scale).astype(np.float32),
    )


def train_arm(
    spatial: np.ndarray,
    motion: np.ndarray,
    labels: np.ndarray,
    eligibility: np.ndarray,
    sources: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_everything(seed)
    train_spatial = torch.from_numpy(spatial[train_indices]).to(device)
    test_spatial = torch.from_numpy(spatial[test_indices]).to(device)
    train_motion = torch.from_numpy(motion[train_indices]).to(device)
    test_motion = torch.from_numpy(motion[test_indices]).to(device)
    train_y = torch.from_numpy(
        labels[train_indices].astype(np.float32)
    ).to(device)
    train_mask = torch.from_numpy(
        eligibility[train_indices].astype(np.float32)
    ).to(device)
    train_weight = torch.from_numpy(
        masked_source_weights(
            sources[train_indices],
            eligibility[train_indices],
        )
    ).to(device)
    positive = torch.sum(train_y * train_mask, dim=0)
    negative = torch.sum((1.0 - train_y) * train_mask, dim=0)
    if torch.any(positive == 0) or torch.any(negative == 0):
        raise ValueError("D14 train target is single-class")
    positive_weight = negative / positive

    model = CurrentMotionOnsetHead().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    losses = []
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_spatial, train_motion, arm=arm)
        per_target = nnf.binary_cross_entropy_with_logits(
            logits,
            train_y,
            pos_weight=positive_weight,
            reduction="none",
        )
        weighted = per_target * train_mask * train_weight
        loss = torch.mean(
            torch.sum(weighted, dim=0)
            / torch.sum(train_mask * train_weight, dim=0)
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(
            model(test_spatial, test_motion, arm=arm)
        ).cpu().numpy()
    return probability, {
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def evaluate(
    labels: np.ndarray,
    eligibility: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for index, target in enumerate(TARGETS):
        mask = eligibility[:, index]
        result[target] = {
            **binary_metrics(
                labels[mask, index],
                probability[mask, index],
            ),
            "eligible_count": int(np.sum(mask)),
            "positive_count": int(np.sum(labels[mask, index])),
        }
    return result


def metric_value(result: dict[str, Any], path: str) -> float:
    target, metric = path.split(".")
    value = result[target][metric]
    if value is None:
        raise ValueError(f"D14 metric is not evaluable: {path}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--spatial-features",
        type=Path,
        default=DEFAULT_SPATIAL_FEATURES,
    )
    parser.add_argument(
        "--motion-features",
        type=Path,
        default=DEFAULT_MOTION_FEATURES,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = Path(str(args.output) + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite D14 canary")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    expected_ids = np.asarray(
        [record["sample_id"] for record in records],
        dtype=str,
    )
    if len(records) != 1078:
        raise ValueError("Expected 1,078 D12 samples")
    spatial_history = load_features(
        args.spatial_features,
        expected_ids,
        (1078, 5, 576, 4, 7),
    )
    spatial = spatial_history[:, -1]
    del spatial_history
    motion = load_features(
        args.motion_features,
        expected_ids,
        (1078, 4, 8, 3, 6),
    )
    folds = np.asarray([int(record["fold"]) for record in records])
    sources = np.asarray(
        [str(record["source_session_id"]) for record in records]
    )
    labels = np.asarray(
        [
            (
                int(
                    record["future_onset_target"][
                        "proximity_onset"
                    ]
                ),
                int(
                    record["future_onset_target"][
                        "corridor_onset"
                    ]
                ),
            )
            for record in records
        ],
        dtype=np.int64,
    )
    eligibility = np.asarray(
        [
            (
                bool(
                    record["future_onset_target"][
                        "proximity_eligible"
                    ]
                ),
                bool(
                    record["future_onset_target"][
                        "corridor_eligible"
                    ]
                ),
            )
            for record in records
        ],
        dtype=bool,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    metric_paths = (
        "proximity.auroc",
        "proximity.average_precision",
        "corridor.auroc",
        "corridor.average_precision",
    )
    by_fold_delta = {
        fold: {path: [] for path in metric_paths}
        for fold in range(5)
    }
    units = []
    for fold in range(5):
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        if set(sources[test_indices]) & set(sources[train_indices]):
            raise ValueError("Source-session leakage across D14 folds")
        normalized_spatial, normalized_motion = normalize_fold(
            spatial,
            motion,
            train_indices,
        )
        for seed in SEEDS:
            arm_results = {}
            diagnostics = {}
            for arm in ("current", "motion"):
                probability, diagnostic = train_arm(
                    normalized_spatial,
                    normalized_motion,
                    labels,
                    eligibility,
                    sources,
                    train_indices,
                    test_indices,
                    arm,
                    seed,
                    device,
                )
                arm_results[arm] = evaluate(
                    labels[test_indices],
                    eligibility[test_indices],
                    probability,
                )
                diagnostics[arm] = diagnostic
            delta = {}
            for path in metric_paths:
                value = (
                    metric_value(arm_results["motion"], path)
                    - metric_value(arm_results["current"], path)
                )
                delta[path] = value
                by_fold_delta[fold][path].append(value)
            units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "current": arm_results["current"],
                    "motion": arm_results["motion"],
                    "motion_minus_current": delta,
                    "training": diagnostics,
                }
            )

    fold_rows = []
    aggregate_values = {path: [] for path in metric_paths}
    for fold in range(5):
        mean_delta = {}
        for path in metric_paths:
            value = float(np.mean(by_fold_delta[fold][path]))
            mean_delta[path] = value
            aggregate_values[path].append(value)
        fold_rows.append(
            {
                "fold": fold,
                "seed_count": len(SEEDS),
                "mean_motion_minus_current": mean_delta,
            }
        )
    aggregate = {
        path: summarize(values)
        for path, values in aggregate_values.items()
    }
    floors = {
        "proximity.auroc": AUROC_MEAN_FLOOR,
        "proximity.average_precision": AP_MEAN_FLOOR,
        "corridor.auroc": AUROC_MEAN_FLOOR,
        "corridor.average_precision": AP_MEAN_FLOOR,
    }
    supported = all(
        aggregate[path]["mean"] is not None
        and float(aggregate[path]["mean"]) >= floors[path]
        and aggregate[path]["median"] is not None
        and float(aggregate[path]["median"]) > 0.0
        and int(aggregate[path]["positive_count"]) >= 3
        for path in metric_paths
    )
    status = (
        "D14_EXPLICIT_MOTION_FUTURE_ONSET_INCREMENT_SUPPORTED"
        if supported
        else "D14_EXPLICIT_MOTION_FUTURE_ONSET_INCREMENT_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development explicit-motion onset canary",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "spatial_features_path": str(
                args.spatial_features.resolve()
            ),
            "spatial_features_sha256": sha256(args.spatial_features),
            "motion_features_path": str(
                args.motion_features.resolve()
            ),
            "motion_features_sha256": sha256(args.motion_features),
        },
        "design": {
            "split": "fixed SHA-256(source_session_id) modulo 5",
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "current_input": "frozen MobileNet current 576x4x7 map",
            "candidate_input": (
                "four direction-preserving raw+residual RAFT 8x3x6 maps"
            ),
            "capacity_control": (
                "identical current/motion 49,490-parameter model; current "
                "arm receives an all-zero motion tensor"
            ),
            "loss": (
                "target-masked source-balanced BCEWithLogits with train-fold "
                "positive weights"
            ),
            "selection": "fixed final epoch; no heldout model selection",
            "success_gate": {
                "auroc_mean_floor": AUROC_MEAN_FLOOR,
                "average_precision_mean_floor": AP_MEAN_FLOOR,
                "all_metric_medians_positive": True,
                "minimum_positive_folds_per_metric": 3,
            },
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(set(sources)),
            "folds": 5,
            "seeds": len(SEEDS),
            "training_units": len(units) * 2,
        },
        "device": str(device),
        "units": units,
        "folds": fold_rows,
        "aggregate_seed_mean_motion_minus_current": aggregate,
        "next_action": (
            "advance explicit motion to a lightweight temporal student"
            if supported
            else (
                "retain the true-onset estimand and D13 weak signal, but "
                "stop this fixed RAFT grid recipe without tuning"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    sidecar.write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "aggregate_seed_mean_motion_minus_current": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
