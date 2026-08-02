#!/usr/bin/env python3
"""Evaluate equal-capacity RGB history on TartanGround true future onset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
    summarize,
)
from materialize_stage_c_d16_tartanground_future_onset import (
    HEIGHT_INDICES,
    HORIZONS,
)
from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    masked_source_weights,
)
from run_stage_c_d8_thor_magni_equal_capacity_temporal_head import (
    EPOCHS,
    LEARNING_RATE,
    SEEDS,
    WEIGHT_DECAY,
    TemporalSpatialActionabilityHead,
    normalize_fold,
    seed_everything,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d16_tartanground_"
    "future_onset_equal_capacity_temporal_spatial_baseline_v0"
)
DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d16-tartanground-future-onset-v0/samples.jsonl"
)
DEFAULT_FEATURES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d16-tartanground-spatial-features-v0/features.npz"
)
TARGETS = tuple(
    f"{horizon}_{height}"
    for horizon in HORIZONS
    for height in HEIGHT_INDICES
)
AUROC_MEAN_FLOOR = 0.01
AP_MEAN_FLOOR = 0.005


def load_features(path: Path, records: list[dict[str, Any]]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str)
        features = payload["features"].astype(np.float32)
    expected = np.asarray(
        [record["sample_id"] for record in records],
        dtype=str,
    )
    if not np.array_equal(sample_ids, expected):
        raise ValueError("D16 feature sample ordering mismatch")
    if features.shape != (495, 5, 576, 4, 7):
        raise ValueError(f"Unexpected D16 features: {features.shape}")
    return features


def train_arm(
    features: np.ndarray,
    labels: np.ndarray,
    eligibility: np.ndarray,
    environments: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_everything(seed)
    train_values = features[train_indices]
    test_values = features[test_indices]
    if arm == "current":
        train_values = np.repeat(train_values[:, -1:], 5, axis=1)
        test_values = np.repeat(test_values[:, -1:], 5, axis=1)
    elif arm != "history":
        raise ValueError(f"Unknown D16 arm: {arm}")
    train_x = torch.from_numpy(train_values).to(device)
    test_x = torch.from_numpy(test_values).to(device)
    train_y = torch.from_numpy(
        labels[train_indices].astype(np.float32)
    ).to(device)
    train_mask = torch.from_numpy(
        eligibility[train_indices].astype(np.float32)
    ).to(device)
    train_weight = torch.from_numpy(
        masked_source_weights(
            environments[train_indices],
            eligibility[train_indices],
        )
    ).to(device)
    positive = torch.sum(train_y * train_mask, dim=0)
    negative = torch.sum((1.0 - train_y) * train_mask, dim=0)
    if torch.any(positive == 0) or torch.any(negative == 0):
        raise ValueError("D16 train target is single-class")
    positive_weight = negative / positive
    model = TemporalSpatialActionabilityHead(
        channels=576,
        height=4,
        width=7,
        output_count=len(TARGETS),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    losses = []
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
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
        probability = torch.sigmoid(model(test_x)).cpu().numpy()
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
    target, metric = path.rsplit(".", 1)
    value = result[target][metric]
    if value is None:
        raise ValueError(f"D16 metric is not evaluable: {path}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = Path(str(args.output) + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite D16 baseline")
    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 495:
        raise ValueError("Expected 495 D16 samples")
    features = load_features(args.features, records)
    folds = np.asarray(
        [int(record["environment_fold"]) for record in records]
    )
    environments = np.asarray(
        [str(record["environment"]) for record in records]
    )
    labels = np.asarray(
        [
            [
                int(
                    record["future_onset_target"]["sample_onset"][
                        target
                    ]
                )
                for target in TARGETS
            ]
            for record in records
        ],
        dtype=np.int64,
    )
    eligibility = np.asarray(
        [
            [
                bool(
                    record["future_onset_target"]["sample_eligible"][
                        target
                    ]
                )
                for target in TARGETS
            ]
            for record in records
        ],
        dtype=bool,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    metric_paths = tuple(
        f"{target}.{metric}"
        for target in TARGETS
        for metric in ("auroc", "average_precision")
    )
    by_fold_delta = {
        fold: {path: [] for path in metric_paths}
        for fold in range(3)
    }
    units = []
    for fold in range(3):
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        if set(environments[test_indices]) & set(
            environments[train_indices]
        ):
            raise ValueError("Environment leakage across D16 folds")
        normalized = normalize_fold(features, train_indices)
        for seed in SEEDS:
            arm_results = {}
            diagnostics = {}
            for arm in ("current", "history"):
                probability, diagnostic = train_arm(
                    normalized,
                    labels,
                    eligibility,
                    environments,
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
                    metric_value(arm_results["history"], path)
                    - metric_value(arm_results["current"], path)
                )
                delta[path] = value
                by_fold_delta[fold][path].append(value)
            units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "heldout_environments": sorted(
                        set(environments[test_indices])
                    ),
                    "current": arm_results["current"],
                    "history": arm_results["history"],
                    "history_minus_current": delta,
                    "training": diagnostics,
                }
            )
    fold_rows = []
    aggregate_values = {path: [] for path in metric_paths}
    for fold in range(3):
        mean_delta = {}
        for path in metric_paths:
            value = float(np.mean(by_fold_delta[fold][path]))
            mean_delta[path] = value
            aggregate_values[path].append(value)
        fold_rows.append(
            {
                "fold": fold,
                "seed_count": len(SEEDS),
                "mean_history_minus_current": mean_delta,
            }
        )
    aggregate = {
        path: summarize(values)
        for path, values in aggregate_values.items()
    }
    floors = {
        path: (
            AUROC_MEAN_FLOOR
            if path.endswith(".auroc")
            else AP_MEAN_FLOOR
        )
        for path in metric_paths
    }
    supported = all(
        aggregate[path]["mean"] is not None
        and float(aggregate[path]["mean"]) >= floors[path]
        and int(aggregate[path]["positive_count"]) == 3
        for path in metric_paths
    )
    status = (
        "D16_TARTANGROUND_FUTURE_ONSET_HISTORY_INCREMENT_SUPPORTED"
        if supported
        else "D16_TARTANGROUND_FUTURE_ONSET_HISTORY_INCREMENT_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development synthetic onset representation baseline",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "features_path": str(args.features.resolve()),
            "features_sha256": sha256(args.features),
        },
        "design": {
            "folds": "inherited D5 3-fold environment assignments",
            "targets": list(TARGETS),
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": (
                "frozen MobileNet 5x576x4x7 features; identical current/"
                "history 14,484-parameter temporal-spatial heads"
            ),
            "loss": (
                "target-masked environment-balanced BCEWithLogits with "
                "train-fold positive weights"
            ),
            "selection": "fixed final epoch; no heldout model selection",
            "success_gate": {
                "auroc_mean_floor": AUROC_MEAN_FLOOR,
                "average_precision_mean_floor": AP_MEAN_FLOOR,
                "positive_folds_per_metric": 3,
                "metric_count": len(metric_paths),
            },
        },
        "counts": {
            "samples": len(records),
            "environments": len(set(environments)),
            "folds": 3,
            "seeds": len(SEEDS),
            "training_units": len(units) * 2,
        },
        "device": str(device),
        "units": units,
        "folds": fold_rows,
        "aggregate_seed_mean_history_minus_current": aggregate,
        "next_action": (
            "use the synthetic onset task for pretraining before real-source "
            "transfer"
            if supported
            else (
                "retain the onset corpus but do not advance this frozen "
                "history representation"
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
                "aggregate_seed_mean_history_minus_current": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
