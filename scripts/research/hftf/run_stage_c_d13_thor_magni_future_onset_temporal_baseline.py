#!/usr/bin/env python3
"""Compare current/history frozen spatial features on true future onset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
from run_stage_c_d8_thor_magni_equal_capacity_temporal_head import (
    EPOCHS,
    LEARNING_RATE,
    SEEDS,
    WEIGHT_DECAY,
    TemporalSpatialActionabilityHead,
    load_aligned_features,
    normalize_fold,
    seed_everything,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d13_thor_magni_"
    "future_onset_equal_capacity_temporal_spatial_baseline_v0"
)
DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d12-thor-magni-future-onset-v0/samples.jsonl"
)
DEFAULT_FEATURES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d8-thor-magni-spatial-features-v0/features.npz"
)
TARGETS = ("proximity", "corridor")


def masked_source_weights(
    sources: np.ndarray,
    eligibility: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(eligibility.shape, dtype=np.float32)
    for target_index in range(eligibility.shape[1]):
        eligible_sources = sources[eligibility[:, target_index]]
        counts = Counter(str(value) for value in eligible_sources)
        for row_index, source in enumerate(sources):
            if eligibility[row_index, target_index]:
                weights[row_index, target_index] = (
                    1.0 / counts[str(source)]
                )
        positive = weights[:, target_index] > 0
        weights[positive, target_index] /= np.mean(
            weights[positive, target_index]
        )
    return weights


def train_arm(
    normalized_features: np.ndarray,
    labels: np.ndarray,
    eligibility: np.ndarray,
    sources: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if arm not in {"current", "history"}:
        raise ValueError(f"Unknown arm: {arm}")
    seed_everything(seed)
    train_values = normalized_features[train_indices]
    test_values = normalized_features[test_indices]
    if arm == "current":
        train_values = np.repeat(train_values[:, -1:], 5, axis=1)
        test_values = np.repeat(test_values[:, -1:], 5, axis=1)

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
            sources[train_indices],
            eligibility[train_indices],
        )
    ).to(device)
    positive = torch.sum(train_y * train_mask, dim=0)
    negative = torch.sum((1.0 - train_y) * train_mask, dim=0)
    if torch.any(positive == 0) or torch.any(negative == 0):
        raise ValueError("Training onset target is single-class")
    positive_weight = negative / positive

    model = TemporalSpatialActionabilityHead(
        channels=normalized_features.shape[2],
        height=normalized_features.shape[3],
        width=normalized_features.shape[4],
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
    for target_index, target in enumerate(TARGETS):
        mask = eligibility[:, target_index]
        result[target] = {
            **binary_metrics(
                labels[mask, target_index],
                probability[mask, target_index],
            ),
            "eligible_count": int(np.sum(mask)),
            "positive_count": int(
                np.sum(labels[mask, target_index])
            ),
        }
    return result


def metric_value(result: dict[str, Any], path: str) -> float:
    target, metric = path.split(".")
    value = result[target][metric]
    if value is None:
        raise ValueError(f"Onset metric is not evaluable: {path}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = Path(str(args.output) + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite D13 onset baseline")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 1078:
        raise ValueError("Expected 1,078 D12 samples")
    features = load_aligned_features(args.features, records)
    if features.shape != (1078, 5, 576, 4, 7):
        raise ValueError(f"Expected spatial features, got {features.shape}")
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
            raise ValueError("Source-session leakage across D13 folds")
        normalized = normalize_fold(features, train_indices)
        for seed in SEEDS:
            arm_results = {}
            diagnostics = {}
            for arm in ("current", "history"):
                probability, diagnostic = train_arm(
                    normalized,
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
                    metric_value(arm_results["history"], path)
                    - metric_value(arm_results["current"], path)
                )
                delta[path] = value
                by_fold_delta[fold][path].append(value)
            units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "current": arm_results["current"],
                    "history": arm_results["history"],
                    "history_minus_current": delta,
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
                "mean_history_minus_current": mean_delta,
            }
        )
    aggregate = {
        path: summarize(values)
        for path, values in aggregate_values.items()
    }
    supported = all(
        aggregate[path]["median"] is not None
        and float(aggregate[path]["median"]) > 0.0
        and int(aggregate[path]["positive_count"]) >= 3
        for path in metric_paths
    )
    status = (
        "D13_FUTURE_ONSET_TEMPORAL_SPATIAL_INCREMENT_SUPPORTED"
        if supported
        else "D13_FUTURE_ONSET_TEMPORAL_SPATIAL_INCREMENT_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development true-future onset baseline",
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
            "split": "fixed SHA-256(source_session_id) modulo 5",
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "target": (
                "future onset among samples whose exact current geometric "
                "state is safe; target-specific masks"
            ),
            "architecture": (
                "frozen MobileNet 5x576x4x7 features; identical current/"
                "history 13,586-parameter temporal-spatial heads"
            ),
            "loss": (
                "target-masked source-balanced BCEWithLogits with train-fold "
                "positive weights"
            ),
            "selection": "fixed final epoch; no heldout model selection",
            "success_gate": (
                "seed-mean fold delta median > 0 and at least 3/5 positive "
                "folds for onset proximity/corridor AUROC and AP"
            ),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(set(sources)),
            "folds": 5,
            "seeds": len(SEEDS),
            "training_units": len(units) * 2,
            "eligible": {
                target: int(np.sum(eligibility[:, index]))
                for index, target in enumerate(TARGETS)
            },
            "positive": {
                target: int(np.sum(labels[:, index]))
                for index, target in enumerate(TARGETS)
            },
        },
        "device": str(device),
        "units": units,
        "folds": fold_rows,
        "aggregate_seed_mean_history_minus_current": aggregate,
        "next_action": (
            "advance onset target to an explicit motion-aware representation"
            if supported
            else (
                "retain the corrected onset estimand but do not claim frozen "
                "MobileNet temporal benefit; diagnose motion observability"
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
