#!/usr/bin/env python3
"""Run an equal-capacity current/history THOR actionability head."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_SAMPLES,
    binary_metrics,
    load_jsonl,
    sha256,
    summarize,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d8_thor_magni_"
    "equal_capacity_temporal_actionability_head_v0"
)
DEFAULT_FEATURES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d8-thor-magni-rgb-history-screen-v0/features.npz"
)
SEEDS = (17, 23, 41)
EPOCHS = 120
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3


class TemporalActionabilityHead(nn.Module):
    """Fuse temporal residuals while preserving a current-frame identity path."""

    def __init__(self, channels: int, output_count: int = 2) -> None:
        super().__init__()
        self.temporal_residual_weight = nn.Parameter(
            torch.zeros(4, channels)
        )
        self.normalization = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, output_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 5:
            raise ValueError("Expected [batch,5,channel] features")
        current = features[:, -1]
        residual = features[:, :4] - current[:, None]
        weighted_residual = (
            residual
            * torch.tanh(self.temporal_residual_weight)[None]
        ).mean(dim=1)
        return self.head(
            self.normalization(current + weighted_residual)
        )


class TemporalSpatialActionabilityHead(nn.Module):
    """Preserve the frozen spatial grid under the same arm capacity."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        hidden_channels: int = 16,
        output_count: int = 2,
    ) -> None:
        super().__init__()
        self.temporal_residual_weight = nn.Parameter(
            torch.zeros(4, channels, 1, 1)
        )
        self.normalization = nn.GroupNorm(32, channels)
        self.projection = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
        )
        self.head = nn.Linear(
            hidden_channels * height * width,
            output_count,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or features.shape[1] != 5:
            raise ValueError("Expected [batch,5,channel,height,width] features")
        current = features[:, -1]
        residual = features[:, :4] - current[:, None]
        weighted_residual = (
            residual
            * torch.tanh(self.temporal_residual_weight)[None]
        ).mean(dim=1)
        fused = self.normalization(current + weighted_residual)
        hidden = nnf.hardswish(self.projection(fused))
        return self.head(hidden.flatten(1))


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


def source_balanced_weights(sources: np.ndarray) -> np.ndarray:
    counts = Counter(str(value) for value in sources)
    weights = np.asarray(
        [1.0 / counts[str(value)] for value in sources],
        dtype=np.float32,
    )
    return weights / np.mean(weights)


def normalize_fold(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    if features.ndim == 3:
        train = features[train_indices].reshape(-1, features.shape[-1])
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0)
        scale[scale < 1e-6] = 1.0
        return ((features - mean) / scale).astype(np.float32)
    if features.ndim == 5:
        train = features[train_indices]
        mean = np.mean(train, axis=(0, 1, 3, 4), keepdims=True)
        scale = np.std(train, axis=(0, 1, 3, 4), keepdims=True)
        scale[scale < 1e-6] = 1.0
        return ((features - mean) / scale).astype(np.float32)
    raise ValueError(f"Unsupported feature tensor rank: {features.ndim}")


def train_arm(
    normalized_features: np.ndarray,
    labels: np.ndarray,
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
        train_values = np.repeat(
            train_values[:, -1:],
            5,
            axis=1,
        )
        test_values = np.repeat(
            test_values[:, -1:],
            5,
            axis=1,
        )

    train_x = torch.from_numpy(train_values).to(device)
    test_x = torch.from_numpy(test_values).to(device)
    train_y = torch.from_numpy(
        labels[train_indices].astype(np.float32)
    ).to(device)
    sample_weight = torch.from_numpy(
        source_balanced_weights(sources[train_indices])
    ).to(device)
    positive = torch.sum(train_y, dim=0)
    negative = train_y.shape[0] - positive
    positive_weight = negative / torch.clamp(positive, min=1.0)

    if normalized_features.ndim == 3:
        model: nn.Module = TemporalActionabilityHead(
            channels=normalized_features.shape[-1]
        ).to(device)
    elif normalized_features.ndim == 5:
        model = TemporalSpatialActionabilityHead(
            channels=normalized_features.shape[2],
            height=normalized_features.shape[3],
            width=normalized_features.shape[4],
        ).to(device)
    else:
        raise ValueError("Unsupported temporal-head feature rank")
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
        loss = torch.mean(per_target * sample_weight[:, None])
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
        "train_source_weight_sum_range": [
            float(
                np.min(
                    [
                        np.sum(
                            source_balanced_weights(
                                sources[train_indices]
                            )[sources[train_indices] == source]
                        )
                        for source in sorted(
                            set(sources[train_indices].tolist())
                        )
                    ]
                )
            ),
            float(
                np.max(
                    [
                        np.sum(
                            source_balanced_weights(
                                sources[train_indices]
                            )[sources[train_indices] == source]
                        )
                        for source in sorted(
                            set(sources[train_indices].tolist())
                        )
                    ]
                )
            ),
        ],
    }


def evaluate(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    return {
        "proximity": binary_metrics(target[:, 0], probability[:, 0]),
        "corridor": binary_metrics(target[:, 1], probability[:, 1]),
    }


def metric_value(result: dict[str, Any], path: str) -> float | None:
    current: Any = result
    for key in path.split("."):
        current = current[key]
    return None if current is None else float(current)


def load_aligned_features(
    path: Path,
    records: list[dict[str, Any]],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str)
        features = payload["features"].astype(np.float32)
    expected = np.asarray(
        [record["sample_id"] for record in records],
        dtype=str,
    )
    if not np.array_equal(sample_ids, expected):
        raise ValueError("Feature cache sample ordering mismatch")
    allowed_shapes = {
        (len(records), 5, 576),
        (len(records), 5, 576, 4, 7),
    }
    if features.shape not in allowed_shapes:
        raise ValueError(
            f"Unexpected frozen feature shape: {features.shape}"
        )
    return features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--experiment",
        choices=("d8-thor", "d9-jrdb"),
        default="d8-thor",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("Refusing to overwrite temporal-head result")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    expected_samples = 1078 if args.experiment == "d8-thor" else 104
    if len(records) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples for {args.experiment}"
        )
    features = load_aligned_features(args.features, records)
    feature_kind = "pooled" if features.ndim == 3 else "spatial"
    folds = np.asarray([int(record["fold"]) for record in records])
    sources = np.asarray(
        [str(record["source_session_id"]) for record in records]
    )
    labels = np.asarray(
        [
            (
                int(record["target"]["future_proximity_le_1_25m"]),
                int(record["target"]["future_corridor_intrusion"]),
            )
            for record in records
        ],
        dtype=np.int64,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    fold_ids = sorted(set(folds.tolist()))
    expected_folds = (
        list(range(5))
        if args.experiment == "d8-thor"
        else [0, 1]
    )
    if fold_ids != expected_folds:
        raise ValueError(
            f"Unexpected folds for {args.experiment}: {fold_ids}"
        )

    metric_paths = (
        "proximity.auroc",
        "proximity.average_precision",
        "corridor.auroc",
        "corridor.average_precision",
    )
    units = []
    by_fold_delta: dict[int, dict[str, list[float]]] = {
        fold: {path: [] for path in metric_paths}
        for fold in fold_ids
    }
    for fold in fold_ids:
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        train_sources = set(sources[train_indices].tolist())
        test_sources = set(sources[test_indices].tolist())
        if train_sources & test_sources:
            raise ValueError("Source-session leakage across folds")
        normalized = normalize_fold(features, train_indices)
        for seed in SEEDS:
            arm_results = {}
            diagnostics = {}
            for arm in ("current", "history"):
                probability, diagnostic = train_arm(
                    normalized,
                    labels,
                    sources,
                    train_indices,
                    test_indices,
                    arm,
                    seed,
                    device,
                )
                arm_results[arm] = evaluate(
                    labels[test_indices],
                    probability,
                )
                diagnostics[arm] = diagnostic
            delta = {}
            for path in metric_paths:
                history_value = metric_value(
                    arm_results["history"],
                    path,
                )
                current_value = metric_value(
                    arm_results["current"],
                    path,
                )
                value = (
                    history_value - current_value
                    if history_value is not None
                    and current_value is not None
                    else None
                )
                delta[path] = value
                if value is not None:
                    by_fold_delta[fold][path].append(value)
            units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "train_sample_count": len(train_indices),
                    "heldout_sample_count": len(test_indices),
                    "train_source_count": len(train_sources),
                    "heldout_source_count": len(test_sources),
                    "heldout_sources": sorted(test_sources),
                    "current": arm_results["current"],
                    "history": arm_results["history"],
                    "history_minus_current": delta,
                    "training": diagnostics,
                }
            )

    fold_rows = []
    aggregate_values: dict[str, list[float]] = {
        path: [] for path in metric_paths
    }
    for fold in fold_ids:
        mean_delta = {}
        for path in metric_paths:
            values = by_fold_delta[fold][path]
            value = float(np.mean(values)) if values else None
            mean_delta[path] = value
            if value is not None:
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
    if args.experiment == "d8-thor":
        supported = all(
            aggregate[path]["median"] is not None
            and float(aggregate[path]["median"]) > 0.0
            and int(aggregate[path]["positive_count"]) >= 3
            for path in metric_paths
        )
        status_prefix = (
            "D8_EQUAL_CAPACITY_TEMPORAL_ACTIONABILITY"
            if feature_kind == "pooled"
            else "D8_EQUAL_CAPACITY_TEMPORAL_SPATIAL_ACTIONABILITY"
        )
        status = (
            f"{status_prefix}_INCREMENT_SUPPORTED"
            if supported
            else f"{status_prefix}_INCREMENT_NOT_STABLE"
        )
        success_gate = (
            "seed-mean fold delta median > 0 and at least 3/5 "
            "positive folds for proximity/corridor AUROC and AP"
        )
    else:
        primary_paths = (
            "corridor.auroc",
            "corridor.average_precision",
        )
        supported = all(
            aggregate[path]["mean"] is not None
            and float(aggregate[path]["mean"]) > 0.0
            and int(aggregate[path]["positive_count"]) == len(fold_ids)
            and sum(
                1
                for unit in units
                if unit["history_minus_current"][path] > 0.0
            )
            >= 4
            for path in primary_paths
        )
        status = (
            "D9_JRDB_TEMPORAL_SPATIAL_CORRIDOR_REPLICATION_SUPPORTED"
            if supported
            else "D9_JRDB_TEMPORAL_SPATIAL_CORRIDOR_REPLICATION_NOT_SUPPORTED"
        )
        success_gate = (
            "corridor AUROC and AP seed-mean delta > 0 in both fixed "
            "source-pair folds and at least 4/6 fold-seed units positive; "
            "proximity is a negative control"
        )

    report = {
        "schema": (
            SCHEMA
            if args.experiment == "d8-thor"
            else (
                "blindassist_hftf_stage_c_d9_jrdb_equal_capacity_"
                "temporal_spatial_corridor_replication_v0"
            )
        ),
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development coarse-actionability model canary",
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
            "split": (
                "fixed SHA-256(source_session_id) modulo 5"
                if args.experiment == "d8-thor"
                else (
                    "two fixed source-pair folds: clark+gates and "
                    "meyer+stlc"
                )
            ),
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "backbone": (
                "frozen cached MobileNetV3-small "
                f"{list(features.shape[1:])} {feature_kind} features"
            ),
            "feature_kind": feature_kind,
            "architecture": (
                (
                    "current identity plus learned per-time/per-channel "
                    "bounded residual fusion, LayerNorm, Linear(576,2)"
                )
                if feature_kind == "pooled"
                else (
                    "current identity plus learned per-time/per-channel "
                    "bounded spatial residual fusion, GroupNorm, "
                    "Conv1x1(576,16), Linear(16x4x7,2)"
                )
            ),
            "capacity_control": (
                "current repeats its current feature five times; both arms "
                "instantiate and optimize the identical architecture"
            ),
            "loss": (
                "source-balanced full-batch BCEWithLogits with train-fold "
                "positive weights"
            ),
            "selection": "fixed final epoch; no heldout model selection",
            "success_gate": success_gate,
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(set(sources.tolist())),
            "folds": len(fold_ids),
            "seeds": len(SEEDS),
            "training_units": len(units) * 2,
        },
        "device": str(device),
        "units": units,
        "folds": fold_rows,
        "aggregate_seed_mean_history_minus_current": aggregate,
        "next_action": (
            (
                "seek another independent source before any App or field work"
                if args.experiment == "d9-jrdb"
                else (
                    "seek independent-source replication before any field "
                    "or App work"
                )
            )
            if supported
            else (
                "close this independent-source replication"
                if args.experiment == "d9-jrdb"
                else (
                    "stop the compact temporal head on this representation "
                    "and do not use the earlier screen as proof"
                )
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    Path(str(args.output) + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "device": str(device),
                "aggregate_seed_mean_history_minus_current": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
