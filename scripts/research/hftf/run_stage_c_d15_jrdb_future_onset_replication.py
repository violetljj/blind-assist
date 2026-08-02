#!/usr/bin/env python3
"""Replicate the THOR future-onset history signal on JRDB."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
    summarize,
)
from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    SEEDS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    evaluate,
    metric_value,
    train_arm,
)
from run_stage_c_d8_thor_magni_equal_capacity_temporal_head import (
    load_aligned_features,
    normalize_fold,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d15_jrdb_"
    "future_onset_equal_capacity_history_replication_v0"
)
DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d15-jrdb-future-onset-v0/samples.jsonl"
)
DEFAULT_FEATURES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d9-jrdb-spatial-features-v0/features.npz"
)
PRIMARY_PATHS = (
    "corridor.auroc",
    "corridor.average_precision",
)


def replication_supported(
    aggregate: dict[str, dict[str, Any]],
    units: list[dict[str, Any]],
) -> bool:
    return all(
        aggregate[path]["mean"] is not None
        and float(aggregate[path]["mean"]) > 0.0
        and int(aggregate[path]["positive_count"]) == 2
        and sum(
            1
            for unit in units
            if float(unit["history_minus_current"][path]) > 0.0
        )
        >= 4
        for path in PRIMARY_PATHS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = Path(str(args.output) + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite D15 replication")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 104:
        raise ValueError("Expected 104 JRDB onset samples")
    features = load_aligned_features(args.features, records)
    if features.shape != (104, 5, 576, 4, 7):
        raise ValueError(f"Expected JRDB spatial features: {features.shape}")
    folds = np.asarray([int(record["fold"]) for record in records])
    sources = np.asarray(
        [str(record["source_session_id"]) for record in records]
    )
    labels = np.asarray(
        [
            (
                int(
                    record["transition_target"]["proximity_onset"]
                ),
                int(
                    record["transition_target"]["corridor_onset"]
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
                    record["transition_target"][
                        "proximity_eligible"
                    ]
                ),
                bool(
                    record["transition_target"]["corridor_eligible"]
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
        for fold in (0, 1)
    }
    units = []
    for fold in (0, 1):
        test_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        if set(sources[test_indices]) & set(sources[train_indices]):
            raise ValueError("Source-session leakage across D15 folds")
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
                    "heldout_sources": sorted(
                        set(sources[test_indices])
                    ),
                    "current": arm_results["current"],
                    "history": arm_results["history"],
                    "history_minus_current": delta,
                    "training": diagnostics,
                }
            )

    fold_rows = []
    aggregate_values = {path: [] for path in metric_paths}
    for fold in (0, 1):
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
    supported = replication_supported(aggregate, units)
    status = (
        "D15_JRDB_FUTURE_ONSET_HISTORY_REPLICATION_SUPPORTED"
        if supported
        else "D15_JRDB_FUTURE_ONSET_HISTORY_REPLICATION_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development independent-dataset onset replication",
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
            "folds": (
                "two fixed D9 source-pair folds: clark+gates and meyer+stlc"
            ),
            "primary_target": "current-safe future corridor onset",
            "negative_control": "current-safe future proximity onset",
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
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
                "corridor AUROC and AP seed-mean delta > 0 in both fixed "
                "source-pair folds and at least 4/6 fold-seed units positive; "
                "proximity remains a negative control"
            ),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(set(sources)),
            "folds": 2,
            "seeds": len(SEEDS),
            "training_units": len(units) * 2,
            "eligible": {
                "proximity": int(np.sum(eligibility[:, 0])),
                "corridor": int(np.sum(eligibility[:, 1])),
            },
            "positive": {
                "proximity": int(np.sum(labels[:, 0])),
                "corridor": int(np.sum(labels[:, 1])),
            },
        },
        "device": str(device),
        "units": units,
        "folds": fold_rows,
        "aggregate_seed_mean_history_minus_current": aggregate,
        "next_action": (
            "seek a larger independent onset-rich source before system work"
            if supported
            else (
                "retain the corrected onset estimand but do not advance the "
                "current frozen MobileNet history representation"
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
