#!/usr/bin/env python3
"""Test whether train-only synthetic pairs fix a frozen profile direction.

Every real static pair is held out by parent source. Synthetic descendants of
that source are excluded from the fold. Unit pair deltas form a deterministic
prototype direction; no endpoint classifier or threshold is fitted. Rice is an
external clear-risk-clear pressure test after the fold audit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_synthetic_pair_delta_probe_v1"


def prototype_direction(deltas: Sequence[np.ndarray]) -> np.ndarray:
    unit: list[np.ndarray] = []
    for delta in deltas:
        value = np.asarray(delta, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if norm > 1e-12:
            unit.append(value / norm)
    if not unit:
        raise ValueError("prototype direction needs a non-degenerate delta")
    direction = np.mean(unit, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("prototype deltas cancel to zero")
    return direction / norm


def fold_rows(
    real_rows: Sequence[dict[str, Any]],
    synthetic_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for held in real_rows:
        held_source = held["parent_source_id"]
        real_train = [row for row in real_rows if row["parent_source_id"] != held_source]
        synthetic_train = [row for row in synthetic_rows if row["parent_source_id"] != held_source]
        baseline_direction = prototype_direction([row["delta"] for row in real_train])
        augmented_direction = prototype_direction([row["delta"] for row in real_train + synthetic_train])
        results.append({
            "held_out_pair_id": held["pair_id"],
            "held_out_parent_source_id": held_source,
            "training_real_pair_ids": [row["pair_id"] for row in real_train],
            "training_synthetic_pair_ids": [row["pair_id"] for row in synthetic_train],
            "held_out_source_descendants_excluded": all(row["parent_source_id"] != held_source for row in real_train + synthetic_train),
            "baseline_projection": float(held["delta"] @ baseline_direction),
            "augmented_projection": float(held["delta"] @ augmented_direction),
            "baseline_ordered": bool(held["delta"] @ baseline_direction > 0.0),
            "augmented_ordered": bool(held["delta"] @ augmented_direction > 0.0),
        })
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.real_report, args.synthetic_report, args.dataset_root, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    real = lifecycle.verify_json_sidecar(args.real_report.resolve())
    synthetic = lifecycle.verify_json_sidecar(args.synthetic_report.resolve())
    fields = tuple(synthetic["frozen_response_contract"]["response_fields"])
    if synthetic.get("parent_source_isolated_representation_short_run_authorized") is not True:
        raise ValueError("synthetic response audit did not authorize a source-isolated short run")
    manifest = [
        json.loads(line)
        for line in (args.dataset_root.resolve() / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    synthetic_source = {
        row["attributes"]["counterfactual_pair_id"]: row["source"]["parent_source_id"]
        for row in manifest if row["attributes"].get("synthetic") is True
    }
    parent_episode_source = {
        row["source"]["parent_episode_id"]: row["source"]["parent_source_id"]
        for row in manifest
    }
    real_rows = [{
        "pair_id": row["counterfactual_pair_id"],
        "parent_source_id": parent_episode_source[row["no_alert_episode_id"]],
        "delta": np.asarray([float(row["alert_scores"][field]) - float(row["no_alert_scores"][field]) for field in fields]),
    } for row in real["legacy_static_pairs"]]
    synthetic_rows = [{
        "pair_id": row["counterfactual_pair_id"],
        "parent_source_id": synthetic_source[row["counterfactual_pair_id"]],
        "delta": np.asarray([float(row["risk_minus_clear"][field]) for field in fields]),
    } for row in synthetic["counterfactual_responses"]]
    folds = fold_rows(real_rows, synthetic_rows)
    baseline_rate = sum(row["baseline_ordered"] for row in folds) / len(folds)
    augmented_rate = sum(row["augmented_ordered"] for row in folds) / len(folds)

    final_direction = prototype_direction([row["delta"] for row in real_rows + synthetic_rows])
    rice = real["rice_street_windows"]
    pre = np.asarray([float(rice["pre_clear"][field]) for field in fields])
    risk = np.asarray([float(rice["risk"][field]) for field in fields])
    post = np.asarray([float(rice["post_clear"][field]) for field in fields])
    rice_open = float((risk - pre) @ final_direction)
    rice_close = float((risk - post) @ final_direction)
    gate = bool(
        augmented_rate == 1.0
        and augmented_rate > baseline_rate
        and rice_open > 0.0
        and rice_close > 0.0
        and all(row["held_out_source_descendants_excluded"] for row in folds)
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "real_report_sha256": common.sha256_file(args.real_report),
            "synthetic_report_sha256": common.sha256_file(args.synthetic_report),
            "manifest_sha256": common.sha256_file(args.dataset_root / "manifest.jsonl"),
        },
        "feature_contract": {
            "fields": list(fields),
            "pair_delta": "risk_minus_clear",
            "prototype": "unit-normalize each training pair delta, mean, unit-normalize",
            "threshold_fitted": False,
            "trainable_parameters": 0,
        },
        "source_isolated_folds": folds,
        "real_pair_metrics": {
            "pair_count": len(folds),
            "real_only_ordering_rate": baseline_rate,
            "real_plus_synthetic_ordering_rate": augmented_rate,
        },
        "rice_external_pressure": {
            "open_projection": rice_open,
            "close_projection": rice_close,
            "open_ordered": rice_open > 0.0,
            "close_ordered": rice_close > 0.0,
            "used_for_training": False,
        },
        "frozen_head_augmentation_gate": {
            "passed": gate,
            "requirements": {
                "all_real_pairs_ordered": True,
                "strictly_improves_real_only": True,
                "rice_open_and_close_ordered": True,
                "held_out_parent_descendants_excluded": True,
            },
        },
        "interpretation_if_failed": "Clean synthetic counterfactuals excite the frozen channels, but adding them to a frozen pair prototype does not transfer. Increase scene diversity and train the representation; do not tune the endpoint head.",
        "representation_finetune_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--synthetic-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({
        "ok": True,
        "gate_passed": payload["frozen_head_augmentation_gate"]["passed"],
        "real_plus_synthetic_ordering_rate": payload["real_pair_metrics"]["real_plus_synthetic_ordering_rate"],
        "rice_open_ordered": payload["rice_external_pressure"]["open_ordered"],
        "rice_close_ordered": payload["rice_external_pressure"]["close_ordered"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
