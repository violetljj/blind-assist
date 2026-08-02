#!/usr/bin/env python3
"""Aggregate D22 seed17 with D23 seeds23/41 for proximity robustness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import sha256


SCHEMA = "blindassist_hftf_stage_c_d23_thor_magni_proximity_multiseed_v0"
DEFAULT_CANARY = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d22-thor-magni-dense-flow-transfer-v0/report.json"
)
DEFAULT_ADDITIONAL = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d23-thor-magni-proximity-multiseed-v0/"
    "additional_seeds_report.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d23-thor-magni-proximity-multiseed-v0/report.json"
)
METRIC_PATHS = (
    "by_target.proximity.source_macro.auroc",
    "by_target.proximity.source_macro.average_precision",
    "by_target.proximity.pooled.auroc",
    "by_target.proximity.pooled.average_precision",
    "by_target.corridor.source_macro.auroc",
    "by_target.corridor.source_macro.average_precision",
)


def nested(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return float(value)


def summarize(
    units: list[dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    values = [
        nested(unit["history_minus_current"], path) for unit in units
    ]
    seeds = sorted({int(unit["seed"]) for unit in units})
    folds = sorted({int(unit["fold"]) for unit in units})
    by_seed = [
        float(
            np.mean(
                [
                    nested(unit["history_minus_current"], path)
                    for unit in units
                    if int(unit["seed"]) == seed
                ]
            )
        )
        for seed in seeds
    ]
    by_fold = [
        float(
            np.mean(
                [
                    nested(unit["history_minus_current"], path)
                    for unit in units
                    if int(unit["fold"]) == fold
                ]
            )
        )
        for fold in folds
    ]
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_units": int(sum(value > 0 for value in values)),
        "seeds": seeds,
        "by_seed_mean": by_seed,
        "positive_seeds": int(sum(value > 0 for value in by_seed)),
        "folds": folds,
        "by_fold_seed_mean": by_fold,
        "positive_folds": int(sum(value > 0 for value in by_fold)),
    }


def build_gate(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    auroc = aggregate["by_target.proximity.source_macro.auroc"]
    ap = aggregate[
        "by_target.proximity.source_macro.average_precision"
    ]
    pooled_auroc = aggregate["by_target.proximity.pooled.auroc"]
    pooled_ap = aggregate[
        "by_target.proximity.pooled.average_precision"
    ]
    checks = {
        "source_macro_auroc_effect": auroc["mean"] >= 0.01,
        "source_macro_ap_effect": ap["mean"] >= 0.005,
        "source_macro_auroc_all_seeds_positive": (
            auroc["positive_seeds"] == 3
        ),
        "source_macro_ap_all_seeds_positive": (
            ap["positive_seeds"] == 3
        ),
        "source_macro_auroc_positive_folds": (
            auroc["positive_folds"] >= 3
        ),
        "source_macro_ap_positive_folds": ap["positive_folds"] >= 3,
        "source_macro_auroc_positive_units": (
            auroc["positive_units"] >= 10
        ),
        "source_macro_ap_positive_units": ap["positive_units"] >= 10,
        "pooled_auroc_noninferiority": pooled_auroc["mean"] >= -0.005,
        "pooled_ap_noninferiority": pooled_ap["mean"] >= -0.005,
    }
    return {
        "frozen_thresholds": {
            "source_macro_auroc_mean_floor": 0.01,
            "source_macro_ap_mean_floor": 0.005,
            "positive_seeds": 3,
            "positive_folds": 3,
            "positive_units": 10,
            "pooled_metric_noninferiority_floor": -0.005,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def validate_reports(
    canary: dict[str, Any],
    additional: dict[str, Any],
) -> list[dict[str, Any]]:
    if canary["status"] != (
        "D22_THOR_MAGNI_DENSE_FLOW_TRANSFER_CANARY_NOT_SUPPORTED"
    ):
        raise ValueError("Unexpected D23 seed17 source status")
    if canary["design"]["seeds"] != [17]:
        raise ValueError("D23 seed17 report binding mismatch")
    if additional["design"]["seeds"] != [23, 41]:
        raise ValueError("D23 additional seed report binding mismatch")
    for key in (
        "samples_sha256",
        "rgb_cache_sha256",
        "pretrained_sha256",
        "flow_sha256",
    ):
        if canary["inputs"][key] != additional["inputs"][key]:
            raise ValueError(f"D23 input mismatch: {key}")
    units = [*canary["units"], *additional["units"]]
    identities = {
        (int(unit["fold"]), int(unit["seed"])) for unit in units
    }
    if identities != {
        (fold, seed) for fold in range(5) for seed in (17, 23, 41)
    }:
        raise ValueError("D23 requires the exact 5 folds x 3 seeds")
    return sorted(
        units,
        key=lambda unit: (int(unit["fold"]), int(unit["seed"])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", type=Path, default=DEFAULT_CANARY)
    parser.add_argument(
        "--additional",
        type=Path,
        default=DEFAULT_ADDITIONAL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D23 aggregate output is non-overwriting")
    canary = json.loads(args.canary.read_text(encoding="utf-8"))
    additional = json.loads(args.additional.read_text(encoding="utf-8"))
    units = validate_reports(canary, additional)
    aggregate = {
        path: summarize(units, path) for path in METRIC_PATHS
    }
    gate = build_gate(aggregate)
    status = (
        "D23_THOR_MAGNI_PROXIMITY_MULTI_SEED_ROBUSTNESS_SUPPORTED"
        if gate["supported"]
        else (
            "D23_THOR_MAGNI_PROXIMITY_MULTI_SEED_"
            "ROBUSTNESS_NOT_SUPPORTED"
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development post-hypothesis target-specific robustness"
            ),
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "seed17_report_path": str(args.canary.resolve()),
            "seed17_report_sha256": sha256(args.canary),
            "additional_seeds_report_path": str(
                args.additional.resolve()
            ),
            "additional_seeds_report_sha256": sha256(args.additional),
            **{
                key: canary["inputs"][key]
                for key in (
                    "samples_sha256",
                    "rgb_cache_sha256",
                    "pretrained_sha256",
                    "flow_sha256",
                )
            },
        },
        "design": {
            "hypothesis_origin": (
                "proximity source-macro AUROC/AP were both positive in "
                "4/5 D22 seed17 folds; D23 was frozen after observing D22"
            ),
            "training": (
                "same dual-target D22 training; only seeds 23/41 are newly "
                "executed and combined with the already observed seed17"
            ),
            "primary_target": "proximity true-future onset",
            "seeds": [17, 23, 41],
            "folds": 5,
            "selection": "fixed D22 final epoch and frozen D23 gate",
        },
        "counts": {
            "folds": 5,
            "seeds": 3,
            "paired_units": len(units),
            "training_runs": len(units) * 2,
        },
        "gate": gate,
        "aggregate": aggregate,
        "units": units,
        "next_action": (
            "freeze a real-event proximity-onset decision test"
            if gate["supported"]
            else (
                "retain the D22 seed17 proximity observation but stop "
                "target-specific dense-flow transfer"
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
                "gate": gate,
                "aggregate": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
