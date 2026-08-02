#!/usr/bin/env python3
"""Summarize early-pair structured-field canaries against references."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from train_stage_c_d5_tartanground_development_student import (
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_early_pair_"
    "structured_field_canary_summary_v1"
)
DEFAULT_REFERENCE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-cross-environment-v1"
)
DEFAULT_CANDIDATE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d6-early-pair-structured-field-canary-v0"
)
SEEDS = (17, 29, 43)
FOLDS = (0, 1, 2)


METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "selection_environment_macro_future_body_head_f1": (
        lambda report: float(report["selected_selection_score"])
    ),
    "aggregate_future_body_head_macro_f1": (
        lambda report: float(
            report["selected_dev_metrics"][
                "future_body_head_macro_f1"
            ]
        )
    ),
    "future_body_head_f1": (
        lambda report: float(
            report["selected_dev_metrics"][
                "risk_future_body_head"
            ]["f1"]
        )
    ),
    "future_body_head_auroc": (
        lambda report: float(
            report["selected_dev_metrics"][
                "risk_future_body_head"
            ]["auroc"]
        )
    ),
    "future_body_head_average_precision": (
        lambda report: float(
            report["selected_dev_metrics"][
                "risk_future_body_head"
            ]["average_precision"]
        )
    ),
    "future_body_head_recall": (
        lambda report: float(
            report["selected_dev_metrics"][
                "risk_future_body_head"
            ]["recall"]
        )
    ),
    "future_body_head_false_positive_rate": (
        lambda report: float(
            report["selected_dev_metrics"][
                "risk_future_body_head"
            ]["false_positive_rate"]
        )
    ),
    "known_accuracy": (
        lambda report: float(
            report["selected_dev_metrics"]["known_accuracy"]
        )
    ),
}


def summarize_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_count": int((array > 0.0).sum()),
        "zero_count": int((array == 0.0).sum()),
        "negative_count": int((array < 0.0).sum()),
    }


def reference_report_path(
    root: Path,
    seed: int,
    fold: int,
) -> Path:
    return (
        root
        / "training"
        / f"fold-{fold}"
        / f"directional-single-seed{seed}"
        / "report.json"
    )


def candidate_report_path(
    root: Path,
    seed: int,
    fold: int,
) -> Path:
    return root / f"seed-{seed}" / f"fold-{fold}" / "report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError("Refusing to overwrite early-pair summary")

    units = []
    metric_deltas: dict[str, list[float]] = {
        name: [] for name in METRICS
    }
    seed_metric_deltas: dict[
        str, dict[str, list[float]]
    ] = {
        str(seed): {name: [] for name in METRICS}
        for seed in args.seeds
    }
    environment_rows = []
    pair_constraint_modes = set()
    candidate_temporal_modes = set()
    for seed in args.seeds:
        for fold in FOLDS:
            reference_path = reference_report_path(
                args.reference_root,
                seed,
                fold,
            )
            candidate_path = candidate_report_path(
                args.candidate_root,
                seed,
                fold,
            )
            reference = json.loads(
                reference_path.read_text(encoding="utf-8")
            )
            candidate = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            if (
                candidate["temporal_mode"]
                not in {"early_pair", "early_pair_risk_veto"}
                or candidate["optimization"]["mode"]
                != "early_pair_only"
                or candidate["optimization"][
                    "initial_checkpoint_sha256"
                ]
                != reference["checkpoint"]["sha256"]
            ):
                raise ValueError(
                    f"Candidate/reference mismatch: seed {seed} "
                    f"fold {fold}"
                )
            pair_constraint_modes.add(
                candidate["optimization"].get(
                    "pair_constraint_mode",
                    "none",
                )
            )
            candidate_temporal_modes.add(candidate["temporal_mode"])
            deltas = {}
            for name, getter in METRICS.items():
                delta = getter(candidate) - getter(reference)
                deltas[name] = delta
                metric_deltas[name].append(delta)
                seed_metric_deltas[str(seed)][name].append(delta)
            units.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_epoch": candidate["selected_epoch"],
                    "reference_report_path": str(
                        reference_path.resolve()
                    ),
                    "reference_report_sha256": sha256(
                        reference_path
                    ),
                    "candidate_report_path": str(
                        candidate_path.resolve()
                    ),
                    "candidate_report_sha256": sha256(
                        candidate_path
                    ),
                    "metric_deltas": deltas,
                }
            )
            reference_environments = reference[
                "selected_dev_metrics_by_environment"
            ]
            candidate_environments = candidate[
                "selected_dev_metrics_by_environment"
            ]
            if set(reference_environments) != set(
                candidate_environments
            ):
                raise ValueError(
                    f"Environment mismatch: seed {seed} fold {fold}"
                )
            for environment in sorted(reference_environments):
                baseline = float(
                    reference_environments[environment][
                        "future_body_head_macro_f1"
                    ]
                )
                value = float(
                    candidate_environments[environment][
                        "future_body_head_macro_f1"
                    ]
                )
                environment_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        "reference": baseline,
                        "candidate": value,
                        "delta": value - baseline,
                    }
                )

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "EARLY_PAIR_STRUCTURED_FIELD_CANARY_SUMMARY_COMPLETE"
        ),
        "design": {
            "seeds": list(args.seeds),
            "folds": list(FOLDS),
            "unit_count": len(units),
            "environment_unit_count": len(environment_rows),
            "reference": (
                "directional single checkpoint selected by "
                "environment-macro future body/head F1"
            ),
            "candidate": (
                "zero-initialized early RGB-pair residual, "
                "early-pair parameters only"
            ),
            "pair_constraint_modes": sorted(pair_constraint_modes),
            "candidate_temporal_modes": sorted(
                candidate_temporal_modes
            ),
            "metric_search": False,
            "threshold_search": False,
        },
        "units": units,
        "metric_delta_summary": {
            name: summarize_values(values)
            for name, values in metric_deltas.items()
        },
        "by_seed_metric_delta_summary": {
            seed: {
                name: summarize_values(values)
                for name, values in metrics.items()
            }
            for seed, metrics in seed_metric_deltas.items()
        },
        "environment_future_body_head_macro_f1": {
            "summary": summarize_values(
                [row["delta"] for row in environment_rows]
            ),
            "rows": environment_rows,
        },
        "evidence_limit": (
            "Outcome-open synthetic Development representation "
            "comparison. Dev folds select epochs and do not provide "
            "fresh confirmation, real-event utility, mainline, App, "
            "production, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(str(args.output) + ".sha256").write_text(
        sha256(args.output) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "metric_delta_summary": (
                    report["metric_delta_summary"]
                ),
                "environment_summary": report[
                    "environment_future_body_head_macro_f1"
                ]["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
