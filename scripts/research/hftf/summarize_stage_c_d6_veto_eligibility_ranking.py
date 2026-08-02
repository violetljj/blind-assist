#!/usr/bin/env python3
"""Summarize confidence-anchored veto-eligibility ranking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from summarize_stage_c_d6_early_pair_structured_field_canary import (
    FOLDS,
    SEEDS,
    summarize_values,
)
from train_stage_c_d5_tartanground_development_student import sha256


DEFAULT_CANDIDATE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d6-veto-eligibility-confidence-residual-canary-v0"
)


def macro_metric(
    environments: dict,
    model: str,
    metric: str,
) -> float:
    values = [
        row[model][metric] for row in environments.values()
    ]
    if any(value is None for value in values):
        raise ValueError(f"Undefined {model} {metric}")
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError("Refusing to overwrite ranking summary")

    units = []
    environment_rows = []
    values = {
        "macro_auroc": [],
        "macro_auroc_delta": [],
        "macro_average_precision_delta": [],
        "pooled_auroc_delta": [],
        "pooled_average_precision_delta": [],
    }
    for seed in SEEDS:
        for fold in FOLDS:
            path = (
                args.candidate_root
                / f"seed-{seed}"
                / f"fold-{fold}"
                / "report.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["task"]["ranking_mode"] != "confidence_residual":
                raise ValueError(
                    f"Unexpected ranking mode: seed={seed} fold={fold}"
                )
            environments = report[
                "selected_dev_metrics_by_environment"
            ]
            candidate_macro_auroc = macro_metric(
                environments,
                "candidate",
                "auroc",
            )
            comparator_macro_auroc = macro_metric(
                environments,
                "baseline_inverse_risk_confidence",
                "auroc",
            )
            candidate_macro_ap = macro_metric(
                environments,
                "candidate",
                "average_precision",
            )
            comparator_macro_ap = macro_metric(
                environments,
                "baseline_inverse_risk_confidence",
                "average_precision",
            )
            overall = report["selected_dev_metrics"]
            row = {
                "seed": seed,
                "fold": fold,
                "selected_epoch": report["selected_epoch"],
                "report_sha256": sha256(path),
                "macro_candidate_auroc": candidate_macro_auroc,
                "macro_comparator_auroc": comparator_macro_auroc,
                "macro_auroc_delta": (
                    candidate_macro_auroc - comparator_macro_auroc
                ),
                "macro_candidate_average_precision": (
                    candidate_macro_ap
                ),
                "macro_comparator_average_precision": (
                    comparator_macro_ap
                ),
                "macro_average_precision_delta": (
                    candidate_macro_ap - comparator_macro_ap
                ),
                "pooled_auroc_delta": overall[
                    "candidate_auroc_delta"
                ],
                "pooled_average_precision_delta": overall[
                    "candidate_average_precision_delta"
                ],
            }
            units.append(row)
            values["macro_auroc"].append(
                row["macro_candidate_auroc"]
            )
            for name in values:
                if name == "macro_auroc":
                    continue
                values[name].append(row[name])
            for environment, metrics in environments.items():
                environment_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        "auroc_delta": metrics[
                            "candidate_auroc_delta"
                        ],
                        "average_precision_delta": metrics[
                            "candidate_average_precision_delta"
                        ],
                    }
                )

    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_veto_eligibility_"
            "ranking_summary_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VETO_ELIGIBILITY_RANKING_SUMMARY_COMPLETE",
        "design": {
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "unit_count": len(units),
            "environment_unit_count": len(environment_rows),
            "ranking_mode": "confidence_residual",
            "comparator": "1 - frozen baseline risk probability",
        },
        "units": units,
        "summary": {
            **{
                name: summarize_values(metric_values)
                for name, metric_values in values.items()
            },
            "environment_auroc_delta": summarize_values(
                [row["auroc_delta"] for row in environment_rows]
            ),
            "environment_average_precision_delta": summarize_values(
                [
                    row["average_precision_delta"]
                    for row in environment_rows
                ]
            ),
        },
        "environment_rows": environment_rows,
        "evidence_limit": (
            "Outcome-open synthetic Development ranking only; no "
            "execution or promotion claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    Path(str(args.output) + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
