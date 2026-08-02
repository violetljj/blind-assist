#!/usr/bin/env python3
"""Evaluate veto-eligibility ranking on outcome-unseen environments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from summarize_stage_c_d6_early_pair_structured_field_canary import (
    FOLDS,
    SEEDS,
    summarize_values,
)
from train_stage_c_d5_tartanground_development_student import (
    HftfDataset,
    load_jsonl,
    sha256,
)
from train_stage_c_d6_veto_eligibility_ranking import (
    VetoEligibilityStudent,
    evaluate,
    load_reference,
)


DEFAULT_CANDIDATE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d6-veto-eligibility-confidence-residual-canary-v0"
)
DEFAULT_REFERENCE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-cross-environment-v1"
)
DEFAULT_OUTCOME_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-outcome-unseen-transfer-v0"
)
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)


def candidate_report_path(
    root: Path,
    seed: int,
    fold: int,
) -> Path:
    return root / f"seed-{seed}" / f"fold-{fold}" / "report.json"


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


def mean_metric(
    rows: dict[str, dict[str, Any]],
    model: str,
    metric: str,
) -> float:
    values = [
        row[model][metric]
        for row in rows.values()
    ]
    if any(value is None for value in values):
        raise ValueError(
            f"Every environment needs {model} {metric}"
        )
    return float(np.mean(values))


def paired_environment_metric(
    rows: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    evaluable = [
        environment
        for environment, row in rows.items()
        if row["candidate"][metric] is not None
        and row["baseline_inverse_risk_confidence"][metric]
        is not None
    ]
    not_evaluable = sorted(set(rows) - set(evaluable))
    if not evaluable:
        raise ValueError(
            f"No outcome environment is evaluable for {metric}"
        )
    candidate = float(
        np.mean(
            [
                rows[environment]["candidate"][metric]
                for environment in evaluable
            ]
        )
    )
    comparator = float(
        np.mean(
            [
                rows[environment][
                    "baseline_inverse_risk_confidence"
                ][metric]
                for environment in evaluable
            ]
        )
    )
    return {
        "candidate": candidate,
        "comparator": comparator,
        "delta": candidate - comparator,
        "evaluable_environments": evaluable,
        "not_evaluable_environments": not_evaluable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--outcome-root",
        type=Path,
        default=DEFAULT_OUTCOME_ROOT,
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError("Refusing to overwrite outcome ranking report")

    samples_path = args.outcome_root / "samples.jsonl"
    records = [
        row
        for row in load_jsonl(samples_path)
        if row["role"] == "transfer"
    ]
    environments = sorted(
        {row["environment"] for row in records}
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    overall_loader = DataLoader(
        HftfDataset(
            records,
            "history",
            train=False,
            seed=0,
        ),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    environment_loaders = {
        environment: DataLoader(
            HftfDataset(
                [
                    row
                    for row in records
                    if row["environment"] == environment
                ],
                "history",
                train=False,
                seed=0,
            ),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for environment in environments
    }

    units = []
    macro_auroc_deltas = []
    macro_ap_deltas = []
    pooled_auroc_deltas = []
    pooled_ap_deltas = []
    environment_rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            candidate_path = candidate_report_path(
                args.candidate_root,
                seed,
                fold,
            )
            reference_path = reference_report_path(
                args.reference_root,
                seed,
                fold,
            )
            candidate_report = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            reference_report = json.loads(
                reference_path.read_text(encoding="utf-8")
            )
            if (
                candidate_report["task"]["ranking_mode"]
                != "confidence_residual"
                or candidate_report[
                    "reference_checkpoint_sha256"
                ]
                != reference_report["checkpoint"]["sha256"]
            ):
                raise ValueError(
                    f"Candidate/reference mismatch: seed={seed} "
                    f"fold={fold}"
                )
            reference_checkpoint = Path(
                candidate_report["reference_checkpoint_path"]
            )
            reference, _ = load_reference(
                args.pretrained,
                reference_checkpoint,
                device,
            )
            candidate_checkpoint_path = Path(
                candidate_report["checkpoint"]["path"]
            )
            candidate_checkpoint = torch.load(
                candidate_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if (
                candidate_checkpoint.get("ranking_mode")
                != "confidence_residual"
            ):
                raise ValueError("Unexpected candidate ranking mode")
            student = VetoEligibilityStudent(zero_head=True)
            student.load_state_dict(
                candidate_checkpoint["model_state_dict"],
                strict=True,
            )
            student.to(device).eval()
            overall = evaluate(
                student,
                reference,
                overall_loader,
                device,
                "confidence_residual",
            )
            by_environment = {
                environment: evaluate(
                    student,
                    reference,
                    loader,
                    device,
                    "confidence_residual",
                )
                for environment, loader in environment_loaders.items()
            }
            macro_auroc = paired_environment_metric(
                by_environment,
                "auroc",
            )
            macro_ap = paired_environment_metric(
                by_environment,
                "average_precision",
            )
            candidate_macro_auroc = macro_auroc["candidate"]
            comparator_macro_auroc = macro_auroc["comparator"]
            candidate_macro_ap = macro_ap["candidate"]
            comparator_macro_ap = macro_ap["comparator"]
            macro_auroc_delta = macro_auroc["delta"]
            macro_ap_delta = macro_ap["delta"]
            macro_auroc_deltas.append(macro_auroc_delta)
            macro_ap_deltas.append(macro_ap_delta)
            pooled_auroc_deltas.append(
                overall["candidate_auroc_delta"]
            )
            pooled_ap_deltas.append(
                overall["candidate_average_precision_delta"]
            )
            for environment, metrics in by_environment.items():
                auroc_delta = (
                    metrics["candidate_auroc_delta"]
                )
                ap_delta = metrics[
                    "candidate_average_precision_delta"
                ]
                environment_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        "eligible_cells": metrics["candidate"][
                            "known_cells"
                        ],
                        "false_alert_prevalence": metrics[
                            "false_alert_prevalence"
                        ],
                        "auroc_delta": auroc_delta,
                        "average_precision_delta": ap_delta,
                        "auroc_evaluable": auroc_delta is not None,
                        "average_precision_evaluable": (
                            ap_delta is not None
                        ),
                    }
                )
            units.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_epoch": candidate_report[
                        "selected_epoch"
                    ],
                    "candidate_report_sha256": sha256(
                        candidate_path
                    ),
                    "candidate_checkpoint_sha256": sha256(
                        candidate_checkpoint_path
                    ),
                    "reference_report_sha256": sha256(
                        reference_path
                    ),
                    "macro_candidate_auroc": candidate_macro_auroc,
                    "macro_comparator_auroc": (
                        comparator_macro_auroc
                    ),
                    "macro_auroc_delta": macro_auroc_delta,
                    "macro_candidate_average_precision": (
                        candidate_macro_ap
                    ),
                    "macro_comparator_average_precision": (
                        comparator_macro_ap
                    ),
                    "macro_average_precision_delta": macro_ap_delta,
                    "macro_auroc_evaluable_environments": (
                        macro_auroc["evaluable_environments"]
                    ),
                    "macro_auroc_not_evaluable_environments": (
                        macro_auroc["not_evaluable_environments"]
                    ),
                    "macro_ap_evaluable_environments": (
                        macro_ap["evaluable_environments"]
                    ),
                    "macro_ap_not_evaluable_environments": (
                        macro_ap["not_evaluable_environments"]
                    ),
                    "overall": overall,
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "macro_auroc_delta": macro_auroc_delta,
                        "macro_average_precision_delta": (
                            macro_ap_delta
                        ),
                    }
                ),
                flush=True,
            )

    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_veto_eligibility_"
            "outcome_unseen_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OUTCOME_UNSEEN_VETO_ELIGIBILITY_RANKING_COMPLETE",
        "policy": {
            "outcome_open": True,
            "promotion_evidence": False,
            "system_output_connected": False,
            "threshold_search": False,
        },
        "design": {
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "environments": environments,
            "unit_count": len(units),
            "environment_unit_count": len(environment_rows),
            "ranking_mode": "confidence_residual",
            "comparator": "1 - frozen baseline risk probability",
        },
        "samples_path": str(samples_path.resolve()),
        "samples_sha256": sha256(samples_path),
        "units": units,
        "summary": {
            "macro_auroc_delta": summarize_values(
                macro_auroc_deltas
            ),
            "macro_average_precision_delta": summarize_values(
                macro_ap_deltas
            ),
            "pooled_auroc_delta": summarize_values(
                pooled_auroc_deltas
            ),
            "pooled_average_precision_delta": summarize_values(
                pooled_ap_deltas
            ),
            "environment_auroc_delta": summarize_values(
                [
                    row["auroc_delta"]
                    for row in environment_rows
                    if row["auroc_delta"] is not None
                ]
            ),
            "environment_average_precision_delta": summarize_values(
                [
                    row["average_precision_delta"]
                    for row in environment_rows
                    if row["average_precision_delta"] is not None
                ]
            ),
        },
        "environment_rows": environment_rows,
        "evidence_limit": (
            "Outcome-unseen synthetic Development ranking only; no "
            "threshold, event action, real-source, or promotion claim."
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
    print(
        json.dumps(
            {"ok": True, "summary": report["summary"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
