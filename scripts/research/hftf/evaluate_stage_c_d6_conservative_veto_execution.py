#!/usr/bin/env python3
"""Test a zero-training-true-alert-veto execution rule."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_stage_c_d6_veto_eligibility_outcome_unseen import (
    DEFAULT_CANDIDATE_ROOT,
    DEFAULT_OUTCOME_ROOT,
    DEFAULT_PRETRAINED,
    DEFAULT_REFERENCE_ROOT,
    candidate_report_path,
    reference_report_path,
)
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
    collect_ranking_arrays,
    load_reference,
)


def zero_training_true_veto_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
) -> float:
    true_alert = eligible.astype(bool) & (target < 0.5)
    if not np.any(true_alert):
        raise ValueError("Threshold calibration needs true alerts")
    maximum = np.asarray(
        np.max(probability[true_alert]),
        dtype=probability.dtype,
    )
    infinity = np.asarray(np.inf, dtype=probability.dtype)
    return float(np.nextafter(maximum, infinity))


def execution_stats(
    probability: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> dict[str, int | float | None]:
    eligible_mask = eligible.astype(bool)
    false_alert = eligible_mask & (target >= 0.5)
    true_alert = eligible_mask & (target < 0.5)
    veto = eligible_mask & (probability >= threshold)
    false_veto = int(np.sum(veto & false_alert))
    true_veto = int(np.sum(veto & true_alert))
    false_total = int(np.sum(false_alert))
    true_total = int(np.sum(true_alert))
    return {
        "eligible_cells": int(np.sum(eligible_mask)),
        "false_alert_cells": false_total,
        "true_alert_cells": true_total,
        "vetoed_false_alert_cells": false_veto,
        "vetoed_true_alert_cells": true_veto,
        "false_alert_veto_coverage": (
            false_veto / false_total if false_total else None
        ),
        "true_alert_veto_rate": (
            true_veto / true_total if true_total else None
        ),
        "net_correct_veto_cells": false_veto - true_veto,
    }


def make_loader(
    records: list[dict[str, Any]],
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        HftfDataset(
            records,
            "history",
            train=False,
            seed=seed,
        ),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


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
        raise ValueError("Refusing to overwrite veto execution report")

    outcome_samples_path = args.outcome_root / "samples.jsonl"
    outcome_records = [
        row
        for row in load_jsonl(outcome_samples_path)
        if row["role"] == "transfer"
    ]
    environments = np.asarray(
        [row["environment"] for row in outcome_records],
        dtype=str,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = torch.cuda.is_available()
    outcome_loader = make_loader(
        outcome_records,
        seed=0,
        pin_memory=pin_memory,
    )
    train_records_by_fold = {}
    train_loaders = {}
    for fold in FOLDS:
        samples_path = (
            args.reference_root / f"fold-{fold}" / "samples.jsonl"
        )
        train_records = [
            row
            for row in load_jsonl(samples_path)
            if row["role"] == "train"
        ]
        train_records_by_fold[fold] = {
            "records": train_records,
            "samples_path": samples_path,
        }
        train_loaders[fold] = make_loader(
            train_records,
            seed=0,
            pin_memory=pin_memory,
        )

    units = []
    environment_rows = []
    false_coverages = []
    true_veto_rates = []
    net_correct = []
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
            reference_checkpoint_path = Path(
                candidate_report["reference_checkpoint_path"]
            )
            reference, _ = load_reference(
                args.pretrained,
                reference_checkpoint_path,
                device,
            )
            candidate_checkpoint_path = Path(
                candidate_report["checkpoint"]["path"]
            )
            checkpoint = torch.load(
                candidate_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            student = VetoEligibilityStudent(zero_head=True)
            student.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            student.to(device).eval()
            train_arrays = collect_ranking_arrays(
                student,
                reference,
                train_loaders[fold],
                device,
                "confidence_residual",
            )
            threshold = zero_training_true_veto_threshold(
                train_arrays["probability"],
                train_arrays["target"],
                train_arrays["eligible"],
            )
            train_stats = execution_stats(
                train_arrays["probability"],
                train_arrays["target"],
                train_arrays["eligible"],
                threshold,
            )
            if train_stats["vetoed_true_alert_cells"] != 0:
                raise RuntimeError(
                    "Calibrated threshold vetoed a training true alert"
                )
            outcome_arrays = collect_ranking_arrays(
                student,
                reference,
                outcome_loader,
                device,
                "confidence_residual",
            )
            outcome_stats = execution_stats(
                outcome_arrays["probability"],
                outcome_arrays["target"],
                outcome_arrays["eligible"],
                threshold,
            )
            if outcome_stats["false_alert_veto_coverage"] is not None:
                false_coverages.append(
                    outcome_stats["false_alert_veto_coverage"]
                )
            if outcome_stats["true_alert_veto_rate"] is not None:
                true_veto_rates.append(
                    outcome_stats["true_alert_veto_rate"]
                )
            net_correct.append(
                float(outcome_stats["net_correct_veto_cells"])
            )
            for environment in sorted(set(environments.tolist())):
                selected = environments == environment
                stats = execution_stats(
                    outcome_arrays["probability"][selected],
                    outcome_arrays["target"][selected],
                    outcome_arrays["eligible"][selected],
                    threshold,
                )
                environment_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        **stats,
                    }
                )
            units.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_epoch": candidate_report[
                        "selected_epoch"
                    ],
                    "threshold": threshold,
                    "threshold_rule": (
                        "nextafter(max training true-alert score, +inf)"
                    ),
                    "train": train_stats,
                    "outcome_unseen": outcome_stats,
                    "candidate_report_sha256": sha256(
                        candidate_path
                    ),
                    "candidate_checkpoint_sha256": sha256(
                        candidate_checkpoint_path
                    ),
                    "reference_report_sha256": sha256(
                        reference_path
                    ),
                    "train_samples_sha256": sha256(
                        train_records_by_fold[fold]["samples_path"]
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "threshold": threshold,
                        "false_alert_veto_coverage": outcome_stats[
                            "false_alert_veto_coverage"
                        ],
                        "true_alert_veto_rate": outcome_stats[
                            "true_alert_veto_rate"
                        ],
                    }
                ),
                flush=True,
            )

    evaluable_environment_false = [
        row["false_alert_veto_coverage"]
        for row in environment_rows
        if row["false_alert_veto_coverage"] is not None
    ]
    evaluable_environment_true = [
        row["true_alert_veto_rate"]
        for row in environment_rows
        if row["true_alert_veto_rate"] is not None
    ]
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_conservative_"
            "veto_execution_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CONSERVATIVE_VETO_EXECUTION_CANARY_COMPLETE",
        "policy": {
            "outcome_open": True,
            "promotion_evidence": False,
            "system_output_connected": False,
            "threshold_search": False,
            "threshold_uses_outcome_labels": False,
        },
        "design": {
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "unit_count": len(units),
            "environment_unit_count": len(environment_rows),
            "threshold_rule": (
                "nextafter(max training true-alert score, +inf)"
            ),
            "training_true_alert_veto_budget": 0,
            "outcome_environments": sorted(set(environments.tolist())),
        },
        "outcome_samples_sha256": sha256(outcome_samples_path),
        "units": units,
        "summary": {
            "outcome_false_alert_veto_coverage": summarize_values(
                false_coverages
            ),
            "outcome_true_alert_veto_rate": summarize_values(
                true_veto_rates
            ),
            "outcome_net_correct_veto_cells": summarize_values(
                net_correct
            ),
            "environment_false_alert_veto_coverage": (
                summarize_values(evaluable_environment_false)
            ),
            "environment_true_alert_veto_rate": summarize_values(
                evaluable_environment_true
            ),
        },
        "environment_rows": environment_rows,
        "evidence_limit": (
            "Consumed synthetic Development execution canary only; "
            "not a production threshold, event utility result, or "
            "real-source promotion claim."
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
