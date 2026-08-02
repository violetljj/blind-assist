#!/usr/bin/env python3
"""Evaluate early-pair fields on outcome-unseen TartanGround."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from evaluate_stage_c_d5_tartanground_event_proxy import (
    model_metrics,
    predict,
)
from summarize_stage_c_d6_early_pair_structured_field_canary import (
    DEFAULT_CANDIDATE_ROOT,
    DEFAULT_REFERENCE_ROOT,
    FOLDS,
    SEEDS,
    candidate_report_path,
    reference_report_path,
    summarize_values,
)
from train_stage_c_d5_tartanground_development_student import (
    decode_labels,
    load_jsonl,
    sha256,
    summarize_metrics,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_early_pair_"
    "outcome_unseen_transfer_v1"
)
DECISION_POLICY = "height_spatiotemporal_selective_v2"
DEFAULT_OUTCOME_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-outcome-unseen-transfer-v0"
)
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)


CELL_METRICS: dict[
    str, Callable[[dict[str, Any]], float]
] = {
    "future_body_head_macro_f1": (
        lambda metrics: float(
            metrics["future_body_head_macro_f1"]
        )
    ),
    "future_body_head_f1": (
        lambda metrics: float(
            metrics["risk_future_body_head"]["f1"]
        )
    ),
    "future_body_head_auroc": (
        lambda metrics: float(
            metrics["risk_future_body_head"]["auroc"]
        )
    ),
    "future_body_head_average_precision": (
        lambda metrics: float(
            metrics["risk_future_body_head"]["average_precision"]
        )
    ),
    "future_body_head_recall": (
        lambda metrics: float(
            metrics["risk_future_body_head"]["recall"]
        )
    ),
    "future_body_head_false_positive_rate": (
        lambda metrics: float(
            metrics["risk_future_body_head"][
                "false_positive_rate"
            ]
        )
    ),
}
EVENT_METRICS = (
    "event_recall",
    "false_active_lane_frame_rate",
    "clearance_rate",
    "hit_event_count",
    "false_alert_event_count",
    "cleared_event_count",
)


def existing_cell_path(
    outcome_root: Path,
    seed: int,
    fold: int,
) -> Path:
    return (
        outcome_root
        / "evaluation"
        / "cell-level"
        / f"seed-{seed}"
        / f"fold-{fold}.json"
    )


def existing_event_path(
    outcome_root: Path,
    seed: int,
    fold: int,
) -> Path:
    return (
        outcome_root
        / "evaluation"
        / "height-spatiotemporal-selective-v2"
        / f"seed-{seed}"
        / f"fold-{fold}.json"
    )


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
        raise ValueError(
            "Refusing to overwrite early-pair outcome transfer"
        )

    samples_path = args.outcome_root / "samples.jsonl"
    records = [
        record
        for record in load_jsonl(samples_path)
        if record["role"] == "transfer"
    ]
    records.sort(
        key=lambda record: (
            record["parent_id"],
            record["anchor_frame_id"],
        )
    )
    truth_rows = []
    truth_known_rows = []
    for record in records:
        risk, known = decode_labels(record)
        truth_rows.append(risk.numpy())
        truth_known_rows.append(known.numpy())
    truth = np.stack(truth_rows)
    truth_known = np.stack(truth_known_rows)
    environments = np.asarray(
        [record["environment"] for record in records],
        dtype=str,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    units = []
    cell_deltas: dict[str, list[float]] = {
        name: [] for name in CELL_METRICS
    }
    event_deltas: dict[str, list[float]] = {
        name: [] for name in EVENT_METRICS
    }
    environment_cell_rows = []
    environment_event_rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            reference_report = json.loads(
                reference_report_path(
                    args.reference_root,
                    seed,
                    fold,
                ).read_text(encoding="utf-8")
            )
            candidate_report = json.loads(
                candidate_report_path(
                    args.candidate_root,
                    seed,
                    fold,
                ).read_text(encoding="utf-8")
            )
            cell_path = existing_cell_path(
                args.outcome_root,
                seed,
                fold,
            )
            event_path = existing_event_path(
                args.outcome_root,
                seed,
                fold,
            )
            existing_cell = json.loads(
                cell_path.read_text(encoding="utf-8")
            )
            existing_event = json.loads(
                event_path.read_text(encoding="utf-8")
            )
            reference_sha = reference_report["checkpoint"]["sha256"]
            if (
                existing_cell["models"]["directional"][
                    "checkpoint_sha256"
                ]
                != reference_sha
                or existing_event["models"]["directional"][
                    "checkpoint_sha256"
                ]
                != reference_sha
                or existing_cell["samples"]["sha256"]
                != sha256(samples_path)
                or existing_event["samples_sha256"]
                != sha256(samples_path)
                or candidate_report["optimization"][
                    "initial_checkpoint_sha256"
                ]
                != reference_sha
            ):
                raise ValueError(
                    f"Outcome reference mismatch: seed {seed} "
                    f"fold {fold}"
                )
            candidate_checkpoint = Path(
                candidate_report["checkpoint"]["path"]
            )
            probability, known_probability = predict(
                records,
                candidate_checkpoint,
                args.pretrained,
                device,
                input_arm="history",
            )
            candidate_cell = summarize_metrics(
                probability,
                known_probability,
                truth,
                truth_known,
            )
            candidate_event = model_metrics(
                records,
                probability,
                known_probability,
                decision_policy=DECISION_POLICY,
            )
            reference_cell = existing_cell["models"][
                "directional"
            ]["overall"]
            reference_event = existing_event["models"][
                "directional"
            ]
            unit_cell_deltas = {}
            for name, getter in CELL_METRICS.items():
                delta = (
                    getter(candidate_cell)
                    - getter(reference_cell)
                )
                unit_cell_deltas[name] = delta
                cell_deltas[name].append(delta)
            unit_event_deltas = {}
            for name in EVENT_METRICS:
                reference_value = reference_event["overall"][name]
                candidate_value = candidate_event["overall"][name]
                delta = (
                    float(candidate_value) - float(reference_value)
                    if reference_value is not None
                    and candidate_value is not None
                    else None
                )
                unit_event_deltas[name] = delta
                if delta is not None:
                    event_deltas[name].append(delta)
            units.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_epoch": candidate_report[
                        "selected_epoch"
                    ],
                    "candidate_checkpoint_sha256": sha256(
                        candidate_checkpoint
                    ),
                    "cell_deltas": unit_cell_deltas,
                    "event_deltas": unit_event_deltas,
                }
            )
            for environment in sorted(set(environments.tolist())):
                selected = environments == environment
                candidate_environment_cell = summarize_metrics(
                    probability[selected],
                    known_probability[selected],
                    truth[selected],
                    truth_known[selected],
                )
                baseline_environment_cell = existing_cell[
                    "models"
                ]["directional"]["by_environment"][environment]
                environment_cell_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        "delta": (
                            candidate_environment_cell[
                                "future_body_head_macro_f1"
                            ]
                            - baseline_environment_cell[
                                "future_body_head_macro_f1"
                            ]
                        ),
                    }
                )
                environment_event_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        "delta": (
                            candidate_event["by_environment"][
                                environment
                            ]["event_recall"]
                            - reference_event["by_environment"][
                                environment
                            ]["event_recall"]
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "future_body_head_macro_f1_delta": (
                            unit_cell_deltas[
                                "future_body_head_macro_f1"
                            ]
                        ),
                        "event_recall_delta": unit_event_deltas[
                            "event_recall"
                        ],
                    }
                ),
                flush=True,
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "EARLY_PAIR_OUTCOME_UNSEEN_TRANSFER_COMPLETE"
        ),
        "design": {
            "sample_count": len(records),
            "environment_count": len(set(environments.tolist())),
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "unit_count": len(units),
            "input_arm": "history",
            "decision_policy": DECISION_POLICY,
            "candidate_checkpoints_reinferred": True,
            "reference_reports_reused": True,
            "threshold_search": False,
        },
        "samples_path": str(samples_path.resolve()),
        "samples_sha256": sha256(samples_path),
        "units": units,
        "cell_metric_delta_summary": {
            name: summarize_values(values)
            for name, values in cell_deltas.items()
        },
        "event_metric_delta_summary": {
            name: (
                summarize_values(values)
                if values
                else None
            )
            for name, values in event_deltas.items()
        },
        "environment_cell_future_body_head_macro_f1": (
            summarize_values(
                [row["delta"] for row in environment_cell_rows]
            )
        ),
        "environment_event_recall": summarize_values(
            [row["delta"] for row in environment_event_rows]
        ),
        "environment_cell_rows": environment_cell_rows,
        "environment_event_rows": environment_event_rows,
        "evidence_limit": (
            "Previously consumed outcome-unseen synthetic "
            "Development transfer. This is not fresh confirmation, "
            "human real-event utility, mainline, App, production, "
            "or safety evidence."
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
                "cell_metric_delta_summary": (
                    report["cell_metric_delta_summary"]
                ),
                "event_metric_delta_summary": (
                    report["event_metric_delta_summary"]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
