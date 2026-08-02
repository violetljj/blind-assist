#!/usr/bin/env python3
"""Evaluate selected early-pair checkpoints on the fixed event proxy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    load_jsonl,
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_early_pair_"
    "structured_field_event_proxy_v1"
)
DECISION_POLICY = "height_spatiotemporal_selective_v2"
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)
EVENT_METRICS = (
    "event_recall",
    "false_active_lane_frame_rate",
    "clearance_rate",
    "hit_event_count",
    "false_alert_event_count",
    "cleared_event_count",
)


def existing_reference_event_path(
    root: Path,
    seed: int,
    fold: int,
) -> Path:
    return (
        root
        / "event-proxy"
        / DECISION_POLICY.replace("_", "-")
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
            "Refusing to overwrite early-pair event proxy"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    units = []
    metric_deltas: dict[str, list[float]] = {
        name: [] for name in EVENT_METRICS
    }
    environment_deltas = []
    for seed in SEEDS:
        for fold in FOLDS:
            samples_path = (
                args.reference_root
                / f"fold-{fold}"
                / "samples.jsonl"
            )
            records = [
                record
                for record in load_jsonl(samples_path)
                if record["role"] == "dev"
            ]
            records.sort(
                key=lambda record: (
                    record["parent_id"],
                    record["anchor_frame_id"],
                )
            )
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
            existing_path = existing_reference_event_path(
                args.reference_root,
                seed,
                fold,
            )
            existing = json.loads(
                existing_path.read_text(encoding="utf-8")
            )
            if (
                existing["definition"]["decision_policy"]
                != DECISION_POLICY
                or existing["samples_sha256"]
                != sha256(samples_path)
                or existing["models"]["directional"][
                    "checkpoint_sha256"
                ]
                != reference_report["checkpoint"]["sha256"]
                or candidate_report["optimization"][
                    "initial_checkpoint_sha256"
                ]
                != reference_report["checkpoint"]["sha256"]
            ):
                raise ValueError(
                    f"Event reference mismatch: seed {seed} "
                    f"fold {fold}"
                )
            candidate_checkpoint = Path(
                candidate_report["checkpoint"]["path"]
            )
            risk, known = predict(
                records,
                candidate_checkpoint,
                args.pretrained,
                device,
                input_arm="history",
            )
            candidate_metrics = model_metrics(
                records,
                risk,
                known,
                decision_policy=DECISION_POLICY,
            )
            reference_metrics = existing["models"]["directional"]
            deltas = {}
            for name in EVENT_METRICS:
                reference_value = reference_metrics["overall"][name]
                candidate_value = candidate_metrics["overall"][name]
                delta = (
                    float(candidate_value) - float(reference_value)
                    if reference_value is not None
                    and candidate_value is not None
                    else None
                )
                deltas[name] = delta
                if delta is not None:
                    metric_deltas[name].append(delta)
            units.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selected_epoch": candidate_report[
                        "selected_epoch"
                    ],
                    "samples_sha256": sha256(samples_path),
                    "reference_event_report_path": str(
                        existing_path.resolve()
                    ),
                    "reference_event_report_sha256": sha256(
                        existing_path
                    ),
                    "candidate_checkpoint_path": str(
                        candidate_checkpoint.resolve()
                    ),
                    "candidate_checkpoint_sha256": sha256(
                        candidate_checkpoint
                    ),
                    "reference_overall": reference_metrics["overall"],
                    "candidate_overall": candidate_metrics["overall"],
                    "deltas": deltas,
                }
            )
            for environment in sorted(
                reference_metrics["by_environment"]
            ):
                reference_value = reference_metrics[
                    "by_environment"
                ][environment]["event_recall"]
                candidate_value = candidate_metrics[
                    "by_environment"
                ][environment]["event_recall"]
                environment_deltas.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "environment": environment,
                        "reference_event_recall": reference_value,
                        "candidate_event_recall": candidate_value,
                        "delta": (
                            float(candidate_value)
                            - float(reference_value)
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "event_recall_delta": deltas[
                            "event_recall"
                        ],
                        "false_active_lane_frame_rate_delta": (
                            deltas["false_active_lane_frame_rate"]
                        ),
                    }
                ),
                flush=True,
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "EARLY_PAIR_STRUCTURED_FIELD_EVENT_PROXY_COMPLETE"
        ),
        "design": {
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "unit_count": len(units),
            "decision_policy": DECISION_POLICY,
            "reference_event_metrics_reused": True,
            "candidate_checkpoints_reinferred": True,
            "threshold_search": False,
        },
        "units": units,
        "metric_delta_summary": {
            name: (
                summarize_values(values)
                if values
                else None
            )
            for name, values in metric_deltas.items()
        },
        "environment_event_recall": {
            "summary": summarize_values(
                [row["delta"] for row in environment_deltas]
            ),
            "rows": environment_deltas,
        },
        "evidence_limit": (
            "Synthetic teacher-derived Development event proxy. "
            "It is not human event truth, source-fresh confirmation, "
            "real-event utility, mainline, App, production, or safety "
            "evidence."
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
                "environment_event_recall": report[
                    "environment_event_recall"
                ]["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
