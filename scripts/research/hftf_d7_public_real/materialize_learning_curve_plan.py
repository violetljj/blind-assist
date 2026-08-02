#!/usr/bin/env python3
"""Write the fixed post-completion learning-curve protocol without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import load_json, utc_now, write_json


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_root).resolve()
    validation_path = root / "reports" / "d7_validation_report.json"
    validation = load_json(validation_path) if validation_path.is_file() else {}
    plan = {
        "schema": "hftf_d7_public_real_learning_curve_plan_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "generated_at_utc": utc_now(),
        "status": "BLOCKED_UNTIL_DATASET_COMPLETE",
        "dataset_validation_status": validation.get("status", "UNKNOWN"),
        "dataset_admitted_parent_events": validation.get("counts", {}).get("adjudicated_parent_events", 0),
        "training_authorized": False,
        "model_changes_authorized": False,
        "fixed_model_contract": {
            "yolo": "frozen_existing_checkpoint",
            "hftf": "frozen_existing_directional_v2_and_spatial_relation_head",
            "thresholds": "frozen_existing_values",
            "confirmation_length": "frozen_existing_value",
            "backbone": "frozen_existing_encoder",
        },
        "training_event_counts": [250, 500, 1000, 2000, 4000, 6000],
        "seeds_per_count": 3,
        "primary_metrics": [
            "positive_event_hits",
            "false_alert_events",
            "clearance",
            "response_delay",
            "worst_source_performance",
            "seed_variance",
            "yolo_miss_recovery",
            "hftf_only_false_alerts",
        ],
        "required_controls": [
            "source-session-disjoint roles",
            "frozen encoder/checkpoint/threshold/confirmation length",
            "pre-registered optimizer and operating point",
            "no candidate rescue by threshold/model/feature/seed search",
            "report NOT_EVALUABLE and source-specific missingness",
        ],
        "execution_gate": "Only open after d7_validation_report.status=COMPLETE and independent review receipts are hash-bound.",
        "unfreezing_gate": "Only consider the minimal near-output backbone subset if fixed-encoder curves continue improving and approach saturation across seeds and worst-source slices.",
    }
    write_json(root / "manifests" / "learning_curve_plan.json", plan)
    (root / "reports" / "learning_curve_plan.md").write_text(
        "\n".join([
            "# HFTF D7 fixed-model learning-curve plan",
            "",
            f"Status: `{plan['status']}`.",
            "",
            "This plan is materialized for later use; no model, threshold, backbone, or confirmation-length change was made and no training run was opened.",
            "",
            "- Event counts: `250 / 500 / 1,000 / 2,000 / 4,000 / 6,000`.",
            "- Seeds per count: `3`.",
            "- Gate: `d7_validation_report.status=COMPLETE` plus hash-bound review receipts.",
            "- Metrics: positive-event hits, false-alert events, clearance, response delay, worst-source performance, seed variance, YOLO-miss recovery, and HFTF-only false alerts.",
            "- Unfreezing is not authorized by this plan; it requires a separate post-curve decision.",
        ]) + "\n",
        encoding="utf-8",
    )
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=False, default="d7-r1-learning-curve-plan-20260802")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
