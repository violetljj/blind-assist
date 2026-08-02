#!/usr/bin/env python3
"""Compare causal HFTF decision kernels on one fixed checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_stage_c_d5_tartanground_event_proxy import (
    DECISION_POLICIES,
    comparison,
    decision_policy_spec,
    model_metrics,
    predict,
)
from train_stage_c_d5_tartanground_development_student import (
    load_jsonl,
    sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decision-policy",
        action="append",
        choices=DECISION_POLICIES,
        dest="decision_policies",
    )
    parser.add_argument(
        "--reference",
        choices=DECISION_POLICIES,
        default="hard_known_and_risk",
    )
    parser.add_argument("--role", default="dev")
    args = parser.parse_args()

    decision_policies = args.decision_policies or list(DECISION_POLICIES)
    if len(decision_policies) != len(set(decision_policies)):
        parser.error("--decision-policy values must be unique")
    if args.reference not in decision_policies:
        parser.error("--reference must name a requested decision policy")

    records = [
        record
        for record in load_jsonl(args.samples)
        if record["role"] == args.role
    ]
    records.sort(
        key=lambda record: (
            record["parent_id"],
            record["anchor_frame_id"],
        )
    )
    if not records:
        raise ValueError("No records for requested role")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    risk, known = predict(
        records,
        args.checkpoint,
        args.pretrained,
        device,
    )
    kernels = {
        policy: {
            "definition": decision_policy_spec(policy),
            **model_metrics(
                records,
                risk,
                known,
                decision_policy=policy,
            ),
        }
        for policy in decision_policies
    }
    reference = kernels[args.reference]
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d5_tartanground_"
            "decision_kernel_comparison_v0"
        ),
        "status": "DEVELOPMENT_DECISION_KERNEL_COMPARISON_COMPLETE",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "human_event_truth": False,
            "system_claim": False,
            "promotion_evidence": False,
        },
        "truth_definition": {
            "unit": "environment_x_horizon_x_height_x_direction_lane",
            "positive": (
                "any teacher-known distance cell with risk_score >= 0.5"
            ),
            "negative": (
                "all six distance cells teacher-known and none positive"
            ),
            "unknown": "otherwise",
            "anchor_period_s": 0.2,
        },
        "samples_path": str(args.samples.resolve()),
        "samples_sha256": sha256(args.samples),
        "sample_count": len(records),
        "role": args.role,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "reference": args.reference,
        "kernels": kernels,
        "comparisons": {
            policy: comparison(metrics, reference)
            for policy, metrics in kernels.items()
            if policy != args.reference
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    for policy, metrics in kernels.items():
        print(
            json.dumps(
                {
                    "decision_policy": policy,
                    "event_recall": metrics["overall"]["event_recall"],
                    "false_active_lane_frame_rate": metrics["overall"][
                        "false_active_lane_frame_rate"
                    ],
                    "clearance_rate": metrics["overall"][
                        "clearance_rate"
                    ],
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
