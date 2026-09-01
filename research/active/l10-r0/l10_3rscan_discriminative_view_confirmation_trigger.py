#!/usr/bin/env python3
"""Authorize the fixed action from initial scorer margins only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-confirmation-trigger-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-discriminative-view-confirmation-trigger-result-v1"


def _rank(row: dict[str, Any], minimum_iou: float) -> int | None:
    return next(
        (
            index
            for index, candidate in enumerate(row["ranked_candidates"], start=1)
            if float(candidate["target_metrics_evaluation_only"]["iou"]) >= minimum_iou
        ),
        None,
    )


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    scorer_protocol_row = protocol["scorer_protocol"]
    scorer_protocol_path = HERE / scorer_protocol_row["path"]
    pixel.require(pixel.sha256(scorer_protocol_path) == scorer_protocol_row["sha256"], "SCORER_PROTOCOL_HASH")
    scorer_row = protocol["scorer_result"]
    scorer = pixel.load_json(HERE / scorer_row["path"])
    pixel.require(scorer["schema"] == scorer_row["required_schema"], "SCORER_SCHEMA")
    pixel.require(scorer["conclusion"] == scorer_row["required_conclusion"], "SCORER_CONCLUSION")
    pixel.require(scorer["protocol_sha256"] == scorer_protocol_row["sha256"], "SCORER_PROTOCOL_RECEIPT")

    minimum_iou = float(protocol["evaluation"]["minimum_iou"])
    keys = protocol["policy"]["initial_query_keys"]
    rows = []
    for key in keys:
        receipt = scorer["query_receipts"][key]
        rank = _rank(receipt, minimum_iou)
        rows.append(
            {
                "query_key": key,
                "top1_minus_top2_fused_margin": float(receipt["top1_layer18_nids_fused_margin"]),
                "first_target_proposal_rank_evaluation_only": rank,
                "target_top1_evaluation_only": rank == 1,
                "target_recall_at_3_evaluation_only": rank is not None and rank <= 3,
                "top1_target_iou_evaluation_only": float(receipt["selected"]["target_metrics_evaluation_only"]["iou"]),
                "refined_top1_target_iou_evaluation_only": float(receipt["refined_mask_bbox_target_metrics_evaluation_only"]["iou"]),
            }
        )
    trigger_value = min(row["top1_minus_top2_fused_margin"] for row in rows)
    threshold = float(protocol["policy"]["truth_free_trigger_threshold"])
    triggered = trigger_value <= threshold
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_INITIAL_SCORE_ONLY_ACTION_AUTHORIZATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_DISCRIMINATIVE_VIEW_CONFIRMATION_ACTION_TRIGGERED"
            if triggered
            else "L10_3RSCAN_DISCRIMINATIVE_VIEW_CONFIRMATION_ACTION_NOT_TRIGGERED"
        ),
        "trigger_uses_target_truth": False,
        "trigger_statistic": "minimum initial Top-1-minus-Top-2 fused-score margin",
        "trigger_threshold": threshold,
        "trigger_value": trigger_value,
        "triggered": triggered,
        "action_authorized": triggered,
        "authorized_action": protocol["policy"]["fixed_action"] if triggered else None,
        "authorized_query_key": protocol["policy"]["action_query_key"] if triggered else None,
        "initial_views_evaluation_only": rows,
        "initial_target_top1": sum(row["target_top1_evaluation_only"] for row in rows),
        "initial_target_recall_at_3": sum(row["target_recall_at_3_evaluation_only"] for row in rows),
        "initial_wrong_top1_count": sum(not row["target_top1_evaluation_only"] for row in rows),
        "next_action": protocol["next_action"]["triggered" if triggered else "not_triggered"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
