#!/usr/bin/env python3
"""Evaluate the preregistered tiered outcome after one authorized action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-discriminative-view-confirmation-result-v1"


def _load_hashed(row: dict[str, Any], label: str) -> dict[str, Any]:
    path = HERE / row["path"]
    pixel.require(pixel.sha256(path) == row["sha256"], f"{label}_HASH")
    value = pixel.load_json(path)
    if "required_conclusion" in row:
        pixel.require(value["conclusion"] == row["required_conclusion"], f"{label}_CONCLUSION")
    return value


def _rank(receipt: dict[str, Any], minimum_iou: float) -> int | None:
    return next(
        (
            index
            for index, candidate in enumerate(receipt["ranked_candidates"], start=1)
            if float(candidate["target_metrics_evaluation_only"]["iou"]) >= minimum_iou
        ),
        None,
    )


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    trigger = _load_hashed(protocol["trigger_result"], "TRIGGER_RESULT")
    initial = _load_hashed(protocol["initial_scorer_result"], "INITIAL_SCORER_RESULT")
    action_protocol_row = protocol["action_scorer_protocol"]
    pixel.require(pixel.sha256(HERE / action_protocol_row["path"]) == action_protocol_row["sha256"], "ACTION_PROTOCOL_HASH")
    action_row = protocol["action_scorer_result"]
    action = pixel.load_json(HERE / action_row["path"])
    pixel.require(action["schema"] == action_row["required_schema"], "ACTION_RESULT_SCHEMA")
    pixel.require(action["conclusion"] == action_row["required_conclusion"], "ACTION_RESULT_CONCLUSION")
    pixel.require(action["protocol_sha256"] == action_protocol_row["sha256"], "ACTION_PROTOCOL_RECEIPT")
    pixel.require(bool(trigger["action_authorized"]), "ACTION_NOT_AUTHORIZED")

    minimum_iou = float(protocol["evaluation"]["minimum_iou"])
    action_key = protocol["policy"]["action_query_key"]
    action_receipt = action["query_receipts"][action_key]
    action_rank = _rank(action_receipt, minimum_iou)
    action_margin = float(action_receipt["top1_layer18_nids_fused_margin"])
    action_iou = float(action_receipt["selected"]["target_metrics_evaluation_only"]["iou"])
    action_refined_iou = float(action_receipt["refined_mask_bbox_target_metrics_evaluation_only"]["iou"])
    initial_rows = trigger["initial_views_evaluation_only"]
    initial_margins = [float(row["top1_minus_top2_fused_margin"]) for row in initial_rows]
    initial_wrong = int(trigger["initial_wrong_top1_count"])
    initial_recall3 = int(trigger["initial_target_recall_at_3"])

    wrong_recovery_evaluable = initial_wrong > 0
    wrong_recovery_met = (
        wrong_recovery_evaluable
        and initial_recall3 == len(initial_rows)
        and action_rank == 1
        and action_iou >= minimum_iou
        and action_margin > 0.0
    )
    margin_sharpening_evaluable = initial_wrong == 0
    margin_sharpening_met = (
        margin_sharpening_evaluable
        and action_rank == 1
        and action_margin > max(initial_margins)
        and action_iou > float(initial_rows[-1]["top1_target_iou_evaluation_only"])
    )
    if wrong_recovery_met:
        conclusion = "L10_3RSCAN_DISCRIMINATIVE_VIEW_WRONG_TOP1_RECOVERY_FRESH_CONFIRMATION_GATE_MET"
    elif margin_sharpening_met:
        conclusion = "L10_3RSCAN_DISCRIMINATIVE_VIEW_MARGIN_SHARPENING_FRESH_CONFIRMATION_GATE_MET"
    else:
        conclusion = "L10_3RSCAN_DISCRIMINATIVE_VIEW_FRESH_CONFIRMATION_GATE_NOT_MET"
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_PRE_FROZEN_QUEUE_TRUTH_FREE_TRIGGER_ONE_AUTHORIZED_ACTION_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": conclusion,
        "gate_met": wrong_recovery_met or margin_sharpening_met,
        "policy": {
            "trigger_uses_target_truth": False,
            "trigger_threshold": float(trigger["trigger_threshold"]),
            "trigger_value": float(trigger["trigger_value"]),
            "triggered": bool(trigger["triggered"]),
            "fixed_action": trigger["authorized_action"],
            "action_query_key": trigger["authorized_query_key"],
            "action_executed_after_authorization": True,
        },
        "tiered_gates": {
            "wrong_top1_recovery": {
                "evaluable": wrong_recovery_evaluable,
                "status": (
                    "GATE_MET"
                    if wrong_recovery_met
                    else "NOT_EVALUABLE_NO_INITIAL_WRONG_TOP1"
                    if not wrong_recovery_evaluable
                    else "GATE_NOT_MET"
                ),
                "gate_met": wrong_recovery_met,
            },
            "margin_sharpening": {
                "evaluable": margin_sharpening_evaluable,
                "status": "GATE_MET" if margin_sharpening_met else "GATE_NOT_MET",
                "gate_met": margin_sharpening_met,
            },
        },
        "metrics": {
            "initial_target_top1": int(trigger["initial_target_top1"]),
            "initial_target_recall_at_3": initial_recall3,
            "initial_wrong_top1_count": initial_wrong,
            "initial_margins": initial_margins,
            "minimum_initial_margin": min(initial_margins),
            "maximum_initial_margin": max(initial_margins),
            "action_first_target_rank_evaluation_only": action_rank,
            "action_target_top1": int(action_rank == 1),
            "action_target_recall_at_3": int(action_rank is not None and action_rank <= 3),
            "action_margin": action_margin,
            "action_margin_gain_over_trigger_view": action_margin - min(initial_margins),
            "action_margin_gain_over_initial_maximum": action_margin - max(initial_margins),
            "action_top1_target_iou": action_iou,
            "action_top1_iou_gain_over_latest_initial": action_iou - float(initial_rows[-1]["top1_target_iou_evaluation_only"]),
            "action_refined_top1_target_iou": action_refined_iou,
            "wrong_top1_exclusion_fraction": (
                1.0 if wrong_recovery_met else "NOT_APPLICABLE_NO_INITIAL_WRONG_TOP1" if not wrong_recovery_evaluable else 0.0
            ),
        },
        "initial_views_evaluation_only": initial_rows,
        "action_view_evaluation_only": {
            "query_key": action_key,
            "top1_minus_top2_fused_margin": action_margin,
            "first_target_proposal_rank": action_rank,
            "top1_target_iou": action_iou,
            "refined_top1_target_iou": action_refined_iou,
            "candidate_count": len(action_receipt["ranked_candidates"]),
        },
        "next_action": protocol["next_action"][
            "wrong_recovery_met" if wrong_recovery_met else "margin_sharpening_met" if margin_sharpening_met else "not_met"
        ],
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
