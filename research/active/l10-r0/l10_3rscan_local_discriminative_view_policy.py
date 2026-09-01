#!/usr/bin/env python3
"""Summarize the frozen truth-free trigger and fixed next-frame action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-local-discriminative-view-policy-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-local-discriminative-view-policy-result-v1"


def _view(row: dict[str, Any], minimum_iou: float) -> dict[str, Any]:
    candidates = row["ranked_candidates"]
    first_target_rank = next(
        (
            index
            for index, candidate in enumerate(candidates, start=1)
            if float(candidate["target_metrics_evaluation_only"]["iou"]) >= minimum_iou
        ),
        None,
    )
    return {
        "top1_minus_top2_fused_margin": float(row["top1_layer18_nids_fused_margin"]),
        "top1_target_iou_evaluation_only": float(row["selected"]["target_metrics_evaluation_only"]["iou"]),
        "refined_top1_target_iou_evaluation_only": float(row["refined_mask_bbox_target_metrics_evaluation_only"]["iou"]),
        "best_reachable_target_iou_evaluation_only": float(row["best_reachable_target_iou_evaluation_only"]),
        "first_target_proposal_rank_evaluation_only": first_target_rank,
        "target_top1_evaluation_only": first_target_rank == 1,
        "target_recall_at_3_evaluation_only": first_target_rank is not None and first_target_rank <= 3,
        "candidate_count": len(candidates),
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    scorer_row = protocol["scorer_result"]
    scorer_path = HERE / scorer_row["path"]
    pixel.require(pixel.sha256(scorer_path) == scorer_row["sha256"], "SCORER_HASH")
    scorer = pixel.load_json(scorer_path)
    pixel.require(scorer["conclusion"] == scorer_row["required_conclusion"], "SCORER_CONCLUSION")

    minimum_iou = float(protocol["evaluation"]["minimum_iou"])
    initial_keys = protocol["policy"]["initial_query_keys"]
    action_key = protocol["policy"]["action_query_key"]
    views = {
        key: _view(scorer["query_receipts"][key], minimum_iou)
        for key in [*initial_keys, action_key]
    }
    initial = [views[key] for key in initial_keys]
    action = views[action_key]
    initial_margins = [row["top1_minus_top2_fused_margin"] for row in initial]
    threshold = float(protocol["policy"]["truth_free_trigger_threshold"])
    trigger_value = min(initial_margins)
    triggered = trigger_value <= threshold
    action_margin = action["top1_minus_top2_fused_margin"]
    margin_gain_over_trigger = action_margin - trigger_value
    initial_top1_ious = [row["top1_target_iou_evaluation_only"] for row in initial]
    initial_refined_ious = [row["refined_top1_target_iou_evaluation_only"] for row in initial]
    wrong_initial = sum(not row["target_top1_evaluation_only"] for row in initial)
    observed = (
        triggered
        and action["target_top1_evaluation_only"]
        and action_margin > max(initial_margins)
        and action["top1_target_iou_evaluation_only"] > initial[-1]["top1_target_iou_evaluation_only"]
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "POSTHOC_SUMMARY_OF_FROZEN_LOCAL_ACTIVE_VIEW_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_LOCAL_DISCRIMINATIVE_VIEW_TRIGGERED_MARGIN_SHARPENING_DEVELOPMENT_OBSERVED"
            if observed
            else "L10_3RSCAN_LOCAL_DISCRIMINATIVE_VIEW_TRIGGERED_MARGIN_SHARPENING_DEVELOPMENT_NOT_OBSERVED"
        ),
        "observed": observed,
        "policy": {
            "trigger_uses_target_truth": False,
            "trigger_statistic": "minimum initial Top-1-minus-Top-2 fused-score margin",
            "trigger_threshold": threshold,
            "trigger_value": trigger_value,
            "triggered": triggered,
            "fixed_action": "NEXT_FRAME_FORWARD",
            "action_executed": triggered,
        },
        "metrics": {
            "initial_query_count": len(initial),
            "initial_target_top1": sum(row["target_top1_evaluation_only"] for row in initial),
            "initial_target_recall_at_3": sum(row["target_recall_at_3_evaluation_only"] for row in initial),
            "initial_wrong_top1_count": wrong_initial,
            "initial_margins": initial_margins,
            "minimum_initial_margin": min(initial_margins),
            "mean_initial_margin": float(np.mean(initial_margins)),
            "maximum_initial_margin": max(initial_margins),
            "action_target_top1": int(action["target_top1_evaluation_only"]),
            "action_target_recall_at_3": int(action["target_recall_at_3_evaluation_only"]),
            "action_margin": action_margin,
            "action_margin_gain_over_trigger_view": margin_gain_over_trigger,
            "action_margin_relative_gain_over_trigger_view": margin_gain_over_trigger / trigger_value,
            "action_margin_gain_over_initial_mean": action_margin - float(np.mean(initial_margins)),
            "action_margin_gain_over_initial_maximum": action_margin - max(initial_margins),
            "initial_top1_target_ious": initial_top1_ious,
            "mean_initial_top1_target_iou": float(np.mean(initial_top1_ious)),
            "action_top1_target_iou": action["top1_target_iou_evaluation_only"],
            "action_top1_iou_gain_over_latest_initial": action["top1_target_iou_evaluation_only"] - initial[-1]["top1_target_iou_evaluation_only"],
            "mean_initial_refined_top1_target_iou": float(np.mean(initial_refined_ious)),
            "action_refined_top1_target_iou": action["refined_top1_target_iou_evaluation_only"],
            "wrong_top1_exclusion_fraction": "NOT_APPLICABLE_NO_INITIAL_WRONG_TOP1",
        },
        "views": views,
        "interpretation": protocol["interpretation"],
        "next_action": protocol["next_action"],
        "literature_motivation": protocol["literature_motivation"],
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
