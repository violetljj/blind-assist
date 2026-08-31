#!/usr/bin/env python3
"""Posthoc four-surface responsibility split for 3RScan target transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-four-surface-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-four-surface-posthoc-result-v1"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    base.require(protocol["implementation"]["sha256"] == base.sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    for row in protocol["frozen_inputs"]:
        source = HERE / row["path"]
        base.require(base.sha256(source) == row["sha256"], f"INPUT_HASH:{row['path']}")
    predecessor = base.load_json(HERE / protocol["predecessor"]["result_path"])
    base.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    return protocol


def replay(protocol_path: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    predecessor = base.load_json(predecessor_path)
    episode_ids = sorted(predecessor["prompt_receipts"])
    rows: dict[str, Any] = {}
    for episode_id in episode_ids:
        prompt = predecessor["prompt_receipts"][episode_id]
        query_support = predecessor["proposal_receipts"][episode_id]
        bilateral = predecessor["pair_support"][episode_id]
        rows[episode_id] = {
            "reference_identity_support": predecessor["reference_proposal_receipts"][episode_id],
            "reference_geometric_extent_xyxy": predecessor["input_receipts"][f"{episode_id}:reference"]
            ["target_bbox_xyxy_evaluation_only"],
            "query_identity_support": query_support,
            "query_geometric_extent_xyxy": prompt["prompt_box_xyxy"],
            "query_extent_target_bbox_iou_evaluation_only": prompt["target_bbox_iou_evaluation_only"],
            "query_extent_target_bbox_recall_evaluation_only": prompt["target_bbox_recall_evaluation_only"],
            "query_support_bbox_iou_diagnostic_only": query_support["target_bbox_iou_evaluation_only"],
            "bilateral_absolute_support": bilateral["absolute_support"],
        }
    gate = protocol["decision_gate"]
    minimum_extent_iou = min(float(row["query_extent_target_bbox_iou_evaluation_only"]) for row in rows.values())
    supported = sum(bool(row["bilateral_absolute_support"]) for row in rows.values())
    gate_met = (
        len(rows) == int(gate["required_target_pairs"])
        and minimum_extent_iou >= float(gate["minimum_query_extent_target_bbox_iou"])
        and supported >= int(gate["minimum_bilateral_absolute_support"])
    )
    base.roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_FOUR_SURFACE_RESPONSIBILITY_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "predecessor_result_path": predecessor_path.name,
            "predecessor_result_sha256": base.sha256(predecessor_path),
            "conclusion": (
                "L10_3RSCAN_ROMA_CYCLE_PROMPT_FOUR_SURFACE_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met else "L10_3RSCAN_ROMA_CYCLE_PROMPT_FOUR_SURFACE_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "target_pairs": len(rows),
                "minimum_query_extent_target_bbox_iou": minimum_extent_iou,
                "bilateral_absolute_support": supported,
                "minimum_query_support_bbox_iou_diagnostic_only": min(
                    float(row["query_support_bbox_iou_diagnostic_only"]) for row in rows.values()
                ),
            },
            "pairs": rows,
            "model_calls": 0,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
