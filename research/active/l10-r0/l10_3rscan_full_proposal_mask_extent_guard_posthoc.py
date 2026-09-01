#!/usr/bin/env python3
"""Audit a full-proposal mask-extent guard on consumed scale-prior receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-full-proposal-mask-extent-guard-posthoc-protocol-v1"


def _area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    input_path = HERE / protocol["input"]["path"]
    pixel.require(pixel.sha256(input_path) == protocol["input"]["sha256"], "INPUT_HASH")
    source = pixel.load_json(input_path)
    minimum_retention = float(protocol["selection"]["minimum_mask_extent_retention"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    family_receipts: dict[str, Any] = {}
    all_ious: list[float] = []
    successes = 0
    guarded = 0
    for family, family_row in source["families"].items():
        receipts: dict[str, Any] = {}
        family_ious: list[float] = []
        family_successes = 0
        for query, row in family_row["query_receipts"].items():
            proposal_box = row["selected"]["box_xyxy"]
            mask_box = row["refined_mask_bbox_xyxy"]
            retention = _area(mask_box) / _area(proposal_box)
            apply_guard = row["selected"]["tile_id"] == "full" and retention < minimum_retention
            evaluation = (
                row["selected"]["target_metrics_evaluation_only"]
                if apply_guard else
                row["refined_mask_bbox_target_metrics_evaluation_only"]
            )
            final_box = proposal_box if apply_guard else mask_box
            iou = float(evaluation["iou"])
            all_ious.append(iou)
            family_ious.append(iou)
            successes += int(iou >= minimum_iou)
            family_successes += int(iou >= minimum_iou)
            guarded += int(apply_guard)
            receipts[query] = {
                "query_truth_used_for_selection": False,
                "proposal_tile_id": row["selected"]["tile_id"],
                "mask_extent_retention": retention,
                "guard_applied": apply_guard,
                "final_box_source": "FULL_PROPOSAL" if apply_guard else "SAM_MASK_BBOX",
                "final_box_xyxy": final_box,
                "final_target_metrics_evaluation_only": evaluation,
            }
        family_receipts[family] = {
            "metrics": {
                "query_count": len(receipts),
                "iou_gate_queries": family_successes,
                "minimum_iou": min(family_ious),
                "mean_iou": sum(family_ious) / len(family_ious),
            },
            "query_receipts": receipts,
        }
    required = int(protocol["decision_gate"]["required_queries"])
    gate_met = successes == required
    result = {
        "schema": "blindassist-l10-3rscan-full-proposal-mask-extent-guard-posthoc-result-v1",
        "authority": "CONSUMED_THREE_FAMILY_FULL_PROPOSAL_MASK_EXTENT_GUARD_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_FULL_PROPOSAL_MASK_EXTENT_GUARD_THREE_FAMILY_DEVELOPMENT_GATE_MET"
            if gate_met else
            "L10_3RSCAN_FULL_PROPOSAL_MASK_EXTENT_GUARD_THREE_FAMILY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "selection": protocol["selection"],
        "metrics": {
            "family_count": len(family_receipts),
            "query_count": len(all_ious),
            "iou_gate_queries": successes,
            "guard_applied_queries": guarded,
            "minimum_iou": min(all_ious),
            "mean_iou": sum(all_ious) / len(all_ious),
        },
        "families": family_receipts,
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
