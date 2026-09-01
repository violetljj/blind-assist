#!/usr/bin/env python3
"""Test a smaller fixed corner-tile scale for D15 proposal reachability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_mask_ffa_memory_posthoc as ffa  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_tiled_mask_ffa_posthoc as tiled  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-small-tile-proposal-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-small-tile-proposal-posthoc-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    predecessor_path = HERE / protocol["predecessor"]["path"]
    pixel.require(pixel.sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    pixel.require(pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"], "PREDECESSOR_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    proposal = protocol["proposal"]
    pixel.require(pixel.sha256(Path(__file__).resolve().parents[3] / proposal["model_path"]) == proposal["model_sha256"], "PROPOSAL_MODEL_HASH")
    cohort = pixel.load_json(cohort_path)
    images, image_rows = ffa._load_images(protocol, cohort)
    proposals, runtime = tiled._tiled_proposals(protocol, images)
    minimum = float(protocol["decision_gate"]["minimum_iou"])
    opportunities = 0
    receipts = {}
    for key in protocol["evaluation"]["query_images"]:
        truth = image_rows[key]["bbox_xyxy"]
        rows = [
            {**row, "target_metrics_evaluation_only": base._bbox_metrics(row["box_xyxy"], truth)}
            for row in proposals[key]
        ]
        rows.sort(key=lambda row: (-row["target_metrics_evaluation_only"]["iou"], -row["objectness_score"]))
        best = rows[0]
        opportunity = float(best["target_metrics_evaluation_only"]["iou"]) >= minimum
        opportunities += int(opportunity)
        receipts[key] = {
            "query_truth_used_for_proposal_generation": False,
            "opportunity_at_iou_gate": opportunity,
            "best": best,
            "proposal_count": len(rows),
        }
    gate_met = opportunities == int(protocol["decision_gate"]["required_opportunity_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D15_SMALL_FIXED_TILE_PROPOSAL_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_SMALL_TILE_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_SMALL_TILE_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {"query_count": len(receipts), "opportunity_queries": opportunities},
        "query_receipts": receipts,
        "runtime": runtime,
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
