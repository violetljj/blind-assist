#!/usr/bin/env python3
"""Spatially coherent cycle-component support for 3RScan target transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cycle-component-support-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cycle-component-support-posthoc-result-v1"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    base.require(protocol["implementation"]["sha256"] == base.sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    for row in protocol["frozen_inputs"]:
        base.require(base.sha256(HERE / row["path"]) == row["sha256"], f"INPUT_HASH:{row['path']}")
    predecessor = base.load_json(HERE / protocol["predecessor"]["result_path"])
    base.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    return protocol


def replay(protocol_path: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    predecessor = base.load_json(predecessor_path)
    gate = protocol["decision_gate"]
    rows: dict[str, Any] = {}
    for episode_id, prompt in sorted(predecessor["prompt_receipts"].items()):
        cycle_fraction = float(prompt["all_cycle_fraction"])
        component_dominance = float(prompt["selected_component_fraction_of_cycles"])
        extent_iou = float(prompt["target_bbox_iou_evaluation_only"])
        coherent = (
            cycle_fraction >= float(gate["minimum_reference_cycle_fraction"])
            and component_dominance >= float(gate["minimum_dominant_component_cycle_fraction"])
        )
        rows[episode_id] = {
            "query_extent_target_bbox_iou_evaluation_only": extent_iou,
            "reference_cycle_fraction": cycle_fraction,
            "dominant_component_cycle_fraction": component_dominance,
            "dominant_component_pixels": int(prompt["component"]["selected_pixels"]),
            "affine_mean_residual_normalized": prompt["affine_mean_residual_normalized"],
            "coherent_component_support": coherent,
            "legacy_bilateral_global_purity_support_diagnostic_only": predecessor["pair_support"][episode_id]["absolute_support"],
            "query_sam_support_bbox_iou_diagnostic_only": predecessor["proposal_receipts"][episode_id]["target_bbox_iou_evaluation_only"],
        }
    minimum_extent_iou = min(row["query_extent_target_bbox_iou_evaluation_only"] for row in rows.values())
    supported = sum(bool(row["coherent_component_support"]) for row in rows.values())
    gate_met = (
        len(rows) == int(gate["required_target_pairs"])
        and minimum_extent_iou >= float(gate["minimum_query_extent_target_bbox_iou"])
        and supported >= int(gate["minimum_coherent_component_support"])
    )
    base.roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_SPATIALLY_COHERENT_CYCLE_COMPONENT_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "predecessor_result_path": predecessor_path.name,
            "predecessor_result_sha256": base.sha256(predecessor_path),
            "conclusion": (
                "L10_3RSCAN_CYCLE_COMPONENT_SUPPORT_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met else "L10_3RSCAN_CYCLE_COMPONENT_SUPPORT_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "target_pairs": len(rows),
                "minimum_query_extent_target_bbox_iou": minimum_extent_iou,
                "coherent_component_support": supported,
                "minimum_reference_cycle_fraction": min(row["reference_cycle_fraction"] for row in rows.values()),
                "minimum_dominant_component_cycle_fraction": min(row["dominant_component_cycle_fraction"] for row in rows.values()),
                "legacy_bilateral_global_purity_support": sum(bool(row["legacy_bilateral_global_purity_support_diagnostic_only"]) for row in rows.values()),
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
