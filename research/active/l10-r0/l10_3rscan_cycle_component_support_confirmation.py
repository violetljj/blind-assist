#!/usr/bin/env python3
"""Direct confirmation of spatially coherent cycle-component support."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_four_surface_confirmation as four  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cycle-component-support-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cycle-component-support-confirmation-result-v1"


@contextmanager
def confirmation_context():
    saved_schema = four.PROTOCOL_SCHEMA
    saved_result = four.RESULT_SCHEMA
    saved_file = four.__file__
    four.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    four.RESULT_SCHEMA = RESULT_SCHEMA
    four.__file__ = str(Path(__file__).resolve())
    try:
        yield
    finally:
        four.PROTOCOL_SCHEMA = saved_schema
        four.RESULT_SCHEMA = saved_result
        four.__file__ = saved_file


def replay(protocol_path: Path, output_path: Path) -> None:
    with confirmation_context():
        four.replay(protocol_path, output_path)
    protocol = base.load_json(protocol_path)
    result = base.load_json(output_path)
    gate = protocol["decision_gate"]
    coherent: dict[str, dict[str, float | int | bool]] = {}
    for episode_id, prompt in sorted(result["prompt_receipts"].items()):
        cycle_fraction = float(prompt["all_cycle_fraction"])
        dominance = float(prompt["selected_component_fraction_of_cycles"])
        supported = (
            cycle_fraction >= float(gate["minimum_reference_cycle_fraction"])
            and dominance >= float(gate["minimum_dominant_component_cycle_fraction"])
        )
        coherent[episode_id] = {
            "reference_cycle_fraction": cycle_fraction,
            "dominant_component_cycle_fraction": dominance,
            "dominant_component_pixels": int(prompt["component"]["selected_pixels"]),
            "affine_mean_residual_normalized": float(prompt["affine_mean_residual_normalized"]),
            "coherent_component_support": supported,
            "legacy_bilateral_global_purity_support_diagnostic_only": bool(
                result["pair_support"][episode_id]["absolute_support"]
            ),
        }
    minimum_extent_iou = min(
        float(row["target_bbox_iou_evaluation_only"])
        for row in result["prompt_receipts"].values()
    )
    supported_count = sum(bool(row["coherent_component_support"]) for row in coherent.values())
    gate_met = (
        len(coherent) == int(gate["required_prompt_boxes"])
        and len(result["proposal_receipts"]) == int(gate["required_sam_masks"])
        and minimum_extent_iou >= float(gate["minimum_prompt_target_bbox_iou"])
        and supported_count >= int(gate["minimum_coherent_component_support"])
    )
    result["authority"] = "PHYSICAL_TARGET_DISJOINT_SAME_PROVIDER_CYCLE_COMPONENT_SUPPORT_CONFIRMATION_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_3RSCAN_CYCLE_COMPONENT_SUPPORT_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_MET"
        if gate_met else "L10_3RSCAN_CYCLE_COMPONENT_SUPPORT_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_NOT_MET"
    )
    result["gate_met"] = gate_met
    result["metrics"] = {
        "prompt_boxes": len(result["prompt_receipts"]),
        "sam_support_masks": len(result["proposal_receipts"]),
        "minimum_query_extent_target_bbox_iou": minimum_extent_iou,
        "coherent_component_support": supported_count,
        "required_true_pairs": len(coherent),
        "minimum_reference_cycle_fraction": min(float(row["reference_cycle_fraction"]) for row in coherent.values()),
        "minimum_dominant_component_cycle_fraction": min(float(row["dominant_component_cycle_fraction"]) for row in coherent.values()),
        "legacy_bilateral_global_purity_support_diagnostic_only": sum(
            bool(row["legacy_bilateral_global_purity_support_diagnostic_only"])
            for row in coherent.values()
        ),
        "minimum_query_support_bbox_iou_diagnostic_only": min(
            float(row["target_bbox_iou_evaluation_only"])
            for row in result["proposal_receipts"].values()
        ),
    }
    result["coherent_component_support"] = coherent
    base.roma_base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
