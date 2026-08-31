#!/usr/bin/env python3
"""Direct physical-target-disjoint confirmation of the four-surface state."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path


import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_dual_surface_confirmation as confirmation  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-four-surface-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-four-surface-confirmation-result-v1"


@contextmanager
def four_surface_context():
    saved_schema = confirmation.PROTOCOL_SCHEMA
    saved_result = confirmation.RESULT_SCHEMA
    saved_file = confirmation.__file__
    confirmation.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    confirmation.RESULT_SCHEMA = RESULT_SCHEMA
    confirmation.__file__ = str(Path(__file__).resolve())
    try:
        yield
    finally:
        confirmation.PROTOCOL_SCHEMA = saved_schema
        confirmation.RESULT_SCHEMA = saved_result
        confirmation.__file__ = saved_file


def replay(protocol_path: Path, output_path: Path) -> None:
    with four_surface_context():
        confirmation.replay(protocol_path, output_path)
    result = base.load_json(output_path)
    minimum_support_iou = min(
        float(row["target_bbox_iou_evaluation_only"])
        for row in result["proposal_receipts"].values()
    )
    result["authority"] = "PHYSICAL_TARGET_DISJOINT_SAME_PROVIDER_FOUR_SURFACE_CONFIRMATION_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_3RSCAN_ROMA_CYCLE_PROMPT_FOUR_SURFACE_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_MET"
        if result["gate_met"]
        else "L10_3RSCAN_ROMA_CYCLE_PROMPT_FOUR_SURFACE_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_NOT_MET"
    )
    result["metrics"]["minimum_query_extent_target_bbox_iou"] = result["metrics"].pop(
        "minimum_prompt_target_bbox_iou"
    )
    result["metrics"]["minimum_query_support_bbox_iou_diagnostic_only"] = minimum_support_iou
    result["representation"] = {
        "reference_identity_support": "native_reference_sam2_mask",
        "reference_geometric_extent": "privileged_initial_reference_bbox",
        "query_identity_support": "native_query_sam2_mask",
        "query_geometric_extent": "cycle_affine_transported_bbox",
    }
    base.roma_base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
