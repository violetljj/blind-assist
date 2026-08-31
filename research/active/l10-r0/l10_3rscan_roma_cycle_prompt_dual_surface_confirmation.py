#!/usr/bin/env python3
"""Physical-target-disjoint confirmation of 3RScan dual-surface prompts."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-dual-surface-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-dual-surface-confirmation-result-v1"


@contextmanager
def confirmation_surface(
    reference_masks: dict[tuple[Any, ...], np.ndarray],
    extent_masks: dict[str, np.ndarray],
):
    with dual.dual_surface(reference_masks, extent_masks):
        base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
        base.RESULT_SCHEMA = RESULT_SCHEMA
        base.__file__ = str(Path(__file__).resolve())
        yield


def replay(protocol_path: Path, output_path: Path) -> None:
    with confirmation_surface({}, {}):
        protocol = base.load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    images, inputs = base.load_images(protocol, cohort)
    reference_masks, reference_receipts, masker_device = reference_base.make_reference_masks(
        protocol, cohort, images, inputs
    )
    extent_masks: dict[str, np.ndarray] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        key = f"{episode_id}:reference"
        image = images[key]
        bbox = inputs[key]["target_bbox_xyxy_evaluation_only"]
        lookup_key = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
        support_mask = reference_masks[lookup_key]
        extent_masks[base.mask_sha256(support_mask)] = base.rectangle_mask(image.size, bbox)
    with confirmation_surface(reference_masks, extent_masks):
        base.replay(protocol_path, output_path)
    result = base.load_json(output_path)
    result["authority"] = "PHYSICAL_TARGET_DISJOINT_SAME_PROVIDER_DUAL_SURFACE_CONFIRMATION_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_3RSCAN_ROMA_CYCLE_PROMPT_DUAL_SURFACE_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_MET"
        if result["gate_met"]
        else "L10_3RSCAN_ROMA_CYCLE_PROMPT_DUAL_SURFACE_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_NOT_MET"
    )
    result["reference_proposal_receipts"] = reference_receipts
    result["runtime"]["sam2_calls"] = int(result["runtime"]["sam2_calls"]) + len(reference_receipts)
    result["runtime"]["reference_masker_device"] = masker_device
    base.roma_base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
