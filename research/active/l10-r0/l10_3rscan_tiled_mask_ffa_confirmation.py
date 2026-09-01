#!/usr/bin/env python3
"""Confirm the frozen tiled target-only FFA chain on a new family."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_mask_ffa_target_only_confirmation as confirmation  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_tiled_mask_ffa_posthoc as tiled  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-tiled-mask-ffa-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-tiled-mask-ffa-confirmation-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    original_proposals = base._objectness_proposals
    original_file = confirmation.__file__
    original_protocol_schema = confirmation.PROTOCOL_SCHEMA
    original_result_schema = confirmation.RESULT_SCHEMA
    try:
        base._objectness_proposals = tiled._tiled_proposals
        confirmation.__file__ = __file__
        confirmation.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
        confirmation.RESULT_SCHEMA = RESULT_SCHEMA
        confirmation.run(protocol_path, output_path)
    finally:
        base._objectness_proposals = original_proposals
        confirmation.__file__ = original_file
        confirmation.PROTOCOL_SCHEMA = original_protocol_schema
        confirmation.RESULT_SCHEMA = original_result_schema
    result = pixel.load_json(output_path)
    result["authority"] = "FROZEN_PRE_RGB_PRE_MODEL_NEW_FAMILY_SAME_PROVIDER_TILED_TARGET_ONLY_FFA_CONFIRMATION"
    result["conclusion"] = (
        "L10_3RSCAN_TILED_MASK_FFA_NEW_FAMILY_CONFIRMATION_GATE_MET"
        if result["gate_met"] else "L10_3RSCAN_TILED_MASK_FFA_NEW_FAMILY_CONFIRMATION_GATE_NOT_MET"
    )
    result["tiling"] = pixel.load_json(protocol_path)["tiling"]
    pixel.atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
