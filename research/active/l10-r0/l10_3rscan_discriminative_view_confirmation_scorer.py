#!/usr/bin/env python3
"""Run the unchanged scorer on the fresh active-view confirmation cohort."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_foundpose_layer18_nids_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-confirmation-scorer-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-discriminative-view-confirmation-scorer-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    base_row = protocol["base_implementation"]
    pixel.require(pixel.sha256(HERE / base_row["path"]) == base_row["sha256"], "BASE_IMPLEMENTATION_HASH")

    compatibility_protocol = deepcopy(protocol)
    compatibility_protocol["schema"] = base.PROTOCOL_SCHEMA
    compatibility_protocol["implementation"] = base_row
    temporary_protocol = output_path.with_name(f".{output_path.name}.base-protocol.tmp.json")
    temporary_result = output_path.with_name(f".{output_path.name}.base-result.tmp.json")
    try:
        pixel.atomic_write_json(temporary_protocol, compatibility_protocol)
        base.run(temporary_protocol, temporary_result)
        result = pixel.load_json(temporary_result)
        base_conclusion = result["conclusion"]
        result.update(
            {
                "schema": RESULT_SCHEMA,
                "authority": "FRESH_PRE_FROZEN_QUEUE_UNCHANGED_SCORER_ACTIVE_VIEW_CONFIRMATION",
                "protocol_path": protocol_path.name,
                "protocol_sha256": pixel.sha256(protocol_path),
                "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
                "base_implementation": base_row,
                "base_conclusion": base_conclusion,
                "conclusion": "L10_3RSCAN_DISCRIMINATIVE_VIEW_CONFIRMATION_SCORER_MATERIALIZED",
                "claim_boundary": protocol["claim_boundary"],
            }
        )
        pixel.atomic_write_json(output_path, result)
        print(json.dumps(result, indent=2))
    finally:
        temporary_protocol.unlink(missing_ok=True)
        temporary_result.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
