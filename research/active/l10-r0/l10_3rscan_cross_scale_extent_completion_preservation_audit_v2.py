#!/usr/bin/env python3
"""Audit extent completion with geometry and cross-scale appearance consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cross_scale_extent_completion_preservation_audit as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cross-scale-extent-completion-preservation-audit-protocol-v2"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = base.pixel.load_json(protocol_path)
    base.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    base.pixel.require(
        base.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    base_row = protocol["base_implementation"]
    base.pixel.require(
        base.pixel.sha256(HERE / base_row["path"]) == base_row["sha256"],
        "BASE_IMPLEMENTATION_HASH",
    )
    minimum_retention = float(protocol["selection"]["minimum_local_score_retention"])

    def select(
        candidates: list[dict[str, Any]], minimum_containment: float
    ) -> tuple[dict[str, Any], bool, float]:
        ranked = base._rank(candidates)
        anchor = ranked[0]
        if anchor["tile_id"] == "full":
            return anchor, False, 1.0
        completions = [
            (row, base._containment(anchor["box_xyxy"], row["box_xyxy"]))
            for row in ranked
            if row["tile_id"] == "full"
            and row["winning_target_reference"] == anchor["winning_target_reference"]
            and float(row["layer18_local_appearance_score"])
            >= minimum_retention * float(anchor["layer18_local_appearance_score"])
        ]
        completions = [row for row in completions if row[1] >= minimum_containment]
        if not completions:
            return anchor, False, 0.0
        return completions[0][0], True, float(completions[0][1])

    base._select = select
    proxy = dict(protocol)
    proxy["schema"] = base.PROTOCOL_SCHEMA
    proxy["implementation"] = base_row
    proxy_path = output_path.with_name(output_path.stem + ".proxy.tmp.json")
    proxy_result_path = output_path.with_name(output_path.stem + ".proxy-result.tmp.json")
    try:
        base.pixel.atomic_write_json(proxy_path, proxy)
        base.run(proxy_path, proxy_result_path)
        mechanical = base.pixel.load_json(proxy_result_path)
    finally:
        proxy_path.unlink(missing_ok=True)
        proxy_result_path.unlink(missing_ok=True)
    gate_met = bool(mechanical["gate_met"])
    result = {
        **mechanical,
        "schema": "blindassist-l10-3rscan-cross-scale-extent-completion-preservation-audit-result-v2",
        "authority": "CONSUMED_TWO_FAMILY_GEOMETRY_APPEARANCE_CONSENSUS_AUDIT_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": base.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_CROSS_SCALE_EXTENT_COMPLETION_V2_TWO_FAMILY_PRESERVATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_CROSS_SCALE_EXTENT_COMPLETION_V2_TWO_FAMILY_PRESERVATION_GATE_NOT_MET"
        ),
        "selection": protocol["selection"],
        "claim_boundary": protocol["claim_boundary"],
    }
    base.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
