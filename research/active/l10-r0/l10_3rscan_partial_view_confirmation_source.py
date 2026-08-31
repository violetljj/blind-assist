#!/usr/bin/env python3
"""Freeze maximum-visible anti-sliver views for the third 3RScan family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_3rscan_multiview_observation_portfolio_posthoc as portfolio  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-partial-view-confirmation-source-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-partial-view-confirmation-source-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    candidate_path = HERE / protocol["candidate"]["path"]
    pixel.require(pixel.sha256(candidate_path) == protocol["candidate"]["sha256"], "CANDIDATE_HASH")
    candidate_receipt = pixel.load_json(candidate_path)
    candidate = candidate_receipt["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    target_id = int(candidate["target_instance_id"])
    reference_portfolio = portfolio._portfolio(
        data_root, str(candidate["reference_scan_id"]), target_id, protocol["partial_view_rules"], 1
    )
    query_portfolio = portfolio._portfolio(
        data_root, str(candidate["rescan_id"]), target_id, protocol["partial_view_rules"], 1
    )
    reference = reference_portfolio["selected"][0] if reference_portfolio["selected"] else None
    query = query_portfolio["selected"][0] if query_portfolio["selected"] else None
    evaluable = reference is not None and query is not None
    conclusion = (
        "L10_3RSCAN_PARTIAL_VIEW_CONFIRMATION_SOURCE_EVALUABLE"
        if evaluable
        else "L10_3RSCAN_PARTIAL_VIEW_CONFIRMATION_SOURCE_NOT_EVALUABLE"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_NEW_FAMILY_PRE_RGB_PRE_MODEL_PARTIAL_VIEW_SOURCE_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "conclusion": conclusion,
        "source_evaluable": evaluable,
        "reference": reference,
        "query": query,
        "reference_diagnostic": reference_portfolio,
        "query_diagnostic": query_portfolio,
        "rgb_members_opened": 0,
        "model_calls": 0,
        "next_action": protocol["next_action"]["evaluable" if evaluable else "not_evaluable"],
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
