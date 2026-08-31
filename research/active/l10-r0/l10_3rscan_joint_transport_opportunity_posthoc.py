#!/usr/bin/env python3
"""Rerank consumed PV28 view pairs by overlap times viewpoint agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-joint-transport-opportunity-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-joint-transport-opportunity-posthoc-result-v1"


def _score(row: dict) -> float:
    return float(row["joint_visible_surface_fraction"]) * max(0.0, float(row["view_direction_cosine"]))


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("joint_diagnostic", "matcher_challenge"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    joint = pixel.load_json(HERE / protocol["joint_diagnostic"]["path"])
    rows = joint["top_pairs"]
    pixel.require(len(rows) > 0, "NO_RANKED_PAIRS")
    ranked = sorted(rows, key=lambda row: (_score(row), -int(row["reference"]["frame"]), -int(row["query"]["frame"])), reverse=True)
    selected = {**ranked[0], "transport_opportunity_score": _score(ranked[0])}
    baseline = {**joint["independent_maximum_visible_baseline"], "transport_opportunity_score": _score(joint["independent_maximum_visible_baseline"])}
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SOURCE_GEOMETRY_DEPTH_ONLY_VIEWPOINT_TRANSPORT_OPPORTUNITY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": "L10_3RSCAN_JOINT_TRANSPORT_OPPORTUNITY_POSTHOC_DIAGNOSTIC_ONLY",
        "score": protocol["score"],
        "baseline": baseline,
        "selected": selected,
        "selected_minus_baseline_score": float(selected["transport_opportunity_score"] - baseline["transport_opportunity_score"]),
        "rgb_members_opened": 0,
        "model_calls": 0,
        "literature_motivation": protocol["literature_motivation"],
        "next_action": protocol["next_action"],
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
