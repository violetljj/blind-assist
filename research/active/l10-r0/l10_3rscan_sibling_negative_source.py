#!/usr/bin/env python3
"""Select a geometry-only same-scene sibling negative before descriptor access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_joint_covisibility_selector_posthoc as views  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-sibling-negative-source-protocol-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    candidate_row = protocol["candidate_source"]
    pixel.require(pixel.sha256(HERE / candidate_row["path"]) == candidate_row["sha256"], "CANDIDATE_HASH")
    candidate = pixel.load_json(HERE / candidate_row["path"])["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    scan_id = str(candidate["rescan_id"])
    sibling_id = int(protocol["sibling_instance_id"])
    points, candidates, opened = views._candidates(
        artifact_root / "datasets/3rscan", scan_id, sibling_id, protocol["candidate_view_rules"]
    )
    pixel.require(len(candidates) > 0, "NO_SIBLING_VIEW")
    selected = max(
        (row for row, _, _ in candidates),
        key=lambda row: (
            int(row["visible_target_vertices"]), float(row["bbox_short_side_fraction"]),
            float(row["depth_visible_ratio"]), -int(row["frame"]),
        ),
    )
    result = {
        "schema": "blindassist-l10-3rscan-sibling-negative-source-result-v1",
        "authority": "CONSUMED_PV28_SAME_SCENE_SIBLING_GEOMETRY_DEPTH_ONLY_SOURCE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "scan_id": scan_id,
        "sibling_instance_id": sibling_id,
        "sibling_label": protocol["sibling_label"],
        "target_vertices": int(len(points)),
        "candidate_views": len(candidates),
        "selected": selected,
        "opened": opened,
        "rgb_members_opened": 0,
        "model_calls": 0,
        "conclusion": "L10_3RSCAN_SAME_SCENE_SIBLING_NEGATIVE_SOURCE_EVALUABLE",
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
