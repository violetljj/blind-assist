#!/usr/bin/env python3
"""Freeze target memory views and a same-scene sibling before descriptor confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_joint_covisibility_selector_posthoc as views  # noqa: E402
import l10_3rscan_multiview_observation_portfolio_posthoc as portfolio  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-set-memory-confirmation-source-protocol-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    candidate_row = protocol["candidate"]
    pixel.require(pixel.sha256(HERE / candidate_row["path"]) == candidate_row["sha256"], "CANDIDATE_HASH")
    candidate = pixel.load_json(HERE / candidate_row["path"])["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    data_root = artifact_root / "datasets/3rscan"
    target_id = int(candidate["target_instance_id"])
    budget = int(protocol["memory_budget"])
    reference = portfolio._portfolio(
        data_root, str(candidate["reference_scan_id"]), target_id,
        protocol["candidate_view_rules"], budget,
    )
    query = portfolio._portfolio(
        data_root, str(candidate["rescan_id"]), target_id,
        protocol["candidate_view_rules"], budget,
    )
    sibling = None
    for sibling_id in candidate["rescan_door_instance_ids"]:
        sibling_id = int(sibling_id)
        if sibling_id == target_id:
            continue
        points, candidates, opened = views._candidates(
            data_root, str(candidate["rescan_id"]), sibling_id, protocol["candidate_view_rules"]
        )
        if not candidates:
            continue
        selected = max(
            (row for row, _, _ in candidates),
            key=lambda row: (
                int(row["visible_target_vertices"]), float(row["bbox_short_side_fraction"]),
                float(row["depth_visible_ratio"]), -int(row["frame"]),
            ),
        )
        sibling = {
            "instance_id": sibling_id,
            "label": "door_or_doorframe",
            "target_vertices": int(len(points)),
            "candidate_views": len(candidates),
            "selected": selected,
            "opened": opened,
        }
        break
    minimum = int(protocol["decision_gate"]["minimum_memory_views_per_side"])
    evaluable = len(reference["selected"]) >= minimum and len(query["selected"]) >= minimum and sibling is not None
    result = {
        "schema": "blindassist-l10-3rscan-set-memory-confirmation-source-result-v1",
        "authority": "FROZEN_NEW_FAMILY_PRE_RGB_PRE_MODEL_SET_MEMORY_CONFIRMATION_SOURCE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "conclusion": (
            "L10_3RSCAN_SET_MEMORY_CONFIRMATION_SOURCE_EVALUABLE"
            if evaluable else "L10_3RSCAN_SET_MEMORY_CONFIRMATION_SOURCE_NOT_EVALUABLE"
        ),
        "source_evaluable": evaluable,
        "reference_memory": reference,
        "query_memory": query,
        "same_scene_sibling": sibling,
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
