#!/usr/bin/env python3
"""Append one geometry-frozen same-scene sibling query to a memory cohort."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-sibling-negative-freeze-protocol-v1"


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("cohort", "sibling_source"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    cohort = deepcopy(pixel.load_json(HERE / protocol["cohort"]["path"]))
    source = pixel.load_json(HERE / protocol["sibling_source"]["path"])
    pixel.require(source["conclusion"] == "L10_3RSCAN_SAME_SCENE_SIBLING_NEGATIVE_SOURCE_EVALUABLE", "SOURCE_CONCLUSION")
    episode_id = str(protocol["episode_id"])
    view = deepcopy(source["selected"])
    cohort["episodes"].append({
        "episode_id": episode_id,
        "reference_scan_id": source["scan_id"],
        "rescan_id": source["scan_id"],
        "target_instance_id": int(source["sibling_instance_id"]),
        "target_label": source["sibling_label"],
        "reference": None,
        "query": view,
        "negative_authority": "SAME_RESCAN_DIFFERENT_INSTANCE_ID_FROM_PV28",
    })
    cohort["images"][f"{episode_id}_query"] = {
        "episode_id": episode_id,
        "role": "query",
        "scan_id": source["scan_id"],
        "target_instance_id": int(source["sibling_instance_id"]),
        "target_label": source["sibling_label"],
        "frame": int(view["frame"]),
        "color_size": view["color_size"],
        "bbox_xyxy": view["bbox_xyxy"],
        "zip_member": view["zip_member"],
    }
    cohort["schema"] = "blindassist-l10-3rscan-multiview-memory-sibling-negative-cohort-v1"
    cohort["authority"] = "CONSUMED_PV28_MULTIVIEW_MEMORY_WITH_PRE_DESCRIPTOR_SAME_SCENE_SIBLING_NEGATIVE"
    cohort["protocol_path"] = protocol_path.name
    cohort["protocol_sha256"] = pixel.sha256(protocol_path)
    cohort["implementation"] = {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))}
    cohort["sibling_negative"] = {
        "episode_id": episode_id,
        "source_path": protocol["sibling_source"]["path"],
        "source_sha256": protocol["sibling_source"]["sha256"],
        "rgb_members_opened_before_freeze": 0,
        "model_calls_before_freeze": 0,
    }
    cohort["claim_boundary"] = protocol["claim_boundary"]
    pixel.require(len(cohort["episodes"]) == 6 and len(cohort["images"]) == 11, "COHORT_CARDINALITY")
    pixel.atomic_write_json(output_path, cohort)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
