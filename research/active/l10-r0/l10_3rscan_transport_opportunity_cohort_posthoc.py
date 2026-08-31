#!/usr/bin/env python3
"""Materialize the consumed PV28 viewpoint-opportunity pair with fixed controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-transport-opportunity-cohort-posthoc-protocol-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("base_cohort", "selection_result"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    cohort = deepcopy(pixel.load_json(HERE / protocol["base_cohort"]["path"]))
    selection = pixel.load_json(HERE / protocol["selection_result"]["path"])["selected"]
    episode = next(row for row in cohort["episodes"] if row["episode_id"] == "PV28")
    pixel.require(int(selection["reference"]["frame"]) == int(episode["reference"]["frame"]), "REFERENCE_CHANGED")
    episode["query"] = deepcopy(selection["query"])
    episode["selection_authority"] = "CONSUMED_GEOMETRY_DEPTH_JOINT_SURFACE_TIMES_VIEW_DIRECTION_OPPORTUNITY"
    image = cohort["images"]["PV28_query"]
    image["frame"] = int(selection["query"]["frame"])
    image["color_size"] = selection["query"]["color_size"]
    image["bbox_xyxy"] = selection["query"]["bbox_xyxy"]
    image["zip_member"] = selection["query"]["zip_member"]
    cohort["schema"] = "blindassist-l10-3rscan-transport-opportunity-cohort-posthoc-v1"
    cohort["authority"] = "CONSUMED_SOURCE_PRE_MODEL_TRANSPORT_OPPORTUNITY_POSTHOC_COHORT"
    cohort["protocol_path"] = protocol_path.name
    cohort["protocol_sha256"] = pixel.sha256(protocol_path)
    cohort["implementation"] = {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))}
    cohort["selection"] = {
        "episode_id": "PV28",
        "reference_frame": int(selection["reference"]["frame"]),
        "query_frame": int(selection["query"]["frame"]),
        "transport_opportunity_score": float(selection["transport_opportunity_score"]),
        "joint_visible_surface_fraction": float(selection["joint_visible_surface_fraction"]),
        "view_direction_cosine": float(selection["view_direction_cosine"]),
        "opened_rgb_members_before_selection": 0,
        "model_calls_before_selection": 0
    }
    cohort["claim_boundary"] = protocol["claim_boundary"]
    pixel.atomic_write_json(output_path, cohort)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
