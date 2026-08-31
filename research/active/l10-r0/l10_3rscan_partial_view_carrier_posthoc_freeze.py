#!/usr/bin/env python3
"""Freeze the geometry-selected OA2 partial-view pair plus fixed controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-partial-view-carrier-posthoc-freeze-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-partial-view-carrier-posthoc-cohort-v1"


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("portfolio", "source_result", "controls"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    portfolio = pixel.load_json(HERE / protocol["portfolio"]["path"])
    panel = next(row for row in portfolio["panels"] if row["id"] == protocol["selection"]["panel_id"])
    source = pixel.load_json(HERE / protocol["source_result"]["path"])
    candidate = source["candidate"]
    reference = deepcopy(panel["reference"]["selected"][0])
    query = deepcopy(panel["query"]["selected"][0])
    episode_id = str(protocol["selection"]["episode_id"])
    episode = {
        "episode_id": episode_id,
        **candidate,
        "reference": reference,
        "query": query,
        "active_query": None,
        "selection_authority": "GEOMETRY_DEPTH_ONLY_MAXIMUM_NEW_VISIBLE_TARGET_VERTICES_WITH_ANTI_SLIVER_GATE",
    }
    images = {}
    for role, scan_key, frame in (
        ("reference", "reference_scan_id", reference),
        ("query", "rescan_id", query),
    ):
        images[f"{episode_id}_{role}"] = {
            "episode_id": episode_id,
            "role": role,
            "scan_id": candidate[scan_key],
            "target_instance_id": int(candidate["target_instance_id"]),
            "target_label": candidate["target_label"],
            "frame": int(frame["frame"]),
            "color_size": frame["color_size"],
            "bbox_xyxy": frame["bbox_xyxy"],
            "zip_member": frame["zip_member"],
        }

    controls = pixel.load_json(HERE / protocol["controls"]["path"])
    control_episodes = {str(row["episode_id"]): row for row in controls["episodes"]}
    control_images = {
        str(row["episode_id"]) + ":" + str(row["role"]): row for row in controls["images"].values()
    }
    episodes = [episode]
    for control in protocol["selection"]["negative_query_controls"]:
        control_id = str(control["episode_id"])
        source_id = str(control["source_episode_id"])
        copied = deepcopy(control_episodes[source_id])
        copied["episode_id"] = control_id
        copied["negative_query_control_provenance"] = {
            "cohort_path": protocol["controls"]["path"],
            "source_episode_id": source_id,
            "pixels_consumed_for_another_target": True,
        }
        episodes.append(copied)
        for role in ("reference", "query"):
            image = deepcopy(control_images[f"{source_id}:{role}"])
            image["episode_id"] = control_id
            images[f"{control_id}_{role}"] = image

    output = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_CONSUMED_SOURCE_PRE_RGB_MODEL_FRESH_PARTIAL_VIEW_CARRIER_POSTHOC_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str((ROOT / protocol["source"]["artifact_root"]).resolve()),
        "source_manifest": protocol["source"]["sequence_zips"],
        "selection": {
            **protocol["selection"],
            "opened_rgb_members": 0,
            "model_calls": 0,
            "reference_visible_target_fraction": reference["cumulative_visible_target_fraction"],
            "query_visible_target_fraction": query["cumulative_visible_target_fraction"],
        },
        "episodes": episodes,
        "images": images,
        "evaluation": protocol["evaluation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.require(len(episodes) == 3 and len(images) == 6, "COHORT_CARDINALITY")
    pixel.atomic_write_json(output_path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
