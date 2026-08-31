#!/usr/bin/env python3
"""Freeze a new-source transport-opportunity pair with fixed controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-transport-opportunity-confirmation-freeze-protocol-v1"


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("source_result", "controls"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    source = pixel.load_json(HERE / protocol["source_result"]["path"])
    pixel.require(source["source_evaluable"] is True, "SOURCE_NOT_EVALUABLE")
    selected = source["selected"]
    candidate = source["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["sequence_zips"].values():
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    episode_id = str(protocol["selection"]["episode_id"])
    reference, query = deepcopy(selected["reference"]), deepcopy(selected["query"])
    episode = {
        "episode_id": episode_id,
        **candidate,
        "reference": reference,
        "query": query,
        "active_query": None,
        "selection_authority": "PRE_DOWNLOAD_FROZEN_NEW_FAMILY_JOINT_SURFACE_TIMES_VIEW_DIRECTION_OPPORTUNITY",
    }
    images = {}
    for role, scan_key, view in (("reference", "reference_scan_id", reference), ("query", "rescan_id", query)):
        images[f"{episode_id}_{role}"] = {
            "episode_id": episode_id,
            "role": role,
            "scan_id": candidate[scan_key],
            "target_instance_id": int(candidate["target_instance_id"]),
            "target_label": candidate["target_label"],
            "frame": int(view["frame"]),
            "color_size": view["color_size"],
            "bbox_xyxy": view["bbox_xyxy"],
            "zip_member": view["zip_member"],
        }
    controls = pixel.load_json(HERE / protocol["controls"]["path"])
    control_episodes = {str(row["episode_id"]): row for row in controls["episodes"]}
    control_images = {str(row["episode_id"]) + ":" + str(row["role"]): row for row in controls["images"].values()}
    episodes = [episode]
    for control in protocol["selection"]["negative_query_controls"]:
        control_id, source_id = str(control["episode_id"]), str(control["source_episode_id"])
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
        "schema": "blindassist-l10-3rscan-transport-opportunity-confirmation-cohort-v1",
        "authority": "FROZEN_NEW_SOURCE_PRE_RGB_MODEL_TRANSPORT_OPPORTUNITY_CONFIRMATION_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest": protocol["source"]["sequence_zips"],
        "selection": {
            **protocol["selection"],
            "joint_visible_surface_fraction": selected["joint_visible_surface_fraction"],
            "view_direction_cosine": selected["view_direction_cosine"],
            "transport_opportunity_score": selected["transport_opportunity_score"],
            "opened_rgb_members": 0,
            "model_calls": 0,
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
