#!/usr/bin/env python3
"""Freeze a new-family set-memory confirmation cohort with fixed controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-set-memory-confirmation-freeze-protocol-v1"


def _image(episode_id: str, role: str, scan_id: str, candidate: dict, view: dict) -> dict:
    return {
        "episode_id": episode_id, "role": role, "scan_id": scan_id,
        "target_instance_id": int(candidate["target_instance_id"]),
        "target_label": candidate["target_label"], "frame": int(view["frame"]),
        "color_size": view["color_size"], "bbox_xyxy": view["bbox_xyxy"],
        "zip_member": view["zip_member"],
    }


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("source_result", "controls"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    source = pixel.load_json(HERE / protocol["source_result"]["path"])
    pixel.require(source["source_evaluable"] is True, "SOURCE_NOT_EVALUABLE")
    candidate = source["candidate"]
    references = source["reference_memory"]["selected"]
    queries = source["query_memory"]["selected"]
    ids = [str(value) for value in protocol["selection"]["memory_episode_ids"]]
    pixel.require(len(references) == len(queries) == len(ids) == 3, "MEMORY_CARDINALITY")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["sequence_zips"].values():
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    episodes, images = [], {}
    for episode_id, reference, query in zip(ids, references, queries):
        episodes.append({
            "episode_id": episode_id, **candidate, "reference": deepcopy(reference),
            "query": deepcopy(query), "active_query": None,
            "selection_authority": "PRE_MODEL_GREEDY_COMPLEMENTARY_SURFACE_SET_MEMORY",
        })
        images[f"{episode_id}_reference"] = _image(
            episode_id, "reference", candidate["reference_scan_id"], candidate, reference
        )
        images[f"{episode_id}_query"] = _image(
            episode_id, "query", candidate["rescan_id"], candidate, query
        )
    sibling = source["same_scene_sibling"]
    sibling_id = str(protocol["selection"]["sibling_episode_id"])
    sibling_candidate = {
        **candidate, "target_instance_id": int(sibling["instance_id"]),
        "target_label": sibling["label"],
    }
    episodes.append({
        "episode_id": sibling_id, "reference_scan_id": candidate["rescan_id"],
        "rescan_id": candidate["rescan_id"], "target_instance_id": int(sibling["instance_id"]),
        "target_label": sibling["label"], "reference": None,
        "query": deepcopy(sibling["selected"]),
        "negative_authority": "SAME_RESCAN_DIFFERENT_INSTANCE_ID_FROM_TARGET",
    })
    images[f"{sibling_id}_query"] = _image(
        sibling_id, "query", candidate["rescan_id"], sibling_candidate, sibling["selected"]
    )

    controls = pixel.load_json(HERE / protocol["controls"]["path"])
    control_episodes = {str(row["episode_id"]): row for row in controls["episodes"]}
    control_images = {str(row["episode_id"]) + ":" + str(row["role"]): row for row in controls["images"].values()}
    for control in protocol["selection"]["negative_query_controls"]:
        control_id, source_id = str(control["episode_id"]), str(control["source_episode_id"])
        copied = deepcopy(control_episodes[source_id])
        copied["episode_id"] = control_id
        copied["negative_query_control_provenance"] = {"cohort_path": protocol["controls"]["path"], "source_episode_id": source_id}
        episodes.append(copied)
        for role in ("reference", "query"):
            image = deepcopy(control_images[f"{source_id}:{role}"])
            image["episode_id"] = control_id
            images[f"{control_id}_{role}"] = image
    output = {
        "schema": "blindassist-l10-3rscan-set-memory-confirmation-cohort-v1",
        "authority": "FROZEN_NEW_FAMILY_PRE_DESCRIPTOR_SET_MEMORY_CONFIRMATION_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest": protocol["source"]["sequence_zips"],
        "selection": {
            **protocol["selection"],
            "reference_cumulative_visible_fraction": source["reference_memory"]["final_cumulative_visible_target_fraction"],
            "query_cumulative_visible_fraction": source["query_memory"]["final_cumulative_visible_target_fraction"],
            "opened_rgb_members_before_freeze": 0, "model_calls_before_freeze": 0,
        },
        "episodes": episodes, "images": images,
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.require(len(episodes) == 6 and len(images) == 11, "COHORT_CARDINALITY")
    pixel.atomic_write_json(output_path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
