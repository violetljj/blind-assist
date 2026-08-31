#!/usr/bin/env python3
"""Freeze a bounded multi-view instance memory with fixed cross-family controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-multiview-memory-freeze-protocol-v1"


def _image(episode_id: str, role: str, scan_id: str, candidate: dict, view: dict) -> dict:
    return {
        "episode_id": episode_id,
        "role": role,
        "scan_id": scan_id,
        "target_instance_id": int(candidate["target_instance_id"]),
        "target_label": candidate["target_label"],
        "frame": int(view["frame"]),
        "color_size": view["color_size"],
        "bbox_xyxy": view["bbox_xyxy"],
        "zip_member": view["zip_member"],
    }


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("memory_source", "candidate_source", "controls"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    memory = pixel.load_json(HERE / protocol["memory_source"]["path"])
    candidate_source = pixel.load_json(HERE / protocol["candidate_source"]["path"])
    candidate = candidate_source["candidate"]
    panel = memory["panels"][0]
    references = panel["reference"]["selected"]
    queries = panel["query"]["selected"]
    episode_ids = [str(value) for value in protocol["selection"]["memory_episode_ids"]]
    pixel.require(len(references) == len(queries) == len(episode_ids) == 3, "MEMORY_CARDINALITY")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["sequence_zips"].values():
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    episodes = []
    images = {}
    for episode_id, reference, query in zip(episode_ids, references, queries):
        episode = {
            "episode_id": episode_id,
            **candidate,
            "reference": deepcopy(reference),
            "query": deepcopy(query),
            "active_query": None,
            "selection_authority": "CONSUMED_GEOMETRY_DEPTH_GREEDY_THREE_VIEW_INSTANCE_MEMORY",
        }
        episodes.append(episode)
        images[f"{episode_id}_reference"] = _image(
            episode_id, "reference", candidate["reference_scan_id"], candidate, reference
        )
        images[f"{episode_id}_query"] = _image(
            episode_id, "query", candidate["rescan_id"], candidate, query
        )

    controls = pixel.load_json(HERE / protocol["controls"]["path"])
    control_episodes = {str(row["episode_id"]): row for row in controls["episodes"]}
    control_images = {
        str(row["episode_id"]) + ":" + str(row["role"]): row
        for row in controls["images"].values()
    }
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
        "schema": "blindassist-l10-3rscan-multiview-memory-cohort-v1",
        "authority": "CONSUMED_POSTHOC_PRE_MODEL_BOUNDED_MULTIVIEW_MEMORY_DEVELOPMENT_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest": protocol["source"]["sequence_zips"],
        "selection": {
            **protocol["selection"],
            "reference_cumulative_visible_fraction": panel["reference"]["final_cumulative_visible_target_fraction"],
            "query_cumulative_visible_fraction": panel["query"]["final_cumulative_visible_target_fraction"],
            "opened_rgb_members_before_freeze": 0,
            "model_calls_before_freeze": 0,
        },
        "episodes": episodes,
        "images": images,
        "evaluation": protocol["evaluation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.require(len(episodes) == 5 and len(images) == 10, "COHORT_CARDINALITY")
    pixel.require(len(protocol["evaluation"]["pairs"]) == 15, "PAIR_CARDINALITY")
    pixel.atomic_write_json(output_path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
