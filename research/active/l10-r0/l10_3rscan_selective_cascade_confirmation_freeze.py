#!/usr/bin/env python3
"""Freeze one new 3RScan target plus two fixed cross-scene negative queries."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-selective-cascade-confirmation-freeze-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-selective-cascade-confirmation-cohort-v1"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = pixel.load_json(path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    candidate_path = HERE / protocol["source"]["candidate_protocol_path"]
    pixel.require(pixel.sha256(candidate_path) == protocol["source"]["candidate_protocol_sha256"], "CANDIDATE_PROTOCOL_HASH")
    control_path = HERE / protocol["source"]["negative_control_cohort_path"]
    pixel.require(pixel.sha256(control_path) == protocol["source"]["negative_control_cohort_sha256"], "CONTROL_COHORT_HASH")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        source = artifact_root / row["path"]
        pixel.require(source.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(source) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    for row in protocol["physical_target_exclusions"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"EXCLUSION_HASH:{row['path']}")
    return protocol


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    candidate_protocol = pixel.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    candidates = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    candidate_keys = {
        (str(row["reference_scan_id"]), str(row["rescan_id"]), int(row["target_instance_id"])): row
        for row in candidates
    }
    consumed: set[tuple[str, int]] = set()
    for source in protocol["physical_target_exclusions"]:
        cohort = pixel.load_json(HERE / source["path"])
        for episode in cohort.get("episodes", []):
            if "reference_scan_id" in episode and "target_instance_id" in episode:
                consumed.add((str(episode["reference_scan_id"]), int(episode["target_instance_id"])))

    target = protocol["selection"]["fresh_target"]
    reference_id = str(target["reference_scan_id"])
    rescan_id = str(target["rescan_id"])
    target_id = int(target["target_instance_id"])
    key = (reference_id, rescan_id, target_id)
    pixel.require(key in candidate_keys, "FRESH_TARGET_NOT_STABLE_CANDIDATE")
    pixel.require((reference_id, target_id) not in consumed, "FRESH_TARGET_ALREADY_CONSUMED")
    rules = protocol["selection"]["frame_rules"]
    reference, reference_opened = pixel.select_frame(data_root, reference_id, target_id, rules)
    query, query_opened = pixel.select_frame(data_root, rescan_id, target_id, rules)
    pixel.require(reference is not None and query is not None, "FRESH_TARGET_PRIMARY_SOURCE_NOT_EVALUABLE")
    fresh_episode = {
        "episode_id": str(target["episode_id"]),
        **candidate_keys[key],
        "reference": reference,
        "query": query,
        "active_query": None,
    }

    control = pixel.load_json(HERE / protocol["source"]["negative_control_cohort_path"])
    old_episodes = {str(row["episode_id"]): row for row in control["episodes"]}
    old_images = {str(row["episode_id"]) + ":" + str(row["role"]): row for row in control["images"].values()}
    episodes = [fresh_episode]
    images: dict[str, dict[str, Any]] = {}
    for role, scan_key in (("reference", "reference_scan_id"), ("query", "rescan_id")):
        frame = fresh_episode[role]
        images[f"{fresh_episode['episode_id']}_{role}"] = {
            "episode_id": fresh_episode["episode_id"],
            "role": role,
            "scan_id": fresh_episode[scan_key],
            "target_instance_id": target_id,
            "target_label": fresh_episode["target_label"],
            "frame": int(frame["frame"]),
            "color_size": frame["color_size"],
            "bbox_xyxy": frame["bbox_xyxy"],
            "zip_member": f"frame-{int(frame['frame']):06d}.color.jpg",
        }

    for control_row in protocol["selection"]["negative_query_controls"]:
        source_id = str(control_row["source_episode_id"])
        episode_id = str(control_row["episode_id"])
        episode = deepcopy(old_episodes[source_id])
        episode["episode_id"] = episode_id
        episode["negative_query_control_provenance"] = {
            "cohort_path": protocol["source"]["negative_control_cohort_path"],
            "source_episode_id": source_id,
            "pixels_consumed_for_another_target": True,
        }
        episodes.append(episode)
        for role in ("reference", "query"):
            image = deepcopy(old_images[f"{source_id}:{role}"])
            image["episode_id"] = episode_id
            images[f"{episode_id}_{role}"] = image

    for episode in episodes[1:]:
        pixel.require(episode["reference_scan_id"] != reference_id, f"NEGATIVE_NOT_CROSS_SCENE:{episode['episode_id']}")
    opened = {
        "pose_members": int(reference_opened["pose_members"] + query_opened["pose_members"]),
        "depth_members": int(reference_opened["depth_members"] + query_opened["depth_members"]),
        "rgb_members": 0,
        "model_calls": 0,
    }
    source_manifest = {
        row["path"].removeprefix("datasets/3rscan/"): {
            "path": row["path"], "bytes": int(row["bytes"]), "sha256": row["sha256"]
        }
        for row in protocol["source"]["files"] if row["path"].endswith("sequence.zip")
    }
    pixel.require(len(episodes) == 3 and len(images) == 6, "COHORT_CARDINALITY")
    pixel.atomic_write_json(output_path, {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_MODEL_ONE_FRESH_TARGET_TWO_CONSUMED_CROSS_SCENE_QUERY_CONTROLS",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str(artifact_root.resolve()),
        "selection": {**protocol["selection"], "opened_members": opened, "excluded_physical_targets": len(consumed)},
        "source_manifest": source_manifest,
        "episodes": episodes,
        "images": images,
        "evaluation": protocol["evaluation"],
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
