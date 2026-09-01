#!/usr/bin/env python3
"""Freeze an ordered multi-family queue for active-view source screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-source-queue-freeze-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-discriminative-view-source-queue-v1"


def _consume(cohort: dict[str, Any], consumed: set[tuple[str, int]], references: set[str]) -> None:
    candidate = cohort.get("candidate")
    if isinstance(candidate, dict):
        reference = str(candidate["reference_scan_id"])
        references.add(reference)
        consumed.add((reference, int(candidate["target_instance_id"])))
    for episode in cohort.get("episodes", []):
        if "reference_scan_id" in episode and "target_instance_id" in episode:
            reference = str(episode["reference_scan_id"])
            references.add(reference)
            consumed.add((reference, int(episode["target_instance_id"])))


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("candidate_protocol", "exclusion_authority"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    for row in protocol["source"]["metadata"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    consumed: set[tuple[str, int]] = set()
    used_references: set[str] = set()
    authority = pixel.load_json(HERE / protocol["exclusion_authority"]["path"])
    for source in authority["physical_target_exclusions"]:
        path = HERE / source["path"]
        pixel.require(pixel.sha256(path) == source["sha256"], f"EXCLUSION_HASH:{source['path']}")
        _consume(pixel.load_json(path), consumed, used_references)
    for source in protocol["additional_exclusions"]:
        path = HERE / source["path"]
        pixel.require(pixel.sha256(path) == source["sha256"], f"ADDITIONAL_EXCLUSION_HASH:{source['path']}")
        _consume(pixel.load_json(path), consumed, used_references)

    candidate_protocol = pixel.load_json(HERE / protocol["candidate_protocol"]["path"])
    candidates = extent.candidate_rows(candidate_protocol, data_root, require_geometry=False)
    required_names = ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip")
    selected: list[dict[str, Any]] = []
    seen_references: set[str] = set()
    considered = 0
    for candidate in candidates:
        reference = str(candidate["reference_scan_id"])
        query = str(candidate["rescan_id"])
        target = int(candidate["target_instance_id"])
        family = (reference, query)
        if reference in used_references or reference in seen_references or (reference, target) in consumed:
            continue
        if len(candidate["rescan_door_instance_ids"]) < 2:
            continue
        considered += 1
        if any((data_root / scan_id / name).exists() for scan_id in family for name in required_names):
            continue
        seen_references.add(reference)
        selected.append(candidate)
        if len(selected) == int(protocol["selection"]["queue_length"]):
            break
    pixel.require(len(selected) == int(protocol["selection"]["queue_length"]), "QUEUE_UNDERSIZED")

    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_PRE_DOWNLOAD_PRE_POSE_DEPTH_PRE_RGB_PRE_MODEL_MULTI_FAMILY_SOURCE_QUEUE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "selection": {
            "ranking": protocol["selection"]["ranking"],
            "queue_length": len(selected),
            "metadata_candidates": len(candidates),
            "eligible_families_considered_before_queue_complete": considered,
            "excluded_physical_targets": len(consumed),
            "excluded_reference_scan_families": len(used_references),
            "opened_pose_members": 0,
            "opened_depth_members": 0,
            "opened_rgb_members": 0,
            "model_calls": 0,
        },
        "ordered_candidates": [
            {
                "queue_index": index,
                "candidate": candidate,
                "download_plan": [
                    {"scan_id": scan_id, "file": name}
                    for scan_id in (candidate["reference_scan_id"], candidate["rescan_id"])
                    for name in required_names
                ],
            }
            for index, candidate in enumerate(selected, start=1)
        ],
        "screening": protocol["screening"],
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
