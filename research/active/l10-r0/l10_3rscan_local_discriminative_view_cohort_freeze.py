#!/usr/bin/env python3
"""Freeze the source-admitted local active-view image cohort."""

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


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-local-discriminative-view-cohort-freeze-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-local-discriminative-view-cohort-v1"


def _image(
    row: dict[str, Any], key: str, role: str, scan_id: str, target_id: int
) -> tuple[str, dict[str, Any]]:
    return key, {
        "episode_id": key.removesuffix("_reference").removesuffix("_query"),
        "role": role,
        "scan_id": scan_id,
        "target_instance_id": target_id,
        "target_label": "door",
        "frame": int(row["frame"]),
        "color_size": row["color_size"],
        "bbox_xyxy": row["bbox_xyxy"],
        "zip_member": row["zip_member"],
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    source_row = protocol["source_result"]
    source_path = HERE / source_row["path"]
    pixel.require(pixel.sha256(source_path) == source_row["sha256"], "SOURCE_RESULT_HASH")
    source = pixel.load_json(source_path)
    pixel.require(source["conclusion"] == source_row["required_conclusion"], "SOURCE_CONCLUSION")
    pixel.require(bool(source["source_evaluable"]), "SOURCE_NOT_EVALUABLE")

    candidate = source["candidate"]
    target_id = int(candidate["target_instance_id"])
    images: dict[str, dict[str, Any]] = {}
    reference_keys: list[str] = []
    for index, row in enumerate(source["reference_memory"]["selected"], start=1):
        key = f"LDV_reference_{index}"
        reference_keys.append(key)
        images.update([_image(row, key, "reference", str(candidate["reference_scan_id"]), target_id)])
    query_keys = ["LDV_initial_1_query", "LDV_initial_2_query", "LDV_action_query"]
    for key, row in zip(query_keys, source["query_triplet"]["selected"], strict=True):
        images.update([_image(row, key, "query", str(candidate["rescan_id"]), target_id)])

    result = {
        "schema": RESULT_SCHEMA,
        "authority": "LOCALLY_MATERIALIZED_SOURCE_FROZEN_ACTIVE_VIEW_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "panel": {
            "reference_keys": reference_keys,
            "initial_query_keys": query_keys[:2],
            "action_query_key": query_keys[2],
            "ordered_query_keys": query_keys,
            "fixed_action": "NEXT_FRAME_FORWARD",
            "truth_free_trigger_threshold": float(protocol["truth_free_trigger_threshold"]),
            "same_scene_hard_negative_instance_id": int(source["same_scene_sibling"]["instance_id"]),
            "selection": "Pose/depth source gate only; no RGB/model or outcome selection.",
        },
        "artifact_root": str((ROOT / "artifacts.local").resolve()),
        "source_manifest": source["source_manifest"],
        "images": images,
        "rgb_members_opened_during_freeze": 0,
        "model_calls_during_freeze": 0,
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
