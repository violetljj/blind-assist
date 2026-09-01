#!/usr/bin/env python3
"""Freeze every source-admitted extra query view for a consumed-family oracle."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import l10_3rscan_joint_covisibility_selector_posthoc as views  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-discriminative-view-oracle-freeze-protocol-v1"
)


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        pixel.require(
            pixel.sha256(HERE / dependency["path"]) == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    for key in ("source_protocol", "source_result", "base_cohort", "baseline"):
        row = protocol[key]
        pixel.require(
            pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )

    source_protocol = pixel.load_json(HERE / protocol["source_protocol"]["path"])
    source_result = pixel.load_json(HERE / protocol["source_result"]["path"])
    base_cohort = pixel.load_json(HERE / protocol["base_cohort"]["path"])
    baseline = pixel.load_json(HERE / protocol["baseline"]["path"])
    pixel.require(
        source_result["conclusion"]
        == protocol["source_result"]["required_conclusion"],
        "SOURCE_CONCLUSION",
    )
    pixel.require(
        baseline["conclusion"] == protocol["baseline"]["required_conclusion"],
        "BASELINE_CONCLUSION",
    )

    scan_id = str(protocol["panel"]["scan_id"])
    target_instance_id = int(protocol["panel"]["target_instance_id"])
    data_root = ROOT / protocol["panel"]["artifact_data_root"]
    _, candidates, opened = views._candidates(
        data_root,
        scan_id,
        target_instance_id,
        source_protocol["candidate_view_rules"],
    )
    initial_frames = {int(value) for value in protocol["panel"]["initial_frames"]}
    admitted_by_frame = {int(row["frame"]): row for row, _, _ in candidates}
    pixel.require(initial_frames <= set(admitted_by_frame), "INITIAL_FRAME_NOT_ADMITTED")

    extra_rows = [
        deepcopy(admitted_by_frame[frame])
        for frame in sorted(admitted_by_frame)
        if frame not in initial_frames
    ]
    pixel.require(
        len(extra_rows) == int(protocol["panel"]["required_extra_views"]),
        "EXTRA_VIEW_COUNT",
    )
    images = {
        key: deepcopy(value)
        for key, value in base_cohort["images"].items()
        if key.endswith("_reference")
    }
    query_keys: list[str] = []
    local_keys: list[str] = []
    wide_keys: list[str] = []
    local_frames = {int(value) for value in protocol["panel"]["local_frames"]}
    for row in extra_rows:
        frame = int(row["frame"])
        key = f"DVO_frame_{frame:06d}_query"
        images[key] = {
            "episode_id": f"DVO_frame_{frame:06d}",
            "role": "query",
            "scan_id": scan_id,
            "target_instance_id": target_instance_id,
            "target_label": "door",
            "frame": frame,
            "color_size": row["color_size"],
            "bbox_xyxy": row["bbox_xyxy"],
            "zip_member": row["zip_member"],
        }
        query_keys.append(key)
        (local_keys if frame in local_frames else wide_keys).append(key)

    result = {
        "schema": "blindassist-l10-3rscan-discriminative-view-oracle-cohort-v1",
        "authority": "CONSUMED_ELEVENTH_FAMILY_ALL_ADMITTED_VIEW_ORACLE_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": pixel.sha256(Path(__file__)),
        },
        "panel": {
            "scan_id": scan_id,
            "target_instance_id": target_instance_id,
            "initial_frames": sorted(initial_frames),
            "admitted_view_count": len(candidates),
            "extra_view_count": len(extra_rows),
            "query_keys": query_keys,
            "local_one_step_query_keys": local_keys,
            "wide_baseline_query_keys": wide_keys,
            "selection": "All source-admitted views except the two consumed initial frames; no RGB/model or outcome selection.",
        },
        "images": images,
        "source_receipt": {
            "opened": opened,
            "rgb_members_opened": 0,
            "model_calls": 0,
            "admitted_rows": extra_rows,
        },
        "literature_motivation": protocol["literature_motivation"],
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
