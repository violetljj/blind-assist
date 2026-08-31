#!/usr/bin/env python3
"""Adjudicate a frozen 3RScan family's source adequacy before RGB/model access."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_3rscan_observation_adequacy_posthoc as adequacy  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-confirmation-source-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-confirmation-source-result-v1"


def _select(
    data_root: Path,
    scan_id: str,
    target_id: int,
    source_rules: dict[str, Any],
    adequacy_rules: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, int], int]:
    scan_root = data_root / scan_id
    points = extent.ply_instance_points(scan_root / "labels.instances.annotated.v2.ply", {target_id})[target_id]
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
    inherited_admitted = 0
    candidates: list[dict[str, Any]] = []
    with zipfile.ZipFile(scan_root / "sequence.zip") as archive:
        info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
        width, height = int(info["color_width"]), int(info["color_height"])
        for frame in pixel.pose_frames(archive):
            try:
                pose = pixel.read_pose(archive, frame)
            except ValueError:
                continue
            opened["pose_members"] += 1
            if not np.isfinite(pose).all():
                continue
            try:
                depth = pixel.decode_depth(archive, frame)
                opened["depth_members"] += 1
                stats = pixel.frame_visibility(
                    points, pose, info, depth, float(source_rules["depth_consistency_metres"])
                )
            except ValueError:
                continue
            if not pixel.eligible(stats, source_rules):
                continue
            inherited_admitted += 1
            shape = adequacy._shape_receipt(stats, width, height)
            if not adequacy._adequate(shape, adequacy_rules):
                continue
            candidates.append(
                {
                    "frame": int(frame),
                    "color_size": [width, height],
                    **stats,
                    **shape,
                    "zip_member": f"frame-{int(frame):06d}.color.jpg",
                }
            )
    selected = None
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                float(row["bbox_short_side_fraction"]),
                float(row["depth_visible_ratio"]),
                float(row["depth_visible_vertices"]),
                float(row["projected_area_pixels"]),
                float(row["image_margin_pixels"]),
                -float(row["frame"]),
            ),
        )
    return selected, opened, inherited_admitted


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    candidate_path = HERE / protocol["candidate"]["path"]
    pixel.require(pixel.sha256(candidate_path) == protocol["candidate"]["sha256"], "CANDIDATE_HASH")
    candidate_receipt = pixel.load_json(candidate_path)
    candidate = candidate_receipt["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    source_rules = protocol["source_rules"]["inherited"]
    adequacy_rules = protocol["source_rules"]["observation_adequacy"]
    target_id = int(candidate["target_instance_id"])
    reference, reference_opened, reference_inherited = _select(
        data_root, str(candidate["reference_scan_id"]), target_id, source_rules, adequacy_rules
    )
    query, query_opened, query_inherited = _select(
        data_root, str(candidate["rescan_id"]), target_id, source_rules, adequacy_rules
    )
    evaluable = reference is not None and query is not None
    conclusion = (
        "L10_3RSCAN_OBSERVATION_ADEQUACY_CONFIRMATION_SOURCE_EVALUABLE"
        if evaluable
        else "L10_3RSCAN_OBSERVATION_ADEQUACY_CONFIRMATION_SOURCE_NOT_EVALUABLE"
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_PRE_MODEL_OBSERVATION_ADEQUACY_SOURCE_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "conclusion": conclusion,
        "source_evaluable": evaluable,
        "reference": reference,
        "query": query,
        "metrics": {
            "reference_inherited_admitted_views": reference_inherited,
            "query_inherited_admitted_views": query_inherited,
            "pose_members_opened": reference_opened["pose_members"] + query_opened["pose_members"],
            "depth_members_opened": reference_opened["depth_members"] + query_opened["depth_members"],
            "rgb_members_opened": 0,
            "model_calls": 0,
        },
        "rules": protocol["source_rules"],
        "next_action": protocol["next_action"]["evaluable" if evaluable else "not_evaluable"],
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
