#!/usr/bin/env python3
"""Audit whether the consumed SC34 view contains enough 2D target support."""

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

import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-posthoc-result-v1"


def _shape_receipt(stats: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    x0, y0, x1, y1 = (float(value) for value in stats["bbox_xyxy"])
    bbox_width = max(0.0, x1 - x0)
    bbox_height = max(0.0, y1 - y0)
    short_side = min(bbox_width, bbox_height)
    long_side = max(bbox_width, bbox_height)
    return {
        "bbox_width_pixels": bbox_width,
        "bbox_height_pixels": bbox_height,
        "bbox_short_side_pixels": short_side,
        "bbox_short_side_fraction": short_side / min(width, height),
        "bbox_aspect_ratio": long_side / max(short_side, 1e-12),
    }


def _adequate(shape: dict[str, Any], rules: dict[str, Any]) -> bool:
    return bool(
        shape["bbox_short_side_fraction"]
        >= float(rules["minimum_projected_bbox_short_side_fraction"])
        and shape["bbox_aspect_ratio"] <= float(rules["maximum_projected_bbox_aspect_ratio"])
    )


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    for key in ("source_cohort", "predecessor"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    cohort = pixel.load_json(HERE / protocol["source_cohort"]["path"])
    episode = next(row for row in cohort["episodes"] if row["episode_id"] == protocol["source"]["episode_id"])
    scan_id = str(episode["rescan_id"])
    target_id = int(episode["target_instance_id"])
    primary_frame = int(episode["query"]["frame"])
    scan_root = artifact_root / "datasets/3rscan" / scan_id
    target_points = extent.ply_instance_points(
        scan_root / "labels.instances.annotated.v2.ply", {target_id}
    )[target_id]
    source_rules = protocol["source"]["inherited_frame_rules"]
    adequacy_rules = protocol["observation_adequacy"]
    maximum_baseline = float(adequacy_rules["maximum_camera_translation_metres"])
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
    admitted: list[dict[str, Any]] = []

    with zipfile.ZipFile(scan_root / "sequence.zip") as archive:
        info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
        width, height = int(info["color_width"]), int(info["color_height"])
        primary_pose = pixel.read_pose(archive, primary_frame)
        primary_position = primary_pose[:3, 3]
        for frame in pixel.pose_frames(archive):
            try:
                pose = pixel.read_pose(archive, frame)
            except ValueError:
                continue
            opened["pose_members"] += 1
            if not np.isfinite(pose).all():
                continue
            baseline = float(np.linalg.norm(pose[:3, 3] - primary_position))
            if baseline > maximum_baseline:
                continue
            try:
                depth = pixel.decode_depth(archive, frame)
                opened["depth_members"] += 1
                stats = pixel.frame_visibility(
                    target_points,
                    pose,
                    info,
                    depth,
                    float(source_rules["depth_consistency_metres"]),
                )
            except ValueError:
                continue
            if not pixel.eligible(stats, source_rules):
                continue
            shape = _shape_receipt(stats, width, height)
            row = {
                "frame": int(frame),
                "baseline_metres": baseline,
                "color_size": [width, height],
                **stats,
                **shape,
                "observation_adequate": _adequate(shape, adequacy_rules),
            }
            admitted.append(row)

    pixel.require(opened["rgb_members"] == 0 and opened["model_calls"] == 0, "SOURCE_ISOLATION")
    primary = next(row for row in admitted if row["frame"] == primary_frame)
    adequate = [row for row in admitted if row["observation_adequate"]]
    best = None
    if adequate:
        best = max(
            adequate,
            key=lambda row: (
                float(row["bbox_short_side_fraction"]),
                float(row["depth_visible_ratio"]),
                float(row["depth_visible_vertices"]),
                float(row["projected_area_pixels"]),
                float(row["image_margin_pixels"]),
                -float(row["baseline_metres"]),
                -float(row["frame"]),
            ),
        )
    if primary["observation_adequate"]:
        conclusion = "L10_3RSCAN_SC34_PRIMARY_OBSERVATION_ADEQUATE"
    elif best is not None:
        conclusion = "L10_3RSCAN_SC34_BOUNDED_ACTIVE_OBSERVATION_RECOVERY_AVAILABLE"
    else:
        conclusion = "L10_3RSCAN_SC34_SOURCE_NOT_EVALUABLE_WITHIN_BOUNDED_ACTIVE_OBSERVATION"

    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SC34_GEOMETRY_ONLY_OBSERVATION_ADEQUACY_POSTHOC_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": conclusion,
        "primary_observation_adequate": bool(primary["observation_adequate"]),
        "bounded_active_recovery_available": bool(best is not None and best["frame"] != primary_frame),
        "metrics": {
            "poses_opened": opened["pose_members"],
            "depth_members_opened": opened["depth_members"],
            "inherited_source_admitted_views": len(admitted),
            "observation_adequate_views": len(adequate),
            "rgb_members_opened": 0,
            "model_calls": 0,
        },
        "rules": adequacy_rules,
        "primary": primary,
        "best_bounded_adequate_view": best,
        "literature_motivation": protocol["literature_motivation"],
        "decision": (
            "Do not use SC34 as positive algorithm-transfer evidence when neither the primary nor any bounded active view supplies adequate projected target support. "
            "Apply the frozen adequacy rule before RGB/model access to successor source selection."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps({"conclusion": conclusion, "primary": primary, "best": best, "metrics": result["metrics"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
