#!/usr/bin/env python3
"""Freeze one geometry-only active query view per 3RScan sibling-door episode."""

from __future__ import annotations

import argparse
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


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-active-query-freeze-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-active-query-freeze-result-v1"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = pixel.load_json(path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    source_path = HERE / protocol["source"]["cohort_path"]
    pixel.require(pixel.sha256(source_path) == protocol["source"]["cohort_sha256"], "SOURCE_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    pixel.require(
        pixel.sha256(predecessor_path) == protocol["predecessor"]["result_sha256"],
        "PREDECESSOR_HASH",
    )
    pixel.require(
        pixel.load_json(predecessor_path)["conclusion"]
        == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        source = artifact_root / row["path"]
        pixel.require(source.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(source) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    return protocol


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    source_path = HERE / protocol["source"]["cohort_path"]
    source = pixel.load_json(source_path)
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    rules = protocol["selection"]["frame_rules"]
    minimum = float(protocol["selection"]["minimum_baseline_metres"])
    maximum = float(protocol["selection"]["maximum_baseline_metres"])
    target_baseline = float(protocol["selection"]["target_baseline_metres"])
    maximum_foreign_inside = int(protocol["selection"]["maximum_foreign_target_inside_vertices"])
    absence_by_episode = {
        str(row["query_episode"]): row for row in protocol["same_scene_absence_constraints"]
    }
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
    episodes: list[dict[str, Any]] = []

    for episode in source["episodes"]:
        episode_id = str(episode["episode_id"])
        scan_id = str(episode["rescan_id"])
        target_id = int(episode["target_instance_id"])
        primary_frame = int(episode["query"]["frame"])
        absence = absence_by_episode.get(episode_id)
        foreign_id = int(absence["foreign_target_instance_id"]) if absence else None
        wanted_ids = {target_id} | ({foreign_id} if foreign_id is not None else set())
        scan_root = data_root / scan_id
        points_by_id = extent.ply_instance_points(
            scan_root / "labels.instances.annotated.v2.ply", wanted_ids
        )
        target_points = points_by_id[target_id]
        foreign_points = points_by_id.get(foreign_id) if foreign_id is not None else None
        candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        eligible_without_absence = 0
        with zipfile.ZipFile(scan_root / "sequence.zip") as archive:
            info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
            primary_pose = pixel.read_pose(archive, primary_frame)
            primary_position = primary_pose[:3, 3]
            for frame in pixel.pose_frames(archive):
                if frame == primary_frame:
                    continue
                try:
                    pose = pixel.read_pose(archive, frame)
                except ValueError:
                    continue
                opened["pose_members"] += 1
                if not np.isfinite(pose).all():
                    continue
                baseline = float(np.linalg.norm(pose[:3, 3] - primary_position))
                if baseline < minimum or baseline > maximum:
                    continue
                try:
                    depth = pixel.decode_depth(archive, frame)
                    opened["depth_members"] += 1
                    stats = pixel.frame_visibility(
                        target_points, pose, info, depth, float(rules["depth_consistency_metres"])
                    )
                except ValueError:
                    continue
                if not pixel.eligible(stats, rules):
                    continue
                eligible_without_absence += 1
                foreign_inside = 0
                if foreign_points is not None:
                    _, _, inside = pixel.project_points(
                        foreign_points,
                        pose,
                        info["color_intrinsic"],
                        int(info["color_width"]),
                        int(info["color_height"]),
                    )
                    foreign_inside = int(np.count_nonzero(inside))
                    if foreign_inside > maximum_foreign_inside:
                        continue
                key = (
                    abs(baseline - target_baseline),
                    abs(frame - primary_frame),
                    -float(stats["depth_visible_ratio"]),
                    -float(stats["depth_visible_vertices"]),
                    -float(stats["projected_area_pixels"]),
                    float(frame),
                )
                candidates.append(
                    (
                        key,
                        {
                            "frame": int(frame),
                            "color_size": [int(info["color_width"]), int(info["color_height"])],
                            "baseline_metres": baseline,
                            "frame_gap": abs(int(frame) - primary_frame),
                            **stats,
                            "foreign_target_instance_id": foreign_id,
                            "foreign_target_inside_vertices": foreign_inside if foreign_id is not None else None,
                            "zip_member": f"frame-{int(frame):06d}.color.jpg",
                        },
                    )
                )
        selected = min(candidates, key=lambda row: row[0])[1] if candidates else None
        episodes.append(
            {
                "episode_id": episode_id,
                "query_scan_id": scan_id,
                "target_instance_id": target_id,
                "target_label": episode["target_label"],
                "primary_query_frame": primary_frame,
                "active_query": selected,
                "eligible_frames_before_absence_constraint": eligible_without_absence,
                "eligible_frames_after_absence_constraint": len(candidates),
            }
        )

    evaluable = len(episodes) == int(protocol["selection"]["required_targets"]) and all(
        row["active_query"] is not None for row in episodes
    )
    pixel.require(opened["rgb_members"] == 0 and opened["model_calls"] == 0, "PRE_RGB_MODEL_VIOLATION")
    pixel.atomic_write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "FROZEN_PRE_RGB_MODEL_BLIND_ACTIVE_QUERY_SOURCE_ASSESSMENT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": pixel.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
            "source_cohort_path": source_path.name,
            "source_cohort_sha256": pixel.sha256(source_path),
            "conclusion": (
                "L10_3RSCAN_ACTIVE_QUERY_SOURCE_EVALUABLE" if evaluable
                else "L10_3RSCAN_ACTIVE_QUERY_SOURCE_NOT_EVALUABLE"
            ),
            "evaluable": evaluable,
            "selection": protocol["selection"],
            "episodes": episodes,
            "opened_members": opened,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
