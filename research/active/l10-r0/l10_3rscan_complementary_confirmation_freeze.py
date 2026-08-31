#!/usr/bin/env python3
"""Freeze a physical-target-disjoint active-query sibling-door confirmation panel."""

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


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-complementary-confirmation-freeze-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-complementary-confirmation-cohort-v1"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = pixel.load_json(path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    candidate_path = HERE / protocol["source"]["candidate_protocol_path"]
    pixel.require(pixel.sha256(candidate_path) == protocol["source"]["candidate_protocol_sha256"], "CANDIDATE_PROTOCOL_HASH")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        source = artifact_root / row["path"]
        pixel.require(source.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(source) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    for row in protocol["physical_target_exclusions"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"EXCLUSION_HASH:{row['path']}")
    return protocol


def active_query(
    data_root: Path,
    scan_id: str,
    target_id: int,
    primary_frame: int,
    foreign_id: int | None,
    rules: dict[str, Any],
    minimum: float,
    maximum: float,
    target_baseline: float,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    scan_root = data_root / scan_id
    wanted = {target_id} | ({foreign_id} if foreign_id is not None else set())
    points = extent.ply_instance_points(scan_root / "labels.instances.annotated.v2.ply", wanted)
    target_points = points[target_id]
    foreign_points = points.get(foreign_id) if foreign_id is not None else None
    opened = {"pose_members": 0, "depth_members": 0}
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
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
            foreign_inside = 0
            if foreign_points is not None:
                _, _, inside = pixel.project_points(
                    foreign_points, pose, info["color_intrinsic"],
                    int(info["color_width"]), int(info["color_height"]),
                )
                foreign_inside = int(np.count_nonzero(inside))
                if foreign_inside != 0:
                    continue
            key = (
                abs(baseline - target_baseline), abs(frame - primary_frame),
                -float(stats["depth_visible_ratio"]), -float(stats["depth_visible_vertices"]),
                -float(stats["projected_area_pixels"]), float(frame),
            )
            candidates.append((key, {
                "frame": int(frame),
                "color_size": [int(info["color_width"]), int(info["color_height"])],
                "baseline_metres": baseline,
                "frame_gap": abs(int(frame) - primary_frame),
                **stats,
                "foreign_target_instance_id": foreign_id,
                "foreign_target_inside_vertices": foreign_inside if foreign_id is not None else None,
                "zip_member": f"frame-{int(frame):06d}.color.jpg",
            }))
    return (min(candidates, key=lambda row: row[0])[1] if candidates else None), opened


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

    rules = protocol["selection"]["frame_rules"]
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
    episodes: list[dict[str, Any]] = []
    for frozen in protocol["selection"]["episodes"]:
        episode_id = str(frozen["episode_id"])
        reference_id = str(frozen["reference_scan_id"])
        rescan_id = str(frozen["rescan_id"])
        target_id = int(frozen["target_instance_id"])
        key = (reference_id, rescan_id, target_id)
        pixel.require(key in candidate_keys, f"NOT_STABLE_DOOR_CANDIDATE:{episode_id}")
        pixel.require((reference_id, target_id) not in consumed, f"PHYSICAL_TARGET_CONSUMED:{episode_id}")
        reference, reference_opened = pixel.select_frame(data_root, reference_id, target_id, rules)
        query, query_opened = pixel.select_frame(data_root, rescan_id, target_id, rules)
        for counts in (reference_opened, query_opened):
            for name, count in counts.items():
                opened[name] += count
        pixel.require(reference is not None and query is not None, f"PRIMARY_SOURCE_NOT_EVALUABLE:{episode_id}")
        foreign_id = frozen.get("active_foreign_target_instance_id")
        active, active_opened = active_query(
            data_root, rescan_id, target_id, int(query["frame"]),
            int(foreign_id) if foreign_id is not None else None,
            rules,
            float(protocol["selection"]["minimum_active_baseline_metres"]),
            float(protocol["selection"]["maximum_active_baseline_metres"]),
            float(protocol["selection"]["target_active_baseline_metres"]),
        )
        for name, count in active_opened.items():
            opened[name] += count
        pixel.require(active is not None, f"ACTIVE_SOURCE_NOT_EVALUABLE:{episode_id}")
        row = candidate_keys[key]
        episodes.append({
            "episode_id": episode_id,
            **row,
            "reference": reference,
            "query": query,
            "active_query": active,
        })

    by_id = {str(row["episode_id"]): row for row in episodes}
    absence_receipts: dict[str, Any] = {}
    for pair in protocol["evaluation"]["pairs"]:
        if pair["label"] != "target_absent":
            continue
        reference = by_id[str(pair["reference_episode"])]
        query = by_id[str(pair["query_episode"])]
        target_id = int(reference["target_instance_id"])
        scan_root = data_root / str(query["rescan_id"])
        points = extent.ply_instance_points(scan_root / "labels.instances.annotated.v2.ply", {target_id})[target_id]
        with zipfile.ZipFile(scan_root / "sequence.zip") as archive:
            info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
            primary_pose = pixel.read_pose(archive, int(query["query"]["frame"]))
            active_pose = pixel.read_pose(archive, int(query["active_query"]["frame"]))
        counts = {}
        for role, pose in (("primary_query", primary_pose), ("active_query", active_pose)):
            _, _, inside = pixel.project_points(
                points, pose, info["color_intrinsic"], int(info["color_width"]), int(info["color_height"])
            )
            counts[role] = int(np.count_nonzero(inside))
            pixel.require(counts[role] == 0, f"NEGATIVE_TARGET_VISIBLE:{pair['id']}:{role}")
        absence_receipts[str(pair["id"])] = {
            "reference_target_instance_id": target_id,
            "query_sibling_target_instance_id": int(query["target_instance_id"]),
            "query_scan_id": query["rescan_id"],
            "primary_query_frame": int(query["query"]["frame"]),
            "active_query_frame": int(query["active_query"]["frame"]),
            "primary_projected_inside_vertices": counts["primary_query"],
            "active_projected_inside_vertices": counts["active_query"],
        }

    source_manifest = {
        row["path"].removeprefix("datasets/3rscan/"): {
            "path": row["path"], "bytes": int(row["bytes"]), "sha256": row["sha256"]
        }
        for row in protocol["source"]["files"]
    }
    images: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        for role, scan_key in (("reference", "reference_scan_id"), ("query", "rescan_id")):
            frame = episode[role]
            images[f"{episode['episode_id']}_{role}"] = {
                "episode_id": episode["episode_id"], "role": role,
                "scan_id": episode[scan_key], "target_instance_id": episode["target_instance_id"],
                "target_label": episode["target_label"], "frame": frame["frame"],
                "color_size": frame["color_size"], "bbox_xyxy": frame["bbox_xyxy"],
                "image_margin_pixels": frame["image_margin_pixels"],
                "inside_vertex_fraction": frame["inside_vertex_fraction"],
                "depth_visible_ratio": frame["depth_visible_ratio"],
                "zip_member": f"frame-{int(frame['frame']):06d}.color.jpg",
            }
    pixel.require(opened["rgb_members"] == 0 and opened["model_calls"] == 0, "PRE_RGB_MODEL_VIOLATION")
    pixel.atomic_write_json(output_path, {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_MODEL_BLIND_PHYSICAL_TARGET_DISJOINT_COMPLEMENTARY_CONFIRMATION_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str(artifact_root.resolve()),
        "selection": {**protocol["selection"], "opened_members": opened, "excluded_physical_targets": len(consumed)},
        "source_manifest": source_manifest,
        "episodes": episodes,
        "images": images,
        "evaluation": protocol["evaluation"],
        "sibling_absence_receipts": absence_receipts,
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
