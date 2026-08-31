#!/usr/bin/env python3
"""Select consumed PV28 views by cross-scan physical-surface covisibility."""

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
import l10_3rscan_multiview_observation_portfolio_posthoc as portfolio  # noqa: E402
import l10_3rscan_observation_adequacy_posthoc as adequacy  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-joint-covisibility-selector-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-joint-covisibility-selector-posthoc-result-v1"


def _candidates(
    data_root: Path,
    scan_id: str,
    target_id: int,
    rules: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[dict[str, Any], np.ndarray, np.ndarray]], dict[str, int]]:
    scan_root = data_root / scan_id
    points = extent.ply_instance_points(scan_root / "labels.instances.annotated.v2.ply", {target_id})[target_id]
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
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
                stats = pixel.frame_visibility(points, pose, info, depth, float(rules["depth_consistency_metres"]))
            except ValueError:
                continue
            shape = adequacy._shape_receipt(stats, width, height)
            if not (
                stats["inside_vertex_fraction"] >= float(rules["minimum_per_view_inside_vertex_fraction"])
                and stats["projected_area_pixels"] >= float(rules["minimum_projected_area_pixels"])
                and stats["image_margin_pixels"] >= float(rules["minimum_image_margin_pixels"])
                and stats["depth_compared_vertices"] >= int(rules["minimum_depth_compared_vertices"])
                and stats["depth_visible_vertices"] >= int(rules["minimum_depth_visible_vertices"])
                and stats["depth_visible_ratio"] >= float(rules["minimum_depth_visible_ratio"])
                and shape["bbox_short_side_fraction"]
                >= float(rules["minimum_projected_bbox_short_side_fraction"])
                and shape["bbox_aspect_ratio"] <= float(rules["maximum_projected_bbox_aspect_ratio"])
            ):
                continue
            visible = portfolio._visible_mask(
                points, pose, info, depth, float(rules["depth_consistency_metres"])
            )
            candidates.append(
                (
                    {
                        "frame": int(frame),
                        "color_size": [width, height],
                        **stats,
                        **shape,
                        "visible_target_vertices": int(np.count_nonzero(visible)),
                        "zip_member": f"frame-{int(frame):06d}.color.jpg",
                    },
                    visible,
                    pose,
                )
            )
    return points, candidates, opened


def _unit_view(camera_to_scan: np.ndarray, target_centroid: np.ndarray) -> np.ndarray:
    direction = target_centroid - camera_to_scan[:3, 3]
    norm = float(np.linalg.norm(direction))
    pixel.require(norm > 0.0, "ZERO_VIEW_DIRECTION")
    return direction / norm


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("source_result", "terminal_receipt"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    source = pixel.load_json(HERE / protocol["source_result"]["path"])
    candidate = source["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    data_root = artifact_root / "datasets/3rscan"
    target_id = int(candidate["target_instance_id"])
    ref_points, ref_candidates, ref_opened = _candidates(
        data_root, str(candidate["reference_scan_id"]), target_id, protocol["candidate_rules"]
    )
    query_points, query_candidates, query_opened = _candidates(
        data_root, str(candidate["rescan_id"]), target_id, protocol["candidate_rules"]
    )
    matrix = extent.provider_matrix(candidate["transform"])
    query_in_reference = extent.transform_points(query_points, matrix)
    distance = np.linalg.norm(ref_points[:, None, :] - query_in_reference[None, :, :], axis=2)
    ref_to_query = np.argmin(distance, axis=1)
    query_to_ref = np.argmin(distance, axis=0)
    ref_indices = np.arange(len(ref_points), dtype=np.int64)
    mutual = query_to_ref[ref_to_query] == ref_indices
    mutual &= distance[ref_indices, ref_to_query] <= float(protocol["surface_correspondence"]["maximum_distance_metres"])
    matched_ref = ref_indices[mutual]
    matched_query = ref_to_query[mutual]
    pixel.require(len(matched_ref) > 0, "NO_MUTUAL_TARGET_SURFACE_CORRESPONDENCES")
    target_centroid = np.mean(ref_points[matched_ref], axis=0)

    ranked = []
    for ref_row, ref_visible, ref_pose in ref_candidates:
        ref_direction = _unit_view(ref_pose, target_centroid)
        for query_row, query_visible, query_pose in query_candidates:
            common = int(np.count_nonzero(ref_visible[matched_ref] & query_visible[matched_query]))
            query_pose_in_reference = matrix @ query_pose
            query_direction = _unit_view(query_pose_in_reference, target_centroid)
            direction_cosine = float(np.dot(ref_direction, query_direction))
            receipt = {
                "reference": ref_row,
                "query": query_row,
                "mutual_surface_vertices": int(len(matched_ref)),
                "joint_visible_surface_vertices": common,
                "joint_visible_surface_fraction": float(common / len(matched_ref)),
                "view_direction_cosine": direction_cosine,
            }
            key = (
                common,
                direction_cosine,
                min(float(ref_row["bbox_short_side_fraction"]), float(query_row["bbox_short_side_fraction"])),
                min(float(ref_row["depth_visible_ratio"]), float(query_row["depth_visible_ratio"])),
                -int(ref_row["frame"]),
                -int(query_row["frame"]),
            )
            ranked.append((key, receipt))
    pixel.require(len(ranked) > 0, "NO_JOINT_VIEW_CANDIDATES")
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[0][1]
    baseline = next(
        row
        for _, row in ranked
        if int(row["reference"]["frame"]) == int(source["reference"]["frame"])
        and int(row["query"]["frame"]) == int(source["query"]["frame"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SOURCE_GEOMETRY_DEPTH_ONLY_JOINT_COVISIBILITY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "conclusion": "L10_3RSCAN_JOINT_COVISIBILITY_SELECTOR_POSTHOC_DIAGNOSTIC_ONLY",
        "reference_candidate_views": len(ref_candidates),
        "query_candidate_views": len(query_candidates),
        "joint_candidate_pairs": len(ranked),
        "mutual_target_surface_vertices": int(len(matched_ref)),
        "independent_maximum_visible_baseline": baseline,
        "selected": selected,
        "selected_minus_baseline_joint_visible_vertices": int(
            selected["joint_visible_surface_vertices"] - baseline["joint_visible_surface_vertices"]
        ),
        "top_pairs": [row for _, row in ranked[: int(protocol["report_top_pairs"])]],
        "opened": {"reference": ref_opened, "query": query_opened},
        "literature_motivation": protocol["literature_motivation"],
        "next_action": protocol["next_action"],
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
