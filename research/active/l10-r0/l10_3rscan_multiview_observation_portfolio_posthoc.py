#!/usr/bin/env python3
"""Measure whether bounded multi-view portfolios repair single-view source scarcity."""

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


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-multiview-observation-portfolio-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-multiview-observation-portfolio-posthoc-result-v1"


def _visible_mask(
    points: np.ndarray,
    pose: np.ndarray,
    info: dict[str, Any],
    depth: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    camera, pixels, inside = pixel.project_points(
        points,
        pose,
        info["depth_intrinsic"],
        int(info["depth_width"]),
        int(info["depth_height"]),
    )
    result = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(inside)
    if not len(indices):
        return result
    xs = np.rint(pixels[indices, 0]).astype(np.int32).clip(0, int(info["depth_width"]) - 1)
    ys = np.rint(pixels[indices, 1]).astype(np.int32).clip(0, int(info["depth_height"]) - 1)
    observed = depth[ys, xs].astype(np.float64) / 1000.0
    valid = observed > 0.0
    visible_indices = indices[valid][np.abs(observed[valid] - camera[indices[valid], 2]) <= tolerance]
    result[visible_indices] = True
    return result


def _portfolio(
    data_root: Path,
    scan_id: str,
    target_id: int,
    rules: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    scan_root = data_root / scan_id
    points = extent.ply_instance_points(scan_root / "labels.instances.annotated.v2.ply", {target_id})[target_id]
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []
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
                stats = pixel.frame_visibility(
                    points, pose, info, depth, float(rules["depth_consistency_metres"])
                )
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
            visible = _visible_mask(
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
                )
            )

    covered = np.zeros(len(points), dtype=bool)
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    for _ in range(budget):
        if not remaining:
            break
        index, (row, mask) = max(
            enumerate(remaining),
            key=lambda item: (
                int(np.count_nonzero(item[1][1] & ~covered)),
                float(item[1][0]["bbox_short_side_fraction"]),
                float(item[1][0]["depth_visible_ratio"]),
                -float(item[1][0]["frame"]),
            ),
        )
        marginal = int(np.count_nonzero(mask & ~covered))
        if marginal == 0:
            break
        covered |= mask
        selected.append(
            {
                **row,
                "marginal_visible_target_vertices": marginal,
                "cumulative_visible_target_vertices": int(np.count_nonzero(covered)),
                "cumulative_visible_target_fraction": float(np.count_nonzero(covered) / len(points)),
            }
        )
        remaining.pop(index)
    return {
        "scan_id": scan_id,
        "target_instance_id": target_id,
        "target_vertices": len(points),
        "eligible_partial_views": len(candidates),
        "portfolio_budget": budget,
        "selected": selected,
        "final_cumulative_visible_target_vertices": int(np.count_nonzero(covered)),
        "final_cumulative_visible_target_fraction": float(np.count_nonzero(covered) / len(points)),
        "opened": opened,
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    results = []
    for panel in protocol["panels"]:
        source_protocol_path = HERE / panel["source_protocol"]["path"]
        source_result_path = HERE / panel["source_result"]["path"]
        pixel.require(pixel.sha256(source_protocol_path) == panel["source_protocol"]["sha256"], f"SOURCE_PROTOCOL_HASH:{panel['id']}")
        pixel.require(pixel.sha256(source_result_path) == panel["source_result"]["sha256"], f"SOURCE_RESULT_HASH:{panel['id']}")
        source_protocol = pixel.load_json(source_protocol_path)
        source_result = pixel.load_json(source_result_path)
        for row in source_protocol["source"]["files"]:
            path = artifact_root / row["path"]
            pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
            pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
        candidate = source_result["candidate"]
        target_id = int(candidate["target_instance_id"])
        results.append(
            {
                "id": panel["id"],
                "reference": _portfolio(
                    data_root, str(candidate["reference_scan_id"]), target_id,
                    protocol["partial_view_rules"], int(protocol["portfolio_budget"]),
                ),
                "query": _portfolio(
                    data_root, str(candidate["rescan_id"]), target_id,
                    protocol["partial_view_rules"], int(protocol["portfolio_budget"]),
                ),
            }
        )
    output = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SOURCE_GEOMETRY_ONLY_MULTIVIEW_PORTFOLIO_POSTHOC_DIAGNOSTIC",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": "L10_3RSCAN_MULTIVIEW_OBSERVATION_PORTFOLIO_POSTHOC_DIAGNOSTIC_ONLY",
        "panels": results,
        "decision_threshold": None,
        "next_decision": protocol["next_decision"],
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, output)
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
