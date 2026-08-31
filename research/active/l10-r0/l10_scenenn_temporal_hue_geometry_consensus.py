#!/usr/bin/env python3
"""Posthoc threshold-free Hue and geometry reciprocal consensus on frozen SceneNN RGB-D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_observed_extent_support as parent  # noqa: E402
import l10_scenenn_temporal_fused_extent_support as temporal  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-temporal-hue-geometry-consensus-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-temporal-hue-geometry-consensus-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return parent.sha256(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    for member in ("protocol", "cohort", "receipt", "result"):
        dependency = HERE / protocol["predecessor"][f"{member}_path"]
        require(dependency.is_file(), f"PREDECESSOR_MISSING:{dependency}")
        require(
            sha256(dependency) == protocol["predecessor"][f"{member}_sha256"],
            f"PREDECESSOR_HASH:{member}",
        )
    predecessor = load_json(HERE / protocol["predecessor"]["result_path"])
    require(
        predecessor.get("conclusion") == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    return protocol


def colored_observation(
    renderer: Any,
    pose: np.ndarray,
    depth_path: Path,
    image_path: Path,
    intrinsic: dict[str, float | int],
    tolerance_m: float,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    require(depth is not None and depth.dtype == np.uint16 and depth.shape == (480, 640), f"DEPTH_FORMAT:{depth_path}")
    require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{image_path}")
    visible_device, target_depth_device = renderer.visible_mask(pose)
    visible_mask = renderer.cp.asnumpy(visible_device)
    target_depth = renderer.cp.asnumpy(target_depth_device)
    actual_depth = depth.astype(np.float32) / 1000.0
    valid = visible_mask & (depth > 0) & np.isfinite(target_depth) & (np.abs(actual_depth - target_depth) <= tolerance_m)
    points_camera = parent.base.backproject_mask(depth, valid, intrinsic)
    points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
    colors = image[valid]
    require(len(points_world) == len(colors), "COLOR_POINT_ALIGNMENT")
    raw_count = len(points_world)
    if raw_count > maximum_points:
        indices = np.linspace(0, raw_count - 1, maximum_points, dtype=np.int64)
        points_world = points_world[indices]
        colors = colors[indices]
    return (
        np.ascontiguousarray(points_world, dtype=np.float64).reshape((-1, 3)),
        np.ascontiguousarray(colors, dtype=np.uint8).reshape((-1, 3)),
        {
            "depth_consistent_target_pixels": int(np.count_nonzero(valid)),
            "retained_points": int(len(points_world)),
            "visible_mask_sha256": parent.visible.mask_sha256(visible_mask),
        },
    )


def fuse(points: list[np.ndarray], colors: list[np.ndarray], maximum_points: int) -> tuple[np.ndarray, np.ndarray]:
    fused_points = np.concatenate(points, axis=0)
    fused_colors = np.concatenate(colors, axis=0)
    require(len(fused_points) == len(fused_colors), "FUSED_COLOR_POINT_ALIGNMENT")
    if len(fused_points) > maximum_points:
        indices = np.linspace(0, len(fused_points) - 1, maximum_points, dtype=np.int64)
        fused_points = fused_points[indices]
        fused_colors = fused_colors[indices]
    return np.ascontiguousarray(fused_points), np.ascontiguousarray(fused_colors)


def hue_descriptor(colors: np.ndarray) -> np.ndarray:
    require(len(colors) > 0, "EMPTY_HUE_OBSERVATION")
    hsv = cv2.cvtColor(colors.reshape((-1, 1, 3)), cv2.COLOR_BGR2HSV).reshape((-1, 3))
    hue_bins = np.minimum(hsv[:, 0].astype(np.int64) // 10, 17)
    saturation = hsv[:, 1].astype(np.float64) / 255.0
    descriptor = np.zeros(19, dtype=np.float64)
    descriptor[:18] = np.bincount(hue_bins, weights=saturation, minlength=18)
    descriptor[18] = np.sum(1.0 - saturation)
    descriptor /= np.sum(descriptor)
    return descriptor


def hue_matrix(references: list[np.ndarray], queries: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(references), len(queries)), dtype=np.float64)
    for row, reference in enumerate(references):
        for column, query in enumerate(queries):
            matrix[row, column] = float(np.minimum(reference, query).sum())
    return matrix


def replay(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    predecessor = protocol["predecessor"]
    cohort_path = HERE / predecessor["cohort_path"]
    receipt_path = HERE / predecessor["receipt_path"]
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    predecessor_result = load_json(HERE / predecessor["result_path"])
    scene_id = str(cohort["scene_id"])
    paths = parent.source_paths(source_root, scene_id)
    intrinsic = parent.base.parse_intrinsic(paths["intrinsic"])
    xyz, labels, faces = parent.visible.read_mesh(paths["ply"])
    poses = {int(row["frame"]): row["camera_to_world"] for row in parent.base.parse_poses(paths["trajectory"])}
    carrier = cohort["temporal_carrier"]
    tolerance = float(carrier["depth_consistency_tolerance_metres"])
    frame_cap = int(carrier["maximum_points_per_frame"])
    fused_cap = int(carrier["maximum_points_per_fused_observation"])
    minimum = int(carrier["minimum_points_per_fused_observation"])
    reference_points: list[np.ndarray] = []
    query_points: list[np.ndarray] = []
    reference_hue: list[np.ndarray] = []
    query_hue: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for episode in cohort["episodes"]:
        renderer = parent.visible.VisibilityRenderer(xyz, labels, faces, int(episode["target_instance_id"]), intrinsic, protocol.get("renderer", load_json(HERE / predecessor["protocol_path"])["renderer"]))
        role_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        episode_diagnostic: dict[str, Any] = {"episode_id": episode["episode_id"]}
        for role in ("reference", "query"):
            point_frames: list[np.ndarray] = []
            color_frames: list[np.ndarray] = []
            frame_diagnostics: list[dict[str, Any]] = []
            for frozen_frame in episode[role]["window"]:
                frame = int(frozen_frame["trajectory_frame"])
                depth_path = temporal.temporal_path(source_root, scene_id, "depth", frame)
                image_path = temporal.temporal_path(source_root, scene_id, "image", frame)
                sealed = receipt["sealed_frames"][str(frame)]
                require(sha256(depth_path) == sealed["depth_sha256"], f"DEPTH_HASH:{frame}")
                require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{frame}")
                points, colors, frame_diagnostic = colored_observation(renderer, poses[frame], depth_path, image_path, intrinsic, tolerance, frame_cap)
                require(frame_diagnostic["visible_mask_sha256"] == frozen_frame["visible_mask_sha256"], f"VISIBLE_MASK_HASH:{episode['episode_id']}:{role}:{frame}")
                point_frames.append(points)
                color_frames.append(colors)
                frame_diagnostics.append({"frame": frame, **frame_diagnostic})
            points, colors = fuse(point_frames, color_frames, fused_cap)
            require(len(points) >= minimum, f"FUSED_POINTS_BELOW_MINIMUM:{episode['episode_id']}:{role}")
            role_values[role] = (points, hue_descriptor(colors))
            episode_diagnostic[role] = {"retained_fused_points": int(len(points)), "frames": frame_diagnostics}
        reference_points.append(role_values["reference"][0])
        query_points.append(role_values["query"][0])
        reference_hue.append(role_values["reference"][1])
        query_hue.append(role_values["query"][1])
        diagnostics.append(episode_diagnostic)
    geometry_scores, _ = parent.score_matrix(reference_points, query_points)
    expected_geometry = np.asarray(predecessor_result["metrics"]["full_temporal_surface_score_matrix"], dtype=np.float64)
    require(np.allclose(geometry_scores, expected_geometry, atol=5e-7, rtol=0.0), "GEOMETRY_REPRODUCTION")
    hue_scores = hue_matrix(reference_hue, query_hue)
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        geometry = geometry_scores[np.ix_(rows, columns)]
        hue = hue_scores[np.ix_(rows, columns)]
        geometry_matches = parent.open_zero.reciprocal_zero_assignment(geometry)
        hue_matches = parent.open_zero.reciprocal_zero_assignment(hue)
        hue_set = set(hue_matches)
        consensus_matches = [pair for pair in geometry_matches if pair in hue_set]
        scenarios.append({
            **scenario,
            "geometry_score_matrix": geometry.round(6).tolist(),
            "hue_histogram_intersection_matrix": hue.round(6).tolist(),
            "methods": {
                "geometry_reciprocal": parent.open_zero.evaluate_matches(references, queries, geometry_matches),
                "hue_reciprocal": parent.open_zero.evaluate_matches(references, queries, hue_matches),
                "hue_geometry_consensus": parent.open_zero.evaluate_matches(references, queries, consensus_matches),
            },
        })
    aggregates = {name: parent.open_zero.aggregate(scenarios, name) for name in scenarios[0]["methods"]}
    baseline = aggregates["geometry_reciprocal"]
    consensus = aggregates["hue_geometry_consensus"]
    expected_true = int(cohort["counts"]["truth_matches_across_scenarios"])
    gate_met = (
        consensus["true_positive"] == expected_true
        and consensus["false_positive"] == 0
        and consensus["false_negative"] == 0
        and consensus["zero_assignment_exact_scenarios"] == len(scenarios)
        and consensus["true_positive"] >= baseline["true_positive"]
        and consensus["f1"] >= baseline["f1"]
    )
    parent.write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SCENE_POSTHOC_HUE_GEOMETRY_CONSENSUS_MECHANISM_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": sha256(receipt_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "conclusion": "L10_SCENENN_TEMPORAL_HUE_GEOMETRY_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_TEMPORAL_HUE_GEOMETRY_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "metrics": {
            "aggregate": aggregates,
            "scenarios": scenarios,
            "full_geometry_score_matrix": geometry_scores.round(6).tolist(),
            "full_hue_histogram_intersection_matrix": hue_scores.round(6).tolist(),
            "reference_hue_descriptors": [value.round(8).tolist() for value in reference_hue],
            "query_hue_descriptors": [value.round(8).tolist() for value in query_hue],
            "observation_diagnostics": diagnostics,
        },
        "incremental_gain_over_geometry": {
            "true_positive_delta": consensus["true_positive"] - baseline["true_positive"],
            "false_positive_reduction": baseline["false_positive"] - consensus["false_positive"],
            "false_negative_reduction": baseline["false_negative"] - consensus["false_negative"],
            "f1_delta": round(consensus["f1"] - baseline["f1"], 6),
            "exact_scenario_gain": consensus["zero_assignment_exact_scenarios"] - baseline["zero_assignment_exact_scenarios"],
        },
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", choices=["replay"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.source_root, args.output)


if __name__ == "__main__":
    main()
