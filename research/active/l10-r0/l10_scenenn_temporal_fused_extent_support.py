#!/usr/bin/env python3
"""Posthoc mechanism test of three-frame registered SceneNN depth fusion."""

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


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-temporal-fused-extent-support-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-temporal-fused-extent-support-cohort-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-temporal-fused-extent-support-rgbd-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-temporal-fused-extent-support-result-v1"
TRANSPORT_REPAIR_SCHEMA = "blindassist-l10-scenenn-temporal-fused-extent-support-transport-repair-v1"
TRANSPORT_REPAIR_PATH = HERE / "l10_scenenn_temporal_fused_extent_support_transport_repair_v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    parent.write_json(path, value)


def sha256(path: Path) -> str:
    return parent.sha256(path)


def authorize_entrypoint(cohort: dict[str, Any]) -> dict[str, Any] | None:
    frozen_hash = str(cohort["entrypoint_sha256"])
    current_hash = sha256(Path(__file__))
    if frozen_hash == current_hash:
        return None
    repair = load_json(TRANSPORT_REPAIR_PATH)
    require(repair.get("schema") == TRANSPORT_REPAIR_SCHEMA, "TRANSPORT_REPAIR_SCHEMA")
    require(repair.get("original_entrypoint_sha256") == frozen_hash, "TRANSPORT_REPAIR_ORIGINAL_HASH")
    require(repair.get("repaired_entrypoint_sha256") == current_hash, "TRANSPORT_REPAIR_CURRENT_HASH")
    require(repair.get("scientific_fields_changed") == [], "TRANSPORT_REPAIR_SCIENTIFIC_FIELDS")
    require(repair.get("additional_rgbd_access") == 0, "TRANSPORT_REPAIR_RGBD_ACCESS")
    return repair


def load_protocol(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    for prefix in ("protocol", "cohort", "receipt", "result", "implementation"):
        dependency = HERE / value["predecessor"][f"{prefix}_path"]
        require(dependency.is_file(), f"PREDECESSOR_MISSING:{dependency}")
        require(
            sha256(dependency) == value["predecessor"][f"{prefix}_sha256"],
            f"PREDECESSOR_HASH:{dependency}",
        )
    result = load_json(HERE / value["predecessor"]["result_path"])
    require(result.get("conclusion") == value["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    return value


def temporal_path(source_root: Path, scene_id: str, kind: str, frame: int) -> Path:
    return source_root / "payload" / scene_id / "temporal" / kind / f"frame.{frame:04d}.png"


def freeze(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    predecessor = protocol["predecessor"]
    parent_cohort = load_json(HERE / predecessor["cohort_path"])
    require(parent_cohort.get("schema") == parent.COHORT_SCHEMA, "PARENT_COHORT_SCHEMA")
    require(not list(source_root.glob("**/temporal/**/*.png")), "TEMPORAL_RGBD_OPENED_BEFORE_FREEZE")
    scene_id = str(parent_cohort["selection"]["scene_id"])
    paths = parent.source_paths(source_root, scene_id)
    intrinsic = parent.base.parse_intrinsic(paths["intrinsic"])
    xyz, labels, faces = parent.visible.read_mesh(paths["ply"])
    poses = {int(row["frame"]): row for row in parent.base.parse_poses(paths["trajectory"])}
    offsets = [int(value) for value in protocol["temporal_carrier"]["trajectory_frame_offsets"]]
    episodes: list[dict[str, Any]] = []
    all_frames: set[int] = set()
    for episode in parent_cohort["episodes"]:
        target_id = int(episode["target_instance_id"])
        renderer = parent.visible.VisibilityRenderer(
            xyz,
            labels,
            faces,
            target_id,
            intrinsic,
            protocol["renderer"],
        )
        roles: dict[str, Any] = {}
        for role in ("reference", "query"):
            anchor = int(episode[role]["frame"])
            rows: list[dict[str, Any]] = []
            for offset in offsets:
                frame = anchor + offset
                require(frame in poses, f"TEMPORAL_FRAME_OUTSIDE_TRAJECTORY:{frame}")
                stats, mask = renderer.statistics(poses[frame]["camera_to_world"], return_mask=True)
                require(mask is not None, f"TEMPORAL_MASK_MISSING:{frame}")
                rows.append(
                    {
                        "trajectory_frame": frame,
                        "offset_from_anchor": offset,
                        "visible_mask_sha256": parent.visible.mask_sha256(mask),
                        **stats,
                    }
                )
                all_frames.add(frame)
            roles[role] = {"anchor_frame": anchor, "window": rows}
        episodes.append(
            {
                "episode_id": episode["episode_id"],
                "scene_id": scene_id,
                "target_instance_id": target_id,
                **roles,
            }
        )
    write_json(
        output_path,
        {
            "schema": COHORT_SCHEMA,
            "authority": "FROZEN_PRE_ADDITIONAL_RGBD_CONSUMED_SCENE_TEMPORAL_MECHANISM_COHORT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "entrypoint_path": Path(__file__).name,
            "entrypoint_sha256": sha256(Path(__file__)),
            "parent_cohort_path": predecessor["cohort_path"],
            "parent_cohort_sha256": predecessor["cohort_sha256"],
            "scene_id": scene_id,
            "episodes": episodes,
            "selected_trajectory_frames": sorted(all_frames),
            "scenarios": parent_cohort["scenarios"],
            "counts": {
                **parent_cohort["counts"],
                "temporal_window_frames_per_role": len(offsets),
                "unique_rgbd_frames": len(all_frames),
            },
            "temporal_carrier": protocol["temporal_carrier"],
            "additional_rgb_members_opened_at_freeze": 0,
            "additional_depth_members_opened_at_freeze": 0,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def seal(
    cohort_path: Path,
    source_root: Path,
    extraction_root: Path,
    extractor_exe: Path,
    output_path: Path,
) -> None:
    cohort = load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__)), "COHORT_ENTRYPOINT_HASH")
    scene_id = str(cohort["scene_id"])
    frames = {int(value) for value in cohort["selected_trajectory_frames"]}
    expected_names = {f"frame.{frame:04d}.png" for frame in frames}
    image_files = sorted((extraction_root / "image").glob("*.png"))
    depth_files = sorted((extraction_root / "depth").glob("*.png"))
    require({path.name for path in image_files} == expected_names, "EXTRACTED_IMAGE_SET")
    require({path.name for path in depth_files} == expected_names, "EXTRACTED_DEPTH_SET")
    timestamps = parent.visible.parse_selected_timestamps(extraction_root / "selected_timestamp.txt")
    summary = parent.visible.parse_extraction_summary(extraction_root / "summary.txt")
    require(set(timestamps) == frames, "EXTRACTED_TIMESTAMP_SET")
    require(summary["requested"] == len(frames) and summary["saved"] == len(frames), "EXTRACTED_COUNT")
    sealed_frames: dict[str, Any] = {}
    for frame in sorted(frames):
        source_image = extraction_root / "image" / f"frame.{frame:04d}.png"
        source_depth = extraction_root / "depth" / f"frame.{frame:04d}.png"
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
        require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{frame}")
        require(depth is not None and depth.shape == (480, 640) and depth.dtype == np.uint16, f"DEPTH_FORMAT:{frame}")
        target_image = temporal_path(source_root, scene_id, "image", frame)
        target_depth = temporal_path(source_root, scene_id, "depth", frame)
        parent.base.atomic_write(target_image, source_image.read_bytes())
        parent.base.atomic_write(target_depth, source_depth.read_bytes())
        timestamp = timestamps[frame]
        require(int(timestamp["playback_index"]) == frame + 1, f"PLAYBACK_INDEX:{frame}")
        sealed_frames[str(frame)] = {
            "trajectory_frame": frame,
            "timestamp": timestamp,
            "image_path": parent.relative(target_image, source_root),
            "image_sha256": sha256(target_image),
            "depth_path": parent.relative(target_depth, source_root),
            "depth_sha256": sha256(target_depth),
            "valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
        }
    write_json(
        output_path,
        {
            "schema": RECEIPT_SCHEMA,
            "authority": "POST_TEMPORAL_COHORT_FREEZE_SPARSE_RGBD_RECEIPT",
            "cohort_path": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "extractor_executable": str(extractor_exe.resolve()),
            "extractor_sha256": sha256(extractor_exe),
            "extractor_source_sha256": sha256(parent.visible.EXTRACTOR_SOURCE_PATH),
            "extraction_summary": summary,
            "sealed_frames": sealed_frames,
        },
    )


def fuse_clouds(clouds: list[np.ndarray], maximum_points: int) -> tuple[np.ndarray, int]:
    fused = np.concatenate(clouds, axis=0)
    raw_count = len(fused)
    if raw_count > maximum_points:
        indices = np.linspace(0, raw_count - 1, maximum_points, dtype=np.int64)
        fused = fused[indices]
    return np.ascontiguousarray(fused, dtype=np.float64), raw_count


def observed_points_allow_empty(
    renderer: Any,
    pose: np.ndarray,
    depth_path: Path,
    intrinsic: dict[str, float | int],
    tolerance_m: float,
    maximum_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the frozen single-frame carrier while allowing a zero-point temporal contribution."""
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    require(
        depth is not None and depth.dtype == np.uint16 and depth.shape == (480, 640),
        f"DEPTH_FORMAT:{depth_path}",
    )
    visible_device, target_depth_device = renderer.visible_mask(pose)
    cp = renderer.cp
    visible_mask = cp.asnumpy(visible_device)
    target_depth = cp.asnumpy(target_depth_device)
    actual_depth = depth.astype(np.float32) / 1000.0
    valid = (
        visible_mask
        & (depth > 0)
        & np.isfinite(target_depth)
        & (np.abs(actual_depth - target_depth) <= tolerance_m)
    )
    points_camera = parent.base.backproject_mask(depth, valid, intrinsic)
    points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
    raw_count = len(points_world)
    if raw_count > maximum_points:
        indices = np.linspace(0, raw_count - 1, maximum_points, dtype=np.int64)
        points_world = points_world[indices]
    return np.ascontiguousarray(points_world, dtype=np.float64).reshape((-1, 3)), {
        "renderer_visible_pixels": int(np.count_nonzero(visible_mask)),
        "valid_sensor_depth_pixels": int(np.count_nonzero(depth)),
        "depth_consistent_target_pixels": int(np.count_nonzero(valid)),
        "retained_points": int(len(points_world)),
        "depth_consistency_tolerance_m": tolerance_m,
        "visible_mask_sha256": parent.visible.mask_sha256(visible_mask),
    }


def replay(
    protocol_path: Path,
    cohort_path: Path,
    receipt_path: Path,
    source_root: Path,
    output_path: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_HASH")
    transport_repair = authorize_entrypoint(cohort)
    require(receipt["cohort_sha256"] == sha256(cohort_path), "RECEIPT_COHORT_HASH")
    scene_id = str(cohort["scene_id"])
    paths = parent.source_paths(source_root, scene_id)
    intrinsic = parent.base.parse_intrinsic(paths["intrinsic"])
    xyz, labels, faces = parent.visible.read_mesh(paths["ply"])
    poses = {int(row["frame"]): row["camera_to_world"] for row in parent.base.parse_poses(paths["trajectory"])}
    carrier = protocol["temporal_carrier"]
    tolerance = float(carrier["depth_consistency_tolerance_metres"])
    frame_cap = int(carrier["maximum_points_per_frame"])
    fused_cap = int(carrier["maximum_points_per_fused_observation"])
    minimum = int(carrier["minimum_points_per_fused_observation"])
    reference_clouds: list[np.ndarray] = []
    query_clouds: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for episode in cohort["episodes"]:
        target_id = int(episode["target_instance_id"])
        renderer = parent.visible.VisibilityRenderer(
            xyz, labels, faces, target_id, intrinsic, protocol["renderer"]
        )
        episode_diagnostic: dict[str, Any] = {
            "episode_id": episode["episode_id"],
            "target_instance_id": target_id,
        }
        role_clouds: dict[str, np.ndarray] = {}
        for role in ("reference", "query"):
            clouds: list[np.ndarray] = []
            frame_diagnostics: list[dict[str, Any]] = []
            for frozen_frame in episode[role]["window"]:
                frame = int(frozen_frame["trajectory_frame"])
                depth_path = temporal_path(source_root, scene_id, "depth", frame)
                image_path = temporal_path(source_root, scene_id, "image", frame)
                sealed = receipt["sealed_frames"][str(frame)]
                require(sha256(depth_path) == sealed["depth_sha256"], f"DEPTH_HASH:{frame}")
                require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{frame}")
                cloud, diagnostic = observed_points_allow_empty(
                    renderer,
                    poses[frame],
                    depth_path,
                    intrinsic,
                    tolerance,
                    frame_cap,
                )
                require(
                    diagnostic["visible_mask_sha256"] == frozen_frame["visible_mask_sha256"],
                    f"VISIBLE_MASK_HASH:{episode['episode_id']}:{role}:{frame}",
                )
                clouds.append(cloud)
                frame_diagnostics.append({"frame": frame, **diagnostic})
            fused, raw_count = fuse_clouds(clouds, fused_cap)
            require(len(fused) >= minimum, f"TEMPORAL_OBSERVATION_POINTS_BELOW_MINIMUM:{episode['episode_id']}:{role}")
            role_clouds[role] = fused
            episode_diagnostic[role] = {
                "frames": frame_diagnostics,
                "raw_fused_points": raw_count,
                "retained_fused_points": int(len(fused)),
            }
        reference_clouds.append(role_clouds["reference"])
        query_clouds.append(role_clouds["query"])
        diagnostics.append(episode_diagnostic)
    scores, surface_diagnostics = parent.score_matrix(reference_clouds, query_clouds)
    supports, support_diagnostics = parent.support_matrix(reference_clouds, query_clouds)
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenario_results: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        scenario_scores = scores[np.ix_(rows, columns)]
        scenario_supports = supports[np.ix_(rows, columns)]
        rank_matches = parent.open_zero.reciprocal_zero_assignment(scenario_scores)
        support_matches = [
            (row, column)
            for row, column in rank_matches
            if float(scenario_supports[row, column]) > 0.0
        ]
        scenario_results.append(
            {
                **scenario,
                "temporal_surface_score_matrix": scenario_scores.round(6).tolist(),
                "temporal_extent_iou_matrix": scenario_supports.round(6).tolist(),
                "methods": {
                    "complete_temporal_surface_hungarian": parent.open_zero.evaluate_matches(
                        references, queries, parent.open_zero.complete_assignment(scenario_scores)
                    ),
                    "rank_only_temporal_surface_zero": parent.open_zero.evaluate_matches(
                        references, queries, rank_matches
                    ),
                    "extent_support_temporal_surface_zero": parent.open_zero.evaluate_matches(
                        references, queries, support_matches
                    ),
                },
            }
        )
    method_names = list(scenario_results[0]["methods"])
    aggregates = {
        name: parent.open_zero.aggregate(scenario_results, name) for name in method_names
    }
    rank_only = aggregates["rank_only_temporal_surface_zero"]
    upgraded = aggregates["extent_support_temporal_surface_zero"]
    expected_true = int(cohort["counts"]["truth_matches_across_scenarios"])
    diagonal_support = [float(supports[index, index]) for index in range(len(episode_ids))]
    gate_met = (
        all(value > 0.0 for value in diagonal_support)
        and upgraded["true_positive"] == expected_true
        and upgraded["false_positive"] == 0
        and upgraded["false_negative"] == 0
        and upgraded["zero_assignment_exact_scenarios"] == len(scenario_results)
        and upgraded["true_positive"] >= rank_only["true_positive"]
        and upgraded["f1"] >= rank_only["f1"]
    )
    write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_SCENE_POSTHOC_TEMPORAL_FUSION_MECHANISM_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "cohort_path": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "receipt_path": receipt_path.name,
            "receipt_sha256": sha256(receipt_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "transport_repair": (
                None
                if transport_repair is None
                else {
                    "path": TRANSPORT_REPAIR_PATH.name,
                    "sha256": sha256(TRANSPORT_REPAIR_PATH),
                    "observed_failure": transport_repair["observed_failure"],
                }
            ),
            "conclusion": (
                "L10_SCENENN_TEMPORAL_FUSED_EXTENT_SUPPORT_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_SCENENN_TEMPORAL_FUSED_EXTENT_SUPPORT_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "counts": cohort["counts"],
            "temporal_carrier": carrier,
            "metrics": {
                "aggregate": aggregates,
                "scenarios": scenario_results,
                "full_temporal_surface_score_matrix": scores.round(6).tolist(),
                "full_temporal_extent_iou_matrix": supports.round(6).tolist(),
                "true_pair_extent_iou": [round(value, 6) for value in diagonal_support],
                "observation_diagnostics": diagnostics,
                "surface_pair_diagnostics": surface_diagnostics,
                "extent_support_diagnostics": support_diagnostics,
            },
            "incremental_gain_over_rank_only": {
                "true_positive_delta": upgraded["true_positive"] - rank_only["true_positive"],
                "false_positive_reduction": rank_only["false_positive"] - upgraded["false_positive"],
                "false_negative_reduction": rank_only["false_negative"] - upgraded["false_negative"],
                "f1_delta": round(upgraded["f1"] - rank_only["f1"], 6),
                "exact_zero_assignment_scenario_gain": (
                    upgraded["zero_assignment_exact_scenarios"]
                    - rank_only["zero_assignment_exact_scenarios"]
                ),
            },
            "support_rule": "unchanged strictly positive registered planar convex-hull intersection; no IoU magnitude threshold",
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, required=True)
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.source_root, args.output)
    elif args.action == "seal":
        seal(args.cohort, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.cohort, args.receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()
