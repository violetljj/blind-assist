#!/usr/bin/env python3
"""Test registered planar support on real SceneNN RGB-D target observations.

The target masks and provider trajectory are privileged Development inputs.  The
identity score and planar support predicate operate only on the depth points
actually observed in two selected RGB-D frames, not on the complete target mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_scenenn_visible_portal_transfer as visible  # noqa: E402

# Torch-backed 3RScan helpers load a different bundled CUDA major.  Force the
# already-frozen SceneNN CuPy renderer to bind its own cuBLAS first; this is a
# runtime transport order only and does not alter any scientific calculation.
import cupy as cp  # noqa: E402

_CUPY_CUBLAS_PRELOAD = cp.eye(1, dtype=cp.float32) @ cp.eye(1, dtype=cp.float32)
cp.cuda.Stream.null.synchronize()
del _CUPY_CUBLAS_PRELOAD

import l10_3rscan_open_roster_zero_assignment as open_zero  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402
import l10_3rscan_registered_surface_zero_assignment as surface  # noqa: E402


base = visible.base
PROTOCOL_SCHEMA = "blindassist-l10-scenenn-observed-extent-support-protocol-v1"
ADMISSION_SCHEMA = "blindassist-l10-scenenn-observed-extent-support-source-admission-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-observed-extent-support-cohort-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-observed-extent-support-rgbd-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-observed-extent-support-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def sha256(path: Path) -> str:
    return base.sha256(path)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    predecessor = value["predecessor"]
    for prefix in ("retained_core_protocol", "retained_core_result", "retained_core_implementation", "scenenn_negative_control_result"):
        dependency = HERE / predecessor[f"{prefix}_path"]
        require(dependency.is_file(), f"PREDECESSOR_MISSING:{dependency}")
        require(sha256(dependency) == predecessor[f"{prefix}_sha256"], f"PREDECESSOR_HASH:{dependency}")
    retained_result = load_json(HERE / predecessor["retained_core_result_path"])
    require(
        retained_result.get("conclusion") == predecessor["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    return value


def normalize_label(value: Any) -> str:
    return "".join(str(value).casefold().split())


def is_door_label(value: Any) -> bool:
    return re.fullmatch(r"door\d*", normalize_label(value)) is not None


def source_paths(source_root: Path, scene_id: str) -> dict[str, Path]:
    scene_root = source_root / "payload" / scene_id
    return {
        "ply": scene_root / f"{scene_id}.ply",
        "xml": scene_root / f"{scene_id}.xml",
        "trajectory": scene_root / "trajectory.log",
        "oni": scene_root / f"{scene_id}.oni",
        "intrinsic": source_root / "payload" / "intrinsic" / "asus.ini",
    }


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def validate_file(path: Path, expected: dict[str, Any], need_md5: bool = True) -> dict[str, Any]:
    require(path.is_file(), f"SOURCE_MISSING:{path}")
    require(path.stat().st_size == int(expected["content_length"]), f"SOURCE_LENGTH:{path}")
    receipt: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if "sha256" in expected:
        require(receipt["sha256"] == str(expected["sha256"]).casefold(), f"SOURCE_SHA256:{path}")
    if need_md5:
        digest = md5(path)
        require(digest == str(expected["md5"]).casefold(), f"SOURCE_MD5:{path}")
        receipt["md5"] = digest
    return receipt


def admit(protocol_path: Path, inventory_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    consumed = set(protocol["source_selector"]["consumed_scene_ids"])
    minimum_targets = int(protocol["source_selector"]["physical_targets"])
    audits: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for xml_path in sorted(inventory_root.glob("*.xml"), key=lambda path: path.stem):
        scene_id = xml_path.stem
        labels = base.parse_xml_labels(xml_path)
        doors = [
            {"target_instance_id": int(instance_id), **row}
            for instance_id, row in sorted(labels.items())
            if is_door_label(row.get("text", ""))
        ]
        audit = {
            "scene_id": scene_id,
            "consumed": scene_id in consumed,
            "door_instances": len(doors),
            "eligible": scene_id not in consumed and len(doors) >= minimum_targets,
        }
        audits.append(audit)
        if audit["eligible"]:
            selected = {
                "scene_id": scene_id,
                "inventory_xml_path": str(xml_path.resolve()),
                "inventory_xml_bytes": xml_path.stat().st_size,
                "inventory_xml_sha256": sha256(xml_path),
                "targets": doors[:minimum_targets],
            }
            break
    require(selected is not None, "SCENENN_FOUR_DOOR_SOURCE_NOT_EVALUABLE")
    expected_scene = str(protocol["source_selector"]["expected_first_eligible_scene"])
    require(selected["scene_id"] == expected_scene, "SOURCE_ORDER_DRIFT")
    write_json(
        output_path,
        {
            "schema": ADMISSION_SCHEMA,
            "authority": "FROZEN_PRE_GEOMETRY_PRE_RGBD_PROVIDER_DISJOINT_SOURCE_ADMISSION",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "entrypoint_path": Path(__file__).name,
            "entrypoint_sha256": sha256(Path(__file__)),
            "selection_rule": protocol["source_selector"],
            "audits_through_selection": audits,
            "selected": selected,
            "geometry_members_opened": 0,
            "rgb_members_opened": 0,
            "depth_members_opened": 0,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def frame_rows(
    renderer: visible.VisibilityRenderer,
    poses: list[dict[str, Any]],
    target_centroid: np.ndarray,
    rules: dict[str, Any],
    stride: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    sampled = poses[::stride]
    if sampled[-1]["frame"] != poses[-1]["frame"]:
        sampled.append(poses[-1])
    for pose_row in sampled:
        stats, _ = renderer.statistics(pose_row["camera_to_world"])
        camera_center = pose_row["camera_to_world"][:3, 3]
        distance = float(np.linalg.norm(camera_center - target_centroid))
        eligible, reason = visible.eligible_frame(stats, distance, rules)
        reasons[reason] += 1
        rows.append(
            {
                "frame": int(pose_row["frame"]),
                "trajectory_header": pose_row["header"],
                "camera_center_world": [float(value) for value in camera_center],
                "camera_to_target_centroid_m": distance,
                "eligible": eligible,
                "reason": reason,
                **stats,
            }
        )
    return rows, dict(sorted(reasons.items()))


def scenarios(episode_ids: list[str]) -> list[dict[str, Any]]:
    rows = [
        {"id": "closed-four", "reference_targets": episode_ids, "query_targets": episode_ids},
        {"id": "query-extra", "reference_targets": episode_ids[:-1], "query_targets": episode_ids},
        {"id": "reference-extra", "reference_targets": episode_ids, "query_targets": episode_ids[:-1]},
    ]
    for missing_reference in episode_ids:
        for missing_query in episode_ids:
            if missing_reference == missing_query:
                continue
            rows.append(
                {
                    "id": f"balanced-swap-{missing_reference}-{missing_query}",
                    "reference_targets": [value for value in episode_ids if value != missing_reference],
                    "query_targets": [value for value in episode_ids if value != missing_query],
                }
            )
    return rows


def freeze(
    protocol_path: Path,
    admission_path: Path,
    source_root: Path,
    output_path: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    admission = load_json(admission_path)
    require(admission.get("schema") == ADMISSION_SCHEMA, "ADMISSION_SCHEMA_MISMATCH")
    require(admission["protocol_sha256"] == sha256(protocol_path), "ADMISSION_PROTOCOL_HASH")
    require(admission["entrypoint_sha256"] == sha256(Path(__file__)), "ADMISSION_ENTRYPOINT_HASH")
    require(not list(source_root.glob("**/*.oni")), "RGBD_ONI_OPENED_BEFORE_COHORT_FREEZE")
    require(not list(source_root.glob("**/selected/**/*.png")), "RGBD_FRAME_OPENED_BEFORE_COHORT_FREEZE")

    scene_id = str(admission["selected"]["scene_id"])
    paths = source_paths(source_root, scene_id)
    expected_files = protocol["provider"]["files"]
    manifest = {
        relative(paths[name], source_root): validate_file(paths[name], expected_files[name])
        for name in ("ply", "xml", "trajectory")
    }
    manifest[relative(paths["intrinsic"], source_root)] = validate_file(
        paths["intrinsic"], protocol["provider"]["intrinsic"], need_md5=False
    )
    require(
        sha256(paths["xml"]) == admission["selected"]["inventory_xml_sha256"],
        "ADMISSION_XML_HASH_DRIFT",
    )
    intrinsic = base.parse_intrinsic(paths["intrinsic"])
    xyz, labels, faces = visible.read_mesh(paths["ply"])
    poses = base.parse_poses(paths["trajectory"])
    selector = protocol["pre_rgbd_selector"]
    stride = int(selector["trajectory_stride"])
    episodes: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    renderer_runtime: dict[str, Any] | None = None
    for index, frozen_target in enumerate(admission["selected"]["targets"], start=1):
        target_id = int(frozen_target["target_instance_id"])
        target_points = xyz[labels == target_id]
        require(len(target_points) >= 4, f"TARGET_POINTS_MISSING:{target_id}")
        renderer = visible.VisibilityRenderer(
            xyz, labels, faces, target_id, intrinsic, selector["renderer"]
        )
        if renderer_runtime is None:
            renderer_runtime = renderer.runtime_identity()
        rows, reasons = frame_rows(
            renderer,
            poses,
            np.mean(target_points.astype(np.float64), axis=0),
            selector["eligible_frame"],
            stride,
        )
        pair = visible.select_pair(
            rows,
            minimum_baseline=float(selector["minimum_pair_baseline_metres"]),
            minimum_gap=int(selector["minimum_pair_frame_gap"]),
        )
        require(pair is not None, f"TARGET_PAIR_NOT_EVALUABLE:{scene_id}:{target_id}")
        reference, query, baseline = pair
        pose_by_frame = {int(row["frame"]): row for row in poses}
        reference_stats, reference_mask = renderer.statistics(
            pose_by_frame[int(reference["frame"])]["camera_to_world"], return_mask=True
        )
        query_stats, query_mask = renderer.statistics(
            pose_by_frame[int(query["frame"])]["camera_to_world"], return_mask=True
        )
        require(reference_mask is not None and query_mask is not None, "SELECTED_MASK_MISSING")
        reference = {**reference, "visible_mask_sha256": visible.mask_sha256(reference_mask)}
        query = {**query, "visible_mask_sha256": visible.mask_sha256(query_mask)}
        require(reference_stats["visible_pixels"] == reference["visible_pixels"], "REFERENCE_STATS_DRIFT")
        require(query_stats["visible_pixels"] == query["visible_pixels"], "QUERY_STATS_DRIFT")
        episode_id = f"SO{index:02d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "scene_id": scene_id,
                "target_instance_id": target_id,
                "target_xml": frozen_target,
                "target_mesh_vertices": int(len(target_points)),
                "strict_target_faces": int(renderer.target_face_count),
                "reference": reference,
                "query": query,
                "camera_baseline_m": float(baseline),
            }
        )
        audits.append(
            {
                "episode_id": episode_id,
                "target_instance_id": target_id,
                "trajectory_frames": len(poses),
                "sampled_frames": len(rows),
                "eligible_frames": sum(bool(row["eligible"]) for row in rows),
                "reason_counts": reasons,
                "reference_frame": int(reference["frame"]),
                "query_frame": int(query["frame"]),
                "camera_baseline_m": float(baseline),
            }
        )
    episode_ids = [row["episode_id"] for row in episodes]
    scenario_rows = scenarios(episode_ids)
    truth_matches = sum(
        len(set(row["reference_targets"]) & set(row["query_targets"]))
        for row in scenario_rows
    )
    truth_unmatched = sum(
        len(set(row["reference_targets"]) ^ set(row["query_targets"]))
        for row in scenario_rows
    )
    selected_frames = sorted(
        {
            int(episode[role]["frame"])
            for episode in episodes
            for role in ("reference", "query")
        }
    )
    write_json(
        output_path,
        {
            "schema": COHORT_SCHEMA,
            "authority": "FROZEN_PRE_RGBD_FOUR_TARGET_REAL_OBSERVATION_COHORT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "source_admission_path": admission_path.name,
            "source_admission_sha256": sha256(admission_path),
            "entrypoint_path": Path(__file__).name,
            "entrypoint_sha256": sha256(Path(__file__)),
            "source_root": str(source_root.resolve()),
            "source_manifest": dict(sorted(manifest.items())),
            "selection": {
                "scene_id": scene_id,
                "geometry_members_opened": 3,
                "rgb_members_opened": 0,
                "depth_members_opened": 0,
                "selector": selector,
                "runtime": renderer_runtime,
                "audits": audits,
            },
            "episodes": episodes,
            "selected_trajectory_frames": selected_frames,
            "scenarios": scenario_rows,
            "counts": {
                "physical_targets": len(episodes),
                "selected_rgbd_frames": len(selected_frames),
                "scenarios": len(scenario_rows),
                "balanced_swap_scenarios": len(scenario_rows) - 3,
                "truth_matches_across_scenarios": truth_matches,
                "truth_unmatched_nodes_across_scenarios": truth_unmatched,
            },
            "materialize_after_freeze": protocol["provider"]["files"]["oni"],
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def selected_path(source_root: Path, scene_id: str, kind: str, frame: int) -> Path:
    return source_root / "payload" / scene_id / "selected" / kind / f"frame.{frame:04d}.png"


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
    scene_id = str(cohort["selection"]["scene_id"])
    frames = {int(value) for value in cohort["selected_trajectory_frames"]}
    expected_names = {f"frame.{frame:04d}.png" for frame in frames}
    image_files = sorted((extraction_root / "image").glob("*.png"))
    depth_files = sorted((extraction_root / "depth").glob("*.png"))
    require({path.name for path in image_files} == expected_names, "EXTRACTED_IMAGE_SET")
    require({path.name for path in depth_files} == expected_names, "EXTRACTED_DEPTH_SET")
    timestamps = visible.parse_selected_timestamps(extraction_root / "selected_timestamp.txt")
    summary = visible.parse_extraction_summary(extraction_root / "summary.txt")
    require(set(timestamps) == frames, "EXTRACTED_TIMESTAMP_SET")
    require(summary["requested"] == len(frames) and summary["saved"] == len(frames), "EXTRACTED_COUNT")
    paths = source_paths(source_root, scene_id)
    oni_receipt = validate_file(paths["oni"], cohort["materialize_after_freeze"])
    sealed_frames: dict[str, Any] = {}
    for frame in sorted(frames):
        source_image = extraction_root / "image" / f"frame.{frame:04d}.png"
        source_depth = extraction_root / "depth" / f"frame.{frame:04d}.png"
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
        require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{frame}")
        require(depth is not None and depth.shape == (480, 640) and depth.dtype == np.uint16, f"DEPTH_FORMAT:{frame}")
        target_image = selected_path(source_root, scene_id, "image", frame)
        target_depth = selected_path(source_root, scene_id, "depth", frame)
        base.atomic_write(target_image, source_image.read_bytes())
        base.atomic_write(target_depth, source_depth.read_bytes())
        timestamp = timestamps[frame]
        require(int(timestamp["playback_index"]) == frame + 1, f"PLAYBACK_INDEX:{frame}")
        sealed_frames[str(frame)] = {
            "trajectory_frame": frame,
            "timestamp": timestamp,
            "image_path": relative(target_image, source_root),
            "image_sha256": sha256(target_image),
            "depth_path": relative(target_depth, source_root),
            "depth_sha256": sha256(target_depth),
            "valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
        }
    write_json(
        output_path,
        {
            "schema": RECEIPT_SCHEMA,
            "authority": "POST_COHORT_FREEZE_SPARSE_RGBD_RECEIPT",
            "cohort_path": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "extractor_executable": str(extractor_exe.resolve()),
            "extractor_sha256": sha256(extractor_exe),
            "extractor_source_path": visible.EXTRACTOR_SOURCE_PATH.name,
            "extractor_source_sha256": sha256(visible.EXTRACTOR_SOURCE_PATH),
            "extraction_summary": summary,
            "oni": {"path": relative(paths["oni"], source_root), **oni_receipt},
            "sealed_frames": sealed_frames,
        },
    )


def portal_frame_y_up(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(points) >= 4, "OBSERVATION_POINTS_TOO_FEW")
    origin = np.mean(points, axis=0)
    horizontal_xz = points[:, [0, 2]] - origin[[0, 2]]
    covariance = horizontal_xz.T @ horizontal_xz / max(len(horizontal_xz), 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
        direction = -direction
    horizontal = np.array([direction[0], 0.0, direction[1]], dtype=np.float64)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return origin, horizontal, vertical


def observed_points(
    renderer: visible.VisibilityRenderer,
    pose: np.ndarray,
    depth_path: Path,
    intrinsic: dict[str, float | int],
    tolerance_m: float,
    maximum_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    require(depth is not None and depth.dtype == np.uint16 and depth.shape == (480, 640), f"DEPTH_FORMAT:{depth_path}")
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
    points_camera = base.backproject_mask(depth, valid, intrinsic)
    points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
    raw_count = len(points_world)
    require(raw_count >= 4, f"OBSERVATION_EMPTY:{depth_path}")
    if raw_count > maximum_points:
        indices = np.linspace(0, raw_count - 1, maximum_points, dtype=np.int64)
        points_world = points_world[indices]
    return np.ascontiguousarray(points_world, dtype=np.float64), {
        "renderer_visible_pixels": int(np.count_nonzero(visible_mask)),
        "valid_sensor_depth_pixels": int(np.count_nonzero(depth)),
        "depth_consistent_target_pixels": int(np.count_nonzero(valid)),
        "retained_points": int(len(points_world)),
        "depth_consistency_tolerance_m": tolerance_m,
        "visible_mask_sha256": visible.mask_sha256(visible_mask),
    }


def support_matrix(
    reference_points: list[np.ndarray], query_points: list[np.ndarray]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    size = len(reference_points)
    matrix = np.zeros((size, size), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    for row, reference in enumerate(reference_points):
        frame = portal_frame_y_up(reference)
        reference_hull = extent.convex_hull(extent.project_uv(reference, *frame))
        for column, query in enumerate(query_points):
            query_hull = extent.convex_hull(extent.project_uv(query, *frame))
            value = extent.polygon_iou(reference_hull, query_hull)
            matrix[row, column] = value
            diagnostics.append(
                {
                    "reference_index": row,
                    "query_index": column,
                    "registered_planar_extent_iou": round(value, 6),
                    "positive_support": bool(value > 0.0),
                }
            )
    return matrix, diagnostics


def score_matrix(
    reference_points: list[np.ndarray], query_points: list[np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.zeros((len(reference_points), len(query_points)), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for row, reference in enumerate(reference_points):
        for column, query in enumerate(query_points):
            distance = surface.symmetric_surface_distance(reference, query)
            matrix[row, column] = -distance
            diagnostics[f"{row}->{column}"] = {
                "symmetric_median_observed_surface_distance_metres": round(distance, 6)
            }
    return matrix, diagnostics


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
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__)), "COHORT_ENTRYPOINT_HASH")
    require(receipt["cohort_sha256"] == sha256(cohort_path), "RECEIPT_COHORT_HASH")
    scene_id = str(cohort["selection"]["scene_id"])
    paths = source_paths(source_root, scene_id)
    for relative_path, frozen in cohort["source_manifest"].items():
        path = source_root / relative_path
        require(path.is_file() and sha256(path) == frozen["sha256"], f"FROZEN_INPUT_HASH:{path}")
    intrinsic = base.parse_intrinsic(paths["intrinsic"])
    xyz, labels, faces = visible.read_mesh(paths["ply"])
    poses = {int(row["frame"]): row["camera_to_world"] for row in base.parse_poses(paths["trajectory"])}
    tolerance = float(protocol["observation_carrier"]["depth_consistency_tolerance_metres"])
    maximum_points = int(protocol["observation_carrier"]["maximum_points_per_observation"])
    minimum_points = int(protocol["observation_carrier"]["minimum_points_per_observation"])
    reference_clouds: list[np.ndarray] = []
    query_clouds: list[np.ndarray] = []
    observation_diagnostics: list[dict[str, Any]] = []
    for episode in cohort["episodes"]:
        target_id = int(episode["target_instance_id"])
        renderer = visible.VisibilityRenderer(
            xyz, labels, faces, target_id, intrinsic, protocol["pre_rgbd_selector"]["renderer"]
        )
        role_clouds: dict[str, np.ndarray] = {}
        role_diagnostics: dict[str, Any] = {}
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            depth_path = selected_path(source_root, scene_id, "depth", frame)
            image_path = selected_path(source_root, scene_id, "image", frame)
            frozen_frame = receipt["sealed_frames"][str(frame)]
            require(sha256(depth_path) == frozen_frame["depth_sha256"], f"DEPTH_HASH:{frame}")
            require(sha256(image_path) == frozen_frame["image_sha256"], f"IMAGE_HASH:{frame}")
            cloud, diagnostic = observed_points(
                renderer, poses[frame], depth_path, intrinsic, tolerance, maximum_points
            )
            require(
                diagnostic["visible_mask_sha256"] == episode[role]["visible_mask_sha256"],
                f"VISIBLE_MASK_HASH:{episode['episode_id']}:{role}",
            )
            require(len(cloud) >= minimum_points, f"OBSERVATION_POINTS_BELOW_MINIMUM:{episode['episode_id']}:{role}")
            role_clouds[role] = cloud
            role_diagnostics[role] = {"frame": frame, **diagnostic}
        reference_clouds.append(role_clouds["reference"])
        query_clouds.append(role_clouds["query"])
        observation_diagnostics.append(
            {"episode_id": episode["episode_id"], "target_instance_id": target_id, **role_diagnostics}
        )
    surface_scores, surface_diagnostics = score_matrix(reference_clouds, query_clouds)
    supports, support_diagnostics = support_matrix(reference_clouds, query_clouds)
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    target_index = {target: index for index, target in enumerate(episode_ids)}
    scenario_results: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        scores = surface_scores[np.ix_(rows, columns)]
        support = supports[np.ix_(rows, columns)]
        rank_matches = open_zero.reciprocal_zero_assignment(scores)
        support_matches = [
            (row, column)
            for row, column in rank_matches
            if float(support[row, column]) > 0.0
        ]
        scenario_results.append(
            {
                **scenario,
                "observed_surface_score_matrix": scores.round(6).tolist(),
                "observed_extent_iou_matrix": support.round(6).tolist(),
                "methods": {
                    "complete_observed_surface_hungarian": open_zero.evaluate_matches(
                        references, queries, open_zero.complete_assignment(scores)
                    ),
                    "rank_only_observed_surface_zero": open_zero.evaluate_matches(
                        references, queries, rank_matches
                    ),
                    "extent_support_observed_surface_zero": open_zero.evaluate_matches(
                        references, queries, support_matches
                    ),
                },
            }
        )
    method_names = list(scenario_results[0]["methods"])
    aggregates = {
        name: open_zero.aggregate(scenario_results, name) for name in method_names
    }
    rank_only = aggregates["rank_only_observed_surface_zero"]
    upgraded = aggregates["extent_support_observed_surface_zero"]
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
            "authority": "FRESH_PROVIDER_DISJOINT_REAL_RGBD_PARTIAL_SURFACE_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "cohort_path": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "receipt_path": receipt_path.name,
            "receipt_sha256": sha256(receipt_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "conclusion": (
                "L10_SCENENN_OBSERVED_EXTENT_SUPPORT_PROVIDER_DISJOINT_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_SCENENN_OBSERVED_EXTENT_SUPPORT_PROVIDER_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "counts": cohort["counts"],
            "observation_carrier": protocol["observation_carrier"],
            "metrics": {
                "aggregate": aggregates,
                "scenarios": scenario_results,
                "full_observed_surface_score_matrix": surface_scores.round(6).tolist(),
                "full_observed_extent_iou_matrix": supports.round(6).tolist(),
                "true_pair_extent_iou": [round(value, 6) for value in diagonal_support],
                "observation_diagnostics": observation_diagnostics,
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
            "support_rule": "strictly positive registered planar convex-hull intersection; no IoU magnitude threshold",
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    admit_parser = subparsers.add_parser("admit")
    admit_parser.add_argument("--protocol", type=Path, required=True)
    admit_parser.add_argument("--inventory-root", type=Path, required=True)
    admit_parser.add_argument("--output", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--admission", type=Path, required=True)
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
    if args.action == "admit":
        admit(args.protocol, args.inventory_root, args.output)
    elif args.action == "freeze":
        freeze(args.protocol, args.admission, args.source_root, args.output)
    elif args.action == "seal":
        seal(args.cohort, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.cohort, args.receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()
