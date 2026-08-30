#!/usr/bin/env python3
"""Freeze and replay SceneNN real RGB-D metric portal transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from plyfile import PlyData
from scipy.spatial import ConvexHull, QhullError


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-scenenn-real-posed-portal-protocol-v2"
COHORT_SCHEMA = "blindassist-l10-scenenn-real-posed-portal-cohort-v2"
RESULT_SCHEMA = "blindassist-l10-scenenn-real-posed-portal-result-v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict[str, Any]:
    override = load_json(path)
    require(override.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    base_path = path.with_name(override["base_protocol_path"])
    require(base_path.is_file(), f"BASE_PROTOCOL_MISSING:{base_path}")
    require(sha256(base_path) == override["base_protocol_sha256"], "BASE_PROTOCOL_HASH_MISMATCH")
    resolved = load_json(base_path)
    resolved.update(override)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def parse_intrinsic(path: Path) -> dict[str, float | int]:
    require(path.is_file(), f"MISSING_INTRINSIC:{path}")
    values: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) >= 2:
            try:
                values[parts[0]] = float(parts[-1])
            except ValueError:
                continue
    required = ("depth_width", "depth_height", "fx", "fy", "cx", "cy")
    require(all(name in values for name in required), "INTRINSIC_FIELDS_MISSING")
    return {
        "width": int(values["depth_width"]),
        "height": int(values["depth_height"]),
        "fx": values["fx"],
        "fy": values["fy"],
        "cx": values["cx"],
        "cy": values["cy"],
    }


def intrinsic_matrix(values: dict[str, float | int]) -> np.ndarray:
    return np.array(
        [
            [float(values["fx"]), 0.0, float(values["cx"])],
            [0.0, float(values["fy"]), float(values["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def parse_poses(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"MISSING_TRAJECTORY:{path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) % 5 == 0, f"TRAJECTORY_LINE_COUNT:{len(lines)}")
    records: list[dict[str, Any]] = []
    for offset in range(0, len(lines), 5):
        header = [int(value) for value in lines[offset].split()]
        require(len(header) == 3, f"TRAJECTORY_HEADER:{lines[offset]}")
        matrix = np.array(
            [[float(value) for value in lines[offset + row].split()] for row in range(1, 5)],
            dtype=np.float64,
        )
        require(matrix.shape == (4, 4), f"TRAJECTORY_MATRIX:{header[0]}")
        require(np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]), f"TRAJECTORY_AFFINE:{header[0]}")
        records.append({"frame": header[0], "header": header, "camera_to_world": matrix})
    require([row["frame"] for row in records] == list(range(len(records))), "TRAJECTORY_FRAME_SEQUENCE")
    return records


def parse_xml_labels(path: Path) -> dict[int, dict[str, Any]]:
    require(path.is_file(), f"MISSING_XML:{path}")
    root = ET.parse(path).getroot()
    labels: dict[int, dict[str, Any]] = {}
    for element in root.iter("label"):
        instance_id = int(element.attrib["id"])
        labels[instance_id] = {
            "instance_id": instance_id,
            "text": element.attrib.get("text", ""),
            "color_rgb": [int(value) for value in element.attrib["color"].split()],
            "area": int(element.attrib.get("area", "0")),
        }
    require(labels, f"XML_LABELS_EMPTY:{path}")
    return labels


def read_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    require(path.is_file(), f"MISSING_PLY:{path}")
    vertices = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64, copy=False
    )
    labels = np.asarray(vertices["label"], dtype=np.int64)
    require(np.isfinite(xyz).all(), f"PLY_NONFINITE:{path}")
    return xyz, labels


def boundary_vertices(points: np.ndarray) -> np.ndarray:
    require(len(points) >= 4, f"TOO_FEW_TARGET_VERTICES:{len(points)}")
    try:
        indices = ConvexHull(points).vertices
        return points[np.asarray(indices, dtype=np.int64)]
    except QhullError:
        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane_xy = centered @ vh[:2].T
        indices = cv2.convexHull(
            plane_xy.astype(np.float32), returnPoints=False
        ).reshape(-1)
        return points[indices]


def project_world(points: np.ndarray, camera_to_world: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    positive = camera[:, 2] > 1e-6
    camera = camera[positive]
    if not len(camera):
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    uvw = camera @ k.T
    pixels = uvw[:, :2] / uvw[:, 2:3]
    return pixels, camera[:, 2]


def envelope_from_pixels(
    pixels: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if len(pixels) < 3 or not np.isfinite(pixels).all():
        return None
    clipped = np.column_stack(
        (
            np.clip(pixels[:, 0], 0, width - 1),
            np.clip(pixels[:, 1], 0, height - 1),
        )
    )
    contour = cv2.convexHull(np.rint(clipped).astype(np.int32)).reshape(-1, 2)
    if len(contour) < 3:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 1)
    pixel_count = int(mask.sum())
    if not pixel_count:
        return None
    ys, xs = np.where(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    stats = {
        "pixels": pixel_count,
        "image_fraction": float(pixel_count / (width * height)),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_width": int(x1 - x0),
        "bbox_height": int(y1 - y0),
        "bbox_margin": int(min(x0, y0, width - x1, height - y1)),
    }
    return mask.astype(bool), contour, stats


def mesh_envelope(
    boundary: np.ndarray,
    camera_to_world: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    pixels, _ = project_world(boundary, camera_to_world, k)
    return envelope_from_pixels(pixels, width, height)


def eligible_frame(stats: dict[str, Any], distance_m: float, rules: dict[str, Any]) -> tuple[bool, str]:
    checks = (
        (stats["pixels"] >= int(rules["minimum_projected_envelope_pixels"]), "PIXELS"),
        (stats["bbox_width"] >= int(rules["minimum_bbox_width_pixels"]), "WIDTH"),
        (stats["bbox_height"] >= int(rules["minimum_bbox_height_pixels"]), "HEIGHT"),
        (stats["image_fraction"] <= float(rules["maximum_projected_image_fraction"]), "FRACTION"),
        (distance_m <= float(rules["maximum_camera_to_target_centroid_metres"]), "RANGE"),
    )
    failures = [name for passed, name in checks if not passed]
    return not failures, "+".join(failures) if failures else "ELIGIBLE"


def select_pair(rows: list[dict[str, Any]], minimum_baseline: float) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    eligible = [row for row in rows if row["eligible"]]
    best: tuple[tuple[float, int, int, int], dict[str, Any], dict[str, Any], float] | None = None
    for index, reference in enumerate(eligible[:-1]):
        later = [row for row in eligible[index + 1 :] if row["frame"] > reference["frame"]]
        if not later:
            continue
        centres = np.array([row["camera_center_world"] for row in later], dtype=np.float64)
        distances = np.linalg.norm(centres - np.asarray(reference["camera_center_world"]), axis=1)
        if float(np.max(distances)) < minimum_baseline:
            continue
        rounded = np.round(distances, 9)
        for query_index in np.flatnonzero(rounded == np.max(rounded)).tolist():
            query = later[query_index]
            baseline = float(distances[query_index])
            key = (
                round(baseline, 9),
                min(int(reference["pixels"]), int(query["pixels"])),
                -int(reference["frame"]),
                -int(query["frame"]),
            )
            if best is None or key > best[0]:
                best = (key, reference, query, baseline)
    if best is None:
        return None
    return best[1], best[2], best[3]


def input_paths(source_root: Path, scene_id: str) -> dict[str, Path]:
    scene = source_root / "payload" / scene_id
    return {
        "ply": scene / f"{scene_id}.ply",
        "xml": scene / f"{scene_id}.xml",
        "trajectory": scene / "trajectory.log",
        "intrinsic": source_root / "payload" / "intrinsic" / "asus.ini",
    }


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def freeze_cohort(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    materialized = sorted((source_root / "payload").glob("**/*.oni"))
    materialized += sorted((source_root / "payload").glob("**/image*.png"))
    materialized += sorted((source_root / "payload").glob("**/depth*.png"))
    require(not materialized, f"RGBD_MATERIALIZED_BEFORE_FREEZE:{len(materialized)}")

    intrinsic_path = source_root / "payload" / "intrinsic" / "asus.ini"
    intrinsic = parse_intrinsic(intrinsic_path)
    require(intrinsic["width"] == protocol["provider"]["image_width"], "WIDTH_MISMATCH")
    require(intrinsic["height"] == protocol["provider"]["image_height"], "HEIGHT_MISMATCH")
    k = intrinsic_matrix(intrinsic)
    rules = protocol["pre_rgbd_selector"]["eligible_frame"]
    episodes: list[dict[str, Any]] = []
    source_manifest: dict[str, dict[str, Any]] = {}
    selection_audit: list[dict[str, Any]] = []

    for frozen in protocol["frozen_scenes"]:
        scene_id = frozen["scene_id"]
        target_id = int(frozen["target_door_instance_id"])
        paths = input_paths(source_root, scene_id)
        for key, path in paths.items():
            if key == "intrinsic" and relative(path, source_root) in source_manifest:
                continue
            require(path.is_file(), f"MISSING_SOURCE:{path}")
            source_manifest[relative(path, source_root)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        expected = frozen["files"]
        for name, row in expected.items():
            if name.endswith(".oni"):
                continue
            path = paths["trajectory"] if name == "trajectory.log" else paths["xml" if name.endswith(".xml") else "ply"]
            require(path.stat().st_size == int(row["content_length"]), f"CONTENT_LENGTH:{path}")

        labels_by_id = parse_xml_labels(paths["xml"])
        require(target_id in labels_by_id, f"TARGET_NOT_IN_XML:{scene_id}:{target_id}")
        require(labels_by_id[target_id]["text"].casefold() == "door", f"TARGET_NOT_DOOR:{scene_id}:{target_id}")
        require(labels_by_id[target_id]["color_rgb"] == frozen["target_xml_color_rgb"], f"TARGET_COLOR:{scene_id}")
        xyz, labels = read_vertices(paths["ply"])
        target_points = xyz[labels == target_id]
        require(len(target_points) > 0, f"TARGET_NOT_IN_PLY:{scene_id}:{target_id}")
        boundary = boundary_vertices(target_points)
        target_centroid = np.mean(target_points, axis=0)
        poses = parse_poses(paths["trajectory"])
        records: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        for pose_row in poses:
            pose = pose_row["camera_to_world"]
            envelope = mesh_envelope(
                boundary,
                pose,
                k,
                int(intrinsic["width"]),
                int(intrinsic["height"]),
            )
            if envelope is None:
                reasons["NO_PROJECTED_ENVELOPE"] += 1
                continue
            _, _, stats = envelope
            camera_center = pose[:3, 3]
            distance_m = float(np.linalg.norm(camera_center - target_centroid))
            is_eligible, reason = eligible_frame(stats, distance_m, rules)
            reasons[reason] += 1
            records.append(
                {
                    "frame": int(pose_row["frame"]),
                    "trajectory_header": pose_row["header"],
                    "camera_center_world": [float(value) for value in camera_center],
                    "camera_to_target_centroid_m": distance_m,
                    "eligible": is_eligible,
                    "reason": reason,
                    **stats,
                }
            )
        pair = select_pair(records, minimum_baseline=0.5)
        selection_audit.append(
            {
                "scene_id": scene_id,
                "target_door_instance_id": target_id,
                "trajectory_frames": len(poses),
                "target_ply_vertices": int(len(target_points)),
                "target_convex_boundary_vertices": int(len(boundary)),
                "eligible_frames": sum(1 for row in records if row["eligible"]),
                "frame_reason_counts": dict(sorted(reasons.items())),
                "ordered_pair_found": pair is not None,
            }
        )
        require(pair is not None, f"NOT_EVALUABLE_NO_ORDERED_PAIR:{scene_id}")
        reference, query, baseline = pair
        episodes.append(
            {
                "episode_id": frozen["episode_id"],
                "scene_id": scene_id,
                "target_door_instance_id": target_id,
                "target_xml": labels_by_id[target_id],
                "target_ply_vertices": int(len(target_points)),
                "reference": reference,
                "query": query,
                "camera_baseline_m": baseline,
                "playback_mapping": {
                    "reference": f"playback frame {reference['frame'] + 1:05d} -> sealed frame {reference['frame']:04d}",
                    "query": f"playback frame {query['frame'] + 1:05d} -> sealed frame {query['frame']:04d}",
                },
            }
        )

    require(len(episodes) == int(protocol["pre_rgbd_selector"]["cohort_size"]), "COHORT_SIZE")
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGBD_REAL_INDOOR_CONFIRMATION_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "provider": protocol["provider"],
            "input_manifest": source_manifest,
            "rgbd_files_at_freeze": 0,
        },
        "selection": protocol["pre_rgbd_selector"],
        "selection_audit": selection_audit,
        "episodes": episodes,
        "materialize_after_freeze": {
            "oni": [
                {
                    "scene_id": row["scene_id"],
                    "url": row["files"][f"{row['scene_id']}.oni"]["url"],
                    "content_length": row["files"][f"{row['scene_id']}.oni"]["content_length"],
                    "etag": row["files"][f"{row['scene_id']}.oni"]["etag"],
                }
                for row in protocol["frozen_scenes"]
            ],
            "sealed_frame_layout": "payload/<scene>/selected/{image,depth}/frame.<trajectory-index:04d>.png",
            "playback_index_rule": "Official Playback.exe output index is trajectory frame index plus one.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output_path, cohort)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
    require(len(points) >= 64, f"TOO_FEW_DEPTH_POINTS:{len(points)}")

    def fit(values: np.ndarray) -> tuple[np.ndarray, float]:
        centre = np.mean(values, axis=0)
        _, _, vh = np.linalg.svd(values - centre, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        return normal, -float(np.dot(normal, centre))

    initial_normal, initial_offset = fit(points)
    residual = np.abs(points @ initial_normal + initial_offset)
    keep_count = max(64, int(math.floor(0.70 * len(points))))
    keep = np.argpartition(residual, keep_count - 1)[:keep_count]
    normal, offset = fit(points[keep])
    final_residual = np.abs(points[keep] @ normal + offset)
    return normal, offset, {
        "input_points": int(len(points)),
        "retained_points": int(len(keep)),
        "median_residual_m": float(np.median(final_residual)),
        "p90_residual_m": float(np.quantile(final_residual, 0.90)),
    }


def backproject_mask(depth_mm: np.ndarray, mask: np.ndarray, intrinsic: dict[str, float | int]) -> np.ndarray:
    valid = mask & (depth_mm > 0)
    ys, xs = np.where(valid)
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    x = (xs.astype(np.float64) - float(intrinsic["cx"])) * z / float(intrinsic["fx"])
    y = (ys.astype(np.float64) - float(intrinsic["cy"])) * z / float(intrinsic["fy"])
    return np.column_stack((x, y, z))


def contour_plane_points(
    mask: np.ndarray,
    normal: np.ndarray,
    offset: float,
    intrinsic: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    require(contours, "REFERENCE_CONTOUR_EMPTY")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    rays = np.column_stack(
        (
            (contour[:, 0] - float(intrinsic["cx"])) / float(intrinsic["fx"]),
            (contour[:, 1] - float(intrinsic["cy"])) / float(intrinsic["fy"]),
            np.ones(len(contour), dtype=np.float64),
        )
    )
    denominator = rays @ normal
    scale = -offset / denominator
    valid = np.isfinite(scale) & (np.abs(denominator) > 1e-9) & (scale > 0.0)
    require(np.count_nonzero(valid) >= 3, "PLANE_CONTOUR_INTERSECTION_EMPTY")
    return contour[valid], rays[valid] * scale[valid, None]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask.astype(np.uint8))
    require(moments["m00"] > 0.0, "MASK_CENTROID_EMPTY")
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def mesh_plane_normal(points: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
    return vh[-1] / np.linalg.norm(vh[-1])


def selected_frame_path(source_root: Path, scene_id: str, kind: str, frame: int) -> Path:
    return source_root / "payload" / scene_id / "selected" / kind / f"frame.{frame:04d}.png"


def seal_selected_frames(
    cohort_path: Path,
    source_root: Path,
    extraction_root: Path,
    receipt_path: Path,
) -> None:
    cohort = load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    oni_by_scene = {
        row["scene_id"]: row for row in cohort["materialize_after_freeze"]["oni"]
    }
    receipt_scenes: list[dict[str, Any]] = []
    for episode in cohort["episodes"]:
        scene_id = episode["scene_id"]
        extracted = extraction_root / scene_id
        image_files = sorted((extracted / "image").glob("image*.png"))
        depth_files = sorted((extracted / "depth").glob("depth*.png"))
        timestamp_path = extracted / "timestamp.txt"
        require(timestamp_path.is_file(), f"TIMESTAMP_MISSING:{scene_id}")
        timestamp_lines = [
            line for line in timestamp_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        require(len(image_files) == len(depth_files) == len(timestamp_lines), f"PLAYBACK_COUNT_MISMATCH:{scene_id}")
        pose_count = len(parse_poses(input_paths(source_root, scene_id)["trajectory"]))
        require(pose_count == (len(image_files) // 100) * 100, f"POSE_PLAYBACK_FRAGMENT_ALIGNMENT:{scene_id}:{pose_count}:{len(image_files)}")
        oni_path = source_root / "payload" / scene_id / f"{scene_id}.oni"
        require(oni_path.is_file(), f"ONI_MISSING:{oni_path}")
        expected_oni = oni_by_scene[scene_id]
        require(oni_path.stat().st_size == int(expected_oni["content_length"]), f"ONI_SIZE:{scene_id}")
        sealed: dict[str, dict[str, Any]] = {}
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            playback_index = frame + 1
            source_image = extracted / "image" / f"image{playback_index:05d}.png"
            source_depth = extracted / "depth" / f"depth{playback_index:05d}.png"
            require(source_image.is_file() and source_depth.is_file(), f"SELECTED_PLAYBACK_FRAME_MISSING:{scene_id}:{frame}")
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
            require(image is not None and image.shape == (480, 640, 3), f"SELECTED_IMAGE_FORMAT:{scene_id}:{frame}")
            require(depth is not None and depth.shape == (480, 640) and depth.dtype == np.uint16, f"SELECTED_DEPTH_FORMAT:{scene_id}:{frame}")
            target_image = selected_frame_path(source_root, scene_id, "image", frame)
            target_depth = selected_frame_path(source_root, scene_id, "depth", frame)
            atomic_write(target_image, source_image.read_bytes())
            atomic_write(target_depth, source_depth.read_bytes())
            sealed[role] = {
                "trajectory_frame": frame,
                "playback_index": playback_index,
                "timestamp_row": timestamp_lines[playback_index - 1],
                "image_path": relative(target_image, source_root),
                "image_sha256": sha256(target_image),
                "depth_path": relative(target_depth, source_root),
                "depth_sha256": sha256(target_depth),
                "valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
            }
        receipt_scenes.append(
            {
                "scene_id": scene_id,
                "pose_frames": pose_count,
                "synchronized_playback_frames": len(image_files),
                "discarded_unposed_tail_frames": len(image_files) - pose_count,
                "oni_path": relative(oni_path, source_root),
                "oni_bytes": oni_path.stat().st_size,
                "oni_sha256": sha256(oni_path),
                "sealed": sealed,
            }
        )
    receipt = {
        "schema": "blindassist-l10-scenenn-selected-rgbd-receipt-v2",
        "authority": "POST_COHORT_FREEZE_MATERIALIZATION_RECEIPT",
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "playback_index_rule": "trajectory frame n maps to synchronized Playback.exe output n+1; each pose log is the largest complete 100-frame prefix",
        "scenes": receipt_scenes,
    }
    write_json(receipt_path, receipt)


def instance_envelopes(
    xyz: np.ndarray,
    labels: np.ndarray,
    xml_labels: dict[int, dict[str, Any]],
    camera_to_world: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
) -> dict[int, np.ndarray]:
    outputs: dict[int, np.ndarray] = {}
    for instance_id in sorted(xml_labels):
        points = xyz[labels == instance_id]
        if len(points) < 3:
            continue
        pixels, _ = project_world(points, camera_to_world, k)
        envelope = envelope_from_pixels(pixels, width, height)
        if envelope is not None:
            outputs[instance_id] = envelope[0]
    return outputs


def overlay_preview(
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_mask: np.ndarray,
    predicted_mask: np.ndarray,
    truth_mask: np.ndarray,
    episode_id: str,
) -> bytes:
    left = reference_rgb.copy()
    right = query_rgb.copy()
    left[reference_mask] = (0.45 * left[reference_mask] + 0.55 * np.array([255, 180, 0])).astype(np.uint8)
    right[predicted_mask] = (0.45 * right[predicted_mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    right[truth_mask] = (0.55 * right[truth_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.putText(left, f"{episode_id} reference credential", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "query: prediction magenta / truth green", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.concatenate((left, right), axis=1)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    require(ok, "PREVIEW_ENCODE_FAILED")
    return encoded.tobytes()


def replay(
    protocol_path: Path,
    cohort_path: Path,
    source_root: Path,
    output_path: Path,
    preview_dir: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    cohort = load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "PROTOCOL_HASH_MISMATCH")
    intrinsic_path = source_root / "payload" / "intrinsic" / "asus.ini"
    intrinsic = parse_intrinsic(intrinsic_path)
    k = intrinsic_matrix(intrinsic)
    width, height = int(intrinsic["width"]), int(intrinsic["height"])
    results: list[dict[str, Any]] = []

    for episode in cohort["episodes"]:
        scene_id = episode["scene_id"]
        target_id = int(episode["target_door_instance_id"])
        paths = input_paths(source_root, scene_id)
        for relative_path, frozen in cohort["source"]["input_manifest"].items():
            if f"payload/{scene_id}/" not in relative_path and not relative_path.endswith("asus.ini"):
                continue
            current = source_root / relative_path
            require(current.is_file(), f"MISSING_FROZEN_INPUT:{current}")
            require(current.stat().st_size == int(frozen["bytes"]), f"FROZEN_INPUT_SIZE:{current}")
            require(sha256(current) == frozen["sha256"], f"FROZEN_INPUT_HASH:{current}")
        xyz, labels = read_vertices(paths["ply"])
        xml_labels = parse_xml_labels(paths["xml"])
        target_points = xyz[labels == target_id]
        target_boundary = boundary_vertices(target_points)
        poses = {row["frame"]: row["camera_to_world"] for row in parse_poses(paths["trajectory"])}
        reference_frame = int(episode["reference"]["frame"])
        query_frame = int(episode["query"]["frame"])
        reference_pose = poses[reference_frame]
        query_pose = poses[query_frame]
        reference_envelope = mesh_envelope(target_boundary, reference_pose, k, width, height)
        query_envelope = mesh_envelope(target_boundary, query_pose, k, width, height)
        require(reference_envelope is not None and query_envelope is not None, f"FROZEN_ENVELOPE_MISSING:{scene_id}")
        reference_mask = reference_envelope[0]
        truth_mask = query_envelope[0]

        reference_depth_path = selected_frame_path(source_root, scene_id, "depth", reference_frame)
        reference_rgb_path = selected_frame_path(source_root, scene_id, "image", reference_frame)
        query_rgb_path = selected_frame_path(source_root, scene_id, "image", query_frame)
        query_depth_path = selected_frame_path(source_root, scene_id, "depth", query_frame)
        for path in (reference_depth_path, reference_rgb_path, query_rgb_path, query_depth_path):
            require(path.is_file(), f"MISSING_SELECTED_FRAME:{path}")
        depth = cv2.imread(str(reference_depth_path), cv2.IMREAD_UNCHANGED)
        reference_rgb = cv2.imread(str(reference_rgb_path), cv2.IMREAD_COLOR)
        query_rgb = cv2.imread(str(query_rgb_path), cv2.IMREAD_COLOR)
        require(depth is not None and depth.dtype == np.uint16 and depth.shape == (height, width), f"DEPTH_FORMAT:{reference_depth_path}")
        require(reference_rgb is not None and reference_rgb.shape[:2] == (height, width), f"REFERENCE_RGB_FORMAT:{reference_rgb_path}")
        require(query_rgb is not None and query_rgb.shape[:2] == (height, width), f"QUERY_RGB_FORMAT:{query_rgb_path}")

        reference_points_camera = backproject_mask(depth, reference_mask, intrinsic)
        normal_camera, offset_camera, plane_stats = fit_plane(reference_points_camera)
        _, contour_camera = contour_plane_points(reference_mask, normal_camera, offset_camera, intrinsic)
        contour_world = contour_camera @ reference_pose[:3, :3].T + reference_pose[:3, 3]
        predicted_pixels, _ = project_world(contour_world, query_pose, k)
        predicted_envelope = envelope_from_pixels(predicted_pixels, width, height)
        prediction_visible = predicted_envelope is not None
        if predicted_envelope is None:
            predicted_mask = np.zeros((height, width), dtype=bool)
            predicted_stats = {
                "pixels": 0,
                "image_fraction": 0.0,
                "bbox_xyxy": None,
                "bbox_width": 0,
                "bbox_height": 0,
                "bbox_margin": None,
            }
        else:
            predicted_mask, _, predicted_stats = predicted_envelope

        candidates = instance_envelopes(xyz, labels, xml_labels, query_pose, k, width, height)
        require(target_id in candidates, f"TARGET_NOT_VISIBLE:{scene_id}:{target_id}")
        instance_iou = {str(instance_id): mask_iou(predicted_mask, mask) for instance_id, mask in candidates.items()}
        selected_id = min(candidates, key=lambda instance_id: (-instance_iou[str(instance_id)], instance_id))
        target_iou = instance_iou[str(target_id)]
        intersection = np.count_nonzero(predicted_mask & truth_mask)
        predicted_pixels_count = np.count_nonzero(predicted_mask)
        truth_pixels_count = np.count_nonzero(truth_mask)
        truth_centroid = mask_centroid(truth_mask)
        if prediction_visible:
            predicted_centroid = mask_centroid(predicted_mask)
            centroid_error = float(np.linalg.norm(predicted_centroid - truth_centroid))
            rounded_centroid = np.rint(predicted_centroid).astype(int)
            centroid_inside = bool(
                0 <= rounded_centroid[0] < width
                and 0 <= rounded_centroid[1] < height
                and truth_mask[rounded_centroid[1], rounded_centroid[0]]
            )
        else:
            predicted_centroid = None
            centroid_error = None
            centroid_inside = False
        normal_world = reference_pose[:3, :3] @ normal_camera
        target_normal = mesh_plane_normal(target_points)
        world_centroid_error = float(np.linalg.norm(np.mean(contour_world, axis=0) - np.mean(target_points, axis=0)))

        preview_path = preview_dir / f"{episode['episode_id'].lower()}-real-rgbd-portal-transfer.jpg"
        atomic_write(
            preview_path,
            overlay_preview(
                reference_rgb,
                query_rgb,
                reference_mask,
                predicted_mask,
                truth_mask,
                episode["episode_id"],
            ),
        )
        results.append(
            {
                "episode_id": episode["episode_id"],
                "scene_id": scene_id,
                "target_door_instance_id": target_id,
                "selected_instance_id": int(selected_id),
                "selected_instance_text": xml_labels[selected_id]["text"],
                "correct_target_instance": selected_id == target_id,
                "wrong_instance_commit": selected_id != target_id,
                "visible_query_instance_count": len(candidates),
                "prediction_visible_in_query": prediction_visible,
                "prediction_collapse_reason": (
                    None
                    if prediction_visible
                    else "PROJECTED_CONTOUR_OUTSIDE_OR_DEGENERATE_IN_QUERY"
                ),
                "target_envelope_iou": target_iou,
                "target_precision": (
                    float(intersection / predicted_pixels_count)
                    if predicted_pixels_count
                    else 0.0
                ),
                "target_recall": float(intersection / truth_pixels_count),
                "centroid_error_pixels": centroid_error,
                "centroid_error_image_diagonal_fraction": (
                    float(centroid_error / math.hypot(width, height))
                    if centroid_error is not None
                    else None
                ),
                "centroid_inside_target_envelope": centroid_inside,
                "metric_world_centroid_error_m": world_centroid_error,
                "reference_plane_normal_error_degrees": angle_degrees(normal_world, target_normal),
                "reference_plane": plane_stats,
                "camera_baseline_m": float(episode["camera_baseline_m"]),
                "reference_frame": reference_frame,
                "query_frame": query_frame,
                "predicted_contour_points_in_front": int(len(predicted_pixels)),
                "predicted_envelope_pixels": int(predicted_stats["pixels"]),
                "target_truth_envelope_pixels": int(truth_pixels_count),
                "instance_envelope_iou": instance_iou,
                "preview_path": str(preview_path.resolve()),
                "input_sha256": {
                    relative(path, source_root): sha256(path)
                    for path in (
                        paths["ply"],
                        paths["xml"],
                        paths["trajectory"],
                        intrinsic_path,
                        reference_depth_path,
                        reference_rgb_path,
                        query_depth_path,
                        query_rgb_path,
                    )
                },
            }
        )

    visible_centroid_errors = [
        row["centroid_error_pixels"]
        for row in results
        if row["centroid_error_pixels"] is not None
    ]
    summary = {
        "episodes": len(results),
        "correct_target_instance": sum(row["correct_target_instance"] for row in results),
        "wrong_instance_commit": sum(row["wrong_instance_commit"] for row in results),
        "prediction_visible_in_query": sum(row["prediction_visible_in_query"] for row in results),
        "centroid_inside_target_envelope": sum(row["centroid_inside_target_envelope"] for row in results),
        "median_target_envelope_iou": float(np.median([row["target_envelope_iou"] for row in results])),
        "minimum_target_envelope_iou": float(min(row["target_envelope_iou"] for row in results)),
        "mean_centroid_error_pixels_visible_predictions": (
            float(np.mean(visible_centroid_errors)) if visible_centroid_errors else None
        ),
        "median_metric_world_centroid_error_m": float(np.median([row["metric_world_centroid_error_m"] for row in results])),
        "mean_reference_plane_normal_error_degrees": float(np.mean([row["reference_plane_normal_error_degrees"] for row in results])),
        "median_camera_baseline_m": float(np.median([row["camera_baseline_m"] for row in results])),
    }
    gates = {
        "correct_target_instance_3_of_3": summary["correct_target_instance"] == 3,
        "wrong_instance_commit_0_of_3": summary["wrong_instance_commit"] == 0,
        "centroid_inside_target_envelope_3_of_3": summary["centroid_inside_target_envelope"] == 3,
        "median_target_envelope_iou_at_least_0_5": summary["median_target_envelope_iou"] >= 0.5,
        "median_metric_world_centroid_error_m_at_most_0_25": summary["median_metric_world_centroid_error_m"] <= 0.25,
    }
    gate_met = all(gates.values())
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONFIRMATION_REAL_INDOOR_RGBD_MECHANISM_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "conclusion": (
            "L10_SCENENN_REAL_RGBD_PARTIAL_METRIC_PORTAL_TRANSFER_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_SCENENN_REAL_RGBD_PARTIAL_METRIC_PORTAL_TRANSFER_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "gates": gates,
        "summary": summary,
        "episodes": results,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, default=HERE / "l10_scenenn_real_posed_portal_protocol_v2.json")
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, default=HERE / "l10_scenenn_real_posed_portal_cohort_v2.json")
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, default=HERE / "l10_scenenn_real_posed_portal_cohort_v2.json")
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, default=HERE / "l10_scenenn_real_posed_portal_protocol_v2.json")
    replay_parser.add_argument("--cohort", type=Path, default=HERE / "l10_scenenn_real_posed_portal_cohort_v2.json")
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, default=HERE / "l10_scenenn_real_posed_portal_result_v2.json")
    replay_parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_cohort(args.protocol, args.source_root, args.output)
    elif args.command == "seal":
        seal_selected_frames(args.cohort, args.source_root, args.extraction_root, args.receipt)
    else:
        replay(args.protocol, args.cohort, args.source_root, args.output, args.preview_dir)


if __name__ == "__main__":
    main()
