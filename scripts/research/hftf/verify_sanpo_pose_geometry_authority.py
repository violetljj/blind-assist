#!/usr/bin/env python3
"""Verify SANPO pose/frame authority and a source-derived HFTF proxy frame.

This is a source-specific Development verifier. It binds the official SANPO
loader to the local GCS object inventory, proves that pose row ``frame_num`` is
used with the same-numbered RGB/depth/mask frame, and then uses metric-depth
reprojection to identify the transform convention. A ground-plane canary may
admit a standard-body *proxy* centered on the camera ground projection.

It never authenticates a physical camera-to-person calibration, participant
body dimensions, assistive-event truth, Android behavior, or production use.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import itertools
import json
import math
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "blindassist_hftf_sanpo_pose_geometry_authority_r0"
EXPECTED_REPLAY_SCHEMA = "blindassist_sanpo_synthetic_replay_v1"
EXPECTED_OFFICIAL_ORIGIN = (
    "https://github.com/google-research-datasets/sanpo_dataset.git"
)
DEFAULT_OFFICIAL_COMMIT = "11faca999b5c223b804cd3196541a1427834918b"
DEFAULT_COMMON_SHA256 = (
    "25f93fbe61a61fff61cccf29c4bb0047cbbc120eea3f51b67c64dd123412043e"
)
COMMON_RELATIVE_PATH = Path("sanpo_dataset/lib/common.py")
OFFICIAL_CODE_MARKERS = (
    "FILENAME_RGB = '{frame_num:06d}.png'",
    "FILENAME_DEPTH = '{frame_num:06d}.float16.gz'",
    "FEATURE_CAMERA_TRANSLATIONS = 'camera_translation_in_m'",
    "FEATURE_CAMERA_QUATERNIONS = "
    "'camera_quaternions_right_handed_y_up'",
    "for frame_num in range(self.n_frames(sensor_name)):",
    "self.camera_poses(sensor_name)[",
    "FILENAME_RGB.format(frame_num=self.frame_num)",
    "FILENAME_DEPTH.format(frame_num=self.frame_num)",
    "[line['q_x'], line['q_y'], line['q_z'], line['q_w']]",
)
EXCLUDED_REPROJECTION_CLASSES = {0, 10, 11, 12, 13, 14, 21, 27}
GROUND_PROXY_CLASSES = {1, 3, 5, 6, 17, 30}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root.resolve() / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes replay root: {relative}") from exc
    return candidate


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _official_code_contract(
    repo: Path,
    expected_commit: str,
    expected_common_sha256: str,
) -> dict[str, Any]:
    common_path = repo / COMMON_RELATIVE_PATH
    result: dict[str, Any] = {
        "origin_url": None,
        "commit": None,
        "common_py_sha256": None,
        "expected_markers_present": False,
        "clean_tracked_tree": False,
        "ok": False,
        "errors": [],
    }
    try:
        origin = _git(repo, "remote", "get-url", "origin")
        commit = _git(repo, "rev-parse", "HEAD")
        dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        result["errors"].append(f"official_repo_git_error:{exc}")
        return result
    result.update(
        {
            "origin_url": origin,
            "commit": commit,
            "clean_tracked_tree": not dirty,
        }
    )
    if not common_path.is_file():
        result["errors"].append("official_common_py_missing")
        return result
    common_sha = _sha256(common_path)
    text = common_path.read_text(encoding="utf-8")
    missing_markers = [
        marker for marker in OFFICIAL_CODE_MARKERS if marker not in text
    ]
    result.update(
        {
            "common_py_sha256": common_sha,
            "expected_markers_present": not missing_markers,
            "missing_markers": missing_markers,
        }
    )
    if origin.rstrip("/") != EXPECTED_OFFICIAL_ORIGIN.rstrip("/"):
        result["errors"].append("official_repo_origin_mismatch")
    if commit != expected_commit:
        result["errors"].append("official_repo_commit_mismatch")
    if common_sha != expected_common_sha256:
        result["errors"].append("official_common_py_hash_mismatch")
    if dirty:
        result["errors"].append("official_repo_tracked_tree_dirty")
    if missing_markers:
        result["errors"].append("official_loader_contract_markers_missing")
    result["ok"] = not result["errors"]
    return result


def _fetch_gcs_metadata(object_name: str, timeout_seconds: float) -> dict[str, Any]:
    encoded = urllib.parse.quote(object_name, safe="")
    url = (
        "https://storage.googleapis.com/storage/v1/b/gresearch/o/"
        f"{encoded}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BlindAssist-HFTF-source-verifier/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GCS metadata response is not an object")
    return value


def _inventory_matches_live(
    inventory: dict[str, Any],
    live: dict[str, Any],
) -> bool:
    return all(
        (
            inventory.get("name") == live.get("name"),
            str(inventory.get("generation")) == str(live.get("generation")),
            str(inventory.get("size")) == str(live.get("size")),
            inventory.get("md5_base64") == live.get("md5Hash"),
            inventory.get("crc32c_base64") == live.get("crc32c"),
        )
    )


def _quaternion_matrix_xyzw(values: list[float]) -> np.ndarray:
    x, y, z, w = values
    norm_squared = x * x + y * y + z * z + w * w
    if not math.isfinite(norm_squared) or norm_squared <= 0:
        raise ValueError("Quaternion norm must be finite and positive")
    scale = 2.0 / norm_squared
    return np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _read_depth(path: Path, width: int, height: int) -> np.ndarray:
    values = np.frombuffer(gzip.decompress(path.read_bytes()), dtype="<f2")
    if values.size != width * height + 2:
        raise ValueError(f"Unexpected depth payload size: {path}")
    if int(values[0]) != height or int(values[1]) != width:
        raise ValueError(f"Depth header mismatch: {path}")
    depth = values[2:].astype(np.float32).reshape(height, width)
    return depth


def _read_semantic_class(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        value = np.asarray(image)
    if value.shape[:2] != (height, width):
        raise ValueError(f"Mask dimensions mismatch: {path}")
    return value[..., 0] if value.ndim == 3 else value


def _source_pose_contract(
    root: Path,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    live_description: dict[str, Any],
    live_poses: dict[str, Any],
) -> dict[str, Any]:
    inventory = spec.get("source_inventory", {})
    description_inventory = (
        inventory.get("description", {}) if isinstance(inventory, dict) else {}
    )
    pose_inventory = (
        inventory.get("camera_poses", {}) if isinstance(inventory, dict) else {}
    )
    description_path = root / "source_metadata/source_session_description.json"
    pose_path = root / "source_metadata/camera_poses.csv"
    result: dict[str, Any] = {
        "gcs_description_authenticated": False,
        "gcs_camera_poses_authenticated": False,
        "official_frame_pose_row_contract": (
            "frame_num indexes same-numbered RGB/depth/mask filenames and "
            "camera_poses[frame_num]"
        ),
        "position_unit": "meter",
        "quaternion_order": "xyzw",
        "bindings": [],
        "ok": False,
        "errors": [],
    }
    if not all(
        isinstance(item, dict)
        for item in (description_inventory, pose_inventory)
    ):
        result["errors"].append("source_inventory_missing")
        return result
    if not description_path.is_file() or not pose_path.is_file():
        result["errors"].append("source_metadata_file_missing")
        return result
    result["gcs_description_authenticated"] = (
        _inventory_matches_live(description_inventory, live_description)
        and description_path.stat().st_size
        == int(description_inventory.get("size", -1))
        and _md5_base64(description_path)
        == description_inventory.get("md5_base64")
    )
    result["gcs_camera_poses_authenticated"] = (
        _inventory_matches_live(pose_inventory, live_poses)
        and pose_path.stat().st_size == int(pose_inventory.get("size", -1))
        and _md5_base64(pose_path) == pose_inventory.get("md5_base64")
    )
    if not result["gcs_description_authenticated"]:
        result["errors"].append("description_gcs_identity_or_local_md5_failed")
    if not result["gcs_camera_poses_authenticated"]:
        result["errors"].append("camera_poses_gcs_identity_or_local_md5_failed")

    description = _load_json(description_path)
    sampling = spec.get("sampling", {})
    source_fps = (
        sampling.get("source_fps") if isinstance(sampling, dict) else None
    )
    details = description.get("session_camera_details", [])
    declared_fps = details[0].get("fps") if details else None
    if (
        not isinstance(source_fps, (int, float))
        or isinstance(source_fps, bool)
        or not math.isfinite(float(source_fps))
        or source_fps <= 0
        or float(source_fps) != float(declared_fps)
    ):
        result["errors"].append("source_fps_contract_failed")
        return result

    with pose_path.open("r", encoding="utf-8-sig", newline="") as handle:
        pose_rows = list(csv.DictReader(handle))
    selected = (
        sampling.get("selected_source_frames", [])
        if isinstance(sampling, dict)
        else []
    )
    manifest_indices = [row.get("source_frame_index") for row in rows]
    if manifest_indices != selected:
        result["errors"].append("manifest_indices_do_not_match_sampling")
    if not manifest_indices or any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(pose_rows)
        for index in manifest_indices
    ):
        result["errors"].append("pose_row_index_out_of_range")
        return result

    bindings: list[dict[str, Any]] = []
    for manifest_row, source_index in zip(rows, manifest_indices):
        pose_row = pose_rows[source_index]
        try:
            position = [
                float(pose_row[key]) for key in ("pos_x", "pos_y", "pos_z")
            ]
            quaternion = [
                float(pose_row[key]) for key in ("q_x", "q_y", "q_z", "q_w")
            ]
        except (KeyError, TypeError, ValueError):
            result["errors"].append(
                f"{manifest_row.get('id')}:pose_values_invalid"
            )
            continue
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        expected_timestamp = int(
            round(source_index * 1000.0 / float(source_fps))
        )
        modality_names = [
            manifest_row.get("modalities", {})
            .get(name, {})
            .get("name")
            for name in ("rgb", "panoptic_mask", "metric_depth")
        ]
        modality_indices_match = all(
            isinstance(name, str)
            and Path(name).name.startswith(f"{source_index:06d}.")
            for name in modality_names
        )
        binding_ok = all(
            (
                pose_row.get("tracking_state") == "TrackingState.READY",
                all(math.isfinite(value) for value in position),
                all(math.isfinite(value) for value in quaternion),
                abs(quaternion_norm - 1.0) <= 1e-3,
                manifest_row.get("source_timestamp_ms") == expected_timestamp,
                modality_indices_match,
            )
        )
        if not binding_ok:
            result["errors"].append(
                f"{manifest_row.get('id')}:frame_pose_binding_failed"
            )
        bindings.append(
            {
                "manifest_id": manifest_row.get("id"),
                "source_frame_index": source_index,
                "source_timestamp_ms": expected_timestamp,
                "raw_pose_row_index": source_index,
                "tracking_state": pose_row.get("tracking_state"),
                "position_m": position,
                "quaternion_xyzw": quaternion,
                "ok": binding_ok,
            }
        )
    result.update(
        {
            "pose_row_count": len(pose_rows),
            "binding_count": len(bindings),
            "source_fps": float(source_fps),
            "bindings": bindings,
        }
    )
    result["ok"] = (
        not result["errors"]
        and len(bindings) == len(rows)
        and all(binding["ok"] for binding in bindings)
    )
    return result


def _signed_permutation_rotations() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row_index, column_index in enumerate(permutation):
                matrix[row_index, column_index] = signs[row_index]
            if round(float(np.linalg.det(matrix))) == 1:
                matrices.append(matrix)
    return matrices


def _pose_arrays(binding: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(binding["position_m"], dtype=np.float64)
    rotation = _quaternion_matrix_xyzw(binding["quaternion_xyzw"])
    return position, rotation


def _transform_points(
    points_camera: np.ndarray,
    source_pose: tuple[np.ndarray, np.ndarray],
    target_pose: tuple[np.ndarray, np.ndarray],
    camera_basis: np.ndarray,
    orientation: str,
) -> np.ndarray:
    source_translation, source_rotation = source_pose
    target_translation, target_rotation = target_pose
    points_pose_camera = camera_basis @ points_camera
    if orientation == "R":
        points_world = (
            source_rotation @ points_pose_camera + source_translation[:, None]
        )
        target_pose_camera = target_rotation.T @ (
            points_world - target_translation[:, None]
        )
    else:
        points_world = (
            source_rotation.T @ points_pose_camera
            + source_translation[:, None]
        )
        target_pose_camera = target_rotation @ (
            points_world - target_translation[:, None]
        )
    return camera_basis.T @ target_pose_camera


def _transform_authority_canary(
    root: Path,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    sample_stride: int,
    evaluation_mode: str,
) -> dict[str, Any]:
    camera = spec.get("camera", {})
    fx, fy, cx, cy = [
        float(camera[key]) for key in ("fx", "fy", "cx", "cy")
    ]
    hypotheses = [
        (orientation, basis)
        for basis in _signed_permutation_rotations()
        for orientation in ("R", "RT")
    ]
    relative_errors: list[list[float]] = [[] for _ in hypotheses]
    valid_counts = [0 for _ in hypotheses]
    source_counts = [0 for _ in hypotheses]

    for index in range(len(rows) - 1):
        source_row = rows[index]
        target_row = rows[index + 1]
        if (
            source_row.get("session_id") != target_row.get("session_id")
            or source_row.get("sequence_id") != target_row.get("sequence_id")
        ):
            continue
        width, height = int(source_row["width"]), int(source_row["height"])
        source_depth = _read_depth(
            _resolve_inside(root, str(source_row["source_depth_path"])),
            width,
            height,
        )
        target_depth = _read_depth(
            _resolve_inside(root, str(target_row["source_depth_path"])),
            width,
            height,
        )
        source_semantic = _read_semantic_class(
            _resolve_inside(root, str(source_row["source_mask_path"])),
            width,
            height,
        )
        target_semantic = _read_semantic_class(
            _resolve_inside(root, str(target_row["source_mask_path"])),
            width,
            height,
        )
        y_grid = np.arange(sample_stride // 2, height, sample_stride)
        x_grid = np.arange(sample_stride // 2, width, sample_stride)
        u, v = np.meshgrid(x_grid, y_grid)
        z = source_depth[v, u]
        source_valid = (
            np.isfinite(z)
            & (z > 0.1)
            & (z < 80.0)
            & ~np.isin(
                source_semantic[v, u],
                list(EXCLUDED_REPROJECTION_CLASSES),
            )
        )
        u = u[source_valid].astype(np.float64)
        v = v[source_valid].astype(np.float64)
        z = z[source_valid].astype(np.float64)
        if z.size == 0:
            continue
        points = np.stack(
            ((u - cx) * z / fx, (v - cy) * z / fy, z),
            axis=0,
        )
        source_pose = _pose_arrays(bindings[index])
        target_pose = _pose_arrays(bindings[index + 1])

        for hypothesis_index, (orientation, basis) in enumerate(hypotheses):
            target_points = _transform_points(
                points,
                source_pose,
                target_pose,
                basis,
                orientation,
            )
            predicted_depth = target_points[2]
            projected_u = np.rint(
                fx * target_points[0] / predicted_depth + cx
            ).astype(np.int64)
            projected_v = np.rint(
                fy * target_points[1] / predicted_depth + cy
            ).astype(np.int64)
            inside = (
                np.isfinite(predicted_depth)
                & (predicted_depth > 0.1)
                & (projected_u >= 0)
                & (projected_u < width)
                & (projected_v >= 0)
                & (projected_v < height)
            )
            source_counts[hypothesis_index] += int(z.size)
            if not inside.any():
                continue
            observed = target_depth[
                projected_v[inside], projected_u[inside]
            ]
            target_class = target_semantic[
                projected_v[inside], projected_u[inside]
            ]
            predicted = predicted_depth[inside]
            valid = (
                np.isfinite(observed)
                & (observed > 0.1)
                & (observed < 80.0)
                & ~np.isin(
                    target_class,
                    list(EXCLUDED_REPROJECTION_CLASSES),
                )
            )
            if not valid.any():
                continue
            relative = np.abs(predicted[valid] - observed[valid]) / np.maximum(
                observed[valid], 0.1
            )
            finite = relative[np.isfinite(relative)]
            relative_errors[hypothesis_index].extend(finite.tolist())
            valid_counts[hypothesis_index] += int(finite.size)

    summaries: list[dict[str, Any]] = []
    for hypothesis_index, (orientation, basis) in enumerate(hypotheses):
        errors = np.asarray(
            relative_errors[hypothesis_index], dtype=np.float64
        )
        summaries.append(
            {
                "orientation_hypothesis": orientation,
                "camera_basis_rows": basis.astype(int).tolist(),
                "sample_count": int(errors.size),
                "coverage": (
                    valid_counts[hypothesis_index]
                    / source_counts[hypothesis_index]
                    if source_counts[hypothesis_index]
                    else 0.0
                ),
                "median_relative_depth_error": (
                    float(np.median(errors)) if errors.size else None
                ),
                "p75_relative_depth_error": (
                    float(np.quantile(errors, 0.75)) if errors.size else None
                ),
            }
        )
    ranked = sorted(
        (
            summary
            for summary in summaries
            if summary["median_relative_depth_error"] is not None
        ),
        key=lambda summary: float(summary["median_relative_depth_error"]),
    )
    best = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    canonical = next(
        (
            summary
            for summary in summaries
            if summary["orientation_hypothesis"] == "R"
            and summary["camera_basis_rows"]
            == np.eye(3, dtype=int).tolist()
        ),
        None,
    )
    canonical_rank = (
        ranked.index(canonical) + 1
        if canonical is not None and canonical in ranked
        else None
    )
    minimum_samples = max(200, (len(rows) - 1) * 20)
    best_error = (
        float(best["median_relative_depth_error"]) if best else math.inf
    )
    second_error = (
        float(second["median_relative_depth_error"]) if second else math.inf
    )
    unique_margin = (
        best is not None
        and second is not None
        and second_error - best_error >= 0.01
        and second_error >= 5.0 * max(best_error, 1e-6)
    )
    discovery_admitted = bool(
        best
        and best["sample_count"] >= minimum_samples
        and best["coverage"] >= 0.25
        and best["median_relative_depth_error"] <= 0.02
        and best["p75_relative_depth_error"] <= 0.05
        and unique_margin
    )
    canonical_replication_admitted = bool(
        canonical
        and canonical_rank == 1
        and canonical["sample_count"] >= minimum_samples
        and canonical["coverage"] >= 0.25
        and canonical["median_relative_depth_error"] <= 0.02
        and canonical["p75_relative_depth_error"] <= 0.05
    )
    admitted = (
        discovery_admitted
        if evaluation_mode == "discovery"
        else canonical_replication_admitted
    )
    return {
        "hypothesis_count": len(hypotheses),
        "sample_stride": sample_stride,
        "minimum_samples": minimum_samples,
        "evaluation_mode": evaluation_mode,
        "best": best,
        "runner_up": second,
        "frozen_canonical_hypothesis": canonical,
        "frozen_canonical_rank": canonical_rank,
        "unique_median_error_rule": {
            "absolute_gap_at_least_0_01": (
                bool(second_error - best_error >= 0.01)
                if best and second
                else False
            ),
            "runner_up_at_least_5x_best": (
                bool(second_error >= 5.0 * max(best_error, 1e-6))
                if best and second
                else False
            ),
            "ok": unique_margin,
        },
        "top_five": ranked[:5],
        "discovery_transform_direction_admitted": discovery_admitted,
        "frozen_canonical_replication_admitted": (
            canonical_replication_admitted
        ),
        "transform_direction_admitted": admitted,
        "admitted_semantics": (
            "p_world = R_xyzw @ p_opencv_camera + camera_translation_m"
            if admitted
            and (
                evaluation_mode == "frozen_canonical_replication"
                or (
                    best["orientation_hypothesis"] == "R"
                    and best["camera_basis_rows"]
                    == np.eye(3, dtype=int).tolist()
                )
            )
            else "SOURCE_DERIVED_NONCANONICAL_HYPOTHESIS"
            if admitted
            else None
        ),
        "ok": admitted,
    }


def _camera_to_world(
    points_camera: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    transform: dict[str, Any],
) -> np.ndarray:
    best = transform["best"]
    basis = np.asarray(best["camera_basis_rows"], dtype=np.float64)
    translation, rotation = pose
    pose_points = basis @ points_camera
    if best["orientation_hypothesis"] == "R":
        return rotation @ pose_points + translation[:, None]
    return rotation.T @ pose_points + translation[:, None]


def _fit_local_ground_plane(
    world: np.ndarray,
    camera_position: np.ndarray,
    seed: int,
) -> dict[str, Any] | None:
    points = world.T
    if points.shape[0] < 20:
        return None
    rng = np.random.default_rng(seed)
    candidate_count = min(384, max(64, points.shape[0] // 4))
    best_inliers: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    for _ in range(candidate_count):
        indices = rng.choice(points.shape[0], size=3, replace=False)
        a, b, c = points[indices]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            continue
        normal /= norm
        residuals = np.abs((points - a) @ normal)
        inliers = residuals <= 0.08
        count = int(inliers.sum())
        if count < 20:
            continue
        median = float(np.median(residuals[inliers]))
        score = (count, -median)
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers is None:
        return None
    minimum_inliers = max(20, math.ceil(points.shape[0] * 0.35))
    if int(best_inliers.sum()) < minimum_inliers:
        return None
    inlier_points = points[best_inliers]
    centroid = np.median(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(
        inlier_points - centroid,
        full_matrices=False,
    )
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    residuals = np.abs((points - centroid) @ normal)
    refined = residuals <= 0.10
    if int(refined.sum()) >= minimum_inliers:
        inlier_points = points[refined]
        centroid = np.median(inlier_points, axis=0)
        _, _, vh = np.linalg.svd(
            inlier_points - centroid,
            full_matrices=False,
        )
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
    signed_clearance = float((camera_position - centroid) @ normal)
    if signed_clearance < 0:
        normal = -normal
        signed_clearance = -signed_clearance
    inlier_residuals = np.abs((inlier_points - centroid) @ normal)
    projection = camera_position - signed_clearance * normal
    return {
        "input_sample_count": int(points.shape[0]),
        "inlier_count": int(inlier_points.shape[0]),
        "inlier_fraction": float(inlier_points.shape[0] / points.shape[0]),
        "normal_toward_camera": normal.tolist(),
        "plane_point_m": centroid.tolist(),
        "median_inlier_residual_m": float(np.median(inlier_residuals)),
        "p75_inlier_residual_m": float(
            np.quantile(inlier_residuals, 0.75)
        ),
        "camera_clearance_m": signed_clearance,
        "camera_ground_projection_m": projection.tolist(),
    }


def _ground_proxy_canary(
    root: Path,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    transform: dict[str, Any],
    sample_stride: int,
) -> dict[str, Any]:
    camera = spec["camera"]
    fx, fy, cx, cy = [
        float(camera[key]) for key in ("fx", "fy", "cx", "cy")
    ]
    frame_axis_rows: list[list[dict[str, Any]]] = [[], [], []]
    per_frame: list[dict[str, Any]] = []
    for row, binding in zip(rows, bindings):
        width, height = int(row["width"]), int(row["height"])
        depth = _read_depth(
            _resolve_inside(root, str(row["source_depth_path"])),
            width,
            height,
        )
        semantic = _read_semantic_class(
            _resolve_inside(root, str(row["source_mask_path"])),
            width,
            height,
        )
        y_grid = np.arange(int(height * 0.55), height, sample_stride)
        x_grid = np.arange(sample_stride // 2, width, sample_stride)
        u, v = np.meshgrid(x_grid, y_grid)
        z = depth[v, u]
        valid = (
            np.isfinite(z)
            & (z >= 0.5)
            & (z <= 8.0)
            & np.isin(semantic[v, u], list(GROUND_PROXY_CLASSES))
        )
        u = u[valid].astype(np.float64)
        v = v[valid].astype(np.float64)
        z = z[valid].astype(np.float64)
        if z.size < 20:
            continue
        points = np.stack(
            ((u - cx) * z / fx, (v - cy) * z / fy, z),
            axis=0,
        )
        pose = _pose_arrays(binding)
        world = _camera_to_world(points, pose, transform)
        frame_result: dict[str, Any] = {
            "manifest_id": row.get("id"),
            "ground_sample_count": int(z.size),
            "axes": {},
        }
        for axis in range(3):
            ground_coordinate = float(np.median(world[axis]))
            mad = float(
                np.median(np.abs(world[axis] - ground_coordinate))
            )
            raw_height = float(pose[0][axis] - ground_coordinate)
            axis_row = {
                "ground_coordinate": ground_coordinate,
                "ground_mad_m": mad,
                "signed_camera_clearance_m": raw_height,
            }
            frame_axis_rows[axis].append(axis_row)
            frame_result["axes"][str(axis)] = axis_row
        plane = _fit_local_ground_plane(
            world,
            pose[0],
            int(binding["source_frame_index"]),
        )
        frame_result["local_ground_plane"] = plane
        per_frame.append(frame_result)

    axis_summaries: list[dict[str, Any]] = []
    for axis, rows_for_axis in enumerate(frame_axis_rows):
        clearances = np.asarray(
            [
                abs(row["signed_camera_clearance_m"])
                for row in rows_for_axis
            ],
            dtype=np.float64,
        )
        signed = np.asarray(
            [
                row["signed_camera_clearance_m"]
                for row in rows_for_axis
            ],
            dtype=np.float64,
        )
        mads = np.asarray(
            [row["ground_mad_m"] for row in rows_for_axis],
            dtype=np.float64,
        )
        axis_summaries.append(
            {
                "axis_index": axis,
                "frame_count": len(rows_for_axis),
                "median_ground_mad_m": (
                    float(np.median(mads)) if mads.size else None
                ),
                "median_camera_clearance_m": (
                    float(np.median(clearances))
                    if clearances.size
                    else None
                ),
                "camera_clearance_iqr_m": (
                    float(np.quantile(clearances, 0.75))
                    - float(np.quantile(clearances, 0.25))
                    if clearances.size
                    else None
                ),
                "positive_axis_points_up": (
                    bool(float(np.median(signed)) > 0)
                    if signed.size
                    else None
                ),
            }
        )
    planes = [
        frame["local_ground_plane"]
        for frame in per_frame
        if frame.get("local_ground_plane") is not None
    ]
    normals = np.asarray(
        [plane["normal_toward_camera"] for plane in planes],
        dtype=np.float64,
    )
    clearances = np.asarray(
        [plane["camera_clearance_m"] for plane in planes],
        dtype=np.float64,
    )
    residuals = np.asarray(
        [plane["median_inlier_residual_m"] for plane in planes],
        dtype=np.float64,
    )
    alignments = (
        np.median(np.abs(normals), axis=0)
        if normals.size
        else np.zeros(3, dtype=np.float64)
    )
    ranked_axes = np.argsort(-alignments)
    best_axis = int(ranked_axes[0])
    alignment_margin = float(
        alignments[ranked_axes[0]] - alignments[ranked_axes[1]]
    )
    required_frames = max(3, math.ceil(len(rows) * 0.8))
    plane_contract_ok = bool(
        len(planes) >= required_frames
        and alignments[best_axis] >= 0.85
        and alignment_margin >= 0.20
        and residuals.size
        and float(np.median(residuals)) <= 0.05
        and clearances.size
        and 0.5 <= float(np.median(clearances)) <= 2.5
        and (
            float(np.quantile(clearances, 0.75))
            - float(np.quantile(clearances, 0.25))
        )
        <= 0.40
    )
    chosen = (
        {
            "axis_index": best_axis,
            "frame_count": len(planes),
            "median_axis_alignment": float(alignments[best_axis]),
            "axis_alignment_margin": alignment_margin,
            "median_ground_mad_m": float(np.median(residuals)),
            "median_camera_clearance_m": float(
                np.median(clearances)
            ),
            "camera_clearance_iqr_m": float(
                np.quantile(clearances, 0.75)
                - np.quantile(clearances, 0.25)
            ),
            "positive_axis_points_up": bool(
                float(np.median(normals[:, best_axis])) > 0
            ),
        }
        if plane_contract_ok
        else None
    )
    admitted = chosen is not None
    axis_names = ("X", "Y", "Z")
    vertical = (
        (
            "+"
            if chosen["positive_axis_points_up"]
            else "-"
        )
        + axis_names[chosen["axis_index"]]
        if chosen
        else None
    )
    return {
        "ground_class_ids": sorted(GROUND_PROXY_CLASSES),
        "sample_stride": sample_stride,
        "frame_count_with_ground": len(per_frame),
        "coordinate_axis_summaries": axis_summaries,
        "local_ground_plane_frame_count": len(planes),
        "local_ground_normal_axis_alignments": alignments.tolist(),
        "eligible_axis_count": 1 if plane_contract_ok else 0,
        "vertical_axis": vertical,
        "chosen_axis": chosen,
        "per_frame": per_frame,
        "physical_camera_to_body_calibration_admitted": False,
        "standard_body_proxy_center_rule": (
            "camera world position orthogonally projected onto the "
            "source-derived per-frame local ground plane"
            if admitted
            else None
        ),
        "standard_body_proxy_frame_admitted_for_h1": admitted,
        "ok": admitted,
    }


def audit(
    replay_root: Path,
    official_repo: Path,
    *,
    expected_official_commit: str,
    expected_common_sha256: str,
    live_description_metadata: dict[str, Any],
    live_pose_metadata: dict[str, Any],
    reprojection_stride: int = 64,
    ground_stride: int = 16,
    evaluation_mode: str = "discovery",
) -> dict[str, Any]:
    if evaluation_mode not in (
        "discovery",
        "frozen_canonical_replication",
    ):
        raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")
    root = replay_root.resolve()
    spec = _load_json(root / "dataset_spec.json")
    rows = _load_jsonl(root / "manifest.replay.jsonl")
    if spec.get("schema") != EXPECTED_REPLAY_SCHEMA or not rows:
        raise ValueError("Replay schema is unsupported or manifest is empty")
    session_ids = sorted(
        {
            str(row.get("session_id"))
            for row in rows
            if isinstance(row.get("session_id"), str)
            and row.get("session_id")
        }
    )
    if len(session_ids) != 1:
        raise ValueError("Exactly one non-empty source session is required")

    official = _official_code_contract(
        official_repo.resolve(),
        expected_official_commit,
        expected_common_sha256,
    )
    source = _source_pose_contract(
        root,
        spec,
        rows,
        live_description_metadata,
        live_pose_metadata,
    )
    transform = (
        _transform_authority_canary(
            root,
            spec,
            rows,
            source["bindings"],
            reprojection_stride,
            evaluation_mode,
        )
        if official["ok"] and source["ok"]
        else {"ok": False, "transform_direction_admitted": False}
    )
    ground = (
        _ground_proxy_canary(
            root,
            spec,
            rows,
            source["bindings"],
            transform,
            ground_stride,
        )
        if transform["ok"]
        else {
            "ok": False,
            "physical_camera_to_body_calibration_admitted": False,
            "standard_body_proxy_frame_admitted_for_h1": False,
        }
    )

    mapping_admitted = official["ok"] and source["ok"]
    proxy_frame_admitted = mapping_admitted and transform["ok"] and ground["ok"]
    if proxy_frame_admitted:
        terminal = (
            "HFTF_H0_1_SANPO_PROXY_FRAME_ADMITTED"
            if evaluation_mode == "discovery"
            else "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED"
        )
    elif mapping_admitted:
        terminal = (
            "HFTF_H0_1_POSE_MAPPING_ONLY"
            if evaluation_mode == "discovery"
            else "HFTF_H0_2_CANONICAL_PROXY_NOT_REPLICATED"
        )
    else:
        terminal = "HFTF_H0_1_SOURCE_AUTHORITY_NOT_EVALUABLE"

    official_feature_label = "right_handed_y_up"
    vertical_axis = ground.get("vertical_axis")
    label_conflict = bool(
        vertical_axis and vertical_axis not in ("+Y", "-Y")
    )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evaluation_mode": evaluation_mode,
        "claim_ceiling": "SOURCE_SPECIFIC_GEOMETRY_PROXY_ONLY",
        "source_session_ids": session_ids,
        "manifest_frame_count": len(rows),
        "mainline_changed": False,
        "default_app_changed": False,
        "input_hashes": {
            "verifier_sha256": _sha256(Path(__file__).resolve()),
            "dataset_spec_sha256": _sha256(root / "dataset_spec.json"),
            "manifest_sha256": _sha256(root / "manifest.replay.jsonl"),
            "camera_poses_sha256": _sha256(
                root / "source_metadata/camera_poses.csv"
            ),
        },
        "official_loader_authority": official,
        "source_pose_authority": source,
        "transform_direction_canary": transform,
        "ground_and_body_proxy_canary": ground,
        "coordinate_label_reconciliation": {
            "official_feature_label": official_feature_label,
            "source_derived_vertical_axis": vertical_axis,
            "label_conflict_for_this_replay": label_conflict,
            "decision": (
                "USE_SOURCE_DERIVED_VERTICAL_AXIS_FOR_THIS_DEVELOPMENT_"
                "EVIDENCE_VERSION_ONLY"
                if label_conflict and proxy_frame_admitted
                else "NO_CONFLICT"
                if proxy_frame_admitted
                else "NOT_EVALUABLE"
            ),
            "not_allowed": (
                "Do not generalize the source-derived axis to other SANPO "
                "versions or claim the official feature label is globally wrong."
            ),
        },
        "capability_decisions": {
            "official_frame_pose_row_mapping": (
                "ELIGIBLE" if mapping_admitted else "NOT_EVALUABLE"
            ),
            "source_derived_transform_direction": (
                "ELIGIBLE" if transform["ok"] else "NOT_EVALUABLE"
            ),
            "source_derived_ground_proxy_frame": (
                "ELIGIBLE" if ground["ok"] else "NOT_EVALUABLE"
            ),
            "standard_body_proxy_for_h1_geometry_mechanics": (
                "ELIGIBLE" if proxy_frame_admitted else "NOT_EVALUABLE"
            ),
            "physical_camera_to_person_calibration": "NOT_EVALUABLE",
            "participant_specific_body_dimensions": "NOT_EVALUABLE",
            "student_or_event_effect": "NOT_EVALUABLE",
        },
        "allowed_next_step": (
            (
                "H0_2_INDEPENDENT_SANPO_SYNTHETIC_SESSION_EXPANSION"
                if evaluation_mode == "discovery"
                else "H0_2_COHORT_AGGREGATION"
            )
            if proxy_frame_admitted
            else "REPAIR_SOURCE_MAPPING_OR_PROXY_FRAME_BEFORE_H0_2"
        ),
        "prohibited_inferences": [
            "source-derived proxy is physical participant calibration",
            "geometry proxy is human collision or safety truth",
            "teacher mechanics proves student effect",
            "research mainline promotion",
            "Android or production authorization",
        ],
    }


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument(
        "--expected-official-commit",
        default=DEFAULT_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-common-sha256",
        default=DEFAULT_COMMON_SHA256,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--reprojection-stride", type=int, default=64)
    parser.add_argument("--ground-stride", type=int, default=16)
    parser.add_argument(
        "--evaluation-mode",
        choices=("discovery", "frozen_canonical_replication"),
        default="discovery",
    )
    args = parser.parse_args()
    try:
        if (
            args.network_timeout_seconds <= 0
            or args.reprojection_stride <= 0
            or args.ground_stride <= 0
        ):
            raise ValueError("Timeout and strides must be positive")
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        spec = _load_json(args.replay_root / "dataset_spec.json")
        inventory = spec.get("source_inventory", {})
        if not isinstance(inventory, dict):
            raise ValueError("source_inventory is missing")
        description_inventory = inventory.get("description", {})
        pose_inventory = inventory.get("camera_poses", {})
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            for item in (description_inventory, pose_inventory)
        ):
            raise ValueError("Description/pose source inventory is incomplete")
        live_description = _fetch_gcs_metadata(
            description_inventory["name"],
            args.network_timeout_seconds,
        )
        live_poses = _fetch_gcs_metadata(
            pose_inventory["name"],
            args.network_timeout_seconds,
        )
        report = audit(
            args.replay_root,
            args.official_repo,
            expected_official_commit=args.expected_official_commit,
            expected_common_sha256=args.expected_common_sha256,
            live_description_metadata=live_description,
            live_pose_metadata=live_poses,
            reprojection_stride=args.reprojection_stride,
            ground_stride=args.ground_stride,
            evaluation_mode=args.evaluation_mode,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "allowed_next_step": report["allowed_next_step"],
                    "output": str(output),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
