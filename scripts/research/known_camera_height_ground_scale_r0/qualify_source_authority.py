"""Qualify the frozen gravity-plane height proxy without running DA or effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import core
from download_locked_assets import timestamp_from_stem


WORLD_VERTICAL = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
MAXIMUM_POSE_DIFFERENCE_S = 0.05
DEPTH_RANGE_M = (0.25, 6.0)
HEIGHT_RANGE_M = (0.80, 2.20)
HISTOGRAM_BIN_WIDTH_M = 0.04
SUPPORT_HALF_WIDTH_M = 0.08
MINIMUM_VALID_PROXY_FRAMES_PER_PARENT = 90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_intrinsics(path: Path, width: int, height: int) -> np.ndarray:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    if len(values) != 6:
        raise ValueError(f"invalid pincam: {path}")
    source_width, source_height, fx, fy, cx, cy = values
    if int(source_width) != width or int(source_height) != height:
        raise ValueError("intrinsics/image size mismatch")
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if not core.validate_intrinsics(matrix, width, height):
        raise ValueError("invalid intrinsics values")
    return matrix


def read_trajectory(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split()
        if len(tokens) != 7:
            raise ValueError(f"{path}:{line_number}: expected seven columns")
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}:{line_number}: non-finite trajectory")
        rotation_world_to_pose, _ = cv2.Rodrigues(values[1:4])
        extrinsics = np.eye(4, dtype=np.float64)
        extrinsics[:3, :3] = rotation_world_to_pose
        extrinsics[:3, 3] = values[4:7]
        camera_to_world = np.linalg.inv(extrinsics)
        up_camera = camera_to_world[:3, :3].T @ WORLD_VERTICAL
        up_camera /= np.linalg.norm(up_camera)
        rows.append((float(values[0]), up_camera))
    if not rows:
        raise ValueError("empty trajectory")
    return sorted(rows, key=lambda row: row[0])


def nearest_up_camera(
    timestamp: float, trajectory: list[tuple[float, np.ndarray]]
) -> tuple[np.ndarray | None, str | None, float | None]:
    differences = [(abs(value - timestamp), value, up) for value, up in trajectory]
    difference, _, up = min(differences, key=lambda row: (row[0], row[1]))
    if difference > MAXIMUM_POSE_DIFFERENCE_S:
        return None, "POSE_TIMESTAMP_OUT_OF_RANGE", difference
    if sum(abs(value - timestamp) == difference for value, _ in trajectory) != 1:
        return None, "POSE_TIMESTAMP_AMBIGUOUS", difference
    return up, None, difference


def fit_gravity_plane_height_proxy(
    depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    up_camera: np.ndarray,
) -> dict[str, object]:
    depth = np.asarray(depth_m, dtype=np.float64)
    conf = np.asarray(confidence)
    if depth.shape != conf.shape or depth.ndim != 2:
        return {"status": "UNKNOWN", "reason": "DEPTH_CONFIDENCE_SHAPE_MISMATCH"}
    usable = (
        np.isfinite(depth)
        & (depth >= DEPTH_RANGE_M[0])
        & (depth <= DEPTH_RANGE_M[1])
        & (conf == 2)
    )
    filtered = np.where(usable, depth, 0.0)
    points, pixels = core.relative_depth_to_points(filtered, intrinsics, stride=4)
    candidates = points[
        pixels[:, 1] >= core.LOWER_ROI_START_FRACTION * depth.shape[0]
    ]
    if len(candidates) < core.MINIMUM_CANDIDATES:
        return {"status": "UNKNOWN", "reason": "INSUFFICIENT_PROXY_CANDIDATES"}
    up = np.asarray(up_camera, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm <= 0.0:
        return {"status": "UNKNOWN", "reason": "INVALID_TRAJECTORY_UP"}
    up /= norm
    offsets = -(candidates @ up)
    valid = (
        np.isfinite(offsets)
        & (offsets >= HEIGHT_RANGE_M[0])
        & (offsets <= HEIGHT_RANGE_M[1])
    )
    required = max(
        core.MINIMUM_INLIERS,
        int(np.ceil(core.MINIMUM_INLIER_FRACTION * len(candidates))),
    )
    if int(np.sum(valid)) < required:
        return {"status": "UNKNOWN", "reason": "INSUFFICIENT_PROXY_HEIGHT_SUPPORT"}
    bins = np.arange(
        HEIGHT_RANGE_M[0],
        HEIGHT_RANGE_M[1] + HISTOGRAM_BIN_WIDTH_M + 1e-9,
        HISTOGRAM_BIN_WIDTH_M,
    )
    counts, edges = np.histogram(offsets[valid], bins=bins)
    if not len(counts) or int(np.max(counts)) < core.MINIMUM_INLIERS:
        return {"status": "UNKNOWN", "reason": "NO_PROXY_HEIGHT_MODE"}
    mode = int(np.argmax(counts))
    mode_center = float((edges[mode] + edges[mode + 1]) / 2.0)
    support = valid & (np.abs(offsets - mode_center) <= SUPPORT_HALF_WIDTH_M)
    count = int(np.sum(support))
    if count < required:
        return {"status": "UNKNOWN", "reason": "INSUFFICIENT_PROXY_MODE_SUPPORT"}
    height = float(np.median(offsets[support]))
    normalized_residual = float(
        np.median(np.abs(offsets[support] - height)) / height
    )
    if normalized_residual > core.MAXIMUM_NORMALIZED_PLANE_RESIDUAL:
        return {"status": "UNKNOWN", "reason": "PROXY_RESIDUAL_REJECTED"}
    return {
        "status": "VALID",
        "height_proxy_m": height,
        "normalized_median_residual": normalized_residual,
        "candidate_count": len(candidates),
        "support_count": count,
        "support_fraction": count / len(candidates),
        "up_camera": up.tolist(),
    }


def _by_stem(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        stem = Path(row["path"]).stem
        if stem in result:
            raise ValueError(f"duplicate extracted stem: {stem}")
        result[stem] = row
    return result


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--roster-lock", required=True, type=Path)
    parser.add_argument("--media-manifest", required=True, type=Path)
    parser.add_argument("--authority-amendment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.media_manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != sha256(arguments.protocol):
        raise ValueError("media protocol mismatch")
    if manifest.get("roster_lock_sha256") != sha256(arguments.roster_lock):
        raise ValueError("media roster mismatch")
    amendment = json.loads(arguments.authority_amendment.read_text(encoding="utf-8"))
    if amendment.get("status") != "FROZEN_BEFORE_DA_OR_EFFECT_OUTCOME_AUTHORITY_DOWNGRADED":
        raise ValueError("authority amendment is not frozen")

    parent_results = []
    all_records = []
    for video in manifest["videos"]:
        maps = {
            name: _by_stem(video["extracted"][name])
            for name in ("lowres_wide", "lowres_depth", "confidence")
        }
        intrinsics_rows = video["extracted"]["lowres_wide_intrinsics"]
        intrinsics_candidates = sorted(
            (timestamp_from_stem(Path(row["path"]).stem), row)
            for row in intrinsics_rows
        )
        trajectory_receipt = next(
            row for row in video["source_assets"] if row["asset"] == "lowres_wide.traj"
        )
        trajectory_path = (
            Path(maps["lowres_wide"][video["selected_frame_stems"][0]]["path"])
            .parent.parent
            / "lowres_wide.traj"
        )
        if sha256(trajectory_path) != trajectory_receipt["archive_sha256"]:
            raise ValueError("trajectory hash mismatch")
        trajectory = read_trajectory(trajectory_path)
        records = []
        for stem in video["selected_frame_stems"]:
            rows = {name: maps[name][stem] for name in maps}
            for row in rows.values():
                path = Path(row["path"])
                if sha256(path) != row["sha256"]:
                    raise ValueError(f"extracted hash mismatch: {path}")
            rgb = cv2.imread(rows["lowres_wide"]["path"], cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(rows["lowres_depth"]["path"], cv2.IMREAD_UNCHANGED)
            confidence = cv2.imread(rows["confidence"]["path"], cv2.IMREAD_UNCHANGED)
            if rgb is None or depth_raw is None or confidence is None:
                raise ValueError("media decode failed")
            if rgb.shape[:2] != depth_raw.shape or depth_raw.shape != confidence.shape:
                raise ValueError("RGB/depth/confidence shape mismatch")
            timestamp = timestamp_from_stem(stem)
            difference, intrinsics_row = min(
                (
                    abs(candidate_timestamp - timestamp),
                    row,
                )
                for candidate_timestamp, row in intrinsics_candidates
            )
            if difference > 0.0015:
                raise ValueError("intrinsics timestamp out of range")
            intrinsics_path = Path(intrinsics_row["path"])
            if sha256(intrinsics_path) != intrinsics_row["sha256"]:
                raise ValueError("intrinsics hash mismatch")
            intrinsics = read_intrinsics(
                intrinsics_path, depth_raw.shape[1], depth_raw.shape[0]
            )
            up_camera, pose_reason, pose_difference = nearest_up_camera(
                timestamp, trajectory
            )
            if up_camera is None:
                proxy = {"status": "UNKNOWN", "reason": pose_reason}
            else:
                proxy = fit_gravity_plane_height_proxy(
                    depth_raw.astype(np.float64) / 1000.0,
                    confidence,
                    intrinsics,
                    up_camera,
                )
            record = {
                "visit_id": video["visit_id"],
                "video_id": video["video_id"],
                "frame_stem": stem,
                "pose_timestamp_difference_s": pose_difference,
                "proxy": proxy,
            }
            records.append(record)
            all_records.append(record)
        valid = [row for row in records if row["proxy"]["status"] == "VALID"]
        reasons: dict[str, int] = {}
        for row in records:
            if row["proxy"]["status"] != "VALID":
                reason = str(row["proxy"]["reason"])
                reasons[reason] = reasons.get(reason, 0) + 1
        parent_results.append(
            {
                "visit_id": video["visit_id"],
                "video_id": video["video_id"],
                "frame_count": len(records),
                "valid_proxy_frame_count": len(valid),
                "valid_proxy_fraction": len(valid) / len(records),
                "unknown_reason_counts": reasons,
                "height_proxy_median": float(
                    np.median([row["proxy"]["height_proxy_m"] for row in valid])
                )
                if valid
                else None,
                "height_proxy_iqr": float(
                    np.quantile(
                        [row["proxy"]["height_proxy_m"] for row in valid], 0.75
                    )
                    - np.quantile(
                        [row["proxy"]["height_proxy_m"] for row in valid], 0.25
                    )
                )
                if valid
                else None,
                "source_authority_passed": len(valid)
                >= MINIMUM_VALID_PROXY_FRAMES_PER_PARENT,
            }
        )
    passed = all(row["source_authority_passed"] for row in parent_results)
    result = {
        "schema": "blindassist_known_camera_height_ground_scale_r0_source_authority_qualification",
        "protocol_sha256": sha256(arguments.protocol),
        "roster_lock_sha256": sha256(arguments.roster_lock),
        "media_manifest_sha256": sha256(arguments.media_manifest),
        "authority_amendment_sha256": sha256(arguments.authority_amendment),
        "candidate_or_da_outputs_read": False,
        "effect_metrics_computed": False,
        "parent_results": parent_results,
        "records": all_records,
        "terminal": "GRAVITY_PLANE_PROXY_SOURCE_AUTHORITY_QUALIFIED"
        if passed
        else "HOLD_SOURCE_AUTHORITY_NO_REPLACEMENT",
    }
    write_json_new(arguments.output, result)
    print(json.dumps({**result, "records": f"{len(all_records)} records"}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
