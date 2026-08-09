#!/usr/bin/env python3
"""Perform a label-blind integrity audit of pose-covered B0 ARKitScenes media."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    bound_file,
    load_json,
    require,
    sha256_file,
)
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem  # noqa: E402


PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_integrity_audit_protocol_v1"
MANIFEST_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1"
RECEIPT_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_integrity_audit_v1"
MODALITY_MODES = {
    "lowres_wide": {"RGB"},
    "lowres_depth": {"I;16", "I;16L"},
    "confidence": {"L"},
}


def sha256_path_entry(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"materialized file missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"materialized file size drift: {path}")
    require(sha256_file(path) == entry["sha256"], f"materialized file SHA drift: {path}")
    return path


def parse_intrinsics(path: Path) -> tuple[int, int, float, float, float, float]:
    fields = path.read_text(encoding="utf-8").split()
    require(len(fields) == 6, f"intrinsics must have six fields: {path}")
    values = [float(field) for field in fields]
    require(all(math.isfinite(value) for value in values), f"non-finite intrinsics: {path}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height, f"non-integral intrinsics dimensions: {path}")
    fx, fy, cx, cy = values[2:]
    require(width > 0 and height > 0 and fx > 0 and fy > 0, f"invalid intrinsics scale: {path}")
    require(0 <= cx < width and 0 <= cy < height, f"principal point outside image: {path}")
    return width, height, fx, fy, cx, cy


def parse_trajectory(path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 7, f"trajectory row {line_number} must have seven fields: {path}")
        values = tuple(float(field) for field in fields)
        require(all(math.isfinite(value) for value in values), f"non-finite trajectory row {line_number}: {path}")
        rows.append(values)
    require(len(rows) >= 2, f"trajectory has fewer than two rows: {path}")
    require(all(left[0] < right[0] for left, right in zip(rows, rows[1:])), f"trajectory timestamps are not strictly increasing: {path}")
    return rows


def maximum_bracketing_gap(frame_times: list[float], trajectory_times: list[float]) -> float:
    maximum = 0.0
    for timestamp in frame_times:
        right = bisect.bisect_left(trajectory_times, timestamp)
        if right < len(trajectory_times) and trajectory_times[right] == timestamp:
            continue
        require(0 < right < len(trajectory_times), f"frame outside trajectory interpolation domain: {timestamp}")
        maximum = max(maximum, trajectory_times[right] - trajectory_times[right - 1])
    return maximum


def decode_image(path: Path, modality: str) -> tuple[tuple[int, int], str, float, int, int]:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = image.size
        require(mode in MODALITY_MODES[modality], f"unexpected {modality} image mode {mode}: {path}")
        array = np.asarray(image)
    require(array.size > 0, f"empty decoded image: {path}")
    require(np.isfinite(array).all(), f"non-finite decoded image: {path}")
    nonzero_ratio = float(np.count_nonzero(array) / array.size)
    return size, mode, nonzero_ratio, int(array.min()), int(array.max())


def audit_video(video: dict[str, Any], frame_count: int, maximum_pose_gap: float) -> dict[str, Any]:
    selected_stems = [str(stem) for stem in video["selected_frame_stems"]]
    require(len(selected_stems) == frame_count, "selected frame count drift")
    frame_times = [timestamp_from_stem(stem) for stem in selected_stems]
    require(all(left < right for left, right in zip(frame_times, frame_times[1:])), "selected frame timestamps are not strictly increasing")
    require(all(0 < right - left <= 0.5 for left, right in zip(frame_times, frame_times[1:])), "selected frame continuity drift")

    trajectory_entry = video["trajectory"]
    trajectory_path = Path(trajectory_entry["path"])
    require(trajectory_path.is_file(), f"trajectory missing: {trajectory_path}")
    require(trajectory_path.stat().st_size == int(trajectory_entry["bytes"]), f"trajectory size drift: {trajectory_path}")
    require(sha256_file(trajectory_path) == trajectory_entry["sha256"], f"trajectory SHA drift: {trajectory_path}")
    trajectory_rows = parse_trajectory(trajectory_path)
    trajectory_times = [row[0] for row in trajectory_rows]
    require(frame_times[0] >= trajectory_times[0] and frame_times[-1] <= trajectory_times[-1], "selected window is not pose-covered")
    bracketing_gap = maximum_bracketing_gap(frame_times, trajectory_times)
    require(bracketing_gap <= maximum_pose_gap, f"trajectory interpolation gap exceeds {maximum_pose_gap} seconds")

    extracted = video["extracted"]
    for modality in (*MODALITY_MODES, "lowres_wide_intrinsics"):
        require(len(extracted.get(modality, [])) == frame_count, f"{modality} entry count drift")

    modality_sizes: dict[str, set[tuple[int, int]]] = {name: set() for name in MODALITY_MODES}
    modality_modes: dict[str, set[str]] = {name: set() for name in MODALITY_MODES}
    depth_nonzero_min = 1.0
    depth_min = 2**31 - 1
    depth_max = 0
    confidence_values: set[int] = set()
    rgb_entries = extracted["lowres_wide"]
    depth_entries = extracted["lowres_depth"]
    confidence_entries = extracted["confidence"]
    intrinsics_entries = extracted["lowres_wide_intrinsics"]
    for index in range(frame_count):
        decoded: dict[str, tuple[tuple[int, int], str, float, int, int]] = {}
        for modality, entries in (
            ("lowres_wide", rgb_entries),
            ("lowres_depth", depth_entries),
            ("confidence", confidence_entries),
        ):
            path = sha256_path_entry(entries[index])
            decoded[modality] = decode_image(path, modality)
            modality_sizes[modality].add(decoded[modality][0])
            modality_modes[modality].add(decoded[modality][1])
        image_size = decoded["lowres_wide"][0]
        require(decoded["lowres_depth"][0] == image_size, "RGB/depth dimensions differ")
        require(decoded["confidence"][0] == image_size, "RGB/confidence dimensions differ")

        intrinsics_path = sha256_path_entry(intrinsics_entries[index])
        width, height, _fx, _fy, _cx, _cy = parse_intrinsics(intrinsics_path)
        require((width, height) == image_size, "intrinsics/image dimensions differ")

        depth_nonzero_min = min(depth_nonzero_min, decoded["lowres_depth"][2])
        depth_min = min(depth_min, decoded["lowres_depth"][3])
        depth_max = max(depth_max, decoded["lowres_depth"][4])
        confidence_values.update((decoded["confidence"][3], decoded["confidence"][4]))

    require(depth_max > 0, "all decoded depth is zero")
    require(confidence_values.issubset({0, 1, 2}), f"unexpected confidence values: {sorted(confidence_values)}")
    return {
        "role": str(video["role"]),
        "visit_id": str(video["visit_id"]),
        "video_id": str(video["video_id"]),
        "frame_count": frame_count,
        "decoded_image_count": frame_count * 3,
        "intrinsics_mapping_count": frame_count,
        "unique_intrinsics_file_count": len({entry["path"] for entry in intrinsics_entries}),
        "trajectory_row_count": len(trajectory_rows),
        "maximum_pose_bracketing_gap_seconds": bracketing_gap,
        "image_sizes": {key: [list(value) for value in sorted(values)] for key, values in modality_sizes.items()},
        "image_modes": {key: sorted(values) for key, values in modality_modes.items()},
        "minimum_depth_nonzero_ratio": depth_nonzero_min,
        "depth_observed_min_raw": depth_min,
        "depth_observed_max_raw": depth_max,
        "confidence_observed_endpoint_values": sorted(confidence_values),
        "integrity_pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    manifest_path = bound_file(root, protocol["manifest"])
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    require(manifest.get("terminal") == "B0_ARKIT_POSE_COVERED_MEDIA_DOWNLOADED_INTEGRITY_AUDIT_PENDING", "manifest terminal drift")
    require(manifest.get("task_outcome_opened") is False, "task outcome boundary drift")
    require(manifest.get("model_outputs_read") is False, "model output boundary drift")
    require(int(manifest["video_count"]) == int(protocol["expected_video_count"]), "video count drift")
    frame_count = int(protocol["expected_frame_count_per_video"])
    roles = [str(video["role"]) for video in manifest["videos"]]
    require({role: roles.count(role) for role in set(roles)} == protocol["expected_role_counts"], "role count drift")
    require(len({str(video["visit_id"]) for video in manifest["videos"]}) == len(roles), "visit overlap")
    require(len({str(video["video_id"]) for video in manifest["videos"]}) == len(roles), "video overlap")

    videos: list[dict[str, Any]] = []
    for index, video in enumerate(manifest["videos"], start=1):
        result = audit_video(video, frame_count, float(protocol["maximum_pose_bracketing_gap_seconds"]))
        videos.append(result)
        print(json.dumps({"audited": index, "total": len(manifest["videos"]), "role": result["role"], "video_id": result["video_id"]}), flush=True)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "producer": protocol["producer"],
        "manifest": protocol["manifest"],
        "video_count": len(videos),
        "frame_count": len(videos) * frame_count,
        "decoded_image_count": sum(video["decoded_image_count"] for video in videos),
        "intrinsics_mapping_count": sum(video["intrinsics_mapping_count"] for video in videos),
        "role_counts": {role: roles.count(role) for role in ("TRAIN", "DEVELOPMENT", "CONFIRMATION")},
        "minimum_depth_nonzero_ratio": min(video["minimum_depth_nonzero_ratio"] for video in videos),
        "maximum_pose_bracketing_gap_seconds": max(video["maximum_pose_bracketing_gap_seconds"] for video in videos),
        "videos": videos,
        "task_outcome_opened": False,
        "model_outputs_read": False,
        "confirmation_used_for_selection": False,
        "terminal": "B0_ARKIT_POSE_COVERED_MEDIA_LABEL_BLIND_INTEGRITY_PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "videos"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
