#!/usr/bin/env python3
"""Freeze seven previously unused TUM parents for the AG-ST third-Teacher R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from ag_st_tum_rgbd import (
    COHORT_SCHEMA,
    _read_member,
    _tar_member_map,
    interpolate_camera_to_world,
    pair_rgb_depth_unique,
    parse_tum_index,
    parse_tum_poses,
)
from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
TOKEN = "AG_ST_TUM_THIRD_TEACHER_R2_V1"
DEFAULT_SOURCE_DIR = REPO_ROOT / "artifacts.local/downloads/ag-st-tum-third-teacher-r2"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_THIRD_TEACHER_COHORT_R2_2026-08-10.json"
)
PARENTS = (
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg1_desk2",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg1_plant",
    "rgbd_dataset_freiburg2_xyz",
    "rgbd_dataset_freiburg3_cabinet",
    "rgbd_dataset_freiburg3_teddy",
)
INTRINSICS = {
    "freiburg1": [517.3, 516.5, 318.6, 255.3],
    "freiburg2": [520.9, 521.0, 325.1, 249.7],
    "freiburg3": [535.4, 539.2, 320.1, 247.6],
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def parent_roles() -> dict[str, str]:
    ordered = sorted(PARENTS, key=lambda parent: _digest(f"{TOKEN}|PARENT|{parent}"))
    return {parent: ("fit" if index < 4 else "evaluation") for index, parent in enumerate(ordered)}


def _select_three(
    parent: str,
    role: str,
    rgb_rows: list[Any],
    depth_rows: list[Any],
    pose_rows: list[Any],
) -> tuple[list[int], list[int], int]:
    pairing = pair_rgb_depth_unique(rgb_rows, depth_rows)
    rgb_by_index = {row.row_index: row for row in rgb_rows}
    eligible: list[Any] = []
    for rgb_index in sorted(pairing, key=lambda index: rgb_by_index[index].timestamp_seconds):
        rgb = rgb_by_index[rgb_index]
        try:
            interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
        except ValueError:
            continue
        eligible.append(rgb)
    require(len(eligible) >= 3, f"insufficient pose-bound pairs: {parent}")

    selected: list[Any] = []
    count = len(eligible)
    for tercile in range(3):
        start = tercile * count // 3
        end = (tercile + 1) * count // 3
        bucket = eligible[start:end]
        require(bucket, f"empty frame tercile: {parent}/{tercile}")
        selected.append(
            min(
                bucket,
                key=lambda rgb: _digest(
                    f"{TOKEN}|FRAME|{role}|{parent}|{tercile}|"
                    f"{rgb.relative_path}|{pairing[rgb.row_index].relative_path}"
                ),
            )
        )
    return (
        [rgb.row_index for rgb in selected],
        [pairing[rgb.row_index].row_index for rgb in selected],
        len(pairing),
    )


def build_manifest(source_dir: Path) -> dict[str, Any]:
    roles = parent_roles()
    rows: dict[str, list[dict[str, Any]]] = {"fit": [], "evaluation": []}
    for parent in PARENTS:
        archive_path = source_dir / f"{parent}.tgz"
        require(archive_path.is_file(), f"TUM archive missing: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _tar_member_map(archive, parent)
            require("groundtruth.txt" in members, f"TUM groundtruth missing: {parent}")
            rgb_rows = parse_tum_index(_read_member(archive, members["rgb.txt"]).decode("utf-8"))
            depth_rows = parse_tum_index(_read_member(archive, members["depth.txt"]).decode("utf-8"))
            pose_rows = parse_tum_poses(
                _read_member(archive, members["groundtruth.txt"]).decode("utf-8")
            )
            rgb_rows = [row for row in rgb_rows if row.relative_path in members]
            depth_rows = [row for row in depth_rows if row.relative_path in members]
            role = roles[parent]
            rgb_indices, depth_indices, pair_count = _select_three(
                parent, role, rgb_rows, depth_rows, pose_rows
            )
        family = next(key for key in INTRINSICS if key in parent)
        rows[role].append(
            {
                "parent_id": parent,
                "storage_kind": "tgz",
                "source_path": archive_path.absolute().relative_to(REPO_ROOT.absolute()).as_posix(),
                "source_url": (
                    f"https://cvg.cit.tum.de/rgbd/dataset/{family}/"
                    f"{parent}.tgz"
                ),
                "source_bytes": archive_path.stat().st_size,
                "source_sha256": sha256_file(archive_path),
                "intrinsics_fx_fy_cx_cy": INTRINSICS[family],
                "rgb_row_indices_zero_based": rgb_indices,
                "depth_row_indices_zero_based": depth_indices,
                "unique_pair_count": pair_count,
            }
        )
    for role in rows:
        rows[role].sort(key=lambda row: _digest(f"{TOKEN}|PARENT|{row['parent_id']}"))
    require(len(rows["fit"]) == 4 and len(rows["evaluation"]) == 3, "TUM role count drift")
    return {
        "schema": COHORT_SCHEMA,
        "status": "FROZEN_BEFORE_THIRD_TEACHER_MODEL_EXECUTION",
        "token": TOKEN,
        "selection": {
            "parent_order": "SHA256({TOKEN}|PARENT|{parent_id}) ascending; first 4 FIT, remaining 3 EVALUATION",
            "frame_order": "within each pose-bound paired-frame tercile choose minimum SHA256 token",
            "frames_per_parent": 3,
        },
        "sensor_contract": {
            "publisher": "Technical University of Munich Computer Vision Group",
            "dataset": "TUM RGB-D benchmark",
            "native_resolution_wh": [640, 480],
            "depth_scale_divisor": 5000.0,
            "invalid_depth": "uint16 == 0",
            "official_format_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats",
            "official_license": "CC BY 4.0",
            "official_license_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
        },
        "fit_parents": rows["fit"],
        "evaluation_parents": rows["evaluation"],
        "role_boundary": {
            "all_parents_absent_from_repository_text_before_r2": True,
            "evaluation_may_influence_fit_or_selector_choice": False,
            "all_seven_parents_are_one_tum_kinect_benchmark_domain": True,
            "support_boundary_obstacle": "UNKNOWN",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"cohort output already exists: {args.output}")
    payload = build_manifest(args.source_dir.absolute())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "fit": 4, "evaluation": 3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
