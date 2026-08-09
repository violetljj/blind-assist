#!/usr/bin/env python3
"""Materialize frozen B0 ARKitScenes windows fully covered by camera poses."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (
    ASSETS,
    bound_file,
    load_json,
    lookup_preflight,
    require,
    roster_rows,
    sha256_file,
    write_json_exclusive,
)

from scripts.research.spatial_calibration_head_r1.download_locked_assets import (
    download_file,
    extract_named_members,
    nearest_pincam_member_names,
    pincam_members,
    png_members_by_stem,
    remove_empty_archive_tree,
    safe_delete_archive,
)
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem


PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_download_protocol_v1"
MANIFEST_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1"


def read_trajectory_bounds(path: Path) -> tuple[float, float, int]:
    timestamps: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 7, f"trajectory row {line_number} does not have 7 fields: {path}")
        values = [float(field) for field in fields]
        require(all(math.isfinite(value) for value in values), f"non-finite trajectory row {line_number}: {path}")
        timestamps.append(values[0])
    require(len(timestamps) >= 2, f"trajectory has fewer than two rows: {path}")
    require(all(left < right for left, right in zip(timestamps, timestamps[1:])), f"trajectory is not strictly increasing: {path}")
    return timestamps[0], timestamps[-1], len(timestamps)


def pose_covered_common_stems(
    member_maps: dict[str, dict[str, str]],
    count: int,
    trajectory_start: float,
    trajectory_end: float,
) -> list[str]:
    common = set.intersection(*(set(value) for value in member_maps.values()))
    covered = [
        stem
        for stem in common
        if trajectory_start <= timestamp_from_stem(stem) <= trajectory_end
    ]
    ordered = sorted(covered, key=lambda stem: (timestamp_from_stem(stem), stem))
    require(len(ordered) >= count, f"fewer than {count} pose-covered common RGB-depth-confidence frames")
    selected = ordered[:count]
    times = [timestamp_from_stem(stem) for stem in selected]
    require(
        all(0 < right - left <= 0.5 for left, right in zip(times, times[1:])),
        "earliest pose-covered common window violates 500 ms gap",
    )
    require(times[0] >= trajectory_start and times[-1] <= trajectory_end, "pose coverage invariant failed")
    return selected


def validate_pose_covered_receipt(
    path: Path,
    parent: dict[str, str],
    frame_count: int,
) -> dict[str, Any]:
    value = load_json(path)
    for key in ("role", "visit_id", "video_id", "official_fold"):
        require(value.get(key) == parent[key], f"receipt {key} drift: {path}")
    require(value.get("selected_frame_count") == frame_count, f"receipt frame count drift: {path}")
    coverage = value.get("pose_coverage", {})
    require(coverage.get("all_selected_frames_pose_covered") is True, f"receipt pose coverage drift: {path}")
    for outputs in value.get("extracted", {}).values():
        for output in outputs:
            output_path = Path(output["path"])
            require(output_path.is_file(), f"receipt output missing: {output_path}")
            require(output_path.stat().st_size == output["bytes"], f"receipt output size drift: {output_path}")
    trajectory_path = Path(value["trajectory"]["path"])
    require(trajectory_path.is_file(), f"receipt trajectory missing: {trajectory_path}")
    require(trajectory_path.stat().st_size == value["trajectory"]["bytes"], f"receipt trajectory size drift: {trajectory_path}")
    return value


def source_asset_receipt(row: dict[str, Any], digest: str, attempts: int) -> dict[str, Any]:
    return {
        "asset": row["asset"],
        "url": row["url"],
        "bytes": int(row["content_length_bytes"]),
        "sha256": digest,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        bound_file(root, dependency)
    roster = load_json(bound_file(root, protocol["roster"]))
    preflight = load_json(bound_file(root, protocol["asset_preflight"]))
    authorization = load_json(bound_file(root, protocol["authorization_receipt"]))
    require(authorization["interpreted_scope"]["new_source_media_download"] is True, "media download is not authorized")
    require(
        authorization["interpreted_scope"]["arkitscenes_roster"]["sha256"] == protocol["roster"]["sha256"],
        "authorization roster mismatch",
    )
    require(preflight["terminal"] == "B0_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED", "asset preflight terminal mismatch")
    lookup = lookup_preflight(preflight)
    rows = roster_rows(roster)
    free = shutil.disk_usage(args.output_root.parent).free
    by_video: dict[str, int] = {}
    for row in preflight["assets"]:
        video_id = str(row["video_id"])
        by_video[video_id] = by_video.get(video_id, 0) + int(row["content_length_bytes"])
    require(free >= max(by_video.values()) * 3 + 2_000_000_000, "insufficient bounded working space")

    attempt = {
        "schema": "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_download_attempt_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "roster_sha256": protocol["roster"]["sha256"],
        "asset_preflight_sha256": protocol["asset_preflight"]["sha256"],
        "authorization_receipt_sha256": protocol["authorization_receipt"]["sha256"],
        "output_root": str(args.output_root.resolve()),
    }
    attempt_path = args.output_root / "attempt.json"
    if args.output_root.exists():
        require(args.resume, f"output root exists; explicit --resume required: {args.output_root}")
        require(attempt_path.is_file(), "resume attempt receipt missing")
        require(load_json(attempt_path) == attempt, "resume attempt receipt drift")
        require(not (args.output_root / "manifest.json").exists(), "completed manifest already exists")
    else:
        args.output_root.mkdir(parents=True)
        write_json_exclusive(attempt_path, attempt)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir(exist_ok=True)

    videos: list[dict[str, Any]] = []
    frame_count = int(protocol["continuous_frame_count_per_video"])
    for index, parent in enumerate(rows, start=1):
        role = parent["role"]
        video_id = parent["video_id"]
        official_fold = parent["official_fold"]
        receipt_path = args.output_root / "receipts" / f"{index:02d}-{role}-{video_id}.json"
        if receipt_path.exists():
            videos.append(validate_pose_covered_receipt(receipt_path, parent, frame_count))
            print(json.dumps({"completed": index, "total": len(rows), "role": role, "video_id": video_id, "resumed": True}), flush=True)
            continue

        video_root = args.output_root / "raw" / official_fold / video_id
        source_assets: list[dict[str, Any]] = []
        trajectory_row = lookup[(video_id, "lowres_wide.traj")]
        trajectory_path = video_root / "lowres_wide.traj"
        digest, attempts = download_file(
            trajectory_row["url"],
            trajectory_path,
            int(trajectory_row["content_length_bytes"]),
        )
        source_assets.append(source_asset_receipt(trajectory_row, digest, attempts))
        trajectory_start, trajectory_end, trajectory_rows = read_trajectory_bounds(trajectory_path)

        archives: dict[str, Path] = {}
        for asset in ASSETS[:3]:
            row = lookup[(video_id, asset)]
            archive_path = archive_root / video_id / asset
            digest, attempts = download_file(row["url"], archive_path, int(row["content_length_bytes"]))
            archives[asset] = archive_path
            source_assets.append(source_asset_receipt(row, digest, attempts))
        maps = {asset: png_members_by_stem(path) for asset, path in archives.items()}
        selected = pose_covered_common_stems(maps, frame_count, trajectory_start, trajectory_end)
        extracted: dict[str, Any] = {}
        for asset, folder in (
            ("lowres_wide.zip", "lowres_wide"),
            ("lowres_depth.zip", "lowres_depth"),
            ("confidence.zip", "confidence"),
        ):
            extracted[folder] = extract_named_members(
                archives[asset],
                [maps[asset][stem] for stem in selected],
                video_root / folder,
            )
            safe_delete_archive(archives[asset], archive_root)

        intrinsics_row = lookup[(video_id, "lowres_wide_intrinsics.zip")]
        intrinsics_archive = archive_root / video_id / "lowres_wide_intrinsics.zip"
        digest, attempts = download_file(
            intrinsics_row["url"],
            intrinsics_archive,
            int(intrinsics_row["content_length_bytes"]),
        )
        source_assets.append(source_asset_receipt(intrinsics_row, digest, attempts))
        intrinsics_members = nearest_pincam_member_names(pincam_members(intrinsics_archive), selected)
        extracted["lowres_wide_intrinsics"] = extract_named_members(
            intrinsics_archive,
            intrinsics_members,
            video_root / "lowres_wide_intrinsics",
        )
        safe_delete_archive(intrinsics_archive, archive_root)

        selected_times = [timestamp_from_stem(stem) for stem in selected]
        video_receipt = parent | {
            "selected_frame_stems": selected,
            "selected_frame_count": len(selected),
            "pose_coverage": {
                "all_selected_frames_pose_covered": True,
                "trajectory_start_timestamp": trajectory_start,
                "trajectory_end_timestamp": trajectory_end,
                "trajectory_row_count": trajectory_rows,
                "selected_start_timestamp": selected_times[0],
                "selected_end_timestamp": selected_times[-1],
            },
            "trajectory": {
                "path": str(trajectory_path.resolve()),
                "bytes": trajectory_path.stat().st_size,
                "sha256": sha256_file(trajectory_path),
            },
            "source_assets": source_assets,
            "extracted": extracted,
        }
        write_json_exclusive(receipt_path, video_receipt)
        videos.append(video_receipt)
        print(json.dumps({"completed": index, "total": len(rows), "role": role, "video_id": video_id}), flush=True)

    remove_empty_archive_tree(archive_root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "roster_sha256": protocol["roster"]["sha256"],
        "asset_preflight_sha256": protocol["asset_preflight"]["sha256"],
        "authorization_receipt_sha256": protocol["authorization_receipt"]["sha256"],
        "video_count": len(videos),
        "continuous_frame_count_per_video": frame_count,
        "frame_window_policy": "EARLIEST_COMMON_RGB_DEPTH_CONFIDENCE_FULLY_INSIDE_TRAJECTORY_DOMAIN",
        "videos": videos,
        "task_outcome_opened": False,
        "model_outputs_read": False,
        "temporary_archives_retained": False,
        "terminal": "B0_ARKIT_POSE_COVERED_MEDIA_DOWNLOADED_INTEGRITY_AUDIT_PENDING",
    }
    manifest_path = args.output_root / "manifest.json"
    write_json_exclusive(manifest_path, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
