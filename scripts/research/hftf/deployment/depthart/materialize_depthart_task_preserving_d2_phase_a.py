#!/usr/bin/env python3
"""Materialize pose-derived portrait continuity for the D2 Phase-A pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
SPATIAL_HELPER_ROOT = REPO_ROOT / "scripts" / "research" / "spatial_calibration_head_r1"
sys.path.insert(0, str(SPATIAL_HELPER_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    interpolate_camera_to_world,
    orientation_index,
    parse_trajectory,
)
from scripts.research.spatial_calibration_head_r1.download_locked_assets import (
    download_file,
    extract_named_members,
    pincam_members,
    remove_empty_archive_tree,
    safe_delete_archive,
)
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_body_protocol_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d2_source_support_pool_lock_v1"
HEAD_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_asset_header_preflight_v1"
LICENSE_SCHEMA = "blindassist_depthart_task_preserving_d2_arkit_source_scope_receipt_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_manifest_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def continuous_window(stems: list[str], count: int, maximum_gap: float) -> list[str]:
    ordered = sorted(stems, key=lambda stem: (timestamp_from_stem(stem), stem))
    run: list[str] = []
    previous: float | None = None
    for stem in ordered:
        timestamp = timestamp_from_stem(stem)
        if previous is None or 0 < timestamp - previous <= maximum_gap:
            run.append(stem)
        else:
            run = [stem]
        previous = timestamp
        if len(run) == count:
            return run
    raise ValueError(f"fewer than {count} continuous eligible frames")


def portrait_candidates(
    stems: list[str],
    trajectory: Any,
    maximum_pose_gap: float,
    portrait_indices: set[int],
) -> tuple[list[str], dict[str, Any]]:
    portrait: list[str] = []
    orientation_counts = {str(index): 0 for index in range(4)}
    pose_rejected = 0
    maximum_observed_pose_gap = 0.0
    for stem in sorted(stems, key=lambda item: (timestamp_from_stem(item), item)):
        try:
            pose, metadata = interpolate_camera_to_world(
                trajectory, timestamp_from_stem(stem), maximum_pose_gap
            )
        except ValueError:
            pose_rejected += 1
            continue
        index = orientation_index(pose)
        orientation_counts[str(index)] += 1
        maximum_observed_pose_gap = max(
            maximum_observed_pose_gap, metadata["bracketing_gap_seconds"]
        )
        if index in portrait_indices:
            portrait.append(stem)
    return portrait, {
        "intrinsics_frame_count": len(stems),
        "pose_rejected_frame_count": pose_rejected,
        "pose_covered_orientation_counts": orientation_counts,
        "portrait_pose_covered_frame_count": len(portrait),
        "maximum_observed_pose_bracketing_gap_seconds": maximum_observed_pose_gap,
    }


def member_map(archive: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for _, member in pincam_members(archive):
        stem = Path(member).stem
        require(stem not in result, f"duplicate intrinsics stem: {stem}")
        result[stem] = member
    require(result, "intrinsics ZIP contains no .pincam members")
    return result


def lookup_assets(preflight: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in preflight["assets"]:
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate HEAD row: {key}")
        require(row["http_status"] == 200 and int(row["content_length_bytes"]) > 0, f"unavailable: {key}")
        lookup[key] = row
    require(len(lookup) == 64, "HEAD row count drift")
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"overwrite forbidden: {args.output_root}")
    protocol = load_json(args.protocol)
    roster = load_json(args.roster)
    head_result = load_json(args.head_result)
    license_receipt = load_json(args.license_receipt)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(head_result.get("schema") == HEAD_SCHEMA, "HEAD schema drift")
    require(license_receipt.get("schema") == LICENSE_SCHEMA, "license schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        path = Path(dependency["path"])
        require(path.is_file(), f"dependency missing: {path}")
        require(path.stat().st_size == int(dependency["bytes"]), f"dependency size drift: {path}")
        require(sha256_file(path) == dependency["sha256"], f"dependency SHA drift: {path}")
    for name, path in (
        ("roster", args.roster),
        ("head_result", args.head_result),
        ("license_receipt", args.license_receipt),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(head_result["terminal"] == "D2_PHASE_A_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED", "HEAD terminal drift")
    require(license_receipt["authority"]["phase_a_body_after_head_pass"] is True, "Phase-A body not authorized")
    pool = roster["pool"]
    require(len(pool) == 32 and [int(row["pool_order"]) for row in pool] == list(range(1, 33)), "pool drift")
    lookup = lookup_assets(head_result)
    maximum_video_bytes = max(
        sum(int(lookup[(str(row["video_id"]), asset)]["content_length_bytes"]) for asset in protocol["assets"])
        for row in pool
    )
    require(
        shutil.disk_usage(args.output_root.parent).free >= maximum_video_bytes * 3 + 500_000_000,
        "insufficient bounded working space",
    )
    args.output_root.mkdir(parents=True)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir()
    receipts_root = args.output_root / "receipts"
    selected_count = int(protocol["selected_identity_count"])
    frame_count = int(protocol["continuous_portrait_frame_count"])
    maximum_frame_gap = float(protocol["maximum_adjacent_frame_gap_seconds"])
    maximum_pose_gap = float(protocol["maximum_pose_bracketing_gap_seconds"])
    portrait_indices = {int(value) for value in protocol["portrait_orientation_indices"]}
    processed: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for parent in pool:
        if len(eligible) == selected_count:
            break
        video_id = str(parent["video_id"])
        video_root = args.output_root / "raw" / "Training" / video_id
        trajectory_row = lookup[(video_id, "lowres_wide.traj")]
        trajectory_path = video_root / "lowres_wide.traj"
        trajectory_sha, trajectory_attempts = download_file(
            trajectory_row["url"], trajectory_path, int(trajectory_row["content_length_bytes"])
        )
        trajectory = parse_trajectory(trajectory_path)
        intrinsics_row = lookup[(video_id, "lowres_wide_intrinsics.zip")]
        archive = archive_root / video_id / "lowres_wide_intrinsics.zip"
        intrinsics_sha, intrinsics_attempts = download_file(
            intrinsics_row["url"], archive, int(intrinsics_row["content_length_bytes"])
        )
        members = member_map(archive)
        candidates, coverage = portrait_candidates(
            list(members), trajectory, maximum_pose_gap, portrait_indices
        )
        try:
            selected = continuous_window(candidates, frame_count, maximum_frame_gap)
            reason = "PASS"
        except ValueError as error:
            selected = []
            reason = str(error)
        extracted: list[dict[str, Any]] = []
        if selected:
            extracted = extract_named_members(
                archive,
                [members[stem] for stem in selected],
                video_root / "lowres_wide_intrinsics",
            )
        safe_delete_archive(archive, archive_root)
        times = [timestamp_from_stem(stem) for stem in selected]
        value = {
            **parent,
            "eligible": bool(selected),
            "eligibility_reason": reason,
            "selected_frame_stems": selected,
            "selected_frame_count": len(selected),
            "selected_start_timestamp": times[0] if times else None,
            "selected_end_timestamp": times[-1] if times else None,
            "maximum_selected_adjacent_gap_seconds": (
                max(right - left for left, right in zip(times, times[1:])) if len(times) > 1 else None
            ),
            "coverage": coverage,
            "source_assets": [
                {
                    "asset": "lowres_wide.traj",
                    "url": trajectory_row["url"],
                    "bytes": int(trajectory_row["content_length_bytes"]),
                    "sha256": trajectory_sha,
                    "attempts": trajectory_attempts,
                },
                {
                    "asset": "lowres_wide_intrinsics.zip",
                    "url": intrinsics_row["url"],
                    "bytes": int(intrinsics_row["content_length_bytes"]),
                    "sha256": intrinsics_sha,
                    "attempts": intrinsics_attempts,
                },
            ],
            "trajectory": {
                "path": str(trajectory_path.resolve()),
                "bytes": trajectory_path.stat().st_size,
                "sha256": sha256_file(trajectory_path),
            },
            "extracted_intrinsics": extracted,
            "rgb_depth_confidence_read": False,
            "truth_or_model_output_read": False,
        }
        write_json_exclusive(receipts_root / f"{int(parent['pool_order']):02d}-{video_id}.json", value)
        processed.append(value)
        if value["eligible"]:
            eligible.append(value)
        print(
            json.dumps(
                {
                    "processed": len(processed),
                    "pool": len(pool),
                    "eligible": len(eligible),
                    "target": selected_count,
                    "video_id": video_id,
                    "status": reason,
                }
            ),
            flush=True,
        )

    remove_empty_archive_tree(archive_root)
    passed = len(eligible) == selected_count
    selected_phase_b = [
        {
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
            "selected_frame_stems": row["selected_frame_stems"],
        }
        for row in eligible
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "head_result_sha256": sha256_file(args.head_result),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "pool_count": len(pool),
        "processed_identity_count": len(processed),
        "eligible_identity_count": len(eligible),
        "selected_phase_b": selected_phase_b,
        "processed": processed,
        "unprocessed_pool_orders": [int(row["pool_order"]) for row in pool[len(processed) :]],
        "media_body_scope": ["lowres_wide_intrinsics.zip", "lowres_wide.traj"],
        "rgb_depth_confidence_read": False,
        "truth_or_model_output_read": False,
        "train_development_roles_assigned": False,
        "r2_cohort_access": "NONE",
        "terminal": (
            "D2_PHASE_A_PORTRAIT_CONTINUITY_PASS_16_IDENTITIES_LOCKED"
            if passed
            else "D2_PHASE_A_FAIL_FEWER_THAN_16_ELIGIBLE_IDENTITIES"
        ),
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "processed"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
