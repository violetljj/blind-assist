#!/usr/bin/env python3
"""Materialize and audit label-blind portrait continuity for DepthART D1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    interpolate_camera_to_world,
    orientation_index,
    parse_trajectory,
)
from scripts.research.assistive_geometry.audit_b0_arkitscenes_integrity import (  # noqa: E402
    MODALITY_MODES,
    decode_image,
    parse_intrinsics,
    sha256_path_entry,
)
from scripts.research.spatial_calibration_head_r1.download_locked_assets import (  # noqa: E402
    download_file,
    extract_named_members,
    nearest_pincam_member_names,
    pincam_members,
    png_members_by_stem,
    remove_empty_archive_tree,
    safe_delete_archive,
)
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem  # noqa: E402


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_body_preflight_protocol_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_development_roster_lock_v1"
HEAD_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_asset_header_preflight_v1"
LICENSE_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_license_scope_receipt_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_body_preflight_manifest_v1"
ASSETS = (
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def bound_file(root: Path, binding: dict[str, Any]) -> Path:
    path = (root / binding["path"]).resolve()
    require(path.is_file(), f"bound file missing: {path}")
    require(path.stat().st_size == int(binding["bytes"]), f"bound file size drift: {path}")
    require(sha256_file(path) == binding["sha256"], f"bound file SHA drift: {path}")
    return path


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def roster_rows(roster: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ("primary", "reserve"):
        for order, parent in enumerate(roster[role], start=1):
            rows.append(
                {
                    "role": role.upper(),
                    "frozen_order": order,
                    "visit_id": str(parent["visit_id"]),
                    "video_id": str(parent["video_id"]),
                    "fold": str(parent["fold"]),
                }
            )
    require(len(rows) == 16, "expected exactly 16 identities")
    return rows


def preflight_lookup(preflight: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in preflight["assets"]:
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate HEAD asset: {key}")
        require(row["http_status"] == 200 and int(row["content_length_bytes"]) > 0, f"unavailable asset: {key}")
        lookup[key] = row
    require(len(lookup) == 80, "HEAD asset count drift")
    return lookup


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
    member_maps: dict[str, dict[str, str]],
    trajectory: np.ndarray,
    maximum_pose_gap: float,
    portrait_indices: set[int],
) -> tuple[list[str], dict[str, Any]]:
    common = set.intersection(*(set(mapping) for mapping in member_maps.values()))
    portrait: list[str] = []
    orientation_counts = {str(index): 0 for index in range(4)}
    pose_rejected = 0
    maximum_observed_pose_gap = 0.0
    for stem in sorted(common, key=lambda item: (timestamp_from_stem(item), item)):
        try:
            pose, metadata = interpolate_camera_to_world(
                trajectory,
                timestamp_from_stem(stem),
                maximum_pose_gap,
            )
        except ValueError:
            pose_rejected += 1
            continue
        index = orientation_index(pose)
        orientation_counts[str(index)] += 1
        maximum_observed_pose_gap = max(maximum_observed_pose_gap, metadata["bracketing_gap_seconds"])
        if index in portrait_indices:
            portrait.append(stem)
    return portrait, {
        "common_rgb_depth_confidence_frame_count": len(common),
        "pose_rejected_frame_count": pose_rejected,
        "pose_covered_orientation_counts": orientation_counts,
        "portrait_pose_covered_frame_count": len(portrait),
        "maximum_observed_pose_bracketing_gap_seconds": maximum_observed_pose_gap,
    }


def source_receipt(row: dict[str, Any], digest: str, attempts: int) -> dict[str, Any]:
    return {
        "asset": str(row["asset"]),
        "url": str(row["url"]),
        "bytes": int(row["content_length_bytes"]),
        "sha256": digest,
        "attempts": attempts,
    }


def audit_extracted(extracted: dict[str, Any], frame_count: int) -> dict[str, Any]:
    for modality in (*MODALITY_MODES, "lowres_wide_intrinsics"):
        require(len(extracted.get(modality, [])) == frame_count, f"{modality} entry count drift")
    sizes: dict[str, set[tuple[int, int]]] = {name: set() for name in MODALITY_MODES}
    modes: dict[str, set[str]] = {name: set() for name in MODALITY_MODES}
    depth_nonzero_min = 1.0
    depth_max = 0
    confidence_values: set[int] = set()
    for index in range(frame_count):
        decoded = {}
        for modality in MODALITY_MODES:
            decoded[modality] = decode_image(sha256_path_entry(extracted[modality][index]), modality)
            sizes[modality].add(decoded[modality][0])
            modes[modality].add(decoded[modality][1])
        image_size = decoded["lowres_wide"][0]
        require(decoded["lowres_depth"][0] == image_size, "RGB/depth dimensions differ")
        require(decoded["confidence"][0] == image_size, "RGB/confidence dimensions differ")
        width, height, *_ = parse_intrinsics(sha256_path_entry(extracted["lowres_wide_intrinsics"][index]))
        require((width, height) == image_size, "intrinsics/image dimensions differ")
        depth_nonzero_min = min(depth_nonzero_min, decoded["lowres_depth"][2])
        depth_max = max(depth_max, decoded["lowres_depth"][4])
        confidence_values.update((decoded["confidence"][3], decoded["confidence"][4]))
    require(depth_max > 0, "all decoded depth is zero")
    require(confidence_values.issubset({0, 1, 2}), f"unexpected confidence values: {sorted(confidence_values)}")
    return {
        "decoded_image_count": frame_count * 3,
        "intrinsics_mapping_count": frame_count,
        "image_sizes": {key: [list(value) for value in sorted(values)] for key, values in sizes.items()},
        "image_modes": {key: sorted(values) for key, values in modes.items()},
        "minimum_depth_nonzero_ratio": depth_nonzero_min,
        "depth_observed_max_raw": depth_max,
        "confidence_observed_endpoint_values": sorted(confidence_values),
    }


def select_final(videos: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    primaries = sorted((row for row in videos if row["role"] == "PRIMARY"), key=lambda row: row["frozen_order"])
    reserves = iter(sorted((row for row in videos if row["role"] == "RESERVE" and row["eligible"]), key=lambda row: row["frozen_order"]))
    selected: list[dict[str, str]] = []
    replacements: list[dict[str, str]] = []
    for primary in primaries:
        if primary["eligible"]:
            selected.append({key: str(primary[key]) for key in ("visit_id", "video_id")})
            continue
        try:
            reserve = next(reserves)
        except StopIteration:
            continue
        selected.append({key: str(reserve[key]) for key in ("visit_id", "video_id")})
        replacements.append(
            {
                "primary_visit_id": str(primary["visit_id"]),
                "primary_video_id": str(primary["video_id"]),
                "reserve_visit_id": str(reserve["visit_id"]),
                "reserve_video_id": str(reserve["video_id"]),
                "basis": "LABEL_BLIND_MEDIA_INTEGRITY_AND_PRODUCT_PORTRAIT_POSE_RGBD_CONTINUITY_ONLY",
            }
        )
    return selected, replacements


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
    license_receipt = load_json(bound_file(root, protocol["license_receipt"]))
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(preflight.get("schema") == HEAD_SCHEMA, "HEAD preflight schema drift")
    require(license_receipt.get("schema") == LICENSE_SCHEMA, "license schema drift")
    require(license_receipt["bounded_media_download_authorized"] is True, "media download not authorized")
    require(preflight["terminal"] == "D1_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED", "HEAD terminal drift")
    lookup = preflight_lookup(preflight)
    rows = roster_rows(roster)
    free = shutil.disk_usage(args.output_root.parent).free
    largest_video = max(
        sum(int(lookup[(row["video_id"], asset)]["content_length_bytes"]) for asset in ASSETS)
        for row in rows
    )
    require(free >= largest_video * 3 + 2_000_000_000, "insufficient bounded working space")

    attempt = {
        "schema": "blindassist_depthart_task_preserving_d1_arkit_body_preflight_attempt_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "output_root": str(args.output_root.resolve()),
    }
    attempt_path = args.output_root / "attempt.json"
    if args.output_root.exists():
        require(args.resume, f"output root exists; explicit --resume required: {args.output_root}")
        require(attempt_path.is_file() and load_json(attempt_path) == attempt, "resume attempt drift")
        require(not (args.output_root / "manifest.json").exists(), "completed manifest already exists")
    else:
        args.output_root.mkdir(parents=True)
        write_json_exclusive(attempt_path, attempt)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir(exist_ok=True)
    frame_count = int(protocol["continuous_portrait_frame_count_per_video"])
    maximum_frame_gap = float(protocol["maximum_adjacent_frame_gap_seconds"])
    maximum_pose_gap = float(protocol["maximum_pose_bracketing_gap_seconds"])
    portrait_indices = {int(value) for value in protocol["portrait_orientation_indices"]}
    videos: list[dict[str, Any]] = []

    for index, parent in enumerate(rows, start=1):
        receipt_path = args.output_root / "receipts" / f"{index:02d}-{parent['role']}-{parent['video_id']}.json"
        if receipt_path.exists():
            value = load_json(receipt_path)
            require(all(str(value[key]) == str(parent[key]) for key in ("role", "visit_id", "video_id")), "resume identity drift")
            videos.append(value)
            print(json.dumps({"completed": index, "total": len(rows), "video_id": parent["video_id"], "resumed": True}), flush=True)
            continue
        video_root = args.output_root / "raw" / parent["fold"] / parent["video_id"]
        source_assets: list[dict[str, Any]] = []
        trajectory_row = lookup[(parent["video_id"], "lowres_wide.traj")]
        trajectory_path = video_root / "lowres_wide.traj"
        digest, attempts = download_file(trajectory_row["url"], trajectory_path, int(trajectory_row["content_length_bytes"]))
        source_assets.append(source_receipt(trajectory_row, digest, attempts))
        trajectory = parse_trajectory(trajectory_path)
        archives: dict[str, Path] = {}
        for asset in ASSETS[:3]:
            row = lookup[(parent["video_id"], asset)]
            archive = archive_root / parent["video_id"] / asset
            digest, attempts = download_file(row["url"], archive, int(row["content_length_bytes"]))
            archives[asset] = archive
            source_assets.append(source_receipt(row, digest, attempts))
        maps = {asset: png_members_by_stem(path) for asset, path in archives.items()}
        candidates, coverage = portrait_candidates(maps, trajectory, maximum_pose_gap, portrait_indices)
        try:
            selected = continuous_window(candidates, frame_count, maximum_frame_gap)
            eligibility_reason = "PASS"
        except ValueError as error:
            selected = []
            eligibility_reason = str(error)
        extracted: dict[str, Any] = {}
        integrity: dict[str, Any] = {}
        if selected:
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
            intrinsics_row = lookup[(parent["video_id"], "lowres_wide_intrinsics.zip")]
            intrinsics_archive = archive_root / parent["video_id"] / "lowres_wide_intrinsics.zip"
            digest, attempts = download_file(
                intrinsics_row["url"], intrinsics_archive, int(intrinsics_row["content_length_bytes"])
            )
            source_assets.append(source_receipt(intrinsics_row, digest, attempts))
            extracted["lowres_wide_intrinsics"] = extract_named_members(
                intrinsics_archive,
                nearest_pincam_member_names(pincam_members(intrinsics_archive), selected),
                video_root / "lowres_wide_intrinsics",
            )
            safe_delete_archive(intrinsics_archive, archive_root)
            try:
                integrity = audit_extracted(extracted, frame_count)
            except ValueError as error:
                eligibility_reason = f"integrity audit failed: {error}"
        for archive in archives.values():
            safe_delete_archive(archive, archive_root)
        eligible = bool(selected) and bool(integrity) and eligibility_reason == "PASS"
        times = [timestamp_from_stem(stem) for stem in selected]
        value = parent | {
            "eligible": eligible,
            "eligibility_reason": eligibility_reason,
            "selected_frame_stems": selected,
            "selected_frame_count": len(selected),
            "selected_start_timestamp": times[0] if times else None,
            "selected_end_timestamp": times[-1] if times else None,
            "maximum_selected_adjacent_gap_seconds": max(np.diff(times)).item() if len(times) > 1 else None,
            "coverage": coverage,
            "trajectory": {
                "path": str(trajectory_path.resolve()),
                "bytes": trajectory_path.stat().st_size,
                "sha256": sha256_file(trajectory_path),
            },
            "source_assets": source_assets,
            "extracted": extracted,
            "integrity": integrity,
            "truth_model_or_task_outcome_read": False,
        }
        write_json_exclusive(receipt_path, value)
        videos.append(value)
        print(json.dumps({"completed": index, "total": len(rows), "role": parent["role"], "video_id": parent["video_id"], "eligible": eligible, "reason": eligibility_reason}), flush=True)

    remove_empty_archive_tree(archive_root)
    selected, replacements = select_final(videos)
    passed = len(selected) == 8
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "roster_sha256": protocol["roster"]["sha256"],
        "asset_preflight_sha256": protocol["asset_preflight"]["sha256"],
        "license_receipt_sha256": protocol["license_receipt"]["sha256"],
        "identity_count": len(videos),
        "eligible_primary_count": sum(row["role"] == "PRIMARY" and row["eligible"] for row in videos),
        "eligible_reserve_count": sum(row["role"] == "RESERVE" and row["eligible"] for row in videos),
        "continuous_portrait_frame_count_per_selected_video": frame_count,
        "selected": selected,
        "replacements": replacements,
        "videos": videos,
        "source_media_content_decoded_for_integrity_only": True,
        "rgb_visual_review_performed": False,
        "truth_model_or_task_outcome_read": False,
        "candidate_selection_performed": False,
        "r2_cohort_accessed": False,
        "terminal": (
            "D1_ARKIT_LABEL_BLIND_MEDIA_PREFLIGHT_PASS_FINAL_DEVELOPMENT_ROSTER_LOCKED"
            if passed
            else "D1_ARKIT_LABEL_BLIND_MEDIA_PREFLIGHT_FAIL_INSUFFICIENT_ELIGIBLE_IDENTITIES"
        ),
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
