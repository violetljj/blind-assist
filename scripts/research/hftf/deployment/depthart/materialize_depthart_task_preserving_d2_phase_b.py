#!/usr/bin/env python3
"""Materialize D2 depth/confidence and audit truth support without model output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[5]
SPATIAL_HELPER_ROOT = REPO_ROOT / "scripts" / "research" / "spatial_calibration_head_r1"
sys.path.insert(0, str(SPATIAL_HELPER_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    WORLD_UP,
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    parse_pincam,
    parse_trajectory,
)
from scripts.research.spatial_calibration_head_r1.download_locked_assets import (  # noqa: E402
    download_file,
    extract_named_members,
    png_members_by_stem,
    remove_empty_archive_tree,
    safe_delete_archive,
)
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem  # noqa: E402


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_b_body_protocol_v1"
PHASE_A_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_manifest_v1"
HEAD_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_b_asset_header_preflight_v1"
LICENSE_SCHEMA = "blindassist_depthart_task_preserving_d2_arkit_source_scope_receipt_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_b_manifest_v1"
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)


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


def asset_lookup(head_result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in head_result["assets"]:
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate HEAD row: {key}")
        require(row["http_status"] == 200 and int(row["content_length_bytes"]) > 0, f"unavailable: {key}")
        lookup[key] = row
    require(len(lookup) == 32, "HEAD asset count drift")
    return lookup


def summarize_truth(truth: dict[str, Any]) -> dict[str, Any]:
    result = {
        "known_cells": 0,
        "clear_cells": 0,
        "occupied_cells": 0,
        "valid_band_clearances": 0,
        "known_by_grid": {f"{band}@{horizon:.1f}m": 0 for band in BANDS for horizon in HORIZONS},
    }
    for band in BANDS:
        band_result = truth.get("bands", {}).get(band)
        if not band_result:
            continue
        if band_result.get("clearance_m") is not None:
            result["valid_band_clearances"] += 1
        occupied = band_result.get("occupied_by_horizon", {})
        for horizon in HORIZONS:
            value = occupied.get(str(horizon))
            if value is None:
                continue
            result["known_cells"] += 1
            result["known_by_grid"][f"{band}@{horizon:.1f}m"] += 1
            if value:
                result["occupied_cells"] += 1
            else:
                result["clear_cells"] += 1
    return result


def add_counts(total: dict[str, Any], frame: dict[str, Any]) -> None:
    for key in ("known_cells", "clear_cells", "occupied_cells", "valid_band_clearances"):
        total[key] += int(frame[key])
    for key, value in frame["known_by_grid"].items():
        total["known_by_grid"][key] += int(value)


def qualifies(counts: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for count_key, threshold_key in (
        ("known_cells", "minimum_truth_known_cells_per_identity"),
        ("clear_cells", "minimum_truth_clear_cells_per_identity"),
        ("occupied_cells", "minimum_truth_occupied_cells_per_identity"),
        ("valid_band_clearances", "minimum_valid_band_clearances_per_identity"),
    ):
        if int(counts[count_key]) < int(thresholds[threshold_key]):
            failures.append(f"{count_key}={counts[count_key]}<{thresholds[threshold_key]}")
    minimum_grid = int(thresholds["minimum_truth_known_cells_per_band_horizon"])
    for key, value in sorted(counts["known_by_grid"].items()):
        if int(value) < minimum_grid:
            failures.append(f"{key}_known={value}<{minimum_grid}")
    return not failures, failures


def exact_entries(entries: list[dict[str, Any]], selected_stems: list[str], modality: str) -> list[dict[str, Any]]:
    by_stem = {Path(entry["path"]).stem: entry for entry in entries}
    require(len(by_stem) == len(entries), f"duplicate {modality} extracted stem")
    missing = [stem for stem in selected_stems if stem not in by_stem]
    require(not missing, f"missing {modality} stems: {missing[:3]}")
    return [by_stem[stem] for stem in selected_stems]


def audit_identity(
    *,
    selected_stems: list[str],
    intrinsics_entries: list[dict[str, Any]],
    depth_entries: list[dict[str, Any]],
    confidence_entries: list[dict[str, Any]],
    trajectory_path: Path,
    policy: TruthReaderPolicy,
) -> dict[str, Any]:
    intrinsics_entries = exact_entries(intrinsics_entries, selected_stems, "intrinsics")
    depth_entries = exact_entries(depth_entries, selected_stems, "depth")
    confidence_entries = exact_entries(confidence_entries, selected_stems, "confidence")
    trajectory = parse_trajectory(trajectory_path)
    counts = {
        "known_cells": 0,
        "clear_cells": 0,
        "occupied_cells": 0,
        "valid_band_clearances": 0,
        "known_by_grid": {f"{band}@{horizon:.1f}m": 0 for band in BANDS for horizon in HORIZONS},
    }
    orientation_counts = {str(index): 0 for index in range(4)}
    depth_sizes: set[tuple[int, int]] = set()
    confidence_values: set[int] = set()
    for index, stem in enumerate(selected_stems):
        with Image.open(depth_entries[index]["path"]) as image:
            depth_raw = np.asarray(image).copy()
        with Image.open(confidence_entries[index]["path"]) as image:
            confidence = np.asarray(image).copy()
        require(depth_raw.ndim == 2 and np.issubdtype(depth_raw.dtype, np.integer), "invalid depth raster")
        require(confidence.shape == depth_raw.shape, "depth/confidence size mismatch")
        require(np.issubdtype(confidence.dtype, np.integer), "invalid confidence raster")
        depth_sizes.add((int(depth_raw.shape[1]), int(depth_raw.shape[0])))
        confidence_values.update(int(value) for value in np.unique(confidence))
        intrinsics, source_size = parse_pincam(Path(intrinsics_entries[index]["path"]))
        require(source_size == (depth_raw.shape[1], depth_raw.shape[0]), "intrinsics/depth size mismatch")
        pose, _ = interpolate_camera_to_world(
            trajectory, timestamp_from_stem(stem), policy.maximum_pose_bracketing_gap_seconds
        )
        dummy_rgb = np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
        canonical = canonicalize_frame(dummy_rgb, depth_raw, confidence, intrinsics, pose)
        orientation_counts[str(canonical["rotation_index"])] += 1
        require(canonical["rotation_index"] in (1, 3), "Phase-A portrait orientation drift")
        up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
        truth = derive_assistive_truth(
            depth_mm_to_metres(canonical["depth_raw_mm"]),
            canonical["confidence"],
            canonical["intrinsics"],
            up_camera,
            policy,
        )
        add_counts(counts, summarize_truth(truth))
    require(confidence_values.issubset({0, 1, 2}), f"confidence values drift: {sorted(confidence_values)}")
    return {
        **counts,
        "frame_count": len(selected_stems),
        "depth_sizes_wh": [list(value) for value in sorted(depth_sizes)],
        "confidence_values": sorted(confidence_values),
        "orientation_counts": orientation_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"overwrite forbidden: {args.output_root}")
    protocol = load_json(args.protocol)
    phase_a = load_json(args.phase_a_manifest)
    head_result = load_json(args.head_result)
    receipt = load_json(args.license_receipt)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(phase_a.get("schema") == PHASE_A_SCHEMA, "Phase-A schema drift")
    require(head_result.get("schema") == HEAD_SCHEMA, "HEAD schema drift")
    require(receipt.get("schema") == LICENSE_SCHEMA, "license schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        path = Path(dependency["path"])
        require(path.stat().st_size == int(dependency["bytes"]), f"dependency size drift: {path}")
        require(sha256_file(path) == dependency["sha256"], f"dependency SHA drift: {path}")
    for name, path in (
        ("phase_a_manifest", args.phase_a_manifest),
        ("head_result", args.head_result),
        ("license_receipt", args.license_receipt),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(phase_a["terminal"] == "D2_PHASE_A_PORTRAIT_CONTINUITY_PASS_16_IDENTITIES_LOCKED", "Phase-A terminal drift")
    require(head_result["terminal"] == "D2_PHASE_B_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED", "HEAD terminal drift")
    require(receipt["authority"]["phase_b_head_and_body_after_phase_a_lock"] is True, "Phase-B not authorized")
    selected = phase_a["selected_phase_b"]
    require(len(selected) == 16, "Phase-B identity count drift")
    phase_a_by_video = {str(row["video_id"]): row for row in phase_a["processed"]}
    lookup = asset_lookup(head_result)
    maximum_video_bytes = max(
        sum(int(lookup[(str(row["video_id"]), asset)]["content_length_bytes"]) for asset in protocol["assets"])
        for row in selected
    )
    require(
        shutil.disk_usage(args.output_root.parent).free >= maximum_video_bytes * 3 + 2_000_000_000,
        "insufficient bounded working space",
    )
    args.output_root.mkdir(parents=True)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir()
    policy = TruthReaderPolicy()
    thresholds = protocol["truth_support_thresholds"]
    videos: list[dict[str, Any]] = []

    for index, selected_row in enumerate(selected, start=1):
        video_id = str(selected_row["video_id"])
        phase_a_row = phase_a_by_video[video_id]
        selected_stems = [str(value) for value in selected_row["selected_frame_stems"]]
        require(len(selected_stems) == 300, "selected frame count drift")
        video_root = args.output_root / "raw" / "Training" / video_id
        archives: dict[str, Path] = {}
        source_assets: list[dict[str, Any]] = []
        maps: dict[str, dict[str, str]] = {}
        for asset in protocol["assets"]:
            row = lookup[(video_id, asset)]
            archive = archive_root / video_id / asset
            digest, attempts = download_file(row["url"], archive, int(row["content_length_bytes"]))
            archives[asset] = archive
            maps[asset] = png_members_by_stem(archive)
            source_assets.append(
                {
                    "asset": asset,
                    "url": row["url"],
                    "bytes": int(row["content_length_bytes"]),
                    "sha256": digest,
                    "attempts": attempts,
                }
            )
        missing = {
            asset: [stem for stem in selected_stems if stem not in maps[asset]]
            for asset in protocol["assets"]
        }
        if any(missing.values()):
            extracted = {"lowres_depth": [], "confidence": []}
            audit = None
            qualified = False
            failures = [f"{asset}_missing_exact_stems={len(values)}" for asset, values in missing.items() if values]
        else:
            extracted = {
                "lowres_depth": extract_named_members(
                    archives["lowres_depth.zip"],
                    [maps["lowres_depth.zip"][stem] for stem in selected_stems],
                    video_root / "lowres_depth",
                ),
                "confidence": extract_named_members(
                    archives["confidence.zip"],
                    [maps["confidence.zip"][stem] for stem in selected_stems],
                    video_root / "confidence",
                ),
            }
            audit = audit_identity(
                selected_stems=selected_stems,
                intrinsics_entries=phase_a_row["extracted_intrinsics"],
                depth_entries=extracted["lowres_depth"],
                confidence_entries=extracted["confidence"],
                trajectory_path=Path(phase_a_row["trajectory"]["path"]),
                policy=policy,
            )
            qualified, failures = qualifies(audit, thresholds)
        for archive in archives.values():
            safe_delete_archive(archive, archive_root)
        value = {
            "phase_a_order": index,
            "pool_order": int(selected_row["pool_order"]),
            "visit_id": str(selected_row["visit_id"]),
            "video_id": video_id,
            "fold": str(selected_row["fold"]),
            "selected_frame_stems": selected_stems,
            "exact_stem_missing_counts": {asset: len(values) for asset, values in missing.items()},
            "source_assets": source_assets,
            "extracted": extracted,
            "truth_support": audit,
            "qualified": qualified,
            "qualification_failures": failures,
            "model_output_read": False,
        }
        write_json_exclusive(args.output_root / "receipts" / f"{index:02d}-{video_id}.json", value)
        videos.append(value)
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(selected),
                    "video_id": video_id,
                    "qualified": qualified,
                    "known": audit["known_cells"] if audit else None,
                    "clear": audit["clear_cells"] if audit else None,
                    "occupied": audit["occupied_cells"] if audit else None,
                }
            ),
            flush=True,
        )

    remove_empty_archive_tree(archive_root)
    qualified_rows = [row for row in videos if row["qualified"]]
    passed = len(qualified_rows) >= 8
    selected_roles = qualified_rows[:8] if passed else []
    role_assignments = [
        {
            "role_order": index + 1 if index < 4 else index - 3,
            "role": "D2_TRAIN" if index < 4 else "D2_DEVELOPMENT_SEALED",
            "pool_order": row["pool_order"],
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
        }
        for index, row in enumerate(selected_roles)
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "phase_a_manifest_sha256": sha256_file(args.phase_a_manifest),
        "head_result_sha256": sha256_file(args.head_result),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "identity_count": len(videos),
        "qualified_identity_count": len(qualified_rows),
        "truth_support_thresholds": thresholds,
        "role_assignments": role_assignments,
        "videos": videos,
        "truth_support_counts_read": True,
        "per_frame_truth_rows_saved": False,
        "model_output_read": False,
        "d2_development_model_outcome_opened": False,
        "r2_cohort_access": "NONE",
        "terminal": (
            "D2_PHASE_B_TRUTH_SUPPORT_PASS_4_TRAIN_4_DEVELOPMENT_ROLES_LOCKED"
            if passed
            else "D2_PHASE_B_FAIL_FEWER_THAN_8_SUPPORT_QUALIFIED_IDENTITIES"
        ),
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
