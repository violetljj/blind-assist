#!/usr/bin/env python3
"""Download and extract the locked P3 R0.2.1 ARKit validation identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

HELPER_ROOT = Path(__file__).resolve().parents[1] / "spatial_calibration_head_r1"
sys.path.insert(0, str(HELPER_ROOT))

from download_locked_assets import (  # noqa: E402
    download_file,
    extract_named_members,
    nearest_pincam_member_names,
    pincam_members,
    png_members_by_stem,
    remove_empty_archive_tree,
    safe_delete_archive,
)
from materialize_cache import timestamp_from_stem  # noqa: E402


PROTOCOL_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_download_protocol"
MANIFEST_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_media_manifest"
ASSETS = ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip", "lowres_wide_intrinsics.zip", "lowres_wide.traj")


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


def bound_file(root: Path, binding: dict[str, Any]) -> Path:
    path = (root / binding["path"]).resolve()
    require(path.is_file() and sha256_file(path) == binding["sha256"], f"bound file mismatch: {path}")
    return path


def lookup_preflight(preflight: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in preflight["assets"]:
        key = (str(row["video_id"]), str(row["asset"]))
        require(key not in result, f"duplicate preflight asset: {key}")
        require(row["http_status"] == 200 and int(row["content_length_bytes"]) > 0, f"asset unavailable: {key}")
        result[key] = row
    return result


def earliest_common_stems(member_maps: dict[str, dict[str, str]], count: int) -> list[str]:
    common = set.intersection(*(set(value) for value in member_maps.values()))
    ordered = sorted(common, key=lambda stem: (timestamp_from_stem(stem), stem))
    require(len(ordered) >= count, f"fewer than {count} common RGB-depth-confidence frames")
    selected = ordered[:count]
    times = [timestamp_from_stem(stem) for stem in selected]
    require(all(0 < right - left <= 0.5 for left, right in zip(times, times[1:])), "earliest common window violates 500 ms gap")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    require(not args.output_root.exists(), f"output root exists: {args.output_root}")
    protocol = load_json(args.protocol)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        bound_file(root, dependency)
    roster = load_json(bound_file(root, protocol["roster"]))
    preflight = load_json(bound_file(root, protocol["asset_preflight"]))
    license_receipt = load_json(bound_file(root, protocol["license_receipt"]))
    require(license_receipt["media_download_authorized"] is True and license_receipt["roster"]["sha256"] == protocol["roster"]["sha256"], "license scope mismatch")
    lookup = lookup_preflight(preflight)
    free = shutil.disk_usage(args.output_root.parent).free
    by_video = {}
    for row in preflight["assets"]:
        by_video[str(row["video_id"])] = by_video.get(str(row["video_id"]), 0) + int(row["content_length_bytes"])
    require(free >= max(by_video.values()) * 3 + 2_000_000_000, "insufficient bounded working space")

    args.output_root.mkdir(parents=True)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir()
    videos = []
    for parent in roster["selected"]:
        visit_id, video_id = str(parent["visit_id"]), str(parent["video_id"])
        video_root = args.output_root / "raw" / "Validation" / video_id
        archives, source_assets = {}, []
        for asset in ASSETS[:3]:
            row = lookup[(video_id, asset)]
            path = archive_root / video_id / asset
            digest, attempts = download_file(row["url"], path, int(row["content_length_bytes"]))
            archives[asset] = path
            source_assets.append({"asset": asset, "url": row["url"], "bytes": row["content_length_bytes"], "sha256": digest, "attempts": attempts})
        maps = {asset: png_members_by_stem(path) for asset, path in archives.items()}
        selected = earliest_common_stems(maps, int(protocol["continuous_frame_count_per_video"]))
        extracted = {}
        for asset, folder in (("lowres_wide.zip", "lowres_wide"), ("lowres_depth.zip", "lowres_depth"), ("confidence.zip", "confidence")):
            extracted[folder] = extract_named_members(archives[asset], [maps[asset][stem] for stem in selected], video_root / folder)
            safe_delete_archive(archives[asset], archive_root)

        asset = "lowres_wide_intrinsics.zip"
        row = lookup[(video_id, asset)]
        intrinsics_archive = archive_root / video_id / asset
        digest, attempts = download_file(row["url"], intrinsics_archive, int(row["content_length_bytes"]))
        source_assets.append({"asset": asset, "url": row["url"], "bytes": row["content_length_bytes"], "sha256": digest, "attempts": attempts})
        intrinsics_members = nearest_pincam_member_names(pincam_members(intrinsics_archive), selected)
        extracted["lowres_wide_intrinsics"] = extract_named_members(intrinsics_archive, intrinsics_members, video_root / "lowres_wide_intrinsics")
        safe_delete_archive(intrinsics_archive, archive_root)

        asset = "lowres_wide.traj"
        row = lookup[(video_id, asset)]
        trajectory = video_root / asset
        digest, attempts = download_file(row["url"], trajectory, int(row["content_length_bytes"]))
        source_assets.append({"asset": asset, "url": row["url"], "bytes": row["content_length_bytes"], "sha256": digest, "attempts": attempts})
        videos.append({
            "visit_id": visit_id, "video_id": video_id, "official_fold": "Validation",
            "selected_frame_stems": selected, "selected_frame_count": len(selected),
            "source_assets": source_assets, "extracted": extracted,
        })
    remove_empty_archive_tree(archive_root)
    manifest = {
        "schema": MANIFEST_SCHEMA, "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": protocol["roster"]["sha256"], "asset_preflight_sha256": protocol["asset_preflight"]["sha256"],
        "license_receipt_sha256": protocol["license_receipt"]["sha256"], "video_count": len(videos),
        "continuous_frame_count_per_video": protocol["continuous_frame_count_per_video"], "videos": videos,
        "labels_opened": False, "model_outputs_read": False, "temporary_archives_retained": False,
        "terminal": "P3_R0_2_1_ARKIT_VALIDATION_MEDIA_DOWNLOADED_CONTINUITY_AUDIT_PENDING",
    }
    path = args.output_root / "manifest.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
