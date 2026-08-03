#!/usr/bin/env python3
"""Download only frozen R1 members, one video at a time, after license receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from materialize_cache import nearest_intrinsics, sample_150, timestamp_from_stem
from validate_protocol import DEFAULT_PROTOCOL, sha256, validate


def download_file(url: str, output: Path, expected_length: int, retries: int = 3) -> tuple[str, int]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(1, retries + 1):
        partial = output.with_name(f"{output.name}.attempt-{attempt}.partial")
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-R1-scoped-download"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if size != expected_length:
                raise ValueError(f"content length mismatch: {size} != {expected_length}")
            os.replace(partial, output)
            return digest.hexdigest().upper(), attempt
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            if partial.exists():
                partial.unlink()
    raise OSError(f"download failed after {retries} attempts: {errors}")


def png_members_by_stem(archive: Path) -> dict[str, str]:
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
        result = {}
        for name in bundle.namelist():
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
            if pure.suffix.lower() == ".png":
                if pure.stem in result:
                    raise ValueError(f"duplicate PNG stem in ZIP: {pure.stem}")
                result[pure.stem] = name
    return result


def pincam_members(archive: Path) -> list[tuple[float, str]]:
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
        rows = []
        for name in bundle.namelist():
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
            if pure.suffix.lower() == ".pincam":
                rows.append((timestamp_from_stem(pure.stem), name))
        return sorted(rows)


def nearest_pincam_member_names(candidates: list[tuple[float, str]], selected_stems: list[str]) -> list[str]:
    """Return ZIP member names without converting archive separators on Windows."""
    if not candidates:
        raise ValueError("no intrinsics candidates")
    candidate_paths = [(timestamp, Path(name)) for timestamp, name in candidates]
    canonical_by_path = {str(Path(name)): name for _, name in candidates}
    selected_names = []
    for stem in selected_stems:
        matched_path = nearest_intrinsics(timestamp_from_stem(stem), candidate_paths)
        selected_names.append(canonical_by_path[str(matched_path)])
    return sorted(set(selected_names))


def extract_named_members(archive: Path, member_names: list[str], destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    with zipfile.ZipFile(archive) as bundle:
        for member_name in member_names:
            target = destination / Path(member_name).name
            if target.exists():
                raise FileExistsError(target)
            digest = hashlib.sha256()
            with bundle.open(member_name) as source, target.open("xb") as stream:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                stream.flush()
                os.fsync(stream.fileno())
            outputs.append({"member": member_name, "path": str(target.resolve()), "sha256": digest.hexdigest().upper(), "bytes": target.stat().st_size})
    return outputs


def safe_delete_archive(path: Path, archive_root: Path) -> None:
    resolved = path.resolve()
    root = archive_root.resolve()
    if root not in resolved.parents or resolved.suffix not in (".zip", ".partial"):
        raise ValueError(f"refusing archive cleanup outside scoped root: {resolved}")
    path.unlink()


def remove_empty_archive_tree(archive_root: Path) -> None:
    """Remove only empty scoped directories; fail if any file remains."""
    remaining_files = [path for path in archive_root.rglob("*") if path.is_file()]
    if remaining_files:
        raise ValueError(f"temporary archive files remain: {remaining_files}")
    for directory in sorted(
        (path for path in archive_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    archive_root.rmdir()


def entry_lookup(preflight: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in preflight["assets"]:
        key = (str(row["video_id"]), str(row["asset"]))
        if key in result:
            raise ValueError(f"duplicate preflight asset: {key}")
        if row["http_status"] != 200 or not row["content_length_bytes"]:
            raise ValueError(f"unavailable preflight asset: {key}")
        result[key] = row
    return result


def process_zip(
    row: dict[str, Any], archive_root: Path, video_id: str, asset: str
) -> tuple[Path, dict[str, Any]]:
    archive = archive_root / video_id / asset
    digest, attempts = download_file(row["url"], archive, int(row["content_length_bytes"]))
    return archive, {
        "asset": asset, "url": row["url"], "content_length_bytes": row["content_length_bytes"],
        "archive_sha256": digest, "download_attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--roster-lock", type=Path, required=True)
    parser.add_argument("--asset-preflight", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError(f"protocol invalid: {errors}")
    lock = json.loads(args.roster_lock.read_text(encoding="utf-8"))
    preflight = json.loads(args.asset_preflight.read_text(encoding="utf-8"))
    license_receipt = json.loads(args.license_receipt.read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol)
    lock_hash = sha256(args.roster_lock)
    preflight_hash = sha256(args.asset_preflight)
    if lock.get("protocol_sha256") != protocol_hash:
        raise ValueError("roster authority mismatch")
    if preflight.get("terminal") != "SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADERS_AVAILABLE" or preflight.get("protocol_sha256") != protocol_hash or preflight.get("roster_lock_sha256") != lock_hash:
        raise ValueError("asset preflight authority mismatch")
    if license_receipt.get("media_download_authorized") is not True or license_receipt.get("protocol_sha256") != protocol_hash or license_receipt.get("roster_lock_sha256") != lock_hash or license_receipt.get("asset_preflight_sha256") != preflight_hash:
        raise ValueError("license receipt authority mismatch")
    if license_receipt.get("authorized_scope", {}).get("sealed_metric_assets") is not False:
        raise ValueError("sealed metric asset scope must remain forbidden")

    free_bytes = shutil.disk_usage(args.output_root.parent).free
    maximum_video_archives = 0
    by_video: dict[str, int] = {}
    for row in preflight["assets"]:
        by_video[str(row["video_id"])] = by_video.get(str(row["video_id"]), 0) + int(row["content_length_bytes"])
    maximum_video_archives = max(by_video.values())
    if free_bytes < maximum_video_archives * 3 + 2_000_000_000:
        raise OSError("insufficient free space for bounded one-video working set")

    args.output_root.mkdir(parents=True)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir()
    lookup = entry_lookup(preflight)
    videos = []
    for role in ("train", "validation"):
        for roster_row in lock["roles"][role]:
            video_id = str(roster_row["video_id"])
            video_root = args.output_root / "raw" / "Training" / video_id
            archives = {}
            receipts = []
            for asset in ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip"):
                archives[asset], receipt = process_zip(lookup[(video_id, asset)], archive_root, video_id, asset)
                receipts.append(receipt)
            member_maps = {asset: png_members_by_stem(path) for asset, path in archives.items()}
            common = sorted(set.intersection(*(set(value) for value in member_maps.values())), key=lambda stem: (timestamp_from_stem(stem), stem))
            selected = sample_150(common)
            extracted = {}
            for asset, folder in (("lowres_wide.zip", "lowres_wide"), ("lowres_depth.zip", "lowres_depth"), ("confidence.zip", "confidence")):
                extracted[folder] = extract_named_members(archives[asset], [member_maps[asset][stem] for stem in selected], video_root / folder)
                safe_delete_archive(archives[asset], archive_root)

            intrinsics_archive, intrinsics_receipt = process_zip(lookup[(video_id, "lowres_wide_intrinsics.zip")], archive_root, video_id, "lowres_wide_intrinsics.zip")
            receipts.append(intrinsics_receipt)
            candidates = pincam_members(intrinsics_archive)
            selected_members = nearest_pincam_member_names(candidates, selected)
            extracted["lowres_wide_intrinsics"] = extract_named_members(intrinsics_archive, selected_members, video_root / "lowres_wide_intrinsics")
            safe_delete_archive(intrinsics_archive, archive_root)

            trajectory_row = lookup[(video_id, "lowres_wide.traj")]
            trajectory_path = video_root / "lowres_wide.traj"
            trajectory_hash, attempts = download_file(trajectory_row["url"], trajectory_path, int(trajectory_row["content_length_bytes"]))
            receipts.append({"asset": "lowres_wide.traj", "url": trajectory_row["url"], "content_length_bytes": trajectory_row["content_length_bytes"], "archive_sha256": trajectory_hash, "download_attempts": attempts})
            videos.append({
                "role": role, "visit_id": roster_row["visit_id"], "video_id": video_id,
                "selected_frame_stems": selected, "matched_source_stem_count": len(common),
                "source_assets": receipts, "extracted": extracted,
            })

    for roster_row in lock["roles"]["sealed"]:
        video_id = str(roster_row["video_id"])
        archive, receipt = process_zip(lookup[(video_id, "lowres_wide.zip")], archive_root, video_id, "lowres_wide.zip")
        members = png_members_by_stem(archive)
        ordered = sorted(members, key=lambda stem: (timestamp_from_stem(stem), stem))
        selected = sample_150(ordered)
        destination = args.output_root / "identity_only" / "Validation" / video_id / "lowres_wide"
        extracted = extract_named_members(archive, [members[stem] for stem in selected], destination)
        safe_delete_archive(archive, archive_root)
        videos.append({
            "role": "sealed_identity_only", "visit_id": roster_row["visit_id"], "video_id": video_id,
            "selected_frame_stems": selected, "source_rgb_stem_count": len(ordered),
            "source_assets": [receipt], "extracted": {"lowres_wide": extracted},
            "sealed_metric_assets_read": False,
        })
    remove_empty_archive_tree(archive_root)
    manifest = {
        "schema": "blindassist_spatial_calibration_head_r1_scoped_media_manifest",
        "protocol_sha256": protocol_hash,
        "roster_lock_sha256": lock_hash,
        "asset_preflight_sha256": preflight_hash,
        "license_receipt_sha256": sha256(args.license_receipt),
        "videos": videos,
        "development_video_count": 20,
        "sealed_identity_video_count": 4,
        "development_selected_frame_count": 3000,
        "sealed_identity_selected_frame_count": 600,
        "sealed_metric_assets_read": False,
        "temporary_archives_retained": False,
        "terminal": "SPATIAL_CALIBRATION_HEAD_R1_SCOPED_MEDIA_DOWNLOADED_ASSET_QUALIFICATION_PENDING",
    }
    manifest_path = args.output_root / "manifest.json"
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
