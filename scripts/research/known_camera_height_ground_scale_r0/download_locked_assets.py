"""Download and extract only the frozen four-video fresh cohort."""

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

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def timestamp_from_stem(stem: str) -> float:
    try:
        return float(stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"cannot parse timestamp from {stem}") from error


def sample_150(stems: list[str]) -> list[str]:
    if len(stems) < 150:
        raise ValueError("fewer than 150 matched frame triples")
    indices = np.round(np.linspace(0, len(stems) - 1, 150)).astype(int)
    if len(set(indices.tolist())) != 150:
        raise ValueError("sampling produced duplicate indices")
    return [stems[index] for index in indices]


def nearest_intrinsics(timestamp: float, candidates: list[tuple[float, str]]) -> str:
    differences = [(abs(value - timestamp), name) for value, name in candidates]
    if not differences:
        raise ValueError("no intrinsics files")
    difference, name = min(differences, key=lambda row: (row[0], row[1]))
    if difference > 0.0015:
        raise ValueError(f"no intrinsics within 1.5 ms of {timestamp}")
    if sum(abs(value - timestamp) == difference for value, _ in candidates) != 1:
        raise ValueError("ambiguous nearest intrinsics")
    return name


def download_file(
    url: str, output: Path, expected_length: int, retries: int = 3
) -> tuple[str, int]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(1, retries + 1):
        partial = output.with_name(f"{output.name}.attempt-{attempt}.partial")
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BlindAssist-known-height-r0-scoped-download"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                "xb"
            ) as stream:
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


def _safe_members(bundle: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    bad = bundle.testzip()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    result = {}
    for name in bundle.namelist():
        pure = Path(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe ZIP member: {name}")
        if pure.suffix.lower() == suffix:
            if pure.stem in result:
                raise ValueError(f"duplicate member stem: {pure.stem}")
            result[pure.stem] = name
    return result


def member_map(archive: Path, suffix: str) -> dict[str, str]:
    with zipfile.ZipFile(archive) as bundle:
        return _safe_members(bundle, suffix)


def extract_named_members(
    archive: Path, names: list[str], destination: Path
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    with zipfile.ZipFile(archive) as bundle:
        for name in names:
            target = destination / Path(name).name
            if target.exists():
                raise FileExistsError(target)
            digest = hashlib.sha256()
            with bundle.open(name) as source, target.open("xb") as stream:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                stream.flush()
                os.fsync(stream.fileno())
            outputs.append(
                {
                    "member": name,
                    "path": str(target.resolve()),
                    "sha256": digest.hexdigest().upper(),
                    "bytes": target.stat().st_size,
                }
            )
    return outputs


def safe_delete_archive(path: Path, archive_root: Path) -> None:
    resolved = path.resolve()
    root = archive_root.resolve()
    if root not in resolved.parents or resolved.suffix not in (".zip", ".partial"):
        raise ValueError(f"refusing archive cleanup outside scoped root: {resolved}")
    path.unlink()


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


def process_asset(
    row: dict[str, Any], archive_root: Path, video_id: str, asset: str
) -> tuple[Path, dict[str, Any]]:
    archive = archive_root / video_id / asset
    digest, attempts = download_file(
        row["url"], archive, int(row["content_length_bytes"])
    )
    return archive, {
        "asset": asset,
        "url": row["url"],
        "content_length_bytes": row["content_length_bytes"],
        "archive_sha256": digest,
        "download_attempts": attempts,
    }


def write_json_new(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--asset-preflight", required=True, type=Path)
    parser.add_argument("--license-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output_root.exists():
        raise FileExistsError(arguments.output_root)
    lock = json.loads(arguments.roster_lock.read_text(encoding="utf-8"))
    preflight = json.loads(arguments.asset_preflight.read_text(encoding="utf-8"))
    license_receipt = json.loads(arguments.license_receipt.read_text(encoding="utf-8"))
    protocol_hash = sha256(arguments.protocol)
    lock_hash = sha256(arguments.roster_lock)
    preflight_hash = sha256(arguments.asset_preflight)
    if lock.get("protocol_sha256") != protocol_hash:
        raise ValueError("roster authority mismatch")
    if (
        preflight.get("terminal") != "KNOWN_HEIGHT_R0_ASSET_HEADERS_AVAILABLE"
        or preflight.get("protocol_sha256") != protocol_hash
        or preflight.get("roster_lock_sha256") != lock_hash
    ):
        raise ValueError("asset preflight authority mismatch")
    if (
        license_receipt.get("media_download_authorized") is not True
        or license_receipt.get("protocol_sha256") != protocol_hash
        or license_receipt.get("roster_lock_sha256") != lock_hash
        or license_receipt.get("asset_preflight_sha256") != preflight_hash
    ):
        raise ValueError("license scope receipt mismatch")
    if set(license_receipt["authorized_scope"]["video_ids"]) != {
        str(row["video_id"]) for row in lock["fresh_evaluation"]
    }:
        raise ValueError("license video scope mismatch")

    lookup = entry_lookup(preflight)
    by_video: dict[str, int] = {}
    for row in preflight["assets"]:
        video_id = str(row["video_id"])
        by_video[video_id] = by_video.get(video_id, 0) + int(
            row["content_length_bytes"]
        )
    free_bytes = shutil.disk_usage(arguments.output_root.parent).free
    if free_bytes < max(by_video.values()) * 3 + 2_000_000_000:
        raise OSError("insufficient free space for bounded one-video working set")

    arguments.output_root.mkdir(parents=True)
    archive_root = arguments.output_root / "_temporary_archives"
    archive_root.mkdir()
    videos = []
    for roster_row in lock["fresh_evaluation"]:
        video_id = str(roster_row["video_id"])
        video_root = arguments.output_root / "raw" / "Validation" / video_id
        archives = {}
        receipts = []
        for asset in ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip"):
            archives[asset], receipt = process_asset(
                lookup[(video_id, asset)], archive_root, video_id, asset
            )
            receipts.append(receipt)
        maps = {asset: member_map(path, ".png") for asset, path in archives.items()}
        common = sorted(
            set.intersection(*(set(values) for values in maps.values())),
            key=lambda stem: (timestamp_from_stem(stem), stem),
        )
        selected = sample_150(common)
        extracted = {}
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

        intrinsics_asset = "lowres_wide_intrinsics.zip"
        intrinsics_archive, receipt = process_asset(
            lookup[(video_id, intrinsics_asset)],
            archive_root,
            video_id,
            intrinsics_asset,
        )
        receipts.append(receipt)
        intrinsics_map = member_map(intrinsics_archive, ".pincam")
        intrinsics_candidates = sorted(
            (timestamp_from_stem(stem), name)
            for stem, name in intrinsics_map.items()
        )
        selected_intrinsics = sorted(
            {
                nearest_intrinsics(timestamp_from_stem(stem), intrinsics_candidates)
                for stem in selected
            }
        )
        extracted["lowres_wide_intrinsics"] = extract_named_members(
            intrinsics_archive,
            selected_intrinsics,
            video_root / "lowres_wide_intrinsics",
        )
        safe_delete_archive(intrinsics_archive, archive_root)

        trajectory_row = lookup[(video_id, "lowres_wide.traj")]
        trajectory_path = video_root / "lowres_wide.traj"
        trajectory_hash, attempts = download_file(
            trajectory_row["url"],
            trajectory_path,
            int(trajectory_row["content_length_bytes"]),
        )
        receipts.append(
            {
                "asset": "lowres_wide.traj",
                "url": trajectory_row["url"],
                "content_length_bytes": trajectory_row["content_length_bytes"],
                "archive_sha256": trajectory_hash,
                "download_attempts": attempts,
            }
        )
        videos.append(
            {
                "role": "fresh_evaluation",
                "visit_id": roster_row["visit_id"],
                "video_id": video_id,
                "selected_frame_stems": selected,
                "matched_source_stem_count": len(common),
                "source_assets": receipts,
                "extracted": extracted,
            }
        )

    remaining = [path for path in archive_root.rglob("*") if path.is_file()]
    if remaining:
        raise ValueError(f"temporary archive files remain: {remaining}")
    for directory in sorted(
        (path for path in archive_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    archive_root.rmdir()
    manifest = {
        "schema": "blindassist_known_camera_height_ground_scale_r0_scoped_media_manifest",
        "protocol_sha256": protocol_hash,
        "roster_lock_sha256": lock_hash,
        "asset_preflight_sha256": preflight_hash,
        "license_scope_receipt_sha256": sha256(arguments.license_receipt),
        "videos": videos,
        "fresh_video_count": len(videos),
        "selected_frame_count": sum(
            len(video["selected_frame_stems"]) for video in videos
        ),
        "temporary_archives_retained": False,
        "outcomes_evaluated": False,
        "terminal": "KNOWN_HEIGHT_R0_SCOPED_MEDIA_DOWNLOADED_QUALIFICATION_PENDING",
    }
    write_json_new(arguments.output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "videos"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
