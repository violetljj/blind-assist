#!/usr/bin/env python3
"""Download the frozen B0 ARKitScenes rosters one video at a time."""

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


PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_download_protocol_v1"
MANIFEST_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_media_manifest_v1"
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
    require(sha256_file(path) == binding["sha256"], f"bound file SHA mismatch: {path}")
    return path


def lookup_preflight(preflight: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in preflight["assets"]:
        key = (str(row["video_id"]), str(row["asset"]))
        require(key not in result, f"duplicate preflight asset: {key}")
        require(
            row["http_status"] == 200 and int(row["content_length_bytes"]) > 0,
            f"asset unavailable: {key}",
        )
        result[key] = row
    return result


def earliest_common_stems(member_maps: dict[str, dict[str, str]], count: int) -> list[str]:
    common = set.intersection(*(set(value) for value in member_maps.values()))
    ordered = sorted(common, key=lambda stem: (timestamp_from_stem(stem), stem))
    require(len(ordered) >= count, f"fewer than {count} common RGB-depth-confidence frames")
    selected = ordered[:count]
    times = [timestamp_from_stem(stem) for stem in selected]
    require(
        all(0 < right - left <= 0.5 for left, right in zip(times, times[1:])),
        "earliest common window violates 500 ms gap",
    )
    return selected


def roster_rows(roster: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for role in ("TRAIN", "DEVELOPMENT", "CONFIRMATION"):
        for parent in roster["roles"][role]:
            rows.append(
                {
                    "role": role,
                    "visit_id": str(parent["visit_id"]),
                    "video_id": str(parent["video_id"]),
                    "official_fold": str(parent["official_fold"]),
                }
            )
    require(len(rows) == 32, "expected exactly 32 roster rows")
    require(len({row["visit_id"] for row in rows}) == 32, "visit overlap")
    require(len({row["video_id"] for row in rows}) == 32, "video overlap")
    return rows


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_video_receipt(path: Path, parent: dict[str, str], frame_count: int) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("role") == parent["role"], f"receipt role drift: {path}")
    require(value.get("visit_id") == parent["visit_id"], f"receipt visit drift: {path}")
    require(value.get("video_id") == parent["video_id"], f"receipt video drift: {path}")
    require(value.get("official_fold") == parent["official_fold"], f"receipt fold drift: {path}")
    require(value.get("selected_frame_count") == frame_count, f"receipt frame count drift: {path}")
    for outputs in value.get("extracted", {}).values():
        for output in outputs:
            output_path = Path(output["path"])
            require(output_path.is_file(), f"receipt output missing: {output_path}")
            require(output_path.stat().st_size == output["bytes"], f"receipt output size drift: {output_path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    protocol = load_json(args.protocol)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        bound_file(root, dependency)
    roster = load_json(bound_file(root, protocol["roster"]))
    preflight = load_json(bound_file(root, protocol["asset_preflight"]))
    authorization = load_json(bound_file(root, protocol["authorization_receipt"]))
    require(
        authorization["interpreted_scope"]["new_source_media_download"] is True,
        "media download is not authorized",
    )
    require(
        authorization["interpreted_scope"]["arkitscenes_roster"]["sha256"]
        == protocol["roster"]["sha256"],
        "authorization roster mismatch",
    )
    require(
        preflight["terminal"] == "B0_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED",
        "asset preflight terminal mismatch",
    )
    lookup = lookup_preflight(preflight)
    rows = roster_rows(roster)
    free = shutil.disk_usage(args.output_root.parent).free
    by_video: dict[str, int] = {}
    for row in preflight["assets"]:
        video_id = str(row["video_id"])
        by_video[video_id] = by_video.get(video_id, 0) + int(row["content_length_bytes"])
    require(free >= max(by_video.values()) * 3 + 2_000_000_000, "insufficient bounded working space")

    attempt = {
        "schema": "blindassist_assistive_geometry_b0_arkitscenes_download_attempt_v1",
        "protocol_sha256": sha256_file(args.protocol),
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
    for index, parent in enumerate(rows, start=1):
        role = parent["role"]
        visit_id = parent["visit_id"]
        video_id = parent["video_id"]
        official_fold = parent["official_fold"]
        receipt_path = args.output_root / "receipts" / f"{index:02d}-{role}-{video_id}.json"
        if receipt_path.exists():
            videos.append(
                validate_video_receipt(
                    receipt_path,
                    parent,
                    int(protocol["continuous_frame_count_per_video"]),
                )
            )
            print(json.dumps({"completed": index, "total": len(rows), "role": role, "video_id": video_id, "resumed": True}), flush=True)
            continue
        video_root = args.output_root / "raw" / official_fold / video_id
        archives: dict[str, Path] = {}
        source_assets: list[dict[str, Any]] = []
        for asset in ASSETS[:3]:
            row = lookup[(video_id, asset)]
            path = archive_root / video_id / asset
            digest, attempts = download_file(row["url"], path, int(row["content_length_bytes"]))
            archives[asset] = path
            source_assets.append(
                {"asset": asset, "url": row["url"], "bytes": row["content_length_bytes"], "sha256": digest, "attempts": attempts}
            )
        maps = {asset: png_members_by_stem(path) for asset, path in archives.items()}
        selected = earliest_common_stems(maps, int(protocol["continuous_frame_count_per_video"]))
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

        asset = "lowres_wide_intrinsics.zip"
        row = lookup[(video_id, asset)]
        intrinsics_archive = archive_root / video_id / asset
        digest, attempts = download_file(row["url"], intrinsics_archive, int(row["content_length_bytes"]))
        source_assets.append(
            {"asset": asset, "url": row["url"], "bytes": row["content_length_bytes"], "sha256": digest, "attempts": attempts}
        )
        intrinsics_members = nearest_pincam_member_names(pincam_members(intrinsics_archive), selected)
        extracted["lowres_wide_intrinsics"] = extract_named_members(
            intrinsics_archive,
            intrinsics_members,
            video_root / "lowres_wide_intrinsics",
        )
        safe_delete_archive(intrinsics_archive, archive_root)

        asset = "lowres_wide.traj"
        row = lookup[(video_id, asset)]
        trajectory = video_root / asset
        digest, attempts = download_file(row["url"], trajectory, int(row["content_length_bytes"]))
        source_assets.append(
            {"asset": asset, "url": row["url"], "bytes": row["content_length_bytes"], "sha256": digest, "attempts": attempts}
        )
        video_receipt = {
            "role": role,
            "visit_id": visit_id,
            "video_id": video_id,
            "official_fold": official_fold,
            "selected_frame_stems": selected,
            "selected_frame_count": len(selected),
            "source_assets": source_assets,
            "extracted": extracted,
        }
        write_json_exclusive(receipt_path, video_receipt)
        videos.append(video_receipt)
        print(json.dumps({"completed": index, "total": len(rows), "role": role, "video_id": video_id}), flush=True)
    remove_empty_archive_tree(archive_root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": protocol["roster"]["sha256"],
        "asset_preflight_sha256": protocol["asset_preflight"]["sha256"],
        "authorization_receipt_sha256": protocol["authorization_receipt"]["sha256"],
        "video_count": len(videos),
        "continuous_frame_count_per_video": protocol["continuous_frame_count_per_video"],
        "videos": videos,
        "task_outcome_opened": False,
        "model_outputs_read": False,
        "temporary_archives_retained": False,
        "terminal": "B0_ARKIT_MEDIA_DOWNLOADED_LABEL_BLIND_INTEGRITY_AUDIT_PENDING",
    }
    path = args.output_root / "manifest.json"
    write_json_exclusive(path, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
