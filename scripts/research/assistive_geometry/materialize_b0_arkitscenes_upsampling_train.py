#!/usr/bin/env python3
"""Materialize TRAIN-only ARKitScenes FARO/AppleDepth validation pairs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
HELPER_ROOT = Path(__file__).resolve().parents[1] / "spatial_calibration_head_r1"
sys.path.insert(0, str(HELPER_ROOT))

from download_locked_assets import (  # noqa: E402
    download_file,
    extract_named_members,
    nearest_pincam_member_names,
    pincam_members,
    remove_empty_archive_tree,
    safe_delete_archive,
)
from scripts.research.assistive_geometry.arkitscenes_truth_reader import parse_trajectory  # noqa: E402
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    bound_file,
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_pose_covered_assets import (  # noqa: E402
    pose_covered_common_stems,
)


PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_upsampling_train_protocol_v1"
MANIFEST_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_upsampling_train_manifest_v1"
MODALITIES = ("wide", "highres_depth", "lowres_depth", "confidence")


def upsampling_png_members(archive: Path) -> dict[str, dict[str, str]]:
    """Index repeated frame stems independently within each official modality."""
    result: dict[str, dict[str, str]] = {modality: {} for modality in MODALITIES}
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        require(bad is None, f"ZIP CRC failure: {bad}")
        for name in bundle.namelist():
            pure = Path(name)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {name}")
            if pure.suffix.lower() != ".png" or len(pure.parts) < 2:
                continue
            modality = pure.parts[-2]
            if modality not in result:
                continue
            require(pure.stem not in result[modality], f"duplicate {modality} PNG stem: {pure.stem}")
            result[modality][pure.stem] = name
    require(all(result.values()), "upsampling modality missing")
    return result


def download_or_reuse(url: str, output: Path, expected_length: int) -> tuple[str, int]:
    """Resume only an exact-size completed download; partials remain fail-closed."""
    if output.exists():
        require(output.is_file() and output.stat().st_size == expected_length, f"existing download size drift: {output}")
        return sha256_file(output), 0
    return download_file(url, output, expected_length)


def metadata_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, "upsampling metadata is empty")
    required = {"video_id", "fold", "sky_direction"}
    require(required.issubset(rows[0]), "upsampling metadata schema drift")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        require(video_id not in result, f"duplicate upsampling metadata video: {video_id}")
        result[video_id] = row
    return result


def train_video_lookup(manifest: dict[str, Any], expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    videos = {
        str(video["video_id"]): video
        for video in manifest["videos"]
        if video["role"] == "TRAIN"
    }
    require(len(videos) == 16, "main manifest TRAIN count drift")
    require(len(expected_ids) == len(set(expected_ids)), "duplicate expected upsampling video")
    missing = sorted(set(expected_ids) - set(videos))
    require(not missing, f"upsampling videos outside frozen TRAIN role: {missing}")
    return {video_id: videos[video_id] for video_id in expected_ids}


def _source_receipt(source: dict[str, Any], digest: str, attempts: int) -> dict[str, Any]:
    return {
        "asset": source["asset"],
        "url": source["url"],
        "bytes": int(source["content_length_bytes"]),
        "sha256": digest,
        "attempts": attempts,
    }


def validate_receipt(path: Path, video_id: str) -> dict[str, Any]:
    receipt = load_json(path)
    require(receipt.get("video_id") == video_id, f"receipt video drift: {path}")
    require(receipt.get("role") == "TRAIN", f"receipt role drift: {path}")
    require(int(receipt.get("selected_frame_count", 0)) >= 1, f"receipt has no frames: {path}")
    for entries in receipt["extracted"].values():
        for entry in entries:
            output = Path(entry["path"])
            require(output.is_file() and output.stat().st_size == int(entry["bytes"]), f"receipt output drift: {output}")
            require(sha256_file(output) == entry["sha256"], f"receipt output SHA drift: {output}")
    return receipt


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
    manifest_path = bound_file(root, protocol["main_manifest"])
    main_manifest = load_json(manifest_path)
    authorization = load_json(bound_file(root, protocol["authorization_receipt"]))
    require(authorization["interpreted_scope"]["purpose"] == "TRAIN-only independent source-depth and registration validation", "authorization purpose drift")
    expected_ids = [str(value) for value in protocol["video_ids"]]
    videos = train_video_lookup(main_manifest, expected_ids)
    sources = {str(row["video_id"]): row for row in protocol["sources"]}
    require(set(sources) == set(expected_ids), "source/video identity drift")
    free = shutil.disk_usage(args.output_root.parent).free
    source_bytes = [sum(int(asset["content_length_bytes"]) for asset in row["assets"]) for row in sources.values()]
    require(free >= sum(source_bytes) + max(source_bytes) * 3 + 2_000_000_000, "insufficient working space")

    attempt = {
        "schema": "blindassist_assistive_geometry_b0_arkitscenes_upsampling_train_attempt_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "main_manifest_sha256": protocol["main_manifest"]["sha256"],
        "authorization_receipt_sha256": protocol["authorization_receipt"]["sha256"],
        "output_root": str(args.output_root.resolve()),
    }
    attempt_path = args.output_root / "attempt.json"
    if args.output_root.exists():
        require(args.resume, "existing output requires explicit --resume")
        require(attempt_path.is_file() and load_json(attempt_path) == attempt, "resume attempt drift")
        require(not (args.output_root / "manifest.json").exists(), "completed manifest already exists")
    else:
        args.output_root.mkdir(parents=True)
        write_json_exclusive(attempt_path, attempt)
    archive_root = args.output_root / "_temporary_archives"
    archive_root.mkdir(exist_ok=True)

    metadata_source = protocol["metadata"]
    metadata_path = args.output_root / "metadata.csv"
    digest, attempts = download_or_reuse(
        metadata_source["url"], metadata_path, int(metadata_source["content_length_bytes"])
    )
    metadata_receipt = _source_receipt(metadata_source, digest, attempts)
    metadata = metadata_rows(metadata_path)

    receipts: list[dict[str, Any]] = []
    for index, video_id in enumerate(expected_ids, start=1):
        receipt_path = args.output_root / "receipts" / f"{index:02d}-TRAIN-{video_id}.json"
        if receipt_path.exists():
            receipts.append(validate_receipt(receipt_path, video_id))
            print(json.dumps({"completed": index, "total": len(expected_ids), "video_id": video_id, "resumed": True}), flush=True)
            continue
        video = videos[video_id]
        source = sources[video_id]
        require(video_id in metadata, f"upsampling metadata missing video: {video_id}")
        metadata_row = metadata[video_id]
        require(metadata_row["fold"] == "Training", f"upsampling metadata role drift: {video_id}")
        require(metadata_row["sky_direction"] in ("Up", "Down", "Left", "Right"), f"invalid sky direction: {video_id}")
        by_asset = {asset["asset"]: asset for asset in source["assets"]}
        require(set(by_asset) == {"upsampling.zip", "lowres_wide_intrinsics.zip"}, f"asset set drift: {video_id}")

        upsampling_source = by_asset["upsampling.zip"]
        upsampling_archive = archive_root / video_id / "upsampling.zip"
        digest, attempts = download_or_reuse(
            upsampling_source["url"],
            upsampling_archive,
            int(upsampling_source["content_length_bytes"]),
        )
        source_receipts = [_source_receipt(upsampling_source, digest, attempts)]
        maps = upsampling_png_members(upsampling_archive)
        trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
        selected = pose_covered_common_stems(
            maps,
            1,
            float(trajectory[0, 0]),
            float(trajectory[-1, 0]),
        )
        common = set.intersection(*(set(value) for value in maps.values()))
        selected = sorted(
            [stem for stem in common if float(trajectory[0, 0]) <= float(stem.rsplit("_", 1)[1]) <= float(trajectory[-1, 0])],
            key=lambda stem: (float(stem.rsplit("_", 1)[1]), stem),
        )
        require(selected, f"no pose-covered upsampling frames: {video_id}")
        video_root = args.output_root / "raw" / video_id
        extracted: dict[str, Any] = {}
        for modality in MODALITIES:
            extracted[modality] = extract_named_members(
                upsampling_archive,
                [maps[modality][stem] for stem in selected],
                video_root / modality,
            )
        safe_delete_archive(upsampling_archive, archive_root)

        intrinsics_source = by_asset["lowres_wide_intrinsics.zip"]
        intrinsics_archive = archive_root / video_id / "lowres_wide_intrinsics.zip"
        digest, attempts = download_or_reuse(
            intrinsics_source["url"],
            intrinsics_archive,
            int(intrinsics_source["content_length_bytes"]),
        )
        source_receipts.append(_source_receipt(intrinsics_source, digest, attempts))
        intrinsics_members = nearest_pincam_member_names(pincam_members(intrinsics_archive), selected)
        extracted["lowres_wide_intrinsics"] = extract_named_members(
            intrinsics_archive,
            intrinsics_members,
            video_root / "lowres_wide_intrinsics",
        )
        safe_delete_archive(intrinsics_archive, archive_root)

        receipt = {
            "role": "TRAIN",
            "visit_id": str(video["visit_id"]),
            "video_id": video_id,
            "sky_direction": metadata_row["sky_direction"],
            "selected_frame_stems": selected,
            "selected_frame_count": len(selected),
            "trajectory": video["trajectory"],
            "source_assets": source_receipts,
            "extracted": extracted,
        }
        write_json_exclusive(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps({"completed": index, "total": len(expected_ids), "video_id": video_id, "frames": len(selected)}), flush=True)

    remove_empty_archive_tree(archive_root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "main_manifest_sha256": protocol["main_manifest"]["sha256"],
        "authorization_receipt_sha256": protocol["authorization_receipt"]["sha256"],
        "metadata": metadata_receipt,
        "video_count": len(receipts),
        "frame_count": sum(int(receipt["selected_frame_count"]) for receipt in receipts),
        "videos": receipts,
        "development_or_confirmation_opened": False,
        "model_outputs_read": False,
        "terminal": "B0_ARKIT_UPSAMPLING_TRAIN_MATERIALIZED_VALIDATION_PENDING",
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
