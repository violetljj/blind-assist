#!/usr/bin/env python3
"""Materialize a label-blind source catalog for clearance-fusion R0.

Only archive member names, source timestamps and cryptographic bindings are
read. Image/depth bytes, labels and model outputs are never decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_source_catalog"
MEDIA_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_media_manifest"
ZIP_PNG_RE = re.compile(r"_(\d+(?:\.\d+)?)\.png$", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_tum_index(payload: str, prefix: str) -> dict[float, str]:
    result: dict[float, str] = {}
    for raw in payload.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"malformed {prefix} index row")
        timestamp = float(parts[0])
        require(timestamp not in result, f"duplicate {prefix} timestamp")
        result[timestamp] = parts[1]
    require(result, f"empty {prefix} index")
    return result


def tum_frames(archive_path: Path, parent_id: str) -> list[dict[str, Any]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        require("rgb.txt" in members and "depth.txt" in members, f"TUM metadata missing: {parent_id}")
        rgb = parse_tum_index(archive.extractfile(members["rgb.txt"]).read().decode("utf-8"), "rgb")
        depth = parse_tum_index(archive.extractfile(members["depth.txt"]).read().decode("utf-8"), "depth")
    depth_times = sorted(depth)
    pairs: list[tuple[float, float]] = []
    for timestamp in sorted(rgb):
        nearest = min(depth_times, key=lambda value: abs(value - timestamp))
        require(abs(nearest - timestamp) <= 0.02, f"RGB-depth association gap exceeds 20 ms: {parent_id}")
        pairs.append((timestamp, nearest))
    require(pairs, f"no RGB-depth timestamp association: {parent_id}")
    return [
        {
            "frame_id": f"TUM_RGBD:{parent_id}:{timestamp:.9f}",
            "dataset": "TUM_RGBD",
            "parent_id": parent_id,
            "video_id": parent_id,
            "timestamp_ns": int(round(timestamp * 1_000_000_000)),
            "source_timestamp_s": timestamp,
            "rgb_member": rgb[timestamp],
            "depth_member": depth[depth_timestamp],
            "confidence_member": None,
        }
        for timestamp, depth_timestamp in pairs
    ]


def arkit_frames(repo_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == MEDIA_SCHEMA, "ARKit media manifest schema drift")
    require(manifest.get("labels_opened") is False and manifest.get("model_outputs_read") is False, "media boundary violated")
    by_video: dict[str, dict[str, Any]] = {}
    for row in manifest["files"]:
        by_video.setdefault(str(row["video_id"]), {})[str(row["asset"])] = row
    result: list[dict[str, Any]] = []
    media_root = repo_root / "artifacts.local" / "datasets" / "quality-gated-clearance-fusion-r0-1-arkit-381644-20260806" / "raw" / "Validation"
    for video_id, assets in sorted(by_video.items()):
        paths = {asset: media_root / video_id / asset for asset in ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip")}
        for asset, path in paths.items():
            require(path.is_file() and sha256_file(path) == assets[asset]["sha256"], f"ARKit asset SHA mismatch: {video_id}/{asset}")
        maps: dict[str, dict[float, str]] = {}
        for asset, path in paths.items():
            with zipfile.ZipFile(path) as archive:
                values: dict[float, str] = {}
                for name in archive.namelist():
                    match = ZIP_PNG_RE.search(Path(name).name)
                    if not match:
                        continue
                    timestamp = float(match.group(1))
                    require(timestamp not in values, f"duplicate ARKit timestamp: {video_id}/{asset}")
                    values[timestamp] = name
                maps[asset] = values
        common = sorted(set(maps["lowres_wide"]) & set(maps["lowres_depth"]) & set(maps["confidence"]))
        require(common, f"no ARKit RGB-depth-confidence intersection: {video_id}")
        result.extend({
            "frame_id": f"ARKitScenes:381644:{video_id}:{timestamp:.3f}",
            "dataset": "ARKitScenes",
            "parent_id": "381644",
            "video_id": video_id,
            "timestamp_ns": int(round(timestamp * 1_000_000_000)),
            "source_timestamp_s": timestamp,
            "rgb_member": maps["lowres_wide"][timestamp],
            "depth_member": maps["lowres_depth"][timestamp],
            "confidence_member": maps["confidence"][timestamp],
        } for timestamp in common)
    return result


def nonoverlap_clips(frames: list[dict[str, Any]], length: int = 4, max_gap_ns: int = 500_000_000) -> list[dict[str, Any]]:
    ordered = sorted(frames, key=lambda row: int(row["timestamp_ns"]))
    clips = []
    index = 0
    while index + length <= len(ordered):
        window = ordered[index:index + length]
        if all(0 < int(right["timestamp_ns"]) - int(left["timestamp_ns"]) <= max_gap_ns for left, right in zip(window, window[1:])):
            clips.append({"clip_id": f"{window[0]['parent_id']}:{window[0]['frame_id']}", "parent_id": window[0]["parent_id"], "video_id": window[0]["video_id"], "frame_ids": [row["frame_id"] for row in window]})
            index += length
        else:
            index += 1
    return clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--arkit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "overwrite forbidden")
    root = args.repo_root.resolve()
    tum_root = args.tum_root.resolve()
    arkit_manifest = args.arkit_manifest.resolve()
    tum_specs = [
        ("rgbd_dataset_freiburg3_sitting_halfsphere", tum_root / "rgbd_dataset_freiburg3_sitting_halfsphere.tgz"),
        ("rgbd_dataset_freiburg3_sitting_rpy", tum_root / "rgbd_dataset_freiburg3_sitting_rpy.tgz"),
    ]
    frames = []
    archive_bindings = []
    for parent_id, archive in tum_specs:
        require(archive.is_file(), f"missing TUM archive: {archive}")
        archive_bindings.append({"parent_id": parent_id, "path": str(archive), "sha256": sha256_file(archive), "bytes": archive.stat().st_size})
        frames.extend(tum_frames(archive, parent_id))
    frames.extend(arkit_frames(root, arkit_manifest))
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_parent.setdefault(str(frame["parent_id"]), []).append(frame)
    clips = []
    for parent_id, rows in sorted(by_parent.items()):
        parent_clips = nonoverlap_clips(rows)
        require(len(parent_clips) >= 8, f"clip capacity below minimum: {parent_id}")
        clips.extend(parent_clips)
    result = {
        "schema": RESULT_SCHEMA,
        "source_contract": "label-blind identities and source-native timestamps only",
        "archive_bindings": archive_bindings,
        "arkit_manifest_path": str(arkit_manifest),
        "arkit_manifest_sha256": sha256_file(arkit_manifest),
        "parents": sorted({row["parent_id"] for row in frames}),
        "frame_count": len(frames),
        "clip_count": len(clips),
        "frames": frames,
        "clips": clips,
        "labels_opened": False,
        "image_or_depth_bytes_decoded": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "terminal": "QUALITY_GATED_CLEARANCE_FUSION_R0_SOURCE_CATALOG_MATERIALIZED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: value for key, value in result.items() if key not in {"frames", "clips"}}, indent=2))


if __name__ == "__main__":
    main()
