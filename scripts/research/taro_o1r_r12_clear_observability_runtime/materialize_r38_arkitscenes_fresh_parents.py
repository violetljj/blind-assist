#!/usr/bin/env python3
"""Materialize the source-locked R38 ARKitScenes parents without decoding pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_ROOT = REPO_ROOT / "scripts/research/spatial_calibration_head_r1"
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

from scripts.research.assistive_geometry import arkitscenes_truth_reader as arkit
from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    arkitscenes_balanced_pose_source_frontdoor as arkit_frontdoor,
)


LOCK_SCHEMA = "blindassist.taro.r38_arkitscenes_fresh_parent_source_lock.v1"
MANIFEST_SCHEMA = "blindassist.taro.r38_arkitscenes_fresh_parent_materialization.v1"
ASSETS = (
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
)
FRAME_COUNT = 300
MAXIMUM_POSE_BRACKET_S = 0.25


class R38MaterializationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R38MaterializationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def head(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "BlindAssist-TARO-R38-source-materializer"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        require(int(response.status) == 200 and length is not None, f"asset HEAD unavailable: {url}")
        return {
            "url": url,
            "http_status": int(response.status),
            "content_length_bytes": int(length),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def trajectory_rows(path: Path) -> np.ndarray:
    value = arkit.parse_trajectory(path)
    require(value.shape[0] >= 2, f"trajectory too short: {path}")
    return value


def select_landscape_pose_covered_stems(
    member_maps: Mapping[str, Mapping[str, str]],
    trajectory: np.ndarray,
    count: int = FRAME_COUNT,
) -> tuple[list[str], dict[str, Any]]:
    common = set.intersection(*(set(value) for value in member_maps.values()))
    ordered = sorted(common, key=lambda stem: (timestamp_from_stem(stem), stem))
    selected: list[str] = []
    orientation_counts = {str(index): 0 for index in range(4)}
    outside_trajectory = 0
    excessive_bracket = 0
    previous_timestamp: float | None = None
    for stem in ordered:
        timestamp = timestamp_from_stem(stem)
        if timestamp < float(trajectory[0, 0]) or timestamp > float(trajectory[-1, 0]):
            outside_trajectory += 1
            continue
        try:
            pose, _receipt = arkit.interpolate_camera_to_world(
                trajectory, timestamp, MAXIMUM_POSE_BRACKET_S
            )
        except Exception:
            excessive_bracket += 1
            continue
        orientation = arkit.orientation_index(pose)
        orientation_counts[str(orientation)] += 1
        if orientation not in arkit_frontdoor.ALLOWED_ORIENTATION_INDICES:
            continue
        if previous_timestamp is not None and timestamp - previous_timestamp > 0.5:
            selected = []
        selected.append(stem)
        previous_timestamp = timestamp
        if len(selected) == count:
            break
    require(len(selected) == count, "fewer than 300 continuous landscape common frames")
    timestamps = [timestamp_from_stem(stem) for stem in selected]
    require(
        all(0.0 < right - left <= 0.5 for left, right in zip(timestamps, timestamps[1:])),
        "selected landscape window is not continuous",
    )
    return selected, {
        "common_stem_count": len(common),
        "orientation_counts_before_stop": orientation_counts,
        "outside_trajectory_stem_count_before_stop": outside_trajectory,
        "excessive_pose_bracket_stem_count_before_stop": excessive_bracket,
        "selected_start_timestamp": timestamps[0],
        "selected_end_timestamp": timestamps[-1],
        "selected_frame_count": len(selected),
    }


def validate_receipt(path: Path, parent: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value["video_id"] == parent["video_id"], f"receipt video drift: {path}")
    require(value["visit_id"] == parent["visit_id"], f"receipt visit drift: {path}")
    require(value["selected_frame_count"] == FRAME_COUNT, f"receipt frame count drift: {path}")
    trajectory = Path(value["trajectory"]["path"])
    require(trajectory.is_file(), f"receipt trajectory absent: {trajectory}")
    require(sha256_file(trajectory) == parent["trajectory_sha256"], f"trajectory hash drift: {trajectory}")
    for entries in value["extracted"].values():
        for entry in entries:
            output = Path(entry["path"])
            require(output.is_file() and output.stat().st_size == entry["bytes"], f"output drift: {output}")
            require(sha256_file(output) == entry["sha256"], f"output hash drift: {output}")
    return value


def materialize(
    source_lock_path: Path,
    trajectory_root: Path,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA, "R38 source lock schema drift")
    require(
        lock.get("status") == "FROZEN_AFTER_METADATA_TRAJECTORY_AND_HEAD_ONLY_BEFORE_MEDIA_BODY_OR_TASK_OUTCOME",
        "R38 source lock status drift",
    )
    parents = list(lock["parents"])
    require(len(parents) == 12, "R38 parent count drift")
    require(len({row["video_id"] for row in parents}) == len(parents), "duplicate R38 video")
    require(len({row["visit_id"] for row in parents}) == len(parents), "duplicate R38 visit")
    base_url = lock["official_source"]["base_url"]
    head_rows: list[dict[str, Any]] = []
    for parent in parents:
        for asset in ASSETS:
            row = head(f"{base_url}/Validation/{parent['video_id']}/{asset}")
            head_rows.append(
                row
                | {
                    "video_id": parent["video_id"],
                    "visit_id": parent["visit_id"],
                    "official_fold": "Validation",
                    "asset": asset,
                }
            )
    require(len(head_rows) == 60, "R38 HEAD request count drift")
    by_key = {(row["video_id"], row["asset"]): row for row in head_rows}
    attempt = {
        "schema": "blindassist.taro.r38_arkitscenes_fresh_parent_materialization_attempt.v1",
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": sha256_file(source_lock_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "asset_head_rows": head_rows,
        "asset_head_identity_sha256": hashlib.sha256(canonical_json_bytes(head_rows)).hexdigest().upper(),
        "asset_head_total_content_length_bytes": sum(row["content_length_bytes"] for row in head_rows),
        "output_root": str(output_root),
        "candidate_sensor_depth_pixel_decodes": 0,
        "task_outcome_reads": 0,
    }
    attempt_path = output_root / "attempt.json"
    if output_root.exists():
        require(resume, f"output exists; --resume required: {output_root}")
        require(attempt_path.is_file(), "R38 resume attempt absent")
        previous = json.loads(attempt_path.read_text(encoding="utf-8"))
        require(previous == attempt, "R38 resume attempt drift")
        require(not (output_root / "manifest.json").exists(), "R38 materialization already complete")
    else:
        output_root.mkdir(parents=True)
        write_json_exclusive(attempt_path, attempt)
    archive_root = output_root / "_temporary_archives"
    archive_root.mkdir(exist_ok=True)
    videos: list[dict[str, Any]] = []
    for index, parent in enumerate(parents, start=1):
        video_id = parent["video_id"]
        receipt_path = output_root / "receipts" / f"{index:02d}-{video_id}.json"
        if receipt_path.exists():
            videos.append(validate_receipt(receipt_path, parent))
            print(json.dumps({"completed": index, "total": len(parents), "video_id": video_id, "resumed": True}), flush=True)
            continue
        source_trajectory = trajectory_root / f"{video_id}.traj"
        require(source_trajectory.is_file(), f"source trajectory absent: {source_trajectory}")
        require(source_trajectory.stat().st_size == parent["trajectory_bytes"], f"trajectory size drift: {video_id}")
        require(sha256_file(source_trajectory) == parent["trajectory_sha256"], f"trajectory hash drift: {video_id}")
        video_root = output_root / "raw" / "Validation" / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        trajectory_path = video_root / "lowres_wide.traj"
        shutil.copyfile(source_trajectory, trajectory_path)
        trajectory = trajectory_rows(trajectory_path)
        archives: dict[str, Path] = {}
        source_assets: list[dict[str, Any]] = []
        for asset in ASSETS[:3]:
            row = by_key[(video_id, asset)]
            archive = archive_root / video_id / asset
            digest, attempts = download_file(row["url"], archive, row["content_length_bytes"])
            archives[asset] = archive
            source_assets.append(
                {
                    "asset": asset,
                    "url": row["url"],
                    "bytes": row["content_length_bytes"],
                    "sha256": digest,
                    "attempts": attempts,
                }
            )
        member_maps = {asset: png_members_by_stem(path) for asset, path in archives.items()}
        selected, selection_receipt = select_landscape_pose_covered_stems(member_maps, trajectory)
        extracted: dict[str, list[dict[str, Any]]] = {}
        for asset, folder in (
            ("lowres_wide.zip", "lowres_wide"),
            ("lowres_depth.zip", "lowres_depth"),
            ("confidence.zip", "confidence"),
        ):
            extracted[folder] = extract_named_members(
                archives[asset],
                [member_maps[asset][stem] for stem in selected],
                video_root / folder,
            )
            safe_delete_archive(archives[asset], archive_root)
        intrinsics_row = by_key[(video_id, "lowres_wide_intrinsics.zip")]
        intrinsics_archive = archive_root / video_id / "lowres_wide_intrinsics.zip"
        digest, attempts = download_file(
            intrinsics_row["url"], intrinsics_archive, intrinsics_row["content_length_bytes"]
        )
        source_assets.append(
            {
                "asset": "lowres_wide_intrinsics.zip",
                "url": intrinsics_row["url"],
                "bytes": intrinsics_row["content_length_bytes"],
                "sha256": digest,
                "attempts": attempts,
            }
        )
        names = nearest_pincam_member_names(pincam_members(intrinsics_archive), selected)
        extracted["lowres_wide_intrinsics"] = extract_named_members(
            intrinsics_archive, names, video_root / "lowres_wide_intrinsics"
        )
        safe_delete_archive(intrinsics_archive, archive_root)
        receipt = {
            "video_id": video_id,
            "visit_id": parent["visit_id"],
            "official_fold": "Validation",
            "rank_sha256": parent["rank_sha256"],
            "selected_frame_stems": selected,
            "selected_frame_count": len(selected),
            "selection": selection_receipt,
            "trajectory": {
                "path": str(trajectory_path.resolve()),
                "bytes": trajectory_path.stat().st_size,
                "sha256": sha256_file(trajectory_path),
                "row_count": len(trajectory),
            },
            "source_assets": source_assets,
            "extracted": extracted,
            "candidate_sensor_depth_pixel_decodes": 0,
            "task_outcome_reads": 0,
        }
        write_json_exclusive(receipt_path, receipt)
        videos.append(receipt)
        print(json.dumps({"completed": index, "total": len(parents), "video_id": video_id}), flush=True)
    remove_empty_archive_tree(archive_root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": sha256_file(source_lock_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "parent_count": len(videos),
        "selected_frame_count": sum(row["selected_frame_count"] for row in videos),
        "selected_frame_count_per_parent": FRAME_COUNT,
        "parents": videos,
        "reference_input_preflight_opened": False,
        "candidate_sensor_depth_pixel_decodes": 0,
        "task_outcome_opened": False,
        "model_outputs_read": False,
        "temporary_archives_retained": False,
        "terminal": "TARO_R38_ARKITSCENES_FRESH_PARENT_MATERIALIZED_SOURCE_ONLY",
    }
    manifest["content_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest().upper()
    write_json_exclusive(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = materialize(
        args.source_lock.resolve(),
        args.trajectory_root.resolve(),
        args.output_root.resolve(),
        args.resume,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "parents"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
