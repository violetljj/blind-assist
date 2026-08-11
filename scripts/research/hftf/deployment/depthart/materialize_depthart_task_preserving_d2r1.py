#!/usr/bin/env python3
"""Scan D2R1 full portrait runs for source-truth support without model output."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    orientation_index,
    parse_trajectory,
)
from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2_phase_b import (  # noqa: E402
    qualifies,
    summarize_truth,
)
from scripts.research.spatial_calibration_head_r1.download_locked_assets import download_file  # noqa: E402
from scripts.research.spatial_calibration_head_r1.materialize_cache import timestamp_from_stem  # noqa: E402


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d2r1_body_protocol_v1"
RECOVERY_SCHEMA = "blindassist_depthart_task_preserving_d2r1_target_support_window_recovery_protocol_v1"
PHASE_A_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_manifest_v1"
HEAD_SCHEMA = "blindassist_depthart_task_preserving_d2r1_asset_header_preflight_v1"
LICENSE_SCHEMA = "blindassist_depthart_task_preserving_d2r1_source_scope_receipt_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d2r1_manifest_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d2r1_identity_checkpoint_v1"
ASSETS = ("lowres_wide_intrinsics.zip", "lowres_wide.traj", "lowres_depth.zip", "confidence.zip")
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
COUNT_KEYS = (
    "known_cells",
    "clear_cells",
    "occupied_cells",
    "valid_band_clearances",
    *(f"{band}@{horizon:.1f}m" for band in BANDS for horizon in HORIZONS),
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


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_bytes_exclusive(path, json_bytes(value))


def safe_member_map(archive: Path, suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename
            pure = Path(name)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {name}")
            if pure.suffix.lower() != suffix:
                continue
            require(pure.stem not in result, f"duplicate {suffix} stem: {pure.stem}")
            result[pure.stem] = name
    require(result, f"ZIP contains no {suffix} members: {archive}")
    return result


def parse_pincam_payload(payload: bytes, label: str) -> tuple[np.ndarray, tuple[int, int]]:
    values = [float(value) for value in payload.decode("utf-8").split()]
    require(len(values) == 6 and all(math.isfinite(value) for value in values), f"invalid pincam: {label}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height and width > 0 and height > 0, f"invalid pincam dimensions: {label}")
    fx, fy, cx, cy = values[2:]
    require(fx > 0 and fy > 0 and 0 <= cx < width and 0 <= cy < height, f"invalid pincam values: {label}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (width, height)


def split_continuous_portrait_runs(
    classified: list[dict[str, Any]], maximum_adjacent_gap_seconds: float
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in classified:
        if not row["portrait"]:
            if current:
                runs.append(current)
                current = []
            continue
        timestamp = float(row["timestamp"])
        if current:
            gap = timestamp - float(current[-1]["timestamp"])
            if not 0 < gap <= maximum_adjacent_gap_seconds:
                runs.append(current)
                current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def portrait_runs(
    common_stems: list[str],
    trajectory: Any,
    maximum_pose_gap_seconds: float,
    maximum_adjacent_gap_seconds: float,
    portrait_indices: set[int],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    orientation_counts = {str(index): 0 for index in range(4)}
    pose_rejected = 0
    maximum_observed_pose_gap = 0.0
    for stem in sorted(common_stems, key=lambda value: (timestamp_from_stem(value), value)):
        timestamp = timestamp_from_stem(stem)
        try:
            pose, metadata = interpolate_camera_to_world(trajectory, timestamp, maximum_pose_gap_seconds)
        except ValueError:
            pose_rejected += 1
            classified.append({"stem": stem, "timestamp": timestamp, "portrait": False, "pose": None})
            continue
        index = orientation_index(pose)
        orientation_counts[str(index)] += 1
        maximum_observed_pose_gap = max(maximum_observed_pose_gap, float(metadata["bracketing_gap_seconds"]))
        classified.append(
            {"stem": stem, "timestamp": timestamp, "portrait": index in portrait_indices, "pose": pose}
        )
    runs = split_continuous_portrait_runs(classified, maximum_adjacent_gap_seconds)
    return runs, {
        "common_stem_count": len(common_stems),
        "pose_rejected_frame_count": pose_rejected,
        "pose_covered_orientation_counts": orientation_counts,
        "portrait_frame_count": sum(len(run) for run in runs),
        "portrait_run_lengths": [len(run) for run in runs],
        "candidate_window_count": sum(max(0, len(run) - 299) for run in runs),
        "maximum_observed_pose_bracketing_gap_seconds": maximum_observed_pose_gap,
    }


def counts_to_vector(counts: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            int(counts["known_cells"]),
            int(counts["clear_cells"]),
            int(counts["occupied_cells"]),
            int(counts["valid_band_clearances"]),
            *(int(counts["known_by_grid"][key]) for key in COUNT_KEYS[4:]),
        ],
        dtype=np.int64,
    )


def vector_to_counts(vector: np.ndarray) -> dict[str, Any]:
    values = [int(value) for value in vector]
    return {
        "known_cells": values[0],
        "clear_cells": values[1],
        "occupied_cells": values[2],
        "valid_band_clearances": values[3],
        "known_by_grid": {key: values[index] for index, key in enumerate(COUNT_KEYS[4:], start=4)},
    }


def earliest_qualified_window(
    runs: list[list[dict[str, Any]]],
    frame_counter: Callable[[dict[str, Any]], dict[str, Any]],
    window_frames: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    decoded = 0
    windows_tested = 0
    for run_index, run in enumerate(runs):
        if len(run) < window_frames:
            continue
        prefix = [np.zeros(len(COUNT_KEYS), dtype=np.int64)]
        for end_index, row in enumerate(run):
            prefix.append(prefix[-1] + counts_to_vector(frame_counter(row)))
            decoded += 1
            if end_index + 1 < window_frames:
                continue
            start_index = end_index + 1 - window_frames
            counts = vector_to_counts(prefix[end_index + 1] - prefix[start_index])
            windows_tested += 1
            passed, failures = qualifies(counts, thresholds)
            if passed:
                selected = run[start_index : end_index + 1]
                return {
                    "qualified": True,
                    "selected_frame_stems": [str(value["stem"]) for value in selected],
                    "selected_start_timestamp": float(selected[0]["timestamp"]),
                    "selected_end_timestamp": float(selected[-1]["timestamp"]),
                    "selected_run_index": run_index,
                    "truth_support": counts,
                    "qualification_failures": [],
                    "decoded_frame_count": decoded,
                    "windows_tested": windows_tested,
                }
        # Prefix storage is per-run and never serialized.
    return {
        "qualified": False,
        "selected_frame_stems": [],
        "selected_start_timestamp": None,
        "selected_end_timestamp": None,
        "selected_run_index": None,
        "truth_support": None,
        "qualification_failures": ["no_300_frame_window_passed_all_frozen_support_thresholds"],
        "decoded_frame_count": decoded,
        "windows_tested": windows_tested,
    }


def head_lookup(head_result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in head_result["assets"]:
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate HEAD row: {key}")
        require(row["http_status"] == 200 and int(row["content_length_bytes"]) > 0, f"unavailable: {key}")
        lookup[key] = row
    require(len(lookup) == 64, "HEAD asset count drift")
    return lookup


def safe_remove_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    require(resolved_root in resolved.parents and resolved != resolved_root, f"unsafe temporary cleanup: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def role_assignments(qualified_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(qualified_rows) < 8:
        return []
    return [
        {
            "role_order": index + 1 if index < 4 else index - 3,
            "role": "D2_TRAIN" if index < 4 else "D2_DEVELOPMENT_SEALED",
            "phase_a_order": int(row["phase_a_order"]),
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "selected_frame_stems": list(row["selected_frame_stems"]),
        }
        for index, row in enumerate(qualified_rows[:8])
    ]


def read_checkpoint(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256.json")
    require(sidecar.is_file(), f"checkpoint sidecar missing: {sidecar}")
    seal = load_json(sidecar)
    require(path.stat().st_size == int(seal["bytes"]), f"checkpoint bytes drift: {path}")
    require(sha256_file(path) == seal["sha256"], f"checkpoint SHA drift: {path}")
    value = load_json(path)
    require(value.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema drift")
    for key in ("phase_a_order", "pool_order", "visit_id", "video_id", "fold"):
        require(str(value[key]) == str(expected_identity[key]), f"checkpoint identity drift: {key}")
    require(value["model_output_read"] is False and value["r2_cohort_access"] == "NONE", "checkpoint authority drift")
    return value


def write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    payload = json_bytes(value)
    write_bytes_exclusive(path, payload)
    write_json_exclusive(
        path.with_suffix(".sha256.json"),
        {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
    )


def make_frame_counter(
    intrinsics_bundle: zipfile.ZipFile,
    depth_bundle: zipfile.ZipFile,
    confidence_bundle: zipfile.ZipFile,
    intrinsics_map: dict[str, str],
    depth_map: dict[str, str],
    confidence_map: dict[str, str],
    policy: TruthReaderPolicy,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def frame_counter(row: dict[str, Any]) -> dict[str, Any]:
        stem = str(row["stem"])
        intrinsics, source_size = parse_pincam_payload(
            intrinsics_bundle.read(intrinsics_map[stem]), intrinsics_map[stem]
        )
        with depth_bundle.open(depth_map[stem]) as stream, Image.open(stream) as image:
            depth_raw = np.asarray(image).copy()
        with confidence_bundle.open(confidence_map[stem]) as stream, Image.open(stream) as image:
            confidence = np.asarray(image).copy()
        require(depth_raw.ndim == 2 and np.issubdtype(depth_raw.dtype, np.integer), "invalid depth raster")
        require(confidence.shape == depth_raw.shape and np.issubdtype(confidence.dtype, np.integer), "invalid confidence raster")
        require(source_size == (depth_raw.shape[1], depth_raw.shape[0]), "intrinsics/depth size mismatch")
        confidence_values = {int(value) for value in np.unique(confidence)}
        require(confidence_values.issubset({0, 1, 2}), f"confidence values drift: {sorted(confidence_values)}")
        dummy_rgb = np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
        canonical = canonicalize_frame(dummy_rgb, depth_raw, confidence, intrinsics, row["pose"])
        require(canonical["rotation_index"] in (1, 3), "portrait orientation drift")
        up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
        truth = derive_assistive_truth(
            depth_mm_to_metres(canonical["depth_raw_mm"]),
            canonical["confidence"],
            canonical["intrinsics"],
            up_camera,
            policy,
        )
        return summarize_truth(truth)

    return frame_counter


def expected_attempt(
    protocol_path: Path,
    recovery_path: Path,
    phase_a_path: Path,
    head_path: Path,
    receipt_path: Path,
    identities: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "blindassist_depthart_task_preserving_d2r1_attempt_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "recovery_protocol_sha256": sha256_file(recovery_path),
        "phase_a_manifest_sha256": sha256_file(phase_a_path),
        "head_result_sha256": sha256_file(head_path),
        "license_receipt_sha256": sha256_file(receipt_path),
        "identities": identities,
        "truth_support_thresholds": thresholds,
        "window_frames": 300,
        "rgb_read": False,
        "model_output_read": False,
        "r2_cohort_access": "NONE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    recovery = load_json(args.recovery_protocol)
    phase_a = load_json(args.phase_a_manifest)
    head_result = load_json(args.head_result)
    receipt = load_json(args.license_receipt)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(recovery.get("schema") == RECOVERY_SCHEMA, "recovery protocol schema drift")
    require(phase_a.get("schema") == PHASE_A_SCHEMA, "Phase-A schema drift")
    require(head_result.get("schema") == HEAD_SCHEMA, "HEAD schema drift")
    require(receipt.get("schema") == LICENSE_SCHEMA, "license schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        path = Path(dependency["path"])
        require(path.stat().st_size == int(dependency["bytes"]), f"dependency size drift: {path}")
        require(sha256_file(path) == dependency["sha256"], f"dependency SHA drift: {path}")
    for name, path in (
        ("recovery_protocol", args.recovery_protocol),
        ("phase_a_manifest", args.phase_a_manifest),
        ("head_result", args.head_result),
        ("license_receipt", args.license_receipt),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(head_result["terminal"] == "D2R1_ASSET_HEADERS_AVAILABLE_BODY_UNOPENED", "HEAD terminal drift")
    require(receipt["authority"]["d2r1_body_download"] is True, "D2R1 body not authorized")
    require(receipt["authority"]["d2r1_aggregate_window_scan"] is True, "D2R1 scan not authorized")
    require(tuple(protocol["assets"]) == ASSETS, "asset drift")
    require(protocol["truth_support_thresholds"] == recovery["unchanged_truth_support_thresholds"], "threshold drift")
    selected = [
        {
            "phase_a_order": index + 1,
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
        }
        for index, row in enumerate(phase_a["selected_phase_b"])
    ]
    require(len(selected) == 16, "identity count drift")
    lookup = head_lookup(head_result)
    total_body_bytes = sum(int(row["content_length_bytes"]) for row in lookup.values())
    require(total_body_bytes <= int(receipt["source_scope"]["maximum_total_body_bytes"]), "body bound drift")
    require(total_body_bytes == int(protocol["authorized_total_body_bytes"]), "authorized body total drift")
    maximum_video_bytes = max(
        sum(int(lookup[(row["video_id"], asset)]["content_length_bytes"]) for asset in ASSETS)
        for row in selected
    )
    require(
        shutil.disk_usage(args.output_root.parent).free >= maximum_video_bytes * 3 + 2_000_000_000,
        "insufficient bounded working space",
    )
    attempt = expected_attempt(
        args.protocol,
        args.recovery_protocol,
        args.phase_a_manifest,
        args.head_result,
        args.license_receipt,
        selected,
        protocol["truth_support_thresholds"],
    )
    if args.output_root.exists():
        require(args.resume, f"output exists; --resume required: {args.output_root}")
        require(load_json(args.output_root / "attempt.json") == attempt, "resume attempt binding drift")
        require(not (args.output_root / "manifest.json").exists(), "completed manifest already exists")
    else:
        require(not args.resume, "--resume requires existing output root")
        args.output_root.mkdir(parents=True)
        write_json_exclusive(args.output_root / "attempt.json", attempt)
    temporary_root = args.output_root / "_temporary_downloads"
    temporary_root.mkdir(exist_ok=True)
    run_temporary_root = temporary_root / uuid.uuid4().hex
    run_temporary_root.mkdir()
    policy = TruthReaderPolicy()
    serialized_policy = json.loads(json.dumps(asdict(policy)))
    require(serialized_policy == protocol["truth_reader_policy"], "truth reader policy drift")
    thresholds = protocol["truth_support_thresholds"]
    videos: list[dict[str, Any]] = []
    try:
        for identity in selected:
            checkpoint_path = args.output_root / "receipts" / f"{identity['phase_a_order']:02d}-{identity['video_id']}.json"
            if checkpoint_path.exists():
                value = read_checkpoint(checkpoint_path, identity)
                videos.append(value)
                print(json.dumps({"resumed": identity["phase_a_order"], "video_id": identity["video_id"], "qualified": value["qualified"]}), flush=True)
                continue
            video_id = identity["video_id"]
            video_temp = run_temporary_root / f"{identity['phase_a_order']:02d}-{video_id}"
            video_temp.mkdir()
            paths: dict[str, Path] = {}
            source_assets: list[dict[str, Any]] = []
            for asset in ASSETS:
                row = lookup[(video_id, asset)]
                target = video_temp / asset
                digest, attempts = download_file(row["url"], target, int(row["content_length_bytes"]))
                paths[asset] = target
                source_assets.append(
                    {
                        "asset": asset,
                        "url": row["url"],
                        "bytes": int(row["content_length_bytes"]),
                        "sha256": digest,
                        "attempts": attempts,
                    }
                )
            intrinsics_map = safe_member_map(paths["lowres_wide_intrinsics.zip"], ".pincam")
            depth_map = safe_member_map(paths["lowres_depth.zip"], ".png")
            confidence_map = safe_member_map(paths["confidence.zip"], ".png")
            common_stems = sorted(
                set(intrinsics_map) & set(depth_map) & set(confidence_map),
                key=lambda value: (timestamp_from_stem(value), value),
            )
            require(common_stems, f"no exact common stems: {video_id}")
            trajectory = parse_trajectory(paths["lowres_wide.traj"])
            runs, coverage = portrait_runs(
                common_stems,
                trajectory,
                float(protocol["maximum_pose_bracketing_gap_seconds"]),
                float(protocol["maximum_adjacent_frame_gap_seconds"]),
                {int(value) for value in protocol["portrait_orientation_indices"]},
            )
            with (
                zipfile.ZipFile(paths["lowres_wide_intrinsics.zip"]) as intrinsics_bundle,
                zipfile.ZipFile(paths["lowres_depth.zip"]) as depth_bundle,
                zipfile.ZipFile(paths["confidence.zip"]) as confidence_bundle,
            ):
                scan = earliest_qualified_window(
                    runs,
                    make_frame_counter(
                        intrinsics_bundle,
                        depth_bundle,
                        confidence_bundle,
                        intrinsics_map,
                        depth_map,
                        confidence_map,
                        policy,
                    ),
                    int(protocol["window_frames"]),
                    thresholds,
                )
            value = {
                "schema": CHECKPOINT_SCHEMA,
                **identity,
                "source_assets": source_assets,
                "intrinsics_stem_count": len(intrinsics_map),
                "depth_stem_count": len(depth_map),
                "confidence_stem_count": len(confidence_map),
                "coverage": coverage,
                **scan,
                "per_frame_truth_rows_saved": False,
                "rgb_read": False,
                "model_output_read": False,
                "r2_cohort_access": "NONE",
            }
            write_checkpoint(checkpoint_path, value)
            videos.append(value)
            print(
                json.dumps(
                    {
                        "completed": identity["phase_a_order"],
                        "total": len(selected),
                        "video_id": video_id,
                        "qualified": value["qualified"],
                        "decoded_frames": value["decoded_frame_count"],
                        "windows_tested": value["windows_tested"],
                    }
                ),
                flush=True,
            )
            safe_remove_tree(video_temp, run_temporary_root)
    finally:
        safe_remove_tree(run_temporary_root, temporary_root)
    qualified_rows = [row for row in videos if row["qualified"]]
    roles = role_assignments(qualified_rows)
    passed = len(roles) == 8
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "recovery_protocol_sha256": sha256_file(args.recovery_protocol),
        "phase_a_manifest_sha256": sha256_file(args.phase_a_manifest),
        "head_result_sha256": sha256_file(args.head_result),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "identity_count": len(videos),
        "qualified_identity_count": len(qualified_rows),
        "truth_support_thresholds": thresholds,
        "role_assignments": roles,
        "videos": videos,
        "per_frame_truth_rows_saved": False,
        "rgb_read": False,
        "model_output_read": False,
        "d2_development_model_outcome_opened": False,
        "r2_cohort_access": "NONE",
        "terminal": (
            "D2R1_SOURCE_SUPPORT_PASS_4_TRAIN_4_DEVELOPMENT_ROLES_LOCKED"
            if passed
            else "D2R1_SOURCE_SUPPORT_FAIL_FEWER_THAN_8_QUALIFIED_IDENTITIES"
        ),
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
