#!/usr/bin/env python3
"""Evaluate D3R3 source-truth support with fixed-roster UNKNOWN handling.

The exact 300-frame plan remains the estimand.  If either depth or confidence
is absent for an exact stem, that frame contributes SOURCE_UNAVAILABLE_UNKNOWN:
zero clear/occupied cells and zero valid clearance observations.  No neighbor,
replacement, interpolation, or denominator rewrite is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    WORLD_UP,
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    parse_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
MINIMUM_SOURCE_AVAILABLE_FRAMES = 297
THRESHOLDS = {
    "minimum_truth_known_cells": 1800,
    "minimum_truth_clear_cells": 270,
    "minimum_truth_occupied_cells": 900,
    "minimum_truth_clear_cells_per_band_horizon": 30,
    "minimum_truth_occupied_cells_per_band_horizon": 30,
    "minimum_valid_band_clearances": 450,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "exclusive write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def selected_stem_sha256(stems: list[str]) -> str:
    return hashlib.sha256(("\n".join(stems) + "\n").encode("ascii")).hexdigest().upper()


def timestamp_from_stem(stem: str) -> float:
    try:
        value = Decimal(stem.rsplit("_", 1)[-1])
    except (InvalidOperation, IndexError) as error:
        raise ValueError(f"invalid frame timestamp: {stem}") from error
    require(value.is_finite(), f"non-finite frame timestamp: {stem}")
    result = float(value)
    require(math.isfinite(result), f"non-finite frame timestamp: {stem}")
    return result


def parse_pincam_payload(payload: bytes, label: str) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        values = [float(value) for value in payload.decode("utf-8").split()]
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid pincam payload: {label}") from error
    require(len(values) == 6 and all(math.isfinite(value) for value in values), f"invalid pincam: {label}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height and width > 0 and height > 0, f"invalid pincam dimensions: {label}")
    fx, fy, cx, cy = values[2:]
    require(fx > 0 and fy > 0 and 0 <= cx < width and 0 <= cy < height, f"invalid pincam values: {label}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (width, height)


def member_map(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    seen_names: set[str] = set()
    for info in archive.infolist():
        pure = PurePosixPath(info.filename.replace("\\", "/"))
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {info.filename}")
        require(info.filename not in seen_names, f"duplicate ZIP member name: {info.filename}")
        seen_names.add(info.filename)
        if info.is_dir() or pure.suffix.lower() != suffix:
            continue
        require(pure.stem not in result, f"duplicate ZIP frame stem: {pure.stem}")
        result[pure.stem] = info.filename
    require(result, f"no {suffix} members: {archive.filename}")
    return result


def resolve_frame_availability(
    selected_stems: list[str], depth_stems: set[str], confidence_stems: set[str]
) -> dict[str, list[str]]:
    depth_missing = [stem for stem in selected_stems if stem not in depth_stems]
    confidence_missing = [stem for stem in selected_stems if stem not in confidence_stems]
    unavailable_set = set(depth_missing) | set(confidence_missing)
    unavailable = [stem for stem in selected_stems if stem in unavailable_set]
    available = [stem for stem in selected_stems if stem not in unavailable_set]
    return {
        "available": available,
        "source_unavailable": unavailable,
        "depth_missing": depth_missing,
        "confidence_missing": confidence_missing,
    }


def empty_counts() -> dict[str, Any]:
    grid = {f"{band}@{horizon:.1f}m": 0 for band in BANDS for horizon in HORIZONS}
    return {
        "known_cells": 0,
        "clear_cells": 0,
        "occupied_cells": 0,
        "valid_band_clearances": 0,
        "known_by_grid": dict(grid),
        "clear_by_grid": dict(grid),
        "occupied_by_grid": dict(grid),
    }


def summarize_truth(truth: dict[str, Any]) -> dict[str, Any]:
    result = empty_counts()
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
            key = f"{band}@{horizon:.1f}m"
            result["known_cells"] += 1
            result["known_by_grid"][key] += 1
            if bool(value):
                result["occupied_cells"] += 1
                result["occupied_by_grid"][key] += 1
            else:
                result["clear_cells"] += 1
                result["clear_by_grid"][key] += 1
    return result


def add_counts(total: dict[str, Any], frame: dict[str, Any]) -> None:
    for key in ("known_cells", "clear_cells", "occupied_cells", "valid_band_clearances"):
        total[key] += int(frame[key])
    for group in ("known_by_grid", "clear_by_grid", "occupied_by_grid"):
        for key, value in frame[group].items():
            total[group][key] += int(value)


def qualification_failures(counts: dict[str, Any]) -> list[str]:
    require(counts["known_cells"] == counts["clear_cells"] + counts["occupied_cells"], "known identity drift")
    failures: list[str] = []
    for count_key, threshold_key in (
        ("known_cells", "minimum_truth_known_cells"),
        ("clear_cells", "minimum_truth_clear_cells"),
        ("occupied_cells", "minimum_truth_occupied_cells"),
        ("valid_band_clearances", "minimum_valid_band_clearances"),
    ):
        if int(counts[count_key]) < THRESHOLDS[threshold_key]:
            failures.append(f"{count_key}={counts[count_key]}<{THRESHOLDS[threshold_key]}")
    for key in sorted(counts["known_by_grid"]):
        require(
            counts["known_by_grid"][key]
            == counts["clear_by_grid"][key] + counts["occupied_by_grid"][key],
            f"grid identity drift: {key}",
        )
        if counts["clear_by_grid"][key] < THRESHOLDS["minimum_truth_clear_cells_per_band_horizon"]:
            failures.append(f"{key}_clear={counts['clear_by_grid'][key]}<30")
        if counts["occupied_by_grid"][key] < THRESHOLDS["minimum_truth_occupied_cells_per_band_horizon"]:
            failures.append(f"{key}_occupied={counts['occupied_by_grid'][key]}<30")
    return failures


def truth_policy_dict(policy: TruthReaderPolicy) -> dict[str, Any]:
    value = asdict(policy)
    value["horizons_m"] = list(value["horizons_m"])
    return value


def load_fixed_plan(
    phase_a_manifest: dict[str, Any], coverage_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    require(coverage_manifest.get("terminal") == "D3R3_PHASE_B_EXACT64_MEMBER_COVERAGE_CENSUS_COMPLETE_NO_MEMBER_PAYLOAD_OR_TRUTH_READ", "coverage terminal drift")
    require(coverage_manifest.get("paired_exact_present_frame_count") == 9597, "coverage present count drift")
    require(coverage_manifest.get("paired_exact_missing_frame_count") == 3, "coverage missing count drift")
    selected = phase_a_manifest.get("selected_phase_a")
    coverage = coverage_manifest.get("processed")
    require(isinstance(selected, list) and len(selected) == 32, "Phase-A selection drift")
    require(isinstance(coverage, list) and len(coverage) == 32, "coverage identity drift")
    phase_a_processed = {
        (int(row["pool_order"]), str(row["visit_id"]), str(row["video_id"])): row
        for row in phase_a_manifest["processed"]
    }
    result: list[dict[str, Any]] = []
    for order, (identity, coverage_row) in enumerate(zip(selected, coverage, strict=True), start=1):
        require(identity["selection_order"] == order, "selection order drift")
        for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold"):
            require(identity[key] == coverage_row[key], f"coverage plan drift: {key}")
        stems = list(identity["selected_frame_stems"])
        require(len(stems) == 300 and len(set(stems)) == 300, "selected stem plan drift")
        require(selected_stem_sha256(stems) == coverage_row["selected_frame_plan_sha256"], "selected stem digest drift")
        key = (int(identity["pool_order"]), str(identity["visit_id"]), str(identity["video_id"]))
        require(key in phase_a_processed, "Phase-A processed identity missing")
        result.append(identity | {
            "phase_a_checkpoint": phase_a_processed[key],
            "coverage": coverage_row,
        })
    return result


def evaluate_identity(
    identity: dict[str, Any], phase_a_root: Path, coverage_root: Path,
    policy: TruthReaderPolicy,
) -> dict[str, Any]:
    video_id = str(identity["video_id"])
    selected_stems = list(identity["selected_frame_stems"])
    intrinsics_path = phase_a_root / "source" / "Training" / video_id / "lowres_wide_intrinsics.zip"
    trajectory_path = phase_a_root / "source" / "Training" / video_id / "lowres_wide.traj"
    depth_path = coverage_root / "source" / "Training" / video_id / "lowres_depth.zip"
    confidence_path = coverage_root / "source" / "Training" / video_id / "confidence.zip"
    for path in (intrinsics_path, trajectory_path, depth_path, confidence_path):
        require(path.is_file(), f"source missing: {path}")
    trajectory = parse_trajectory(trajectory_path)
    counts = empty_counts()
    orientation_counts = {str(index): 0 for index in range(4)}
    confidence_values: set[int] = set()
    depth_sizes: set[tuple[int, int]] = set()
    maximum_pose_gap = 0.0
    with (
        zipfile.ZipFile(intrinsics_path) as intrinsics_zip,
        zipfile.ZipFile(depth_path) as depth_zip,
        zipfile.ZipFile(confidence_path) as confidence_zip,
    ):
        intrinsics_map = member_map(intrinsics_zip, ".pincam")
        depth_map = member_map(depth_zip, ".png")
        confidence_map = member_map(confidence_zip, ".png")
        require(all(stem in intrinsics_map for stem in selected_stems), "selected intrinsics coverage drift")
        availability = resolve_frame_availability(selected_stems, set(depth_map), set(confidence_map))
        coverage = identity["coverage"]
        require(availability["source_unavailable"] == coverage["paired_exact_missing_stems"], "coverage manifest/source drift")
        require(availability["depth_missing"] == coverage["lowres_depth"]["selected_missing_stems"], "depth coverage drift")
        require(availability["confidence_missing"] == coverage["confidence"]["selected_missing_stems"], "confidence coverage drift")
        for stem in availability["available"]:
            intrinsics, source_size = parse_pincam_payload(
                intrinsics_zip.read(intrinsics_map[stem]), intrinsics_map[stem]
            )
            with depth_zip.open(depth_map[stem]) as stream, Image.open(stream) as image:
                depth_raw = np.asarray(image).copy()
            with confidence_zip.open(confidence_map[stem]) as stream, Image.open(stream) as image:
                confidence = np.asarray(image).copy()
            require(depth_raw.ndim == 2 and np.issubdtype(depth_raw.dtype, np.integer), "invalid depth raster")
            require(confidence.ndim == 2 and np.issubdtype(confidence.dtype, np.integer), "invalid confidence raster")
            require(confidence.shape == depth_raw.shape, "depth/confidence shape drift")
            require(source_size == (depth_raw.shape[1], depth_raw.shape[0]), "intrinsics/depth shape drift")
            depth_sizes.add((int(depth_raw.shape[1]), int(depth_raw.shape[0])))
            confidence_values.update(int(value) for value in np.unique(confidence))
            pose, pose_meta = interpolate_camera_to_world(
                trajectory, timestamp_from_stem(stem), policy.maximum_pose_bracketing_gap_seconds
            )
            maximum_pose_gap = max(maximum_pose_gap, float(pose_meta["bracketing_gap_seconds"]))
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
    failures = qualification_failures(counts)
    available_count = len(availability["available"])
    coverage_evaluable = available_count >= MINIMUM_SOURCE_AVAILABLE_FRAMES
    if not coverage_evaluable:
        failures.append(f"source_available_frames={available_count}<{MINIMUM_SOURCE_AVAILABLE_FRAMES}")
    support_qualified = not failures
    return {
        "selection_order": int(identity["selection_order"]),
        "pool_order": int(identity["pool_order"]),
        "visit_id": str(identity["visit_id"]),
        "video_id": video_id,
        "fold": "Training",
        "fixed_frame_count": 300,
        "selected_frame_plan_sha256": selected_stem_sha256(selected_stems),
        "source_available_frame_count": available_count,
        "source_unavailable_frame_count": len(availability["source_unavailable"]),
        "source_unavailable_stems": availability["source_unavailable"],
        "source_unavailable_unknown_cell_count": len(availability["source_unavailable"]) * 9,
        "source_unavailable_invalid_clearance_count": len(availability["source_unavailable"]) * 3,
        "source_unavailable_clear_count": 0,
        "source_unavailable_occupied_count": 0,
        "neighbor_substitution_used": False,
        "denominator_rewritten": False,
        "coverage_evaluable": coverage_evaluable,
        "truth_support": counts,
        "source_truth_support_qualified": support_qualified,
        "strict_complete_case_qualified": support_qualified and not availability["source_unavailable"],
        "qualification_failures": failures,
        "trajectory_row_count": int(trajectory.shape[0]),
        "maximum_pose_bracketing_gap_seconds": maximum_pose_gap,
        "orientation_counts_of_available_frames": orientation_counts,
        "depth_sizes_wh": [list(value) for value in sorted(depth_sizes)],
        "confidence_values": sorted(confidence_values),
        "per_frame_truth_retained": False,
    }


def select_first(processed: list[dict[str, Any]], key: str, target: int = 16) -> list[dict[str, Any]]:
    qualified = [row for row in processed if row[key]]
    if len(qualified) < target:
        return []
    return [
        {
            "phase_b_selection_order": order,
            "phase_a_selection_order": row["selection_order"],
            "pool_order": row["pool_order"],
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
            "fixed_frame_count": 300,
            "selected_frame_plan_sha256": row["selected_frame_plan_sha256"],
            "source_unavailable_frame_count": row["source_unavailable_frame_count"],
        }
        for order, row in enumerate(qualified[:target], start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--coverage-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase_a_manifest = load_json(args.phase_a_manifest)
    coverage_manifest = load_json(args.coverage_manifest)
    coverage_validation = load_json(args.coverage_validation)
    require(coverage_validation.get("status") == "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_VALIDATION_PASS", "coverage validation drift")
    require(coverage_validation.get("manifest_sha256") == sha256_file(args.coverage_manifest), "coverage validation binding drift")
    selected = load_fixed_plan(phase_a_manifest, coverage_manifest)
    policy = TruthReaderPolicy()
    policy.validate()
    phase_a_root = args.phase_a_manifest.parent
    coverage_root = args.coverage_manifest.parent
    processed: list[dict[str, Any]] = []
    for identity in selected:
        row = evaluate_identity(identity, phase_a_root, coverage_root, policy)
        processed.append(row)
        print(json.dumps({
            "completed": len(processed),
            "total": 32,
            "video_id": row["video_id"],
            "available": row["source_available_frame_count"],
            "qualified": row["source_truth_support_qualified"],
        }, sort_keys=True), flush=True)
    primary = select_first(processed, "source_truth_support_qualified")
    strict = select_first(processed, "strict_complete_case_qualified")
    result = {
        "schema": "blindassist_depthart_task_preserving_d3r3_phase_b_source_truth_unknown_experiment_v1",
        "status": "D3R3_PHASE_B_SOURCE_TRUTH_SUPPORT_PASS" if primary else "D3R3_PHASE_B_SOURCE_TRUTH_SUPPORT_NOT_EVALUABLE",
        "input_bindings": {
            "phase_a_manifest_sha256": sha256_file(args.phase_a_manifest),
            "coverage_manifest_sha256": sha256_file(args.coverage_manifest),
            "coverage_validation_sha256": sha256_file(args.coverage_validation),
        },
        "problem": "Preserve the fixed 300-frame task while three exact depth/confidence stems are unavailable.",
        "hypothesis": "Treating source absence as UNKNOWN rather than negative or replacement preserves usable support without hiding coverage loss.",
        "primary_policy": {
            "name": "FIXED_300_SOURCE_UNAVAILABLE_UNKNOWN",
            "fixed_frame_count": 300,
            "minimum_source_available_frames": MINIMUM_SOURCE_AVAILABLE_FRAMES,
            "missing_frame_contribution": "9 UNKNOWN cells and 3 invalid clearances; zero clear and zero occupied",
            "neighbor_substitution": False,
            "denominator_rewrite": False,
        },
        "sensitivity_policy": "STRICT_300_OF_300_COMPLETE_CASE_IDENTITY",
        "truth_reader_policy": truth_policy_dict(policy),
        "truth_support_thresholds": THRESHOLDS,
        "processed_identity_count": 32,
        "fixed_frame_count": 9600,
        "source_available_frame_count": sum(row["source_available_frame_count"] for row in processed),
        "source_unavailable_frame_count": sum(row["source_unavailable_frame_count"] for row in processed),
        "source_unavailable_unknown_cell_count": sum(row["source_unavailable_unknown_cell_count"] for row in processed),
        "primary_qualified_identity_count": sum(bool(row["source_truth_support_qualified"]) for row in processed),
        "strict_complete_case_qualified_identity_count": sum(bool(row["strict_complete_case_qualified"]) for row in processed),
        "primary_selected_phase_b": primary,
        "strict_complete_case_selected_phase_b": strict,
        "selection_policy_sensitive": [row["video_id"] for row in primary] != [row["video_id"] for row in strict],
        "processed": processed,
        "per_frame_truth_retained": False,
        "rgb_read": False,
        "model_output_read": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "android_default_authority": False,
        "production_authority": False,
        "safety_authority": False,
        "next_action": "REGISTER_EXACT16_PHASE_C_RGB_SCOPE" if primary else None,
    }
    write_json_exclusive(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "qualified": result["primary_qualified_identity_count"],
        "strict_qualified": result["strict_complete_case_qualified_identity_count"],
        "selection_policy_sensitive": result["selection_policy_sensitive"],
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
