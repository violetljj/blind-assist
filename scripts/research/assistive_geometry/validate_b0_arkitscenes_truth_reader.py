#!/usr/bin/env python3
"""Validate the Assistive Geometry B0 truth reader on frozen TRAIN data only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    WORLD_UP,
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    load_manifest_frame,
    orientation_index,
    parse_pincam,
    parse_trajectory,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (
    bound_file,
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)


PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b0_truth_reader_validation_protocol_v1"
RECEIPT_SCHEMA = "blindassist_assistive_geometry_b0_truth_reader_validation_receipt_v1"
SKY_DIRECTION_INDEX = {"Up": 0, "Left": 1, "Down": 2, "Right": 3}
BANDS = ("left", "center", "right")


def _timestamp(stem: str) -> float:
    return float(stem.rsplit("_", 1)[1])


def _entry_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        stem = Path(entry["path"]).stem
        require(stem not in result, f"duplicate extracted stem: {stem}")
        result[stem] = entry
    return result


def _nearest_entry(entries: list[dict[str, Any]], stem: str) -> tuple[dict[str, Any], float]:
    require(entries, "no intrinsics entries")
    target = _timestamp(stem)
    difference, entry = min(
        ((abs(_timestamp(Path(entry["path"]).stem) - target), entry) for entry in entries),
        key=lambda value: value[0],
    )
    require(difference <= 0.05, f"intrinsics timestamp gap {difference} exceeds 50 ms")
    return entry, difference


def _read_image(entry: dict[str, Any]) -> np.ndarray:
    path = Path(entry["path"])
    require(path.is_file() and path.stat().st_size == int(entry["bytes"]), f"image receipt drift: {path}")
    with Image.open(path) as image:
        return np.asarray(image).copy()


def _derive(depth_raw_mm: np.ndarray, confidence: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> dict[str, Any]:
    rgb = np.zeros((*depth_raw_mm.shape, 3), dtype=np.uint8)
    canonical = canonicalize_frame(rgb, depth_raw_mm, confidence, intrinsics, pose)
    depth_m = depth_mm_to_metres(canonical["depth_raw_mm"])
    up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
    return derive_assistive_truth(depth_m, canonical["confidence"], canonical["intrinsics"], up_camera)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _evaluate_upsampling(
    manifest: dict[str, Any],
    per_frame_gates: dict[str, float] | None,
) -> dict[str, Any]:
    require(manifest.get("schema") == "blindassist_assistive_geometry_b0_arkitscenes_upsampling_train_manifest_v1", "upsampling manifest schema drift")
    require(manifest.get("terminal") == "B0_ARKIT_UPSAMPLING_TRAIN_MATERIALIZED_VALIDATION_PENDING", "upsampling manifest terminal drift")
    require(manifest.get("development_or_confirmation_opened") is False, "upsampling role firewall drift")
    require(manifest.get("model_outputs_read") is False, "upsampling model-output firewall drift")
    frame_rows: list[dict[str, Any]] = []
    ground_differences: list[float] = []
    clearance_differences: list[float] = []
    occupancy_agreements: list[bool] = []
    intrinsics_gaps: list[float] = []
    video_ids: list[str] = []

    for video in manifest["videos"]:
        require(video["role"] == "TRAIN", f"non-TRAIN upsampling video admitted: {video['video_id']}")
        video_id = str(video["video_id"])
        video_ids.append(video_id)
        expected_orientation = SKY_DIRECTION_INDEX[str(video["sky_direction"])]
        maps = {name: _entry_map(video["extracted"][name]) for name in ("wide", "highres_depth", "lowres_depth", "confidence")}
        selected = [str(value) for value in video["selected_frame_stems"]]
        require(all(set(selected) == set(mapping) for mapping in maps.values()), f"modality stem drift: {video_id}")
        trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
        for stem in selected:
            wide = _read_image(maps["wide"][stem])
            high_raw = _read_image(maps["highres_depth"][stem])
            low_raw = _read_image(maps["lowres_depth"][stem])
            confidence = _read_image(maps["confidence"][stem])
            require(wide.shape[:2] == (1440, 1920), f"high-resolution RGB shape drift: {stem}")
            require(high_raw.shape == (1440, 1920), f"FARO depth shape drift: {stem}")
            require(low_raw.shape == (192, 256) and confidence.shape == low_raw.shape, f"AppleDepth shape drift: {stem}")
            require(np.issubdtype(high_raw.dtype, np.integer) and np.issubdtype(low_raw.dtype, np.integer), f"depth dtype drift: {stem}")
            intrinsics_entry, intrinsics_gap = _nearest_entry(video["extracted"]["lowres_wide_intrinsics"], stem)
            intrinsics_gaps.append(intrinsics_gap)
            intrinsics, source_size = parse_pincam(Path(intrinsics_entry["path"]))
            require(source_size == (256, 192), f"pincam size drift: {stem}")
            timestamp = _timestamp(stem)
            pose, interpolation = interpolate_camera_to_world(
                trajectory,
                timestamp,
                TruthReaderPolicy().maximum_pose_bracketing_gap_seconds,
            )
            observed_orientation = orientation_index(pose)

            high_down = cv2.resize(high_raw, (256, 192), interpolation=cv2.INTER_NEAREST)
            low_m = depth_mm_to_metres(low_raw)
            high_m = depth_mm_to_metres(high_down)
            overlap = (
                (confidence >= TruthReaderPolicy().minimum_sensor_confidence)
                & (low_m >= TruthReaderPolicy().depth_min_m)
                & (low_m <= TruthReaderPolicy().depth_max_m)
                & (high_m >= TruthReaderPolicy().depth_min_m)
                & (high_m <= TruthReaderPolicy().depth_max_m)
            )
            overlap_fraction = float(np.mean(overlap))
            require(np.any(overlap), f"no valid AppleDepth/FARO overlap: {stem}")
            absolute = np.abs(low_m[overlap] - high_m[overlap])
            ratio = low_m[overlap] / high_m[overlap]
            median_absolute = float(np.quantile(absolute, 0.5))
            p90_absolute = float(np.quantile(absolute, 0.9))
            median_ratio = float(np.quantile(ratio, 0.5))

            low_truth = _derive(low_raw, confidence, intrinsics, pose)
            high_truth = _derive(high_down, confidence, intrinsics, pose)
            if low_truth["ground_plane"] is not None and high_truth["ground_plane"] is not None:
                ground_differences.append(
                    abs(float(low_truth["ground_plane"]["camera_height_m"]) - float(high_truth["ground_plane"]["camera_height_m"]))
                )
            for band in BANDS:
                low_band = low_truth.get("bands", {}).get(band)
                high_band = high_truth.get("bands", {}).get(band)
                if not low_band or not high_band:
                    continue
                low_clearance = low_band.get("clearance_m")
                high_clearance = high_band.get("clearance_m")
                if low_clearance is not None and high_clearance is not None:
                    clearance_differences.append(abs(float(low_clearance) - float(high_clearance)))
                for horizon in TruthReaderPolicy().horizons_m:
                    low_occupied = low_band["occupied_by_horizon"][str(horizon)]
                    high_occupied = high_band["occupied_by_horizon"][str(horizon)]
                    if low_occupied is not None and high_occupied is not None:
                        occupancy_agreements.append(bool(low_occupied == high_occupied))

            frame_pass = None
            if per_frame_gates is not None:
                frame_pass = bool(
                    overlap_fraction >= per_frame_gates["minimum_overlap_fraction"]
                    and median_absolute <= per_frame_gates["maximum_median_absolute_error_m"]
                    and p90_absolute <= per_frame_gates["maximum_p90_absolute_error_m"]
                    and per_frame_gates["minimum_median_depth_ratio"] <= median_ratio <= per_frame_gates["maximum_median_depth_ratio"]
                    and observed_orientation == expected_orientation
                    and float(interpolation["bracketing_gap_seconds"]) <= TruthReaderPolicy().maximum_pose_bracketing_gap_seconds
                )
            frame_rows.append({
                "video_id": video_id,
                "frame_stem": stem,
                "overlap_fraction": overlap_fraction,
                "median_absolute_error_m": median_absolute,
                "p90_absolute_error_m": p90_absolute,
                "median_apple_to_faro_ratio": median_ratio,
                "expected_orientation_index": expected_orientation,
                "observed_orientation_index": observed_orientation,
                "pose_bracketing_gap_seconds": float(interpolation["bracketing_gap_seconds"]),
                "frame_gate_pass": frame_pass,
            })

    require(len(video_ids) == len(set(video_ids)), "duplicate upsampling video")
    return {
        "video_count": len(video_ids),
        "video_ids": video_ids,
        "frame_count": len(frame_rows),
        "orientation_agreement_fraction": float(np.mean([row["observed_orientation_index"] == row["expected_orientation_index"] for row in frame_rows])),
        "frame_gate_pass_fraction": None if per_frame_gates is None else float(np.mean([row["frame_gate_pass"] for row in frame_rows])),
        "maximum_intrinsics_timestamp_gap_seconds": max(intrinsics_gaps, default=None),
        "overlap_fraction": _summary([float(row["overlap_fraction"]) for row in frame_rows]),
        "frame_median_absolute_error_m": _summary([float(row["median_absolute_error_m"]) for row in frame_rows]),
        "frame_p90_absolute_error_m": _summary([float(row["p90_absolute_error_m"]) for row in frame_rows]),
        "frame_median_apple_to_faro_ratio": _summary([float(row["median_apple_to_faro_ratio"]) for row in frame_rows]),
        "ground_height_absolute_difference_m": _summary(ground_differences),
        "clearance_absolute_difference_m": _summary(clearance_differences),
        "occupied_decision_pair_count": len(occupancy_agreements),
        "occupied_decision_agreement_fraction": float(np.mean(occupancy_agreements)) if occupancy_agreements else None,
        "frames": frame_rows,
    }


def _evaluate_main_train(manifest: dict[str, Any], stride: int) -> dict[str, Any]:
    require(stride > 0, "main sample stride must be positive")
    require(manifest.get("schema") == "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1", "main manifest schema drift")
    require(manifest.get("task_outcome_opened") is False, "main manifest task-outcome firewall drift")
    require(manifest.get("model_outputs_read") is False, "main manifest model-output firewall drift")
    train_videos = [video for video in manifest["videos"] if video["role"] == "TRAIN"]
    require(len(train_videos) == 16, "main manifest TRAIN count drift")
    rows: list[dict[str, Any]] = []
    videos_with_ground: set[str] = set()
    unknown_clearance_leak_count = 0
    for video in train_videos:
        trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
        for index in range(0, len(video["selected_frame_stems"]), stride):
            frame = load_manifest_frame(video, index, trajectory)
            truth = frame["truth"]
            if truth["ground_plane"] is not None:
                videos_with_ground.add(str(video["video_id"]))
            if truth["status"] == "UNKNOWN" and any(
                band.get("clearance_m") is not None for band in truth.get("bands", {}).values()
            ):
                unknown_clearance_leak_count += 1
            all_bands_known = bool(truth.get("bands")) and all(
                truth["bands"].get(band, {}).get("status") == "KNOWN" for band in BANDS
            )
            rows.append({
                "video_id": str(video["video_id"]),
                "frame_stem": frame["identity"]["frame_stem"],
                "status": truth["status"],
                "ground_available": truth["ground_plane"] is not None,
                "all_bands_known": all_bands_known,
                "pose_bracketing_gap_seconds": float(frame["pose_interpolation"]["bracketing_gap_seconds"]),
            })
    return {
        "train_video_count": len(train_videos),
        "evaluated_frame_count": len(rows),
        "sample_stride": stride,
        "ground_available_fraction": float(np.mean([row["ground_available"] for row in rows])),
        "all_bands_known_fraction": float(np.mean([row["all_bands_known"] for row in rows])),
        "videos_with_ground_count": len(videos_with_ground),
        "maximum_pose_bracketing_gap_seconds": max(float(row["pose_bracketing_gap_seconds"]) for row in rows),
        "unknown_clearance_leak_count": unknown_clearance_leak_count,
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in ("VALID", "PARTIAL_UNKNOWN", "UNKNOWN")},
    }


def _judge(metrics: dict[str, Any], gates: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    upsampling = metrics["upsampling_train"]
    main = metrics["main_train_capability"]
    checks = {
        "upsampling_video_count": upsampling["video_count"] == gates["upsampling"]["expected_video_count"],
        "upsampling_frame_count": upsampling["frame_count"] >= gates["upsampling"]["minimum_frame_count"],
        "orientation_agreement": upsampling["orientation_agreement_fraction"] >= gates["upsampling"]["minimum_orientation_agreement_fraction"],
        "frame_gate_pass_fraction": upsampling["frame_gate_pass_fraction"] >= gates["upsampling"]["minimum_frame_gate_pass_fraction"],
        "intrinsics_gap": upsampling["maximum_intrinsics_timestamp_gap_seconds"] <= gates["upsampling"]["maximum_intrinsics_timestamp_gap_seconds"],
        "dual_ground_count": upsampling["ground_height_absolute_difference_m"]["count"] >= gates["upsampling"]["minimum_dual_ground_frame_count"],
        "ground_height_median": upsampling["ground_height_absolute_difference_m"]["median"] <= gates["upsampling"]["maximum_ground_height_median_absolute_difference_m"],
        "clearance_pair_count": upsampling["clearance_absolute_difference_m"]["count"] >= gates["upsampling"]["minimum_clearance_pair_count"],
        "clearance_median": upsampling["clearance_absolute_difference_m"]["median"] <= gates["upsampling"]["maximum_clearance_median_absolute_difference_m"],
        "occupied_agreement": upsampling["occupied_decision_agreement_fraction"] >= gates["upsampling"]["minimum_occupied_decision_agreement_fraction"],
        "main_frame_count": main["evaluated_frame_count"] == gates["main_train_capability"]["expected_frame_count"],
        "main_ground_fraction": main["ground_available_fraction"] >= gates["main_train_capability"]["minimum_ground_available_fraction"],
        "main_band_fraction": main["all_bands_known_fraction"] >= gates["main_train_capability"]["minimum_all_bands_known_fraction"],
        "main_video_coverage": main["videos_with_ground_count"] == gates["main_train_capability"]["expected_videos_with_ground_count"],
        "main_pose_gap": main["maximum_pose_bracketing_gap_seconds"] <= gates["main_train_capability"]["maximum_pose_bracketing_gap_seconds"],
        "unknown_fail_closed": main["unknown_clearance_leak_count"] == 0,
    }
    return checks, all(checks.values())


def _binding(root: Path, protocol: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    value = protocol[key]
    return bound_file(root, value), value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--main-manifest", type=Path, required=True)
    parser.add_argument("--upsampling-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-stride", type=int, default=10)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    main_path = args.main_manifest.resolve()
    upsampling_path = args.upsampling_manifest.resolve()
    protocol: dict[str, Any] | None = None
    per_frame_gates = None
    if args.protocol is not None:
        require(args.output is not None, "final validation requires --output")
        protocol = load_json(args.protocol.resolve())
        require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
        require(protocol["validator"]["sha256"] == sha256_file(Path(__file__)), "validator SHA drift")
        for key in ("validator_test", "truth_reader", "truth_reader_test"):
            bound_file(root, protocol[key])
        for binding in protocol["source_bindings"]:
            bound_file(root, binding)
        bound_main, _ = _binding(root, protocol, "main_manifest")
        bound_upsampling, _ = _binding(root, protocol, "upsampling_manifest")
        require(bound_main == main_path and bound_upsampling == upsampling_path, "CLI manifest binding drift")
        per_frame_gates = protocol["gates"]["upsampling"]["per_frame"]
        require(args.sample_stride == int(protocol["main_train_sample_stride"]), "sample stride drift")
    else:
        require(args.output is None, "exploration cannot write a final receipt")

    main_manifest = load_json(main_path)
    upsampling_manifest = load_json(upsampling_path)
    metrics = {
        "upsampling_train": _evaluate_upsampling(upsampling_manifest, per_frame_gates),
        "main_train_capability": _evaluate_main_train(main_manifest, args.sample_stride),
    }
    if protocol is None:
        compact = {**metrics, "terminal": "TRAIN_EXPLORATION_ONLY_THRESHOLDS_NOT_FROZEN"}
        compact["upsampling_train"].pop("frames")
        print(json.dumps(compact, indent=2, sort_keys=True))
        return 0

    checks, passed = _judge(metrics, protocol["gates"])
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "validator_sha256": protocol["validator"]["sha256"],
        "truth_reader_sha256": protocol["truth_reader"]["sha256"],
        "main_manifest_sha256": protocol["main_manifest"]["sha256"],
        "upsampling_manifest_sha256": protocol["upsampling_manifest"]["sha256"],
        "truth_reader_policy": asdict(TruthReaderPolicy()),
        "metrics": metrics,
        "gate_checks": checks,
        "firewalls": {
            "train_only": True,
            "development_content_or_outcome_opened": False,
            "confirmation_content_or_outcome_opened": False,
            "model_outputs_read": False,
        },
        "authority": "TRAIN-only sensor-derived geometry validation; not human safety truth and no Development, Confirmation, deployment, product, production or safety authority.",
        "terminal": "B0_TRUTH_READER_AND_REGISTRATION_LOCK_PASS" if passed else "B0_TRUTH_READER_AND_REGISTRATION_LOCK_FAIL",
    }
    write_json_exclusive(args.output.resolve(), receipt)
    print(json.dumps({"terminal": receipt["terminal"], "gate_checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
