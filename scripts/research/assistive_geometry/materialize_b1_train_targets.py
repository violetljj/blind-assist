#!/usr/bin/env python3
"""Materialize compact, source-resolution B1 targets for frozen TRAIN frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    load_manifest_frame,
    parse_trajectory,
)
from scripts.research.assistive_geometry.audit_b1_orientation_geometry import (  # noqa: E402
    tensor_hw_for_orientation,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)


MANIFEST_SCHEMA = "blindassist_assistive_geometry_b1_train_target_manifest_v1"
SOURCE_MANIFEST_SHA256 = "F51041594FF34EE922F92A7A49AAD7CA44551B5C0A8D77A81693ECFFDD226D87"


def frozen_train_videos(manifest: dict[str, Any], expected: list[dict[str, str]]) -> list[dict[str, Any]]:
    lookup = {str(video["video_id"]): video for video in manifest["videos"] if video["role"] == "TRAIN"}
    require(len(lookup) == 16, "source TRAIN count drift")
    require(len(expected) == 16, "protocol TRAIN count drift")
    rows: list[dict[str, Any]] = []
    for identity in expected:
        video_id = str(identity["video_id"])
        require(video_id in lookup, f"protocol TRAIN video missing: {video_id}")
        video = lookup[video_id]
        require(str(video["visit_id"]) == str(identity["visit_id"]), f"TRAIN visit drift: {video_id}")
        rows.append(video)
    require(len({str(video["visit_id"]) for video in rows}) == 16, "TRAIN visit overlap")
    return rows


def tensor_intrinsics(intrinsics: np.ndarray, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    source_height, source_width = source_hw
    target_height, target_width = target_hw
    require(source_height > 0 and source_width > 0 and target_height > 0 and target_width > 0, "invalid tensor size")
    output = np.asarray(intrinsics, dtype=np.float64).copy()
    output[0, :] *= target_width / source_width
    output[1, :] *= target_height / source_height
    output[2, :] = [0.0, 0.0, 1.0]
    return output.astype(np.float32)


def build_band_targets(truth: dict[str, Any]) -> dict[str, np.ndarray]:
    clearance = np.zeros(3, dtype=np.float32)
    clearance_valid = np.zeros(3, dtype=np.bool_)
    occupancy = np.zeros((3, 3), dtype=np.float32)
    occupancy_valid = np.zeros((3, 3), dtype=np.bool_)
    horizons = TruthReaderPolicy().horizons_m
    for band_index, band in enumerate(("left", "center", "right")):
        value = truth.get("bands", {}).get(band)
        if not value:
            continue
        if value.get("clearance_m") is not None:
            clearance[band_index] = float(value["clearance_m"])
            clearance_valid[band_index] = True
        for horizon_index, horizon in enumerate(horizons):
            occupied = value["occupied_by_horizon"][str(horizon)]
            if occupied is not None:
                occupancy[band_index, horizon_index] = float(bool(occupied))
                occupancy_valid[band_index, horizon_index] = True
    confidence_valid = np.all(occupancy_valid, axis=1)
    return {
        "clearance_m": clearance,
        "clearance_valid": clearance_valid,
        "occupancy": occupancy,
        "occupancy_valid": occupancy_valid,
        "band_confidence_valid": confidence_valid,
    }


def resize_cached_dense(array: np.ndarray, target_hw: tuple[int, int], *, mask: bool) -> np.ndarray:
    height, width = target_hw
    resized = cv2.resize(array.astype(np.uint8 if mask else np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.bool_ if mask else np.float32)


def _save_npz_exclusive(path: Path, values: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    require(not path.exists() and not temporary.exists(), f"target output already exists: {path}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_video_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require(receipt.get("role") == "TRAIN", f"non-TRAIN target receipt: {path}")
    for frame in receipt["frames"]:
        target = Path(frame["target"]["path"])
        require(target.is_file() and target.stat().st_size == int(frame["target"]["bytes"]), f"target size drift: {target}")
        require(sha256_file(target) == frame["target"]["sha256"], f"target SHA drift: {target}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--overlay-protocol", type=Path, required=True)
    parser.add_argument("--implementation-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base_path = args.base_protocol.resolve()
    overlay_path = args.overlay_protocol.resolve()
    implementation_path = args.implementation_protocol.resolve()
    source_path = args.source_manifest.resolve()
    base = load_json(base_path)
    overlay = load_json(overlay_path)
    implementation = load_json(implementation_path)
    source = load_json(source_path)
    require(overlay["base_protocol"]["sha256"] == sha256_file(base_path), "base protocol hash drift")
    require(implementation.get("schema") == "blindassist_assistive_geometry_b1_implementation_lock_protocol_v1", "implementation protocol schema drift")
    require(implementation["target_materializer"]["sha256"] == sha256_file(Path(__file__)), "target materializer SHA drift")
    require(implementation["base_protocol"]["sha256"] == sha256_file(base_path), "implementation/base binding drift")
    require(implementation["overlay_protocol"]["sha256"] == sha256_file(overlay_path), "implementation/overlay binding drift")
    require(sha256_file(source_path) == SOURCE_MANIFEST_SHA256, "source manifest hash drift")
    require(overlay["authority"]["train_target_materialization"] is True, "target materialization not authorized")
    videos = frozen_train_videos(source, base["data_roles"]["TRAIN"])

    attempt = {
        "schema": "blindassist_assistive_geometry_b1_train_target_attempt_v1",
        "base_protocol_sha256": sha256_file(base_path),
        "overlay_protocol_sha256": sha256_file(overlay_path),
        "implementation_protocol_sha256": sha256_file(implementation_path),
        "source_manifest_sha256": sha256_file(source_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "output_root": str(args.output_root.resolve()),
    }
    attempt_path = args.output_root / "attempt.json"
    if args.output_root.exists():
        require(args.resume, "existing target root requires --resume")
        require(attempt_path.is_file() and load_json(attempt_path) == attempt, "target resume attempt drift")
        require(not (args.output_root / "manifest.json").exists(), "target manifest already complete")
    else:
        args.output_root.mkdir(parents=True)
        write_json_exclusive(attempt_path, attempt)

    receipts: list[dict[str, Any]] = []
    for video_index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"])
        receipt_path = args.output_root / "receipts" / f"{video_index:02d}-TRAIN-{video_id}.json"
        if receipt_path.exists():
            receipts.append(_validate_video_receipt(receipt_path))
            print(json.dumps({"completed": video_index, "total": 16, "video_id": video_id, "resumed": True}), flush=True)
            continue
        trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
        frames: list[dict[str, Any]] = []
        for frame_index in range(len(video["selected_frame_stems"])):
            frame = load_manifest_frame(video, frame_index, trajectory)
            identity = frame["identity"]
            truth = frame["truth"]
            orientation = int(frame["orientation"]["rotation_index"])
            target_hw = tensor_hw_for_orientation(orientation)
            depth = np.asarray(frame["depth_m_upright"], dtype=np.float32)
            depth_valid = np.asarray(truth["depth_valid"], dtype=np.bool_)
            ground_plane_valid = truth["ground_plane"] is not None
            ground_label_valid = depth_valid if ground_plane_valid else np.zeros_like(depth_valid)
            bands = build_band_targets(truth)
            target_values = {
                "depth_m_source": depth,
                "depth_valid_source": depth_valid,
                "ground_probability_source": np.asarray(truth["ground_probability"], dtype=np.float32),
                "ground_label_valid_source": ground_label_valid,
                "intrinsics_source": np.asarray(frame["intrinsics_upright"], dtype=np.float32),
                "intrinsics_tensor": tensor_intrinsics(frame["intrinsics_upright"], depth.shape, target_hw),
                "up_camera": np.asarray(frame["orientation"]["up_camera"], dtype=np.float32),
                "camera_height_m": np.asarray(float(truth["ground_plane"]["camera_height_m"]) if ground_plane_valid else np.nan, dtype=np.float32),
                "ground_plane_valid": np.asarray(ground_plane_valid, dtype=np.bool_),
                "clearance_m": bands["clearance_m"],
                "clearance_valid": bands["clearance_valid"],
                "occupancy": bands["occupancy"],
                "occupancy_valid": bands["occupancy_valid"],
                "band_confidence_valid": bands["band_confidence_valid"],
                "target_hw": np.asarray(target_hw, dtype=np.int32),
                "orientation_index": np.asarray(orientation, dtype=np.int8),
            }
            output = args.output_root / "targets" / video_id / f"{identity['frame_stem']}.npz"
            target_receipt = _save_npz_exclusive(output, target_values)
            rgb_entry = video["extracted"]["lowres_wide"][frame_index]
            frames.append({
                "frame_index": frame_index,
                "frame_stem": identity["frame_stem"],
                "orientation_index": orientation,
                "orientation_family": "portrait" if orientation in (1, 3) else "landscape",
                "source_hw": list(depth.shape),
                "target_hw": list(target_hw),
                "truth_status": truth["status"],
                "ground_plane_valid": ground_plane_valid,
                "rgb_source": rgb_entry,
                "target": target_receipt,
                "confidence_target_materialized": False,
            })
        receipt = {
            "role": "TRAIN",
            "visit_id": str(video["visit_id"]),
            "video_id": video_id,
            "frame_count": len(frames),
            "portrait_frame_count": sum(frame["orientation_family"] == "portrait" for frame in frames),
            "landscape_frame_count": sum(frame["orientation_family"] == "landscape" for frame in frames),
            "frames": frames,
        }
        write_json_exclusive(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps({"completed": video_index, "total": 16, "video_id": video_id, "frames": len(frames)}), flush=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        **{key: value for key, value in attempt.items() if key != "schema"},
        "storage": "compact source-upright dense targets plus exact tensor transform receipt",
        "dense_tensor_resize": "INTER_NEAREST at training load time",
        "video_count": len(receipts),
        "frame_count": sum(receipt["frame_count"] for receipt in receipts),
        "portrait_frame_count": sum(receipt["portrait_frame_count"] for receipt in receipts),
        "landscape_frame_count": sum(receipt["landscape_frame_count"] for receipt in receipts),
        "videos": receipts,
        "development_or_confirmation_content_opened": False,
        "model_outputs_read": False,
        "confidence_target_materialized": False,
        "terminal": "B1_TRAIN_TARGETS_MATERIALIZED_MODEL_IMPLEMENTATION_PENDING",
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "videos"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
