#!/usr/bin/env python3
"""Fail-closed validation for the frozen B1 TRAIN target materialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)
from scripts.research.assistive_geometry.materialize_b1_train_targets import tensor_intrinsics  # noqa: E402


REQUIRED_KEYS = {
    "depth_m_source",
    "depth_valid_source",
    "ground_probability_source",
    "ground_label_valid_source",
    "intrinsics_source",
    "intrinsics_tensor",
    "up_camera",
    "camera_height_m",
    "ground_plane_valid",
    "clearance_m",
    "clearance_valid",
    "occupancy",
    "occupancy_valid",
    "band_confidence_valid",
    "target_hw",
    "orientation_index",
}


def validate_target_arrays(frame: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, int]:
    require(set(arrays) == REQUIRED_KEYS, f"target key drift: {frame['frame_stem']}")
    source_hw = tuple(int(value) for value in frame["source_hw"])
    target_hw = tuple(int(value) for value in frame["target_hw"])
    orientation = int(frame["orientation_index"])
    require(source_hw in ((192, 256), (256, 192)), f"source shape drift: {frame['frame_stem']}")
    require(target_hw == ((608, 448) if orientation in (1, 3) else (448, 608)), f"target shape drift: {frame['frame_stem']}")
    require(frame["orientation_family"] == ("portrait" if orientation in (1, 3) else "landscape"), f"orientation family drift: {frame['frame_stem']}")

    dense = arrays["depth_m_source"]
    depth_valid = arrays["depth_valid_source"]
    ground = arrays["ground_probability_source"]
    ground_valid = arrays["ground_label_valid_source"]
    require(dense.shape == source_hw and dense.dtype == np.float32, f"depth layout drift: {frame['frame_stem']}")
    require(depth_valid.shape == source_hw and depth_valid.dtype == np.bool_, f"depth mask drift: {frame['frame_stem']}")
    require(ground.shape == source_hw and ground.dtype == np.float32, f"ground layout drift: {frame['frame_stem']}")
    require(ground_valid.shape == source_hw and ground_valid.dtype == np.bool_, f"ground mask drift: {frame['frame_stem']}")
    require(np.all(np.isfinite(dense[depth_valid])) and np.all(dense[depth_valid] > 0.0), f"invalid valid-depth domain: {frame['frame_stem']}")
    require(np.all(np.isfinite(ground)) and np.all((ground >= 0.0) & (ground <= 1.0)), f"ground probability domain drift: {frame['frame_stem']}")

    plane_valid = bool(arrays["ground_plane_valid"].item())
    require(plane_valid == bool(frame["ground_plane_valid"]), f"ground-plane receipt drift: {frame['frame_stem']}")
    camera_height = float(arrays["camera_height_m"].item())
    if plane_valid:
        require(np.isfinite(camera_height) and camera_height > 0.0, f"invalid camera height: {frame['frame_stem']}")
        require(np.array_equal(ground_valid, depth_valid), f"ground support mask drift: {frame['frame_stem']}")
    else:
        require(np.isnan(camera_height), f"UNKNOWN camera height must be NaN: {frame['frame_stem']}")
        require(not np.any(ground_valid), f"UNKNOWN ground mask leak: {frame['frame_stem']}")

    intrinsics_source = arrays["intrinsics_source"]
    intrinsics_tensor = arrays["intrinsics_tensor"]
    require(intrinsics_source.shape == (3, 3) and np.all(np.isfinite(intrinsics_source)), f"source K drift: {frame['frame_stem']}")
    require(intrinsics_tensor.shape == (3, 3) and np.all(np.isfinite(intrinsics_tensor)), f"tensor K drift: {frame['frame_stem']}")
    expected_tensor = tensor_intrinsics(intrinsics_source, source_hw, target_hw)
    intrinsics_difference = np.abs(intrinsics_tensor - expected_tensor)
    one_ulp = np.spacing(np.maximum(np.abs(intrinsics_tensor), np.abs(expected_tensor)))
    require(np.all(intrinsics_difference <= one_ulp), f"tensor K scaling drift exceeds one FP32 ULP: {frame['frame_stem']}")
    up = arrays["up_camera"]
    require(up.shape == (3,) and np.all(np.isfinite(up)), f"up-camera drift: {frame['frame_stem']}")
    require(abs(float(np.linalg.norm(up)) - 1.0) <= 1e-5, f"up-camera normalization drift: {frame['frame_stem']}")

    clearance = arrays["clearance_m"]
    clearance_valid = arrays["clearance_valid"]
    occupancy = arrays["occupancy"]
    occupancy_valid = arrays["occupancy_valid"]
    confidence_valid = arrays["band_confidence_valid"]
    require(clearance.shape == (3,) and clearance.dtype == np.float32, f"clearance layout drift: {frame['frame_stem']}")
    require(clearance_valid.shape == (3,) and clearance_valid.dtype == np.bool_, f"clearance mask drift: {frame['frame_stem']}")
    require(occupancy.shape == (3, 3) and occupancy.dtype == np.float32, f"occupancy layout drift: {frame['frame_stem']}")
    require(occupancy_valid.shape == (3, 3) and occupancy_valid.dtype == np.bool_, f"occupancy mask drift: {frame['frame_stem']}")
    require(confidence_valid.shape == (3,) and confidence_valid.dtype == np.bool_, f"confidence-valid layout drift: {frame['frame_stem']}")
    require(np.all(np.isfinite(clearance[clearance_valid])) and np.all(clearance[clearance_valid] >= 0.0), f"clearance domain drift: {frame['frame_stem']}")
    require(not np.any(clearance[~clearance_valid]), f"UNKNOWN clearance filled as clear: {frame['frame_stem']}")
    require(np.all(np.isin(occupancy[occupancy_valid], (0.0, 1.0))), f"occupancy domain drift: {frame['frame_stem']}")
    require(not np.any(occupancy[~occupancy_valid]), f"UNKNOWN occupancy filled: {frame['frame_stem']}")
    require(np.array_equal(confidence_valid, np.all(occupancy_valid, axis=1)), f"confidence validity drift: {frame['frame_stem']}")
    require(np.array_equal(arrays["target_hw"], np.asarray(target_hw, dtype=np.int32)), f"target HW payload drift: {frame['frame_stem']}")
    require(int(arrays["orientation_index"].item()) == orientation, f"orientation payload drift: {frame['frame_stem']}")
    return {
        "ground_plane_valid": int(plane_valid),
        "clearance_known_bands": int(np.count_nonzero(clearance_valid)),
        "occupancy_known_cells": int(np.count_nonzero(occupancy_valid)),
        "confidence_known_bands": int(np.count_nonzero(confidence_valid)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.execution_protocol.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == "blindassist_assistive_geometry_b1_execution_lock_protocol_v1", "execution protocol schema drift")
    require(protocol["target_validator"]["sha256"] == sha256_file(Path(__file__)), "target validator SHA drift")
    manifest_path = args.manifest.resolve()
    require(manifest_path.is_file() and sha256_file(manifest_path) == protocol["target_manifest"]["sha256"], "target manifest binding drift")
    manifest = load_json(manifest_path)
    gates = protocol["target_gates"]
    for key in ("video_count", "frame_count", "portrait_frame_count", "landscape_frame_count"):
        require(int(manifest[key]) == int(gates[key]), f"manifest {key} drift")
    require(manifest["development_or_confirmation_content_opened"] is False, "role firewall drift")
    require(manifest["model_outputs_read"] is False, "model-output firewall drift")
    require(manifest["confidence_target_materialized"] is False, "prediction-dependent confidence target leak")

    root = manifest_path.parent.resolve()
    identities: set[tuple[str, str]] = set()
    visits: set[str] = set()
    stats = {"ground_plane_valid": 0, "clearance_known_bands": 0, "occupancy_known_cells": 0, "confidence_known_bands": 0}
    frame_count = portrait_count = landscape_count = 0
    for video in manifest["videos"]:
        require(video["role"] == "TRAIN", f"non-TRAIN receipt: {video['video_id']}")
        require(str(video["visit_id"]) not in visits, f"duplicate TRAIN visit: {video['visit_id']}")
        visits.add(str(video["visit_id"]))
        for frame in video["frames"]:
            identity = (str(video["video_id"]), str(frame["frame_stem"]))
            require(identity not in identities, f"duplicate frame identity: {identity}")
            identities.add(identity)
            require(frame["confidence_target_materialized"] is False, f"confidence target leak: {identity}")
            target = Path(frame["target"]["path"]).resolve()
            require(target.is_relative_to(root), f"target escaped materialization root: {target}")
            require(target.is_file() and target.stat().st_size == int(frame["target"]["bytes"]), f"target size drift: {target}")
            require(sha256_file(target) == frame["target"]["sha256"], f"target SHA drift: {target}")
            with np.load(target, allow_pickle=False) as payload:
                row = validate_target_arrays(frame, {key: payload[key] for key in payload.files})
            for key, value in row.items():
                stats[key] += value
            frame_count += 1
            portrait_count += int(frame["orientation_family"] == "portrait")
            landscape_count += int(frame["orientation_family"] == "landscape")
    require(len(visits) == gates["video_count"] and frame_count == gates["frame_count"], "validated identity count drift")
    require(portrait_count == gates["portrait_frame_count"] and landscape_count == gates["landscape_frame_count"], "validated orientation count drift")

    receipt = {
        "schema": "blindassist_assistive_geometry_b1_train_target_validation_v1",
        "execution_protocol_sha256": sha256_file(protocol_path),
        "producer_sha256": protocol["target_validator"]["sha256"],
        "manifest": {"path": str(manifest_path), "bytes": manifest_path.stat().st_size, "sha256": protocol["target_manifest"]["sha256"]},
        "video_count": len(visits),
        "frame_count": frame_count,
        "portrait_frame_count": portrait_count,
        "landscape_frame_count": landscape_count,
        **stats,
        "all_target_sizes_and_sha256_verified": True,
        "all_npz_semantics_verified": True,
        "unknown_clearance_not_filled_as_clear": True,
        "development_or_confirmation_content_opened": False,
        "model_outputs_read": False,
        "confidence_target_materialized": False,
        "terminal": "B1_TRAIN_TARGET_INTEGRITY_AND_ROLE_FIREWALL_PASS",
        "authority": "TRAIN target integrity only; no model quality, Development, Confirmation, deployment, product or safety authority.",
    }
    write_json_exclusive(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
