#!/usr/bin/env python3
"""Materialize training-only S1.1 geometry targets from the A4 RGB-D stream.

This utility is deliberately unable to read a model-variant gate roster.  It
uses only the 3,000-record A4 teacher manifest, per-frame low-resolution ARKit
depth/confidence and the corresponding ``.pincam`` intrinsics.  It is not an
evaluator and it neither trains a model nor opens the consumed 120-frame gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_metric3d_clearance_field_a0 import clearance_field  # noqa: E402


SCHEMA = "blindassist_hftf_clearance_student_mobile_s1_1_geometry_targets"
RECEIPT_SCHEMA = "blindassist_hftf_clearance_student_mobile_s1_1_geometry_target_preflight"
EXPECTED_RECORD_COUNT = 3000
BANDS = ("left", "center", "right")
OCCUPANCY_HORIZON_M = "1.5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def paths_for(record: dict[str, Any]) -> tuple[Path, Path, Path]:
    rgb = Path(str(record["rgb_path"]))
    root, stem = rgb.parent.parent, rgb.stem
    return (
        root / "lowres_depth" / f"{stem}.png",
        root / "confidence" / f"{stem}.png",
        root / "lowres_wide_intrinsics" / f"{stem}.pincam",
    )


def intrinsics_from_pincam(path: Path) -> np.ndarray:
    values = path.read_text(encoding="utf-8").split()
    if len(values) != 6:
        raise ValueError(f"invalid pincam (expected six values): {path}")
    width, height, fx, fy, cx, cy = (float(value) for value in values)
    if not (width > 0 and height > 0 and fx > 0 and fy > 0):
        raise ValueError(f"non-positive pincam values: {path}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def coverage(values: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(values, dtype=bool)
    fraction = float(valid.mean()) if valid.size else 0.0
    return {"known_count": int(valid.sum()), "known_fraction": fraction}


def materialize(protocol_path: Path, teacher_manifest_path: Path, output_root: Path, limit: int | None = None) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "clearance-student-mobile-s1-1":
        raise ValueError("S1.1 protocol binding mismatch")
    if manifest.get("truth_inputs_opened") is not False:
        raise ValueError("teacher manifest truth firewall mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_RECORD_COUNT} A4 records")
    if {str(row.get("role")) for row in records} != {"train", "validation"}:
        raise ValueError("A4 roles must be exactly train and validation")
    # Refuse accidental application to the consumed model-variant gate.
    if any("model-variant-gate-r0" in str(row.get("rgb_path", "")).lower() for row in records):
        raise ValueError("consumed 120-frame gate is forbidden for S1.1 targets")

    if limit is not None and (limit <= 0 or limit > len(records)):
        raise ValueError("limit must be within the A4 record count")
    records = records[:limit] if limit is not None else records
    count = len(records)
    plane = np.full((count, 4), np.nan, dtype=np.float32)  # normal xyz, offset
    camera_height = np.full(count, np.nan, dtype=np.float32)
    clearance = np.full((count, 3), np.nan, dtype=np.float32)
    # -1 is unknown; 0 clear; 1 occupied at frozen 1.5 m horizon.
    occupancy = np.full((count, 3), -1, dtype=np.int8)
    ground_valid = np.zeros(count, dtype=np.uint8)
    clearance_valid = np.zeros((count, 3), dtype=np.uint8)
    depth_valid_fraction = np.zeros(count, dtype=np.float32)
    role = np.empty(count, dtype="U10")
    frame_ids: list[str] = []

    for index, record in enumerate(records):
        depth_path, confidence_path, pincam_path = paths_for(record)
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence_map = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if raw_depth is None or confidence_map is None or not pincam_path.is_file():
            raise OSError(f"A4 geometry input missing for {record.get('frame_id')}")
        if raw_depth.shape != confidence_map.shape:
            raise ValueError(f"depth/confidence shape mismatch for {record.get('frame_id')}")
        depth_m = raw_depth.astype(np.float32) / 1000.0
        valid = (confidence_map == 2) & np.isfinite(depth_m) & (depth_m >= 0.25) & (depth_m <= 6.0)
        # Invalid/low-confidence truth must not participate in plane, clearance,
        # or occupancy targets.  This is the same pixel eligibility contracted
        # for the S1.1 depth, gradient and scale losses.
        masked_depth = np.where(valid, depth_m, np.nan).astype(np.float32)
        intrinsics = intrinsics_from_pincam(pincam_path)
        # Fit once and pass the frozen plane into clearance_field.  This keeps
        # the serialized plane and derived clearances bit-consistent and avoids
        # running the 240-iteration RANSAC twice per frame.
        from evaluate_metric3d_clearance_field_a0 import depth_to_points, fit_ground_plane
        points, pixels = depth_to_points(masked_depth, intrinsics)
        fitted = fit_ground_plane(points, pixels, masked_depth.shape[0])
        field = clearance_field(
            masked_depth,
            intrinsics,
            plane_override=fitted,
            confidence_map=confidence_map,
        )
        role[index] = str(record["role"])
        frame_ids.append(str(record["frame_id"]))
        depth_valid_fraction[index] = float(valid.mean())
        if field.get("status") != "VALID":
            continue
        height = field.get("camera_height_m")
        if height is None or not np.isfinite(float(height)):
            continue
        if fitted is None:
            raise RuntimeError(f"clearance/plane determinism mismatch: {record.get('frame_id')}")
        normal, offset, _ = fitted
        plane[index] = np.asarray([*normal, offset], dtype=np.float32)
        camera_height[index] = float(height)
        ground_valid[index] = 1
        for band_index, name in enumerate(BANDS):
            band = field["bands"][name]
            value = band.get("clearance_m")
            state = band.get("occupied_by_horizon", {}).get(OCCUPANCY_HORIZON_M)
            if value is not None and np.isfinite(float(value)) and state is not None:
                clearance[index, band_index] = float(value)
                occupancy[index, band_index] = int(bool(state))
                clearance_valid[index, band_index] = 1

    output_root.mkdir(parents=True)
    target_path = output_root / "geometry_targets.npz"
    np.savez_compressed(
        target_path, plane=plane, camera_height_m=camera_height, clearance_m=clearance,
        occupancy_1_5m=occupancy, ground_valid=ground_valid,
        clearance_valid=clearance_valid, depth_valid_fraction=depth_valid_fraction,
        role=role, frame_id=np.asarray(frame_ids, dtype="U64"),
    )
    per_role: dict[str, Any] = {}
    for name in ("train", "validation"):
        selected = role == name
        per_role[name] = {
            "record_count": int(selected.sum()),
            "mean_confidence_2_valid_depth_fraction": float(depth_valid_fraction[selected].mean()) if selected.any() else 0.0,
            "ground_plane_normal_known": coverage(ground_valid[selected] == 1),
            "camera_height_known": coverage(np.isfinite(camera_height[selected])),
            "clearance_known_by_band": {band: coverage(clearance_valid[selected, i] == 1) for i, band in enumerate(BANDS)},
            "occupancy_known_by_band": {band: coverage(occupancy[selected, i] >= 0) for i, band in enumerate(BANDS)},
        }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "terminal": "S1_1_GEOMETRY_TARGET_PREFLIGHT_COMPLETE_DEVELOPMENT_ONLY",
        "protocol_sha256": sha256_file(protocol_path),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "target_path": str(target_path.resolve()),
        "target_sha256": sha256_file(target_path),
        "record_count": count,
        "consumed_120_frame_cohort_opened": False,
        "coverage": {
            "ground_plane": coverage(ground_valid == 1)["known_fraction"],
            "camera_height": coverage(np.isfinite(camera_height))["known_fraction"],
            "clearance_any_band": coverage(np.any(clearance_valid == 1, axis=1))["known_fraction"],
        },
        "roles": per_role,
        "target_contract": {
            "truth_mask": "confidence_equals_2_and_depth_0.25_to_6.0_m",
            "ground_plane": "depth_ransac_on_masked_metric_depth",
            "plane_layout": "normal_x_normal_y_normal_z_offset_m",
            "occupancy_horizon_m": 1.5,
            "unknown_occupancy_value": -1,
            "consumed_120_frame_gate_read": False,
            "training_performed": False,
            "qnn_qat_android_performed": False,
        },
    }
    (output_root / "preflight_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    materialize(args.protocol.resolve(), args.teacher_manifest.resolve(), args.output_root.resolve(), args.limit)


if __name__ == "__main__":
    main()
