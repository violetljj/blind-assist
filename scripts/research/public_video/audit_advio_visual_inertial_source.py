#!/usr/bin/env python3
"""Create a hash-bound, fail-closed audit receipt for one ADVIO sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "blindassist_public_visual_inertial_acquisition_audit_v1"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path, columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"unexpected CSV shape for {path}: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite value in {path}")
    return values


def modality_summary(path: Path, values: np.ndarray) -> dict[str, Any]:
    timestamps = values[:, 0]
    deltas = np.diff(timestamps)
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_hash(path),
        "row_count": int(len(values)),
        "column_count": int(values.shape[1]),
        "first_timestamp_seconds": float(timestamps[0]),
        "last_timestamp_seconds": float(timestamps[-1]),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "timestamps_strictly_increasing": bool(np.all(deltas > 0)),
        "median_sample_hz": float(1.0 / np.median(deltas)),
    }


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".sha256").exists():
        raise ValueError("refusing to overwrite acquisition audit")
    contract = load_json(contract_path)
    registry_path = Path(contract["source_registry"]["path"])
    if file_hash(registry_path) != contract["source_registry"]["sha256"]:
        raise ValueError("source registry hash mismatch")
    registry = load_json(registry_path)
    archive_path = Path(contract["archive"]["path"])
    archive_size = archive_path.stat().st_size
    archive_md5 = file_hash(archive_path, "md5")
    archive_sha256 = file_hash(archive_path)
    archive_checks = {
        "actual_size_matches_receipt": archive_size == int(contract["archive"]["actual_size_bytes"]),
        "official_md5_matches": archive_md5 == contract["archive"]["official_md5"],
        "sha256_matches": archive_sha256 == contract["archive"]["sha256"],
    }
    root = Path(contract["extracted_root"])
    summaries: dict[str, dict[str, Any]] = {}
    loaded: dict[str, np.ndarray] = {}
    for name, spec in contract["modalities"].items():
        path = root / spec["path"]
        if not path.is_file():
            raise ValueError(f"missing modality: {path}")
        if "columns" in spec:
            values = load_csv(path, int(spec["columns"]))
            loaded[name] = values
            summaries[name] = modality_summary(path, values)
        else:
            summaries[name] = {
                "path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": file_hash(path)
            }
    timed = [summary for summary in summaries.values() if "first_timestamp_seconds" in summary]
    shared_start = max(item["first_timestamp_seconds"] for item in timed)
    shared_end = min(item["last_timestamp_seconds"] for item in timed)
    shared_duration = shared_end - shared_start
    pose = loaded["ground_truth_pose"]
    quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(pose[:, 4:8], axis=1) - 1.0)))
    xyz_std = np.std(pose[:, 1:4], axis=0)
    vertical_axis = int(np.argmin(xyz_std))
    checks_spec = contract["checks"]
    checks = {
        **archive_checks,
        "all_timestamps_strictly_increasing": all(
            item.get("timestamps_strictly_increasing", True) for item in summaries.values()
        ),
        "minimum_shared_duration": shared_duration >= float(checks_spec["minimum_shared_duration_seconds"]),
        "minimum_gyro_rate": summaries["iphone_gyro"]["median_sample_hz"] >= float(checks_spec["minimum_gyro_hz"]),
        "minimum_frame_timestamp_rate": summaries["iphone_frames"]["median_sample_hz"] >= float(checks_spec["minimum_frame_timestamp_hz"]),
        "minimum_ground_truth_rate": summaries["ground_truth_pose"]["median_sample_hz"] >= float(checks_spec["minimum_ground_truth_hz"]),
        "quaternion_norm": quaternion_norm_error <= float(checks_spec["maximum_quaternion_norm_error"]),
    }
    report = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": contract_path.as_posix(),
        "contract_sha256": file_hash(contract_path),
        "source_id": registry["source_id"],
        "license": registry["license"],
        "allowed_use": registry["allowed_use"],
        "archive": {"path": archive_path.as_posix(), "size_bytes": archive_size, "md5": archive_md5, "sha256": archive_sha256},
        "modalities": summaries,
        "synchronization": {"shared_start_seconds": shared_start, "shared_end_seconds": shared_end, "shared_duration_seconds": shared_duration},
        "ground_truth_geometry": {
            "translation_axis_std": xyz_std.tolist(),
            "inferred_vertical_axis_index": vertical_axis,
            "horizontal_axis_indices": [index for index in range(3) if index != vertical_axis],
            "maximum_quaternion_norm_error": quaternion_norm_error,
        },
        "checks": checks,
        "audit_passed": bool(all(checks.values())),
        "isolation": contract["isolation"],
        "limitations": [
            "The source is licensed CC BY-NC 4.0 and is isolated from commercial or production use.",
            "This receipt verifies integrity and synchronization, not obstacle risk or actionability truth.",
            "The inferred vertical axis is a sequence-level geometry diagnostic, not a device-frame convention claim."
        ],
    }
    if not report["audit_passed"]:
        raise ValueError(f"ADVIO audit failed: {checks}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = file_hash(output_path)
    Path(str(output_path) + ".sha256").write_text(f"{digest}  {output_path.name}\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.output)
    print(json.dumps({"audit_passed": report["audit_passed"], "source_id": report["source_id"]}))


if __name__ == "__main__":
    main()
