from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def qualify(receipt_path: Path, protocol_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    phase = receipt.get("phase")
    phase_contract = protocol.get("phases", {}).get(phase)
    if phase_contract is None:
        failures.append("UNKNOWN_PHASE")
        phase_contract = {}
    if receipt.get("protocol_id") != protocol.get("protocol_id"):
        failures.append("PROTOCOL_ID_MISMATCH")
    if receipt.get("model_id") != protocol.get("model_id"):
        failures.append("MODEL_ID_MISMATCH")

    common = protocol["common"]
    height = receipt.get("camera_height_m")
    uncertainty = receipt.get("camera_height_uncertainty_m")
    height_range = common["camera_height_range_m"]
    if not isinstance(height, (int, float)) or not math.isfinite(height) or not height_range[0] <= height <= height_range[1]:
        failures.append("INVALID_MEASURED_CAMERA_HEIGHT")
    if not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty < 0 or uncertainty > common["maximum_camera_height_uncertainty_m"]:
        failures.append("INVALID_CAMERA_HEIGHT_UNCERTAINTY")
    for field in ("session_id", "device_serial", "camera_id", "mount_profile_id", "intrinsics_sha256"):
        if not receipt.get(field):
            failures.append(f"MISSING_{field.upper()}")

    base = receipt_path.parent
    manifest_value = receipt.get("frame_manifest")
    manifest_path = (base / manifest_value).resolve() if isinstance(manifest_value, str) else None
    rows: list[dict[str, Any]] = []
    if manifest_path is None or not manifest_path.is_file():
        failures.append("MISSING_FRAME_MANIFEST")
    else:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            failures.append("INVALID_FRAME_MANIFEST")
            rows = []
    minimum = phase_contract.get("minimum_admitted_frames_per_session", phase_contract.get("fixed_anchors_per_session", 0))
    if len(rows) < minimum:
        failures.append("INSUFFICIENT_FRAMES_OR_ANCHORS")
    timestamps: list[int] = []
    for index, row in enumerate(rows):
        timestamp = row.get("capture_timestamp_ns")
        if not isinstance(timestamp, int):
            failures.append(f"FRAME_{index}_INVALID_TIMESTAMP")
        else:
            timestamps.append(timestamp)
        image_value = row.get("rgb_file")
        image_path = (base / image_value).resolve() if isinstance(image_value, str) else None
        if image_path is None or not image_path.is_file():
            failures.append(f"FRAME_{index}_MISSING_RGB")
        elif _sha256(image_path) != str(row.get("rgb_sha256", "")).upper():
            failures.append(f"FRAME_{index}_RGB_SHA256_MISMATCH")
    if len(timestamps) > 1 and any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        failures.append("NON_MONOTONIC_CAPTURE_TIMESTAMPS")

    reference_value = receipt.get("reference_manifest")
    reference_path = (base / reference_value).resolve() if isinstance(reference_value, str) else None
    if reference_path is None or not reference_path.is_file():
        failures.append("MISSING_INDEPENDENT_REFERENCE_MANIFEST")
    elif _sha256(reference_path) != str(receipt.get("reference_manifest_sha256", "")).upper():
        failures.append("REFERENCE_MANIFEST_SHA256_MISMATCH")

    unique_failures = list(dict.fromkeys(failures))
    return {
        "schema": "blindassist_known_height_phone_session_preflight_v1",
        "session_id": receipt.get("session_id"),
        "phase": phase,
        "status": "ADMITTED" if not unique_failures else "HOLD",
        "frame_or_anchor_count": len(rows),
        "failures": unique_failures,
        "authorization": {"offline_shadow_evaluation": not unique_failures, "app_runtime": False, "production": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify(args.receipt, args.protocol)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "ADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
