"""Fail-closed contract for fresh AtomS3R-M12 RGB/BMI270 G0 captures."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "blindassist.vitg_g0.capture_manifest.v1"
CAMERA_SCHEMA = "blindassist.vitg_g0.camera_slot.v1"
IMU_SCHEMA = "blindassist.vitg_g0.imu_sample.v1"
PROTOCOL_SCHEMA = "blindassist.vitg_g0.protocol.v1"
SURFACE_STRATA = {
    "matte_light_hard_floor",
    "dark_low_reflectance_floor_or_mat",
    "textured_or_carpet_floor",
    "specular_or_bright_tile_floor",
}
EPISODE_TYPES = {
    "EXCITED_WALK_TURN",
    "NATURAL_HEAD_MOTION",
    "STRAIGHT_LOW_EXCITATION_CONTROL",
    "STATIC_CONTROL",
    "POSTURE_HEIGHT_CHANGE",
    "STAIRS_OR_RAMP",
    "LOW_TEXTURE_OR_DYNAMIC_OCCLUDER",
}
REQUIRED_BINDINGS = {
    "firmware_sha256",
    "camera_intrinsics_sha256",
    "imu_to_camera_extrinsics_sha256",
    "camera_imu_clock_validation_sha256",
    "height_reference_instrument_id",
    "height_reference_calibration_sha256",
    "capture_writer_sha256",
    "truth_writer_sha256",
    "artifact_root",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"schema mismatch: {path}")
    return value


def _load_jsonl(path: Path, schema: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != schema:
            raise ValueError(f"{path}:{line_number}: schema mismatch")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty VITG G0 stream: {path}")
    return rows


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_roster(manifest: dict[str, Any]) -> None:
    parents = manifest.get("frozen_parent_records", [])
    if not isinstance(parents, list) or len(parents) != 8:
        raise ValueError("VITG G0 requires exactly eight parent records")
    parent_ids = [str(row.get("parent_id", "")) for row in parents]
    if any(not value for value in parent_ids) or len(set(parent_ids)) != 8:
        raise ValueError("VITG G0 parent IDs must be nonempty and unique")
    sites = {str(row.get("site_id", "")) for row in parents if str(row.get("site_id", ""))}
    if len(sites) < 2:
        raise ValueError("VITG G0 requires at least two physical sites")
    strata = Counter(str(row.get("surface_stratum", "")) for row in parents)
    if set(strata) != SURFACE_STRATA or any(strata[name] != 2 for name in SURFACE_STRATA):
        raise ValueError("VITG G0 requires exactly two parents in every frozen surface stratum")
    episodes = manifest.get("frozen_episode_records", [])
    if not isinstance(episodes, list) or len(episodes) != len(EPISODE_TYPES):
        raise ValueError("VITG G0 episode roster is incomplete")
    episode_ids = [str(row.get("episode_id", "")) for row in episodes]
    if any(not value for value in episode_ids) or len(set(episode_ids)) != len(EPISODE_TYPES):
        raise ValueError("VITG G0 episode IDs must be nonempty and unique")
    if {str(row.get("episode_type", "")) for row in episodes} != EPISODE_TYPES:
        raise ValueError("VITG G0 episode types drifted")
    if any(float(row.get("duration_seconds", 0.0)) != 20.0 for row in episodes):
        raise ValueError("VITG G0 freezes every episode at 20 seconds")


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("parent_id", "")), str(row.get("episode_id", ""))


def _assert_no_range_fields(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        if any("tof" in str(key).lower() or "range_m" in str(key).lower() for key in row):
            raise ValueError("VITG G0 capture contains a forbidden ToF/range field")


def validate_camera_slots(
    rows: list[dict[str, Any]], expected_pairs: set[tuple[str, str]], clock_domain: str
) -> dict[tuple[str, str], list[int]]:
    _assert_no_range_fields(rows)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = _identity(row)
        if pair not in expected_pairs:
            raise ValueError("camera slot lies outside the frozen roster")
        if row.get("clock_domain") != clock_domain:
            raise ValueError("camera slot clock-domain drift")
        sequence = row.get("frame_sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError("camera frame sequence must be a nonnegative integer")
        if row.get("status") not in {"VALID_JPEG", "LOST_BEFORE_WRITER", "INVALID_JPEG"}:
            raise ValueError("unsupported camera slot status")
        timestamp = row.get("capture_timestamp_us")
        if row["status"] == "VALID_JPEG":
            if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
                raise ValueError("valid camera slot requires a device capture timestamp")
            if not isinstance(row.get("frame_id"), str) or not row["frame_id"]:
                raise ValueError("valid camera slot requires a frame ID")
        elif timestamp is not None and (not isinstance(timestamp, int) or isinstance(timestamp, bool)):
            raise ValueError("invalid camera slot timestamp")
        by_pair[pair].append(row)
    if set(by_pair) != expected_pairs:
        raise ValueError("camera stream does not cover every frozen parent/episode")
    valid_timestamps: dict[tuple[str, str], list[int]] = {}
    for pair, pair_rows in by_pair.items():
        pair_rows.sort(key=lambda row: int(row["frame_sequence"]))
        sequences = [int(row["frame_sequence"]) for row in pair_rows]
        if sequences != list(range(sequences[0], sequences[-1] + 1)):
            raise ValueError("camera frame loss was deleted instead of materialized as a slot")
        timestamps = [int(row["capture_timestamp_us"]) for row in pair_rows if row["status"] == "VALID_JPEG"]
        if len(timestamps) < 250 or len(timestamps) > 350:
            raise ValueError("camera valid-frame count is outside the frozen 20 s admission range")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("camera timestamps are not strictly increasing")
        duration = timestamps[-1] - timestamps[0]
        if not 18_000_000 <= duration <= 22_000_000:
            raise ValueError("camera episode duration is outside the frozen 20 s window")
        valid_timestamps[pair] = timestamps
    return valid_timestamps


def validate_imu_samples(
    rows: list[dict[str, Any]],
    expected_pairs: set[tuple[str, str]],
    clock_domain: str,
    camera_timestamps: dict[tuple[str, str], list[int]],
) -> None:
    _assert_no_range_fields(rows)
    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        pair = _identity(row)
        if pair not in expected_pairs:
            raise ValueError("IMU sample lies outside the frozen roster")
        if row.get("clock_domain") != clock_domain:
            raise ValueError("IMU sample clock-domain drift")
        timestamp = row.get("timestamp_us")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ValueError("IMU timestamp must be a nonnegative integer")
        _finite_vector(row.get("accelerometer_mps2"), 3, "accelerometer_mps2")
        _finite_vector(row.get("gyroscope_rad_s"), 3, "gyroscope_rad_s")
        by_pair[pair].append(timestamp)
    if set(by_pair) != expected_pairs:
        raise ValueError("IMU stream does not cover every frozen parent/episode")
    for pair, timestamps in by_pair.items():
        timestamps.sort()
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("duplicate IMU timestamps")
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if not gaps or max(gaps) > 20_000:
            raise ValueError("IMU stream has a gap above 20 ms")
        duration_s = (timestamps[-1] - timestamps[0]) / 1_000_000.0
        rate_hz = (len(timestamps) - 1) / duration_s if duration_s > 0 else 0.0
        if not 100.0 <= rate_hz <= 400.0:
            raise ValueError("IMU effective rate lies outside 100-400 Hz")
        index = 0
        for camera_timestamp in camera_timestamps[pair]:
            while index + 1 < len(timestamps) and timestamps[index + 1] < camera_timestamp:
                index += 1
            if index + 1 >= len(timestamps):
                raise ValueError("camera frame lacks a following IMU bracket")
            left, right = timestamps[index], timestamps[index + 1]
            if not left <= camera_timestamp <= right:
                raise ValueError("camera frame is outside its IMU bracket")
            if right - left > 10_000 or min(camera_timestamp - left, right - camera_timestamp) > 5_000:
                raise ValueError("camera/IMU synchronization exceeds the frozen 5 ms nearest-sample gate")


def validate_capture(manifest_path: Path, protocol_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path, MANIFEST_SCHEMA)
    protocol = _load_json(protocol_path, PROTOCOL_SCHEMA)
    if str(manifest.get("protocol_sha256", "")).upper() != sha256_file(protocol_path):
        raise ValueError("capture manifest is not bound to the supplied frozen protocol")
    if manifest.get("evidence_role") != "FRESH_PHYSICAL_VITG_G0":
        raise ValueError("VITG G0 requires fresh physical evidence")
    suite = manifest.get("sensor_suite", {})
    required_suite = {
        "device": "M5STACK_ATOMS3R_M12",
        "camera": "OV3660",
        "imu": "BMI270",
        "same_rigid_body": True,
        "same_device_clock": True,
        "external_tof_present": False,
        "phone_imu_used": False,
    }
    if any(suite.get(key) != value for key, value in required_suite.items()):
        raise ValueError("VITG G0 sensor-suite identity or no-ToF boundary mismatch")
    bindings = manifest.get("bindings", {})
    if not isinstance(bindings, dict) or any(not str(bindings.get(key, "")).strip() for key in REQUIRED_BINDINGS):
        raise ValueError("VITG G0 calibration/writer bindings are incomplete")
    validate_roster(manifest)
    parents = [str(row["parent_id"]) for row in manifest["frozen_parent_records"]]
    episodes = [str(row["episode_id"]) for row in manifest["frozen_episode_records"]]
    expected_pairs = {(parent, episode) for parent in parents for episode in episodes}
    streams = manifest.get("streams", {})
    camera_identity, imu_identity = streams.get("camera"), streams.get("imu")
    if not isinstance(camera_identity, dict) or not isinstance(imu_identity, dict):
        raise ValueError("VITG G0 camera and IMU streams are required")
    camera_path = (manifest_path.parent / str(camera_identity.get("path", ""))).resolve()
    imu_path = (manifest_path.parent / str(imu_identity.get("path", ""))).resolve()
    for path, identity in ((camera_path, camera_identity), (imu_path, imu_identity)):
        if sha256_file(path) != str(identity.get("sha256", "")).upper():
            raise ValueError(f"VITG G0 stream hash mismatch: {path}")
    clock_domain = str(manifest.get("device_clock_domain", ""))
    if not clock_domain.startswith("esp32_boot_monotonic:"):
        raise ValueError("VITG G0 requires the AtomS3R boot-monotonic device clock")
    camera_rows = _load_jsonl(camera_path, CAMERA_SCHEMA)
    imu_rows = _load_jsonl(imu_path, IMU_SCHEMA)
    camera_timestamps = validate_camera_slots(camera_rows, expected_pairs, clock_domain)
    validate_imu_samples(imu_rows, expected_pairs, clock_domain, camera_timestamps)
    return {
        "schema": "blindassist.vitg_g0.capture_validation.v1",
        "status": "VITG_G0_RGB_IMU_CAPTURE_CONTRACT_VALID",
        "parents": len(parents),
        "episodes_per_parent": len(episodes),
        "camera_slots": len(camera_rows),
        "imu_samples": len(imu_rows),
        "outcome_access_authorized": False,
        "claim_ceiling": "synchronized fresh physical RGB-IMU source admission only; no VIO, metric height, clearance, safety, or product result",
    }
