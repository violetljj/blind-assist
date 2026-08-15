#!/usr/bin/env python3
"""Run GA-SATOM G0 from separately bound measurement and evaluator-truth streams."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import GroundAnchorPolicy, MeasurementFrame, TruthFrame, ZoneMeasurement, evaluate_g0


MANIFEST_SCHEMA = "blindassist.ga_satom_g0.manifest.v1"
PROTOCOL_SCHEMA = "blindassist.ga_satom_g0.protocol.v1"
ACTIVATION_SCHEMA = "blindassist.ga_satom_g0.activation.v1"
MEASUREMENT_SCHEMA = "blindassist.ga_satom_g0.measurement.v1"
TRUTH_SCHEMA = "blindassist.ga_satom_g0.truth.v1"
SURFACE_STRATA = {
    "matte_light_hard_floor",
    "dark_low_reflectance_floor_or_mat",
    "textured_or_carpet_floor",
    "specular_or_bright_tile_floor",
}
EXPECTED_EPISODE_GRID = {
    (height_m, pitch_degrees)
    for height_m in (1.2, 1.5, 1.8)
    for pitch_degrees in (-5.0, 0.0, 5.0)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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
        raise ValueError(f"empty G0 stream: {path}")
    return rows


def validate_expected_schedule(
    measurements: list[MeasurementFrame],
    truth: list[TruthFrame],
    frozen_parent_ids: list[str],
    frozen_episode_ids: list[str],
    frozen_frame_ids: list[str],
) -> None:
    if len(frozen_parent_ids) != 8 or len(set(frozen_parent_ids)) != 8:
        raise ValueError("G0 requires exactly eight unique frozen parents")
    if len(frozen_episode_ids) != 9 or len(set(frozen_episode_ids)) != 9:
        raise ValueError("G0 requires exactly nine unique frozen episodes per parent")
    if not frozen_frame_ids or len(frozen_frame_ids) != len(set(frozen_frame_ids)):
        raise ValueError("G0 requires unique frozen frame-slot IDs")
    expected_keys = {
        (parent_id, episode_id, frame_id)
        for parent_id in frozen_parent_ids
        for episode_id in frozen_episode_ids
        for frame_id in frozen_frame_ids
    }
    measurement_keys = [(row.parent_id, row.episode_id, row.frame_id) for row in measurements]
    truth_keys = [(row.parent_id, row.episode_id, row.frame_id) for row in truth]
    if len(measurement_keys) != len(set(measurement_keys)):
        raise ValueError("duplicate G0 measurement identity")
    if len(truth_keys) != len(set(truth_keys)):
        raise ValueError("duplicate G0 truth identity")
    if set(measurement_keys) != set(truth_keys):
        raise ValueError("G0 measurement/truth identity ledger mismatch")
    if set(measurement_keys) != expected_keys:
        raise ValueError(
            "G0 requires every frozen time slot; capture loss must be materialized as an INVALID 64-zone frame"
        )
    frame_order = {frame_id: index for index, frame_id in enumerate(frozen_frame_ids)}
    by_pair: dict[tuple[str, str], list[MeasurementFrame]] = {}
    for frame in measurements:
        by_pair.setdefault((frame.parent_id, frame.episode_id), []).append(frame)
    for pair_frames in by_pair.values():
        pair_frames.sort(key=lambda frame: frame_order[frame.frame_id])
        timestamps = [frame.timestamp_ns for frame in pair_frames]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("G0 host-monotonic timestamps must follow the frozen frame-slot order")


def _load_bound_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"schema mismatch: {path}")
    return value


def validate_activation_roster(activation: dict[str, Any]) -> None:
    parent_ids = [str(value) for value in activation.get("frozen_parent_ids", [])]
    episode_ids = [str(value) for value in activation.get("frozen_episode_ids", [])]
    records = activation.get("parent_records", [])
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("G0 activation requires eight parent records")
    if {str(record.get("parent_id", "")) for record in records} != set(parent_ids):
        raise ValueError("G0 activation parent records do not match the frozen parent IDs")
    sites = {str(record.get("site_id", "")) for record in records if str(record.get("site_id", ""))}
    if len(sites) < 2:
        raise ValueError("G0 activation requires at least two physical sites")
    strata = Counter(str(record.get("surface_stratum", "")) for record in records)
    if set(strata) != SURFACE_STRATA or any(strata[name] < 2 for name in SURFACE_STRATA):
        raise ValueError("G0 activation does not satisfy the frozen surface strata")
    if any(str(record.get("occluder_episode_id", "")) not in set(episode_ids) for record in records):
        raise ValueError("G0 activation must predeclare one occluder episode for every parent")
    episode_records = activation.get("episode_records", [])
    if not isinstance(episode_records, list) or len(episode_records) != 9:
        raise ValueError("G0 activation requires nine episode records")
    if {str(record.get("episode_id", "")) for record in episode_records} != set(episode_ids):
        raise ValueError("G0 activation episode records do not match the frozen episode IDs")
    grid = {
        (float(record.get("reference_rgb_camera_height_m")), float(record.get("rig_pitch_degrees")))
        for record in episode_records
    }
    if grid != EXPECTED_EPISODE_GRID:
        raise ValueError("G0 activation episode grid drift")


def load_manifest(
    path: Path,
    protocol_path: Path,
    activation_path: Path,
) -> tuple[list[MeasurementFrame], list[TruthFrame]]:
    protocol = _load_bound_json(protocol_path, PROTOCOL_SCHEMA)
    activation = _load_bound_json(activation_path, ACTIVATION_SCHEMA)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("G0 manifest schema mismatch")
    if manifest.get("evidence_role") != "FRESH_PHYSICAL_G0":
        raise ValueError("G0 requires a fresh physical evidence role")
    if manifest.get("sensor_family") != "ST_VL53L8CX_8X8" or manifest.get("simulated") is not False:
        raise ValueError("G0 requires the frozen physical VL53L8CX source")
    if manifest.get("arm_evaluation_authorized") is not False or manifest.get("training_authorized") is not False:
        raise ValueError("G0 cannot authorize SATOM arms or training")
    protocol_hash = sha256_file(protocol_path)
    activation_hash = sha256_file(activation_path)
    if str(manifest.get("protocol_sha256", "")).upper() != protocol_hash:
        raise ValueError("G0 manifest is not bound to the supplied frozen protocol")
    if str(manifest.get("activation_receipt_sha256", "")).upper() != activation_hash:
        raise ValueError("G0 manifest is not bound to the supplied activation receipt")
    if str(activation.get("protocol_sha256", "")).upper() != protocol_hash:
        raise ValueError("G0 activation receipt protocol hash mismatch")
    if activation.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("G0 activation receipt protocol identity mismatch")
    if activation.get("status") != "ACTIVATED_FOR_FRESH_PHYSICAL_CAPTURE":
        raise ValueError("G0 physical capture was not activated")
    if activation.get("outcome_access_authorized") is not True:
        raise ValueError("G0 activation does not authorize outcome access")
    validate_activation_roster(activation)
    required_bindings = {
        "device_serial", "firmware_sha256", "source_sha256", "rgb_intrinsics_sha256",
        "tof_to_rgb_registration_sha256", "imu_to_rgb_registration_sha256",
        "reference_instrument_id", "calibration_receipt_sha256",
        "measurement_writer_sha256", "truth_writer_sha256", "artifact_root",
    }
    bindings = activation.get("bindings", {})
    if not isinstance(bindings, dict) or any(not str(bindings.get(key, "")).strip() for key in required_bindings):
        raise ValueError("G0 activation receipt has incomplete physical/calibration/writer bindings")
    streams = manifest["streams"]
    if manifest.get("artifact_root") != bindings["artifact_root"]:
        raise ValueError("G0 manifest artifact root drift from activation")
    if str(streams["measurements"].get("writer_sha256", "")).upper() != str(bindings["measurement_writer_sha256"]).upper():
        raise ValueError("G0 measurement writer drift from activation")
    if str(streams["truth"].get("writer_sha256", "")).upper() != str(bindings["truth_writer_sha256"]).upper():
        raise ValueError("G0 truth writer drift from activation")
    measurement_path = (path.parent / streams["measurements"]["path"]).resolve()
    truth_path = (path.parent / streams["truth"]["path"]).resolve()
    for stream_path, identity in (
        (measurement_path, streams["measurements"]), (truth_path, streams["truth"]),
    ):
        if sha256_file(stream_path) != str(identity["sha256"]).upper():
            raise ValueError(f"G0 stream hash mismatch: {stream_path}")
    measurements = []
    for row in _load_jsonl(measurement_path, MEASUREMENT_SCHEMA):
        measurements.append(
            MeasurementFrame(
                parent_id=str(row["parent_id"]), episode_id=str(row["episode_id"]),
                frame_id=str(row["frame_id"]), timestamp_ns=int(row["timestamp_ns"]),
                gravity_down_rgb_unit=np.asarray(row["gravity_down_rgb_unit"], dtype=np.float64),
                zones=tuple(
                    ZoneMeasurement(
                        zone_id=str(zone["zone_id"]),
                        origin_rgb_m=np.asarray(zone["origin_rgb_m"], dtype=np.float64),
                        ray_rgb_unit=np.asarray(zone["ray_rgb_unit"], dtype=np.float64),
                        range_m=None if zone.get("range_m") is None else float(zone["range_m"]),
                        sigma_m=None if zone.get("sigma_m") is None else float(zone["sigma_m"]),
                        status=str(zone["status"]),
                    )
                    for zone in row["zones"]
                ),
            )
        )
    truth = [
        TruthFrame(
            parent_id=str(row["parent_id"]), episode_id=str(row["episode_id"]),
            frame_id=str(row["frame_id"]),
            reference_rgb_camera_height_m=float(row["reference_rgb_camera_height_m"]),
            reference_height_uncertainty_m=float(row["reference_height_uncertainty_m"]),
            ground_labels={str(key): str(value) for key, value in row["ground_labels"].items()},
        )
        for row in _load_jsonl(truth_path, TRUTH_SCHEMA)
    ]
    expected_parents = [str(value) for value in activation["frozen_parent_ids"]]
    expected_episodes = [str(value) for value in activation["frozen_episode_ids"]]
    if manifest.get("frozen_parent_ids") != activation.get("frozen_parent_ids"):
        raise ValueError("G0 manifest parent roster drift from activation")
    if manifest.get("frozen_episode_ids") != activation.get("frozen_episode_ids"):
        raise ValueError("G0 manifest episode roster drift from activation")
    expected_frames = int(activation.get("expected_frames_per_episode", -1))
    if expected_frames != 300:
        raise ValueError("G0 activation must retain 300 registered time slots per episode")
    if manifest.get("expected_frames_per_episode") != expected_frames:
        raise ValueError("G0 manifest time-slot count drift from activation")
    expected_frame_ids = [str(value) for value in activation.get("frozen_frame_ids", [])]
    if len(expected_frame_ids) != expected_frames:
        raise ValueError("G0 activation frame-slot identity ledger must contain exactly 300 IDs")
    if manifest.get("frozen_frame_ids") != activation.get("frozen_frame_ids"):
        raise ValueError("G0 manifest frame-slot identity drift from activation")
    validate_expected_schedule(measurements, truth, expected_parents, expected_episodes, expected_frame_ids)
    occluder_episode_by_parent = {
        str(record["parent_id"]): str(record["occluder_episode_id"])
        for record in activation["parent_records"]
    }
    for parent_id, episode_id in occluder_episode_by_parent.items():
        if not any(
            row.parent_id == parent_id
            and row.episode_id == episode_id
            and "NON_GROUND" in row.ground_labels.values()
            for row in truth
        ):
            raise ValueError("G0 predeclared occluder episode lacks evaluator-confirmed NON_GROUND anchor content")
    return measurements, truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurements, truth = load_manifest(
        args.manifest.resolve(), args.protocol.resolve(), args.activation_receipt.resolve()
    )
    result = evaluate_g0(measurements, truth, GroundAnchorPolicy())
    if args.output.exists():
        raise FileExistsError(f"G0 output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
