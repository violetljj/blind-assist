#!/usr/bin/env python3
"""Validate causal video/clock/frame/route binding for one USTRF episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    pass


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local(root: Path, relative: str, *, where: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{where} escapes manifest root") from error
    if not path.is_file():
        raise ContractError(f"{where} is not a local file")
    return path


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _hash_bound(row: dict[str, Any], root: Path, stem: str, *, where: str) -> tuple[dict[str, Any], str]:
    relative = _text(row.get(f"{stem}_path"), where=f"{where}.{stem}_path")
    expected = row.get(f"{stem}_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ContractError(f"{where}.{stem}_sha256 must be a lowercase SHA-256")
    path = _local(root, relative, where=f"{where}.{stem}_path")
    if _sha(path) != expected:
        raise ContractError(f"{where}.{stem} SHA-256 mismatch")
    return _json(path), expected


def validate_episode_binding(
    row: dict[str, Any], *, root: Path, policy: dict[str, Any], endpoint_tolerance_ms: int, where: str,
) -> dict[str, Any]:
    """Recompute the atomic frame clock and causal route bindings."""
    context = row.get("capture_context")
    if not isinstance(context, dict):
        raise ContractError(f"{where}.capture_context must be an object")
    camera_frame = _text(context.get("camera_frame"), where=f"{where}.capture_context.camera_frame")
    calibration_id = _text(context.get("calibration_id"), where=f"{where}.capture_context.calibration_id")

    ledger, ledger_sha = _hash_bound(row, root, "capture_frame_ledger", where=where)
    if ledger.get("schema") != policy.get("ledger_schema", "blindassist_capture_frame_ledger_v1"):
        raise ContractError(f"{where}.capture frame ledger has unexpected schema")
    if ledger.get("episode_id") != row.get("episode_id") or ledger.get("source_video_sha256") != row.get("video_sha256"):
        raise ContractError(f"{where}.capture frame ledger episode/video binding mismatch")
    if ledger.get("camera_frame") != camera_frame:
        raise ContractError(f"{where}.capture frame ledger camera_frame mismatch")
    receipt_id = _text(ledger.get("clock_receipt_id"), where=f"{where}.capture_frame_ledger.clock_receipt_id")
    frames = ledger.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"{where}.capture frame ledger frames must be non-empty")
    ids: list[str] = []
    timestamps: list[int] = []
    pts_values: list[int] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        frame_where = f"{where}.capture_frame_ledger.frames[{index}]"
        if not isinstance(frame, dict) or frame.get("frame_index") != index:
            raise ContractError(f"{frame_where}.frame_index must be contiguous from zero")
        frame_id = _text(frame.get("frame_id"), where=f"{frame_where}.frame_id")
        timestamp = frame.get("capture_timestamp_ns")
        pts_ms = frame.get("video_pts_ms")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ContractError(f"{frame_where}.capture_timestamp_ns must be a non-negative integer")
        if not isinstance(pts_ms, int) or isinstance(pts_ms, bool) or pts_ms < 0 or pts_ms > row.get("duration_ms", -1):
            raise ContractError(f"{frame_where}.video_pts_ms must be an in-range integer")
        if frame.get("episode_time_ms") != pts_ms:
            raise ContractError(f"{frame_where}.episode_time_ms must equal video_pts_ms")
        if frame.get("camera_frame") != camera_frame or frame.get("dropped") is not False or frame.get("duplicate") is not False:
            raise ContractError(f"{frame_where} must be a decoded, unique frame in the configured camera frame")
        payload = frame.get("frame_payload_sha256")
        if not isinstance(payload, str) or len(payload) != 64 or any(char not in "0123456789abcdef" for char in payload):
            raise ContractError(f"{frame_where}.frame_payload_sha256 must be a lowercase SHA-256")
        ids.append(frame_id)
        timestamps.append(timestamp)
        pts_values.append(pts_ms)
        by_id[frame_id] = frame
    if len(ids) != len(set(ids)):
        raise ContractError(f"{where}.capture frame IDs must be unique")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ContractError(f"{where}.capture timestamps must be strictly increasing")
    if any(current <= previous for previous, current in zip(pts_values, pts_values[1:])):
        raise ContractError(f"{where}.video PTS values must be strictly increasing")
    if pts_values[0] > endpoint_tolerance_ms or pts_values[-1] < row["duration_ms"] - endpoint_tolerance_ms:
        raise ContractError(f"{where}.capture frame ledger does not cover both episode endpoints")
    capture_gaps = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
    maximum_gap_ns = max(capture_gaps, default=0)
    allowed_gap_ns = policy.get("maximum_capture_gap_ns")
    if not isinstance(allowed_gap_ns, int) or allowed_gap_ns <= 0 or maximum_gap_ns > allowed_gap_ns:
        raise ContractError(f"{where}.capture frame gap exceeds policy")
    alignment_tolerance_ns = policy.get("maximum_clock_pts_alignment_error_ns")
    if not isinstance(alignment_tolerance_ns, int) or alignment_tolerance_ns < 0:
        raise ContractError("capture clock/PTS alignment tolerance is invalid")
    for timestamp, pts_ms in zip(timestamps, pts_values):
        capture_delta_ns = timestamp - timestamps[0]
        pts_delta_ns = (pts_ms - pts_values[0]) * 1_000_000
        if abs(capture_delta_ns - pts_delta_ns) > alignment_tolerance_ns:
            raise ContractError(f"{where}.capture clock and video PTS drift above policy")

    clock, _ = _hash_bound(row, root, "capture_clock_receipt", where=where)
    expected_clock = {
        "schema": policy.get("clock_receipt_schema", "blindassist_capture_clock_receipt_v1"),
        "receipt_id": receipt_id,
        "episode_id": row.get("episode_id"),
        "video_sha256": row.get("video_sha256"),
        "capture_frame_ledger_sha256": ledger_sha,
        "camera_frame": camera_frame,
        "timestamp_unit": "nanoseconds",
        "timestamps_strictly_monotonic": True,
        "frame_count": len(frames),
        "decoded_frame_count": len(frames),
        "first_capture_timestamp_ns": timestamps[0],
        "last_capture_timestamp_ns": timestamps[-1],
        "first_video_pts_ms": pts_values[0],
        "last_video_pts_ms": pts_values[-1],
        "maximum_capture_gap_ns": maximum_gap_ns,
    }
    for key, expected in expected_clock.items():
        if clock.get(key) != expected:
            raise ContractError(f"{where}.capture clock receipt {key} does not match recomputed ledger evidence")
    _text(clock.get("clock_domain"), where=f"{where}.capture_clock_receipt.clock_domain")
    _text(clock.get("clock_source"), where=f"{where}.capture_clock_receipt.clock_source")

    route, _ = _hash_bound(row, root, "route_intent", where=where)
    if route.get("capture_frame_ledger_sha256") != ledger_sha or route.get("input_video_sha256") != row.get("video_sha256"):
        raise ContractError(f"{where}.route intent is not bound to frame ledger and input video")
    if route.get("camera_frame") != camera_frame or route.get("calibration_id") != calibration_id:
        raise ContractError(f"{where}.route intent camera/calibration binding mismatch")
    route_plan_id = _text(route.get("route_plan_id"), where=f"{where}.route_intent.route_plan_id")
    provider = route.get("provider")
    if not isinstance(provider, dict):
        raise ContractError(f"{where}.route_intent.provider must be an object")
    for key in ("implementation_sha256", "config_sha256"):
        value = provider.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ContractError(f"{where}.route_intent.provider.{key} must be a lowercase SHA-256")
    coordinate = route.get("coordinate_contract")
    if not isinstance(coordinate, dict):
        raise ContractError(f"{where}.route_intent.coordinate_contract must be an object")
    projection_path = _text(coordinate.get("projection_receipt_path"), where=f"{where}.route_intent.coordinate_contract.projection_receipt_path")
    projection_sha = coordinate.get("projection_receipt_sha256")
    if not isinstance(projection_sha, str) or len(projection_sha) != 64 or any(char not in "0123456789abcdef" for char in projection_sha):
        raise ContractError(f"{where}.projection receipt SHA must be lowercase SHA-256")
    projection_file = _local(root, projection_path, where=f"{where}.projection_receipt_path")
    if _sha(projection_file) != projection_sha:
        raise ContractError(f"{where}.projection receipt SHA-256 mismatch")
    projection = _json(projection_file)
    for key, expected in {
        "schema": "blindassist_route_projection_receipt_v1",
        "episode_id": row.get("episode_id"),
        "input_video_sha256": row.get("video_sha256"),
        "capture_frame_ledger_sha256": ledger_sha,
        "camera_frame": camera_frame,
        "calibration_id": calibration_id,
    }.items():
        if projection.get(key) != expected:
            raise ContractError(f"{where}.projection receipt {key} mismatch")

    samples = route.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError(f"{where}.route samples must be non-empty")
    for index, sample in enumerate(samples):
        sample_where = f"{where}.route_intent.samples[{index}]"
        if not isinstance(sample, dict):
            raise ContractError(f"{sample_where} must be an object")
        source_id = _text(sample.get("source_frame_id"), where=f"{sample_where}.source_frame_id")
        consuming_id = _text(sample.get("consuming_frame_id"), where=f"{sample_where}.consuming_frame_id")
        if source_id not in by_id or consuming_id not in by_id:
            raise ContractError(f"{sample_where} references a frame outside the ledger")
        source_ts = by_id[source_id]["capture_timestamp_ns"]
        consuming_ts = by_id[consuming_id]["capture_timestamp_ns"]
        generated_ts = sample.get("generated_at_timestamp_ns")
        if sample.get("source_capture_timestamp_ns") != source_ts:
            raise ContractError(f"{sample_where}.source_capture_timestamp_ns mismatch")
        if not isinstance(generated_ts, int) or isinstance(generated_ts, bool) or not source_ts <= generated_ts <= consuming_ts:
            raise ContractError(f"{sample_where} violates causal source/generated/consuming order")
        if sample.get("timestamp_ms") != by_id[consuming_id]["episode_time_ms"]:
            raise ContractError(f"{sample_where}.timestamp_ms is not bound to consuming frame")
    return {
        "frame_count": len(frames),
        "route_plan_id": route_plan_id,
        "provider_policy": {
            key: provider.get(key) for key in ("type", "provider_id", "implementation_sha256", "config_sha256", "input_space")
        },
        "route_choice": context.get("route_choice"),
    }
