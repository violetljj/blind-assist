"""Fail-closed R1.1 target-ledger and per-frame projection contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    from .contract import load_json, require_false_flags, sha256_file
    from .projected_corridor_geometry import validate_polygon
except ImportError:
    from contract import load_json, require_false_flags, sha256_file
    from projected_corridor_geometry import validate_polygon


TARGET_LEDGER_SCHEMA = "blindassist_ustrf_crosscam_target_instance_ledger_v1"
PROJECTION_SCHEMA = "blindassist_ustrf_crosscam_frame_projection_receipt_v2"
ORACLE_SCHEMA = "blindassist_ustrf_crosscam_target_oracle_geometry_v1"
ANDROID_SCHEMA = "blindassist_ustrf_crosscam_target_aware_android_output_v2"
ATTRIBUTION_SCHEMA = "blindassist_ustrf_crosscam_r11_attribution_report_v1"
UNCERTAINTY_RATIOS = [0.01, 0.02, 0.03]
DIAGNOSTIC_ROLE = "seen_diagnostic_not_held_out"
HELD_OUT_UNSCORED_ROLE = "new_held_out_unscored"
ALLOWED_DATASET_ROLES = {DIAGNOSTIC_ROLE, HELD_OUT_UNSCORED_ROLE}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_pair(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 2, f"{label} must contain xy")
    result = [float(item) for item in value]
    require(all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result), f"{label} is outside normalized image space")
    return result


def _normalized_box(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 4, f"{label} must contain xyxy")
    result = [float(item) for item in value]
    require(all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result), f"{label} is outside normalized image space")
    require(result[0] < result[2] and result[1] < result[3], f"{label} has invalid extent")
    return result


def load_target_ledger(path: Path) -> dict[str, Any]:
    ledger = load_json(path)
    require(ledger.get("schema") == TARGET_LEDGER_SCHEMA, "target ledger schema mismatch")
    require(ledger.get("diagnostic_set_role") in ALLOWED_DATASET_ROLES, "unsupported target-ledger dataset role")
    require(ledger.get("uncertainty_frame_ratios") == UNCERTAINTY_RATIOS, "uncertainty profiles drifted")
    require_false_flags(ledger["authority"], "target_ledger.authority")
    events = ledger.get("events")
    require(isinstance(events, list) and events, "target ledger needs events")
    event_ids: set[str] = set()
    source_ids: set[str] = set()
    for event in events:
        event_id = event.get("event_id")
        source_id = event.get("source_id")
        require(isinstance(event_id, str) and event_id and event_id not in event_ids, "target ledger repeats event_id")
        require(isinstance(source_id, str) and source_id and source_id not in source_ids, "target ledger repeats source_id")
        event_ids.add(event_id)
        source_ids.add(source_id)
        window = event.get("window_ms")
        require(isinstance(window, list) and len(window) == 2 and 0 <= window[0] < window[1], f"{event_id}: invalid window")
        target = event.get("target_instance")
        require(isinstance(target, dict), f"{event_id}: missing target instance")
        require(isinstance(target.get("target_instance_id"), str) and target["target_instance_id"], f"{event_id}: missing target id")
        require(target.get("expected_route_relation") in ("inside", "outside"), f"{event_id}: invalid expected relation")
        allowlist = target.get("detector_label_allowlist")
        require(isinstance(allowlist, list) and all(isinstance(item, str) and item for item in allowlist), f"{event_id}: invalid label allowlist")
        frames = target.get("frames")
        require(isinstance(frames, list) and frames, f"{event_id}: target needs frozen frames")
        frame_ids: set[str] = set()
        timestamps: set[int] = set()
        for frame in frames:
            frame_id = frame.get("frame_id")
            timestamp = frame.get("timestamp_ms")
            require(isinstance(frame_id, str) and frame_id and frame_id not in frame_ids, f"{event_id}: repeated frame_id")
            require(isinstance(timestamp, int) and window[0] <= timestamp < window[1] and timestamp not in timestamps, f"{event_id}: invalid/repeated timestamp")
            frame_ids.add(frame_id)
            timestamps.add(timestamp)
            require(isinstance(frame.get("frame_sha256"), str) and len(frame["frame_sha256"]) == 64, f"{event_id}/{frame_id}: invalid frame SHA")
            require(isinstance(frame.get("frame_width"), int) and frame["frame_width"] > 0, f"{event_id}/{frame_id}: invalid width")
            require(isinstance(frame.get("frame_height"), int) and frame["frame_height"] > 0, f"{event_id}/{frame_id}: invalid height")
            visibility = frame.get("visibility")
            require(visibility in ("visible", "occluded", "absent"), f"{event_id}/{frame_id}: invalid visibility")
            if visibility == "visible":
                box = _normalized_box(frame.get("bbox_xyxy_norm"), f"{event_id}/{frame_id} bbox")
                contact = _normalized_pair(frame.get("contact_xy_norm"), f"{event_id}/{frame_id} contact")
                require(box[0] <= contact[0] <= box[2], f"{event_id}/{frame_id}: contact x is outside bbox")
                require(abs(contact[1] - box[3]) <= 0.02, f"{event_id}/{frame_id}: contact must lie at bbox bottom")
            else:
                require(frame.get("bbox_xyxy_norm") is None and frame.get("contact_xy_norm") is None,
                        f"{event_id}/{frame_id}: hidden target cannot carry geometry")
    return ledger


def load_projection(path: Path, ledger_path: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    projection = load_json(path)
    require(projection.get("schema") == PROJECTION_SCHEMA, "projection schema mismatch")
    require(projection.get("diagnostic_set_role") == ledger.get("diagnostic_set_role"), "projection role mismatch")
    require(projection.get("target_ledger_sha256") == sha256_file(ledger_path), "projection is not bound to target ledger")
    require_false_flags(projection["authority"], "projection.authority")
    expected = {event["event_id"]: event for event in ledger["events"]}
    actual = projection.get("events")
    require(isinstance(actual, list) and {event.get("event_id") for event in actual} == set(expected), "projection event inventory mismatch")
    for event in actual:
        event_id = event["event_id"]
        require(event.get("projection_mode") in ("per_frame", "stable_windows"), f"{event_id}: static full-window projection is forbidden")
        target_frames = {frame["frame_id"]: frame for frame in expected[event_id]["target_instance"]["frames"] if frame["visibility"] == "visible"}
        frames = event.get("frames")
        require(isinstance(frames, list), f"{event_id}: projection frames missing")
        require({frame.get("frame_id") for frame in frames} == set(target_frames), f"{event_id}: projection must exactly cover visible target frames")
        for frame in frames:
            target = target_frames[frame["frame_id"]]
            require(frame.get("timestamp_ms") == target["timestamp_ms"] and frame.get("frame_sha256") == target["frame_sha256"],
                    f"{event_id}/{frame['frame_id']}: projection frame identity mismatch")
            require(frame.get("status") == "admitted", f"{event_id}/{frame['frame_id']}: projection is not admitted")
            validate_polygon(frame.get("route_polygon_xy_norm"))
    return projection
