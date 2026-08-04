#!/usr/bin/env python3
"""Fail-closed source admission for fresh metric snapshot layered intrusion R0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_hftf_fresh_metric_snapshot_layered_intrusion_r0_protocol_v1"
MANIFEST_SCHEMA = "blindassist_hftf_fresh_metric_snapshot_source_package_v1"
REPORT_SCHEMA = "blindassist_hftf_fresh_metric_snapshot_source_admission_v1"
FORBIDDEN_KEYS = {
    "arm_outputs",
    "prediction",
    "predictions",
    "b0_output",
    "b1_output",
    "c1_output",
    "selected_threshold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} escapes package root") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {name}: {candidate}")
    return candidate


def _verify_bound_file(root: Path, item: dict[str, Any], prefix: str) -> None:
    path = _bounded_path(root, item.get("path"), f"{prefix}.path")
    expected = item.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{prefix}.sha256 must have 64 hex characters")
    if sha256(path) != expected.upper():
        raise ValueError(f"{prefix} SHA-256 mismatch")


def _load_bound_json(root: Path, item: dict[str, Any], prefix: str) -> dict[str, Any]:
    _verify_bound_file(root, item, prefix)
    return load_json(_bounded_path(root, item.get("path"), f"{prefix}.path"))


def _walk_forbidden(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden outcome key at {location}.{key}")
            _walk_forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{location}[{index}]")


def _expected_class(session_id: str) -> str:
    if session_id.startswith("clear_all_"):
        return "CLEAR_ALL"
    if session_id.startswith("foot_only_"):
        return "FOOT_ONLY"
    if session_id.startswith("body_only_"):
        return "BODY_ONLY"
    if session_id.startswith("head_only_"):
        return "HEAD_ONLY"
    if session_id.startswith("multi_layer_"):
        return "MULTI_LAYER"
    if session_id.startswith("left_right_height_competition_"):
        return "LEFT_RIGHT_HEIGHT_COMPETITION"
    raise ValueError(f"unrecognized frozen session id: {session_id}")


def _validate_truth_shape(session: dict[str, Any], scenario: str) -> None:
    layers = session.get("expected_intrusion_by_height")
    if not isinstance(layers, dict) or set(layers) != {"foot", "body", "head"}:
        raise ValueError("expected_intrusion_by_height must contain foot/body/head exactly")
    if any(not isinstance(value, bool) for value in layers.values()):
        raise ValueError("expected layer values must be boolean")
    expected = {
        "CLEAR_ALL": {"foot": False, "body": False, "head": False},
        "FOOT_ONLY": {"foot": True, "body": False, "head": False},
        "BODY_ONLY": {"foot": False, "body": True, "head": False},
        "HEAD_ONLY": {"foot": False, "body": False, "head": True},
    }
    if scenario in expected and layers != expected[scenario]:
        raise ValueError(f"layer truth conflicts with {scenario}")
    if scenario == "MULTI_LAYER" and sum(layers.values()) < 2:
        raise ValueError("MULTI_LAYER requires at least two positive heights")
    cells = session.get("truth_cells")
    if not isinstance(cells, list):
        raise ValueError("truth_cells must be a list")
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("truth cell must be object")
        if cell.get("direction_deg") not in {-25, 0, 25}:
            raise ValueError("truth cell direction is outside frozen centers")
        if cell.get("height") not in {"foot", "body", "head"}:
            raise ValueError("truth cell height is invalid")
        distance = _finite_number(cell.get("distance_m"), "truth cell distance_m")
        if min(abs(distance - center) for center in (1.0, 1.5, 2.0)) > 0.10 + 1e-9:
            raise ValueError("truth cell distance is outside frozen centers")
    if scenario == "CLEAR_ALL" and cells:
        raise ValueError("CLEAR_ALL cannot contain intrusion truth cells")
    if scenario != "CLEAR_ALL" and not cells:
        raise ValueError(f"{scenario} requires intrusion truth cells")
    if scenario == "LEFT_RIGHT_HEIGHT_COMPETITION":
        directions = {cell["direction_deg"] for cell in cells}
        heights = {cell["height"] for cell in cells}
        if not {-25, 25}.issubset(directions) or len(heights) < 2:
            raise ValueError("competition session requires left/right and distinct heights")
    positive_heights = {height for height, positive in layers.items() if positive}
    cell_heights = {cell["height"] for cell in cells}
    if positive_heights != cell_heights:
        raise ValueError("truth cell heights differ from expected intrusion layers")


def _validate_calibration(
    root: Path, item: dict[str, Any], protocol: dict[str, Any]
) -> None:
    document = _load_bound_json(root, item, "camera_calibration")
    if document.get("schema") != "blindassist_hftf_camera_calibration_v1":
        raise ValueError("unexpected camera calibration schema")
    if document.get("device") != protocol["source"]["device"]:
        raise ValueError("calibration device mismatch")
    intrinsics = document.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise ValueError("calibration intrinsics must contain four values")
    if any(_finite_number(value, "intrinsic") <= 0 for value in intrinsics[:2]):
        raise ValueError("focal lengths must be positive")
    _finite_number(intrinsics[2], "cx"); _finite_number(intrinsics[3], "cy")
    error = _finite_number(document.get("reprojection_error_px"), "reprojection_error_px")
    if error > float(protocol["source"]["camera_calibration_reprojection_error_px_max"]):
        raise ValueError("camera calibration reprojection error exceeds protocol")
    if document.get("sealed_before_collection") is not True:
        raise ValueError("camera calibration document was not sealed before collection")


def _validate_physical_truth(
    root: Path,
    item: dict[str, Any],
    session: dict[str, Any],
    scenario: str,
    protocol: dict[str, Any],
) -> None:
    document = _load_bound_json(root, item, f"{session['session_id']}.physical_truth")
    if document.get("schema") != "blindassist_hftf_physical_intrusion_truth_v1":
        raise ValueError("unexpected physical truth schema")
    if document.get("session_id") != session["session_id"]:
        raise ValueError("physical truth session mismatch")
    if document.get("qnn_used_for_truth") is not False:
        raise ValueError("physical truth must be independent of QNN")
    if document.get("supports_outside_evaluated_envelopes") is not True:
        raise ValueError("obstacle supports are not proven outside evaluated envelopes")
    tools = document.get("measurement_tools")
    if not isinstance(tools, list) or not {"rigid_ruler_or_tape", "laser_distance_meter", "level", "fiducial_pose_board"}.issubset(set(tools)):
        raise ValueError("physical truth measurement tools are incomplete")
    limits = protocol["source"]["physical_truth"]
    if _finite_number(document.get("distance_error_m"), "distance_error_m") > float(limits["maximum_distance_error_m"]):
        raise ValueError("physical truth distance error exceeds protocol")
    if _finite_number(document.get("height_lateral_error_m"), "height_lateral_error_m") > float(limits["maximum_height_or_lateral_error_m"]):
        raise ValueError("physical truth height/lateral error exceeds protocol")
    prisms = document.get("obstacle_prisms")
    if not isinstance(prisms, list):
        raise ValueError("obstacle_prisms must be a list")
    if scenario == "CLEAR_ALL" and prisms:
        raise ValueError("CLEAR_ALL physical truth cannot contain obstacle prisms")
    if scenario != "CLEAR_ALL" and not prisms:
        raise ValueError(f"{scenario} physical truth requires obstacle prisms")
    for prism in prisms:
        if not isinstance(prism, dict):
            raise ValueError("obstacle prism must be object")
        values = [_finite_number(prism.get(key), key) for key in ("x_min_m", "x_max_m", "y_min_m", "y_max_m", "z_min_m", "z_max_m")]
        if not (values[0] < values[1] and values[2] < values[3] and values[4] < values[5]):
            raise ValueError("obstacle prism bounds must be strictly increasing")
    if document.get("expected_intrusion_by_height") != session.get("expected_intrusion_by_height"):
        raise ValueError("physical truth height labels differ from manifest")
    if document.get("truth_cells") != session.get("truth_cells"):
        raise ValueError("physical truth cells differ from manifest")


def validate(protocol_path: Path, package_root: Path, manifest_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected protocol schema")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected source manifest schema")
    protocol_hash = sha256(protocol_path)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("manifest does not bind current protocol SHA-256")
    _walk_forbidden(manifest)
    if manifest.get("package_role") != "FORMAL_DECISION_COHORT":
        raise ValueError("package_role must be FORMAL_DECISION_COHORT")
    if manifest.get("device") != protocol["source"]["device"]:
        raise ValueError("device differs from frozen protocol")
    calibration = manifest.get("camera_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("camera_calibration must be object")
    _validate_calibration(package_root, calibration, protocol)
    if calibration.get("sealed_before_collection") is not True:
        raise ValueError("camera calibration was not sealed before collection")

    expected_ids = list(protocol["cohort"]["session_ids"])
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("sessions must be list")
    actual_ids = [session.get("session_id") for session in sessions if isinstance(session, dict)]
    if actual_ids != expected_ids:
        raise ValueError("session roster/order differs from frozen protocol")
    parent_ids = [session.get("parent_capture_id") for session in sessions]
    if any(not isinstance(value, str) or not value for value in parent_ids):
        raise ValueError("every parent_capture_id must be non-empty")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("parent_capture_id values must be unique")

    snapshot_total = 0
    class_counts: dict[str, int] = {}
    for session in sessions:
        session_id = session["session_id"]
        scenario = _expected_class(session_id)
        if session.get("scenario_class") != scenario:
            raise ValueError(f"scenario mismatch: {session_id}")
        class_counts[scenario] = class_counts.get(scenario, 0) + 1
        height = _finite_number(session.get("lens_height_m"), "lens_height_m")
        lower, upper = protocol["source"]["lens_height_m_range"]
        if not lower <= height <= upper:
            raise ValueError(f"lens height out of protocol: {session_id}")
        for angle in ("pitch_error_deg", "roll_error_deg", "yaw_error_deg"):
            if abs(_finite_number(session.get(angle), angle)) > float(
                protocol["source"]["pitch_roll_yaw_abs_error_deg_max"]
            ):
                raise ValueError(f"{angle} out of protocol: {session_id}")
        if session.get("truth_sealed_before_qnn_output") is not True:
            raise ValueError(f"truth was not sealed before QNN output: {session_id}")
        truth = session.get("physical_truth")
        if not isinstance(truth, dict):
            raise ValueError("physical_truth must be object")
        _validate_physical_truth(package_root, truth, session, scenario, protocol)
        _validate_truth_shape(session, scenario)
        sealed_ns = int(session.get("truth_sealed_timestamp_ns", -1))
        frames = session.get("frames")
        if not isinstance(frames, list) or len(frames) != int(
            protocol["cohort"]["snapshots_per_session"]
        ):
            raise ValueError(f"wrong frame count: {session_id}")
        timestamps = []
        for expected_index, frame in enumerate(frames):
            if frame.get("frame_index") != expected_index:
                raise ValueError(f"non-canonical frame index: {session_id}")
            capture_ns = int(frame.get("capture_timestamp_ns", -1))
            depth_capture_ns = int(frame.get("depth_capture_timestamp_ns", -2))
            completed_ns = int(frame.get("depth_completed_timestamp_ns", -1))
            if capture_ns <= 0 or depth_capture_ns != capture_ns or completed_ns < capture_ns:
                raise ValueError(f"invalid same-frame timestamps: {session_id}/{expected_index}")
            if sealed_ns <= 0 or sealed_ns >= completed_ns:
                raise ValueError(f"truth not sealed before QNN completion: {session_id}")
            if frame.get("rgb_shape") != protocol["source"]["required_rgb_capture_shape"]:
                raise ValueError(f"RGB shape mismatch: {session_id}/{expected_index}")
            if frame.get("depth_shape") != protocol["source"]["required_depth_shape"]:
                raise ValueError(f"depth shape mismatch: {session_id}/{expected_index}")
            if frame.get("depth_source") != "CAMERAX_SAME_FRAME_QNN_METRIC_DEPTH":
                raise ValueError(f"depth source mismatch: {session_id}/{expected_index}")
            _verify_bound_file(package_root, frame.get("rgb", {}), "rgb")
            _verify_bound_file(package_root, frame.get("depth", {}), "depth")
            timestamps.append(capture_ns)
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise ValueError(f"timestamps are not strictly increasing: {session_id}")
        snapshot_total += len(frames)

    expected_classes = set(protocol["cohort"]["scenario_classes"])
    if set(class_counts) != expected_classes or any(
        count != int(protocol["cohort"]["sessions_per_class"])
        for count in class_counts.values()
    ):
        raise ValueError("scenario class counts differ from frozen protocol")
    if snapshot_total != int(protocol["cohort"]["exact_snapshot_count"]):
        raise ValueError("snapshot total differs from frozen protocol")
    return {
        "schema": REPORT_SCHEMA,
        "terminal": "FRESH_SNAPSHOT_SOURCE_PACKAGE_ADMITTED_OUTCOME_NOT_OPENED",
        "protocol_sha256": protocol_hash,
        "manifest_sha256": sha256(manifest_path),
        "session_count": len(sessions),
        "snapshot_count": snapshot_total,
        "scenario_class_counts": dict(sorted(class_counts.items())),
        "arm_outputs_opened": False,
        "effect_evaluation_authorized": True,
        "claim_ceiling": "source admission only; arm effect remains unopened",
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args.protocol, args.package_root, args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {
            "schema": REPORT_SCHEMA,
            "terminal": "FRESH_SNAPSHOT_SOURCE_PACKAGE_NOT_EVALUABLE",
            "error": str(error),
            "effect_evaluation_authorized": False,
        }
    write_new(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["effect_evaluation_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
