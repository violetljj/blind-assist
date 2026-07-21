#!/usr/bin/env python3
"""Validate a complete target-device metric-geometry evidence bundle.

Passing this gate permits an isolated geometry shadow experiment only. It never authorizes a
user-facing route instruction, a production-model replacement, or reuse on another device/mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_ustrf_sc_device_metric_geometry_evidence_v1"
REQUIRED_ARTIFACTS = {
    "calibration",
    "frame_clock",
    "body_local_ground_truth",
    "route_event_truth",
    "target_device_benchmark",
}
ARTIFACT_SCHEMAS = {
    name: f"blindassist_ustrf_sc_metric_geometry_{name}_artifact_v1"
    for name in REQUIRED_ARTIFACTS
}
ALLOWED_STATUSES = {"not_collected", "in_progress", "blocked", "complete"}


class ContractError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError("evidence manifest root must be an object")
    return value


def require_text(row: dict[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or value.startswith("REQUIRED_"):
        raise ContractError(f"{where}.{key} must be a concrete non-empty string")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_receipt(root: Path, receipt: dict[str, Any], where: str) -> Path:
    relative = require_text(receipt, "path", where)
    expected = require_text(receipt, "sha256", where)
    if len(expected) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected):
        raise ContractError(f"{where}.sha256 must be a SHA256 hex string")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{where}.path escapes manifest root") from error
    if not path.is_file() or sha256(path) != expected.lower():
        raise ContractError(f"{where} artifact is missing or has a mismatched SHA256")
    return path


def artifact_binding(device: dict[str, Any], calibration: dict[str, Any]) -> dict[str, str]:
    return {
        "device_id": require_text(device, "device_id", "device_identity"),
        "hardware_revision": require_text(device, "hardware_revision", "device_identity"),
        "mount_revision": require_text(device, "mount_revision", "device_identity"),
        "camera_frame": require_text(device, "camera_frame", "device_identity"),
        "body_frame": require_text(device, "body_frame", "device_identity"),
        "calibration_id": require_text(calibration, "calibration_id", "calibration"),
    }


def verify_source_evidence(root: Path, artifact: dict[str, Any], where: str) -> None:
    sources = artifact.get("source_evidence")
    if not isinstance(sources, list) or not sources:
        raise ContractError(f"{where}.source_evidence must contain at least one hash-bound raw or gate artifact")
    for index, receipt in enumerate(sources):
        if not isinstance(receipt, dict):
            raise ContractError(f"{where}.source_evidence[{index}] must be an object")
        verify_file_receipt(root, receipt, f"{where}.source_evidence[{index}]")


def load_verified_artifact(root: Path, receipt: dict[str, Any], where: str) -> dict[str, Any]:
    path = verify_file_receipt(root, receipt, where)
    artifact = load_json(path)
    verify_source_evidence(root, artifact, where)
    return artifact


def verify_artifact_semantics(
    artifact: dict[str, Any], *, name: str, binding: dict[str, str], metrics: dict[str, Any], where: str,
) -> None:
    if artifact.get("schema") != ARTIFACT_SCHEMAS[name]:
        raise ContractError(f"{where}.schema does not match the frozen {name} artifact schema")
    if artifact.get("artifact_kind") != name:
        raise ContractError(f"{where}.artifact_kind mismatch")
    if artifact.get("bundle_binding") != binding:
        raise ContractError(f"{where}.bundle_binding does not match device/mount/calibration identity")
    if artifact.get("metrics") != metrics:
        raise ContractError(f"{where}.metrics do not exactly match the bundle summary")


def numeric(row: dict[str, Any], key: str, where: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{where}.{key} must be numeric")
    return float(value)


def validate_partial(value: dict[str, Any], *, root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    device = value.get("device_identity")
    calibration = value.get("calibration")
    if not isinstance(device, dict) or not device:
        blockers.append("MISSING_DEVICE_IDENTITY")
    if not isinstance(calibration, dict) or not calibration:
        blockers.append("MISSING_CALIBRATION")
    for name in ("frame_clock", "body_local_ground_truth", "route_event_truth", "target_device_benchmark"):
        if not isinstance(value.get(name), dict) or not value[name]:
            blockers.append(f"MISSING_{name.upper()}")

    artifacts = value.get("evidence_artifacts")
    if not isinstance(artifacts, dict):
        blockers.append("MISSING_EVIDENCE_ARTIFACTS")
    else:
        unknown = set(artifacts) - REQUIRED_ARTIFACTS
        if unknown:
            raise ContractError(f"evidence_artifacts contains unknown kinds: {sorted(unknown)}")
        binding: dict[str, str] | None = None
        if isinstance(device, dict) and device and isinstance(calibration, dict) and calibration:
            try:
                binding = artifact_binding(device, calibration)
            except ContractError:
                binding = None
        for name in sorted(REQUIRED_ARTIFACTS):
            receipt = artifacts.get(name)
            if not isinstance(receipt, dict) or not receipt:
                blockers.append(f"MISSING_{name.upper()}_ARTIFACT")
                continue
            where = f"evidence_artifacts.{name}"
            artifact = load_verified_artifact(root, receipt, where)
            if artifact.get("schema") != ARTIFACT_SCHEMAS[name] or artifact.get("artifact_kind") != name:
                raise ContractError(f"{where} does not match the frozen {name} artifact contract")
            metrics = value.get(name)
            if isinstance(metrics, dict) and metrics and artifact.get("metrics") != metrics:
                raise ContractError(f"{where}.metrics do not exactly match the bundle summary")
            if binding is not None and artifact.get("bundle_binding") != binding:
                raise ContractError(f"{where}.bundle_binding does not match device/mount/calibration identity")

    if value.get("status") == "blocked":
        require_text(value, "blocker_code", "bundle")
        blocker_evidence = value.get("blocker_evidence")
        if not isinstance(blocker_evidence, list) or not blocker_evidence:
            raise ContractError("blocked bundle needs at least one blocker_evidence receipt")
        for index, receipt in enumerate(blocker_evidence):
            if not isinstance(receipt, dict):
                raise ContractError(f"blocker_evidence[{index}] must be an object")
            verify_file_receipt(root, receipt, f"blocker_evidence[{index}]")

    return {
        "ok": True,
        "status": value["status"],
        "device_metric_geometry_admitted": False,
        "geometry_shadow_authorized": False,
        "blockers": blockers or ["EVIDENCE_BUNDLE_NOT_COMPLETE"],
        "declared_blocker_code": value.get("blocker_code"),
        "production_authority": False,
    }


def validate(value: dict[str, Any], *, root: Path, require_complete: bool) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise ContractError("unexpected device metric-geometry schema")
    if value.get("production_authority") is not False:
        raise ContractError("production_authority must remain false")
    if value.get("status") not in ALLOWED_STATUSES:
        raise ContractError("status must be not_collected, in_progress, blocked, or complete")
    complete = value.get("status") == "complete"
    if not complete:
        if require_complete:
            raise ContractError("--require-complete needs status=complete")
        return validate_partial(value, root=root)

    device = value.get("device_identity")
    if not isinstance(device, dict):
        raise ContractError("device_identity must be an object")
    for key in ("device_id", "hardware_revision", "mount_revision", "camera_frame", "body_frame"):
        require_text(device, key, "device_identity")
    if device.get("device_stage") not in {"handheld_experiment", "fixed_body_mount", "glasses"}:
        raise ContractError("device_identity.device_stage is invalid")
    if device.get("device_stage") == "glasses" and device.get("reused_phone_evidence") is not False:
        raise ContractError("glasses evidence must not reuse phone calibration or device receipts")

    calibration = value.get("calibration")
    if not isinstance(calibration, dict):
        raise ContractError("calibration must be an object")
    for key in ("calibration_id", "camera_calibration_version", "collector_id", "reviewer_id"):
        require_text(calibration, key, "calibration")
    if calibration.get("collector_id") == calibration.get("reviewer_id"):
        raise ContractError("calibration collector and reviewer must differ")
    if calibration.get("independent_review_approved") is not True:
        raise ContractError("calibration independent review is not approved")
    if calibration.get("camera_frame") != device["camera_frame"] or calibration.get("body_frame") != device["body_frame"]:
        raise ContractError("calibration frame does not match device identity")
    if numeric(calibration, "sample_count", "calibration") < 30:
        raise ContractError("calibration needs at least 30 samples")
    if numeric(calibration, "pose_coverage_bins", "calibration") < 5:
        raise ContractError("calibration needs at least 5 pose coverage bins")
    if numeric(calibration, "intrinsics_p95_reprojection_px", "calibration") > 1.5:
        raise ContractError("intrinsics P95 reprojection exceeds 1.5 px")
    if numeric(calibration, "depth_registration_p95_error_m", "calibration") > 0.03:
        raise ContractError("depth registration P95 exceeds 0.03 m")
    if numeric(calibration, "mount_translation_repeatability_m", "calibration") > 0.01:
        raise ContractError("mount translation repeatability exceeds 0.01 m")
    if numeric(calibration, "mount_rotation_repeatability_deg", "calibration") > 1.0:
        raise ContractError("mount rotation repeatability exceeds 1 degree")

    clock = value.get("frame_clock")
    if not isinstance(clock, dict):
        raise ContractError("frame_clock must be an object")
    if clock.get("camera_frame") != device["camera_frame"]:
        raise ContractError("frame_clock camera_frame does not match device")
    if clock.get("capture_timestamps_strictly_monotonic") is not True:
        raise ContractError("capture timestamps must be strictly monotonic")
    if clock.get("pose_reference_mode") != "INTER_FRAME_STABLE":
        raise ContractError("pose reference must be INTER_FRAME_STABLE")
    if numeric(clock, "source_aligned_metric_depth_pair_count", "frame_clock") < 100:
        raise ContractError("frame_clock needs at least 100 source-aligned metric-depth pairs")
    if numeric(clock, "source_aligned_metric_depth_fraction", "frame_clock") < 0.95:
        raise ContractError("source-aligned metric-depth fraction is below 0.95")
    if numeric(clock, "maximum_cross_sensor_sync_error_ms", "frame_clock") > 20.0:
        raise ContractError("cross-sensor sync error exceeds 20 ms")

    ground = value.get("body_local_ground_truth")
    if not isinstance(ground, dict):
        raise ContractError("body_local_ground_truth must be an object")
    if ground.get("body_frame") != device["body_frame"]:
        raise ContractError("ground-truth body frame does not match device")
    if ground.get("independent_review_approved") is not True or ground.get("collector_id") == ground.get("reviewer_id"):
        raise ContractError("body-local ground truth lacks independent review")
    if numeric(ground, "sample_count", "body_local_ground_truth") < 30:
        raise ContractError("body-local ground truth needs at least 30 samples")
    if numeric(ground, "p95_plane_distance_error_m", "body_local_ground_truth") > 0.03:
        raise ContractError("body-local ground P95 error exceeds 0.03 m")
    if ground.get("clear_obstacle_head_drop_and_missing_depth_covered") is not True:
        raise ContractError("body-local ground set lacks required positive and negative geometry coverage")

    event_truth = value.get("route_event_truth")
    if not isinstance(event_truth, dict):
        raise ContractError("route_event_truth must be an object")
    if event_truth.get("route_conditioned_truth_eligible") is not True:
        raise ContractError("route-conditioned human event truth is not eligible")
    if numeric(event_truth, "episode_count", "route_event_truth") < 120:
        raise ContractError("route event truth needs the frozen 120-episode matrix")
    if numeric(event_truth, "matched_pair_count", "route_event_truth") < 60:
        raise ContractError("route event truth needs the frozen 60 matched pairs")

    benchmark = value.get("target_device_benchmark")
    if not isinstance(benchmark, dict):
        raise ContractError("target_device_benchmark must be an object")
    for key in ("device_id", "mount_revision", "calibration_id"):
        expected = device[key] if key in device else calibration[key]
        if benchmark.get(key) != expected:
            raise ContractError(f"target benchmark {key} does not match the admitted evidence")
    if numeric(benchmark, "failure_count", "target_device_benchmark") != 0:
        raise ContractError("target-device benchmark has failures")
    if numeric(benchmark, "pipeline_p95_ms", "target_device_benchmark") > 70.0:
        raise ContractError("target-device pipeline P95 exceeds 70 ms")
    if numeric(benchmark, "stale_output_count", "target_device_benchmark") != 0:
        raise ContractError("target-device benchmark emitted stale output")
    if benchmark.get("latest_only_queue_verified") is not True:
        raise ContractError("target-device benchmark did not verify latest-only freshness")
    if numeric(benchmark, "thermal_throttle_count", "target_device_benchmark") != 0:
        raise ContractError("target-device benchmark observed thermal throttling")

    artifacts = value.get("evidence_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ContractError("evidence_artifacts must contain the exact frozen evidence set")
    binding = artifact_binding(device, calibration)
    for name in sorted(REQUIRED_ARTIFACTS):
        receipt = artifacts[name]
        if not isinstance(receipt, dict):
            raise ContractError(f"evidence_artifacts.{name} must be an object")
        where = f"evidence_artifacts.{name}"
        artifact = load_verified_artifact(root, receipt, where)
        metrics = value[name]
        verify_artifact_semantics(artifact, name=name, binding=binding, metrics=metrics, where=where)

    return {
        "ok": True,
        "status": "complete",
        "device_id": device["device_id"],
        "mount_revision": device["mount_revision"],
        "calibration_id": calibration["calibration_id"],
        "device_metric_geometry_admitted": bool(require_complete),
        "geometry_shadow_authorized": bool(require_complete),
        "production_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(load_json(args.manifest), root=args.manifest.parent, require_complete=args.require_complete)
    except ContractError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
