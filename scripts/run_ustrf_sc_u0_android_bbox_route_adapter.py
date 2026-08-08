#!/usr/bin/env python3
"""Run the USTRF U0 detector+bbox+explicit-route arm on Android.

The host only validates and transports hash-bound inputs. Android performs video
decode, shipped YOLO inference, causal explicit-route bbox gating and the shared
AssistDecisionKernel. The pulled output is independently checked, including a
recalculation of every route sample selection and bbox/corridor decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Mapping

import run_ustrf_sc_u0_android_baseline_adapter as common
from validate_explicit_route_intent_episode import validate_episode


ARM_ID = "detector_bbox_explicit_route"
ADAPTER_ID = "detector_bbox_explicit_route_adapter_v1"
BACKEND_ID = "android_kotlin_assist_decision_kernel_v1"
REQUEST_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_request_v1"
OUTPUT_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_output_v1"
MANIFEST_SCHEMA = "blindassist_ustrf_sc_u0_sanitized_inference_manifest_v1"
CONFIG_SCHEMA = "blindassist_ustrf_sc_u0_android_bbox_route_adapter_config_v1"
RECEIPT_SCHEMA = "blindassist_ustrf_sc_u0_android_backend_receipt_v1"
ROUTE_RECEIPT_SCHEMA = "blindassist_ustrf_sc_u0_route_conditioning_receipt_v1"
ROUTE_SCHEMA = "blindassist_explicit_route_intent_episode_v1"
ROUTE_POLICY = "episode_explicit_causal_route_v1"
GATE_CONTRACT_ID = "bbox_bottom_footprint_polyline_corridor_v1"
UNKNOWN_ROUTE_POLICY = "context_attention_only_empty_detection_gate_v1"
FRAME_DECODE_POLICY = "android_media_metadata_retriever_closest_v1"
DECODED_PAYLOAD_CONTRACT = "rgba8888_row_major_android_getpixels_v1"
DETECTOR_RUNTIME = "tflite_cpu_4_threads_v1"
TARGET_PACKAGE = "com.linnan.blindassist"
INSTRUMENTATION_COMPONENT = "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.linnan.blindassist.benchmark.UstrfU0BaselineAdapterDeviceTest"
TEST_METHOD = f"{TEST_CLASS}#runDetectorBBoxExplicitRouteAdapter"
DEVICE_SOURCE = Path(
    "apps/benchmarks/device-benchmark/src/main/java/com/linnan/blindassist/benchmark/UstrfU0BaselineAdapterDeviceTest.kt"
)
GATE_SOURCE = Path(
    "apps/benchmarks/device-benchmark/src/main/java/com/linnan/blindassist/benchmark/UstrfU0ExplicitRouteDetectionGate.kt"
)
REQUIRED_HORIZONS = [1_000, 2_000, 3_000]
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


AdapterError = common.AdapterError
sha256_file = common.sha256_file
normalized_text_sha256 = common.normalized_text_sha256
load_json = common.load_json
require_sha = common.require_sha
require_text = common.require_text
confined_file = common.confined_file


def _require_exact(value: Mapping[str, Any], expected: Mapping[str, Any], *, where: str) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AdapterError(f"{where} {key} mismatch")


def validate_inputs(
    *,
    request_path: Path,
    manifest_path: Path,
    inference_root: Path,
    artifact_path: Path,
    threshold_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, Path]:
    for path, where in (
        (request_path, "request"),
        (manifest_path, "sanitized inference manifest"),
        (artifact_path, "fixed-arm artifact"),
        (threshold_path, "threshold config"),
    ):
        if not path.resolve().is_file():
            raise AdapterError(f"{where} is not a local file: {path}")
    request = load_json(request_path, where="request")
    manifest = load_json(manifest_path, where="sanitized inference manifest")
    config = load_json(threshold_path, where="threshold config")
    _require_exact(request, {
        "schema": REQUEST_SCHEMA,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "kernel_execution_backend_id": BACKEND_ID,
        "decision_profile_id": "STANDARD",
        "fit_policy": "fixed_no_fit_v1",
        "event_identity_policy": "kernel_native_optional_v1",
        "route_input_policy": ROUTE_POLICY,
        "synthetic_fixture": False,
        "blind_accessed": False,
        "future_inputs_used": False,
        "production_model_replacement_authorized": False,
    }, where="request")
    frames = request.get("frames")
    if not isinstance(frames, list) or not frames:
        raise AdapterError("request frames must be non-empty")
    cadence = request.get("decision_cadence")
    if not isinstance(cadence, dict) or cadence.get("canonical_step_ms") != 500 or cadence.get(
        "route_sample_policy"
    ) != "latest_valid_generated_at_or_before_frame_v1":
        raise AdapterError("request does not use the frozen causal 500ms cadence")
    if sha256_file(manifest_path) != request.get("sanitized_inference_manifest_sha256"):
        raise AdapterError("sanitized inference manifest SHA-256 mismatch")
    _require_exact(manifest, {
        "schema": MANIFEST_SCHEMA,
        "arm_id": ARM_ID,
        "episode_id": request.get("episode_id"),
        "route_input_policy": ROUTE_POLICY,
        "adapter_route_input_sha256": request.get("adapter_route_input_sha256"),
        "adapter_route_source_episode_id": request.get("episode_id"),
        "blind_accessed": False,
        "future_inputs_used": False,
        "review_fields_present": False,
        "adjudication_fields_present": False,
        "event_label_fields_present": False,
    }, where="sanitized inference manifest")
    if manifest.get("frames") != frames:
        raise AdapterError("sanitized inference frames differ from request")
    video_path = confined_file(inference_root, manifest.get("input_video_path"), where="input video path")
    ledger_path = confined_file(
        inference_root, manifest.get("capture_frame_ledger_path"), where="capture frame ledger path"
    )
    route_path = confined_file(
        inference_root, manifest.get("adapter_route_input_path"), where="explicit route input path"
    )
    if sha256_file(video_path) != request.get("input_video_sha256") or sha256_file(video_path) != manifest.get(
        "input_video_sha256"
    ):
        raise AdapterError("input video SHA-256 mismatch")
    if sha256_file(ledger_path) != request.get("source_capture_frame_ledger_sha256") or sha256_file(
        ledger_path
    ) != manifest.get("capture_frame_ledger_sha256"):
        raise AdapterError("capture frame ledger SHA-256 mismatch")
    route_sha = sha256_file(route_path)
    if any(route_sha != expected for expected in (
        request.get("adapter_route_input_sha256"),
        request.get("truth_route_intent_sha256"),
        manifest.get("adapter_route_input_sha256"),
        manifest.get("truth_route_intent_sha256"),
    )):
        raise AdapterError("explicit route input SHA-256 mismatch")
    route = load_json(route_path, where="explicit route input")
    try:
        validate_episode(route, runtime=True)
    except (KeyError, TypeError, ValueError) as error:
        raise AdapterError(f"explicit route input violates runtime contract: {error}") from error
    if route.get("episode_id") != request.get("episode_id"):
        raise AdapterError("explicit route episode identity mismatch")
    ledger = load_json(ledger_path, where="capture frame ledger")
    if ledger.get("schema") != "blindassist_capture_frame_ledger_v1" or ledger.get("episode_id") != request.get(
        "episode_id"
    ):
        raise AdapterError("capture frame ledger identity mismatch")
    ledger_frames = ledger.get("frames")
    binding_keys = ("frame_id", "frame_index", "capture_timestamp_ns", "video_pts_ms", "frame_payload_sha256")
    if not isinstance(ledger_frames, list) or len(ledger_frames) != len(frames):
        raise AdapterError("capture frame ledger inventory mismatch")
    for index, (ledger_frame, request_frame) in enumerate(zip(ledger_frames, frames)):
        if not isinstance(ledger_frame, dict) or any(
            ledger_frame.get(key) != request_frame.get(key) for key in binding_keys
        ):
            raise AdapterError(f"capture frame ledger frame {index} differs from request")
    if sha256_file(artifact_path) != request.get("fold_artifact_sha256"):
        raise AdapterError("fixed-arm artifact SHA-256 mismatch")
    if sha256_file(threshold_path) != request.get("threshold_config_sha256"):
        raise AdapterError("threshold config SHA-256 mismatch")
    _require_exact(config, {
        "schema": CONFIG_SCHEMA,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "fit_policy": "fixed_no_fit_v1",
        "event_identity_policy": "kernel_native_optional_v1",
        "route_input_policy": ROUTE_POLICY,
        "kernel_execution_backend_id": BACKEND_ID,
        "decision_profile_id": "STANDARD",
        "assist_scenario": "GENERAL",
        "model_asset_name": "yolo11n_fp16_320.tflite",
        "labels_asset_name": "coco_labels.txt",
        "input_size": 320,
        "confidence_threshold": 0.35,
        "iou_threshold": 0.45,
        "detector_runtime": DETECTOR_RUNTIME,
        "frame_decode_policy": FRAME_DECODE_POLICY,
        "decoded_payload_contract": DECODED_PAYLOAD_CONTRACT,
        "route_gate_contract_id": GATE_CONTRACT_ID,
        "unknown_route_policy": UNKNOWN_ROUTE_POLICY,
        "minimum_route_confidence": 0.5,
        "maximum_route_age_ms": 1_000,
        "corridor_half_width_frame_ratio": 0.08,
        "obstacle_footprint_height_ratio": 0.25,
        "instrumentation_component": INSTRUMENTATION_COMPONENT,
        "instrumentation_test_class": TEST_METHOD,
        "blind_accessed": False,
        "future_inputs_used": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }, where="threshold config")
    for key in (
        "model_asset_sha256", "labels_asset_sha256", "device_adapter_implementation_sha256",
        "route_gate_implementation_sha256", "host_adapter_implementation_sha256",
    ):
        require_sha(config.get(key), where=f"threshold config {key}")
    maximum_decode_error = config.get("maximum_decode_pts_error_us")
    if (
        not isinstance(maximum_decode_error, int)
        or isinstance(maximum_decode_error, bool)
        or not 0 <= maximum_decode_error <= 20_000
    ):
        raise AdapterError("threshold config maximum_decode_pts_error_us must be an integer in [0, 20000]")
    repo = Path(__file__).resolve().parents[1]
    if normalized_text_sha256(Path(__file__).resolve()) != config.get("host_adapter_implementation_sha256"):
        raise AdapterError("host adapter implementation differs from threshold config")
    if normalized_text_sha256(repo / DEVICE_SOURCE) != config.get("device_adapter_implementation_sha256"):
        raise AdapterError("Android device adapter implementation differs from threshold config")
    if normalized_text_sha256(repo / GATE_SOURCE) != config.get("route_gate_implementation_sha256"):
        raise AdapterError("route gate implementation differs from threshold config")
    return request, manifest, config, route, video_path, ledger_path, route_path


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def _segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    def cross(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    projections_overlap = (
        max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
        and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))
    )
    return projections_overlap and cross(a, b, c) * cross(a, b, d) <= 0.0 and cross(c, d, a) * cross(c, d, b) <= 0.0


def _segment_rectangle_distance(start: tuple[float, float], end: tuple[float, float], box: list[float]) -> float:
    left, top, right, bottom = box
    def inside(point: tuple[float, float]) -> bool:
        return left <= point[0] <= right and top <= point[1] <= bottom
    if inside(start) or inside(end):
        return 0.0
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    if any(_segments_intersect(start, end, corners[index], corners[(index + 1) % 4]) for index in range(4)):
        return 0.0
    def point_box(point: tuple[float, float]) -> float:
        dx = max(left - point[0], 0.0, point[0] - right)
        dy = max(top - point[1], 0.0, point[1] - bottom)
        return math.hypot(dx, dy)
    return min(point_box(start), point_box(end), *(_point_segment_distance(corner, start, end) for corner in corners))


def _expected_route_selection(route: Mapping[str, Any], frame_ms: int, config: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    selected: dict[str, Any] | None = None
    selected_index: int | None = None
    for index, sample in enumerate(route["samples"]):
        if sample["timestamp_ms"] <= frame_ms:
            selected, selected_index = sample, index
        else:
            break
    if selected is None:
        return None, "NO_CAUSAL_ROUTE_SAMPLE"
    selected = {**selected, "sample_index": selected_index}
    if not selected["route_valid"]:
        return selected, "ROUTE_MARKED_INVALID"
    if frame_ms > selected["valid_until_timestamp_ms"]:
        return selected, "ROUTE_STALE"
    if frame_ms - selected["timestamp_ms"] > config["maximum_route_age_ms"]:
        return selected, "ROUTE_TOO_OLD"
    if selected["confidence"] < config["minimum_route_confidence"]:
        return selected, "ROUTE_LOW_CONFIDENCE"
    horizons = [row["horizon_ms"] for row in selected["horizon_waypoints"]]
    if horizons != REQUIRED_HORIZONS:
        return selected, "ROUTE_WAYPOINT_CONTRACT_INVALID"
    return selected, "ROUTE_USABLE"


def validate_route_receipt(
    receipt: Mapping[str, Any], *, request: Mapping[str, Any], route: Mapping[str, Any],
    config: Mapping[str, Any], decoded_frames: list[Mapping[str, Any]],
) -> None:
    _require_exact(receipt, {
        "schema": ROUTE_RECEIPT_SCHEMA,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "route_input_policy": ROUTE_POLICY,
        "route_input_sha256": request.get("adapter_route_input_sha256"),
        "route_episode_id": request.get("episode_id"),
        "route_parent_source_id": route.get("parent_source_id"),
        "route_provider_type": route.get("provider", {}).get("type"),
        "route_provider_id": route.get("provider", {}).get("provider_id"),
        "projection_receipt_id": route.get("coordinate_contract", {}).get("projection_receipt_id"),
        "route_gate_contract_id": GATE_CONTRACT_ID,
        "route_gate_implementation_sha256": config.get("route_gate_implementation_sha256"),
        "unknown_route_policy": UNKNOWN_ROUTE_POLICY,
        "route_sample_policy": "latest_valid_generated_at_or_before_frame_v1",
        "future_inputs_used": False,
        "risk_model_inferred_route": False,
        "frame_count": len(request["frames"]),
    }, where="route conditioning receipt")
    frames = receipt.get("frames")
    if not isinstance(frames, list) or len(frames) != len(request["frames"]):
        raise AdapterError("route conditioning frame inventory mismatch")
    for index, (row, requested, decoded) in enumerate(zip(frames, request["frames"], decoded_frames)):
        if not isinstance(row, dict):
            raise AdapterError(f"route conditioning frame {index} must be an object")
        frame_ms = requested["video_pts_ms"]
        expected_sample, reason = _expected_route_selection(route, frame_ms, config)
        _require_exact(row, {
            "frame_id": requested["frame_id"],
            "frame_timestamp_ms": frame_ms,
            "route_episode_id": request["episode_id"],
            "route_parent_source_id": route["parent_source_id"],
            "route_provider_type": route["provider"]["type"],
            "route_provider_id": route["provider"]["provider_id"],
            "projection_receipt_id": route["coordinate_contract"]["projection_receipt_id"],
            "route_usable": reason == "ROUTE_USABLE",
            "gate_reason": reason,
            "selected_sample_index": expected_sample["sample_index"] if expected_sample else None,
            "selected_sample_timestamp_ms": expected_sample["timestamp_ms"] if expected_sample else None,
            "selected_valid_until_timestamp_ms": expected_sample["valid_until_timestamp_ms"] if expected_sample else None,
            "selected_route_confidence": expected_sample["confidence"] if expected_sample else None,
            "selected_waypoints": expected_sample["horizon_waypoints"] if expected_sample else [],
            "gate_contract_id": GATE_CONTRACT_ID,
            "unknown_route_policy": UNKNOWN_ROUTE_POLICY,
            "minimum_route_confidence": config["minimum_route_confidence"],
            "maximum_route_age_ms": config["maximum_route_age_ms"],
            "corridor_half_width_frame_ratio": config["corridor_half_width_frame_ratio"],
            "obstacle_footprint_height_ratio": config["obstacle_footprint_height_ratio"],
            "input_detection_count": decoded["detection_count"] if reason == "ROUTE_USABLE" else 0,
            "retained_detection_count": decoded["kernel_input_detection_count"],
        }, where=f"route conditioning frame {index}")
        detections = row.get("detections")
        if not isinstance(detections, list) or len(detections) != row["input_detection_count"]:
            raise AdapterError(f"route conditioning frame {index} detection inventory mismatch")
        if reason != "ROUTE_USABLE":
            if detections or row["retained_detection_count"] != 0:
                raise AdapterError(f"route conditioning frame {index} unknown route did not fail closed")
            continue
        width, height = decoded["width"], decoded["height"]
        route_points = [(width / 2.0, float(height))] + [
            (waypoint["xy_norm"][0] * width, waypoint["xy_norm"][1] * height)
            for waypoint in expected_sample["horizon_waypoints"]
        ]
        corridor = config["corridor_half_width_frame_ratio"] * width
        kept_count = 0
        for detection_index, detection in enumerate(detections):
            if detection.get("detection_index") != detection_index:
                raise AdapterError(f"route conditioning frame {index} detection order mismatch")
            source = detection.get("source_box_xyxy_px")
            footprint = detection.get("footprint_box_xyxy_px")
            if not isinstance(source, list) or len(source) != 4 or not all(isinstance(v, (int, float)) for v in source):
                raise AdapterError(f"route conditioning frame {index} source bbox is invalid")
            if not isinstance(footprint, list) or len(footprint) != 4:
                raise AdapterError(f"route conditioning frame {index} footprint bbox is invalid")
            expected_footprint = [
                source[0], source[3] - (source[3] - source[1]) * config["obstacle_footprint_height_ratio"],
                source[2], source[3],
            ]
            if any(abs(actual - expected) > 1e-4 for actual, expected in zip(footprint, expected_footprint)):
                raise AdapterError(f"route conditioning frame {index} footprint bbox mismatch")
            distance = min(
                _segment_rectangle_distance(start, end, footprint)
                for start, end in zip(route_points, route_points[1:])
            )
            kept = distance <= corridor
            if abs(detection.get("minimum_route_distance_px") - distance) > 1e-4:
                raise AdapterError(f"route conditioning frame {index} route distance mismatch")
            if abs(detection.get("corridor_half_width_px") - corridor) > 1e-4 or detection.get("kept") is not kept:
                raise AdapterError(f"route conditioning frame {index} gate decision mismatch")
            kept_count += int(kept)
        if kept_count != row["retained_detection_count"]:
            raise AdapterError(f"route conditioning frame {index} retained count mismatch")


def validate_device_output(
    output: Mapping[str, Any], *, request: Mapping[str, Any], route: Mapping[str, Any],
    config: Mapping[str, Any], request_sha256: str,
) -> None:
    copied = {key: value for key, value in request.items() if key not in {"schema", "frames"}}
    _require_exact(output, {
        **copied,
        "schema": OUTPUT_SCHEMA,
        "execution_completed": True,
        "failure_count": 0,
        "abstained": False,
    }, where="device output")
    frames = output.get("frames")
    if not isinstance(frames, list) or len(frames) != len(request["frames"]):
        raise AdapterError("device output frame inventory mismatch")
    for index, (actual, requested) in enumerate(zip(frames, request["frames"])):
        if not isinstance(actual, dict) or not isinstance(actual.get("decision"), dict):
            raise AdapterError(f"device output frame {index} lacks decision")
        if any(actual.get(key) != value for key, value in requested.items()):
            raise AdapterError(f"device output frame {index} identity mismatch")
    receipt = output.get("android_backend_receipt")
    if not isinstance(receipt, dict):
        raise AdapterError("device output lacks android backend receipt")
    _require_exact(receipt, {
        "schema": RECEIPT_SCHEMA,
        "backend_id": BACKEND_ID,
        "adapter_output_origin": "on_device_instrumentation_v1",
        "instrumentation_component": INSTRUMENTATION_COMPONENT,
        "instrumentation_test_class": TEST_METHOD,
        "target_package": TARGET_PACKAGE,
        "model_asset_name": config["model_asset_name"],
        "model_asset_sha256": config["model_asset_sha256"],
        "labels_asset_name": config["labels_asset_name"],
        "labels_asset_sha256": config["labels_asset_sha256"],
        "input_size": config["input_size"],
        "confidence_threshold": config["confidence_threshold"],
        "iou_threshold": config["iou_threshold"],
        "detector_runtime": DETECTOR_RUNTIME,
        "shared_decision_kernel_contract_id": request["shared_decision_kernel_contract_id"],
        "device_adapter_implementation_sha256": config["device_adapter_implementation_sha256"],
        "host_adapter_implementation_sha256": config["host_adapter_implementation_sha256"],
        "request_sha256": request_sha256,
        "sanitized_inference_manifest_sha256": request["sanitized_inference_manifest_sha256"],
        "input_video_sha256": request["input_video_sha256"],
        "source_capture_frame_ledger_sha256": request["source_capture_frame_ledger_sha256"],
        "frame_decode_policy": FRAME_DECODE_POLICY,
        "decoded_payload_contract": DECODED_PAYLOAD_CONTRACT,
        "maximum_decode_pts_error_us": config["maximum_decode_pts_error_us"],
        "requested_frame_count": len(request["frames"]),
        "decoded_frame_count": len(request["frames"]),
    }, where="android backend receipt")
    for key in ("target_apk_sha256", "test_apk_sha256", "build_fingerprint_sha256"):
        require_sha(receipt.get(key), where=f"android backend receipt {key}")
    require_text(receipt.get("device_model"), where="android backend receipt device_model")
    decoded = receipt.get("decoded_frames")
    if not isinstance(decoded, list) or len(decoded) != len(request["frames"]):
        raise AdapterError("decoded frame receipt inventory mismatch")
    for index, (row, requested) in enumerate(zip(decoded, request["frames"])):
        if row.get("frame_id") != requested["frame_id"] or row.get("video_pts_ms") != requested["video_pts_ms"]:
            raise AdapterError(f"decoded frame receipt {index} identity mismatch")
        for key in ("decoded_rgba8888_sha256", "encoded_sample_sha256"):
            require_sha(row.get(key), where=f"decoded frame receipt {index}.{key}")
        if not isinstance(row.get("detection_count"), int) or not isinstance(row.get("kernel_input_detection_count"), int):
            raise AdapterError(f"decoded frame receipt {index} detection counts are invalid")
        if row["kernel_input_detection_count"] > row["detection_count"]:
            raise AdapterError(f"decoded frame receipt {index} gate increased detections")
    route_receipt = output.get("route_conditioning_receipt")
    if not isinstance(route_receipt, dict):
        raise AdapterError("device output lacks route conditioning receipt")
    validate_route_receipt(route_receipt, request=request, route=route, config=config, decoded_frames=decoded)


def execute_on_device(
    *, request_path: Path, manifest_path: Path, video_path: Path, ledger_path: Path,
    route_path: Path, artifact_path: Path, threshold_path: Path,
) -> bytes:
    adb = common.locate_adb()
    serial = common.select_serial(adb)
    stage_id = f"u0-route-{sha256_file(request_path)[:16]}-{secrets.token_hex(4)}"
    shell_root = f"/data/local/tmp/blindassist-{stage_id}"
    app_root = f"ustrf-u0/{stage_id}"
    output_relative = f"{app_root}/adapter-output.json"
    video_suffix = video_path.suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,8}", video_path.suffix.lower()) else ".video"
    names = {
        "request": "request.json", "manifest": "inference-manifest.json", "video": f"input-video{video_suffix}",
        "ledger": "capture-frame-ledger.json", "route": "explicit-route-intent.json",
        "artifact": "fold-artifact.bin", "threshold": "threshold-config.json",
    }
    common.run_process(common.adb_command(adb, serial, "shell", "mkdir", "-p", shell_root), timeout=30)
    try:
        common.run_process(
            common.adb_command(adb, serial, "shell", "run-as", TARGET_PACKAGE, "mkdir", "-p", f"files/{app_root}"),
            timeout=30,
        )
        for source, key in (
            (request_path, "request"), (manifest_path, "manifest"), (video_path, "video"),
            (ledger_path, "ledger"), (route_path, "route"), (artifact_path, "artifact"),
            (threshold_path, "threshold"),
        ):
            common.stage_file(adb, serial, source, shell_root=shell_root, app_root=app_root, name=names[key])
        arguments = [
            "shell", "am", "instrument", "-w", "-r", "-e", "class", TEST_METHOD,
            "-e", "ustrfU0BBoxRouteRequired", "true",
            "-e", "ustrfU0Request", f"{app_root}/{names['request']}",
            "-e", "ustrfU0InferenceManifest", f"{app_root}/{names['manifest']}",
            "-e", "ustrfU0Video", f"{app_root}/{names['video']}",
            "-e", "ustrfU0Ledger", f"{app_root}/{names['ledger']}",
            "-e", "ustrfU0ExplicitRoute", f"{app_root}/{names['route']}",
            "-e", "ustrfU0Artifact", f"{app_root}/{names['artifact']}",
            "-e", "ustrfU0ThresholdConfig", f"{app_root}/{names['threshold']}",
            "-e", "ustrfU0Output", output_relative, INSTRUMENTATION_COMPONENT,
        ]
        completed = common.run_process(common.adb_command(adb, serial, *arguments), timeout=270, check=False)
        transcript = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0 or "FAILURES!!!" in transcript or "INSTRUMENTATION_FAILED" in transcript or "OK (1 test)" not in transcript:
            raise AdapterError(f"USTRF bbox-route instrumentation failed: {transcript.strip()}")
        pulled = common.run_process(
            common.adb_command(adb, serial, "exec-out", "run-as", TARGET_PACKAGE, "cat", f"files/{output_relative}"),
            timeout=30, binary=True,
        )
        if not pulled.stdout:
            raise AdapterError("device adapter produced empty output")
        return bytes(pulled.stdout)
    finally:
        common.run_process(common.adb_command(adb, serial, "shell", "rm", "-rf", shell_root), timeout=30, check=False)
        common.run_process(
            common.adb_command(adb, serial, "shell", "run-as", TARGET_PACKAGE, "rm", "-rf", f"files/{app_root}"),
            timeout=30, check=False,
        )


def run_adapter(
    *, request_path: Path, manifest_path: Path, inference_root: Path,
    artifact_path: Path, threshold_path: Path, output_path: Path,
) -> None:
    if output_path.exists():
        raise AdapterError(f"refusing to overwrite adapter output: {output_path}")
    request, _, config, route, video, ledger, route_path = validate_inputs(
        request_path=request_path.resolve(), manifest_path=manifest_path.resolve(),
        inference_root=inference_root.resolve(), artifact_path=artifact_path.resolve(),
        threshold_path=threshold_path.resolve(),
    )
    payload = execute_on_device(
        request_path=request_path.resolve(), manifest_path=manifest_path.resolve(), video_path=video,
        ledger_path=ledger, route_path=route_path, artifact_path=artifact_path.resolve(),
        threshold_path=threshold_path.resolve(),
    )
    try:
        output = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"device output is not UTF-8 JSON: {error}") from error
    if not isinstance(output, dict):
        raise AdapterError("device output must be a JSON object")
    validate_device_output(output, request=request, route=route, config=config, request_sha256=sha256_file(request_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--inference-root", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--threshold-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_adapter(
            request_path=args.request, manifest_path=args.inference_manifest,
            inference_root=args.inference_root, artifact_path=args.artifact,
            threshold_path=args.threshold_config, output_path=args.output,
        )
    except AdapterError as error:
        print(f"USTRF U0 Android bbox-route adapter failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "complete", "arm_id": ARM_ID, "backend_id": BACKEND_ID,
        "output": str(args.output.resolve()), "u0_authority_granted": False,
        "production_model_replacement_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
