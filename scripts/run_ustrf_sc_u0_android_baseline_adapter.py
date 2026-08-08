#!/usr/bin/env python3
"""Run the fixed USTRF U0 baseline through Android TFLite + shared Kotlin kernel.

This is a host transport adapter, not a model implementation. It validates the
sanitized U0 request, stages only hash-bound inference inputs in the target
app's private files directory, invokes one instrumentation class, preserves
the byte-identical on-device JSON output, and fails closed on any ADB, device,
hash, instrumentation, or backend-receipt mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


ARM_ID = "baseline_yolo_geometry"
ADAPTER_ID = "yolo_geometry_adapter_v1"
BACKEND_ID = "android_kotlin_assist_decision_kernel_v1"
REQUEST_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_request_v1"
OUTPUT_SCHEMA = "blindassist_ustrf_sc_u0_candidate_adapter_output_v1"
MANIFEST_SCHEMA = "blindassist_ustrf_sc_u0_sanitized_inference_manifest_v1"
CONFIG_SCHEMA = "blindassist_ustrf_sc_u0_android_baseline_adapter_config_v1"
RECEIPT_SCHEMA = "blindassist_ustrf_sc_u0_android_backend_receipt_v1"
FRAME_DECODE_POLICY = "android_media_metadata_retriever_closest_v1"
DECODED_PAYLOAD_CONTRACT = "rgba8888_row_major_android_getpixels_v1"
DETECTOR_RUNTIME = "tflite_cpu_4_threads_v1"
TARGET_PACKAGE = "com.linnan.blindassist"
INSTRUMENTATION_COMPONENT = (
    "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner"
)
TEST_CLASS = "com.linnan.blindassist.benchmark.UstrfU0BaselineAdapterDeviceTest"
TEST_METHOD = f"{TEST_CLASS}#runBaselineYoloGeometryAdapter"
DEVICE_SOURCE_RELATIVE = Path(
    "apps/benchmarks/device-benchmark/src/main/java/com/linnan/blindassist/benchmark/UstrfU0BaselineAdapterDeviceTest.kt"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class AdapterError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise AdapterError(f"{where} must contain a JSON object")
    return value


def require_sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise AdapterError(f"{where} must be a lowercase SHA-256")
    return value


def require_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{where} must be a non-empty string")
    return value


def confined_file(root: Path, relative: Any, *, where: str) -> Path:
    text = require_text(relative, where=where)
    root = root.resolve()
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AdapterError(f"{where} escapes inference root") from error
    if not path.is_file():
        raise AdapterError(f"{where} is not a file")
    return path


def validate_inputs(
    *,
    request_path: Path,
    manifest_path: Path,
    inference_root: Path,
    artifact_path: Path,
    threshold_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
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
    expected_request = {
        "schema": REQUEST_SCHEMA,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "kernel_execution_backend_id": BACKEND_ID,
        "decision_profile_id": "STANDARD",
        "fit_policy": "fixed_no_fit_v1",
        "event_identity_policy": "kernel_native_optional_v1",
        "route_input_policy": "no_route_input_v1",
        "synthetic_fixture": False,
        "blind_accessed": False,
        "future_inputs_used": False,
        "production_model_replacement_authorized": False,
    }
    for key, expected in expected_request.items():
        if request.get(key) != expected:
            raise AdapterError(f"request {key} mismatch")
    frames = request.get("frames")
    if not isinstance(frames, list) or not frames:
        raise AdapterError("request frames must be non-empty")
    if request.get("decision_cadence", {}).get("canonical_step_ms") != 500:
        raise AdapterError("request decision cadence is not the frozen 500ms grid")
    if sha256_file(manifest_path) != request.get("sanitized_inference_manifest_sha256"):
        raise AdapterError("sanitized inference manifest SHA-256 mismatch")
    expected_manifest = {
        "schema": MANIFEST_SCHEMA,
        "arm_id": ARM_ID,
        "episode_id": request.get("episode_id"),
        "route_input_policy": "no_route_input_v1",
        "adapter_route_input_path": None,
        "adapter_route_input_sha256": None,
        "adapter_route_source_episode_id": None,
        "blind_accessed": False,
        "future_inputs_used": False,
        "review_fields_present": False,
        "adjudication_fields_present": False,
        "event_label_fields_present": False,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise AdapterError(f"sanitized inference manifest {key} mismatch")
    if manifest.get("frames") != frames:
        raise AdapterError("sanitized inference frames differ from request")
    video_path = confined_file(inference_root, manifest.get("input_video_path"), where="input video path")
    ledger_path = confined_file(
        inference_root,
        manifest.get("capture_frame_ledger_path"),
        where="capture frame ledger path",
    )
    video_sha = sha256_file(video_path)
    if video_sha != request.get("input_video_sha256") or video_sha != manifest.get("input_video_sha256"):
        raise AdapterError("input video SHA-256 mismatch")
    ledger_sha = sha256_file(ledger_path)
    if (
        ledger_sha != request.get("source_capture_frame_ledger_sha256")
        or ledger_sha != manifest.get("capture_frame_ledger_sha256")
    ):
        raise AdapterError("capture frame ledger SHA-256 mismatch")
    ledger = load_json(ledger_path, where="capture frame ledger")
    if ledger.get("schema") != "blindassist_capture_frame_ledger_v1" or ledger.get("episode_id") != request.get("episode_id"):
        raise AdapterError("capture frame ledger identity mismatch")
    binding_keys = ("frame_id", "frame_index", "capture_timestamp_ns", "video_pts_ms", "frame_payload_sha256")
    ledger_frames = ledger.get("frames")
    if not isinstance(ledger_frames, list) or len(ledger_frames) != len(frames):
        raise AdapterError("capture frame ledger inventory mismatch")
    for index, (ledger_frame, request_frame) in enumerate(zip(ledger_frames, frames)):
        if not isinstance(ledger_frame, dict) or any(ledger_frame.get(key) != request_frame.get(key) for key in binding_keys):
            raise AdapterError(f"capture frame ledger frame {index} differs from request")
    if sha256_file(artifact_path) != request.get("fold_artifact_sha256"):
        raise AdapterError("fixed-arm fold artifact SHA-256 mismatch")
    if sha256_file(threshold_path) != request.get("threshold_config_sha256"):
        raise AdapterError("threshold config SHA-256 mismatch")
    expected_config = {
        "schema": CONFIG_SCHEMA,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "kernel_execution_backend_id": BACKEND_ID,
        "decision_profile_id": "STANDARD",
        "model_asset_name": "yolo11n_fp16_320.tflite",
        "labels_asset_name": "coco_labels.txt",
        "input_size": 320,
        "confidence_threshold": 0.35,
        "iou_threshold": 0.45,
        "detector_runtime": DETECTOR_RUNTIME,
        "frame_decode_policy": FRAME_DECODE_POLICY,
        "decoded_payload_contract": DECODED_PAYLOAD_CONTRACT,
        "instrumentation_component": INSTRUMENTATION_COMPONENT,
        "instrumentation_test_class": TEST_METHOD,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise AdapterError(f"threshold config {key} mismatch")
    require_sha(config.get("model_asset_sha256"), where="threshold config model_asset_sha256")
    require_sha(config.get("labels_asset_sha256"), where="threshold config labels_asset_sha256")
    maximum_decode_pts_error_us = config.get("maximum_decode_pts_error_us")
    if (
        not isinstance(maximum_decode_pts_error_us, int)
        or isinstance(maximum_decode_pts_error_us, bool)
        or not 0 <= maximum_decode_pts_error_us <= 20_000
    ):
        raise AdapterError("threshold config maximum_decode_pts_error_us must be an integer in [0, 20000]")
    require_sha(
        config.get("device_adapter_implementation_sha256"),
        where="threshold config device_adapter_implementation_sha256",
    )
    expected_host_sha = normalized_text_sha256(Path(__file__).resolve())
    if config.get("host_adapter_implementation_sha256") != expected_host_sha:
        raise AdapterError("host adapter implementation differs from threshold config")
    device_source = Path(__file__).resolve().parents[1] / DEVICE_SOURCE_RELATIVE
    if not device_source.is_file() or normalized_text_sha256(device_source) != config.get(
        "device_adapter_implementation_sha256"
    ):
        raise AdapterError("Android device adapter implementation differs from threshold config")
    return request, manifest, config, video_path, ledger_path


def parse_connected_devices(output: str) -> list[str]:
    devices: list[str] = []
    for raw in output.splitlines()[1:]:
        fields = raw.strip().split()
        if len(fields) >= 2 and fields[1] == "device":
            devices.append(fields[0])
    return devices


def locate_adb() -> Path:
    override = os.environ.get("USTRF_U0_ADB")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    repo_adb = Path(__file__).resolve().parents[1] / ".android-sdk" / "platform-tools" / (
        "adb.exe" if os.name == "nt" else "adb"
    )
    candidates.append(repo_adb)
    discovered = shutil.which("adb")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise AdapterError("adb not found; set USTRF_U0_ADB or restore the project Android SDK junction")


def run_process(
    command: Sequence[str],
    *,
    timeout: int,
    binary: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[Any]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
            errors=None if binary else "replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AdapterError(f"command failed to execute: {command[0]}: {error}") from error
    if check and completed.returncode != 0:
        stderr = completed.stderr if isinstance(completed.stderr, str) else completed.stderr.decode("utf-8", "replace")
        raise AdapterError(f"command exited {completed.returncode}: {' '.join(command[:4])}: {stderr.strip()}")
    return completed


def select_serial(adb: Path) -> str:
    listed = run_process([str(adb), "devices", "-l"], timeout=15)
    devices = parse_connected_devices(listed.stdout)
    requested = os.environ.get("ANDROID_SERIAL")
    if requested:
        if requested not in devices:
            raise AdapterError(f"ANDROID_SERIAL is not an online device: {requested}")
        return requested
    if len(devices) != 1:
        raise AdapterError(f"expected exactly one online Android device, found {len(devices)}")
    return devices[0]


def adb_command(adb: Path, serial: str, *arguments: str) -> list[str]:
    return [str(adb), "-s", serial, *arguments]


def stage_file(adb: Path, serial: str, source: Path, *, shell_root: str, app_root: str, name: str) -> None:
    shell_path = f"{shell_root}/{name}"
    app_path = f"files/{app_root}/{name}"
    run_process(adb_command(adb, serial, "push", str(source.resolve()), shell_path), timeout=120)
    run_process(
        adb_command(adb, serial, "shell", "run-as", TARGET_PACKAGE, "cp", shell_path, app_path),
        timeout=30,
    )


def validate_device_output(
    output: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    config: Mapping[str, Any],
    request_sha256: str,
) -> None:
    copied = {key: value for key, value in request.items() if key not in {"schema", "frames"}}
    expected = {
        **copied,
        "schema": OUTPUT_SCHEMA,
        "execution_completed": True,
        "failure_count": 0,
        "abstained": False,
    }
    for key, value in expected.items():
        if output.get(key) != value:
            raise AdapterError(f"device output {key} mismatch")
    output_frames = output.get("frames")
    request_frames = request.get("frames")
    if not isinstance(output_frames, list) or len(output_frames) != len(request_frames):
        raise AdapterError("device output does not contain every requested frame")
    for index, (actual, requested) in enumerate(zip(output_frames, request_frames)):
        if not isinstance(actual, dict) or not isinstance(actual.get("decision"), dict):
            raise AdapterError(f"device output frame {index} lacks a decision")
        for key, value in requested.items():
            if actual.get(key) != value:
                raise AdapterError(f"device output frame {index}.{key} mismatch")
    receipt = output.get("android_backend_receipt")
    if not isinstance(receipt, dict):
        raise AdapterError("device output lacks android_backend_receipt")
    receipt_expected = {
        "schema": RECEIPT_SCHEMA,
        "backend_id": BACKEND_ID,
        "adapter_output_origin": "on_device_instrumentation_v1",
        "instrumentation_component": INSTRUMENTATION_COMPONENT,
        "instrumentation_test_class": TEST_METHOD,
        "target_package": TARGET_PACKAGE,
        "model_asset_name": config.get("model_asset_name"),
        "model_asset_sha256": config.get("model_asset_sha256"),
        "labels_asset_name": config.get("labels_asset_name"),
        "labels_asset_sha256": config.get("labels_asset_sha256"),
        "input_size": config.get("input_size"),
        "confidence_threshold": config.get("confidence_threshold"),
        "iou_threshold": config.get("iou_threshold"),
        "detector_runtime": DETECTOR_RUNTIME,
        "shared_decision_kernel_contract_id": request.get("shared_decision_kernel_contract_id"),
        "device_adapter_implementation_sha256": config.get("device_adapter_implementation_sha256"),
        "host_adapter_implementation_sha256": config.get("host_adapter_implementation_sha256"),
        "request_sha256": request_sha256,
        "sanitized_inference_manifest_sha256": request.get("sanitized_inference_manifest_sha256"),
        "input_video_sha256": request.get("input_video_sha256"),
        "source_capture_frame_ledger_sha256": request.get("source_capture_frame_ledger_sha256"),
        "frame_decode_policy": FRAME_DECODE_POLICY,
        "decoded_payload_contract": DECODED_PAYLOAD_CONTRACT,
        "maximum_decode_pts_error_us": config.get("maximum_decode_pts_error_us"),
        "requested_frame_count": len(request_frames),
        "decoded_frame_count": len(request_frames),
    }
    for key, value in receipt_expected.items():
        if receipt.get(key) != value:
            raise AdapterError(f"android backend receipt {key} mismatch")
    require_text(receipt.get("device_model"), where="android backend receipt device_model")
    require_text(receipt.get("target_version_name"), where="android backend receipt target_version_name")
    if not isinstance(receipt.get("target_version_code"), int) or isinstance(receipt.get("target_version_code"), bool):
        raise AdapterError("android backend receipt target_version_code must be an integer")
    for key in ("target_apk_sha256", "test_apk_sha256"):
        require_sha(receipt.get(key), where=f"android backend receipt {key}")
    if (
        not isinstance(receipt.get("device_api_level"), int)
        or isinstance(receipt.get("device_api_level"), bool)
        or receipt["device_api_level"] <= 0
    ):
        raise AdapterError("android backend receipt device_api_level must be positive integer")
    require_sha(receipt.get("build_fingerprint_sha256"), where="android backend receipt build_fingerprint_sha256")
    decoded = receipt.get("decoded_frames")
    if not isinstance(decoded, list) or len(decoded) != len(request_frames):
        raise AdapterError("android backend receipt decoded frame inventory mismatch")
    for index, (row, requested) in enumerate(zip(decoded, request_frames)):
        if not isinstance(row, dict):
            raise AdapterError(f"decoded frame receipt {index} must be an object")
        if row.get("frame_id") != requested.get("frame_id") or row.get("video_pts_ms") != requested.get("video_pts_ms"):
            raise AdapterError(f"decoded frame receipt {index} identity mismatch")
        require_sha(row.get("decoded_rgba8888_sha256"), where=f"decoded frame receipt {index} bitmap")
        require_sha(row.get("encoded_sample_sha256"), where=f"decoded frame receipt {index} encoded sample")
        expected_pts_us = requested.get("video_pts_ms") * 1_000
        if row.get("requested_pts_us") != expected_pts_us:
            raise AdapterError(f"decoded frame receipt {index}.requested_pts_us mismatch")
        selected_pts_us = row.get("selected_source_pts_us")
        pts_error_us = row.get("pts_error_us")
        if not isinstance(selected_pts_us, int) or isinstance(selected_pts_us, bool) or selected_pts_us < 0:
            raise AdapterError(f"decoded frame receipt {index}.selected_source_pts_us must be non-negative integer")
        if (
            not isinstance(pts_error_us, int)
            or isinstance(pts_error_us, bool)
            or pts_error_us != abs(selected_pts_us - expected_pts_us)
            or pts_error_us > config.get("maximum_decode_pts_error_us")
        ):
            raise AdapterError(f"decoded frame receipt {index}.pts_error_us violates config")
        for key in ("width", "height"):
            if not isinstance(row.get(key), int) or isinstance(row.get(key), bool) or row[key] <= 0:
                raise AdapterError(f"decoded frame receipt {index}.{key} must be positive integer")
        for key in ("decode_duration_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "total_detect_ms"):
            if not isinstance(row.get(key), (int, float)) or isinstance(row.get(key), bool) or row[key] < 0:
                raise AdapterError(f"decoded frame receipt {index}.{key} must be non-negative number")
        if not isinstance(row.get("detection_count"), int) or isinstance(row.get("detection_count"), bool) or row["detection_count"] < 0:
            raise AdapterError(f"decoded frame receipt {index}.detection_count must be non-negative integer")


def build_instrumentation_command(
    *, adb: Path, serial: str, app_root: str, names: Mapping[str, str], output_relative: str
) -> list[str]:
    return adb_command(
        adb,
        serial,
        "shell",
        "am",
        "instrument",
        "-w",
        "-r",
        "-e",
        "class",
        TEST_METHOD,
        "-e",
        "ustrfU0BaselineRequired",
        "true",
        "-e",
        "ustrfU0Request",
        f"{app_root}/{names['request']}",
        "-e",
        "ustrfU0InferenceManifest",
        f"{app_root}/{names['manifest']}",
        "-e",
        "ustrfU0Video",
        f"{app_root}/{names['video']}",
        "-e",
        "ustrfU0Ledger",
        f"{app_root}/{names['ledger']}",
        "-e",
        "ustrfU0Artifact",
        f"{app_root}/{names['artifact']}",
        "-e",
        "ustrfU0ThresholdConfig",
        f"{app_root}/{names['threshold']}",
        "-e",
        "ustrfU0Output",
        output_relative,
        INSTRUMENTATION_COMPONENT,
    )


def execute_on_device(
    *,
    adb: Path,
    serial: str,
    request_path: Path,
    manifest_path: Path,
    video_path: Path,
    ledger_path: Path,
    artifact_path: Path,
    threshold_path: Path,
) -> bytes:
    stage_id = f"u0-{sha256_file(request_path)[:16]}-{secrets.token_hex(4)}"
    shell_root = f"/data/local/tmp/blindassist-{stage_id}"
    app_root = f"ustrf-u0/{stage_id}"
    output_relative = f"{app_root}/adapter-output.json"
    video_suffix = video_path.suffix.lower() if video_path.suffix else ".video"
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", video_suffix):
        video_suffix = ".video"
    names = {
        "request": "request.json",
        "manifest": "inference-manifest.json",
        "video": f"input-video{video_suffix}",
        "ledger": "capture-frame-ledger.json",
        "artifact": "fold-artifact.bin",
        "threshold": "threshold-config.json",
    }
    run_process(adb_command(adb, serial, "shell", "mkdir", "-p", shell_root), timeout=30)
    try:
        run_process(
            adb_command(adb, serial, "shell", "run-as", TARGET_PACKAGE, "mkdir", "-p", f"files/{app_root}"),
            timeout=30,
        )
        for source, key in (
            (request_path, "request"),
            (manifest_path, "manifest"),
            (video_path, "video"),
            (ledger_path, "ledger"),
            (artifact_path, "artifact"),
            (threshold_path, "threshold"),
        ):
            stage_file(adb, serial, source, shell_root=shell_root, app_root=app_root, name=names[key])
        instrument = build_instrumentation_command(
            adb=adb,
            serial=serial,
            app_root=app_root,
            names=names,
            output_relative=output_relative,
        )
        completed = run_process(instrument, timeout=270, check=False)
        transcript = f"{completed.stdout}\n{completed.stderr}"
        if (
            completed.returncode != 0
            or "FAILURES!!!" in transcript
            or "INSTRUMENTATION_FAILED" in transcript
            or "OK (1 test)" not in transcript
        ):
            raise AdapterError(f"USTRF baseline instrumentation failed: {transcript.strip()}")
        pulled = run_process(
            adb_command(
                adb,
                serial,
                "exec-out",
                "run-as",
                TARGET_PACKAGE,
                "cat",
                f"files/{output_relative}",
            ),
            timeout=30,
            binary=True,
        )
        if not pulled.stdout:
            raise AdapterError("device adapter produced an empty output")
        return bytes(pulled.stdout)
    finally:
        run_process(adb_command(adb, serial, "shell", "rm", "-rf", shell_root), timeout=30, check=False)
        run_process(
            adb_command(adb, serial, "shell", "run-as", TARGET_PACKAGE, "rm", "-rf", f"files/{app_root}"),
            timeout=30,
            check=False,
        )


def run_adapter(
    *,
    request_path: Path,
    manifest_path: Path,
    inference_root: Path,
    artifact_path: Path,
    threshold_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise AdapterError(f"refusing to overwrite adapter output: {output_path}")
    request, _, config, video_path, ledger_path = validate_inputs(
        request_path=request_path.resolve(),
        manifest_path=manifest_path.resolve(),
        inference_root=inference_root.resolve(),
        artifact_path=artifact_path.resolve(),
        threshold_path=threshold_path.resolve(),
    )
    adb = locate_adb()
    serial = select_serial(adb)
    payload = execute_on_device(
        adb=adb,
        serial=serial,
        request_path=request_path.resolve(),
        manifest_path=manifest_path.resolve(),
        video_path=video_path,
        ledger_path=ledger_path,
        artifact_path=artifact_path.resolve(),
        threshold_path=threshold_path.resolve(),
    )
    try:
        output = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"device output is not UTF-8 JSON: {error}") from error
    if not isinstance(output, dict):
        raise AdapterError("device output must be a JSON object")
    validate_device_output(
        output,
        request=request,
        config=config,
        request_sha256=sha256_file(request_path),
    )
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
            request_path=args.request,
            manifest_path=args.inference_manifest,
            inference_root=args.inference_root,
            artifact_path=args.artifact,
            threshold_path=args.threshold_config,
            output_path=args.output,
        )
    except AdapterError as error:
        print(f"USTRF U0 Android baseline adapter failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "complete",
        "arm_id": ARM_ID,
        "backend_id": BACKEND_ID,
        "output": str(args.output.resolve()),
        "u0_authority_granted": False,
        "production_model_replacement_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
