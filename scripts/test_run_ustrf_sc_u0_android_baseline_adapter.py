from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import run_ustrf_sc_u0_android_baseline_adapter as subject


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class UstrfU0AndroidBaselineAdapterTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict, dict, dict, dict, dict]:
        video = root / "inference/video.mp4"
        ledger_path = root / "inference/capture-frame-ledger.json"
        artifact = root / "artifact.bin"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"protocol-video")
        artifact.write_bytes(b"fixed-no-fit")
        frames = [{
            "frame_id": "f-000",
            "frame_index": 0,
            "capture_timestamp_ns": 1,
            "video_pts_ms": 0,
            "frame_payload_sha256": "1" * 64,
        }]
        save(ledger_path, {
            "schema": "blindassist_capture_frame_ledger_v1",
            "episode_id": "episode-1",
            "frames": frames,
        })
        manifest = {
            "schema": subject.MANIFEST_SCHEMA,
            "arm_id": subject.ARM_ID,
            "episode_id": "episode-1",
            "input_video_path": "inference/video.mp4",
            "input_video_sha256": sha(video),
            "capture_frame_ledger_path": "inference/capture-frame-ledger.json",
            "capture_frame_ledger_sha256": sha(ledger_path),
            "route_input_policy": "no_route_input_v1",
            "adapter_route_input_path": None,
            "adapter_route_input_sha256": None,
            "adapter_route_source_episode_id": None,
            "frames": frames,
            "blind_accessed": False,
            "future_inputs_used": False,
            "review_fields_present": False,
            "adjudication_fields_present": False,
            "event_label_fields_present": False,
        }
        manifest_path = root / "manifest.json"
        save(manifest_path, manifest)
        config = {
            "schema": subject.CONFIG_SCHEMA,
            "arm_id": subject.ARM_ID,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "kernel_execution_backend_id": subject.BACKEND_ID,
            "decision_profile_id": "STANDARD",
            "model_asset_name": "yolo11n_fp16_320.tflite",
            "model_asset_sha256": "2" * 64,
            "labels_asset_name": "coco_labels.txt",
            "labels_asset_sha256": "3" * 64,
            "input_size": 320,
            "confidence_threshold": 0.35,
            "iou_threshold": 0.45,
            "detector_runtime": subject.DETECTOR_RUNTIME,
            "frame_decode_policy": subject.FRAME_DECODE_POLICY,
            "decoded_payload_contract": subject.DECODED_PAYLOAD_CONTRACT,
            "maximum_decode_pts_error_us": 20_000,
            "instrumentation_component": subject.INSTRUMENTATION_COMPONENT,
            "instrumentation_test_class": subject.TEST_METHOD,
            "device_adapter_implementation_sha256": subject.normalized_text_sha256(
                Path(subject.__file__).resolve().parents[1] / subject.DEVICE_SOURCE_RELATIVE
            ),
            "host_adapter_implementation_sha256": subject.normalized_text_sha256(Path(subject.__file__).resolve()),
        }
        config_path = root / "threshold.json"
        save(config_path, config)
        request = {
            "schema": subject.REQUEST_SCHEMA,
            "contract_id": "ustrf_sc_u0_teacher_upper_bound_v1",
            "arm_id": subject.ARM_ID,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "adapter_runtime_id": "python_subprocess_v1",
            "shared_decision_kernel_contract_id": "blindassist_shared_decision_kernel_v1",
            "shared_decision_kernel_implementation_sha256": "5" * 64,
            "kernel_execution_backend_id": subject.BACKEND_ID,
            "decision_profile_id": "STANDARD",
            "feedback_adapter_id": "offline_u0_feedback_receipt_v1",
            "kernel_trace_order": ["analyzer", "temporal", "stabilizer", "event", "confirmation", "feedback"],
            "decision_cadence": {"canonical_step_ms": 500},
            "fit_policy": "fixed_no_fit_v1",
            "event_identity_policy": "kernel_native_optional_v1",
            "route_input_policy": "no_route_input_v1",
            "implementation_sha256": "6" * 64,
            "artifact_inventory_sha256": "7" * 64,
            "threshold_config_sha256": sha(config_path),
            "fold_held_out_session_id": "session-1",
            "fold_artifact_sha256": sha(artifact),
            "fold_training_input_manifest_sha256": "8" * 64,
            "fold_training_receipt_sha256": "9" * 64,
            "episode_id": "episode-1",
            "sanitized_inference_manifest_sha256": sha(manifest_path),
            "input_video_sha256": sha(video),
            "truth_route_intent_sha256": "a" * 64,
            "adapter_route_input_sha256": None,
            "adapter_route_source_episode_id": None,
            "source_capture_frame_ledger_sha256": sha(ledger_path),
            "synthetic_fixture": False,
            "blind_accessed": False,
            "future_inputs_used": False,
            "production_model_replacement_authorized": False,
            "frames": frames,
        }
        request_path = root / "request.json"
        save(request_path, request)
        decision = {
            "raw_risk_level": "NONE",
            "stable_risk_level": "NONE",
            "event_id": None,
            "event_state": None,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "feedback_receipt": {
                "outcome": "NO_ALERT",
                "kernel_feedback_reason": "NO_FEEDBACK_RISK",
                "delivered": False,
                "adapter_id": "offline_u0_feedback_receipt_v1",
                "kernel_contract_id": "blindassist_shared_decision_kernel_v1",
            },
        }
        decoded = {
            "frame_id": "f-000",
            "video_pts_ms": 0,
            "requested_pts_us": 0,
            "selected_source_pts_us": 1_000,
            "pts_error_us": 1_000,
            "decode_duration_ms": 2.5,
            "decoded_rgba8888_sha256": "c" * 64,
            "encoded_sample_sha256": "0" * 64,
            "width": 320,
            "height": 240,
            "preprocess_ms": 1,
            "inference_ms": 10,
            "postprocess_ms": 1,
            "total_detect_ms": 12,
            "detection_count": 0,
        }
        receipt = {
            "schema": subject.RECEIPT_SCHEMA,
            "backend_id": subject.BACKEND_ID,
            "adapter_output_origin": "on_device_instrumentation_v1",
            "instrumentation_component": subject.INSTRUMENTATION_COMPONENT,
            "instrumentation_test_class": subject.TEST_METHOD,
            "target_package": subject.TARGET_PACKAGE,
            "target_version_name": "1.0",
            "target_version_code": 1,
            "target_apk_sha256": "e" * 64,
            "test_apk_sha256": "f" * 64,
            "device_model": "device",
            "device_api_level": 35,
            "build_fingerprint_sha256": "d" * 64,
            "model_asset_name": config["model_asset_name"],
            "model_asset_sha256": config["model_asset_sha256"],
            "labels_asset_name": config["labels_asset_name"],
            "labels_asset_sha256": config["labels_asset_sha256"],
            "input_size": 320,
            "confidence_threshold": 0.35,
            "iou_threshold": 0.45,
            "detector_runtime": subject.DETECTOR_RUNTIME,
            "shared_decision_kernel_contract_id": "blindassist_shared_decision_kernel_v1",
            "device_adapter_implementation_sha256": config["device_adapter_implementation_sha256"],
            "host_adapter_implementation_sha256": config["host_adapter_implementation_sha256"],
            "request_sha256": sha(request_path),
            "sanitized_inference_manifest_sha256": sha(manifest_path),
            "input_video_sha256": sha(video),
            "source_capture_frame_ledger_sha256": sha(ledger_path),
            "frame_decode_policy": subject.FRAME_DECODE_POLICY,
            "decoded_payload_contract": subject.DECODED_PAYLOAD_CONTRACT,
            "maximum_decode_pts_error_us": 20_000,
            "requested_frame_count": 1,
            "decoded_frame_count": 1,
            "decoded_frames": [decoded],
        }
        output = {key: value for key, value in request.items() if key not in {"schema", "frames"}}
        output.update({
            "schema": subject.OUTPUT_SCHEMA,
            "execution_completed": True,
            "failure_count": 0,
            "abstained": False,
            "frames": [{**frames[0], "decision": decision}],
            "android_backend_receipt": receipt,
        })
        return request, manifest, config, output, receipt

    def test_input_and_device_receipt_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, _, config, output, _ = self.fixture(root)
            loaded, _, loaded_config, video, ledger = subject.validate_inputs(
                request_path=root / "request.json",
                manifest_path=root / "manifest.json",
                inference_root=root,
                artifact_path=root / "artifact.bin",
                threshold_path=root / "threshold.json",
            )
            self.assertEqual(request, loaded)
            self.assertEqual(config, loaded_config)
            self.assertEqual(root / "inference/video.mp4", video)
            self.assertEqual(root / "inference/capture-frame-ledger.json", ledger)
            subject.validate_device_output(
                json.loads(json.dumps(output)),
                request=loaded,
                config=loaded_config,
                request_sha256=sha(root / "request.json"),
            )

    def test_label_injection_and_pts_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, manifest, config, output, _ = self.fixture(root)
            manifest["event_label_fields_present"] = True
            save(root / "manifest.json", manifest)
            request["sanitized_inference_manifest_sha256"] = sha(root / "manifest.json")
            save(root / "request.json", request)
            with self.assertRaisesRegex(subject.AdapterError, "event_label_fields_present"):
                subject.validate_inputs(
                    request_path=root / "request.json",
                    manifest_path=root / "manifest.json",
                    inference_root=root,
                    artifact_path=root / "artifact.bin",
                    threshold_path=root / "threshold.json",
                )
            output["sanitized_inference_manifest_sha256"] = request["sanitized_inference_manifest_sha256"]
            output["android_backend_receipt"]["sanitized_inference_manifest_sha256"] = request[
                "sanitized_inference_manifest_sha256"
            ]
            output["android_backend_receipt"]["request_sha256"] = sha(root / "request.json")
            output["android_backend_receipt"]["decoded_frames"][0]["pts_error_us"] = 20_001
            with self.assertRaisesRegex(subject.AdapterError, "pts_error_us"):
                subject.validate_device_output(
                    output,
                    request=request,
                    config=config,
                    request_sha256=sha(root / "request.json"),
                )

    def test_connected_device_parser_ignores_offline_and_unauthorized(self) -> None:
        output = "List of devices attached\nserial-a device product:x\nserial-b offline\nserial-c unauthorized\n"
        self.assertEqual(["serial-a"], subject.parse_connected_devices(output))

    def test_instrumentation_command_selects_only_baseline_method(self) -> None:
        names = {
            "request": "request.json",
            "manifest": "manifest.json",
            "video": "video.mp4",
            "ledger": "ledger.json",
            "artifact": "artifact.bin",
            "threshold": "threshold.json",
        }
        command = subject.build_instrumentation_command(
            adb=Path("adb"),
            serial="serial-a",
            app_root="ustrf-u0/stage",
            names=names,
            output_relative="ustrf-u0/stage/output.json",
        )
        class_index = command.index("class")
        self.assertEqual(subject.TEST_METHOD, command[class_index + 1])
        self.assertNotIn(subject.TEST_CLASS, command)


if __name__ == "__main__":
    unittest.main()
