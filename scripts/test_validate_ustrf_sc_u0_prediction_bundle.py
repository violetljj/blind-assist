from __future__ import annotations

import base64
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import validate_ustrf_sc_u0_prediction_bundle as subject
import test_run_ustrf_sc_u0_android_baseline_adapter as android_fixture


SCRIPTS = Path(__file__).resolve().parent
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "u0_evaluator_fixture",
    SCRIPTS / "test_evaluate_ustrf_sc_u0_teacher_upper_bound.py",
)
assert FIXTURE_SPEC and FIXTURE_SPEC.loader
fixture_module = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture_module)


def dense_field_fixture() -> tuple[dict, dict]:
    scale = subject.DENSE_SCALE
    active = [True] * 4 + [False] * 12

    def encoded(values: list[int]) -> str:
        return base64.b64encode(b"".join(struct.pack("<I", value) for value in values)).decode("ascii")

    payload = {
        "grid_width": 4,
        "grid_height": 4,
        "quantization": subject.DENSE_QUANTIZATION,
        "fields_base64": {
            "local_obstacle_field": encoded([800_000 if value else 0 for value in active]),
            "walkability_field": encoded([200_000 if value else scale for value in active]),
            "boundary_field": encoded([0] * 16),
            "unknown_field": encoded([100_000 if value else 0 for value in active]),
            "route_weight_field": encoded([scale if value else 0 for value in active]),
            "route_relative_risk_field": encoded([400_000 if value else 0 for value in active]),
        },
    }
    return payload, subject._dense_summary_from_payload(payload, where="fixture")


class UstrfU0PredictionBundleTest(unittest.TestCase):
    def test_valid_bundle_recomputes_every_frame_and_alert(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "prediction-bundle"
            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            report = subject.validate_bundle(
                contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle,
            )
            self.assertEqual(6, report["arm_count"])
            self.assertEqual(12, report["episode_trace_count"])
            self.assertEqual(252, report["frame_trace_count"])
            self.assertEqual(2, report["trace_derived_alert_count"])
            self.assertTrue(report["all_alerts_trace_derived"])
            self.assertFalse(report["u0_authority_granted"])

    def test_manual_alert_summary_and_file_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "prediction-bundle"
            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            causal = next(row for row in predictions["arms"] if row["arm_id"] == "teacher_dense_explicit_route_causal")
            positive = next(row for row in causal["episodes"] if row["alert_timestamps_ms"])
            positive["alert_timestamps_ms"] = [1000]
            with self.assertRaisesRegex(subject.ContractError, "differs from trace-derived"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            dependency = predictions["shared_decision_kernel_dependencies"][0]
            dependency_path = bundle / dependency["path"]
            dependency_path.write_text(dependency_path.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
            with self.assertRaisesRegex(subject.ContractError, "kernel_dependencies.*SHA-256 mismatch"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            arm = predictions["arms"][0]
            artifact = bundle / arm["artifact_path"]
            artifact.write_bytes(artifact.read_bytes() + b"tamper")
            with self.assertRaisesRegex(subject.ContractError, "artifact_inventory SHA-256 mismatch"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            arm = predictions["arms"][0]
            trace = bundle / arm["episodes"][0]["prediction_trace_path"]
            trace.write_text(trace.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(subject.ContractError, "prediction_trace SHA-256 mismatch"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

    def test_missing_frame_adapter_drift_and_execution_failure_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "prediction-bundle"
            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            arm = predictions["arms"][0]
            summary = arm["episodes"][0]
            trace_path = bundle / summary["prediction_trace_path"]
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["frames"].pop()
            trace_path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")
            summary["prediction_trace_sha256"] = fixture_module.sha(trace_path)
            receipt_path = bundle / arm["execution_receipt_path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["prediction_trace_sha256_by_episode"][summary["episode_id"]] = summary["prediction_trace_sha256"]
            receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            arm["execution_receipt_sha256"] = fixture_module.sha(receipt_path)
            with self.assertRaisesRegex(subject.ContractError, "every truth frame exactly once"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            predictions["arms"][0]["candidate_adapter_id"] = "bypass_adapter"
            with self.assertRaisesRegex(subject.ContractError, "preregistered adapter"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            arm = predictions["arms"][0]
            receipt_path = bundle / arm["execution_receipt_path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["failure_count"] = 1
            receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            arm["execution_receipt_sha256"] = fixture_module.sha(receipt_path)
            with self.assertRaisesRegex(subject.ContractError, "failure_count mismatch"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

            contract, _, truth_manifest, predictions, _, _ = fixture_module.UstrfU0TeacherUpperBoundTest.fixture(root)
            arm = predictions["arms"][0]
            summary = arm["episodes"][0]
            manifest_path = bundle / summary["sanitized_inference_manifest_path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["should_alert"] = True
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            summary["sanitized_inference_manifest_sha256"] = fixture_module.sha(manifest_path)
            with self.assertRaisesRegex(subject.ContractError, "key inventory is not sanitized"):
                subject.validate_bundle(contract, truth_manifest, predictions, truth_root=root, prediction_root=bundle)

    def test_kernel_native_event_identity_and_feedback_reason_mapping(self) -> None:
        contract = json.loads((SCRIPTS.parent / "configs/ustrf_sc_u0_teacher_upper_bound_v1.json").read_text(encoding="utf-8"))
        evidence = contract["prediction_evidence_contract"]
        decision = {
            "raw_risk_level": "MEDIUM",
            "stable_risk_level": "MEDIUM",
            "event_id": None,
            "event_state": None,
            "candidate_adapter_id": "yolo_geometry_adapter_v1",
            "feedback_receipt": {
                "outcome": "TRIGGERED",
                "kernel_feedback_reason": "TRIGGERED",
                "delivered": True,
                "adapter_id": evidence["feedback_adapter_id"],
                "kernel_contract_id": evidence["shared_decision_kernel_contract_id"],
            },
        }
        self.assertTrue(subject._validate_decision(
            decision,
            contract=evidence,
            adapter_id="yolo_geometry_adapter_v1",
            event_identity_policy="kernel_native_optional_v1",
            where="optional",
        ))
        with self.assertRaisesRegex(subject.ContractError, "kernel-native event identity"):
            subject._validate_decision(
                decision,
                contract=evidence,
                adapter_id="yolo_geometry_adapter_v1",
                event_identity_policy="kernel_native_required_v1",
                where="required",
            )
        invalid_state = json.loads(json.dumps(decision))
        invalid_state["event_id"] = "seg-1"
        invalid_state["event_state"] = "ACTIVE"
        with self.assertRaisesRegex(subject.ContractError, "event_state"):
            subject._validate_decision(
                invalid_state,
                contract=evidence,
                adapter_id="yolo_geometry_adapter_v1",
                event_identity_policy="kernel_native_optional_v1",
                where="state",
            )
        invalid_reason = json.loads(json.dumps(decision))
        invalid_reason["feedback_receipt"]["kernel_feedback_reason"] = "NO_FEEDBACK_RISK"
        with self.assertRaisesRegex(subject.ContractError, "reason/outcome"):
            subject._validate_decision(
                invalid_reason,
                contract=evidence,
                adapter_id="yolo_geometry_adapter_v1",
                event_identity_policy="kernel_native_optional_v1",
                where="reason",
            )

    def test_android_backend_receipt_is_required_and_pts_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, _, threshold, output, _ = android_fixture.UstrfU0AndroidBaselineAdapterTest().fixture(root)
            contract = json.loads(
                (SCRIPTS.parent / "configs/ustrf_sc_u0_teacher_upper_bound_v1.json").read_text(encoding="utf-8")
            )
            evidence = contract["prediction_evidence_contract"]
            subject._validate_android_backend_receipt(
                output,
                request=request,
                request_sha256=android_fixture.sha(root / "request.json"),
                threshold_config=threshold,
                evidence=evidence,
                where="android",
            )
            missing = json.loads(json.dumps(output))
            missing.pop("android_backend_receipt")
            with self.assertRaisesRegex(subject.ContractError, "is required"):
                subject._validate_android_backend_receipt(
                    missing,
                    request=request,
                    request_sha256=android_fixture.sha(root / "request.json"),
                    threshold_config=threshold,
                    evidence=evidence,
                    where="android",
                )
            tampered = json.loads(json.dumps(output))
            tampered["android_backend_receipt"]["decoded_frames"][0]["selected_source_pts_us"] = 25_000
            with self.assertRaisesRegex(subject.ContractError, "PTS binding"):
                subject._validate_android_backend_receipt(
                    tampered,
                    request=request,
                    request_sha256=android_fixture.sha(root / "request.json"),
                    threshold_config=threshold,
                    evidence=evidence,
                    where="android",
                )

    def test_bbox_route_receipt_is_causal_gate_bound_and_required_on_android(self) -> None:
        contract = json.loads(
            (SCRIPTS.parent / "configs/ustrf_sc_u0_teacher_upper_bound_v1.json").read_text(encoding="utf-8")
        )
        evidence = contract["prediction_evidence_contract"]
        threshold = {
            "route_gate_contract_id": "bbox_bottom_footprint_polyline_corridor_v1",
            "route_gate_implementation_sha256": "a" * 64,
            "unknown_route_policy": "context_attention_only_empty_detection_gate_v1",
            "minimum_route_confidence": 0.5,
            "maximum_route_age_ms": 1000,
            "corridor_half_width_frame_ratio": 0.08,
            "obstacle_footprint_height_ratio": 0.25,
        }
        request = {
            "arm_id": "detector_bbox_explicit_route",
            "candidate_adapter_id": "detector_bbox_explicit_route_adapter_v1",
            "kernel_execution_backend_id": "android_kotlin_assist_decision_kernel_v1",
            "route_input_policy": "episode_explicit_causal_route_v1",
            "adapter_route_input_sha256": "b" * 64,
            "episode_id": "episode-1",
            "decision_cadence": {"route_sample_policy": "latest_valid_generated_at_or_before_frame_v1"},
            "frames": [{"frame_id": "frame-0", "video_pts_ms": 500}],
        }
        receipt = {
            "schema": "blindassist_ustrf_sc_u0_route_conditioning_receipt_v1",
            "arm_id": request["arm_id"],
            "candidate_adapter_id": request["candidate_adapter_id"],
            "route_input_policy": request["route_input_policy"],
            "route_input_sha256": request["adapter_route_input_sha256"],
            "route_episode_id": request["episode_id"],
            "route_parent_source_id": "source-1",
            "route_provider_type": "navigation",
            "route_provider_id": "provider-1",
            "projection_receipt_id": "projection-1",
            **threshold,
            "route_sample_policy": "latest_valid_generated_at_or_before_frame_v1",
            "future_inputs_used": False,
            "risk_model_inferred_route": False,
            "frame_count": 1,
            "frames": [{
                "frame_id": "frame-0",
                "frame_timestamp_ms": 500,
                "route_usable": True,
                "gate_reason": "ROUTE_USABLE",
                "selected_sample_timestamp_ms": 500,
                "selected_valid_until_timestamp_ms": 1000,
                "selected_route_confidence": 1.0,
                "input_detection_count": 1,
                "retained_detection_count": 1,
                "detections": [{
                    "detection_index": 0,
                    "minimum_route_distance_px": 10.0,
                    "corridor_half_width_px": 20.0,
                    "kept": True,
                }],
            }],
        }
        output = {
            "android_backend_receipt": {"decoded_frames": [{"detection_count": 1, "kernel_input_detection_count": 1}]},
            "route_conditioning_receipt": receipt,
        }
        subject._validate_bbox_route_conditioning_receipt(
            output, request=request, threshold_config=threshold, evidence=evidence, where="route",
        )
        missing = json.loads(json.dumps(output))
        missing.pop("route_conditioning_receipt")
        with self.assertRaisesRegex(subject.ContractError, "is required"):
            subject._validate_bbox_route_conditioning_receipt(
                missing, request=request, threshold_config=threshold, evidence=evidence, where="route",
            )
        future = json.loads(json.dumps(output))
        future["route_conditioning_receipt"]["frames"][0]["selected_sample_timestamp_ms"] = 501
        with self.assertRaisesRegex(subject.ContractError, "future route sample"):
            subject._validate_bbox_route_conditioning_receipt(
                future, request=request, threshold_config=threshold, evidence=evidence, where="route",
            )
        fake_gate = json.loads(json.dumps(output))
        fake_gate["route_conditioning_receipt"]["frames"][0]["detections"][0]["kept"] = False
        with self.assertRaisesRegex(subject.ContractError, "gate arithmetic"):
            subject._validate_bbox_route_conditioning_receipt(
                fake_gate, request=request, threshold_config=threshold, evidence=evidence, where="route",
            )

    def test_dense_risk_receipt_binds_teacher_provenance_fold_and_normalization(self) -> None:
        contract = json.loads(
            (SCRIPTS.parent / "configs/ustrf_sc_u0_teacher_upper_bound_v1.json").read_text(encoding="utf-8")
        )
        evidence = contract["prediction_evidence_contract"]
        threshold = {
            "dense_field_contract_id": "ustrf_sc_u0_dense_teacher_field_v2",
            "normalization_contract_id": "ustrf_u0_dense_route_intrusion_to_kernel_risk_v1",
            "shared_risk_evidence_input_contract_id": "blindassist_shared_decision_kernel_risk_evidence_input_v1",
            "low_threshold": 0.35,
            "medium_threshold": 0.55,
            "high_threshold": 0.75,
            "local_peak_weight": 0.75,
            "maximum_route_unknown_fraction": 0.5,
        }
        request = {
            "arm_id": "teacher_dense_explicit_route",
            "candidate_adapter_id": "teacher_dense_explicit_route_adapter_v1",
            "kernel_execution_backend_id": "android_kotlin_assist_decision_kernel_v1",
            "route_input_policy": "episode_explicit_causal_route_v1",
            "adapter_route_input_sha256": "a" * 64,
            "episode_id": "episode-1",
            "fold_artifact_sha256": "b" * 64,
            "fold_training_input_manifest_sha256": "c" * 64,
            "fold_training_receipt_sha256": "d" * 64,
            "frames": [{
                "frame_id": "frame-0",
                "video_pts_ms": 500,
                "frame_payload_sha256": "e" * 64,
            }],
        }
        field_payload, field_summary = dense_field_fixture()
        receipt = {
            "schema": "blindassist_ustrf_sc_u0_dense_risk_evidence_receipt_v2",
            "arm_id": request["arm_id"],
            "candidate_adapter_id": request["candidate_adapter_id"],
            "route_input_policy": request["route_input_policy"],
            "route_input_sha256": request["adapter_route_input_sha256"],
            "episode_id": request["episode_id"],
            "fold_artifact_sha256": request["fold_artifact_sha256"],
            "fold_training_input_manifest_sha256": request["fold_training_input_manifest_sha256"],
            "fold_training_receipt_sha256": request["fold_training_receipt_sha256"],
            **threshold,
            "teacher_output_role": "auxiliary_only_not_human_truth",
            "blind_accessed": False,
            "future_inputs_used": False,
            "human_event_truth_used": False,
            "teacher_provenance": {
                "model_name": "depth-teacher",
                "model_version": "v1",
                "license_identifier": "Apache-2.0",
                "inference_runtime": "onnxruntime-cpu",
                "model_weights_sha256": "f" * 64,
                "inference_implementation_sha256": "1" * 64,
            },
            "frame_count": 1,
            "frames": [{
                "frame_id": "frame-0",
                "frame_timestamp_ms": 500,
                "observed_at_ms": 500,
                "valid_until_ms": 1000,
                "evidence_status": "AVAILABLE",
                "source_frame_payload_sha256": "e" * 64,
                **field_summary,
                "field_payload": field_payload,
                "route_intent_id": "route-1",
                "event_key": "episode-1:route-1",
                "risk_sources": ["depth-gradient", "relative-depth"],
                "normalized_risk_score": 0.6,
                "raw_risk_level": "MEDIUM",
                "risk_direction": "CENTER",
            }],
        }
        output = {
            "frames": [{"decision": {"raw_risk_level": "MEDIUM"}}],
            "dense_risk_evidence_receipt": receipt,
        }
        subject._validate_dense_risk_evidence_receipt(
            output, request=request, threshold_config=threshold, evidence=evidence, where="dense",
        )
        tampered = json.loads(json.dumps(output))
        tampered["dense_risk_evidence_receipt"]["frames"][0]["normalized_risk_score"] = 0.59
        with self.assertRaisesRegex(subject.ContractError, "arithmetic mismatch"):
            subject._validate_dense_risk_evidence_receipt(
                tampered, request=request, threshold_config=threshold, evidence=evidence, where="dense",
            )
        leaked = json.loads(json.dumps(output))
        leaked["dense_risk_evidence_receipt"]["human_event_truth_used"] = True
        with self.assertRaisesRegex(subject.ContractError, "human_event_truth_used mismatch"):
            subject._validate_dense_risk_evidence_receipt(
                leaked, request=request, threshold_config=threshold, evidence=evidence, where="dense",
            )


if __name__ == "__main__":
    unittest.main()
