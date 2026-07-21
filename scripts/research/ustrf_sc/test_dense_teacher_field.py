from __future__ import annotations

import unittest

from dense_teacher_field import (
    ARTIFACT_SCHEMA,
    DenseTeacherConfig,
    DenseTeacherError,
    MODEL_LICENSE,
    MODEL_INPUT_CONTRACT,
    MODEL_NAME,
    MODEL_OUTPUT_CONTRACT,
    MODEL_VERSION,
    build_frame_field,
    select_route_sample,
    validate_serialized_field_summary,
    validate_loso_artifact,
)


class DenseTeacherFieldTest(unittest.TestCase):
    def artifact(self) -> dict:
        return {
            "schema": ARTIFACT_SCHEMA,
            "fit_policy": "leave_one_session_out_fit_v1",
            "held_out_session_id": "held-out",
            "training_session_ids": ["train-a", "train-b"],
            "training_input_manifest_sha256": "1" * 64,
            "teacher_model_sha256": "3" * 64,
            "teacher_model_name": MODEL_NAME,
            "teacher_model_version": MODEL_VERSION,
            "teacher_model_license": MODEL_LICENSE,
            "teacher_model_input_contract": MODEL_INPUT_CONTRACT,
            "teacher_model_output_contract": MODEL_OUTPUT_CONTRACT,
            "teacher_decode_policy": "opencv_video_capture_requested_pts_v1",
            "teacher_inference_runtime": "onnxruntime_cpu_v1",
            "calibration_input_schema": "blindassist_ustrf_sc_u0_dense_teacher_calibration_inputs_v1",
            "fit_implementation_file_sha256": "6" * 64,
            "fit_implementation_sha256": "4" * 64,
            "training_sample_inventory_sha256": "5" * 64,
            "blind_accessed": False,
            "future_inputs_used": False,
            "human_event_truth_used": False,
            "production_authorized": False,
            "calibration": {
                "raw_depth_lower_quantile": 1.0,
                "raw_depth_upper_quantile": 3.0,
                "gradient_upper_quantile": 1.0,
                "training_frame_count": 1,
            },
            "training_samples": [{
                "session_id": "train-a",
                "episode_id": "episode-a",
                "frame_id": "frame-a",
                "video_pts_ms": 0,
                "teacher_decoded_rgb_sha256": "7" * 64,
                "raw_depth_sha256": "8" * 64,
            }],
        }

    def route(self, *, future: bool = False) -> dict:
        return {
            "provider": {"provider_id": "p", "inferred_by_risk_model": False},
            "coordinate_contract": {"space": "normalized_current_camera_frame_xy"},
            "samples": [{
                "timestamp_ms": 501 if future else 500,
                "valid_until_timestamp_ms": 1_000,
                "confidence": 1.0,
                "route_valid": True,
                "horizon_waypoints": [
                    {"horizon_ms": 1_000, "xy_norm": [0.5, 0.9]},
                    {"horizon_ms": 2_000, "xy_norm": [0.5, 0.8]},
                    {"horizon_ms": 3_000, "xy_norm": [0.5, 0.7]},
                ],
            }],
        }

    def test_loso_artifact_binds_holdout_training_and_model(self) -> None:
        validate_loso_artifact(
            self.artifact(), held_out_session_id="held-out",
            training_manifest_sha256="1" * 64,
            model_sha256="3" * 64,
        )
        leaked = self.artifact()
        leaked["training_session_ids"].append("held-out")
        with self.assertRaisesRegex(DenseTeacherError, "training session inventory"):
            validate_loso_artifact(
                leaked, held_out_session_id="held-out",
                training_manifest_sha256="1" * 64,
                model_sha256="3" * 64,
            )

    def test_route_selection_rejects_future_and_uses_current_sample(self) -> None:
        index, sample = select_route_sample(self.route(), 500, DenseTeacherConfig())
        self.assertEqual(0, index)
        self.assertEqual(500, sample["timestamp_ms"])
        with self.assertRaisesRegex(DenseTeacherError, "no causal"):
            select_route_sample(self.route(future=True), 500, DenseTeacherConfig())

    def test_dense_field_contains_all_auxiliary_outputs_and_route_changes_evidence(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed in dependency-free test runtime")
        depth = np.tile(np.linspace(1.0, 3.0, 252, dtype=np.float32), (252, 1))
        center = self.route()["samples"][0]
        left = self.route()["samples"][0] | {
            "horizon_waypoints": [
                {"horizon_ms": 1_000, "xy_norm": [0.05, 0.9]},
                {"horizon_ms": 2_000, "xy_norm": [0.05, 0.8]},
                {"horizon_ms": 3_000, "xy_norm": [0.05, 0.7]},
            ]
        }
        center_payload, center_summary = build_frame_field(
            depth, artifact=self.artifact(), route_sample=center, config=DenseTeacherConfig()
        )
        left_payload, left_summary = build_frame_field(
            depth, artifact=self.artifact(), route_sample=left, config=DenseTeacherConfig()
        )
        self.assertEqual(
            {"local_obstacle_field", "walkability_field", "boundary_field", "unknown_field", "route_weight_field", "route_relative_risk_field"},
            set(center_payload["fields_base64"]),
        )
        self.assertNotEqual(center_summary["route_intrusion_score"], left_summary["route_intrusion_score"])
        self.assertEqual(center_summary["source_field_sha256"], left_summary["source_field_sha256"])
        self.assertNotEqual(
            center_summary["route_interaction_field_sha256"],
            left_summary["route_interaction_field_sha256"],
        )
        self.assertEqual(center_payload["quantization"], left_payload["quantization"])
        validate_serialized_field_summary(center_payload, center_summary)
        tampered_summary = center_summary | {"route_intrusion_score": center_summary["route_intrusion_score"] + 0.01}
        with self.assertRaisesRegex(DenseTeacherError, "summary mismatch"):
            validate_serialized_field_summary(center_payload, tampered_summary)
        self.assertEqual(sorted(center_summary["risk_sources"]), center_summary["risk_sources"])


if __name__ == "__main__":
    unittest.main()
