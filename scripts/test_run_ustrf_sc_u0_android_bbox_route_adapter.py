import copy
import unittest

import run_ustrf_sc_u0_android_bbox_route_adapter as subject


class UstrfU0AndroidBBoxRouteAdapterTest(unittest.TestCase):
    def fixture(self):
        waypoints = [
            {"horizon_ms": 1000, "xy_norm": [0.5, 0.9]},
            {"horizon_ms": 2000, "xy_norm": [0.5, 0.8]},
            {"horizon_ms": 3000, "xy_norm": [0.5, 0.7]},
        ]
        route = {
            "episode_id": "episode-1",
            "parent_source_id": "source-1",
            "provider": {"type": "navigation", "provider_id": "provider-1"},
            "coordinate_contract": {"projection_receipt_id": "projection-1"},
            "samples": [{
                "timestamp_ms": 0,
                "valid_until_timestamp_ms": 500,
                "confidence": 1.0,
                "route_valid": True,
                "horizon_waypoints": waypoints,
            }],
        }
        request = {
            "arm_id": subject.ARM_ID,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "route_input_policy": subject.ROUTE_POLICY,
            "adapter_route_input_sha256": "b" * 64,
            "episode_id": "episode-1",
            "decision_cadence": {"route_sample_policy": "latest_valid_generated_at_or_before_frame_v1"},
            "frames": [{"frame_id": "frame-0", "video_pts_ms": 0}],
        }
        config = {
            "route_gate_implementation_sha256": "a" * 64,
            "minimum_route_confidence": 0.5,
            "maximum_route_age_ms": 1000,
            "corridor_half_width_frame_ratio": 0.08,
            "obstacle_footprint_height_ratio": 0.25,
        }
        frame = {
            "frame_id": "frame-0",
            "frame_timestamp_ms": 0,
            "route_episode_id": "episode-1",
            "route_parent_source_id": "source-1",
            "route_provider_type": "navigation",
            "route_provider_id": "provider-1",
            "projection_receipt_id": "projection-1",
            "route_usable": True,
            "gate_reason": "ROUTE_USABLE",
            "selected_sample_index": 0,
            "selected_sample_timestamp_ms": 0,
            "selected_valid_until_timestamp_ms": 500,
            "selected_route_confidence": 1.0,
            "selected_waypoints": waypoints,
            "gate_contract_id": subject.GATE_CONTRACT_ID,
            "unknown_route_policy": subject.UNKNOWN_ROUTE_POLICY,
            "minimum_route_confidence": 0.5,
            "maximum_route_age_ms": 1000,
            "corridor_half_width_frame_ratio": 0.08,
            "obstacle_footprint_height_ratio": 0.25,
            "input_detection_count": 1,
            "retained_detection_count": 1,
            "detections": [{
                "detection_index": 0,
                "label": "person",
                "confidence": 0.9,
                "source_box_xyxy_px": [450.0, 600.0, 550.0, 1000.0],
                "footprint_box_xyxy_px": [450.0, 900.0, 550.0, 1000.0],
                "minimum_route_distance_px": 0.0,
                "corridor_half_width_px": 80.0,
                "kept": True,
            }],
        }
        receipt = {
            "schema": subject.ROUTE_RECEIPT_SCHEMA,
            "arm_id": subject.ARM_ID,
            "candidate_adapter_id": subject.ADAPTER_ID,
            "route_input_policy": subject.ROUTE_POLICY,
            "route_input_sha256": "b" * 64,
            "route_episode_id": "episode-1",
            "route_parent_source_id": "source-1",
            "route_provider_type": "navigation",
            "route_provider_id": "provider-1",
            "projection_receipt_id": "projection-1",
            "route_gate_contract_id": subject.GATE_CONTRACT_ID,
            "route_gate_implementation_sha256": "a" * 64,
            "unknown_route_policy": subject.UNKNOWN_ROUTE_POLICY,
            "route_sample_policy": "latest_valid_generated_at_or_before_frame_v1",
            "future_inputs_used": False,
            "risk_model_inferred_route": False,
            "frame_count": 1,
            "frames": [frame],
        }
        decoded = [{"width": 1000, "height": 1000, "detection_count": 1, "kernel_input_detection_count": 1}]
        return receipt, request, route, config, decoded

    def test_host_recomputes_causal_sample_footprint_distance_and_keep(self):
        receipt, request, route, config, decoded = self.fixture()
        subject.validate_route_receipt(
            receipt, request=request, route=route, config=config, decoded_frames=decoded,
        )

        tampered = copy.deepcopy(receipt)
        tampered["frames"][0]["detections"][0]["minimum_route_distance_px"] = 1.0
        with self.assertRaisesRegex(subject.AdapterError, "route distance mismatch"):
            subject.validate_route_receipt(
                tampered, request=request, route=route, config=config, decoded_frames=decoded,
            )

    def test_future_sample_is_not_selected(self):
        route = {"samples": [{
            "timestamp_ms": 501,
            "valid_until_timestamp_ms": 1000,
            "confidence": 1.0,
            "route_valid": True,
            "horizon_waypoints": [],
        }]}
        selected, reason = subject._expected_route_selection(route, 500, {
            "minimum_route_confidence": 0.5, "maximum_route_age_ms": 1000,
        })
        self.assertIsNone(selected)
        self.assertEqual("NO_CAUSAL_ROUTE_SAMPLE", reason)

    def test_collinear_but_disjoint_segment_does_not_fake_intersection(self):
        distance = subject._segment_rectangle_distance((0.0, 0.0), (1.0, 0.0), [2.0, 0.0, 3.0, 1.0])
        self.assertEqual(1.0, distance)


if __name__ == "__main__":
    unittest.main()
