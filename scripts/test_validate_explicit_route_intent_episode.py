import copy
import unittest

import validate_explicit_route_intent_episode as subject


def valid_episode():
    return {
        "schema": subject.SCHEMA,
        "episode_id": "episode-1",
        "parent_source_id": "source-1",
        "provider": {"type": "navigation", "provider_id": "nav-1",
                     "inferred_by_risk_model": False, "input_space": "current_camera_frame"},
        "coordinate_contract": {"space": "normalized_current_camera_frame_xy",
                                "projection_receipt_id": "projection-1",
                                "device_to_world_alignment_receipt_id": None},
        "samples": [{"timestamp_ms": 0, "valid_until_timestamp_ms": 500, "confidence": 0.9,
                     "route_valid": True, "horizon_waypoints": [
                         {"horizon_ms": 1000, "xy_norm": [0.4, 0.9]},
                         {"horizon_ms": 2000, "xy_norm": [0.5, 0.8]},
                         {"horizon_ms": 3000, "xy_norm": [0.6, 0.7]},
                     ]}],
        "fallback": {"missing_stale_or_low_confidence_route": "context_attention_only",
                     "directional_instruction_allowed": False, "intervention_upgrade_allowed": False},
        "training_isolation": {"future_video_teacher_allowed_in_eval_or_runtime": False},
    }


class ExplicitRouteIntentValidatorTest(unittest.TestCase):
    def test_valid_runtime_episode(self) -> None:
        result = subject.validate_episode(valid_episode(), runtime=True)
        self.assertEqual(1, result["valid_route_sample_count"])

    def test_rejects_risk_model_as_intent_provider(self) -> None:
        value = valid_episode()
        value["provider"]["inferred_by_risk_model"] = True
        with self.assertRaisesRegex(ValueError, "must not be inferred"):
            subject.validate_episode(value, runtime=True)

    def test_rejects_future_oracle_at_runtime(self) -> None:
        value = valid_episode()
        value["provider"]["type"] = "train_only_future_video_oracle"
        with self.assertRaisesRegex(ValueError, "not allowed at runtime"):
            subject.validate_episode(value, runtime=True)

    def test_world_waypoints_require_alignment_receipt(self) -> None:
        value = valid_episode()
        value["provider"]["input_space"] = "world_waypoints"
        with self.assertRaisesRegex(ValueError, "device_to_world_alignment_receipt_id"):
            subject.validate_episode(value, runtime=False)

    def test_rejects_stale_and_out_of_bounds_samples(self) -> None:
        stale = valid_episode()
        stale["samples"][0]["valid_until_timestamp_ms"] = 2000
        with self.assertRaisesRegex(ValueError, "validity"):
            subject.validate_episode(stale, runtime=True)
        outside = valid_episode()
        outside["samples"][0]["horizon_waypoints"][0]["xy_norm"] = [-0.1, 0.9]
        with self.assertRaisesRegex(ValueError, "inside"):
            subject.validate_episode(outside, runtime=True)

    def test_invalid_route_sample_carries_no_waypoints(self) -> None:
        value = valid_episode()
        value["samples"][0]["route_valid"] = False
        with self.assertRaisesRegex(ValueError, "must not contain"):
            subject.validate_episode(value, runtime=True)


if __name__ == "__main__":
    unittest.main()
