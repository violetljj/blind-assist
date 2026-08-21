from __future__ import annotations

import unittest

import cv2
import numpy as np

from opencv_provider import FrozenRgbProvider
from select_roster import classify_episode
from stage_a import FrameEvidence, SpatialEvidence, adjudicate_stage_a, step


def spatial(reference: str, **changes) -> SpatialEvidence:
    values = {
        "reference_frame": reference,
        "geometry_supported": True,
        "motion_observable": True,
        "translation_overreach": False,
        "geometry_degenerate": False,
        "bearing_estimate_deg": 12.0,
        "bearing_uncertainty_deg": 2.0,
        "compatibility": "SUPPORTED",
    }
    values.update(changes)
    return SpatialEvidence(**values)


def frame(**changes) -> FrameEvidence:
    values = {
        "frame_id": "f1",
        "observation_supported": True,
        "candidate_region_xyxy": (10.0, 20.0, 30.0, 40.0),
        "independent_identity_confirmation": "SUPPORTED",
        "observability_reason": "IN_VIEW_CANDIDATE",
        "c0_spatial": spatial("CAMERA_RELATIVE"),
        "t0_spatial": spatial("KEYFRAME_RELATIVE"),
    }
    values.update(changes)
    return FrameEvidence(**values)


class StageAMechanicsTest(unittest.TestCase):
    def test_reacquisition_requires_both_channels(self):
        value = frame(independent_identity_confirmation="REJECTED")
        self.assertEqual(step("W1-T0", "r1", value, previous_observation_state="NONE")["reacquisition_status"], "NOT_REACQUIRED")
        value = frame(t0_spatial=spatial("KEYFRAME_RELATIVE", compatibility="REJECTED"))
        self.assertEqual(step("W1-T0", "r1", value, previous_observation_state="NONE")["reacquisition_status"], "NOT_REACQUIRED")
        self.assertEqual(step("W1-T0", "r1", frame(), previous_observation_state="NONE")["reacquisition_status"], "REACQUIRED")
        self.assertEqual(step("W1-T0", "r1", frame(), previous_observation_state="SUPPORTED")["reacquisition_status"], "NOT_REACQUIRED")

    def test_no_observation_never_retains_bbox(self):
        value = frame(
            observation_supported=False,
            candidate_region_xyxy=None,
            observability_reason="NO_OBSERVATION",
            independent_identity_confirmation="INSUFFICIENT",
        )
        result = step("C0", "r1", value, previous_observation_state="SUPPORTED")
        self.assertEqual(result["observation_state"], "NONE")
        self.assertIsNone(result["candidate_region"])
        self.assertEqual(result["identity_state"], "VALID")

    def test_translation_overreach_stales_without_deleting_identity(self):
        value = frame(t0_spatial=spatial("KEYFRAME_RELATIVE", translation_overreach=True))
        result = step("W1-T0", "r1", value, previous_observation_state="NONE")
        self.assertEqual(result["spatial_anchor_state"], "STALE")
        self.assertEqual(result["identity_state"], "VALID")
        self.assertFalse(result["directional_guidance_authorized"])
        self.assertIsNone(result["bearing_estimate"])
        self.assertEqual(result["reacquisition_status"], "NOT_REACQUIRED")

    def test_geometry_degenerate_stales(self):
        value = frame(t0_spatial=spatial("KEYFRAME_RELATIVE", geometry_degenerate=True))
        self.assertEqual(step("W1-T0", "r1", value, previous_observation_state="NONE")["spatial_anchor_state"], "STALE")

    def test_arm_reference_frames_cannot_be_swapped(self):
        value = frame(c0_spatial=spatial("KEYFRAME_RELATIVE"))
        with self.assertRaisesRegex(ValueError, "CAMERA_RELATIVE"):
            step("C0", "r1", value, previous_observation_state="NONE")

    def test_candidate_contract_rejects_fabricated_region(self):
        value = frame(observation_supported=False)
        with self.assertRaisesRegex(ValueError, "candidate_region"):
            step("C0", "r1", value, previous_observation_state="NONE")

    def test_frozen_rgb_provider_is_causal_and_shared(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        rng = np.random.default_rng(7)
        for _ in range(120):
            x, y = rng.integers(10, 310), rng.integers(10, 230)
            color = tuple(int(item) for item in rng.integers(80, 255, size=3))
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
        box = (80.0, 60.0, 240.0, 190.0)
        provider = FrozenRgbProvider(image, box)
        evidence = provider.evidence("f0", image.copy())
        self.assertTrue(evidence.observation_supported)
        self.assertEqual(evidence.independent_identity_confirmation, "SUPPORTED")
        self.assertEqual(step("C0", "r", evidence, previous_observation_state="SUPPORTED")["reference_frame"], "CAMERA_RELATIVE")
        self.assertEqual(step("W1-T0", "r", evidence, previous_observation_state="SUPPORTED")["reference_frame"], "KEYFRAME_RELATIVE")

    def test_outcome_blind_selector_tags_motion_and_confuser_support(self):
        half_angle = np.deg2rad(10.0)
        trajectory = {
            0: {"tx_world_device": 0.0, "ty_world_device": 0.0, "tz_world_device": 0.0,
                "qx_world_device": 0.0, "qy_world_device": 0.0, "qz_world_device": 0.0, "qw_world_device": 1.0},
            1: {"tx_world_device": 0.05, "ty_world_device": 0.0, "tz_world_device": 0.0,
                "qx_world_device": 0.0, "qy_world_device": 0.0, "qz_world_device": float(np.sin(half_angle)), "qw_world_device": float(np.cos(half_angle))},
        }
        episode = {
            "frames": [{"timestamp_ns": 0, "target_visible": True}, {"timestamp_ns": 1, "target_visible": False}],
            "temporal_mode_tags": ["OUT_OF_VIEW_RETURN"],
            "candidate_distractor_instance_ids": ["opaque-private-id"],
        }
        buckets, _ = classify_episode(episode, trajectory)
        self.assertIn("ROTATION_DOMINANT", buckets)
        self.assertIn("IDENTITY_CONFUSER", buckets)
        self.assertIn("OBSERVATION_LOSS", buckets)

    def test_adjudicator_requires_utility_without_safety_regression(self):
        support = {bucket: 1 for bucket in (
            "ROTATION_DOMINANT", "SMALL_TRANSLATION", "TRANSLATION_BEYOND_TIER0",
            "OCCLUSION_OR_REAPPEARANCE", "IDENTITY_CONFUSER", "OBSERVATION_LOSS", "GEOMETRY_DEGENERATE",
        )}
        base = {metric: 0 for metric in (
            "fabricated_observation", "single_channel_reacquisition", "stale_anchor_guidance_use",
            "post_initialization_truth_leakage", "future_frame_access",
        )}
        c0 = {**base, "false_reacquisition": 1, "false_continuity": 3,
              "identity_confirmed_reacquisition": 2, "bearing_compatibility_rate": 0.8,
              "usable_anchor_coverage": 4, "abstention_count": 5}
        t0 = {**base, "false_reacquisition": 1, "false_continuity": 2,
              "identity_confirmed_reacquisition": 3, "bearing_compatibility_rate": 0.8,
              "usable_anchor_coverage": 6, "abstention_count": 4,
              "translation_overreach_timely_stale": True, "geometry_degenerate_timely_stale": True}
        self.assertEqual(adjudicate_stage_a(c0, t0, support), "W1_T0_WORLD_REFERENT_SIGNAL_ESTABLISHED")
        t0["stale_anchor_guidance_use"] = 1
        self.assertEqual(adjudicate_stage_a(c0, t0, support), "W1_T0_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
