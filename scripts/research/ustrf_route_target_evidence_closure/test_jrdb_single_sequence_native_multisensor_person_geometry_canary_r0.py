#!/usr/bin/env python3
"""Focused stdlib tests for JRDB P2 geometry canary invariants."""
from __future__ import annotations

import math
import unittest

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0 import (
    GateError,
    apply_transform,
    audit_packet,
    compose,
    interpolate,
    inverse,
    q_slerp,
    transform,
)


class GeometryCanaryTests(unittest.TestCase):
    def test_transform_inverse_roundtrip(self) -> None:
        value = transform(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0],
        )
        identity = compose(value, inverse(value))
        result = apply_transform(identity, [4.0, 5.0, 6.0])
        self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(result, [4.0, 5.0, 6.0])))

    def test_slerp_normalized_and_short_arc(self) -> None:
        result = q_slerp([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0], 0.5)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in result)), 1.0)
        self.assertAlmostEqual(abs(result[3]), 1.0)

    def test_interpolation_bound_is_fail_closed(self) -> None:
        samples = [{"timestamp_ns": 0}, {"timestamp_ns": 100_000_000}]
        with self.assertRaises(GateError):
            interpolate(samples, 50_000_000, 0.05, 0.025, "interpolation")

    def test_label_join_failure_skips_motion(self) -> None:
        config = {
            "canary": {"frame_count": 1},
            "gates": {
                "require_all_3d_labels_join_2d": True,
                "minimum_joined_person_frames": 1,
                "minimum_valid_motion_pairs": 1,
                "minimum_motion_track_count": 1,
                "maximum_person_motion_gap_seconds": 0.2,
            },
            "authority": {
                "route_risk": False,
                "event_lifecycle": False,
                "alert_logic": False,
                "android": False,
                "human_safety": False,
                "production": False,
                "commit": False,
                "push": False,
            },
        }
        packet = {
            "schema": "blindassist_ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0_observation_packet",
            "stage": "JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R0",
            "config_sha256": "0" * 64,
            "sequence": "synthetic",
            "window": {"first_frame": 0, "last_frame": 0, "frame_count": 1},
            "frames": [
                {
                    "frame_index": 0,
                    "time": {
                        "image_timestamp_ns": 0,
                        "upper_pointcloud_timestamp_ns": 0,
                        "lower_pointcloud_timestamp_ns": 0,
                        "bag_rgb_delta_seconds": 0.0,
                        "bag_upper_delta_seconds": 0.0,
                        "bag_lower_delta_seconds": 0.0,
                    },
                    "pose": {"bracket_seconds": 0.01},
                    "imu": {"bracket_seconds": 0.05},
                    "labels": {
                        "joined_count": 0,
                        "labels_2d_only": [],
                        "labels_3d_without_2d": ["pedestrian:1"],
                        "joined": [],
                    },
                }
            ],
        }
        receipt = audit_packet(config, packet, "1" * 64)
        self.assertEqual(receipt["terminal_state"], "FAIL_CLOSED_LABEL_JOIN")
        self.assertEqual(receipt["motion_pairs"], [])
        self.assertFalse(receipt["claims"]["source_native_person_motion_available"])


if __name__ == "__main__":
    unittest.main()
