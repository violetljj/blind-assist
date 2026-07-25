from __future__ import annotations

import unittest
from types import SimpleNamespace

from audit_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0 import (
    nearest_delta_seconds,
    norm_frame,
    norm_topic,
    stamp_ns,
    summarize,
)


class JrdbSingleRosbagAuthorityTest(unittest.TestCase):
    def test_frame_and_stamp_normalization(self) -> None:
        self.assertEqual(norm_frame("/odom"), "odom")
        self.assertEqual(norm_topic("/tf"), "tf")
        self.assertEqual(stamp_ns(SimpleNamespace(sec=12, nanosec=34)), 12_000_000_034)

    def test_nearest_delta(self) -> None:
        result = nearest_delta_seconds([1_000_000_000, 2_000_000_000], [1.1, 1.9])
        self.assertAlmostEqual(result["maximum"], 0.1)

    def test_summary_detects_coverage_and_variation(self) -> None:
        samples = [
            {"header_ns": 1_000_000_000, "bag_ns": 1_010_000_000, "frame_id": "imu", "signature": (0.0,)},
            {"header_ns": 1_100_000_000, "bag_ns": 1_110_000_000, "frame_id": "imu", "signature": (1.0,)},
            {"header_ns": 1_200_000_000, "bag_ns": 1_210_000_000, "frame_id": "imu", "signature": (2.0,)},
        ]
        result = summarize(samples, (1.0, 1.2), 0.25, 0.01)
        self.assertTrue(result["covers_external_window"])
        self.assertEqual(result["unique_measurement_signatures"], 3)
        self.assertEqual(result["backward_header_steps"], 0)


if __name__ == "__main__":
    unittest.main()
