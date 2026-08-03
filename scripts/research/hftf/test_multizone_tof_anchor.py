#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multizone_tof_anchor import (
    FRAME_SCHEMA,
    REGISTRATION_SCHEMA,
    TofAnchorPolicy,
    TofFrameStream,
    estimate_tof_scale_anchor,
    load_registration,
    load_tof_frames,
)


class MultizoneTofAnchorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registration_payload = {
            "schema": REGISTRATION_SCHEMA,
            "admitted": True,
            "registration_id": "rig-r0",
            "tof_sensor_id": "tof-8x8",
            "rgb_calibration_id": "rgb-cal-r0",
            "transform_rgb_from_tof": np.eye(4).tolist(),
            "zones": [
                {"zone_id": "left", "ray_tof_unit": [-0.6, 0.0, 0.8]},
                {"zone_id": "center", "ray_tof_unit": [0.0, 0.0, 1.0]},
                {"zone_id": "right", "ray_tof_unit": [0.6, 0.0, 0.8]},
            ],
        }
        self.frame = {
            "schema": FRAME_SCHEMA,
            "sequence_id": "seq",
            "timestamp_ns": 90,
            "clock_domain": "host_monotonic",
            "tof_sensor_id": "tof-8x8",
            "registration_id": "rig-r0",
            "zones": [
                {"zone_id": "left", "range_m": 2.5, "sigma_m": 0.01, "status": "VALID"},
                {"zone_id": "center", "range_m": 2.0, "sigma_m": 0.01, "status": "VALID"},
                {"zone_id": "right", "range_m": 2.5, "sigma_m": 0.01, "status": "VALID"},
            ],
        }
        self.policy = TofAnchorPolicy(
            max_rgb_tof_skew_ns=20,
            max_sigma_m=0.05,
            minimum_zones=3,
            minimum_bands=3,
            maximum_scale_mad=0.05,
            depth_patch_radius_px=0,
        )

    def registration(self, root: Path):
        path = root / "registration.json"
        path.write_text(json.dumps(self.registration_payload), encoding="utf-8")
        return load_registration(path)

    def test_registered_z_depth_recovers_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registration = self.registration(Path(directory))
        depth = np.full((101, 101), 4.0, dtype=np.float32)
        anchor, diagnostic = estimate_tof_scale_anchor(
            depth,
            [50.0, 50.0, 50.0, 50.0],
            100,
            "host_monotonic",
            self.frame,
            registration,
            "rgb-cal-r0",
            self.policy,
        )
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor.scale, 0.5)
        self.assertEqual(anchor.pair_count, 3)
        self.assertEqual(diagnostic["covered_bands"], ["center", "left", "right"])

    def test_clock_skew_and_quality_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registration = self.registration(Path(directory))
        depth = np.full((101, 101), 4.0, dtype=np.float32)
        anchor, diagnostic = estimate_tof_scale_anchor(
            depth,
            [50.0, 50.0, 50.0, 50.0],
            100,
            "other_clock",
            self.frame,
            registration,
            "rgb-cal-r0",
            self.policy,
        )
        self.assertIsNone(anchor)
        self.assertEqual(diagnostic["status"], "UNKNOWN_TOF_RGB_CLOCK_DOMAIN_MISMATCH")
        noisy = {**self.frame, "zones": [{**zone, "sigma_m": 0.2} for zone in self.frame["zones"]]}
        anchor, diagnostic = estimate_tof_scale_anchor(
            depth,
            [50.0, 50.0, 50.0, 50.0],
            100,
            "host_monotonic",
            noisy,
            registration,
            "rgb-cal-r0",
            self.policy,
        )
        self.assertIsNone(anchor)
        self.assertEqual(diagnostic["status"], "UNKNOWN_INSUFFICIENT_TOF_ZONES")

    def test_frame_transport_is_ordered_and_causal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tof.jsonl"
            second = {**self.frame, "timestamp_ns": 110}
            path.write_text(
                json.dumps(self.frame) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            stream = TofFrameStream(load_tof_frames(path))
        self.assertEqual(len(stream.take_available("seq", 100)), 1)
        self.assertEqual(stream.take_available("seq", 105), [])
        self.assertEqual(len(stream.take_available("seq", 110)), 1)


if __name__ == "__main__":
    unittest.main()
