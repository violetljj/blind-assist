"""Small source-contract checks; no scoring and no stored experiment outputs."""
import unittest
import numpy as np
from mz90_observation_source import build_source, materialize, corridor_intersection, geometric_hazard, DT, FRAMES


class SourceTests(unittest.TestCase):
    def fixture(self):
        source = build_source()[:2]
        source[1]["objects"] = [dict(x=0.0, z=3.0, vx=0.0, vz=0.0, reflectivity=1.0)]
        return source

    def test_source_reproducible_and_bounded(self):
        self.assertEqual(build_source(), build_source())
        source = build_source()
        self.assertEqual(len(source), 48)
        self.assertEqual(len({s["episode_id"] for s in source}), 48)
        self.assertTrue(all(not source[i]["objects"] for i in range(0, 48, 8)))

    def test_sensor_frame_and_no_privileged_fields(self):
        source = self.fixture()
        raw, _ = materialize(source, "ideal")
        j = FRAMES+10
        yaw = np.cumsum(raw["delta_yaw"][FRAMES:2*FRAMES])[10]
        self.assertAlmostEqual(raw["radar_angle"][j, 0], -yaw)
        self.assertAlmostEqual(raw["radar_velocity"][j, 0], -0.7)
        self.assertEqual(set(raw), {"episode_id", "time_s", "tof_packet_received", "tof_range_m", "tof_theta_deg", "tof_range_sigma_m", "tof_status", "radar_packet_received", "radar_range_m", "radar_velocity", "radar_angle", "radar_valid", "delta_yaw", "delta_pitch", "imu_valid"})

    def test_missing_packet_distinct_from_no_return(self):
        raw, _ = materialize(self.fixture(), "sensor_proxy")
        absent = ~raw["tof_packet_received"]
        self.assertTrue(absent.any())
        self.assertTrue(np.all(raw["tof_status"][absent] == 0))
        empty_received = raw["tof_packet_received"][:FRAMES]
        self.assertTrue(np.all(raw["tof_status"][:FRAMES][empty_received] == 255))
        self.assertTrue(np.isnan(raw["tof_range_m"][absent]).all())

    def test_pair_truth_and_reproducibility(self):
        src = self.fixture()
        a, truth_a = materialize(src, "ideal")
        b, truth_b = materialize(src, "sensor_proxy")
        c, _ = materialize(src, "sensor_proxy")
        for key in truth_a:
            np.testing.assert_array_equal(truth_a[key], truth_b[key])
        for key in b:
            np.testing.assert_array_equal(b[key], c[key])
        self.assertTrue(np.all(truth_a["truth_height"] == "UNKNOWN"))
        self.assertFalse(truth_a["truth"][:FRAMES].any())
        self.assertEqual(a["tof_range_m"].shape, (80, 8))

    def test_continuous_crossing_not_endpoint_only(self):
        # Both endpoints outside; the segment crosses the corridor between them.
        self.assertTrue(corridor_intersection(-1, 2, 2, 0))
        self.assertTrue(geometric_hazard(-1, 2, 2, 0))
        self.assertFalse(corridor_intersection(-1, 2, -2, 0))
        self.assertFalse(corridor_intersection(0, 0.2, 0, 0))
        self.assertFalse(geometric_hazard(-1, 4, 2, 0))
        self.assertFalse(geometric_hazard(0, 0.1, 0, 1))


if __name__ == "__main__":
    unittest.main()
