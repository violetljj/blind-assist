import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("align_revel_detector_failures_with_vicon.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("revel_vicon_alignment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RevelDetectorViconAlignmentTest(unittest.TestCase):
    @staticmethod
    def _track(timestamps_ns, positions):
        count = len(timestamps_ns)
        return {
            "timestamps_ns": np.asarray(timestamps_ns, dtype=np.int64),
            "positions": np.asarray(positions, dtype=np.float64),
            "quaternions": np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]]), (count, 1)),
        }

    def test_identity_quaternion_rotation(self):
        matrix = MODULE._rotation_matrix(np.asarray([[0.0, 0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(matrix[0], np.eye(3), atol=1e-12)

    def test_summary_stratifies_recall_by_range_and_outcome(self):
        boxes = [
            {"vicon_available": True, "sensor_local_range_m": 1.0, "matched_at_fixed_score": True, "class_name": "green", "stratum": "large", "source_motion_available": True, "source_motion_unavailable_reason": None, "source_radial_motion": "approaching", "source_radial_range_rate_mps": -1.0, "source_ttc_proxy_s": .8},
            {"vicon_available": True, "sensor_local_range_m": 3.5, "matched_at_fixed_score": False, "class_name": "yellow", "stratum": "small", "source_motion_available": True, "source_motion_unavailable_reason": None, "source_radial_motion": "receding", "source_radial_range_rate_mps": .5, "source_ttc_proxy_s": None},
            {"vicon_available": False, "sensor_local_range_m": None, "matched_at_fixed_score": False, "class_name": "yellow", "stratum": "small", "source_motion_available": False, "source_motion_unavailable_reason": "no_strict_vicon_bracket", "source_radial_motion": None, "source_radial_range_rate_mps": None, "source_ttc_proxy_s": None},
        ]
        summary = MODULE._summarize(boxes, ["green", "yellow"])
        self.assertEqual(2, summary["vicon_aligned_box_count"])
        self.assertEqual(1.0, summary["recall_by_sensor_local_range"]["0-2m"]["recall"])
        self.assertEqual(0.0, summary["recall_by_sensor_local_range"]["3-4m"]["recall"])
        self.assertEqual(3.5, summary["sensor_local_range_by_outcome_m"]["small_missed"]["median"])
        self.assertEqual(0.5, summary["document_range_summary"]["within_0_5m"]["recall"])
        self.assertIsNotNone(summary["document_range_summary"]["within_0_5m"]["recall_wilson95"])
        self.assertEqual(2, summary["source_motion_aligned_box_count"])
        self.assertEqual(1.0, summary["recall_by_source_radial_motion"]["approaching"]["recall"])
        self.assertEqual(0.0, summary["recall_by_source_radial_motion"]["receding"]["recall"])
        self.assertIsNone(summary["recall_by_source_radial_motion"]["quasi_static"]["recall"])
        self.assertEqual(1, summary["source_motion_unavailable_reasons"]["no_strict_vicon_bracket"])

    def test_native_vicon_pairs_classify_radial_motion_and_ttc_proxy(self):
        timestamps = [1_000_000_000, 1_010_000_000, 1_020_000_000, 1_030_000_000]
        sensor = self._track(timestamps, [[1.0, 0.0, 0.0]] * 4)
        person = self._track(timestamps, [[4.0, 0.0, 0.0], [3.99, 0.0, 0.0], [3.99, 0.0, 0.0], [4.0, 0.0, 0.0]])
        radial = MODULE._source_radial_motion(person, sensor)
        self.assertTrue(np.all(radial["valid"]))
        np.testing.assert_allclose(radial["range_rate_mps"], [-1.0, 0.0, 1.0], atol=1e-9)
        self.assertEqual(["approaching", "quasi_static", "receding"], list(radial["state"]))
        self.assertAlmostEqual(2.995, radial["ttc_proxy_s"][0], places=9)
        self.assertTrue(np.isnan(radial["ttc_proxy_s"][1]))

    def test_ego_matched_motion_has_zero_radial_rate(self):
        timestamps = [1_000_000_000, 1_010_000_000]
        sensor = self._track(timestamps, [[1.0, 0.0, 0.0], [1.01, 0.0, 0.0]])
        person = self._track(timestamps, [[4.0, 0.0, 0.0], [4.01, 0.0, 0.0]])
        radial = MODULE._source_radial_motion(person, sensor)
        self.assertTrue(radial["valid"][0])
        self.assertAlmostEqual(0.0, radial["range_rate_mps"][0], places=12)
        self.assertEqual("quasi_static", radial["state"][0])

    def test_bracketing_uses_native_vicon_pair_without_extrapolation(self):
        person_times = np.asarray([0, 10, 20], dtype=np.int64)
        query = np.asarray([5, 10, 15, 25], dtype=np.int64)
        np.testing.assert_array_equal([0, 0, 1, -1], MODULE._bracketing_pair_indices(query, person_times))

    def test_deadband_boundaries_and_long_gap_fail_closed(self):
        self.assertEqual("approaching", MODULE._radial_state(-MODULE.RADIAL_DEADBAND_MPS))
        self.assertEqual("receding", MODULE._radial_state(MODULE.RADIAL_DEADBAND_MPS))
        self.assertEqual("quasi_static", MODULE._radial_state(0.099))
        timestamps = [1_000_000_000, 1_100_000_000]
        sensor = self._track(timestamps, [[1.0, 0.0, 0.0]] * 2)
        person = self._track(timestamps, [[4.0, 0.0, 0.0], [3.9, 0.0, 0.0]])
        self.assertFalse(MODULE._source_radial_motion(person, sensor)["valid"][0])


if __name__ == "__main__":
    unittest.main()
