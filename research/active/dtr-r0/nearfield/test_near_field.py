"""Focused evidence-contract checks, not empirical obstacle accuracy tests."""
import unittest

import numpy as np
import torch

from near_field import Camera, NearFieldEncoder


class EvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.camera = Camera(64, 48, 90, 1.6)
        cls.encoder = NearFieldEncoder(cls.camera)

    def test_missing_observation_is_unknown_and_has_no_alert(self):
        for value in (0., np.nan, np.inf):
            evidence = self.encoder.encode(np.full((48, 64), value, np.float32))
            self.assertFalse(evidence.alerts().any())
            self.assertEqual({s for row in evidence.region_state for s in row}, {"UNKNOWN"})

    def test_thin_vertical_surface_survives_and_input_is_unchanged(self):
        frame = np.full((48, 64), 8., np.float32)
        frame[25:36, 31] = 2.
        before = frame.copy()
        evidence = self.encoder.encode(frame)
        self.assertTrue(evidence.alerts()[1, 1])
        self.assertEqual(evidence.region_distance_m[1, 1], 2.)
        np.testing.assert_array_equal(before, frame)

    def test_isolated_near_pixel_is_unknown_instead_of_far_clearance(self):
        frame = np.full((48, 64), 8., np.float32)
        frame[31, 31] = 2.
        evidence = self.encoder.encode(frame)
        self.assertFalse(evidence.alerts().any())
        self.assertEqual(evidence.region_state[1][1], "UNKNOWN")

    def test_two_directions_inside_one_tile_both_survive(self):
        # At this calibration columns 37 and 38 straddle +10deg and share tile 2.
        frame = np.full((48, 64), 8., np.float32)
        frame[26:33, 37] = 1.8
        frame[26:33, 38] = 2.2
        evidence = self.encoder.encode(frame)
        self.assertTrue(evidence.alerts()[1, 1])
        self.assertTrue(evidence.alerts()[2, 1])

    def test_no_implicit_calibration_or_shape_fallback(self):
        with self.assertRaises(ValueError):
            Camera(64, 48, 90, float("nan"))
        with self.assertRaises(ValueError):
            self.encoder.encode(np.zeros((24, 32), np.float32))


if __name__ == "__main__":
    unittest.main()
