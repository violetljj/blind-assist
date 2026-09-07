"""Small evaluator contracts; no cohort or model execution."""
import json
import unittest

import numpy as np

from boundary_retention import diagnose_boundary
from near_field import Camera


class BoundaryRetentionTest(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(40, 30, 90., 1.6, 0.)
        self.native = np.full((30, 40), 1.8, dtype=np.float32)
        self.native[:, 20:] = 2.2

    def test_identical_and_positive_global_scale(self):
        baseline = diagnose_boundary(self.native, self.native, self.camera)
        scaled = diagnose_boundary(self.native, self.native * 17., self.camera)
        self.assertGreater(baseline["native_edges"], 0)
        self.assertEqual(baseline["existence_recall"], 1.)
        for key in ("matched_native_edges", "extraneous_predicted_edges", "predicted_edges_in_neighborhood"):
            self.assertEqual(baseline[key], scaled[key])
        self.assertAlmostEqual(scaled["matched_log_contrast_ratio"]["p50"], 1., places=5)
        json.dumps(scaled, allow_nan=False)

    def test_flat_invalid_and_input_immutability(self):
        original = self.native.copy()
        flat = diagnose_boundary(self.native, np.full_like(self.native, 2.), self.camera)
        self.assertEqual(flat["matched_native_edges"], 0)
        invalid = diagnose_boundary(np.full_like(self.native, np.nan), self.native, self.camera)
        self.assertEqual(invalid["native_edges"], 0)
        self.assertIsNone(invalid["existence_recall"])
        self.assertEqual(invalid["predicted_edges_in_neighborhood"], 0)
        flat_source = np.full_like(self.native, 2.)
        self.assertEqual(diagnose_boundary(flat_source, flat_source, self.camera)["native_edges"], 0)
        np.testing.assert_array_equal(self.native, original)

    def test_radius_and_signed_orientation(self):
        shifted = np.full_like(self.native, 1.8)
        shifted[:, 22:] = 2.2
        nearby = diagnose_boundary(self.native, shifted, self.camera)
        self.assertEqual(nearby["existence_recall"], 1.)
        self.assertEqual(nearby["matched_localization_px"]["p50"], 2.)
        shifted[:, 22] = 1.8  # Boundary now three pixels away.
        self.assertEqual(diagnose_boundary(self.native, shifted, self.camera)["matched_native_edges"], 0)
        reversed_jump = 4. - self.native
        self.assertEqual(diagnose_boundary(self.native, reversed_jump, self.camera)["matched_native_edges"], 0)


if __name__ == "__main__":
    unittest.main()
