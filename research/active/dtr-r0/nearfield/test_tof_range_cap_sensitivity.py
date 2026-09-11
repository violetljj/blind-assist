import unittest

import numpy as np

from tof_range_cap_sensitivity import apply_range_cap, zone_caps


class TofRangeCapSensitivityTests(unittest.TestCase):
    def test_center_four_and_corners_match_endpoints(self):
        caps = zone_caps(1.55, 1.40).reshape(8, 8)
        np.testing.assert_allclose(caps[3:5, 3:5], 1.55)
        np.testing.assert_allclose(caps[[0, 0, 7, 7], [0, 7, 0, 7]], 1.40)
        self.assertTrue(np.all((caps >= 1.40) & (caps <= 1.55)))

    def test_cap_removes_only_over_range_usable_slots(self):
        ranges = np.zeros((1, 64, 2), np.float32)
        valid = np.zeros_like(ranges, dtype=bool)
        ranges[0, 27, :] = [1.50, 1.60]
        valid[0, 27, :] = True
        capped, kept, removed = apply_range_cap(ranges, valid, 1.55, 1.40)
        self.assertTrue(kept[0, 27, 0])
        self.assertTrue(removed[0, 27, 1])
        self.assertEqual(capped[0, 27, 1], 0.0)

    def test_existing_invalid_return_stays_invalid_not_removed(self):
        ranges = np.ones((1, 64, 2), np.float32) * 3.0
        valid = np.zeros_like(ranges, dtype=bool)
        _, kept, removed = apply_range_cap(ranges, valid, 1.0, 0.8)
        self.assertFalse(kept.any())
        self.assertFalse(removed.any())

    def test_invalid_configuration_and_shape_fail(self):
        with self.assertRaises(ValueError):
            zone_caps(1.0, 1.2)
        with self.assertRaises(ValueError):
            apply_range_cap(np.zeros((1, 8, 8)), np.zeros((1, 8, 8), bool), 1.0, 0.8)


if __name__ == '__main__':
    unittest.main()
