import unittest

import numpy as np

from tof_geometry_temporal_expert import causal_gap_fill, predict, project_events, zone_center_rays


class TofGeometryTemporalExpertTests(unittest.TestCase):
    def packet(self, n=8):
        return np.zeros((n, 64, 2), np.float32), np.zeros((n, 64, 2), bool)

    def test_zone_rays_are_unit_and_row_major(self):
        rays = zone_center_rays()
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)
        self.assertGreater(rays[0, 2], rays[-1, 2])
        self.assertLess(rays[0, 1], rays[7, 1])

    def test_invalid_is_not_projected_as_clear_support(self):
        ranges, valid = self.packet(1)
        margins, support = project_events(ranges, valid)
        np.testing.assert_array_equal(margins, -np.ones((1, 4)))
        np.testing.assert_array_equal(support, np.zeros((1, 4)))

    def test_direct_head_and_body_projection(self):
        ranges, valid = self.packet(1)
        # Central-upper and central-lower zones at 1 m reach head/body boxes.
        ranges[0, 3 * 8 + 3, 0] = 1.0
        valid[0, 3 * 8 + 3, 0] = True
        ranges[0, 7 * 8 + 3, 0] = 1.0
        valid[0, 7 * 8 + 3, 0] = True
        margins, _ = project_events(ranges, valid)
        self.assertEqual(margins[0, 0], 1.0)
        self.assertEqual(margins[0, 2], 1.0)

    def test_causal_closing_fill_stops_after_three_frames(self):
        ranges, valid = self.packet(7)
        ranges[:3, 27, 0] = [2.0, 1.9, 1.8]
        valid[:3, 27, 0] = True
        clip = np.array(['a'] * 7)
        time = np.arange(7) / 10
        filled, filled_valid, imputed = causal_gap_fill(ranges, valid, clip, time)
        np.testing.assert_allclose(filled[3:6, 27, 0], [1.7, 1.6, 1.5], atol=1e-6)
        self.assertTrue(imputed[3:6, 27, 0].all())
        self.assertFalse(filled_valid[6, 27, 0])

    def test_static_receding_and_cross_clip_returns_are_not_filled(self):
        ranges, valid = self.packet(6)
        ranges[:4, 27, 0] = [1.0, 1.0, 1.1, 1.2]
        valid[:4, 27, 0] = True
        clip = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
        time = np.array([0, .1, .2, .3, 0, .1])
        _, filled_valid, imputed = causal_gap_fill(ranges, valid, clip, time)
        self.assertFalse(imputed.any())
        self.assertFalse(filled_valid[4:].any())

    def test_predict_never_removes_direct_alerts(self):
        ranges, valid = self.packet(5)
        ranges[:, 27, 0] = np.linspace(2.0, 1.6, 5)
        valid[:, 27, 0] = True
        result = predict(ranges, valid, np.array(['a'] * 5), np.arange(5) / 10)
        self.assertTrue(np.all(result['GEOMETRY_TEMPORAL'] >= result['GEOMETRY_FRAME']))


if __name__ == '__main__':
    unittest.main()
