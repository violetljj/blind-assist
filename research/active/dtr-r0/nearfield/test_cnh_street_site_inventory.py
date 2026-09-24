import unittest
import numpy as np
from cnh_street_site_inventory import clips_at, query_envelope, overlaps, regions


class StreetInventoryTest(unittest.TestCase):
    def test_sweep_covers_travel_lateral_channels_and_margin(self):
        lo, hi = query_envelope(clips_at(0., 0., 1.6, 0))
        np.testing.assert_allclose(lo, [.15, -1.05, .55])
        np.testing.assert_allclose(hi, [7.15, 1.05, 1.95])

    def test_touching_margin_rejects(self):
        lo, hi = query_envelope(clips_at(0., 0., 1.6, 0))
        lows = np.array([[hi[0], 0., 1.], [hi[0]+.001, 0., 1.]])
        highs = lows + .01
        self.assertEqual(overlaps(lo, hi, lows, highs).tolist(), [True, False])

    def test_rotated_full_sweep(self):
        lo, hi = query_envelope(clips_at(0., 0., 1.6, 90))
        np.testing.assert_allclose(lo, [-1.05, .15, .55], atol=1e-10)
        np.testing.assert_allclose(hi, [1.05, 7.15, 1.95], atol=1e-10)

    def test_authored_regions_do_not_invent_alley(self):
        recipe = dict(graphs=[dict(label='BA sidewalk', location_m=[10, 20, 0], yaw=90,
            parameters=[dict(name='Width', value=5400), dict(name='Length', value=900)]),
            dict(label='BA plaza', location_m=[55, 42, 0])])
        areas = regions(recipe)
        self.assertEqual([r['category'] for r in areas], ['intersection', 'sidewalk', 'plaza'])
        self.assertEqual(areas[1]['low'], [5.5, -7.])
        self.assertEqual(areas[1]['high'], [14.5, 47.])


if __name__ == '__main__':
    unittest.main()
