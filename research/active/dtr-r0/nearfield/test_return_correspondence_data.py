"""Synthetic-only fixed-ray and observed-return teacher checks."""
import unittest
from unittest.mock import patch

import numpy as np

from ba_camera_corridor import sample_indices, sample_native
from inherit_spatial_model import QUERIES
from query_occupancy_data import observation_tokens
from return_correspondence_data import FOCAL640, RAYS, ray_layout, teacher
from tof_fov45_core import boxes45, simulate

IDENTITY = 'query-occupancy/synthetic-correspondence-only'


def public(depth):
    values, traces = simulate(sample_native(depth), IDENTITY, boxes45())
    return observation_tokens(values, boxes45()), traces


class ReturnCorrespondenceDataTests(unittest.TestCase):
    def test_interval_possible_does_not_require_nominal_or_native_point_inside(self):
        from return_correspondence_check import interval_counts
        # y slope about .21: nominal Z=1.9 lies below BODY, but the original
        # interval still intersects BODY. Perfect angular association is insufficient.
        values = np.full(64, 1.9, np.float32)
        tof = observation_tokens(values, boxes45())
        native = np.array([236*640+320])
        self.assertLess(((236.5-180)/FOCAL640)*1.9, .42)
        self.assertEqual(interval_counts(native, np.array([0]), tof)[1], 1)
        self.assertEqual(interval_counts(native, np.array([0]), tof, np.array([False])), [0]*6)

    def test_unobserved_zone_cannot_supply_contributor_support(self):
        from return_correspondence_check import interval_counts
        tof = observation_tokens(np.full(64, np.nan, np.float32), boxes45())
        with self.assertRaisesRegex(AssertionError, 'unobserved'):
            interval_counts(np.array([180*640+320]), np.array([0]), tof)

    def test_unique_fixed_mapping_and_exact_native_pixel_centres(self):
        layout = ray_layout()
        self.assertEqual(RAYS, 1024)
        for key in ('zone', 'sensor_indices', 'native_indices', 'rgb_indices', 'ax', 'ay'):
            self.assertEqual(layout[key].shape, (1024,))
        np.testing.assert_array_equal(np.bincount(layout['zone']), np.full(64, 16))
        sy, sx = np.divmod(layout['sensor_indices'], 256)
        ny, nx = sample_indices()
        np.testing.assert_array_equal(layout['native_indices'], ny[sy]*640+nx[sx])
        np.testing.assert_array_equal(layout['rgb_indices'], (ny[sy]//2)*320+nx[sx]//2)
        np.testing.assert_array_equal(layout['ax'], (nx[sx]+.5-320)/FOCAL640)
        np.testing.assert_array_equal(layout['ay'], (ny[sy]+.5-180)/FOCAL640)
        for zone, (y0, x0, y1, x1) in enumerate(boxes45()):
            keep = layout['zone'] == zone
            np.testing.assert_array_equal(np.unique(sy[keep]), np.linspace(y0, y1-1, 4).round())
            np.testing.assert_array_equal(np.unique(sx[keep]), np.linspace(x0, x1-1, 4).round())
        self.assertEqual(len(np.unique(layout['native_indices'])), 1024)

    def test_flat_plane_observed_winner_labels_and_query_coordinates(self):
        depth = np.full((360, 640), 1.5, np.float32)
        tof, _ = public(depth)
        answer = teacher(depth, tof, IDENTITY)
        layout = ray_layout()
        self.assertTrue(answer['public_tof_exact_parity'])
        self.assertTrue(answer['native_valid'].all())
        np.testing.assert_array_equal(answer['y'], answer['eligible'])
        self.assertTrue(answer['y'].any())
        self.assertEqual(answer['query_inside'].shape, (1024, 6))
        x, y = layout['ax']*1.5, layout['ay']*1.5
        expected = np.stack([(x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
            for xl, xh, yl, yh in QUERIES], axis=1)
        np.testing.assert_array_equal(answer['query_inside'], expected)

    def test_dropout_has_no_contributor_labels_despite_native_visibility(self):
        class Dropout:
            def random(self):
                return 0.
        depth = np.full((360, 640), 1.5, np.float32)
        with patch('tof_fov45_core.np.random.default_rng', return_value=Dropout()):
            tof, traces = public(depth)
            answer = teacher(depth, tof, IDENTITY)
        self.assertTrue(all(t['reason'] == 'SIMULATED_DROPOUT' for t in traces))
        self.assertFalse(answer['y'].any())
        self.assertFalse(answer['eligible'].any())
        self.assertTrue(answer['has_native'].all())
        self.assertTrue(answer['query_inside'].any())

    def test_native_invalid_ray_stays_eligible_in_observed_zone(self):
        depth = np.full((360, 640), 1.5, np.float32)
        tof, _ = public(depth)
        layout = ray_layout()
        ray = int(np.flatnonzero(tof[layout['zone'], 1] == 1)[0])
        depth.ravel()[layout['native_indices'][ray]] = np.nan
        tof, _ = public(depth)
        answer = teacher(depth, tof, IDENTITY)
        self.assertTrue(answer['eligible'][ray])
        self.assertFalse(answer['y'][ray])
        self.assertFalse(answer['native_valid'][ray])
        self.assertFalse(answer['query_inside'][ray].any())

    def test_native_point_depth_is_never_interpolated_and_tof_mismatch_rejected(self):
        depth = np.full((360, 640), 1.5, np.float32)
        layout = ray_layout()
        ray = int(np.flatnonzero((np.abs(layout['ax']) < .1) & (layout['ay'] > .31))[0])
        depth.ravel()[layout['native_indices'][ray]] = 3.5
        sampled = sample_native(depth).ravel()
        self.assertEqual(sampled[layout['sensor_indices'][ray]], 3.5)
        tof, _ = public(depth)
        answer = teacher(depth, tof, IDENTITY)
        self.assertTrue(answer['native_valid'][ray])
        self.assertFalse(answer['query_inside'][ray].any())
        altered = tof.copy()
        altered[0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'parity'):
            teacher(depth, altered, IDENTITY)


if __name__ == '__main__':
    unittest.main()
