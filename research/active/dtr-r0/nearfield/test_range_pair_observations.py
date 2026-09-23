import unittest
import numpy as np

from range_pair_observations import (public_readouts, evaluator_target_mask,
                                     summarize_trace_support)


class RangePairObservationTest(unittest.TestCase):
    def tof(self):
        t = np.zeros((64, 6))
        t[:, 2:] = [.45, .45, .55, .55]
        return t

    def test_lateral_outside_excluded(self):
        t = self.tof()
        t[:2, 1] = 1
        t[:2, 0] = np.array([2., 3.2])/8
        t[0, 2:] = [.45, .9, .55, 1.]
        out = public_readouts(t)
        self.assertEqual(out['global_min']['selected_zone_ids'], [0])
        self.assertEqual(out['corridor_min']['selected_zone_ids'], [1])
        self.assertEqual(out['corridor_min']['estimate_m'], 3.2)

    def test_near_far_are_both_observed_not_boundary_filtered(self):
        for distance in (.2, 2.82, 3.08, 7.9):
            t = self.tof(); t[0, :2] = [distance/8, 1]
            self.assertAlmostEqual(public_readouts(t)['corridor_min']['estimate_m'], distance)

    def test_missing_and_three_return_requirement(self):
        t = self.tof()
        self.assertIsNone(public_readouts(t)['global_min']['estimate_m'])
        t[:2, 1] = 1; t[:2, 0] = [2/8, 3/8]
        self.assertIsNone(public_readouts(t)['corridor_median3']['estimate_m'])
        t[2, :2] = [4/8, 1]
        out = public_readouts(t)['corridor_median3']
        self.assertEqual(out, dict(estimate_m=3., selected_zone_ids=[0, 1, 2]))

    def geo(self, front=2.):
        return dict(declared_camera=dict(x=10., y=1., z=1.82, pitch=0., yaw=0., roll=0.),
                    actual_camera_location_m=[10., 1., 1.82], objects=[
            dict(name='target_a', render_bounds_center_m=[10.+front+.1, 1., 1.82],
                 render_bounds_extent_m=[.1, .25, .25]),
            dict(name='target_b', render_bounds_center_m=[10.+front+.1, 3., 1.82],
                 render_bounds_extent_m=[.1, .1, .1]),
            dict(name='background', render_bounds_center_m=[16.1, 1., 1.82],
                 render_bounds_extent_m=[.1, 20., 20.])])

    def test_background_is_not_target_and_selected_lineage_only(self):
        native = np.full((360, 640), 6., np.float32)
        native[160:200, 300:340] = 2.
        mask = evaluator_target_mask(native, self.geo())
        self.assertTrue(mask.any()); self.assertFalse(mask[0, 0])
        yes = int(np.flatnonzero(mask)[0]); no = int(np.flatnonzero(~mask)[0])
        traces = [dict(zone_id=0, observed=True, pixel_indices=np.array([no])),
                  dict(zone_id=1, observed=True, pixel_indices=np.array([yes, no]))]
        self.assertEqual(summarize_trace_support(traces, [0], mask),
                         dict(target_pixels=0, total_pixels=1, target_witnessed=False))
        self.assertEqual(summarize_trace_support(traces, [1], mask),
                         dict(target_pixels=1, total_pixels=2, target_witnessed=True))
        self.assertEqual(summarize_trace_support(traces, [], mask)['total_pixels'], 0)

    def test_cube_crossing_camera_plane_and_invalid_depth(self):
        geo = self.geo(front=-.1)
        depth = np.full((360, 640), .05, np.float32)
        self.assertTrue(evaluator_target_mask(depth, geo).any())
        depth[:] = 0
        self.assertFalse(evaluator_target_mask(depth, geo).any())
        depth[:] = np.nan
        self.assertFalse(evaluator_target_mask(depth, geo).any())
        geo['declared_camera']['yaw'] = 1.
        with self.assertRaises(ValueError):
            evaluator_target_mask(depth, geo)


if __name__ == '__main__':
    unittest.main()
