"""Synthetic accounting tests; no historical cohort is executed."""
import unittest
import numpy as np
import foundation_geometry_evaluate_20260923 as e


class GeometryAccountingTests(unittest.TestCase):
    def setUp(self):
        self.pose = dict(yaw=0., pitch=0., roll=0., camera_in_body_m=[0., 0., 1.6])

    def depth(self):
        return np.full((360, 640), np.nan, dtype=np.float32)

    def test_missing_prediction_counts_as_accuracy_failure(self):
        native, pred = self.depth(), self.depth()
        native[180, 320] = 1.
        row = e.geometry_rows(pred, native, self.pose)[1]
        self.assertEqual(row['coverage']['native_query_pixels'], 1)
        self.assertEqual(row['coverage']['missing'], 1)
        self.assertEqual(row['coverage']['abs_z_20cm'], 0)

    def test_origins_partition_far_near_outside_and_true(self):
        native, pred = self.depth(), self.depth()
        native[180, 320] = 1.; pred[180, 320] = 1.
        native[180, 321] = 10.; pred[180, 321] = 1.
        native[180, 400] = 3.5; pred[180, 400] = .8
        origin = e.geometry_rows(pred, native, self.pose)[1]['origin']
        self.assertEqual(origin, dict(pixels=3, native_in_query=1, native_far=1,
                                     native_near_outside=1, other=0))

    def test_readout_presence_matches_pixel_membership(self):
        rng = np.random.default_rng(7)
        depth = rng.uniform(.5, 4, (360, 640)).astype('float32')
        for yaw, pitch in [(0, 0), (12, -8), (-20, 10)]:
            pose = dict(self.pose, yaw=yaw, pitch=pitch)
            mask = e.membership(depth, pose)
            np.testing.assert_array_equal(mask.any(axis=(0, 1)),
                e.m.readout(e.m.depth_points(depth), pose, common_fov=True)[0] > 0)

    def test_out_of_range_prediction_missing_without_altering_source(self):
        native, pred = self.depth(), self.depth()
        native[180, 320] = 1.; pred[180, 320] = 8.
        self.assertEqual(e.geometry_rows(pred, native, self.pose)[1]['coverage']['missing'], 1)
        self.assertEqual(pred[180, 320], 8.)

    def test_correct_alert_wrong_geometry_is_reported(self):
        native, pred = self.depth(), self.depth()
        native[180, 320] = 8.; pred[180, 320] = 1.
        row = dict(model='candidate', family='small_head', part='HEAD', truth=True,
                   **e.geometry_rows(pred, native, self.pose)[1])
        result = e.aggregate_geometry([row])['candidate']['all']
        self.assertEqual(result['supported_queries']['task_tp_without_native_in_query'], 1)
        self.assertIsNone(result['coverage_rates']['abs_z_5cm'])


if __name__ == '__main__':
    unittest.main()
