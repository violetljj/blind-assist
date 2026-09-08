"""Projection and UNKNOWN gates, using CPU tensors only for small unit fixtures."""
import unittest
import torch
import verify_city_native_targets as labels


class NativeTargetLabelTest(unittest.TestCase):
    def setUp(self):
        self.native = torch.full((64, 64), 2., dtype=torch.float32)
        self.camera = dict(x=-299., y=-1., z=1.7, pitch=0., yaw=0., roll=0.)

    def test_yaw_translation_invariant_wearer_projection(self):
        first, _ = labels.in_query(self.native, self.camera, 0.)
        turned = dict(self.camera, yaw=93., x=1800., y=-1200.)
        second, _ = labels.in_query(self.native, turned, 0.)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(first[1].any()))

    def test_missing_floor_and_native_stay_unknown(self):
        mask, near = labels.scene_labels(self.native, self.camera, None)
        self.assertEqual(near, [-1, -1])
        self.assertTrue(bool((mask == -1).all()))
        mask, near = labels.scene_labels(torch.zeros_like(self.native), self.camera, 0.)
        self.assertEqual(near, [-1, -1])
        self.assertTrue(bool((mask == -1).all()))

    def test_agreement_reliable_but_closer_clone_unknown(self):
        mask, row = labels.target_label(self.native, self.native.clone(), self.camera, 0.)
        self.assertEqual(row['status'], 'EVALUABLE')
        self.assertGreater(row['reliable_in_query_pixels'][1], 2)
        isolated = self.native.clone()
        isolated[0, 0] -= .1
        mask, row = labels.target_label(self.native, isolated, self.camera, 0.)
        self.assertEqual(row['unexplained_nearer_clone_pixels'], 1)
        self.assertEqual(row['status'], 'UNKNOWN')
        self.assertTrue(bool((mask == -1).all()))
        self.assertEqual(row['uncertainty_reasons'], ['UNEXPLAINED_NEARER_CLONE'])

    def test_occluded_target_is_not_visible_positive(self):
        mask, row = labels.target_label(self.native, self.native + 1., self.camera, 0.)
        self.assertEqual(row['occluded_pixels'], 64 * 64)
        self.assertEqual(row['status'], 'UNKNOWN')
        self.assertEqual(int((mask == 1).sum()), 0)

    def test_collision_failure_keeps_native_identity_diagnostic(self):
        mask, row = labels.target_label(self.native, self.native.clone(), self.camera, 0., 'FAIL')
        self.assertEqual(row['native_agree_pixels'], 64 * 64)
        self.assertEqual(row['raycheck'], 'FAIL')
        self.assertEqual(row['uncertainty_reasons'], ['INDEPENDENT_RAYCHECK_UNRESOLVED'])
        self.assertEqual(row['status'], 'UNKNOWN')
        self.assertTrue(bool((mask == -1).all()))


if __name__ == '__main__':
    unittest.main()
