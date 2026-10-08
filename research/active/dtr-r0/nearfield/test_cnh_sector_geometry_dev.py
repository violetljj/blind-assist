"""Focused CPU geometry checks: identity, extents, coordinate frame, support."""
import unittest
import numpy as np
import cnh_sector_geometry_dev as E
import cnh_sequence_observed_geometry as G


def box(lo, hi):
    return dict(lo=list(lo), hi=list(hi), rho=.5)


def poses():
    p = np.repeat(np.eye(4)[None], 16, axis=0)
    p[:, 2, 3] = np.linspace(-2.4, 0., 16)
    return p


class ContactObjectLocalization(unittest.TestCase):
    def test_contact_object_need_not_be_deadline_box(self):
        boxes = [box([1., -.1, .6], [1.1, .26, .7]),
                 box([.1, -.1, .6], [.2, .26, .7])]
        travel = poses()
        _, ref = G.deadline_reference(G.corners(boxes[0]), travel[3:])
        deadline = int(np.searchsorted(E.OUTPUT_FRAMES, ref['reference_fraction']+1e-8, side='right')-1)
        details = E.localization_truth(boxes, travel, travel, deadline, True, return_details=True)
        mask = details['mask']
        self.assertEqual(details['target_ids'], [1])
        self.assertEqual(mask.shape, (13, 3))
        self.assertTrue(mask[:, 1].all())

    def test_box_extent_crosses_boundary(self):
        target = box([.1, -.1, 1.], [.25, .26, 1.1])
        np.testing.assert_array_equal(E.box_sector_mask(target, np.eye(4)), [False, True, True])
        # Its center alone is in CENTER; RIGHT support must survive.
        center = (np.asarray(target['lo'])+np.asarray(target['hi']))/2
        self.assertLess(np.degrees(np.arctan2(center[0], center[2])), 10.)

    def test_sensor_yaw_translation_and_frame_alignment(self):
        target = box([-.03, -.1, 1.], [.03, .26, 1.1])
        sensor = np.eye(4)
        sensor[:3, :3] = G.S.ry(20.)
        np.testing.assert_array_equal(E.box_sector_mask(target, sensor), [True, False, False])
        sensor = np.eye(4); sensor[0, 3] = -.5
        np.testing.assert_array_equal(E.box_sector_mask(target, sensor), [False, False, True])
        pitched = sensor.copy()
        a = np.radians(-10.)
        pitched[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        np.testing.assert_array_equal(E.box_sector_mask(target, pitched), E.box_sector_mask(target, sensor))

    def test_behind_and_multiple_contact_objects(self):
        np.testing.assert_array_equal(E.box_sector_mask(box([-.1, -.1, -2.], [.1, .26, -1.]), np.eye(4)), [False]*3)
        travel = poses()
        boxes = [box([-.2, -.1, .6], [-.1, .26, .7]),
                 box([.1, .5, .6], [.2, .8, .7])]
        _, ref = G.deadline_reference(G.corners(boxes[0]), travel[3:])
        deadline = int(np.searchsorted(E.OUTPUT_FRAMES, ref['reference_fraction']+1e-8, side='right')-1)
        details = E.localization_truth(boxes, travel, travel, deadline, True, return_details=True)
        self.assertEqual(details['target_ids'], [0, 1])
        self.assertEqual(details['target_count'], 2)

    def test_inconsistent_event_is_not_assigned_to_box_zero(self):
        travel = poses(); boxes = [box([1., -.1, .6], [1.1, .26, .7])]
        _, ref = G.deadline_reference(G.corners(boxes[0]), travel[3:])
        deadline = int(np.searchsorted(E.OUTPUT_FRAMES, ref['reference_fraction']+1e-8, side='right')-1)
        with self.assertRaisesRegex(ValueError, 'no original contact'):
            E.localization_truth(boxes, travel, travel, deadline, True)
        with self.assertRaisesRegex(ValueError, 'differs from geometry'):
            E.localization_truth(boxes, travel, travel, deadline-1, True)
        details = E.localization_truth(boxes, travel, travel, deadline, False, return_details=True)
        self.assertFalse(details['mask'].any()); self.assertEqual(details['status'], 'NOT_CONTACT')


if __name__ == '__main__':
    unittest.main()
